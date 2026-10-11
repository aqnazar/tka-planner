"""The parametric TKA demo: the planner, with the implant built in Fusion from the plan.

Private (patient-derived). Runs the planner's own server (sessions, plan controls,
commit, trial reduction, arc) with the demo app as its web root, and adds:

  GET  /api/demo/knees                  the three demo knees
  GET  /api/demo/knees/<name>           a knee's built parts, Fusion parameters, checks,
                                        tray rule, J-curves (demo_app/data/manifest.json)
  POST /api/sessions/<id>/parametric    the parameters derived from the session's current
                                        (adjusted) plan, the natural J-curves, and what
                                        differs from the parts last built
  POST /api/sessions/<id>/build         build the knee in Fusion from that derivation
  GET  /api/demo/build                  the build's progress
  POST /api/sessions/<id>/kinematics    the flexion arc on the built parts (kinematics.py):
                                        poses and readings, {max_deg, steps, model}
  POST /api/sessions/<id>/trial         as the planner's, but flexion follows the same
                                        kinematics ({model} picks the profile)
  POST /api/demo/server/restart         restart this server (a fresh process on the same
                                        port: new code, all sessions dropped)
  POST /api/demo/server/stop            stop it

  python demo_server.py [--port 8770]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from tka_planner.server import httpd  # noqa: E402
from tka_planner.server.api import Api, ApiError  # noqa: E402
from tka_planner.server.sessions import SessionManager  # noqa: E402

APP = HERE / "demo_app"
BUILDS = HERE / "patient_builds"
JOBS = BUILDS / "jobs_demo.json"
MANIFEST = APP / "data" / "manifest.json"
COMPARE = {"femoral": ("Femur_ML", "C_Distal_cut_surface", "B_Posterior_chamfer_projection",
                       "D_Anterior_chamfer_projection"),
           "plate": ("Tibial_Plateau_ML", "Tibial_Plateau_AP", "Left_Condyle_Diameter", "Right_Condyle_Diameter",
                     "Tray_Offset_X", "Tray_Offset_Y"),
           "insert": ("Insert_Thickness", "Distal_Cut")}


def mm(v):
    return float(str(v).split()[0])


class DemoApi(Api):
    def __init__(self, manager):
        super().__init__(manager)
        self.lock = threading.Lock()
        self.derived = {}                      # session id -> last derivation
        self.build_state = {"state": "idle", "log": []}
        self.blocks = {}                       # block STL -> (mtime, mesh)
        self.block_used = {}                   # (session id, bone) -> (mesh id, pose) last cut with
        self.components = {}                   # femoral STL -> (mtime, CAD vertices)
        self.motion = {}                       # session id -> kinematics model name
        self.leaving = threading.Event()       # set by a restart or stop from the page

    # -- routing ------------------------------------------------------------

    def _route(self, method, parts, body):
        rest = parts[1:] if parts and parts[0] == "api" else []
        if rest[:1] == ["demo"]:
            return self._demo(method, rest[1:], body)
        if len(rest) == 3 and rest[0] == "sessions" and rest[2] == "poses" and method == "GET":
            return self.poses(self.manager.get(rest[1]))
        if len(rest) == 3 and rest[0] == "sessions" and rest[2] in ("parametric", "build") and method == "POST":
            entry = self.manager.get(rest[1])
            return self.parametric(entry, body) if rest[2] == "parametric" else self.build(entry, body)
        if len(rest) == 3 and rest[0] == "sessions" and rest[2] == "commit" and method == "POST":
            self.use_built_block(self.manager.get(rest[1]))
        if len(rest) == 3 and rest[0] == "sessions" and rest[2] == "kinematics" and method == "POST":
            return self.kinematics(self.manager.get(rest[1]), body or {})
        if rest[:1] == ["sessions"] and rest[2:] in (["trial"], ["trial", "reset"]) and method == "POST":
            entry = self.manager.get(rest[1])
            out = super()._route(method, parts, body)
            return self.trial(entry, out, (body or {}).get("model"))
        if rest == ["sessions"] and method == "POST":
            # A knee is known by its case id, not by its folder's name (the cohort's
            # folders are named Left and Right).
            out = super()._route(method, parts, body)
            if (body or {}).get("name"):
                entry = self.manager.get(out["session_id"])
                entry.session.case_id = body["name"]
                self.use_built_block(entry)          # the knee's own blocks, from the start
                out = self.snapshot(entry)
            return out
        return super()._route(method, parts, body)

    # -- kinematics on the built parts (kinematics.py) ----------------------------

    def knee_motion(self, entry):
        """The kinematic model of this session's knee on its built femoral component, at
        the current plan's poses; None for a knee with no built component."""
        from kinematics import Knee
        from tka_planner.core.meshio import read_stl
        s = entry.session
        stl = BUILDS / s.case_id / "femoral.stl"
        e = next((k for k in self.manifest()["knees"] if k["name"] == s.case_id), None)
        if not stl.exists() or not e or not e.get("jcurve"):
            return None
        stamp = stl.stat().st_mtime
        if self.components.get(str(stl), (None,))[0] != stamp:
            self.components[str(stl)] = (stamp, read_stl(stl).vertices)
        poses = self.poses(entry)
        ys = {k: v["y_mm"] for k, v in e["jcurve"]["condyles"].items()}
        return Knee(self.components[str(stl)][1], ys, poses["femoral"], poses["tibial"])

    def stress(self, entry, knee):
        """The trial's varus-valgus stress, drawer and distraction, at rest, about a pivot
        between the two condyles' contacts (the planner's own pivot is the epicondyles',
        which a misplaced epicondyle can put far up the shaft)."""
        import numpy as np
        from tka_planner.scene import motion
        s = entry.session
        c = np.mean([knee.rest[n] for n in knee.rest], axis=0)
        pivot = motion.pivot_frame(knee.F[:3, :3] @ c + knee.F[:3, 3], knee.F[:3, 1])
        t = s.trial
        return motion.trial_pose(pivot, flexion_deg=0.0, varus_valgus_deg=t.varus_valgus_deg,
                                 drawer_ap_mm=t.drawer_ap_mm, distraction_mm=t.distraction_mm,
                                 distal=-s.measurement.tibial_frame.z_proximal)

    def trial(self, entry, out, model):
        """Re-pose the tibia after the planner's trial route: flexion by the kinematics."""
        from tka_planner.scene.model import TIBIAL
        if model:
            self.motion[entry.session_id] = model
        knee = self.knee_motion(entry)
        if knee is None:
            return out
        model = self.motion.get(entry.session_id, "published")
        P, info = knee.pose(entry.session.trial.flexion_deg, model)
        pose = P @ self.stress(entry, knee)
        entry.session.scene.set_pose(TIBIAL, pose)
        out.setdefault("delta", {}).setdefault("poses", {})["TibialSet"] = pose.tolist()
        out["kinematics"] = dict(info, model=model)
        return out

    def kinematics(self, entry, body):
        from kinematics import PUBLISHED
        knee = self.knee_motion(entry)
        if knee is None:
            raise ApiError(400, "This knee has no built femoral component to move on.")
        model = body.get("model") or self.motion.get(entry.session_id, "published")
        self.motion[entry.session_id] = model
        steps, top = max(2, int(body.get("steps", 60))), float(body.get("max_deg", 120.0))
        stress = self.stress(entry, knee)
        frames = []
        for i in range(steps + 1):
            P, info = knee.pose(top * i / steps, model)
            frames.append(dict(info, pose=(P @ stress).tolist()))
        return {"frames": frames, "model": model, "published": PUBLISHED}

    # -- the knee's own cutting block ------------------------------------------

    BUILT_BLOCKS = (("Femur", "femoral", "femoral_block.stl"),      # femoral_block.py
                    ("Tibia", "tibial", "tibial_block.stl"))        # tibial_block.py

    def use_built_block(self, entry):
        """Cut each bone, and take its shell, with the knee's Fusion-built block instead of
        the library block scaled to size: the femoral from its femoral's own J-curve, the
        tibial from its plate (the cut and the keel), at the built parts' poses. The shell
        is taken with the same block (author 2026-10-10): the library's shells differed
        from their blocks only by the 0.4 mm face offset, which isn't used. A knee with no
        built block keeps the library one."""
        import numpy as np
        from tka_planner.core.meshio import read_stl
        from tka_planner.scene.build import BLOCK_COLOUR
        from tka_planner.scene.model import FEMORAL, TIBIAL, Node
        s = entry.session
        if s.scene is None:
            return
        poses = self.poses(entry)
        for bone, group, file in self.BUILT_BLOCKS:
            stl = BUILDS / s.case_id / file
            if not stl.exists():
                continue
            stamp = stl.stat().st_mtime
            if self.blocks.get(str(stl), (None,))[0] != stamp:
                self.blocks[str(stl)] = (stamp, read_stl(stl))
            mesh = self.blocks[str(stl)][1]
            rest = np.asarray(poses[group], float)
            for name, shell in ((f"{group}_cutting_block", False), (f"{group}_cutting_block_shell", True)):
                if name in s.scene.nodes:
                    s.scene.replace_mesh(name, mesh)
                    s.scene.nodes[name].rest = rest
                else:
                    s.scene.add(Node(name=name, set_id=FEMORAL if group == "femoral" else TIBIAL, rest=rest,
                                     colour=BLOCK_COLOUR, visible=s.controls.show_cutting_blocks,
                                     tags={"component": True, "group": group, "cutting_block": True,
                                           "shell": shell}), mesh)
                s.scene.nodes[name].tags["built"] = file
            # What the page draws: the block trimmed to a box around its part, so it reads
            # as a block over the cut bone rather than the 300 mm tool it is.
            view = f"{group}_cutting_block_view"
            vmesh = self.block_view(stl, stamp, mesh, group)
            if view in s.scene.nodes:
                s.scene.replace_mesh(view, vmesh)
                s.scene.nodes[view].rest = rest
            else:
                s.scene.add(Node(name=view, set_id=FEMORAL if group == "femoral" else TIBIAL, rest=rest,
                                 colour=BLOCK_COLOUR, visible=s.controls.show_cutting_blocks,
                                 tags={"component": True, "group": group, "cutting_block": True,
                                       "display": True}), vmesh)
            used = (s.scene.nodes[f"{group}_cutting_block"].mesh_id, rest.round(6).tobytes())
            # The commit cache keys on the plan, not on the block; a new block re-cuts.
            if self.block_used.get((entry.session_id, bone)) != used:
                entry.committed.pop(bone, None)
                self.block_used[(entry.session_id, bone)] = used

    def block_view(self, stl, stamp, mesh, group):
        """The block within 5 mm of its part (the femoral or the plate, same frame), from
        just under the part to 12 mm over the tibial cut. Cached per block file."""
        import numpy as np
        from tka_planner.core.meshio import read_stl
        from tka_planner.geom import mesh as gm
        from tka_planner.geom.kernel import default_kernel
        key = ("view", str(stl))
        if self.blocks.get(key, (None,))[0] != stamp:
            part = read_stl(stl.with_name("femoral.stl" if group == "femoral" else "plate.stl")).vertices
            lo, hi = part.min(0) - 5.0, part.max(0) + 5.0
            if group == "tibial":
                lo[2], hi[2] = part[:, 2].min() - 1.0, part[:, 2].max() + 12.0
            else:
                lo[2], hi[2] = part[:, 2].min() - 3.0, part[:, 2].max() + 3.0
            box = gm.box(hi - lo, gm.translation((hi + lo) / 2))
            self.blocks[key] = (stamp, default_kernel().intersect(mesh, box).mesh)
        return self.blocks[key][1]

    def _demo(self, method, rest, body):
        if rest == ["knees"] and method == "GET":
            from demo_cases import PATIENTS
            return {"knees": [dict({k: e.get(k) for k in ("name", "label", "side", "folder", "scale", "anatomy",
                                                          "plan_controls")},
                                   group="patient" if e["name"] in PATIENTS else "reference",
                                   built=bool(e["parts"]),
                                   versions={p: f.get("version") for p, f in e["fusion"].items()})
                              for e in self.manifest()["knees"]]}
        if len(rest) == 2 and rest[0] == "knees" and method == "GET":
            for e in self.manifest()["knees"]:
                if e["name"] == rest[1]:
                    return e
            raise ApiError(404, f"No demo knee {rest[1]!r}")
        if rest == ["build"] and method == "GET":
            return self.build_state
        if rest == ["sweep"] and method == "GET":
            return {"sweep": self.manifest()["sweep"]}
        if rest[:1] == ["server"] and rest[1:] in (["restart"], ["stop"]) and method == "POST":
            return self.server_control(rest[1])
        raise ApiError(404, f"No demo route {'/'.join(rest)}")

    @staticmethod
    def manifest():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    @staticmethod
    def built_job(name):
        for j in json.loads(JOBS.read_text()):
            if j["name"] == name:
                return j
        raise ApiError(404, f"No built job for {name!r}")

    def poses(self, entry):
        """The components' poses on the current plan, with the built knee's ML slide (the
        femoral fit) applied, so the built parts can follow a plan as it is adjusted."""
        import numpy as np
        s = entry.session
        # Copies: the slide below must not land in the plan itself (np.asarray of a float
        # array is the same array, and each call slid the plan's poses again).
        F = np.array(s.plan.components["femoral_component"], float)
        T = np.array(s.plan.components["tibial_component"], float)
        try:
            shift = self.built_job(s.case_id)["femoral_fit"]["ml_shift_mm"]
        except ApiError:
            shift = 0.0
        world = F[:3, 1] * shift
        F[:3, 3] += world
        T[:3, 3] += T[:3, :2] @ (T[:3, :2].T @ world)
        return {"femoral": F.tolist(), "tibial": T.tolist()}

    # -- the server itself (from the page) ------------------------------------------

    def server_control(self, action):
        """Answer first, then go: the page gets its reply before the socket closes. A
        restart starts the same command in a new process once the port is free; a build
        in progress is not cut short."""
        if self.build_state.get("state") == "running":
            raise ApiError(409, "A Fusion build is running; wait for it to finish.")

        self.leaving.set()

        def go():
            time.sleep(0.4)
            self.server.stop()
            if action == "restart":
                flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:],
                                 cwd=str(HERE), creationflags=flags)
            os._exit(0)

        threading.Thread(target=go, daemon=True).start()
        return {"state": "restarting" if action == "restart" else "stopped", "pid": os.getpid()}

    # -- the derivation -------------------------------------------------------

    def parametric(self, entry, body):
        from demo_cases import derive
        from jcurve import run as jcurve_run
        name = body.get("name") or entry.session.case_id
        s = entry.session
        t0 = time.time()
        job = derive(name, s.side, s.measurement, s.plan, s.sizing, s.library)
        built = self.built_job(name)
        job["label"], job["bones"] = built.get("label", name), built["bones"]
        natural = jcurve_run(job, femur=s.measurement.femur, comp=None, write=False)
        changed = {}
        for part, keys in COMPARE.items():
            for k in keys:
                a, b = mm(built[part][k]), mm(job[part][k])
                if abs(a - b) > 0.05:
                    changed[f"{part}.{k}"] = [round(a, 2), round(b, 2)]
        self.derived[entry.session_id] = job
        return {"job": job, "natural": natural, "changed": changed, "seconds": round(time.time() - t0, 2)}

    # -- Fusion ------------------------------------------------------------

    def build(self, entry, body):
        job = self.derived.get(entry.session_id)
        if job is None:
            raise ApiError(400, "Derive the parameters first (Implant).")
        with self.lock:
            if self.build_state["state"] == "running":
                raise ApiError(409, "A build is running.")
            self.build_state = {"state": "running", "name": job["name"], "started": time.time(), "log": []}
        threading.Thread(target=self._build, args=(job,), daemon=True).start()
        return self.build_state

    def _log(self, line):
        self.build_state["log"].append(f"{time.strftime('%H:%M:%S')}  {line}")

    def _build(self, job):
        from demo_build import knee_entry
        from fit_check import check
        from implant_pipeline import build
        from jcurve import run as jcurve_run
        name = job["name"]
        try:
            live = BUILDS / "jobs_live.json"
            live.write_text(json.dumps([job], indent=1), encoding="utf-8")
            self._log(f"Fusion: femoral, plate and insert for {name} (Patient Builds/{name})")
            build([name], live)
            rep = json.loads((BUILDS / name / "build.json").read_text())
            if rep.get("error"):
                raise RuntimeError(rep["error"])
            for part in ("femoral", "plate", "insert"):
                r = rep.get(part, {})
                self._log(f"{part}: v{r.get('version')} {'saved' if r.get('saved') else 'NOT saved'}"
                          f"{'; ' + ', '.join(h[0] for h in r.get('health') or []) if r.get('health') else ''}")
            jobs = [job if j["name"] == name else j for j in json.loads(JOBS.read_text())]
            JOBS.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
            self._log("J-curves and fit on the new parts")
            jcurve_run(job)
            fc = BUILDS / "fit_check.json"
            fits = json.loads(fc.read_text()) if fc.exists() else {}
            fits[name] = check(job)
            fc.write_text(json.dumps(fits, indent=1))
            man = self.manifest()
            man["knees"] = [knee_entry(job) if e["name"] == name else e for e in man["knees"]]
            MANIFEST.write_text(json.dumps(man, indent=1), encoding="utf-8")
            self._log("done")
            self.build_state["state"] = "done"
        except Exception as ex:          # noqa: BLE001 - reported to the page
            self._log(f"failed: {type(ex).__name__}: {ex}")
            self.build_state["trace"] = traceback.format_exc()[-2000:]
            self.build_state["state"] = "error"
        self.build_state["finished"] = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    os.chdir(HERE)                   # knee paths in the public copy are relative to eval/
    from eval_stage_b import LIBRARY
    manager = SessionManager(store=None, library=str(LIBRARY))
    server = httpd.PlannerServer(manager, port=args.port, web_root=APP)
    api = DemoApi(manager)
    api.server = server
    server.api = api
    server._server.RequestHandlerClass = httpd._make_handler(api, APP)
    print(f"Parametric TKA demo at {server.url}", flush=True)
    server.serve_forever()
    if api.leaving.is_set():         # restart or stop from the page: its thread ends the process
        time.sleep(30)


if __name__ == "__main__":
    main()
