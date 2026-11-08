"""The implant routes: the derived parameters, the Fusion build, the motion on the parts.

  POST /api/sessions/<id>/parametric     the parameters the rules derive from the current
                                         plan, each with its rule, what differs from the
                                         knee's build, and the J-curves (bone, component)
  POST /api/sessions/<id>/build          build the derived parts in Fusion (in the
                                         background); answers at once when Fusion is not
                                         reachable on 127.0.0.1:27182
  GET  /api/build                        the build's progress and log
  POST /api/sessions/<id>/implant/refresh  put the knee's (new) build in the scene
  POST /api/sessions/<id>/kinematics     the flexion arc on the femoral component shown
                                         (medial pivot or J-curve rolling): poses, readings
  GET  /api/sessions/<id>/anatomy        the landmarks with what each is used for

From ``eval/demo_server.py``, which ran these beside the planner's routes; here they are
the planner's own. Every number is worked out here; the viewer draws it.
"""

from __future__ import annotations

import threading
import time
import traceback

import numpy as np

__all__ = ["ImplantRoutes", "MOTION_MODELS"]

# The Reduce motion: the planner's trajectory control, or the kinematic model on the
# femoral component shown (tka_planner.scene.kinematics).
MOTION_MODELS = ("planner", "published", "fixed")


class ImplantRoutes:
    """Mixed into :class:`~tka_planner.server.api.Api`."""

    def _implant_init(self):
        self.build_lock = threading.Lock()
        self.build_state = {"state": "idle", "log": []}
        self.fusion_config_path = None

    # -- routing -----------------------------------------------------------------

    def _implant_route(self, method, entry, tail, body):
        """The session's implant routes, or ``None`` when ``tail`` is not one."""
        if tail == ["parametric"] and method == "POST":
            return self.parametric(entry)
        if tail == ["build"] and method == "POST":
            return self.build(entry)
        if tail == ["implant", "refresh"] and method == "POST":
            from .api import _envelope
            return _envelope(entry, entry.session.refresh_implant())
        if tail == ["kinematics"] and method == "POST":
            return self.kinematics(entry, body)
        if tail == ["anatomy"] and method == "GET":
            return self.anatomy(entry)
        return None

    # -- the derivation -----------------------------------------------------------

    def parametric(self, entry) -> dict:
        from tka_planner.implant.explain import parameter_rows
        from tka_planner.implant.jcurve import run as jcurve

        from .api import ApiError

        s = entry.session
        start = time.time()
        job = s.derived
        if job is None:
            raise ApiError(400, f"The rules cannot size this plan: {s.derived_reason}")
        stale = s.stale_parameters() if s.build_record is not None else []
        changed = {f"{r['part']}.{r['name']}": [r["built"], r["now"]] for r in stale}
        rows = parameter_rows(job)
        for row in rows:
            row["changed"] = changed.get(f"{row['part']}.{row['name']}")
        try:
            curves = jcurve(job, s.measurement.femur, self._component_in(entry, job))
            curves_note = None
        except (ValueError, IndexError, KeyError) as error:
            curves, curves_note = None, f"The J-curves could not be drawn: {error}"
        build = s.build_record
        return {
            "job": job, "parameters": rows, "changed": changed, "stale": stale,
            "jcurve": curves, "jcurve_note": curves_note,
            "tray_rule": job["tray_rule"], "side": str(s.side),
            "implant": s.implant_state(),
            "build": None if build is None else {
                "versions": build.versions, "health": build.health,
                "built_at": build.built_at, "parameters": build.parameters},
            "insert_minimum": {k: s.plan.diagnostics.get(k) for k in (
                "insert_minimum_mm", "tibial_resection_auto_mm", "insert_of_record_mm",
                "insert_minimum_clamped")},
            "seconds": round(time.time() - start, 2),
        }

    def _component_in(self, entry, job):
        """The femoral component shown, in the derivation's femoral CAD frame (its
        slid pose): the built part or the stand-in, as the scene places it."""
        s = entry.session
        node = s.scene.nodes.get("femoral_component") if s.scene is not None else None
        if node is None:
            return None
        from tka_planner.core.meshio import Mesh

        mesh = s.scene.meshes[node.mesh_id]
        F = np.asarray(job["plan"]["femoral_pose"], float)
        M = np.linalg.inv(F) @ np.asarray(node.rest, float)
        return Mesh(vertices=mesh.vertices @ M[:3, :3].T + M[:3, 3], faces=mesh.faces)

    # -- Fusion -----------------------------------------------------------------

    def build(self, entry) -> dict:
        from tka_planner.fusion import FusionConfig, FusionUnavailable, ping

        from .api import ApiError

        s = entry.session
        if self.manager.builds is None:
            raise ApiError(400, "No builds folder (start the server with a store or --builds).")
        job = s.derived
        if job is None:
            raise ApiError(400, f"The rules cannot size this plan: {s.derived_reason}")
        config = FusionConfig.load(self.fusion_config_path)
        try:
            ping(config.url)
        except FusionUnavailable as error:
            raise ApiError(503, str(error)) from None
        try:
            config.require()
        except FileNotFoundError as error:
            raise ApiError(400, str(error)) from None
        with self.build_lock:
            if self.build_state.get("state") == "running":
                raise ApiError(409, "A Fusion build is running; wait for it to finish.")
            self.build_state = {"state": "running", "name": job["name"],
                                "session_id": entry.session_id, "started": time.time(),
                                "log": []}
        folder = self.manager.builds.folder_for(s.case_id, s.side)
        threading.Thread(target=self._build, args=(dict(job), folder, config),
                         daemon=True).start()
        return self.build_state

    def _log(self, line):
        self.build_state["log"].append(f"{time.strftime('%H:%M:%S')}  {line}")

    def _build(self, job, folder, config):
        from tka_planner.fusion import build_case

        try:
            result = build_case(job, folder, config, log=self._log)
            self.build_state["result"] = result.to_dict()
            self.build_state["state"] = "done" if result.ok else "error"
        except Exception as error:      # noqa: BLE001 - reported to the page
            self._log(f"failed: {type(error).__name__}: {error}")
            self.build_state["trace"] = traceback.format_exc()[-2000:]
            self.build_state["state"] = "error"
        self.build_state["finished"] = time.time()

    def build_progress(self) -> dict:
        return self.build_state

    # -- motion on the parts -----------------------------------------------------

    def knee_motion(self, entry):
        """The kinematic model of this knee on the femoral component shown, at the
        current plan's poses, or ``None`` when the rules cannot size the plan."""
        from tka_planner.implant.jcurve import run as jcurve
        from tka_planner.scene.kinematics import Knee

        s = entry.session
        job = s.derived
        if job is None or s.scene is None:
            return None
        key = (id(s.plan), s.scene.nodes["femoral_component"].mesh_id)
        cached = getattr(s, "_knee_motion", None)
        if cached is None or cached[0] != key:
            comp = self._component_in(entry, job)
            curves = jcurve(job, s.measurement.femur)
            ys = {name: c["y_mm"] for name, c in curves["condyles"].items()}
            T = np.asarray(job["plan"]["tray_pose"], float)
            F = np.asarray(job["plan"]["femoral_pose"], float)
            cached = (key, Knee(comp.vertices, ys, F, T))
            s._knee_motion = cached
        return cached[1]

    def _stress(self, entry, knee):
        """The trial's stress, drawer and distraction at rest, about a pivot between the
        two condyles' contacts."""
        from tka_planner.scene import motion

        s = entry.session
        c = np.mean([knee.rest[n] for n in knee.rest], axis=0)
        pivot = motion.pivot_frame(knee.F[:3, :3] @ c + knee.F[:3, 3], knee.F[:3, 1])
        t = s.trial
        return motion.trial_pose(pivot, flexion_deg=0.0, varus_valgus_deg=t.varus_valgus_deg,
                                 drawer_ap_mm=t.drawer_ap_mm, distraction_mm=t.distraction_mm,
                                 distal=-s.measurement.tibial_frame.z_proximal)

    def kinematic_trial(self, entry, out, model):
        """After the planner's trial route: with a kinematic model chosen, flexion
        follows it on the femoral component shown."""
        from tka_planner.scene.model import TIBIAL

        s = entry.session
        if model:
            if model not in MOTION_MODELS:
                raise ValueError(f"motion model must be one of {', '.join(MOTION_MODELS)}")
            s.motion_model = model
        model = getattr(s, "motion_model", "planner")
        if model == "planner":
            return out
        knee = self.knee_motion(entry)
        if knee is None:
            out["kinematics_note"] = f"No kinematic model: {s.derived_reason}"
            return out
        P, info = knee.pose(s.trial.flexion_deg, model)
        pose = P @ self._stress(entry, knee)
        s.scene.set_pose(TIBIAL, pose)
        out.setdefault("delta", {}).setdefault("poses", {})["TibialSet"] = pose.tolist()
        out["kinematics"] = dict(info, model=model)
        return out

    def kinematics(self, entry, body) -> dict:
        from tka_planner.scene.kinematics import PUBLISHED

        from .api import ApiError

        s = entry.session
        model = body.get("model") or getattr(s, "motion_model", "planner")
        if model == "planner":
            model = "published"
        if model not in ("published", "fixed"):
            raise ApiError(400, "model must be published or fixed")
        try:
            knee = self.knee_motion(entry)
        except ValueError as error:
            raise ApiError(400, f"No kinematic model on these parts: {error}") from None
        if knee is None:
            raise ApiError(400, f"No kinematic model: {s.derived_reason}")
        steps, top = max(2, int(body.get("steps", 60))), float(body.get("max_deg", 120.0))
        stress = self._stress(entry, knee)
        frames = []
        for i in range(steps + 1):
            P, info = knee.pose(top * i / steps, model)
            frames.append(dict(info, pose=(P @ stress).tolist()))
        return {"frames": frames, "model": model, "published": PUBLISHED,
                "parts": s.implant_state()["label"]}

    # -- the anatomy -------------------------------------------------------------

    def anatomy(self, entry) -> dict:
        from tka_planner.implant.explain import landmark_label

        s = entry.session
        m = s.measurement
        landmarks = []
        for landmark in m.landmarks:
            if not landmark.is_usable:
                continue
            name, role = landmark_label(landmark.id)
            landmarks.append({"id": landmark.id, "name": name, "role": role,
                              "status": landmark.status.value,
                              "bone": landmark.id.split(".")[0],
                              "position_mm": [round(float(v), 2) for v in landmark.position_mm]})
        job = s.derived
        facts = {
            "femur_in_scan_mm": round(float(np.ptp(m.femur.vertices @ m.femoral_frame.z_proximal)), 0),
            "tibia_in_scan_mm": round(float(np.ptp(m.tibia.vertices @ m.tibial_frame.z_proximal)), 0),
            "femoral_ml_mm": round(float(m.femoral_measure.ml_mm), 1),
            "nearest_catalogue_size": s.sizing.nearest_discrete_size,
        }
        if job is not None:
            facts.update({k: job["anatomy"][k] for k in ("tibial_cut_ml_mm", "tibial_cut_ap_mm")})
        return {"landmarks": landmarks, "facts": facts}
