"""Data for the parametric TKA demo app (private: patient-derived; served locally by
demo_server.py, which runs the planner session for the bones and the plan).

For each demo knee (patient_builds/jobs_demo.json): the Fusion-built femoral component,
plate and insert placed on the plan (binary glTF, scan frame, metres), and an entry with
each part's Fusion parameters, version and health, the checks measured on the parts, the
femoral fit, the tray rule and the J-curves. Also the femoral size sweep.
(demo_build_v1.py made the first, static page with bones cut here.)

  python demo_build.py            -> demo_app/data/, demo_app/vendor/, demo_app/fonts/
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from tka_planner.core.meshio import Mesh, read_stl  # noqa: E402
from tka_planner.geom.gltf import write_glb  # noqa: E402

APP = HERE / "demo_app"
DATA = APP / "data"
BUILDS = HERE / "patient_builds"
JOBS = BUILDS / "jobs_demo.json"
FONTS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts"
PARTS = (("femoral", "femoral", "Femoral component"),
         ("plate", "tibial", "Tibial plate"),
         ("insert", "tibial", "Insert"))


def transform(mesh, pose):
    P = np.asarray(pose)
    return Mesh(vertices=mesh.vertices @ P[:3, :3].T + P[:3, 3], faces=mesh.faces)


def glb(mesh, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_glb(mesh, name=path.stem))
    return path.name


def vendor():
    (APP / "vendor").mkdir(parents=True, exist_ok=True)
    for f in ("three.module.js", "OrbitControls.js", "LICENSE-three.txt"):
        shutil.copy(REPO / "web/vendor" / f, APP / "vendor" / f)
    shutil.copy(REPO / "web/glb.js", APP / "glb.js")
    (APP / "fonts").mkdir(exist_ok=True)
    for f in ("IBMPlexMono-Regular.ttf", "IBMPlexMono-SemiBold.ttf"):
        # Once: a running server holds the font open, and Windows won't overwrite it.
        if (FONTS / f).exists() and not (APP / "fonts" / f).exists():
            shutil.copy(FONTS / f, APP / "fonts" / f)


def knee_entry(job):
    """Write one knee's part meshes and return its entry."""
    name = job["name"]
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    build = BUILDS / name / "build.json"
    rep = json.loads(build.read_text()) if build.exists() else {}
    F, T = np.array(job["plan"]["femoral_pose"]), np.array(job["plan"]["tray_pose"])
    parts = []
    for part, group, label in PARTS:
        stl = BUILDS / name / f"{part}.stl"
        if stl.exists():
            pose = F if part == "femoral" else T
            parts.append({"id": part, "label": label, "set": group,
                          "file": f"data/{name}/" + glb(transform(read_stl(stl), pose), d / f"{part}.glb"),
                          "stamp": int(stl.stat().st_mtime)})
    checks = {}
    fs = BUILDS / name / "femoral.stl"
    if fs.exists():
        v = read_stl(fs).vertices
        ml = float(job["femoral"]["Femur_ML"].split()[0])
        checks["femoral width"] = [round(float(np.ptp(v[:, 1])), 2), f"ML - 1 = {ml - 1:.2f}"]
        checks["femoral distal thickness"] = [round(float(-v[:, 2].min()), 2),
                                              f"9 x box/52.1 = {job['plan']['distal_mm']:.2f}"]
    p = job["plan"]
    checks["tray coverage of the cut"] = [f"{p['tray_coverage']:.0%}", "no overhang, 1 mm inset"]
    checks["insert thickness"] = [round(p["insert_mm"], 2),
                                  "below 6: +%.1f mm tibial cut reaches 6" % p["extra_tibial_resection_for_min_mm"]
                                  if p["insert_below_min"] else "6 mm or more"]
    checks["femoral to tray tilt"] = [p["femoral_tray_tilt_deg"], "deg; the CAD insert assumes 0"]
    fc = BUILDS / "fit_check.json"
    fit = json.loads(fc.read_text()).get(name) if fc.exists() else None
    jc = BUILDS / name / "jcurve.json"
    scale = Path(job["bones"][0]).with_name("scale.json")
    return {
        "name": name, "label": job.get("label", name), "side": job["side"], "parts": parts,
        "folder": str(Path(job["bones"][0]).parent), "anatomy": job.get("anatomy"),
        "scale": json.loads(scale.read_text())["scale_ml_ap_si"] if scale.exists() else None,
        "poses": {"femoral": F.tolist(), "tibial": T.tolist()},
        "fusion": {part: {"folder": rep.get("folder"), "version": rep.get(part, {}).get("version"),
                          "health": rep.get(part, {}).get("health") or [], "parameters": job[part]}
                   for part in ("femoral", "plate", "insert")},
        "plan": {k: v for k, v in p.items() if not k.endswith("_pose")},
        "box": job.get("femoral_box"), "femoral_fit": job.get("femoral_fit"), "tray_rule": job.get("tray_rule"),
        "checks": checks, "fit": fit, "plan_controls": job.get("plan_controls") or {},
        "jcurve": json.loads(jc.read_text()) if jc.exists() else None,
    }


def main():
    vendor()
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {"knees": [], "sweep": []}
    for job in json.loads(JOBS.read_text()):
        manifest["knees"].append(knee_entry(job))
        print(job["name"], [p["id"] for p in manifest["knees"][-1]["parts"]], flush=True)
    sweep = HERE / "audit_author_v19"
    (DATA / "sweep").mkdir(exist_ok=True)
    for stl in sorted(sweep.glob("ml*.stl"), key=lambda s: int(s.stem[2:])):
        manifest["sweep"].append({"ml": int(stl.stem[2:]),
                                  "file": "data/sweep/" + glb(read_stl(stl), DATA / "sweep" / f"{stl.stem}.glb")})
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", DATA)


if __name__ == "__main__":
    main()
