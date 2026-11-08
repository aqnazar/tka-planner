"""The demo's three knees: the reference knee and two synthetic extremes made from it.

Author 2026-10-09: one baseline plus a too-small and a too-big knee, the extremes made by
scaling the reference bones X, Y, Z independently, kept anatomically plausible. The scale
is applied in the reference plan's tibial frame (X medio-lateral, Y antero-posterior,
Z along the tibia) about the joint (midway between the femoral and tibial origins), to
femur and tibia together, so the joint stays congruent. Each knee then goes through the
whole pipeline as a patient would: measure, plan, femoral box, tray rule, insert, Fusion.

  python demo_cases.py bones    # write the scaled bones to demo_cases/<name>/
  python demo_cases.py jobs [knee ...]   # plan them, write patient_builds/jobs_demo.json
                                         # (named knees only: the others' jobs are kept)
  python demo_cases.py build [knee ...]  # build them in Fusion (Patient Builds/<name>/)

The demo also carries five cohort knees as they come from the scanner (knee-only CT,
about 150 mm of each bone), by case id only (author 2026-10-11).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from tka_planner.core.meshio import Mesh, read_stl  # noqa: E402
from tka_planner.geom.mesh import write_stl  # noqa: E402

FEMORAL_INSET = 1.0   # mm inside the bone's cut surfaces, each side (as the tray)
# The patients' default plans leave 2.4 to 5.9 mm of insert, and the insert design no
# longer builds below about 7 (its Condylar Profile fails to solve, 2026-10-11). Their
# plans take the planner's own remedy, more tibial resection, to this thickness; the
# app opens each knee on the plan it was built for (plan_controls).
INSERT_TARGET_MM = 8.0
BONES = HERE / "demo_cases"
JOBS = HERE / "patient_builds" / "jobs_demo.json"
REF = ("REF_KJ2", REPO / "data/cases/REF_KJ2/FD1Right.stl", REPO / "data/cases/REF_KJ2/TD1Right.stl", "right")
# (ML, AP, SI) scale of the reference knee. Small: a small knee with the narrow,
# AP-long proportions of a small female knee. Large: a broad, ML-wide large male knee.
SCALES = {
    "DEMO_SMALL": {"scale": (0.76, 0.82, 0.88), "label": "Small (reference scaled 0.76 ML, 0.82 AP, 0.88 SI)"},
    "DEMO_LARGE": {"scale": (1.12, 1.05, 1.06), "label": "Large (reference scaled 1.12 ML, 1.05 AP, 1.06 SI)"},
}
PATIENTS = ("P007L", "P007R", "P009L", "P029R", "P047R")
LABELS = dict({"REF_KJ2": "Baseline (reference knee)"}, **{k: v["label"] for k, v in SCALES.items()},
              **{k: f"Patient {k[:4]}, {'left' if k[-1] == 'L' else 'right'} knee" for k in PATIENTS})


def plan(name, femur, tibia, side, controls=None):
    from eval_stage_b import LIBRARY
    from tka_planner.core.planning import Adjustments
    from tka_planner.pipeline import femoral_cut_box_case, implant_spec_case, measure_case, plan_case
    m = measure_case(femur, tibia, side, case_id=name)
    p, sizing = plan_case(m, library=LIBRARY, adjustments=Adjustments(**(controls or {})))
    return m, p, sizing, LIBRARY, femoral_cut_box_case, implant_spec_case


def bones():
    name, femur, tibia, side = REF
    _, p, *_ = plan(name, femur, tibia, side)
    F, T = np.asarray(p.components["femoral_component"]), np.asarray(p.components["tibial_component"])
    R = T[:3, :3]
    centre = (F[:3, 3] + T[:3, 3]) / 2
    for case, spec in SCALES.items():
        A = R @ np.diag(spec["scale"]) @ R.T
        for src, dst in ((femur, "FD.stl"), (tibia, "TD.stl")):
            m = read_stl(src)
            write_stl(Mesh(vertices=(m.vertices - centre) @ A.T + centre, faces=m.faces), BONES / case / dst)
        (BONES / case / "scale.json").write_text(json.dumps(
            {"from": name, "scale_ml_ap_si": spec["scale"], "frame": R.tolist(), "centre_mm": centre.tolist()}, indent=1))
        print("wrote", BONES / case)


def cases():
    from eval_stage_b import cases as cohort
    yield REF
    for case in SCALES:
        yield case, BONES / case / "FD.stl", BONES / case / "TD.stl", REF[3]
    for case in cohort():
        if case[0] in PATIENTS:
            yield case


def femoral_fit(femur, F, box):
    """Femur_ML and the ML slide on the cut surfaces: the planner's rule
    (tka_planner.implant.femoral_fit)."""
    from tka_planner.implant import femoral_fit as rule
    return rule(femur, F, box)


def derive(name, side, m, p, sizing, library):
    """The three parts' Fusion parameters from one measured, planned knee. The rules
    live in the planner now (tka_planner.implant.derive_parameters); this is a call."""
    from tka_planner.implant import derive_parameters
    return derive_parameters(m, p, sizing, library, name=name, side=side)


def jobs(names=()):
    from implant_pipeline import print_jobs
    out = [] if not names else [j for j in json.loads(JOBS.read_text()) if j["name"] not in names]
    for name, femur, tibia, side in cases():
        if names and name not in names:
            continue
        m, p, sizing, library, *_ = plan(name, femur, tibia, side)
        j = derive(name, side, m, p, sizing, library)
        controls = {}
        if name in PATIENTS and j["plan"]["insert_mm"] < INSERT_TARGET_MM:
            extra = np.ceil((INSERT_TARGET_MM - j["plan"]["insert_mm"]) * 2) / 2
            controls = {"tibial_resection_delta_mm": float(extra)}
            m, p, sizing, library, *_ = plan(name, femur, tibia, side, controls)
            j = derive(name, side, m, p, sizing, library)
        j["plan_controls"] = controls
        j["label"] = LABELS[name]
        j["bones"] = [str(femur), str(tibia)]
        out.append(j)
        print(name, "planned:", j["anatomy"], "femoral fit", j["femoral_fit"], flush=True)
    JOBS.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_jobs(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bones":
        bones()
    elif cmd == "jobs":
        jobs(sys.argv[2:])
    else:
        from implant_pipeline import build
        build(sys.argv[2:] or [c[0] for c in cases()], JOBS)
