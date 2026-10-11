"""Femoral rotation diagnosis (private: reads the cohort). Angles in the femoral cut
plane, positive = toward the patient's lateral side (external rotation of an anterior
direction), measured from the anterior perpendicular of the posterior condylar axis."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists()))); sys.path.insert(0, str(HERE))
from eval_stage_b import cases, LIBRARY
from tka_planner.pipeline import measure_case, plan_case

def inplane(v, n):
    v = v - np.dot(v, n) * n
    return v / np.linalg.norm(v)

def signed(a, b, n, lateral):
    """Angle from a to b about n, positive when b turns from a toward lateral."""
    ang = np.degrees(np.arctan2(np.dot(np.cross(a, b), n), np.dot(a, b)))
    sense = np.sign(np.dot(np.cross(a, lateral), n)) or 1.0
    return ang * sense

print(f"{'knee':8} {'frame method':38} {'twist':>6} {'frame_ant':>9} {'comp':>6}")
for name, femur, tibia, side in cases():
    try:
        m = measure_case(femur, tibia, side, case_id=name)
        plan, _ = plan_case(m, library=LIBRARY)
    except Exception as e:
        print(name, "failed", type(e).__name__); continue
    fr, lm = m.femoral_frame, m.landmarks
    pose = plan.components["femoral_component"]
    R = pose[:3, :3] / np.linalg.norm(pose[:3, :3], axis=0)
    n = R[:, 2]
    lat = inplane(fr.lateral, n)
    pm, pl = lm.require("femur.condyle_posterior_medial", "femur.condyle_posterior_lateral")
    pca = inplane(pl - pm, n)
    a_pca = inplane(np.cross(n, pca), n)
    if np.dot(a_pca, fr.x_anterior) < 0: a_pca = -a_pca
    tea_names = [k for k in ("femur.epicondyle_medial_sulcus", "femur.epicondyle_medial_prominence") if lm.available(k)]
    twist = float("nan")
    if tea_names and lm.available("femur.epicondyle_lateral"):
        em, el = lm.require(tea_names[0], "femur.epicondyle_lateral")
        tea = inplane(el - em, n)
        a_tea = inplane(np.cross(n, tea), n)
        if np.dot(a_tea, a_pca) < 0: a_tea = -a_tea
        twist = signed(a_pca, a_tea, n, lat)
    fa = signed(a_pca, inplane(fr.x_anterior, n), n, lat)
    comp = signed(a_pca, inplane(R[:, 0], n), n, lat)
    print(f"{name:8} {fr.method[:38]:38} {twist:+6.2f} {fa:+9.2f} {comp:+6.2f}  TEA={tea_names[:1]}")
