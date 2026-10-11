"""Left vs right: the component's rotation from the PCA about the scanner's superior axis
(LPS +Z, seen from above), and in anatomical terms. Private: reads the cohort."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists()))); sys.path.insert(0, str(HERE))
from eval_stage_b import cases, LIBRARY
from tka_planner.pipeline import measure_case, plan_case

UP = np.array([0.0, 0.0, 1.0])   # LPS superior
def flat(v):
    v = v - np.dot(v, UP) * UP; return v / np.linalg.norm(v)
print(f"{'knee':8} {'about scanner +Z (CCW seen from above)':>40} {'anatomical (+ = external)':>28}")
for name, femur, tibia, side in cases():
    try:
        m = measure_case(femur, tibia, side, case_id=name); plan, _ = plan_case(m, library=LIBRARY)
    except Exception as e:
        print(name, "failed"); continue
    pose = plan.components["femoral_component"]
    R = pose[:3, :3] / np.linalg.norm(pose[:3, :3], axis=0)
    pm, pl = m.landmarks.require("femur.condyle_posterior_medial", "femur.condyle_posterior_lateral")
    sq = flat(np.cross(UP, flat(pl - pm)))
    if np.dot(sq, m.femoral_frame.x_anterior) < 0: sq = -sq
    ant = flat(R[:, 0])
    world = np.degrees(np.arctan2(np.dot(np.cross(sq, ant), UP), np.dot(sq, ant)))
    lateral = flat(m.femoral_frame.lateral); lateral = lateral - np.dot(lateral, sq) * sq; lateral /= np.linalg.norm(lateral)
    anat = np.degrees(np.arctan2(np.dot(ant, lateral), np.dot(ant, sq)))
    print(f"{name:8} {world:+40.2f} {anat:+28.2f}")
