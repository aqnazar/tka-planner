"""Stage (b) evaluation: the femoral cut box on the reference femur and the cohort.

Private: it reads the patient dataset, whose folder names carry identities. Knees are
printed by case id only.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))

from tka_planner.core.femoral_box import femoral_cut_box  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402
from tka_planner.pipeline import (  # noqa: E402
    box_design, femoral_cut_box_case, measure_case, plan_case,
)

# As the launcher: TKA_IMPLANT_LIBRARY, else data/implants, else the author's folder.
LIBRARY = Path(os.environ.get("TKA_IMPLANT_LIBRARY") or next(
    (p for p in (REPO / "data" / "implants",) if p.exists()), REPO.parent / "BP" / "Scaled_STLs"))
CASE_IDS = {"Patient_001": "P007", "Patient_002": "P009", "Patient_003": "P029",
            "Patient_004": "P037", "Patient_005": "P047"}
V57 = {"posterior_cut_x_mm": -23.9, "anterior_cut_x_at_distal_mm": 25.53,
       "flange_height_mm": 42.6, "cad_origin_x_mm": 0.0,
       "distal_face_mm": 26.6, "posterior_chamfer_mm": 10.6, "anterior_chamfer_mm": 15.0}


def cases():
    yield "REF_KJ2", REPO / "data/cases/REF_KJ2/FD1Right.stl", \
        REPO / "data/cases/REF_KJ2/TD1Right.stl", "right"
    root = REPO / "Patient Dataset"
    for folder, case_id in CASE_IDS.items():
        for side in ("Left", "Right"):
            knee = root / folder / side
            if not knee.exists():
                knee = next(p for p in (root / folder).iterdir()
                            if p.is_dir() and p.name.lower().startswith(side.lower()))
            femur = next(p for p in knee.glob("*.stl") if p.name.upper().startswith("FD"))
            tibia = next(p for p in knee.glob("*.stl") if p.name.upper().startswith("TD"))
            yield f"{case_id}{side[0]}", femur, tibia, side.lower()


def main():
    out = []
    design = box_design(LIBRARY)
    Df = np.load(Path(__file__).with_name("design_poses_kj.npz"))["Df"]
    ref = femoral_cut_box(read_stl(REPO / "data/cases/REF_KJ2/FD1Right.stl"), Df,
                          medial_sign=1.0, design=design).to_dict()
    out.append("## Reference femur at the author's design pose, against v57\n")
    out.append("| Quantity | Rule | v57 | Rule − v57 |\n|---|---|---|---|")
    for key, value in V57.items():
        out.append(f"| {key} | {ref[key]:.2f} | {value:.2f} | {ref[key] - value:+.2f} |")
    r = ref["resections_mm"]
    out.append(f"\nResections at the design pose: posterior medial {r['posterior_medial']:.2f}, "
               f"lateral {r['posterior_lateral']:.2f}, anterior {r['anterior']:.2f}; "
               f"PF lowering {ref['patellofemoral_lowering_mm']:+.2f}; notch "
               f"{ref['anterior_notch_mm']:.2f}; shaft slope {ref['shaft_slope_deg']:.2f} deg.\n")

    out.append("## Planner resection table (mm), mechanical alignment, default controls\n")
    out.append("| Knee | Distal M | Distal L | Post M | Post L | Anterior | PF lowering | "
               "Flange h | Notch | Box AP | Origin X | Box time s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, femur, tibia, side in cases():
        try:
            m = measure_case(femur, tibia, side, case_id=name)
            plan, _ = plan_case(m, library=LIBRARY)
            start = time.perf_counter()
            box = femoral_cut_box_case(m, plan, LIBRARY)
            took = time.perf_counter() - start
        except Exception as error:  # noqa: BLE001 - report every knee
            out.append(f"| {name} | failed: {type(error).__name__}: {error} |")
            continue
        cut = plan.resections["femoral_distal"]
        if not box["available"]:
            out.append(f"| {name} | {cut.medial_depth_mm:.1f} | {cut.lateral_depth_mm:.1f} | "
                       f"not placed: {box['reason']} |")
            continue
        r = box["resections_mm"]
        out.append(
            f"| {name} | {cut.medial_depth_mm:.1f} | {cut.lateral_depth_mm:.1f} | "
            f"{r['posterior_medial']:.1f} | {r['posterior_lateral']:.1f} | "
            f"{r['anterior']:.1f} | {box['patellofemoral_lowering_mm']:+.1f} | "
            f"{box['flange_height_mm']:.0f} | {box['anterior_notch_mm']:.1f} | "
            f"{box['box_ap_mm']:.1f} | {box['cad_origin_x_mm']:+.1f} | {took:.2f} |")
        print(name, "done", flush=True)
    text = "\n".join(out) + "\n"
    Path(__file__).with_name("stage_b_results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
