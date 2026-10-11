"""Tray and insert inputs per knee from the planner: the tibial cut section in the tray
frame, the solved insert, the femoral ML. Private (patient-derived).

  python tibial_inputs.py   -> tray_driver/inputs.json and a table
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
OUT = HERE / "tray_driver"


def main():
    from eval_stage_b import LIBRARY, cases
    from tka_planner.pipeline import implant_spec_case, measure_case, plan_case
    OUT.mkdir(exist_ok=True)
    femoral_jobs = {j["name"]: j for j in json.loads((HERE / "v57_driver/jobs.json").read_text())}
    rows = []
    for name, femur, tibia, side in cases():
        try:
            m = measure_case(femur, tibia, side, case_id=name)
            plan, sizing = plan_case(m, library=LIBRARY)
            spec = implant_spec_case(m, plan, sizing, LIBRARY, with_cut_box=False)
        except Exception as error:  # noqa: BLE001
            print(name, "failed:", type(error).__name__, error, flush=True)
            continue
        t = spec["tibial_component"]
        rows.append({
            "name": name, "side": side,
            "femur_ml_mm": femoral_jobs.get(name, {}).get("femur_ml_mm"),
            "tibial": t["dimensions_mm"],
            "section_centre_mm": t["diagnostics"]["section_centre_mm"],
            "outline_mm": t["outline_mm"],
            "insert_thickness_mm": spec["insert"]["thickness_mm"],
            "insert_solved": spec["insert"]["solved"],
            "tray_pose": np.asarray(plan.components["tibial_component"]).tolist(),
            "femoral_pose": np.asarray(plan.components["femoral_component"]).tolist(),
            "nearest_size": spec["catalogue_equivalent"]["nearest_size"],
        })
        d = t["dimensions_mm"]
        c = t["diagnostics"]["section_centre_mm"]
        print(f"{name:7s} {side:5s} size {rows[-1]['nearest_size']:3s} femML {rows[-1]['femur_ml_mm']}  "
              f"tibML {d['ml']:5.1f} AP {d['ap']:5.1f} medAP {d['medial_ap']:5.1f} latAP {d['lateral_ap']:5.1f} "
              f"PCL w {d['pcl_notch_width']:4.1f} d {d['pcl_notch_depth']:4.1f} tray {d.get('tray_thickness')} "
              f"insert {spec['insert']['thickness_mm']}  section centre {np.round(c, 1).tolist()}", flush=True)
    (OUT / "inputs.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
