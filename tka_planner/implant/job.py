"""One knee's build job: the three parts' Fusion parameters, as the build sets them.

From ``eval/fusion_driver.job`` (the femoral box) and ``eval/implant_pipeline.make_job``
(the three parameter sets and the insert thickness), unchanged.

Insert thickness (author 2026-10-09): the vertical distance between the components.
In the insert document the plate's insert floor is at Z = Plate_Thickness and the
femoral origin at Z = floor + Insert_Thickness + Distal_Cut, so
``Insert_Thickness = h - floor - Distal_cut_depth``, h being the femoral origin's height
above the tibial cut along the tray normal.
"""

from __future__ import annotations

import numpy as np

from .design import (
    BOX_PARAMETERS,
    DESIGN_BOX_MM,
    DISTAL_CUT_DESIGN_MM,
    INSERT_FLAG_MM,
    PLATE_FLOOR_MM,
)

__all__ = ["femoral_job", "make_job", "distal_cut_mm", "insert_thickness_mm"]


def femoral_job(name, side, pose, femur_ml, box) -> dict:
    """The femoral part's inputs: Femur_ML and the box lengths at a pose."""
    return {
        "name": name,
        "side": side,
        "pose": np.asarray(pose, dtype=float).tolist(),
        "femur_ml_mm": round(femur_ml, 3),
        "parameters_mm": {k: box["cad_parameters"][k] for k in BOX_PARAMETERS},
        "cad_origin_x_mm": box["cad_origin_x_mm"],
        "posterior_cut_x_mm": box["posterior_cut_x_mm"],
        "anterior_cut_x_at_distal_mm": box["anterior_cut_x_at_distal_mm"],
        "flange_height_mm": box["flange_height_mm"],
    }


def distal_cut_mm(box_parameters: dict) -> float:
    """Distal_cut_depth: the design's 9 mm scaled by the box depth over the design's."""
    box = box_parameters
    return DISTAL_CUT_DESIGN_MM * (box["C_Distal_cut_surface"] + box["B_Posterior_chamfer_projection"]
                                   + box["D_Anterior_chamfer_projection"]) / DESIGN_BOX_MM


def insert_thickness_mm(femoral_pose, tray_pose, box_parameters: dict) -> float:
    """The insert the parts close with: h - plate floor - distal cut."""
    F, T = np.asarray(femoral_pose, float), np.asarray(tray_pose, float)
    h = float(T[:3, 2] @ (F[:3, 3] - T[:3, 3]))
    return h - PLATE_FLOOR_MM - distal_cut_mm(box_parameters)


def make_job(f, r, t) -> dict:
    """One knee's three parameter sets. f: femoral_job, r: the tibial inputs row
    (name, side, femoral_pose, tray_pose), t: the tray rule fit (outline shifted)."""
    ml = f["femur_ml_mm"]
    F, T = np.array(r["femoral_pose"]), np.array(r["tray_pose"])
    n = T[:3, 2]
    h = float(n @ (F[:3, 3] - T[:3, 3]))
    distal = distal_cut_mm(f["parameters_mm"])
    thickness = h - PLATE_FLOOR_MM - distal
    plate = {k: f"{v:.4f} mm" for k, v in t["params"].items()}
    plate["Tray_Offset_X"] = f"{t['shift_mm'][0]:.4f} mm"
    plate["Tray_Offset_Y"] = f"{t['shift_mm'][1]:.4f} mm"
    return {
        "name": r["name"], "side": r["side"],
        "femoral": dict({"Femur_ML": f"{ml:.4f} mm"},
                        **{k: f"{v:.4f} mm" for k, v in f["parameters_mm"].items()}),
        "plate": plate,
        "insert": {"Insert_Thickness": f"{thickness:.4f} mm", "Distal_Cut": f"{distal:.4f} mm",
                   "Femur_ML": f"{ml:.4f} mm",
                   "Tibial_Plateau_ML": plate["Tibial_Plateau_ML"],
                   "Tibial_Plateau_AP": plate["Tibial_Plateau_AP"]},
        "plan": {"h_mm": round(h, 3), "insert_mm": round(thickness, 3), "distal_mm": round(distal, 3),
                 "insert_below_min": thickness < INSERT_FLAG_MM,
                 "extra_tibial_resection_for_min_mm": round(max(0.0, INSERT_FLAG_MM - thickness), 2),
                 "femoral_tray_tilt_deg": round(float(np.degrees(np.arccos(np.clip(F[:3, 2] @ n, -1, 1)))), 2),
                 "tray_coverage": t["coverage"],
                 "femoral_pose": r["femoral_pose"], "tray_pose": r["tray_pose"]},
    }
