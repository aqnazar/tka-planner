"""The three parts' Fusion parameters from one measured, planned knee.

Any plan, adjusted or not: the femoral box and width from the bone at the cuts, the tray
rule on the tibial cut, the insert from the height between the components. This is
``eval/demo_cases.derive`` moved into the planner without changing a number (the parity
test holds it to the fixtures the eval code wrote).
"""

from __future__ import annotations

import numpy as np

from .design import FEMORAL_INSET_MM, FEMORAL_WIDTH_ALLOWANCE_MM
from .facets import facet_extents, to_frame
from .job import femoral_job, make_job
from .tray import fit as tray_fit

__all__ = ["derive_parameters", "femoral_fit", "ml_slide"]


def femoral_fit(femur, F, box) -> dict:
    """Femur_ML and the ML slide that keep the component the inset inside the bone's cut
    surfaces (distal, posterior chamfer, posterior), from their common ML band. The
    component is ML - 1 wide (v19), centred on its origin."""
    cad = box["cad_parameters"]
    ext = facet_extents(to_frame(femur, F), cad["C_Distal_cut_surface"], cad["B_Posterior_chamfer_projection"],
                        cad["D_Anterior_chamfer_projection"])
    lo, hi = max(e[0] for e in ext.values()), min(e[1] for e in ext.values())
    width = hi - lo - 2 * FEMORAL_INSET_MM
    return {"femur_ml_mm": round(width + FEMORAL_WIDTH_ALLOWANCE_MM, 3), "ml_shift_mm": round((lo + hi) / 2, 3),
            "facet_ml_extent_mm": {k: [round(a, 2), round(b, 2)] for k, (a, b) in ext.items()},
            "band_mm": [round(lo, 2), round(hi, 2)]}


def ml_slide(F, T, shift_mm: float):
    """The femoral component slid along its own ML axis by ``shift_mm``, and the tray
    origin taking the same slide in its own plane: (F2, T2, the tray's slide in its
    axes). Every femoral cut plane contains the ML axis, so no resection changes."""
    F, T = np.asarray(F, float), np.asarray(T, float)
    world = F[:3, 1] * shift_mm
    t_shift = T[:3, :2].T @ world
    F2, T2 = F.copy(), T.copy()
    F2[:3, 3] += world
    T2[:3, 3] += T[:3, :2] @ t_shift
    return F2, T2, t_shift


def derive_parameters(measurement, plan, sizing, library, *, name: str | None = None,
                      side: str | None = None, box: dict | None = None) -> dict:
    """The build job for this plan: ``femoral``, ``plate`` and ``insert`` parameter maps
    (Fusion expressions), ``femoral_fit``, ``tray_rule``, ``anatomy``, ``plan`` and the
    femoral box. ``box`` is the femoral cut box on this plan when the caller has it.
    Raises ``ValueError`` when the box cannot be placed or no tray fits."""
    from tka_planner.core.implant_spec import traced_outline
    from tka_planner.pipeline import femoral_cut_box_case, implant_spec_case

    m, p = measurement, plan
    name = name or m.case_id
    side = side or str(m.side)
    if box is None:
        box = femoral_cut_box_case(m, p, library)
    if not box or not box.get("available"):
        raise ValueError(f"{name}: no femoral box: {(box or {}).get('reason')}")
    F = np.asarray(p.components["femoral_component"], float)
    T = np.asarray(p.components["tibial_component"], float)
    # Femoral width from the bone at the cuts, not the catalogue chart; the component
    # slides along its own ML axis onto the cut surfaces (no resection changes: every
    # cut plane contains the ML axis). The tray origin takes the same slide, so the
    # two stay in register (the planner's rule); the tray's own outline then centres
    # on the tibial cut through Tray_Offset.
    ffit = femoral_fit(m.femur, F, box)
    F2, T2, t_shift = ml_slide(F, T, ffit["ml_shift_mm"])
    f = femoral_job(name, side, F2, ffit["femur_ml_mm"], box)
    spec = implant_spec_case(m, p, sizing, library, with_cut_box=False)
    t = spec["tibial_component"]
    row = {"name": name, "side": side, "femur_ml_mm": f["femur_ml_mm"], "tibial": t["dimensions_mm"],
           "section_centre_mm": (np.asarray(t["diagnostics"]["section_centre_mm"]) - t_shift).tolist(),
           "outline_mm": (np.asarray(t["outline_mm"]) - t_shift).tolist(),
           "tray_pose": T2.tolist(), "femoral_pose": F2.tolist()}
    tray = tray_fit(row, row["section_centre_mm"])
    j = make_job(f, row, tray)
    j["femoral_box"] = {k: f[k] for k in ("flange_height_mm", "posterior_cut_x_mm", "anterior_cut_x_at_distal_mm")}
    j["femoral_fit"] = dict(ffit, chart_femur_ml_mm=round(float(sizing.implant_ml_mm), 3),
                            tray_origin_shift_mm=np.round(t_shift, 3).tolist())
    j["anatomy"] = {"tibial_cut_ml_mm": round(t["dimensions_mm"]["ml"], 1),
                    "tibial_cut_ap_mm": round(t["dimensions_mm"]["ap"], 1),
                    "tibial_ap_over_ml": round(t["dimensions_mm"]["ap"] / t["dimensions_mm"]["ml"], 3),
                    "nearest_catalogue_size": spec["catalogue_equivalent"]["nearest_size"]}
    shift = np.asarray(tray["shift_mm"])
    plate = np.asarray(traced_outline(tray["_mask"], tray["_xs"], tray["_ys"])) + shift
    j["tray_rule"] = {"c": tray["c"], "coverage": tray["coverage"], "bone_area_mm2": tray["bone_area_mm2"],
                      "tray_area_mm2": tray["tray_area_mm2"], "outline_mm": row["outline_mm"],
                      "plate_outline_mm": plate.round(2).tolist(), "shift_mm": tray["shift_mm"],
                      "notch_w": tray["params"]["Intercondylar_Notch_Width"]}
    return j
