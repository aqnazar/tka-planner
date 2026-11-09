"""The words beside the numbers: what each derived parameter comes from, and what each
landmark is used for. Text only; served so the viewer shows it and computes nothing."""

from __future__ import annotations

from .design import DESIGN_BOX_MM

__all__ = ["parameter_rows", "LANDMARK_ROLES", "landmark_label"]

PART_LABELS = {"femoral": "Femoral component", "plate": "Tibial plate", "insert": "Insert"}


def _mm(value):
    try:
        return float(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def parameter_rows(job: dict) -> list:
    """Each derived Fusion parameter with the rule that set it: rows ``{part, name,
    value, rule}`` in the order the Implant view lists them."""
    ff, plan, fem, plate, ins = (job.get("femoral_fit") or {}), job.get("plan") or {}, \
        job["femoral"], job["plate"], job["insert"]
    band = ff.get("band_mm") or [0.0, 0.0]
    box = sum(_mm(fem[k]) for k in ("C_Distal_cut_surface", "B_Posterior_chamfer_projection",
                                    "D_Anterior_chamfer_projection"))
    rows = [
        ("femoral", "Femur_ML", fem["Femur_ML"],
         f"bone at the cuts {band[1] - band[0]:.1f} mm - 2 x 1 mm inset + 1 "
         f"(the size chart said {ff.get('chart_femur_ml_mm', 0):.1f})"),
        ("femoral", "C_Distal_cut_surface", fem["C_Distal_cut_surface"], "planned cut box on the bone"),
        ("femoral", "B_Posterior_chamfer_projection", fem["B_Posterior_chamfer_projection"],
         "planned cut box on the bone"),
        ("femoral", "D_Anterior_chamfer_projection", fem["D_Anterior_chamfer_projection"],
         "planned cut box on the bone"),
        ("femoral", "Sagittal_Scale", f"{box / DESIGN_BOX_MM:.4f}",
         "(C + B + D) / 52.1: the J-curve radii and thicknesses (design v20)"),
        ("femoral", "ML slide", f"{ff.get('ml_shift_mm', 0.0):.3f} mm",
         "the component centred on the distal, posterior chamfer and posterior cut surfaces"),
        ("plate", "Tibial_Plateau_ML", plate["Tibial_Plateau_ML"],
         "largest two-circle outline 1 mm inside the tibial cut, no overhang"),
        ("plate", "Tibial_Plateau_AP", plate["Tibial_Plateau_AP"], "condyle diameter + 4 (design)"),
        ("plate", "Left_Condyle_Diameter", plate["Left_Condyle_Diameter"],
         "the +X (patient-left) compartment, on its own"),
        ("plate", "Right_Condyle_Diameter", plate["Right_Condyle_Diameter"],
         "the -X (patient-right) compartment, on its own"),
        ("plate", "Intercondylar_Notch_Width", plate["Intercondylar_Notch_Width"],
         "the bone's PCL notch + 2 x 1 mm inset, never less than the design's"),
        ("plate", "Internal_Notch_Offset", plate["Internal_Notch_Offset"],
         "the bone's PCL notch depth + 1 mm, never less than the design's 6"),
        ("plate", "d11", plate["d11"], "notch-corner fillet 8 x AP / 44, capped by the edge beside the notch"),
        ("plate", "Tray_Offset_X", plate["Tray_Offset_X"], "the outline centred on the cut section"),
        ("plate", "Tray_Offset_Y", plate["Tray_Offset_Y"], "the outline centred on the cut section"),
        ("insert", "Insert_Thickness", ins["Insert_Thickness"],
         f"femoral origin {plan.get('h_mm', 0):.1f} mm above the tibial cut - 5 plate - "
         f"{plan.get('distal_mm', 0):.1f} distal"),
        ("insert", "Distal_Cut", ins["Distal_Cut"], "9 mm x (C + B + D) / 52.1"),
        ("insert", "Outline", "plate pocket", "this knee's plate section, - 0.6 mm clearance"),
        ("insert", "Articular surface", "femoral J-curve", "swept from this knee's femoral component"),
    ]
    return [{"part": part, "part_label": PART_LABELS[part], "name": name, "value": value,
             "rule": rule} for part, name, value, rule in rows]


# Landmark -> (name, what the plan uses it for). From the demo app's Anatomy step.
LANDMARK_ROLES = {
    "femur.condyle_distal_medial": ("Distal medial condyle", "depth of the distal cut"),
    "femur.condyle_distal_lateral": ("Distal lateral condyle", "depth of the distal cut"),
    "femur.condyle_posterior_medial": ("Posterior medial condyle", "posterior cut, femoral rotation"),
    "femur.condyle_posterior_lateral": ("Posterior lateral condyle", "posterior cut, femoral rotation"),
    "femur.epicondyle_lateral": ("Lateral epicondyle", "transepicondylar axis"),
    "femur.epicondyle_medial_prominence": ("Medial epicondyle", "transepicondylar axis"),
    "femur.epicondyle_medial_sulcus": ("Medial epicondylar sulcus", "surgical epicondylar axis"),
    "femur.notch_centre": ("Intercondylar notch", "femoral mechanical axis, distal end"),
    "femur.canal_centre_distal": ("Femoral canal, distal", "anatomical axis"),
    "femur.canal_centre_proximal": ("Femoral canal, proximal", "anatomical axis"),
    "femur.head_centre": ("Femoral head centre", "femoral mechanical axis, proximal end"),
    "tibia.plateau_medial_lowest": ("Medial plateau, lowest", "depth of the tibial cut"),
    "tibia.plateau_lateral_lowest": ("Lateral plateau, lowest", "depth of the tibial cut"),
    "tibia.spine_medial": ("Medial tibial spine", "tibial centre"),
    "tibia.spine_lateral": ("Lateral tibial spine", "tibial centre"),
    "tibia.pcl_insertion_midpoint": ("PCL insertion", "tibial rotation (AP axis)"),
    "tibia.tubercle_medial_third": ("Tubercle, medial third", "tibial rotation (AP axis)"),
    "tibia.tubercle_patellar_tendon_medial_border": ("Patellar tendon, medial border", "tibial rotation"),
    "tibia.plateau_medial_anterior": ("Medial plateau, anterior", "posterior slope"),
    "tibia.plateau_medial_posterior": ("Medial plateau, posterior", "posterior slope"),
    "tibia.canal_centre_proximal": ("Tibial canal, proximal", "tibial mechanical axis"),
    "tibia.canal_centre_distal": ("Tibial canal, distal", "tibial mechanical axis"),
}


def landmark_label(landmark_id: str) -> tuple:
    """(name, role) of a landmark; the id made readable when it is not in the table."""
    if landmark_id in LANDMARK_ROLES:
        return LANDMARK_ROLES[landmark_id]
    return landmark_id.split(".")[-1].replace("_", " ").capitalize(), ""
