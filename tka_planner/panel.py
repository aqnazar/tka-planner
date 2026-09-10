"""The control panel, described once so every front end can build it.

The add-on carried this in Blender property definitions, where a browser cannot read it.
A viewer that invented its own ranges would be a second place for a clinical limit to
live, and two places drift. So the ranges, the wording and the grouping are here, lifted
from the add-on unchanged, and the panel is built from them.

A test asserts that this covers :class:`~tka_planner.session.Controls` exactly, which is
what stops a control being added to one and forgotten in the other.
"""

from __future__ import annotations

__all__ = ["CONTROL_SCHEMA", "TRIAL_SCHEMA", "control_names", "with_sizes"]

CONTROL_SCHEMA = (
    {
        "group": "Alignment",
        "controls": (
            {"name": "philosophy", "label": "Alignment", "kind": "choice",
             "choices": [["mechanical", "Mechanical"], ["kinematic", "Kinematic"]],
             "help": "Mechanical puts the components perpendicular to the mechanical "
                     "axes. Kinematic cuts parallel to the native joint surfaces."},
            {"name": "size_override", "label": "Size", "kind": "choice",
             "choices": [["", "Solve from anatomy"]],
             "help": "Auto solves a continuous size from the measurement. The listed "
                     "sizes are the twelve the chart publishes."},
            {"name": "tibial_resection_mm", "label": "Tibial resection",
             "kind": "float", "min": 0.0, "max": 25.0, "step": 0.5, "unit": "mm",
             "help": "Depth below the higher, less worn plateau."},
            {"name": "coronal_correction_deg", "label": "Varus / valgus",
             "kind": "float", "min": -15.0, "max": 15.0, "step": 0.5, "unit": "deg",
             "help": "Coronal correction applied to both cuts together, so the "
                     "components stay parallel. Positive is valgus."},
        ),
    },
    {
        "group": "Femoral cut",
        "controls": (
            {"name": "femoral_resection_delta_mm", "label": "Distal resection",
             "kind": "float", "min": -6.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Extra bone off the distal femur, beyond the component "
                     "thickness."},
            {"name": "femoral_flexion_delta_deg", "label": "Flexion (cut)",
             "kind": "float", "min": -10.0, "max": 15.0, "step": 0.5, "unit": "deg",
             "help": "Sagittal flexion of the femoral cut and component. This changes "
                     "the cut, unlike the trial flexion in Reduce."},
            {"name": "femoral_varus_delta_deg", "label": "Varus / valgus (this cut)",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.5, "unit": "deg",
             "help": "Coronal angle of the femoral cut alone. This breaks the shared "
                     "mediolateral slope with the tibial cut, and the plan says so."},
            {"name": "femoral_rotation_delta_deg", "label": "Rotation",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.5, "unit": "deg",
             "help": "External rotation beyond the conventional three degrees off the "
                     "posterior condylar axis."},
            {"name": "femoral_shift_ap_mm", "label": "Anterior / posterior",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Slide the component across its cut. Positive is anterior."},
            {"name": "femoral_shift_ml_mm", "label": "Medial / lateral",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Slide the component across its cut. Positive is lateral."},
        ),
    },
    {
        "group": "Tibial cut",
        "controls": (
            {"name": "tibial_resection_delta_mm", "label": "Resection",
             "kind": "float", "min": -6.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Extra bone off the proximal tibia, beyond the set depth."},
            {"name": "tibial_slope_delta_deg", "label": "Posterior slope",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.5, "unit": "deg",
             "help": "Change to the planned posterior slope. Positive is more slope."},
            {"name": "tibial_varus_delta_deg", "label": "Varus / valgus (this cut)",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.5, "unit": "deg",
             "help": "Coronal angle of the tibial cut alone. This breaks the shared "
                     "mediolateral slope with the femoral cut, and the plan says so."},
            {"name": "tibial_rotation_delta_deg", "label": "Rotation",
             "kind": "float", "min": -15.0, "max": 15.0, "step": 0.5, "unit": "deg",
             "help": "External rotation of the tray about its cut normal."},
            {"name": "tibial_shift_ap_mm", "label": "Anterior / posterior",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Slide the tray across its cut. Positive is anterior."},
            {"name": "tibial_shift_ml_mm", "label": "Medial / lateral",
             "kind": "float", "min": -10.0, "max": 10.0, "step": 0.1, "unit": "mm",
             "help": "Slide the tray across its cut. Positive is lateral."},
        ),
    },
    {
        "group": "Insert",
        "controls": (
            {"name": "use_insert", "label": "Insert", "kind": "bool",
             "help": "Show the plastic insert at a set thickness and report the "
                     "extension gap it leaves."},
            {"name": "insert_thickness_mm", "label": "Thickness", "kind": "float",
             "min": 4.0, "max": 20.0, "step": 0.5, "unit": "mm"},
        ),
    },
    {
        "group": "Resection",
        "controls": (
            {"name": "resection_mode", "label": "Resect with", "kind": "choice",
             "choices": [["block", "Cutting block"], ["plane", "Cut plane"]],
             "help": "The block is the instrument that would realise the cut in "
                     "theatre; the plane is what the plan specifies. They are "
                     "alternatives and are never stacked."},
            {"name": "build_bone_shells", "label": "Bone shells", "kind": "bool",
             "help": "Intersect a copy of each bone with its cutting block shell, "
                     "giving the patient-specific mating surface."},
        ),
    },
    {
        "group": "Display",
        "controls": (
            {"name": "show_planes", "label": "Cut planes", "kind": "bool"},
            {"name": "show_axes", "label": "Axes", "kind": "bool"},
            {"name": "show_landmarks", "label": "Landmarks", "kind": "bool"},
            {"name": "show_cutting_blocks", "label": "Cutting blocks", "kind": "bool",
             "help": "Show the instrument that would realise the cut. It is the size "
                     "of the block rather than of the resection, so it stands in "
                     "front of the bone."},
            {"name": "isolate_landmarks", "label": "Isolate landmarks", "kind": "bool",
             "help": "Hide everything but the landmarks, for watching how they move "
                     "relative to each other through a trial pose."},
        ),
    },
)

TRIAL_SCHEMA = (
    {"name": "flexion_deg", "label": "Flexion", "kind": "float",
     "min": -10.0, "max": 150.0, "step": 0.5, "unit": "deg",
     "help": "Flex the implanted construct about the transepicondylar axis. This "
             "poses the committed geometry and does not touch the cut."},
    {"name": "varus_valgus_deg", "label": "Varus / valgus stress", "kind": "float",
     "min": -15.0, "max": 15.0, "step": 0.5, "unit": "deg",
     "help": "Apply a coronal stress at the current flexion angle, the way a manual "
             "laxity exam does."},
    {"name": "drawer_ap_mm", "label": "AP drawer", "kind": "float",
     "min": -15.0, "max": 15.0, "step": 0.1, "unit": "mm",
     "help": "Slide the tibia anteriorly or posteriorly to check a drawer test."},
)


def control_names() -> set:
    """Every control the schema describes."""
    return {
        control["name"] for group in CONTROL_SCHEMA for control in group["controls"]
    }


def with_sizes(labels) -> tuple:
    """The schema with the size menu filled in from the chart that is actually loaded.

    The twelve published sizes are data, not code, so they are not written here. Solving
    from the anatomy stays the first entry because it is the answer the pipeline
    computes; a discrete size is an override of it.
    """
    options = [["", "Solve from anatomy"]] + [[label, label] for label in labels]
    return tuple(
        {
            **group,
            "controls": tuple(
                {**control, "choices": options}
                if control["name"] == "size_override" else control
                for control in group["controls"]
            ),
        }
        for group in CONTROL_SCHEMA
    )
