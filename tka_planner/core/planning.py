"""Alignment planning: resection planes, correction angles and component poses.

This is the step that turns measurements into a surgical plan. It answers the questions
a planning screen exists to answer -- at what angle is the distal femoral cut, how deep
is each resection, where does each component sit -- and it answers them from the
patient's own anatomy rather than from a fixed constant.

The distal femoral valgus cut angle
-----------------------------------
The headline number. A surgeon setting a distal femoral jig dials in a valgus angle,
conventionally about 5 to 7 degrees, because the cut must be perpendicular to the
*mechanical* axis while the jig references the *anatomical* canal. That angle is exactly
the femoral mechanical-anatomical angle, and it is a property of the patient.

The legacy pipeline hard-coded it at 6 degrees for everyone. Here it is computed --
measured outright when the femoral head is in the scan, and otherwise derived from the
declared population assumption, with the plan stating which. Same number in the default
case; the difference is that it is now a stated, per-patient, challengeable quantity.

Alignment philosophies
----------------------
``mechanical``
    Both components perpendicular to their mechanical axes, targeting a neutral limb.
    On a knee-only scan the femoral mechanical axis is itself assumed, so the plan is
    honest that mechanical alignment here is executed *under an assumption*.
``kinematic``
    Resect what the implant replaces: the cut parallels the native joint surface and
    the resection depth equals the component thickness on each side. Fully computable
    from a knee-only scan, since it references only local joint geometry -- which makes
    it the one philosophy this cohort supports without assumption.

Both are supported and nothing else. Restricted and adjusted variants are deliberately
out of scope until there is data to choose between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frames import AnatomicalFrame
from .geometry import angle_between, unit
from .landmarks import LandmarkSet
from .provenance import Quality

__all__ = [
    "AlignmentTarget",
    "ResectionPlane",
    "SurgicalPlan",
    "plan_alignment",
    "MECHANICAL",
    "KINEMATIC",
]


@dataclass(frozen=True)
class AlignmentTarget:
    """What the plan is aiming at."""

    philosophy: str
    description: str
    femoral_slope_deg: float = 3.0      # flexion of the femoral component
    tibial_slope_deg: float | None = None  # None means match the native slope
    femoral_rotation_deg: float = 0.0   # external rotation from the frame reference


MECHANICAL = AlignmentTarget(
    philosophy="mechanical",
    description=(
        "Both components perpendicular to their mechanical axes, targeting a neutral "
        "hip-knee-ankle axis."
    ),
    tibial_slope_deg=3.0,
)

KINEMATIC = AlignmentTarget(
    philosophy="kinematic",
    description=(
        "Resect what the implant replaces: cuts parallel to the native joint surfaces, "
        "restoring the patient's own joint line obliquity."
    ),
    tibial_slope_deg=None,  # reproduce the measured native slope
)


@dataclass(frozen=True)
class ResectionPlane:
    """A cut plane, with the depth it removes at each compartment."""

    name: str
    point: np.ndarray
    normal: np.ndarray
    medial_depth_mm: float
    lateral_depth_mm: float
    reference: str

    def to_dict(self) -> dict:
        return {
            "point_mm": [round(float(v), 3) for v in self.point],
            "normal": [round(float(v), 6) for v in self.normal],
            "medial_depth_mm": round(self.medial_depth_mm, 2),
            "lateral_depth_mm": round(self.lateral_depth_mm, 2),
            "reference": self.reference,
        }


@dataclass(frozen=True)
class SurgicalPlan:
    """The planned correction, resections and component placements."""

    philosophy: str
    distal_femoral_valgus_cut_deg: float
    valgus_source: str
    valgus_quality: Quality
    tibial_slope_deg: float
    resections: dict[str, ResectionPlane]
    components: dict[str, np.ndarray]
    warnings: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "philosophy": self.philosophy,
            "distal_femoral_valgus_cut_deg": round(
                self.distal_femoral_valgus_cut_deg, 2),
            "valgus_source": self.valgus_source,
            "valgus_quality": self.valgus_quality.value,
            "tibial_slope_deg": round(self.tibial_slope_deg, 2),
            "resections": {k: v.to_dict() for k, v in self.resections.items()},
            "components": {
                k: [[round(float(x), 6) for x in row] for row in matrix]
                for k, matrix in self.components.items()
            },
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics,
        }


def _plane_depths(
    landmarks: LandmarkSet,
    point: np.ndarray,
    normal: np.ndarray,
    medial_id: str,
    lateral_id: str,
    *,
    removes: str,
) -> tuple[float, float]:
    """How much bone each compartment loses to a cut plane.

    ``removes`` says which side of the plane the discarded bone lies on, and the two
    joints differ: the distal femur is cut from below, so the removed bone is *distal*
    to the plane, while the proximal tibia is cut from above and its removed bone is
    *proximal*. Applying one convention to both silently reports the tibial depths
    negative -- caught here by the out-of-bone warning on a real case.

    Both compartments are reported separately, because a single "resection depth" hides
    exactly the asymmetry alignment planning exists to manage.
    """
    if removes not in ("distal", "proximal"):
        raise ValueError("removes must be 'distal' or 'proximal'.")

    medial, lateral = landmarks.require(medial_id, lateral_id)
    sign = 1.0 if removes == "distal" else -1.0
    normal = unit(normal)
    return (
        float(sign * np.dot(point - medial, normal)),
        float(sign * np.dot(point - lateral, normal)),
    )


def plan_alignment(
    landmarks: LandmarkSet,
    femoral_frame: AnatomicalFrame,
    tibial_frame: AnatomicalFrame,
    *,
    target: AlignmentTarget = MECHANICAL,
    femoral_thickness_mm: float = 9.0,
    tibial_resection_mm: float = 10.0,
    native_slope_deg: float | None = None,
) -> SurgicalPlan:
    """Compute the correction, the resection planes and the component poses.

    ``femoral_thickness_mm`` is the distal thickness of the femoral component, which is
    a property of the implant and comes from the size chart -- not a surgical choice.
    ``tibial_resection_mm`` is measured from the *higher* (less worn) plateau, which is
    the usual reference because the lower one has lost bone to disease.
    """
    warnings: list[str] = []

    # ---- The distal femoral valgus cut angle -------------------------
    valgus_deg, valgus_source, valgus_quality = _valgus_cut_angle(
        landmarks, femoral_frame
    )

    # ---- Femoral distal resection -----------------------------------
    if target.philosophy == "kinematic":
        # Parallel to the native distal condylar surface.
        medial, lateral = landmarks.require(
            "femur.condyle_distal_medial", "femur.condyle_distal_lateral"
        )
        joint_line = unit(lateral - medial)
        femoral_normal = unit(np.cross(joint_line, femoral_frame.x_anterior))
        if float(np.dot(femoral_normal, femoral_frame.z_proximal)) < 0:
            femoral_normal = -femoral_normal
        femoral_reference = "native distal condylar surface (kinematic)"
    else:
        femoral_normal = femoral_frame.z_proximal
        femoral_reference = "perpendicular to the femoral mechanical axis"

    # Seat the plane so the deeper compartment gives exactly the component thickness.
    distal_points = np.array(landmarks.require(
        "femur.condyle_distal_medial", "femur.condyle_distal_lateral"
    ))
    projections = distal_points @ femoral_normal
    femoral_point = (
        distal_points[int(np.argmin(projections))]
        + femoral_thickness_mm * femoral_normal
    )
    femoral_medial, femoral_lateral = _plane_depths(
        landmarks, femoral_point, femoral_normal,
        "femur.condyle_distal_medial", "femur.condyle_distal_lateral",
        removes="distal",
    )

    # ---- Tibial resection -------------------------------------------
    slope_deg = (
        target.tibial_slope_deg
        if target.tibial_slope_deg is not None
        else (native_slope_deg if native_slope_deg is not None else 3.0)
    )
    tibial_normal = _apply_posterior_slope(
        tibial_frame.z_proximal, tibial_frame, slope_deg
    )

    plateau_points = np.array(landmarks.require(
        "tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"
    ))
    plateau_projections = plateau_points @ tibial_normal
    # Reference the higher (less worn) plateau.
    tibial_point = (
        plateau_points[int(np.argmax(plateau_projections))]
        - tibial_resection_mm * tibial_normal
    )
    tibial_medial, tibial_lateral = _plane_depths(
        landmarks, tibial_point, tibial_normal,
        "tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest",
        removes="proximal",
    )

    for name, medial, lateral in (
        ("femoral", femoral_medial, femoral_lateral),
        ("tibial", tibial_medial, tibial_lateral),
    ):
        if min(medial, lateral) < -0.5:
            warnings.append(
                f"The {name} cut plane lies outside the bone on one compartment "
                f"(medial {medial:.1f} mm, lateral {lateral:.1f} mm); check the "
                f"landmarks and the component thickness."
            )

    # ---- Component poses --------------------------------------------
    components = {
        "femoral_component": _pose(
            femoral_point, femoral_normal, femoral_frame, target.femoral_rotation_deg
        ),
        "tibial_component": _pose(tibial_point, tibial_normal, tibial_frame, 0.0),
    }

    return SurgicalPlan(
        philosophy=target.philosophy,
        distal_femoral_valgus_cut_deg=valgus_deg,
        valgus_source=valgus_source,
        valgus_quality=valgus_quality,
        tibial_slope_deg=slope_deg,
        resections={
            "femoral_distal": ResectionPlane(
                "femoral_distal", femoral_point, femoral_normal,
                femoral_medial, femoral_lateral, femoral_reference,
            ),
            "tibial_proximal": ResectionPlane(
                "tibial_proximal", tibial_point, tibial_normal,
                tibial_medial, tibial_lateral,
                f"{tibial_resection_mm:.0f} mm below the higher plateau, "
                f"{slope_deg:.1f} degrees posterior slope",
            ),
        },
        components=components,
        warnings=tuple(warnings),
        diagnostics={
            "target": target.description,
            "femoral_component_thickness_mm": femoral_thickness_mm,
            "tibial_resection_reference_mm": tibial_resection_mm,
            "femoral_resection_asymmetry_mm": round(
                abs(femoral_medial - femoral_lateral), 2),
            "tibial_resection_asymmetry_mm": round(
                abs(tibial_medial - tibial_lateral), 2),
        },
    )


def _valgus_cut_angle(
    landmarks: LandmarkSet, frame: AnatomicalFrame
) -> tuple[float, str, Quality]:
    """The angle a distal femoral jig would be set to, for this patient.

    Measured directly as the angle between the mechanical and anatomical axes when both
    are available. Otherwise it is whatever the frame assumed, and the plan says so --
    which is the same 6 degrees the legacy pipeline applied, now attributed.
    """
    canal = ("femur.canal_centre_distal", "femur.canal_centre_proximal")
    if landmarks.available(*canal) and frame.quality is Quality.MEASURED:
        distal, proximal = landmarks.require(*canal)
        anatomical = unit(proximal - distal)
        return (
            float(np.degrees(angle_between(anatomical, frame.z_proximal))),
            "measured from this patient's mechanical and anatomical axes",
            Quality.MEASURED,
        )

    assumed = next(
        (a for a in frame.assumptions if a.id == "population_femoral_ama"), None
    )
    if assumed is not None:
        return (
            float(assumed.value),
            f"assumed: {assumed.source.split('.')[0].lower()}",
            Quality.ESTIMATED,
        )
    return (6.0, "default population value", Quality.ESTIMATED)


def _apply_posterior_slope(
    axis: np.ndarray, frame: AnatomicalFrame, slope_deg: float
) -> np.ndarray:
    """Tilt a cut normal posteriorly, without any per-side sign conditional.

    The rotation sense is fixed by asking the frame which way is posterior, so the same
    code is correct on both knees.
    """
    if abs(slope_deg) < 1e-9:
        return unit(axis)

    hinge = frame.y_patient_left
    angle = np.radians(slope_deg)
    rotated = (
        axis * np.cos(angle)
        + np.cross(hinge, axis) * np.sin(angle)
        + hinge * float(np.dot(hinge, axis)) * (1.0 - np.cos(angle))
    )
    # A posterior slope tips the plane's normal anteriorly; pick the sense that does so.
    if float(np.dot(rotated, frame.x_anterior)) < float(
        np.dot(unit(axis), frame.x_anterior)
    ):
        rotated = (
            axis * np.cos(-angle)
            + np.cross(hinge, axis) * np.sin(-angle)
            + hinge * float(np.dot(hinge, axis)) * (1.0 - np.cos(-angle))
        )
    return unit(rotated)


def _pose(
    point: np.ndarray,
    normal: np.ndarray,
    frame: AnatomicalFrame,
    rotation_deg: float,
) -> np.ndarray:
    """A 4x4 component pose: seated on the cut plane, oriented by the frame."""
    z_axis = unit(normal)
    x_axis = frame.x_anterior - np.dot(frame.x_anterior, z_axis) * z_axis
    x_axis = unit(x_axis)

    if abs(rotation_deg) > 1e-9:
        angle = np.radians(rotation_deg)
        x_axis = unit(
            x_axis * np.cos(angle) + np.cross(z_axis, x_axis) * np.sin(angle)
        )

    y_axis = np.cross(z_axis, x_axis)

    matrix = np.eye(4)
    matrix[:3, 0] = x_axis
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = z_axis
    matrix[:3, 3] = point
    return matrix
