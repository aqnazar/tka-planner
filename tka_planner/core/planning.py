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
from .sides import LPS_ANTERIOR
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
    femoral_flexion_deg: float = 0.0    # sagittal flexion of the femoral component
    tibial_slope_deg: float | None = None  # None means match the native slope
    femoral_rotation_deg: float = 0.0   # external rotation from the frame reference


MECHANICAL = AlignmentTarget(
    philosophy="mechanical",
    description=(
        "Both components perpendicular to their mechanical axes, targeting a neutral "
        "hip-knee-ankle axis."
    ),
    tibial_slope_deg=3.0,
    femoral_rotation_deg=3.0,
)

KINEMATIC = AlignmentTarget(
    philosophy="kinematic",
    description=(
        "Resect what the implant replaces: cuts parallel to the native joint surfaces, "
        "restoring the patient's own joint line obliquity."
    ),
    tibial_slope_deg=None,  # reproduce the measured native slope
    femoral_rotation_deg=3.0,
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
    femur_mesh=None,
    tibia_mesh=None,
) -> SurgicalPlan:
    """Compute the correction, the resection planes and the component poses.

    Both cuts are built on a **single coronal reference**, and this is deliberate.

    In mechanical alignment the femoral and tibial mechanical axes are collinear once the
    limb is corrected, so both components are perpendicular to the same line and the two
    cuts necessarily share a mediolateral slope. Deriving each cut from its own
    independently estimated axis breaks that: on a real case here the femoral axis leaned
    1.0 degrees in the coronal plane and the tibial axis 5.4 degrees, a 4.4 degree
    disagreement that is estimation error from short canal segments rather than anatomy.
    Left uncorrected it tilts the two cuts against each other, and the components then
    have to articulate across that error.

    So the femoral mechanical axis supplies the coronal reference for both, and the cuts
    differ only in the sagittal plane -- femoral flexion and tibial posterior slope. The
    sagittal angles are *set from the target* rather than inherited from the axis, for
    the same reason: component flexion and tibial slope are surgical parameters, not
    properties of a diaphyseal fit that happens to lean forward because the leg lay at an
    angle in the scanner.

    ``femoral_thickness_mm`` is the distal thickness of the femoral component, a property
    of the implant taken from the size chart. ``tibial_resection_mm`` is measured from the
    *higher* (less worn) plateau, since the other has lost bone to disease.
    """
    warnings: list[str] = []

    valgus_deg, valgus_source, valgus_quality = _valgus_cut_angle(
        landmarks, femoral_frame
    )

    # ---- The shared coronal reference --------------------------------
    limb_axis = unit(femoral_frame.z_proximal)
    coronal_disagreement = float(np.degrees(angle_between(
        _in_plane_of(limb_axis, femoral_frame.x_anterior),
        _in_plane_of(tibial_frame.z_proximal, femoral_frame.x_anterior),
    )))
    if coronal_disagreement > 3.0:
        warnings.append(
            f"The independently estimated femoral and tibial mechanical axes disagree "
            f"by {coronal_disagreement:.1f} degrees in the coronal plane. The femoral "
            f"axis is used for both cuts so they share a mediolateral slope; the "
            f"disagreement is most likely error in the diaphyseal fits."
        )

    # ---- The shared cut basis ----------------------------------------
    #
    # Both cuts are built from one reference direction and one hinge, so they differ
    # only in their anteroposterior angle and share a mediolateral slope exactly. The
    # components articulate with each other, so a disagreement in mediolateral slope is
    # not a difference of plan but an error the construct has to absorb.
    if target.philosophy == "kinematic":
        medial, lateral = landmarks.require(
            "femur.condyle_distal_medial", "femur.condyle_distal_lateral"
        )
        joint_line = unit(lateral - medial)
        reference = unit(np.cross(joint_line, femoral_frame.x_anterior))
        if float(np.dot(reference, limb_axis)) < 0:
            reference = -reference
        femoral_reference = "parallel to the native distal condylar surface (kinematic)"
    else:
        reference = limb_axis
        femoral_reference = "perpendicular to the femoral mechanical axis"

    # Strip the reference's sagittal tilt, measured against the **patient** frame.
    #
    # A diaphyseal axis fitted to a supine, slightly flexed leg leans forwards by a few
    # degrees -- 4.3 on this case -- and that lean says how the leg lay in the scanner,
    # not what the anatomy is. Component flexion and tibial slope are surgical
    # parameters, so they are set explicitly below rather than inherited from it.
    #
    # The zero has to come from the patient frame rather than the bone frame. A bone
    # frame's anterior axis is perpendicular to its own proximal axis by construction,
    # so projecting the axis onto it removes nothing at all. LPS *is* the patient frame,
    # which is what makes this well defined: +Z superior, -Y anterior.
    reference = unit(reference - np.dot(reference, LPS_ANTERIOR) * LPS_ANTERIOR)

    # One hinge for both cuts: the mediolateral direction perpendicular to the
    # reference. Rotating about it sweeps the normal purely anteroposteriorly and leaves
    # the ratio of the mediolateral to proximal components untouched, so every cut built
    # on this pair carries an identical mediolateral slope however much its own
    # anteroposterior angle differs.
    hinge = unit(np.cross(reference, LPS_ANTERIOR))

    # ---- Femoral distal resection -----------------------------------
    femoral_normal = _tilt_about(
        reference, hinge, target.femoral_flexion_deg, posterior=-LPS_ANTERIOR
    )
    if abs(target.femoral_flexion_deg) > 1e-9:
        femoral_reference += f", {target.femoral_flexion_deg:.1f} degrees flexion"

    distal_points = np.array(landmarks.require(
        "femur.condyle_distal_medial", "femur.condyle_distal_lateral"
    ))
    projections = distal_points @ femoral_normal
    femoral_point = _plane_origin(
        distal_points, femoral_normal,
        seat=distal_points[int(np.argmin(projections))]
        + femoral_thickness_mm * femoral_normal,
        mesh=femur_mesh,
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
    # Same reference and same hinge as the femur, tilted posteriorly. A posterior slope
    # drops the back of the cut, which tilts the plane normal posteriorly; an earlier
    # version tilted it the other way and produced cuts that sloped forwards.
    tibial_normal = _tilt_about(
        reference, hinge, slope_deg, posterior=-LPS_ANTERIOR
    )

    plateau_points = np.array(landmarks.require(
        "tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"
    ))
    plateau_projections = plateau_points @ tibial_normal
    tibial_point = _plane_origin(
        plateau_points, tibial_normal,
        seat=plateau_points[int(np.argmax(plateau_projections))]
        - tibial_resection_mm * tibial_normal,
        mesh=tibia_mesh,
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
    #
    # Femoral component rotation is set off the **posterior condylar axis** with a
    # conventional external rotation, which is how it is set in theatre -- and not off
    # the frame's epicondylar axis, which serves the coronal construction. The two
    # differ by the condylar twist angle, so using the frame's would rotate the
    # component by that much.
    femoral_anterior = _rotational_reference(
        landmarks, femoral_frame, femoral_normal, target.femoral_rotation_deg
    )
    components = {
        "femoral_component": _component_pose(
            femoral_point, femoral_normal, convention="femoral",
            anterior=femoral_anterior,
        ),
        "tibial_component": _component_pose(
            tibial_point, tibial_normal, convention="tibial",
            anterior=_in_plane_of(tibial_frame.x_anterior, tibial_normal),
        ),
    }
    # A cutting block shares its implant's CAD origin, so it shares its pose exactly.
    components["femoral_cutting_block"] = components["femoral_component"]
    components["tibial_cutting_block"] = components["tibial_component"]

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
                f"{slope_deg:.1f} degrees posterior slope, sharing the femoral "
                f"coronal reference",
            ),
        },
        components=components,
        warnings=tuple(warnings),
        diagnostics={
            "target": target.description,
            "femoral_component_thickness_mm": femoral_thickness_mm,
            "tibial_resection_reference_mm": tibial_resection_mm,
            "femoral_flexion_deg": target.femoral_flexion_deg,
            "coronal_axis_disagreement_deg": round(coronal_disagreement, 2),
            "cut_ml_slope_shared": True,
            "reference_sagittal_tilt_removed": True,
            "cut_angle_between_deg": round(float(np.degrees(
                angle_between(femoral_normal, tibial_normal)
            )), 2),
            "femoral_resection_asymmetry_mm": round(
                abs(femoral_medial - femoral_lateral), 2),
            "tibial_resection_asymmetry_mm": round(
                abs(tibial_medial - tibial_lateral), 2),
        },
    )


def _in_plane_of(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """The part of ``vector`` lying in the plane whose normal is ``normal``."""
    normal = unit(normal)
    return unit(np.asarray(vector, dtype=float) - np.dot(vector, normal) * normal)


def _plane_origin(
    points: np.ndarray, normal: np.ndarray, *, seat: np.ndarray, mesh=None
) -> np.ndarray:
    """Centre the cut plane on the anatomy it cuts, at the seated depth.

    The plane's stored point doubles as the component's origin, so it must sit at the
    anatomical centre of the cut surface. Where the bone mesh is available, that centre
    is taken as the **centroid of the actual cross-section** the plane makes through it,
    which is what the component seats on.

    Falling back to the midpoint of the two compartment landmarks -- as this did before a
    mesh could be passed in -- puts the origin wherever those two happen to lie. On the
    tibia that is roughly a centimetre off in both the mediolateral and anteroposterior
    directions, because the deepest point of each plateau is neither centred nor
    symmetric.
    """
    normal = unit(normal)
    seat = np.asarray(seat, dtype=float)

    centre = None
    if mesh is not None:
        offsets = (mesh.vertices - seat) @ normal
        for half_width in (1.5, 3.0, 6.0):
            on_plane = np.abs(offsets) <= half_width
            if int(on_plane.sum()) >= 30:
                centre = mesh.vertices[on_plane].mean(axis=0)
                break
    if centre is None:
        centre = np.asarray(points, dtype=float).mean(axis=0)

    return centre + np.dot(seat - centre, normal) * normal


def _tilt_about(
    axis: np.ndarray,
    hinge: np.ndarray,
    angle_deg: float,
    *,
    posterior: np.ndarray,
) -> np.ndarray:
    """Rotate a cut normal about a supplied hinge, positive meaning posterior slope.

    The hinge is passed in rather than taken from a frame so that several cuts can share
    one, which is what guarantees they end up with identical mediolateral slopes.

    A positive angle drops the posterior edge of the cut, tilting the plane normal
    posteriorly. The sense is resolved by testing the result against the posterior
    direction rather than by a sign convention, so it holds on both knees without a
    conditional.
    """
    axis, hinge = unit(axis), unit(hinge)
    if abs(angle_deg) < 1e-9:
        return axis

    def rotate(theta):
        return unit(
            axis * np.cos(theta)
            + np.cross(hinge, axis) * np.sin(theta)
            + hinge * float(np.dot(hinge, axis)) * (1.0 - np.cos(theta))
        )

    angle = np.radians(abs(angle_deg))
    candidates = (rotate(angle), rotate(-angle))
    aim = np.asarray(posterior, dtype=float)
    if angle_deg < 0:
        aim = -aim
    return max(candidates, key=lambda v: float(np.dot(v, aim)))


def _rotational_reference(
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
    normal: np.ndarray,
    external_rotation_deg: float,
) -> np.ndarray:
    """Anterior direction for the femoral component, from the posterior condylar axis.

    Femoral rotation is set in theatre off the posterior condylar line with a
    conventional external rotation of about three degrees, and that is what is used
    here. The frame's epicondylar axis is a different reference serving the coronal
    construction; the two differ by the condylar twist angle, so borrowing the frame's
    would rotate the component by that much.

    External rotation carries the lateral side of the component anteriorly. The sense is
    resolved by testing against the frame's own lateral and anterior directions, so it
    is right on both knees with no conditional.

    Falls back to the frame's anterior axis when the posterior condyles are unavailable.
    """
    required = ("femur.condyle_posterior_medial", "femur.condyle_posterior_lateral")
    if not landmarks.available(*required):
        return _in_plane_of(frame.x_anterior, normal)

    medial, lateral = landmarks.require(*required)
    condylar_line = lateral - medial
    if float(np.dot(condylar_line, frame.lateral)) < 0:
        condylar_line = -condylar_line
    condylar_line = _in_plane_of(condylar_line, normal)

    # Anterior is a quarter turn from the condylar line; pick the turn that actually
    # points forwards.
    z_axis = unit(normal)
    candidates = (np.cross(condylar_line, z_axis), np.cross(z_axis, condylar_line))
    anterior = max(candidates, key=lambda v: float(np.dot(v, frame.x_anterior)))
    anterior = unit(anterior)

    if abs(external_rotation_deg) > 1e-9:
        angle = np.radians(abs(external_rotation_deg))
        turns = [
            unit(anterior * np.cos(t) + np.cross(z_axis, anterior) * np.sin(t))
            for t in (angle, -angle)
        ]
        # Which way is "external" is settled by the epicondylar axis rather than by
        # geometric reasoning about where the lateral side goes. The clinical rule means
        # something specific: three degrees of external rotation from the posterior
        # condylar line is a stand-in for the surgical epicondylar axis, which sits
        # externally rotated relative to it by the condylar twist angle. So the correct
        # turn is simply the one heading toward the epicondylar reference. Deriving the
        # sense geometrically instead sent it the other way, leaving the component about
        # six degrees short.
        toward = _in_plane_of(frame.x_anterior, normal)
        anterior = max(turns, key=lambda v: float(np.dot(v, toward)))
        if external_rotation_deg < 0:
            anterior = min(turns, key=lambda v: float(np.dot(v, toward)))
    return anterior


def _component_pose(
    point: np.ndarray,
    normal: np.ndarray,
    *,
    convention: str,
    anterior: np.ndarray,
) -> np.ndarray:
    """A 4x4 pose placing a component's own CAD axes onto the cut.

    The implant library uses two axis conventions, a quarter turn apart::

        femoral parts   local +X anterior, +Y patient-left, +Z proximal
        tibial parts    local +X patient-left, +Y posterior, +Z proximal

    Both are right-handed. Mapping each explicitly is what puts a component the right way
    round; assuming a single convention for the whole library leaves the tibial parts
    rotated ninety degrees.

    The translation places the component's **native CAD origin** on the cut. Every part
    belonging to a bone shares that origin, which is precisely why a cutting block and
    its implant, given the same pose, coincide by construction.
    """
    z_axis = unit(normal)
    anterior = unit(np.asarray(anterior, dtype=float)
                    - np.dot(anterior, z_axis) * z_axis)
    patient_left = np.cross(z_axis, anterior)

    if convention == "femoral":
        columns = (anterior, patient_left, z_axis)
    elif convention == "tibial":
        columns = (patient_left, -anterior, z_axis)
    else:
        raise ValueError(f"Unknown component convention {convention!r}.")

    matrix = np.eye(4)
    for index, column in enumerate(columns):
        matrix[:3, index] = column
    matrix[:3, 3] = np.asarray(point, dtype=float)

    if not np.isclose(float(np.linalg.det(matrix[:3, :3])), 1.0, atol=1e-6):
        raise ValueError(
            f"The {convention} component pose is not right-handed; the axis mapping is "
            f"wrong and the part would be mirrored."
        )
    return matrix


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
