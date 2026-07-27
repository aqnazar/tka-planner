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
    "Adjustments",
    "ResectionPlane",
    "SurgicalPlan",
    "plan_alignment",
    "MECHANICAL",
    "KINEMATIC",
]


@dataclass(frozen=True)
class Adjustments:
    """Manual departures from the computed plan, in the surgeon's own terms.

    These are what the planning screen's controls write. They are inputs to
    :func:`plan_alignment` rather than edits applied to its output, which is the whole
    point: an adjusted plan is still a pure function of the landmarks, the frames, the
    target and this object, so it serialises into ``plan.json`` and re-derives exactly.
    Nudging the scene instead would leave the geometry describing something the plan file
    does not.

    Signs are anatomical, never spatial. Positive is valgus, deeper, more slope, more
    external rotation, more anterior and more lateral -- on both knees. A signed rotation
    about a world axis would mean the opposite thing on a right knee, which is the class
    of bug the mirror-invariance tests exist to catch.

    ``insert_thickness_mm`` alters no cut. It sets the construct height the gap report
    is measured against, and ``None`` means no gap is reported at all rather than one
    computed from an assumed insert.
    """

    coronal_correction_deg: float = 0.0       # + valgus, applied to BOTH cuts together

    femoral_resection_delta_mm: float = 0.0   # + removes more bone
    femoral_varus_delta_deg: float = 0.0      # + valgus, this cut only
    femoral_flexion_delta_deg: float = 0.0
    femoral_rotation_delta_deg: float = 0.0   # + external
    femoral_shift_ap_mm: float = 0.0          # + anterior
    femoral_shift_ml_mm: float = 0.0          # + lateral

    tibial_resection_delta_mm: float = 0.0
    tibial_varus_delta_deg: float = 0.0
    tibial_slope_delta_deg: float = 0.0
    tibial_rotation_delta_deg: float = 0.0
    tibial_shift_ap_mm: float = 0.0
    tibial_shift_ml_mm: float = 0.0

    insert_thickness_mm: float | None = None

    def to_dict(self) -> dict:
        return {
            "coronal_correction_deg": self.coronal_correction_deg,
            "femoral_resection_delta_mm": self.femoral_resection_delta_mm,
            "femoral_varus_delta_deg": self.femoral_varus_delta_deg,
            "femoral_flexion_delta_deg": self.femoral_flexion_delta_deg,
            "femoral_rotation_delta_deg": self.femoral_rotation_delta_deg,
            "femoral_shift_ap_mm": self.femoral_shift_ap_mm,
            "femoral_shift_ml_mm": self.femoral_shift_ml_mm,
            "tibial_resection_delta_mm": self.tibial_resection_delta_mm,
            "tibial_varus_delta_deg": self.tibial_varus_delta_deg,
            "tibial_slope_delta_deg": self.tibial_slope_delta_deg,
            "tibial_rotation_delta_deg": self.tibial_rotation_delta_deg,
            "tibial_shift_ap_mm": self.tibial_shift_ap_mm,
            "tibial_shift_ml_mm": self.tibial_shift_ml_mm,
            "insert_thickness_mm": self.insert_thickness_mm,
        }

    @property
    def is_identity(self) -> bool:
        """Whether anything was actually moved, so a report can say "as planned"."""
        return self == Adjustments(insert_thickness_mm=self.insert_thickness_mm)


NO_ADJUSTMENT = Adjustments()


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
    adjustments: Adjustments = NO_ADJUSTMENT

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
            "adjustments": self.adjustments.to_dict(),
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
    tibial_tray_thickness_mm: float = 0.0,
    native_slope_deg: float | None = None,
    adjustments: Adjustments = NO_ADJUSTMENT,
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

    ``adjustments`` carries whatever a surgeon has changed by hand. It enters here rather
    than being applied to the returned plan so that the plan remains reproducible from
    its inputs; see :class:`Adjustments`.

    ``tibial_tray_thickness_mm`` defaults to zero because the tray's height is not a
    column in the size chart and has not been confirmed against the CAD. Until it is,
    the gap report understates the construct by exactly that height rather than assuming
    a figure -- see ``docs/CLINICAL_QUESTIONS.md`` 3.1.
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

    # ---- Varus/valgus correction -------------------------------------
    #
    # This is the one control that has to move the whole construct, so it is applied to
    # the shared reference *before* the two cuts are separated. Both cuts descend from
    # this direction, so both rotate by the same angle and they go on sharing a
    # mediolateral slope exactly -- which is the guarantee the rest of this function is
    # built around. Rotating the two finished cuts by the same amount afterwards would
    # look identical and would not be: their hinges differ once the sagittal angles are
    # applied, so the slopes would drift apart.
    #
    # Positive is valgus. The sense is resolved by aiming at the frame's own lateral
    # direction, so the same number means the same correction on a left and a right knee.
    if abs(adjustments.coronal_correction_deg) > 1e-9:
        reference = _tilt_about(
            reference, LPS_ANTERIOR, adjustments.coronal_correction_deg,
            posterior=femoral_frame.lateral,
        )
        femoral_reference += (
            f", {adjustments.coronal_correction_deg:+.1f} degrees manual coronal "
            f"correction"
        )

    # One hinge for both cuts: the mediolateral direction perpendicular to the
    # reference. Rotating about it sweeps the normal purely anteroposteriorly and leaves
    # the ratio of the mediolateral to proximal components untouched, so every cut built
    # on this pair carries an identical mediolateral slope however much its own
    # anteroposterior angle differs.
    hinge = unit(np.cross(reference, LPS_ANTERIOR))

    # ---- Femoral distal resection -----------------------------------
    femoral_flexion_deg = (
        target.femoral_flexion_deg + adjustments.femoral_flexion_delta_deg
    )
    femoral_normal = _tilt_about(
        reference, hinge, femoral_flexion_deg, posterior=-LPS_ANTERIOR
    )
    if abs(adjustments.femoral_varus_delta_deg) > 1e-9:
        # A per-cut varus override, deliberately applied after the split. It breaks the
        # shared mediolateral slope, which is why it is reported below rather than
        # absorbed silently.
        femoral_normal = _tilt_about(
            femoral_normal, LPS_ANTERIOR, adjustments.femoral_varus_delta_deg,
            posterior=femoral_frame.lateral,
        )
    if abs(femoral_flexion_deg) > 1e-9:
        femoral_reference += f", {femoral_flexion_deg:.1f} degrees flexion"

    # The manual delta is folded into the seating depth rather than applied to the
    # finished plane, so the plane is still centred on the cross-section it actually
    # makes through the bone at its final level.
    femoral_seat_mm = femoral_thickness_mm + adjustments.femoral_resection_delta_mm
    distal_points = np.array(landmarks.require(
        "femur.condyle_distal_medial", "femur.condyle_distal_lateral"
    ))
    projections = distal_points @ femoral_normal
    femoral_point = _plane_origin(
        distal_points, femoral_normal,
        seat=distal_points[int(np.argmin(projections))]
        + femoral_seat_mm * femoral_normal,
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
    ) + adjustments.tibial_slope_delta_deg
    # Same reference and same hinge as the femur, tilted posteriorly. A posterior slope
    # drops the back of the cut, which tilts the plane normal posteriorly; an earlier
    # version tilted it the other way and produced cuts that sloped forwards.
    tibial_normal = _tilt_about(
        reference, hinge, slope_deg, posterior=-LPS_ANTERIOR
    )
    if abs(adjustments.tibial_varus_delta_deg) > 1e-9:
        tibial_normal = _tilt_about(
            tibial_normal, LPS_ANTERIOR, adjustments.tibial_varus_delta_deg,
            posterior=tibial_frame.lateral,
        )

    tibial_seat_mm = tibial_resection_mm + adjustments.tibial_resection_delta_mm
    plateau_points = np.array(landmarks.require(
        "tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"
    ))
    plateau_projections = plateau_points @ tibial_normal
    tibial_point = _plane_origin(
        plateau_points, tibial_normal,
        seat=plateau_points[int(np.argmax(plateau_projections))]
        - tibial_seat_mm * tibial_normal,
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
        landmarks, femoral_frame, femoral_normal,
        target.femoral_rotation_deg + adjustments.femoral_rotation_delta_deg,
    )
    tibial_anterior = _rotate_in_plane(
        _in_plane_of(tibial_frame.x_anterior, tibial_normal),
        tibial_normal,
        adjustments.tibial_rotation_delta_deg,
        toward=tibial_frame.lateral,
    )
    components = {
        "femoral_component": _component_pose(
            femoral_point, femoral_normal, convention="femoral",
            anterior=femoral_anterior,
        ),
        "tibial_component": _component_pose(
            tibial_point, tibial_normal, convention="tibial",
            anterior=tibial_anterior,
        ),
    }

    # Component position is not resection. A shift slides the implant across the cut it
    # sits on, so it is applied to the pose alone and both shift directions are taken in
    # the plane of that cut -- otherwise moving a component forwards would also lift it
    # off its own cut surface.
    components["femoral_component"] = _shift_component(
        components["femoral_component"], femoral_normal,
        anterior=femoral_anterior, lateral=femoral_frame.lateral,
        ap_mm=adjustments.femoral_shift_ap_mm,
        ml_mm=adjustments.femoral_shift_ml_mm,
    )
    components["tibial_component"] = _shift_component(
        components["tibial_component"], tibial_normal,
        anterior=tibial_anterior, lateral=tibial_frame.lateral,
        ap_mm=adjustments.tibial_shift_ap_mm,
        ml_mm=adjustments.tibial_shift_ml_mm,
    )

    # A cutting block shares its implant's CAD origin, so it shares its pose exactly --
    # including any manual shift, since the block is the instrument that would realise
    # the cut the implant then sits on.
    components["femoral_cutting_block"] = components["femoral_component"]
    components["tibial_cutting_block"] = components["tibial_component"]

    # ---- Did the cuts stay parallel in the coronal plane? -------------
    ml_disagreement = float(np.degrees(angle_between(
        _in_plane_of(femoral_normal, LPS_ANTERIOR),
        _in_plane_of(tibial_normal, LPS_ANTERIOR),
    )))
    if ml_disagreement > 0.01:
        warnings.append(
            f"The femoral and tibial cuts disagree in mediolateral slope by "
            f"{ml_disagreement:.1f} degrees. A per-cut varus override was applied, so "
            f"the construct has to absorb the difference across the articulation."
        )

    gaps = _extension_gaps(
        landmarks,
        femoral_point=femoral_point, femoral_normal=femoral_normal,
        tibial_point=tibial_point, tibial_normal=tibial_normal,
        femoral_thickness_mm=femoral_thickness_mm,
        tray_thickness_mm=tibial_tray_thickness_mm,
        insert_thickness_mm=adjustments.insert_thickness_mm,
    )

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
                f"{tibial_seat_mm:.0f} mm below the higher plateau, "
                f"{slope_deg:.1f} degrees posterior slope, sharing the femoral "
                f"coronal reference",
            ),
        },
        components=components,
        warnings=tuple(warnings),
        adjustments=adjustments,
        diagnostics={
            "target": target.description,
            "femoral_component_thickness_mm": femoral_thickness_mm,
            "tibial_resection_reference_mm": tibial_seat_mm,
            "femoral_flexion_deg": femoral_flexion_deg,
            "coronal_axis_disagreement_deg": round(coronal_disagreement, 2),
            "cut_ml_slope_shared": ml_disagreement <= 0.01,
            "cut_ml_disagreement_deg": round(ml_disagreement, 3),
            **gaps,
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


def _rotate_in_plane(
    vector: np.ndarray,
    normal: np.ndarray,
    angle_deg: float,
    *,
    toward: np.ndarray,
) -> np.ndarray:
    """Turn a direction about a plane's normal, with the sense fixed anatomically.

    Used for component rotation, where positive means external. Which way that is
    depends on the knee, so the sense is settled by asking which turn carries the vector
    toward the frame's lateral direction rather than by a sign convention that would be
    right on one side only.
    """
    vector = _in_plane_of(vector, normal)
    if abs(angle_deg) < 1e-9:
        return vector

    axis = unit(normal)
    angle = np.radians(abs(angle_deg))
    turns = [
        unit(vector * np.cos(t) + np.cross(axis, vector) * np.sin(t))
        for t in (angle, -angle)
    ]
    aim = _in_plane_of(np.asarray(toward, dtype=float), normal)
    chooser = max if angle_deg > 0 else min
    return chooser(turns, key=lambda v: float(np.dot(v, aim)))


def _shift_component(
    pose: np.ndarray,
    normal: np.ndarray,
    *,
    anterior: np.ndarray,
    lateral: np.ndarray,
    ap_mm: float,
    ml_mm: float,
) -> np.ndarray:
    """Slide a component across the cut it sits on.

    Both directions are taken in the plane of the cut, so a shift never changes the
    component's seating depth -- moving an implant forwards must not also lift it off the
    bone. Positive is anterior and lateral, on either knee.
    """
    if abs(ap_mm) < 1e-12 and abs(ml_mm) < 1e-12:
        return pose

    anterior = _in_plane_of(anterior, normal)
    lateral = _in_plane_of(lateral, normal)

    shifted = pose.copy()
    shifted[:3, 3] = pose[:3, 3] + ap_mm * anterior + ml_mm * lateral
    return shifted


def _extension_gaps(
    landmarks: LandmarkSet,
    *,
    femoral_point: np.ndarray,
    femoral_normal: np.ndarray,
    tibial_point: np.ndarray,
    tibial_normal: np.ndarray,
    femoral_thickness_mm: float,
    tray_thickness_mm: float,
    insert_thickness_mm: float | None,
) -> dict:
    """What is left between the two cuts once the construct is in, per compartment.

    The space between the resected surfaces is measured at each compartment, and the
    components that fill it are subtracted: the femoral component's distal thickness
    hangs below the femoral cut, the tray and the insert stack above the tibial one. A
    negative result means the construct overstuffs that compartment.

    Each compartment is measured at its own plateau landmark, projected onto the tibial
    cut, because the whole reason to report a gap is the difference between the two
    sides. A single mid-joint figure would hide it.

    The measurement runs **normal to the tibial cut**. With posterior slope the two cuts
    are not parallel, so there is no single separation between them, and this is the
    direction that matches how the space is filled and judged: the insert seats on the
    tibial cut and its thickness is a dimension perpendicular to that surface, and a
    trial spacer enters the same way. The femoral component's thickness is subtracted as
    though perpendicular to the same direction, which is exact only when the cuts are
    parallel; at a typical 3 degrees of slope the discrepancy is
    ``thickness x (1 - cos 3 deg)``, about 0.01 mm, which is far below anything the
    landmarks themselves support.

    Returns an empty mapping when no insert thickness has been set. That is the
    pipeline's standing rule and it matters more here than usual: a gap computed against
    an assumed insert would look like a measurement of this knee.

    Only the **extension** gap is reported. The flexion gap needs the posterior condylar
    resection, which this pipeline does not plan, so there is no honest way to state it.
    """
    if insert_thickness_mm is None:
        return {}

    required = ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest")
    if not landmarks.available(*required):
        return {}

    femoral_normal = unit(femoral_normal)
    tibial_normal = unit(tibial_normal)
    occupied = float(femoral_thickness_mm) + float(tray_thickness_mm) \
        + float(insert_thickness_mm)

    gaps = {}
    for compartment, landmark_id in (("medial", required[0]),
                                     ("lateral", required[1])):
        landmark = np.asarray(landmarks.position(landmark_id), dtype=float)
        on_tibial_cut = landmark - np.dot(
            landmark - tibial_point, tibial_normal
        ) * tibial_normal
        space = float(np.dot(femoral_point - on_tibial_cut, tibial_normal))
        gaps[f"extension_gap_{compartment}_mm"] = round(space - occupied, 3)

    gaps["extension_gap_construct_mm"] = round(occupied, 3)
    return gaps


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
        # Projected as a matvec minus a scalar rather than as ``(vertices - seat) @
        # normal``. The two are identical arithmetic, but the second form materialises a
        # full copy of the vertex array first, and on a real segmentation that is 2.5
        # million points. This is the whole cost of re-planning -- the anatomy maths is
        # about 1.7 ms and this was 56 ms per cut -- and re-planning happens on every
        # movement of a control.
        offsets = mesh.vertices @ normal
        offsets -= float(np.dot(seat, normal))
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
