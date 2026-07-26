"""Deformity metrics, each reported with how it was obtained.

Every function here returns a :class:`~tka_planner.core.provenance.Metric` rather than a
number. The envelope carries the definition, the sign convention, the versioned method,
the landmarks consumed, any assumption substituted, and the normal range -- so a report
can show not just that mLDFA is 88.3 degrees but what that means, how it was derived,
and whether the femoral head was actually in the scan.

Sign conventions
----------------
Angles here are *anatomical* angles, defined as opening toward a named side, not signed
rotations. That is deliberate. The classical definitions are asymmetric -- mLDFA is
measured **laterally** and MPTA **medially** -- and any codebase that reduces them to a
signed rotation with a per-side correction will get exactly one of them wrong on right
knees. Instead each metric asks the frame for the anatomical direction it needs, and the
frame asks the :class:`~tka_planner.core.sides.Side`. No metric in this module contains a
side conditional.

The reference ranges are population norms used only to flag a value as unusual. Nothing
is ever clamped or corrected to fit them.
"""

from __future__ import annotations

import numpy as np

from .frames import AnatomicalFrame
from .geometry import angle_between, project_out, signed_angle_in_plane, unit
from .landmarks import LandmarkSet, LandmarkStatus
from .provenance import AUTOMATIC_LANDMARK_ESTIMATE, Assumption, Metric, Quality


__all__ = [
    "mldfa",
    "aldfa",
    "mpta",
    "hka",
    "jlca",
    "condylar_twist_angle",
    "posterior_slope_medial",
    "femoral_mechanical_anatomical_angle",
    "compute_all",
]


def _landmark_quality(
    landmarks: LandmarkSet, landmark_ids: tuple[str, ...]
) -> tuple[Quality, tuple[Assumption, ...]]:
    """Quality implied by *how the landmarks were obtained*.

    A metric is only as good as its weakest input, and that includes the provenance of
    the points themselves. A value computed from machine-estimated landmarks is not a
    measurement however sound the arithmetic, so any metric touching an ``ESTIMATED``
    landmark is demoted and carries
    :data:`~tka_planner.core.provenance.AUTOMATIC_LANDMARK_ESTIMATE`.

    Without this the pipeline would label a number ``measured`` when every point behind
    it had been guessed by a bounding-box heuristic -- exactly the unearned confidence
    this system exists to avoid.
    """
    if any(
        landmarks.get(landmark_id).status is LandmarkStatus.ESTIMATED
        for landmark_id in landmark_ids
    ):
        return Quality.ESTIMATED, (AUTOMATIC_LANDMARK_ESTIMATE,)
    return Quality.MEASURED, ()


def _combined_quality(
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
    landmark_ids: tuple[str, ...],
) -> tuple[Quality, tuple[Assumption, ...]]:
    """Landmark quality combined with the frame's axis-ladder quality.

    Used by metrics that depend on a mechanical axis. Metrics referencing only local
    joint geometry -- aLDFA, condylar twist, JLCA -- call :func:`_landmark_quality`
    instead, because the frame's assumed axis does not enter their arithmetic and
    inheriting its tier would understate them.
    """
    quality, assumptions = _landmark_quality(landmarks, landmark_ids)
    return (
        quality.combine(frame.quality),
        tuple(frame.assumptions) + assumptions,
    )


def _joint_line_direction(
    landmarks: LandmarkSet,
    frame: AnatomicalFrame,
    medial_id: str,
    lateral_id: str,
    *,
    towards_lateral: bool,
) -> np.ndarray:
    """Direction along a joint line, projected into the frame's coronal plane.

    Oriented by asking the frame which way is medial or lateral, so the caller never
    reasons about sides.
    """
    medial_point, lateral_point = landmarks.require(medial_id, lateral_id)
    direction = lateral_point - medial_point
    target = frame.lateral if towards_lateral else frame.medial
    if float(np.dot(direction, target)) < 0:
        direction = -direction
    return project_out(direction, frame.coronal_normal)


# ----------------------------------------------------------------------
# Coronal alignment
# ----------------------------------------------------------------------


MLDFA_DEFINITION = (
    "Mechanical lateral distal femoral angle: the angle opening laterally between the "
    "femoral mechanical axis and the distal condylar joint line, in the coronal plane."
)
MLDFA_SIGN = (
    "90 degrees means the joint line is perpendicular to the mechanical axis. "
    "Below 85 indicates a valgus distal femur; above 90 indicates varus."
)


def mldfa(landmarks: LandmarkSet, frame: AnatomicalFrame) -> Metric:
    """Mechanical lateral distal femoral angle.

    Inherits the femoral frame's quality. On a knee-only scan the mechanical axis is
    derived through an assumed mechanical-anatomical angle, so this is an estimate --
    and the report says so, because the number is otherwise indistinguishable from one
    measured against a real femoral head.
    """
    required = ("femur.condyle_distal_medial", "femur.condyle_distal_lateral")
    missing = landmarks.missing_of(*required)
    if missing:
        return Metric.unavailable(
            "mldfa_deg", "deg", MLDFA_DEFINITION,
            missing=tuple(missing),
            reason="the distal condylar landmarks are needed to define the joint line",
            method="metrics.mldfa.v1",
        )

    joint_lateral = _joint_line_direction(
        landmarks, frame, *required, towards_lateral=True
    )
    axis_proximal = project_out(frame.z_proximal, frame.coronal_normal)
    value = np.degrees(angle_between(axis_proximal, joint_lateral))
    quality, assumptions = _combined_quality(landmarks, frame, required)

    return Metric(
        name="mldfa_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=MLDFA_DEFINITION,
        sign_convention=MLDFA_SIGN,
        method="metrics.mldfa.v1",
        inputs=("frames.femoral.z_proximal",) + required,
        assumptions=assumptions,
        reference_range=(85.0, 90.0),
    )


ALDFA_DEFINITION = (
    "Anatomical lateral distal femoral angle: the angle opening laterally between the "
    "femoral ANATOMICAL (diaphyseal) axis and the distal condylar joint line."
)


def aldfa(landmarks: LandmarkSet, frame: AnatomicalFrame) -> Metric:
    """Anatomical lateral distal femoral angle.

    Worth reporting alongside mLDFA precisely because it is **measurable on a knee-only
    scan while mLDFA is not**: it references the diaphysis, which is in the field of
    view, rather than the femoral head, which is not. On this cohort it is the honest
    coronal number, and mLDFA is the modelled one.
    """
    required = ("femur.condyle_distal_medial", "femur.condyle_distal_lateral")
    canal = ("femur.canal_centre_distal", "femur.canal_centre_proximal")
    missing = landmarks.missing_of(*required, *canal)
    if missing:
        return Metric.unavailable(
            "aldfa_deg", "deg", ALDFA_DEFINITION,
            missing=tuple(missing),
            reason="needs both the distal condyles and the diaphyseal canal centres",
            method="metrics.aldfa.v1",
        )

    distal_canal, proximal_canal = landmarks.require(*canal)
    anatomical_axis = project_out(proximal_canal - distal_canal, frame.coronal_normal)
    joint_lateral = _joint_line_direction(
        landmarks, frame, *required, towards_lateral=True
    )
    value = np.degrees(angle_between(anatomical_axis, joint_lateral))
    # Independent of the frame's axis ladder -- it never touches the femoral head -- so
    # the frame's assumed axis does not enter, but the landmarks' provenance does.
    quality, assumptions = _landmark_quality(landmarks, required + canal)

    return Metric(
        name="aldfa_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=ALDFA_DEFINITION,
        sign_convention=(
            "Normally 79-83 degrees. Relates to mLDFA by the mechanical-anatomical "
            "angle: aLDFA = mLDFA - AMA, since the anatomical axis leans laterally "
            "from the mechanical axis and so closes the lateral angle."
        ),
        method="metrics.aldfa.v1",
        inputs=required + canal,
        assumptions=assumptions,
        reference_range=(79.0, 83.0),
    )


MPTA_DEFINITION = (
    "Medial proximal tibial angle: the angle opening medially between the tibial "
    "mechanical axis and the proximal tibial joint line, in the coronal plane."
)
MPTA_SIGN = (
    "90 degrees means the plateau is perpendicular to the mechanical axis. "
    "Below 85 indicates a varus proximal tibia; above 90 indicates valgus."
)


def mpta(landmarks: LandmarkSet, frame: AnatomicalFrame) -> Metric:
    """Medial proximal tibial angle.

    Measured on the *medial* side against the *distal* direction of the mechanical
    axis, mirroring how mLDFA is measured laterally against the proximal direction.
    The asymmetry is in the classical definitions, not an accident here.
    """
    required = ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest")
    missing = landmarks.missing_of(*required)
    if missing:
        return Metric.unavailable(
            "mpta_deg", "deg", MPTA_DEFINITION,
            missing=tuple(missing),
            reason="the plateau landmarks are needed to define the joint line",
            method="metrics.mpta.v1",
        )

    joint_medial = _joint_line_direction(
        landmarks, frame, *required, towards_lateral=False
    )
    axis_distal = project_out(frame.distal, frame.coronal_normal)
    value = np.degrees(angle_between(axis_distal, joint_medial))
    quality, assumptions = _combined_quality(landmarks, frame, required)

    return Metric(
        name="mpta_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=MPTA_DEFINITION,
        sign_convention=MPTA_SIGN,
        method="metrics.mpta.v1",
        inputs=("frames.tibial.z_proximal",) + required,
        assumptions=assumptions,
        reference_range=(85.0, 90.0),
    )


# ----------------------------------------------------------------------
# Global alignment
# ----------------------------------------------------------------------


HKA_DEFINITION = (
    "Hip-knee-ankle angle: the angle at the knee between the femoral and tibial "
    "mechanical axes, in the coronal plane."
)


def hka(
    landmarks: LandmarkSet,
    femoral_frame: AnatomicalFrame,
    tibial_frame: AnatomicalFrame,
) -> Metric:
    """Hip-knee-ankle angle and its deviation from neutral.

    Reported only when **both** mechanical axes were genuinely measured. This is the
    metric the capability model exists to protect: HKA needs the femoral head and the
    malleoli, neither of which is in a knee-only scan, and a value computed through two
    stacked assumptions would be a statement about population averages wearing the
    clothing of a patient measurement.

    Where the axes are estimated, the metric is emitted as not computable, naming the
    missing landmarks and what imaging would supply them. Both angle conventions are
    stored, each labelled, because the literature uses both.
    """
    missing = [
        landmark_id
        for landmark_id, available in (
            ("femur.head_centre", femoral_frame.quality is Quality.MEASURED),
            ("tibia.ankle_centre", tibial_frame.quality is Quality.MEASURED),
        )
        if not available
    ]
    if missing:
        return Metric.unavailable(
            "hka_deviation_deg", "deg", HKA_DEFINITION,
            missing=tuple(missing),
            reason=(
                "the hip and ankle centres are outside a knee-only field of view, so "
                "neither mechanical axis is measured. Deriving HKA through assumed "
                "axes would report a population average as a patient measurement."
            ),
            would_require=(
                "full-limb imaging: a low-dose CT topogram, EOS biplanar radiography, "
                "or a long-leg standing radiograph registered to the distal femur"
            ),
            method="metrics.hka.v1",
        )

    coronal_normal = femoral_frame.coronal_normal
    femoral_proximal = project_out(femoral_frame.z_proximal, coronal_normal)
    tibial_distal = project_out(tibial_frame.distal, coronal_normal)

    angle_deg = 180.0 - np.degrees(angle_between(femoral_proximal, -tibial_distal))

    # Deviation signed so that positive is valgus, expressed by asking the frame which
    # way is lateral rather than by a per-side sign flip.
    signed = np.degrees(
        signed_angle_in_plane(femoral_proximal, -tibial_distal, coronal_normal)
    )
    lateral_reference = np.degrees(
        signed_angle_in_plane(femoral_proximal, femoral_frame.lateral, coronal_normal)
    )
    deviation = signed * (1.0 if lateral_reference > 0 else -1.0)

    return Metric(
        name="hka_deviation_deg",
        value=round(float(deviation), 3),
        unit="deg",
        quality=Quality.MEASURED,
        definition=HKA_DEFINITION,
        sign_convention=(
            "Deviation from neutral: 0 is a straight limb, positive is valgus, "
            "negative is varus. The 180-based angle is in diagnostics as "
            "'hka_angle_deg', where below 180 is varus."
        ),
        method="metrics.hka.v1",
        inputs=("femur.head_centre", "femur.notch_centre",
                "tibia.spine_medial", "tibia.spine_lateral", "tibia.ankle_centre"),
        reference_range=(-3.0, 3.0),
        diagnostics={"hka_angle_deg": round(float(180.0 - abs(deviation)), 3)
                     if deviation < 0 else round(float(180.0 + abs(deviation)), 3)},
    )


JLCA_DEFINITION = (
    "Joint line convergence angle: the angle between the distal femoral joint line and "
    "the proximal tibial joint line, in the coronal plane."
)


def jlca(
    landmarks: LandmarkSet,
    femoral_frame: AnatomicalFrame,
    tibial_frame: AnatomicalFrame,
) -> Metric:
    """Joint line convergence angle.

    Computable on this cohort even though HKA is not, because it compares two joint
    lines to each other and never references the hip or the ankle. It is measurable at
    all only because femur and tibia arrive already registered in the shared CT frame --
    the relationship the legacy pipeline discarded by recentring each bone on its own
    bounding box and reconstructing the gap from a fixed 20 mm constant.
    """
    femoral_required = ("femur.condyle_distal_medial", "femur.condyle_distal_lateral")
    tibial_required = ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest")
    missing = landmarks.missing_of(*femoral_required, *tibial_required)
    if missing:
        return Metric.unavailable(
            "jlca_deg", "deg", JLCA_DEFINITION,
            missing=tuple(missing),
            reason="both joint lines are required",
            method="metrics.jlca.v1",
        )

    coronal_normal = femoral_frame.coronal_normal
    femoral_line = _joint_line_direction(
        landmarks, femoral_frame, *femoral_required, towards_lateral=True
    )
    tibial_line = _joint_line_direction(
        landmarks, tibial_frame, *tibial_required, towards_lateral=True
    )

    signed = np.degrees(
        signed_angle_in_plane(femoral_line, tibial_line, coronal_normal)
    )
    # The sign is fixed by asking the frame which rotational sense carries the proximal
    # axis toward lateral, so the convention holds on both knees without a conditional.
    lateral_reference = np.degrees(
        signed_angle_in_plane(femoral_frame.z_proximal, femoral_frame.lateral,
                              coronal_normal)
    )
    value = signed * (1.0 if lateral_reference > 0 else -1.0)

    quality, assumptions = _landmark_quality(
        landmarks, femoral_required + tibial_required
    )

    return Metric(
        name="jlca_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=JLCA_DEFINITION,
        sign_convention=(
            "Positive when the joint space opens laterally (apex medial). Typically "
            "0-2 degrees."
        ),
        method="metrics.jlca.v1",
        inputs=femoral_required + tibial_required,
        assumptions=assumptions,
        reference_range=(-2.0, 3.0),
    )


# ----------------------------------------------------------------------
# Rotation and slope
# ----------------------------------------------------------------------


TWIST_DEFINITION = (
    "Condylar twist angle: the angle in the transverse plane between the posterior "
    "condylar line and the surgical transepicondylar axis."
)


def condylar_twist_angle(
    landmarks: LandmarkSet, frame: AnatomicalFrame
) -> Metric:
    """Condylar twist angle, measured rather than assumed.

    This replaces the legacy ``TEA_CORRECTION_DEG = 3.0``, whose docstring conflated two
    different angles. The roughly 3 degree figure in the literature is this one -- the
    posterior condylar line against the surgical epicondylar axis -- not the difference
    between the surgical and anatomical epicondylar axes, which is 1 to 2 degrees. The
    constant was also a poor one to fix: reported means differ markedly between sexes,
    so the between-patient variation exceeds the value itself.

    With real epicondylar and posterior condylar landmarks no constant is needed.
    """
    required = (
        "femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus",
        "femur.condyle_posterior_medial", "femur.condyle_posterior_lateral",
    )
    missing = landmarks.missing_of(*required)
    if missing:
        return Metric.unavailable(
            "condylar_twist_deg", "deg", TWIST_DEFINITION,
            missing=tuple(missing),
            reason="needs the surgical epicondylar axis and both posterior condyles",
            method="metrics.condylar_twist.v1",
        )

    lateral_epi, medial_sulcus, posterior_medial, posterior_lateral = (
        landmarks.require(*required)
    )

    stea = _oriented_laterally(lateral_epi - medial_sulcus, frame)
    posterior_line = _oriented_laterally(posterior_lateral - posterior_medial, frame)

    stea = project_out(stea, frame.transverse_normal)
    posterior_line = project_out(posterior_line, frame.transverse_normal)

    signed = np.degrees(
        signed_angle_in_plane(stea, posterior_line, frame.z_proximal)
    )
    # Positive means the sTEA is externally rotated relative to the posterior condylar
    # line. External rotation carries the lateral side anteriorly, so rather than hard-
    # coding a sign per knee, the frame is asked which rotational sense that is: the
    # angle from lateral to anterior about the proximal axis has opposite sign on the
    # two sides, which is exactly the side dependence being cancelled.
    external_sense = np.sign(
        signed_angle_in_plane(frame.lateral, frame.x_anterior, frame.z_proximal)
    )
    value = signed * float(external_sense)
    quality, assumptions = _combined_quality(landmarks, frame, required)
    if not any(a.id == "automatic_landmark_estimate" for a in assumptions):
        quality, assumptions = Quality.MEASURED, ()
    else:
        quality = Quality.ESTIMATED
        assumptions = tuple(a for a in assumptions
                            if a.id == "automatic_landmark_estimate")

    return Metric(
        name="condylar_twist_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=TWIST_DEFINITION,
        sign_convention=(
            "Positive when the surgical epicondylar axis is externally rotated "
            "relative to the posterior condylar line, which is the usual finding."
        ),
        method="metrics.condylar_twist.v1",
        inputs=required,
        assumptions=assumptions,
        reference_range=(0.0, 7.0),
    )


SLOPE_DEFINITION = (
    "Native posterior tibial slope of the medial compartment: the inclination of the "
    "medial plateau in the sagittal plane, relative to the perpendicular to the tibial "
    "axis."
)


def posterior_slope_medial(
    landmarks: LandmarkSet, frame: AnatomicalFrame
) -> Metric:
    """Native medial posterior slope, measured per patient.

    Replaces the legacy fixed ``TIBIA_POSTERIOR_SLOPE_DEG = 5.0``, which was applied as
    a bare number with no stated reference axis. Both facts recorded here matter: slope
    differs between the medial and lateral compartments by 1 to 2 degrees, and the value
    shifts by several degrees depending on whether it is referenced to the mechanical
    axis or the proximal anatomical axis. A slope quoted without both is ambiguous.
    """
    required = ("tibia.plateau_medial_anterior", "tibia.plateau_medial_posterior")
    missing = landmarks.missing_of(*required)
    if missing:
        return Metric.unavailable(
            "posterior_slope_medial_deg", "deg", SLOPE_DEFINITION,
            missing=tuple(missing),
            reason="needs the anterior and posterior rim of the medial plateau",
            method="metrics.posterior_slope.v1",
        )

    anterior_rim, posterior_rim = landmarks.require(*required)
    rim_line = anterior_rim - posterior_rim
    if float(np.dot(rim_line, frame.x_anterior)) < 0:
        rim_line = -rim_line
    rim_line = project_out(rim_line, frame.sagittal_normal)

    # Angle above the transverse plane; positive means anterior is higher, which is a
    # posterior-down slope.
    value = np.degrees(np.arcsin(np.clip(
        float(np.dot(rim_line, frame.z_proximal)), -1.0, 1.0
    )))

    quality, assumptions = _combined_quality(landmarks, frame, required)

    return Metric(
        name="posterior_slope_medial_deg",
        value=round(float(value), 3),
        unit="deg",
        quality=quality,
        definition=SLOPE_DEFINITION,
        sign_convention=(
            "Positive is posterior-down, the normal direction. Typically 5-10 degrees."
        ),
        method="metrics.posterior_slope.v1",
        inputs=("frames.tibial.z_proximal",) + required,
        assumptions=assumptions,
        reference_range=(0.0, 15.0),
        diagnostics={"compartment": "medial",
                     "reference_axis": frame.method},
    )


AMA_DEFINITION = (
    "Femoral mechanical-anatomical angle: the angle between the femoral mechanical "
    "axis and the anatomical axis of the diaphysis, in the coronal plane."
)


def femoral_mechanical_anatomical_angle(
    landmarks: LandmarkSet, frame: AnatomicalFrame
) -> Metric:
    """The patient's own mechanical-anatomical angle, when it can be measured.

    Only computable with the femoral head present, which on this cohort it is not --
    which is precisely why it has to be assumed in the first place. Reporting it when
    full-limb data is available closes the loop: it is the direct check on whether the
    6 degree population value was reasonable for a given patient.
    """
    canal = ("femur.canal_centre_distal", "femur.canal_centre_proximal")
    missing = landmarks.missing_of("femur.head_centre", *canal)
    if missing:
        return Metric.unavailable(
            "femoral_ama_deg", "deg", AMA_DEFINITION,
            missing=tuple(missing),
            reason=(
                "needs the femoral head to define the mechanical axis; this is the "
                "very quantity that must be assumed when the head is out of scan"
            ),
            would_require="full-limb imaging including the hip",
            method="metrics.femoral_ama.v1",
        )

    distal_canal, proximal_canal = landmarks.require(*canal)
    anatomical = project_out(proximal_canal - distal_canal, frame.coronal_normal)
    mechanical = project_out(frame.z_proximal, frame.coronal_normal)

    quality, assumptions = _landmark_quality(landmarks, ("femur.head_centre",) + canal)

    return Metric(
        name="femoral_ama_deg",
        value=round(float(np.degrees(angle_between(anatomical, mechanical))), 3),
        unit="deg",
        quality=quality,
        definition=AMA_DEFINITION,
        sign_convention="Unsigned magnitude. Typically 5-7 degrees.",
        method="metrics.femoral_ama.v1",
        inputs=("femur.head_centre", "femur.notch_centre") + canal,
        assumptions=assumptions,
        reference_range=(3.0, 9.0),
    )


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def compute_all(
    landmarks: LandmarkSet,
    femoral_frame: AnatomicalFrame,
    tibial_frame: AnatomicalFrame,
) -> dict[str, Metric]:
    """Compute every metric, returning them keyed by name.

    Nothing is skipped on failure: a metric that cannot be computed is still returned,
    carrying what is missing and why. The report iterates this dictionary, so an
    omitted entry would be an invisible gap whereas an unavailable one is a visible,
    explained absence.
    """
    metrics = [
        mldfa(landmarks, femoral_frame),
        aldfa(landmarks, femoral_frame),
        femoral_mechanical_anatomical_angle(landmarks, femoral_frame),
        mpta(landmarks, tibial_frame),
        hka(landmarks, femoral_frame, tibial_frame),
        jlca(landmarks, femoral_frame, tibial_frame),
        condylar_twist_angle(landmarks, femoral_frame),
        posterior_slope_medial(landmarks, tibial_frame),
    ]
    return {metric.name: metric for metric in metrics}


def _oriented_laterally(vector: np.ndarray, frame: AnatomicalFrame) -> np.ndarray:
    """Flip a mediolateral vector so it points laterally on this knee."""
    vector = unit(vector)
    return -vector if float(np.dot(vector, frame.lateral)) < 0 else vector
