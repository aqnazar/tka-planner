"""Anatomical coordinate frames for the femur and the tibia.

A frame is what turns a bag of coordinates into anatomy. Once one exists, "how deep is
the medial resection" becomes a dot product, and every metric downstream is a projection
or an angle in a plane the frame defines.

Axes are named for what they mean, never for their letter::

    x_anterior       forwards
    y_patient_left   toward the patient's left  (a fixed spatial direction)
    z_proximal       toward the head            (up the limb)

The set is right-handed, and *patient-left* is deliberately not *medial*: medial points
toward the body midline, so it is patient-right on a left knee and patient-left on a
right one. Frames therefore never encode laterality themselves -- they carry a
:class:`~tka_planner.core.sides.Side` and let it answer that question, which is what
keeps side logic out of every metric.

The method ladder
-----------------
Constructing the proximal axis is where this cohort's constraint bites. The femoral
mechanical axis runs from the femoral head centre to the knee centre, and the head is
outside every scan here. So each frame declares an ordered list of methods, and the
first one whose landmarks are available is used:

**Femur**

1. ``frames.femur.mechanical.v1`` -- head centre to knee centre. Quality: measured.
2. ``frames.femur.ama_assumed.v1`` -- fit the diaphyseal axis, then rotate it medially
   in the coronal plane by an assumed mechanical-anatomical angle. Quality: estimated,
   carrying :data:`~tka_planner.core.provenance.POPULATION_FEMORAL_AMA`.

**Tibia**

1. ``frames.tibia.mechanical.v1`` -- ankle centre to knee centre. Quality: measured.
2. ``frames.tibia.anatomical_proxy.v1`` -- the proximal diaphyseal axis used directly.
   Quality: estimated, but a far better estimate than the femoral fallback: the tibial
   mechanical-anatomical angle is roughly 0-2 degrees against the femur's 5-7. That
   asymmetry is a real finding and worth stating plainly -- MPTA degrades gracefully on
   a knee-only scan, and mLDFA does not.

Whichever method runs is recorded on the frame, so a plan always shows which axis
definition produced its numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    angle_between,
    fit_line,
    orient_towards,
    project_out,
    unit,
)
from .landmarks import LandmarkSet
from .provenance import (
    POPULATION_FEMORAL_AMA,
    TIBIAL_AMA_NEGLIGIBLE,
    Assumption,
    Quality,
)
from .sides import Side

__all__ = [
    "AnatomicalFrame",
    "build_femoral_frame",
    "build_tibial_frame",
    "FrameConstructionError",
]


class FrameConstructionError(ValueError):
    """Raised when no method in a frame's ladder can be satisfied."""

    def __init__(self, bone: str, attempts: list[str]):
        self.bone = bone
        self.attempts = attempts
        super().__init__(
            f"Cannot construct a {bone} frame. Methods tried:\n"
            + "\n".join(f"  - {attempt}" for attempt in attempts)
        )


@dataclass(frozen=True)
class AnatomicalFrame:
    """An orthonormal, right-handed frame anchored at a joint centre.

    Carries its own provenance. Two frames with identical axes but different methods
    are not interchangeable, because the metrics derived from them inherit different
    quality tiers.
    """

    bone: str
    side: Side
    origin: np.ndarray
    x_anterior: np.ndarray
    y_patient_left: np.ndarray
    z_proximal: np.ndarray
    method: str
    quality: Quality
    assumptions: tuple[Assumption, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        axes = np.array([self.x_anterior, self.y_patient_left, self.z_proximal])
        if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-6):
            raise ValueError(f"{self.bone} frame axes are not orthonormal.")
        # A left-handed frame would silently mirror every subsequent measurement.
        if not np.isclose(float(np.linalg.det(axes)), 1.0, atol=1e-6):
            raise ValueError(
                f"{self.bone} frame is left-handed (determinant "
                f"{float(np.linalg.det(axes)):.3f}). Axis order must satisfy "
                f"x_anterior x y_patient_left = z_proximal."
            )

    # -- anatomical directions ----------------------------------------

    @property
    def medial(self) -> np.ndarray:
        """Unit vector pointing toward the body midline for this knee."""
        return self.side.medial_direction(self.y_patient_left)

    @property
    def lateral(self) -> np.ndarray:
        return self.side.lateral_direction(self.y_patient_left)

    @property
    def distal(self) -> np.ndarray:
        return -self.z_proximal

    @property
    def posterior(self) -> np.ndarray:
        return -self.x_anterior

    # -- planes, named by the anatomy they contain --------------------

    @property
    def coronal_normal(self) -> np.ndarray:
        """Normal to the coronal plane, which contains the proximal and lateral axes."""
        return self.x_anterior

    @property
    def sagittal_normal(self) -> np.ndarray:
        """Normal to the sagittal plane, which contains the proximal and anterior axes."""
        return self.y_patient_left

    @property
    def transverse_normal(self) -> np.ndarray:
        """Normal to the transverse plane, which contains the anterior and lateral axes."""
        return self.z_proximal

    # -- coordinate conversion ----------------------------------------

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Express world points in frame coordinates as ``(anterior, left, proximal)``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        basis = np.array([self.x_anterior, self.y_patient_left, self.z_proximal])
        return (points - self.origin) @ basis.T

    def to_world(self, local: np.ndarray) -> np.ndarray:
        local = np.atleast_2d(np.asarray(local, dtype=float))
        basis = np.array([self.x_anterior, self.y_patient_left, self.z_proximal])
        return local @ basis + self.origin

    @property
    def matrix(self) -> np.ndarray:
        """The 4x4 frame-to-world transform, as stored in a plan."""
        transform = np.eye(4)
        transform[:3, 0] = self.x_anterior
        transform[:3, 1] = self.y_patient_left
        transform[:3, 2] = self.z_proximal
        transform[:3, 3] = self.origin
        return transform

    def to_dict(self) -> dict:
        return {
            "bone": self.bone,
            "side": self.side.value,
            "origin_mm": [round(float(v), 4) for v in self.origin],
            "x_anterior": [round(float(v), 6) for v in self.x_anterior],
            "y_patient_left": [round(float(v), 6) for v in self.y_patient_left],
            "z_proximal": [round(float(v), 6) for v in self.z_proximal],
            "handedness": "right",
            "method": self.method,
            "quality": self.quality.value,
            "assumptions": [a.id for a in self.assumptions],
            "diagnostics": self.diagnostics,
        }


# ----------------------------------------------------------------------
# Femur
# ----------------------------------------------------------------------


def build_femoral_frame(
    landmarks: LandmarkSet,
    *,
    ama_assumption: Assumption = POPULATION_FEMORAL_AMA,
) -> AnatomicalFrame:
    """Build the femoral frame, taking the best available axis definition.

    The knee centre is the intercondylar notch, *not* the midpoint of the epicondyles.
    That choice is deliberate: the surgical epicondylar axis is already the frame's
    rotational reference, and reusing it for the coronal origin would couple the two
    constructions so that a single mis-picked epicondyle corrupts both. The errors
    would then be correlated, which would flatter the sensitivity analysis and fail in
    a way that is hard to diagnose.
    """
    side = landmarks.side
    attempts: list[str] = []

    knee_centre = landmarks.position("femur.notch_centre")
    if knee_centre is None:
        raise FrameConstructionError(
            "femoral",
            [f"knee centre requires femur.notch_centre "
             f"({landmarks.get('femur.notch_centre').status.value})"],
        )

    z_proximal, method, quality, assumptions, diagnostics = _femoral_proximal_axis(
        landmarks, knee_centre, side, ama_assumption, attempts
    )

    # Rotational reference: the surgical transepicondylar axis, brought into the
    # transverse plane.
    lateral_epicondyle = landmarks.position("femur.epicondyle_lateral")
    medial_sulcus = landmarks.position("femur.epicondyle_medial_sulcus")
    if lateral_epicondyle is None or medial_sulcus is None:
        raise FrameConstructionError(
            "femoral",
            attempts + [
                "rotational reference requires femur.epicondyle_lateral and "
                "femur.epicondyle_medial_sulcus (surgical TEA)"
            ],
        )

    stea_raw = lateral_epicondyle - medial_sulcus
    # Point it patient-left, so the frame's y axis always means the same direction
    # regardless of which end was picked first or which knee this is.
    stea_raw = orient_towards(stea_raw, _patient_left_reference(side, z_proximal))

    # The component removed by projection is the axis's out-of-plane tilt. It is a
    # useful pick-quality signal, so it is measured before being discarded.
    y_patient_left = project_out(stea_raw, z_proximal)
    diagnostics["stea_out_of_plane_deg"] = round(
        90.0 - np.degrees(angle_between(stea_raw, z_proximal)), 3
    )

    x_anterior = unit(np.cross(y_patient_left, z_proximal))

    return AnatomicalFrame(
        bone="femur",
        side=side,
        origin=knee_centre,
        x_anterior=x_anterior,
        y_patient_left=y_patient_left,
        z_proximal=z_proximal,
        method=method,
        quality=quality,
        assumptions=assumptions,
        diagnostics=diagnostics,
    )


def _femoral_proximal_axis(landmarks, knee_centre, side, ama_assumption, attempts):
    """Walk the femoral axis ladder, returning the first satisfiable method."""
    diagnostics: dict = {}

    # -- Method 1: the true mechanical axis ---------------------------
    head_centre = landmarks.position("femur.head_centre")
    if head_centre is not None:
        z_proximal = unit(head_centre - knee_centre)
        diagnostics["mechanical_axis_length_mm"] = round(
            float(np.linalg.norm(head_centre - knee_centre)), 2
        )
        return (z_proximal, "frames.femur.mechanical.v1", Quality.MEASURED, (),
                diagnostics)

    head_status = landmarks.get("femur.head_centre").status
    attempts.append(
        f"frames.femur.mechanical.v1 needs femur.head_centre ({head_status.value})"
    )

    # -- Method 2: diaphyseal axis plus an assumed AMA ----------------
    anatomical = _diaphyseal_axis(landmarks, "femur", knee_centre)
    if anatomical is None:
        attempts.append(
            "frames.femur.ama_assumed.v1 needs femur.canal_centre_distal and "
            "femur.canal_centre_proximal"
        )
        raise FrameConstructionError("femoral", attempts)

    axis, extent_mm = anatomical
    diagnostics["anatomical_axis_baseline_mm"] = round(extent_mm, 2)

    # The femoral shaft leans laterally as it ascends, so its proximal end sits lateral
    # to the femoral head. Recovering the mechanical axis therefore means rotating the
    # anatomical axis medially, in the coronal plane, by the mechanical-anatomical
    # angle. Rotating the wrong way would double the error rather than remove it.
    medial = side.medial_direction(_patient_left_reference(side, axis))
    coronal_normal = unit(np.cross(medial, axis))
    z_proximal = _rotate_about(axis, coronal_normal,
                               np.radians(ama_assumption.value), towards=medial)

    diagnostics["ama_applied_deg"] = ama_assumption.value
    return (z_proximal, "frames.femur.ama_assumed.v1", Quality.ESTIMATED,
            (ama_assumption,), diagnostics)


# ----------------------------------------------------------------------
# Tibia
# ----------------------------------------------------------------------


def build_tibial_frame(
    landmarks: LandmarkSet,
    *,
    rotational_reference: str = "akagi_2004",
    ama_assumption: Assumption = TIBIAL_AMA_NEGLIGIBLE,
) -> AnatomicalFrame:
    """Build the tibial frame.

    The knee centre is the midpoint of the intercondylar tubercles.

    ``rotational_reference`` selects the anteroposterior reference:

    ``akagi_2004``
        PCL attachment midpoint to the medial border of the patellar tendon
        attachment, as originally defined.
    ``medial_third``
        PCL attachment midpoint to the medial third of the tibial tubercle.

    Both are supported and the choice is recorded, because they are frequently conflated
    in the literature yet differ by roughly 2 to 4 degrees of component rotation -- a
    difference large enough that a plan is ambiguous unless it says which was used.
    """
    side = landmarks.side
    attempts: list[str] = []

    spines = landmarks.missing_of("tibia.spine_medial", "tibia.spine_lateral")
    if spines:
        raise FrameConstructionError(
            "tibial", [f"knee centre requires the tibial spines; missing {spines}"]
        )
    knee_centre = np.mean(
        landmarks.require("tibia.spine_medial", "tibia.spine_lateral"), axis=0
    )

    z_proximal, method, quality, assumptions, diagnostics = _tibial_proximal_axis(
        landmarks, knee_centre, ama_assumption, attempts
    )

    anterior_landmark = (
        "tibia.tubercle_patellar_tendon_medial_border"
        if rotational_reference == "akagi_2004"
        else "tibia.tubercle_medial_third"
    )
    posterior_point = landmarks.position("tibia.pcl_insertion_midpoint")
    anterior_point = landmarks.position(anterior_landmark)
    if posterior_point is None or anterior_point is None:
        raise FrameConstructionError(
            "tibial",
            attempts + [
                f"rotational reference {rotational_reference!r} requires "
                f"tibia.pcl_insertion_midpoint and {anterior_landmark}"
            ],
        )

    ap_raw = anterior_point - posterior_point
    x_anterior = project_out(ap_raw, z_proximal)
    diagnostics["rotational_reference"] = rotational_reference
    diagnostics["ap_axis_out_of_plane_deg"] = round(
        90.0 - np.degrees(angle_between(ap_raw, z_proximal)), 3
    )

    y_patient_left = unit(np.cross(z_proximal, x_anterior))
    # The Akagi line's own direction fixes anterior, and the cross product then fixes
    # patient-left. Confirm it really points patient-left rather than trusting the
    # ordering, since a mis-picked pair would otherwise invert the frame silently.
    if float(np.dot(y_patient_left, _patient_left_reference(side, z_proximal))) < 0:
        raise FrameConstructionError(
            "tibial",
            attempts + [
                "the anteroposterior reference points backwards: check that "
                "tibia.pcl_insertion_midpoint and the tubercle landmark are not "
                "swapped"
            ],
        )

    return AnatomicalFrame(
        bone="tibia",
        side=side,
        origin=knee_centre,
        x_anterior=x_anterior,
        y_patient_left=y_patient_left,
        z_proximal=z_proximal,
        method=method,
        quality=quality,
        assumptions=assumptions,
        diagnostics=diagnostics,
    )


def _tibial_proximal_axis(landmarks, knee_centre, ama_assumption, attempts):
    diagnostics: dict = {}

    ankle_centre = landmarks.position("tibia.ankle_centre")
    if ankle_centre is not None:
        z_proximal = unit(knee_centre - ankle_centre)
        diagnostics["mechanical_axis_length_mm"] = round(
            float(np.linalg.norm(knee_centre - ankle_centre)), 2
        )
        return (z_proximal, "frames.tibia.mechanical.v1", Quality.MEASURED, (),
                diagnostics)

    ankle_status = landmarks.get("tibia.ankle_centre").status
    attempts.append(
        f"frames.tibia.mechanical.v1 needs tibia.ankle_centre ({ankle_status.value})"
    )

    anatomical = _diaphyseal_axis(landmarks, "tibia", knee_centre)
    if anatomical is None:
        attempts.append(
            "frames.tibia.anatomical_proxy.v1 needs tibia.canal_centre_proximal and "
            "tibia.canal_centre_distal"
        )
        raise FrameConstructionError("tibial", attempts)

    axis, extent_mm = anatomical
    diagnostics["anatomical_axis_baseline_mm"] = round(extent_mm, 2)
    # Used directly: the tibial mechanical and anatomical axes are near-collinear, so
    # unlike the femur no angular correction is applied. The assumption is still
    # recorded, because "we assumed the correction is zero" is a modelling decision.
    return (axis, "frames.tibia.anatomical_proxy.v1", Quality.ESTIMATED,
            (ama_assumption,), diagnostics)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _diaphyseal_axis(
    landmarks: LandmarkSet, bone: str, knee_centre: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Fit the shaft axis through the canal centres, oriented proximally.

    Returns the axis and the separation between the canal centres. That separation is
    the fit's lever arm, and it matters: angular uncertainty scales inversely with it,
    so a 150 mm femur segment yields a markedly noisier axis than a 240 mm one even
    when both fit tightly. The value is surfaced in the frame's diagnostics rather than
    left implicit.
    """
    distal_id = f"{bone}.canal_centre_distal"
    proximal_id = f"{bone}.canal_centre_proximal"
    if not landmarks.available(distal_id, proximal_id):
        return None

    distal, proximal = landmarks.require(distal_id, proximal_id)
    fit = fit_line(np.array([distal, proximal]))
    axis = orient_towards(fit.direction, proximal - distal)
    return axis, float(np.linalg.norm(proximal - distal))


def _patient_left_reference(side: Side, proximal_axis: np.ndarray) -> np.ndarray:
    """A world direction that is unambiguously patient-left, for orienting axes.

    Frames are built in the LPS patient frame, where +X is patient-left by definition,
    so that is the reference. It is used only to fix the *sign* of an axis, never its
    direction, so a proximal axis that is not exactly world-vertical costs nothing.
    """
    from .sides import LPS_PATIENT_LEFT

    reference = LPS_PATIENT_LEFT
    if abs(float(np.dot(reference, unit(proximal_axis)))) > 0.9:
        raise ValueError(
            "The proximal axis is nearly parallel to the patient-left direction, so "
            "left and right cannot be distinguished. The bone is probably mis-oriented "
            "or the landmarks are in the wrong coordinate system."
        )
    return reference


def _rotate_about(
    vector: np.ndarray, axis: np.ndarray, angle_rad: float, *, towards: np.ndarray
) -> np.ndarray:
    """Rotate ``vector`` about ``axis`` by ``angle_rad``, toward ``towards``.

    The rotation direction is given by the target rather than by a sign convention the
    caller has to get right. Sign conventions are where side-dependent bugs live; asking
    "which way should this tip?" and answering with a direction removes the question.
    """
    vector = unit(vector)
    axis = unit(axis)

    rotated = _rodrigues(vector, axis, angle_rad)
    # If that turned away from the target, the axis pointed the other way.
    if float(np.dot(rotated, towards)) < float(np.dot(vector, towards)):
        rotated = _rodrigues(vector, axis, -angle_rad)
    return unit(rotated)


def _rodrigues(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * float(np.dot(axis, vector)) * (1.0 - cos_a)
    )
