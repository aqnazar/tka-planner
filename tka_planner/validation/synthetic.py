"""Synthetic knees whose anatomy is known by construction.

Testing anatomical measurement is circular unless the ground truth comes from somewhere
other than the code under test. Recording whatever a function returns and asserting it
does not change catches regressions but never catches a formula that was wrong from the
start -- and a sign error in a coronal angle is exactly the kind of mistake that looks
entirely plausible in its output.

So these builders work backwards. You state the mLDFA, the MPTA, the
mechanical-anatomical angle and the condylar twist you want, and they place landmarks
that realise those angles exactly. The measured value is then compared against the
number that was an *input*, and a wrong formula or a flipped sign fails immediately.

Everything is built directly in the LPS patient frame (+X patient-left, +Y posterior,
+Z superior), the same frame the real segmentations arrive in, so the fixtures exercise
the same code path as real data.

This lives in the package rather than beside the tests because the paper's evaluation
harness sweeps it too (experiment E1), and the Monte Carlo tests perturb it; the test
suite imports it through ``tests/synthetic.py``.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.landmarks import Landmark, LandmarkSet, LandmarkStatus
from tka_planner.core.meshio import Mesh
from tka_planner.core.sides import LPS_PATIENT_LEFT, LPS_SUPERIOR, Side

__all__ = [
    "synthetic_knee",
    "mirror_landmarks",
    "posed",
    "scanner_rotation",
    "bone_like",
    "expected_metrics",
]

# Reference anatomy, roughly mid-population.
DEFAULT_MLDFA_DEG = 87.5
DEFAULT_MPTA_DEG = 87.0
DEFAULT_AMA_DEG = 6.0
DEFAULT_TWIST_DEG = 3.0
DEFAULT_SLOPE_DEG = 7.0
DEFAULT_FEMORAL_WIDTH_MM = 75.0
DEFAULT_TIBIAL_WIDTH_MM = 72.0


def _rodrigues(vector, axis, angle_rad):
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * float(np.dot(axis, vector)) * (1.0 - cos_a)
    )


def _rotate_towards(vector, axis, angle_rad, towards):
    """Rotate about ``axis`` by ``angle_rad`` in whichever sense moves toward
    ``towards``, so callers never have to reason about a sign convention.

    A negative angle turns *away* from ``towards``. Choosing the sense by the target
    alone used to discard the sign, so a requested anterior (negative) slope of -2
    degrees was built as +2 and reported back as a 4 degree error by the E1 sweep.
    """
    if angle_rad < 0:
        return _rotate_towards(vector, axis, -angle_rad, -np.asarray(towards))
    rotated = _rodrigues(vector, axis, angle_rad)
    if float(np.dot(rotated, towards)) < float(np.dot(vector, towards)):
        rotated = _rodrigues(vector, axis, -angle_rad)
    return rotated / np.linalg.norm(rotated)


def _present(landmark_id, position):
    return Landmark(
        id=landmark_id,
        position_mm=np.asarray(position, dtype=float),
        status=LandmarkStatus.PRESENT,
        origin="synthetic",
    )


def _out_of_scan(landmark_id):
    return Landmark(
        id=landmark_id,
        status=LandmarkStatus.OUT_OF_SCAN,
        reason="knee-only field of view",
    )


def synthetic_knee(
    side: "str | Side" = "left",
    *,
    mldfa_deg: float = DEFAULT_MLDFA_DEG,
    mpta_deg: float = DEFAULT_MPTA_DEG,
    ama_deg: float = DEFAULT_AMA_DEG,
    condylar_twist_deg: float = DEFAULT_TWIST_DEG,
    posterior_slope_deg: float = DEFAULT_SLOPE_DEG,
    femoral_width_mm: float = DEFAULT_FEMORAL_WIDTH_MM,
    tibial_width_mm: float = DEFAULT_TIBIAL_WIDTH_MM,
    tibial_ama_deg: float = 0.0,
    tibial_ama_sagittal_deg: float = 0.0,
    include_head: bool = False,
    include_ankle: bool = False,
    case_id: str = "CASE_SYNTHETIC",
) -> LandmarkSet:
    """Build a landmark set realising the requested anatomy exactly.

    Both mechanical axes are made exactly vertical (world +Z), so every constructed
    angle is expressed purely through where the landmarks sit relative to that axis.
    A knee built this way has an HKA of exactly 180 degrees.

    ``include_head`` and ``include_ankle`` default to ``False`` because that is the real
    situation for this cohort: knee-only scans. Turn them on to exercise the measured
    branch of the axis ladder.

    The anatomical (diaphyseal) axes are placed by rotating the mechanical axis
    *laterally* by ``ama_deg``. That is the anatomically correct relationship -- the
    femoral shaft leans laterally as it ascends -- and it makes the test meaningful:
    the frame builder must rotate medially by the same angle to recover the mechanical
    axis, so a rotation in the wrong direction fails by twice the angle rather than
    passing quietly.

    ``tibial_ama_deg`` and ``tibial_ama_sagittal_deg`` tilt the tibial diaphyseal axis
    away from the mechanical one -- distal end toward lateral, and toward anterior --
    which the planner assumes to be zero on a knee-only scan. Both default to zero, the
    planner's assumption; set them to see what that assumption costs MPTA and slope.
    """
    side = Side.parse(side)
    proximal = LPS_SUPERIOR
    lateral = side.lateral_direction(LPS_PATIENT_LEFT)
    medial = -lateral
    anterior = np.array([0.0, -1.0, 0.0])  # -Y is anterior in LPS

    # Put the knee where a real one sits: off the midline, deep in scanner coordinates.
    femoral_knee_centre = 80.0 * LPS_PATIENT_LEFT * side.lateral_sign + np.array(
        [0.0, 0.0, -550.0]
    )
    if side is Side.RIGHT:
        femoral_knee_centre = np.array([-80.0, 0.0, -550.0])
    else:
        femoral_knee_centre = np.array([80.0, 0.0, -550.0])

    landmarks: list[Landmark] = []

    # ---------------- Femur ----------------

    landmarks.append(_present("femur.notch_centre", femoral_knee_centre))

    if include_head:
        landmarks.append(
            _present("femur.head_centre", femoral_knee_centre + 400.0 * proximal)
        )
    else:
        landmarks.append(_out_of_scan("femur.head_centre"))

    # Diaphyseal axis: the mechanical axis tilted laterally by the AMA.
    coronal_normal = np.cross(medial, proximal)
    femoral_anatomical = _rotate_towards(
        proximal, coronal_normal, np.radians(ama_deg), towards=lateral
    )
    landmarks.append(
        _present("femur.canal_centre_distal",
                 femoral_knee_centre + 60.0 * femoral_anatomical)
    )
    landmarks.append(
        _present("femur.canal_centre_proximal",
                 femoral_knee_centre + 200.0 * femoral_anatomical)
    )

    # Surgical transepicondylar axis, exactly perpendicular to the mechanical axis so
    # the frame's out-of-plane diagnostic should read zero.
    half_width = femoral_width_mm / 2.0
    landmarks.append(
        _present("femur.epicondyle_lateral", femoral_knee_centre + half_width * lateral)
    )
    landmarks.append(
        _present("femur.epicondyle_medial_sulcus",
                 femoral_knee_centre + half_width * medial)
    )
    landmarks.append(
        _present("femur.epicondyle_medial_prominence",
                 femoral_knee_centre + (half_width + 4.0) * medial)
    )

    # Distal condyles: the joint line makes exactly mldfa_deg with the mechanical axis,
    # measured on the lateral side.
    joint_lateral = (
        np.cos(np.radians(mldfa_deg)) * proximal
        + np.sin(np.radians(mldfa_deg)) * lateral
    )
    condylar_plane = femoral_knee_centre - 25.0 * proximal
    landmarks.append(
        _present("femur.condyle_distal_lateral", condylar_plane + half_width * joint_lateral)
    )
    landmarks.append(
        _present("femur.condyle_distal_medial", condylar_plane - half_width * joint_lateral)
    )

    # Posterior condylar axis, rotated from the sTEA in the transverse plane. The sign
    # is chosen so that a positive twist means the sTEA is externally rotated relative
    # to the posterior condylar line, which is the standard convention.
    external_rotation_sign = -side.lateral_sign
    posterior_axis = _rodrigues(
        lateral, proximal, external_rotation_sign * np.radians(condylar_twist_deg)
    )
    posterior_plane = femoral_knee_centre - 30.0 * anterior
    landmarks.append(
        _present("femur.condyle_posterior_lateral",
                 posterior_plane + half_width * posterior_axis)
    )
    landmarks.append(
        _present("femur.condyle_posterior_medial",
                 posterior_plane - half_width * posterior_axis)
    )

    landmarks.append(
        _present("femur.trochlear_groove_anterior",
                 femoral_knee_centre + 35.0 * anterior + 10.0 * proximal)
    )

    # ---------------- Tibia ----------------

    tibial_knee_centre = femoral_knee_centre - 8.0 * proximal
    half_tibial = tibial_width_mm / 2.0

    landmarks.append(
        _present("tibia.spine_medial", tibial_knee_centre + 6.0 * medial)
    )
    landmarks.append(
        _present("tibia.spine_lateral", tibial_knee_centre + 6.0 * lateral)
    )

    if include_ankle:
        landmarks.append(
            _present("tibia.ankle_centre", tibial_knee_centre - 380.0 * proximal)
        )
    else:
        landmarks.append(_out_of_scan("tibia.ankle_centre"))

    # Tibial anatomical axis: collinear with the mechanical axis by default, matching
    # the near-zero tibial mechanical-anatomical angle the planner assumes. Tilting it
    # moves the canal centres only, so the mechanical axis -- and therefore every
    # constructed angle -- stays where it was.
    tibial_distal = _rotate_towards(
        -proximal, coronal_normal, np.radians(tibial_ama_deg), towards=lateral
    )
    tibial_distal = _rotate_towards(
        tibial_distal, lateral, np.radians(tibial_ama_sagittal_deg), towards=anterior
    )
    landmarks.append(
        _present("tibia.canal_centre_proximal", tibial_knee_centre + 60.0 * tibial_distal)
    )
    landmarks.append(
        _present("tibia.canal_centre_distal", tibial_knee_centre + 200.0 * tibial_distal)
    )

    # Akagi line: PCL attachment behind, patellar tendon insertion in front.
    landmarks.append(
        _present("tibia.pcl_insertion_midpoint", tibial_knee_centre - 18.0 * anterior)
    )
    landmarks.append(
        _present("tibia.tubercle_patellar_tendon_medial_border",
                 tibial_knee_centre + 42.0 * anterior - 35.0 * proximal)
    )
    landmarks.append(
        _present("tibia.tubercle_medial_third",
                 tibial_knee_centre + 42.0 * anterior - 35.0 * proximal
                 + 4.0 * medial)
    )

    # Plateau: the joint line makes exactly mpta_deg with the mechanical axis, measured
    # medially against the distal direction.
    joint_medial = (
        np.cos(np.radians(mpta_deg)) * (-proximal)
        + np.sin(np.radians(mpta_deg)) * medial
    )
    plateau_plane = tibial_knee_centre - 6.0 * proximal
    landmarks.append(
        _present("tibia.plateau_medial_lowest", plateau_plane + half_tibial * joint_medial)
    )
    landmarks.append(
        _present("tibia.plateau_lateral_lowest", plateau_plane - half_tibial * joint_medial)
    )

    # Medial plateau rim, tilted to realise the requested posterior slope. Positive
    # slope means the posterior rim sits lower than the anterior one.
    sagittal_normal = lateral
    slope_direction = _rotate_towards(
        anterior, sagittal_normal, np.radians(posterior_slope_deg), towards=proximal
    )
    medial_compartment = plateau_plane + (half_tibial * 0.5) * medial
    landmarks.append(
        _present("tibia.plateau_medial_anterior",
                 medial_compartment + 22.0 * slope_direction)
    )
    landmarks.append(
        _present("tibia.plateau_medial_posterior",
                 medial_compartment - 22.0 * slope_direction)
    )

    landmarks.append(
        _present("fibula.head_apex",
                 tibial_knee_centre + 30.0 * lateral - 15.0 * proximal)
    )

    return LandmarkSet(
        case_id=case_id,
        side=side,
        coordinate_system="LPS",
        landmarks=landmarks,
        source={"kind": "synthetic", "constructed": {
            "mldfa_deg": mldfa_deg,
            "mpta_deg": mpta_deg,
            "ama_deg": ama_deg,
            "condylar_twist_deg": condylar_twist_deg,
            "posterior_slope_deg": posterior_slope_deg,
            "femoral_width_mm": femoral_width_mm,
            "tibial_width_mm": tibial_width_mm,
            "tibial_ama_deg": tibial_ama_deg,
            "tibial_ama_sagittal_deg": tibial_ama_sagittal_deg,
        }},
    )


def expected_metrics(landmarks: LandmarkSet) -> dict[str, float]:
    """The value every metric must report on a knee built by :func:`synthetic_knee`.

    Derived from the construction, never from the code under test:

    * mLDFA, MPTA, condylar twist and slope are inputs.
    * aLDFA = mLDFA - AMA, because the shaft leans laterally off the mechanical axis.
    * JLCA = MPTA - mLDFA. Both mechanical axes are vertical, so each joint line's
      lateral end rises by (90 - its angle); the space opens laterally (positive JLCA)
      when the tibial line rises less than the femoral one, i.e. MPTA exceeds mLDFA.
    * HKA deviation is zero: the two mechanical axes are collinear by construction.
    * The femoral AMA is the input AMA.

    These are the *true* values. A knee-only scan cannot see the femoral head, so the
    planner reports mLDFA through an assumed AMA and is off by the difference; that is
    the modelled error the provenance tiers exist to label, not a formula error.
    """
    built = landmarks.source["constructed"]
    mldfa, mpta = built["mldfa_deg"], built["mpta_deg"]
    return {
        "mldfa_deg": mldfa,
        "aldfa_deg": mldfa - built["ama_deg"],
        "femoral_ama_deg": built["ama_deg"],
        "mpta_deg": mpta,
        "hka_deviation_deg": 0.0,
        "jlca_deg": mpta - mldfa,
        "condylar_twist_deg": built["condylar_twist_deg"],
        "posterior_slope_medial_deg": built["posterior_slope_deg"],
    }


def scanner_rotation(flexion_deg: float = 0.0, abduction_deg: float = 0.0,
                     axial_deg: float = 0.0) -> np.ndarray:
    """A rotation matrix for how a leg might lie in the scanner, in LPS.

    Applied as axial (about superior), then abduction (about anterior-posterior), then
    flexion (about patient-left). The signs are spatial, not anatomical: the sweep this
    serves asks only whether a measurement changes when the whole knee is turned, and
    every angle is swept both ways.
    """
    def about(axis, angle_deg):
        axis = np.asarray(axis, dtype=float)
        angle = np.radians(angle_deg)
        k = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)

    return (about((1.0, 0.0, 0.0), flexion_deg)
            @ about((0.0, 1.0, 0.0), abduction_deg)
            @ about((0.0, 0.0, 1.0), axial_deg))


def posed(landmarks: LandmarkSet, rotation: np.ndarray) -> LandmarkSet:
    """The same knee turned rigidly about its femoral knee centre.

    Real knees do not lie square in the scanner, and nothing the planner measures should
    depend on how they lay. Turning a synthetic knee and measuring it again is the direct
    test. The construction record is kept, because the anatomy is unchanged.
    """
    rotation = np.asarray(rotation, dtype=float)
    centre = landmarks.position("femur.notch_centre")
    moved = []
    for landmark in landmarks:
        position = landmark.position_mm
        if position is not None:
            position = (position - centre) @ rotation.T + centre
        moved.append(Landmark(
            id=landmark.id, position_mm=position, status=landmark.status,
            origin=landmark.origin, reason=landmark.reason,
            metadata=dict(landmark.metadata),
        ))
    return LandmarkSet(
        case_id=landmarks.case_id, side=landmarks.side,
        coordinate_system=landmarks.coordinate_system, landmarks=moved,
        source={**landmarks.source, "posed": np.round(rotation, 9).tolist()},
    )


def bone_like(landmarks: LandmarkSet, prefix: str, *, margin_mm: float = 12.0,
              segments: int = 48, rings: int = 40) -> Mesh:
    """A dense ellipsoid enclosing one bone's landmarks.

    Measurement takes a cross-section through a thin slab and needs vertices spread
    through it, so a bounding box with eight corners is not enough. An ellipsoid over
    the landmark cloud gives a mesh the real measurement functions can work on, so a
    synthetic knee can go through :func:`~tka_planner.pipeline.measure_from_landmarks`
    exactly as a real one does. Only sizing reads it; no metric does.
    """
    from tka_planner.geom import mesh as gm

    points = np.array([one.position_mm for one in landmarks
                       if one.is_usable and one.id.startswith(prefix)])
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    radii = (high - low) / 2.0 + margin_mm
    unit_sphere = gm.uv_sphere(1.0, segments=segments, rings=rings)
    return Mesh(vertices=unit_sphere.vertices * radii + centre, faces=unit_sphere.faces)


def mirror_landmarks(landmarks: LandmarkSet) -> LandmarkSet:
    """Reflect a case through the sagittal plane and flip its side.

    The result is the contralateral knee with identical anatomy, which is the basis of
    the mirror-invariance tests: every scalar metric must come out the same, because
    reflecting a knee does not change how varus it is.

    Reflection negates the patient-left component, which in LPS is X.
    """
    reflected = []
    for landmark in landmarks:
        position = landmark.position_mm
        if position is not None:
            position = position * np.array([-1.0, 1.0, 1.0])
        reflected.append(
            Landmark(
                id=landmark.id,
                position_mm=position,
                status=landmark.status,
                origin=landmark.origin,
                reason=landmark.reason,
                metadata=dict(landmark.metadata),
            )
        )

    return LandmarkSet(
        case_id=landmarks.case_id + "_MIRRORED",
        side=landmarks.side.opposite,
        coordinate_system=landmarks.coordinate_system,
        landmarks=reflected,
        source={**landmarks.source, "mirrored_from": landmarks.case_id},
    )
