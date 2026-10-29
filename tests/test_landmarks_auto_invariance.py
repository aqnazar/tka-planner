"""The automatic landmarks describe the bone, not the scan of it.

Paper experiment E2 found the estimators reading the scanner's axes: the tibial AP
reference turned 17 degrees with a 10 degree turn of the leg, a distal condyle jumped
30 mm under a 30 degree tilt, and a 100 mm shift with no turn at all moved a plateau low
point 12.5 mm, because the articular grid was anchored at the scanner origin. Every
search now runs in the bone's own axes, so a rigid motion of the scan must move every
landmark with the bone -- to rounding, not to a tolerance -- and a mirror must mirror
them.

The constructed bones carry a little seeded roughness, as a segmented surface does, so
that no two vertices tie exactly in any search; a tie broken by rounding would otherwise
make an exactly symmetric toy shape look unstable for reasons no real bone has.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks_auto import (
    NOTCH_BAND_MM,
    _notch_roof,
    estimate_femoral_landmarks,
    estimate_tibial_landmarks,
    femoral_axes,
    tibial_axes,
)
from tka_planner.core.meshio import Mesh
from tka_planner.core.sides import Side

from .test_landmarks_auto import proximal_tibia
from .test_landmarks_auto_femur import distal_femur

ROUGHNESS_MM = 0.05


def _rough(mesh: Mesh, seed: int) -> Mesh:
    rng = np.random.default_rng(seed)
    return Mesh(vertices=mesh.vertices + rng.normal(scale=ROUGHNESS_MM,
                                                    size=mesh.vertices.shape),
                faces=mesh.faces)


def _rotation(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _moved(mesh: Mesh, rotation, shift) -> Mesh:
    return Mesh(vertices=mesh.vertices @ rotation.T + shift, faces=mesh.faces)


MOTIONS = {
    "shift_off_grid": (np.eye(3), np.array([100.7, -37.3, 12.9])),
    "sagittal_+30": (_rotation([1, 0, 0], 30.0), np.zeros(3)),
    "sagittal_-30": (_rotation([1, 0, 0], -30.0), np.zeros(3)),
    "coronal_+30": (_rotation([0, 1, 0], 30.0), np.zeros(3)),
    "coronal_-30": (_rotation([0, 1, 0], -30.0), np.zeros(3)),
    "axial_+30": (_rotation([0, 0, 1], 30.0), np.zeros(3)),
    "axial_-30": (_rotation([0, 0, 1], -30.0), np.zeros(3)),
    "oblique_25_and_shift": (_rotation([0.3, -0.5, 0.8], 25.0),
                             np.array([-55.0, 80.0, 31.0])),
}


@pytest.fixture(scope="module")
def femur():
    return _rough(distal_femur(150.0), seed=1)


@pytest.fixture(scope="module")
def tibia():
    return _rough(proximal_tibia("left"), seed=2)


def _positions(landmarks):
    return {lm.id: lm.position_mm for lm in landmarks if lm.position_mm is not None}


@pytest.mark.parametrize("motion", sorted(MOTIONS))
def test_the_femoral_landmarks_move_with_the_femur(femur, motion):
    rotation, shift = MOTIONS[motion]
    expected = _positions(estimate_femoral_landmarks(femur, Side.LEFT))
    found = _positions(estimate_femoral_landmarks(_moved(femur, rotation, shift),
                                                  Side.LEFT))
    assert found.keys() == expected.keys()
    for landmark_id, position in expected.items():
        np.testing.assert_allclose(found[landmark_id], rotation @ position + shift,
                                   atol=1e-6, err_msg=landmark_id)


@pytest.mark.parametrize("motion", sorted(MOTIONS))
def test_the_tibial_landmarks_move_with_the_tibia(tibia, motion):
    rotation, shift = MOTIONS[motion]
    expected = _positions(estimate_tibial_landmarks(tibia, Side.LEFT))
    found = _positions(estimate_tibial_landmarks(_moved(tibia, rotation, shift),
                                                 Side.LEFT))
    assert found.keys() == expected.keys()
    for landmark_id, position in expected.items():
        np.testing.assert_allclose(found[landmark_id], rotation @ position + shift,
                                   atol=1e-6, err_msg=landmark_id)


def test_the_bone_axes_turn_with_the_bone(femur, tibia):
    rotation, shift = MOTIONS["oblique_25_and_shift"]
    for find, mesh in ((lambda m: femoral_axes(m, Side.LEFT), femur), (tibial_axes, tibia)):
        before, after = find(mesh), find(_moved(mesh, rotation, shift))
        np.testing.assert_allclose(after.rotation, before.rotation @ rotation.T, atol=1e-9)
        np.testing.assert_allclose(after.origin, rotation @ before.origin + shift,
                                   atol=1e-9)


def test_the_axes_are_right_handed_and_in_lps_order(femur, tibia):
    for axes in (femoral_axes(femur, Side.LEFT), tibial_axes(tibia)):
        np.testing.assert_allclose(axes.rotation @ axes.rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(axes.rotation) == pytest.approx(1.0)
        # A bone lying square in the scanner keeps the scanner's senses.
        assert axes.patient_left[0] > 0.9 and axes.posterior[1] > 0.9
        assert axes.superior[2] > 0.9


def test_the_femoral_working_axis_is_the_shaft_turned_medially(femur):
    """A straight vertical shaft: the working axis leans 6 degrees from it, its
    proximal end towards the midline -- patient-right on a left knee."""
    left, right = femoral_axes(femur, Side.LEFT), femoral_axes(femur, Side.RIGHT)
    for axes, medial_sign in ((left, -1.0), (right, 1.0)):
        lean = np.degrees(np.arctan2(axes.superior[0], axes.superior[2]))
        assert medial_sign * lean == pytest.approx(6.0, abs=0.3)


@pytest.mark.parametrize("bone", ["femur", "tibia"])
def test_a_mirrored_knee_has_mirrored_landmarks(femur, tibia, bone):
    mirror = np.diag([-1.0, 1.0, 1.0])
    estimate, mesh = ((estimate_femoral_landmarks, femur) if bone == "femur"
                      else (estimate_tibial_landmarks, tibia))
    expected = _positions(estimate(mesh, Side.LEFT))
    # A reflection reverses the winding; flip it back so the surface faces out.
    mirrored = Mesh(vertices=mesh.vertices @ mirror.T, faces=mesh.faces[:, ::-1])
    found = _positions(estimate(mirrored, Side.RIGHT))
    for landmark_id, position in expected.items():
        np.testing.assert_allclose(found[landmark_id], mirror @ position, atol=1e-6,
                                   err_msg=landmark_id)


def test_the_pcl_footprint_is_on_the_midline_not_on_a_condyle():
    """The backs of the condyles bulge either side of the PCL fossa. Version 1 took the
    most posterior point of a 32 mm band, which is on the flank of a condyle at the
    band's edge; the footprint is behind the eminence, between the spines."""
    tibia = proximal_tibia("left")
    vertices = tibia.vertices.copy()
    back = vertices[:, 1] > 0
    x = vertices[back, 0]
    # One condyle bulges further than the other, as they do.
    bulge = 1.0 + np.where(x > 0, 0.16, 0.10) * np.exp(-((np.abs(x) - 20.0) / 6.0) ** 2)
    vertices[back, 1] *= bulge
    found = _positions(estimate_tibial_landmarks(
        _rough(Mesh(vertices=vertices, faces=tibia.faces), seed=3), Side.LEFT))
    spines = 0.5 * (found["tibia.spine_medial"] + found["tibia.spine_lateral"])
    assert abs(found["tibia.pcl_insertion_midpoint"][0] - spines[0]) <= 2.5
    assert found["tibia.pcl_insertion_midpoint"][1] > 20.0     # at the back


def _hook(step=0.25):
    """The midline sagittal outline of a distal femur, as a ribbon of points 6 mm wide.

    Front: the trochlear groove comes down from z = 40 to a flat stretch 10 mm high
    between y = -4 and y = +4; behind it the notch roof climbs steeply to the
    intercondylar line at z = 30, y = 14, and the posterior cortex runs back up to z = 40.
    """
    outline = []
    for y in np.arange(-24.0, -4.0, step):            # trochlea, falling
        outline.append((y, 10.0 + 30.0 * ((-4.0 - y) / 20.0) ** 2))
    for y in np.arange(-4.0, 4.0, step):              # the flat roof
        outline.append((y, 10.0 + 0.02 * y))
    for z in np.arange(10.0, 30.0, step):             # notch roof, climbing
        outline.append((4.0 + 10.0 * (z - 10.0) / 20.0, z))
    for z in np.arange(30.0, 40.0, step):             # posterior cortex
        outline.append((14.0 - 0.3 * (z - 30.0), z))
    points = [(x, y, z) for x in np.arange(-3.0, 3.01, 0.5) for y, z in outline]
    return np.array(points)


def test_the_notch_roof_is_the_middle_of_its_lowest_stretch():
    roof = _notch_roof(_hook(), 0.0, np.array([1.0, 0.0, 0.0]))
    assert roof[2] == pytest.approx(10.0, abs=NOTCH_BAND_MM)
    assert abs(roof[1]) < 2.0         # the middle of the flat, not one end of it
    assert abs(roof[0]) < 0.1


def test_the_notch_roof_is_not_the_ceiling_of_the_search():
    """Version 1 returned the most proximal point near the midline: here z = 40."""
    roof = _notch_roof(_hook(), 0.0, np.array([1.0, 0.0, 0.0]))
    assert roof[2] < 15.0
