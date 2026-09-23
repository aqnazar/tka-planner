"""Automatic tibial landmarks on a constructed proximal tibia.

The block below has two dished plateaus, an eminence between them and vertical sides
running well below the search slab. Version 1 of the dish estimator took the lowest
point in the slab, which on this shape -- as on every real tibia -- is the slab floor on
the side of the bone, whatever the plateaus look like.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks_auto import estimate_tibial_landmarks
from tka_planner.core.meshio import Mesh
from tka_planner.core.sides import Side

MEDIAL_DISH_MM = -6.0
LATERAL_DISH_MM = -3.0
EMINENCE_MM = 4.0
DISH_CENTRE_X = 20.0


def _top(x, y, side):
    """Height of the joint surface. Medial is patient-right (-X) on a left knee."""
    medial_sign = -1.0 if side == "left" else 1.0
    medial_x, lateral_x = medial_sign * DISH_CENTRE_X, -medial_sign * DISH_CENTRE_X
    medial = MEDIAL_DISH_MM + 0.005 * ((x - medial_x) ** 2 + y ** 2)
    lateral = LATERAL_DISH_MM + 0.005 * ((x - lateral_x) ** 2 + y ** 2)
    # Falls to -10 mm away from the midline, below both dishes, so it only shapes the
    # middle of the plateau.
    eminence = (EMINENCE_MM + 10.0) * np.exp(-(x / 4.0) ** 2) \
        * np.exp(-(y / 12.0) ** 2) - 10.0
    return np.maximum(np.minimum(medial, lateral), eminence)


def proximal_tibia(side="left", bottom=-80.0):
    """A closed heightfield block: the joint surface on top, flat sides and base."""
    xs = np.arange(-40.0, 40.01, 1.0)
    ys = np.arange(-25.0, 25.01, 1.0)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    top = np.stack([gx, gy, _top(gx, gy, side)], axis=-1)
    nx, ny = gx.shape

    vertices = [top.reshape(-1, 3)]
    index = np.arange(nx * ny).reshape(nx, ny)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
            faces += [[a, b, c], [a, c, d]]

    # The sides: each boundary vertex of the top joined straight down to the base.
    ring = ([index[i, 0] for i in range(nx)] + [index[nx - 1, j] for j in range(ny)]
            + [index[i, ny - 1] for i in reversed(range(nx))]
            + [index[0, j] for j in reversed(range(ny))])
    flat = vertices[0]
    base_start = len(flat)
    base = flat[ring].copy()
    base[:, 2] = bottom
    vertices.append(base)
    for k in range(len(ring) - 1):
        a, b = ring[k], ring[k + 1]
        c, d = base_start + k + 1, base_start + k
        faces += [[a, b, c], [a, c, d]]
    return Mesh(vertices=np.vstack(vertices), faces=np.array(faces))


@pytest.fixture(params=["left", "right"])
def side(request):
    return request.param


def landmarks_for(side):
    found = estimate_tibial_landmarks(proximal_tibia(side), Side.parse(side))
    return {landmark.id: landmark for landmark in found}


def test_the_plateau_low_points_are_on_the_joint_surface(side):
    landmarks = landmarks_for(side)

    medial = landmarks["tibia.plateau_medial_lowest"].position_mm
    lateral = landmarks["tibia.plateau_lateral_lowest"].position_mm

    assert medial[2] == pytest.approx(MEDIAL_DISH_MM, abs=1.0)
    assert lateral[2] == pytest.approx(LATERAL_DISH_MM, abs=1.0)


def test_the_plateau_low_points_are_in_their_own_compartments(side):
    landmarks = landmarks_for(side)
    medial_sign = -1.0 if side == "left" else 1.0

    medial = landmarks["tibia.plateau_medial_lowest"].position_mm
    lateral = landmarks["tibia.plateau_lateral_lowest"].position_mm

    assert medial[0] == pytest.approx(medial_sign * DISH_CENTRE_X, abs=4.0)
    assert lateral[0] == pytest.approx(-medial_sign * DISH_CENTRE_X, abs=4.0)


def test_the_plateau_low_points_never_sit_on_the_side_of_the_bone(side):
    """The regression: version 1 returned the slab floor, 22 mm below the top."""
    landmarks = landmarks_for(side)
    for landmark_id in ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"):
        landmark = landmarks[landmark_id]
        assert landmark.origin == "auto:plateau_dish.v2"
        assert landmark.position_mm[2] > -10.0


def test_the_medial_rim_points_bound_the_medial_plateau(side):
    landmarks = landmarks_for(side)
    anterior = landmarks["tibia.plateau_medial_anterior"].position_mm
    posterior = landmarks["tibia.plateau_medial_posterior"].position_mm

    # LPS: anterior is -Y. Both on the top surface, not down the sides.
    assert anterior[1] < -15.0 and posterior[1] > 15.0
    for point in (anterior, posterior):
        assert point[2] == pytest.approx(
            float(_top(point[0], point[1], side)), abs=1.0)
