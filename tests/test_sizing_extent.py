"""The femoral sizing extent must not be set by the one or two worst vertices.

A bounding box is the distance between two vertices, so a single stray vertex moves it by
its whole excursion, and surface noise always widens it: on the cohort 0.5 mm of vertex
noise put 2.1 to 3.1 mm on the femoral ML and changed the chart size on five of eleven
knees. The extent is now taken between the 0.1st and 99.9th percentiles, which on the
clean cohort changed no size.
"""

import numpy as np

from tka_planner.core.frames import AnatomicalFrame
from tka_planner.core.measure import EXTENT_PERCENTILE, measure_femoral_ml
from tka_planner.core.meshio import Mesh

from .test_landmarks_auto_femur import _world_frame, distal_femur


def _clean():
    return distal_femur(150.0)


def _ml(mesh: Mesh, frame: AnatomicalFrame, **kwargs) -> float:
    return measure_femoral_ml(mesh, frame, **kwargs).ml_mm


def test_the_default_extent_is_the_approved_percentile():
    assert EXTENT_PERCENTILE == 0.1
    one = measure_femoral_ml(_clean(), _world_frame())
    assert one.method == "measure.femur.frame_aligned_distal_extent.v2"
    assert one.diagnostics["extent_percentile"] == 0.1


def test_a_single_stray_vertex_does_not_move_the_extent():
    frame = _world_frame()
    clean = _clean()
    vertices = np.array(clean.vertices, dtype=float, copy=True)
    # The frame's patient-left is world +X. One vertex at the distal end, 15 mm out.
    lowest = int(np.argmin(vertices[:, 2] - 1e-3 * vertices[:, 0]))
    vertices[lowest, 0] += 15.0
    spiked = Mesh(vertices=vertices, faces=clean.faces)

    assert _ml(spiked, frame, extent_percentile=0.0) - _ml(clean, frame,
                                                           extent_percentile=0.0) > 5.0
    # One rank of shift is all it can cause; on this coarse mesh that is ~0.15 mm.
    assert abs(_ml(spiked, frame) - _ml(clean, frame)) < 0.5


def test_surface_noise_widens_the_percentile_extent_far_less_than_the_box():
    """Averaged over three noise draws: this bone has few vertices at its widest
    section, so a single draw says little. The cohort figures are in the module doc."""
    frame = _world_frame()
    clean = _clean()
    box_growth, robust_growth = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        noisy = Mesh(vertices=clean.vertices + rng.normal(scale=1.0,
                                                          size=clean.vertices.shape),
                     faces=clean.faces)
        box_growth.append(_ml(noisy, frame, extent_percentile=0.0)
                          - _ml(clean, frame, extent_percentile=0.0))
        robust_growth.append(_ml(noisy, frame) - _ml(clean, frame))
    assert np.mean(box_growth) > 1.0
    assert abs(np.mean(robust_growth)) < 0.5 * np.mean(box_growth)


def test_a_clean_bone_measures_within_a_millimetre_of_its_box():
    frame = _world_frame()
    clean = _clean()
    box = measure_femoral_ml(clean, frame, extent_percentile=0.0)
    robust = measure_femoral_ml(clean, frame)
    assert 0.0 <= box.ml_mm - robust.ml_mm < 1.0
    assert 0.0 <= box.ap_mm - robust.ap_mm < 1.0
