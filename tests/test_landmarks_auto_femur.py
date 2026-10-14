"""Automatic femoral landmarks on a constructed distal femur of any length.

The epicondyles are searched in a band above the distal end of the bone. That band used
to be a fixed fraction of the segment's length -- 15 to 35 per cent -- which is about
22 to 87 mm on a knee-only scan but 67 to 157 mm on a full-length femur, where it lies
on the shaft. A full-length reference femur had its lateral epicondyle placed 157 mm
above the joint. The femur below is the same bone at different lengths: a condylar
block with two posterior condyles and an epicondylar prominence 40 mm up, tapering into
a shaft. Where the scan stops must not move the epicondyles.
"""

import numpy as np
import pytest

from tka_planner.core.landmarks_auto import (
    DISTAL_WINDOW_CONDYLAR_WIDTHS,
    condylar_width_mm,
    distal_window_mm,
    estimate_femoral_landmarks,
)
from tka_planner.core.meshio import Mesh
from tka_planner.core.sides import Side

EPICONDYLE_HEIGHT_MM = 40.0
CONDYLAR_HALF_WIDTH_MM = 36.0
SHAFT_RADIUS_MM = 14.0


def _half_width(z):
    """Mediolateral half-width at height ``z`` above the distal end."""
    prominence = 5.0 * np.exp(-((z - EPICONDYLE_HEIGHT_MM) / 7.0) ** 2)
    # The metaphysis narrows into the shaft from 55 mm upwards.
    taper = np.clip((z - 55.0) / 35.0, 0.0, 1.0)
    block = CONDYLAR_HALF_WIDTH_MM + prominence
    return block + (SHAFT_RADIUS_MM - block) * taper


def _half_depth(z):
    taper = np.clip((z - 55.0) / 35.0, 0.0, 1.0)
    return 30.0 + (SHAFT_RADIUS_MM - 30.0) * taper


def distal_femur(length_mm: float, *, rotation_deg: float = 0.0) -> Mesh:
    """A lofted femur, distal end at z = 0, ``length_mm`` long, in LPS (+Y posterior).

    The posterior half of each section bulges into two condyles either side of the
    midline, so the posterior condylar line -- the estimator's first anchor -- exists.
    """
    angles = np.radians(np.arange(0.0, 360.0, 3.0))
    heights = np.arange(0.0, length_mm + 0.5, 1.0)
    rings = []
    for z in heights:
        a, b = _half_width(z), _half_depth(z)
        x = a * np.cos(angles)
        y = b * np.sin(angles)
        lobes = 1.0 + 0.15 * np.exp(-((np.abs(np.cos(angles)) - 0.5) / 0.2) ** 2)
        y = np.where(y > 0, y * lobes, y)
        rings.append(np.stack([x, y, np.full_like(x, z)], axis=1))
    vertices = np.concatenate(rings)
    if rotation_deg:
        c, s = np.cos(np.radians(rotation_deg)), np.sin(np.radians(rotation_deg))
        vertices = vertices @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T

    n = len(angles)
    faces = []
    for k in range(len(heights) - 1):
        for j in range(n):
            a, b = k * n + j, k * n + (j + 1) % n
            c, d = a + n, b + n
            faces += [[a, b, c], [b, d, c]]
    return Mesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))


def _epicondyles(mesh, side=Side.LEFT):
    found = {lm.id: lm.position_mm for lm in estimate_femoral_landmarks(mesh, side)}
    return (found["femur.epicondyle_lateral"],
            found["femur.epicondyle_medial_prominence"])


def test_a_full_length_femur_has_the_epicondyles_of_its_knee_only_scan():
    short = _epicondyles(distal_femur(150.0))
    long = _epicondyles(distal_femur(450.0))
    for near, far in zip(short, long):
        np.testing.assert_allclose(far, near, atol=1e-9)


def test_the_epicondyles_are_on_the_prominence_not_the_shaft():
    for length in (150.0, 450.0):
        lateral, medial = _epicondyles(distal_femur(length))
        for point in (lateral, medial):
            assert point[2] == pytest.approx(EPICONDYLE_HEIGHT_MM, abs=1.5)
        # The prominence is the widest part of the bone, 41 mm either side.
        assert lateral[0] == pytest.approx(41.0, abs=0.5)
        assert medial[0] == pytest.approx(-41.0, abs=0.5)


def test_every_length_beyond_the_window_is_searched_identically():
    """Past the window the scan's length plays no part at all, to the last bit."""
    reference = distal_femur(300.0)
    window = distal_window_mm(reference)
    assert window < 300.0
    expected = {lm.id: lm.position_mm
                for lm in estimate_femoral_landmarks(reference, Side.RIGHT)
                if lm.position_mm is not None and "canal" not in lm.id}
    for length in (400.0, 600.0):
        mesh = distal_femur(length)
        assert distal_window_mm(mesh) == window
        found = {lm.id: lm.position_mm
                 for lm in estimate_femoral_landmarks(mesh, Side.RIGHT)
                 if lm.position_mm is not None and "canal" not in lm.id}
        assert found.keys() == expected.keys()
        for landmark_id, position in expected.items():
            np.testing.assert_array_equal(found[landmark_id], position)


def test_a_knee_only_segment_keeps_its_own_length_as_the_window():
    mesh = distal_femur(150.0)
    assert distal_window_mm(mesh) == pytest.approx(150.0)


def test_the_condylar_width_does_not_depend_on_how_the_femur_lies_in_the_scanner():
    straight = condylar_width_mm(distal_femur(150.0))
    turned = condylar_width_mm(distal_femur(150.0, rotation_deg=25.0))
    assert straight == pytest.approx(2.0 * CONDYLAR_HALF_WIDTH_MM, abs=0.5)
    assert turned == pytest.approx(straight, rel=2e-3)


def test_the_window_is_a_fixed_number_of_condylar_widths():
    mesh = distal_femur(450.0)
    assert distal_window_mm(mesh) == pytest.approx(
        DISTAL_WINDOW_CONDYLAR_WIDTHS * condylar_width_mm(mesh))


def _world_frame():
    from tka_planner.core.frames import AnatomicalFrame
    from tka_planner.core.provenance import Quality

    return AnatomicalFrame(
        bone="femur", side=Side.LEFT, origin=np.zeros(3),
        x_anterior=np.array([0.0, -1.0, 0.0]), y_patient_left=np.array([1.0, 0.0, 0.0]),
        z_proximal=np.array([0.0, 0.0, 1.0]), method="test", quality=Quality.MEASURED,
    )


def test_the_sizing_block_of_a_long_femur_stops_at_the_window():
    """A quarter of 450 mm reaches the shaft; the sizing box must not."""
    from tka_planner.core.measure import measure_femoral_ml

    frame = _world_frame()
    measured = [measure_femoral_ml(distal_femur(length), frame)
                for length in (300.0, 450.0, 600.0)]
    for one in measured[1:]:
        assert one.ml_mm == measured[0].ml_mm
        assert one.ap_mm == measured[0].ap_mm
        assert one.diagnostics["n_vertices"] == measured[0].diagnostics["n_vertices"]
    assert "window" in measured[0].diagnostics["region"]
    # A quarter of the window stays on the condylar block, below the taper at 55 mm.
    assert 0.25 * distal_window_mm(distal_femur(450.0)) < 55.0


def test_a_knee_only_femur_is_sized_on_a_quarter_of_its_length_as_before():
    from tka_planner.core.measure import measure_femoral_ml

    one = measure_femoral_ml(distal_femur(150.0), _world_frame())
    assert one.diagnostics["region"] == "distal 25% of the segment"
    # Rings are 1 mm apart from z = 0, so a quarter of 150 mm holds rings 0 to 37.
    assert one.diagnostics["n_vertices"] == 38 * 120
