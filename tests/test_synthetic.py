"""The synthetic knee generator itself.

Every accuracy test leans on the generator realising what it is asked for, so the
generator gets tests of its own: signed angles keep their sign, the expected values it
publishes are the ones the planner reports when nothing is assumed, the optional tibial
axis tilt moves only what it should, and turning the whole knee changes nothing a
full-length scan measures.
"""

import itertools

import numpy as np
import pytest

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.metrics import compute_all
from tka_planner.pipeline import measure_from_landmarks
from tka_planner.validation.synthetic import (
    bone_like,
    expected_metrics,
    posed,
    scanner_rotation,
    synthetic_knee,
)

SIDES = ["left", "right"]


def metrics(landmarks):
    return compute_all(landmarks, build_femoral_frame(landmarks),
                       build_tibial_frame(landmarks))


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("slope", [-4.0, -2.0, 0.0, 2.0])
def test_an_anterior_slope_keeps_its_sign(side, slope):
    # The generator used to choose the rotation sense from the target direction alone,
    # which built -2 degrees as +2.
    knee = synthetic_knee(side, posterior_slope_deg=slope)
    assert metrics(knee)["posterior_slope_medial_deg"].value == pytest.approx(slope, abs=1e-6)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("mldfa, mpta, ama, twist, slope", [
    (87.5, 87.0, 6.0, 3.0, 7.0),
    (80.0, 95.0, 2.0, -4.0, -2.0),
    (95.0, 80.0, 11.0, 10.0, 18.0),
])
def test_expected_values_are_what_a_full_length_scan_reports(side, mldfa, mpta, ama,
                                                             twist, slope):
    knee = synthetic_knee(side, mldfa_deg=mldfa, mpta_deg=mpta, ama_deg=ama,
                          condylar_twist_deg=twist, posterior_slope_deg=slope,
                          include_head=True, include_ankle=True)
    reported = metrics(knee)
    for name, expected in expected_metrics(knee).items():
        assert reported[name].value == pytest.approx(expected, abs=1e-6), name


def test_the_tibial_axis_tilt_defaults_to_nothing():
    plain = synthetic_knee("left")
    zero = synthetic_knee("left", tibial_ama_deg=0.0, tibial_ama_sagittal_deg=0.0)
    for landmark in plain:
        if landmark.is_usable:
            np.testing.assert_array_equal(landmark.position_mm,
                                          zero.position(landmark.id))


@pytest.mark.parametrize("side", SIDES)
def test_a_tibial_axis_tilt_moves_only_the_canal_centres(side):
    plain = synthetic_knee(side)
    tilted = synthetic_knee(side, tibial_ama_deg=3.0, tibial_ama_sagittal_deg=4.0)
    for landmark in plain:
        if not landmark.is_usable:
            continue
        same = np.allclose(landmark.position_mm, tilted.position(landmark.id))
        assert same != landmark.id.startswith("tibia.canal_centre"), landmark.id


@pytest.mark.parametrize("side", SIDES)
def test_the_tibial_tilt_is_what_a_knee_only_mpta_and_slope_miss_by(side):
    # On a knee-only scan the tibial mechanical axis is the canal axis, assumed
    # collinear; tilting the canal off it by a known angle shifts MPTA and slope by
    # that angle, and the ankle centre removes the error again.
    tilted = synthetic_knee(side, tibial_ama_deg=3.0, tibial_ama_sagittal_deg=4.0)
    knee_only = metrics(tilted)
    assert abs(knee_only["mpta_deg"].value - 87.0) == pytest.approx(3.0, abs=0.02)
    assert abs(knee_only["posterior_slope_medial_deg"].value - 7.0) == pytest.approx(
        4.0, abs=0.02)
    with_ankle = metrics(synthetic_knee(side, tibial_ama_deg=3.0,
                                        tibial_ama_sagittal_deg=4.0, include_ankle=True))
    assert with_ankle["mpta_deg"].value == pytest.approx(87.0, abs=1e-6)
    assert with_ankle["posterior_slope_medial_deg"].value == pytest.approx(7.0, abs=1e-6)


def test_scanner_rotation_is_a_rotation():
    rotation = scanner_rotation(20.0, -15.0, 30.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("side", SIDES)
def test_a_full_length_knee_measures_the_same_however_it_lies(side):
    knee = synthetic_knee(side, include_head=True, include_ankle=True)
    square = metrics(knee)
    for pose in itertools.product((-20.0, 20.0), (-15.0, 15.0), (-30.0, 30.0)):
        turned = metrics(posed(knee, scanner_rotation(*pose)))
        for name, metric in square.items():
            assert turned[name].value == pytest.approx(metric.value, abs=1e-6), (name, pose)


def test_a_synthetic_knee_runs_through_the_pipeline_with_its_bone_like_meshes():
    knee = synthetic_knee("right")
    case = measure_from_landmarks(
        knee, bone_like(knee, "femur"), bone_like(knee, "tibia"),
        femur_path="f.stl", tibia_path="t.stl", case_id="CASE_SYN", side="right",
    )
    assert case.metrics["mldfa_deg"].value == pytest.approx(87.5, abs=1e-6)
    # The sizing box reads the mesh. An ellipsoid narrows towards its tip, and the box
    # spans the distal end of the bone (sized by the condyles, not the scan), so the
    # width is the ellipsoid's there: less than its widest, more than nothing plausible.
    widest = float(np.ptp(bone_like(knee, "femur").vertices[:, 0]))
    assert 60.0 < case.femoral_measure.ml_mm <= widest
