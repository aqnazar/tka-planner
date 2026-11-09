"""Functional hip centre from tracked pivoting, against motions with a known centre."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.hip_centre import (
    DEFAULT_POLICY,
    FUNCTIONAL_SOURCE,
    PERMISSIVE_POLICY,
    PIVOT_METHOD,
    SPHERE_METHOD,
    HipCentreFitError,
    arc_coverage,
    fit_pivot,
    fit_sphere_pivoting,
    functional_head_centre_set,
    head_centre_landmark,
    poses_from_marker_points,
)
from tka_planner.core.landmarks import LandmarkStatus, overlay_landmarks
from tka_planner.core.metrics import compute_all
from tka_planner.core.provenance import Quality
from tka_planner.report.html import _metric_rows

HIP = np.array([25.0, -1400.0, 180.0])
MARKER = np.array([60.0, 0.0, -420.0])          # array origin, femur frame, from the hip
ARRAY = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 80.0],
                  [-30.0, 0.0, 40.0]])


def rotation(axis, angle_rad):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * k + (1 - np.cos(angle_rad)) * k @ k


def cone(half_angle_deg, n=120):
    """Circumduction once round a cone, as a swing without axial twist."""
    azimuth = 2 * np.pi * np.arange(n) / n
    return np.array([rotation([-np.sin(a), np.cos(a), 0.0], np.radians(half_angle_deg))
                     for a in azimuth])


def flexion(range_deg=60.0, n=120):
    """A single plane of motion: every rotation about the same axis."""
    return np.array([rotation([1.0, 0.0, 0.0], np.radians(a))
                     for a in np.linspace(-range_deg / 2, range_deg / 2, n)])


def track(rotations, noise_mm=0.0, seed=0):
    """Sphere positions and poses of the femoral array for a motion about ``HIP``."""
    translations = HIP + rotations @ MARKER
    spheres = np.einsum("nij,kj->nki", rotations, ARRAY) + translations[:, None, :]
    if noise_mm:
        rng = np.random.default_rng(seed)
        spheres = spheres + rng.normal(0.0, noise_mm / np.sqrt(3), spheres.shape)
    return spheres, translations


class TestExactRecovery:
    @pytest.mark.parametrize("half_angle", [5.0, 20.0, 30.0])
    def test_the_sphere_fit_recovers_the_centre_without_noise(self, half_angle):
        spheres, _ = track(cone(half_angle))
        fit = fit_sphere_pivoting(spheres, policy=PERMISSIVE_POLICY)

        assert fit.method == SPHERE_METHOD
        np.testing.assert_allclose(fit.centre_mm, HIP, atol=1e-6)
        assert fit.rms_residual_mm < 1e-8
        assert fit.converged

    def test_one_tracked_point_is_enough_when_its_path_is_not_a_circle(self):
        spheres, _ = track(cone(25.0))
        fit = fit_sphere_pivoting(spheres[:, 1, :])

        np.testing.assert_allclose(fit.centre_mm, HIP, atol=1e-6)

    @pytest.mark.parametrize("half_angle", [5.0, 20.0, 30.0])
    def test_the_pivot_recovers_the_centre_and_the_femur_fixed_point(self, half_angle):
        rotations = cone(half_angle)
        _, translations = track(rotations)
        fit = fit_pivot(rotations, translations, policy=PERMISSIVE_POLICY)

        assert fit.method == PIVOT_METHOD
        np.testing.assert_allclose(fit.centre_mm, HIP, atol=1e-6)
        # The hip in marker coordinates is minus the marker's offset from the hip.
        np.testing.assert_allclose(fit.pivot_in_marker_mm, -MARKER, atol=1e-6)
        assert fit.radius_mm == pytest.approx(np.linalg.norm(MARKER), abs=1e-6)
        assert fit.rms_residual_mm < 1e-8

    def test_poses_are_recovered_from_marker_points(self):
        rotations = cone(20.0, n=30)
        spheres, translations = track(rotations)
        found, moved, fre = poses_from_marker_points(spheres, ARRAY)

        np.testing.assert_allclose(found, rotations, atol=1e-10)
        np.testing.assert_allclose(moved, translations, atol=1e-8)
        assert fre.max() < 1e-8

    def test_both_agree_on_noisy_data_within_a_millimetre(self):
        rotations = cone(25.0, n=300)
        spheres, _ = track(rotations, noise_mm=0.25, seed=3)
        sphere = fit_sphere_pivoting(spheres)
        pivot = fit_pivot(*poses_from_marker_points(spheres, ARRAY)[:2])

        assert np.linalg.norm(sphere.centre_mm - HIP) < 1.0
        assert np.linalg.norm(pivot.centre_mm - HIP) < 1.0
        # The predicted uncertainty is of the same order as the error, not wildly off.
        assert 0.02 < sphere.centre_sd_mm < 1.0
        assert 0.02 < pivot.centre_sd_mm < 1.0


class TestCameraFrameInvariance:
    """Moving the camera moves the answer with it and changes nothing else."""

    def setup_method(self):
        self.rotations = cone(20.0, n=200)
        self.spheres, _ = track(self.rotations, noise_mm=0.3, seed=11)
        self.q = rotation([0.3, -0.8, 0.5], 1.1)
        self.shift = np.array([-310.0, 75.0, 940.0])
        self.moved = self.spheres @ self.q.T + self.shift

    def test_the_sphere_fit(self):
        before = fit_sphere_pivoting(self.spheres)
        after = fit_sphere_pivoting(self.moved)

        np.testing.assert_allclose(after.centre_mm, self.q @ before.centre_mm
                                   + self.shift, atol=1e-6)
        assert after.rms_residual_mm == pytest.approx(before.rms_residual_mm, rel=1e-6)
        assert after.arc_minor_deg == pytest.approx(before.arc_minor_deg, abs=1e-6)
        assert after.centre_sd_mm == pytest.approx(before.centre_sd_mm, rel=1e-5)

    def test_the_pivot(self):
        before = fit_pivot(*poses_from_marker_points(self.spheres, ARRAY)[:2])
        after = fit_pivot(*poses_from_marker_points(self.moved, ARRAY)[:2])

        np.testing.assert_allclose(after.centre_mm, self.q @ before.centre_mm
                                   + self.shift, atol=1e-6)
        np.testing.assert_allclose(after.pivot_in_marker_mm, before.pivot_in_marker_mm,
                                   atol=1e-6)
        assert after.rms_residual_mm == pytest.approx(before.rms_residual_mm, rel=1e-6)
        assert after.centre_sd_mm == pytest.approx(before.centre_sd_mm, rel=1e-5)


class TestRefusal:
    def test_a_single_plane_of_flexion_is_refused_by_both(self):
        rotations = flexion()
        spheres, translations = track(rotations)

        with pytest.raises(HipCentreFitError, match="one axis"):
            fit_sphere_pivoting(spheres, policy=PERMISSIVE_POLICY)
        with pytest.raises(HipCentreFitError, match="one axis"):
            fit_pivot(rotations, translations, policy=PERMISSIVE_POLICY)

    def test_a_noisy_single_plane_is_refused_rather_than_answered(self):
        rotations = flexion()
        spheres, _ = track(rotations, noise_mm=0.25, seed=5)

        for fit in (lambda: fit_sphere_pivoting(spheres),
                    lambda: fit_pivot(*poses_from_marker_points(spheres, ARRAY)[:2])):
            with pytest.raises(HipCentreFitError) as refused:
                fit()
            assert any("arc too small" in r for r in refused.value.reasons)

    def test_a_small_arc_is_refused_with_the_fit_attached(self):
        rotations = cone(3.0)
        spheres, translations = track(rotations, noise_mm=0.25, seed=2)

        with pytest.raises(HipCentreFitError, match="arc too small") as refused:
            fit_pivot(*poses_from_marker_points(spheres, ARRAY)[:2])
        assert refused.value.fit is not None
        assert refused.value.fit.arc_minor_deg == pytest.approx(3.0, abs=0.2)

    def test_too_few_samples_are_refused(self):
        rotations = cone(30.0, n=12)
        spheres, translations = track(rotations)

        with pytest.raises(HipCentreFitError, match="fewer than"):
            fit_pivot(rotations, translations)

    def test_a_still_marker_is_refused(self):
        spheres = np.repeat(ARRAY[None] + HIP, 50, axis=0)
        with pytest.raises(HipCentreFitError, match="did not move"):
            fit_sphere_pivoting(spheres)

    def test_the_permissive_policy_returns_what_the_default_refuses(self):
        rotations = cone(5.0)
        spheres, translations = track(rotations, noise_mm=0.25, seed=4)
        fit = fit_sphere_pivoting(spheres, policy=PERMISSIVE_POLICY)

        assert DEFAULT_POLICY.reasons(fit)
        with pytest.raises(HipCentreFitError):
            DEFAULT_POLICY.check(fit)


class TestArcCoverage:
    def test_a_circular_cone_scores_its_half_angle_both_ways(self):
        _, translations = track(cone(20.0, n=360))
        minor, major, widest = arc_coverage(translations, HIP)

        assert minor == pytest.approx(20.0, abs=1.0)
        assert major == pytest.approx(20.0, abs=1.0)

    def test_a_flexion_arc_has_no_minor_spread(self):
        _, translations = track(flexion(60.0))
        minor, major, widest = arc_coverage(translations, HIP)

        assert minor < 1.0
        assert major > 15.0
        assert widest == pytest.approx(30.0, abs=1.5)


class TestIntoThePlan:
    def fit(self):
        rotations = cone(25.0, n=200)
        spheres, _ = track(rotations, noise_mm=0.2, seed=8)
        return fit_pivot(*poses_from_marker_points(spheres, ARRAY)[:2])

    def registration(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation([0.2, 0.1, 1.0], 0.7)
        transform[:3, 3] = [12.0, -40.0, 1350.0]
        return transform

    def test_the_landmark_is_measured_and_says_where_it_came_from(self):
        fit = self.fit()
        transform = self.registration()
        landmark = head_centre_landmark(fit, transform, registration="reg-piece-1",
                                        registration_rms_mm=0.4)

        assert landmark.id == "femur.head_centre"
        assert landmark.status is LandmarkStatus.PRESENT
        assert landmark.origin == f"functional_pivot:{PIVOT_METHOD}"
        np.testing.assert_allclose(landmark.position_mm,
                                   transform[:3, :3] @ fit.centre_mm + transform[:3, 3])
        assert landmark.metadata["source"] == FUNCTIONAL_SOURCE == "functional, camera"
        assert landmark.metadata["rms_residual_mm"] == pytest.approx(
            fit.rms_residual_mm, abs=1e-4)
        assert landmark.metadata["registration"] == "reg-piece-1"
        assert landmark.metadata["registration_rms_mm"] == 0.4

    @pytest.mark.parametrize("bad", ["reflection", "scale", "shape"])
    def test_a_non_rigid_registration_is_refused(self, bad):
        transform = self.registration()
        if bad == "reflection":
            transform[:3, 0] *= -1
        elif bad == "scale":
            transform[:3, :3] *= 1.02
        else:
            transform = transform[:3]
        with pytest.raises(ValueError):
            head_centre_landmark(self.fit(), transform)

    def test_a_fit_kept_only_for_study_cannot_enter_a_plan(self):
        rotations = cone(5.0)
        spheres, _ = track(rotations, noise_mm=0.25, seed=4)
        fit = fit_sphere_pivoting(spheres, policy=PERMISSIVE_POLICY)

        with pytest.raises(HipCentreFitError):
            head_centre_landmark(fit, np.eye(4))

    def test_the_metrics_and_the_report_carry_the_source(self):
        knee = synthetic_knee("left")              # knee-only: head out of scan
        truth = synthetic_knee("left", include_head=True).position("femur.head_centre")
        fit = self.fit()
        # A registration that puts the fitted centre on the knee's true head centre.
        transform = np.eye(4)
        transform[:3, 3] = truth - fit.centre_mm
        picks = functional_head_centre_set(fit, transform, case_id=knee.case_id,
                                           side=knee.side, registration="test")
        merged = overlay_landmarks(knee, picks)

        femoral = build_femoral_frame(merged)
        metrics = compute_all(merged, femoral, build_tibial_frame(merged))

        assert femoral.method == "frames.femur.mechanical.v1"
        mldfa = metrics["mldfa_deg"]
        assert mldfa.quality is Quality.MEASURED
        source = mldfa.sources["femur.head_centre"]
        assert source["source"] == "functional, camera"
        assert source["origin"].startswith("functional_pivot:")
        assert source["status"] == "present"
        assert "rms_residual_mm" in source
        assert "femur.head_centre" in mldfa.to_dict()["sources"]
        assert "functional, camera" in _metric_rows({"mldfa_deg": mldfa})

    def test_ordinary_landmarks_record_their_origin_but_no_external_source(self):
        knee = synthetic_knee("right", include_head=True)
        metrics = compute_all(knee, build_femoral_frame(knee), build_tibial_frame(knee))

        mldfa = metrics["mldfa_deg"]
        assert mldfa.sources["femur.head_centre"] == {"status": "present",
                                                      "origin": "synthetic"}
        assert mldfa.external_sources == {}
        assert "Source:" not in _metric_rows(metrics)


class TestEnteredFemurLength:
    """A knee-only scan cannot give the femur's length; the author asked that it can be
    entered (2026-10-15). Entered, it turns the fit's uncertainty into this patient's
    axis uncertainty, and an independent length is the one check pelvic motion fails."""

    def test_the_axis_bound_uses_the_entered_length(self):
        from tka_planner.core.hip_centre import femur_length_check

        record = femur_length_check([0, 0, 0], femur_length_mm=400.0, centre_sd_mm=2.0)
        assert record["femur_length_entered_mm"] == 400.0
        assert record["axis_sd_bound_deg"] == pytest.approx(
            np.degrees(np.arctan(2.0 / 400.0)), abs=1e-4)
        assert "warnings" not in record

    def test_a_pivoted_length_far_from_the_entered_one_is_flagged(self):
        from tka_planner.core.hip_centre import femur_length_check

        record = femur_length_check([0, 0, 450.0], [0, 0, 0], femur_length_mm=430.0)
        assert record["femur_length_pivot_mm"] == 450.0
        assert record["femur_length_difference_mm"] == 20.0
        assert "pelvic motion" in record["warnings"][0]

    def test_a_pivoted_length_that_agrees_is_not_flagged(self):
        from tka_planner.core.hip_centre import femur_length_check

        record = femur_length_check([0, 0, 434.0], [0, 0, 0], femur_length_mm=430.0)
        assert record["femur_length_difference_mm"] == 4.0
        assert "warnings" not in record


@pytest.mark.parametrize("length, code", [(None, None), (455.0, "FEMUR_LENGTH_ENTERED"),
                                          (45.5, "FEMUR_LENGTH_IMPLAUSIBLE")])
def test_an_entered_femur_length_is_checked_and_recorded_as_entered(length, code):
    from tka_planner.core.qc import check_entered_femur_length

    report = check_entered_femur_length(length)
    assert [f.code for f in report.findings] == ([code] if code else [])
