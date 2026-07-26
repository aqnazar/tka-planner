"""Geometry primitives, tested against shapes whose answers are known analytically.

Every assertion here compares against a value that was an *input* to the construction,
never against the function's own output. A test that fits a sphere to points sampled
from a sphere and checks the radius it was given back is a real test; one that records
whatever the fitter produced and checks it does not change is a regression guard, not a
correctness proof. Both have value, but only the first catches a wrong formula.
"""

import numpy as np
import pytest

from tka_planner.core.geometry import (
    angle_between,
    bounding_box,
    canonical_direction,
    cross_section_centroid,
    extremal_point,
    fit_line,
    fit_plane,
    fit_sphere,
    fit_sphere_robust,
    orient_towards,
    principal_axes,
    project_out,
    project_to_plane,
    regional_mask,
    signed_angle_in_plane,
    unit,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def sphere_points(centre, radius, n=500, cap_angle_deg=180.0, seed=0):
    """Sample a spherical cap. ``cap_angle_deg=180`` gives the whole sphere.

    A partial cap is the realistic femoral-head case: CT captures the articular
    surface, not the whole ball.
    """
    rng = np.random.default_rng(seed)
    max_polar = np.radians(cap_angle_deg) / 2.0
    # Uniform on the cap by area.
    cos_polar = rng.uniform(np.cos(max_polar), 1.0, n)
    polar = np.arccos(cos_polar)
    azimuth = rng.uniform(0, 2 * np.pi, n)

    directions = np.column_stack([
        np.sin(polar) * np.cos(azimuth),
        np.sin(polar) * np.sin(azimuth),
        np.cos(polar),
    ])
    return np.asarray(centre) + radius * directions


# ----------------------------------------------------------------------
# Direction handling
# ----------------------------------------------------------------------


class TestDirections:
    def test_canonical_direction_is_sign_invariant(self):
        """An axis and its negation must canonicalise identically."""
        vector = np.array([0.3, -0.9, 0.2])
        assert np.allclose(canonical_direction(vector), canonical_direction(-vector))

    def test_canonical_direction_returns_unit_vector(self):
        result = canonical_direction(np.array([3.0, 0.0, 4.0]))
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_canonical_direction_makes_dominant_component_positive(self):
        result = canonical_direction(np.array([0.1, -0.99, 0.05]))
        assert result[1] > 0

    def test_orient_towards_flips_when_opposed(self):
        direction = np.array([0.0, 0.0, 1.0])
        assert np.allclose(orient_towards(direction, np.array([0, 0, -5.0])),
                           [0, 0, -1])
        assert np.allclose(orient_towards(direction, np.array([0, 0, 5.0])),
                           [0, 0, 1])

    def test_unit_rejects_zero_vector(self):
        with pytest.raises(ValueError, match="zero-length"):
            unit(np.zeros(3))


# ----------------------------------------------------------------------
# Sphere fitting
# ----------------------------------------------------------------------


class TestSphereFit:
    def test_recovers_full_sphere(self):
        centre, radius = np.array([12.0, -34.0, 56.0]), 23.0
        fit = fit_sphere(sphere_points(centre, radius))

        assert np.allclose(fit.centre, centre, atol=1e-6)
        assert np.isclose(fit.radius_mm, radius, atol=1e-6)
        assert fit.rms_mm < 1e-6

    def test_recovers_partial_cap(self):
        """The femoral-head case: only an articular cap is available.

        The algebraic fit is biased on partial data; the Gauss-Newton refinement is
        what makes a cap usable. Without refinement this test fails.
        """
        centre, radius = np.array([100.0, 20.0, -450.0]), 22.5
        points = sphere_points(centre, radius, n=800, cap_angle_deg=120.0)

        fit = fit_sphere(points)
        assert np.allclose(fit.centre, centre, atol=1e-5)
        assert np.isclose(fit.radius_mm, radius, atol=1e-5)

    def test_rms_exposes_a_bad_fit(self):
        """Fitting a sphere to something that is not one must show up in the RMS.

        This is the mechanism scan-coverage detection relies on: a femoral head that is
        genuinely present fits tightly, and anything else does not.
        """
        rng = np.random.default_rng(3)
        blob = rng.normal(scale=15.0, size=(400, 3))

        assert fit_sphere(blob).rms_mm > 1.5

    def test_noisy_fit_stays_within_noise(self):
        centre, radius = np.array([0.0, 0.0, 0.0]), 20.0
        rng = np.random.default_rng(7)
        points = sphere_points(centre, radius, n=1000) + rng.normal(
            scale=0.2, size=(1000, 3)
        )

        fit = fit_sphere(points)
        assert np.linalg.norm(fit.centre - centre) < 0.1
        assert abs(fit.radius_mm - radius) < 0.1

    def test_surface_spread_distinguishes_a_ball_from_a_patch(self):
        """Low residual alone proves nothing: a small patch fits some sphere perfectly.

        Coverage is the property that separates a genuine femoral head from an
        incidentally curved piece of bone, so it is measured explicitly.
        """
        centre, radius = np.zeros(3), 20.0

        whole = fit_sphere(sphere_points(centre, radius, cap_angle_deg=360.0))
        hemisphere = fit_sphere(sphere_points(centre, radius, cap_angle_deg=180.0))
        patch = fit_sphere(sphere_points(centre, radius, cap_angle_deg=30.0))

        assert whole.surface_spread > 0.9
        assert 0.35 < hemisphere.surface_spread < 0.65
        assert patch.surface_spread < 0.1

        # All three fit tightly -- which is exactly the point.
        for fit in (whole, hemisphere, patch):
            assert fit.rms_mm < 1e-4

    def test_rejects_coplanar_points(self):
        """Points on a circle admit infinitely many spheres, so refuse to pick one.

        Least squares happily returns one with a positive radius, which is why this
        needs an explicit rank check rather than a sanity check on the result.
        """
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        circle = np.column_stack([
            20.0 * np.cos(angles), 20.0 * np.sin(angles), np.zeros(40)
        ])
        with pytest.raises(ValueError, match="coplanar or collinear"):
            fit_sphere(circle)

    def test_rejects_collinear_points(self):
        line = np.column_stack([
            np.linspace(0, 10, 20), np.linspace(0, 5, 20), np.zeros(20)
        ])
        with pytest.raises(ValueError, match="coplanar or collinear"):
            fit_sphere(line)


# ----------------------------------------------------------------------
# Plane and line fitting
# ----------------------------------------------------------------------


class TestRobustSphereFit:
    """Fitting the head when the window also contains neck, trochanter and shaft."""

    @staticmethod
    def head_among_clutter(seed=0):
        """A sphere surrounded by a cylinder, as a proximal femur window really is."""
        rng = np.random.default_rng(seed)
        centre, radius = np.array([80.0, 0.0, -180.0]), 23.0
        head = sphere_points(centre, radius, n=3000, cap_angle_deg=240.0, seed=seed)

        # Shaft running away from the head.
        z = rng.uniform(-70, 0, 1500)
        theta = rng.uniform(0, 2 * np.pi, 1500)
        shaft = np.column_stack([
            80.0 + 14.0 * np.cos(theta),
            14.0 * np.sin(theta),
            -190.0 + z,
        ])
        return np.vstack([head, shaft]), centre, radius

    def test_recovers_the_sphere_despite_surrounding_anatomy(self):
        points, centre, radius = self.head_among_clutter()

        plain = fit_sphere(points)
        robust = fit_sphere_robust(points)

        # The plain fit is dragged off by the shaft; the robust one is not.
        assert np.linalg.norm(robust.centre - centre) < 1.0
        assert abs(robust.radius_mm - radius) < 1.0
        assert robust.rms_mm < plain.rms_mm

    def test_reports_how_many_points_it_kept(self):
        points, _, _ = self.head_among_clutter()
        robust = fit_sphere_robust(points)

        assert robust.n_inliers is not None
        assert 0.0 < robust.inlier_fraction < 1.0
        assert robust.n_points == len(points)

    def test_leaves_clean_data_essentially_untouched(self):
        centre, radius = np.array([5.0, 5.0, 5.0]), 18.0
        points = sphere_points(centre, radius, n=800)
        robust = fit_sphere_robust(points)

        assert np.allclose(robust.centre, centre, atol=1e-4)
        assert robust.inlier_fraction > 0.95

    def test_is_deterministic(self):
        """A randomised fitter would make plan hashes irreproducible."""
        points, _, _ = self.head_among_clutter()
        rng = np.random.default_rng(31)

        first = fit_sphere_robust(points)
        second = fit_sphere_robust(rng.permutation(points))
        assert np.allclose(first.centre, second.centre, atol=1e-9)
        assert np.isclose(first.radius_mm, second.radius_mm, atol=1e-9)

    def test_a_bare_cylinder_still_fits_badly(self):
        """Trimming must not let a shaft masquerade as a head.

        A cylinder wraps fully around its axis, so it scores high on surface spread;
        the residual is what exposes it.
        """
        rng = np.random.default_rng(41)
        z = rng.uniform(-100, 100, 4000)
        theta = rng.uniform(0, 2 * np.pi, 4000)
        cylinder = np.column_stack([
            14.0 * np.cos(theta), 14.0 * np.sin(theta), z
        ])

        assert fit_sphere_robust(cylinder).rms_mm > 2.0


class TestPlaneFit:
    def test_recovers_known_plane(self):
        rng = np.random.default_rng(1)
        normal = unit(np.array([0.2, -0.3, 0.9]))
        origin = np.array([5.0, 5.0, 5.0])

        # Two in-plane directions spanning the plane.
        u = unit(np.cross(normal, [1.0, 0.0, 0.0]))
        v = np.cross(normal, u)
        coeffs = rng.uniform(-50, 50, size=(300, 2))
        points = origin + coeffs[:, :1] * u + coeffs[:, 1:] * v

        fit = fit_plane(points)
        assert np.allclose(np.abs(fit.normal), np.abs(normal), atol=1e-9)
        assert fit.rms_mm < 1e-9

    def test_residual_reflects_out_of_plane_scatter(self):
        rng = np.random.default_rng(2)
        points = np.column_stack([
            rng.uniform(-50, 50, 400),
            rng.uniform(-50, 50, 400),
            rng.normal(scale=0.5, size=400),
        ])
        assert 0.3 < fit_plane(points).rms_mm < 0.7


class TestLineFit:
    def test_recovers_known_axis(self):
        direction = unit(np.array([0.1, 0.05, 1.0]))
        base = np.array([30.0, -10.0, -500.0])
        t = np.linspace(-80, 80, 200)
        points = base + np.outer(t, direction)

        fit = fit_line(points)
        assert np.allclose(np.abs(fit.direction), np.abs(direction), atol=1e-9)
        assert fit.rms_mm < 1e-9

    def test_extent_reports_available_lever_arm(self):
        """``extent_mm`` is what distinguishes a 150 mm femur from a 240 mm one."""
        direction = np.array([0.0, 0.0, 1.0])
        t = np.linspace(-35, 35, 100)
        fit = fit_line(np.outer(t, direction))

        assert np.isclose(fit.extent_mm, 70.0, atol=1e-9)

    def test_short_baseline_is_angularly_noisier(self):
        """Same scatter over a shorter span gives a worse-determined axis.

        This is why the plan reports the shaft length used for a diaphyseal fit: RMS
        alone would make both look equally good.
        """
        rng = np.random.default_rng(11)
        truth = np.array([0.0, 0.0, 1.0])

        def angular_error(half_length):
            t = np.linspace(-half_length, half_length, 200)
            points = np.outer(t, truth) + rng.normal(scale=0.5, size=(200, 3))
            return angle_between(fit_line(points).direction, truth)

        errors_short = [angular_error(10.0) for _ in range(15)]
        errors_long = [angular_error(100.0) for _ in range(15)]
        assert np.mean(errors_short) > 3 * np.mean(errors_long)


class TestPrincipalAxes:
    def test_orders_by_decreasing_spread(self):
        rng = np.random.default_rng(5)
        points = rng.normal(scale=[30.0, 10.0, 2.0], size=(2000, 3))

        axes, variances = principal_axes(points)
        assert variances[0] > variances[1] > variances[2]
        assert np.allclose(np.abs(axes[0]), [1, 0, 0], atol=0.05)
        assert np.allclose(np.abs(axes[2]), [0, 0, 1], atol=0.05)

    def test_is_deterministic_under_point_reordering(self):
        """Shuffling input must not flip an axis -- plan hashes depend on this."""
        rng = np.random.default_rng(6)
        points = rng.normal(scale=[20.0, 5.0, 1.0], size=(500, 3))

        first, _ = principal_axes(points)
        second, _ = principal_axes(rng.permutation(points))
        assert np.allclose(first, second, atol=1e-9)


# ----------------------------------------------------------------------
# Projection and angles
# ----------------------------------------------------------------------


class TestProjectionAndAngles:
    def test_project_to_plane_removes_normal_component(self):
        normal = unit(np.array([0.0, 0.0, 1.0]))
        origin = np.array([0.0, 0.0, 10.0])
        points = np.array([[1.0, 2.0, 30.0], [-4.0, 5.0, -7.0]])

        projected = project_to_plane(points, origin, normal)
        assert np.allclose(projected[:, 2], 10.0)
        assert np.allclose(projected[:, :2], points[:, :2])

    def test_project_out_gives_in_plane_unit_vector(self):
        axis = np.array([0.0, 0.0, 1.0])
        result = project_out(np.array([3.0, 0.0, 9.0]), axis)

        assert np.allclose(result, [1.0, 0.0, 0.0])
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_project_out_rejects_parallel_vector(self):
        with pytest.raises(ValueError, match="parallel"):
            project_out(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]))

    def test_signed_angle_follows_right_hand_rule(self):
        x, y, z = np.eye(3)
        assert np.isclose(signed_angle_in_plane(x, y, z), np.pi / 2)
        assert np.isclose(signed_angle_in_plane(y, x, z), -np.pi / 2)

    def test_signed_angle_reverses_with_the_normal(self):
        """Sign is a property of the chosen normal, which is how anatomical sign
        conventions are expressed without post-hoc negation."""
        x, y, z = np.eye(3)
        assert np.isclose(signed_angle_in_plane(x, y, z),
                          -signed_angle_in_plane(x, y, -z))

    def test_angle_between_is_stable_at_the_extremes(self):
        v = unit(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(angle_between(v, v), 0.0, atol=1e-7)
        assert np.isclose(angle_between(v, -v), np.pi, atol=1e-7)

    @pytest.mark.parametrize("degrees", [0.5, 5.0, 30.0, 90.0, 179.0])
    def test_angle_between_recovers_constructed_angle(self, degrees):
        radians = np.radians(degrees)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([np.cos(radians), np.sin(radians), 0.0])
        assert np.isclose(angle_between(a, b), radians, atol=1e-9)


# ----------------------------------------------------------------------
# Point selection
# ----------------------------------------------------------------------


class TestPointSelection:
    def test_extremal_point_averages_the_furthest_points(self):
        points = np.array([
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -5.0],
        ])
        result = extremal_point(points, np.array([0.0, 0.0, 1.0]), n_average=2)
        assert np.allclose(result, [0.0, 0.0, 9.5])

    def test_extremal_point_resists_a_single_outlier(self):
        """A lone spike -- a marching-cubes artefact -- must not define a landmark."""
        rng = np.random.default_rng(9)
        surface = np.column_stack([
            rng.uniform(-10, 10, 500),
            rng.uniform(-10, 10, 500),
            rng.normal(scale=0.1, size=500),
        ])
        spiked = np.vstack([surface, [0.0, 0.0, 25.0]])

        naive = spiked[np.argmax(spiked[:, 2])]
        averaged = extremal_point(spiked, np.array([0.0, 0.0, 1.0]), n_average=20)

        assert naive[2] == pytest.approx(25.0)
        assert averaged[2] < 2.0

    def test_extremal_point_is_deterministic_with_ties(self):
        points = np.zeros((50, 3))
        rng = np.random.default_rng(13)
        first = extremal_point(points, np.array([0.0, 0.0, 1.0]), n_average=5)
        second = extremal_point(rng.permutation(points),
                                np.array([0.0, 0.0, 1.0]), n_average=5)
        assert np.allclose(first, second)

    def test_bounding_box(self):
        points = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])
        box = bounding_box(points)

        assert np.allclose(box["min"], [-4, -2, -6])
        assert np.allclose(box["max"], [1, 5, 3])
        assert np.allclose(box["extent"], [5, 7, 9])
        assert np.allclose(box["centre"], [-1.5, 1.5, -1.5])

    def test_regional_mask_selects_the_requested_end(self):
        points = np.column_stack([
            np.zeros(100), np.zeros(100), np.linspace(0, 100, 100)
        ])
        z = np.array([0.0, 0.0, 1.0])

        low = regional_mask(points, z, fraction=0.25, end="low")
        high = regional_mask(points, z, fraction=0.25, end="high")

        assert points[low][:, 2].max() <= 25.0
        assert points[high][:, 2].min() >= 75.0
        assert low.sum() + high.sum() <= len(points)

    def test_regional_mask_validates_arguments(self):
        points = np.zeros((10, 3))
        with pytest.raises(ValueError, match="fraction"):
            regional_mask(points, np.array([0, 0, 1.0]), fraction=1.5)
        with pytest.raises(ValueError, match="end"):
            regional_mask(points, np.array([0, 0, 1.0]), end="middle")

    def test_cross_section_centroid_finds_the_canal_centre(self):
        """A tilted hollow cylinder: the slab centroid must land on its axis."""
        axis = unit(np.array([0.05, 0.0, 1.0]))
        base = np.array([10.0, 20.0, -500.0])
        rng = np.random.default_rng(17)

        # Densely sampled: a 4 mm slab must hold enough points that the circular
        # standard error (radius / sqrt(n)) is well under the tolerance asserted
        # below, otherwise the test measures sampling noise rather than correctness.
        n = 60_000
        heights = rng.uniform(-60, 60, n)
        angles = rng.uniform(0, 2 * np.pi, n)
        # Two in-plane basis vectors.
        u = unit(np.cross(axis, [0.0, 1.0, 0.0]))
        v = np.cross(axis, u)
        radius = 12.0
        points = (
            base
            + np.outer(heights, axis)
            + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
        )

        centroid = cross_section_centroid(
            points, axis, level=30.0, slab_thickness_mm=4.0, origin=base
        )
        expected = base + 30.0 * axis
        assert np.linalg.norm(centroid - expected) < 1.0

    def test_cross_section_centroid_reports_an_empty_slab(self):
        points = np.column_stack([
            np.zeros(50), np.zeros(50), np.linspace(-10, 10, 50)
        ])
        with pytest.raises(ValueError, match="No points within"):
            cross_section_centroid(points, np.array([0.0, 0.0, 1.0]), level=500.0)


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            fit_plane(np.zeros((10, 2)))

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least"):
            fit_sphere(np.zeros((3, 3)))

    def test_rejects_non_finite_coordinates(self):
        points = np.zeros((10, 3))
        points[4, 1] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            fit_plane(points)
