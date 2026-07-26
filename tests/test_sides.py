"""Laterality.

The point of :mod:`tka_planner.core.sides` is that side errors become impossible to
write rather than merely unlikely, so these tests check the anatomy of the abstraction
itself: that medial genuinely points toward the midline on both sides, that compartment
splitting is not the legacy world-position split in disguise, and that mirroring a case
maps cleanly onto the contralateral side.
"""

import numpy as np
import pytest

from tka_planner.core.sides import (
    LPS_ANTERIOR,
    LPS_PATIENT_LEFT,
    LPS_POSTERIOR,
    LPS_SUPERIOR,
    Side,
)


class TestParsing:
    @pytest.mark.parametrize("value", ["L", "l", "left", "Left", " LEFT "])
    def test_accepts_left_spellings(self, value):
        assert Side.parse(value) is Side.LEFT

    @pytest.mark.parametrize("value", ["R", "r", "right", "Right", " RIGHT "])
    def test_accepts_right_spellings(self, value):
        assert Side.parse(value) is Side.RIGHT

    def test_is_idempotent(self):
        assert Side.parse(Side.LEFT) is Side.LEFT

    @pytest.mark.parametrize("value", ["", "lateral", "both", "L/R", "1"])
    def test_refuses_to_guess(self, value):
        """A silently defaulted side is the failure this module exists to prevent."""
        with pytest.raises(ValueError, match="Unrecognised side"):
            Side.parse(value)

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            Side.parse(1)


class TestAnatomicalDirections:
    def test_medial_points_toward_the_midline(self):
        """The core anatomical fact, asserted directly.

        A left knee sits at positive X in LPS, so travelling medially means travelling
        toward smaller X. A right knee is the reverse. Everything else in the module
        follows from this.
        """
        left_medial = Side.LEFT.medial_direction(LPS_PATIENT_LEFT)
        right_medial = Side.RIGHT.medial_direction(LPS_PATIENT_LEFT)

        assert np.allclose(left_medial, [-1, 0, 0])
        assert np.allclose(right_medial, [1, 0, 0])

    def test_medial_and_lateral_are_opposed(self):
        for side in Side:
            medial = side.medial_direction(LPS_PATIENT_LEFT)
            lateral = side.lateral_direction(LPS_PATIENT_LEFT)
            assert np.allclose(medial, -lateral)

    def test_sides_disagree_about_medial(self):
        assert Side.LEFT.medial_sign == -Side.RIGHT.medial_sign

    def test_directions_are_normalised(self):
        """Frame axes arrive unnormalised often enough to be worth handling."""
        scaled = np.array([7.0, 0.0, 0.0])
        assert np.isclose(
            np.linalg.norm(Side.LEFT.medial_direction(scaled)), 1.0
        )

    def test_works_in_a_rotated_frame(self):
        """Directions are relative to a supplied axis, never to world coordinates."""
        rotated_patient_left = np.array([0.0, 1.0, 0.0])
        assert np.allclose(
            Side.LEFT.medial_direction(rotated_patient_left), [0, -1, 0]
        )


class TestCompartmentSplitting:
    def test_labels_compartments_correctly_on_both_sides(self):
        """The regression test for the legacy defect.

        ``find_condyle_contact`` split at the bounding-box midline and called the lower
        half medial regardless of side, which is right for one knee and wrong for the
        other. Here the same physical point must receive opposite labels depending on
        which knee it belongs to.
        """
        origin = np.zeros(3)
        # One point to patient-left of the midline, one to patient-right.
        points = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])

        left_medial, left_lateral = Side.LEFT.split_compartments(
            points, origin, LPS_PATIENT_LEFT
        )
        right_medial, right_lateral = Side.RIGHT.split_compartments(
            points, origin, LPS_PATIENT_LEFT
        )

        # On a left knee the patient-right point (-10) is the medial one.
        assert list(left_medial) == [False, True]
        assert list(left_lateral) == [True, False]

        # On a right knee it is exactly reversed.
        assert list(right_medial) == [True, False]
        assert list(right_lateral) == [False, True]

    def test_masks_are_disjoint_and_exhaustive(self):
        rng = np.random.default_rng(4)
        points = rng.normal(scale=30.0, size=(500, 3))

        for side in Side:
            medial, lateral = side.split_compartments(
                points, np.zeros(3), LPS_PATIENT_LEFT
            )
            assert not np.any(medial & lateral)
            assert np.all(medial | lateral)

    def test_split_respects_a_shifted_midline(self):
        points = np.array([[95.0, 0.0, 0.0], [105.0, 0.0, 0.0]])
        origin = np.array([100.0, 0.0, 0.0])

        medial, _ = Side.LEFT.split_compartments(points, origin, LPS_PATIENT_LEFT)
        # For a left knee, medial is -X, so the point at 95 is medial.
        assert list(medial) == [True, False]

    def test_rejects_malformed_point_arrays(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            Side.LEFT.split_compartments(
                np.zeros((5, 2)), np.zeros(3), LPS_PATIENT_LEFT
            )


class TestMoreLateral:
    def test_picks_the_more_lateral_of_two_points(self):
        a = np.array([50.0, 0.0, 0.0])
        b = np.array([20.0, 0.0, 0.0])

        # Left knee: lateral is +X, so a wins.
        assert np.allclose(Side.LEFT.more_lateral(a, b, LPS_PATIENT_LEFT), a)
        # Right knee: lateral is -X, so b wins.
        assert np.allclose(Side.RIGHT.more_lateral(a, b, LPS_PATIENT_LEFT), b)


class TestMirroring:
    def test_opposite_is_an_involution(self):
        for side in Side:
            assert side.opposite.opposite is side

    def test_opposite_swaps_sides(self):
        assert Side.LEFT.opposite is Side.RIGHT
        assert Side.RIGHT.opposite is Side.LEFT

    def test_mirroring_a_case_reproduces_the_contralateral_anatomy(self):
        """The property the mirror-invariance metric test is built on.

        Reflecting points through the sagittal plane and flipping the side must leave
        medial offsets unchanged: a point 12 mm medial on a left knee is still 12 mm
        medial once mirrored into a right knee.
        """
        rng = np.random.default_rng(21)
        points = rng.normal(scale=40.0, size=(200, 3))
        origin = np.zeros(3)

        medial = Side.LEFT.medial_direction(LPS_PATIENT_LEFT)
        original_offsets = (points - origin) @ medial

        # Reflect through the sagittal plane (negate the patient-left component).
        mirrored = points * np.array([-1.0, 1.0, 1.0])
        mirrored_medial = Side.RIGHT.medial_direction(LPS_PATIENT_LEFT)
        mirrored_offsets = (mirrored - origin) @ mirrored_medial

        assert np.allclose(original_offsets, mirrored_offsets)


class TestLpsConstants:
    def test_axes_are_orthonormal_and_right_handed(self):
        assert np.isclose(np.dot(LPS_PATIENT_LEFT, LPS_POSTERIOR), 0.0)
        assert np.isclose(np.dot(LPS_POSTERIOR, LPS_SUPERIOR), 0.0)
        assert np.allclose(
            np.cross(LPS_PATIENT_LEFT, LPS_POSTERIOR), LPS_SUPERIOR
        )

    def test_anterior_opposes_posterior(self):
        assert np.allclose(LPS_ANTERIOR, -LPS_POSTERIOR)
