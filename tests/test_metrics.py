"""Deformity metrics.

Two families of test carry the weight here.

**Constructed-angle recovery.** The synthetic knee is built to realise a stated mLDFA,
MPTA, condylar twist and posterior slope, so each metric is checked against a number
that was an input rather than against its own output. A wrong formula fails, and so does
a flipped sign -- which is the failure mode that would otherwise ship, because a
plausible number in the right range looks correct.

**Mirror invariance.** Reflecting a knee through the sagittal plane and flipping its
side must leave every scalar unchanged. A knee is not more varus for being a right knee.
This is the test that catches the class of bug the legacy pipeline had, where medial and
lateral were labelled by world position and so were correct on one side only.
"""

import numpy as np
import pytest

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.metrics import (
    aldfa,
    compute_all,
    condylar_twist_angle,
    femoral_mechanical_anatomical_angle,
    hka,
    jlca,
    mldfa,
    mpta,
    posterior_slope_medial,
)
from tka_planner.core.provenance import Quality
from tests.synthetic import mirror_landmarks, synthetic_knee
from tests.test_frames import drop


def frames_for(landmarks):
    return build_femoral_frame(landmarks), build_tibial_frame(landmarks)


# ----------------------------------------------------------------------
# Constructed-angle recovery
# ----------------------------------------------------------------------


class TestCoronalAngles:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("constructed", [82.0, 85.5, 87.5, 90.0, 93.0])
    def test_mldfa_recovers_the_constructed_angle(self, side, constructed):
        """Across the valgus-to-varus range and on both knees."""
        landmarks = synthetic_knee(side, mldfa_deg=constructed, include_head=True)
        femoral, _ = frames_for(landmarks)

        assert mldfa(landmarks, femoral).value == pytest.approx(constructed, abs=0.01)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("constructed", [80.0, 85.0, 87.0, 90.0, 94.0])
    def test_mpta_recovers_the_constructed_angle(self, side, constructed):
        landmarks = synthetic_knee(side, mpta_deg=constructed, include_ankle=True)
        _, tibial = frames_for(landmarks)

        assert mpta(landmarks, tibial).value == pytest.approx(constructed, abs=0.01)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_mldfa_below_ninety_means_valgus(self, side):
        """Pins the direction of the sign convention, not merely its magnitude.

        A metric that returned ``180 - value`` would pass a magnitude check on a
        symmetric case and be exactly backwards clinically.
        """
        valgus = synthetic_knee(side, mldfa_deg=83.0, include_head=True)
        varus = synthetic_knee(side, mldfa_deg=93.0, include_head=True)

        valgus_value = mldfa(valgus, build_femoral_frame(valgus)).value
        varus_value = mldfa(varus, build_femoral_frame(varus)).value

        assert valgus_value < 85.0 < 90.0 < varus_value

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_mpta_below_ninety_means_varus(self, side):
        varus = synthetic_knee(side, mpta_deg=83.0, include_ankle=True)
        valgus = synthetic_knee(side, mpta_deg=94.0, include_ankle=True)

        assert mpta(varus, build_tibial_frame(varus)).value < 85.0
        assert mpta(valgus, build_tibial_frame(valgus)).value > 90.0

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_one_degree_of_construction_moves_the_metric_one_degree(self, side):
        """Sensitivity as well as sign: a scale error would survive a sign test."""
        base = synthetic_knee(side, mldfa_deg=87.0, include_head=True)
        moved = synthetic_knee(side, mldfa_deg=88.0, include_head=True)

        delta = (mldfa(moved, build_femoral_frame(moved)).value
                 - mldfa(base, build_femoral_frame(base)).value)
        assert delta == pytest.approx(1.0, abs=0.01)


class TestAldfa:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_aldfa_is_measurable_without_the_femoral_head(self, side):
        """The point of reporting it: on a knee-only scan this is the honest coronal
        number and mLDFA is the modelled one."""
        landmarks = synthetic_knee(side, include_head=False)
        femoral, _ = frames_for(landmarks)

        femoral_mldfa = mldfa(landmarks, femoral)
        femoral_aldfa = aldfa(landmarks, femoral)

        assert femoral_mldfa.quality is Quality.ESTIMATED
        assert femoral_aldfa.quality is Quality.MEASURED

    @pytest.mark.parametrize("ama_deg", [4.0, 6.0, 8.0])
    def test_aldfa_falls_below_mldfa_by_the_ama(self, ama_deg):
        """The defining relation aLDFA = mLDFA - AMA, checked against construction.

        The anatomical axis leans laterally away from the mechanical axis, so it closes
        the laterally-opening angle rather than widening it. The direction is confirmed
        by the normal ranges: mLDFA is 85-90 and aLDFA is 79-83, roughly six degrees
        lower.
        """
        landmarks = synthetic_knee("left", mldfa_deg=87.5, ama_deg=ama_deg,
                                   include_head=True)
        femoral, _ = frames_for(landmarks)

        measured_mldfa = mldfa(landmarks, femoral).value
        measured_aldfa = aldfa(landmarks, femoral).value

        assert measured_mldfa - measured_aldfa == pytest.approx(ama_deg, abs=0.01)

    def test_aldfa_lands_in_its_normal_range_for_normal_anatomy(self):
        """An independent check on the direction of the relation.

        Were the sign reversed, a textbook-normal knee would produce an aLDFA of about
        93 degrees, far outside the accepted 79-83 range.
        """
        landmarks = synthetic_knee("left", mldfa_deg=87.5, ama_deg=6.0,
                                   include_head=True)
        femoral, _ = frames_for(landmarks)
        metric = aldfa(landmarks, femoral)

        assert 79.0 <= metric.value <= 83.0
        assert not metric.is_outside_reference_range


class TestRotationAndSlope:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("constructed", [0.0, 3.0, 5.5, 8.0])
    def test_condylar_twist_recovers_the_constructed_angle(self, side, constructed):
        """Replaces the legacy fixed 3 degree TEA correction with a measurement."""
        landmarks = synthetic_knee(side, condylar_twist_deg=constructed)
        femoral, _ = frames_for(landmarks)

        assert condylar_twist_angle(landmarks, femoral).value == pytest.approx(
            constructed, abs=0.01
        )

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_positive_twist_means_the_stea_is_externally_rotated(self, side):
        """The standard convention, asserted directionally on both knees."""
        landmarks = synthetic_knee(side, condylar_twist_deg=5.0)
        femoral, _ = frames_for(landmarks)

        assert condylar_twist_angle(landmarks, femoral).value > 0

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("constructed", [0.0, 5.0, 7.0, 12.0])
    def test_posterior_slope_recovers_the_constructed_angle(self, side, constructed):
        landmarks = synthetic_knee(side, posterior_slope_deg=constructed,
                                   include_ankle=True)
        _, tibial = frames_for(landmarks)

        assert posterior_slope_medial(landmarks, tibial).value == pytest.approx(
            constructed, abs=0.01
        )

    def test_slope_records_compartment_and_reference_axis(self):
        """A slope quoted without both is ambiguous by several degrees."""
        landmarks = synthetic_knee("left")
        _, tibial = frames_for(landmarks)
        metric = posterior_slope_medial(landmarks, tibial)

        assert metric.diagnostics["compartment"] == "medial"
        assert metric.diagnostics["reference_axis"] == tibial.method


# ----------------------------------------------------------------------
# The capability model
# ----------------------------------------------------------------------


class TestHkaRefusesToGuess:
    def test_hka_is_not_computable_on_a_knee_only_scan(self):
        """The metric the whole provenance machinery exists to protect."""
        landmarks = synthetic_knee("left", include_head=False, include_ankle=False)
        femoral, tibial = frames_for(landmarks)
        metric = hka(landmarks, femoral, tibial)

        assert metric.quality is Quality.NOT_COMPUTABLE
        assert metric.value is None
        assert set(metric.missing) == {"femur.head_centre", "tibia.ankle_centre"}

    def test_unavailable_hka_explains_what_would_supply_it(self):
        """This text becomes the report's 'what would it take' block."""
        landmarks = synthetic_knee("left")
        femoral, tibial = frames_for(landmarks)
        metric = hka(landmarks, femoral, tibial)

        assert "full-limb" in metric.would_require
        assert "population average" in metric.reason

    def test_hka_is_computed_when_both_axes_are_measured(self):
        landmarks = synthetic_knee("left", include_head=True, include_ankle=True)
        femoral, tibial = frames_for(landmarks)
        metric = hka(landmarks, femoral, tibial)

        assert metric.quality is Quality.MEASURED
        # Both synthetic mechanical axes are vertical, so the limb is straight.
        assert metric.value == pytest.approx(0.0, abs=0.01)

    def test_partial_availability_still_refuses(self):
        """One measured axis is not enough; the tier is the weaker of the two."""
        landmarks = synthetic_knee("left", include_head=True, include_ankle=False)
        femoral, tibial = frames_for(landmarks)
        metric = hka(landmarks, femoral, tibial)

        assert metric.quality is Quality.NOT_COMPUTABLE
        assert metric.missing == ("tibia.ankle_centre",)


class TestQualityPropagation:
    def test_metrics_inherit_their_frame_quality(self):
        landmarks = synthetic_knee("left", include_head=False)
        femoral, _ = frames_for(landmarks)
        metric = mldfa(landmarks, femoral)

        assert metric.quality is Quality.ESTIMATED
        assert [a.id for a in metric.assumptions] == ["population_femoral_ama"]

    def test_measured_frames_yield_measured_metrics_with_no_assumptions(self):
        landmarks = synthetic_knee("left", include_head=True)
        femoral, _ = frames_for(landmarks)
        metric = mldfa(landmarks, femoral)

        assert metric.quality is Quality.MEASURED
        assert metric.assumptions == ()

    def test_missing_landmarks_make_a_metric_unavailable_not_wrong(self):
        landmarks = drop(synthetic_knee("left"), "femur.condyle_distal_medial")
        femoral, _ = frames_for(landmarks)
        metric = mldfa(landmarks, femoral)

        assert metric.quality is Quality.NOT_COMPUTABLE
        assert metric.value is None
        assert metric.missing == ("femur.condyle_distal_medial",)

    def test_ama_cannot_be_measured_without_the_head(self):
        """The circularity made explicit: the quantity that must be assumed is
        exactly the one the missing anatomy would have supplied."""
        landmarks = synthetic_knee("left", include_head=False)
        femoral, _ = frames_for(landmarks)

        assert femoral_mechanical_anatomical_angle(
            landmarks, femoral
        ).quality is Quality.NOT_COMPUTABLE

    @pytest.mark.parametrize("ama_deg", [4.0, 6.0, 8.0])
    def test_measured_ama_recovers_the_construction(self, ama_deg):
        landmarks = synthetic_knee("left", ama_deg=ama_deg, include_head=True)
        femoral, _ = frames_for(landmarks)

        assert femoral_mechanical_anatomical_angle(
            landmarks, femoral
        ).value == pytest.approx(ama_deg, abs=0.01)


class TestJlca:
    def test_jlca_is_computable_on_a_knee_only_scan(self):
        """It compares two joint lines and never references hip or ankle.

        Available only because femur and tibia arrive registered in the shared CT
        frame -- the relationship the legacy pipeline discarded.
        """
        landmarks = synthetic_knee("left", include_head=False, include_ankle=False)
        femoral, tibial = frames_for(landmarks)
        metric = jlca(landmarks, femoral, tibial)

        assert metric.quality is Quality.MEASURED
        assert metric.value is not None

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("angle", [85.0, 87.0, 91.0])
    def test_parallel_joint_lines_give_zero(self, side, angle):
        """Because both angles are measured against their own mechanical axis in the
        same rotational sense, equal values mean parallel lines."""
        landmarks = synthetic_knee(side, mldfa_deg=angle, mpta_deg=angle)
        femoral, tibial = frames_for(landmarks)

        assert jlca(landmarks, femoral, tibial).value == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_positive_jlca_means_the_space_opens_laterally(self, side):
        """Derived physically rather than adopted from the code's output.

        Travelling laterally, the femoral surface rises by cos(mLDFA) per unit and the
        tibial surface by cos(MPTA). The gap therefore widens laterally exactly when
        mLDFA < MPTA, which by the stated convention must read positive.
        """
        opens_laterally = synthetic_knee(side, mldfa_deg=87.0, mpta_deg=93.0)
        opens_medially = synthetic_knee(side, mldfa_deg=93.0, mpta_deg=87.0)

        assert jlca(opens_laterally, *frames_for(opens_laterally)).value > 0
        assert jlca(opens_medially, *frames_for(opens_medially)).value < 0

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_jlca_magnitude_matches_the_joint_line_divergence(self, side):
        landmarks = synthetic_knee(side, mldfa_deg=87.0, mpta_deg=93.0)

        assert abs(jlca(landmarks, *frames_for(landmarks)).value) == pytest.approx(
            6.0, abs=0.01
        )


# ----------------------------------------------------------------------
# Mirror invariance
# ----------------------------------------------------------------------


class TestMirrorInvariance:
    """A knee is not more varus for being a right knee."""

    SCALAR_METRICS = [
        "mldfa_deg", "aldfa_deg", "mpta_deg", "jlca_deg",
        "condylar_twist_deg", "posterior_slope_medial_deg",
    ]

    @pytest.mark.parametrize("metric_name", SCALAR_METRICS)
    def test_metrics_are_unchanged_by_mirroring(self, metric_name):
        original = synthetic_knee("left", mldfa_deg=84.0, mpta_deg=85.5,
                                  condylar_twist_deg=4.5, posterior_slope_deg=9.0)
        mirrored = mirror_landmarks(original)

        left = compute_all(original, *frames_for(original))
        right = compute_all(mirrored, *frames_for(mirrored))

        assert left[metric_name].value == pytest.approx(
            right[metric_name].value, abs=1e-6
        ), f"{metric_name} changed when the knee was mirrored"

    def test_hka_is_unchanged_by_mirroring(self):
        original = synthetic_knee("left", mldfa_deg=84.0, include_head=True,
                                  include_ankle=True)
        mirrored = mirror_landmarks(original)

        left = hka(original, *frames_for(original)).value
        right = hka(mirrored, *frames_for(mirrored)).value
        assert left == pytest.approx(right, abs=1e-6)

    @pytest.mark.parametrize("metric_name", SCALAR_METRICS)
    def test_quality_is_unchanged_by_mirroring(self, metric_name):
        original = synthetic_knee("left")
        mirrored = mirror_landmarks(original)

        left = compute_all(original, *frames_for(original))
        right = compute_all(mirrored, *frames_for(mirrored))
        assert left[metric_name].quality is right[metric_name].quality


# ----------------------------------------------------------------------
# Aggregation and serialisation
# ----------------------------------------------------------------------


class TestComputeAll:
    def test_returns_every_metric_including_unavailable_ones(self):
        """An omitted metric is an invisible gap; an unavailable one is explained."""
        landmarks = synthetic_knee("left", include_head=False, include_ankle=False)
        metrics = compute_all(landmarks, *frames_for(landmarks))

        assert "hka_deviation_deg" in metrics
        assert metrics["hka_deviation_deg"].quality is Quality.NOT_COMPUTABLE

    def test_knee_only_scan_yields_the_expected_mix_of_tiers(self):
        """The honest summary of what this cohort can actually support."""
        landmarks = synthetic_knee("left", include_head=False, include_ankle=False)
        metrics = compute_all(landmarks, *frames_for(landmarks))
        tiers = {name: metric.quality for name, metric in metrics.items()}

        assert tiers["aldfa_deg"] is Quality.MEASURED
        assert tiers["condylar_twist_deg"] is Quality.MEASURED
        assert tiers["jlca_deg"] is Quality.MEASURED
        assert tiers["mldfa_deg"] is Quality.ESTIMATED
        assert tiers["mpta_deg"] is Quality.ESTIMATED
        assert tiers["hka_deviation_deg"] is Quality.NOT_COMPUTABLE
        assert tiers["femoral_ama_deg"] is Quality.NOT_COMPUTABLE

    def test_full_limb_scan_upgrades_the_tiers(self):
        landmarks = synthetic_knee("left", include_head=True, include_ankle=True)
        metrics = compute_all(landmarks, *frames_for(landmarks))

        for name in ("mldfa_deg", "mpta_deg", "hka_deviation_deg", "femoral_ama_deg"):
            assert metrics[name].quality is Quality.MEASURED


class TestSerialisation:
    def test_metric_serialises_with_provenance(self):
        landmarks = synthetic_knee("left", include_head=False)
        record = mldfa(landmarks, build_femoral_frame(landmarks)).to_dict()

        assert record["status"] == "estimated"
        assert record["assumptions"] == ["population_femoral_ama"]
        assert record["unit"] == "deg"
        assert "definition" in record and "sign_convention" in record

    def test_unavailable_metric_serialises_its_reasons(self):
        landmarks = synthetic_knee("left")
        record = hka(landmarks, *frames_for(landmarks)).to_dict()

        assert record["value"] is None
        assert record["status"] == "not_computable"
        assert "missing" in record and "would_require" in record

    def test_reference_range_flags_without_correcting(self):
        landmarks = synthetic_knee("left", mldfa_deg=95.0, include_head=True)
        record = mldfa(landmarks, build_femoral_frame(landmarks)).to_dict()

        assert record["outside_reference_range"] is True
        assert record["value"] == pytest.approx(95.0, abs=0.01)  # never clamped
