"""Parametric implant sizing.

The two properties that matter are that the continuous solver reproduces the discrete
chart exactly at chart points -- without which legacy parity is unprovable -- and that
it keeps working outside the chart, which is the whole reason for having it.
"""

import numpy as np
import pytest

from tka_planner.core.sizing import (
    QUANTISED_COLUMNS,
    load_size_chart,
    select_discrete_size,
    solve_parametric_size,
)

CHART_PATH = "data/SizeChart.csv"


@pytest.fixture(scope="module")
def chart():
    from pathlib import Path

    path = Path(__file__).parent.parent / CHART_PATH
    if not path.exists():
        pytest.skip("SizeChart.csv not present in this checkout.")
    return load_size_chart(path)


class TestChartLoading:
    def test_reads_all_twelve_sizes(self, chart):
        assert chart.n_sizes == 12
        assert chart.labels[0] == "S1"
        assert chart.labels[-1] == "L4"

    def test_converts_metre_columns_to_millimetres(self, chart):
        """The chart mixes units; a resection depth of 0.007 would be nonsense in mm."""
        assert 6.0 < chart.value_at("femur_distal_cut", 0) < 10.0

    def test_records_a_content_hash(self, chart):
        assert chart.sha256 and len(chart.sha256) == 64

    def test_identifies_which_columns_are_genuinely_parametric(self, chart):
        """The femoral family is a true one-parameter family; the tibial one is not.

        Stating this matters: it is the difference between a design that is parametric
        and a table that was hand-adjusted, and the report says which is which.
        """
        for column in ("femur_ML", "femur_AP", "tibia_AP",
                       "femur_distal_cut", "tibia_proximal_cut"):
            assert chart.is_exactly_linear(column), f"{column} should be linear"

        for column in ("tibia_ML", "patella_d", "insert_thin", "insert_thick"):
            assert not chart.is_exactly_linear(column), f"{column} should not be"


class TestParametricSolver:
    @pytest.mark.parametrize("label", ["S1", "S4", "M2", "L1", "L4"])
    def test_reproduces_chart_rows_exactly(self, chart, label):
        """Legacy parity: at a chart width, the solver must return that chart row."""
        parameter = chart.label_parameter(label)
        chart_ml = chart.value_at("femur_ML", parameter)

        decision = solve_parametric_size(chart, measured_ml_mm=chart_ml)

        assert decision.size_parameter == pytest.approx(parameter, abs=1e-9)
        assert decision.nearest_discrete_size == label
        assert decision.implant_ml_mm == pytest.approx(chart_ml, abs=1e-9)

    def test_interpolates_between_chart_sizes(self, chart):
        decision = solve_parametric_size(chart, measured_ml_mm=73.0)

        assert 5.0 < decision.size_parameter < 6.0  # between M2 (72) and M3 (74)
        assert decision.implant_ml_mm == pytest.approx(73.0, abs=1e-9)
        assert not decision.beyond_discrete_range

    def test_extrapolates_past_the_largest_size(self, chart):
        """The case the discrete chart cannot serve.

        Case 005 measures 86.4 mm against a largest chart size of 84.0 mm. The legacy
        pipeline clamped it; a continuous parameter has nothing to clamp against.
        """
        decision = solve_parametric_size(chart, measured_ml_mm=86.4)

        assert decision.implant_ml_mm == pytest.approx(86.4, abs=1e-9)
        assert decision.beyond_discrete_range
        assert "BEYOND_DISCRETE_CHART_RANGE" in decision.flags
        assert decision.size_parameter > 11

    def test_extrapolates_below_the_smallest_size(self, chart):
        decision = solve_parametric_size(chart, measured_ml_mm=58.0)

        assert decision.implant_ml_mm == pytest.approx(58.0, abs=1e-9)
        assert decision.size_parameter < 0

    def test_is_monotonic_in_the_measurement(self, chart):
        widths = np.linspace(55.0, 95.0, 40)
        parameters = [
            solve_parametric_size(chart, measured_ml_mm=w).size_parameter
            for w in widths
        ]
        assert np.all(np.diff(parameters) > 0)

    def test_reports_the_aspect_residual(self, chart):
        """The honest limit of a one-parameter family.

        Matching mediolateral and anteroposterior width independently needs two
        parameters; this reports what the second one would be worth for this patient.
        """
        decision = solve_parametric_size(
            chart, measured_ml_mm=80.0, measured_ap_mm=64.0
        )
        # At 80 mm width the chart's implant depth is 70 mm, so the residual is +6.
        assert decision.ap_mismatch_mm == pytest.approx(6.0, abs=1e-9)
        assert "AP_ASPECT_MISMATCH" in decision.flags

    def test_insert_thickness_stays_quantised(self, chart):
        """Polyethylene inserts come in whole millimetres; interpolating is meaningless."""
        decision = solve_parametric_size(chart, measured_ml_mm=73.0)

        for column in QUANTISED_COLUMNS:
            assert float(decision.insert_thin_mm).is_integer()
            assert float(decision.insert_thick_mm).is_integer()

    def test_scale_factor_is_relative_to_the_reference(self, chart):
        decision = solve_parametric_size(
            chart, measured_ml_mm=chart.value_at("femur_ML", chart.label_parameter("M3")),
            reference_size="M3",
        )
        assert decision.scale_factor == pytest.approx(1.0, abs=1e-9)


class TestDiscreteSelection:
    def test_round_down_matches_the_legacy_default(self, chart):
        decision = select_discrete_size(chart, measured_ml_mm=73.0, rounding="D")
        assert decision.nearest_discrete_size == "M2"  # 72, the largest not exceeding

    def test_round_up(self, chart):
        decision = select_discrete_size(chart, measured_ml_mm=73.0, rounding="U")
        assert decision.nearest_discrete_size == "M3"  # 74, the smallest covering

    def test_exact_chart_width_selects_that_size(self, chart):
        for label in chart.labels:
            width = chart.value_at("femur_ML", chart.label_parameter(label))
            assert select_discrete_size(
                chart, measured_ml_mm=width
            ).nearest_discrete_size == label

    def test_reproduces_the_case_005_clamp(self, chart):
        """The recorded legacy behaviour, now flagged instead of merely printed."""
        decision = select_discrete_size(chart, measured_ml_mm=86.4, rounding="D")

        assert decision.nearest_discrete_size == "L4"
        assert "EXCEEDS_LARGEST_SIZE" in decision.flags
        assert "CLAMPED_TO_CHART_LIMIT" in decision.flags
        assert decision.diagnostics["size_difference_mm"] == pytest.approx(-2.4)
        assert "undersized" in decision.rationale

    def test_parametric_beats_discrete_on_the_clamped_case(self, chart):
        """The concrete gain, asserted rather than claimed."""
        discrete = select_discrete_size(chart, measured_ml_mm=86.4, measured_ap_mm=77.5)
        parametric = solve_parametric_size(
            chart, measured_ml_mm=86.4, measured_ap_mm=77.5
        )

        assert abs(parametric.implant_ml_mm - 86.4) < abs(discrete.implant_ml_mm - 86.4)
        assert abs(parametric.ap_mismatch_mm) < abs(discrete.ap_mismatch_mm)

    def test_rejects_an_unknown_rounding_rule(self, chart):
        with pytest.raises(ValueError, match="rounding must be"):
            select_discrete_size(chart, measured_ml_mm=72.0, rounding="X")


class TestChartAccess:
    def test_unknown_column_is_reported(self, chart):
        with pytest.raises(KeyError, match="Unknown size chart column"):
            chart.value_at("not_a_column", 0.0)

    def test_unknown_label_is_reported(self, chart):
        with pytest.raises(KeyError, match="Unknown size"):
            chart.label_parameter("XL9")

    def test_non_monotonic_columns_cannot_drive_sizing(self, chart):
        """Insert thickness is quantised, so it has flat regions and no unique inverse."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            chart.parameter_for("insert_thin", 10.0)
