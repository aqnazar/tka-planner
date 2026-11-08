"""The insert minimum (author 2026-11-08: 8 mm by default, met by extra tibial resection).

The rule is eval/demo_cases.jobs's: when the insert is thinner than the minimum, the
tibial resection deepens in 0.5 mm steps until it is met, never past the control's own
limit. The knee is stood in for by the resection it was planned with, so each case here
is exact.
"""

import pytest

from tka_planner.core.planning import deepen_for_insert_minimum


def _linear_knee(default_insert_mm, gain=1.0):
    """A knee whose insert grows with the extra tibial resection by ``gain``."""
    def replan(delta_mm):
        return {"delta": delta_mm}

    def insert_of(plan):
        return default_insert_mm + gain * plan["delta"]

    return replan, insert_of


def test_a_thin_insert_deepens_the_tibial_cut_in_half_millimetres():
    replan, insert_of = _linear_knee(5.6)
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=8.0,
                                       user_delta_mm=0.0, limit_mm=10.0)
    assert result.auto_mm == pytest.approx(2.5)
    assert result.insert_mm == pytest.approx(8.1)
    assert result.plan == {"delta": 2.5}
    assert not result.clamped


def test_an_insert_already_above_the_minimum_is_left_alone():
    replan, insert_of = _linear_knee(8.4)
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=8.0,
                                       user_delta_mm=0.0, limit_mm=10.0)
    assert result.auto_mm == 0.0
    assert result.plan == {"delta": 0.0}
    assert result.insert_mm == pytest.approx(8.4)


def test_a_minimum_of_zero_is_off():
    replan, insert_of = _linear_knee(2.0)
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=0.0,
                                       user_delta_mm=1.0, limit_mm=10.0)
    assert result.auto_mm == 0.0 and result.plan == {"delta": 1.0}


def test_the_surgeons_own_resection_counts_toward_the_minimum():
    # 5.6 at no delta; the surgeon took 1 mm more, so 6.6 needs another 1.5.
    replan, insert_of = _linear_knee(5.6)
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=8.0,
                                       user_delta_mm=1.0, limit_mm=10.0)
    assert result.auto_mm == pytest.approx(1.5)
    assert result.plan == {"delta": 2.5}


def test_a_nonlinear_knee_keeps_deepening_until_the_minimum_is_met():
    replan, insert_of = _linear_knee(5.0, gain=0.8)   # the cut gains less than it takes
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=8.0,
                                       user_delta_mm=0.0, limit_mm=10.0)
    assert result.insert_mm >= 8.0 - 1e-9
    assert result.auto_mm == pytest.approx(4.0)       # 3.0 first, then 3.5, 4.0


def test_a_minimum_past_the_resection_limit_clamps_and_warns():
    replan, insert_of = _linear_knee(1.0, gain=0.2)   # 2.2 mm at the surgeon's +6
    result = deepen_for_insert_minimum(replan, insert_of, minimum_mm=8.0,
                                       user_delta_mm=6.0, limit_mm=10.0)
    assert result.plan == {"delta": 10.0}          # never past the control's +10 mm
    assert result.auto_mm == pytest.approx(4.0)
    assert result.clamped
    assert result.insert_mm == pytest.approx(3.0)
    assert "10" in result.warning and "8" in result.warning


def test_no_insert_to_measure_leaves_the_plan_alone():
    result = deepen_for_insert_minimum(lambda d: {"delta": d}, lambda plan: None,
                                       minimum_mm=8.0, user_delta_mm=0.0, limit_mm=10.0)
    assert result.auto_mm == 0.0 and result.insert_mm is None


# -- in the one pipeline -------------------------------------------------------------


def test_the_plan_records_the_extra_resection_and_its_reason(session):
    """On a whole knee: the plan carries the minimum, the extra resection it took and the
    insert it gives, and the report says why the cut is deeper."""
    off = session.replan(insert_minimum_mm=0.0) and session.plan
    base = off.diagnostics["insert_of_record_mm"]
    assert off.diagnostics["tibial_resection_auto_mm"] == 0.0
    session.replan(insert_minimum_mm=round(base + 2.2, 1))
    on = session.plan
    d = on.diagnostics
    assert d["insert_minimum_mm"] == pytest.approx(round(base + 2.2, 1))
    assert d["tibial_resection_auto_mm"] >= 2.0
    assert d["insert_of_record_mm"] >= d["insert_minimum_mm"] - 1e-9
    assert on.resections["tibial_proximal"].medial_depth_mm == pytest.approx(
        off.resections["tibial_proximal"].medial_depth_mm + d["tibial_resection_auto_mm"],
        abs=0.3)
    assert any("tibial resection for the" in line and "insert minimum" in line
               for line in session.report_lines())
    # The surgeon's own control is untouched: the extra is the plan's, not theirs.
    assert session.controls.tibial_resection_delta_mm == 0.0


def test_the_minimum_is_a_panel_control_with_the_default_of_8(folder, library):
    from tka_planner.panel import control_names
    from tka_planner.session import Controls

    assert Controls().insert_minimum_mm == 8.0
    assert "insert_minimum_mm" in control_names()


def test_a_clamped_minimum_is_a_warning_finding(session):
    session.replan(insert_minimum_mm=40.0)
    assert session.plan.diagnostics["insert_minimum_clamped"]
    codes = {f["code"]: f for f in session.findings()}
    assert codes["INSERT_MINIMUM_CLAMPED"]["severity"] == "warning"
    tibial = session.plan.diagnostics["tibial_resection_from_datum_mm"]
    assert tibial <= 9.0 + 10.0 + 1e-9
