"""The tray rotation toggle (author 2026-11-08).

By default the tray follows the femoral component (ruling 2026-09-26): imported together
they sit square on each other, and the tray is off the tibia's own axis by however much
the femur leads it. "tibial" squares the tray to the tibial rotational reference
instead, and the femoral-over-tibial rotation it creates is then what the register
reports.
"""

import pytest

from tka_planner.core.planning import Adjustments

from tests.synthetic import mirror_landmarks, synthetic_knee
from tests.test_planning import plan_for


def _knee():
    return synthetic_knee()


def test_the_default_follows_the_femoral_component_and_changes_nothing():
    landmarks = _knee()
    implicit = plan_for(landmarks)
    explicit = plan_for(landmarks, tray_rotation="femoral")
    for name in implicit.components:
        assert explicit.components[name] == pytest.approx(implicit.components[name])
    assert explicit.diagnostics["component_rotation_mismatch_deg"] == 0.0
    assert explicit.diagnostics["tray_rotation"] == "femoral"


@pytest.mark.parametrize("mirror", [False, True])
def test_tibial_squares_the_tray_to_the_tibia_and_reports_the_register(mirror):
    landmarks = mirror_landmarks(_knee()) if mirror else _knee()
    # The synthetic knee is square, so the femur is turned to lead the tray off the tibia.
    turned = Adjustments(femoral_rotation_delta_deg=5.0)
    femoral = plan_for(landmarks, adjustments=turned)
    offset = femoral.diagnostics["tray_rotation_from_tibial_axis_deg"]
    assert abs(offset) > 0.5, "the synthetic knee must lead the tray off the tibial axis"

    tibial = plan_for(landmarks, tray_rotation="tibial", adjustments=turned)
    d = tibial.diagnostics
    assert d["tray_rotation_from_tibial_axis_deg"] == pytest.approx(0.0, abs=0.01)
    assert d["component_rotation_mismatch_deg"] == pytest.approx(offset, abs=0.02)
    assert d["tray_rotation"] == "tibial"
    # Only the tray turns: the cuts and the femoral component stay where they were.
    assert tibial.components["femoral_component"] == pytest.approx(
        femoral.components["femoral_component"])
    for name in ("femoral_distal", "tibial_proximal"):
        assert tibial.resections[name].point == pytest.approx(femoral.resections[name].point)


def test_the_tibial_rotation_control_still_turns_the_tray_from_there():
    landmarks = _knee()
    plan = plan_for(landmarks, tray_rotation="tibial",
                    adjustments=Adjustments(tibial_rotation_delta_deg=3.0))
    assert plan.diagnostics["tray_rotation_from_tibial_axis_deg"] == pytest.approx(3.0, abs=0.01)


def test_an_unknown_tray_rotation_is_refused():
    with pytest.raises(ValueError):
        plan_for(_knee(), tray_rotation="sideways")


def test_the_toggle_is_a_panel_control_and_reaches_the_plan(session):
    from tka_planner.panel import control_names

    assert "tray_rotation" in control_names()
    assert session.controls.tray_rotation == "femoral"
    session.replan(tray_rotation="tibial")
    assert session.plan.diagnostics["tray_rotation"] == "tibial"
    assert session.bones_affected_by(("tray_rotation",)) == ("Tibia",)
    assert any(line.startswith("Tray rotation|") for line in session.report_lines())
