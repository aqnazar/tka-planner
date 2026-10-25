"""The flexion trajectory (D1) and the maximum-flexion search, on synthetic geometry.

The hinge must be exactly what Reduce mode has always done when the trajectory is
neutral, the scenarios must move each compartment the way their sources say on both
knees, and the search must find a contact angle that is known in closed form.
"""

import numpy as np
import pytest

from tka_planner.core.kinematics import (
    FOLD_DEG,
    NEUTRAL,
    SCENARIOS,
    Contact,
    Trajectory,
    envelope,
    scenario,
    search_max_flexion,
    tibial_motion,
)
from tka_planner.core.sides import LPS_ANTERIOR, LPS_PATIENT_LEFT, Side
from tka_planner.scene import motion

CENTRE = np.array([12.0, -4.0, -550.0])


@pytest.fixture
def pivot():
    return motion.pivot_frame(CENTRE, LPS_PATIENT_LEFT)


def _medial(side):
    return Side.parse(side).medial_direction(LPS_PATIENT_LEFT)


def _femur_moves(trajectory, angle, side, ml_mm):
    """How far a femoral point ``ml_mm`` medial of the centre moves anteriorly on the
    tibia: the inverse of the tibia's motion, measured along the tibia's anterior."""
    medial = _medial(side)
    motion_ = tibial_motion(trajectory, angle, centre=CENTRE, anterior=LPS_ANTERIOR,
                            medial=medial)
    point = np.append(CENTRE + ml_mm * medial, 1.0)
    moved = np.linalg.inv(motion_) @ point
    return float((moved - point)[:3] @ LPS_ANTERIOR)


# ----------------------------------------------------------------------
# S3 is today's hinge
# ----------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 12.5, 45.0, 90.0, 133.0])
def test_the_neutral_trajectory_is_exactly_the_reduce_hinge(pivot, angle):
    hinge = motion.flexion_pose(pivot, flexion_deg=angle)
    pose = motion.trajectory_pose(pivot, angle, "S3", anterior=LPS_ANTERIOR,
                                  medial=_medial("left"))
    assert np.array_equal(pose, hinge)


def test_a_trial_on_the_neutral_trajectory_is_bit_for_bit_the_old_trial(pivot):
    old = motion.trial_pose(pivot, flexion_deg=70.0, varus_valgus_deg=3.0,
                            drawer_ap_mm=2.0)
    new = motion.trial_pose(pivot, flexion_deg=70.0, varus_valgus_deg=3.0,
                            drawer_ap_mm=2.0, trajectory="S3",
                            anterior=LPS_ANTERIOR, medial=_medial("right"))
    assert np.array_equal(old, new)


def test_the_neutral_motion_is_the_identity():
    assert np.array_equal(
        tibial_motion(NEUTRAL, 80.0, centre=CENTRE, anterior=LPS_ANTERIOR,
                      medial=_medial("left")),
        np.eye(4))


def test_a_trajectory_needs_the_tibial_directions(pivot):
    with pytest.raises(ValueError, match="anterior and medial"):
        motion.trial_pose(pivot, flexion_deg=30.0, trajectory="S2")


# ----------------------------------------------------------------------
# What each scenario does to each compartment
# ----------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_native_scenario_reproduces_the_published_displacements(side):
    """S1 is fitted to Pinskerova 2004 and Nakagawa 2000: with the condyle centres 52.8 mm
    apart, the medial centre stays put 30-120 deg and the lateral one goes back 15 mm
    from 20 to 120, then a further 4 mm medially and 15 mm laterally to 162."""
    s1 = SCENARIOS["S1"]
    d = s1.pivot_ml_mm                          # the pivot is the medial centre
    assert d == pytest.approx(26.4)

    def moved(angle, ml):
        return _femur_moves(s1, angle, side, ml)

    assert moved(120.0, +d) - moved(30.0, +d) == pytest.approx(0.0, abs=0.01)
    assert moved(120.0, -d) - moved(20.0, -d) == pytest.approx(-15.0, abs=0.1)
    assert moved(162.0, +d) - moved(120.0, +d) == pytest.approx(-4.0, abs=0.1)
    assert moved(162.0, -d) - moved(120.0, -d) == pytest.approx(-15.0, abs=0.2)
    assert s1.at(120.0)[0] == pytest.approx(20.0)           # about 20 deg by 120


def test_the_native_scenario_holds_its_last_values_beyond_162_degrees():
    s1 = SCENARIOS["S1"]
    assert s1.at(170.0) == s1.at(180.0) == s1.at(162.0)
    assert s1.at(0.0) == (0.0, 0.0)
    # Continuous through the first knot, and linear from extension to it.
    assert s1.at(60.0)[0] == pytest.approx(10.0)
    assert s1.at(120.0) == (20.0, 0.0)


def test_knots_must_rise_in_flexion():
    with pytest.raises(ValueError, match="knots"):
        Trajectory("T", "bad", knots=((90.0, 10.0, 0.0), (60.0, 20.0, 0.0)))
    with pytest.raises(ValueError, match="knots"):
        Trajectory("T", "bad", knots=((0.0, 10.0, 0.0),))


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_observed_cr_scenario_slides_both_condyles_forwards(side):
    s2 = SCENARIOS["S2"]
    angle = s2.reference_flexion_deg
    medial = _femur_moves(s2, angle, side, +22.0)
    lateral = _femur_moves(s2, angle, side, -22.0)
    # Bellemans 2002: 1.97 mm medially, 0.27 mm laterally, both anterior.
    assert medial == pytest.approx(1.97, abs=0.3)
    assert lateral == pytest.approx(0.27, abs=0.3)
    assert medial > lateral


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("name", ["S1", "S2"])
def test_the_closed_form_compartment_translation_matches_the_motion(side, name):
    trajectory = SCENARIOS[name]
    for angle in (30.0, 95.0, 140.0):
        for ml in (-25.0, 0.0, 18.0):
            assert trajectory.compartment_translation_mm(angle, ml) == pytest.approx(
                _femur_moves(trajectory, angle, side, ml), abs=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
def test_internal_rotation_turns_the_tibia_towards_medial(side):
    medial = _medial(side)
    turned = tibial_motion(Trajectory("T", "turn", axial_rotation_deg=10.0),
                           90.0, centre=CENTRE, anterior=LPS_ANTERIOR, medial=medial)
    anterior_after = turned[:3, :3] @ LPS_ANTERIOR
    assert float(anterior_after @ medial) == pytest.approx(np.sin(np.radians(10.0)))


def test_a_negative_translation_is_rollback():
    back = Trajectory("T", "rollback", ap_translation_mm=-6.0,
                      reference_flexion_deg=100.0)
    assert _femur_moves(back, 100.0, "left", 0.0) == pytest.approx(-6.0)
    assert _femur_moves(back, 50.0, "left", 0.0) == pytest.approx(-3.0)


def test_the_motion_grows_at_a_constant_rate():
    s2 = SCENARIOS["S2"]
    r1, t1 = s2.at(50.0)
    r2, t2 = s2.at(100.0)
    assert r2 == pytest.approx(2 * r1) and t2 == pytest.approx(2 * t1)


def test_the_motion_is_rigid():
    m = tibial_motion(SCENARIOS["S1"], 130.0, centre=CENTRE, anterior=LPS_ANTERIOR,
                      medial=_medial("right"))
    assert m[:3, :3] @ m[:3, :3].T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(m[:3, :3]) == pytest.approx(1.0)


def test_the_pivot_point_does_not_translate_with_a_pure_rotation():
    t = Trajectory("T", "pivot", axial_rotation_deg=15.0, pivot_ml_mm=20.0)
    assert _femur_moves(t, 90.0, "left", 20.0) == pytest.approx(0.0, abs=1e-12)


def test_scenarios_round_trip_through_a_plan_record():
    for trajectory in SCENARIOS.values():
        assert Trajectory.from_dict(trajectory.to_dict()) == trajectory
        assert scenario(trajectory.to_dict()) == trajectory


def test_the_native_scenario_is_labelled_literature_and_cites_its_sources():
    s1 = SCENARIOS["S1"]
    assert s1.status == "literature" and "PLACEHOLDER" not in s1.label
    assert "Pinskerova 2004" in s1.source and "Nakagawa 2000" in s1.source
    assert all(s.status != "placeholder" for s in SCENARIOS.values())


def test_an_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="Unknown trajectory"):
        scenario("S9")


# ----------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------


def _contact(pair, angle, kind="impingement"):
    return Contact(pair=pair, flexion_deg=angle, volume_mm3=1.0, depth_mm=0.1,
                   location=np.zeros(3), kind=kind)


def test_the_search_finds_a_known_contact_angle_to_its_resolution():
    limit = search_max_flexion(
        lambda a: [_contact("wall", a)] if a >= 123.4 else [],
        step_deg=5.0, resolution_deg=0.5)
    assert not limit.censored
    assert 123.4 - 0.5 <= limit.max_flexion_deg < 123.4
    assert limit.first_pair == "wall"


def test_nothing_hit_is_reported_as_censored():
    limit = search_max_flexion(lambda a: [], stop_deg=150.0)
    assert limit.censored and limit.max_flexion_deg == 150.0
    assert limit.stop_reason == "no_contact_in_scanned_models"
    assert limit.flexion_limit_deg is None


def test_by_default_the_search_runs_to_the_fold_not_to_a_chosen_angle():
    """Ruling #26: the stop angle is the end of the motion (180 deg), so a contact the
    old 170 deg ceiling would have censored is found."""
    limit = search_max_flexion(lambda a: [_contact("wall", a)] if a >= 176.2 else [])
    assert limit.stop_deg == FOLD_DEG == 180.0
    assert not limit.censored and limit.stop_reason == "contact"
    assert 175.7 <= limit.max_flexion_deg < 176.2
    assert limit.flexion_limit_deg == limit.max_flexion_deg
    record = limit.to_dict()
    assert record["stop_reason"] == "contact"
    assert record["flexion_limit_deg"] == record["max_flexion_deg"]
    assert record["search"]["stop_deg"] == 180.0


def test_nothing_hit_by_the_fold_is_no_contact_and_never_a_flexion_value():
    seen = []
    limit = search_max_flexion(lambda a: seen.append(a) or [])
    assert seen[-1] == 180.0
    record = limit.to_dict()
    # Backward compatible: the old keys keep their meaning (a lower bound, censored).
    assert record["censored"] and record["max_flexion_deg"] == 180.0
    assert record["stop_reason"] == "no_contact_in_scanned_models"
    assert record["flexion_limit_deg"] is None


def test_no_contact_is_shown_as_the_usual_extreme_but_only_as_a_bound():
    """Author ruling 2026-10-25: show 170 deg, flagged, and never fit it."""
    from tka_planner.core.kinematics import USUAL_EXTREME_DEG
    record = search_max_flexion(lambda a: []).to_dict()
    assert USUAL_EXTREME_DEG == 170.0
    assert record["reported_flexion_deg"] == 170.0 and record["reported_is_bound"]
    assert record["flexion_limit_deg"] is None            # what the fits read
    touched = search_max_flexion(lambda a: [_contact("wall", a)] if a >= 120.0 else [])
    shown = touched.to_dict()
    assert not shown["reported_is_bound"]
    assert shown["reported_flexion_deg"] == shown["flexion_limit_deg"]


def test_the_last_step_lands_on_the_bound():
    seen = []
    search_max_flexion(lambda a: seen.append(a) or [], step_deg=7.0)
    assert seen[-1] == 180.0 and seen[-2] == 175.0


def test_a_bound_past_the_fold_is_refused():
    with pytest.raises(ValueError, match="folded"):
        search_max_flexion(lambda a: [], stop_deg=190.0)


def test_articulation_does_not_stop_flexion():
    limit = search_max_flexion(
        lambda a: [_contact("bearing", a, kind="articulation")]
        + ([_contact("rim", a)] if a >= 97.0 else []))
    assert limit.first_pair == "rim"
    assert 96.5 <= limit.max_flexion_deg < 97.0
    assert limit.deepest_articulation.pair == "bearing"


def test_a_pair_blocked_at_extension_is_listed_not_read_as_zero_flexion():
    limit = search_max_flexion(
        lambda a: [_contact("plan_problem", a)]
        + ([_contact("wall", a)] if a >= 60.0 else []))
    assert [c.pair for c in limit.blocked_at_start] == ["plan_problem"]
    assert limit.first_pair == "wall"
    assert 59.5 <= limit.max_flexion_deg < 60.0


def test_a_pair_blocked_at_extension_counts_again_once_it_has_separated():
    def contacts(a):
        seated = a < 12.0 or a >= 88.0      # in contact at first, again from 88 deg
        return [_contact("insert_vs_bone", a)] if seated else []

    limit = search_max_flexion(contacts)
    assert [c.pair for c in limit.blocked_at_start] == ["insert_vs_bone"]
    assert limit.never_cleared == ()
    assert 87.5 <= limit.max_flexion_deg < 88.0


def test_a_pair_that_never_separates_is_named():
    limit = search_max_flexion(lambda a: [_contact("stuck", a)], stop_deg=60.0)
    assert limit.censored and limit.never_cleared == ("stuck",)


def test_the_envelope_is_the_spread_over_scenarios():
    limits = {
        name: search_max_flexion(lambda a, at=at: [_contact("w", a)] if a >= at else [])
        for name, at in (("S1", 140.0), ("S2", 120.0), ("S3", 130.0))
    }
    spread = envelope(limits)
    assert spread["min_trajectory"] == "S2" and spread["max_trajectory"] == "S1"
    assert spread["spread_deg"] == pytest.approx(20.0, abs=1.0)
    assert spread["no_contact_trajectories"] == []
    limits["S1"] = search_max_flexion(lambda a: [])
    assert envelope(limits)["no_contact_trajectories"] == ["S1"]


# ----------------------------------------------------------------------
# The search on real solids: a bar swinging into a wall
# ----------------------------------------------------------------------


def test_a_swinging_bar_hits_a_wall_at_the_angle_geometry_predicts():
    """A 10 x 10 x 40 mm bar hangs below the hinge (z from -50 to -10) and swings
    about +X; a wall stands at y >= 30. The bar's far edge (y = 5, z = -50) reaches the
    wall when 5 cos(t) + 50 sin(t) = 30."""
    from tka_planner.geom import mesh as gm
    from tka_planner.geom.collision import Solid, overlap

    hinge = motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    bar = Solid.from_mesh(gm.box([10.0, 10.0, 40.0], gm.translation([0, 0, -30.0])))
    wall = Solid.from_mesh(gm.box([40.0, 40.0, 200.0], gm.translation([0, 50.0, 0])))

    def contacts(angle):
        pose = motion.flexion_pose(hinge, flexion_deg=angle)
        found = overlap(bar.transformed(pose), wall)
        return [] if found is None or found.volume_mm3 < 1e-3 else [
            Contact("bar_vs_wall", angle, found.volume_mm3, 0.0, found.centre)]

    expected = np.degrees(np.arcsin(30.0 / np.hypot(5.0, 50.0))
                          - np.arctan2(5.0, 50.0))
    limit = search_max_flexion(contacts, step_deg=5.0, resolution_deg=0.5)
    assert limit.max_flexion_deg == pytest.approx(expected, abs=0.6)
    assert limit.max_flexion_deg < expected + 0.05
