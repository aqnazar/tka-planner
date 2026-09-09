"""Flexion and trial reduction as pure matrix composition."""

import numpy as np
import pytest

from tka_planner.scene import motion


@pytest.fixture
def pivot():
    return motion.pivot_frame(
        origin_mm=np.array([0.0, 0.0, 0.0]), direction=np.array([1.0, 0.0, 0.0])
    )


def test_pivot_frame_puts_x_on_the_flexion_axis(pivot):
    assert pivot[:3, 0] == pytest.approx([1.0, 0.0, 0.0])


def test_pivot_frame_is_orthonormal(pivot):
    assert pivot[:3, :3] @ pivot[:3, :3].T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(pivot[:3, :3]) == pytest.approx(1.0)


def test_pivot_frame_sits_at_its_origin():
    frame = motion.pivot_frame(np.array([5.0, -7.0, 12.0]), np.array([0.0, 1.0, 0.0]))
    assert frame[:3, 3] == pytest.approx([5.0, -7.0, 12.0])


def test_zero_flexion_is_the_identity_pose(pivot):
    assert motion.flexion_pose(pivot, flexion_deg=0.0) == pytest.approx(np.eye(4))


def test_flexion_rotates_about_the_axis_through_its_origin(pivot):
    pose = motion.flexion_pose(pivot, flexion_deg=90.0)
    moved = pose @ np.array([0.0, 0.0, -50.0, 1.0])

    assert np.linalg.norm(moved[:3]) == pytest.approx(50.0)
    assert moved[0] == pytest.approx(0.0, abs=1e-9)


def test_flexion_leaves_a_point_on_the_axis_alone(pivot):
    pose = motion.flexion_pose(pivot, flexion_deg=75.0)
    assert (pose @ np.array([10.0, 0.0, 0.0, 1.0]))[:3] == pytest.approx(
        [10.0, 0.0, 0.0]
    )


def test_a_pivot_away_from_the_origin_still_holds_its_own_point_fixed():
    offset = motion.pivot_frame(
        origin_mm=np.array([5.0, -7.0, 12.0]), direction=np.array([0.0, 1.0, 0.0])
    )
    pose = motion.flexion_pose(offset, flexion_deg=30.0)
    assert (pose @ np.array([5.0, -7.0, 12.0, 1.0]))[:3] == pytest.approx(
        [5.0, -7.0, 12.0]
    )


def test_all_trial_controls_at_zero_is_the_identity(pivot):
    assert motion.trial_pose(pivot) == pytest.approx(np.eye(4))


def test_returning_every_trial_control_to_zero_returns_to_the_identity(pivot):
    """The drift bug, encoded. A pose is a function of the control values alone."""
    motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0, drawer_ap_mm=3.0)
    assert motion.trial_pose(
        pivot, flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0
    ) == pytest.approx(np.eye(4))


def test_a_trial_pose_is_the_same_however_many_times_it_is_asked_for(pivot):
    once = motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0)
    twice = motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0)
    assert once == pytest.approx(twice)


def test_the_drawer_slides_along_the_pivot_rest_axis_not_the_flexed_one(pivot):
    """Anterior means the joint's own anterior, not wherever flexion left the tibia."""
    straight = motion.trial_pose(pivot, drawer_ap_mm=10.0)
    flexed = motion.trial_pose(pivot, flexion_deg=90.0, drawer_ap_mm=10.0)
    unflexed = motion.trial_pose(pivot, flexion_deg=90.0, drawer_ap_mm=0.0)

    rest_y = pivot[:3, 1]
    assert straight[:3, 3] == pytest.approx(rest_y * 10.0)
    assert flexed[:3, 3] - unflexed[:3, 3] == pytest.approx(rest_y * 10.0)


def test_the_drawer_alone_does_not_rotate_anything(pivot):
    pose = motion.trial_pose(pivot, drawer_ap_mm=7.0)
    assert pose[:3, :3] == pytest.approx(np.eye(3))


def test_trial_pose_is_a_rigid_transform(pivot):
    pose = motion.trial_pose(pivot, flexion_deg=33.0, varus_valgus_deg=-4.0,
                             drawer_ap_mm=2.0)
    rotation = pose[:3, :3]

    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_a_stress_turns_about_the_already_flexed_tibia(pivot):
    """A stress exam is relative to the tibia's current position, not the femur's frame."""
    flexed = motion.trial_pose(pivot, flexion_deg=90.0)
    stressed = motion.trial_pose(pivot, flexion_deg=90.0, varus_valgus_deg=10.0)

    # The stress axis in world terms is the flexed tibia's own Y, not the pivot's.
    flexed_y = (flexed[:3, :3] @ pivot[:3, :3])[:, 1]
    relative = stressed[:3, :3] @ flexed[:3, :3].T
    angle = np.degrees(np.arccos((np.trace(relative) - 1.0) / 2.0))
    axis = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])
    axis = axis / np.linalg.norm(axis)

    assert angle == pytest.approx(10.0)
    assert abs(float(np.dot(axis, flexed_y / np.linalg.norm(flexed_y)))) == (
        pytest.approx(1.0)
    )


def test_flexion_pose_agrees_with_trial_pose_at_the_same_angle(pivot):
    assert motion.flexion_pose(pivot, flexion_deg=55.0) == pytest.approx(
        motion.trial_pose(pivot, flexion_deg=55.0)
    )


def test_rotation_about_an_arbitrary_axis_is_orthonormal():
    rotation = motion.rotation_about(np.array([0.3, 0.5, -0.8]), 27.0)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)


@pytest.fixture
def knee():
    """A synthetic knee with its frames and a plan, as the builder will supply them."""
    from tests.synthetic import synthetic_knee
    from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
    from tka_planner.core.planning import MECHANICAL, plan_alignment

    landmarks = synthetic_knee()
    femoral_frame = build_femoral_frame(landmarks)
    tibial_frame = build_tibial_frame(landmarks)
    plan = plan_alignment(
        landmarks, femoral_frame, tibial_frame,
        target=MECHANICAL, femoral_thickness_mm=9.0, tibial_resection_mm=8.0,
        native_slope_deg=5.0,
    )
    return landmarks, femoral_frame, plan


def test_the_flexion_axis_is_the_midpoint_of_the_epicondyles(knee):
    landmarks, femoral_frame, plan = knee
    origin, direction = motion.flexion_axis(plan, femoral_frame, landmarks)

    lateral, medial = landmarks.require(
        "femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus"
    )
    assert origin == pytest.approx((np.asarray(lateral) + np.asarray(medial)) / 2.0)
    assert direction == pytest.approx(femoral_frame.y_patient_left)


def test_the_flexion_axis_falls_back_to_the_component_origin_without_epicondyles(knee):
    _, femoral_frame, plan = knee
    origin, _ = motion.flexion_axis(plan, femoral_frame, landmarks=None)

    assert origin == pytest.approx(plan.components["femoral_component"][:3, 3])


def test_a_pivot_on_the_real_flexion_axis_holds_that_axis_fixed(knee):
    landmarks, femoral_frame, plan = knee
    origin, direction = motion.flexion_axis(plan, femoral_frame, landmarks)
    pivot = motion.pivot_frame(origin, direction)
    pose = motion.flexion_pose(pivot, flexion_deg=60.0)

    on_axis = np.append(origin + np.asarray(direction) * 25.0, 1.0)
    assert (pose @ on_axis)[:3] == pytest.approx(on_axis[:3])
