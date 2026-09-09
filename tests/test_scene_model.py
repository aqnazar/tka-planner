"""The scene model, and the rest-basis rule that keeps poses from drifting."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene.model import FEMORAL, TIBIAL, Node, Scene, SceneDelta


@pytest.fixture
def scene():
    built = Scene()
    built.add(Node(name="Femur", set_id=FEMORAL), gm.box(10.0))
    built.add(
        Node(name="Tibia", set_id=TIBIAL, rest=gm.translation((0.0, 0.0, -50.0))),
        gm.box((12.0, 9.0, 40.0)),
    )
    return built


def test_a_node_with_no_set_pose_is_at_its_rest(scene):
    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -50.0)))


def test_a_set_pose_composes_onto_the_rest_basis(scene):
    scene.set_pose(TIBIAL, gm.translation((0.0, 0.0, 5.0)))
    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -45.0)))


def test_a_set_pose_does_not_reach_another_set(scene):
    scene.set_pose(TIBIAL, gm.translation((0.0, 0.0, 5.0)))
    assert scene.world("Femur") == pytest.approx(np.eye(4))


def test_setting_the_same_pose_twice_is_idempotent(scene):
    pose = gm.plane_matrix((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
    scene.set_pose(TIBIAL, pose)
    once = scene.world("Tibia")
    scene.set_pose(TIBIAL, pose)

    assert scene.world("Tibia") == pytest.approx(once)


def test_returning_a_set_pose_to_identity_returns_the_node_to_rest(scene):
    """The trial-rig drift bug, encoded. A pose is assigned, never accumulated."""
    scene.set_pose(TIBIAL, gm.plane_matrix((9.0, 9.0, 9.0), (1.0, 1.0, 0.0)))
    scene.set_pose(TIBIAL, np.eye(4))

    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -50.0)))


def test_a_set_pose_never_alters_a_rest_transform(scene):
    before = scene.nodes["Tibia"].rest.copy()
    scene.set_pose(TIBIAL, gm.plane_matrix((4.0, 0.0, 1.0), (0.0, 1.0, 1.0)))

    assert scene.nodes["Tibia"].rest == pytest.approx(before)


def test_a_set_pose_must_be_four_by_four(scene):
    with pytest.raises(ValueError, match="4x4"):
        scene.set_pose(TIBIAL, np.eye(3))


def test_identical_geometry_is_stored_once(scene):
    """Meshes are content-addressed, so a shell sharing a bone's geometry costs nothing."""
    scene.add(Node(name="FemurCopy", set_id=FEMORAL), gm.box(10.0))

    assert scene.nodes["FemurCopy"].mesh_id == scene.nodes["Femur"].mesh_id
    assert len(scene.meshes) == 2


def test_a_node_can_reference_an_existing_mesh_without_supplying_it(scene):
    scene.add(Node(name="Shell", set_id=FEMORAL,
                   mesh_id=scene.nodes["Femur"].mesh_id))

    assert scene.mesh_for("Shell") is scene.mesh_for("Femur")


def test_replacing_a_mesh_changes_only_that_node(scene):
    before = scene.nodes["Tibia"].mesh_id
    scene.replace_mesh("Femur", gm.box(3.0))

    assert scene.nodes["Femur"].mesh_id != before
    assert scene.nodes["Tibia"].mesh_id == before


def test_nodes_in_lists_only_that_set(scene):
    assert [node.name for node in scene.nodes_in(TIBIAL)] == ["Tibia"]


def test_world_of_an_unknown_node_raises(scene):
    with pytest.raises(KeyError, match="Patella"):
        scene.world("Patella")


def test_a_delta_serialises_to_plain_types():
    delta = SceneDelta(
        poses={"TibialSet": np.eye(4)},
        visibility={"Femur": False},
        scalars={"extension_gap_medial_mm": 9.25},
    )
    payload = delta.to_dict()

    assert payload["poses"]["TibialSet"] == [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ]
    assert payload["visibility"] == {"Femur": False}
    assert payload["scalars"]["extension_gap_medial_mm"] == 9.25


def test_merging_deltas_keeps_the_later_value():
    first = SceneDelta(poses={"A": np.eye(4)}, scalars={"x": 1})
    second = SceneDelta(poses={"A": gm.translation((1.0, 0.0, 0.0))}, scalars={"y": 2})
    merged = first.merge(second)

    assert merged.poses["A"] == pytest.approx(gm.translation((1.0, 0.0, 0.0)))
    assert merged.scalars == {"x": 1, "y": 2}


def test_merging_deltas_leaves_the_originals_alone():
    first = SceneDelta(scalars={"x": 1})
    first.merge(SceneDelta(scalars={"y": 2}))

    assert first.scalars == {"x": 1}
