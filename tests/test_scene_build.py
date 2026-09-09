"""Building a scene from a plan, with no Blender anywhere."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.planning import MECHANICAL, plan_alignment
from tka_planner.geom import mesh as gm
from tka_planner.scene import build as sb
from tka_planner.scene.model import FEMORAL, TIBIAL


@pytest.fixture
def knee(tmp_path):
    """A synthetic knee written to STL, plus its plan and frames."""
    landmarks = synthetic_knee()
    femoral_frame = build_femoral_frame(landmarks)
    tibial_frame = build_tibial_frame(landmarks)
    femur_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "femur.stl")
    tibia_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "tibia.stl")

    plan = plan_alignment(
        landmarks, femoral_frame, tibial_frame,
        target=MECHANICAL, femoral_thickness_mm=9.0, tibial_resection_mm=8.0,
        native_slope_deg=5.0,
    )
    return landmarks, femoral_frame, tibial_frame, plan, femur_path, tibia_path


@pytest.fixture
def build(knee):
    landmarks, femoral_frame, tibial_frame, plan, femur_path, tibia_path = knee

    def make(**overrides):
        options = dict(
            femur_path=femur_path, tibia_path=tibia_path, plan=plan,
            femoral_frame=femoral_frame, tibial_frame=tibial_frame,
            landmarks=landmarks, show_landmarks=True,
        )
        options.update(overrides)
        return sb.build_scene(**options)

    return make


@pytest.fixture
def scene(build):
    return build()


def test_both_bones_are_present_in_their_own_sets(scene):
    assert scene.nodes["Femur"].set_id == FEMORAL
    assert scene.nodes["Tibia"].set_id == TIBIAL


def test_a_cut_plane_is_built_for_every_resection(knee, scene):
    plan = knee[3]
    for name in plan.resections:
        assert name in scene.nodes


def test_a_cut_plane_sits_on_its_resection(knee, scene):
    plan = knee[3]
    resection = plan.resections["femoral_distal"]
    world = scene.world("femoral_distal")

    assert world[:3, 3] == pytest.approx(resection.point)
    assert world[:3, 2] == pytest.approx(
        np.asarray(resection.normal) / np.linalg.norm(resection.normal)
    )


def test_the_femoral_plane_belongs_to_the_femoral_set(scene):
    assert scene.nodes["femoral_distal"].set_id == FEMORAL
    assert scene.nodes["tibial_proximal"].set_id == TIBIAL


def test_planes_can_be_turned_off(build):
    scene = build(show_planes=False)
    assert "femoral_distal" not in scene.nodes


def test_both_mechanical_axes_are_drawn(scene):
    assert "FemoralMechanicalAxis" in scene.nodes
    assert "TibialMechanicalAxis" in scene.nodes


def test_the_axis_lies_along_the_frame_it_was_built_from(knee, scene):
    femoral_frame = knee[1]
    direction = np.asarray(femoral_frame.z_proximal, dtype=float)
    direction = direction / np.linalg.norm(direction)

    assert scene.world("FemoralMechanicalAxis")[:3, 2] == pytest.approx(direction)


def test_axes_can_be_turned_off(build):
    scene = build(show_axes=False)
    assert "FemoralMechanicalAxis" not in scene.nodes


def test_landmarks_become_nodes_tagged_as_landmarks(knee, scene):
    landmarks = knee[0]
    usable = [landmark for landmark in landmarks if landmark.is_usable]
    tagged = [node for node in scene.nodes.values() if node.tags.get("landmark")]

    assert len(tagged) == len(usable)


def test_a_landmark_sits_where_the_landmark_set_says(knee, scene):
    landmarks = knee[0]
    landmark = next(one for one in landmarks if one.is_usable)

    assert scene.world(landmark.id)[:3, 3] == pytest.approx(landmark.position_mm)


def test_landmarks_belong_to_the_set_of_their_bone(scene):
    femoral = [
        node for node in scene.nodes.values()
        if node.tags.get("landmark") and node.name.startswith("femur.")
    ]
    tibial = [
        node for node in scene.nodes.values()
        if node.tags.get("landmark") and node.name.startswith("tibia.")
    ]

    assert femoral and all(node.set_id == FEMORAL for node in femoral)
    assert tibial and all(node.set_id == TIBIAL for node in tibial)


def test_a_landmark_carries_its_provenance_status(scene):
    marker = next(n for n in scene.nodes.values() if n.tags.get("landmark"))
    assert marker.tags["status"] in {"present", "estimated", "derived"}


def test_landmarks_share_one_marker_mesh(scene):
    ids = {
        node.mesh_id for node in scene.nodes.values() if node.tags.get("landmark")
    }
    assert len(ids) == 1


def test_landmarks_can_be_built_hidden(build):
    scene = build(show_landmarks=False)
    marker = next(n for n in scene.nodes.values() if n.tags.get("landmark"))
    assert marker.visible is False


def test_the_pivot_is_recorded_on_the_scene(knee, scene):
    from tka_planner.scene import motion

    landmarks, femoral_frame, _, plan, _, _ = knee
    origin, direction = motion.flexion_axis(plan, femoral_frame, landmarks)

    assert scene.pivot == pytest.approx(motion.pivot_frame(origin, direction))


def test_the_insert_spacer_is_built_at_the_requested_thickness(build):
    scene = build(insert_thickness_mm=9.0, insert_footprint_mm=(60.0, 40.0))
    assert gm.volume(scene.mesh_for(sb.INSERT_SPACER)) == pytest.approx(
        60.0 * 40.0 * 9.0, rel=1e-6
    )


def test_the_insert_spacer_belongs_to_the_tibial_set(build):
    scene = build(insert_thickness_mm=9.0)
    assert scene.nodes[sb.INSERT_SPACER].set_id == TIBIAL


def test_no_insert_spacer_when_the_insert_is_off(scene):
    assert sb.INSERT_SPACER not in scene.nodes


def test_a_missing_component_mesh_is_noted_rather_than_raised(build, tmp_path):
    scene = build(components={
        "femoral_component": {
            "path": str(tmp_path / "nope.stl"), "scale": 1.0, "group": "femoral",
        }
    })
    assert "femoral_component" not in scene.nodes
    assert any("not found" in note for note in scene.notes)


def test_a_component_is_seated_on_its_plan_pose(build, knee, tmp_path):
    plan = knee[3]
    path = gm.write_stl(gm.box(20.0), tmp_path / "femoral.stl")
    scene = build(components={
        "femoral_component": {"path": str(path), "scale": 1.0, "group": "femoral"},
    })

    assert scene.world("femoral_component")[:3, 3] == pytest.approx(
        plan.components["femoral_component"][:3, 3]
    )


def test_a_component_scale_does_not_move_its_seating(build, knee, tmp_path):
    plan = knee[3]
    path = gm.write_stl(gm.box(20.0), tmp_path / "femoral.stl")
    scene = build(components={
        "femoral_component": {"path": str(path), "scale": 1.4, "group": "femoral"},
    })

    assert scene.world("femoral_component")[:3, 3] == pytest.approx(
        plan.components["femoral_component"][:3, 3]
    )


def test_seat_scales_about_the_cad_origin():
    pose = gm.translation((10.0, 20.0, 30.0))
    seated = sb.seat(pose, 2.0)

    assert seated[:3, 3] == pytest.approx([10.0, 20.0, 30.0])
    assert np.linalg.norm(seated[:3, 0]) == pytest.approx(2.0)


def test_building_imports_no_bpy():
    import sys

    assert "bpy" not in sys.modules
