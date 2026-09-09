"""Re-posing a built scene from a re-planned SurgicalPlan."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.planning import MECHANICAL, Adjustments, plan_alignment
from tka_planner.geom import mesh as gm
from tka_planner.scene import build as sb
from tka_planner.scene import update as su


@pytest.fixture
def case(tmp_path):
    landmarks = synthetic_knee()
    femoral_frame = build_femoral_frame(landmarks)
    tibial_frame = build_tibial_frame(landmarks)
    femur_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "femur.stl")
    tibia_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "tibia.stl")
    component_path = gm.write_stl(gm.box(20.0), tmp_path / "femoral.stl")

    def make(**adjustments):
        return plan_alignment(
            landmarks, femoral_frame, tibial_frame,
            target=MECHANICAL, femoral_thickness_mm=9.0, tibial_resection_mm=8.0,
            native_slope_deg=5.0,
            adjustments=Adjustments(**adjustments),
        )

    plan = make()
    scene = sb.build_scene(
        femur_path=femur_path, tibia_path=tibia_path, plan=plan,
        femoral_frame=femoral_frame, tibial_frame=tibial_frame,
        landmarks=landmarks, show_landmarks=True,
        components={
            "femoral_component": {
                "path": str(component_path), "scale": 1.0, "group": "femoral",
                "source_ml": 65.0,
            },
        },
        insert_thickness_mm=9.0, insert_footprint_mm=(60.0, 40.0),
    )
    return scene, plan, make


def test_an_unchanged_plan_moves_the_cut_plane_nowhere(case):
    scene, plan, _ = case
    before = scene.world("femoral_distal").copy()
    su.update_scene(scene, plan)

    assert scene.world("femoral_distal") == pytest.approx(before)


def test_an_unchanged_plan_reports_an_empty_delta(case):
    scene, plan, _ = case
    delta = su.update_scene(scene, plan)

    assert delta.poses == {}


def test_a_deeper_tibial_resection_moves_the_tibial_plane(case):
    scene, _, make = case
    before = scene.world("tibial_proximal").copy()
    su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert scene.world("tibial_proximal")[:3, 3] != pytest.approx(before[:3, 3])


def test_the_delta_names_only_what_moved(case):
    scene, _, make = case
    delta = su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert "tibial_proximal" in delta.poses
    assert "Femur" not in delta.poses
    assert "femoral_distal" not in delta.poses


def test_the_delta_carries_the_plane_pose_as_a_matrix(case):
    scene, _, make = case
    delta = su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert np.asarray(delta.poses["tibial_proximal"]).shape == (4, 4)


def test_a_component_follows_its_plan_pose(case):
    scene, _, make = case
    before = scene.world("femoral_component").copy()
    su.update_scene(scene, make(femoral_resection_delta_mm=2.0))

    assert scene.world("femoral_component")[:3, 3] != pytest.approx(before[:3, 3])


def test_changing_the_implant_size_rescales_without_replacing_the_mesh(case):
    scene, plan, _ = case
    before = scene.nodes["femoral_component"].mesh_id
    delta = su.update_scene(scene, plan, implant_ml_mm=70.0)

    assert delta.meshes == {}
    assert scene.nodes["femoral_component"].mesh_id == before


def test_a_new_size_rescales_from_the_width_the_mesh_was_exported_at(case):
    scene, plan, _ = case
    su.update_scene(scene, plan, implant_ml_mm=71.5)

    assert scene.nodes["femoral_component"].tags["scale"] == pytest.approx(71.5 / 65.0)
    assert np.linalg.norm(scene.world("femoral_component")[:3, 0]) == pytest.approx(
        71.5 / 65.0
    )


def test_the_insert_slab_is_rebuilt_at_a_new_thickness(case):
    scene, plan, _ = case
    delta = su.update_scene(scene, plan, insert_thickness_mm=12.0)

    assert sb.INSERT_SPACER in delta.meshes
    assert gm.volume(scene.mesh_for(sb.INSERT_SPACER)) == pytest.approx(
        60.0 * 40.0 * 12.0, rel=1e-6
    )


def test_the_insert_keeps_its_footprint_when_its_thickness_changes(case):
    scene, plan, _ = case
    su.update_scene(scene, plan, insert_thickness_mm=12.0)
    extent = scene.mesh_for(sb.INSERT_SPACER).extent

    assert extent[0] == pytest.approx(60.0)
    assert extent[1] == pytest.approx(40.0)


def test_turning_the_insert_off_hides_the_slab(case):
    scene, plan, _ = case
    delta = su.update_scene(scene, plan, insert_thickness_mm=None)

    assert delta.visibility[sb.INSERT_SPACER] is False
    assert scene.nodes[sb.INSERT_SPACER].visible is False


def test_turning_the_insert_back_on_shows_the_slab(case):
    scene, plan, _ = case
    su.update_scene(scene, plan, insert_thickness_mm=None)
    delta = su.update_scene(scene, plan, insert_thickness_mm=9.0)

    assert delta.visibility[sb.INSERT_SPACER] is True


def test_isolate_landmarks_hides_everything_else(case):
    scene, _, _ = case
    su.isolate_landmarks(scene, True)
    marker = next(n for n in scene.nodes.values() if n.tags.get("landmark"))

    assert scene.nodes["Femur"].visible is False
    assert marker.visible is True


def test_isolate_landmarks_restores_what_it_hid(case):
    scene, _, _ = case
    su.isolate_landmarks(scene, True)
    su.isolate_landmarks(scene, False)

    assert scene.nodes["Femur"].visible is True


def test_isolate_landmarks_leaves_something_already_hidden_hidden(case):
    scene, plan, _ = case
    su.update_scene(scene, plan, insert_thickness_mm=None)
    su.isolate_landmarks(scene, True)
    su.isolate_landmarks(scene, False)

    assert scene.nodes[sb.INSERT_SPACER].visible is False


def test_set_visibility_reports_only_real_changes(case):
    scene, _, _ = case
    first = su.set_visibility(scene, lambda node: node.name == "Femur", False)
    again = su.set_visibility(scene, lambda node: node.name == "Femur", False)

    assert first.visibility == {"Femur": False}
    assert again.visibility == {}


def test_plan_warnings_reach_the_delta(case):
    scene, _, make = case
    delta = su.update_scene(scene, make(femoral_varus_delta_deg=6.0))

    assert isinstance(delta.notes, list)


def test_the_delta_survives_serialisation(case):
    scene, _, make = case
    payload = su.update_scene(scene, make(tibial_slope_delta_deg=2.0)).to_dict()

    assert isinstance(payload["poses"]["tibial_proximal"], list)
    assert len(payload["poses"]["tibial_proximal"]) == 4


def test_updating_never_touches_a_set_pose(case):
    scene, _, make = case
    from tka_planner.scene.model import TIBIAL

    su.update_scene(scene, make(tibial_resection_delta_mm=3.0))
    assert scene.set_poses[TIBIAL] == pytest.approx(np.eye(4))
