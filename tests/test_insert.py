"""The solved insert and the Reduce-mode distraction, from constructed parts."""

import numpy as np
import pytest

from tka_planner.core.insert import dish_geometry, solve_insert, stretch_matrix
from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.scene import motion

UP = np.array([0.0, 0.0, 1.0])
LATERAL = np.array([1.0, 0.0, 0.0])


def block(size, centre):
    return gm.box(size, gm.translation(centre))


def test_the_clearance_is_what_the_insert_must_grow():
    insert = block((60.0, 40.0, 8.0), (0.0, 0.0, 4.0))      # top at z = 8
    femoral = block((60.0, 40.0, 10.0), (0.0, 0.0, 16.0))   # bottom at z = 11

    solution = solve_insert(femoral, insert, normal=UP, lateral=LATERAL)

    assert solution.change_mm == pytest.approx(3.0, abs=0.05)
    assert solution.thickness_at_contact_mm == pytest.approx(11.0, abs=0.05)


def test_an_overlap_means_the_insert_must_be_thinner():
    insert = block((60.0, 40.0, 8.0), (0.0, 0.0, 4.0))
    femoral = block((60.0, 40.0, 10.0), (0.0, 0.0, 11.0))   # bottom at z = 6

    assert solve_insert(femoral, insert, normal=UP,
                        lateral=LATERAL).change_mm == pytest.approx(-2.0, abs=0.05)


def test_the_tighter_compartment_decides_and_the_other_shows_the_imbalance():
    insert = block((60.0, 40.0, 8.0), (0.0, 0.0, 4.0))
    lateral_condyle = block((25.0, 40.0, 10.0), (17.5, 0.0, 15.0))   # 2 mm clear
    medial_condyle = block((25.0, 40.0, 10.0), (-17.5, 0.0, 17.0))   # 4 mm clear
    femoral = Mesh(
        vertices=np.vstack([lateral_condyle.vertices, medial_condyle.vertices]),
        faces=np.vstack([lateral_condyle.faces,
                         medial_condyle.faces + len(lateral_condyle.vertices)]),
    )

    solution = solve_insert(femoral, insert, normal=UP, lateral=LATERAL)

    assert solution.change_mm == pytest.approx(2.0, abs=0.05)
    assert solution.lateral_clearance_mm == pytest.approx(2.0, abs=0.05)
    assert solution.medial_clearance_mm == pytest.approx(4.0, abs=0.05)


def test_a_flat_insert_is_as_thick_as_its_dish():
    floor, dish = dish_geometry(block((60.0, 40.0, 8.0), (0.0, 0.0, 7.0)))

    assert floor == pytest.approx(3.0)
    assert dish == pytest.approx(11.0, abs=0.01)


def test_the_stretch_leaves_the_floor_on_the_tray():
    matrix = stretch_matrix(3.0, 1.5)

    floor = matrix @ np.array([5.0, 2.0, 3.0, 1.0])
    top = matrix @ np.array([5.0, 2.0, 11.0, 1.0])
    assert floor[:3] == pytest.approx([5.0, 2.0, 3.0])
    assert top[2] == pytest.approx(3.0 + 1.5 * 8.0)


def test_the_viewer_insert_is_stretched_to_the_solved_thickness(session):
    from tka_planner.pipeline import insert_display

    display = insert_display(session.plan, session._components())
    node = session.scene.nodes["tibial_insert"]
    scale = node.tags["scale"]

    assert display["insert_thickness_mm"] == pytest.approx(
        session.plan.diagnostics["insert_thickness_mm"], abs=1e-3)
    assert display["library_thickness_mm"] * display["stretch"] == pytest.approx(
        max(display["insert_thickness_mm"], 0.5), rel=1e-6)
    # The node's own axis is scaled by the stretch on top of the size's uniform scale.
    local_z = node.rest[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.linalg.norm(local_z) == pytest.approx(scale * display["stretch"], rel=1e-6)


def test_a_thicker_insert_stretches_the_viewer_insert(session):
    before = session.scene.nodes["tibial_insert"].rest.copy()
    session.replan(insert_thickness_delta_mm=2.0)

    after = session.scene.nodes["tibial_insert"].rest
    assert np.linalg.norm(after[:3, 2]) > np.linalg.norm(before[:3, 2])


# ----------------------------------------------------------------------
# Distraction
# ----------------------------------------------------------------------

def test_distraction_pulls_the_tibia_along_its_axis():
    pivot = motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    distal = np.array([0.0, 0.0, -1.0])

    pose = motion.trial_pose(pivot, distraction_mm=6.0, distal=distal)

    assert pose[:3, 3] == pytest.approx([0.0, 0.0, -6.0])
    assert pose[:3, :3] == pytest.approx(np.eye(3))


def test_in_flexion_the_distraction_still_runs_along_the_tibia():
    pivot = motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    distal = np.array([0.0, 0.0, -1.0])

    pose = motion.trial_pose(pivot, flexion_deg=90.0, distraction_mm=6.0,
                             distal=distal)
    tibial_axis = pose[:3, :3] @ distal

    assert pose[:3, 3] == pytest.approx(6.0 * tibial_axis, abs=1e-9)


def test_a_distraction_without_a_direction_is_refused():
    pivot = motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="distal"):
        motion.trial_pose(pivot, distraction_mm=2.0)


def test_the_session_opens_a_gap_without_touching_the_plan(session):
    plan_before = session.plan
    delta = session.set_trial(distraction_mm=5.0)

    moved = delta.poses["TibialSet"][:3, 3]
    assert np.linalg.norm(moved) == pytest.approx(5.0, abs=1e-9)
    assert float(moved @ session.measurement.tibial_frame.z_proximal) < 0
    assert session.plan is plan_before
    assert delta.scalars["distraction_mm"] == 5.0
