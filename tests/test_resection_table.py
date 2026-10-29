"""The resection table, on blocks of bone whose answers can be worked out by hand.

The femur here is a box in the femoral component's own frame (the pose is the
identity: +X anterior, +Y patient-left, +Z proximal, origin on the distal cut), so every
cut's thickness is a difference of two coordinates.
"""

import numpy as np
import pytest

from tka_planner.core.insert import surface_samples
from tka_planner.core.resection_table import (
    BoxProfile,
    coronal_angle_deg,
    femoral_resection_rows,
    inside_along_z,
    tibial_resection_rows,
)
from tka_planner.geom import mesh as gm

COS45 = np.cos(np.radians(45.0))


def femur_block(x=(-30.0, 30.0), y=(-35.0, 35.0), z=(-8.0, 60.0)):
    size = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    centre = (sum(x) / 2.0, sum(y) / 2.0, sum(z) / 2.0)
    return gm.box(size, gm.translation(centre))


def profile(flare=0.0):
    """Posterior cut at x = -25, anterior at x = 20, 10 mm posterior chamfer, 20 mm
    distal face, chamfers at 45 degrees, flange 50 mm high."""
    return BoxProfile(posterior_x_mm=-25.0, anterior_x_at_distal_mm=20.0,
                      anterior_flare_deg=flare, flange_height_mm=50.0,
                      posterior_chamfer_mm=10.0, distal_face_mm=20.0, chamfer_deg=45.0)


def rows_for(femur, box=None, medial_sign=1.0):
    samples = surface_samples(femur, per_mm2=4.0)
    rows = femoral_resection_rows(femur, np.eye(4), medial_sign=medial_sign,
                                  samples=samples, profile=box)
    return {row["cut"]: row for row in rows}


def test_each_femoral_cut_takes_what_lies_beyond_it():
    rows = rows_for(femur_block(), profile())

    for side in ("medial_mm", "lateral_mm"):
        assert rows["distal"][side] == pytest.approx(8.0, abs=0.01)
        assert rows["posterior"][side] == pytest.approx(5.0, abs=0.01)
        assert rows["anterior"][side] == pytest.approx(10.0, abs=0.01)


def test_a_filled_corner_is_what_a_chamfer_takes():
    """After the distal and posterior cuts the corner between them is cut bone, and the
    furthest point from the chamfer: 10 mm of chamfer at 45 degrees is 7.07 mm deep."""
    rows = rows_for(femur_block(), profile())

    for side in ("medial_mm", "lateral_mm"):
        assert rows["posterior_chamfer"][side] == pytest.approx(10.0 * COS45, abs=0.01)
        # The anterior chamfer runs from the front of the distal face (x = 5) to the
        # anterior cut (x = 20): 15 mm at 45 degrees.
        assert rows["anterior_chamfer"][side] == pytest.approx(15.0 * COS45, abs=0.01)


def test_a_corner_the_bone_does_not_reach_falls_back_to_the_surface():
    """A condyle that stops 3 mm short of the posterior cut leaves the corner empty."""
    rows = rows_for(femur_block(x=(-22.0, 30.0)), profile())

    assert rows["posterior"]["medial_mm"] == 0.0
    # The bone's back face at x = -22 is 7 mm in front of the chamfer's foot (x = -15
    # is its start on the distal plane), so it is (7 - z) cos 45 deep at height z.
    assert rows["posterior_chamfer"]["medial_mm"] == pytest.approx(7.0 * COS45, abs=0.3)


def test_the_anterior_cut_is_measured_perpendicular_to_its_flare():
    flare = 10.0
    rows = rows_for(femur_block(), profile(flare=flare))
    # The front face (x = 30) is furthest from the flared plane at z = 0, 10 mm in AP.
    assert rows["anterior"]["lateral_mm"] == pytest.approx(
        10.0 * np.cos(np.radians(flare)), abs=0.05)


def test_compartments_follow_the_medial_side():
    """A distal condyle 3 mm longer on patient-left is medial on a right knee."""
    from tka_planner.core.meshio import Mesh

    longer = femur_block(y=(0.0, 35.0), z=(-11.0, 60.0))
    shorter = femur_block(y=(-35.0, 0.0))
    femur = Mesh(np.vstack([longer.vertices, shorter.vertices]),
                 np.vstack([longer.faces, shorter.faces + len(longer.vertices)]))

    right = rows_for(femur, medial_sign=+1.0)["distal"]
    left = rows_for(femur, medial_sign=-1.0)["distal"]

    assert right["medial_mm"] == pytest.approx(11.0, abs=0.01)
    assert right["lateral_mm"] == pytest.approx(8.0, abs=0.01)
    assert left["medial_mm"] == pytest.approx(8.0, abs=0.01)


def test_without_a_box_only_the_distal_cut_is_measured():
    rows = rows_for(femur_block())

    assert rows["distal"]["medial_mm"] == pytest.approx(8.0, abs=0.01)
    for cut in ("posterior", "anterior", "posterior_chamfer", "anterior_chamfer"):
        assert rows[cut]["medial_mm"] is None
        assert "not placed" in rows[cut]["definition"]


def test_each_row_says_where_its_value_was_found():
    rows = rows_for(femur_block(), profile())
    point = rows["posterior"]["points_mm"]["medial"]

    assert point[0] == pytest.approx(-30.0, abs=0.01)  # the back face of the block
    assert point[1] > 0  # medial is +Y with medial_sign +1


def test_a_ray_counts_its_way_out_of_a_closed_surface():
    cube = gm.box(10.0)
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0], [4.0, -4.0, -4.9],
                       [6.0, 0.0, 0.0]])

    inside = inside_along_z(cube, np.eye(4), np.eye(3), points)

    assert inside.tolist() == [True, False, True, False]


# ----------------------------------------------------------------------
# The tibia
# ----------------------------------------------------------------------


def tibia_block():
    """A block whose top is 10 mm above the cut at z = 0, 80 mm wide."""
    return gm.box((80.0, 50.0, 40.0), gm.translation((0.0, 0.0, -10.0)))


def test_the_tibia_is_read_at_each_plateaus_lowest_point():
    rows = tibial_resection_rows(
        tibia_block(), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        medial=(-1.0, 0.0, 0.0), anterior=(0.0, -1.0, 0.0),
        plateau_points={"medial": (-20.0, 0.0, 4.0), "lateral": (20.0, 0.0, 7.5)},
        axis=(0.0, 0.0, 1.0), slope_deg=3.0,
    )
    row = rows[0]

    assert row["medial_mm"] == pytest.approx(4.0)
    assert row["lateral_mm"] == pytest.approx(7.5)
    assert row["highest_mm"] == {"medial": pytest.approx(10.0),
                                 "lateral": pytest.approx(10.0)}
    assert row["angle_deg"]["posterior_slope"] == 3.0
    assert row["angle_deg"]["varus_from_mechanical_axis"] == pytest.approx(0.0)


def test_the_eminence_is_not_read_as_a_plateau():
    """A spine standing 15 mm above the cut at the centre is left out of both halves."""
    from tka_planner.core.meshio import Mesh

    block = tibia_block()
    spine = gm.box((6.0, 10.0, 20.0), gm.translation((0.0, 0.0, 5.0)))
    tibia = Mesh(np.vstack([block.vertices, spine.vertices]),
                 np.vstack([block.faces, spine.faces + len(block.vertices)]))
    row = tibial_resection_rows(
        tibia, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), medial=(-1.0, 0.0, 0.0),
        anterior=(0.0, -1.0, 0.0),
        plateau_points={"medial": (-20.0, 0.0, 4.0), "lateral": (20.0, 0.0, 4.0)},
    )[0]

    assert row["highest_mm"]["medial"] == pytest.approx(10.0)


def test_a_varus_cut_leaves_the_medial_side_lower():
    medial = np.array([-1.0, 0.0, 0.0])
    tilt = np.radians(3.0)
    normal = np.array([0.0, 0.0, 1.0]) * np.cos(tilt) + medial * np.sin(tilt)

    assert coronal_angle_deg(normal, (0.0, 0.0, 1.0), medial) == pytest.approx(3.0)
    # The plane through the origin with that normal is lower on the medial side.
    medial_height = -np.dot(normal[:2], (-20.0, 0.0)) / normal[2]
    assert medial_height < 0.0


# ----------------------------------------------------------------------
# On a planned case
# ----------------------------------------------------------------------


def test_a_planned_case_carries_the_table_in_the_standard_layout(session):
    table = session.resection_table

    assert [row["cut"] for row in table["rows"]] == [
        "distal", "posterior", "anterior", "posterior_chamfer", "anterior_chamfer",
        "tibial"]
    tibial = table["rows"][-1]
    cut = session.plan.resections["tibial_proximal"]
    # At the stylus point the table is the plan's own depth.
    assert tibial["medial_mm"] == pytest.approx(cut.medial_depth_mm, abs=0.01)
    assert tibial["lateral_mm"] == pytest.approx(cut.lateral_depth_mm, abs=0.01)


def test_the_table_is_in_the_plan_document_and_the_readout(session):
    from tka_planner.pipeline import plan_document, validate_plan_document

    document = plan_document(session.measurement, session.plan, session.sizing)
    assert document["resection_table"]["rows"]
    assert validate_plan_document(document) == []
    lines = session.report_lines()
    assert "HEAD|Resection table (medial / lateral)" in lines
    assert any(line.startswith("Distal femur|") for line in lines)
    assert any(line.startswith("Tibia at stylus point|") for line in lines)
