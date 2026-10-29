"""Where a flexion search stops (author ruling #26), on toy bones with known answers.

The search runs until the models meet, bounded only by the fold (180 deg). Toy solids
in LPS (+X patient-left, +Y posterior, +Z proximal), hinge along +X through the origin;
flexion swings the tibia from below the hinge, posteriorly (+Y), and up. Bones are
20 mm wide boxes; the femoral component and the insert are put far out along the hinge,
where a flexion about it can never bring them near anything.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tka_planner.core.frames import AnatomicalFrame
from tka_planner.core.kinematics import FOLD_DEG, search_max_flexion
from tka_planner.core.provenance import Quality
from tka_planner.core.sides import Side
from tka_planner.geom import mesh as gm
from tka_planner.geom.collision import Solid
from tka_planner.pipeline import (FlexionModel, FlexionSettings, no_contact_reason,
                                  scan_extent)
from tka_planner.scene import motion


def _frame(bone):
    return AnatomicalFrame(
        bone=bone, side=Side.LEFT, origin=np.zeros(3),
        x_anterior=np.array([0.0, -1.0, 0.0]),
        y_patient_left=np.array([1.0, 0.0, 0.0]),
        z_proximal=np.array([0.0, 0.0, 1.0]),
        method="synthetic", quality=Quality.MEASURED,
    )


def _box(y, z, x=(-10.0, 10.0)):
    """An axis-aligned box from its spans along x, y (posterior) and z (proximal)."""
    low = np.array([x[0], y[0], z[0]])
    high = np.array([x[1], y[1], z[1]])
    return gm.box((high - low).tolist(), gm.translation(((low + high) / 2.0).tolist()))


def _model(femur, tibia, **settings):
    settings = FlexionSettings(min_volume_mm3=1e-3, **settings)
    measurement = SimpleNamespace(femur=femur, tibia=tibia, femoral_frame=_frame("femur"),
                                  tibial_frame=_frame("tibia"))
    extent = scan_extent(measurement, np.zeros(3), np.array([1.0, 0.0, 0.0]), settings)
    boxes = extent["boxes"]
    insert_pose = gm.translation([-300.0, 0.0, -20.0])
    return FlexionModel(
        pivot=motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0])),
        femoral_frame=_frame("femur"), tibial_frame=_frame("tibia"),
        # The boxes are in the frames' own axes (anterior, patient-left, proximal).
        femur=Solid.from_mesh(femur).cropped(_frame("femur").matrix, *boxes["femur"]),
        tibia=Solid.from_mesh(tibia).cropped(_frame("tibia").matrix, *boxes["tibia"]),
        femoral_component=Solid.from_mesh(gm.box(10.0, gm.translation([300.0, 0, 0]))),
        insert=Solid.from_mesh(gm.transformed(gm.box(10.0), insert_pose)),
        insert_pose=insert_pose, insert_bounds=((-5.0, -5.0), (5.0, 5.0)),
        settings=settings, diagnostics={"scan_extent": extent["extent"]},
        femur_box_mm=boxes["femur"], tibia_box_mm=boxes["tibia"],
    )


def _search(model, **kwargs):
    return search_max_flexion(lambda a: model.contacts("S3", a), trajectory="S3",
                              step_deg=5.0, resolution_deg=0.25, **kwargs)


def test_a_contact_past_the_old_170_deg_ceiling_is_found_where_geometry_puts_it():
    """The tibia's anterior face (8 mm in front of the hinge) folds onto the femur's
    posterior-distal corner (y = 8.5, z = 3). The face stays 8 mm from the hinge, so it
    reaches the corner when -8.5 cos(t) - 3 sin(t) = 8, at about 172 deg: past the
    170 deg the search used to stop at, which would have censored it."""
    model = _model(femur=_box(y=(-10.0, 8.5), z=(3.0, 103.0)),
                   tibia=_box(y=(-30.0, -8.0), z=(-105.0, 0.0)))
    radius, alpha = np.hypot(8.5, 3.0), np.degrees(np.arctan2(3.0, 8.5))
    expected = 180.0 - (np.degrees(np.arccos(8.0 / radius)) - alpha)
    assert 171.5 < expected < 172.5
    limit = _search(model)
    assert limit.stop_reason == "contact" and not limit.censored
    assert limit.first_pair == "tibial_bone_vs_femoral_bone"
    assert limit.flexion_limit_deg == limit.max_flexion_deg
    assert expected - 0.25 <= limit.max_flexion_deg < expected + 0.6
    # The lever: the contact is at the corner, about 9 mm from the hinge axis.
    (contact,) = limit.contacts
    assert contact.diagnostics["hinge_radius_mm"] == pytest.approx(radius, abs=1.0)
    # The corner lies on the faces of the box the femur was kept in, but those faces are
    # the bone's own surface; only a shaft end cut by the scan would limit the result.
    assert contact.diagnostics["crop_limited"] is False
    assert limit.stop_deg == FOLD_DEG


def test_bones_present_where_they_would_meet_but_never_meeting_are_kept_apart_by_motion():
    """A tibia 25 mm in front of the hinge folds to 25 mm behind it, 15 mm clear of a
    femur whose posterior face is at y = 10. Both are scanned over the same length."""
    model = _model(femur=_box(y=(-10.0, 10.0), z=(3.0, 103.0)),
                   tibia=_box(y=(-40.0, -25.0), z=(-103.0, 0.0)))
    limit = _search(model)
    assert limit.censored and limit.stop_reason == "no_contact_in_scanned_models"
    assert limit.flexion_limit_deg is None and limit.max_flexion_deg == 180.0
    reason = no_contact_reason(model, "S3", limit.stop_deg)
    assert reason["reason"] == "kept_apart_by_motion"
    assert reason["gaps_mm"]["tibial_bone_vs_femoral_bone"] == pytest.approx(15.0,
                                                                             abs=0.3)
    assert reason["folded_tibia_reach_from_hinge_mm"] == pytest.approx(103.0, abs=0.5)
    assert "No blocking contact in the scanned models" in reason["text"]
    # The faces run parallel: the gap is no smaller at the femur's scanned end than
    # anywhere else, so bone beyond the scan would not close it.
    assert reason["gaps_to_shaft_end_mm"]["femur"] == pytest.approx(15.0, abs=0.3)
    assert reason["gaps_to_rest_of_shaft_mm"]["femur"] == pytest.approx(15.0, abs=0.3)


def test_a_tibia_folding_past_the_end_of_a_short_femoral_scan_is_named_as_such():
    """The femur is scanned only 40 mm above the hinge; the tibia lies 60 to 105 mm
    below it, in line with the femur. Folded, it would meet the femoral shaft 60 to
    105 mm up, which is not in the scan. (The tibia is kept only to the femur's reach
    from the hinge axis, 41 mm, plus the 20 mm margin: nothing farther out can touch.)"""
    model = _model(femur=_box(y=(-10.0, 10.0), z=(3.0, 40.0)),
                   tibia=_box(y=(-10.0, 10.0), z=(-105.0, -60.0)))
    limit = _search(model)
    assert limit.censored
    reason = no_contact_reason(model, "S3", limit.stop_deg)
    assert reason["reason"] == "bone_not_in_scan"
    assert reason["femur_scanned_from_hinge_mm"] == pytest.approx(40.0, abs=0.1)
    femur_reach = np.hypot(40.0, 10.0)
    assert reason["folded_tibia_reach_from_hinge_mm"] == pytest.approx(
        femur_reach + 20.0, abs=0.5)
    assert reason["tibia_scanned_from_hinge_mm"] == pytest.approx(105.0)
    assert "lies beyond the scan" in reason["text"]
    # The folded tibia comes nearest the femur's proximal end (20 mm, from its top at
    # 60 mm to the femur's end at 40 mm), nearer than to the rest of the femur (which
    # ends 5 mm lower).
    assert reason["gaps_to_shaft_end_mm"]["femur"] == pytest.approx(20.0, abs=0.3)
    assert reason["gaps_to_rest_of_shaft_mm"]["femur"] == pytest.approx(24.0, abs=0.3)


def test_the_bones_are_kept_over_their_scan_and_cut_only_beyond_the_other_bones_reach():
    femur = _box(y=(-10.0, 10.0), z=(3.0, 300.0))
    tibia = _box(y=(-10.0, 10.0), z=(-100.0, 0.0))
    measurement = SimpleNamespace(femur=femur, tibia=tibia, femoral_frame=_frame("femur"),
                                  tibial_frame=_frame("tibia"))
    settings = FlexionSettings()
    extent = scan_extent(measurement, np.zeros(3), np.array([1.0, 0.0, 0.0]), settings)
    record = extent["extent"]
    tibia_reach = np.hypot(100.0, 10.0)
    assert record["tibia"]["radial_reach_mm"] == pytest.approx(tibia_reach, abs=0.1)
    assert record["femur"]["scanned_from_hinge_mm"] == pytest.approx(300.0)
    assert record["femur"]["scanned_from_joint_line_mm"] == pytest.approx(297.0)
    # The femur is kept to the tibia's reach plus the margin; the tibia whole.
    assert record["femur"]["kept_from_hinge_mm"] == pytest.approx(
        tibia_reach + settings.reach_margin_mm, abs=0.1)
    assert record["tibia"]["kept_from_hinge_mm"] == pytest.approx(101.0)
    assert record["meeting_reach_mm"] == pytest.approx(tibia_reach, abs=0.1)
    # A box given by hand is used as given.
    given = FlexionSettings(femur_box_mm=((-80, -80, -60), (80, 80, 90)))
    boxed = scan_extent(measurement, np.zeros(3), np.array([1.0, 0.0, 0.0]), given)
    assert boxed["boxes"]["femur"] == ((-80.0, -80.0, -60.0), (80.0, 80.0, 90.0))
    assert not boxed["extent"]["femur"]["box_from_scan"]
