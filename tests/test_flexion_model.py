"""The rule that tells a femoral component bearing on its insert from one striking it.

Toy solids in LPS (+X patient-left, +Y posterior, +Z proximal): an insert 40 mm wide,
30 mm deep and 10 mm high, its top at z = -25, and a 10 mm cube for the femoral
component, placed where each case needs it. The bones are put far away so only the
component and the insert can touch.
"""

import numpy as np
import pytest

from tka_planner.core.frames import AnatomicalFrame
from tka_planner.core.provenance import Quality
from tka_planner.core.sides import Side
from tka_planner.geom import mesh as gm
from tka_planner.geom.collision import Solid, overlap
from tka_planner.pipeline import FlexionModel, FlexionSettings
from tka_planner.scene import motion


def _frame(bone):
    return AnatomicalFrame(
        bone=bone, side=Side.LEFT, origin=np.zeros(3),
        x_anterior=np.array([0.0, -1.0, 0.0]),
        y_patient_left=np.array([1.0, 0.0, 0.0]),
        z_proximal=np.array([0.0, 0.0, 1.0]),
        method="synthetic", quality=Quality.MEASURED,
    )


def _model(component_centre, baseline=0.0, tolerance=1.0):
    insert_pose = gm.translation([0.0, 0.0, -30.0])
    insert_mesh = gm.box([40.0, 30.0, 10.0])
    return FlexionModel(
        pivot=motion.pivot_frame(np.zeros(3), np.array([1.0, 0.0, 0.0])),
        femoral_frame=_frame("femur"), tibial_frame=_frame("tibia"),
        femur=Solid.from_mesh(gm.box(5.0, gm.translation([0, 0, 300.0]))),
        tibia=Solid.from_mesh(gm.box(5.0, gm.translation([0, 0, -300.0]))),
        femoral_component=Solid.from_mesh(
            gm.box(10.0, gm.translation(component_centre))),
        insert=Solid.from_mesh(gm.transformed(insert_mesh, insert_pose)),
        insert_pose=insert_pose,
        insert_bounds=((-20.0, -15.0), (20.0, 15.0)),
        settings=FlexionSettings(articular_tolerance_mm=tolerance),
        diagnostics={}, baseline_depth_mm=baseline,
    )


def test_a_shallow_contact_in_the_middle_of_the_dish_is_articulation():
    (contact,) = _model([0.0, 0.0, -20.5]).contacts("S3", 0.0)
    assert contact.pair == "femoral_component_vs_insert"
    assert contact.kind == "articulation" and contact.region == "articular"
    assert contact.depth_mm == pytest.approx(0.5, abs=0.05)


def test_contact_at_the_posterior_lip_is_impingement():
    (contact,) = _model([0.0, 13.0, -20.5]).contacts("S3", 0.0)
    assert contact.region == "posterior_rim"
    assert contact.kind == "impingement"


def test_contact_at_the_anterior_lip_is_impingement():
    (contact,) = _model([0.0, -13.0, -20.5]).contacts("S3", 0.0)
    assert contact.region == "anterior_rim" and contact.blocks


def test_sinking_deeper_than_the_seating_plus_the_tolerance_is_impingement():
    model = _model([0.0, 0.0, -22.0], baseline=0.5)
    (contact,) = model.contacts("S3", 0.0)
    assert contact.depth_mm == pytest.approx(2.0, abs=0.05)
    assert contact.region == "articular" and contact.kind == "impingement"


def test_the_tolerance_is_a_setting():
    model = _model([0.0, 0.0, -22.0], baseline=0.5, tolerance=None)
    (contact,) = model.contacts("S3", 0.0)
    assert contact.kind == "articulation"


def test_the_overlap_depth_is_its_thickness_not_its_extent():
    """A thin slab tilted 30 degrees: its extent along z is far more than its
    thickness, which is what penetration means."""
    tilt = np.eye(4)
    tilt[:3, :3] = motion.rotation_about([1.0, 0.0, 0.0], 30.0)
    slab = Solid.from_mesh(gm.box([20.0, 20.0, 1.0], tilt))
    big = Solid.from_mesh(gm.box(100.0))
    shared = overlap(slab, big)
    assert np.ptp(shared.vertices[:, 2]) > 9.0
    # Vertical thickness of a 1 mm slab tilted 30 deg is 1/cos(30) = 1.15 mm, plus at
    # most cell * tan(30) = 0.29 mm from the column width.
    assert 1.1 < shared.thickness_mm([0.0, 0.0, 1.0]) < 1.5
