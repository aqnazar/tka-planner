"""The phantom femur: a schematic head, neck and shaft drawn from the entered length.

It is a population prior and must never read as a measurement, so the tests pin the
geometry (the head sits at the entered length on the mechanical axis; the neck meets the
shaft at the neck-shaft angle; the head lies medial on both sides), the stated error, and
that a plan written without an entered length is untouched.
"""

import dataclasses

import numpy as np
import pytest

from tka_planner.core.frames import AnatomicalFrame
from tka_planner.core.phantom_femur import (
    AMA_DEG,
    NECK_SHAFT_ANGLE_DEG,
    build_phantom_femur,
    head_diameter_mm,
    hip_centre_error_budget_mm,
)
from tka_planner.core.provenance import Quality
from tka_planner.core.sides import Side
from tka_planner.geom.phantom import phantom_mesh
from tka_planner.geom import mesh as gm
from tka_planner.pipeline import phantom_femur_case, plan_document, validate_plan_document


def frame(side: Side) -> AnatomicalFrame:
    return AnatomicalFrame(
        bone="femur", side=side, origin=np.array([10.0, -5.0, 30.0]),
        x_anterior=np.array([1.0, 0.0, 0.0]), y_patient_left=np.array([0.0, 1.0, 0.0]),
        z_proximal=np.array([0.0, 0.0, 1.0]), method="test", quality=Quality.MEASURED)


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_the_head_sits_at_the_entered_length_on_the_mechanical_axis(side):
    f = frame(side)
    phantom = build_phantom_femur(f, 410.0)
    assert np.allclose(phantom.head_centre, f.origin + 410.0 * f.z_proximal)
    assert phantom.quality == "assumed" and phantom.to_dict()["schematic"] is True


@pytest.mark.parametrize("side", [Side.LEFT, Side.RIGHT])
def test_the_head_lies_medial_to_the_shaft_whichever_the_side(side):
    f = frame(side)
    phantom = build_phantom_femur(f, 410.0)
    lateral = f.lateral
    # The shaft leans laterally as it rises, so the head is medial of the shaft's axis.
    from_axis = phantom.head_centre - phantom.shaft_start
    away = from_axis - np.dot(from_axis, phantom.anatomical_axis) * phantom.anatomical_axis
    assert np.linalg.norm(away) == pytest.approx(phantom.offset_mm, rel=1e-6)
    assert float(np.dot(away, lateral)) < 0.0        # medial, on either side
    assert phantom.offset_mm == pytest.approx(410.0 * np.sin(np.radians(AMA_DEG)),
                                              rel=1e-9)


def test_the_neck_meets_the_shaft_at_the_neck_shaft_angle():
    phantom = build_phantom_femur(frame(Side.LEFT), 410.0)
    neck = phantom.head_centre - phantom.neck_foot
    toward_knee = -phantom.anatomical_axis
    angle = np.degrees(np.arccos(np.dot(neck, toward_knee) / np.linalg.norm(neck)))
    assert angle == pytest.approx(NECK_SHAFT_ANGLE_DEG, abs=1e-6)
    assert phantom.neck_length_mm == pytest.approx(
        phantom.offset_mm / np.sin(np.radians(180.0 - NECK_SHAFT_ANGLE_DEG)), rel=1e-9)


def test_the_head_diameter_comes_from_the_sex_or_the_length():
    assert head_diameter_mm(410.0, "male")[0] == pytest.approx(49.65)
    assert head_diameter_mm(410.0, "female")[0] == pytest.approx(44.41)
    diameter, sd = head_diameter_mm(410.0, None)
    assert diameter == pytest.approx(0.100 * (410.0 + 40.0))
    assert sd == pytest.approx(3.9)
    with pytest.raises(ValueError):
        head_diameter_mm(410.0, "other")


def test_a_longer_femur_has_a_bigger_head_when_the_sex_is_unknown():
    assert (head_diameter_mm(460.0)[0] > head_diameter_mm(380.0)[0])


def test_the_hip_centre_error_is_stated_and_about_eleven_millimetres_at_410():
    budget = hip_centre_error_budget_mm(410.0)
    assert budget["medial_lateral"] == pytest.approx(9.0, abs=0.1)
    assert budget["anterior_posterior"] == pytest.approx(5.6, abs=0.1)
    assert 11.0 < budget["combined"] < 12.5
    assert "excludes the error of carrying the shaft axis" in budget["basis"]


def test_the_length_is_checked_against_the_width_and_only_flagged_when_far_out():
    ok = build_phantom_femur(frame(Side.LEFT), 410.0, width_mm=80.0).length_check
    assert ok["available"] and not ok["implausible"]
    far = build_phantom_femur(frame(Side.LEFT), 520.0, width_mm=80.0).length_check
    assert far["implausible"] and far["difference_mm"] > far["limit_mm"]
    assert not build_phantom_femur(frame(Side.LEFT), 410.0).length_check["available"]


def test_only_a_femoral_frame_and_a_real_length_are_accepted():
    tibial = dataclasses.replace(frame(Side.LEFT), bone="tibia")
    with pytest.raises(ValueError):
        build_phantom_femur(tibial, 410.0)
    for bad in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            build_phantom_femur(frame(Side.LEFT), bad)


def test_the_document_says_what_it_assumes_and_leaves_out():
    record = build_phantom_femur(frame(Side.LEFT), 410.0).to_dict()
    assert record["method"] == "phantom_femur.v1" and record["quality"] == "assumed"
    text = " ".join(record["assumptions"])
    assert "entered, not measured" in text and "PMID" in text
    assert any("anteversion" in item for item in record["not_modelled"])
    assert "cannot support alignment decisions" in record["note"]


def test_the_mesh_is_three_closed_shells_with_the_head_volume_in_it():
    phantom = build_phantom_femur(frame(Side.LEFT), 410.0)
    mesh = phantom_mesh(phantom)
    assert len(mesh.faces) > 0 and mesh.metadata["phantom_femur"] == "phantom_femur.v1"
    sphere = (4.0 / 3.0) * np.pi * (phantom.head_diameter_mm / 2.0) ** 3
    assert gm.volume(mesh) > sphere * 0.9
    assert mesh.vertices[:, 2].max() > phantom.head_centre[2]


def test_a_plan_has_the_phantom_only_when_a_length_was_entered(session):
    plain = plan_document(session.measurement, session.plan, session.sizing)
    assert "phantom_femur" not in plain

    entered = dataclasses.replace(session.measurement, entered={"femur_length_mm": 410.0})
    document = plan_document(entered, session.plan, session.sizing)
    assert validate_plan_document(document) == []
    assert document["phantom_femur"]["entered_length_mm"] == 410.0
    assert document["phantom_femur"]["length_check"]["available"]
    assert phantom_femur_case(entered) == document["phantom_femur"]


def test_the_scene_has_a_toggleable_phantom_only_when_a_length_was_entered(session):
    session.build()
    assert "PhantomFemur" not in session.scene.nodes

    session.measurement.entered["femur_length_mm"] = 410.0
    try:
        session.build()
        node = session.scene.nodes["PhantomFemur"]
        assert node.visible and node.tags["schematic"]
        assert any("Phantom femur" in note for note in session.scene.notes)

        session.replan(show_phantom=False)
        assert not session.scene.nodes["PhantomFemur"].visible
        session.replan(show_phantom=True)
        assert session.scene.nodes["PhantomFemur"].visible
    finally:
        session.measurement.entered.pop("femur_length_mm", None)
        session.replan(show_phantom=True)
        session.build()
