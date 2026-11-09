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
    FEMUR_LENGTH_TYPES,
    HEAD_TOP_TO_CONDYLE_OFFSET_MM,
    head_top_to_condyle_offset_mm,
    NECK_SHAFT_ANGLE_DEG,
    build_phantom_femur,
    entered_femur_record,
    femur_length_record,
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
    assert diameter == pytest.approx(0.100 * (410.0 + 32.0))
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


# ----------------------------------------------------------------------
# What the entered length measures, and the sex (author ruling 2026-10-26)
# ----------------------------------------------------------------------


def test_a_head_top_to_condyle_length_loses_the_calibrated_offset():
    # Calibrated on 157 whole femurs (HFValid, VSD), 2026-10-28: women 31.3, men 34.9,
    # pooled 32.0 mm; it replaced an assumed 40 mm.
    assert FEMUR_LENGTH_TYPES == ("hip_to_knee", "head_top_to_condyle")
    assert HEAD_TOP_TO_CONDYLE_OFFSET_MM == 32.0
    assert head_top_to_condyle_offset_mm("female") == (31.3, 2.5)
    assert head_top_to_condyle_offset_mm("male") == (34.9, 1.7)
    assert head_top_to_condyle_offset_mm(None) == (32.0, 2.8)
    record = femur_length_record(442.0, "head_top_to_condyle")
    assert record == {
        "type": "head_top_to_condyle", "entered_mm": 442.0, "hip_to_knee_mm": 410.0,
        "conversion": "calibrated", "offset_mm": 32.0, "offset_sd_mm": 2.8,
        "offset_for": "pooled", "basis": record["basis"],
    }
    assert "157 whole femurs" in record["basis"] and "pooled" in record["basis"]
    female = femur_length_record(450.0, "head_top_to_condyle", "female")
    assert female["hip_to_knee_mm"] == pytest.approx(418.7)
    assert female["offset_for"] == "female"


def test_the_offset_is_refused_for_an_unknown_sex():
    with pytest.raises(ValueError, match="sex"):
        head_top_to_condyle_offset_mm("other")


def test_a_hip_to_knee_length_is_used_as_entered_and_is_the_default():
    record = femur_length_record(410.0, "hip_to_knee")
    assert record == {"type": "hip_to_knee", "entered_mm": 410.0,
                      "hip_to_knee_mm": 410.0, "conversion": "none"}
    assert femur_length_record(410.0) == record
    assert femur_length_record(410.0, "") == record


def test_the_phantom_is_placed_at_the_converted_length():
    f = frame(Side.LEFT)
    converted = build_phantom_femur(f, 442.0, width_mm=80.0,
                                    length_type="head_top_to_condyle")
    direct = build_phantom_femur(f, 410.0, width_mm=80.0)
    assert np.allclose(converted.head_centre, f.origin + 410.0 * f.z_proximal)
    assert np.allclose(converted.head_centre, direct.head_centre)
    assert converted.head_diameter_mm == pytest.approx(direct.head_diameter_mm)
    assert converted.length_check == direct.length_check
    record = converted.to_dict()
    assert record["entered_length_mm"] == 442.0
    assert record["length"]["hip_to_knee_mm"] == 410.0
    assert record["length"]["conversion"] == "calibrated"
    assert "head-top-to-condyle" in record["assumptions"][0]
    assert "entered, not measured" in record["assumptions"][0]


def test_a_converted_length_must_still_be_a_length():
    with pytest.raises(ValueError, match="hip"):
        build_phantom_femur(frame(Side.LEFT), 30.0, length_type="head_top_to_condyle")


def test_the_sex_sets_the_head_and_its_absence_keeps_the_length_rule():
    f = frame(Side.LEFT)
    male = build_phantom_femur(f, 410.0, sex="male")
    female = build_phantom_femur(f, 410.0, sex="female")
    unset = build_phantom_femur(f, 410.0)
    assert male.head_diameter_mm == pytest.approx(49.65)      # 49.7 to one decimal
    assert female.head_diameter_mm == pytest.approx(44.41)    # 44.4
    assert unset.head_diameter_mm == pytest.approx(0.100 * (410.0 + 32.0))
    # A published mean is reported at its source's precision, not rounded to 49.6.
    assert male.to_dict()["head"]["diameter_mm"] == 49.65
    assert female.to_dict()["head"]["diameter_mm"] == 44.41
    assert unset.to_dict()["head"]["diameter_mm"] == 44.2
    assert male.to_dict()["sex"] == "male" and unset.to_dict()["sex"] is None


def test_the_unset_path_records_the_default_length_type_and_nothing_else_moves():
    f = frame(Side.LEFT)
    default = build_phantom_femur(f, 410.0, width_mm=80.0).to_dict()
    explicit = build_phantom_femur(f, 410.0, width_mm=80.0,
                                   length_type="hip_to_knee").to_dict()
    assert default == explicit
    assert default["entered_length_mm"] == 410.0
    assert default["length"] == {"type": "hip_to_knee", "entered_mm": 410.0,
                                 "hip_to_knee_mm": 410.0, "conversion": "none"}
    assert default["assumptions"][0] == (
        "the head centre lies 410 mm from the knee centre along the frame's mechanical "
        "axis (entered, not measured)")


def test_the_entered_record_is_flat_and_says_what_was_entered():
    assert entered_femur_record(None) == {}
    assert entered_femur_record(410.0) == {
        "femur_length_mm": 410.0, "femur_length_type": "hip_to_knee",
        "femur_length_entered_mm": 410.0, "femur_length_conversion": "none"}
    record = entered_femur_record(450.0, "head_top_to_condyle", "female")
    assert record == {
        "femur_length_mm": record["femur_length_mm"],
        "femur_length_type": "head_top_to_condyle",
        "femur_length_entered_mm": 450.0, "femur_length_conversion": "calibrated",
        "femur_length_offset_mm": 31.3, "sex": "female"}
    assert record["femur_length_mm"] == pytest.approx(418.7)


@pytest.mark.parametrize("length, length_type, sex, match", [
    (450.0, "scanogram", None, "length type"),
    (450.0, "HIP_TO_KNEE", None, "length type"),
    (450.0, None, "M", "sex"),
    (450.0, None, "other", "sex"),
    (None, "head_top_to_condyle", None, "without a femur length"),
    (None, None, "male", "without a femur length"),
])
def test_unknown_or_orphaned_femur_inputs_are_refused(length, length_type, sex, match):
    with pytest.raises(ValueError, match=match):
        entered_femur_record(length, length_type, sex)


def test_measuring_refuses_an_unknown_sex_before_reading_the_bones():
    from tka_planner.pipeline import measure_meshes

    with pytest.raises(ValueError, match="sex"):
        measure_meshes(None, None, "left", case_id="X", femur_length_mm=450.0,
                       sex="unknown")


@pytest.mark.parametrize("length, length_type, code", [
    (680.0, "head_top_to_condyle", "FEMUR_LENGTH_ENTERED"),       # 648 hip to knee
    (680.0, "hip_to_knee", "FEMUR_LENGTH_IMPLAUSIBLE"),
    (700.0, "head_top_to_condyle", "FEMUR_LENGTH_IMPLAUSIBLE"),   # 668 hip to knee
    (330.0, "head_top_to_condyle", "FEMUR_LENGTH_IMPLAUSIBLE"),   # 298 hip to knee
])
def test_the_plausible_range_is_checked_on_the_hip_to_knee_length(length, length_type,
                                                                 code):
    from tka_planner.core.qc import check_entered_femur_length

    report = check_entered_femur_length(length, length_type)
    assert [f.code for f in report.findings] == [code]
    if length_type == "head_top_to_condyle":
        finding = report.findings[0]
        assert "head top" in finding.message and "32 mm" in finding.message
        assert finding.context["femur_length_type"] == "head_top_to_condyle"


def test_the_default_length_finding_is_unchanged():
    from tka_planner.core.qc import check_entered_femur_length

    finding = check_entered_femur_length(455.0).findings[0]
    assert finding.message == ("Femur length 455 mm was entered, not measured: the scan "
                               "does not reach the hip.")
    assert finding.context == {"femur_length_mm": 455.0}


def test_a_plan_records_the_length_type_and_the_sex(session):
    entered = dataclasses.replace(
        session.measurement,
        entered=entered_femur_record(450.0, "head_top_to_condyle", "male"))
    document = plan_document(entered, session.plan, session.sizing)
    assert validate_plan_document(document) == []
    assert document["inputs"]["entered"] == {
        "femur_length_mm": pytest.approx(415.1), "femur_length_type": "head_top_to_condyle",
        "femur_length_entered_mm": 450.0, "femur_length_conversion": "calibrated",
        "femur_length_offset_mm": 34.9, "sex": "male"}
    phantom = document["phantom_femur"]
    assert phantom["entered_length_mm"] == 450.0
    assert phantom["length"]["hip_to_knee_mm"] == pytest.approx(415.1)
    assert phantom["sex"] == "male" and phantom["head"]["diameter_mm"] == 49.65


def test_the_scene_phantom_is_the_plan_phantom(session):
    """The app's drawing and the plan's record come from the same entered values."""
    from tka_planner.pipeline import phantom_femur_for

    saved = dict(session.measurement.entered)
    session.measurement.entered.update(
        entered_femur_record(450.0, "head_top_to_condyle", "female"))
    try:
        drawn = session._phantom()
        assert drawn.head_diameter_mm == pytest.approx(44.41)
        assert drawn.length_mm == pytest.approx(418.7)
        assert drawn.to_dict() == phantom_femur_for(session.measurement).to_dict()
    finally:
        session.measurement.entered.clear()
        session.measurement.entered.update(saved)


def test_the_command_line_passes_the_length_type_and_the_sex(monkeypatch, tmp_path):
    from tka_planner import cli

    seen = {}

    def stop(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("measured")

    monkeypatch.setattr(cli, "measure_case", stop)
    base = ["measure", "--femur", "f.stl", "--tibia", "t.stl", "--side", "left",
            "--out", str(tmp_path)]
    with pytest.raises(RuntimeError):
        cli.main(base + ["--femur-length", "450", "--femur-length-type",
                         "head_top_to_condyle", "--sex", "female"])
    assert seen["femur_length_mm"] == 450.0
    assert seen["femur_length_type"] == "head_top_to_condyle"
    assert seen["sex"] == "female"

    seen.clear()
    with pytest.raises(RuntimeError):
        cli.main(base + ["--femur-length", "410"])
    assert seen["femur_length_type"] is None and seen["sex"] is None


@pytest.mark.parametrize("extra", [["--femur-length", "450", "--sex", "M"],
                                   ["--femur-length", "450", "--femur-length-type",
                                    "scanogram"]])
def test_the_command_line_refuses_an_unknown_value(extra, tmp_path, capsys):
    from tka_planner import cli

    with pytest.raises(SystemExit) as stopped:
        cli.main(["measure", "--femur", "f.stl", "--tibia", "t.stl", "--side", "left",
                  "--out", str(tmp_path)] + extra)
    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--sex", "male"],
                                   ["--femur-length-type", "head_top_to_condyle"]])
def test_the_command_line_refuses_a_type_or_sex_without_a_length(extra, tmp_path, capsys):
    from tka_planner import cli

    assert cli.main(["measure", "--femur", "f.stl", "--tibia", "t.stl", "--side", "left",
                     "--out", str(tmp_path)] + extra) == 2
    assert "without a femur length" in capsys.readouterr().err
