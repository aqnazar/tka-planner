"""The patient-specific implant, measured on constructed cuts with known dimensions."""

import json

import numpy as np
import pytest

from tka_planner.core.implant_spec import (
    measure_femoral_depth,
    measure_femoral_section,
    measure_tibial_section,
)
from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm

IDENTITY = np.eye(4)


def combined(*meshes):
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces))


def tibial_block_with_notch(ml=80.0, ap=50.0, notch_width=12.0, notch_depth=8.0,
                            notch_centre=-10.0):
    """A tibia whose section at z = 0 is a rectangle with a notch cut into its back.

    Tray axes: +X patient-left, +Y posterior. Built from boxes that tile the section:
    everything in front of the notch's depth, then the back strip either side of it.
    """
    front_depth = ap - notch_depth
    front = gm.box((ml, front_depth, 40.0),
                   gm.translation((0.0, -ap / 2 + front_depth / 2, 0.0)))
    left_width = (notch_centre - notch_width / 2) + ml / 2
    right_width = ml / 2 - (notch_centre + notch_width / 2)
    back_y = ap / 2 - notch_depth / 2
    left = gm.box((left_width, notch_depth, 40.0),
                  gm.translation((-ml / 2 + left_width / 2, back_y, 0.0)))
    right = gm.box((right_width, notch_depth, 40.0),
                   gm.translation((ml / 2 - right_width / 2, back_y, 0.0)))
    return combined(front, left, right)


def test_the_tray_dimensions_are_the_section_s():
    dims = measure_tibial_section(tibial_block_with_notch(), IDENTITY,
                                  medial_sign=-1.0).dimensions

    assert dims["ml"] == pytest.approx(80.0, abs=0.6)
    assert dims["ap"] == pytest.approx(50.0, abs=0.6)


def test_an_off_centre_pcl_notch_is_found():
    dims = measure_tibial_section(tibial_block_with_notch(), IDENTITY,
                                  medial_sign=-1.0).dimensions

    assert dims["pcl_notch_depth"] == pytest.approx(8.0, abs=0.6)
    assert dims["pcl_notch_width"] == pytest.approx(12.0, abs=1.0)


def test_compartment_depths_follow_the_side():
    """A shorter medial compartment: on a left knee medial is patient-right, -X."""
    long_lateral = combined(
        gm.box((40.0, 50.0, 40.0), gm.translation((20.0, 0.0, 0.0))),
        gm.box((40.0, 40.0, 40.0), gm.translation((-20.0, -5.0, 0.0))),
    )
    left = measure_tibial_section(long_lateral, IDENTITY, medial_sign=-1.0).dimensions
    right = measure_tibial_section(long_lateral, IDENTITY, medial_sign=1.0).dimensions

    assert left["medial_ap"] == pytest.approx(40.0, abs=0.6)
    assert left["lateral_ap"] == pytest.approx(50.0, abs=0.6)
    assert (right["medial_ap"], right["lateral_ap"]) == (left["lateral_ap"],
                                                          left["medial_ap"])


def two_condyles(width=28.0, notch=18.0, ap=50.0, bridge=15.0):
    """A distal femur section: two condyles joined by an anterior bridge.

    Femoral axes: +X anterior, +Y patient-left.
    """
    half = notch / 2 + width / 2
    condyles = [gm.box((ap, width, 40.0), gm.translation((0.0, sign * half, 0.0)))
                for sign in (-1.0, 1.0)]
    anterior = gm.box((bridge, notch, 40.0),
                      gm.translation((ap / 2 - bridge / 2, 0.0, 0.0)))
    return combined(*condyles, anterior)


def test_the_condyles_and_the_notch_are_measured():
    spec = measure_femoral_section(two_condyles(), IDENTITY, medial_sign=-1.0)
    dims = spec.dimensions

    assert dims["ml"] == pytest.approx(2 * 28.0 + 18.0, abs=0.6)
    assert dims["notch_width"] == pytest.approx(18.0, abs=0.6)
    assert dims["medial_condyle_width"] == pytest.approx(28.0, abs=0.6)
    assert dims["lateral_condyle_width"] == pytest.approx(28.0, abs=0.6)
    assert spec.diagnostics["condyles_found"]


def test_the_outline_follows_the_notch_rather_than_failing_in_it():
    """The centroid of a two-condyle section lies in the notch, off the bone."""
    outline = np.array(measure_femoral_section(two_condyles(), IDENTITY,
                                               medial_sign=-1.0).outline_mm)

    assert len(outline) == 96
    assert np.ptp(outline[:, 1]) == pytest.approx(74.0, abs=1.0)
    # Some outline points lie inside the notch's span, at its floor.
    in_notch = np.abs(outline[:, 1]) < 8.0
    assert in_notch.any()


def test_the_femoral_depth_spans_the_component_height():
    femur = gm.box((70.0, 60.0, 80.0), gm.translation((5.0, 0.0, 30.0)))

    depth = measure_femoral_depth(femur, IDENTITY, height_mm=40.0)

    assert depth["ap_overall"] == pytest.approx(70.0)
    assert depth["anterior_mm"] == pytest.approx(40.0)


def test_a_planned_case_gives_a_complete_specification(session):
    from tka_planner.pipeline import implant_spec_case

    spec = implant_spec_case(session.measurement, session.plan, session.sizing,
                             session.library)

    assert spec["schema"] == "tka-planner/implant-spec"
    for key in ("ml", "ap_overall", "distal_thickness"):
        assert spec["femoral_component"]["dimensions_mm"][key] > 0
    for key in ("ml", "ap", "medial_ap", "lateral_ap"):
        assert spec["tibial_component"]["dimensions_mm"][key] > 0
    assert spec["insert"]["thickness_mm"] == pytest.approx(
        session.plan.diagnostics["insert_thickness_mm"], abs=1e-3)


def test_patient_specific_parts_are_scaled_per_axis(session):
    transforms = session.component_scales

    assert set(transforms) == {"scales", "shifts"}
    for group in ("femoral", "tibial"):
        assert len(transforms["scales"][group]) == 3
    session.replan(implant_mode="catalogue")
    assert session.component_scales is None


def test_the_export_writes_the_specification(session, tmp_path):
    from tka_planner.report.bundle import export_session

    written = export_session(session, tmp_path, write_meshes=False)
    spec = json.loads(open(written["implant_spec"]).read())

    assert spec["case_id"] == session.case_id
    assert len(spec["tibial_component"]["outline_mm"]) == 96


@pytest.mark.parametrize("control, component", [
    ("femoral_shift_ap_mm", "femoral_component"),
    ("femoral_shift_ml_mm", "femoral_component"),
    ("tibial_shift_ap_mm", "tibial_component"),
    ("tibial_shift_ml_mm", "tibial_component"),
])
def test_a_slide_moves_the_patient_specific_part_by_exactly_the_slide(
        session, control, component):
    """The centring onto the cut must not swallow the surgeon's slide."""
    before = session.scene.nodes[component].rest[:3, 3].copy()
    session.replan(**{control: 2.0})
    after = session.scene.nodes[component].rest[:3, 3]

    assert np.linalg.norm(after - before) == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("control", ["femoral_shift_ap_mm", "femoral_shift_ml_mm"])
def test_a_femoral_slide_carries_the_patient_specific_tray(session, control):
    """The tray is in register with the femoral component, so it moves with it."""
    before = session.scene.nodes["tibial_component"].rest[:3, 3].copy()
    session.replan(**{control: 2.0})
    after = session.scene.nodes["tibial_component"].rest[:3, 3]

    assert np.linalg.norm(after - before) == pytest.approx(2.0, abs=0.1)


def test_the_patient_specific_parts_are_shown_in_register(session):
    """The displayed CAD origins are coaxial along the tray's normal."""
    femoral = session.scene.nodes["femoral_component"].rest
    tray = session.scene.nodes["tibial_component"].rest
    normal = tray[:3, 2] / np.linalg.norm(tray[:3, 2])
    offset = femoral[:3, 3] - tray[:3, 3]
    in_plane = offset - np.dot(offset, normal) * normal

    assert np.linalg.norm(in_plane) == pytest.approx(0.0, abs=0.05)


# A made-up design, deliberately unlike any real one.
BOX_DESIGN = {"anterior_flare_deg": 7.0, "chamfer_deg": 40.0,
              "posterior_condyle_thickness_mm": 8.0, "flange_thickness_mm": 3.5,
              "distal_face_fraction": 0.45, "posterior_chamfer_fraction": 0.25}


def test_the_cut_box_is_placed_with_the_library_s_design(tmp_path):
    from types import SimpleNamespace

    from tests.test_femoral_box import femur
    from tka_planner.pipeline import femoral_cut_box_case

    (tmp_path / "femoral_box.json").write_text(json.dumps(BOX_DESIGN), encoding="utf-8")
    measurement = SimpleNamespace(femur=femur(), side="right")
    plan = SimpleNamespace(components={"femoral_component": IDENTITY})

    box = femoral_cut_box_case(measurement, plan, tmp_path)

    assert box["available"]
    assert box["resections_mm"]["posterior_medial"] == pytest.approx(8.0, abs=0.3)
    assert box["flange_height_mm"] == pytest.approx(40.0, abs=1.5)


def test_a_library_without_a_box_design_says_so(tmp_path):
    from types import SimpleNamespace

    from tests.test_femoral_box import femur
    from tka_planner.pipeline import femoral_cut_box_case

    measurement = SimpleNamespace(femur=femur(), side="right")
    plan = SimpleNamespace(components={"femoral_component": IDENTITY})

    box = femoral_cut_box_case(measurement, plan, tmp_path)

    assert box["available"] is False
    assert "femoral_box.json" in box["reason"]
    assert femoral_cut_box_case(measurement, plan, None) is None


def test_a_bone_the_box_cannot_be_placed_on_gives_the_reason(tmp_path):
    from types import SimpleNamespace

    from tests.test_femoral_box import femur
    from tka_planner.pipeline import femoral_cut_box_case

    (tmp_path / "femoral_box.json").write_text(json.dumps(BOX_DESIGN), encoding="utf-8")
    measurement = SimpleNamespace(femur=femur(length=70.0), side="right")
    plan = SimpleNamespace(components={"femoral_component": IDENTITY})

    box = femoral_cut_box_case(measurement, plan, tmp_path)

    assert not box["available"]
    assert "anterior cortex" in box["reason"]


def test_the_specification_carries_the_resection_table(session):
    from tka_planner.pipeline import implant_spec_case

    spec = implant_spec_case(session.measurement, session.plan, session.sizing,
                             session.library)
    table = spec["resections"]["table_mm"]
    femoral = session.plan.resections["femoral_distal"]

    assert set(table) == {"femoral_distal_medial", "femoral_distal_lateral",
                          "femoral_posterior_medial", "femoral_posterior_lateral",
                          "femoral_anterior", "tibial_medial", "tibial_lateral"}
    assert table["femoral_distal_medial"] == pytest.approx(femoral.medial_depth_mm,
                                                           abs=0.01)
    assert "available" in spec["femoral_component"]["cut_box"]


@pytest.mark.parametrize("content", [
    '{"anterior_flare_deg": 7.0}',                                  # keys missing
    '{"anterior_flare_deg": 7.0,',                                  # not JSON
    json.dumps({**BOX_DESIGN, "distal_face_fraction": 0.9}),        # impossible box
    json.dumps({**BOX_DESIGN, "chamfer_deg": "forty"}),             # not a number
])
def test_a_malformed_box_design_is_reported_rather_than_stopping_the_plan(
        tmp_path, content):
    from types import SimpleNamespace

    from tests.test_femoral_box import femur
    from tka_planner.pipeline import femoral_cut_box_case

    (tmp_path / "femoral_box.json").write_text(content, encoding="utf-8")
    measurement = SimpleNamespace(femur=femur(), side="right")
    plan = SimpleNamespace(components={"femoral_component": IDENTITY})

    box = femoral_cut_box_case(measurement, plan, tmp_path)

    assert box["available"] is False
    assert "femoral_box.json" in box["reason"]


def test_the_table_takes_each_compartment_from_the_placed_box(session, monkeypatch):
    import tka_planner.pipeline as pipeline

    placed = {"available": True,
              "resections_mm": {"posterior_medial": 6.4, "posterior_lateral": 2.9,
                                "anterior": 9.1}}
    monkeypatch.setattr(pipeline, "femoral_cut_box_case", lambda *a, **k: placed)

    table = pipeline.implant_spec_case(session.measurement, session.plan,
                                       session.sizing, session.library)[
        "resections"]["table_mm"]

    assert table["femoral_posterior_medial"] == 6.4
    assert table["femoral_posterior_lateral"] == 2.9
    assert table["femoral_anterior"] == 9.1


def test_shaping_the_parts_for_the_viewer_does_not_place_the_box(session, monkeypatch):
    """The viewer's scaling reads only the sections; placing the box there as well
    doubled the cost of every change of a control for nothing."""
    import tka_planner.pipeline as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "femoral_cut_box_case",
                        lambda *a, **k: calls.append(1) or None)

    pipeline.component_scales(session.measurement, session.plan, session.sizing,
                              session.library)

    assert calls == []
