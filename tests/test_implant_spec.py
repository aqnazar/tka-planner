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
