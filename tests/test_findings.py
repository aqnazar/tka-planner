"""Findings: one list, each with its place, and the report that carries them."""

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.core.qc import Finding, Severity, assess_mesh_quality
from tka_planner.geom import mesh as gm
from tka_planner.pipeline import findings_case, render_case_report


def test_an_open_mesh_says_where_it_is_open():
    cube = gm.box(10.0)
    open_cube = Mesh(cube.vertices, cube.faces[2:])  # the bottom face is gone

    _, report = assess_mesh_quality(open_cube)
    finding = next(f for f in report.findings if f.code == "MESH_NOT_CLOSED")

    assert finding.context["location_mm"][2] == -5.0  # on the missing bottom face


def test_a_landmark_finding_points_at_its_landmark(session):
    m = session.measurement
    landmark = next(lm for lm in m.landmarks if lm.is_usable and lm.id.startswith("tibia"))
    m.qc.findings.append(Finding("LANDMARKS_OFF_MESH", Severity.WARNING, "test",
                                 {"offenders": [landmark.id]}))

    findings = findings_case(m, session.plan, session.sizing)
    finding = next(f for f in findings if f["code"] == "LANDMARKS_OFF_MESH")

    assert finding["location"]["point_mm"] == [
        round(float(v), 2) for v in landmark.position_mm]
    assert finding["location"]["set"] == "tibial"
    assert finding["location"]["label"] == landmark.id


def test_the_plan_s_own_warnings_are_findings(session):
    session.replan(femoral_varus_delta_deg=4.0)

    findings = findings_case(session.measurement, session.plan, session.sizing)

    assert any(f["code"] == "PLAN_WARNING" for f in findings)
    assert all(f["source"] for f in findings)


def test_a_mesh_finding_names_its_bone():
    """A mesh check does not know which bone it was given; the pipeline says."""
    from tka_planner.pipeline import _for_bone

    cube = gm.box(10.0)
    _, report = assess_mesh_quality(Mesh(cube.vertices, cube.faces[2:]))
    tagged = _for_bone(report, "tibia")

    assert all(f.context["bone"] == "tibia" for f in tagged.findings)
    located = next(f for f in tagged.findings if f.code == "MESH_NOT_CLOSED")
    assert "location_mm" in located.context


def test_the_report_carries_what_a_comparison_needs(session):
    from tka_planner.pipeline import fit_case_with_grids, implant_spec_case

    fit, grids = fit_case_with_grids(session.measurement, session.plan, session.sizing,
                                     session.library)
    page = render_case_report(
        session.measurement, session.plan, session.sizing, fit=fit, fit_grids=grids,
        implant_spec=implant_spec_case(session.measurement, session.plan, session.sizing,
                                       session.library))

    for section in ("Resection table", "Distal femur", "Tibia", "Component register",
                    "Implant", "Fit on the cuts", "anteromedial", "Fit map",
                    "data:image/png;base64,", "Quality control and findings",
                    "provenance", "measured"):
        assert section.lower() in page.lower(), section
    # Still one self-contained file.
    assert "<script" not in page and "http" not in page.replace("http-equiv", "")


def test_a_report_without_a_library_still_has_the_table_and_findings(session):
    page = render_case_report(session.measurement, session.plan, session.sizing)

    assert "Resection table" in page
    assert "Quality control and findings" in page
    assert "Fit on the cuts" not in page


def test_the_figure_is_a_valid_png():
    import base64
    import zlib

    from tka_planner.report.figures import png_bytes

    rgba = np.zeros((3, 4, 4))
    rgba[1, 2] = (255, 0, 0, 255)
    data = png_bytes(rgba)

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    raw = zlib.decompress(data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8])
    assert len(raw) == 3 * (1 + 4 * 4)
    assert base64.b64encode(data)
