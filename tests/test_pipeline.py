"""The one pipeline: the command line and the application plan a case identically.

The synthetic tests run everywhere. The real-case test runs only where a case's STL files
are present, which is never in CI -- patient data is not in the repository -- but always
on a planning machine, where it is the check that matters: the same case through both
front ends gives the same plan.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from tka_planner.pipeline import (
    DOCUMENT_KEYS,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    REQUIRED_KEYS,
    plan_document,
    read_plan,
    validate_plan_document,
    write_plan,
)
from tka_planner.report.bundle import export_session

REAL_CASE = Path(__file__).parent.parent / "data" / "cases" / "P009"


def cli_style(session):
    """What the command line writes for this measurement and plan."""
    return plan_document(session.measurement, session.plan, session.sizing)


def test_a_plan_document_is_versioned_and_valid(session):
    document = cli_style(session)

    assert document["schema"] == PLAN_SCHEMA
    assert document["schema_version"] == PLAN_SCHEMA_VERSION
    assert validate_plan_document(document) == []


def test_the_application_export_has_the_command_line_shape(session, tmp_path):
    written = export_session(session, tmp_path, write_meshes=False)
    exported = json.loads(Path(written["plan"]).read_text())

    assert set(exported) == set(cli_style(session)) == set(DOCUMENT_KEYS)
    assert set(REQUIRED_KEYS) <= set(DOCUMENT_KEYS)
    assert set(exported["sizing"]) == set(cli_style(session)["sizing"])
    assert exported["controls"]["philosophy"] == "mechanical"


def test_the_application_export_carries_the_quality_control(session, tmp_path):
    written = export_session(session, tmp_path, write_meshes=False)
    exported = json.loads(Path(written["plan"]).read_text())

    assert exported["quality_control"] == json.loads(
        json.dumps(session.measurement.qc.to_dict()))
    assert exported["sizing"]["discrete"]["method"].startswith("sizing.discrete")


def test_inputs_are_named_by_file_not_by_path(session):
    inputs = cli_style(session)["inputs"]

    for key in ("femur", "tibia", "size_chart"):
        assert "/" not in inputs[key] and "\\" not in inputs[key]
        assert inputs[f"{key}_sha256"]


@pytest.mark.parametrize("key", ["surgical_plan", "schema_version", "intended_use"])
def test_a_document_missing_a_key_is_invalid(session, key):
    document = cli_style(session)
    del document[key]

    assert any(key in problem for problem in validate_plan_document(document))


def test_another_schema_version_is_refused(session):
    document = cli_style(session)
    document["schema_version"] = PLAN_SCHEMA_VERSION + 1

    assert validate_plan_document(document)


def test_an_invalid_plan_is_never_written(session, tmp_path):
    document = cli_style(session)
    del document["metrics"]

    with pytest.raises(ValueError, match="invalid plan"):
        write_plan(document, tmp_path / "plan.json")
    assert not (tmp_path / "plan.json").exists()


def test_a_written_plan_reads_back(session, tmp_path):
    path = write_plan(cli_style(session), tmp_path / "plan.json")
    assert read_plan(path)["case_id"] == session.case_id


def test_a_written_plan_has_the_same_bytes_on_every_platform(session, tmp_path):
    """LF line endings, never the platform's own: Windows text mode would write CRLF."""
    from tka_planner.pipeline import _jsonable

    document = cli_style(session)
    raw = write_plan(document, tmp_path / "plan.json").read_bytes()

    assert b"\r" not in raw
    assert raw == (json.dumps(document, indent=2, default=_jsonable) + "\n").encode()


def test_a_written_implant_spec_has_lf_line_endings(session, tmp_path):
    from tka_planner.pipeline import write_implant_spec

    spec = cli_style(session)["implant_spec"]
    assert b"\r" not in write_implant_spec(spec, tmp_path / "spec.json").read_bytes()


@pytest.mark.skipif(not (REAL_CASE / "FD1Left.stl").is_file(),
                    reason="patient data is not in the repository")
def test_the_command_line_and_the_application_plan_a_real_case_identically(tmp_path):
    from tka_planner.cli import main
    from tka_planner.session import PlanningSession

    assert main(["measure", "--femur", str(REAL_CASE / "FD1Left.stl"),
                 "--tibia", str(REAL_CASE / "TD1Left.stl"), "--side", "left",
                 "--case-id", "P009", "--out", str(tmp_path / "cli")]) == 0
    from_cli = read_plan(tmp_path / "cli" / "plan.json")

    session = PlanningSession.open(REAL_CASE, side="left")
    session.build()
    written = export_session(session, tmp_path / "app", write_meshes=False)
    from_app = read_plan(written["plan"])

    for key in ("inputs", "frames", "metrics", "surgical_plan", "sizing",
                "quality_control", "landmarks", "scan_coverage"):
        assert from_app[key] == from_cli[key], key


# ----------------------------------------------------------------------
# Flexion (WP-D) and the posterior condylar offset, through the pipeline
# ----------------------------------------------------------------------


def test_a_plan_records_its_trajectory_and_every_scenario(session):
    document = plan_document(session.measurement, session.plan, session.sizing,
                             trajectory="S2")
    kinematics = document["kinematics"]
    assert kinematics["trajectory"] == "S2"
    assert set(kinematics["scenarios"]) == {"S1", "S2", "S3"}
    assert kinematics["scenarios"]["S1"]["status"] == "literature"
    assert kinematics["scenarios"]["S1"]["knots"]
    assert kinematics["flexion"] is None


def _without(landmarks, *ids):
    from tka_planner.core.landmarks import LandmarkSet

    return LandmarkSet(landmarks.case_id, landmarks.side, landmarks.coordinate_system,
                       [one for one in landmarks if one.id not in ids])


@pytest.mark.parametrize("width, twist", [(75.0, 0.0), (60.0, 0.0), (75.0, 5.0)])
def test_the_condyle_centre_spacing_is_the_posterior_condyles_along_the_hinge(
        width, twist):
    """On a synthetic knee the posterior condyles are ``width`` apart along a line
    turned ``twist`` from the epicondylar axis, so along the hinge they are
    ``width cos(twist)`` apart, and S1's pivot goes to half that."""
    from tests.synthetic import synthetic_knee
    from tka_planner.core.frames import build_femoral_frame as femoral_frame
    from tka_planner.pipeline import condyle_centre_spacing

    landmarks = synthetic_knee(femoral_width_mm=width, condylar_twist_deg=twist)
    frame = femoral_frame(landmarks)
    spacing = condyle_centre_spacing(landmarks, frame)
    expected = width * np.cos(np.radians(twist))
    assert spacing["centre_spacing_mm"] == pytest.approx(expected, abs=1e-6)
    assert "posterior condyle" in spacing["measured_from"]

    from tka_planner.core.kinematics import knee_scenarios
    s1 = knee_scenarios(spacing["centre_spacing_mm"])["S1"]
    assert s1.pivot_ml_mm == pytest.approx(26.4 * expected / 52.8, abs=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_spacing_is_positive_on_both_sides(side):
    from tests.synthetic import synthetic_knee
    from tka_planner.core.frames import build_femoral_frame as femoral_frame
    from tka_planner.pipeline import condyle_centre_spacing

    landmarks = synthetic_knee(side=side, condylar_twist_deg=0.0)
    spacing = condyle_centre_spacing(landmarks, femoral_frame(landmarks))
    assert spacing["centre_spacing_mm"] == pytest.approx(75.0, abs=1e-6)


def test_a_knee_without_a_posterior_condyle_keeps_the_fitted_pivot_with_a_warning():
    from types import SimpleNamespace

    from tests.synthetic import synthetic_knee
    from tka_planner.core.frames import build_femoral_frame as femoral_frame
    from tka_planner.pipeline import condyle_centre_spacing, scenarios_case

    full = synthetic_knee()
    landmarks = _without(full, "femur.condyle_posterior_lateral")
    frame = femoral_frame(full)
    spacing = condyle_centre_spacing(landmarks, frame)
    assert spacing["centre_spacing_mm"] is None
    assert "femur.condyle_posterior_lateral" in spacing["reason"]

    s1 = scenarios_case(SimpleNamespace(landmarks=landmarks, femoral_frame=frame))["S1"]
    assert s1.pivot_ml_mm == 26.4 and s1.pivot_scaling == "fallback"
    assert any("femur.condyle_posterior_lateral" in w for w in s1.warnings)


def test_a_plan_records_s1_scaled_to_the_knee(session):
    from tka_planner.pipeline import condyle_centre_spacing

    spacing = condyle_centre_spacing(session.measurement.landmarks,
                                     session.measurement.femoral_frame)
    s1 = plan_document(session.measurement, session.plan,
                       session.sizing)["kinematics"]["scenarios"]["S1"]
    assert s1["pivot_scaling"]["status"] == "assumed"
    assert s1["pivot_scaling"]["centre_spacing_mm"] == spacing["centre_spacing_mm"]
    assert s1["pivot_ml_mm"] == pytest.approx(
        26.4 * spacing["centre_spacing_mm"] / 52.8)
    assert s1["status"] == "literature"


def test_the_application_trial_flexes_s1_about_the_knees_own_pivot(session):
    from tka_planner.core.kinematics import SCENARIOS
    from tka_planner.pipeline import scenarios_case
    from tka_planner.scene import motion

    session.set_trial(trajectory="S1", flexion_deg=110.0)
    frame = session._tibial_frame

    def posed(trajectory):
        return motion.trial_pose(session.scene.pivot, flexion_deg=110.0,
                                 distal=-frame.z_proximal, trajectory=trajectory,
                                 anterior=frame.x_anterior, medial=frame.medial)

    scaled = scenarios_case(session.measurement)["S1"]
    assert scaled.pivot_ml_mm != SCENARIOS["S1"].pivot_ml_mm
    assert np.array_equal(session.trial_pose_at(110.0), posed(scaled))
    assert not np.allclose(session.trial_pose_at(110.0), posed(SCENARIOS["S1"]))


def test_the_preoperative_offset_is_a_metric_with_provenance(session):
    metric = session.measurement.metrics["pco_medial_mm"].to_dict()
    assert metric["status"] in ("measured", "estimated", "not_computable")
    assert "Bellemans" in metric["definition"]


def test_the_fit_reports_the_planned_offset(session):
    from tka_planner.pipeline import fit_case

    fit = fit_case(session.measurement, session.plan, session.sizing, session.library)
    pco = fit["posterior_condylar_offset"]
    assert "available" in pco
    if pco["available"]:
        for compartment in ("medial", "lateral"):
            assert set(pco[compartment]) == {"preoperative_mm", "planned_mm",
                                             "change_mm"}


def test_the_flexion_search_runs_end_to_end_and_writes_into_the_plan(session, tmp_path):
    from tka_planner.pipeline import FlexionSettings, flexion_case

    flexion = flexion_case(session.measurement, session.plan, session.sizing,
                           session.library,
                           settings=FlexionSettings(step_deg=20.0, resolution_deg=2.0))
    assert set(flexion["trajectories"]) == {"S1", "S2", "S3"}
    # S1 was flexed about this knee's own pivot, and the result says so.
    s1 = flexion["trajectories"]["S1"]["parameters"]
    assert s1["pivot_scaling"]["status"] == "assumed"
    assert s1["pivot_ml_mm"] == pytest.approx(
        26.4 * s1["pivot_scaling"]["centre_spacing_mm"] / 52.8)
    assert flexion["warnings"] == []
    for record in flexion["trajectories"].values():
        assert 0.0 <= record["max_flexion_deg"] <= 180.0
        assert record["parameters"]["name"] == record["trajectory"]
        # The stop reason and the reported limit agree with the old censored flag.
        if record["censored"]:
            assert record["stop_reason"] == "no_contact_in_scanned_models"
            assert record["flexion_limit_deg"] is None
            assert record["no_contact"]["reason"] in ("bone_not_in_scan",
                                                      "kept_apart_by_motion")
        else:
            assert record["stop_reason"] == "contact"
            assert record["flexion_limit_deg"] == record["max_flexion_deg"]
            assert "hinge_radius_mm" in record["contacts"][0]["diagnostics"]
    assert flexion["settings"]["stop_deg"] == 180.0
    extent = flexion["scan_extent"]
    for bone in ("femur", "tibia"):
        assert extent[bone]["box_from_scan"]
        assert extent[bone]["kept_from_hinge_mm"] <= extent[bone][
            "scanned_from_hinge_mm"] + 1.01
    assert flexion["envelope"]["min_deg"] <= flexion["envelope"]["max_deg"]
    document = plan_document(session.measurement, session.plan, session.sizing,
                             flexion=flexion)
    assert read_plan(write_plan(document, tmp_path / "plan.json"))[
        "kinematics"]["flexion"]["method"] == "flexion.impingement_search.v1"


# Measuring meshes already in memory
# ----------------------------------------------------------------------


def _boxes_around(landmarks):
    """A femur and a tibia that are boxes around their bones' landmarks.

    Landmark estimation has its own suite; here it is replaced by the synthetic knee, and
    the boxes only have to be bone-sized and to contain the landmarks they carry.
    """
    from tka_planner.geom.mesh import box
    from tka_planner.geom.resample import subdivide

    meshes = {}
    for bone, prefixes in (("femur", ("femur.",)), ("tibia", ("tibia.", "fibula."))):
        points = np.array([lm.position_mm for lm in landmarks
                           if lm.position_mm is not None and lm.id.startswith(prefixes)])
        low, high = points.min(axis=0) - 15.0, points.max(axis=0) + 15.0
        matrix = np.eye(4)
        matrix[:3, 3] = (low + high) / 2.0
        # Subdivided, so the sizing sections below the joint find vertices to measure.
        mesh = box(high - low, matrix)
        for _ in range(6):
            mesh = subdivide(mesh)
        meshes[bone] = mesh
    return meshes["femur"], meshes["tibia"]


@pytest.fixture
def synthetic_estimate(monkeypatch):
    import tka_planner.pipeline as pipeline
    from tests.synthetic import synthetic_knee

    landmarks = synthetic_knee()
    monkeypatch.setattr(pipeline, "estimate_landmarks",
                        lambda femur, tibia, side, *, case_id, **axes: landmarks)
    return landmarks


def test_measuring_meshes_is_measuring_the_files(synthetic_estimate, tmp_path):
    from tka_planner.core.meshio import read_stl
    from tka_planner.geom.mesh import write_stl
    from tka_planner.pipeline import measure_case, measure_meshes

    femur, tibia = _boxes_around(synthetic_estimate)
    write_stl(femur, tmp_path / "FD.stl")
    write_stl(tibia, tmp_path / "TD.stl")
    from_files = measure_case(tmp_path / "FD.stl", tmp_path / "TD.stl", "left",
                              case_id="CASE_TEST")
    in_memory = measure_meshes(read_stl(tmp_path / "FD.stl"),
                               read_stl(tmp_path / "TD.stl"), "left",
                               case_id="CASE_TEST")
    assert {k: m.value for k, m in in_memory.metrics.items()} == \
        {k: m.value for k, m in from_files.metrics.items()}
    assert [f.code for f in in_memory.qc.findings] == \
        [f.code for f in from_files.qc.findings]


def test_supplied_mesh_quality_is_used_instead_of_recomputed(synthetic_estimate,
                                                             monkeypatch):
    import tka_planner.pipeline as pipeline
    from tka_planner.core.qc import QCReport, Severity

    def not_again(mesh):
        raise AssertionError("mesh quality was recomputed")

    monkeypatch.setattr(pipeline, "assess_mesh_quality", not_again)
    supplied = QCReport()
    supplied.add("MESH_WATERTIGHT", Severity.INFO, "from the original")
    femur, tibia = _boxes_around(synthetic_estimate)
    measurement = pipeline.measure_meshes(femur, tibia, "left", case_id="CASE_TEST",
                                          mesh_quality=supplied)
    assert "MESH_WATERTIGHT" in [f.code for f in measurement.qc.findings]


def test_measuring_meshes_still_refuses_an_implausible_bone(synthetic_estimate):
    from tka_planner.core.qc import QualityControlError
    from tka_planner.geom.mesh import box
    from tka_planner.pipeline import measure_meshes

    _, tibia = _boxes_around(synthetic_estimate)
    with pytest.raises(QualityControlError):
        measure_meshes(box(2.0), tibia, "left", case_id="CASE_TEST")
