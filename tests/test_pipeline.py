"""The one pipeline: the command line and the application plan a case identically.

The synthetic tests run everywhere. The real-case test runs only where a case's STL files
are present, which is never in CI -- patient data is not in the repository -- but always
on a planning machine, where it is the check that matters: the same case through both
front ends gives the same plan.
"""

import json
from pathlib import Path

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
    assert kinematics["scenarios"]["S1"]["status"] == "placeholder"
    assert kinematics["flexion"] is None


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
    for record in flexion["trajectories"].values():
        assert 0.0 <= record["max_flexion_deg"] <= 170.0
        assert record["parameters"]["name"] == record["trajectory"]
    assert flexion["envelope"]["min_deg"] <= flexion["envelope"]["max_deg"]
    document = plan_document(session.measurement, session.plan, session.sizing,
                             flexion=flexion)
    assert read_plan(write_plan(document, tmp_path / "plan.json"))[
        "kinematics"]["flexion"]["method"] == "flexion.impingement_search.v1"
