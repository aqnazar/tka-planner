"""The paper harness's case list: which knees an experiment runs on.

A cohort folder is faked with empty files, since only the names are read.
"""

import pytest

from paper import harness


def _knee(root, folder, case):
    knee = root / folder
    knee.mkdir(parents=True)
    for name in ("FD1.stl", "TD1.stl", f"seg_{case[:4]}_{case[4]}_femur.stl"):
        (knee / name).write_bytes(b"")


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path / "cohort"
    for i, case in enumerate(("P001L", "P009L", "P009R")):
        _knee(root, f"Patient_{i}", case)
    reference = tmp_path / "ref"
    reference.mkdir()
    for name in ("FD1Right.stl", "TD1Right.stl"):
        (reference / name).write_bytes(b"")
    monkeypatch.setenv("TKA_COHORT_DIR", str(root))
    monkeypatch.setenv("TKA_REFERENCE_DIR", str(reference))
    # setenv first, so the variable is restored (removed) after the test even when the
    # command line sets it.
    monkeypatch.setenv("TKA_PAPER_CASES", "")
    monkeypatch.delenv("TKA_PAPER_CASES")
    return root


def test_without_a_filter_every_knee_is_listed(cohort):
    assert [c.case_id for c in harness.cohort_cases()] == ["P001L", "P009L", "P009R"]
    assert harness.cohort_cases(include_reference=True)[0].case_id == "REF_KJ2"


def test_the_filter_keeps_only_the_named_knees_and_brings_in_the_reference(
        cohort, monkeypatch):
    monkeypatch.setenv("TKA_PAPER_CASES", "ref_kj2, P009L")
    cases = harness.cohort_cases()
    assert [c.case_id for c in cases] == ["REF_KJ2", "P009L"]
    assert cases[0].reference and cases[1].side == "left"


def test_a_filter_naming_an_unknown_knee_is_refused(cohort, monkeypatch):
    monkeypatch.setenv("TKA_PAPER_CASES", "P009L,P999L")
    with pytest.raises(SystemExit, match="P999L"):
        harness.cohort_cases()


def test_the_command_line_option_sets_the_filter(cohort, monkeypatch):
    from paper import __main__ as cli

    seen = {}

    def fake_run(experiment_id, **_):
        seen[experiment_id] = [c.case_id for c in harness.cohort_cases()]
        return cohort

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    assert cli.main(["run", "E0", "--cases", "P009R"]) == 0
    assert seen == {"E0": ["P009R"]}
