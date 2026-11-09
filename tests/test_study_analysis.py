"""The observer-study analysis (E3 dispersion, E13 verdicts), on synthetic study folders.

The files are written here by hand in the format ``tka_planner.study`` saves, so the
analysis is checked against known offsets without opening a session.
"""

import csv
import json
import math

import numpy as np
import pytest

from paper.study_analysis import (
    landmark_dispersion,
    load_picks,
    main,
    review_summary,
    write_e5_dispersion_csv,
)
from tka_planner.validation.montecarlo import read_dispersion

LAT = "femur.epicondyle_lateral"
NOTCH = "femur.notch_centre"
T = np.array([10.0, 20.0, 30.0])      # where LAT "really" is
N = np.array([0.0, 50.0, -5.0])       # where NOTCH "really" is
AUTO = T + np.array([2.5, 0.0, 4.0])  # the automatic estimate of LAT


def _write_set(study, folder, *, observer, case="SYN01", side="left", mode="blank",
               round_=1, stamp="20261108T100000Z", picks=None, skipped=(),
               observer_key=None, auto=None, complete=True):
    """One pick save: a ``tka-landmarks/1.0.0`` file and its ``.study.json`` record."""
    target = study / "picks" / folder
    target.mkdir(parents=True, exist_ok=True)
    base = f"{case}_{side}_{mode}_r{round_}_{stamp}"
    source = {"kind": "observer_study", "observer": observer, "round": round_,
              "mode": mode, "saved_utc": stamp}
    if observer_key is not None:
        source["observer_key"] = observer_key
    points, records = [], []
    for landmark_id, position in (picks or {}).items():
        position = [float(v) for v in position]
        points.append({"id": landmark_id, "position_mm": position, "status": "present",
                       "origin": f"observer:{observer}"})
        record = {"id": landmark_id, "seconds": 5.0, "position_mm": position}
        if auto is not None and landmark_id in auto:
            record["moved_from_auto_mm"] = float(
                np.linalg.norm(np.asarray(position) - auto[landmark_id]))
        records.append(record)
    for landmark_id in skipped:
        points.append({"id": landmark_id, "position_mm": None, "status": "not_picked",
                       "origin": f"observer:{observer}", "reason": "not identifiable"})
        records.append({"id": landmark_id, "seconds": 3.0, "skipped": "not identifiable"})
    (target / f"{base}.landmarks.json").write_text(json.dumps({
        "schema_version": "tka-landmarks/1.0.0", "case_id": case, "side": side,
        "coordinate_system": "LPS", "units": "mm", "source": source, "points": points,
    }), encoding="utf-8")
    (target / f"{base}.study.json").write_text(json.dumps({
        **source, "case_id": case, "side": side, "complete": complete,
        "started_utc": None, "total_seconds": 60.0,
        "landmark_file": f"{base}.landmarks.json", "picks": records,
    }), encoding="utf-8")


@pytest.fixture
def study(tmp_path):
    """Two observers x two blank rounds on one knee, plus noise the analysis must drop.

    LAT: observer A picks T +- 1 mm in x (own mean T, variance 2 mm^2); observer B picks
    T + (5, +-2, 0) (own mean T + 5x, variance 8 mm^2). Pooled intra variance 5 mm^2;
    observer means T and T + 5x, grand mean T + 2.5x, inter variance 12.5 mm^2.
    """
    auto = {LAT: AUTO}
    # Observer A appears under two folder names (as a renamed slug would) with one key.
    _write_set(tmp_path, "dr-a", observer="Dr. A", observer_key="dr. a", round_=1,
               picks={LAT: T + [1, 0, 0], NOTCH: N + [1, 0, 0]}, auto=auto)
    _write_set(tmp_path, "dr-a-1234abcd", observer="dr.  a", observer_key="dr. a",
               round_=2, picks={LAT: T + [-1, 0, 0], NOTCH: N + [-1, 0, 0]}, auto=auto)
    # Observer B: an old-style record with no key, grouped by its folder.
    _write_set(tmp_path, "B", observer="B", round_=1,
               picks={LAT: T + [5, 2, 0], NOTCH: N + [3, 0, 0]}, auto=auto)
    # B round 2 was saved twice: a partial "Save progress", then the finished set.
    _write_set(tmp_path, "B", observer="B", round_=2, stamp="20261108T090000Z",
               picks={LAT: T + [50, 50, 50]}, auto=auto, complete=False)
    _write_set(tmp_path, "B", observer="B", round_=2, stamp="20261108T091500Z",
               picks={LAT: T + [5, -2, 0]}, skipped=[NOTCH], auto=auto)
    # An adjust-mode set far away: not E3 unless asked for.
    _write_set(tmp_path, "dr-a", observer="Dr. A", observer_key="dr. a", mode="adjust",
               round_=1, picks={LAT: T + [100, 0, 0]})
    return tmp_path


def test_load_picks_reads_every_set(study):
    picks = load_picks(study)
    assert len(picks) == 6
    keys = {p.observer_key for p in picks}
    assert keys == {"dr. a", "B"}
    one = next(p for p in picks if p.observer_key == "B" and p.round == 2
               and p.complete)
    assert NOTCH in one.skipped and NOTCH not in one.positions
    np.testing.assert_allclose(one.positions[LAT], T + [5, -2, 0])


def test_two_observers_two_rounds_give_the_expected_sds(study):
    result = landmark_dispersion(load_picks(study))[LAT]
    assert result["intra_sd_mm"] == pytest.approx(math.sqrt(5.0), abs=1e-6)
    assert result["inter_sd_mm"] == pytest.approx(math.sqrt(12.5), abs=1e-6)
    np.testing.assert_allclose(result["intra_sd_axes_mm"], [1.0, 2.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result["inter_sd_axes_mm"],
                               [math.sqrt(12.5), 0.0, 0.0], atol=1e-6)
    assert result["n_observers"] == 2
    assert result["n_rounds"] == 2
    assert result["n_picks"] == 4


def test_the_auto_estimate_is_compared_with_the_grand_mean(study):
    picks = load_picks(study)
    # From each pick's recorded distance to the estimate (no position stored).
    by_distance = landmark_dispersion(picks)[LAT]
    assert by_distance["auto_minus_mean_mm"] == pytest.approx(4.0, abs=1e-6)
    assert by_distance["auto_source"] == "distances"
    # From the estimate's position, when it is given.
    by_position = landmark_dispersion(
        picks, auto={("SYN01", "left"): {LAT: AUTO + [0, 3, 0]}})[LAT]
    assert by_position["auto_minus_mean_mm"] == pytest.approx(5.0, abs=1e-6)
    assert by_position["auto_source"] == "position"
    # NOTCH has neither: not a number.
    assert math.isnan(landmark_dispersion(picks)[NOTCH]["auto_minus_mean_mm"])


def test_adjust_sets_are_left_out_unless_asked_for(study):
    picks = load_picks(study)
    blank = landmark_dispersion(picks)[LAT]
    both = landmark_dispersion(picks, include_adjust=True)[LAT]
    assert blank["n_picks"] == 4
    assert both["n_picks"] == 5
    assert both["intra_sd_mm"] > blank["intra_sd_mm"]


def test_skipped_landmarks_are_excluded_and_counted(study):
    result = landmark_dispersion(load_picks(study))[NOTCH]
    assert result["n_skipped"] == 1
    assert result["n_picks"] == 3
    # Only observer A has two rounds: x offsets +-1 -> variance 2.
    assert result["intra_sd_mm"] == pytest.approx(math.sqrt(2.0), abs=1e-6)
    # Observer means N and N + 3x: grand mean N + 1.5x, variance 4.5.
    assert result["inter_sd_mm"] == pytest.approx(math.sqrt(4.5), abs=1e-6)


def test_the_e5_table_round_trips_through_e5s_reader(study, tmp_path):
    dispersion = landmark_dispersion(load_picks(study))
    path = write_e5_dispersion_csv(dispersion, tmp_path / "e5_dispersion.csv")
    table = read_dispersion(path)
    # A surface pick's 3D scatter r is an isotropic per-axis SD of r / sqrt(2).
    assert table.sd_mm(LAT) == pytest.approx(math.sqrt(12.5) / math.sqrt(2), abs=1e-6)
    assert table.sd_mm(NOTCH) == pytest.approx(math.sqrt(4.5) / math.sqrt(2), abs=1e-5)
    assert not table.placeholder
    assert "observer" in table.label.lower()
    intra = read_dispersion(write_e5_dispersion_csv(
        dispersion, tmp_path / "intra.csv", which="intra"))
    assert intra.sd_mm(LAT) == pytest.approx(math.sqrt(5.0) / math.sqrt(2), abs=1e-6)


def test_a_landmark_seen_by_one_observer_is_left_out_of_the_e5_table(tmp_path):
    _write_set(tmp_path, "A", observer="A", round_=1, picks={LAT: T})
    _write_set(tmp_path, "A", observer="A", round_=2, picks={LAT: T + [1, 0, 0]})
    dispersion = landmark_dispersion(load_picks(tmp_path))
    assert math.isnan(dispersion[LAT]["inter_sd_mm"])
    path = write_e5_dispersion_csv(dispersion, tmp_path / "e5.csv")
    assert LAT not in read_dispersion(path).covariances
    assert LAT in path.read_text(encoding="utf-8")  # named in the comments


def _review(observer, verdict, *, alignment="mechanical", reasons=(), severity=None,
            seconds=10.0, round_=1, saved="2026-11-08T10:00:00Z", key=None, case="SYN01"):
    record = {"saved_utc": saved, "observer": observer, "round": round_, "case_id": case,
              "side": "left", "alignment": alignment, "verdict": verdict,
              "reasons": list(reasons), "severity": severity, "comment": "",
              "seconds": seconds, "adjusted": False, "stale": False, "controls": {},
              "report": []}
    if key is not None:
        record["observer_key"] = key
    return record


def _write_reviews(study, name, records):
    folder = study / "reviews"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{name}.jsonl").open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_review_summary_counts_verdicts(tmp_path):
    _write_reviews(tmp_path, "dr-a", [
        _review("Dr. A", "ok", seconds=30, key="dr. a"),
        _review("Dr. A", "not_ok", alignment="kinematic", reasons=["rotation"],
                severity="minor", seconds=10, key="dr. a"),
    ])
    _write_reviews(tmp_path, "S02", [
        _review("S02", "not_ok", reasons=["size", "fit"], severity="major", seconds=50),
        # Saved again in the same round: the later verdict stands.
        _review("S02", "not_ok", reasons=["size"], severity="major", seconds=70,
                saved="2026-11-08T10:05:00Z"),
    ])
    # The same surgeon under a second file name (key decides).
    _write_reviews(tmp_path, "dr-a-99", [
        _review("dr. a", "ok", seconds=20, round_=2, key="dr. a"),
    ])
    rows = {(r["case_id"], r["side"], r["alignment"]): r for r in review_summary(tmp_path)}
    mech = rows[("SYN01", "left", "mechanical")]
    assert mech["n_reviews"] == 3 and mech["n_observers"] == 2
    assert mech["ok"] == 2 and mech["not_ok"] == 1
    assert mech["reason_size"] == 1 and mech["reason_fit"] == 0
    assert mech["severity_major"] == 1 and mech["severity_minor"] == 0
    assert mech["median_seconds"] == pytest.approx(30.0)
    assert mech["n_superseded"] == 1
    kin = rows[("SYN01", "left", "kinematic")]
    assert kin["not_ok"] == 1 and kin["reason_rotation"] == 1 and kin["severity_minor"] == 1
    assert kin["n_observers"] == 1


def test_the_command_writes_its_tables(study, tmp_path):
    _write_reviews(study, "B", [_review("B", "ok")])
    out = tmp_path / "out"
    assert main([str(study), "--out", str(out)]) == 0
    for name in ("dispersion.csv", "e5_dispersion.csv", "reviews_summary.csv",
                 "README.txt"):
        assert (out / name).is_file(), name
    with open(out / "dispersion.csv", newline="", encoding="utf-8") as handle:
        rows = {row["landmark_id"]: row for row in csv.DictReader(handle)}
    assert float(rows[LAT]["inter_sd_mm"]) == pytest.approx(math.sqrt(12.5), abs=1e-4)
    assert read_dispersion(out / "e5_dispersion.csv").sd_mm(LAT) > 0
