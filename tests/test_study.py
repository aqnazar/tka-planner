"""The on-screen observer studies through the API: picking (E3) and plan review (E13)."""

import json

import numpy as np
import pytest

from tka_planner.core.landmarks import LandmarkStatus, read_landmark_set
from tka_planner.server import Api, SessionManager
from tka_planner.store import Store
from tka_planner.study import pickable_landmarks


@pytest.fixture
def manager(raw_session, tmp_path, monkeypatch):
    from tka_planner.server import sessions as sessions_module

    def fake_open(cls, folder, *, side, library=None, landmarks=None, **options):
        raw_session.build()
        return raw_session

    monkeypatch.setattr(sessions_module.PlanningSession, "open", classmethod(fake_open))
    return SessionManager(store=Store(tmp_path / "store"), cases_root=tmp_path,
                          study_dir=tmp_path / "study")


@pytest.fixture
def api(manager):
    return Api(manager)


@pytest.fixture
def session_id(api, folder):
    status, payload = api.dispatch("POST", "/api/sessions",
                                   {"folder": str(folder), "side": "left"})
    assert status == 200, payload
    return payload["session_id"]


def call(api, method, path, body=None, status=200):
    got, payload = api.dispatch(method, path, body or {})
    assert got == status, payload
    return payload


def targets(api, session_id, mode):
    return call(api, "POST", f"/api/sessions/{session_id}/study/targets", {"mode": mode})


def on_femur(manager, session_id, index=0):
    entry = manager.get(session_id)
    mesh = entry.session.scene.meshes[entry.pristine["Femur"]]
    return [float(v) for v in mesh.vertices[index]]


# ----------------------------------------------------------------------
# What the surgeon is given
# ----------------------------------------------------------------------


def test_the_study_vocabulary_is_served(api):
    schema = call(api, "GET", "/api/study/schema")
    assert schema["pick_modes"] == ["blank", "adjust"]
    assert ["minor", "Minor: would adjust during surgery"] in schema["severities"]
    assert any(key == "other" for key, _ in schema["reasons"])


def test_only_surface_landmarks_of_the_loaded_bones_are_picked():
    ids = [definition.id for definition in pickable_landmarks()]
    assert ids[0].startswith("femur.") and "femur.epicondyle_lateral" in ids
    assert "tibia.plateau_medial_lowest" in ids and "fibula.head_apex" in ids
    assert not [i for i in ids if "canal" in i or "centre" in i.split(".")[1]
                and i != "femur.notch_centre"]
    assert not [i for i in ids if i.startswith("patella.")]


def test_the_targets_come_numbered_in_sequence_femur_first(api, session_id):
    given = targets(api, session_id, "blank")
    numbers = [t["number"] for t in given["targets"]]
    assert numbers == list(range(1, len(numbers) + 1))
    bones = [t["bone"] for t in given["targets"]]
    assert bones == sorted(bones)      # every Femur before every Tibia
    assert all(t["definition"] and t["name"] for t in given["targets"])


def test_a_blank_start_never_sends_the_automatic_estimate(api, session_id):
    given = targets(api, session_id, "blank")
    assert all("auto_mm" not in t for t in given["targets"])


def test_an_adjust_start_sends_the_automatic_markers(api, session_id, manager):
    given = targets(api, session_id, "adjust")
    automatic = manager.get(session_id).session.measurement.landmarks
    for target in given["targets"]:
        expected = automatic.get(target["id"]).position_mm
        np.testing.assert_allclose(target["auto_mm"], expected)


def test_picking_is_on_the_uncut_bones_even_after_a_commit(api, session_id, manager):
    before = targets(api, session_id, "blank")["bones"]
    call(api, "POST", f"/api/sessions/{session_id}/commit")
    entry = manager.get(session_id)
    assert entry.session.scene.nodes["Femur"].mesh_id != before["Femur"]["mesh"]
    after = targets(api, session_id, "blank")["bones"]
    assert after["Femur"]["mesh"] == before["Femur"]["mesh"] == entry.pristine["Femur"]
    call(api, "GET", f"/api/sessions/{session_id}/meshes/{after['Femur']['mesh']}.glb")


def test_an_unknown_pick_mode_is_refused(api, session_id):
    call(api, "POST", f"/api/sessions/{session_id}/study/targets", {"mode": "x"}, 400)


# ----------------------------------------------------------------------
# Saving picks
# ----------------------------------------------------------------------


def test_a_saved_set_is_an_ordinary_landmark_file(api, session_id, manager):
    point = on_femur(manager, session_id)
    saved = call(api, "POST", f"/api/sessions/{session_id}/study/picks", {
        "observer": "S01", "round": 1, "mode": "blank", "complete": False,
        "picks": [
            {"id": "femur.epicondyle_lateral", "position_mm": point, "seconds": 12.5},
            {"id": "femur.notch_centre", "position_mm": None,
             "reason": "osteophyte covers it"},
        ],
    })
    assert saved["placed"] == 1 and saved["skipped"] == 1
    landmarks = read_landmark_set(saved["landmarks"])
    picked = landmarks.get("femur.epicondyle_lateral")
    assert picked.status is LandmarkStatus.PRESENT
    assert picked.origin == "observer:S01"
    np.testing.assert_allclose(picked.position_mm, point, atol=1e-6)
    skipped = landmarks.get("femur.notch_centre")
    assert skipped.status is LandmarkStatus.NOT_PICKED
    assert skipped.reason == "osteophyte covers it"
    record = json.loads(open(saved["record"], encoding="utf-8").read())
    assert record["observer"] == "S01" and record["mode"] == "blank"
    assert record["picks"][0]["surface_gap_mm"] == pytest.approx(0.0, abs=1e-6)
    assert "study" in saved["landmarks"] and "S01" in saved["landmarks"]


def test_an_adjusted_marker_records_how_far_it_moved(api, session_id, manager):
    automatic = manager.get(session_id).session.measurement.landmarks
    auto = np.asarray(automatic.get("femur.epicondyle_lateral").position_mm)
    point = on_femur(manager, session_id)
    saved = call(api, "POST", f"/api/sessions/{session_id}/study/picks", {
        "observer": "S01", "round": 2, "mode": "adjust",
        "picks": [{"id": "femur.epicondyle_lateral", "position_mm": point}],
    })
    record = json.loads(open(saved["record"], encoding="utf-8").read())
    assert record["picks"][0]["moved_from_auto_mm"] == pytest.approx(
        float(np.linalg.norm(np.asarray(point) - auto)), abs=1e-3)


@pytest.mark.parametrize("body, message", [
    ({"observer": "S 01", "round": 1, "mode": "blank"}, "observer"),
    ({"observer": "S01", "round": 0, "mode": "blank"}, "round"),
    ({"observer": "S01", "round": 1, "mode": "blank",
      "picks": [{"id": "femur.canal_centre_distal", "position_mm": [0, 0, 0]}]}, "study"),
    ({"observer": "S01", "round": 1, "mode": "blank", "picks": []}, "No picks"),
])
def test_a_malformed_set_is_refused(api, session_id, body, message):
    payload = call(api, "POST", f"/api/sessions/{session_id}/study/picks", body, 400)
    assert message in payload["error"]


def test_a_pick_off_the_bone_is_refused(api, session_id, manager):
    point = np.asarray(on_femur(manager, session_id)) + [0.0, 0.0, 500.0]
    payload = call(api, "POST", f"/api/sessions/{session_id}/study/picks", {
        "observer": "S01", "round": 1, "mode": "blank",
        "picks": [{"id": "femur.epicondyle_lateral", "position_mm": point.tolist()}],
    }, 400)
    assert "surface" in payload["error"]


# ----------------------------------------------------------------------
# Plan review
# ----------------------------------------------------------------------


def test_each_verdict_is_a_line_with_the_plan_it_judged(api, session_id, manager):
    path = f"/api/sessions/{session_id}/study/review"
    call(api, "POST", path, {"observer": "S01", "round": 1, "verdict": "ok"})
    call(api, "POST", f"/api/sessions/{session_id}/controls",
         {"changes": {"philosophy": "kinematic"}})
    saved = call(api, "POST", path, {
        "observer": "S01", "round": 1, "verdict": "not_ok",
        "reasons": ["tibial_slope", "size"], "severity": "minor", "comment": "slope",
    })
    lines = [json.loads(line) for line in open(saved["file"], encoding="utf-8")]
    assert [line["alignment"] for line in lines] == ["mechanical", "kinematic"]
    assert [line["verdict"] for line in lines] == ["ok", "not_ok"]
    assert lines[0]["severity"] is None and lines[1]["severity"] == "minor"
    assert lines[1]["reasons"] == ["tibial_slope", "size"]
    assert lines[1]["report"] and lines[1]["adjusted"] is False


def test_a_hand_adjusted_plan_is_marked_as_such(api, session_id):
    call(api, "POST", f"/api/sessions/{session_id}/controls",
         {"changes": {"tibial_slope_delta_deg": 2.0}})
    saved = call(api, "POST", f"/api/sessions/{session_id}/study/review",
                 {"observer": "S01", "round": 1, "verdict": "ok"})
    line = json.loads(open(saved["file"], encoding="utf-8").read().splitlines()[-1])
    assert line["adjusted"] is True


@pytest.mark.parametrize("body", [
    {"observer": "S01", "round": 1, "verdict": "maybe"},
    {"observer": "S01", "round": 1, "verdict": "not_ok", "reasons": ["size"]},
    {"observer": "S01", "round": 1, "verdict": "not_ok", "severity": "major"},
    {"observer": "S01", "round": 1, "verdict": "not_ok", "severity": "major",
     "reasons": ["nonsense"]},
])
def test_an_incomplete_verdict_is_refused(api, session_id, body):
    call(api, "POST", f"/api/sessions/{session_id}/study/review", body, 400)


def test_without_a_study_folder_nothing_is_saved(raw_session, folder, monkeypatch):
    from tka_planner.server import sessions as sessions_module

    monkeypatch.setattr(sessions_module.PlanningSession, "open",
                        classmethod(lambda cls, *a, **k: (raw_session.build(), raw_session)[1]))
    api = Api(SessionManager())
    status, opened = api.dispatch("POST", "/api/sessions", {"folder": str(folder)})
    assert status == 200, opened
    status, payload = api.dispatch(
        "POST", f"/api/sessions/{opened['session_id']}/study/review",
        {"observer": "S01", "round": 1, "verdict": "ok"})
    assert status == 400 and "--study" in payload["error"]
