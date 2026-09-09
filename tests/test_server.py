"""The API, driven with no browser.

This is where the spec's feature-parity list is checked. Every control the add-on panel
carries is moved through the API here, so a regression in the application shows up
without anyone opening a page.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from tka_planner.geom.gltf import MM_PER_M
from tka_planner.scene.model import TIBIAL
from tka_planner.server import Api, PlannerServer, SessionManager
from tka_planner.server.api import Binary
from tka_planner.server.httpd import free_port
from tka_planner.session import ADJUSTMENT_FIELDS
from tka_planner.store import Store


@pytest.fixture
def manager(raw_session, tmp_path, monkeypatch):
    """A manager whose sessions are the stubbed synthetic knee.

    ``PlanningSession.open`` is patched to hand back the fixture rather than measure a
    real bone, which is the same stub the session suite uses. Everything above the
    measurement -- planning, posing, cutting, caching, routing -- is the real code.
    """
    from tka_planner.server import sessions as sessions_module

    def fake_open(cls, folder, *, side, library=None):
        raw_session.build()
        return raw_session

    monkeypatch.setattr(
        sessions_module.PlanningSession, "open", classmethod(fake_open)
    )
    return SessionManager(store=Store(tmp_path / "store"), cases_root=tmp_path)


@pytest.fixture
def api(manager):
    return Api(manager)


@pytest.fixture
def opened(api, folder):
    status, payload = api.dispatch(
        "POST", "/api/sessions", {"folder": str(folder), "side": "left"}
    )
    assert status == 200, payload
    return payload


@pytest.fixture
def session_id(opened):
    return opened["session_id"]


def post(api, path, body=None):
    status, payload = api.dispatch("POST", path, body or {})
    assert status == 200, payload
    return payload


# ----------------------------------------------------------------------
# Opening
# ----------------------------------------------------------------------


def test_health_answers_before_anything_is_open(api):
    status, payload = api.dispatch("GET", "/api/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["sessions"] == []


def test_opening_returns_a_whole_scene(opened):
    names = {node["name"] for node in opened["scene"]["nodes"]}
    assert {"Femur", "Tibia"} <= names
    assert opened["scene"]["set_poses"].keys() == {"femoral", "tibial"}
    assert len(opened["scene"]["pivot"]) == 4


def test_opening_reports_the_plan_before_any_geometry_is_cut(opened):
    assert opened["stale"] is True
    assert any(line.startswith("CASE|") for line in opened["report"])


def test_a_case_folder_is_listed_once_it_is_openable(api, folder, manager):
    manager.cases_root = folder.parent
    status, payload = api.dispatch("GET", "/api/cases")

    assert status == 200
    assert any(case["case_id"] == folder.name for case in payload["cases"])


def test_opening_without_a_folder_is_a_bad_request(api):
    status, payload = api.dispatch("POST", "/api/sessions", {})
    assert status == 400
    assert "folder" in payload["error"]


def test_an_unknown_session_is_not_found(api):
    status, payload = api.dispatch("GET", "/api/sessions/deadbeef")
    assert status == 404
    assert "deadbeef" in payload["error"]


def test_an_unknown_route_is_not_found(api):
    assert api.dispatch("GET", "/api/nonsense")[0] == 404


def test_closing_forgets_the_session(api, session_id, manager):
    status, payload = api.dispatch("DELETE", f"/api/sessions/{session_id}")

    assert status == 200
    assert payload["closed"] == session_id
    assert manager.open_sessions == []


# ----------------------------------------------------------------------
# Plan mode: the thirteen adjustments and the rest of the panel
# ----------------------------------------------------------------------


@pytest.mark.parametrize("field", ADJUSTMENT_FIELDS)
def test_every_adjustment_drives_the_api(api, session_id, field):
    """The spec's list of thirteen, each one moved through the real route."""
    payload = post(
        api, f"/api/sessions/{session_id}/controls", {"changes": {field: 1.5}}
    )

    assert payload["controls"][field] == 1.5
    assert "scalars" in payload["delta"]


@pytest.mark.parametrize(
    "changes",
    [
        {"philosophy": "kinematic"},
        {"size_override": "M2"},
        {"tibial_resection_mm": 10.0},
        {"insert_thickness_mm": 12.0},
        {"use_insert": False},
        {"resection_mode": "plane"},
        {"build_bone_shells": False},
        {"show_planes": False},
        {"show_axes": False},
        {"show_landmarks": True},
        {"isolate_landmarks": True},
    ],
)
def test_every_remaining_plan_control_drives_the_api(api, session_id, changes):
    payload = post(api, f"/api/sessions/{session_id}/controls", {"changes": changes})

    for name, value in changes.items():
        assert payload["controls"][name] == value


def test_an_unknown_control_is_refused(api, session_id):
    status, payload = api.dispatch(
        "POST", f"/api/sessions/{session_id}/controls", {"changes": {"nope": 1}}
    )
    assert status == 400
    assert "nope" in payload["error"]


def test_an_empty_change_set_is_refused(api, session_id):
    status, _ = api.dispatch(
        "POST", f"/api/sessions/{session_id}/controls", {"changes": {}}
    )
    assert status == 400


def test_moving_a_cut_moves_the_plane_and_the_clip(api, session_id, opened):
    before = opened["clip"]["Tibia"]["constant_mm"]

    payload = post(
        api, f"/api/sessions/{session_id}/controls",
        {"changes": {"tibial_resection_delta_mm": 4.0}},
    )

    assert payload["clip"]["Tibia"]["constant_mm"] != before
    assert "tibial_proximal" in payload["delta"]["poses"]


def test_the_clip_keeps_opposite_sides_of_the_two_cuts(opened):
    """The femur keeps the bone above its plane and the tibia the bone below.

    One convention applied to both would invert a resection, and the sign of the two
    normals along the mechanical axis is where that would first show.
    """
    femoral = np.array(opened["clip"]["Femur"]["normal"])
    tibial = np.array(opened["clip"]["Tibia"]["normal"])

    assert float(np.dot(femoral, tibial)) < 0.0


def test_isolating_landmarks_hides_the_bones(api, session_id):
    payload = post(
        api, f"/api/sessions/{session_id}/controls",
        {"changes": {"isolate_landmarks": True}},
    )

    assert payload["delta"]["visibility"]["Femur"] is False
    assert payload["delta"]["visibility"]["Tibia"] is False


def test_leaving_isolation_restores_what_was_visible(api, session_id):
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"isolate_landmarks": True}})
    payload = post(api, f"/api/sessions/{session_id}/controls",
                   {"changes": {"isolate_landmarks": False}})

    assert payload["delta"]["visibility"]["Femur"] is True


def test_reset_returns_every_adjustment_to_zero(api, session_id):
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"femoral_varus_delta_deg": 2.0, "tibial_slope_delta_deg": 3.0}})

    payload = post(api, f"/api/sessions/{session_id}/controls/reset")

    assert all(payload["controls"][name] == 0.0 for name in ADJUSTMENT_FIELDS)


def test_the_report_carries_the_warning_when_the_cuts_stop_agreeing(api, session_id):
    payload = post(
        api, f"/api/sessions/{session_id}/controls",
        {"changes": {"femoral_varus_delta_deg": 3.0}},
    )

    assert any(line.startswith("WARN|") for line in payload["report"])


# ----------------------------------------------------------------------
# Commit
# ----------------------------------------------------------------------


def test_committing_cuts_both_bones(api, session_id):
    payload = post(api, f"/api/sessions/{session_id}/commit")

    assert sorted(payload["commit"]["computed"]) == ["Femur", "Tibia"]
    assert payload["stale"] is False
    assert {"Femur", "Tibia"} <= set(payload["delta"]["meshes"])


def test_committing_records_which_solver_cut_each_bone(api, session_id):
    payload = post(api, f"/api/sessions/{session_id}/commit")

    assert payload["commit"]["records"]["Femur"]["backend"] == "manifold3d"


def test_committing_twice_with_no_change_does_no_work(api, session_id):
    post(api, f"/api/sessions/{session_id}/commit")
    payload = post(api, f"/api/sessions/{session_id}/commit")

    assert sorted(payload["commit"]["unchanged"]) == ["Femur", "Tibia"]
    assert payload["commit"]["computed"] == []


def test_a_tibial_change_recuts_only_the_tibia(api, session_id):
    """The reason the cache is keyed per bone rather than per plan."""
    post(api, f"/api/sessions/{session_id}/commit")
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"tibial_slope_delta_deg": 2.0}})

    payload = post(api, f"/api/sessions/{session_id}/commit")

    assert payload["commit"]["computed"] == ["Tibia"]
    assert payload["commit"]["unchanged"] == ["Femur"]


def test_returning_to_a_committed_plan_reuses_the_stored_geometry(api, session_id):
    first = post(api, f"/api/sessions/{session_id}/commit")
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"tibial_slope_delta_deg": 2.0}})
    post(api, f"/api/sessions/{session_id}/commit")

    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"tibial_slope_delta_deg": 0.0}})
    again = post(api, f"/api/sessions/{session_id}/commit")

    assert again["commit"]["reused"] == ["Tibia"]
    assert again["delta"]["meshes"]["Tibia"] == first["delta"]["meshes"]["Tibia"]


def test_a_recut_starts_from_the_uncut_bone(api, session_id, manager):
    """Cutting the already-cut bone would remove more and could never put any back."""
    from tka_planner.geom.mesh import volume

    entry = manager.get(session_id)
    post(api, f"/api/sessions/{session_id}/commit")
    once = volume(entry.session.scene.mesh_for("Femur"))

    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"femoral_resection_delta_mm": 0.001}})
    post(api, f"/api/sessions/{session_id}/commit")
    twice = volume(entry.session.scene.mesh_for("Femur"))

    assert twice == pytest.approx(once, rel=1e-3)


def test_a_commit_survives_into_a_new_manager(api, session_id, manager, folder):
    """The store, not the process, is what remembers a commit."""
    first = post(api, f"/api/sessions/{session_id}/commit")

    second = Api(SessionManager(store=manager.store, cases_root=manager.cases_root))
    reopened = post(second, "/api/sessions", {"folder": str(folder), "side": "left"})
    payload = post(second, f"/api/sessions/{reopened['session_id']}/commit")

    assert sorted(payload["commit"]["reused"]) == ["Femur", "Tibia"]
    assert payload["delta"]["meshes"]["Femur"] == first["delta"]["meshes"]["Femur"]


def test_plane_mode_cuts_too(api, session_id):
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"resection_mode": "plane"}})

    payload = post(api, f"/api/sessions/{session_id}/commit")

    assert sorted(payload["commit"]["computed"]) == ["Femur", "Tibia"]


def test_the_bone_shells_are_built_in_block_mode(api, session_id, manager):
    post(api, f"/api/sessions/{session_id}/commit")

    names = set(manager.get(session_id).session.scene.nodes)
    assert {"Femur.Shell", "Tibia.Shell"} <= names


def test_shells_can_be_turned_off(api, folder, manager):
    opened = post(
        Api(manager), "/api/sessions", {"folder": str(folder), "side": "left"}
    )
    api = Api(manager)
    post(api, f"/api/sessions/{opened['session_id']}/controls",
         {"changes": {"build_bone_shells": False}})

    post(api, f"/api/sessions/{opened['session_id']}/commit")

    names = set(manager.get(opened["session_id"]).session.scene.nodes)
    assert "Femur.Shell" not in names


# ----------------------------------------------------------------------
# Reduce mode
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"flexion_deg": 60.0},
        {"varus_valgus_deg": 3.0},
        {"drawer_ap_mm": 5.0},
        {"flexion_deg": 90.0, "varus_valgus_deg": -2.0, "drawer_ap_mm": 2.0},
    ],
)
def test_every_trial_control_drives_the_api(api, session_id, changes):
    payload = post(api, f"/api/sessions/{session_id}/trial", {"changes": changes})

    for name, value in changes.items():
        assert payload["trial"][name] == value
    assert "TibialSet" in payload["delta"]["poses"]


def test_the_trial_pose_never_touches_the_plan(api, session_id, opened):
    payload = post(
        api, f"/api/sessions/{session_id}/trial", {"changes": {"flexion_deg": 45.0}}
    )

    assert payload["controls"] == opened["controls"]
    assert payload["clip"] == opened["clip"]


def test_resetting_the_trial_returns_to_extension(api, session_id):
    post(api, f"/api/sessions/{session_id}/trial",
         {"changes": {"flexion_deg": 80.0, "varus_valgus_deg": 4.0}})

    payload = post(api, f"/api/sessions/{session_id}/trial/reset")

    assert payload["trial"] == {
        "flexion_deg": 0.0, "varus_valgus_deg": 0.0, "drawer_ap_mm": 0.0
    }
    assert np.allclose(payload["delta"]["poses"]["TibialSet"], np.eye(4))


def test_an_unknown_trial_control_is_refused(api, session_id):
    status, _ = api.dispatch(
        "POST", f"/api/sessions/{session_id}/trial", {"changes": {"twist_deg": 3}}
    )
    assert status == 400


def test_the_scripted_arc_comes_back_as_poses(api, session_id):
    payload = post(
        api, f"/api/sessions/{session_id}/arc", {"max_deg": 120.0, "steps": 12}
    )

    frames = payload["frames"]
    assert len(frames) == 13
    assert frames[0]["flexion_deg"] == 0.0
    assert frames[-1]["flexion_deg"] == pytest.approx(120.0)
    assert np.allclose(frames[0]["pose"], np.eye(4))


def test_the_arc_carries_a_dialled_stress_through_it(api, session_id, manager):
    """Scrubbing the arc must combine with a stress rather than replace it."""
    post(api, f"/api/sessions/{session_id}/trial",
         {"changes": {"varus_valgus_deg": 5.0}})

    payload = post(api, f"/api/sessions/{session_id}/arc", {"steps": 4})

    assert not np.allclose(payload["frames"][0]["pose"], np.eye(4))


def test_posing_after_a_commit_moves_the_committed_geometry(api, session_id, manager):
    post(api, f"/api/sessions/{session_id}/commit")
    post(api, f"/api/sessions/{session_id}/trial", {"changes": {"flexion_deg": 30.0}})

    scene = manager.get(session_id).session.scene
    assert not np.allclose(scene.set_poses[TIBIAL], np.eye(4))
    assert not np.allclose(scene.world("Tibia"), scene.nodes["Tibia"].rest)


# ----------------------------------------------------------------------
# Meshes and export
# ----------------------------------------------------------------------


def test_a_mesh_is_served_as_binary_gltf(api, session_id, opened):
    femur = next(n for n in opened["scene"]["nodes"] if n["name"] == "Femur")

    status, payload = api.dispatch(
        "GET", f"/api/sessions/{session_id}/meshes/{femur['mesh']}.glb"
    )

    assert status == 200
    assert isinstance(payload, Binary)
    assert payload.data[:4] == b"glTF"
    assert payload.immutable is True


def test_a_missing_mesh_is_not_found(api, session_id):
    status, _ = api.dispatch("GET", f"/api/sessions/{session_id}/meshes/nosuchmesh.glb")
    assert status == 404


def test_a_committed_mesh_is_still_servable_from_the_store(api, session_id, manager):
    payload = post(api, f"/api/sessions/{session_id}/commit")
    digest = payload["delta"]["meshes"]["Femur"]

    second = Api(SessionManager(store=manager.store))
    entry = manager.get(session_id)
    second.manager._entries[session_id] = entry

    status, served = second.dispatch(
        "GET", f"/api/sessions/{session_id}/meshes/{digest}.glb"
    )
    assert status == 200
    assert served.data[:4] == b"glTF"


def test_exporting_writes_the_plan_the_report_and_the_geometry(
    api, session_id, tmp_path
):
    post(api, f"/api/sessions/{session_id}/commit")

    payload = post(
        api, f"/api/sessions/{session_id}/export", {"out": str(tmp_path / "out")}
    )

    written = payload["written"]
    assert json.loads(open(written["plan"]).read())["side"] == "left"
    assert "<html" in open(written["report"], encoding="utf-8").read().lower()
    assert any(path.endswith("Femur.stl") for path in written["meshes"])


def test_the_export_records_the_manual_adjustments(api, session_id, tmp_path):
    post(api, f"/api/sessions/{session_id}/controls",
         {"changes": {"femoral_shift_ap_mm": 2.5}})

    payload = post(
        api, f"/api/sessions/{session_id}/export",
        {"out": str(tmp_path / "out"), "meshes": False},
    )

    plan = json.loads(open(payload["written"]["plan"]).read())
    assert plan["controls"]["femoral_shift_ap_mm"] == 2.5
    assert plan["geometry_matches_plan"] is False


def test_exporting_without_a_folder_is_refused(api, session_id):
    status, _ = api.dispatch("POST", f"/api/sessions/{session_id}/export", {})
    assert status == 400


# ----------------------------------------------------------------------
# Over a real socket
# ----------------------------------------------------------------------


@pytest.fixture
def running(manager, tmp_path):
    server = PlannerServer(manager, port=free_port(), web_root=tmp_path / "web")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<h1>planner</h1>", encoding="utf-8")
    server.start()
    yield server
    server.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read(), dict(response.headers)


def _post(url, body):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def test_the_server_answers_over_a_socket(running):
    status, body, _ = _get(running.url + "api/health")

    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_the_server_serves_the_viewer(running):
    status, body, headers = _get(running.url)

    assert status == 200
    assert b"planner" in body
    assert headers["Content-Type"].startswith("text/html")


def test_a_whole_session_runs_over_the_socket(running, folder):
    status, opened = _post(
        running.url + "api/sessions", {"folder": str(folder), "side": "left"}
    )
    assert status == 200
    session_id = opened["session_id"]

    _post(running.url + f"api/sessions/{session_id}/controls",
          {"changes": {"tibial_resection_delta_mm": 1.0}})
    _, committed = _post(running.url + f"api/sessions/{session_id}/commit", {})
    _, posed = _post(running.url + f"api/sessions/{session_id}/trial",
                     {"changes": {"flexion_deg": 45.0}})

    assert committed["stale"] is False
    assert posed["trial"]["flexion_deg"] == 45.0


def test_a_mesh_downloads_over_the_socket_in_metres(running, folder):
    _, opened = _post(
        running.url + "api/sessions", {"folder": str(folder), "side": "left"}
    )
    femur = next(n for n in opened["scene"]["nodes"] if n["name"] == "Femur")

    status, body, headers = _get(
        running.url
        + f"api/sessions/{opened['session_id']}/meshes/{femur['mesh']}.glb"
    )

    assert status == 200
    assert body[:4] == b"glTF"
    assert headers["Content-Type"] == "model/gltf-binary"
    assert "immutable" in headers["Cache-Control"]
    # The document declares its own bounds; a bone is decimetres, not hundreds.
    document = json.loads(body[20:20 + int.from_bytes(body[12:16], "little")])
    span = max(document["accessors"][0]["max"]) - min(document["accessors"][0]["min"])
    assert span < 1.0 * MM_PER_M


def test_a_path_that_escapes_the_web_root_is_refused(running):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(running.url + "../tka_planner/session.py")

    assert caught.value.code in (403, 404)


def test_an_unknown_file_is_not_found(running):
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(running.url + "nothing-here.js")

    assert caught.value.code == 404


def test_the_server_binds_loopback_only(running):
    assert running.address[0] == "127.0.0.1"


def test_the_real_viewer_is_served_with_types_a_browser_accepts(manager):
    """A JavaScript module served as text/plain is refused outright by the browser."""
    server = PlannerServer(manager, port=free_port())
    server.start()
    try:
        for path, expected in (
            ("", "text/html"),
            ("app.js", "application/javascript"),
            ("style.css", "text/css"),
            ("vendor/three.module.js", "application/javascript"),
        ):
            status, body, headers = _get(server.url + path)
            assert status == 200, path
            assert headers["Content-Type"].startswith(expected), path
            assert body
    finally:
        server.stop()


def test_a_commit_describes_the_shells_it_created(api, session_id):
    """The shells exist nowhere in the snapshot the viewer drew from.

    Without their node descriptions the viewer would receive a mesh id for a name it
    has never heard of, and would silently drop the geometry a commit just spent
    seconds computing.
    """
    payload = post(api, f"/api/sessions/{session_id}/commit")

    nodes = payload["delta"]["nodes"]
    assert "Femur.Shell" in nodes
    assert nodes["Femur.Shell"]["tags"]["shell"] is True
    assert nodes["Femur.Shell"]["set"] == "femoral"
    assert nodes["Femur.Shell"]["mesh"] == payload["delta"]["meshes"]["Femur.Shell"]


def test_a_reused_commit_describes_its_shells_too(api, session_id, manager, folder):
    """A cache hit must land in the viewer the same way a fresh cut does."""
    post(api, f"/api/sessions/{session_id}/commit")

    second = Api(SessionManager(store=manager.store))
    reopened = post(second, "/api/sessions", {"folder": str(folder), "side": "left"})
    payload = post(second, f"/api/sessions/{reopened['session_id']}/commit")

    assert payload["commit"]["reused"]
    assert "Femur.Shell" in payload["delta"]["nodes"]


def test_toggling_the_planes_and_axes_changes_visibility(api, session_id):
    hidden = post(
        api, f"/api/sessions/{session_id}/controls",
        {"changes": {"show_planes": False, "show_axes": False}},
    )
    assert hidden["delta"]["visibility"]["femoral_distal"] is False
    assert hidden["delta"]["visibility"]["FemoralMechanicalAxis"] is False

    shown = post(
        api, f"/api/sessions/{session_id}/controls", {"changes": {"show_planes": True}}
    )
    assert shown["delta"]["visibility"]["femoral_distal"] is True
