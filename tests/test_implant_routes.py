"""The implant routes of the one app: Implant, Build and the motion on the parts.

On the public demo knee, through the routes as the viewer calls them, with Fusion
mocked (or absent, which is the point of one test).
"""

import json
import shutil
import time
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "eval" / "demo_cases" / "DEMO_LARGE"
DEMO_BUILD = REPO / "eval" / "patient_builds" / "DEMO_LARGE"
LIBRARY = REPO / "data" / "implants"

pytestmark = pytest.mark.skipif(
    not (DEMO / "FD.stl").exists() or not (LIBRARY / "femoral_box.json").exists(),
    reason="needs the public demo knee and the catalogue")


@pytest.fixture(scope="module")
def opened(tmp_path_factory):
    """One server-side session on the demo knee, with an empty builds folder."""
    from tka_planner.server.api import Api
    from tka_planner.server.sessions import SessionManager

    builds = tmp_path_factory.mktemp("builds")
    manager = SessionManager(library=LIBRARY, builds=builds)
    api = Api(manager)
    status, snap = api.dispatch("POST", "/api/sessions", {"folder": str(DEMO), "side": "right"})
    assert status == 200, snap
    return api, snap["session_id"], builds


def test_implant_lists_every_parameter_with_its_rule(opened):
    api, sid, _ = opened
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/parametric")
    assert status == 200, out
    names = {(r["part"], r["name"]) for r in out["parameters"]}
    assert {("femoral", "Femur_ML"), ("plate", "Tibial_Plateau_ML"),
            ("insert", "Insert_Thickness")} <= names
    assert all(r["rule"] for r in out["parameters"])
    assert out["implant"]["label"] == "stand-in, not built"
    assert out["changed"] == {}
    # The J-curves: the bone's and the component's, per condyle.
    assert set(out["jcurve"]["condyles"]) == {"medial", "lateral"}
    assert out["jcurve"]["condyles"]["medial"]["bone"]
    assert out["jcurve"]["condyles"]["medial"]["component"]
    assert out["tray_rule"]["plate_outline_mm"]
    # The plan opened with the insert minimum met.
    assert out["insert_minimum"]["insert_of_record_mm"] >= 8.0 - 1e-6


def test_build_without_fusion_is_a_clear_error_and_breaks_nothing(opened, monkeypatch):
    from tka_planner.fusion import transport

    def refused(*a, **k):
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(transport.urllib.request, "urlopen", refused)
    api, sid, builds = opened
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/build")
    assert status == 503
    assert "Fusion is not reachable on 127.0.0.1:27182" in out["error"]
    assert api.dispatch("GET", "/api/build")[1]["state"] == "idle"
    status, snap = api.dispatch("GET", f"/api/sessions/{sid}")
    assert status == 200 and snap["implant"]["source"] == "stand_in"
    assert not any(builds.iterdir())


def test_the_motion_on_the_parts_follows_the_medial_pivot(opened):
    api, sid, _ = opened
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/kinematics",
                               {"model": "published", "max_deg": 120, "steps": 12})
    assert status == 200, out
    frames = out["frames"]
    assert len(frames) == 13 and frames[-1]["flexion_deg"] == 120
    # The lateral condyle rolls back more than the medial: the femur turns external.
    assert frames[9]["lateral_ap_mm"] < frames[9]["medial_ap_mm"]
    assert out["published"]["flexion_deg"][0] == 0


def test_a_trial_with_a_motion_model_reports_the_kinematics(opened):
    api, sid, _ = opened
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/trial",
                               {"changes": {"flexion_deg": 60}, "model": "published"})
    assert status == 200, out
    assert out["kinematics"]["model"] == "published"
    assert out["delta"]["poses"]["TibialSet"]
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/trial", {"model": "planner"})
    assert status == 200 and "kinematics" not in out
    api.dispatch("POST", f"/api/sessions/{sid}/trial/reset")


def test_anatomy_names_each_landmark_and_its_role(opened):
    api, sid, _ = opened
    status, out = api.dispatch("GET", f"/api/sessions/{sid}/anatomy")
    assert status == 200
    by_id = {lm["id"]: lm for lm in out["landmarks"]}
    assert by_id["femur.condyle_distal_medial"]["role"] == "depth of the distal cut"
    assert out["facts"]["tibial_cut_ml_mm"] > 0


def test_a_build_from_the_app_lands_in_the_builds_folder_and_shows(opened, monkeypatch):
    """Fusion mocked: the parts it 'builds' are the demo knee's published ones."""
    from tka_planner.fusion import transport

    if not (DEMO_BUILD / "femoral.stl").exists():
        pytest.skip("no published demo build")
    api, sid, builds = opened
    config = builds.parent / "local.json"
    config.write_text(json.dumps({"sources": {k: f"lineage-{k}" for k in
                                              ("femoral", "plate", "insert", "block", "tblock")}}))
    api.fusion_config_path = config

    def fake(script, read_only, url=None, timeout=None):
        out = Path(eval(script.split("OUT = ", 1)[1].splitlines()[0]))  # noqa: S307
        report = {"name": "DEMO_LARGE"}
        for part, stem in (("femoral", "femoral"), ("plate", "plate"), ("insert", "insert"),
                           ("block", "femoral_block"), ("tblock", "tibial_block")):
            shutil.copy(DEMO_BUILD / f"{stem}.stl", out / f"{stem}.stl")
            report[part] = {"saved": True, "version": 12, "health": []}
        (out / "build.json").write_text(json.dumps(report))
        return '{"name": "DEMO_LARGE", "error": null}'

    monkeypatch.setattr(transport, "ping", lambda url, timeout=5: None)
    monkeypatch.setattr(transport, "call_fusion", fake)
    status, out = api.dispatch("POST", f"/api/sessions/{sid}/build")
    assert status == 200 and out["state"] == "running"
    for _ in range(100):
        state = api.dispatch("GET", "/api/build")[1]
        if state["state"] != "running":
            break
        time.sleep(0.1)
    assert state["state"] == "done", state
    status, env = api.dispatch("POST", f"/api/sessions/{sid}/implant/refresh")
    assert status == 200
    assert env["implant"]["source"] == "built" and env["implant"]["label"] == "built"
    assert env["implant"]["versions"]["femoral"] == 12
    assert "femoral_component" in env["delta"]["meshes"]


def test_requests_that_change_a_session_take_turns():
    """The server is threaded (meshes load while a commit cuts); two requests that plan,
    cut or derive on the same session must not run at once."""
    import threading

    from tka_planner.server.api import Api

    class Manager:
        store = None
        builds = None

    api = Api(Manager())
    inside, overlap = [], []

    def slow(method, parts, body):
        if inside:
            overlap.append(1)
        inside.append(1)
        time.sleep(0.05)
        inside.pop()
        return {}

    api._route = slow
    threads = [threading.Thread(target=api.dispatch, args=("POST", "/api/sessions/x/controls", {}))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap == []
