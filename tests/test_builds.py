"""Built parts in the scene (one app, B4).

The planner shows the Fusion-built parts of a knee when they exist and were built for
this plan; otherwise it shows a stand-in, the library parts at the dimensions the rules
derive, and says so. It never mixes the two silently: a build made for another plan is
still shown (the parts follow the new poses) but listed as stale with every parameter
that differs, and Commit then cuts with the library blocks, not the stale ones.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tka_planner.implant.builds import Build, BuildStore, compare

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "eval" / "demo_cases" / "DEMO_LARGE"
DEMO_BUILD = REPO / "eval" / "patient_builds" / "DEMO_LARGE"
LIBRARY = REPO / "data" / "implants"
PARTS = ("femoral", "plate", "insert", "femoral_block", "tibial_block")


# -- the store -------------------------------------------------------------------


def _write_build(root, name, job, parts=PARTS, build=None):
    from tka_planner.geom import mesh as gm

    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for part in parts:
        gm.write_stl(gm.box(10.0), folder / f"{part}.stl")
    (folder / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (folder / "build.json").write_text(json.dumps(build or {
        "femoral": {"version": 7, "saved": True, "health": []},
        "plate": {"version": 4, "saved": True, "health": []},
        "insert": {"version": 9, "saved": True, "health": []}}), encoding="utf-8")
    return folder


JOB = {"name": "K1", "side": "left",
       "femoral": {"Femur_ML": "70.0000 mm", "C_Distal_cut_surface": "26.5000 mm"},
       "plate": {"Tibial_Plateau_ML": "68.0000 mm", "Tray_Offset_X": "1.2000 mm"},
       "insert": {"Insert_Thickness": "8.1000 mm"},
       "femoral_fit": {"ml_shift_mm": 1.5}}


def test_a_knee_with_no_build_has_none(tmp_path):
    assert BuildStore(tmp_path).get("K1", "left") is None


def test_the_store_finds_a_knee_by_case_and_side(tmp_path):
    _write_build(tmp_path, "K1", JOB)
    build = BuildStore(tmp_path).get("K1", "left")
    assert isinstance(build, Build)
    assert set(build.parts) == set(PARTS)
    assert build.parameters["femoral"]["Femur_ML"] == "70.0000 mm"
    assert build.versions == {"femoral": 7, "plate": 4, "insert": 9}
    assert build.built_at
    assert build.ml_shift_mm == 1.5
    # The other side of the same case id is another knee.
    assert BuildStore(tmp_path).get("K1", "right") is None


def test_a_build_missing_a_part_is_not_a_build(tmp_path):
    _write_build(tmp_path, "K1", JOB, parts=("femoral", "plate"))
    assert BuildStore(tmp_path).get("K1", "left") is None


def test_a_side_qualified_folder_wins(tmp_path):
    _write_build(tmp_path, "K1", dict(JOB, side="right"))
    _write_build(tmp_path, "K1_left", JOB)
    assert BuildStore(tmp_path).get("K1", "left").folder.name == "K1_left"
    assert BuildStore(tmp_path).get("K1", "right").folder.name == "K1"


def test_compare_lists_every_parameter_beyond_a_hundredth(tmp_path):
    _write_build(tmp_path, "K1", JOB)
    build = BuildStore(tmp_path).get("K1", "left")
    derived = {"femoral": {"Femur_ML": "70.0050 mm", "C_Distal_cut_surface": "27.0000 mm"},
               "plate": {"Tibial_Plateau_ML": "68.0000 mm", "Tray_Offset_X": "1.2000 mm"},
               "insert": {"Insert_Thickness": "8.6000 mm"}}
    stale = compare(build, derived)
    assert [(part, name) for part, name, *_ in stale] == [
        ("femoral", "C_Distal_cut_surface"), ("insert", "Insert_Thickness")]
    assert stale[0][2:] == (26.5, 27.0)
    assert compare(build, {k: dict(v) for k, v in build.parameters.items()}) == []


# -- in a session, with no build --------------------------------------------------


def test_with_no_build_the_implant_is_a_labelled_stand_in(session):
    state = session.implant_state()
    assert state["source"] == "stand_in"
    assert state["label"] == "stand-in, not built"
    assert any(f["code"] == "IMPLANT_STAND_IN" for f in session.findings())


def test_the_envelope_says_what_the_implant_is(session):
    from tka_planner.server.api import Api
    from tka_planner.server.sessions import SessionEntry

    class Manager:
        store = None

    entry = SessionEntry(session_id="s1", session=session)
    out = Api(Manager()).snapshot(entry)
    assert out["implant"]["source"] == "stand_in"
    assert out["implant"]["label"] == "stand-in, not built"


# -- on a real knee with its Fusion-built parts --------------------------------------

needs_demo = pytest.mark.skipif(
    not (DEMO / "FD.stl").exists() or not (DEMO_BUILD / "femoral.stl").exists()
    or not (LIBRARY / "femoral_box.json").exists(),
    reason="needs the public demo knee, its built parts and the catalogue")


@pytest.fixture(scope="module")
def demo_builds(tmp_path_factory):
    """The demo knee's built parts, with a job written for the plan the session opens
    on, so they match it (the published job was built on an older plan)."""
    from tka_planner.session import PlanningSession

    root = tmp_path_factory.mktemp("builds")
    shutil.copytree(DEMO_BUILD, root / "DEMO_LARGE")
    probe = PlanningSession.open(DEMO, side="right", library=LIBRARY)
    probe.build()
    job = probe.derived
    assert job is not None, probe.derived_reason
    (root / "DEMO_LARGE" / "job.json").write_text(json.dumps(job), encoding="utf-8")
    return root


@pytest.fixture
def demo(demo_builds):
    from tka_planner.session import PlanningSession

    session = PlanningSession.open(DEMO, side="right", library=LIBRARY,
                                   builds=BuildStore(demo_builds))
    session.build()
    return session


@needs_demo
def test_a_matching_build_is_shown_at_the_derived_poses(demo):
    from tka_planner.core.meshio import read_stl

    state = demo.implant_state()
    assert state["source"] == "built" and state["label"] == "built"
    assert state["stale"] == []
    node = demo.scene.nodes["femoral_component"]
    assert node.tags.get("built")
    built = read_stl(DEMO_BUILD / "femoral.stl")
    assert len(demo.scene.meshes[node.mesh_id].vertices) == len(built.vertices)
    # The built part at the plan's pose, slid along its ML axis by the build's slide.
    F = np.asarray(demo.plan.components["femoral_component"], float)
    s = demo.derived["femoral_fit"]["ml_shift_mm"]
    expected = F.copy()
    expected[:3, 3] += F[:3, 1] * s
    assert node.rest == pytest.approx(expected, abs=1e-9)
    # Its blocks are the knee's own.
    assert demo.scene.nodes["femoral_cutting_block"].tags.get("built")
    assert demo.scene.nodes["tibial_cutting_block"].tags.get("built")


@needs_demo
def test_the_fit_is_measured_on_the_built_parts(demo):
    demo.refresh_fit()
    assert demo.fit["implant_mode"] == "built"
    assert demo.fit["tibial_component"]["coverage_fraction"] > 0.5


@needs_demo
def test_a_plan_moved_after_the_build_keeps_the_parts_but_flags_them(demo):
    from tka_planner.server.sessions import SessionEntry, SessionManager

    demo.replan(tibial_slope_delta_deg=3.0, femoral_rotation_delta_deg=2.0)
    state = demo.implant_state()
    assert state["source"] == "built"
    assert state["stale"], "the parameters moved with the plan"
    names = {row["name"] for row in state["stale"]}
    assert names & {"Tibial_Plateau_ML", "Tibial_Plateau_AP", "Insert_Thickness",
                    "Tray_Offset_X", "Tray_Offset_Y", "Femur_ML"}
    assert demo.scene.nodes["femoral_component"].tags.get("built")
    # Commit does not cut with blocks made for another plan.
    assert not demo.scene.nodes["femoral_cutting_block"].tags.get("built")
    assert not demo.scene.nodes["tibial_cutting_block"].tags.get("built")
    codes = {f["code"]: f for f in demo.findings()}
    assert codes["BUILD_STALE"]["severity"] == "warning"
    assert str(len(state["stale"])) in codes["BUILD_STALE"]["message"]
    manager = SessionManager()
    entry = SessionEntry(session_id="s", session=demo)
    entry.remember_pristine()
    outcome = manager.commit(entry)
    assert set(outcome.computed) == {"Femur", "Tibia"}


@needs_demo
def test_the_stand_in_can_be_chosen_over_the_built_parts(demo):
    demo.replan(implant_display="stand_in")
    state = demo.implant_state()
    assert state["source"] == "stand_in"
    assert state["label"] == "stand-in, not built"
    assert state["built_available"]
    for name in ("femoral_component", "tibial_component", "tibial_insert",
                 "femoral_cutting_block", "tibial_cutting_block"):
        assert not demo.scene.nodes[name].tags.get("built"), name
    demo.replan(implant_display="built")
    assert demo.scene.nodes["femoral_component"].tags.get("built")


@needs_demo
def test_the_stand_in_takes_the_derived_dimensions(demo_builds):
    """Without a build the library parts are stretched to the rules' Femur_ML - 1 and the
    tray rule's ML and AP, not to the section the planner used to measure."""
    from tka_planner.pipeline import library_dimensions
    from tka_planner.scene.build import resolve_component_meshes
    from tka_planner.session import PlanningSession

    session = PlanningSession.open(DEMO, side="right", library=LIBRARY)
    session.build()
    assert session.implant_state()["source"] == "stand_in"
    job = session.derived
    specs = resolve_component_meshes(LIBRARY, chart=session.chart, sizing=session.sizing,
                                     side="right")
    scales = session.component_scales["scales"]
    ml = float(job["femoral"]["Femur_ML"].split()[0]) - 1.0
    lib_f = library_dimensions(specs["femoral_component"], "femoral")
    assert scales["femoral"][1] * lib_f["ml"] == pytest.approx(ml, abs=1e-6)
    lib_t = library_dimensions(specs["tibial_component"], "tibial")
    assert scales["tibial"][0] * lib_t["ml"] == pytest.approx(
        float(job["plate"]["Tibial_Plateau_ML"].split()[0]), abs=1e-6)
    assert scales["tibial"][1] * lib_t["ap"] == pytest.approx(
        float(job["plate"]["Tibial_Plateau_AP"].split()[0]), abs=1e-6)


# -- the server's builds folder ------------------------------------------------------


def test_the_server_takes_a_builds_folder_and_defaults_beside_the_store(tmp_path):
    from tka_planner.server.__main__ import build_parser
    from tka_planner.server.sessions import SessionManager

    args = build_parser().parse_args(["--builds", str(tmp_path / "b")])
    assert args.builds == str(tmp_path / "b")
    assert SessionManager(builds=tmp_path / "b").builds.root == tmp_path / "b"
    assert SessionManager(store=tmp_path / "s").builds.root == tmp_path / "s" / "builds"
    assert SessionManager().builds is None


def test_tka_serve_passes_the_builds_folder_on(tmp_path, monkeypatch):
    import tka_planner.server.__main__ as server_main
    from tka_planner import cli

    seen = {}
    monkeypatch.setattr(server_main, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    cli.main(["serve", "--builds", str(tmp_path), "--no-browser"])
    assert "--builds" in seen["argv"] and str(tmp_path) in seen["argv"]
