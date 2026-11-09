"""Building the parts in Fusion from the planner (one app, B6), with Fusion mocked.

Fusion runs the build scripts through its MCP add-in on 127.0.0.1:27182. None of these
tests needs Fusion: the transport is replaced, and what is checked is what the planner
does around it -- the clear error when Fusion is not there, the save gate, and where the
parts land.
"""

import json
import sys
import urllib.error
from pathlib import Path

import pytest

from tka_planner.fusion import FusionConfig, FusionUnavailable, build_case
from tka_planner.fusion import transport
from tka_planner.fusion.gate import save_gate

REPO = Path(__file__).resolve().parents[1]

JOB = {"name": "K1", "side": "right",
       "femoral": {"Femur_ML": "70.0000 mm"}, "plate": {"Tibial_Plateau_ML": "68.0000 mm"},
       "insert": {"Insert_Thickness": "8.1000 mm"}, "femoral_fit": {"ml_shift_mm": 0.5}}
SOURCES = {k: f"lineage-{k}" for k in ("femoral", "plate", "insert", "block", "tblock")}


def _refused(*a, **k):
    raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))


# -- the transport -------------------------------------------------------------------


def test_fusion_not_running_is_a_clear_error_naming_the_address(monkeypatch):
    monkeypatch.setattr(transport.urllib.request, "urlopen", _refused)
    with pytest.raises(FusionUnavailable) as caught:
        transport.call_fusion("print(1)", read_only=True)
    assert "Fusion is not reachable on 127.0.0.1:27182" in str(caught.value)


def test_a_closed_port_is_unavailable_not_a_crash():
    import socket

    with socket.socket() as s:            # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(FusionUnavailable) as caught:
        transport.ping(f"http://127.0.0.1:{port}/mcp", timeout=2)
    assert f"127.0.0.1:{port}" in str(caught.value)


def test_a_reply_is_unpacked_from_the_event_stream(monkeypatch):
    replies = iter([
        ("sid-1", ""),
        ("sid-1", ""),
        ("sid-1", 'data: {"result": {"content": [{"text": "{\\"message\\": \\"done\\"}"}]}}'),
    ])

    class Reply:
        def __init__(self, sid, body):
            self.headers = {"Mcp-Session-Id": sid}
            self.body = body.encode()

        def read(self):
            return self.body

    monkeypatch.setattr(transport.urllib.request, "urlopen",
                        lambda request, timeout=None: Reply(*next(replies)))
    assert transport.call_fusion("x", read_only=False) == "done"


# -- the configuration ---------------------------------------------------------------


def test_the_fusion_files_come_from_a_local_file(tmp_path):
    path = tmp_path / "local.json"
    path.write_text(json.dumps({"mcp_url": "http://127.0.0.1:27182/mcp", "sources": SOURCES}))
    config = FusionConfig.load(path)
    assert config.sources == SOURCES and config.address == "127.0.0.1:27182"
    with pytest.raises(FileNotFoundError, match="local.example.json"):
        FusionConfig.load(tmp_path / "missing.json").require()


def test_the_example_config_is_committed_and_holds_no_file_identifiers():
    example = json.loads((REPO / "fusion" / "local.example.json").read_text(encoding="utf-8"))
    assert set(example["sources"]) == set(SOURCES)
    assert not any("urn:adsk" in v for v in example["sources"].values())
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "fusion/local.json" in ignored


def test_no_lineage_identifier_is_in_the_package():
    for path in (REPO / "tka_planner" / "fusion").glob("*.py"):
        assert "urn:adsk" not in path.read_text(encoding="utf-8"), path.name


# -- the save gate --------------------------------------------------------------------


def _bodies(vol, size=(60.0, 70.0, 40.0)):
    return {"Main Body": {"vol": vol, "bb": [0, 0, 0, *size]}}


PREV = {"femoral": {"saved": True, "version": 3, "bodies": _bodies(20000.0)}}


def test_a_volume_jump_with_the_parameters_unchanged_is_refused():
    gate = save_gate("femoral", PREV, JOB, JOB, _bodies(20600.0), "Main Body")
    assert not gate["ok"] and gate["checked"]
    assert "3.0%" in gate["reason"]


def test_a_bounding_box_change_over_a_millimetre_is_refused():
    gate = save_gate("femoral", PREV, JOB, JOB, _bodies(20000.0, (61.5, 70.0, 40.0)),
                     "Main Body")
    assert not gate["ok"]
    assert "1.5 mm" in gate["reason"]


def test_small_changes_with_the_parameters_unchanged_pass():
    gate = save_gate("femoral", PREV, JOB, JOB, _bodies(20200.0, (60.5, 70.0, 40.0)),
                     "Main Body")
    assert gate["ok"] and gate["checked"]


def test_changed_parameters_explain_any_change():
    now = dict(JOB, femoral={"Femur_ML": "75.0000 mm"})
    gate = save_gate("femoral", PREV, JOB, now, _bodies(26000.0), "Main Body")
    assert gate["ok"] and not gate["checked"]


def test_the_insert_is_explained_by_its_femoral_and_plate_too():
    prev = {"insert": {"saved": True, "bodies": {"Body": {"vol": 5000.0, "bb": [0] * 3 + [70, 45, 10]}}}}
    now = dict(JOB, plate={"Tibial_Plateau_ML": "70.0000 mm"})
    assert save_gate("insert", prev, JOB, now, {"Body": {"vol": 5600.0, "bb": [0] * 3 + [72, 45, 10]}},
                     "*")["ok"]
    assert not save_gate("insert", prev, JOB, JOB, {"Body": {"vol": 5600.0, "bb": [0] * 3 + [72, 45, 10]}},
                         "*")["ok"]


def test_a_first_build_has_nothing_to_compare_with():
    assert save_gate("plate", {}, {}, JOB, _bodies(1.0), "Body1")["ok"]


# -- the build, with Fusion mocked ------------------------------------------------------


def test_an_unreachable_fusion_builds_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.urllib.request, "urlopen", _refused)
    config = FusionConfig(sources=SOURCES)
    with pytest.raises(FusionUnavailable):
        build_case(JOB, tmp_path / "K1", config, log=lambda line: None)
    assert not (tmp_path / "K1" / "job.json").exists()


def _fake_fusion(saved=("femoral", "plate", "insert", "block", "tblock"), seen=None):
    """Stands in for Fusion: checks the script would run, writes the parts and a report
    into the folder the script names, as the real build does."""
    from tka_planner.geom import mesh as gm

    def fake(script, read_only, url=None, timeout=None):
        compile(script, "fusion-build", "exec")      # it is Python Fusion can run
        if seen is not None:
            seen.append(script)
        out = script.split("OUT = ", 1)[1].splitlines()[0]
        out = Path(eval(out))                         # noqa: S307 - our own repr
        report = {"name": "K1"}
        files = {"femoral": "femoral", "plate": "plate", "insert": "insert",
                 "block": "femoral_block", "tblock": "tibial_block"}
        for part, stem in files.items():
            if part in saved:
                gm.write_stl(gm.box(10.0), out / f"{stem}.stl")
                report[part] = {"saved": True, "version": 5, "health": [],
                                "bodies": _bodies(1000.0)}
            else:
                report[part] = {"saved": False, "refused": True,
                                "gate": {"ok": False, "reason": "volume +9.0%"}}
        (out / "build.json").write_text(json.dumps(report))
        return json.dumps({"name": "K1", "error": None})

    return fake


def test_a_build_writes_the_parts_where_the_planner_reads_them(tmp_path, monkeypatch):
    from tka_planner.implant.builds import BuildStore

    seen, lines = [], []
    monkeypatch.setattr(transport, "ping", lambda url, timeout=5: None)
    monkeypatch.setattr(transport, "call_fusion", _fake_fusion(seen=seen))
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lines.append)
    assert result.ok and set(result.saved) >= {"femoral", "plate", "insert"}
    build = BuildStore(tmp_path).get("K1", "right")
    assert build is not None and build.parameters["insert"] == JOB["insert"]
    # The script carries the gate and the sources, and no identifiers of its own.
    assert "def save_gate" in seen[0] and "lineage-femoral" in seen[0]
    assert any("Fusion" in line for line in lines)


def test_a_refused_part_leaves_the_last_good_build_in_place(tmp_path, monkeypatch):
    from tka_planner.implant.builds import BuildStore

    monkeypatch.setattr(transport, "ping", lambda url, timeout=5: None)
    monkeypatch.setattr(transport, "call_fusion", _fake_fusion())
    build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    again = dict(JOB, insert={"Insert_Thickness": "9.0000 mm"})
    monkeypatch.setattr(transport, "call_fusion", _fake_fusion(saved=("femoral", "block", "tblock")))
    result = build_case(again, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    assert not result.ok
    assert "plate" in result.refused and "insert" in result.refused
    build = BuildStore(tmp_path).get("K1", "right")
    assert build.parameters["insert"] == JOB["insert"]      # the last good build


def test_the_script_blocks_are_eval_s_own():
    sys.path.insert(0, str(REPO / "eval"))
    try:
        import femoral_block
        import implant_pipeline
        import insert_edit
        import insert_finish
        import tibial_block
    except Exception:  # noqa: BLE001 - eval/ retired or its private files absent
        pytest.skip("eval/ scripts not importable")
    finally:
        sys.path.remove(str(REPO / "eval"))
    from tka_planner.fusion import scripts

    assert scripts.OUTLINE == insert_edit.OUTLINE
    assert scripts.FINISH == insert_finish.FINISH
    assert scripts.BLOCK == femoral_block.BLOCK
    assert scripts.TBLOCK == tibial_block.TBLOCK
    assert scripts.BUILD_EVAL == implant_pipeline.BUILD


# -- the insert's Condylar Profile ------------------------------------------------------


def _finish_functions():
    from tka_planner.fusion import scripts

    namespace = {}
    exec(scripts.FINISH, namespace)        # noqa: S102 - definitions only; adsk is used in Fusion
    return namespace


def test_the_design_s_own_condylar_profile_is_told_from_the_remade_box():
    design_profile = _finish_functions()["design_profile"]
    # The source design's sketch: sides tangent to the condyle curve, top perpendicular.
    assert design_profile(["TangentConstraint", "TangentConstraint", "PerpendicularConstraint"])
    # remake_profile's box: ends coincident with the curve, sides vertical, top horizontal.
    assert not design_profile(["CoincidentConstraint", "CoincidentConstraint", "VerticalConstraint",
                               "VerticalConstraint", "HorizontalConstraint"])
    # A box that lost a side's constraint, or no constraints at all, is remade too.
    assert design_profile(["CoincidentConstraint", "VerticalConstraint", "HorizontalConstraint"])
    assert design_profile([])


def test_the_design_s_profile_is_remade_before_the_femoral_is_replaced():
    from tka_planner.fusion.build import BUILD

    repoint = BUILD[BUILD.index("def repoint(des, doc):"):BUILD.index("def block_repoint(")]
    assert repoint.index("profile_before_replace(des)") < repoint.index("o.replace(")
    assert repoint.index("profile_before_replace(des)") < repoint.index("tray_offset(des)")


# -- a stale knee document is retried once from a fresh copy -------------------------------


def _staged_fusion(attempts, calls):
    """Stands in for Fusion over several calls: a build script gets the next report of
    ``attempts`` (part -> "saved" / "failed" / "refused", and "copied"), a rename script
    its renames."""
    from tka_planner.geom import mesh as gm

    attempts = iter(attempts)

    def value(script, name):
        return eval(script.split(name + " = ", 1)[1].splitlines()[0])      # noqa: S307 - our repr

    def fake(script, read_only, url=None, timeout=None):
        compile(script, "fusion-script", "exec")
        if "SUFFIX = " in script:
            parts, suffix = value(script, "PARTS"), value(script, "SUFFIX")
            calls.append(("rename", parts, suffix))
            names = {"plate": "Tibial Plate", "insert": "Tibial Insert New"}
            return json.dumps({"renamed": {p: names[p] + suffix for p in parts}})
        out = Path(value(script, "OUT"))
        job_path = eval(script.split("JOB = json.load(open(", 1)[1].split(", encoding")[0])  # noqa: S307
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        calls.append(("build", job.get("only")))
        spec = next(attempts)
        report = {"name": "K1", "copied": spec["copied"]}
        files = {"femoral": "femoral", "plate": "plate", "insert": "insert",
                 "block": "femoral_block", "tblock": "tibial_block"}
        for part, stem in files.items():
            if part not in spec:
                continue
            if spec[part] == "saved":
                gm.write_stl(gm.box(10.0), out / f"{stem}.stl")
                report[part] = {"saved": True, "version": 5, "health": [], "bodies": _bodies(1000.0)}
            elif spec[part] == "refused":
                report[part] = {"saved": False, "refused": True,
                                "gate": {"ok": False, "reason": "volume +9.0%"}}
            else:
                report[part] = {"saved": False,
                                "health": [["Condylar Profile", 1, "Failed to solve"],
                                           ["Anterior Sweep", 1, "Profile 1 missing"]]}
        (out / "build.json").write_text(json.dumps(report))
        return json.dumps({"name": "K1", "error": None})

    return fake


ALL_REUSED = {p: False for p in ("femoral", "plate", "insert", "block", "tblock")}


def _mock(monkeypatch, attempts, calls):
    monkeypatch.setattr(transport, "ping", lambda url, timeout=5: None)
    monkeypatch.setattr(transport, "call_fusion", _staged_fusion(attempts, calls))


def test_an_insert_failing_on_a_reused_document_is_built_again_from_a_fresh_copy(tmp_path,
                                                                                   monkeypatch):
    from tka_planner.implant.builds import BuildStore

    calls, lines = [], []
    first = dict(femoral="saved", plate="saved", insert="failed", block="saved", tblock="saved",
                 copied=ALL_REUSED)
    second = dict(insert="saved", copied=dict(ALL_REUSED, insert=True))
    _mock(monkeypatch, [first, second], calls)
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lines.append)
    assert [c[0] for c in calls] == ["build", "rename", "build"]
    assert calls[1][1] == ["insert"] and calls[1][2].startswith(" (stale 20")
    assert calls[2] == ("build", ["insert"])            # only what was not saved
    assert result.ok and "insert" in result.saved
    retried = result.report["retried"]
    assert retried["parts"] == ["insert"] and "Profile 1 missing" in retried["first"]["insert"]
    assert retried["renamed"]["insert"].startswith("Tibial Insert New (stale ")
    assert result.report["copied"]["insert"] is True
    assert BuildStore(tmp_path).get("K1", "right") is not None
    # The job kept beside the parts is the knee's job, not the retry's.
    kept = json.loads((tmp_path / "K1" / "job.json").read_text(encoding="utf-8"))
    assert "only" not in kept
    assert any("fresh copy" in line for line in lines)


def test_a_plate_failing_on_a_reused_document_rebuilds_what_stands_on_it(tmp_path, monkeypatch):
    calls = []
    # The plate failed, so the insert and the tibial block were never tried.
    first = dict(femoral="saved", plate="failed", block="saved", copied=ALL_REUSED)
    second = dict(plate="saved", insert="saved", tblock="saved",
                  copied=dict(ALL_REUSED, plate=True))
    _mock(monkeypatch, [first, second], calls)
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    assert calls[1][1] == ["plate"]                     # the insert is not renamed: never tried
    assert calls[2] == ("build", ["plate", "insert", "tblock"])
    assert result.ok and set(result.saved) == {"femoral", "plate", "insert", "block", "tblock"}


def test_a_fresh_copy_that_fails_is_not_retried(tmp_path, monkeypatch):
    calls = []
    first = dict(femoral="saved", plate="saved", insert="failed", block="saved", tblock="saved",
                 copied=dict(ALL_REUSED, insert=True))
    _mock(monkeypatch, [first], calls)
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    assert [c[0] for c in calls] == ["build"]
    assert not result.ok and "insert" in result.refused and "retried" not in result.report


def test_a_gate_refusal_is_not_taken_for_a_stale_document(tmp_path, monkeypatch):
    calls = []
    first = dict(femoral="saved", plate="saved", insert="refused", block="saved", tblock="saved",
                 copied=ALL_REUSED)
    _mock(monkeypatch, [first], calls)
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    assert [c[0] for c in calls] == ["build"] and not result.ok


def test_a_retry_that_fails_again_keeps_the_last_good_build(tmp_path, monkeypatch):
    calls = []
    first = dict(femoral="saved", plate="saved", insert="failed", block="saved", tblock="saved",
                 copied=ALL_REUSED)
    second = dict(insert="failed", copied=dict(ALL_REUSED, insert=True))
    _mock(monkeypatch, [first, second], calls)
    result = build_case(JOB, tmp_path / "K1", FusionConfig(sources=SOURCES), log=lambda line: None)
    assert [c[0] for c in calls] == ["build", "rename", "build"]      # once, not again
    assert not result.ok and not (tmp_path / "K1" / "job.json").exists()
    assert result.report["retried"]["parts"] == ["insert"]
