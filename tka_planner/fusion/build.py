"""Build one knee's parts in Fusion from its job, behind the save gate.

From ``eval/implant_pipeline.build``: the build script copies the five parametric designs
into the knee's Fusion folder (Parametric Work/Patient Builds/<knee>), sets the job's
parameters, rebuilds, checks every feature's health, saves a version and exports the
parts. The package's script is eval's with the save gate (:mod:`.gate`) put in front of
every save.

Where the parts land: the script writes into a staging folder; only when the femoral,
plate and insert were all saved do the parts, the job and the report replace the knee's
folder in the builds store. A refused or failed part leaves the last good build in place,
so the planner never shows parts that disagree with the job beside them.
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import gate as gate_module
from . import scripts, transport
from .config import FusionConfig

__all__ = ["build_case", "BuildResult", "BUILD", "PART_FILES"]

# Script part -> the file it exports.
PART_FILES = {"femoral": "femoral.stl", "plate": "plate.stl", "insert": "insert.stl",
              "block": "femoral_block.stl", "tblock": "tibial_block.stl"}
REQUIRED = ("femoral", "plate", "insert")


def _gated(template: str) -> str:
    """eval's build script with the save gate before each part's save."""
    def once(text, old, new):
        assert text.count(old) == 1, f"build script anchor not found once: {old[:60]!r}"
        return text.replace(old, new)

    gate_source = "\n".join(inspect.getsource(obj) for obj in (gate_module.gate_body,
                                                               gate_module.save_gate))
    constants = (f"GATE_VOLUME_FRACTION = {gate_module.GATE_VOLUME_FRACTION!r}\n"
                 f"GATE_BBOX_MM = {gate_module.GATE_BBOX_MM!r}\n"
                 f"PART_INPUTS = {gate_module.PART_INPUTS!r}\n")
    s = template
    s = once(s, 'OPTIONAL = ("Fillet2",)\n', 'OPTIONAL = ("Fillet2",)\n' + constants + gate_source + r'''
try: PREV_BUILD = json.load(open(OUT + "\\build.json", encoding="utf-8"))
except Exception: PREV_BUILD = {}
try: PREV_JOB = json.load(open(__PREV_JOB__, encoding="utf-8"))
except Exception: PREV_JOB = {}
''')
    s = once(s, "def build_part(app, df, sets, export, msg, prepare=None, strict=True, finish=None, "
                "shift=(0.0, 0.0)):",
             "def build_part(app, df, sets, export, msg, prepare=None, strict=True, finish=None, "
             "shift=(0.0, 0.0), part=None):")
    s = once(s, '''        if errors:
            rec["saved"] = False
''', '''        if not errors and part is not None:
            # The save gate: unchanged parameters must give an unchanged part.
            rec["gate"] = save_gate(part, PREV_BUILD, PREV_JOB, JOB, rec["bodies"],
                                    export[0][0] if export else None)
            if not rec["gate"]["ok"]:
                rec["refused"] = True
                errors = [["save gate", 2, rec["gate"]["reason"]]]
        if errors:
            rec["saved"] = False
''')
    s = once(s, 'prepare=lambda des, doc: sync(des, src_params["femoral"], JOB["femoral"]))',
             'prepare=lambda des, doc: sync(des, src_params["femoral"], JOB["femoral"]), part="femoral")')
    s = once(s, "prepare=plate_prep, shift=tray)", 'prepare=plate_prep, shift=tray, part="plate")')
    s = once(s, "prepare=repoint, strict=False, finish=add_finish, shift=tray)",
             'prepare=repoint, strict=False, finish=add_finish, shift=tray, part="insert")')
    s = once(s, 'finish=lambda des: {"pegs": block_pegs(des, side_body), "state": block_state(des)})',
             'finish=lambda des: {"pegs": block_pegs(des, side_body), "state": block_state(des)}, '
             'part="block")')
    s = once(s, '''finish=lambda des: {"keel": tblock_keel(des), "state": tblock_state(des)},
                shift=tray)''', '''finish=lambda des: {"keel": tblock_keel(des), "state": tblock_state(des)},
                shift=tray, part="tblock")''')
    return s


BUILD = _gated(scripts.BUILD_EVAL)


@dataclass
class BuildResult:
    ok: bool
    folder: Path
    saved: list = field(default_factory=list)
    refused: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "folder": str(self.folder), "saved": self.saved,
                "refused": self.refused, "error": self.error,
                "seconds": round(self.seconds, 1)}


def script_for(job_path: Path, prev_job_path: Path, out: Path, sources: dict) -> str:
    return (BUILD.replace("__OUTLINE__", scripts.OUTLINE).replace("__FINISH__", scripts.FINISH)
            .replace("__BLOCK__", scripts.BLOCK).replace("__TBLOCK__", scripts.TBLOCK)
            .replace("__JOB__", repr(str(job_path)))
            .replace("__PREV_JOB__", repr(str(prev_job_path)))
            .replace("__SOURCES__", repr(dict(sources)))
            .replace("__OUT__", repr(str(out))))


def build_case(job: dict, out, config: FusionConfig, log=print) -> BuildResult:
    """Build ``job``'s parts in Fusion into the knee's builds folder ``out``.

    Raises :class:`~tka_planner.fusion.transport.FusionUnavailable` before touching
    anything when Fusion is not answering, and ``FileNotFoundError`` when the source
    documents are not configured. Otherwise returns what was saved and what the gate or
    Fusion refused; the folder changes only when the three parts were all saved.
    """
    start = time.time()
    out = Path(out)
    transport.ping(config.url)
    config.require()
    stage = out / "staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "job.json").write_text(json.dumps(job), encoding="utf-8")
    if (out / "job.json").is_file():
        shutil.copy2(out / "job.json", stage / "prev_job.json")
    if (out / "build.json").is_file():
        shutil.copy2(out / "build.json", stage / "build.json")
    log(f"Fusion on {config.address}: femoral, plate, insert and both blocks for "
        f"{job.get('name')}")
    reply = transport.call_fusion(script_for(stage / "job.json", stage / "prev_job.json",
                                             stage, config.sources),
                                  read_only=False, url=config.url)
    log(f"Fusion answered: {str(reply).strip().splitlines()[-1] if str(reply).strip() else ''}")
    report = json.loads((stage / "build.json").read_text(encoding="utf-8"))
    saved, refused = [], {}
    for part in PART_FILES:
        rec = report.get(part) or {}
        if rec.get("saved"):
            saved.append(part)
        else:
            refused[part] = ((rec.get("gate") or {}).get("reason") or rec.get("error")
                             or "; ".join(f"{h[0]}: {h[2]}" for h in rec.get("health") or [])
                             or "not saved")
        health = ", ".join(h[0] for h in rec.get("health") or [])
        log(f"{part}: v{rec.get('version')} {'saved' if rec.get('saved') else 'NOT saved'}"
            + (f"; {health}" if health else "")
            + (f"; gate: {rec['gate']['reason']}" if rec.get("gate") else ""))
    ok = not report.get("error") and all(p in saved for p in REQUIRED)
    result = BuildResult(ok=ok, folder=out, saved=saved, refused=refused, report=report,
                         error=report.get("error"))
    if ok:
        for part, name in PART_FILES.items():
            target = out / name
            if part in saved and (stage / name).is_file():
                shutil.copy2(stage / name, target)
            elif target.exists():
                target.unlink()          # a block made for the previous parts
        (out / "job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
        (out / "build.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        shutil.rmtree(stage, ignore_errors=True)
        log("done: the parts are in " + str(out))
    else:
        missing = [p for p in REQUIRED if p not in saved]
        result.error = result.error or (
            f"{', '.join(missing)} not saved; the last good build is kept "
            f"(report in {stage})")
        log("failed: " + result.error)
    result.seconds = time.time() - start
    return result
