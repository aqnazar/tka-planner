"""What every paper experiment shares: the case list, the output folder and the run record.

Each experiment is one module under :mod:`paper.experiments` with a ``run(ctx)``
function, registered with :func:`experiment`. ``python -m paper run E1`` runs it and
writes its tables (CSV plus a Markdown copy for reading) and figures under
``paper/out/<id>/``, next to a ``run.json`` that records the commit, the versions and
the machine. Every number in the paper comes out of one of these folders.

Patient data never enters the repository, so neither the cohort nor the results do. The
cohort is read from ``TKA_COHORT_DIR`` (default: ``Patient Dataset`` at the top of the
checkout), and results go to ``TKA_PAPER_OUT`` (default: ``paper/out``, gitignored).
Knees are named by their case id and side only (``P009L``), taken from the segmentation
file names, never from folder names.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
LIBRARY_DEFAULT = REPO / "data" / "implants"
REFERENCE_DIR = REPO / "data" / "cases" / "REF_KJ2"
SEED = 20261014

_SEGMENT = re.compile(r"(P\d{3})_([LR])", re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    """One knee: its id, side and the two bone meshes."""

    case_id: str          # e.g. "P009L"
    side: str             # "left" or "right"
    femur: Path
    tibia: Path
    reference: bool = False   # the full-length reference knee, not a cohort member


def cohort_root() -> Path:
    return Path(os.environ.get("TKA_COHORT_DIR", REPO / "Patient Dataset"))


def library_root() -> Path | None:
    path = Path(os.environ.get("TKA_IMPLANT_LIBRARY", LIBRARY_DEFAULT))
    return path if (path / "SizeChart.csv").is_file() else None


def _bone(folder: Path, prefix: str) -> Path | None:
    return next((p for p in sorted(folder.glob("*.stl"))
                 if p.name.upper().startswith(prefix)), None)


def cohort_cases(*, include_reference: bool = False) -> list[Case]:
    """Every knee in the cohort folder, sorted by id.

    A knee is a folder holding an ``FD*`` femur and a ``TD*`` tibia, plus the
    segmentation exports whose names carry the case id and side. Folders without them
    are skipped, so stray outputs never become cases.
    """
    cases = []
    root = cohort_root()
    if root.is_dir():
        for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
            femur, tibia = _bone(folder, "FD"), _bone(folder, "TD")
            if femur is None or tibia is None:
                continue
            ids = {m.group(1).upper() + m.group(2).upper()
                   for p in folder.glob("*.stl") if (m := _SEGMENT.search(p.name))}
            if len(ids) != 1:
                continue
            case_id = ids.pop()
            side = "left" if case_id.endswith("L") else "right"
            cases.append(Case(case_id, side, femur, tibia))
    cases.sort(key=lambda c: c.case_id)
    if include_reference and (REFERENCE_DIR / "FD1Right.stl").is_file():
        cases.insert(0, Case("REF_KJ2", "right", REFERENCE_DIR / "FD1Right.stl",
                             REFERENCE_DIR / "TD1Right.stl", reference=True))
    return cases


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _version(module: str) -> str:
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:
        return ""


def run_record() -> dict:
    """The provenance of a result: commit, versions, machine, time."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "tka_planner", "paper")),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "manifold3d": _version("manifold3d"),
        "tka_planner": _version("tka-planner"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": SEED,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


@dataclass
class Context:
    """Handed to every experiment: where to write, which cases, and the seed."""

    experiment_id: str
    out: Path
    cases: list[Case]
    seed: int = SEED
    quick: bool = False      # a smoke run: fewer cases, fewer samples
    record: dict = field(default_factory=dict)

    def path(self, name: str) -> Path:
        return self.out / name

    def table(self, name: str, rows: list[dict], columns: list[str] | None = None,
              *, digits: int = 3) -> Path:
        """Write rows as ``<name>.csv`` and ``<name>.md``; return the CSV path."""
        return write_table(self.out / name, rows, columns, digits=digits)

    def note(self, text: str) -> None:
        """Append a line to the experiment's ``summary.md``, and echo it."""
        with open(self.out / "summary.md", "a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
        print(text)


def _cell(value, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_table(base: Path, rows: list[dict], columns: list[str] | None = None,
                *, digits: int = 3) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            columns.extend(k for k in row if k not in columns)
    csv_path = base.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_cell(row.get(k), digits) for k in columns) + " |"
              for row in rows]
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    title: str
    run: Callable[[Context], object]
    needs_cohort: bool


REGISTRY: dict[str, Experiment] = {}


def experiment(experiment_id: str, title: str, *, needs_cohort: bool = True):
    """Register ``run(ctx)`` as a paper experiment."""
    def register(function):
        REGISTRY[experiment_id] = Experiment(experiment_id, title, function, needs_cohort)
        return function
    return register


def out_root() -> Path:
    return Path(os.environ.get("TKA_PAPER_OUT", REPO / "paper" / "out"))


def run_experiment(experiment_id: str, *, quick: bool = False,
                   include_reference: bool = False) -> Path:
    exp = REGISTRY[experiment_id]
    out = out_root() / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(f"# {exp.experiment_id}: {exp.title}\n\n",
                                    encoding="utf-8")
    cases = cohort_cases(include_reference=include_reference) if exp.needs_cohort else []
    if exp.needs_cohort and not cases:
        raise SystemExit(f"{experiment_id} needs the cohort; none found under "
                         f"{cohort_root()} (set TKA_COHORT_DIR).")
    if quick:
        cases = cases[:2]
    record = run_record()
    ctx = Context(experiment_id, out, cases, quick=quick, record=record)
    t0 = time.perf_counter()
    exp.run(ctx)
    record["seconds"] = round(time.perf_counter() - t0, 2)
    record["cases"] = [c.case_id for c in cases]
    record["quick"] = quick
    (out / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


_MEASURED: dict = {}


def measured(case: Case):
    """``measure_case`` for a knee, once per process; a refusal is returned, not raised.

    Returns a :class:`~tka_planner.pipeline.CaseMeasurement`, or the
    ``QualityControlError`` / ``FrameConstructionError`` the planner refused the knee
    with, so every experiment reports the same refusals as E0.
    """
    from tka_planner.core.frames import FrameConstructionError
    from tka_planner.core.qc import QualityControlError
    from tka_planner.pipeline import measure_case

    if case.case_id not in _MEASURED:
        try:
            _MEASURED[case.case_id] = measure_case(case.femur, case.tibia, case.side,
                                                   case_id=case.case_id)
        except (QualityControlError, FrameConstructionError) as refusal:
            _MEASURED[case.case_id] = refusal
    return _MEASURED[case.case_id]
