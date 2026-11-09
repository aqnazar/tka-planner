"""What every paper experiment shares: the case list, the output folder and the run record.

Each experiment is one module under :mod:`paper.experiments` with a ``run(ctx)``
function, registered with :func:`experiment`. ``python -m paper run E1`` runs it and
writes its tables (CSV plus a Markdown copy for reading) and figures under
``paper/out/<id>/``, next to a ``run.json`` that records the commit, the versions and
the machine. Every number in the paper comes out of one of these folders.

Patient data never enters the repository, so neither the cohort nor the results do. The
cohort is read from ``TKA_COHORT_DIR`` (default: ``Patient Dataset`` at the top of the
checkout; the full-length reference knee from ``TKA_REFERENCE_DIR``), and results go to ``TKA_PAPER_OUT`` (default: ``paper/out``, gitignored).
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


def reference_dir() -> Path:
    return Path(os.environ.get("TKA_REFERENCE_DIR", REPO / "data" / "cases" / "REF_KJ2"))


def library_root() -> Path | None:
    path = Path(os.environ.get("TKA_IMPLANT_LIBRARY", LIBRARY_DEFAULT))
    return path if (path / "SizeChart.csv").is_file() else None


def _bone(folder: Path, prefix: str) -> Path | None:
    return next((p for p in sorted(folder.glob("*.stl"))
                 if p.name.upper().startswith(prefix)), None)


def case_filter() -> set[str] | None:
    """The knees named in ``TKA_PAPER_CASES`` (comma-separated case ids, e.g.
    ``REF_KJ2,P009L``), or ``None`` when it is unset: every knee."""
    names = os.environ.get("TKA_PAPER_CASES", "")
    wanted = {n.strip().upper() for n in names.split(",") if n.strip()}
    return wanted or None


def cohort_cases(*, include_reference: bool = False) -> list[Case]:
    """Every knee in the cohort folder, sorted by id.

    A knee is a folder holding an ``FD*`` femur and a ``TD*`` tibia, plus the
    segmentation exports whose names carry the case id and side. Folders without them
    are skipped, so stray outputs never become cases.

    With ``TKA_PAPER_CASES`` set (``python -m paper run ... --cases``), only the knees it
    names are returned, in the same order; naming ``REF_KJ2`` includes the reference
    knee without ``include_reference``. A name that matches no knee is an error, so a
    typo cannot quietly shrink a run.
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
    wanted = case_filter()
    reference = reference_dir()
    if ((include_reference or (wanted and "REF_KJ2" in wanted))
            and (reference / "FD1Right.stl").is_file()):
        cases.insert(0, Case("REF_KJ2", "right", reference / "FD1Right.stl",
                             reference / "TD1Right.stl", reference=True))
    if wanted is not None:
        missing = wanted - {c.case_id for c in cases}
        if missing:
            raise SystemExit(f"TKA_PAPER_CASES names knees not found: "
                             f"{', '.join(sorted(missing))}.")
        cases = [c for c in cases if c.case_id in wanted]
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


def cpu_name() -> str:
    """The processor's marketing name, which ``platform.processor()`` does not give on
    Windows (it returns the family and stepping) and gives empty on most Linux."""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor()


def memory_gb() -> dict:
    """Installed and currently available RAM, in GB; empty where it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong),
                            ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total, available = status.total, status.available
        elif sys.platform.startswith("linux"):
            info = dict(line.split(":", 1) for line in
                        Path("/proc/meminfo").read_text().splitlines() if ":" in line)
            total = int(info["MemTotal"].split()[0]) * 1024
            available = int(info["MemAvailable"].split()[0]) * 1024
        else:
            return {}
    except Exception:
        return {}
    return {"ram_total_gb": round(total / 2**30, 1),
            "ram_available_gb": round(available / 2**30, 1)}


def cpu_busy_percent(seconds: float = 1.0) -> float | None:
    """How busy the whole machine's CPU is over ``seconds``, from 0 to 100.

    A timing taken while something else is running is not a benchmark. This is
    recorded beside every timing so a reader can tell a quiet machine from a busy one.
    ``None`` where the counters cannot be read.
    """
    def sample():
        if sys.platform == "win32":
            import ctypes
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),
                                                  ctypes.byref(kernel),
                                                  ctypes.byref(user))
            # Kernel time includes idle time.
            return idle.value, kernel.value + user.value
        fields = [int(v) for v in Path("/proc/stat").read_text().split("\n")[0]
                  .split()[1:]]
        return fields[3] + fields[4], sum(fields)

    try:
        idle0, total0 = sample()
        time.sleep(seconds)
        idle1, total1 = sample()
    except Exception:
        return None
    total = total1 - total0
    return round(100.0 * (1.0 - (idle1 - idle0) / total), 1) if total > 0 else None


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
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        **memory_gb(),
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
    needs_cohort: bool | str     # True, False, or "optional"


REGISTRY: dict[str, Experiment] = {}


def experiment(experiment_id: str, title: str, *, needs_cohort: bool | str = True):
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
    # "optional": the cohort is used when present, and the experiment still runs its
    # synthetic part from a plain checkout without it.
    if exp.needs_cohort is True and not cases:
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
