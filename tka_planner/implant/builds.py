"""The parts Fusion built for a knee, and whether they still match its plan.

A builds folder holds one folder per knee, named by its case id (``P009L``,
``DEMO_LARGE``) or by case id and side (``K1_left``), as the Fusion build writes it:
``femoral.stl`` (femoral CAD frame), ``plate.stl`` and ``insert.stl`` (tray frame, the
plate's Tray_Offset applied), the two cutting blocks, ``job.json`` (the parameters it was
built from) and ``build.json`` (each part's version, health and save). Patient builds
never enter the repository; the folder is the server's ``--builds``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Build", "BuildStore", "compare", "PARTS", "REQUIRED", "TOLERANCE"]

PARTS = ("femoral", "plate", "insert", "femoral_block", "tibial_block")
REQUIRED = ("femoral", "plate", "insert")
PARAMETER_PARTS = ("femoral", "plate", "insert")
# A parameter differs when it moves by more than this (mm or degrees).
TOLERANCE = 0.01


@dataclass
class Build:
    """One knee's built parts: paths, the parameters they were built from, versions."""

    case_id: str
    side: str
    folder: Path
    parts: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    built_at: str = ""
    job: dict = field(default_factory=dict)

    @property
    def ml_shift_mm(self) -> float:
        """The femoral component's slide along its ML axis it was built centred with."""
        return float((self.job.get("femoral_fit") or {}).get("ml_shift_mm", 0.0))

    @property
    def stamp(self) -> tuple:
        """Changes whenever a part file is replaced: what a cache of the meshes keys on."""
        return tuple(sorted((name, path.stat().st_mtime_ns) for name, path in self.parts.items()))


class BuildStore:
    """The knees' builds under one folder."""

    def __init__(self, root):
        self.root = Path(root)

    def folder_for(self, case_id: str, side: str) -> Path:
        """Where a knee's build goes: its existing folder, else one named for the case."""
        for folder in self._candidates(case_id, side):
            if (folder / "job.json").is_file():
                return folder
        return self.root / case_id

    def _candidates(self, case_id, side):
        return (self.root / f"{case_id}_{side}", self.root / case_id)

    def get(self, case_id: str, side: str) -> Build | None:
        """The knee's build, or ``None`` when there is none with all three parts."""
        for folder in self._candidates(case_id, str(side)):
            job_file = folder / "job.json"
            if not job_file.is_file():
                continue
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("side") and str(job["side"]).lower() != str(side).lower():
                continue
            parts = {name: folder / f"{name}.stl" for name in PARTS
                     if (folder / f"{name}.stl").is_file()}
            if not all(name in parts for name in REQUIRED):
                continue
            report = {}
            if (folder / "build.json").is_file():
                try:
                    report = json.loads((folder / "build.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    report = {}
            stamp_file = folder / "build.json" if (folder / "build.json").is_file() else job_file
            return Build(
                case_id=case_id, side=str(side), folder=folder, parts=parts,
                parameters={p: dict(job.get(p) or {}) for p in PARAMETER_PARTS},
                versions={p: (report.get(p) or {}).get("version") for p in PARAMETER_PARTS},
                health={p: (report.get(p) or {}).get("health") or [] for p in PARAMETER_PARTS},
                built_at=time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(stamp_file.stat().st_mtime)),
                job=job,
            )
        return None


def _value(expression):
    """A Fusion expression's number ("81.7500 mm" -> 81.75), or the text when not one."""
    if expression is None:
        return None
    try:
        return float(str(expression).split()[0])
    except (ValueError, IndexError):
        return str(expression)


def compare(build: Build, derived: dict) -> list:
    """Every parameter that differs between the build and a derivation, as
    ``(part, name, built, now)`` rows; numbers beyond :data:`TOLERANCE`."""
    rows = []
    for part in PARAMETER_PARTS:
        built = build.parameters.get(part) or {}
        now = (derived or {}).get(part) or {}
        for name in sorted(set(built) | set(now)):
            a, b = _value(built.get(name)), _value(now.get(name))
            if isinstance(a, float) and isinstance(b, float):
                if abs(a - b) > TOLERANCE + 1e-9:
                    rows.append((part, name, a, b))
            elif a != b:
                rows.append((part, name, a, b))
    return rows
