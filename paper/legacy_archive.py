"""The legacy runs archived beside each knee, and which of them Paper 1 used.

Every knee folder in the cohort holds an ``Output/`` tree written by earlier legacy runs
(read only, never written here). It mixes three things, and only one of them is the
Paper 1 configuration:

* runs of the frozen v3 script with ``SIZE_MODE = "ML"`` -- one batch on 2026-04-29,
  18:35 to 18:47, one knee a minute; the archive this module looks for;
* the same script with ``SIZE_MODE = "AP"`` (the ``AP/`` subfolders);
* a later variant with PCA and TEA switched on (2026-05-02, 2026-07-27), whose log
  reports the epicondyles as "Lateral/Medial" where the frozen script says
  "Endpoint A/B", and whose STLs overwrote the frozen run's where the size matched.

Some ``AP/`` folders also hold copies of another knee's run. Each log's ``Patient:``
line names the folder it was run on, so a run is attributed by that line, never by
where the log happens to sit. The STLs of a run are the ones in its folder written
between the log's start (its file name) and its end (its modification time); an STL
written later belongs to a later run and is not used.

Archive locations are reported relative to the knee's ``Output/`` folder (``ML/
tka_log_20260429_183757.txt``); the folder names above it are never printed.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

from .harness import Case, cohort_root
from .legacy_runner import parse_log

_SOURCE = re.compile(r"(Patient_\d+)[\\/](Left|Right)\b")
_STAMP = re.compile(r"tka_log_(\d{8}_\d{6})\.txt$")


def knee_key(case: Case) -> tuple[str, str] | None:
    """``("Patient_00N", "Left")`` for a cohort knee: its folder under the cohort root."""
    try:
        parts = case.femur.parent.relative_to(cohort_root()).parts
    except ValueError:
        return None
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def archived_runs(case: Case, *, slack_s: float = 3.0) -> list[dict]:
    """Every legacy log under the knee's ``Output/``, parsed and attributed."""
    output = case.femur.parent / "Output"
    if not output.is_dir():
        return []
    key = knee_key(case)
    runs = []
    for log in sorted(output.rglob("tka_log_*.txt")):
        m = _STAMP.search(log.name)
        if not m:
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        source = _SOURCE.search(next((line for line in text.splitlines()
                                      if line.startswith("Patient:")), ""))
        start = datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
        end = log.stat().st_mtime
        stls = {p.name: p for p in sorted(log.parent.glob("*.stl"))
                if start - slack_s <= p.stat().st_mtime <= end + slack_s}
        runs.append({
            "id": log.relative_to(output).as_posix(),
            "timestamp": m.group(1),
            "folder": log.parent,
            "this_knee": bool(source) and key is not None
            and (source.group(1), source.group(2)) == key,
            "log": parse_log(text),
            "stls": stls,
        })
    return runs


def paper1_archive(case: Case) -> dict | None:
    """The latest completed frozen-v3 ML run of this knee, or ``None``."""
    candidates = [r for r in archived_runs(case)
                  if r["this_knee"] and r["log"]["frozen_v3_signature"]
                  and r["log"]["size_mode"] == "ML" and r["log"]["complete"]]
    return max(candidates, key=lambda r: r["timestamp"]) if candidates else None
