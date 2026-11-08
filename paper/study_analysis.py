"""The on-screen observer studies, analysed: landmark dispersion (E3) and verdicts (E13).

``python -m paper.study_analysis <study_dir> --out <dir>`` reads what the planner's Pick
and Review modes wrote (:mod:`tka_planner.study`) and writes:

* ``dispersion.csv`` -- per landmark, how far surgeons' picks scatter: within one
  surgeon across rounds (intra), between surgeons (inter), and how far the automatic
  estimate lies from the surgeons' mean (E3).
* ``e5_dispersion.csv`` -- the inter-observer scatter as a table E5 reads with
  :func:`tka_planner.validation.montecarlo.read_dispersion`, the same layout as the
  literature table (``literature_victor2009_inter.csv``) it replaces.
* ``reviews_summary.csv`` -- per knee, side and alignment, the OK / NOT OK counts,
  reasons, severities and time taken (E13).
* ``README.txt`` -- what every column means.

What is read
------------
Picks: every ``picks/<observer>/<case>_<side>_<mode>_r<round>_<UTC>.landmarks.json``
(``tka-landmarks/1.0.0``) with its ``.study.json`` record beside it. Reviews: every line
of ``reviews/<observer>.jsonl``.

An observer is identified by the record's ``observer_key`` (the normalised surgeon
name) when it has one, else by the folder (picks) or file name (reviews) it was saved
under, so the same surgeon saved under two spellings of one name is one observer.

"Save progress" writes the set so far and "Finish and save" writes it again, so one
observer can have several files for one knee, side, mode and round. Each save holds
everything placed up to then, so only the latest (by its UTC stamp) is used; the others
are counted as superseded. The same holds for a verdict saved twice in one round.

Definitions
-----------
**Coordinates.** Ideally picks would be compared in the bone's own anatomical frame.
The pick files hold scan millimetres, and every observer picks on the same mesh of the
same knee, so positions are compared in scan millimetres directly: a distance is the
same in either frame, and only the per-axis SDs (``*_sd_x/y/z_mm``) are along the
scanner's axes rather than the bone's.

**Which sets.** Only ``blank`` sets (no automatic markers shown) count by default, since
an ``adjust`` set starts on the estimate and is anchored on it; ``include_adjust`` (CLI
``--include-adjust``) adds them, pooled with the same observer's blank picks. A landmark
the observer skipped (``not_picked``) is excluded and counted in ``n_skipped``.

**Intra-observer SD.** For one observer on one knee with ``R >= 2`` picks of a landmark,
the 3D variance about their own mean, ``s^2 = sum |p_r - m|^2 / (R - 1)`` (ddof = 1).
These are pooled over observers and knees as a pooled variance (weighted by ``R - 1``,
which is the root mean of the variances when everyone has the same number of rounds),
and ``intra_sd_mm`` is its square root: the RMS 3D distance of a pick from that
observer's own mean. ``intra_sd_x/y/z_mm`` are the same, axis by axis; their squares sum
to ``intra_sd_mm^2``.

**Inter-observer SD.** On one knee, each observer's mean position ``m_o`` (over their
rounds), the grand mean ``g`` (the mean of the observer means, so every observer counts
once), and ``s^2 = sum |m_o - g|^2 / (n_obs - 1)``. Pooled over knees the same way
(weighted by ``n_obs - 1``); ``inter_sd_mm`` is the RMS 3D distance of one observer's
mean from the grand mean. It needs two observers on a knee.

**Automatic minus mean.** ``auto_minus_mean_mm`` is the distance between the planner's
automatic estimate and the observers' grand mean ``g``, averaged over knees. The
estimate's position comes from the ``auto`` argument (CLI ``--auto``: landmark files of
the automatic estimates) or from an ``auto_mm`` entry in a pick record. Failing that,
it is recovered from the distance each pick was recorded to have moved from the
estimate (``moved_from_auto_mm``, which Pick stores for every pick it has an estimate
for): with weights ``w`` summing to one and ``g`` their weighted mean,
``sum w |p - a|^2 = sum w |p - g|^2 + |g - a|^2`` exactly, so
``|g - a| = sqrt(mean_w d^2 - mean_w |p - g|^2)``. The distances are stored to 0.001
mm, so a recovered value under about 0.1 mm is not resolved. ``auto_source`` says which
(``position``, ``distances``, ``mixed``); with neither it is NaN.

**For E5.** E5 draws a landmark's error as an isotropic Gaussian with ``sd_mm`` per axis
and, for a point on the surface, removes the component along the surface normal, so
the drawn 3D scatter is ``sd_mm * sqrt(2)``. Every pickable landmark is a surface point,
so the measured RMS 3D scatter ``r`` is written as ``sd_mm = r / sqrt(2)`` (the
conversion :mod:`tka_planner.validation.montecarlo` gives for a surface study). The
table is isotropic: E5's covariance columns are in the bone frame, which the pick files
do not carry. The inter-observer SD is written by default, as the literature table it
replaces is inter-observer (Victor 2009); ``which="intra"`` writes the intra SD.
A landmark with no value (one observer only) has no row, which E5 reads as held fixed,
and is named in the table's comments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tka_planner.core.landmarks import read_landmark_set
from tka_planner.study import REVIEW_REASONS, REVIEW_SEVERITIES

__all__ = [
    "PickSet",
    "load_picks",
    "latest_sets",
    "landmark_dispersion",
    "write_dispersion_csv",
    "write_e5_dispersion_csv",
    "load_reviews",
    "review_summary",
    "write_review_summary_csv",
    "main",
]

_AXES = ("x", "y", "z")


# ----------------------------------------------------------------------
# Picks
# ----------------------------------------------------------------------


@dataclass
class PickSet:
    """One saved pick set: one observer, knee, side, mode and round, at one moment."""

    path: Path
    observer: str            # the name as saved
    observer_key: str        # who it is: the record's key, else the folder name
    case_id: str
    side: str
    mode: str
    round: int
    saved_utc: str
    complete: bool
    positions: dict = field(default_factory=dict)        # landmark id -> (3,) mm
    skipped: dict = field(default_factory=dict)          # landmark id -> reason
    moved_from_auto: dict = field(default_factory=dict)  # landmark id -> mm
    auto: dict = field(default_factory=dict)             # landmark id -> (3,) mm

    @property
    def key(self) -> tuple:
        return (self.observer_key, self.case_id, self.side, self.mode, self.round)


def load_picks(study_dir) -> list[PickSet]:
    """Every pick set under ``<study_dir>/picks``, superseded saves included."""
    sets = []
    for path in sorted(Path(study_dir).glob("picks/*/*.landmarks.json")):
        landmarks = read_landmark_set(path)
        source = dict(landmarks.source or {})
        record_path = path.with_name(path.name[: -len(".landmarks.json")] + ".study.json")
        record = (json.loads(record_path.read_text(encoding="utf-8-sig"))
                  if record_path.is_file() else {})
        info = {**source, **record}
        observer = str(info.get("observer") or path.parent.name)
        key = info.get("observer_key") or path.parent.name
        pick = PickSet(
            path=path, observer=observer, observer_key=str(key),
            case_id=landmarks.case_id, side=landmarks.side.value,
            mode=str(info.get("mode") or ""), round=int(info.get("round") or 0),
            saved_utc=str(info.get("saved_utc") or ""),
            complete=bool(info.get("complete", False)))
        for landmark in landmarks:
            if landmark.status.has_position and landmark.position_mm is not None:
                pick.positions[landmark.id] = np.asarray(landmark.position_mm, float)
            else:
                pick.skipped[landmark.id] = landmark.reason or ""
        for entry in record.get("picks") or []:
            landmark_id = entry.get("id")
            if entry.get("moved_from_auto_mm") is not None:
                pick.moved_from_auto[landmark_id] = float(entry["moved_from_auto_mm"])
            if entry.get("auto_mm") is not None:
                pick.auto[landmark_id] = np.asarray(entry["auto_mm"], float)
        sets.append(pick)
    return sets


def latest_sets(picks) -> tuple[list[PickSet], int]:
    """The latest save per observer, knee, side, mode and round, and how many were dropped."""
    latest = {}
    for pick in picks:
        held = latest.get(pick.key)
        if held is None or (pick.saved_utc, str(pick.path)) > (held.saved_utc, str(held.path)):
            latest[pick.key] = pick
    return list(latest.values()), len(picks) - len(latest)


def _pooled(variances_and_dof) -> np.ndarray:
    """Pooled per-axis variance from (per-axis variance, dof) pairs; NaN without any."""
    total = sum(dof for _, dof in variances_and_dof)
    if total <= 0:
        return np.full(3, np.nan)
    return sum(var * dof for var, dof in variances_and_dof) / total


def landmark_dispersion(picks, *, include_adjust: bool = False, auto=None) -> dict:
    """Per landmark: intra- and inter-observer SD, counts, and automatic minus mean.

    ``auto`` optionally maps ``(case_id, side)`` to ``{landmark_id: (3,) mm}``, the
    automatic estimates. See the module docstring for every definition.
    """
    modes = ("blank", "adjust") if include_adjust else ("blank",)
    used, _ = latest_sets([p for p in picks if p.mode in modes])
    auto = auto or {}

    # landmark -> (case, side) -> observer -> list of (position, distance or None, auto)
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped = defaultdict(int)
    rounds = defaultdict(set)
    for pick in used:
        for landmark_id in pick.skipped:
            skipped[landmark_id] += 1
        for landmark_id, position in pick.positions.items():
            table[landmark_id][(pick.case_id, pick.side)][pick.observer_key].append(
                (position, pick.moved_from_auto.get(landmark_id),
                 pick.auto.get(landmark_id)))
            rounds[landmark_id].add(pick.round)

    result = {}
    for landmark_id in sorted(set(table) | set(skipped)):
        intra_parts, inter_parts, auto_values, sources = [], [], [], set()
        observers, n_picks = set(), 0
        for (case_id, side), by_observer in table.get(landmark_id, {}).items():
            means, weighted = [], []
            for observer, entries in by_observer.items():
                observers.add(observer)
                points = np.array([entry[0] for entry in entries])
                n_picks += len(points)
                mean = points.mean(axis=0)
                means.append(mean)
                if len(points) >= 2:
                    intra_parts.append((points.var(axis=0, ddof=1), len(points) - 1))
                for entry in entries:
                    weighted.append((entry, 1.0 / len(entries)))
            means = np.array(means)
            grand = means.mean(axis=0)
            if len(means) >= 2:
                inter_parts.append((means.var(axis=0, ddof=1), len(means) - 1))

            estimate = (auto.get((case_id, side)) or {}).get(landmark_id)
            if estimate is None:
                given = [entry[2] for entry, _ in weighted if entry[2] is not None]
                estimate = given[0] if given else None
            if estimate is not None:
                auto_values.append(float(np.linalg.norm(np.asarray(estimate) - grand)))
                sources.add("position")
            elif all(entry[1] is not None for entry, _ in weighted):
                weights = np.array([w for _, w in weighted]) / len(means)
                d2 = np.array([entry[1] ** 2 for entry, _ in weighted])
                p2 = np.array([float(np.sum((entry[0] - grand) ** 2))
                               for entry, _ in weighted])
                auto_values.append(math.sqrt(max(float(weights @ d2 - weights @ p2), 0.0)))
                sources.add("distances")

        intra_axes = _pooled(intra_parts)
        inter_axes = _pooled(inter_parts)
        result[landmark_id] = {
            "intra_sd_mm": float(np.sqrt(np.sum(intra_axes))),
            "inter_sd_mm": float(np.sqrt(np.sum(inter_axes))),
            "intra_sd_axes_mm": tuple(float(v) for v in np.sqrt(intra_axes)),
            "inter_sd_axes_mm": tuple(float(v) for v in np.sqrt(inter_axes)),
            "n_observers": len(observers),
            "n_rounds": len(rounds.get(landmark_id, ())),
            "n_cases": len(table.get(landmark_id, {})),
            "n_picks": n_picks,
            "n_skipped": skipped.get(landmark_id, 0),
            "auto_minus_mean_mm": (float(np.mean(auto_values)) if auto_values
                                   else float("nan")),
            "auto_source": ("mixed" if len(sources) > 1 else next(iter(sources), "")),
        }
    return result


def _cell(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4f}"
    return str(value)


def write_dispersion_csv(dispersion: dict, path) -> Path:
    """``dispersion.csv``: one row per landmark, every quantity of the analysis."""
    path = Path(path)
    columns = ["landmark_id", "intra_sd_mm", "inter_sd_mm",
               *(f"intra_sd_{a}_mm" for a in _AXES), *(f"inter_sd_{a}_mm" for a in _AXES),
               "n_observers", "n_rounds", "n_cases", "n_picks", "n_skipped",
               "auto_minus_mean_mm", "auto_source"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for landmark_id, row in dispersion.items():
            writer.writerow([
                landmark_id, _cell(row["intra_sd_mm"]), _cell(row["inter_sd_mm"]),
                *map(_cell, row["intra_sd_axes_mm"]), *map(_cell, row["inter_sd_axes_mm"]),
                row["n_observers"], row["n_rounds"], row["n_cases"], row["n_picks"],
                row["n_skipped"], _cell(row["auto_minus_mean_mm"]), row["auto_source"]])
    return path


def write_e5_dispersion_csv(dispersion: dict, path, *, which: str = "inter",
                            comments: tuple[str, ...] = ()) -> Path:
    """The dispersion as E5's table: ``landmark_id,sd_mm,cov_aa,...,cov_pp``.

    The layout of ``tka_planner/validation/dispersion/literature_victor2009_inter.csv``,
    read by :func:`~tka_planner.validation.montecarlo.read_dispersion`: isotropic rows,
    ``sd_mm = r / sqrt(2)`` for a measured RMS 3D scatter ``r`` (surface picks), the
    covariance cells empty. ``which`` is ``"inter"`` or ``"intra"``.
    """
    if which not in ("inter", "intra"):
        raise ValueError("which is 'inter' or 'intra'.")
    column = f"{which}_sd_mm"
    rows, missing = [], []
    for landmark_id, row in dispersion.items():
        value = row[column]
        if math.isnan(value) or value <= 0:
            missing.append(landmark_id)
        else:
            rows.append(f"{landmark_id},{value / math.sqrt(2.0):.9g},,,,,,")
    observers = max((row["n_observers"] for row in dispersion.values()), default=0)
    lines = [
        f"Measured: E3 on-screen observer study, {which}-observer, up to {observers} "
        f"observers (paper/study_analysis.py)",
        "sd_mm per axis = RMS 3D scatter / sqrt(2): every landmark is picked on the bone "
        "surface; isotropic, as the picks are in scan mm and carry no bone frame.",
        *comments,
    ]
    if missing:
        lines.append("No value (fewer than two observers or rounds), so held fixed: "
                     + ", ".join(missing))
    text = "".join(f"# {line}\n" for line in lines)
    text += "landmark_id,sd_mm,cov_aa,cov_al,cov_ap,cov_ll,cov_lp,cov_pp\n"
    text += "".join(f"{row}\n" for row in rows)
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def read_auto_landmarks(paths) -> dict:
    """Landmark files of automatic estimates as ``{(case_id, side): {id: (3,) mm}}``."""
    auto = {}
    for path in paths:
        landmarks = read_landmark_set(path)
        auto.setdefault((landmarks.case_id, landmarks.side.value), {}).update({
            landmark.id: np.asarray(landmark.position_mm, float)
            for landmark in landmarks
            if landmark.status.has_position and landmark.position_mm is not None})
    return auto


# ----------------------------------------------------------------------
# Reviews
# ----------------------------------------------------------------------


def load_reviews(study_dir) -> list[dict]:
    """Every verdict under ``<study_dir>/reviews``, each with its ``observer_key`` set."""
    records = []
    for path in sorted(Path(study_dir).glob("reviews/*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            record["observer_key"] = str(record.get("observer_key") or path.stem)
            record["_order"] = (str(record.get("saved_utc") or ""), str(path), number)
            records.append(record)
    return records


def review_summary(study_dir) -> list[dict]:
    """Per case, side and alignment: verdict, reason and severity counts, median time.

    One verdict per observer, case, side, alignment and round counts: the latest saved.
    """
    latest, superseded = {}, defaultdict(int)
    for record in load_reviews(study_dir):
        key = (record["observer_key"], record.get("case_id"), record.get("side"),
               record.get("alignment"), record.get("round"))
        held = latest.get(key)
        group = key[1:4]
        if held is not None:
            superseded[group] += 1
        if held is None or record["_order"] > held["_order"]:
            latest[key] = record

    groups = defaultdict(list)
    for record in latest.values():
        groups[(record.get("case_id"), record.get("side"), record.get("alignment"))].append(
            record)
    reasons = [key for key, _ in REVIEW_REASONS]
    seen = {r for records in groups.values() for record in records
            for r in record.get("reasons") or []}
    reasons += sorted(seen - set(reasons))
    rows = []
    for group in sorted(groups, key=lambda g: tuple(str(v) for v in g)):
        records = groups[group]
        seconds = [float(r["seconds"]) for r in records if r.get("seconds") is not None]
        row = {
            "case_id": group[0], "side": group[1], "alignment": group[2],
            "n_reviews": len(records),
            "n_observers": len({r["observer_key"] for r in records}),
            "ok": sum(r.get("verdict") == "ok" for r in records),
            "not_ok": sum(r.get("verdict") == "not_ok" for r in records),
        }
        for reason in reasons:
            row[f"reason_{reason}"] = sum(reason in (r.get("reasons") or [])
                                          for r in records)
        for severity, _ in REVIEW_SEVERITIES:
            row[f"severity_{severity}"] = sum(r.get("severity") == severity
                                              for r in records)
        row["median_seconds"] = float(np.median(seconds)) if seconds else float("nan")
        row["n_adjusted"] = sum(bool(r.get("adjusted")) for r in records)
        row["n_stale"] = sum(bool(r.get("stale")) for r in records)
        row["n_superseded"] = superseded.get(group, 0)
        rows.append(row)
    return rows


def write_review_summary_csv(rows: list[dict], path) -> Path:
    path = Path(path)
    columns = list(rows[0]) if rows else [
        "case_id", "side", "alignment", "n_reviews", "n_observers", "ok", "not_ok"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c, "")) for c in columns])
    return path


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

_README = """\
Observer study analysis (paper/study_analysis.py), from {study}
Pick sets read: {n_sets} ({n_used} used: {modes}; {n_adjust} adjust set(s) {adjust_note};
{n_superseded} superseded by a later save of the same observer, knee, side, mode, round).
Verdicts read: {n_reviews}.

Picks are compared in scan millimetres: every observer picks on the same mesh of the
same knee, so distances equal those in the bone frame; only the per-axis SDs are along
the scanner's axes. Skipped landmarks (not_picked) are excluded and counted.

dispersion.csv -- one row per landmark (E3)
  intra_sd_mm         RMS 3D distance of a pick from the same observer's own mean over
                      rounds: per observer and knee, variance with ddof=1, pooled over
                      observers and knees weighted by (rounds - 1).
  inter_sd_mm         RMS 3D distance of an observer's mean from the grand mean (the
                      mean of the observer means): per knee, ddof=1, pooled over knees
                      weighted by (observers - 1). Needs two observers on a knee.
  intra_sd_x/y/z_mm, inter_sd_x/y/z_mm
                      the same per scanner axis; their squares sum to the 3D value's.
  n_observers         observers who placed the landmark (grouped by observer_key).
  n_rounds            distinct round numbers among the picks used.
  n_cases             knee-sides it was placed on.
  n_picks             picks used.
  n_skipped           picks skipped as not identifiable (excluded from everything).
  auto_minus_mean_mm  distance from the planner's automatic estimate to the grand mean,
                      averaged over knees; empty when unknown.
  auto_source         position: from the estimate's position; distances: recovered from
                      each pick's recorded distance to the estimate (exact identity,
                      resolved to about 0.1 mm); mixed: both over knees.

e5_dispersion.csv -- E5's dispersion table (read_dispersion), the layout of
  tka_planner/validation/dispersion/literature_victor2009_inter.csv
  landmark_id, sd_mm  isotropic per-axis SD = inter_sd_mm / sqrt(2) (surface picks:
                      E5 removes the normal component, leaving 2 of 3 axes).
  cov_*               empty (the bone frame is not in the pick files).
  A landmark with no inter-observer value has no row (E5 holds it fixed) and is named in
  the comments.

reviews_summary.csv -- one row per case, side and alignment (E13)
  n_reviews           verdicts counted (one per observer and round: the latest saved).
  n_observers         distinct observers (observer_key, else the file name).
  ok, not_ok          verdict counts.
  reason_<key>        NOT OK verdicts giving that reason (a verdict can give several).
  severity_minor/major  NOT OK verdicts by severity.
  median_seconds      median time on the verdict.
  n_adjusted          verdicts on a plan moved by hand before the verdict.
  n_stale             verdicts on a plan marked stale.
  n_superseded        earlier verdicts replaced by a later one in the same round.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paper.study_analysis",
        description="E3 landmark dispersion and E13 verdict summary from a study folder.")
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-adjust", action="store_true",
                        help="also use adjust-mode pick sets (default: blank only)")
    parser.add_argument("--auto", type=Path, action="append", default=[],
                        help="a landmark file of automatic estimates (repeatable)")
    args = parser.parse_args(argv)

    picks = load_picks(args.study_dir)
    modes = ("blank", "adjust") if args.include_adjust else ("blank",)
    used, superseded = latest_sets([p for p in picks if p.mode in modes])
    dispersion = landmark_dispersion(picks, include_adjust=args.include_adjust,
                                     auto=read_auto_landmarks(args.auto))
    rows = review_summary(args.study_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    write_dispersion_csv(dispersion, args.out / "dispersion.csv")
    write_e5_dispersion_csv(dispersion, args.out / "e5_dispersion.csv")
    write_review_summary_csv(rows, args.out / "reviews_summary.csv")
    n_adjust = sum(p.mode == "adjust" for p in picks)
    (args.out / "README.txt").write_text(_README.format(
        study=args.study_dir, n_sets=len(picks), n_used=len(used),
        modes=" + ".join(modes), n_adjust=n_adjust,
        adjust_note="included" if args.include_adjust else "excluded",
        n_superseded=superseded, n_reviews=len(load_reviews(args.study_dir))),
        encoding="utf-8")
    print(f"{len(used)} pick set(s) used of {len(picks)}, {len(dispersion)} landmark(s); "
          f"{len(rows)} review group(s). Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
