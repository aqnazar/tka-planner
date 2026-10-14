"""REG: does a code change leave the cohort exactly where it was?

A fix aimed at one knee must not move the others, and "the tests pass" does not show
that: the tests run on synthetic bones. This experiment snapshots, per knee, everything
the planner derives from the meshes -- every automatic landmark, both frames, every
metric with its tier, the default plan (mechanical alignment, untouched controls) and
the size -- and compares a later run against it, number by number.

Freeze a baseline on the code you trust, then change the code and run again::

    TKA_REG_FREEZE=1 python -m paper run REG --reference    # write missing baselines
    python -m paper run REG --reference                     # compare against them

``TKA_REG_FREEZE=overwrite`` replaces existing baselines; plain ``1`` never does, so a
baseline frozen before a change cannot be lost by re-running the freeze after it.

The baselines go to ``TKA_PAPER_OUT/baselines/<case>.json``, outside the repository,
because they are derived from patient scans. Positions compare to 1e-6 mm and
directions to 1e-6 degrees; anything larger is a difference and is listed.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from ..harness import Context, experiment, library_root, measured, out_root, run_record

LENGTH_TOL_MM = 1e-6
ANGLE_TOL_DEG = 1e-6
SCALAR_TOL = 1e-6        # metric values (mm or degrees) and sizing scalars


def baseline_dir():
    return out_root() / "baselines"


# ----------------------------------------------------------------------
# The snapshot
# ----------------------------------------------------------------------


def _vector(value) -> list[float]:
    return [float(v) for v in np.asarray(value, dtype=float).ravel()]


def snapshot(case) -> dict:
    """Everything the planner derives from one knee, in plain JSON, at full precision.

    Every leaf is tagged with how it compares (``point``, ``direction``, ``scalar`` or
    ``exact``), so the comparison needs no knowledge of what the numbers mean.
    """
    from tka_planner.pipeline import discrete_sizing, plan_case

    m = measured(case)
    if isinstance(m, Exception):
        return {"case": case.case_id, "accepted": False,
                "refusal": {"kind": "exact", "value": type(m).__name__},
                "reason": {"kind": "exact", "value": str(m).splitlines()[0]}}

    landmarks = {}
    for landmark in m.landmarks:
        record = {"status": {"kind": "exact", "value": landmark.status.value},
                  "origin": {"kind": "exact", "value": landmark.origin}}
        if landmark.position_mm is not None:
            record["position_mm"] = {"kind": "point",
                                     "value": _vector(landmark.position_mm)}
        landmarks[landmark.id] = record

    frames = {}
    for name, frame in (("femoral", m.femoral_frame), ("tibial", m.tibial_frame)):
        frames[name] = {
            "origin_mm": {"kind": "point", "value": _vector(frame.origin)},
            "x_anterior": {"kind": "direction", "value": _vector(frame.x_anterior)},
            "y_patient_left": {"kind": "direction",
                               "value": _vector(frame.y_patient_left)},
            "z_proximal": {"kind": "direction", "value": _vector(frame.z_proximal)},
            "method": {"kind": "exact", "value": frame.method},
            "quality": {"kind": "exact", "value": frame.quality.value},
        }

    metrics = {
        name: {"value": {"kind": "scalar", "value": metric.value},
               "tier": {"kind": "exact", "value": metric.quality.value}}
        for name, metric in m.metrics.items()
    }

    plan, sizing = plan_case(m, philosophy="mechanical", library=library_root())
    resections = {
        name: {"point_mm": {"kind": "point", "value": _vector(cut.point)},
               "normal": {"kind": "direction", "value": _vector(cut.normal)},
               "medial_depth_mm": {"kind": "scalar", "value": float(cut.medial_depth_mm)},
               "lateral_depth_mm": {"kind": "scalar",
                                    "value": float(cut.lateral_depth_mm)},
               "reference": {"kind": "exact", "value": cut.reference}}
        for name, cut in plan.resections.items()
    }
    components = {}
    for name, matrix in plan.components.items():
        matrix = np.asarray(matrix, dtype=float)
        components[name] = {
            "x": {"kind": "direction", "value": _vector(matrix[:3, 0])},
            "y": {"kind": "direction", "value": _vector(matrix[:3, 1])},
            "z": {"kind": "direction", "value": _vector(matrix[:3, 2])},
            "translation_mm": {"kind": "point", "value": _vector(matrix[:3, 3])},
        }
    discrete = discrete_sizing(m)
    return {
        "case": case.case_id,
        "accepted": True,
        "landmarks": landmarks,
        "frames": frames,
        "metrics": metrics,
        "plan": {
            "valgus_cut_deg": {"kind": "scalar",
                               "value": float(plan.distal_femoral_valgus_cut_deg)},
            "tibial_slope_deg": {"kind": "scalar", "value": float(plan.tibial_slope_deg)},
            "warnings": {"kind": "exact", "value": list(plan.warnings)},
            "resections": resections,
            "components": components,
        },
        "sizing": {
            "femoral_ml_mm": {"kind": "scalar", "value": float(m.femoral_measure.ml_mm)},
            "femoral_ap_mm": {"kind": "scalar", "value": float(m.femoral_measure.ap_mm)},
            "size_parameter": {"kind": "scalar", "value": float(sizing.size_parameter)},
            "nearest_discrete_size": {"kind": "exact",
                                      "value": sizing.nearest_discrete_size},
            "scale_factor": {"kind": "scalar", "value": float(sizing.scale_factor)},
            "femoral_thickness_mm": {"kind": "scalar",
                                     "value": float(sizing.femoral_thickness_mm)},
            "discrete_size": {"kind": "exact", "value": discrete.nearest_discrete_size},
        },
    }


# ----------------------------------------------------------------------
# The comparison
# ----------------------------------------------------------------------


def _angle_deg(a, b) -> float:
    """The angle between two directions, accurate near zero (atan2, not arccos)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))),
                                   float(np.dot(a, b))))


def _leaf(node) -> bool:
    return isinstance(node, dict) and set(node) == {"kind", "value"}


def compare(baseline: dict, current: dict, path: str = "") -> list[dict]:
    """Every difference between two snapshots beyond tolerance, as rows."""
    rows = []
    if _leaf(baseline) and _leaf(current):
        kind, old, new = baseline["kind"], baseline["value"], current["value"]
        if kind == "exact" or old is None or new is None:
            if old != new:
                rows.append({"path": path, "kind": kind, "baseline": old,
                             "current": new, "delta": None, "unit": ""})
            return rows
        if kind == "point":
            delta, tol, unit = float(np.linalg.norm(np.subtract(new, old))), \
                LENGTH_TOL_MM, "mm"
        elif kind == "direction":
            delta, tol, unit = _angle_deg(old, new), ANGLE_TOL_DEG, "deg"
        else:
            delta, tol, unit = abs(float(new) - float(old)), SCALAR_TOL, ""
        if delta > tol:
            rows.append({"path": path, "kind": kind, "baseline": _short(old),
                         "current": _short(new), "delta": delta, "unit": unit})
        return rows
    if isinstance(baseline, dict) and isinstance(current, dict):
        for key in sorted(set(baseline) | set(current)):
            sub = f"{path}/{key}" if path else key
            if key not in current:
                rows.append({"path": sub, "kind": "removed", "baseline": "present",
                             "current": "absent", "delta": None, "unit": ""})
            elif key not in baseline:
                rows.append({"path": sub, "kind": "added", "baseline": "absent",
                             "current": "present", "delta": None, "unit": ""})
            else:
                rows += compare(baseline[key], current[key], sub)
        return rows
    if baseline != current:
        rows.append({"path": path, "kind": "exact", "baseline": baseline,
                     "current": current, "delta": None, "unit": ""})
    return rows


def _short(value):
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
    return value


def _category(path: str) -> str:
    return path.split("/", 1)[0]


@experiment("REG", "Cohort regression check against a frozen baseline")
def run(ctx: Context) -> None:
    mode = os.environ.get("TKA_REG_FREEZE", "").strip().lower()
    freeze = mode in ("1", "true", "yes", "overwrite")
    folder = baseline_dir()
    folder.mkdir(parents=True, exist_ok=True)
    current_dir = ctx.path("current")
    current_dir.mkdir(exist_ok=True)

    summary, diffs = [], []
    for case in ctx.cases:
        current = snapshot(case)
        (current_dir / f"{case.case_id}.json").write_text(
            json.dumps(current, indent=1), encoding="utf-8")
        target = folder / f"{case.case_id}.json"
        if freeze:
            if target.exists() and mode != "overwrite":
                summary.append({"case": case.case_id, "result": "baseline kept",
                                "accepted": current["accepted"]})
                continue
            record = {"frozen": run_record(), "snapshot": current}
            target.write_text(json.dumps(record, indent=1), encoding="utf-8")
            summary.append({"case": case.case_id, "result": "frozen",
                            "accepted": current["accepted"]})
            continue
        if not target.exists():
            summary.append({"case": case.case_id, "result": "no baseline",
                            "accepted": current["accepted"]})
            continue
        frozen = json.loads(target.read_text(encoding="utf-8"))
        rows = compare(frozen["snapshot"], current)
        for row in rows:
            diffs.append({"case": case.case_id, **row})
        by_category: dict[str, int] = {}
        for row in rows:
            by_category[_category(row["path"])] = by_category.get(
                _category(row["path"]), 0) + 1
        summary.append({
            "case": case.case_id,
            "result": "identical" if not rows else "differs",
            "accepted": current["accepted"],
            "baseline_commit": frozen["frozen"].get("commit", "")[:7],
            "differences": len(rows),
            **{f"n_{k}": v for k, v in sorted(by_category.items())},
        })

    ctx.table("summary", summary, digits=6)
    if not freeze:
        ctx.table("differences", diffs,
                  ["case", "path", "kind", "baseline", "current", "delta", "unit"],
                  digits=9)
    if freeze:
        ctx.note(f"Baselines in {folder}: "
                 + ", ".join(f"{r['case']} {r['result']}" for r in summary))
        return
    ctx.note(f"Tolerances: positions {LENGTH_TOL_MM} mm, directions {ANGLE_TOL_DEG} deg, "
             f"scalars {SCALAR_TOL}; strings and tiers exactly.")
    for row in summary:
        line = f"- {row['case']}: {row['result']}"
        if row.get("differences"):
            line += f" ({row['differences']} differences)"
        ctx.note(line)
