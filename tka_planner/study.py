"""Observer studies on screen: landmark picking (E3) and plan review (E13).

Both run in the planner's own window, on the case it has open, and need nothing but the
screen: no tracking, no navigation system. What a surgeon does there is recorded here,
on the server, so the browser holds no study data it could lose and computes nothing.

**Picking.** The surgeon is given the landmarks one at a time, in the registry's order,
each with its definition, on the whole uncut femur and tibia. A marker is placed on the
bone surface and can be slid over it. There are two ways to start:

* ``blank`` -- no markers. The automatic estimate is not sent to the browser at all, so
  a blinded pick cannot be anchored on it.
* ``adjust`` -- the automatic markers are placed, and the surgeon moves the ones that
  are wrong. Each automatic position stays on screen as a ghost, and a marker can be
  reset to it.

A finished set is written as an ordinary landmark file (``tka-landmarks/1.0.0``,
status ``present``, origin ``observer:<id>``), so it opens with ``--landmarks`` like any
other, beside a study record with the timings, what was skipped and why, and in
``adjust`` mode how far each marker was moved from the estimate.

**Review.** The surgeon looks at the plan the planner proposes, mechanical or
kinematic, and marks it acceptable or not, with the reasons, a severity and a comment.
Each verdict is a line of a per-observer JSON Lines file, with the plan it judged: the
alignment, every control value, and the report as the surgeon saw it.

Study output holds patient geometry and observer identities. It goes to the study
folder the server is given, never into the repository.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tka_planner.core.landmarks import (
    REGISTRY,
    Landmark,
    LandmarkSet,
    LandmarkStatus,
    write_landmark_set,
)
from tka_planner.session import ADJUSTMENT_FIELDS

__all__ = [
    "PICK_MODES",
    "REVIEW_VERDICTS",
    "REVIEW_REASONS",
    "REVIEW_SEVERITIES",
    "pickable_landmarks",
    "pick_targets",
    "save_picks",
    "save_review",
    "study_schema",
]

PICK_MODES = ("blank", "adjust")

# Not on a surface the surgeon can click (canal and joint centres), outside every
# knee-only scan, or on a bone the planner does not load (the patella).
_NOT_PICKABLE = re.compile(r"canal_centre|head_centre|ankle_centre|^patella\.")

# The mesh a landmark is picked on. The fibular head is part of the tibial segmentation
# where it is in the scan.
_BONE_FOR_PREFIX = {"femur": "Femur", "tibia": "Tibia", "fibula": "Tibia"}

REVIEW_VERDICTS = ("ok", "not_ok")
REVIEW_REASONS = (
    ("femoral_depth", "Distal femoral resection depth"),
    ("tibial_depth", "Tibial resection depth"),
    ("coronal", "Coronal alignment (varus / valgus)"),
    ("femoral_flexion", "Femoral component flexion"),
    ("tibial_slope", "Tibial slope"),
    ("rotation", "Component rotation"),
    ("size", "Component size"),
    ("fit", "Fit on the cut: overhang or undercoverage"),
    ("position", "Component position (AP or ML shift)"),
    ("insert", "Insert thickness"),
    ("landmarks", "A landmark the plan rests on is wrong"),
    ("other", "Other (say in the comment)"),
)
REVIEW_SEVERITIES = (
    ("minor", "Minor: would adjust during surgery"),
    ("major", "Major: would not use this plan"),
)

_OBSERVER = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# A pick further than this from its bone's surface is a fault in the client, not a pick.
MAX_SURFACE_GAP_MM = 5.0


def pickable_landmarks() -> list:
    """The landmarks a surgeon is asked to place, in the registry's order."""
    return [definition for definition in REGISTRY.values()
            if not _NOT_PICKABLE.search(definition.id)
            and definition.id.split(".")[0] in _BONE_FOR_PREFIX]


def study_schema() -> dict:
    """The vocabulary the study panels are built from, served like the control schema."""
    return {
        "pick_modes": list(PICK_MODES),
        "verdicts": list(REVIEW_VERDICTS),
        "reasons": [list(pair) for pair in REVIEW_REASONS],
        "severities": [list(pair) for pair in REVIEW_SEVERITIES],
    }


def pick_targets(entry, mode: str) -> dict:
    """The bones to pick on and the landmarks to place, in order.

    The bones are the meshes as they were read, before any commit cut them, because a
    landmark on a resected surface would be a pick on bone that is no longer there. In
    ``blank`` mode no automatic position is sent.
    """
    mode = _mode(mode)
    session = entry.session
    scene = session.scene
    bones = {}
    for name in ("Femur", "Tibia"):
        node = scene.nodes.get(name)
        if node is None:
            continue
        bones[name] = {
            "mesh": entry.pristine.get(name, node.mesh_id),
            "set": node.set_id,
            "rest": np.asarray(node.rest, dtype=float).tolist(),
        }
    automatic = session.measurement.landmarks
    targets = []
    for definition in pickable_landmarks():
        bone = _BONE_FOR_PREFIX[definition.id.split(".")[0]]
        if bone not in bones:
            continue
        target = {
            "id": definition.id,
            "bone": bone,
            "name": definition.display_name,
            "definition": definition.definition,
        }
        if mode == "adjust":
            target["auto_mm"] = _position(automatic, definition.id)
        targets.append(target)
    for number, target in enumerate(targets, start=1):
        target["number"] = number
    return {
        "case_id": session.case_id,
        "side": _side(session),
        "mode": mode,
        "coordinate_system": automatic.coordinate_system.value,
        "bones": bones,
        "targets": targets,
    }


def save_picks(entry, study_dir, body: dict) -> dict:
    """Write one observer's finished (or partial) set: a landmark file and a record."""
    observer = _observer(body.get("observer"))
    round_ = _round(body.get("round"))
    mode = _mode(body.get("mode"))
    session = entry.session
    allowed = {definition.id for definition in pickable_landmarks()}
    automatic = session.measurement.landmarks
    meshes = {name: session.scene.meshes.get(entry.pristine.get(name))
              for name in ("Femur", "Tibia")}

    points, records = [], []
    for item in body.get("picks") or []:
        landmark_id = item.get("id")
        if landmark_id not in allowed:
            raise ValueError(f"Not a landmark of the study: {landmark_id!r}")
        record = {"id": landmark_id, "seconds": _seconds(item.get("seconds"))}
        position = item.get("position_mm")
        if position is None:
            reason = str(item.get("reason") or "not identifiable")[:200]
            points.append(Landmark(id=landmark_id, status=LandmarkStatus.NOT_PICKED,
                                   origin=f"observer:{observer}", reason=reason))
            record["skipped"] = reason
        else:
            position = _point(position, landmark_id)
            mesh = meshes.get(_BONE_FOR_PREFIX[landmark_id.split(".")[0]])
            if mesh is not None:
                gap = _surface_gap(mesh.vertices, position)
                if gap > MAX_SURFACE_GAP_MM:
                    raise ValueError(
                        f"{landmark_id} lies {gap:.1f} mm from its bone's surface; a "
                        f"pick must be on the bone.")
                record["surface_gap_mm"] = round(gap, 3)
            points.append(Landmark(id=landmark_id, position_mm=position,
                                   status=LandmarkStatus.PRESENT,
                                   origin=f"observer:{observer}"))
            record["position_mm"] = [round(float(v), 4) for v in position]
            auto = _position(automatic, landmark_id)
            if auto is not None:
                record["moved_from_auto_mm"] = round(
                    float(np.linalg.norm(position - np.asarray(auto))), 3)
        records.append(record)
    if not records:
        raise ValueError("No picks were given.")

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    side = _side(session)
    folder = Path(study_dir) / "picks" / observer
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{session.case_id}_{side}_{mode}_r{round_}_{stamp}"
    source = {"kind": "observer_study", "observer": observer, "round": round_,
              "mode": mode, "saved_utc": stamp}
    landmark_path = write_landmark_set(
        LandmarkSet(case_id=session.case_id, side=side,
                    coordinate_system=automatic.coordinate_system,
                    landmarks=points, source=source),
        folder / f"{base}.landmarks.json")
    record_path = folder / f"{base}.study.json"
    record_path.write_text(json.dumps({
        **source,
        "case_id": session.case_id,
        "side": side,
        "complete": bool(body.get("complete", False)),
        "started_utc": _text(body.get("started_utc")),
        "total_seconds": _seconds(body.get("total_seconds")),
        "landmark_file": landmark_path.name,
        "picks": records,
    }, indent=2), encoding="utf-8")
    return {
        "landmarks": str(landmark_path),
        "record": str(record_path),
        "placed": sum(1 for r in records if "position_mm" in r),
        "skipped": sum(1 for r in records if "skipped" in r),
    }


def save_review(entry, study_dir, body: dict) -> dict:
    """Append one verdict on the plan as it stands now."""
    observer = _observer(body.get("observer"))
    round_ = _round(body.get("round"))
    verdict = body.get("verdict")
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"A verdict is one of {', '.join(REVIEW_VERDICTS)}.")
    known = {key for key, _ in REVIEW_REASONS}
    reasons = list(body.get("reasons") or [])
    unknown = [reason for reason in reasons if reason not in known]
    if unknown:
        raise ValueError(f"Unknown reasons: {', '.join(map(str, unknown))}")
    severity = body.get("severity") or None
    if verdict == "not_ok":
        if severity not in {key for key, _ in REVIEW_SEVERITIES}:
            raise ValueError("A plan marked not OK needs a severity.")
        if not reasons:
            raise ValueError("A plan marked not OK needs at least one reason.")
    else:
        severity = None
    session = entry.session
    controls = asdict(session.controls)
    record = {
        "saved_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observer": observer,
        "round": round_,
        "case_id": session.case_id,
        "side": _side(session),
        "alignment": controls.get("philosophy"),
        "verdict": verdict,
        "reasons": reasons,
        "severity": severity,
        "comment": str(body.get("comment") or "")[:2000],
        "seconds": _seconds(body.get("seconds")),
        # The plan as the planner proposed it, or one moved by hand before the verdict.
        "adjusted": any(float(controls.get(name) or 0.0) != 0.0
                        for name in (*ADJUSTMENT_FIELDS, "insert_thickness_delta_mm")),
        "stale": bool(session.stale),
        "controls": controls,
        "report": session.report_lines(),
    }
    folder = Path(study_dir) / "reviews"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{observer}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return {"file": str(path), "record": {k: record[k] for k in (
        "case_id", "side", "alignment", "verdict", "reasons", "severity", "round")}}


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------


def _mode(value) -> str:
    if value not in PICK_MODES:
        raise ValueError(f"A pick mode is one of {', '.join(PICK_MODES)}.")
    return value


def _observer(value) -> str:
    if not isinstance(value, str) or not _OBSERVER.match(value.strip()):
        raise ValueError("Give the observer as 1-32 letters, digits, '-' or '_'.")
    return value.strip()


def _round(value) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Not a round number: {value!r}") from None
    if not 1 <= number <= 9:
        raise ValueError("The round is 1 to 9.")
    return number


def _seconds(value):
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Not a duration: {value!r}")
    return round(number, 1)


def _text(value):
    return None if value is None else str(value)[:64]


def _point(value, landmark_id: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{landmark_id}: a position is three finite millimetre values.")
    return point


def _position(landmarks, landmark_id: str):
    if landmark_id not in landmarks:
        return None
    landmark = landmarks.get(landmark_id)
    if not landmark.status.has_position or landmark.position_mm is None:
        return None
    return [float(v) for v in landmark.position_mm]


def _surface_gap(vertices: np.ndarray, point: np.ndarray) -> float:
    """Distance to the nearest vertex: an upper bound on the distance to the surface."""
    best = math.inf
    for start in range(0, len(vertices), 200_000):
        chunk = vertices[start:start + 200_000] - point
        best = min(best, float(np.min(np.einsum("ij,ij->i", chunk, chunk))))
    return math.sqrt(best)


def _side(session) -> str:
    side = session.side
    return getattr(side, "value", side)
