"""Writing a session out as a folder somebody can keep.

The command line already produces `plan.json`, `report.html` and `landmarks.json` from a
measurement. A session has all of that plus two things the command line never had: the
manual adjustments a surgeon dialled in, and the geometry a commit produced. So the
bundle is the command line's output with those added, which keeps one format rather than
inventing a second.

Geometry is written as STL because that is what the rest of the theatre chain reads.
The viewer's glTF is a transport format and is not written here.
"""

from __future__ import annotations

import json
from pathlib import Path

from tka_planner import __version__
from tka_planner.core.landmarks import write_landmark_set
from tka_planner.geom import mesh as gm

from .html import render_report, write_report

__all__ = ["export_session"]


def export_session(session, out_dir, *, commit=None, write_meshes: bool = True) -> dict:
    """Write the plan, the report, the landmarks and the geometry into a folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    written["landmarks"] = str(
        write_landmark_set(session._landmarks, out_dir / "landmarks.json")
    )

    summary = {
        "case_id": session.case_id,
        "side": str(session.side),
        "tool_version": __version__,
        "inputs": {
            "femur": str(session.femur_path),
            "tibia": str(session.tibia_path),
            "femur_sha256": session._femur.sha256,
            "tibia_sha256": session._tibia.sha256,
            "library": str(session.library) if session.library else None,
        },
        "frames": {
            "femoral": session._femoral_frame.to_dict(),
            "tibial": session._tibial_frame.to_dict(),
        },
        "metrics": {
            name: metric.to_dict() for name, metric in session._metrics.items()
        },
        "surgical_plan": session.plan.to_dict(),
        "sizing": {
            "parametric": session.sizing.to_dict(),
            "femoral_measurement": session._femoral_measure.to_dict(),
            "tibial_measurement": session._tibial_measure.to_dict(),
        },
        # The controls are the part the command line has no equivalent for: without them
        # the plan cannot be reproduced, because a manual adjustment is an input.
        "controls": _controls(session),
        "trial": {
            "flexion_deg": session.trial.flexion_deg,
            "varus_valgus_deg": session.trial.varus_valgus_deg,
            "drawer_ap_mm": session.trial.drawer_ap_mm,
        },
        "geometry": commit.to_dict() if commit is not None else None,
        "geometry_matches_plan": not session.stale,
        "intended_use": "research_and_demonstration_only__not_a_medical_device",
    }
    plan_path = out_dir / "plan.json"
    plan_path.write_text(
        json.dumps(summary, indent=2, default=_jsonable) + "\n", encoding="utf-8"
    )
    written["plan"] = str(plan_path)

    document = render_report(
        case_id=session.case_id,
        side=str(session.side),
        metrics=session._metrics,
        femoral_frame=session._femoral_frame,
        tibial_frame=session._tibial_frame,
        sizing=session.sizing,
        surgical=session.plan,
        measurements={
            "femur": session._femoral_measure.to_dict(),
            "tibia": session._tibial_measure.to_dict(),
        },
        inputs=[
            {"role": "femur", "path": str(session.femur_path),
             "sha256": session._femur.sha256},
            {"role": "tibia", "path": str(session.tibia_path),
             "sha256": session._tibia.sha256},
        ],
        tool_version=__version__,
    )
    written["report"] = str(write_report(document, out_dir / "report.html"))

    if write_meshes and session.scene is not None:
        meshes = out_dir / "meshes"
        paths = []
        for node in session.scene.nodes.values():
            if node.mesh_id is None or node.tags.get("landmark"):
                continue
            mesh = session.scene.meshes[node.mesh_id]
            # Written in world space, so the folder can be opened in any viewer and the
            # parts land where the plan put them.
            placed = gm.transformed(mesh, session.scene.world(node.name))
            paths.append(str(gm.write_stl(placed, meshes / f"{node.name}.stl")))
        written["meshes"] = sorted(paths)

    return written


def _controls(session) -> dict:
    from dataclasses import asdict

    return asdict(session.controls)


def _jsonable(value):
    """Convert numpy scalars and arrays to plain Python, as the command line does."""
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
