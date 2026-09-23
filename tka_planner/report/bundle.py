"""Writing a session out as a folder somebody can keep.

The command line already produces `plan.json`, `report.html` and `landmarks.json` from a
measurement. A session has all of that plus two things the command line never had: the
manual adjustments a surgeon dialled in, and the geometry a commit produced. Both are
written through :mod:`tka_planner.pipeline`, so the bundle's plan is the command line's
format with those keys filled in, not a second format.

Geometry is written as STL because that is what the rest of the theatre chain reads.
The viewer's glTF is a transport format and is not written here.
"""

from __future__ import annotations

from pathlib import Path

from tka_planner.core.landmarks import write_landmark_set
from tka_planner.geom import mesh as gm
from tka_planner.pipeline import (
    fit_case,
    plan_document,
    render_case_report,
    write_plan,
)

from .html import write_report

__all__ = ["export_session"]


def export_session(session, out_dir, *, commit=None, write_meshes: bool = True) -> dict:
    """Write the plan, the report, the landmarks and the geometry into a folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    written["landmarks"] = str(
        write_landmark_set(session._landmarks, out_dir / "landmarks.json")
    )

    # The command line's plan format, with the application's controls, trial pose and
    # last commit filled in where the command line writes None.
    document = plan_document(
        session.measurement, session.plan, session.sizing,
        fit=fit_case(session.measurement, session.plan, session.sizing,
                     session.library),
        controls=_controls(session),
        trial={
            "flexion_deg": session.trial.flexion_deg,
            "varus_valgus_deg": session.trial.varus_valgus_deg,
            "drawer_ap_mm": session.trial.drawer_ap_mm,
            "distraction_mm": session.trial.distraction_mm,
        },
        geometry=commit.to_dict() if commit is not None else None,
        geometry_matches_plan=not session.stale,
    )
    written["plan"] = str(write_plan(document, out_dir / "plan.json"))

    report = render_case_report(session.measurement, session.plan, session.sizing)
    written["report"] = str(write_report(report, out_dir / "report.html"))

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
