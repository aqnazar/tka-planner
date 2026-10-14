"""A knee planned and posed exactly as the application does it, for the geometry
experiments.

E8 cuts the bones with two solvers and E9 times the cut, and both must cut what the
application cuts: the same scene, the same cutting blocks posed by the same plan.
So they go through :class:`~tka_planner.session.PlanningSession` rather than
assembling the tools themselves. The session is given the harness's shared
measurement instead of measuring again, and is never opened from a folder, because
``PlanningSession.open`` names the case after its folder and cohort folders carry
patient names.
"""

from __future__ import annotations

from tka_planner.scene import resect
from tka_planner.session import PlanningSession

from .harness import Case

BONES = ("Femur", "Tibia")
GROUP = {"Femur": "femoral", "Tibia": "tibial"}
PLANE = {"Femur": "femoral_distal", "Tibia": "tibial_proximal"}


def planned_session(case: Case, measurement, library) -> PlanningSession:
    """A session on ``measurement`` with its first plan built: the state Commit starts
    from."""
    session = PlanningSession(femur_path=case.femur, tibia_path=case.tibia,
                              side=case.side, library=library, case_id=case.case_id)
    session.measurement = measurement
    session.build()
    return session


def commit_tools(session: PlanningSession, bone: str) -> dict:
    """The three tools a commit can cut ``bone`` with, in world space.

    ``block`` and ``shell`` are what block mode (the application's default) uses: the
    shell is intersected with the uncut bone, then the block is subtracted. ``plane`` is
    plane mode's discard box. Missing tools are left out.
    """
    group = GROUP[bone]
    tools = {}
    shell = resect._tool(session.scene, f"{group}_cutting_block_shell")
    if shell is not None:
        tools["shell"] = ("intersect", shell)
    block = resect._tool(session.scene, f"{group}_cutting_block")
    if block is not None:
        tools["block"] = ("difference", block)
    resection = session.plan.resections[PLANE[bone]]
    tools["plane"] = ("difference", resect.cutter_for(
        resection.point, resection.normal, keep=resect.KEEP[PLANE[bone]]))
    return tools


def bone_mesh(session: PlanningSession, bone: str):
    """The uncut bone as the scene holds it before Commit."""
    return session.scene.mesh_for(bone)

