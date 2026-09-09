"""Committing a plan's resection into geometry.

This is the only module in the engine that runs a boolean, and it runs one only when
asked. That is the whole point of the Plan / Commit / Reduce split: an exact boolean
against a real segmentation costs seconds, which cannot live inside a slider drag, so
it is moved out of the drag entirely rather than hidden behind a debounce.

Two modes, and they are alternatives rather than a stack. **Block** subtracts the
cutting block itself, which is the instrument that would realise the cut in theatre,
and intersects a copy of the bone with the block's shell to give the patient-specific
mating surface. **Plane** removes everything beyond the planned plane instead, which is
what the plan actually specifies. Stacking them would let the plane swallow the surfaces
the block is shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernel import MeshKernel, default_kernel

from .build import SHELL_COLOUR
from .model import FEMORAL, TIBIAL, Node, Scene, SceneDelta

__all__ = ["CUTTER_SIZE_MM", "cutter_for", "CommitResult", "commit_resection"]

# Large enough to swallow the discarded side of any bone in the cohort.
CUTTER_SIZE_MM = 400.0

# Which side of each plane is kept. The distal femur is cut from below, so the femur
# keeps the bone above its plane; the proximal tibia is cut from above and keeps the
# bone below. Applying one convention to both silently inverts a resection.
KEEP = {"femoral_distal": "proximal", "tibial_proximal": "distal"}
BONE_FOR = {"femoral_distal": "Femur", "tibial_proximal": "Tibia"}


def cutter_for(point_mm, normal_mm, *, keep: str) -> Mesh:
    """The discard box for a resection plane.

    Its near face lies on the plane and its bulk sits on the side being thrown away, so
    a boolean difference leaves exactly the retained bone.

    The cut is made with a box rather than with the cutting block, because the block is
    the instrument that would realise the cut in theatre while the plane is what the
    plan actually specifies. The two are different by design and are meant to agree;
    where they do not, that is worth seeing.
    """
    if keep not in ("proximal", "distal"):
        raise ValueError("keep must be 'proximal' or 'distal'.")

    normal = np.asarray(normal_mm, dtype=float)
    normal = normal / np.linalg.norm(normal)
    discard = -normal if keep == "proximal" else normal
    centre = np.asarray(point_mm, dtype=float) + discard * (CUTTER_SIZE_MM / 2.0)

    return gm.box(CUTTER_SIZE_MM, gm.translation(centre))


@dataclass
class CommitResult:
    """The cut scene, what the kernel did, and what changed."""

    scene: Scene
    records: dict = field(default_factory=dict)
    delta: SceneDelta = field(default_factory=SceneDelta)
    notes: tuple = ()

    def to_dict(self) -> dict:
        """The geometry provenance, shaped to sit in ``plan.json``.

        The plan already records how every number was obtained. This says how the
        geometry was obtained: which solver cut each bone, what it had to repair first,
        and whether it fell back.
        """
        return {
            "resections": {
                name: record.to_dict() for name, record in self.records.items()
            },
            "notes": list(self.notes),
        }


def commit_resection(
    scene: Scene,
    plan,
    *,
    mode: str = "block",
    kernel: MeshKernel | None = None,
    bones: tuple = ("Femur", "Tibia"),
    build_shells: bool = True,
) -> CommitResult:
    """Cut the named bones and return the scene with committed geometry.

    ``bones`` is what makes a re-commit cheap: adjusting the tibial slope re-cuts the
    tibia and leaves an untouched femur alone. The caller decides which bones a change
    reached; see :meth:`tka_planner.session.PlanningSession.bones_affected_by`.
    """
    kernel = kernel or default_kernel()
    result = CommitResult(scene=scene)

    if mode == "none":
        return result
    if mode not in ("block", "plane"):
        raise ValueError(f"Unknown resection mode {mode!r}; expected block or plane.")

    notes: list = []
    for plane_name, resection in plan.resections.items():
        bone_name = BONE_FOR.get(plane_name)
        if bone_name is None or bone_name not in bones:
            continue
        bone = scene.mesh_for(bone_name)
        if bone is None:
            continue

        if mode == "block":
            _commit_block(
                scene, kernel, bone_name, bone, result, notes,
                build_shells=build_shells,
            )
        else:
            tool = cutter_for(
                resection.point, resection.normal,
                keep=KEEP.get(plane_name, "proximal"),
            )
            cut = kernel.difference(bone, tool)
            result.delta.meshes[bone_name] = scene.replace_mesh(bone_name, cut.mesh)
            result.records[bone_name] = cut.record

    result.notes = tuple(notes)
    result.delta.notes.extend(notes)
    return result


def _commit_block(scene, kernel, bone_name, bone, result, notes, *, build_shells):
    """Subtract the cutting block, and take the shell before doing so."""
    group = "femoral" if bone_name == "Femur" else "tibial"
    set_id = FEMORAL if bone_name == "Femur" else TIBIAL

    block = _tool(scene, f"{group}_cutting_block")
    if block is None:
        notes.append(f"{bone_name}: no cutting block, left whole")
        return

    # The shell copy is taken **before** the bone is resected, and that ordering is the
    # whole trick. A shell cut from an already-resected femur would be missing the
    # condylar surface it is supposed to mate with, so it would fit nothing.
    if build_shells:
        shell_tool = _tool(scene, f"{group}_cutting_block_shell")
        if shell_tool is None:
            notes.append(f"{bone_name}: no block shell, shell not built")
        else:
            shell = kernel.intersect(bone, shell_tool)
            name = f"{bone_name}.Shell"
            scene.add(
                Node(name=name, set_id=set_id, colour=SHELL_COLOUR,
                     tags={"shell": True, "group": group}),
                shell.mesh,
            )
            result.records[name] = shell.record
            result.delta.meshes[name] = scene.nodes[name].mesh_id

    cut = kernel.difference(bone, block)
    result.delta.meshes[bone_name] = scene.replace_mesh(bone_name, cut.mesh)
    result.records[bone_name] = cut.record


def _tool(scene: Scene, name: str) -> Mesh | None:
    """A component's mesh in world space, so it can be used as a boolean tool.

    Components are stored at the origin with their seating held in the node's rest
    transform, but a kernel takes plain geometry, so the transform is applied here.
    That is also what makes the block cut where it is posed: move the implant and the
    tool that shapes the bone moves with it, which is deliberate.
    """
    node = scene.nodes.get(name)
    if node is None or node.mesh_id is None:
        return None
    return gm.transformed(scene.meshes[node.mesh_id], scene.world(name))
