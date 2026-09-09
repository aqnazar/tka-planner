"""Re-pose a built scene from a re-planned :class:`SurgicalPlan`.

This is the path every plan control takes. Nothing is imported, created or deleted: the
cut planes, the components and the insert are already in the scene and only their rest
transforms change. Planning itself is a few dozen numpy operations, so the whole round
trip is fast enough to run on every change of a value rather than on a button press.

Nothing here cuts anything. That is the Plan-mode guarantee: no boolean runs while a
control is moving, so the update cost is a handful of matrix assignments regardless of
how dense the segmentation is.

Rebuilding instead would mean re-reading the meshes, re-estimating the landmarks and
re-applying the booleans, which is seconds of work on every change of a value.
"""

from __future__ import annotations

import numpy as np

from tka_planner.geom import mesh as gm

from .build import INSERT_SPACER, seat
from .model import Scene, SceneDelta

__all__ = ["update_scene", "set_visibility", "isolate_landmarks"]

# Below this a pose has not meaningfully moved, and reporting it would put noise on the
# wire on every frame of a drag. A micron is far below any tolerance a resection cares
# about.
POSE_TOLERANCE = 1e-9


def update_scene(
    scene: Scene,
    plan,
    *,
    insert_thickness_mm: float | None = None,
    implant_ml_mm: float | None = None,
) -> SceneDelta:
    """Apply a plan to a scene and report what moved."""
    delta = SceneDelta()

    for name, resection in plan.resections.items():
        node = scene.nodes.get(name)
        if node is None:
            continue
        matrix = gm.plane_matrix(resection.point, resection.normal)
        if not np.allclose(matrix, node.rest, atol=POSE_TOLERANCE):
            node.rest = matrix
            delta.poses[name] = matrix

    for node in scene.nodes.values():
        if not node.tags.get("component"):
            continue
        pose = plan.components.get(f"{node.tags['group']}_component")
        if pose is None:
            continue

        # A change of implant size needs no re-import. Because the library is one master
        # geometry under a uniform scale, re-deriving each part's factor from the width
        # its own mesh was exported at means the part already in the scene *is* the new
        # size rather than a stand-in for it.
        source_ml = node.tags.get("source_ml")
        scale = (
            implant_ml_mm / source_ml
            if implant_ml_mm and source_ml
            else node.tags.get("scale", 1.0)
        )
        matrix = seat(pose, scale)
        if not np.allclose(matrix, node.rest, atol=POSE_TOLERANCE):
            node.rest = matrix
            delta.poses[node.name] = matrix
        node.tags["scale"] = float(scale)

    spacer = scene.nodes.get(INSERT_SPACER)
    if spacer is not None:
        _update_insert(scene, spacer, plan, insert_thickness_mm, delta)

    delta.notes.extend(plan.warnings)
    return delta


def _update_insert(scene, spacer, plan, thickness_mm, delta) -> None:
    """Show, hide or re-stretch the insert slab.

    The slab is rebuilt rather than re-scaled because its footprint must not change
    with its thickness: the gap it fills is what the number means.
    """
    if thickness_mm is None:
        if spacer.visible:
            spacer.visible = False
            delta.visibility[INSERT_SPACER] = False
        return

    if not spacer.visible:
        spacer.visible = True
        delta.visibility[INSERT_SPACER] = True

    ml_mm, ap_mm = spacer.tags.get("footprint_mm", (70.0, 48.0))
    rebuilt = gm.slab(ml_mm, ap_mm, thickness_mm)
    new_id = scene.replace_mesh(INSERT_SPACER, rebuilt)
    delta.meshes[INSERT_SPACER] = new_id

    matrix = seat(plan.components["tibial_component"], 1.0)
    if not np.allclose(matrix, spacer.rest, atol=POSE_TOLERANCE):
        spacer.rest = matrix
        delta.poses[INSERT_SPACER] = matrix


def set_visibility(scene: Scene, predicate, visible: bool) -> SceneDelta:
    """Show or hide every node matching ``predicate``, reporting only real changes."""
    delta = SceneDelta()
    for node in scene.nodes.values():
        if predicate(node) and node.visible != visible:
            node.visible = visible
            delta.visibility[node.name] = visible
    return delta


def isolate_landmarks(scene: Scene, enabled: bool) -> SceneDelta:
    """Show the landmarks alone, or restore everything.

    What was hidden is remembered on the node rather than inferred on the way back, so
    something already hidden for its own reason, such as the insert when it is turned
    off, stays hidden when isolation ends.
    """
    delta = SceneDelta()
    for node in scene.nodes.values():
        if node.tags.get("landmark"):
            continue

        if enabled:
            node.tags.setdefault("visible_before_isolate", node.visible)
            if node.visible:
                node.visible = False
                delta.visibility[node.name] = False
        else:
            restored = node.tags.pop("visible_before_isolate", node.visible)
            if node.visible != restored:
                node.visible = restored
                delta.visibility[node.name] = restored
    return delta
