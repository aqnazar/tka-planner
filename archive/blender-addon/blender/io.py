"""Blender import, export and scene helpers.

Every ``bpy.ops`` call in the project lives here or in :mod:`tka_planner.blender.mesh`.
Confining them matters because Blender renames operators between releases -- the STL
importer moved from ``bpy.ops.import_mesh.stl`` to ``bpy.ops.wm.stl_import`` in 4.x --
and the legacy script scattered a try/except around every call site. One shim, chosen
from ``bpy.app.version``, replaces all of them.

This is also the only boundary where units change. The planning core works in
millimetres throughout; Blender scenes are built in metres because its solvers and
viewport clipping behave badly at a scale of hundreds. The conversion happens here and
nowhere else.
"""

from __future__ import annotations

from pathlib import Path

import bpy

__all__ = [
    "MM_TO_BU",
    "BU_TO_MM",
    "to_blender",
    "from_blender",
    "clear_scene",
    "import_stl",
    "export_stl",
    "ensure_collection",
    "set_material",
]

# One Blender unit is one metre; the core speaks millimetres.
MM_TO_BU = 0.001
BU_TO_MM = 1000.0


def to_blender(value_mm):
    """Millimetres to Blender units."""
    return value_mm * MM_TO_BU


def from_blender(value_bu):
    """Blender units to millimetres."""
    return value_bu * BU_TO_MM


def _is_4x() -> bool:
    return bpy.app.version >= (4, 0, 0)


def clear_scene() -> None:
    """Empty the scene, including orphaned data, so repeated runs do not accumulate.

    Removes objects directly through ``bpy.data`` rather than selecting and deleting
    them. ``select_all`` silently skips anything with ``hide_viewport`` set, which the
    plane cutters always have and the flexion animation's editing objects have while
    the timeline sits away from frame 1 -- either one left a hidden object out of the
    old select-and-delete, orphaned to survive the "clear" and duplicate on the next
    Plan press, which is what surfaced as extra cutting blocks in the scene.
    """
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.collections):
        for item in list(collection):
            if getattr(item, "users", 0) == 0:
                collection.remove(item)


def import_stl(path: "str | Path", name: str):
    """Import an STL and return the object, scaled from millimetres to Blender units.

    The importer is selected from the Blender version rather than discovered by
    exception, and the resulting object is found by diffing the scene rather than
    trusting ``selected_objects[0]`` -- which the legacy script relied on and which
    breaks whenever an import leaves anything else selected.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"STL not found: {path}")

    before = set(bpy.data.objects)
    bpy.ops.object.select_all(action="DESELECT")

    if _is_4x():
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.stl(filepath=str(path))

    created = set(bpy.data.objects) - before
    if not created:
        raise RuntimeError(f"Blender imported no object from {path.name}")

    obj = created.pop()
    obj.name = name
    obj.scale = (MM_TO_BU, MM_TO_BU, MM_TO_BU)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def export_stl(obj, path: "str | Path") -> Path:
    """Export one object to STL, converting back to millimetres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if _is_4x():
        bpy.ops.wm.stl_export(
            filepath=str(path), export_selected_objects=True,
            global_scale=BU_TO_MM,
        )
    else:
        bpy.ops.export_mesh.stl(
            filepath=str(path), use_selection=True, global_scale=BU_TO_MM,
        )
    return path


def ensure_collection(name: str):
    """Get or create a named collection linked to the scene."""
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def move_to_collection(obj, collection) -> None:
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    collection.objects.link(obj)


def set_material(obj, name: str, colour, alpha: float = 1.0):
    """Give an object a simple coloured material, so the viewport reads clearly."""
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True

    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (*colour, alpha)
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = alpha
        if "Roughness" in principled.inputs:
            principled.inputs["Roughness"].default_value = 0.45

    if alpha < 1.0:
        material.blend_method = "BLEND"

    obj.data.materials.clear()
    obj.data.materials.append(material)
    return material
