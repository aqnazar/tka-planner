"""Draw an engine scene in Blender, and nothing else.

This replaces everything ``build.py`` used to decide. The engine computes the plan,
places every node and cuts the bones; this module turns that into Blender objects and
then moves them when a delta arrives. It creates no geometry of its own, runs no
boolean, and holds no plan state.

There are no modifiers in the scene it builds. That is the whole difference: the old
builder left live boolean modifiers on the bones, which is why a slider drag had to
mute them and why playback needed a baked copy of everything. With the resection
committed in the engine, a control change is a matrix assignment and nothing
re-evaluates.

Unit conversion happens here and only here. The engine works in millimetres throughout;
Blender scenes are built in metres because its viewport clipping behaves badly at a
scale of hundreds.
"""

from __future__ import annotations

import numpy as np

from .io import MM_TO_BU, clear_scene, ensure_collection, move_to_collection, set_material

__all__ = ["render_scene", "apply_delta", "apply_scene", "objects", "frame_view"]

# Name to Blender object, for the life of one rendered scene. Blender properties hold
# only primitives, so there is nowhere in a PropertyGroup to keep these; the module
# scope survives between operator invocations for as long as the add-on stays loaded.
_OBJECTS: dict = {}

_COLLECTION_FOR = (
    ("bone", "Bones"),
    ("plane", "Planning"),
    ("axis", "Planning"),
    ("landmark", "Landmarks"),
    ("shell", "Implants"),
    ("insert", "Implants"),
    ("component", "Implants"),
)


def render_scene(scene, *, clear: bool = True) -> dict:
    """Create one Blender object per scene node and return them by name."""
    import bpy

    if clear:
        clear_scene()
    _OBJECTS.clear()

    for name, node in scene.nodes.items():
        mesh = scene.meshes.get(node.mesh_id) if node.mesh_id else None
        if mesh is None:
            continue

        obj = bpy.data.objects.new(name, _to_blender_mesh(mesh, name))
        bpy.context.scene.collection.objects.link(obj)
        move_to_collection(obj, ensure_collection(_collection_for(node)))
        set_material(obj, f"TKA_{_material_key(node)}", node.colour, node.alpha)

        obj.matrix_world = _to_blender_matrix(scene.world(name))
        _set_visible(obj, node.visible)
        _OBJECTS[name] = obj

    ensure_material_shading()
    frame_view()
    return dict(_OBJECTS)


def apply_delta(delta) -> None:
    """Move, re-mesh and re-hide objects from a scene delta.

    A pose change is an assignment, which is why a drag stays responsive: no modifier
    re-evaluates, because there are no modifiers left in this scene.

    Set poses are skipped here. They move a whole set at once, so they are applied
    through :func:`apply_scene`, which recomposes every node's world transform.
    """
    import bpy

    for name, matrix in delta.poses.items():
        if name.endswith("Set"):
            continue
        obj = _OBJECTS.get(name)
        if obj is not None:
            obj.matrix_world = _to_blender_matrix(np.asarray(matrix, dtype=float))

    for name, visible in delta.visibility.items():
        obj = _OBJECTS.get(name)
        if obj is not None:
            _set_visible(obj, visible)

    bpy.context.view_layer.update()


def apply_scene(scene) -> None:
    """Re-sync every object's transform and mesh from the scene.

    Used after a commit or a set-pose change, where a per-node delta would have to
    enumerate a whole set anyway. Nodes the scene has gained since the last render, such
    as a bone shell that only exists after a commit, are created here.
    """
    import bpy

    for name, node in scene.nodes.items():
        mesh = scene.meshes.get(node.mesh_id) if node.mesh_id else None
        if mesh is None:
            continue

        obj = _OBJECTS.get(name)
        if obj is None:
            obj = bpy.data.objects.new(name, _to_blender_mesh(mesh, name))
            bpy.context.scene.collection.objects.link(obj)
            move_to_collection(obj, ensure_collection(_collection_for(node)))
            set_material(obj, f"TKA_{_material_key(node)}", node.colour, node.alpha)
            _OBJECTS[name] = obj
        elif len(obj.data.vertices) != mesh.n_vertices:
            old = obj.data
            obj.data = _to_blender_mesh(mesh, name)
            if old.users == 0:
                bpy.data.meshes.remove(old)

        obj.matrix_world = _to_blender_matrix(scene.world(name))
        _set_visible(obj, node.visible)

    bpy.context.view_layer.update()


def objects() -> dict:
    return dict(_OBJECTS)


# ----------------------------------------------------------------------
# Blender plumbing
# ----------------------------------------------------------------------


def _to_blender_mesh(mesh, name: str):
    """A Blender mesh datablock in metres, from an engine mesh in millimetres."""
    import bpy

    data = bpy.data.meshes.new(name)
    data.from_pydata(
        (mesh.vertices * MM_TO_BU).tolist(),
        [],
        mesh.faces.tolist(),
    )
    data.update()
    return data


def _to_blender_matrix(matrix: np.ndarray):
    """A millimetre 4x4 as a Blender metre matrix."""
    import mathutils

    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, 3] *= MM_TO_BU
    return mathutils.Matrix([[float(v) for v in row] for row in scaled])


def _set_visible(obj, visible: bool) -> None:
    obj.hide_viewport = not visible
    obj.hide_render = not visible


def _collection_for(node) -> str:
    for tag, collection in _COLLECTION_FOR:
        if node.tags.get(tag):
            if tag == "component" and "cutting_block" in node.name:
                return "CuttingBlocks"
            return collection
    return "Planning"


def _material_key(node) -> str:
    """Group materials by colour rather than by object.

    Two hundred landmarks sharing one yellow material is two hundred fewer datablocks,
    and the colour is the only thing the material carries.
    """
    return "_".join(f"{channel:.3f}" for channel in (*node.colour, node.alpha))


def ensure_material_shading() -> None:
    """Show object colours in the viewport.

    Blender's Solid shading defaults to a single flat colour for everything, which hides
    the whole point of colouring the scene by category. Headless runs have no 3D area at
    all, so this is a no-op there rather than an error.
    """
    import bpy

    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.color_type = "MATERIAL"


def frame_view() -> None:
    """Frame the construct, so pressing Plan puts the anatomy on screen."""
    import bpy

    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(
                        window=window, area=area, region=region
                    ):
                        bpy.ops.view3d.view_all(center=False)
                    return


def set_clean_viewport(enabled: bool) -> None:
    """A white background with no grid or overlays, for figures and screenshots."""
    import bpy

    for window in getattr(bpy.context.window_manager, "windows", []):
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type != "VIEW_3D":
                    continue
                space.overlay.show_overlays = not enabled
                space.shading.color_type = "MATERIAL"
                space.shading.background_type = "VIEWPORT" if enabled else "THEME"
                if enabled:
                    space.shading.background_color = (1.0, 1.0, 1.0)
