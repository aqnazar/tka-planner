"""Turn a computed plan into a Blender scene.

Blender's role here is to *render* decisions, not to make them. Everything visible in
the scene -- where each component sits, how deep each cut goes, the correction angle --
was computed in :mod:`tka_planner.core` before this module ran. That inversion is what
keeps the planning logic testable without Blender and reproducible with it.

The scene is organised so a viewer can read it at a glance: bones in bone colour, cut
planes as translucent discs, components in implant grey, and the anatomical axes drawn
as lines so the correction is visible rather than merely tabulated.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np

from .io import (
    MM_TO_BU,
    clear_scene,
    ensure_collection,
    import_stl,
    move_to_collection,
    set_material,
)

__all__ = ["build_scene", "SceneResult"]

BONE_COLOUR = (0.88, 0.85, 0.78)
RESECTED_COLOUR = (0.92, 0.72, 0.62)
IMPLANT_COLOUR = (0.62, 0.66, 0.72)
PLANE_COLOUR = (0.20, 0.60, 0.95)
AXIS_COLOUR = (0.95, 0.35, 0.25)


class SceneResult:
    """Handles to what was built, so an operator can report on it."""

    def __init__(self):
        self.objects: dict = {}
        self.collections: dict = {}
        self.notes: list[str] = []


def _v(point_mm) -> tuple:
    """A millimetre point as a Blender-unit tuple."""
    return tuple(float(c) * MM_TO_BU for c in point_mm)


def build_scene(
    *,
    femur_path: "str | Path",
    tibia_path: "str | Path",
    plan,
    femoral_frame,
    tibial_frame,
    landmarks=None,
    show_planes: bool = True,
    show_axes: bool = True,
    show_landmarks: bool = False,
    clear: bool = True,
) -> SceneResult:
    """Build the full scene from an already-computed plan."""
    result = SceneResult()
    if clear:
        clear_scene()

    bones = ensure_collection("Bones")
    planning = ensure_collection("Planning")

    femur = import_stl(femur_path, "Femur")
    tibia = import_stl(tibia_path, "Tibia")
    for obj in (femur, tibia):
        set_material(obj, "Bone", BONE_COLOUR)
        move_to_collection(obj, bones)
    result.objects["femur"] = femur
    result.objects["tibia"] = tibia

    if show_planes:
        for name, resection in plan.resections.items():
            plane = _make_plane(
                name, resection.point, resection.normal, radius_mm=55.0
            )
            set_material(plane, "CutPlane", PLANE_COLOUR, alpha=0.35)
            move_to_collection(plane, planning)
            result.objects[name] = plane

    if show_axes:
        for label, frame, length in (
            ("FemoralMechanicalAxis", femoral_frame, 150.0),
            ("TibialMechanicalAxis", tibial_frame, 150.0),
        ):
            axis = _make_axis(
                label, frame.origin, frame.z_proximal,
                length_mm=length if "Femoral" in label else -length,
            )
            set_material(axis, "Axis", AXIS_COLOUR)
            move_to_collection(axis, planning)
            result.objects[label] = axis

    if show_landmarks and landmarks is not None:
        cloud = _make_landmark_cloud(landmarks)
        if cloud is not None:
            set_material(cloud, "Landmark", (0.95, 0.80, 0.20))
            move_to_collection(cloud, planning)
            result.objects["landmarks"] = cloud

    result.collections = {"bones": bones, "planning": planning}
    result.notes.append(
        f"{plan.philosophy} alignment, distal femoral valgus cut "
        f"{plan.distal_femoral_valgus_cut_deg:.1f} deg"
    )
    _frame_view()
    return result


def _make_plane(name: str, point_mm, normal_mm, *, radius_mm: float):
    """A translucent disc lying on a cut plane."""
    bpy.ops.mesh.primitive_circle_add(
        vertices=64, radius=radius_mm * MM_TO_BU, fill_type="NGON",
        location=_v(point_mm),
    )
    plane = bpy.context.active_object
    plane.name = name
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = _rotation_to(normal_mm)
    return plane


def _make_axis(name: str, origin_mm, direction_mm, *, length_mm: float):
    """A thin cylinder along an anatomical axis, so the correction is visible."""
    direction = np.asarray(direction_mm, dtype=float)
    direction = direction / np.linalg.norm(direction)
    midpoint = np.asarray(origin_mm, dtype=float) + direction * (length_mm / 2.0)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=1.2 * MM_TO_BU,
        depth=abs(length_mm) * MM_TO_BU, location=_v(midpoint),
    )
    axis = bpy.context.active_object
    axis.name = name
    axis.rotation_mode = "QUATERNION"
    axis.rotation_quaternion = _rotation_to(direction)
    return axis


def _make_landmark_cloud(landmarks, radius_mm: float = 2.5):
    """One small sphere per usable landmark, joined into a single object."""
    spheres = []
    for landmark in landmarks:
        if not landmark.is_usable:
            continue
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius_mm * MM_TO_BU, segments=12, ring_count=8,
            location=_v(landmark.position_mm),
        )
        obj = bpy.context.active_object
        obj.name = landmark.id
        spheres.append(obj)

    if not spheres:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for obj in spheres:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = spheres[0]
    if len(spheres) > 1:
        bpy.ops.object.join()

    joined = bpy.context.active_object
    joined.name = "Landmarks"
    return joined


def _rotation_to(direction):
    """Quaternion taking +Z onto ``direction``."""
    import mathutils

    vector = mathutils.Vector(
        (float(direction[0]), float(direction[1]), float(direction[2]))
    )
    if vector.length < 1e-9:
        return mathutils.Quaternion()
    return mathutils.Vector((0.0, 0.0, 1.0)).rotation_difference(vector.normalized())


def _frame_view() -> None:
    """Zoom the viewport to the scene, ignoring failures in headless mode."""
    try:
        for area in bpy.context.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type == "WINDOW":
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_all()
                    return
    except Exception:
        pass  # headless, or no 3D view open
