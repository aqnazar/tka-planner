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

__all__ = ["build_scene", "SceneResult", "add_component",
           "resolve_component_meshes"]

BONE_COLOUR = (0.88, 0.85, 0.78)
RESECTED_COLOUR = (0.92, 0.72, 0.62)
IMPLANT_COLOUR = (0.62, 0.66, 0.72)
PLANE_COLOUR = (0.20, 0.60, 0.95)
AXIS_COLOUR = (0.95, 0.35, 0.25)
BLOCK_COLOUR = (0.35, 0.72, 0.45)
# Landmark colour encodes provenance, so a glance says which were picked and which
# the estimator guessed.
LANDMARK_COLOURS = {
    "present": (0.20, 0.80, 0.35),
    "estimated": (0.95, 0.75, 0.15),
    "derived": (0.35, 0.65, 0.95),
}


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
    components: dict | None = None,
    show_planes: bool = True,
    show_axes: bool = True,
    show_landmarks: bool = False,
    clear: bool = True,
) -> SceneResult:
    """Build the full scene from an already-computed plan.

    ``components`` maps a component name to ``{"path": ..., "scale": ..., "seat": ...}``
    so the implants appear seated on their cut planes. Omit it to show the anatomy and
    the planned cuts alone.
    """
    result = SceneResult()
    if clear:
        clear_scene()

    bones = ensure_collection("Bones")
    planning = ensure_collection("Planning")
    implants = ensure_collection("Implants")

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

    for component_name, spec in (components or {}).items():
        pose = plan.components.get(component_name)
        if pose is None or not Path(spec["path"]).is_file():
            result.notes.append(f"{component_name}: mesh not found, skipped")
            continue
        obj = add_component(
            spec["path"], component_name,
            pose=pose,
            scale_factor=spec.get("scale", 1.0),
        )
        colour = BLOCK_COLOUR if "cutting_block" in component_name else IMPLANT_COLOUR
        set_material(obj, "Block" if "cutting_block" in component_name else "Implant",
                     colour)
        move_to_collection(obj, implants)
        result.objects[component_name] = obj

    if show_landmarks and landmarks is not None:
        marks = ensure_collection("Landmarks")
        result.objects["landmarks"] = _make_landmarks(landmarks, marks)

    result.collections = {"bones": bones, "planning": planning}
    result.notes.append(
        f"{plan.philosophy} alignment, distal femoral valgus cut "
        f"{plan.distal_femoral_valgus_cut_deg:.1f} deg"
    )
    _frame_view()
    return result


def add_component(
    path: "str | Path",
    name: str,
    *,
    pose,
    scale_factor: float = 1.0,
):
    """Load an implant component, scale it parametrically, and seat it on a cut plane.

    ``scale_factor`` is what makes the family parametric: the exported library is a
    single master geometry under a uniform scale (every femoral implant STL shares the
    same aspect ratio to four decimal places), so any size between or beyond the twelve
    published ones is that master at the appropriate factor.

    Placement uses the component's **native CAD origin**, untouched. The library is
    built so that every part belonging to a bone shares one origin sitting on the cut
    surface, which is what makes a cutting block and its implant coincide exactly when
    given the same pose.

    An earlier version re-originned each mesh to its bounding box instead, seating the
    top or bottom face on the plane. That is wrong for these parts and badly so: a
    femoral component wraps the distal femur, so its highest points are the anterior
    flange and the posterior condyles while its mating surface sits in between. Seating
    by bounding box put the femoral component 42 mm below its cut and 32 mm off to the
    side, and the tibial tray 40 mm above its own.
    """
    import mathutils

    obj = import_stl(path, name)

    pose = np.asarray(pose, dtype=float)
    matrix = mathutils.Matrix(
        [[float(v) for v in row] for row in pose]
    )
    # Scale about the CAD origin, so the parametric factor never shifts the seating.
    matrix = matrix @ mathutils.Matrix.Scale(scale_factor, 4)
    matrix.translation = mathutils.Vector(
        tuple(float(c) * MM_TO_BU for c in pose[:3, 3])
    )
    obj.matrix_world = matrix
    return obj


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


def _make_landmarks(landmarks, collection, radius_mm: float = 2.5):
    """One named sphere per usable landmark.

    Kept as separate objects rather than joined into a single mesh, so each carries its
    anatomical id in the outliner and can be clicked, hidden and checked individually.
    Joining them was faster to draw and useless to inspect.
    """
    created = []
    for landmark in landmarks:
        if not landmark.is_usable:
            continue
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius_mm * MM_TO_BU, segments=12, ring_count=8,
            location=_v(landmark.position_mm),
        )
        obj = bpy.context.active_object
        obj.name = landmark.id
        obj.show_name = True
        set_material(obj, f"Landmark_{landmark.status.value}",
                     LANDMARK_COLOURS.get(landmark.status.value, (0.9, 0.8, 0.2)))
        move_to_collection(obj, collection)
        created.append(obj)
    return created


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


def resolve_component_meshes(
    library: "str | Path",
    *,
    chart,
    sizing,
    side: str,
) -> dict:
    """Find the implant meshes for a plan and the scale that realises its exact size.

    The library holds twelve discrete exports, and the plan asks for a continuous size.
    Because those exports are one master geometry under a uniform scale, the nearest
    discrete mesh multiplied by the ratio of the two widths *is* the requested size --
    not an approximation to it. When the plan lands on a chart size the ratio is one and
    the mesh is used untouched.
    """
    library = Path(library)
    label = sizing.nearest_discrete_size
    folder = library / label
    if not folder.is_dir():
        return {}

    discrete_ml = chart.value_at("femur_ML", chart.label_parameter(label))
    scale = sizing.implant_ml_mm / discrete_ml
    is_left = str(side).lower().startswith("l")

    # Filenames are matched on content rather than exact spelling, because the exported
    # library is not internally consistent: the M2 folder holds
    # "Implant(Femoral)Left_M2.stl" while L4 holds "Implant_Femoral_Left.stl" -- same
    # component, different separators and no size suffix. Normalising to alphanumerics
    # makes both resolve.
    def normalise(name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())

    files = [(normalise(p.name), p) for p in folder.glob("*.stl")]

    def find(*, must: tuple[str, ...], side_token: str | None) -> Path | None:
        matches = [p for key, p in files if all(term in key for term in must)]
        if side_token:
            sided = [p for p in matches if side_token in normalise(p.name)]
            other = "right" if side_token == "left" else "left"
            sided = [p for p in sided if other not in normalise(p.name)]
            if sided:
                return sided[0]
            # Fall back to a laterality-neutral export, e.g. "(L&R)".
            neutral = [
                p for p in matches
                if "left" not in normalise(p.name) and "right" not in normalise(p.name)
            ]
            return neutral[0] if neutral else None
        return matches[0] if matches else None

    resolved = {}
    femoral = find(
        must=("implant", "femoral"), side_token="left" if is_left else "right"
    )
    if femoral is not None:
        resolved["femoral_component"] = {"path": str(femoral), "scale": scale}

    tibial = find(must=("tibial", "plate"), side_token=None)
    if tibial is not None:
        resolved["tibial_component"] = {"path": str(tibial), "scale": scale}

    # Cutting blocks share their implant's CAD origin, so the same pose seats them.
    for name, must in (
        ("femoral_cutting_block", ("cuttingblock", "femoral")),
        ("tibial_cutting_block", ("cuttingblock", "tibia")),
    ):
        block = find(must=must, side_token=None)
        if block is not None and "shell" not in normalise(block.name):
            resolved[name] = {"path": str(block), "scale": scale}
    return resolved
