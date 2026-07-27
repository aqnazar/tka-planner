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

__all__ = ["build_scene", "SceneResult", "add_component", "update_scene",
           "resolve_component_meshes"]

# Object names the live update path looks for. Keeping them in one place is what lets
# `update_scene` find what `build_scene` made without either holding a reference, which
# matters because the two run in different operator invocations and Blender properties
# cannot carry Python objects between them.
INSERT_SPACER = "TibialInsertSpacer"
CUT_PLANES = {"femoral_distal": "femoral", "tibial_proximal": "tibial"}

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
    perform_cuts: bool = True,
    live_cuts: bool = True,
    cut_solver: str = "EXACT",
    animate_flexion: bool = True,
    max_flexion_deg: float = 120.0,
    insert_thickness_mm: float | None = None,
    insert_footprint_mm: tuple | None = None,
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

    # Every part takes the pose of its bone group, because the library gives all parts
    # of a bone one shared CAD origin. That is what puts a cutting block exactly on its
    # implant, and the insert exactly on the tray, without any per-part offsets.
    blocks = ensure_collection("CuttingBlocks")
    for component_name, spec in (components or {}).items():
        group = spec.get("group", "femoral")
        pose = plan.components.get(f"{group}_component")
        if pose is None or not Path(spec["path"]).is_file():
            result.notes.append(f"{component_name}: mesh not found, skipped")
            continue
        obj = add_component(
            spec["path"], component_name,
            pose=pose, scale_factor=spec.get("scale", 1.0), group=group,
            source_ml_mm=spec.get("source_ml"),
        )
        is_block = "cutting_block" in component_name
        set_material(
            obj,
            "BlockShell" if is_block and "shell" in component_name
            else "Block" if is_block else "Implant",
            BLOCK_COLOUR if is_block else IMPLANT_COLOUR,
            alpha=0.45 if "shell" in component_name else 1.0,
        )
        move_to_collection(obj, blocks if is_block else implants)
        result.objects[component_name] = obj

    if show_landmarks and landmarks is not None:
        marks = ensure_collection("Landmarks")
        result.objects["landmarks"] = _make_landmarks(landmarks, marks)

    # ---- The insert -----------------------------------------------------
    #
    # Drawn before the cuts so it joins the tibial group that flexes.
    if insert_thickness_mm is not None and "tibial_component" in plan.components:
        footprint = insert_footprint_mm or (70.0, 48.0)
        spacer = make_insert_spacer(
            plan.components["tibial_component"],
            ml_mm=footprint[0], ap_mm=footprint[1],
            thickness_mm=insert_thickness_mm,
        )
        set_material(spacer, "Insert", (0.90, 0.90, 0.82), alpha=0.65)
        move_to_collection(spacer, implants)
        result.objects["tibial_insert_spacer"] = spacer

    # ---- Resection ---------------------------------------------------
    cutters = {}
    if perform_cuts:
        for bone_obj, resection_name, keep in (
            (femur, "femoral_distal", "proximal"),
            (tibia, "tibial_proximal", "distal"),
        ):
            resection = plan.resections[resection_name]
            outcome, cutter = resect_bone(
                bone_obj, resection.point, resection.normal,
                keep=keep, name=resection_name,
                live=live_cuts, solver=cut_solver,
            )
            if cutter is not None:
                cutters[resection_name] = cutter
                move_to_collection(cutter, planning)
            if outcome is False:
                result.notes.append(f"{resection_name}: boolean failed, bone left whole")
            elif outcome == "fast":
                result.notes.append(
                    f"{resection_name}: exact boolean failed, used the fast solver "
                    f"(the resulting geometry differs)"
                )
            set_material(bone_obj, "BoneCut", RESECTED_COLOUR)
        if live_cuts:
            result.notes.append(
                "cuts are live: the resection follows the plan as it is adjusted"
            )

    # ---- Flexion ------------------------------------------------------
    if animate_flexion:
        axis_point, axis_direction = _flexion_axis(
            plan, femoral_frame, landmarks
        )
        moving = [tibia]
        moving += [
            obj for name, obj in result.objects.items()
            if name.startswith("tibial")
        ]
        # The tibial discard box travels with the tibia. Left behind, a live boolean
        # would carve a fixed plane through a bone that is moving, and the tibia would
        # appear to melt away as it flexed.
        if "tibial_proximal" in cutters:
            moving.append(cutters["tibial_proximal"])
        pivot = add_flexion_animation(
            moving, axis_point_mm=axis_point, axis_direction=axis_direction,
            max_flexion_deg=max_flexion_deg,
        )
        move_to_collection(pivot, planning)
        result.objects["flexion_axis"] = pivot
        result.notes.append(
            f"flexion animated 0 to {max_flexion_deg:.0f} degrees about the "
            f"transepicondylar axis"
        )

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
    group: str = "femoral",
    source_ml_mm: float | None = None,
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
    obj = import_stl(path, name)

    # Remembered on the object itself, so a later live update can re-pose this part
    # without being told anything about it. Blender custom properties survive between
    # operator runs, which Python references do not.
    obj["tka_scale"] = float(scale_factor)
    obj["tka_group"] = str(group)
    if source_ml_mm:
        # The width the exported mesh was drawn at. Because the library is one master
        # geometry under a uniform scale, a size change needs no re-import at all: the
        # mesh already in the scene is the same part, and rescaling it by the ratio of
        # widths *is* the new size rather than an approximation to it.
        obj["tka_source_ml"] = float(source_ml_mm)

    seat_component(obj, pose, scale_factor=scale_factor)
    return obj


def seat_component(obj, pose, *, scale_factor: float = 1.0) -> None:
    """Put a component's CAD origin on a cut, at a parametric scale.

    Split out of :func:`add_component` because the live controls re-pose parts that are
    already in the scene. Both paths must compute the world matrix identically, or a
    component would shift the first time a slider moved.
    """
    import mathutils

    pose = np.asarray(pose, dtype=float)
    matrix = mathutils.Matrix([[float(v) for v in row] for row in pose])
    # Scale about the CAD origin, so the parametric factor never shifts the seating.
    matrix = matrix @ mathutils.Matrix.Scale(scale_factor, 4)
    matrix.translation = mathutils.Vector(
        tuple(float(c) * MM_TO_BU for c in pose[:3, 3])
    )
    obj.matrix_world = matrix


def _flexion_axis(plan, femoral_frame, landmarks):
    """Where the knee hinges, and about what.

    The flexion axis is taken as the **transepicondylar axis**, through the midpoint of
    the epicondyles. That is the standard approximation, and the reason it works is
    geometric: the femoral condyles are close to circular in the sagittal plane and the
    epicondyles sit near the centres of those circles, so the tibia rides around them at
    a nearly constant radius and stays in contact through the arc.

    Using the posterior condyles instead puts the axis on the condylar *surface* rather
    than at its centre of curvature, roughly two centimetres out. The joint then swings
    apart as it flexes -- the component separation grew from 33 mm to 93 mm over 120
    degrees before this was corrected.

    Simplifications worth stating: the axis is held fixed, so femoral rollback and the
    screw-home rotation near extension are not modelled, and the tibia is treated as a
    rigid body hinging in one plane.
    """
    direction = femoral_frame.y_patient_left

    for pair in (
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus"),
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_prominence"),
    ):
        if landmarks is not None and landmarks.available(*pair):
            lateral, medial = landmarks.require(*pair)
            return (np.asarray(lateral) + np.asarray(medial)) / 2.0, direction

    return plan.components["femoral_component"][:3, 3], direction


CUTTER_SIZE_MM = 400.0


def resect_bone(
    obj, point_mm, normal_mm, *, keep: str, name: str,
    live: bool = False, solver: str = "EXACT",
):
    """Cut a bone along a resection plane, keeping one side.

    The cut is made with a box large enough to swallow the discarded side, rather than
    with the cutting block. The block is the instrument that would realise the cut in
    theatre and is shown as such; the plane is what the plan actually specifies, so
    cutting with it shows the plan rather than the tooling.

    ``keep="proximal"`` retains the bone above the plane, which is the femur; the tibia
    keeps ``"distal"``.

    With ``live=True`` the boolean is left unapplied and the cutter box is kept, hidden,
    in the scene. Moving that box then re-cuts the bone by itself, which is what lets a
    resection-depth slider change the bone in the viewport rather than only the plane.
    The cost is that the solver re-runs on every change, so on a dense segmentation the
    exact solver may not keep up with a drag -- hence the solver choice, and hence the
    panel's ability to turn live cutting off and still move everything else.

    Returns ``(outcome, cutter)``. ``outcome`` is ``True``, ``"fast"`` when the exact
    solver failed and the fast one stood in, or ``False`` when both failed and the bone
    was left whole. ``cutter`` is the retained box in live mode and ``None`` otherwise.
    """
    normal = np.asarray(normal_mm, dtype=float)
    normal = normal / np.linalg.norm(normal)
    discard = -normal if keep == "proximal" else normal

    bpy.ops.mesh.primitive_cube_add(size=CUTTER_SIZE_MM * MM_TO_BU)
    cutter = bpy.context.active_object
    cutter.name = f"_{name}_cutter"
    cutter["tka_cutter_for"] = name
    cutter["tka_keep"] = keep
    place_cutter(cutter, point_mm, normal_mm, keep=keep)

    def attach(which: str):
        modifier = obj.modifiers.new(name="Resection", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = which
        return modifier

    if live:
        attach(solver)
        cutter.hide_viewport = True
        cutter.hide_render = True
        cutter.display_type = "WIRE"
        return True, cutter

    modifier = attach("EXACT")
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        applied = True
    except RuntimeError:
        # Exact can fail on a non-manifold segmentation; fall back and say so, since
        # the fast solver produces different geometry.
        obj.modifiers.remove(modifier)
        modifier = attach("FAST")
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            applied = "fast"
        except RuntimeError:
            obj.modifiers.remove(modifier)
            applied = False

    bpy.data.objects.remove(cutter, do_unlink=True)
    return applied, None


def place_cutter(cutter, point_mm, normal_mm, *, keep: str) -> None:
    """Put the discard box against a resection plane.

    The box's near face lies on the plane and its bulk sits on the side being thrown
    away, so the boolean difference leaves exactly the retained bone.

    The pose is assigned as a world matrix rather than as a location and rotation,
    because the tibial cutter is parented to the flexion pivot so that the cut travels
    with the bone through the arc. Setting ``location`` on a parented object would place
    it in the parent's space and slide the cut through the tibia.
    """
    import mathutils

    normal = np.asarray(normal_mm, dtype=float)
    normal = normal / np.linalg.norm(normal)
    discard = -normal if keep == "proximal" else normal
    centre = np.asarray(point_mm, dtype=float) + discard * (CUTTER_SIZE_MM / 2.0)

    cutter.rotation_mode = "QUATERNION"
    cutter.matrix_world = mathutils.Matrix.LocRotScale(
        mathutils.Vector(_v(centre)), _rotation_to(discard), (1.0, 1.0, 1.0)
    )


def add_flexion_animation(
    tibial_objects,
    *,
    axis_point_mm,
    axis_direction,
    max_flexion_deg: float = 120.0,
    frames: int = 120,
):
    """Animate the tibia and its components flexing about the knee.

    The femur is held still and the tibia swings, which is how a planning screen shows
    range of motion: the femoral component is the reference and the articulation is what
    moves.

    Rotation is about the **transepicondylar axis**, the standard approximation to the
    knee's flexion axis, positioned through the posterior condyles. That is better than
    a midpoint between the two bones and no harder: the femoral condyles are very nearly
    circular in the sagittal plane and the tibia rides around their centre. It ignores
    the femoral rollback and the screw-home rotation that accompany real flexion, which
    is a stated simplification rather than an oversight.

    The moving parts are parented to an empty at the axis, so one keyframed rotation
    carries the tibia and every tibial component together and their relative placement
    cannot drift.
    """
    import mathutils

    pivot = bpy.data.objects.new("FlexionAxis", None)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(pivot)

    direction = np.asarray(axis_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Orient the empty so its local X lies along the flexion axis, then keep it in
    # quaternion mode throughout. Switching an object from quaternion to euler does not
    # convert the value -- the euler simply reads whatever it held, usually identity --
    # so orienting by quaternion and then keyframing the euler silently discards the
    # orientation and the animation does nothing.
    pivot.rotation_mode = "QUATERNION"
    base = mathutils.Vector((1.0, 0.0, 0.0)).rotation_difference(
        mathutils.Vector(tuple(float(c) for c in direction))
    )
    pivot.location = _v(axis_point_mm)
    pivot.rotation_quaternion = base
    bpy.context.view_layer.update()

    # Parent with the inverse baked in, which is Blender's keep-transform parenting.
    # Assigning matrix_world afterwards would fight it.
    parent_inverse = pivot.matrix_world.inverted()
    for obj in tibial_objects:
        if obj is None:
            continue
        obj.parent = pivot
        obj.matrix_parent_inverse = parent_inverse

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames

    for frame, fraction in ((1, 0.0), (frames // 2, 1.0), (frames, 0.0)):
        angle = np.radians(max_flexion_deg * fraction)
        # Flexion is a turn about the pivot's own local X, so it composes on the right.
        pivot.rotation_quaternion = (
            base @ mathutils.Quaternion((1.0, 0.0, 0.0), float(angle))
        )
        pivot.keyframe_insert(data_path="rotation_quaternion", frame=frame)

    if pivot.animation_data and pivot.animation_data.action:
        for curve in pivot.animation_data.action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "BEZIER"

    scene.frame_set(1)
    return pivot


def make_insert_spacer(pose, *, ml_mm: float, ap_mm: float, thickness_mm: float):
    """A slab standing in for the plastic insert, at the planned thickness.

    The parametric insert model is not in the library yet, so what matters now is the
    number and the space it occupies: a block of the requested thickness sitting on the
    tibial cut, in the tray's own footprint, so the joint gap it fills is visible rather
    than only tabulated.

    Built from eight vertices rather than a scaled primitive because the thickness has to
    read exactly off the ruler. The base sits at local ``z = 0``, which is the tibial
    CAD origin and therefore the cut surface, so the slab's top face is the articulating
    surface the femoral component meets.
    """
    mesh = bpy.data.meshes.new(INSERT_SPACER)
    half_ml, half_ap = ml_mm / 2.0, ap_mm / 2.0
    # Tibial parts use local +X patient-left, +Y posterior, +Z proximal.
    corners = [
        (x * MM_TO_BU, y * MM_TO_BU, z * MM_TO_BU)
        for z in (0.0, thickness_mm)
        for x, y in ((-half_ml, -half_ap), (half_ml, -half_ap),
                     (half_ml, half_ap), (-half_ml, half_ap))
    ]
    faces = [
        (0, 1, 2, 3), (7, 6, 5, 4),
        (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    mesh.from_pydata(corners, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(INSERT_SPACER, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["tka_group"] = "tibial"
    obj["tka_ml_mm"] = float(ml_mm)
    obj["tka_ap_mm"] = float(ap_mm)
    seat_component(obj, pose)
    return obj


def update_scene(
    plan,
    *,
    insert_thickness_mm: float | None = None,
    live_cuts: bool = True,
    implant_ml_mm: float | None = None,
) -> None:
    """Re-pose an existing scene from a re-planned :class:`SurgicalPlan`.

    This is the path every slider takes. Nothing is imported, created or deleted: the cut
    planes, the components, the discard boxes and the insert slab are all already in the
    scene and only their transforms change. Planning itself is a few dozen numpy
    operations, so the whole round trip is fast enough to run on each change of a value
    rather than on a button press.

    Rebuilding instead would mean re-reading the meshes, re-estimating the landmarks and
    re-applying the booleans -- seconds of work, and it would also discard the user's
    viewport selection on every drag.

    A change of implant size needs no re-import either. ``implant_ml_mm`` re-derives each
    part's scale from the width its own mesh was exported at, and because the library is
    one master geometry under a uniform scale, the part already in the scene at the new
    factor *is* the new size rather than a stand-in for it.
    """
    scene_objects = bpy.data.objects

    # The plan is expressed in the extended pose, so the arc has to be at extension for
    # the world transforms it carries to mean anything.
    if bpy.context.scene.frame_current != bpy.context.scene.frame_start:
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)

    for name, resection in plan.resections.items():
        plane = scene_objects.get(name)
        if plane is not None:
            plane.rotation_mode = "QUATERNION"
            plane.matrix_world = _plane_matrix(resection.point, resection.normal)

        cutter = scene_objects.get(f"_{name}_cutter")
        if cutter is not None and live_cuts:
            place_cutter(
                cutter, resection.point, resection.normal,
                keep=cutter.get("tka_keep", "proximal"),
            )

    for obj in scene_objects:
        group = obj.get("tka_group")
        if group is None or obj.name == INSERT_SPACER:
            continue
        pose = plan.components.get(f"{group}_component")
        if pose is None:
            continue
        source_ml = obj.get("tka_source_ml")
        scale = (
            implant_ml_mm / source_ml
            if implant_ml_mm and source_ml
            else obj.get("tka_scale", 1.0)
        )
        seat_component(obj, pose, scale_factor=scale)

    spacer = scene_objects.get(INSERT_SPACER)
    if spacer is not None:
        if insert_thickness_mm is None:
            spacer.hide_viewport = True
        else:
            spacer.hide_viewport = False
            _restretch_spacer(spacer, insert_thickness_mm)
            seat_component(spacer, plan.components["tibial_component"])


def _restretch_spacer(spacer, thickness_mm: float) -> None:
    """Set the slab's height in place, leaving its footprint alone.

    The mesh is built base-first, so vertices 0-3 lie on the cut at ``z = 0`` and 4-7
    are the top face. Only the top four move.
    """
    height = thickness_mm * MM_TO_BU
    for vertex in spacer.data.vertices[4:]:
        vertex.co.z = height
    spacer.data.update()


def _plane_matrix(point_mm, normal_mm):
    """World matrix putting a disc on a cut plane."""
    import mathutils

    return mathutils.Matrix.LocRotScale(
        mathutils.Vector(_v(point_mm)), _rotation_to(normal_mm), (1.0, 1.0, 1.0)
    )


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
    if not (library / label).is_dir():
        return {}
    is_left = str(side).lower().startswith("l")

    # The exported folders are not uniformly complete -- L4, for instance, holds the
    # implants but no cutting blocks. Rather than drop those parts, search outwards from
    # the nearest size for a folder that does export them, and scale from whichever size
    # was found. Because the library is one master geometry under a uniform scale, a part
    # taken from a neighbouring size and rescaled is the same part.
    order = sorted(
        (lbl for lbl in chart.labels if (library / lbl).is_dir()),
        key=lambda lbl: abs(chart.label_parameter(lbl)
                            - chart.label_parameter(label)),
    )

    # Filenames are matched on content rather than exact spelling, because the exported
    # library is not internally consistent: the M2 folder holds
    # "Implant(Femoral)Left_M2.stl" while L4 holds "Implant_Femoral_Left.stl" -- same
    # component, different separators and no size suffix. Normalising to alphanumerics
    # makes both resolve.
    def normalise(name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())

    def find_in(folder_label, *, must, side_token, exclude=()):
        files = [
            (normalise(p.name), p)
            for p in (library / folder_label).glob("*.stl")
        ]
        matches = [
            p for key, p in files
            if all(term in key for term in must)
            and not any(term in key for term in exclude)
        ]
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

    def find(*, must, side_token, exclude=()):
        """Search outwards from the requested size until the part turns up."""
        for folder_label in order:
            found = find_in(
                folder_label, must=must, side_token=side_token, exclude=exclude
            )
            if found is not None:
                return found, folder_label
        return None, None

    resolved = {}

    # Every part belonging to a bone shares that bone's CAD origin, so they all take the
    # same pose. That is the library's own guarantee and it is what makes a cutting block
    # land exactly on its implant.
    wanted = (
        ("femoral_component", ("implant", "femoral"), "femoral", True),
        ("femoral_cutting_block", ("cuttingblock", "femoral"), "femoral", False),
        ("femoral_cutting_block_shell", ("cuttingblock", "femoral", "shell"),
         "femoral", False),
        ("tibial_component", ("tibial", "plate"), "tibial", False),
        ("tibial_insert", ("tibial", "insert"), "tibial", False),
        ("tibial_cutting_block", ("cuttingblock", "tibia"), "tibial", False),
        ("tibial_cutting_block_shell", ("cuttingblock", "tibia", "shell"),
         "tibial", False),
    )
    for name, must, group, sided in wanted:
        path, found_label = find(
            must=must,
            side_token=("left" if is_left else "right") if sided else None,
            exclude=() if "shell" in name else ("shell",),
        )
        if path is not None:
            source_ml = chart.value_at(
                "femur_ML", chart.label_parameter(found_label)
            )
            resolved[name] = {
                "path": str(path),
                "scale": sizing.implant_ml_mm / source_ml,
                "group": group,
                "source_size": found_label,
                "source_ml": source_ml,
            }

    # The plastic insert is sometimes exported without "tibial" in its name.
    if "tibial_insert" not in resolved:
        path, found_label = find(
            must=("insert",), side_token=None, exclude=("shell",)
        )
        if path is not None:
            source_ml = chart.value_at(
                "femur_ML", chart.label_parameter(found_label)
            )
            resolved["tibial_insert"] = {
                "path": str(path),
                "scale": sizing.implant_ml_mm / source_ml,
                "group": "tibial",
                "source_size": found_label,
                "source_ml": source_ml,
            }
    return resolved
