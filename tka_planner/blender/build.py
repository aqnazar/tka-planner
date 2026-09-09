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
           "resolve_component_meshes", "bake_animation_bones",
           "sync_animation_visibility", "unregister_animation_visibility_handler",
           "set_trial_pose", "set_clean_viewport"]

# Object names the live update path looks for. Keeping them in one place is what lets
# `update_scene` find what `build_scene` made without either holding a reference, which
# matters because the two run in different operator invocations and Blender properties
# cannot carry Python objects between them.
INSERT_SPACER = "TibialInsertSpacer"
# Every boolean the resection uses, by name, so the planning screen can mute them
# all while a control is moving. The exact solver costs seconds on a real
# segmentation, which is far too long to sit inside a slider drag.
RESECTION_MODIFIERS = ("BlockResection", "Resection", "Shell")
CUT_PLANES = {"femoral_distal": "femoral", "tibial_proximal": "tibial"}

# Names of the two animation-ready bones -- see `bake_animation_bones`.
BAKED_FEMUR = "Femur.Baked"
BAKED_TIBIA = "Tibia.Baked"
# Custom properties used to swap between the editing scene (live booleans, muted
# while a control moves, exact once it settles) and the animation scene (two
# plain baked meshes, nothing to re-solve). Set on objects rather than tracked by
# name, so the swap needs no knowledge of which objects the current resection
# mode happened to create.
EDIT_ONLY = "tka_edit_only"
BAKED_BONE = "tka_baked_bone"
# Landmarks are neither: they are relevant while editing and while animated alike, so
# they are not swapped by the rule above at all -- see `sync_animation_visibility`. This
# only marks them for the separate "isolate landmarks" display mode.
LANDMARK = "tka_landmark"

# One colour per category, chosen to stay distinct from every other category at a
# glance: bone grey, implant blue, cutting-block green, shell red, plane cyan, axis
# orange, landmark yellow. Cut bone keeps its own slightly warmer grey rather than the
# implant or shell colour, so a resected surface still reads as *bone*.
BONE_COLOUR = (0.62, 0.62, 0.62)
RESECTED_COLOUR = (0.55, 0.52, 0.50)
IMPLANT_COLOUR = (0.18, 0.38, 0.85)
PLANE_COLOUR = (0.25, 0.80, 0.85)
AXIS_COLOUR = (0.95, 0.55, 0.10)
BLOCK_COLOUR = (0.30, 0.70, 0.35)
SHELL_COLOUR = (0.85, 0.18, 0.18)
# Landmark colour is yellow throughout, per-status shade rather than a different hue, so
# provenance (present/estimated/derived) still reads at a glance without leaving the
# yellow family the category is known by.
LANDMARK_COLOURS = {
    "present": (1.00, 0.85, 0.05),
    "estimated": (0.85, 0.60, 0.05),
    "derived": (1.00, 0.95, 0.45),
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
    cut_with_blocks: bool = True,
    build_bone_shells: bool = True,
    cuts_visible: bool = True,
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
        if animate_flexion:
            obj[EDIT_ONLY] = True
    result.objects["femur"] = femur
    result.objects["tibia"] = tibia

    if show_planes:
        for name, resection in plan.resections.items():
            plane = _make_plane(
                name, resection.point, resection.normal, radius_mm=55.0
            )
            set_material(plane, "CutPlane", PLANE_COLOUR, alpha=0.35)
            move_to_collection(plane, planning)
            if animate_flexion:
                plane[EDIT_ONLY] = True
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
            if animate_flexion:
                axis[EDIT_ONLY] = True
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
        # The cutting blocks are the instrument used to make the cut, not the implant
        # itself, so they belong to the editing scene only -- an animation shows the
        # resected bone and the implant, not the tooling that shaped it.
        if is_block and animate_flexion:
            obj[EDIT_ONLY] = True
        result.objects[component_name] = obj

    if show_landmarks and landmarks is not None:
        marks = ensure_collection("Landmarks")
        made = _make_landmarks(landmarks, marks)
        for obj in made:
            obj[LANDMARK] = True
        result.objects["landmarks"] = made

    # ---- Cutting blocks against the bone ------------------------------
    #
    # Done before any plane resection, because the shells must be taken from the intact
    # bone: a shell intersected with an already-resected femur is missing the surface it
    # is meant to sit on.
    shells = ensure_collection("BoneShells")
    cut_by_block: set = set()
    if cut_with_blocks:
        for bone_obj, bone_name, group in (
            (femur, "Femur", "femoral"),
            (tibia, "Tibia", "tibial"),
        ):
            shell_block = result.objects.get(f"{group}_cutting_block_shell")
            if shell_block is not None and build_bone_shells:
                shell = make_bone_shell(
                    bone_obj, shell_block, f"{bone_name}.Shell", live=live_cuts,
                    visible=cuts_visible,
                )
                set_material(shell, "BoneShell", SHELL_COLOUR, alpha=0.55)
                move_to_collection(shell, shells)
                if animate_flexion:
                    shell[EDIT_ONLY] = True
                result.objects[f"{group}_bone_shell"] = shell
            else:
                result.notes.append(
                    f"{bone_name}: no cutting block shell in the library, "
                    f"no bone shell made"
                )

            block = result.objects.get(f"{group}_cutting_block")
            if block is not None:
                cut_with_block(bone_obj, block, live=live_cuts,
                               visible=cuts_visible)
                set_material(bone_obj, "BoneCut", RESECTED_COLOUR)
                cut_by_block.add(bone_name)
            else:
                result.notes.append(
                    f"{bone_name}: no cutting block in the library, not cut by block"
                )

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
    if perform_cuts:
        for bone_obj, resection_name, keep in (
            (femur, "femoral_distal", "proximal"),
            (tibia, "tibial_proximal", "distal"),
        ):
            # A bone already cut by its block is not cut again by the plane box. The two
            # are alternatives rather than a stack: the box removes everything beyond
            # the plane, so it would swallow the very surfaces the block was shaping and
            # leave the block boolean with nothing to do.
            if bone_obj.name in cut_by_block:
                continue
            resection = plan.resections[resection_name]
            outcome, cutter = resect_bone(
                bone_obj, resection.point, resection.normal,
                keep=keep, name=resection_name,
                live=live_cuts, solver=cut_solver, visible=cuts_visible,
            )
            if cutter is not None:
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
        # A boolean modifier left unapplied re-solves in full on every frame Blender
        # evaluates it -- not only when its cutter moves, the dependency graph does not
        # cache a modifier's result across a frame change at all. That is invisible while
        # editing, where only one frame is ever shown, and it is why playing the flexion
        # animation used to freeze: the tibia rides the pivot with its resection boolean
        # still attached, so every one of 120 frames re-cut a real segmentation. Baking
        # first removes the modifier stack entirely, so playback pulls an
        # already-resolved mesh instead of re-cutting it every frame.
        baked_femur, baked_tibia = bake_animation_bones()
        if baked_femur is not None:
            move_to_collection(baked_femur, bones)
            result.objects["femoral_baked"] = baked_femur
        if baked_tibia is not None:
            move_to_collection(baked_tibia, bones)
            result.objects["tibial_baked"] = baked_tibia

        axis_point, axis_direction = _flexion_axis(
            plan, femoral_frame, landmarks
        )
        moving = [baked_tibia if baked_tibia is not None else tibia]
        moving += [
            obj for name, obj in result.objects.items()
            if name.startswith("tibial") and obj not in moving
            and not obj.get(EDIT_ONLY)
        ]
        # Landmarks on the tibia (and the fibula, which travels with it) are points on
        # that bone, so they have to ride the same rig or they would stay behind in
        # space while the tibia they were measured on swings away underneath them.
        # Femoral landmarks are omitted: the femur never moves, so leaving them
        # unparented is both correct and simpler.
        moving += [
            obj for obj in (result.objects.get("landmarks") or [])
            if obj.name.startswith("tibia.") or obj.name.startswith("fibula.")
        ]
        pivot, trial = add_flexion_animation(
            moving, axis_point_mm=axis_point, axis_direction=axis_direction,
            max_flexion_deg=max_flexion_deg,
        )
        move_to_collection(pivot, planning)
        move_to_collection(trial, planning)
        result.objects["flexion_axis"] = pivot
        result.objects["trial_axis"] = trial
        _register_animation_visibility_handler()
        sync_animation_visibility(bpy.context.scene)
        result.notes.append(
            f"flexion animated 0 to {max_flexion_deg:.0f} degrees about the "
            f"transepicondylar axis, playing back two baked (modifier-free) bones"
        )

    result.collections = {"bones": bones, "planning": planning}
    result.notes.append(
        f"{plan.philosophy} alignment, distal femoral valgus cut "
        f"{plan.distal_femoral_valgus_cut_deg:.1f} deg"
    )
    _frame_view()
    _ensure_material_shading()
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


# Block and shell booleans are always exact, and the panel's solver choice does not
# reach them. The fast solver does not merely lose precision on these pairs, it returns
# confident nonsense: on the real cohort it reduced a 409k-vertex femur to 4k, and
# returned a "shell" 227 mm across -- larger than the bone it was intersected with --
# without raising. A cut that looks like a cut and is not is exactly the failure this
# project keeps running into, so the choice is withheld rather than offered.
BOOLEAN_SOLVER = "EXACT"


def cut_with_block(bone_obj, block_obj, *, live: bool = True,
                   visible: bool = True):
    """Resect a bone with the cutting block itself, as the first pipeline did.

    The plane says what the plan specifies; the block is the instrument that would
    realise it in theatre, and subtracting it shows what the bone would actually look
    like afterwards -- the slot, the captured surfaces, and whatever the block does not
    reach. The two are different by design and are meant to agree; where they do not,
    that is worth seeing.

    Left unapplied by default so the resection follows the block as the plan is adjusted.
    """
    modifier = bone_obj.modifiers.new(name="BlockResection", type="BOOLEAN")
    modifier.operation = "DIFFERENCE"
    modifier.object = block_obj
    modifier.solver = BOOLEAN_SOLVER
    modifier.show_viewport = visible
    if live:
        return True

    bpy.context.view_layer.objects.active = bone_obj
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return True
    except RuntimeError:
        bone_obj.modifiers.remove(modifier)
        return False


def make_bone_shell(bone_obj, shell_obj, name: str, *, live: bool = True,
                    visible: bool = True):
    """The patient-specific shell: the bone intersected with the block's shell.

    A copy of the bone kept only where the shell encloses it, which is the mating
    surface a printed guide would sit on.

    The copy is taken **before** the bone is resected, and that ordering is the whole
    trick. A shell cut from an already-resected femur would be missing the condylar
    surface it is supposed to mate with, so it would fit nothing. Taking it early also
    means the object copy carries an empty modifier stack, and the intersection is the
    only boolean on it.

    The copy shares the bone's mesh datablock rather than duplicating two and a half
    million vertices; a modifier belongs to the object, so the shared mesh is untouched.
    """
    shell = bone_obj.copy()
    shell.name = name
    shell.parent = None
    bpy.context.scene.collection.objects.link(shell)

    modifier = shell.modifiers.new(name="Shell", type="BOOLEAN")
    modifier.operation = "INTERSECT"
    modifier.object = shell_obj
    modifier.solver = BOOLEAN_SOLVER
    modifier.show_viewport = visible
    if live:
        return shell

    bpy.context.view_layer.objects.active = shell
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except RuntimeError:
        shell.modifiers.remove(modifier)
    return shell


CUTTER_SIZE_MM = 400.0


def resect_bone(
    obj, point_mm, normal_mm, *, keep: str, name: str,
    live: bool = False, solver: str = "EXACT", visible: bool = True,
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
    cutter[EDIT_ONLY] = True
    place_cutter(cutter, point_mm, normal_mm, keep=keep)

    def attach(which: str):
        modifier = obj.modifiers.new(name="Resection", type="BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.object = cutter
        modifier.solver = which
        return modifier

    if live:
        attach(solver).show_viewport = visible
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


def _action_fcurves(action):
    """Every F-curve in an action, regardless of Blender's action data model.

    Blender 4.x exposed a flat ``action.fcurves``. 5.0 removed that shim: an action's
    curves live nested under ``action.layers[*].strips[*].channelbags[*].fcurves``, its
    "layered action" model, with no flat accessor left at all. Checking for the
    attribute rather than the version keeps this working if a future release changes
    the threshold, or if a 4.x action was authored under the layered model already.
    """
    if hasattr(action, "fcurves"):
        yield from action.fcurves
        return
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                yield from channelbag.fcurves


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

    The moving parts are parented to a **second** empty, ``TrialAxis``, itself a child of
    this one rather than parented here directly. That is what lets a surgeon manually
    pose the finished, implanted construct -- flexion, a varus/valgus stress, an AP
    drawer -- to check fit and impingement, entirely separately from this scripted
    preview: the two are different objects with different rotations, so a manual pose
    and the keyframed arc never fight over the same property. See `set_trial_pose`.
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

    # ---- The trial rig -------------------------------------------------
    #
    # `base` above is the *minimal* rotation onto the flexion axis, which leaves its own
    # roll about that axis unconstrained -- fine for a pivot that only ever turns about
    # its local X, useless for one a control also has to turn about "anterior" or slide
    # along it. So the trial pivot gets an explicit basis instead: local X the flexion
    # axis, local Y anterior, local Z whatever completes a right-handed frame. That is
    # what lets `set_trial_pose` mean "rotate about local X" as flexion and "translate
    # along local Y" as a drawer test, unconditionally.
    anterior_world = np.array([0.0, -1.0, 0.0])  # LPS: -Y is anterior
    y_axis = anterior_world - np.dot(anterior_world, direction) * direction
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(direction, y_axis)
    trial_rotation = mathutils.Matrix((
        (direction[0], y_axis[0], z_axis[0]),
        (direction[1], y_axis[1], z_axis[1]),
        (direction[2], y_axis[2], z_axis[2]),
    )).to_quaternion()

    trial = bpy.data.objects.new("TrialAxis", None)
    trial.empty_display_type = "PLAIN_AXES"
    trial.empty_display_size = 0.04
    bpy.context.scene.collection.objects.link(trial)
    trial.rotation_mode = "QUATERNION"
    trial.location = _v(axis_point_mm)
    trial.rotation_quaternion = trial_rotation
    # Remembered so `set_trial_pose` can always rebuild the pose from this fixed rest
    # state rather than accumulating drift onto whatever the sliders last left behind.
    trial["tka_trial_rest_location"] = tuple(trial.location)
    trial["tka_trial_rest_y_axis"] = tuple(float(c) for c in y_axis)
    # `parent_inverse` below is computed from this rotation, so `set_trial_pose` must
    # compose onto it rather than replace it -- overwriting `rotation_quaternion` with
    # just the flexion/stress turn discards the anatomical basis, and "zero" on every
    # trial slider would then no longer be the actual rest pose the children were
    # parented against, silently leaving the construct offset from it.
    trial["tka_trial_rest_rotation"] = tuple(trial_rotation)
    bpy.context.view_layer.update()

    trial.parent = pivot
    trial.matrix_parent_inverse = pivot.matrix_world.inverted()

    # Parent with the inverse baked in, which is Blender's keep-transform parenting.
    # Assigning matrix_world afterwards would fight it.
    parent_inverse = trial.matrix_world.inverted()
    for obj in tibial_objects:
        if obj is None:
            continue
        obj.parent = trial
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
        for curve in _action_fcurves(pivot.animation_data.action):
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "BEZIER"

    scene.frame_set(1)
    return pivot, trial


def set_trial_pose(
    *, flexion_deg: float = 0.0, varus_valgus_deg: float = 0.0, drawer_ap_mm: float = 0.0,
) -> bool:
    """Pose the trial rig: a pure rigid transform of the baked, implanted tibia relative
    to the femur, for checking range of motion, stability and impingement on the
    finished construct by hand. Never touches a boolean and never re-plans -- this is
    the same reason the animation plays on the baked pair rather than the live one.

    Flexion turns the trial pivot about its local X (the transepicondylar axis); a
    varus/valgus stress then turns about its *own*, already-flexed local Y, which is
    what a real stress exam is relative to -- the tibia's current position, not the
    femur's fixed frame. The drawer test instead slides along the axis' fixed rest
    direction, because "anterior" for that test means the joint's own anterior, not
    wherever flexion happened to leave the tibia pointing.

    Returns ``False`` if the scene has no trial rig -- Plan was run with the flexion
    animation off, or has not been run at all -- so the caller can report that rather
    than silently do nothing.
    """
    import mathutils

    trial = bpy.data.objects.get("TrialAxis")
    if trial is None:
        return False

    rest_location = trial.get("tka_trial_rest_location")
    rest_y_axis = trial.get("tka_trial_rest_y_axis")
    rest_rotation = trial.get("tka_trial_rest_rotation")
    if rest_location is None or rest_y_axis is None or rest_rotation is None:
        return False

    # `parent_inverse` was baked against `rest_rotation`, so it has to stay the
    # left-most term here: it is what "zero" on every slider must return to, not an
    # arbitrary starting point flexion and stress then turn away from.
    basis = mathutils.Quaternion(tuple(rest_rotation))
    flexion = mathutils.Quaternion((1.0, 0.0, 0.0), np.radians(flexion_deg))
    stress = mathutils.Quaternion((0.0, 1.0, 0.0), np.radians(varus_valgus_deg))
    trial.rotation_quaternion = basis @ flexion @ stress

    offset = mathutils.Vector(tuple(rest_y_axis)) * (drawer_ap_mm * MM_TO_BU)
    trial.location = mathutils.Vector(tuple(rest_location)) + offset
    # Without this, a script reading a child's matrix_world immediately after (as
    # verification does) sees the pre-move value: Blender defers propagating a parent's
    # transform to its children until the next depsgraph evaluation, which an
    # interactive viewport gets for free on its next redraw but a caller with no redraw
    # loop does not.
    bpy.context.view_layer.update()
    return True


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


def _ensure_material_shading() -> None:
    """Make Solid shading actually show the colours this module just assigned.

    Solid shading -- Blender's default -- has its own "Color" setting independent of
    any material, and it is not guaranteed to be ``MATERIAL``: a fresh scene or a
    different startup file can leave it on ``SINGLE``, which paints every object the
    same flat grey regardless of what `set_material` gave it. Bone grey, implant blue,
    shell red and the rest are only ever visible with this set correctly, so it is
    forced here rather than documented as a manual step -- the alternative is a person
    wondering why a colour scheme they were just given is not there.

    Left alone if shading is already in Material Preview or Rendered, both of which
    show real material colours regardless of this setting.
    """
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return  # headless: no screen to touch

    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type != "VIEW_3D":
                continue
            if space.shading.type == "SOLID":
                space.shading.color_type = "MATERIAL"


def set_clean_viewport(enabled: bool) -> None:
    """Flip every open 3D viewport between its normal look and a plain white background
    with the floor grid, axis lines and empty gizmos hidden -- for a screenshot or a
    view in front of someone who does not need Blender's own scaffolding competing with
    the anatomy.

    A display preference, not scene state: there is no single "the" background in
    Blender, each viewport draws its own, so this touches whatever 3D viewports happen
    to be open rather than anything that would need rebuilding or re-planning.
    Restoring hands back Blender's own theme background and default overlays, rather
    than remembering whatever a user had before -- simpler, and what turning it back off
    should reasonably mean.
    """
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return  # headless: no screen to touch

    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type != "VIEW_3D":
                continue
            if space.shading.type == "SOLID":
                space.shading.color_type = "MATERIAL"
            space.shading.background_type = "VIEWPORT" if enabled else "THEME"
            space.shading.background_color = (1.0, 1.0, 1.0)
            space.overlay.show_floor = not enabled
            space.overlay.show_axis_x = not enabled
            space.overlay.show_axis_y = not enabled
            space.overlay.show_axis_z = not enabled
            space.overlay.show_extras = not enabled


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


def set_cuts_visible(visible: bool) -> int:
    """Mute or restore every resection boolean in the scene.

    The exact solver takes seconds on a real segmentation, so it cannot run inside a
    slider drag. Muting the modifiers leaves the planes, components and blocks moving at
    full speed -- which is what alignment is actually judged on -- and the cut is
    recomputed once the controls come to rest.
    """
    changed = 0
    for obj in bpy.data.objects:
        for modifier in obj.modifiers:
            if modifier.type == "BOOLEAN" and modifier.name in RESECTION_MODIFIERS:
                if modifier.show_viewport != visible:
                    modifier.show_viewport = visible
                    changed += 1
    return changed


def bake_animation_bones(bones=("Femur", "Tibia")) -> tuple:
    """Apply every live resection modifier on Femur and Tibia into two plain meshes.

    ``bones`` restricts the bake to the bones actually asked for -- a settle where only
    the tibial cut moved has no reason to also re-copy an unchanged 400k-vertex femur
    into a fresh mesh datablock, which is not free even when its own boolean has
    nothing new to solve. Skipped bones simply keep whatever ``Femur.Baked`` /
    ``Tibia.Baked`` already held.

    Forces the resection modifiers visible for the bake regardless of whether the
    controls are mid-drag and the cuts are currently muted, so the baked shape is always
    the true current cut, then restores whatever visibility the caller had. Reuses the
    ``Femur.Baked`` / ``Tibia.Baked`` objects across calls rather than recreating them, so
    repeated settling during a session does not accumulate orphan meshes.

    Returns the two baked objects, or ``None`` for a bone that is not in the scene (or
    not in ``bones`` and has never been baked before).
    """
    baked = {}
    for source_name, baked_name in (("Femur", BAKED_FEMUR), ("Tibia", BAKED_TIBIA)):
        if source_name not in bones:
            existing = bpy.data.objects.get(baked_name)
            if existing is not None:
                baked[baked_name] = existing
            continue
        source = bpy.data.objects.get(source_name)
        if source is None:
            continue

        resection_modifiers = [
            modifier for modifier in source.modifiers
            if modifier.type == "BOOLEAN" and modifier.name in RESECTION_MODIFIERS
        ]
        saved_visibility = [modifier.show_viewport for modifier in resection_modifiers]
        for modifier in resection_modifiers:
            modifier.show_viewport = True
        bpy.context.view_layer.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated)
        mesh.name = baked_name

        for modifier, visible in zip(resection_modifiers, saved_visibility):
            modifier.show_viewport = visible

        obj = bpy.data.objects.get(baked_name)
        if obj is None:
            obj = bpy.data.objects.new(baked_name, mesh)
            bpy.context.scene.collection.objects.link(obj)
        else:
            old_mesh = obj.data
            obj.data = mesh
            if old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        obj.matrix_world = source.matrix_world.copy()
        # Parent is left alone rather than reset: on the object's first bake it is
        # freshly created and unparented, and `add_flexion_animation` parents it to the
        # pivot once, from `build_scene`. Every later rebake (a control settling) must
        # not touch that parenting, or the tibia would drop out of the animation the
        # moment it was next adjusted.
        obj[BAKED_BONE] = True
        set_material(
            obj, "BoneCut" if resection_modifiers else "Bone",
            RESECTED_COLOUR if resection_modifiers else BONE_COLOUR,
        )
        baked[baked_name] = obj

    return baked.get(BAKED_FEMUR), baked.get(BAKED_TIBIA)


def sync_animation_visibility(scene) -> None:
    """Show the editing objects at rest, and the baked pair everywhere else.

    "At rest" means frame 1 *and* no manual trial pose dialled in -- a surgeon posing
    the implanted construct by hand, at frame 1, still needs to see the baked pair the
    trial pivot actually carries, not the live editing objects sitting underneath it
    unmoved. Registered as a ``frame_change_pre`` handler so scrubbing or playing the
    scripted animation swaps automatically; also called directly after building,
    re-baking, setting a trial pose or toggling isolate-landmarks, so the state is
    correct before the handler next fires. Driven by the ``EDIT_ONLY`` and
    ``BAKED_BONE`` custom properties set when each object was created, so this needs no
    knowledge of which objects the current resection mode happened to make.

    Landmarks sit outside that swap entirely -- relevant whether editing or animated,
    so left showing through both -- except under "isolate landmarks", which inverts the
    whole rule: landmarks are the only thing left visible, everything else hidden,
    for watching how they move relative to each other through flexion without the
    bones and implants in the way.
    """
    properties = getattr(scene, "tka_planner", None)
    trial_active = properties is not None and (
        abs(properties.trial_flexion_deg) > 1e-9
        or abs(properties.trial_varus_valgus_deg) > 1e-9
        or abs(properties.trial_drawer_ap_mm) > 1e-9
    )
    isolate_landmarks = bool(properties is not None and properties.isolate_landmarks)
    at_rest = scene.frame_current == scene.frame_start and not trial_active
    for obj in bpy.data.objects:
        if obj.get(LANDMARK):
            obj.hide_viewport = False
        elif isolate_landmarks:
            obj.hide_viewport = True
        elif obj.get(EDIT_ONLY):
            obj.hide_viewport = not at_rest
        elif obj.get(BAKED_BONE):
            obj.hide_viewport = at_rest


def _register_animation_visibility_handler() -> None:
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == sync_animation_visibility.__name__:
            handlers.remove(handler)
    handlers.append(sync_animation_visibility)


def unregister_animation_visibility_handler() -> None:
    """Drop the frame-change handler. Called when the add-on unregisters, so a Blender
    session that disables and re-enables it does not accumulate duplicate handlers."""
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == sync_animation_visibility.__name__:
            handlers.remove(handler)
