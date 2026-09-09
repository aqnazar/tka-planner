"""Blender add-on: the planning screen.

Pick a patient folder, choose an alignment philosophy, press Plan. The bones load, the
resection planes and mechanical axes appear in the viewport, and the sizing and
positioning numbers appear in the sidebar next to them. From there every cut can be
adjusted by hand and the scene follows as the value changes.

All the work happens in :mod:`tka_planner.core`, which does not import ``bpy``. This
add-on loads meshes, calls the planner, and draws the answer. That separation is why the
same computation can run headless in the test suite, and why nothing shown here can
disagree with what the command line produces. Manual adjustments are no exception: they
are inputs to the planner, not edits to the scene, so an adjusted plan still re-derives
from its own file.

Install by building ``tka_planner.zip`` with ``build_addon.py`` and installing it from
disk, then press N in the 3D viewport and open the "TKA" tab.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy.utils import register_class, unregister_class

bl_info = {
    "name": "TKA Planner",
    "author": "Satbayev University",
    "version": (0, 2, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > TKA",
    "description": (
        "Open pre-operative planning for total knee arthroplasty. "
        "Research and demonstration only - not a medical device."
    ),
    "category": "Object",
}


def _ensure_package_importable() -> None:
    """Put the repository and its vendored wheels on ``sys.path``.

    Blender runs an add-on from inside the package, so the parent directory has to be
    added explicitly unless the package was pip-installed into Blender's own Python.

    ``vendor/`` carries the boolean kernel built for Blender's own Python version.
    Blender ships neither pip nor manifold3d, and installing into its site-packages
    needs administrator rights on Windows, so the wheel is unpacked beside the
    repository instead. See ``vendor/README.md``.
    """
    repository = Path(__file__).resolve().parent.parent.parent
    for path in (repository, repository / "vendor"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


# ----------------------------------------------------------------------
# The session
# ----------------------------------------------------------------------
#
# Everything expensive about a case -- reading two STLs, estimating a full landmark set,
# building both frames, measuring and sizing -- is done once when Plan is pressed and
# kept here. Re-planning then costs a few dozen numpy operations on a handful of points,
# which is what makes a slider able to move the whole construct as it is dragged.
#
# It lives at module scope because Blender properties hold only primitives: there is
# nowhere in a PropertyGroup to put a mesh or a LandmarkSet. The dictionary survives
# between operator invocations for exactly as long as the add-on stays loaded, and a
# stale entry is harmless because Plan overwrites it.

_SESSION: dict = {}


def _session():
    return _SESSION.get("current")


# ----------------------------------------------------------------------
# The panel's controls
# ----------------------------------------------------------------------
#
# The thirteen manual adjustments and the three trial controls are named here only so
# the panel can enumerate them. What they mean, and what they do to a plan, belongs to
# `tka_planner.session`, which is where the values are actually sent.

_ensure_package_importable()
from tka_planner.session import ADJUSTMENT_FIELDS  # noqa: E402

ADJUSTMENT_PROPERTIES = ADJUSTMENT_FIELDS

TRIAL_PROPERTIES = (
    "trial_flexion_deg",
    "trial_varus_valgus_deg",
    "trial_drawer_ap_mm",
)

# Panel properties that are plan inputs rather than adjustments. Sent to the session
# alongside the adjustments on every change.
PLAN_PROPERTIES = (
    "philosophy",
    "size_override",
    "tibial_resection_mm",
    "insert_thickness_mm",
    "use_insert",
    "resection_mode",
    "build_bone_shells",
    "show_landmarks",
    "isolate_landmarks",
)


# Blender keeps no reference to the strings an EnumProperty items callback returns, so a
# list built fresh each call can be garbage collected while the menu is open and show
# corrupt entries. Caching it at module scope is the documented way round that.
_SIZE_ITEMS: list = []


def _size_items(self, context):
    if not _SIZE_ITEMS:
        _SIZE_ITEMS.append(("", "Solve from anatomy", "Size the implant from the "
                            "measured femoral width"))
        try:
            _ensure_package_importable()
            from tka_planner.core.sizing import load_size_chart
            from tka_planner.session import find_size_chart

            for label in load_size_chart(find_size_chart()).labels:
                _SIZE_ITEMS.append((label, label, f"Force size {label}"))
        except Exception:
            traceback.print_exc()
    return _SIZE_ITEMS


# ----------------------------------------------------------------------
# Driving the session
# ----------------------------------------------------------------------


def _controls_from(properties) -> dict:
    """Every panel value the session understands, as one dictionary."""
    return {
        name: getattr(properties, name)
        for name in (*ADJUSTMENT_PROPERTIES, *PLAN_PROPERTIES)
    }


def _replan(properties, context) -> None:
    """Push the panel's control values into the session and draw the result.

    Deliberately silent when there is nothing to update: the properties are also written
    when a blend file loads and when Plan itself runs, and neither should trigger a
    half-built scene. Any failure lands in the panel's status line rather than being
    raised, because an exception inside a property callback leaves Blender's UI in a bad
    state and says nothing useful to the person holding the slider.

    No boolean runs on this path. That is what makes a drag responsive, and it is why
    the mute-and-settle debounce this function used to carry is gone: there is nothing
    left to mute.
    """
    session = _session()
    if session is None or not properties.has_scene:
        return

    try:
        from tka_planner.blender import render

        delta = session.replan(**_controls_from(properties))
        render.apply_delta(delta)

        properties.result_lines = "\n".join(session.report_lines())
        properties.status_is_error = False
        base = "Adjusted by hand" if delta.scalars.get("adjusted") else "As planned"
        properties.status = (
            f"{base} - press Commit to cut" if session.stale else base
        )
    except Exception as error:
        traceback.print_exc()
        properties.status_is_error = True
        properties.status = f"{type(error).__name__}: {error}"


def _apply_trial_pose(properties, context) -> None:
    """Pose the finished, implanted construct by hand.

    Flexion, a varus/valgus stress and an AP drawer, the way a surgeon checks range of
    motion, ligament balance and impingement intraoperatively. A rigid transform of the
    committed geometry: no boolean is involved, nothing is re-planned, and it is instant
    regardless of resection mode.

    Distinct from Femoral's "Flexion (cut)", which changes the femoral cut's sagittal
    angle and therefore the plan.
    """
    session = _session()
    if session is None or not properties.has_scene:
        return

    try:
        from tka_planner.blender import render

        session.set_trial(
            flexion_deg=properties.trial_flexion_deg,
            varus_valgus_deg=properties.trial_varus_valgus_deg,
            drawer_ap_mm=properties.trial_drawer_ap_mm,
        )
        render.apply_scene(session.scene)
    except Exception as error:
        traceback.print_exc()
        properties.status_is_error = True
        properties.status = f"{type(error).__name__}: {error}"


def _toggle_isolate_landmarks(properties, context) -> None:
    """Show the landmarks alone, or restore the rest of the scene."""
    session = _session()
    if session is None or not properties.has_scene:
        return

    from tka_planner.blender import render
    from tka_planner.scene import update as scene_update

    delta = scene_update.isolate_landmarks(
        session.scene, properties.isolate_landmarks
    )
    render.apply_delta(delta)


def _toggle_clean_viewport(properties, context) -> None:
    """White background, no overlays, for figures and screenshots."""
    from tka_planner.blender import render

    render.set_clean_viewport(properties.clean_viewport)


class TKAPlannerProperties(PropertyGroup):
    """Everything the panel remembers between presses."""

    # ---- Case -------------------------------------------------------
    patient_dir: StringProperty(
        name="Patient folder",
        description="Folder holding the segmented femur and tibia STL files",
        subtype="DIR_PATH",
        default="",
    )
    side: EnumProperty(
        name="Side",
        items=[("left", "Left", "Left knee"), ("right", "Right", "Right knee")],
        default="left",
    )
    implant_library: StringProperty(
        name="Implant library",
        description=(
            "Folder of size subfolders (S1..L4) holding the implant STL exports. "
            "Leave empty to show anatomy and cut planes only"
        ),
        subtype="DIR_PATH",
        default="",
    )

    # ---- Plan -------------------------------------------------------
    philosophy: EnumProperty(
        name="Alignment",
        items=[
            ("mechanical", "Mechanical",
             "Components perpendicular to the mechanical axes"),
            ("kinematic", "Kinematic",
             "Cuts parallel to the native joint surfaces"),
        ],
        default="mechanical",
        update=_replan,
    )
    tibial_resection_mm: FloatProperty(
        name="Tibial resection",
        description="Depth below the higher (less worn) plateau, in millimetres",
        default=10.0, min=0.0, max=25.0, step=25, precision=1,
        update=_replan,
    )
    size_override: EnumProperty(
        name="Size",
        description=(
            "Implant size. Auto solves a continuous size from the measurement; the "
            "listed sizes are the twelve the chart publishes"
        ),
        items=_size_items,
        update=_replan,
    )

    # ---- Femoral cut ------------------------------------------------
    coronal_correction_deg: FloatProperty(
        name="Varus / valgus",
        description=(
            "Coronal correction applied to BOTH cuts together, so the femoral and "
            "tibial components stay parallel. Positive is valgus"
        ),
        default=0.0, min=-15.0, max=15.0, step=25, precision=1,
        update=_replan,
    )
    femoral_resection_delta_mm: FloatProperty(
        name="Distal resection",
        description="Extra bone off the distal femur, beyond the component thickness",
        default=0.0, min=-6.0, max=10.0, step=10, precision=1,
        update=_replan,
    )
    femoral_flexion_delta_deg: FloatProperty(
        name="Flexion (cut)",
        description=(
            "Sagittal flexion of the femoral cut and component; positive flexes it. "
            "Changes the cut, unlike Trial reduction's Flexion below, which only poses "
            "the already-cut construct"
        ),
        default=0.0, min=-10.0, max=15.0, step=25, precision=1,
        update=_replan,
    )
    femoral_varus_delta_deg: FloatProperty(
        name="Varus / valgus (this cut)",
        description=(
            "Coronal angle of the femoral cut alone. This breaks the shared "
            "mediolateral slope with the tibial cut, and the plan says so"
        ),
        default=0.0, min=-10.0, max=10.0, step=25, precision=1,
        update=_replan,
    )
    femoral_rotation_delta_deg: FloatProperty(
        name="Rotation",
        description=(
            "External rotation of the femoral component, beyond the conventional "
            "3 degrees off the posterior condylar axis"
        ),
        default=0.0, min=-10.0, max=10.0, step=25, precision=1,
        update=_replan,
    )
    femoral_shift_ap_mm: FloatProperty(
        name="Anterior / posterior",
        description="Slide the femoral component across its cut; positive is anterior",
        default=0.0, min=-10.0, max=10.0, step=10, precision=1,
        update=_replan,
    )
    femoral_shift_ml_mm: FloatProperty(
        name="Medial / lateral",
        description="Slide the femoral component across its cut; positive is lateral",
        default=0.0, min=-10.0, max=10.0, step=10, precision=1,
        update=_replan,
    )

    # ---- Tibial cut -------------------------------------------------
    tibial_resection_delta_mm: FloatProperty(
        name="Resection",
        description="Extra bone off the proximal tibia, beyond the set depth",
        default=0.0, min=-6.0, max=10.0, step=10, precision=1,
        update=_replan,
    )
    tibial_slope_delta_deg: FloatProperty(
        name="Posterior slope",
        description="Change to the planned posterior slope; positive is more slope",
        default=0.0, min=-10.0, max=10.0, step=25, precision=1,
        update=_replan,
    )
    tibial_varus_delta_deg: FloatProperty(
        name="Varus / valgus (this cut)",
        description=(
            "Coronal angle of the tibial cut alone. This breaks the shared "
            "mediolateral slope with the femoral cut, and the plan says so"
        ),
        default=0.0, min=-10.0, max=10.0, step=25, precision=1,
        update=_replan,
    )
    tibial_rotation_delta_deg: FloatProperty(
        name="Rotation",
        description="External rotation of the tibial tray about its cut normal",
        default=0.0, min=-15.0, max=15.0, step=25, precision=1,
        update=_replan,
    )
    tibial_shift_ap_mm: FloatProperty(
        name="Anterior / posterior",
        description="Slide the tray across its cut; positive is anterior",
        default=0.0, min=-10.0, max=10.0, step=10, precision=1,
        update=_replan,
    )
    tibial_shift_ml_mm: FloatProperty(
        name="Medial / lateral",
        description="Slide the tray across its cut; positive is lateral",
        default=0.0, min=-10.0, max=10.0, step=10, precision=1,
        update=_replan,
    )

    # ---- Insert -----------------------------------------------------
    use_insert: BoolProperty(
        name="Insert",
        description=(
            "Show the plastic insert at a set thickness and report the extension gap "
            "it leaves. A placeholder slab until the parametric insert is imported"
        ),
        default=True,
        update=_replan,
    )
    insert_thickness_mm: FloatProperty(
        name="Thickness",
        description="Insert thickness, in millimetres",
        default=9.0, min=4.0, max=20.0, step=25, precision=1,
        update=_replan,
    )

    # ---- Display ----------------------------------------------------
    show_planes: BoolProperty(name="Cut planes", default=True)
    show_axes: BoolProperty(name="Axes", default=True)
    show_landmarks: BoolProperty(name="Show landmarks", default=True)
    isolate_landmarks: BoolProperty(
        name="Isolate landmarks", default=False,
        description=(
            "Hide everything but the landmarks -- bones, implants, blocks, shells, "
            "planes and axes -- for watching how they move relative to each other "
            "through flexion or a trial pose without anything else in the way"
        ),
        update=_toggle_isolate_landmarks,
    )
    resection_mode: EnumProperty(
        name="Resect with",
        description="What removes the bone",
        items=[
            ("block", "Cutting block",
             "Subtract the cutting block itself, and intersect a copy of the bone "
             "with the block's shell. Needs the implant library"),
            ("plane", "Cut plane",
             "Remove everything beyond the planned resection plane"),
            ("none", "Nothing", "Leave both bones whole"),
        ],
        default="block",
    )
    build_bone_shells: BoolProperty(
        name="Bone shells", default=True,
        description=(
            "Intersect a copy of each bone with its cutting block shell, giving the "
            "patient-specific mating surface. Roughly triples the cost of re-cutting"
        ),
    )

    # ---- Trial reduction ---------------------------------------------
    #
    # A hand exam of the finished, implanted construct -- flexion, a coronal stress,
    # an AP drawer -- checking fit and impingement rather than adjusting the plan. Pure
    # rigid transforms of the baked pair, so unlike everything above they carry no
    # boolean cost and need no deferring: `_apply_trial_pose` just moves an empty.
    trial_flexion_deg: FloatProperty(
        name="Flexion (trial)", default=0.0, min=-10.0, max=150.0, step=25, precision=1,
        description=(
            "Flex the implanted construct by hand, about the transepicondylar axis -- "
            "instant, poses the baked construct without touching the cut, unlike "
            "Femoral's Flexion (cut) above. Independent of the scripted flexion "
            "animation -- leave the timeline at frame 1 while using this, or the two "
            "add together"
        ),
        update=_apply_trial_pose,
    )
    trial_varus_valgus_deg: FloatProperty(
        name="Varus / valgus stress", default=0.0, min=-15.0, max=15.0,
        step=25, precision=1,
        description=(
            "Apply a coronal stress at the current trial flexion angle, the way a "
            "manual laxity exam does. Rigid: this does not touch the cuts or the plan"
        ),
        update=_apply_trial_pose,
    )
    trial_drawer_ap_mm: FloatProperty(
        name="AP drawer", default=0.0, min=-15.0, max=15.0, step=10, precision=1,
        description="Slide the tibia anterior/posterior to check an AP drawer test",
        update=_apply_trial_pose,
    )

    # ---- Panel state ------------------------------------------------
    show_femoral: BoolProperty(name="Femoral", default=True)
    show_tibial: BoolProperty(name="Tibial", default=True)
    show_trial: BoolProperty(name="Trial reduction", default=True)
    show_display: BoolProperty(name="Display", default=False)
    clean_viewport: BoolProperty(
        name="White background, no lines", default=False,
        description=(
            "Plain white background with the floor grid and axis lines hidden, in "
            "every open 3D viewport -- a display preference, applies immediately, "
            "needs no plan built"
        ),
        update=_toggle_clean_viewport,
    )

    # Results, written by the operator and read by the panel.
    has_result: BoolProperty(default=False)
    has_scene: BoolProperty(default=False)
    result_lines: StringProperty(default="")
    status: StringProperty(default="")
    status_is_error: BoolProperty(default=False)


class TKA_OT_plan(Operator):
    """Measure the case, plan the alignment and build the scene."""

    bl_idname = "tka.plan"
    bl_label = "Plan"
    bl_description = "Measure this knee, plan the resections, and show the result"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ensure_package_importable()
        properties = context.scene.tka_planner

        try:
            session = self._open(properties)
        except Exception as error:  # surfaced in the panel, not only the console
            traceback.print_exc()
            properties.has_result = False
            properties.has_scene = False
            properties.status_is_error = True
            properties.status = f"{type(error).__name__}: {error}"
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        _SESSION["current"] = session
        properties.result_lines = "\n".join(session.report_lines())
        properties.has_result = True
        properties.has_scene = True
        properties.status_is_error = False
        properties.status = "Plan complete - press Commit to cut"
        return {"FINISHED"}

    @staticmethod
    def _open(properties):
        """Open a session on the patient folder and draw its scene.

        The bones arrive uncut. Cutting is a separate, explicit step, because an exact
        boolean against a real segmentation costs seconds and pressing Plan should put
        the anatomy on screen rather than hold the button for the length of a solve.
        """
        from tka_planner.blender import render
        from tka_planner.session import PlanningSession

        folder = Path(bpy.path.abspath(properties.patient_dir))
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        library = Path(bpy.path.abspath(properties.implant_library or ""))
        session = PlanningSession.open(
            folder,
            side=properties.side,
            library=library if library.is_dir() else None,
        )
        session.controls = replace(session.controls, **_controls_from(properties))
        session.build()
        render.render_scene(session.scene)
        return session


class TKA_OT_commit(Operator):
    """Cut the bones along the planned resections."""

    bl_idname = "tka.commit"
    bl_label = "Commit cuts"
    bl_description = (
        "Run the resection at full resolution. The only step that cuts geometry"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ensure_package_importable()
        properties = context.scene.tka_planner
        session = _session()
        if session is None or not properties.has_scene:
            self.report({"ERROR"}, "Press Plan first.")
            return {"CANCELLED"}

        try:
            from tka_planner.blender import render

            result = session.commit()
            render.apply_scene(session.scene)
        except Exception as error:
            traceback.print_exc()
            properties.status_is_error = True
            properties.status = f"{type(error).__name__}: {error}"
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        properties.result_lines = "\n".join(session.report_lines())
        properties.status_is_error = False
        cut = ", ".join(sorted(result.records)) or "nothing"
        properties.status = f"Committed: {cut}"
        for note in result.notes:
            self.report({"WARNING"}, note)
        return {"FINISHED"}


class TKA_OT_reset(Operator):
    """Put every manual control back to the computed plan."""

    bl_idname = "tka.reset_adjustments"
    bl_label = "Reset to plan"
    bl_description = "Return every manual adjustment to zero"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        properties = context.scene.tka_planner
        for name in ADJUSTMENT_PROPERTIES:
            properties.property_unset(name)
        _replan(properties, context)
        return {"FINISHED"}


class TKA_OT_reset_trial(Operator):
    """Put the trial rig back to extension, no stress, no drawer."""

    bl_idname = "tka.reset_trial"
    bl_label = "Reset trial pose"
    bl_description = "Return the hand exam -- flexion, stress, drawer -- to zero"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        properties = context.scene.tka_planner
        for name in TRIAL_PROPERTIES:
            properties.property_unset(name)
        _apply_trial_pose(properties, context)
        return {"FINISHED"}


class TKA_OT_clear(Operator):
    """Empty the scene."""

    bl_idname = "tka.clear"
    bl_label = "Clear scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ensure_package_importable()
        from tka_planner.blender.io import clear_scene

        clear_scene()
        _SESSION.pop("current", None)
        context.scene.tka_planner.has_result = False
        context.scene.tka_planner.has_scene = False
        context.scene.tka_planner.status = ""
        return {"FINISHED"}


class TKA_PT_panel(Panel):
    """The planning sidebar."""

    bl_label = "TKA Planner"
    bl_idname = "TKA_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "TKA"

    def draw(self, context):
        layout = self.layout
        properties = context.scene.tka_planner

        box = layout.box()
        box.label(text="Research use only", icon="ERROR")
        box.label(text="Not a medical device.")

        layout.separator()
        layout.label(text="Patient", icon="FILE_FOLDER")
        layout.prop(properties, "patient_dir", text="")
        layout.prop(properties, "side", expand=True)
        layout.label(text="Implant library (optional)")
        layout.prop(properties, "implant_library", text="")

        layout.separator()
        run = layout.row()
        run.scale_y = 1.6
        run.enabled = bool(properties.patient_dir)
        run.operator("tka.plan", icon="PLAY")

        if properties.status:
            box = layout.box()
            box.label(
                text=properties.status[:58],
                icon="ERROR" if properties.status_is_error else "CHECKMARK",
            )

        layout.separator()
        layout.label(text="Plan", icon="MODIFIER")
        layout.prop(properties, "philosophy", text="")
        layout.prop(properties, "size_override")
        layout.prop(properties, "tibial_resection_mm")

        column = layout.column(align=True)
        column.scale_y = 1.2
        column.prop(properties, "coronal_correction_deg", slider=True)

        self._draw_femoral(layout, properties)
        self._draw_tibial(layout, properties)
        self._draw_insert(layout, properties)
        self._draw_trial(layout, properties)
        self._draw_display(layout, properties)

        layout.separator()
        row = layout.row(align=True)
        row.operator("tka.reset_adjustments", icon="LOOP_BACK")
        row.operator("tka.clear", icon="TRASH")

        if properties.has_result and properties.result_lines:
            layout.separator()
            self._draw_results(layout, properties)

    @staticmethod
    def _header(layout, properties, flag: str, text: str, icon: str):
        box = layout.box()
        row = box.row()
        row.prop(
            properties, flag, text="", emboss=False,
            icon="DISCLOSURE_TRI_DOWN" if getattr(properties, flag)
            else "DISCLOSURE_TRI_RIGHT",
        )
        row.label(text=text, icon=icon)
        return box

    def _draw_femoral(self, layout, properties):
        box = self._header(layout, properties, "show_femoral", "Femoral", "BONE_DATA")
        if not properties.show_femoral:
            return
        column = box.column(align=True)
        for name in ("femoral_resection_delta_mm", "femoral_flexion_delta_deg",
                     "femoral_varus_delta_deg", "femoral_rotation_delta_deg"):
            column.prop(properties, name, slider=True)
        column = box.column(align=True)
        column.label(text="Position")
        column.prop(properties, "femoral_shift_ap_mm", slider=True)
        column.prop(properties, "femoral_shift_ml_mm", slider=True)

    def _draw_tibial(self, layout, properties):
        box = self._header(layout, properties, "show_tibial", "Tibial", "BONE_DATA")
        if not properties.show_tibial:
            return
        column = box.column(align=True)
        for name in ("tibial_resection_delta_mm", "tibial_slope_delta_deg",
                     "tibial_varus_delta_deg", "tibial_rotation_delta_deg"):
            column.prop(properties, name, slider=True)
        column = box.column(align=True)
        column.label(text="Position")
        column.prop(properties, "tibial_shift_ap_mm", slider=True)
        column.prop(properties, "tibial_shift_ml_mm", slider=True)

    def _draw_insert(self, layout, properties):
        box = layout.box()
        row = box.row()
        row.prop(properties, "use_insert")
        if properties.use_insert:
            box.prop(properties, "insert_thickness_mm", slider=True)
            box.label(text="Placeholder slab", icon="INFO")

    def _draw_trial(self, layout, properties):
        box = self._header(
            layout, properties, "show_trial", "Trial reduction", "ARMATURE_DATA"
        )
        if not properties.show_trial:
            return
        box.label(
            text="Hand exam of the cut, implanted construct -- instant, no re-plan"
        )
        column = box.column(align=True)
        column.prop(properties, "trial_flexion_deg", slider=True)
        column.prop(properties, "trial_varus_valgus_deg", slider=True)
        column.prop(properties, "trial_drawer_ap_mm", slider=True)
        box.operator("tka.reset_trial", icon="LOOP_BACK")

    def _draw_display(self, layout, properties):
        box = self._header(layout, properties, "show_display", "Display", "HIDE_OFF")
        if not properties.show_display:
            return
        box.prop(properties, "clean_viewport", toggle=True)
        row = box.row(align=True)
        row.prop(properties, "show_planes", toggle=True)
        row.prop(properties, "show_axes", toggle=True)
        box.prop(properties, "show_landmarks", toggle=True)
        if properties.show_landmarks:
            box.prop(properties, "isolate_landmarks", toggle=True)
        box.label(text="Resect with")
        box.prop(properties, "resection_mode", text="")
        if properties.resection_mode == "block" and not properties.implant_library:
            box.label(text="Needs the implant library", icon="ERROR")
        if properties.resection_mode == "block":
            box.prop(properties, "build_bone_shells", toggle=True)
        box.label(text="Press Commit to apply", icon="INFO")

    @staticmethod
    def _draw_results(layout, properties):
        for line in properties.result_lines.split("\n"):
            if not line:
                layout.separator(factor=0.4)
            elif line.startswith("HEAD|"):
                layout.label(text=line[5:], icon="DOT")
            elif line.startswith("WARN|"):
                layout.box().label(text=line[5:][:56], icon="ERROR")
            elif line.startswith("CASE|"):
                layout.label(text=line[5:], icon="OUTLINER_OB_ARMATURE")
            else:
                label, _, value = line.partition("|")
                split = layout.split(factor=0.60)
                split.label(text=label)
                split.label(text=value)


_CLASSES = (
    TKAPlannerProperties,
    TKA_OT_plan,
    TKA_OT_commit,
    TKA_OT_reset,
    TKA_OT_reset_trial,
    TKA_OT_clear,
    TKA_PT_panel,
)


def register():
    _ensure_package_importable()
    for cls in _CLASSES:
        register_class(cls)
    bpy.types.Scene.tka_planner = PointerProperty(type=TKAPlannerProperties)


def unregister():
    # No handlers to unregister any more. The old scene needed a depsgraph handler to
    # swap the editing geometry for the baked pair whenever the timeline left frame 1;
    # with the resection committed there is only one set of geometry and nothing to swap.
    del bpy.types.Scene.tka_planner
    for cls in reversed(_CLASSES):
        unregister_class(cls)
    _SESSION.clear()
    _SIZE_ITEMS.clear()


if __name__ == "__main__":
    register()
