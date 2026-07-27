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
import time
import traceback
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
    """Put the repository on ``sys.path`` so ``tka_planner.core`` imports.

    Blender runs an add-on from inside the package, so the parent directory has to be
    added explicitly unless the package was pip-installed into Blender's own Python.
    """
    repository = Path(__file__).resolve().parent.parent.parent
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))


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
# Deferring the cut
# ----------------------------------------------------------------------
#
# Moving a plane or a component costs milliseconds. Re-cutting the bone does not: an
# exact boolean against a real segmentation takes about 6 seconds for the plane box and
# 25 with the cutting blocks and shells, measured on Patient_005. That cannot run inside
# a slider drag, and the first version of this panel appeared to manage it only because
# it was timed headless, where Blender never evaluates a modifier nobody is looking at.
#
# So the resection booleans are muted the moment a control moves and restored once the
# controls have been still for a moment. Alignment is judged on the planes, the axes and
# the components, and all of those keep moving at full speed.

RECUT_IDLE_SECONDS = 0.6
_LAST_CHANGE = [0.0]
_TIMER_ARMED = [False]


def _restore_cuts_when_idle():
    """Timer callback: put the cuts back once the controls have settled.

    Returns the seconds to wait before being called again, or ``None`` to stop -- which
    is Blender's timer protocol, and is what lets one timer debounce a whole drag
    instead of one being registered per change.
    """
    remaining = RECUT_IDLE_SECONDS - (time.monotonic() - _LAST_CHANGE[0])
    if remaining > 0.0:
        return remaining

    _TIMER_ARMED[0] = False
    try:
        from tka_planner.blender.build import set_cuts_visible
        set_cuts_visible(True)
    except Exception:
        traceback.print_exc()
    return None


def _defer_cuts() -> None:
    """Hide the cuts now, and arrange for them to come back."""
    from tka_planner.blender.build import set_cuts_visible

    _LAST_CHANGE[0] = time.monotonic()
    set_cuts_visible(False)
    if not _TIMER_ARMED[0]:
        _TIMER_ARMED[0] = True
        bpy.app.timers.register(
            _restore_cuts_when_idle, first_interval=RECUT_IDLE_SECONDS
        )


ADJUSTMENT_PROPERTIES = (
    "coronal_correction_deg",
    "femoral_resection_delta_mm",
    "femoral_flexion_delta_deg",
    "femoral_varus_delta_deg",
    "femoral_rotation_delta_deg",
    "femoral_shift_ap_mm",
    "femoral_shift_ml_mm",
    "tibial_resection_delta_mm",
    "tibial_slope_delta_deg",
    "tibial_varus_delta_deg",
    "tibial_rotation_delta_deg",
    "tibial_shift_ap_mm",
    "tibial_shift_ml_mm",
)


# Blender keeps no reference to the strings an EnumProperty items callback returns, so a
# list built fresh each call can be garbage collected while the menu is open and show
# corrupt entries. Caching it at module scope is the documented way round that.
_SIZE_ITEMS: list = []


def _size_items(self, context):
    global _SIZE_ITEMS
    items = [("auto", "Auto (continuous)",
              "Solve a continuous size from the measured width")]
    try:
        chart = _load_chart()
        items += [
            (label, label, f"Chart size {label}") for label in chart.labels
        ]
    except Exception:
        pass
    _SIZE_ITEMS = items
    return _SIZE_ITEMS


def _find_size_chart() -> Path:
    """Locate SizeChart.csv, whether running from the repository or from an install.

    Installed as a zip the package sits in Blender's add-ons directory and there is no
    repository beside it, so the chart bundled inside the package is used. Running from
    a checkout, the repository copy wins so edits to it take effect without rebuilding.
    """
    package = Path(__file__).resolve().parent.parent
    candidates = [
        package.parent / "data" / "SizeChart.csv",  # repository checkout
        package / "data" / "SizeChart.csv",         # bundled in the installed add-on
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "SizeChart.csv not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


_CHART: list = []


def _load_chart():
    """The size chart, read once per session."""
    if not _CHART:
        from tka_planner.core.sizing import load_size_chart
        _CHART.append(load_size_chart(_find_size_chart()))
    return _CHART[0]


def _find_bone_files(folder: Path, side: str) -> tuple[Path, Path]:
    """Locate the femur and tibia surfaces in a patient folder.

    Tries the project's ``FD1Left.stl`` / ``TD1Left.stl`` convention first, then falls
    back to any STL named for the bone, so a folder exported straight out of 3D Slicer
    also works.
    """
    side_word = side.capitalize()
    femur = folder / f"FD1{side_word}.stl"
    tibia = folder / f"TD1{side_word}.stl"
    if femur.is_file() and tibia.is_file():
        return femur, tibia

    candidates = sorted(folder.glob("*.stl"))
    femur = next((p for p in candidates if "femur" in p.name.lower()
                  or p.name.upper().startswith("FD")), None)
    tibia = next((p for p in candidates if "tibia" in p.name.lower()
                  or p.name.upper().startswith("TD")), None)
    if femur is None or tibia is None:
        raise FileNotFoundError(
            f"Could not find a femur and a tibia STL in {folder}. Expected "
            f"FD1{side_word}.stl and TD1{side_word}.stl, or files named for the bone."
        )
    return femur, tibia


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------


def _adjustments_from(properties):
    """Gather the panel's controls into the core's adjustment record."""
    from tka_planner.core.planning import Adjustments

    return Adjustments(
        **{name: getattr(properties, name) for name in ADJUSTMENT_PROPERTIES},
        insert_thickness_mm=(
            properties.insert_thickness_mm if properties.use_insert else None
        ),
    )


def _sizing_for(session, properties):
    """The implant size, solved from the anatomy or set by hand."""
    from tka_planner.core.sizing import solve_parametric_size

    chart = session["chart"]
    label = properties.size_override

    # An EnumProperty whose items come from a callback has no default until its menu is
    # first opened, and reads back as an empty string until then. Anything that is not a
    # chart size means solve it from the measurement, which is also the right answer if
    # a saved file names a size the chart no longer publishes.
    override = (
        chart.label_parameter(label) if label in chart.labels else None
    )

    return solve_parametric_size(
        chart,
        measured_ml_mm=session["femoral_measure"].ml_mm,
        measured_ap_mm=session["femoral_measure"].ap_mm,
        parameter_override=override,
    )


def _plan_from(session, properties):
    """Re-plan from the cached measurement. Pure numpy, so it is cheap enough to run on
    every change of a slider rather than on a button press."""
    from tka_planner.core.planning import KINEMATIC, MECHANICAL, plan_alignment

    sizing = _sizing_for(session, properties)
    plan = plan_alignment(
        session["landmarks"], session["femoral_frame"], session["tibial_frame"],
        target=(MECHANICAL if properties.philosophy == "mechanical" else KINEMATIC),
        femoral_thickness_mm=sizing.femoral_thickness_mm,
        tibial_resection_mm=properties.tibial_resection_mm,
        native_slope_deg=session["metrics"]["posterior_slope_medial_deg"].value,
        adjustments=_adjustments_from(properties),
        femur_mesh=session["femur"], tibia_mesh=session["tibia"],
    )
    return plan, sizing


def _replan(properties, context) -> None:
    """Re-plan and re-pose the scene in place. This is what every control calls.

    Deliberately silent when there is nothing to update: the properties are also written
    when a blend file loads and when Plan itself runs, and neither should trigger a
    half-built scene. Any failure is reported into the panel's status line rather than
    raised, because an exception inside a property callback leaves Blender's UI in a bad
    state and says nothing useful to the person holding the slider.
    """
    session = _session()
    if session is None or not properties.has_scene:
        return

    try:
        from tka_planner.blender.build import update_scene

        if properties.defer_cuts:
            _defer_cuts()

        plan, sizing = _plan_from(session, properties)
        update_scene(
            plan,
            insert_thickness_mm=(
                properties.insert_thickness_mm if properties.use_insert else None
            ),
            live_cuts=properties.live_cuts and properties.resection_mode != "none",
            implant_ml_mm=sizing.implant_ml_mm,
        )
        properties.result_lines = "\n".join(_report_lines(session, plan, sizing))
        properties.status_is_error = False
        properties.status = (
            "As planned" if plan.adjustments.is_identity else "Adjusted by hand"
        )
    except Exception as error:
        traceback.print_exc()
        properties.status_is_error = True
        properties.status = f"{type(error).__name__}: {error}"


def _report_lines(session, plan, sizing) -> list[str]:
    """The sidebar readout, rebuilt on every change so it can never lag the geometry."""
    femoral = plan.resections["femoral_distal"]
    tibial = plan.resections["tibial_proximal"]
    metrics = session["metrics"]
    diagnostics = plan.diagnostics

    lines = [
        f"CASE|{session['case_id']} ({session['side']})",
        "",
        "HEAD|Correction",
        f"Valgus cut|{plan.distal_femoral_valgus_cut_deg:.1f} deg",
        f"Posterior slope|{plan.tibial_slope_deg:.1f} deg",
        f"Femoral flexion|{diagnostics['femoral_flexion_deg']:.1f} deg",
        f"Philosophy|{plan.philosophy}",
        f"Cuts share ML slope|"
        f"{'yes' if diagnostics['cut_ml_slope_shared'] else 'no'}",
        "",
        "HEAD|Resection depths",
        f"Femoral medial|{femoral.medial_depth_mm:.1f} mm",
        f"Femoral lateral|{femoral.lateral_depth_mm:.1f} mm",
        f"Tibial medial|{tibial.medial_depth_mm:.1f} mm",
        f"Tibial lateral|{tibial.lateral_depth_mm:.1f} mm",
    ]

    if diagnostics.get("extension_gap_medial_mm") is not None:
        lines += [
            "",
            "HEAD|Extension gap",
            f"Medial|{diagnostics['extension_gap_medial_mm']:.1f} mm",
            f"Lateral|{diagnostics['extension_gap_lateral_mm']:.1f} mm",
            f"Construct|{diagnostics['extension_gap_construct_mm']:.1f} mm",
        ]

    lines += [
        "",
        "HEAD|Sizing",
        f"Femoral ML|{session['femoral_measure'].ml_mm:.1f} mm",
        f"Plateau AP|{session['tibial_measure'].ap_mm:.1f} mm",
        f"Size parameter|{sizing.size_parameter:.2f}",
        f"Implant width|{sizing.implant_ml_mm:.1f} mm",
        f"Nearest size|{sizing.nearest_discrete_size}",
        "",
        "HEAD|Alignment",
    ]
    for name in ("mldfa_deg", "aldfa_deg", "mpta_deg", "jlca_deg"):
        metric = metrics[name]
        value = f"{metric.value:.1f}" if metric.value is not None else "n/a"
        lines.append(
            f"{name.replace('_deg', '').upper()}|{value} ({metric.quality.value})"
        )

    if metrics["hka_deviation_deg"].value is None:
        lines += ["", "WARN|HKA needs hip and ankle: outside this scan"]
    if sizing.flags:
        lines.append(f"WARN|Sizing: {', '.join(sizing.flags)}")
    for warning in plan.warnings:
        lines.append(f"WARN|{warning}")

    return lines


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
        name="Flexion",
        description="Sagittal flexion of the femoral component; positive flexes it",
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
    show_landmarks: BoolProperty(name="Show landmarks", default=False)
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
    defer_cuts: BoolProperty(
        name="Hide cuts while adjusting", default=True,
        description=(
            "Mute the resection while a control is moving and recompute it once you "
            "stop. An exact boolean takes seconds on a dense segmentation, so leaving "
            "it on during a drag makes the whole panel unresponsive"
        ),
    )
    build_bone_shells: BoolProperty(
        name="Bone shells", default=True,
        description=(
            "Intersect a copy of each bone with its cutting block shell, giving the "
            "patient-specific mating surface. Roughly triples the cost of re-cutting"
        ),
    )
    live_cuts: BoolProperty(
        name="Live cuts", default=True,
        description=(
            "Leave the resection boolean unapplied so the bone re-cuts as the plan is "
            "adjusted. Turn off on a dense segmentation if dragging feels heavy"
        ),
    )
    cut_solver: EnumProperty(
        name="Solver",
        items=[
            ("EXACT", "Exact", "Accurate; slower on dense meshes"),
            ("FAST", "Fast", "Responsive; the resulting geometry differs"),
        ],
        default="EXACT",
    )
    animate_flexion: BoolProperty(
        name="Animate flexion", default=True,
        description="Swing the tibia through flexion about the transepicondylar axis",
    )
    max_flexion_deg: FloatProperty(
        name="Max flexion", default=120.0, min=0.0, max=150.0,
        description="Peak flexion angle for the animation, in degrees",
    )

    # ---- Panel state ------------------------------------------------
    show_femoral: BoolProperty(name="Femoral", default=True)
    show_tibial: BoolProperty(name="Tibial", default=True)
    show_display: BoolProperty(name="Display", default=False)

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
            lines = self._run(properties)
        except Exception as error:  # surfaced in the panel, not only the console
            traceback.print_exc()
            properties.has_result = False
            properties.has_scene = False
            properties.status_is_error = True
            properties.status = f"{type(error).__name__}: {error}"
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if properties.defer_cuts:
            _defer_cuts()

        properties.result_lines = "\n".join(lines)
        properties.has_result = True
        properties.has_scene = True
        properties.status_is_error = False
        properties.status = "Plan complete"
        return {"FINISHED"}

    @staticmethod
    def _run(properties) -> list[str]:
        from tka_planner.blender.build import build_scene, resolve_component_meshes
        from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
        from tka_planner.core.landmarks_auto import estimate_landmarks
        from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
        from tka_planner.core.meshio import read_stl
        from tka_planner.core.metrics import compute_all

        folder = Path(bpy.path.abspath(properties.patient_dir))
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        femur_path, tibia_path = _find_bone_files(folder, properties.side)
        femur = read_stl(femur_path)
        tibia = read_stl(tibia_path)
        case_id = folder.name

        landmarks = estimate_landmarks(femur, tibia, properties.side, case_id=case_id)
        femoral_frame = build_femoral_frame(landmarks)
        tibial_frame = build_tibial_frame(landmarks)

        session = {
            "case_id": case_id,
            "side": properties.side,
            "femur": femur,
            "tibia": tibia,
            "femur_path": femur_path,
            "tibia_path": tibia_path,
            "landmarks": landmarks,
            "femoral_frame": femoral_frame,
            "tibial_frame": tibial_frame,
            "metrics": compute_all(landmarks, femoral_frame, tibial_frame),
            "femoral_measure": measure_femoral_ml(femur, femoral_frame),
            "tibial_measure": measure_tibial_plateau(tibia, tibial_frame),
            "chart": _load_chart(),
        }
        _SESSION["current"] = session

        plan, sizing = _plan_from(session, properties)

        components = {}
        library = Path(bpy.path.abspath(properties.implant_library or ""))
        if library.is_dir():
            components = resolve_component_meshes(
                library, chart=session["chart"], sizing=sizing, side=properties.side
            )

        build_scene(
            femur_path=femur_path, tibia_path=tibia_path,
            components=components,
            plan=plan, femoral_frame=femoral_frame, tibial_frame=tibial_frame,
            landmarks=landmarks,
            show_planes=properties.show_planes,
            show_axes=properties.show_axes,
            show_landmarks=properties.show_landmarks,
            perform_cuts=properties.resection_mode == "plane",
            cut_with_blocks=properties.resection_mode == "block",
            build_bone_shells=properties.build_bone_shells,
            # Built muted when deferring, so pressing Plan returns as soon as the
            # anatomy is on screen and the resection fills in a moment later,
            # rather than holding the button for the length of a boolean.
            cuts_visible=not properties.defer_cuts,
            live_cuts=properties.live_cuts,
            cut_solver=properties.cut_solver,
            animate_flexion=properties.animate_flexion,
            max_flexion_deg=properties.max_flexion_deg,
            insert_thickness_mm=(
                properties.insert_thickness_mm if properties.use_insert else None
            ),
            insert_footprint_mm=(
                sizing.diagnostics.get("tibia_ML_mm", 70.0),
                sizing.diagnostics.get("tibia_AP_mm", 48.0),
            ),
        )

        lines = _report_lines(session, plan, sizing)
        lines.append(f"Implants shown|{len(components)}")
        return lines


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

    def _draw_display(self, layout, properties):
        box = self._header(layout, properties, "show_display", "Display", "HIDE_OFF")
        if not properties.show_display:
            return
        row = box.row(align=True)
        row.prop(properties, "show_planes", toggle=True)
        row.prop(properties, "show_axes", toggle=True)
        box.prop(properties, "show_landmarks", toggle=True)
        box.label(text="Resect with")
        box.prop(properties, "resection_mode", text="")
        if properties.resection_mode == "block" and not properties.implant_library:
            box.label(text="Needs the implant library", icon="ERROR")
        if properties.resection_mode != "none":
            box.prop(properties, "live_cuts", toggle=True)
            box.prop(properties, "defer_cuts", toggle=True)
            if properties.resection_mode == "block":
                box.prop(properties, "build_bone_shells", toggle=True)
            if properties.live_cuts and properties.resection_mode == "plane":
                box.prop(properties, "cut_solver", expand=True)
        box.prop(properties, "animate_flexion", toggle=True)
        if properties.animate_flexion:
            box.prop(properties, "max_flexion_deg")
        box.label(text="Rebuild to apply", icon="INFO")

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
    TKA_OT_reset,
    TKA_OT_clear,
    TKA_PT_panel,
)


def register():
    _ensure_package_importable()
    for cls in _CLASSES:
        register_class(cls)
    bpy.types.Scene.tka_planner = PointerProperty(type=TKAPlannerProperties)


def unregister():
    del bpy.types.Scene.tka_planner
    for cls in reversed(_CLASSES):
        unregister_class(cls)
    _SESSION.clear()
    _CHART.clear()


if __name__ == "__main__":
    register()
