"""Blender add-on: the planning screen.

Pick a patient folder, choose an alignment philosophy, press Plan. The bones load, the
resection planes and mechanical axes appear in the viewport, and the sizing and
positioning numbers appear in the sidebar next to them.

All the work happens in :mod:`tka_planner.core`, which does not import ``bpy``. This
add-on loads meshes, calls the planner, and draws the answer. That separation is why the
same computation can run headless in the test suite, and why nothing shown here can
disagree with what the command line produces.

Install by pointing Blender's script path at this repository and enabling
"TKA Planner", then press N in the 3D viewport and open the "TKA" tab.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy.utils import register_class, unregister_class

bl_info = {
    "name": "TKA Planner",
    "author": "Satbayev University",
    "version": (0, 1, 0),
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


class TKAPlannerProperties(PropertyGroup):
    """Everything the panel remembers between presses."""

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
    philosophy: EnumProperty(
        name="Alignment",
        items=[
            ("mechanical", "Mechanical",
             "Components perpendicular to the mechanical axes"),
            ("kinematic", "Kinematic",
             "Cuts parallel to the native joint surfaces"),
        ],
        default="mechanical",
    )
    tibial_resection_mm: FloatProperty(
        name="Tibial resection",
        description="Depth below the higher (less worn) plateau, in millimetres",
        default=10.0, min=0.0, max=20.0,
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
    show_planes: BoolProperty(name="Cut planes", default=True)
    show_axes: BoolProperty(name="Axes", default=True)
    show_landmarks: BoolProperty(name="Show landmarks", default=False)

    # Results, written by the operator and read by the panel.
    has_result: BoolProperty(default=False)
    result_lines: StringProperty(default="")
    status: StringProperty(default="")
    status_is_error: BoolProperty(default=False)


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
            properties.status_is_error = True
            properties.status = f"{type(error).__name__}: {error}"
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        properties.result_lines = "\n".join(lines)
        properties.has_result = True
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
        from tka_planner.core.planning import KINEMATIC, MECHANICAL, plan_alignment
        from tka_planner.core.sizing import (
            load_size_chart,
            select_discrete_size,
            solve_parametric_size,
        )

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
        metrics = compute_all(landmarks, femoral_frame, tibial_frame)

        femoral_measure = measure_femoral_ml(femur, femoral_frame)
        tibial_measure = measure_tibial_plateau(tibia, tibial_frame)

        chart = load_size_chart(_find_size_chart())
        sizing = solve_parametric_size(
            chart,
            measured_ml_mm=femoral_measure.ml_mm,
            measured_ap_mm=femoral_measure.ap_mm,
        )
        discrete = select_discrete_size(chart, measured_ml_mm=femoral_measure.ml_mm)

        plan = plan_alignment(
            landmarks, femoral_frame, tibial_frame,
            target=(MECHANICAL if properties.philosophy == "mechanical"
                    else KINEMATIC),
            femoral_thickness_mm=sizing.femoral_thickness_mm,
            tibial_resection_mm=properties.tibial_resection_mm,
            native_slope_deg=metrics["posterior_slope_medial_deg"].value,
        )

        components = {}
        library = Path(bpy.path.abspath(properties.implant_library or ""))
        if library.is_dir():
            components = resolve_component_meshes(
                library, chart=chart, sizing=sizing, side=properties.side
            )

        build_scene(
            femur_path=femur_path, tibia_path=tibia_path,
            components=components,
            plan=plan, femoral_frame=femoral_frame, tibial_frame=tibial_frame,
            landmarks=landmarks,
            show_planes=properties.show_planes,
            show_axes=properties.show_axes,
            show_landmarks=properties.show_landmarks,
        )

        femoral = plan.resections["femoral_distal"]
        tibial = plan.resections["tibial_proximal"]

        lines = [
            f"CASE|{case_id} ({properties.side})",
            "",
            "HEAD|Correction",
            f"Valgus cut|{plan.distal_femoral_valgus_cut_deg:.1f} deg",
            f"Posterior slope|{plan.tibial_slope_deg:.1f} deg",
            f"Philosophy|{plan.philosophy}",
            f"Cuts share ML slope|yes",
            "",
            "HEAD|Resection depths",
            f"Femoral medial|{femoral.medial_depth_mm:.1f} mm",
            f"Femoral lateral|{femoral.lateral_depth_mm:.1f} mm",
            f"Tibial medial|{tibial.medial_depth_mm:.1f} mm",
            f"Tibial lateral|{tibial.lateral_depth_mm:.1f} mm",
            "",
            "HEAD|Sizing",
            f"Femoral ML|{femoral_measure.ml_mm:.1f} mm",
            f"Plateau AP|{tibial_measure.ap_mm:.1f} mm",
            f"Plateau ML|{tibial_measure.ml_mm:.1f} mm",
            f"Size parameter|{sizing.size_parameter:.2f}",
            f"Implant width|{sizing.implant_ml_mm:.1f} mm",
            f"Nearest size|{discrete.nearest_discrete_size}",
            f"Implants shown|{len(components)}",
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
        if discrete.flags:
            lines.append(f"WARN|Chart: {', '.join(discrete.flags)}")
        for warning in plan.warnings:
            lines.append(f"WARN|{warning}")

        return lines


class TKA_OT_clear(Operator):
    """Empty the scene."""

    bl_idname = "tka.clear"
    bl_label = "Clear scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ensure_package_importable()
        from tka_planner.blender.io import clear_scene

        clear_scene()
        context.scene.tka_planner.has_result = False
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

        layout.separator()
        layout.label(text="Plan", icon="MODIFIER")
        layout.prop(properties, "philosophy", text="")
        layout.prop(properties, "tibial_resection_mm")
        layout.label(text="Implant library (optional)")
        layout.prop(properties, "implant_library", text="")

        row = layout.row(align=True)
        row.prop(properties, "show_planes", toggle=True)
        row.prop(properties, "show_axes", toggle=True)
        layout.prop(properties, "show_landmarks", toggle=True)

        layout.separator()
        run = layout.row()
        run.scale_y = 1.6
        run.enabled = bool(properties.patient_dir)
        run.operator("tka.plan", icon="PLAY")
        layout.operator("tka.clear", icon="TRASH")

        if properties.status:
            box = layout.box()
            box.label(
                text=properties.status[:58],
                icon="ERROR" if properties.status_is_error else "CHECKMARK",
            )

        if properties.has_result and properties.result_lines:
            layout.separator()
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


_CLASSES = (TKAPlannerProperties, TKA_OT_plan, TKA_OT_clear, TKA_PT_panel)


def register():
    _ensure_package_importable()
    for cls in _CLASSES:
        register_class(cls)
    bpy.types.Scene.tka_planner = bpy.props.PointerProperty(type=TKAPlannerProperties)


def unregister():
    del bpy.types.Scene.tka_planner
    for cls in reversed(_CLASSES):
        unregister_class(cls)


if __name__ == "__main__":
    register()
