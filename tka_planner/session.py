"""One planning session, with no user interface attached.

This is the part that did not exist before. The add-on's property callbacks held it:
the cached measurement, the current control values, the re-plan on every change, the
decision about which bones a change actually reaches. All of that was tangled with
Blender's UI lifecycle, which meant it could only be driven by a person holding a
slider.

Pulled out here, the same session is driven by a browser, by a test, or later by a
tracker feed, and none of them need to know about each other.

The session is the Plan / Commit / Reduce split made concrete. ``replan`` never cuts,
``commit`` is the only thing that does, and ``set_trial`` touches neither the plan nor
the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.landmarks_auto import estimate_landmarks
from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
from tka_planner.core.meshio import read_stl
from tka_planner.core.metrics import compute_all
from tka_planner.core.planning import KINEMATIC, MECHANICAL, Adjustments, plan_alignment
from tka_planner.core.sizing import load_size_chart, solve_parametric_size
from tka_planner.scene import build as scene_build
from tka_planner.scene import motion
from tka_planner.scene import resect as scene_resect
from tka_planner.scene import update as scene_update
from tka_planner.scene.model import TIBIAL, SceneDelta

__all__ = [
    "ADJUSTMENT_FIELDS", "Controls", "TrialControls", "PlanningSession",
    "find_bone_files", "find_size_chart",
]

ADJUSTMENT_FIELDS = (
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

# Which bones a control's change actually reaches, so a commit re-cuts only what moved.
# Adjusting the tibial slope must not re-solve an untouched femur.
FEMORAL_ONLY = frozenset(
    name for name in ADJUSTMENT_FIELDS if name.startswith("femoral_")
)
TIBIAL_ONLY = frozenset(
    name for name in ADJUSTMENT_FIELDS if name.startswith("tibial_")
)
BOTH_BONES = frozenset({
    "coronal_correction_deg", "philosophy", "size_override", "tibial_resection_mm",
    "resection_mode", "build_bone_shells",
})
# Controls that move nothing a boolean depends on, in any mode.
NO_BONES = frozenset({
    "insert_thickness_mm", "use_insert", "show_planes", "show_axes",
    "show_landmarks", "isolate_landmarks",
})
# Controls that move a component without moving a resection plane.
COMPONENT_ONLY = ("_shift_", "_rotation_")


@dataclass
class Controls:
    """Every plan control, matching the add-on's panel one for one."""

    coronal_correction_deg: float = 0.0
    femoral_resection_delta_mm: float = 0.0
    femoral_flexion_delta_deg: float = 0.0
    femoral_varus_delta_deg: float = 0.0
    femoral_rotation_delta_deg: float = 0.0
    femoral_shift_ap_mm: float = 0.0
    femoral_shift_ml_mm: float = 0.0
    tibial_resection_delta_mm: float = 0.0
    tibial_slope_delta_deg: float = 0.0
    tibial_varus_delta_deg: float = 0.0
    tibial_rotation_delta_deg: float = 0.0
    tibial_shift_ap_mm: float = 0.0
    tibial_shift_ml_mm: float = 0.0

    philosophy: str = "mechanical"
    size_override: str = ""
    tibial_resection_mm: float = 8.0
    insert_thickness_mm: float = 9.0
    use_insert: bool = True
    resection_mode: str = "block"
    build_bone_shells: bool = True
    show_planes: bool = True
    show_axes: bool = True
    show_landmarks: bool = False
    isolate_landmarks: bool = False


@dataclass
class TrialControls:
    """The three Reduce-mode controls. They never touch the plan."""

    flexion_deg: float = 0.0
    varus_valgus_deg: float = 0.0
    drawer_ap_mm: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.flexion_deg == 0.0
            and self.varus_valgus_deg == 0.0
            and self.drawer_ap_mm == 0.0
        )


@dataclass
class PlanningSession:
    """A patient, a plan, a scene, and the controls that move both."""

    femur_path: Path
    tibia_path: Path
    side: str
    library: Path | None = None
    controls: Controls = field(default_factory=Controls)
    trial: TrialControls = field(default_factory=TrialControls)
    case_id: str = ""

    scene = None
    plan = None
    sizing = None
    chart = None
    stale: bool = False

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, folder, *, side: str, library=None) -> "PlanningSession":
        """Find the two bone meshes in a patient folder and measure them."""
        femur_path, tibia_path = find_bone_files(Path(folder), side)
        session = cls(
            femur_path=femur_path, tibia_path=tibia_path, side=side,
            library=Path(library) if library else None,
            case_id=Path(folder).name,
        )
        session._measure()
        return session

    def _measure(self) -> None:
        """Read the meshes, estimate the landmarks, build the frames and the metrics.

        Cached for the life of the session: this is the expensive part, and no control
        changes it. Re-planning reuses it and costs a few dozen numpy operations.

        This is the call sequence the add-on's Plan operator used, lifted out unchanged.
        """
        self._femur = read_stl(self.femur_path)
        self._tibia = read_stl(self.tibia_path)
        self._landmarks = estimate_landmarks(
            self._femur, self._tibia, self.side, case_id=self.case_id
        )
        self._femoral_frame = build_femoral_frame(self._landmarks)
        self._tibial_frame = build_tibial_frame(self._landmarks)
        self._metrics = compute_all(
            self._landmarks, self._femoral_frame, self._tibial_frame
        )
        self._femoral_measure = measure_femoral_ml(self._femur, self._femoral_frame)
        self._tibial_measure = measure_tibial_plateau(self._tibia, self._tibial_frame)
        self._native_slope_deg = self._metrics["posterior_slope_medial_deg"].value

    # ------------------------------------------------------------------
    # Plan mode
    # ------------------------------------------------------------------

    def build(self) -> SceneDelta:
        """Compute the plan and build the scene. The equivalent of pressing Plan."""
        self.plan, self.sizing = self._plan()
        self.scene = scene_build.build_scene(
            femur_path=self.femur_path,
            tibia_path=self.tibia_path,
            plan=self.plan,
            femoral_frame=self._femoral_frame,
            tibial_frame=self._tibial_frame,
            landmarks=self._landmarks,
            components=self._components(),
            show_planes=self.controls.show_planes,
            show_axes=self.controls.show_axes,
            show_landmarks=self.controls.show_landmarks,
            insert_thickness_mm=self._insert_thickness(),
            insert_footprint_mm=self._insert_footprint(),
        )
        # The scene holds uncut bones until Commit runs, so it does not yet match the
        # plan's resections.
        self.stale = True
        return SceneDelta(scalars=self._scalars(), notes=list(self.scene.notes))

    def replan(self, **changes) -> SceneDelta:
        """Apply control changes, re-plan, and re-pose the scene.

        No boolean runs here. That is the Plan-mode guarantee, and it is what lets this
        be called on every change of a value rather than on a button press.
        """
        unknown = set(changes) - {f.name for f in fields(Controls)}
        if unknown:
            raise ValueError(f"Unknown control(s): {sorted(unknown)}")

        self.controls = replace(self.controls, **changes)
        self.plan, self.sizing = self._plan()

        delta = scene_update.update_scene(
            self.scene,
            self.plan,
            insert_thickness_mm=self._insert_thickness(),
            implant_ml_mm=self.sizing.implant_ml_mm,
        )
        if "isolate_landmarks" in changes:
            delta = delta.merge(
                scene_update.isolate_landmarks(
                    self.scene, self.controls.isolate_landmarks
                )
            )
        if "show_landmarks" in changes and not self.controls.isolate_landmarks:
            delta = delta.merge(
                scene_update.set_visibility(
                    self.scene,
                    lambda node: bool(node.tags.get("landmark")),
                    self.controls.show_landmarks,
                )
            )

        if self.bones_affected_by(tuple(changes)):
            self.stale = True

        delta.scalars.update(self._scalars())
        return delta

    def reset_adjustments(self) -> SceneDelta:
        """Return every manual control to the computed plan."""
        return self.replan(**{name: 0.0 for name in ADJUSTMENT_FIELDS})

    def bones_affected_by(self, changed: tuple) -> tuple:
        """Which bones a set of control changes actually reaches.

        In cutting-block mode a component's own pose moves the block that cuts it, so a
        shift or an in-plane rotation is not free: the block shares its implant's pose
        exactly, on purpose, because the block is what realises the cut. In plane mode
        those same controls move nothing a cutter depends on.
        """
        bones: set = set()
        block_mode = self.controls.resection_mode == "block"

        for name in changed:
            if name in NO_BONES:
                continue
            if any(token in name for token in COMPONENT_ONLY) and not block_mode:
                continue

            if name in FEMORAL_ONLY:
                bones.add("Femur")
            elif name in TIBIAL_ONLY:
                bones.add("Tibia")
            elif name in BOTH_BONES or name in ADJUSTMENT_FIELDS:
                bones.update({"Femur", "Tibia"})
        return tuple(sorted(bones))

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, kernel=None, *, bones=None):
        """Cut the bones. The only place in the session a boolean runs."""
        result = scene_resect.commit_resection(
            self.scene,
            self.plan,
            mode=self.controls.resection_mode,
            kernel=kernel,
            bones=bones or ("Femur", "Tibia"),
            build_shells=self.controls.build_bone_shells,
        )
        self.stale = False
        return result

    # ------------------------------------------------------------------
    # Reduce mode
    # ------------------------------------------------------------------

    def set_trial(self, **changes) -> SceneDelta:
        """Pose the implanted construct. Never re-plans, never cuts."""
        unknown = set(changes) - {f.name for f in fields(TrialControls)}
        if unknown:
            raise ValueError(f"Unknown trial control(s): {sorted(unknown)}")

        self.trial = replace(self.trial, **changes)
        pose = motion.trial_pose(
            self.scene.pivot,
            flexion_deg=self.trial.flexion_deg,
            varus_valgus_deg=self.trial.varus_valgus_deg,
            drawer_ap_mm=self.trial.drawer_ap_mm,
        )
        self.scene.set_pose(TIBIAL, pose)
        return SceneDelta(poses={"TibialSet": pose})

    def reset_trial(self) -> SceneDelta:
        """Return all three trial controls to zero, and the tibia to extension."""
        return self.set_trial(flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_lines(self) -> list:
        """The sidebar readout, rebuilt on every change so it can never lag the plan.

        Ported from the add-on, including its ``KEY|VALUE`` shape, so the existing panel
        keeps rendering it unchanged and a browser can parse the same lines.
        """
        plan, sizing = self.plan, self.sizing
        femoral = plan.resections["femoral_distal"]
        tibial = plan.resections["tibial_proximal"]
        diagnostics = plan.diagnostics

        lines = [
            f"CASE|{self.case_id} ({self.side})",
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
            f"Femoral ML|{self._femoral_measure.ml_mm:.1f} mm",
            f"Plateau AP|{self._tibial_measure.ap_mm:.1f} mm",
            f"Size parameter|{sizing.size_parameter:.2f}",
            f"Implant width|{sizing.implant_ml_mm:.1f} mm",
            f"Nearest size|{sizing.nearest_discrete_size}",
            "",
            "HEAD|Alignment",
        ]
        for name in ("mldfa_deg", "aldfa_deg", "mpta_deg", "jlca_deg"):
            metric = self._metrics[name]
            value = f"{metric.value:.1f}" if metric.value is not None else "n/a"
            lines.append(
                f"{name.replace('_deg', '').upper()}|{value} ({metric.quality.value})"
            )

        if self._metrics["hka_deviation_deg"].value is None:
            lines += ["", "WARN|HKA needs hip and ankle: outside this scan"]
        if sizing.flags:
            lines.append(f"WARN|Sizing: {', '.join(sizing.flags)}")
        for warning in plan.warnings:
            lines.append(f"WARN|{warning}")

        if self.stale:
            lines.append("WARN|Geometry does not match the plan: press Commit")
        return lines

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _plan(self):
        """Re-plan from the cached measurement. Pure numpy, so it is cheap enough to
        run on every change of a control rather than on a button press."""
        sizing = self._sizing()
        plan = plan_alignment(
            self._landmarks, self._femoral_frame, self._tibial_frame,
            target=(
                MECHANICAL if self.controls.philosophy == "mechanical" else KINEMATIC
            ),
            femoral_thickness_mm=sizing.femoral_thickness_mm,
            tibial_resection_mm=self.controls.tibial_resection_mm,
            native_slope_deg=self._native_slope_deg,
            adjustments=self._adjustments(),
            femur_mesh=self._femur, tibia_mesh=self._tibia,
        )
        return plan, sizing

    def _adjustments(self) -> Adjustments:
        return Adjustments(
            **{name: getattr(self.controls, name) for name in ADJUSTMENT_FIELDS},
            insert_thickness_mm=self._insert_thickness(),
        )

    def _sizing(self):
        """The implant size, solved from the anatomy or set by hand.

        An empty or unrecognised ``size_override`` means solve it from the measurement,
        which is also the right answer if a saved case names a size the chart no longer
        publishes.
        """
        chart = self._size_chart()
        label = self.controls.size_override
        override = chart.label_parameter(label) if label in chart.labels else None
        return solve_parametric_size(
            chart,
            measured_ml_mm=self._femoral_measure.ml_mm,
            measured_ap_mm=self._femoral_measure.ap_mm,
            parameter_override=override,
        )

    def _size_chart(self):
        if self.chart is None:
            self.chart = load_size_chart(find_size_chart())
        return self.chart

    def _components(self) -> dict:
        if self.library is None:
            return {}
        return scene_build.resolve_component_meshes(
            self.library, chart=self._size_chart(), sizing=self.sizing, side=self.side
        )

    def _insert_thickness(self) -> float | None:
        return self.controls.insert_thickness_mm if self.controls.use_insert else None

    def _insert_footprint(self) -> tuple:
        """The tray's footprint, from the sizing decision rather than from a mesh.

        Taken from the resection cross-section the sizing solved, which is where the
        add-on took it from too.
        """
        return (
            self.sizing.diagnostics.get("tibia_ML_mm", 70.0),
            self.sizing.diagnostics.get("tibia_AP_mm", 48.0),
        )

    def _scalars(self) -> dict:
        return {
            "distal_femoral_valgus_cut_deg": float(
                self.plan.distal_femoral_valgus_cut_deg
            ),
            "tibial_slope_deg": float(self.plan.tibial_slope_deg),
            "adjusted": not self.plan.adjustments.is_identity,
            "stale": self.stale,
        }


def find_size_chart() -> Path:
    """Locate SizeChart.csv, whether running from the repository or from an install.

    Ported from the add-on. Running from a checkout the repository copy wins, so edits
    take effect without rebuilding; installed, the bundled copy is used.
    """
    package = Path(__file__).resolve().parent
    candidates = [
        package.parent / "data" / "SizeChart.csv",
        package / "data" / "SizeChart.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "SizeChart.csv not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def find_bone_files(folder: Path, side: str) -> tuple:
    """Locate the femur and tibia surfaces in a patient folder.

    Tries the project's ``FD1Left.stl`` / ``TD1Left.stl`` convention first, then falls
    back to any STL named for the bone, so a folder exported straight out of 3D Slicer
    also works. Ported from the add-on unchanged.
    """
    folder = Path(folder)
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
