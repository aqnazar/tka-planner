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

from tka_planner.core.planning import Adjustments
from tka_planner.pipeline import find_size_chart, measure_case, plan_case
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
TIBIAL_ONLY = TIBIAL_ONLY | {"tibial_reference"}
BOTH_BONES = frozenset({
    "coronal_correction_deg", "philosophy", "size_override", "implant_mode",
    "resection_mode", "build_bone_shells",
})
# Controls that move nothing a boolean depends on, in any mode.
NO_BONES = frozenset({
    "insert_thickness_delta_mm", "show_planes", "show_axes",
    "show_landmarks", "isolate_landmarks", "show_cutting_blocks",
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
    tibial_reference: str = "less_affected_plateau"
    # Patient-specific: the implant takes the patient's own dimensions from the cuts.
    # Catalogue: the library part at the size's single scale, as the legacy pipeline.
    implant_mode: str = "patient_specific"
    # The insert is solved to close the joint; this is the surgeon's thicker or thinner.
    insert_thickness_delta_mm: float = 0.0
    resection_mode: str = "block"
    build_bone_shells: bool = True
    show_planes: bool = True
    show_axes: bool = True
    show_landmarks: bool = False
    isolate_landmarks: bool = False
    # The blocks are the size of the instrument, not of the cut, so they stand in front
    # of the bone they are cutting. Off by default: the first thing a plan should show
    # is the anatomy and where the implant sits on it.
    show_cutting_blocks: bool = False


@dataclass
class TrialControls:
    """The Reduce-mode controls. They never touch the plan.

    The plan closes the joint with no gap; ``distraction_mm`` opens it, rigidly, to show
    the gap a given pull would leave.
    """

    flexion_deg: float = 0.0
    varus_valgus_deg: float = 0.0
    drawer_ap_mm: float = 0.0
    distraction_mm: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.flexion_deg == 0.0
            and self.varus_valgus_deg == 0.0
            and self.drawer_ap_mm == 0.0
            and self.distraction_mm == 0.0
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
    landmarks_path: Path | None = None

    measurement = None

    scene = None
    plan = None
    sizing = None
    stale: bool = False

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, folder, *, side: str, library=None,
             landmarks=None) -> "PlanningSession":
        """Find the two bone meshes in a patient folder and measure them.

        ``landmarks`` is a landmark file laid over the automatic estimate, exactly as
        the command line's ``--landmarks`` is.
        """
        femur_path, tibia_path = find_bone_files(Path(folder), side)
        session = cls(
            femur_path=femur_path, tibia_path=tibia_path, side=side,
            library=Path(library) if library else None,
            case_id=Path(folder).name,
            landmarks_path=Path(landmarks) if landmarks else None,
        )
        session._measure()
        return session

    def _measure(self) -> None:
        """Read, check and measure the case, through the same pipeline as the command
        line: quality control, landmarks, frames, metrics and the sizing measurement.

        Cached for the life of the session: this is the expensive part, and no control
        changes it. Re-planning reuses it and costs a few dozen numpy operations.
        """
        self.measurement = measure_case(
            self.femur_path, self.tibia_path, self.side,
            case_id=self.case_id or None, landmarks_path=self.landmarks_path,
        )

    # The measurement's parts, under the names the scene, the report and the export
    # have always read. They are views of the one measurement, never copies of it.
    @property
    def chart(self):
        return self.measurement.chart

    @property
    def _femur(self):
        return self.measurement.femur

    @property
    def _tibia(self):
        return self.measurement.tibia

    @property
    def _landmarks(self):
        return self.measurement.landmarks

    @property
    def _femoral_frame(self):
        return self.measurement.femoral_frame

    @property
    def _tibial_frame(self):
        return self.measurement.tibial_frame

    @property
    def _metrics(self):
        return self.measurement.metrics

    @property
    def _femoral_measure(self):
        return self.measurement.femoral_measure

    @property
    def _tibial_measure(self):
        return self.measurement.tibial_measure

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
            show_cutting_blocks=self.controls.show_cutting_blocks,
            insert_thickness_mm=self._insert_thickness(),
            insert_footprint_mm=self._insert_footprint(),
            insert_stretch=self._insert_stretch(),
            component_scales=self.component_scales,
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
            insert_stretch=self._insert_stretch(),
            component_scales=self.component_scales,
        )
        if "isolate_landmarks" in changes:
            delta = delta.merge(
                scene_update.isolate_landmarks(
                    self.scene, self.controls.isolate_landmarks
                )
            )
        # Isolation owns visibility while it is on, so the individual toggles stand
        # aside rather than fight it. What they set is remembered and applied when
        # isolation ends, which is why they are still written to the controls.
        if not self.controls.isolate_landmarks:
            for control, tag in (
                ("show_landmarks", "landmark"),
                ("show_planes", "plane"),
                ("show_axes", "axis"),
                ("show_cutting_blocks", "cutting_block"),
            ):
                if control not in changes:
                    continue
                delta = delta.merge(
                    scene_update.set_visibility(
                        self.scene,
                        lambda node, tag=tag: bool(node.tags.get(tag)),
                        getattr(self.controls, control),
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
            distraction_mm=self.trial.distraction_mm,
            distal=-self._tibial_frame.z_proximal,
        )
        self.scene.set_pose(TIBIAL, pose)
        return SceneDelta(poses={"TibialSet": pose},
                          scalars={"distraction_mm": self.trial.distraction_mm})

    def reset_trial(self) -> SceneDelta:
        """Return every trial control to zero, and the tibia to extension."""
        return self.set_trial(flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0,
                              distraction_mm=0.0)

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
            f"Tibia measured from|{diagnostics['tibial_resection_datum']}",
        ]

        if diagnostics.get("insert_thickness_mm") is not None:
            lines += [
                "",
                "HEAD|Insert and extension gap",
                f"Insert|{diagnostics['insert_thickness_mm']:.1f} mm"
                + (" (solved)" if diagnostics.get("insert_solved") else ""),
                f"Tray|{diagnostics['tray_thickness_mm']:.1f} mm",
                f"Gap medial|{diagnostics['extension_gap_medial_mm']:.1f} mm",
                f"Gap lateral|{diagnostics['extension_gap_lateral_mm']:.1f} mm",
            ]
        if diagnostics.get("component_rotation_mismatch_deg") is not None:
            lines += [
                "",
                "HEAD|Femoral over tibial component",
                f"Rotation|{diagnostics['component_rotation_mismatch_deg']:+.1f} deg",
                f"Anterior offset|{diagnostics['component_offset_anterior_mm']:+.1f} mm",
                f"Lateral offset|{diagnostics['component_offset_lateral_mm']:+.1f} mm",
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
        return plan_case(
            self.measurement,
            philosophy=self.controls.philosophy,
            size_label=self.controls.size_override,
            adjustments=self._adjustments(),
            tibial_reference=self.controls.tibial_reference,
            library=self.library,
        )

    def _adjustments(self) -> Adjustments:
        return Adjustments(
            **{name: getattr(self.controls, name) for name in ADJUSTMENT_FIELDS},
            insert_thickness_delta_mm=self.controls.insert_thickness_delta_mm,
        )

    def _size_chart(self):
        return self.measurement.chart

    def _components(self) -> dict:
        if self.library is None:
            return {}
        return scene_build.resolve_component_meshes(
            self.library, chart=self._size_chart(), sizing=self.sizing, side=self.side
        )

    def _insert_thickness(self) -> float | None:
        """The solved insert, for the placeholder slab shown when no library insert is.

        With a library the real insert is in the scene and the slab is not needed."""
        if self.plan is None or any(
            name == "tibial_insert" for name in self._component_names()
        ):
            return None
        return self.plan.diagnostics.get("insert_thickness_mm")

    @property
    def component_scales(self) -> dict | None:
        """Per-axis scales of each bone's library parts, for a patient-specific
        implant; ``None`` for the catalogue implant or with no library."""
        from tka_planner.pipeline import component_scales

        if self.library is None or self.plan is None:
            return None
        # Measuring the cuts takes a few tenths of a second, and this is read several
        # times per re-plan, so it is kept for as long as the plan it measured.
        key = (id(self.plan), self.controls.implant_mode)
        cached = getattr(self, "_transforms_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, component_scales(self.measurement, self.plan, self.sizing,
                                            self.library, self.controls.implant_mode))
            self._transforms_cache = cached
        return cached[1]

    def _insert_stretch(self) -> dict | None:
        """The library insert stretched to the solved thickness, so no gap shows."""
        from tka_planner.pipeline import insert_display

        if self.library is None or self.plan is None:
            return None
        return insert_display(self.plan, self._components())

    def _component_names(self) -> list:
        if self.scene is None:
            return list(self._components())
        return [node.name for node in self.scene.nodes.values()
                if node.tags.get("component")]

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
