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
from tka_planner.pipeline import (
    find_size_chart, measure_case, plan_case, trajectory_case,
)
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
    "show_landmarks", "isolate_landmarks", "show_cutting_blocks", "show_fit_map",
    "show_phantom",
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
    # The phantom femur, drawn only when a femur length was entered.
    show_phantom: bool = True
    # The fit on each cut as a colour map, drawn once Commit has measured it.
    show_fit_map: bool = True


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
    # The flexion trajectory (WP-D level D1): S3 is the pure hinge Reduce mode has
    # always used; S1 and S2 add the scenarios' rollback or slide and axial rotation.
    # See tka_planner.core.kinematics.
    trajectory: str = "S3"

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
    # Remove every detached piece of a bone, not only those too small to be anatomy.
    # Quality control reports the large ones; this is the option it offers.
    remove_large_islands: bool = False
    # Hip centre to knee centre, entered: a knee-only scan cannot give it.
    femur_length_mm: float | None = None

    measurement = None

    scene = None
    plan = None
    sizing = None
    stale: bool = False
    # The fit and its maps, measured at the last commit, and the plan they were
    # measured for: a re-plan that moves a component makes them out of date.
    fit = None
    fit_grids = None
    fit_key = None
    fit_seconds = None
    fit_note = None

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, folder, *, side: str, library=None,
             landmarks=None, remove_large_islands: bool = False,
             femur_length_mm: float | None = None) -> "PlanningSession":
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
            remove_large_islands=remove_large_islands,
            femur_length_mm=femur_length_mm,
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
            remove_large_islands=self.remove_large_islands,
            femur_length_mm=self.femur_length_mm,
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

    def _phantom(self):
        """The schematic femur at the entered length, or ``None`` without one."""
        length = self.measurement.entered.get("femur_length_mm")
        if length is None:
            return None
        from tka_planner.core.phantom_femur import build_phantom_femur

        return build_phantom_femur(
            self._femoral_frame, length,
            width_mm=self.measurement.femoral_measure.ml_mm)

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
            phantom=self._phantom(),
            show_phantom=self.controls.show_phantom,
            insert_thickness_mm=self._insert_thickness(),
            insert_footprint_mm=self._insert_footprint(),
            insert_stretch=self._insert_stretch(),
            component_scales=self.component_scales,
            femur_mesh=self._femur,
            tibia_mesh=self._tibia,
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
                ("show_phantom", "phantom"),
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
        if not self.controls.isolate_landmarks:
            delta = delta.merge(self._fit_map_visibility())

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

        if "trajectory" in changes:
            # Refuse an unknown name here, before it is stored.
            from tka_planner.core.kinematics import scenario
            scenario(changes["trajectory"])
        self.trial = replace(self.trial, **changes)
        pose = self.trial_pose_at(self.trial.flexion_deg)
        self.scene.set_pose(TIBIAL, pose)
        return SceneDelta(poses={"TibialSet": pose},
                          scalars={"distraction_mm": self.trial.distraction_mm})

    def trial_pose_at(self, flexion_deg: float):
        """The tibial set pose at a flexion angle under the other trial controls.

        One function for the slider and the scripted arc, so the two cannot disagree
        about the trajectory or the stress.
        """
        frame = self._tibial_frame
        return motion.trial_pose(
            self.scene.pivot,
            flexion_deg=flexion_deg,
            varus_valgus_deg=self.trial.varus_valgus_deg,
            drawer_ap_mm=self.trial.drawer_ap_mm,
            distraction_mm=self.trial.distraction_mm,
            distal=-frame.z_proximal,
            # Resolved for this knee, as the flexion search and the plan file do: S1's
            # pivot is scaled to its condyle-centre spacing.
            trajectory=trajectory_case(self.measurement, self.trial.trajectory),
            anterior=frame.x_anterior,
            medial=frame.medial,
        )

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
            "HEAD|Cut depths at the datum landmarks",
            f"Femoral medial|{femoral.medial_depth_mm:.1f} mm",
            f"Femoral lateral|{femoral.lateral_depth_mm:.1f} mm",
            f"Tibial medial|{tibial.medial_depth_mm:.1f} mm",
            f"Tibial lateral|{tibial.lateral_depth_mm:.1f} mm",
            f"Tibia measured from|{diagnostics['tibial_resection_datum']}",
        ]
        box = self.femoral_cut_box
        if box is not None and box.get("available"):
            cuts = box["resections_mm"]
            lines += [
                f"Femoral posterior medial|{cuts['posterior_medial']:.1f} mm",
                f"Femoral posterior lateral|{cuts['posterior_lateral']:.1f} mm",
                f"Femoral anterior|{cuts['anterior']:.1f} mm",
                f"Patellofemoral lowering|{box['patellofemoral_lowering_mm']:+.1f} mm",
                f"Flange height|{box['flange_height_mm']:.1f} mm",
            ]
        elif box is not None:
            lines.append(f"Femoral box|not placed: {box['reason']}")

        lines += self._table_lines()
        lines += self._fit_lines()

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
        if diagnostics.get("tray_rotation_from_tibial_axis_deg") is not None:
            lines += [
                "",
                "HEAD|Tray on the tibia",
                f"Rotation off tibial axis|"
                f"{diagnostics['tray_rotation_from_tibial_axis_deg']:+.1f} deg",
                f"Offset from cut centre AP|"
                f"{diagnostics['tray_offset_from_tibial_cut_centre_anterior_mm']:+.1f} mm",
                f"Offset from cut centre ML|"
                f"{diagnostics['tray_offset_from_tibial_cut_centre_lateral_mm']:+.1f} mm",
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
        if box is not None and box.get("available") and box["anterior_notch_mm"] > 0.5:
            lines.append(f"WARN|The anterior femoral cut notches the cortex by "
                         f"{box['anterior_notch_mm']:.1f} mm")

        if self.stale:
            lines.append("WARN|Geometry does not match the plan: press Commit")
        return lines

    def _table_lines(self) -> list:
        """The resection table in the sidebar: medial / lateral, then the angles."""
        from tka_planner.core.resection_table import ROW_LABELS

        table = self.resection_table
        if table is None or not table.get("rows"):
            return []

        def pair(medial, lateral):
            def one(value):
                return "-" if value is None else f"{value:.1f}"
            return f"{one(medial)} / {one(lateral)} mm"

        lines = ["", "HEAD|Resection table (medial / lateral)"]
        angles = ["", "HEAD|Cut angles"]
        for row in table["rows"]:
            label = ROW_LABELS.get(row["cut"], row["cut"])
            if row["cut"] == "tibial":
                label = "Tibia at stylus point"
            lines.append(f"{label}|{pair(row['medial_mm'], row['lateral_mm'])}")
            if row.get("highest_mm"):
                highest = row["highest_mm"]
                lines.append(f"Tibia highest bone|"
                             f"{pair(highest['medial'], highest['lateral'])}")
            a = row.get("angle_deg") or {}
            if row["cut"] == "distal":
                angles.append(f"Distal|varus {a.get('varus_from_mechanical_axis', 0):+.1f}"
                              f", flexion {a.get('flexion', 0):+.1f} deg")
            elif row["cut"] == "posterior" and a:
                angles.append(f"Posterior, ER from PCA|"
                              f"{a['external_rotation_from_posterior_condylar_axis']:+.1f}"
                              f" deg")
            elif row["cut"] == "anterior" and a:
                angles.append(f"Anterior flare|{a['flare_from_component_axis']:.1f} deg")
            elif row["cut"] == "tibial":
                angles.append(f"Tibial|varus {a.get('varus_from_mechanical_axis', 0):+.1f}"
                              f", slope {a.get('posterior_slope', 0):.1f} deg")
        return lines + angles

    def _fit_lines(self) -> list:
        """The fit at the last commit, and whether it still describes this plan."""
        if self.fit is None:
            if self.fit_note:
                return ["", "HEAD|Fit on the cuts", f"WARN|{self.fit_note}"]
            return []
        current = self.fit_is_current
        lines = ["", "HEAD|Fit on the cuts" + ("" if current else " (out of date)")]
        for name, label in (("tibial_component", "Tray"),
                            ("femoral_component", "Femoral")):
            record = self.fit.get(name)
            if not record:
                continue
            lines += [
                f"{label} coverage|{100 * record['coverage_fraction']:.1f}%",
                f"{label} overhang max|{record['max_overhang_mm']:.1f} mm",
                f"{label} bone uncovered max|{record['max_underhang_mm']:.1f} mm",
            ]
        if not current:
            lines.append("WARN|The fit is of the last committed plan: press Commit")
        return lines

    # ------------------------------------------------------------------
    # The fit, the resection table and the findings
    # ------------------------------------------------------------------

    @property
    def resection_table(self) -> dict | None:
        """The resection table on the current plan, measured on the bone.

        Kept until the femoral component or the tibial cut moves: a display toggle or an
        insert change re-plans without moving either.
        """
        import numpy as np

        from tka_planner import pipeline

        if self.plan is None:
            return None
        tibial = self.plan.resections["tibial_proximal"]
        box = self.femoral_cut_box
        key = (id(self.measurement), id(box),
               np.asarray(self.plan.components["femoral_component"], float).tobytes(),
               np.asarray(tibial.point, float).tobytes(),
               np.asarray(tibial.normal, float).tobytes())
        cached = getattr(self, "_table_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, pipeline.resection_table_case(
                self.measurement, self.plan, self.library, box))
            self._table_cache = cached
        return cached[1]

    def refresh_fit(self) -> SceneDelta:
        """Measure the fit on the current plan and draw it as a colour map on each cut.

        Run on Commit rather than on every change of a control: it slices both bones and
        both components, a second or two on a real knee. The maps are the fit's own
        grids, so what is drawn is exactly what the numbers were read from.
        """
        import time

        from tka_planner.pipeline import fit_case_with_grids
        from tka_planner.scene.fitmap import FIT_MAP_NODES, fit_map_mesh
        from tka_planner.scene.model import Node

        delta = SceneDelta()
        if self.library is None or self.plan is None:
            return delta
        start = time.perf_counter()
        try:
            fit, grids = fit_case_with_grids(self.measurement, self.plan, self.sizing,
                                             self.library, self.controls.implant_mode,
                                             transforms=self.component_scales)
        except ValueError as error:
            fit, grids = None, {}
            self.fit_note = f"The fit could not be measured: {error}"
        self.fit, self.fit_grids, self.fit_key = fit, grids, self._fit_key()
        if fit is not None:
            self.fit_note = None

        for component, (name, set_id) in FIT_MAP_NODES.items():
            grid = grids.get(component)
            mesh = fit_map_mesh(grid) if grid is not None else None
            node = self.scene.nodes.get(name)
            if mesh is None:
                if node is not None and node.visible:
                    node.visible = False
                    delta.visibility[name] = False
                continue
            if node is None:
                node = Node(name=name, set_id=set_id, colour=(1.0, 1.0, 1.0),
                            visible=False,
                            tags={"fit_map": True, "group": set_id,
                                  "resolution_mm": mesh.metadata["resolution_mm"]})
                self.scene.add(node, mesh)
                delta.meshes[name] = node.mesh_id
            else:
                delta.meshes[name] = self.scene.replace_mesh(name, mesh)
        self.fit_seconds = time.perf_counter() - start
        return delta.merge(self._fit_map_visibility(force=True))

    def _fit_key(self) -> str:
        """What the fit depends on: the components' poses, the cuts, the implant."""
        import hashlib

        import numpy as np

        digest = hashlib.sha256()
        for name in sorted(self.plan.components):
            digest.update(np.asarray(self.plan.components[name], float).tobytes())
        for name in sorted(self.plan.resections):
            resection = self.plan.resections[name]
            digest.update(np.asarray(resection.point, float).tobytes())
            digest.update(np.asarray(resection.normal, float).tobytes())
        digest.update(f"{self.controls.implant_mode}|{self.sizing.size_parameter!r}|"
                      f"{self.plan.diagnostics.get('insert_thickness_mm')!r}"
                      .encode("utf-8"))
        return digest.hexdigest()

    @property
    def fit_is_current(self) -> bool:
        """Whether the last measured fit is of the plan as it stands."""
        return (self.fit is not None and self.plan is not None
                and self.fit_key == self._fit_key())

    def _fit_map_visibility(self, force: bool = False) -> SceneDelta:
        """Show the maps when asked for and still true of the plan; hide them otherwise."""
        from tka_planner.scene.fitmap import FIT_MAP_NODES

        delta = SceneDelta()
        if self.scene is None:
            return delta
        show = bool(self.controls.show_fit_map) and self.fit_is_current
        for name, _ in FIT_MAP_NODES.values():
            node = self.scene.nodes.get(name)
            if node is not None and (node.visible != show or force):
                node.visible = show
                delta.visibility[name] = show
        return delta

    def fit_map_state(self) -> dict:
        """What the viewer needs beside the maps themselves: the legend and its status."""
        from tka_planner.core.fit import OVERHANG_FLAG_MM
        from tka_planner.scene.fitmap import FIT_MAP_NODES, legend

        drawn = [name for name, _ in FIT_MAP_NODES.values()
                 if self.scene is not None and name in self.scene.nodes]
        record = legend(OVERHANG_FLAG_MM)
        if drawn:
            record["resolution_mm"] = self.scene.nodes[drawn[0]].tags.get(
                "resolution_mm")
        return {
            "available": bool(drawn) and self.fit is not None,
            "current": self.fit_is_current,
            "visible": any(self.scene.nodes[name].visible for name in drawn),
            "nodes": drawn,
            "legend": record,
            "seconds": None if self.fit_seconds is None else round(self.fit_seconds, 2),
            "note": self.fit_note,
        }

    def findings(self, fit=None) -> list:
        """Every finding on the current plan, each with its place where it has one.

        The fit's findings come from ``fit`` when given, and otherwise from the last
        commit's fit while it still describes this plan.
        """
        from tka_planner.pipeline import findings_case

        if self.plan is None:
            return []
        if fit is None and self.fit_is_current:
            fit = self.fit
        return findings_case(self.measurement, self.plan, self.sizing,
                             box=self.femoral_cut_box, fit=fit)

    def report_html(self) -> str:
        """The HTML report on the current plan, with the fit measured if it is not."""
        from tka_planner.pipeline import (
            fit_case_with_grids,
            implant_spec_case,
            render_case_report,
        )

        fit, grids = (self.fit, self.fit_grids) if self.fit_is_current else (None, None)
        if fit is None and self.library is not None:
            try:
                fit, grids = fit_case_with_grids(self.measurement, self.plan,
                                                 self.sizing, self.library,
                                                 self.controls.implant_mode,
                                                 transforms=self.component_scales)
            except ValueError:
                fit, grids = None, None
        return render_case_report(
            self.measurement, self.plan, self.sizing, fit=fit, fit_grids=grids,
            implant_spec=implant_spec_case(self.measurement, self.plan, self.sizing,
                                           self.library),
            resection_table=self.resection_table,
            findings=self.findings(fit),
        )

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

    @property
    def femoral_cut_box(self) -> dict | None:
        """The femoral cut box on the current plan.

        It depends only on the femur and the femoral component's pose, so it is kept
        until one of those changes: a tibial control or a display toggle re-plans without
        moving it, and does not pay to place it again.
        """
        import numpy as np

        from tka_planner import pipeline

        if self.library is None or self.plan is None:
            return None
        pose = np.asarray(self.plan.components["femoral_component"], dtype=float)
        key = (id(self.measurement), str(self.library), pose.tobytes())
        cached = getattr(self, "_cut_box_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, pipeline.femoral_cut_box_case(
                self.measurement, self.plan, self.library))
            self._cut_box_cache = cached
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
