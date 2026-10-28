"""One planning pipeline for every front end.

The command line and the application used to assemble a plan separately. The command
line ran quality control and could read a landmark file; the application did neither,
and the two wrote ``plan.json`` in different shapes. A paper comparing plans cannot rest
on two pipelines that agree only by coincidence, so both now call the functions here:

* :func:`measure_case` -- read the meshes, check them, place the landmarks, build the
  frames and the metrics, and measure the bone for sizing. The expensive part, run once
  per case.
* :func:`plan_case` -- size the implant and plan the cuts. Cheap, run on every change of
  a control.
* :func:`plan_document` -- the one ``plan.json`` format, versioned and validated.
* :func:`render_case_report` -- the one HTML report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .core.condylar_offset import (
    planned_condylar_offset,
    posterior_condylar_offset_metrics,
)
from .core.frames import AnatomicalFrame, build_femoral_frame, build_tibial_frame
from .core.landmarks import LandmarkSet, load_landmarks, overlay_landmarks
from .core.landmarks_auto import estimate_landmarks, femoral_axes, tibial_axes
from .core.measure import measure_femoral_ml, measure_tibial_plateau
from .core.meshio import Mesh, read_stl
from .core.metrics import compute_all
from .core.planning import (
    KINEMATIC,
    MECHANICAL,
    NO_ADJUSTMENT,
    TIBIAL_REFERENCE_DEFAULT_MM,
    Adjustments,
    SurgicalPlan,
    plan_alignment,
)
from .core.qc import (
    QCReport,
    assess_femur_coverage,
    assess_mesh_quality,
    assess_tibia_coverage,
    check_landmarks_on_mesh,
    check_laterality,
    check_plausible_bone_scale,
    check_entered_femur_length,
    check_scan_orientation,
    check_repairable,
    clean_islands,
)
from .core.sides import Side
from .geom.repair import remember_welded
from .core.sizing import (
    SizeChart,
    SizingDecision,
    load_size_chart,
    select_discrete_size,
    solve_parametric_size,
)

__all__ = [
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "CaseMeasurement",
    "measure_case",
    "measure_meshes",
    "measure_from_landmarks",
    "plan_case",
    "discrete_sizing",
    "plan_document",
    "validate_plan_document",
    "write_plan",
    "read_plan",
    "render_case_report",
    "find_size_chart",
    "fit_case",
    "fit_case_with_grids",
    "findings_case",
    "resection_table_case",
    "phantom_femur_case",
    "box_design",
    "femoral_cut_box_case",
    "implant_spec_case",
    "component_scales",
    "IMPLANT_MODES",
    "write_implant_spec",
    "FLEXION_PAIRS",
    "FlexionSettings",
    "FlexionModel",
    "flexion_model",
    "flexion_case",
    "scan_extent",
    "no_contact_reason",
]

# How the components are shaped. A patient-specific implant takes its width and depth
# from the patient's own cuts; the catalogue implant is the library part at one scale.
IMPLANT_MODES = ("patient_specific", "catalogue")

PLAN_SCHEMA = "tka-planner/plan"
# Version 1 was the unversioned pair of shapes the command line and the application
# wrote before they shared this module.
PLAN_SCHEMA_VERSION = 2
INTENDED_USE = "research_and_demonstration_only__not_a_medical_device"

# Every plan document carries these, whichever front end wrote it.
REQUIRED_KEYS = (
    "schema", "schema_version", "case_id", "side", "tool_version", "inputs",
    "scan_coverage", "frames", "metrics", "surgical_plan", "sizing",
    "quality_control", "landmarks", "implant_spec", "fit", "controls", "trial",
    "geometry",
    "geometry_matches_plan", "kinematics", "intended_use",
)
# Written into every document since they were added within version 2, but not required
# of one, so a version-2 file written before them still reads.
ADDED_KEYS = ("findings", "resection_table")
DOCUMENT_KEYS = REQUIRED_KEYS + ADDED_KEYS
REQUIRED_SIZING_KEYS = (
    "parametric", "discrete", "femoral_measurement", "tibial_measurement",
)


@dataclass
class CaseMeasurement:
    """Everything about a case that no planning control can change."""

    case_id: str
    side: Side
    femur_path: Path
    tibia_path: Path
    femur: Mesh
    tibia: Mesh
    landmarks: LandmarkSet
    femoral_frame: AnatomicalFrame
    tibial_frame: AnatomicalFrame
    metrics: dict
    femoral_measure: object
    tibial_measure: object
    coverage: dict
    qc: QCReport
    chart: SizeChart
    chart_path: Path
    # Values the operator entered rather than the scan gave, e.g. ``femur_length_mm``.
    entered: dict = field(default_factory=dict)

    @property
    def native_slope_deg(self) -> float | None:
        return self.metrics["posterior_slope_medial_deg"].value

    def landmark_counts(self) -> dict:
        counts: dict[str, int] = {}
        for landmark in self.landmarks:
            counts[landmark.status.value] = counts.get(landmark.status.value, 0) + 1
        return counts


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


def measure_case(
    femur_path,
    tibia_path,
    side,
    *,
    case_id: str | None = None,
    landmarks_path=None,
    rater: str | None = None,
    session: str | None = None,
    size_chart=None,
    remove_large_islands: bool = False,
    femur_length_mm: float | None = None,
) -> CaseMeasurement:
    """Read, check and measure a case.

    Quality control runs here, so no front end can skip it. An error-level finding
    raises :class:`~tka_planner.core.qc.QualityControlError`; a frame that cannot be
    justified raises :class:`~tka_planner.core.frames.FrameConstructionError`. Both are
    refusals rather than failures, and the caller decides how to say so.

    The automatic landmark estimate always runs. A landmark file is laid over it, so
    each pick replaces its estimate and everything else stays estimated and says so.
    """
    femur_path, tibia_path = Path(femur_path), Path(tibia_path)
    return measure_meshes(
        read_stl(femur_path), read_stl(tibia_path), side,
        case_id=case_id or femur_path.parent.name,
        femur_path=femur_path, tibia_path=tibia_path,
        landmarks_path=landmarks_path, rater=rater, session=session,
        size_chart=size_chart, remove_large_islands=remove_large_islands,
        femur_length_mm=femur_length_mm,
    )


def measure_meshes(
    femur: Mesh,
    tibia: Mesh,
    side,
    *,
    case_id: str,
    femur_path=None,
    tibia_path=None,
    landmarks_path=None,
    rater: str | None = None,
    session: str | None = None,
    size_chart=None,
    mesh_quality: QCReport | None = None,
    remove_large_islands: bool = False,
    femur_length_mm: float | None = None,
) -> CaseMeasurement:
    """:func:`measure_case` for meshes already in memory.

    The metamorphic tests of the paper move, mirror, decimate and perturb a bone and
    measure it again; writing each variant to disk only to read it back would test the
    STL writer as much as the planner. Everything after reading the files is here, so a
    variant is measured exactly as the application measures the original.

    ``mesh_quality`` is the mesh-quality findings for both bones when the caller already
    has them for these exact triangles. Welding and counting edges is most of the cost of
    measuring a knee, and a rigid motion cannot change the topology it reports on, so a
    caller re-measuring rigidly moved copies may pass the original's findings. Anything
    that moves vertices non-rigidly must leave it ``None``.

    ``remove_large_islands`` removes every detached piece of a bone, not only those too
    small to be anatomy; the large ones are otherwise kept and reported.

    ``femur_length_mm`` is an entered femur length (hip centre to knee centre), which a
    knee-only scan cannot give. It is checked for plausibility and recorded as entered.
    """
    side = Side.parse(side)
    qc = QCReport()
    qc.extend(check_entered_femur_length(femur_length_mm))
    # Detached pieces too small to be anatomy are removed first: every estimator reads
    # the bone's extremes, and a speck beyond the bone becomes one. The weld is shared
    # with the mesh-quality check, which needs it too. A mesh that is not a closed
    # manifold is repaired now, as the kernel will repair it at Commit, and refused here
    # if it cannot be: better than accepting the knee and failing at the first cut.
    bones = {}
    for mesh, bone in ((femur, "femur"), (tibia, "tibia")):
        mesh, welded, fragment_report = clean_islands(
            mesh, bone, remove_large=remove_large_islands)
        qc.extend(_for_bone(fragment_report, bone))
        bones[bone] = mesh
        qc.extend(_for_bone(check_plausible_bone_scale(mesh, bone), bone))
        qc.extend(_for_bone(check_laterality(mesh, side), bone))
        if mesh_quality is None:
            quality, mesh_report = assess_mesh_quality(welded, already_welded=True)
            qc.extend(_for_bone(mesh_report, bone))
            if not quality.watertight or quality.n_degenerate_faces \
                    or quality.n_duplicate_faces:
                qc.extend(_for_bone(check_repairable(mesh, bone), bone))
            else:
                # Welded here already; the first Commit need not weld it again.
                remember_welded(mesh, welded)
    if mesh_quality is not None:
        qc.extend(mesh_quality)
    femur, tibia = bones["femur"], bones["tibia"]
    qc.raise_if_errors()

    # Each bone's own axes, which the landmark searches run in. A knee lying too far
    # from LPS for them to be trusted is refused here, before anything is built on it.
    femur_axes, tibia_axes = femoral_axes(femur, side), tibial_axes(tibia)
    qc.extend(check_scan_orientation(femur_axes, tibia_axes))
    qc.raise_if_errors()

    femur_coverage, femur_report = assess_femur_coverage(
        femur, superior=femur_axes.superior)
    tibia_coverage, tibia_report = assess_tibia_coverage(
        tibia, superior=tibia_axes.superior)
    qc.extend(_for_bone(femur_report, "femur"))
    qc.extend(_for_bone(tibia_report, "tibia"))

    landmarks = estimate_landmarks(femur, tibia, side, case_id=case_id,
                                   femur_axes=femur_axes, tibia_axes=tibia_axes)
    if landmarks_path:
        picks = load_landmarks(landmarks_path, case_id=case_id, side=side,
                               rater=rater, session=session)
        landmarks = overlay_landmarks(landmarks, picks)
    qc.extend(check_landmarks_on_mesh(landmarks, {"femur": femur, "tibia": tibia}))
    qc.raise_if_errors()

    measurement = measure_from_landmarks(
        landmarks, femur, tibia,
        femur_path=femur_path or femur.source_path or "",
        tibia_path=tibia_path or tibia.source_path or "",
        case_id=case_id, side=side,
        qc=qc, coverage={"femur": femur_coverage, "tibia": tibia_coverage},
        size_chart=size_chart,
    )
    if femur_length_mm is not None:
        measurement.entered["femur_length_mm"] = float(femur_length_mm)
    return measurement


def _for_bone(report: QCReport, bone: str) -> QCReport:
    """The same findings, each saying which bone it is about.

    A mesh check does not know which bone it was handed; without this a reader of the
    plan, or a viewer asked to show the finding, could not tell the femur's open edge
    from the tibia's.
    """
    from dataclasses import replace

    tagged = QCReport()
    tagged.findings = [replace(finding, context={"bone": bone, **finding.context})
                       for finding in report.findings]
    return tagged


def measure_from_landmarks(
    landmarks: LandmarkSet,
    femur: Mesh,
    tibia: Mesh,
    *,
    femur_path,
    tibia_path,
    case_id: str,
    side,
    qc: QCReport | None = None,
    coverage: dict | None = None,
    size_chart=None,
) -> CaseMeasurement:
    """The second half of :func:`measure_case`, for landmarks already in hand.

    Frames, metrics and the sizing measurement are built here and nowhere else, so a
    caller that brings its own landmarks -- a test, or a batch run on reviewed picks --
    still measures the case exactly as the application and the command line do.
    """
    femoral_frame = build_femoral_frame(landmarks)
    tibial_frame = build_tibial_frame(landmarks)
    metrics = compute_all(landmarks, femoral_frame, tibial_frame)
    # The one metric read off the surface rather than the landmarks: the femoral shaft's
    # posterior cortex is part of its definition.
    metrics.update(posterior_condylar_offset_metrics(femur.vertices, landmarks,
                                                     femoral_frame))

    chart_path = Path(size_chart) if size_chart else find_size_chart()
    return CaseMeasurement(
        case_id=case_id,
        side=Side.parse(side),
        femur_path=Path(femur_path),
        tibia_path=Path(tibia_path),
        femur=femur,
        tibia=tibia,
        landmarks=landmarks,
        femoral_frame=femoral_frame,
        tibial_frame=tibial_frame,
        metrics=metrics,
        femoral_measure=measure_femoral_ml(femur, femoral_frame),
        tibial_measure=measure_tibial_plateau(tibia, tibial_frame),
        coverage=coverage or {},
        qc=qc or QCReport(),
        chart=load_size_chart(chart_path),
        chart_path=chart_path,
    )


def plan_case(
    measurement: CaseMeasurement,
    *,
    philosophy: str = "mechanical",
    size_label: str | None = None,
    adjustments: Adjustments = NO_ADJUSTMENT,
    tibial_reference: str = "less_affected_plateau",
    library=None,
) -> tuple[SurgicalPlan, SizingDecision]:
    """Size the implant and plan the cuts for one set of controls.

    ``size_label`` picks a chart size by hand; empty, or a size the chart does not
    publish, means solve it from the anatomy.

    The tibial depth is the commercial default for ``tibial_reference`` -- 9 mm below
    the less affected plateau, or 2 mm below the more affected one -- and the chart's
    depth for the size when the reference is the top of the tibia, the legacy datum.
    The femoral depth is the component's distal thickness for the size. The surgeon's
    resection deltas move both from there.

    With ``library`` the tray thickness is measured from the implant library, so the
    solved insert is the polyethylene alone; without it the solved insert stands for tray
    and insert together.
    """
    if philosophy not in ("mechanical", "kinematic"):
        raise ValueError("philosophy must be 'mechanical' or 'kinematic'.")
    chart = measurement.chart
    override = chart.label_parameter(size_label) if size_label in chart.labels else None
    sizing = solve_parametric_size(
        chart,
        measured_ml_mm=measurement.femoral_measure.ml_mm,
        measured_ap_mm=measurement.femoral_measure.ap_mm,
        parameter_override=override,
    )
    default_depth = TIBIAL_REFERENCE_DEFAULT_MM.get(tibial_reference, 0.0)
    plan = plan_alignment(
        measurement.landmarks, measurement.femoral_frame, measurement.tibial_frame,
        target=MECHANICAL if philosophy == "mechanical" else KINEMATIC,
        femoral_thickness_mm=sizing.femoral_thickness_mm,
        tibial_resection_mm=(
            sizing.tibial_resection_mm if default_depth is None else default_depth
        ),
        tibial_reference=tibial_reference,
        tibial_tray_thickness_mm=tray_thickness(measurement, sizing, library),
        native_slope_deg=measurement.native_slope_deg,
        adjustments=adjustments,
        femur_mesh=measurement.femur,
        tibia_mesh=measurement.tibia,
    )
    return plan, sizing


def tray_thickness(measurement: CaseMeasurement, sizing: SizingDecision,
                   library) -> float:
    """The tray's thickness under the insert, from the implant library at this size.

    The tray and the insert share a CAD origin on the tibial cut, so the insert's lowest
    point is the top of the tray. Zero without a library.
    """
    from .scene.build import resolve_component_meshes

    if library is None:
        return 0.0
    specs = resolve_component_meshes(
        library, chart=measurement.chart, sizing=sizing, side=str(measurement.side)
    )
    insert = specs.get("tibial_insert")
    if insert is None:
        return 0.0
    return float(_cached_component(insert).vertices[:, 2].min()) * insert["scale"]


def _cached_stl(path) -> Mesh:
    """Library parts are read once per process: re-planning happens on every slider
    movement, and the parts never change underneath it."""
    key = str(path)
    if key not in _STL_CACHE:
        _STL_CACHE[key] = read_stl(key)
    return _STL_CACHE[key]


def _cached_component(spec: dict) -> Mesh:
    """A resolved library part, read once per process, mirrored if the resolver said so."""
    from .scene.build import read_component

    key = ("part", spec["path"], bool(spec.get("mirror")))
    if key not in _STL_CACHE:
        _STL_CACHE[key] = read_component(spec, reader=_cached_stl)
    return _STL_CACHE[key]


_STL_CACHE: dict = {}


def insert_display(plan: SurgicalPlan, specs: dict) -> dict | None:
    """How to stretch the library insert so its dish is the solved thickness.

    The library holds one insert height per size; the parametric insert is made to the
    solved thickness. Until Fusion builds it, the library part is stretched along its own
    axis about its floor, so the viewer shows the planned construct with no gap.
    """
    from .core.insert import dish_geometry

    spec = specs.get("tibial_insert")
    solved = plan.diagnostics.get("insert_thickness_mm")
    if spec is None or solved is None:
        return None
    key = ("dish", spec["path"], bool(spec.get("mirror")))
    if key not in _STL_CACHE:
        _STL_CACHE[key] = dish_geometry(_cached_component(spec))
    floor, dish = _STL_CACHE[key]
    library_mm = (dish - floor) * spec["scale"]
    factor = max(float(solved), 0.5) / library_mm
    return {
        "floor_cad_mm": floor,
        "stretch": factor,
        "library_thickness_mm": round(library_mm, 3),
        "insert_thickness_mm": round(float(solved), 3),
    }


def library_dimensions(spec: dict, group: str) -> dict:
    """A library part's own width and depth where it meets the bone, in CAD units.

    Measured like the patient: the femoral component 1 mm below its distal box face and
    over its full height, the tray 1 mm above its seating face.
    """
    from .core.implant_spec import section_mask

    key = ("dims", spec["path"], bool(spec.get("mirror")))
    if key not in _STL_CACHE:
        mesh = _cached_component(spec)
        offset = -1.0 if group == "femoral" else 1.0
        mask, xs, ys = section_mask(mesh, np.eye(4), offset_mm=offset)
        gx, gy = np.meshgrid(xs, ys)
        if group == "femoral":
            x = mesh.vertices[:, 0]
            dims = {"ml": float(np.ptp(gy[mask])), "ap_distal": float(np.ptp(gx[mask])),
                    "ap_overall": float(np.ptp(x)),
                    "height": float(mesh.vertices[:, 2].max()),
                    # Box centres in CAD X (AP, over the whole part) and Y (ML, at
                    # the distal footprint).
                    "centre": [float(x.min() + x.max()) / 2.0,
                               float(gy[mask].min() + gy[mask].max()) / 2.0]}
        else:
            dims = {"ml": float(np.ptp(gx[mask])), "ap": float(np.ptp(gy[mask])),
                    "centre": [float(gx[mask].min() + gx[mask].max()) / 2.0,
                               float(gy[mask].min() + gy[mask].max()) / 2.0]}
        _STL_CACHE[key] = dims
    return _STL_CACHE[key]


def box_design(library):
    """The femoral component's box design, from ``femoral_box.json`` in the implant
    library, or ``None``. It is design data, kept with the implant rather than in the
    planner."""
    from .core.femoral_box import BoxDesign

    if library is None:
        return None
    path = Path(library) / "femoral_box.json"
    if not path.exists():
        return None
    return BoxDesign.from_dict(json.loads(path.read_text(encoding="utf-8")))


def femoral_cut_box_case(measurement: CaseMeasurement, plan: SurgicalPlan,
                         library) -> dict | None:
    """The femoral cut box on this plan's femoral component, or why it was not placed.

    ``None`` with no implant library: without the implant there is no box to place.
    """
    from .core.femoral_box import femoral_cut_box

    if library is None:
        return None
    # The design file is written by hand, so a typo in it must not stop the plan: it
    # is reported like any other reason the box could not be placed.
    try:
        design = box_design(library)
    except (ValueError, KeyError, TypeError) as error:
        return {"available": False,
                "reason": f"The implant library's femoral_box.json cannot be used: "
                          f"{type(error).__name__}: {error}"}
    if design is None:
        return {"available": False,
                "reason": "The implant library has no femoral_box.json, the femoral "
                          "component's box design."}
    medial_sign = 1.0 if str(measurement.side).lower().startswith("r") else -1.0
    try:
        box = femoral_cut_box(measurement.femur,
                              plan.components["femoral_component"],
                              medial_sign=medial_sign, design=design,
                              samples=_femur_samples(measurement.femur))
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    return {"available": True, **box.to_dict()}


def _femur_samples(femur: Mesh) -> np.ndarray:
    """The femur's surface samples, kept for the femur last asked about.

    Sampling is half the cost of placing the box and does not depend on the plan, so it
    is done once per femur rather than on every change of a control. The mesh itself is
    held, so its identity cannot be reused by another while it is cached.
    """
    from .core.femoral_box import femur_samples

    if _SAMPLES.get("femur") is not femur:
        _SAMPLES.update(femur=femur, samples=femur_samples(femur))
    return _SAMPLES["samples"]


_SAMPLES: dict = {}


def phantom_femur_case(measurement: CaseMeasurement, *, sex: str | None = None) -> dict:
    """The schematic femur drawn from the entered length (see ``core.phantom_femur``).

    Needs ``femur_length_mm`` in ``measurement.entered``. The width the length is checked
    against is the planner's own femoral ML.
    """
    from .core.phantom_femur import build_phantom_femur

    return build_phantom_femur(
        measurement.femoral_frame, measurement.entered["femur_length_mm"],
        width_mm=measurement.femoral_measure.ml_mm, sex=sex).to_dict()


def resection_table_case(measurement: CaseMeasurement, plan: SurgicalPlan,
                         library=None, box: dict | None = None) -> dict:
    """The resection table in the standard layout, measured on this patient's bone.

    How each value is measured is in :mod:`tka_planner.core.resection_table`. ``box`` is
    :func:`femoral_cut_box_case`'s result, passed in by a caller that already has it;
    it is placed here otherwise. Without a placed box only the distal and tibial rows
    are measured.
    """
    from .core.resection_table import BoxProfile, ResectionTable

    m = measurement
    if box is None:
        box = femoral_cut_box_case(m, plan, library)
    profile = None
    if box is not None and box.get("available"):
        design = box_design(library)
        try:
            profile = BoxProfile.from_box(box, design.chamfer_deg) if design else None
        except KeyError:
            profile = None

    femoral_cut = plan.resections["femoral_distal"]
    tibial_cut = plan.resections["tibial_proximal"]
    frame = m.femoral_frame
    try:
        rows = _resection_rows(m, plan, profile, femoral_cut, tibial_cut, frame)
    except ValueError as error:
        return ResectionTable(rows=(), notes=(f"Not measured: {error}",)).to_dict()
    notes = []
    if profile is None:
        notes.append(box["reason"] if box and not box.get("available") else
                     "No implant library with a box design, so only the distal and "
                     "tibial cuts are planned.")
    return ResectionTable(rows=tuple(rows), notes=tuple(notes)).to_dict()


def _resection_rows(m, plan, profile, femoral_cut, tibial_cut, frame) -> list:
    from .core.resection_table import (
        coronal_angle_deg,
        femoral_resection_rows,
        tibial_resection_rows,
    )

    rows = femoral_resection_rows(
        m.femur, plan.components["femoral_component"],
        medial_sign=m.side.medial_sign, samples=_femur_samples(m.femur),
        profile=profile,
        distal_angles={
            "varus_from_mechanical_axis": round(coronal_angle_deg(
                femoral_cut.normal, frame.z_proximal, frame.medial), 2),
            "valgus_from_anatomical_axis": round(
                float(plan.distal_femoral_valgus_cut_deg), 2),
            "flexion": round(float(plan.diagnostics.get("femoral_flexion_deg", 0.0)), 2),
        },
    )
    tibial = m.tibial_frame
    rows += tibial_resection_rows(
        m.tibia, tibial_cut.point, tibial_cut.normal,
        medial=tibial.medial, anterior=tibial.x_anterior, axis=tibial.z_proximal,
        plateau_points=dict(zip(("medial", "lateral"), m.landmarks.require(
            "tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"))),
        slope_deg=plan.tibial_slope_deg,
    )
    return rows


def implant_spec_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library=None,
    *,
    with_cut_box: bool = True,
) -> dict:
    """The patient-specific implant, measured on this plan's cuts: what the CAD model
    is given to build.

    Dimensions are in each component's own CAD frame, with the origin where the plan
    seats it. The femoral depth is taken over the height of the component's anterior
    flange: the library's, at this size, when a library is loaded, and 40 mm otherwise.
    """
    from .core.implant_spec import (
        measure_femoral_depth,
        measure_femoral_section,
        measure_tibial_section,
    )
    from .scene.build import resolve_component_meshes

    m = measurement
    # Local +Y is patient-left on the femoral component and local +X on the tray. The
    # medial side is patient-right on a left knee.
    medial_sign = 1.0 if str(m.side).lower().startswith("r") else -1.0
    femoral = measure_femoral_section(
        m.femur, plan.components["femoral_component"], medial_sign=medial_sign)
    tibial = measure_tibial_section(
        m.tibia, plan.components["tibial_component"], medial_sign=medial_sign)

    height = 40.0
    if library is not None:
        specs = resolve_component_meshes(library, chart=m.chart, sizing=sizing,
                                         side=str(m.side))
        if "femoral_component" in specs:
            spec = specs["femoral_component"]
            height = library_dimensions(spec, "femoral")["height"] * spec["scale"]
    depth = measure_femoral_depth(m.femur, plan.components["femoral_component"],
                                  height_mm=height)

    femoral_record = femoral.to_dict()
    # The AP box centre is over the flange height, the ML centre at the distal cut.
    femoral_record["diagnostics"]["section_centre_mm"][0] = round(
        (depth["anterior_mm"] + depth["posterior_mm"]) / 2.0, 3)
    femoral_record["dimensions_mm"].update({
        "ap_overall": round(depth["ap_overall"], 2),
        "flange_height": round(depth["height_mm"], 2),
        "distal_thickness": round(float(sizing.femoral_thickness_mm), 2),
    })
    # The viewer's scaling reads only the sections, so it skips placing the box.
    box = femoral_cut_box_case(m, plan, library) if with_cut_box else None
    femoral_record["cut_box"] = box
    placed = box is not None and box.get("available")
    femoral_cut = plan.resections["femoral_distal"]
    tibial_cut = plan.resections["tibial_proximal"]
    table = {
        "femoral_distal_medial": round(femoral_cut.medial_depth_mm, 2),
        "femoral_distal_lateral": round(femoral_cut.lateral_depth_mm, 2),
        "femoral_posterior_medial": (box["resections_mm"]["posterior_medial"]
                                     if placed else None),
        "femoral_posterior_lateral": (box["resections_mm"]["posterior_lateral"]
                                      if placed else None),
        "femoral_anterior": box["resections_mm"]["anterior"] if placed else None,
        "tibial_medial": round(tibial_cut.medial_depth_mm, 2),
        "tibial_lateral": round(tibial_cut.lateral_depth_mm, 2),
    }
    tibial_record = tibial.to_dict()
    tibial_record["dimensions_mm"]["tray_thickness"] = plan.diagnostics.get(
        "tray_thickness_mm")
    diagnostics = plan.diagnostics
    return {
        "schema": "tka-planner/implant-spec",
        "schema_version": 1,
        "case_id": m.case_id,
        "side": str(m.side),
        "units": "mm",
        "femoral_component": femoral_record,
        "tibial_component": tibial_record,
        "insert": {
            "thickness_mm": diagnostics.get("insert_thickness_mm"),
            "solved": diagnostics.get("insert_solved"),
            "definition": "polyethylene under the condyle, closing the tighter "
                          "compartment in extension with no gap",
        },
        "resections": {
            "femoral_distal_mm": diagnostics.get("femoral_resection_from_datum_mm"),
            "femoral_datum": diagnostics.get("femoral_resection_datum"),
            "tibial_mm": diagnostics.get("tibial_resection_from_datum_mm"),
            "tibial_datum": diagnostics.get("tibial_resection_datum"),
            "table_mm": table,
        },
        "catalogue_equivalent": {
            "size_parameter": round(float(sizing.size_parameter), 4),
            "nearest_size": sizing.nearest_discrete_size,
        },
    }


def component_scales(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
    mode: str = "patient_specific",
    spec: dict | None = None,
) -> dict | None:
    """How each bone's set of library parts is shaped and placed for this implant.

    ``patient_specific`` stretches the library parts to the patient's own width and
    depth, measured on the cuts, keeping the size's scale along the component's axis,
    and slides each footprint in its own plane onto the centre of the section. Returns
    ``{"scales": {group: [sx, sy, sz]}, "shifts": {group: [dx, dy, 0]}}``, the shifts in
    the component's CAD axes in millimetres. ``catalogue`` returns ``None``: the parts
    keep the single scale of the size and the plan's placement. Until the CAD model
    builds the patient's implant, this is what the viewer shows and what the fit is
    checked against.
    """
    from .scene.build import resolve_component_meshes

    if mode not in IMPLANT_MODES:
        raise ValueError(f"implant mode must be one of {', '.join(IMPLANT_MODES)}.")
    if mode == "catalogue" or library is None:
        return None
    specs = resolve_component_meshes(library, chart=measurement.chart, sizing=sizing,
                                     side=str(measurement.side))
    if "femoral_component" not in specs or "tibial_component" not in specs:
        return None
    spec = spec or implant_spec_case(measurement, plan, sizing, library,
                                     with_cut_box=False)
    femoral = spec["femoral_component"]["dimensions_mm"]
    tibial = spec["tibial_component"]["dimensions_mm"]
    lib_f = library_dimensions(specs["femoral_component"], "femoral")
    lib_t = library_dimensions(specs["tibial_component"], "tibial")
    scales = {
        "femoral": np.array([femoral["ap_overall"] / lib_f["ap_overall"],
                             femoral["ml"] / lib_f["ml"],
                             specs["femoral_component"]["scale"]]),
        "tibial": np.array([tibial["ml"] / lib_t["ml"],
                            tibial["ap"] / lib_t["ap"],
                            specs["tibial_component"]["scale"]]),
    }
    # The library part's footprint is not centred on its CAD origin, so once it is the
    # patient's size the femoral part is slid in its own plane until its box centre is
    # the section's.
    record = spec["femoral_component"]
    centre = np.asarray(record["diagnostics"]["section_centre_mm"], dtype=float)
    shifts = {"femoral": np.array([
        *(centre - scales["femoral"][:2] * np.asarray(lib_f["centre"])), 0.0])}

    # The centring above follows the pose, so on its own it would undo the surgeon's
    # slide of the femoral component across its cut. The slide is put back on top of it,
    # in the component's own axes: +X anterior, +Y patient-left. Lateral is patient-left
    # on a left knee.
    lateral_sign = -1.0 if str(measurement.side).lower().startswith("r") else 1.0
    a = plan.adjustments
    shifts["femoral"] += np.array([a.femoral_shift_ap_mm,
                                   lateral_sign * a.femoral_shift_ml_mm, 0.0])

    # The tray is in register with the femoral component: the plan puts its origin
    # directly under the femoral one. So it takes the femoral part's slide rather than
    # being centred on its own cut. Centring each part on its own cut would pull the two
    # apart again. The femoral slide is carried over in world terms and re-expressed in
    # the tray's axes (+X patient-left, +Y posterior). The surgeon's own tray slide is
    # already in the tray's pose.
    femoral_pose = np.asarray(plan.components["femoral_component"], dtype=float)
    tray_pose = np.asarray(plan.components["tibial_component"], dtype=float)
    world = femoral_pose[:3, :2] @ shifts["femoral"][:2]
    shifts["tibial"] = np.array([*(tray_pose[:3, :2].T @ world), 0.0])
    return {"scales": scales, "shifts": shifts}


def fit_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
    mode: str = "patient_specific",
) -> dict | None:
    """How the planned components fit the bone, on each cut.

    The components are taken from the implant library at the plan's size and posed
    exactly as the viewer poses them, so the fit is of what the surgeon sees. Returns
    ``None`` with no library, and names any component the library lacks rather than
    guessing its shape.

    The posterior condylar offset is measured here too, on the femoral component as it
    is placed: preoperative, planned and the change, per compartment.
    """
    result, _ = fit_case_with_grids(measurement, plan, sizing, library, mode)
    return result


def fit_case_with_grids(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
    mode: str = "patient_specific",
    *,
    transforms=None,
) -> tuple[dict | None, dict]:
    """:func:`fit_case`, and the signed-distance grid of each component's fit.

    ``transforms`` is :func:`component_scales`' result for this plan and mode, for a
    caller that already has it (the application keeps it per plan); measuring the cuts
    again for it costs over half a second on a real knee.

    The grids are :class:`~tka_planner.core.fit.FitField` objects by component name,
    the fields the fit's maxima were read from, for drawing the fit as a colour map.
    """
    from .core.fit import section_fit
    from .geom import mesh as gm
    from .scene.build import place, resolve_component_meshes

    grids: dict = {}
    if library is None:
        return None, grids
    m = measurement
    specs = resolve_component_meshes(
        library, chart=m.chart, sizing=sizing, side=str(m.side)
    )
    if transforms is None:
        transforms = component_scales(m, plan, sizing, library, mode)
    transforms = transforms or {}
    scales = transforms.get("scales", {})
    shifts = transforms.get("shifts", {})
    result = {"implant_mode": mode if scales else "catalogue",
              "library_sizes": {}, "missing": []}
    for name, cut, bone, frame, side_of_cut in (
        ("tibial_component", "tibial_proximal", m.tibia, m.tibial_frame, 1.0),
        ("femoral_component", "femoral_distal", m.femur, m.femoral_frame, -1.0),
    ):
        spec = specs.get(name)
        if spec is None:
            result["missing"].append(name)
            continue
        group = "tibial" if name.startswith("tibial") else "femoral"
        posed = gm.transformed(
            _cached_component(spec),
            place(plan.components[name], scales.get(group, spec["scale"]),
                  shifts.get(group)),
        )
        if name == "femoral_component":
            result["posterior_condylar_offset"] = planned_condylar_offset(
                m.femur.vertices, posed.vertices, m.landmarks, m.femoral_frame)
        resection = plan.resections[cut]
        fit = section_fit(
            cut=cut, bone=bone, component=posed,
            point=resection.point, normal=resection.normal,
            component_side=side_of_cut,
            lateral=frame.lateral, anterior=frame.x_anterior, keep_grid=True,
        )
        result[name] = fit.to_dict()
        grids[name] = fit.grid
        result["library_sizes"][name] = {
            "source_size": spec["source_size"], "scale": round(spec["scale"], 6),
        }
        if group in scales:
            result["library_sizes"][name]["scale_xyz"] = [
                round(float(v), 6) for v in scales[group]]
            result["library_sizes"][name]["shift_mm"] = [
                round(float(v), 3) for v in shifts[group]]
    if "femoral_component" not in specs:
        result["posterior_condylar_offset"] = {
            "available": False,
            "reason": "the implant library has no femoral component for this side"}
    result["insert"] = _insert_check(plan, specs, m, transforms)
    return result, grids


def _insert_check(plan: SurgicalPlan, specs: dict, m: CaseMeasurement,
                  transforms: dict | None = None) -> dict | None:
    """The solved insert as the viewer shows it, and whether the parts collide.

    After the stretch the insert's dish is the solved thickness. Anywhere the femoral
    component still reaches below the insert's surface, the parts run into each other;
    that is reported with where it happens, rather than hidden by a thinner insert.
    """
    from .core.insert import solve_insert
    from .geom import mesh as gm
    from .scene.build import insert_stretch_matrix, place

    display = insert_display(plan, specs)
    if display is None or "femoral_component" not in specs:
        return None
    scales = (transforms or {}).get("scales", {})
    shifts = (transforms or {}).get("shifts", {})
    femoral_spec, insert_spec = specs["femoral_component"], specs["tibial_insert"]
    femoral = gm.transformed(
        _cached_component(femoral_spec),
        place(plan.components["femoral_component"],
              scales.get("femoral", femoral_spec["scale"]), shifts.get("femoral")),
    )
    insert_pose = (place(plan.components["tibial_component"],
                         scales.get("tibial", insert_spec["scale"]), shifts.get("tibial"))
                   @ insert_stretch_matrix(display))
    insert = gm.transformed(_cached_component(insert_spec), insert_pose)
    try:
        clearance = solve_insert(
            femoral, insert, normal=plan.resections["tibial_proximal"].normal,
            lateral=m.tibial_frame.lateral,
        )
    except ValueError as error:
        return {**display, "clearance": None, "note": str(error)}

    contact_local = np.linalg.inv(insert_pose) @ np.append(clearance.contact_point, 1.0)
    flags = []
    if clearance.change_mm < -0.5:
        where = "anterior" if contact_local[1] < 0 else "posterior"
        flags.append("COMPONENT_COLLISION")
        note = (f"The femoral component runs {-clearance.change_mm:.1f} mm into the "
                f"insert, towards its {where} edge. The components are "
                f"{abs(plan.diagnostics.get('component_rotation_mismatch_deg', 0.0)):.1f}"
                f" degrees and "
                f"{abs(plan.diagnostics.get('component_offset_anterior_mm', 0.0)):.1f} mm "
                f"(AP) out of register.")
    else:
        note = "No collision: the insert meets the femoral component."
    return {
        **display,
        "clearance": clearance.to_dict(),
        "contact_in_insert_mm": [round(float(v), 2) for v in contact_local[:3]],
        "flags": flags,
        "note": note,
    }


_SET_FOR_BONE = {"femur": "femoral", "tibia": "tibial", "fibula": "tibial"}


def findings_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    *,
    box: dict | None = None,
    fit: dict | None = None,
) -> list:
    """Every finding about this plan, in one list, each with where it is if it has a
    place.

    The scan's quality control comes first, then what the plan, the sizing, the femoral
    box and the fit found. Each entry is ``{code, severity, message, source, location}``;
    ``location`` is ``None`` or ``{"point_mm", "set", "label"}``, a point in scan
    millimetres and the bone set it moves with, so a viewer can turn to it. The viewer
    shows this list as it comes; nothing in it is worked out there.
    """
    m = measurement
    by_id = {landmark.id: landmark for landmark in m.landmarks}
    findings = []

    def add(code, severity, message, source, point=None, bone=None, label=None):
        location = None
        if point is not None:
            location = {"point_mm": [round(float(v), 2) for v in point],
                        "set": _SET_FOR_BONE.get(bone, bone or "femoral"),
                        "label": label or code}
        findings.append({"code": code, "severity": severity, "message": message,
                         "source": source, "location": location})

    for finding in m.qc.findings:
        context = finding.context
        point, bone, label = context.get("location_mm"), context.get("bone"), None
        offenders = [by_id[i] for i in context.get("offenders", []) if i in by_id]
        if point is None and offenders:
            point = offenders[0].position_mm
            bone = offenders[0].definition.bone
            label = offenders[0].id
        message = finding.message
        if context.get("bone") and finding.code.startswith(("MESH_", "LATERALITY_")):
            message = f"{context['bone'].capitalize()}: {message}"
        add(finding.code, finding.severity.value, message, "scan", point, bone, label)

    estimated = [lm for lm in m.landmarks if lm.status.value == "estimated"]
    if estimated:
        add("LANDMARKS_ESTIMATED", "info",
            f"{len(estimated)} landmarks are machine estimates awaiting review, not "
            f"picks.", "landmarks")
    for warning in plan.warnings:
        add("PLAN_WARNING", "warning", warning, "plan")
    for flag in sizing.flags:
        add(f"SIZING_{str(flag).upper()}", "warning", f"Sizing: {flag}.", "sizing")

    if box is not None and not box.get("available"):
        add("FEMORAL_BOX_NOT_PLACED", "warning",
            f"The femoral cut box was not placed: {box.get('reason')}", "plan")
    elif box is not None:
        for warning in box.get("warnings", []):
            add("FEMORAL_BOX_WARNING", "warning", warning, "plan")
        if box.get("anterior_notch_mm", 0.0) > 0.5:
            add("ANTERIOR_NOTCH", "warning",
                f"The anterior femoral cut notches the cortex by "
                f"{box['anterior_notch_mm']:.1f} mm.", "plan")

    for name, set_id, words in (("tibial_component", "tibial", "tibial tray"),
                                ("femoral_component", "femoral", "femoral component")):
        record = (fit or {}).get(name)
        if not record:
            continue
        for flag in record.get("flags", []):
            zone = flag.removeprefix("OVERHANG_").lower()
            over = record["zones"].get(zone, {}).get("max_overhang_mm")
            add(flag, "warning",
                f"The {words} overhangs the bone by up to {over:.1f} mm in the {zone} "
                f"zone (flagged above {record['diagnostics']['overhang_flag_mm']:g} mm).",
                "fit", record.get("extreme_points_mm", {}).get(f"overhang_{zone}"),
                set_id, f"{words} {zone} overhang")
    insert = (fit or {}).get("insert") or {}
    if "COMPONENT_COLLISION" in insert.get("flags", []):
        add("COMPONENT_COLLISION", "warning", insert["note"], "fit",
            (insert.get("clearance") or {}).get("contact_point_mm"), "tibia",
            "insert contact")
    for name in (fit or {}).get("missing", []):
        add("COMPONENT_NOT_IN_LIBRARY", "warning",
            f"The implant library has no {name.replace('_', ' ')}, so its fit is not "
            f"checked.", "fit")
    return findings


FLEXION_PAIRS = (
    "insert_vs_femoral_bone",
    "femoral_component_vs_insert",
    "femoral_component_vs_tibial_bone",
    "tibial_bone_vs_femoral_bone",
)


@dataclass(frozen=True)
class FlexionSettings:
    """How the maximum-flexion search is run, recorded with every result.

    ``articular_tolerance_mm`` is part of the rule that tells the intended bearing
    contact from impingement between the femoral component and the insert (see
    :class:`FlexionModel`). ``None``, the default, lets overlap in the articular region
    never stop flexion; a number makes it stop when deeper than the plan's own seating
    by more than that.

    **Where a search stops** (author ruling #26). ``stop_deg`` is the end of the motion,
    180 deg, where the tibia lies folded onto the femur
    (:data:`~tka_planner.core.kinematics.FOLD_DEG`); it is a safety bound, not a
    parameter of the result, since the scanned bones meet before it. A search that
    reaches it untouched is reported as "no contact in the scanned models", with the
    reason measured from the models (:func:`flexion_case`).

    **What geometry is searched.** The bones are the patient's own scans, not a fixed
    box around the joint: by default (``None``) each bone is kept over its whole
    scanned extent, in the femoral and tibial frames' axes about the hinge, except that
    a bone longer than the other bone's reach from the hinge axis (plus
    ``reach_margin_mm``, for the trajectories' own translation and axial rotation) is
    cut there, because a rigid flexion about the hinge keeps every point at its
    distance from the axis, so nothing farther out can ever be touched. A full-length
    scan is therefore kept whole only as far as the other bone could reach. A box may
    still be given as ``((low), (high))`` in mm.
    """

    start_deg: float = 0.0
    stop_deg: float = 180.0
    step_deg: float = 5.0
    resolution_deg: float = 0.5
    min_volume_mm3: float = 1.0
    articular_tolerance_mm: float | None = None
    rim_margin_mm: float = 2.0
    simplify_mm: float = 0.2
    femur_box_mm: tuple | None = None
    tibia_box_mm: tuple | None = None
    reach_margin_mm: float = 20.0

    def to_dict(self) -> dict:
        return {k: (list(map(list, v)) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}


def _frame_at(frame: AnatomicalFrame, origin) -> np.ndarray:
    matrix = frame.matrix.copy()
    matrix[:3, 3] = np.asarray(origin, dtype=float)
    return matrix


def _radial_reach(points: np.ndarray, origin, direction) -> float:
    """The largest distance of any point from the hinge axis (mm)."""
    axis = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    d = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return float(np.linalg.norm(d - np.outer(d @ axis, axis), axis=1).max())


def scan_extent(measurement: CaseMeasurement, origin, direction,
                settings: FlexionSettings | None = None) -> dict:
    """How far each scanned bone extends from the joint, and the boxes it is kept in.

    Coordinates are in each bone's frame axes (anterior, patient-left, proximal), mm
    from the hinge ``origin``. The joint line is each bone's own articular end: the
    most distal point of the femur, the most proximal point of the tibia. The radial
    reach is the farthest scanned point from the hinge axis: under a rigid flexion about
    the hinge every point keeps that distance, so one bone can only ever meet the other
    within the smaller of the two reaches. These are measurements of the scans, the
    reason a search that meets nothing gives (:func:`flexion_case`).
    """
    settings = settings or FlexionSettings()
    m = measurement
    origin = np.asarray(origin, dtype=float)
    record, boxes = {}, {}
    reach = {"femur": _radial_reach(m.femur.vertices, origin, direction),
             "tibia": _radial_reach(m.tibia.vertices, origin, direction)}
    for name, mesh, frame, given in (
            ("femur", m.femur, m.femoral_frame, settings.femur_box_mm),
            ("tibia", m.tibia, m.tibial_frame, settings.tibia_box_mm)):
        local = (mesh.vertices - origin) @ frame.matrix[:3, :3]
        low, high = local.min(axis=0), local.max(axis=0)
        other = reach["tibia" if name == "femur" else "femur"]
        if given is not None:
            box = (tuple(map(float, given[0])), tuple(map(float, given[1])))
        else:
            # The scan's own extent, one millimetre clear of it, cut at the other bone's
            # reach along the shaft.
            box_low, box_high = low - 1.0, high + 1.0
            bound = other + settings.reach_margin_mm
            if name == "femur":
                box_high[2] = min(box_high[2], bound)
            else:
                box_low[2] = max(box_low[2], -bound)
            box = (tuple(round(float(v), 2) for v in box_low),
                   tuple(round(float(v), 2) for v in box_high))
        boxes[name] = box
        shaft = high[2] if name == "femur" else -low[2]
        record[name] = {
            "scanned_from_hinge_mm": round(float(shaft), 1),
            "scanned_from_joint_line_mm": round(float(high[2] - low[2]), 1),
            "radial_reach_mm": round(reach[name], 1),
            "box_mm": [list(box[0]), list(box[1])],
            "box_from_scan": given is None,
            "kept_from_hinge_mm": round(float(box[1][2] if name == "femur"
                                              else -box[0][2]), 1),
        }
    record["meeting_reach_mm"] = round(min(reach.values()), 1)
    return {"extent": record, "boxes": boxes}


def _cropped_bone(mesh: Mesh, frame: np.ndarray, box) -> "object":
    """A bone as a solid, cropped to the reach of a flexing knee. Kept for the bone last
    asked about: it is the slow step (a million triangles into manifold3d) and depends
    on neither the plan nor the trajectory."""
    from .geom.collision import Solid

    key = (id(mesh), np.asarray(frame).tobytes(), repr(box))
    cached = _BONE_SOLIDS.get(key)
    if cached is None or cached[0] is not mesh:
        # Two entries -- one knee's femur and tibia -- since each holds a bone's worth
        # of geometry and the mesh it came from.
        while len(_BONE_SOLIDS) >= 2:
            _BONE_SOLIDS.pop(next(iter(_BONE_SOLIDS)))
        cached = (mesh, Solid.from_mesh(mesh).cropped(frame, *box))
        _BONE_SOLIDS[key] = cached
    return cached[1]


_BONE_SOLIDS: dict = {}


@dataclass
class FlexionModel:
    """The planned construct as solids, ready to be flexed.

    The femoral side (resected femur, femoral component) stays where the plan put it;
    the tibial side (resected tibia, stretched insert) moves by the trajectory's pose.
    The hinge is the one Reduce mode uses (:func:`~tka_planner.scene.motion.flexion_axis`).

    **Articulation or impingement.** The femoral component is meant to bear on the
    insert, and a prescribed trajectory is not solved from contact, so the two will
    overlap, or part, where they articulate: a hinge that does not run through the
    centres of this component's condyles drives them into the dish or lifts them off
    it. That is the motion's error, not impingement. So an overlap of the pair is
    impingement only when any of it lies within ``rim_margin_mm`` of the insert's edge
    -- its anterior or posterior lip or its sides, in the insert's own axes -- which is
    contact outside the articular surface. Overlap inside the articular surface is the
    articulation; it does not stop flexion, and the deepest of it over the arc is
    reported with every search as a check on how well the motion follows the parts.
    With ``articular_tolerance_mm`` set, articular overlap deeper than the plan's own
    seating at extension by more than that also counts as impingement. Every other pair -- insert against the resected femur, the
    femoral component against the resected tibia, bone against bone -- stops flexion at
    any overlap above ``min_volume_mm3``.
    """

    pivot: np.ndarray
    femoral_frame: AnatomicalFrame
    tibial_frame: AnatomicalFrame
    femur: object
    tibia: object
    femoral_component: object
    insert: object
    insert_pose: np.ndarray
    insert_bounds: tuple
    settings: FlexionSettings
    diagnostics: dict
    baseline_depth_mm: float = 0.0
    femur_box_mm: tuple | None = None   # the boxes the bones were kept in (scan extent)
    tibia_box_mm: tuple | None = None
    # This knee's scenarios, S1 scaled to it (:func:`scenarios_case`); ``None``: the
    # defined ones.
    scenarios: dict | None = None

    def trajectory(self, name_or_trajectory):
        """A scenario name resolved for this knee, or a trajectory passed through."""
        from .core.kinematics import scenario

        return scenario(name_or_trajectory, self.scenarios)

    def pose(self, trajectory, flexion_deg: float) -> np.ndarray:
        from .scene.motion import trajectory_pose

        return trajectory_pose(self.pivot, flexion_deg, self.trajectory(trajectory),
                               anterior=self.tibial_frame.x_anterior,
                               medial=self.tibial_frame.medial)

    def _articular(self, shared, pose: np.ndarray) -> tuple:
        """Where on the insert an overlap lies, and how deep it reaches along the
        insert's axis (mm, :meth:`~tka_planner.geom.collision.Overlap.thickness_mm`)."""
        points = shared.vertices
        insert_pose = pose @ self.insert_pose
        cad = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(insert_pose).T)[:, :3]
        sx, sy = (np.linalg.norm(insert_pose[:3, i]) for i in (0, 1))
        (x0, y0), (x1, y1) = self.insert_bounds
        margin = self.settings.rim_margin_mm
        # CAD axes of tibial parts: +X patient-left, +Y posterior, +Z proximal.
        region = "articular"
        if np.any((y1 - cad[:, 1]) * sy < margin):
            region = "posterior_rim"
        elif np.any((cad[:, 1] - y0) * sy < margin):
            region = "anterior_rim"
        elif np.any(np.minimum(cad[:, 0] - x0, x1 - cad[:, 0]) * sx < margin):
            region = "side_rim"
        return region, shared.thickness_mm(insert_pose[:3, 2])

    def contacts(self, trajectory, flexion_deg: float) -> list:
        """Every overlap above the volume threshold at one angle, as
        :class:`~tka_planner.core.kinematics.Contact`."""
        from .core.kinematics import Contact
        from .geom.collision import overlap

        s = self.settings
        pose = self.pose(trajectory, flexion_deg)
        insert = self.insert.transformed(pose)
        tibia = self.tibia.transformed(pose)
        found = []
        for pair, a, b in (
            ("insert_vs_femoral_bone", insert, self.femur),
            ("femoral_component_vs_insert", self.femoral_component, insert),
            ("femoral_component_vs_tibial_bone", self.femoral_component, tibia),
            ("tibial_bone_vs_femoral_bone", tibia, self.femur),
        ):
            shared = overlap(a, b)
            if shared is None or shared.volume_mm3 < s.min_volume_mm3:
                continue
            local = self.femoral_frame.to_local(shared.vertices)
            location = local.mean(axis=0)
            medial_sign = float(np.dot(self.femoral_frame.medial,
                                       self.femoral_frame.y_patient_left))
            diagnostics = {
                "crop_limited": self._at_crop(shared.vertices, pose),
                "compartment": "medial" if location[1] * medial_sign > 0 else "lateral",
                **self._lever(shared.vertices),
            }
            if pair == "femoral_component_vs_insert":
                region, depth = self._articular(shared, pose)
                tolerance = s.articular_tolerance_mm
                too_deep = (tolerance is not None
                            and depth > self.baseline_depth_mm + tolerance)
                kind = ("impingement" if region != "articular" or too_deep
                        else "articulation")
                diagnostics["baseline_depth_mm"] = round(self.baseline_depth_mm, 3)
            else:
                region = "anterior" if location[0] > 0.0 else "posterior"
                depth = float(np.ptp(local, axis=0).min())
                kind = "impingement"
            found.append(Contact(
                pair=pair, flexion_deg=float(flexion_deg),
                volume_mm3=shared.volume_mm3, depth_mm=depth, location=location,
                kind=kind, region=region, diagnostics=diagnostics,
            ))
        return found

    def _lever(self, points: np.ndarray) -> dict:
        """Where an overlap lies relative to the hinge, for reading a limit's
        sensitivity: its centre in the femoral frame's axes from the hinge (anterior,
        patient-left, proximal), its distance from the hinge axis, and its thinnest
        direction (the smallest principal axis of the overlap's surface, an estimate of
        the contact normal), in the femoral frame's axes."""
        origin, axis = self.pivot[:3, 3], self.pivot[:3, 0] / np.linalg.norm(
            self.pivot[:3, 0])
        centre = points.mean(axis=0)
        d = centre - origin
        radius = float(np.linalg.norm(d - (d @ axis) * axis))
        basis = self.femoral_frame.matrix[:3, :3]
        out = {"from_hinge_mm": [round(float(v), 2) for v in d @ basis],
               "hinge_radius_mm": round(radius, 2)}
        if len(points) >= 4:
            centred = points - centre
            _, vectors = np.linalg.eigh(centred.T @ centred)
            normal = vectors[:, 0] @ basis
            if normal[0] < 0.0:      # sign-free: report it pointing anteriorly
                normal = -normal
            out["thin_direction"] = [round(float(v), 4) for v in normal]
        return out

    def _at_crop(self, points: np.ndarray, pose: np.ndarray) -> bool:
        """Whether an overlap touches a face of the box a bone was kept in (by default
        the scan's own extent): if so, bone beyond it could have been hit first, and the
        result says so."""
        s = self.settings
        femoral = (points - self.pivot[:3, 3]) @ self.femoral_frame.matrix[:3, :3]
        tibial_points = (np.c_[points, np.ones(len(points))]
                         @ np.linalg.inv(pose).T)[:, :3]
        tibial = (tibial_points - self.pivot[:3, 3]) @ self.tibial_frame.matrix[:3, :3]
        for local, given, kept, end in (
                (femoral, s.femur_box_mm, self.femur_box_mm, 1),
                (tibial, s.tibia_box_mm, self.tibia_box_mm, 0)):
            box = kept or given
            if box is None:
                continue
            low, high = (np.asarray(v, dtype=float) for v in box)
            if given is None:
                # A box taken from the scan touches the bone's own surface on every
                # side but one: only the shaft end (the femur's proximal face, the
                # tibia's distal one) is where the scan, or the reach, cut the bone.
                face = high[2] if end else low[2]
                near = np.abs(local[:, 2] - face) < 1.0
            else:
                near = (np.abs(local - low) < 1.0) | (np.abs(local - high) < 1.0)
            if near.any():
                return True
        return False


def flexion_model(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
    mode: str = "patient_specific",
    *,
    settings: FlexionSettings | None = None,
    femoral_offset_mm: float = 0.0,
) -> FlexionModel | None:
    """Build the planned construct as solids for the flexion search.

    The parts are the library parts posed and shaped exactly as the viewer and
    :func:`fit_case` pose them; the bones are cut by the plan's cutting blocks, as
    Commit cuts them. ``None`` with no library, or when it lacks a part. A bone surface
    that is not closed raises ``ValueError``, as Commit does: the search needs solids,
    and holes are reported rather than filled.

    ``femoral_offset_mm`` moves the femoral component that far anteriorly along its own
    AP axis, relative to the femur, with the joint re-seated on the component: the
    tibial set and the hinge keep their place relative to the component, so only the
    femur moves (by the opposite amount) before it is cut. That is what a change of
    posterior condylar offset does to a knee that articulates on its implant. Zero is
    the plan as it stands.
    """
    import time

    from .geom import mesh as gm
    from .geom.collision import Solid, overlap
    from .scene.build import insert_stretch_matrix, place, resolve_component_meshes
    from .scene.motion import flexion_axis, pivot_frame

    if library is None:
        return None
    settings = settings or FlexionSettings()
    m = measurement
    specs = resolve_component_meshes(library, chart=m.chart, sizing=sizing,
                                     side=str(m.side))
    needed = ("femoral_component", "femoral_cutting_block", "tibial_component",
              "tibial_insert", "tibial_cutting_block")
    if any(name not in specs for name in needed):
        return None
    display = insert_display(plan, specs)
    if display is None:
        return None
    transforms = component_scales(m, plan, sizing, library, mode) or {}
    scales, shifts = transforms.get("scales", {}), transforms.get("shifts", {})

    def posed(name):
        spec = specs[name]
        group = spec.get("group", "femoral")
        matrix = place(plan.components[f"{group}_component"],
                       scales.get(group, spec["scale"]), shifts.get(group))
        if name == "tibial_insert":
            matrix = matrix @ insert_stretch_matrix(display)
        return matrix, _cached_component(spec)

    t0 = time.perf_counter()
    origin, direction = flexion_axis(plan, m.femoral_frame, m.landmarks)
    pivot = pivot_frame(origin, direction)
    extent = scan_extent(m, origin, direction, settings)
    femur = _cropped_bone(m.femur, _frame_at(m.femoral_frame, origin),
                          extent["boxes"]["femur"])
    tibia = _cropped_bone(m.tibia, _frame_at(m.tibial_frame, origin),
                          extent["boxes"]["tibia"])
    t_bones = time.perf_counter() - t0

    anterior = np.asarray(plan.components["femoral_component"], dtype=float)[:3, 0]
    anterior = anterior / np.linalg.norm(anterior)
    if femoral_offset_mm:
        femur = femur.translated(-float(femoral_offset_mm) * anterior)
        # The box the femur was kept in moves with it.
        shift = (-float(femoral_offset_mm) * anterior) @ m.femoral_frame.matrix[:3, :3]
        low, high = extent["boxes"]["femur"]
        extent["boxes"]["femur"] = (tuple(np.add(low, shift)), tuple(np.add(high, shift)))

    blocks = {}
    for name in ("femoral_cutting_block", "tibial_cutting_block"):
        matrix, mesh = posed(name)
        blocks[name] = Solid.from_mesh(gm.transformed(mesh, matrix))
    femur = femur.minus(blocks["femoral_cutting_block"]).simplified(settings.simplify_mm)
    tibia = tibia.minus(blocks["tibial_cutting_block"]).simplified(settings.simplify_mm)

    component_matrix, component_mesh = posed("femoral_component")
    component = Solid.from_mesh(gm.transformed(component_mesh, component_matrix))
    insert_matrix, insert_mesh = posed("tibial_insert")
    insert = Solid.from_mesh(gm.transformed(insert_mesh, insert_matrix))
    bounds = (tuple(insert_mesh.vertices[:, :2].min(axis=0)),
              tuple(insert_mesh.vertices[:, :2].max(axis=0)))

    model = FlexionModel(
        pivot=pivot, femoral_frame=m.femoral_frame, tibial_frame=m.tibial_frame,
        femur=femur, tibia=tibia, femoral_component=component, insert=insert,
        insert_pose=insert_matrix, insert_bounds=bounds, settings=settings,
        femur_box_mm=extent["boxes"]["femur"], tibia_box_mm=extent["boxes"]["tibia"],
        scenarios=scenarios_case(m),
        diagnostics={
            "implant_mode": mode if scales else "catalogue",
            "hinge_origin_mm": [round(float(v), 3) for v in origin],
            "hinge_direction": [round(float(v), 6) for v in direction],
            "femoral_offset_mm": float(femoral_offset_mm),
            "scan_extent": extent["extent"],
            "triangles": {"femur": femur.n_triangles, "tibia": tibia.n_triangles,
                          "femoral_component": component.n_triangles,
                          "insert": insert.n_triangles},
            "bone_preparation_s": round(t_bones, 2),
        },
    )
    seated = overlap(component, insert)
    if seated is not None:
        _, model.baseline_depth_mm = model._articular(seated, np.eye(4))
    model.diagnostics["baseline_articular_depth_mm"] = round(model.baseline_depth_mm, 3)
    model.diagnostics["model_build_s"] = round(time.perf_counter() - t0, 2)
    return model


def flexion_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
    mode: str = "patient_specific",
    *,
    trajectories=("S1", "S2", "S3"),
    settings: FlexionSettings | None = None,
    femoral_offset_mm: float = 0.0,
    model: FlexionModel | None = None,
) -> dict | None:
    """The impingement-limited maximum flexion of a plan, under each trajectory, and the
    envelope across them (E10a).

    For each trajectory the knee is flexed until two parts first overlap in a way that
    blocks it (see :class:`FlexionModel` for the rule); the result names the pair, where
    it touched (femoral frame, mm from the knee centre: anterior, patient-left,
    proximal) and how much. ``None`` with no implant library.

    This is a geometric limit: no soft tissue, no thigh-calf contact, and a prescribed
    rather than contact-solved motion. It bounds what the parts allow, not what a
    patient will reach.
    """
    import time

    from .core.kinematics import envelope, search_max_flexion

    settings = settings or (model.settings if model is not None else FlexionSettings())
    if model is None:
        model = flexion_model(measurement, plan, sizing, library, mode,
                              settings=settings, femoral_offset_mm=femoral_offset_mm)
    if model is None:
        return None
    m = measurement
    lateral_ml, medial_ml = _compartment_ml(m, model.pivot[:3, 3])
    limits, records, warnings = {}, {}, []
    for name in trajectories:
        # Resolved for this knee: S1's pivot is scaled to its condyle-centre spacing.
        trajectory = model.trajectory(name)
        warnings += [w for w in trajectory.warnings if w not in warnings]
        t0 = time.perf_counter()
        limit = search_max_flexion(
            lambda angle, t=trajectory: model.contacts(t, angle),
            trajectory=trajectory.name, start_deg=settings.start_deg,
            stop_deg=settings.stop_deg, step_deg=settings.step_deg,
            resolution_deg=settings.resolution_deg,
        )
        limits[trajectory.name] = limit
        theta = limit.max_flexion_deg
        records[trajectory.name] = {
            **limit.to_dict(),
            "parameters": trajectory.to_dict(),
            "at_limit": {
                "internal_rotation_deg": round(trajectory.at(theta)[0], 3),
                "medial_ap_mm": round(trajectory.compartment_translation_mm(
                    theta, medial_ml), 3),
                "lateral_ap_mm": round(trajectory.compartment_translation_mm(
                    theta, lateral_ml), 3),
            },
            "seconds": round(time.perf_counter() - t0, 2),
        }
        if limit.censored:
            records[trajectory.name]["no_contact"] = no_contact_reason(
                model, trajectory, limit.stop_deg)
    component = np.asarray(plan.components["femoral_component"], dtype=float)
    return {
        "method": "flexion.impingement_search.v1",
        "provenance": "estimated",
        "note": ("Geometric limit of the parts under a prescribed trajectory: no soft "
                 "tissue, no thigh-calf contact, not contact-solved (WP-D level D1)."),
        "stop_rule": ("Each search runs until the scanned models make blocking contact, "
                      f"bounded only by the end of the motion ({settings.stop_deg:g} deg, "
                      "the tibia folded onto the femur). A search that reaches the bound "
                      "untouched reports stop_reason 'no_contact_in_scanned_models' and "
                      "flexion_limit_deg null, with the reason under 'no_contact'."),
        "scan_extent": model.diagnostics.get("scan_extent"),
        "warnings": warnings,
        "pairs": list(FLEXION_PAIRS),
        "trajectories": records,
        "envelope": envelope(limits),
        "compartment_ml_mm": {"medial": round(medial_ml, 2),
                              "lateral": round(lateral_ml, 2)},
        "settings": settings.to_dict(),
        "diagnostics": model.diagnostics,
        "femoral_component_anterior": [round(float(v), 6) for v in
                                       component[:3, 0] / np.linalg.norm(component[:3, 0])],
    }


def _min_gap_staged(a, b, search_mm: float):
    """:meth:`Solid.min_gap` with the search radius widened in steps.

    The kernel's cost grows with the radius it is asked to search, and most gaps are small,
    so ask within 10 mm, then 30, then the full radius. The answer is the same as asking
    at the full radius: a gap found within a smaller radius is the minimum.
    """
    for radius in (10.0, 30.0):
        if radius >= search_mm:
            break
        gap = a.min_gap(b, radius)
        if gap is not None:
            return gap
    return a.min_gap(b, search_mm)


def no_contact_reason(model: FlexionModel, trajectory, angle_deg: float,
                      search_mm: float = 100.0) -> dict:
    """Why a search met nothing by ``angle_deg``, measured from the models.

    At the bound the tibial set is posed as the trajectory puts it, and the models are
    asked where they come closest. ``gaps_mm`` is how close each pair still is
    (``None``: farther than ``search_mm``). The same is measured against the last
    ``end_slab_mm`` of each shaft as kept (the femur's proximal end, the tibia's distal
    end), where the scan -- or the other bone's reach -- cut the bone, and against the
    rest of that shaft. If the other side comes nearer the shaft's end than anywhere
    else on it (by more than 0.5 mm), the gap is closing towards the end of the scan and
    the bone that would close it lies beyond: the reason is ``bone_not_in_scan``.
    Otherwise the parts come closest inside the scans, so the bone that would meet is in
    the models and it is the prescribed motion that keeps it clear:
    ``kept_apart_by_motion``. The scan extents (:func:`scan_extent`)
    and how far the folded tibia reaches proximally of the hinge are given with it.
    """
    end_slab_mm = 5.0
    trajectory = model.trajectory(trajectory)
    pose = model.pose(trajectory, angle_deg)
    tibia, insert = model.tibia.transformed(pose), model.insert.transformed(pose)
    gaps = {}
    for pair, a, b in (
        ("insert_vs_femoral_bone", insert, model.femur),
        ("femoral_component_vs_insert", model.femoral_component, insert),
        ("femoral_component_vs_tibial_bone", model.femoral_component, tibia),
        ("tibial_bone_vs_femoral_bone", tibia, model.femur),
    ):
        gap = _min_gap_staged(a, b, search_mm)
        gaps[pair] = None if gap is None else round(gap, 2)
    origin = model.pivot[:3, 3]
    basis = model.femoral_frame.matrix[:3, :3]
    folded = np.vstack([tibia.to_mesh().vertices, insert.to_mesh().vertices])
    reach = float(((folded - origin) @ basis)[:, 2].max())

    # Each shaft's last few millimetres as kept, against the rest of it: if the other
    # side comes nearer the end than anywhere else on that bone, the gap is closing
    # towards the end of the scan.
    def nearest(solids, others):
        found = [_min_gap_staged(a, b, search_mm) for a in solids for b in others]
        return min((g for g in found if g is not None), default=None)

    ends, interiors = {}, {}
    if model.femur_box_mm is not None:
        low, high = (np.asarray(v, dtype=float) for v in model.femur_box_mm)
        frame = _frame_at(model.femoral_frame, origin)
        cut = high[2] - end_slab_mm
        slab = model.femur.cropped(frame, (low[0], low[1], cut), high)
        rest = model.femur.cropped(frame, low, (high[0], high[1], cut))
        if not slab.is_empty():
            ends["femur"] = nearest([tibia, insert], [slab])
            interiors["femur"] = (None if rest.is_empty()
                                  else nearest([tibia, insert], [rest]))
    if model.tibia_box_mm is not None:
        low, high = (np.asarray(v, dtype=float) for v in model.tibia_box_mm)
        frame = _frame_at(model.tibial_frame, origin)
        cut = low[2] + end_slab_mm
        slab = model.tibia.cropped(frame, low, (high[0], high[1], cut))
        rest = model.tibia.cropped(frame, (low[0], low[1], cut), high)
        if not slab.is_empty():
            femoral_side = [model.femur, model.femoral_component]
            ends["tibia"] = nearest([slab.transformed(pose)], femoral_side)
            interiors["tibia"] = nearest(
                [insert] + ([] if rest.is_empty() else [rest.transformed(pose)]),
                femoral_side)

    extent = model.diagnostics.get("scan_extent") or {}
    closest = min(((g, p) for p, g in gaps.items() if g is not None), default=None)
    at_end = [bone for bone, g in ends.items() if g is not None
              and (interiors.get(bone) is None or g < interiors[bone] - 0.5)]
    words = [f"No blocking contact in the scanned models by {angle_deg:g} deg."]
    if closest is not None:
        words.append(f"There the closest pair ({closest[1]}) is still "
                     f"{closest[0]:.1f} mm apart")
    else:
        words.append(f"There no pair is within {search_mm:g} mm")
    if at_end:
        bone = at_end[0]
        record = extent.get(bone, {})
        words[-1] += (f"; the gap closes towards the "
                      f"{'proximal' if bone == 'femur' else 'distal'} end of the {bone} "
                      f"as scanned ({ends[bone]:.1f} mm there, against "
                      + ("nothing nearer" if interiors.get(bone) is None
                         else f"{interiors[bone]:.1f} mm")
                      + f" on the rest of it; scanned {record.get('scanned_from_hinge_mm')}"
                      f" mm from the hinge, kept to {record.get('kept_from_hinge_mm')} "
                      "mm): the bone that would close it lies beyond the scan.")
    else:
        words[-1] += (", inside both scans (femur scanned "
                      f"{extent.get('femur', {}).get('scanned_from_hinge_mm')} mm above "
                      f"the hinge, tibia {extent.get('tibia', {}).get('scanned_from_hinge_mm')}"
                      " mm below it): the bone that would meet is in the models, and the "
                      "prescribed motion keeps it clear.")
    return {
        "reason": "bone_not_in_scan" if at_end else "kept_apart_by_motion",
        "at_deg": float(angle_deg),
        "gaps_mm": gaps,
        "gaps_to_shaft_end_mm": {k: (None if v is None else round(v, 2))
                                 for k, v in ends.items()},
        "gaps_to_rest_of_shaft_mm": {k: (None if v is None else round(v, 2))
                                     for k, v in interiors.items()},
        "folded_tibia_reach_from_hinge_mm": round(reach, 1),
        "femur_scanned_from_hinge_mm": extent.get("femur", {}).get(
            "scanned_from_hinge_mm"),
        "femur_kept_from_hinge_mm": extent.get("femur", {}).get("kept_from_hinge_mm"),
        "tibia_scanned_from_hinge_mm": extent.get("tibia", {}).get(
            "scanned_from_hinge_mm"),
        "text": " ".join(words),
    }


def _compartment_ml(m: CaseMeasurement, centre) -> tuple[float, float]:
    """The posterior condyles' positions along the medial axis, from the hinge midpoint
    (mm, + medial): where the trajectory's per-compartment translation is reported.
    Falls back to 22 mm either side when the landmarks are missing."""
    medial = m.femoral_frame.medial
    values = []
    for compartment, default in (("lateral", -22.0), ("medial", 22.0)):
        landmark = f"femur.condyle_posterior_{compartment}"
        if m.landmarks.available(landmark):
            point = np.asarray(m.landmarks.require(landmark)[0], dtype=float)
            values.append(float((point - np.asarray(centre)) @ medial))
        else:
            values.append(default)
    return values[0], values[1]


#: The points whose spacing scales S1's pivot (:func:`condyle_centre_spacing`).
CENTRE_SPACING_LANDMARKS = ("femur.condyle_posterior_medial",
                            "femur.condyle_posterior_lateral")


def condyle_centre_spacing(landmarks, femoral_frame) -> dict:
    """How far apart this knee's medial and lateral condyle centres are along the
    flexion axis (mm): what S1's fitted 52.8 mm stands for.

    S1 was fitted to the motion of the flexion-facet centres (Pinskerova 2004): the
    centres of the circles the posterior condyles follow from about 20 to 120 deg,
    one in each condyle's sagittal plane. Along the flexion axis such a centre lies
    where its condyle's posterior surface does, so the spacing is that of the posterior
    condyle points (``femur.condyle_posterior_medial`` and ``_lateral``, the most
    posterior point of each condyle), measured along the hinge's direction, the femoral
    frame's mediolateral axis (:func:`~tka_planner.scene.motion.flexion_axis`). Not the
    epicondyles, which lie outside the condyles (the fit puts its centres in a femur 80
    to 90 mm wide), and not the distal condyle points, which bear in extension.

    ``centre_spacing_mm`` is ``None``, with ``reason``, when either point is missing or
    the medial one does not lie medial of the lateral one.
    """
    record = {"centre_spacing_mm": None, "landmarks": list(CENTRE_SPACING_LANDMARKS),
              "measured_from": "", "reason": ""}
    missing = landmarks.missing_of(*CENTRE_SPACING_LANDMARKS)
    if missing:
        record["reason"] = f"{', '.join(missing)} not available"
        return record
    medial_point, lateral_point = (np.asarray(p, dtype=float)
                                   for p in landmarks.require(*CENTRE_SPACING_LANDMARKS))
    spacing = float((medial_point - lateral_point) @ femoral_frame.medial)
    statuses = sorted({landmarks.get(i).status.value for i in CENTRE_SPACING_LANDMARKS})
    if not np.isfinite(spacing) or spacing <= 0.0:
        record["reason"] = (f"the medial posterior condyle lies {-spacing:.1f} mm "
                            "lateral of the lateral one along the flexion axis")
        return record
    record["centre_spacing_mm"] = spacing
    record["measured_from"] = (
        f"posterior condyle points ({' and '.join(statuses)}), {spacing:.1f} mm apart "
        "along the flexion axis (the femoral frame's mediolateral axis)")
    return record


def scenarios_case(measurement: CaseMeasurement) -> dict:
    """The flexion scenarios for this knee: S1 with its pivot scaled to the knee's
    condyle-centre spacing (author ruling 2026-10-26; :func:`condyle_centre_spacing`,
    :func:`~tka_planner.core.kinematics.native_for_knee`), S2 and S3 as defined.

    The one place a knee's scenarios come from, for the flexion search, the plan file
    and the application's trial alike.
    """
    from .core.kinematics import knee_scenarios

    spacing = condyle_centre_spacing(measurement.landmarks, measurement.femoral_frame)
    return knee_scenarios(spacing["centre_spacing_mm"],
                          measured_from=spacing["measured_from"],
                          reason=spacing["reason"])


def trajectory_case(measurement: CaseMeasurement, name_or_trajectory):
    """A scenario name resolved for this knee (:func:`scenarios_case`), or a trajectory
    passed through unchanged."""
    from .core.kinematics import scenario

    if not isinstance(name_or_trajectory, str):
        return scenario(name_or_trajectory)
    return scenario(name_or_trajectory, scenarios_case(measurement))


def discrete_sizing(measurement: CaseMeasurement) -> SizingDecision:
    """The legacy round-down chart size, kept beside every plan for comparison."""
    return select_discrete_size(
        measurement.chart,
        measured_ml_mm=measurement.femoral_measure.ml_mm,
        measured_ap_mm=measurement.femoral_measure.ap_mm,
    )


def plan_document(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    *,
    fit: dict | None = None,
    implant_spec: dict | None = None,
    controls: dict | None = None,
    trial: dict | None = None,
    geometry: dict | None = None,
    geometry_matches_plan: bool | None = None,
    resection_table: dict | None = None,
    findings: list | None = None,
    trajectory: str = "S3",
    flexion: dict | None = None,
) -> dict:
    """The one ``plan.json`` shape.

    Inputs are named by file name and content hash, never by full path: a path carries
    the folder layout of whoever planned the case, which can hold identifying names.

    ``fit`` is :func:`fit_case`'s result, ``None`` when no implant library was given.
    ``controls`` and ``trial`` are the application's settings, and ``geometry`` its
    last commit. The command line has none of them and writes ``None``, so the keys are
    always present and a reader never has to ask which front end wrote the file.

    ``resection_table`` (:func:`resection_table_case`) and ``findings``
    (:func:`findings_case`) are measured here when not given. Both were added within
    schema version 2: they are new keys beside the old ones, so a reader of an earlier
    version-2 file loses nothing and :func:`validate_plan_document` does not require
    them.

    ``kinematics`` records the flexion trajectory the plan was previewed with and the
    parameters of every scenario, so a plan says which motion its flexion numbers
    assumed; ``flexion`` is :func:`flexion_case`'s result when it was run.
    """
    from .core.kinematics import scenario

    m = measurement
    scenarios = scenarios_case(m)
    if resection_table is None:
        resection_table = resection_table_case(m, plan)
    if findings is None:
        findings = findings_case(m, plan, sizing, fit=fit)
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "case_id": m.case_id,
        "side": str(m.side),
        "tool_version": __version__,
        "inputs": {
            "femur": m.femur_path.name,
            "femur_sha256": m.femur.sha256,
            "tibia": m.tibia_path.name,
            "tibia_sha256": m.tibia.sha256,
            "size_chart": m.chart_path.name,
            "size_chart_sha256": m.chart.sha256,
            **({"entered": dict(m.entered)} if m.entered else {}),
        },
        "scan_coverage": m.coverage,
        "frames": {"femoral": m.femoral_frame.to_dict(),
                   "tibial": m.tibial_frame.to_dict()},
        "metrics": {name: metric.to_dict() for name, metric in m.metrics.items()},
        "surgical_plan": plan.to_dict(),
        "sizing": {
            "parametric": sizing.to_dict(),
            "discrete": discrete_sizing(m).to_dict(),
            "femoral_measurement": m.femoral_measure.to_dict(),
            "tibial_measurement": m.tibial_measure.to_dict(),
        },
        "quality_control": m.qc.to_dict(),
        "findings": findings,
        "resection_table": resection_table,
        "landmarks": {
            "counts": m.landmark_counts(),
            "source": _plain(m.landmarks.source),
        },
        "implant_spec": (implant_spec if implant_spec is not None
                         else implant_spec_case(m, plan, sizing)),
        "fit": fit,
        "controls": controls,
        "trial": trial,
        "geometry": geometry,
        "geometry_matches_plan": geometry_matches_plan,
        "kinematics": {
            "trajectory": scenario(trajectory, scenarios).name,
            # This knee's: S1's pivot scaled to it, with the spacing and the assumption.
            "scenarios": {name: t.to_dict() for name, t in scenarios.items()},
            "flexion": flexion,
        },
        # Only when a femur length was entered, so a plan without one is byte-for-byte
        # what it was before the phantom existed.
        **({"phantom_femur": phantom_femur_case(m)}
           if m.entered.get("femur_length_mm") is not None else {}),
        "intended_use": INTENDED_USE,
    }


def validate_plan_document(document: dict) -> list[str]:
    """What is wrong with a plan document, as a list; empty means it is valid."""
    if not isinstance(document, dict):
        return ["A plan document must be a JSON object."]
    problems = [f"Missing key {key!r}." for key in REQUIRED_KEYS if key not in document]
    if document.get("schema") not in (None, PLAN_SCHEMA):
        problems.append(f"Schema is {document.get('schema')!r}, not {PLAN_SCHEMA!r}.")
    version = document.get("schema_version")
    if version is not None and version != PLAN_SCHEMA_VERSION:
        problems.append(
            f"Schema version {version!r} is not supported; this tool reads version "
            f"{PLAN_SCHEMA_VERSION}."
        )
    sizing = document.get("sizing")
    if isinstance(sizing, dict):
        problems += [f"Missing sizing key {key!r}."
                     for key in REQUIRED_SIZING_KEYS if key not in sizing]
    plan = document.get("surgical_plan")
    if isinstance(plan, dict):
        for cut in ("femoral_distal", "tibial_proximal"):
            if cut not in plan.get("resections", {}):
                problems.append(f"The surgical plan has no {cut!r} resection.")
    if document.get("intended_use") not in (None, INTENDED_USE):
        problems.append("The intended-use statement has been changed.")
    return problems


def write_plan(document: dict, path) -> Path:
    """Validate and write a plan document. Refuses to write an invalid one.

    Line endings are LF on every platform. Text mode would write CRLF on Windows, and
    the same plan would then hash differently there than on Linux, which is exactly
    the cross-platform byte identity the plan file is meant to have.
    """
    problems = validate_plan_document(document)
    if problems:
        raise ValueError("Refusing to write an invalid plan: " + " ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=_jsonable) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def write_implant_spec(spec: dict, path) -> Path:
    """Write the patient-specific implant specification the CAD model is built from."""
    if spec.get("schema") != "tka-planner/implant-spec":
        raise ValueError("Not an implant specification.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF everywhere, for the same reason as the plan: one specification, one hash.
    path.write_text(json.dumps(spec, indent=2, default=_jsonable) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def read_plan(path) -> dict:
    """Read and validate a plan document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_plan_document(document)
    if problems:
        raise ValueError(f"{Path(path).name} is not a valid plan: " + " ".join(problems))
    return document


def render_case_report(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    *,
    fit: dict | None = None,
    fit_grids: dict | None = None,
    implant_spec: dict | None = None,
    resection_table: dict | None = None,
    findings: list | None = None,
) -> str:
    """The one HTML report, with the same sections whichever front end asks.

    The resection table and the findings are measured here when not given, so even a
    report written without an implant library carries them. The fit, its map and the
    implant's dimensions need the library, and a caller with one passes them in.
    """
    from .report.html import render_report

    m = measurement
    if resection_table is None:
        resection_table = resection_table_case(m, plan)
    if findings is None:
        findings = findings_case(m, plan, sizing, fit=fit)
    return render_report(
        case_id=m.case_id,
        side=str(m.side),
        metrics=m.metrics,
        femoral_frame=m.femoral_frame,
        tibial_frame=m.tibial_frame,
        sizing=sizing,
        comparison_sizing=discrete_sizing(m),
        surgical=plan,
        measurements={"femur": m.femoral_measure.to_dict(),
                      "tibia": m.tibial_measure.to_dict()},
        coverage=m.coverage,
        qc_findings=m.qc.to_dict(),
        inputs=[
            {"role": "femur mesh", "name": m.femur_path.name, "sha256": m.femur.sha256},
            {"role": "tibia mesh", "name": m.tibia_path.name, "sha256": m.tibia.sha256},
            {"role": "size chart", "name": m.chart_path.name, "sha256": m.chart.sha256},
        ],
        landmark_summary=m.landmark_counts(),
        tool_version=__version__,
        resection_table=resection_table,
        implant_spec=implant_spec,
        fit=fit,
        fit_grids=fit_grids,
        findings=findings,
    )


def _plain(value):
    """A JSON-safe deep copy of a landmark set's ``source`` record."""
    return json.loads(json.dumps(value or {}, default=_jsonable))


def _jsonable(value):
    """Convert numpy scalars and arrays to plain Python.

    Numpy's ``bool_``, ``float64`` and friends are not JSON serialisable, and in a
    numpy-heavy codebase they leak into result dictionaries from comparisons and
    reductions almost anywhere. Handling them centrally is more reliable than casting at
    every site.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.name
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
