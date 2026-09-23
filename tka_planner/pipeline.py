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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .core.frames import AnatomicalFrame, build_femoral_frame, build_tibial_frame
from .core.landmarks import LandmarkSet, load_landmarks, overlay_landmarks
from .core.landmarks_auto import estimate_landmarks
from .core.measure import measure_femoral_ml, measure_tibial_plateau
from .core.meshio import Mesh, read_stl
from .core.metrics import compute_all
from .core.planning import (
    KINEMATIC,
    MECHANICAL,
    NO_ADJUSTMENT,
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
)
from .core.sides import Side
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
]

PLAN_SCHEMA = "tka-planner/plan"
# Version 1 was the unversioned pair of shapes the command line and the application
# wrote before they shared this module.
PLAN_SCHEMA_VERSION = 2
INTENDED_USE = "research_and_demonstration_only__not_a_medical_device"

# Every plan document carries these, whichever front end wrote it.
REQUIRED_KEYS = (
    "schema", "schema_version", "case_id", "side", "tool_version", "inputs",
    "scan_coverage", "frames", "metrics", "surgical_plan", "sizing",
    "quality_control", "landmarks", "fit", "controls", "trial", "geometry",
    "geometry_matches_plan", "intended_use",
)
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
) -> CaseMeasurement:
    """Read, check and measure a case.

    Quality control runs here, so no front end can skip it. An error-level finding
    raises :class:`~tka_planner.core.qc.QualityControlError`; a frame that cannot be
    justified raises :class:`~tka_planner.core.frames.FrameConstructionError`. Both are
    refusals rather than failures, and the caller decides how to say so.

    The automatic landmark estimate always runs. A landmark file is laid over it, so
    each pick replaces its estimate and everything else stays estimated and says so.
    """
    side = Side.parse(side)
    femur_path, tibia_path = Path(femur_path), Path(tibia_path)
    case_id = case_id or femur_path.parent.name
    femur = read_stl(femur_path)
    tibia = read_stl(tibia_path)

    qc = QCReport()
    for mesh, bone in ((femur, "femur"), (tibia, "tibia")):
        qc.extend(check_plausible_bone_scale(mesh, bone))
        qc.extend(check_laterality(mesh, side))
        _, mesh_report = assess_mesh_quality(mesh)
        qc.extend(mesh_report)
    qc.raise_if_errors()

    femur_coverage, femur_report = assess_femur_coverage(femur)
    tibia_coverage, tibia_report = assess_tibia_coverage(tibia)
    qc.extend(femur_report)
    qc.extend(tibia_report)

    landmarks = estimate_landmarks(femur, tibia, side, case_id=case_id)
    if landmarks_path:
        picks = load_landmarks(landmarks_path, case_id=case_id, side=side,
                               rater=rater, session=session)
        landmarks = overlay_landmarks(landmarks, picks)
    qc.extend(check_landmarks_on_mesh(landmarks, {"femur": femur, "tibia": tibia}))
    qc.raise_if_errors()

    return measure_from_landmarks(
        landmarks, femur, tibia,
        femur_path=femur_path, tibia_path=tibia_path, case_id=case_id, side=side,
        qc=qc, coverage={"femur": femur_coverage, "tibia": tibia_coverage},
        size_chart=size_chart,
    )


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
    tibial_tray_thickness_mm: float = 0.0,
) -> tuple[SurgicalPlan, SizingDecision]:
    """Size the implant and plan the cuts for one set of controls.

    ``size_label`` picks a chart size by hand; empty, or a size the chart does not
    publish, means solve it from the anatomy. Both resection depths follow the size.
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
    plan = plan_alignment(
        measurement.landmarks, measurement.femoral_frame, measurement.tibial_frame,
        target=MECHANICAL if philosophy == "mechanical" else KINEMATIC,
        femoral_thickness_mm=sizing.femoral_thickness_mm,
        tibial_resection_mm=sizing.tibial_resection_mm,
        tibial_tray_thickness_mm=tibial_tray_thickness_mm,
        native_slope_deg=measurement.native_slope_deg,
        adjustments=adjustments,
        femur_mesh=measurement.femur,
        tibia_mesh=measurement.tibia,
    )
    return plan, sizing


def fit_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library,
) -> dict | None:
    """How the planned components fit the bone, on each cut.

    The components are taken from the implant library at the plan's size and posed
    exactly as the viewer poses them, so the fit is of what the surgeon sees. Returns
    ``None`` with no library, and names any component the library lacks rather than
    guessing its shape.
    """
    from .core.fit import section_fit
    from .geom import mesh as gm
    from .scene.build import resolve_component_meshes, seat

    if library is None:
        return None
    m = measurement
    specs = resolve_component_meshes(
        library, chart=m.chart, sizing=sizing, side=str(m.side)
    )
    result = {"library_sizes": {}, "missing": []}
    for name, cut, bone, frame, side_of_cut in (
        ("tibial_component", "tibial_proximal", m.tibia, m.tibial_frame, 1.0),
        ("femoral_component", "femoral_distal", m.femur, m.femoral_frame, -1.0),
    ):
        spec = specs.get(name)
        if spec is None:
            result["missing"].append(name)
            continue
        posed = gm.transformed(
            read_stl(spec["path"]), seat(plan.components[name], spec["scale"])
        )
        resection = plan.resections[cut]
        fit = section_fit(
            cut=cut, bone=bone, component=posed,
            point=resection.point, normal=resection.normal,
            component_side=side_of_cut,
            lateral=frame.lateral, anterior=frame.x_anterior,
        )
        result[name] = fit.to_dict()
        result["library_sizes"][name] = {
            "source_size": spec["source_size"], "scale": round(spec["scale"], 6),
        }
    return result


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
    controls: dict | None = None,
    trial: dict | None = None,
    geometry: dict | None = None,
    geometry_matches_plan: bool | None = None,
) -> dict:
    """The one ``plan.json`` shape.

    Inputs are named by file name and content hash, never by full path: a path carries
    the folder layout of whoever planned the case, which can hold identifying names.

    ``fit`` is :func:`fit_case`'s result, ``None`` when no implant library was given.
    ``controls`` and ``trial`` are the application's settings, and ``geometry`` its
    last commit. The command line has none of them and writes ``None``, so the keys are
    always present and a reader never has to ask which front end wrote the file.
    """
    m = measurement
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
        "landmarks": {
            "counts": m.landmark_counts(),
            "source": _plain(m.landmarks.source),
        },
        "fit": fit,
        "controls": controls,
        "trial": trial,
        "geometry": geometry,
        "geometry_matches_plan": geometry_matches_plan,
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
    """Validate and write a plan document. Refuses to write an invalid one."""
    problems = validate_plan_document(document)
    if problems:
        raise ValueError("Refusing to write an invalid plan: " + " ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, default=_jsonable) + "\n",
                    encoding="utf-8")
    return path


def read_plan(path) -> dict:
    """Read and validate a plan document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_plan_document(document)
    if problems:
        raise ValueError(f"{Path(path).name} is not a valid plan: " + " ".join(problems))
    return document


def render_case_report(
    measurement: CaseMeasurement, plan: SurgicalPlan, sizing: SizingDecision
) -> str:
    """The one HTML report, with the same sections whichever front end asks."""
    from .report.html import render_report

    m = measurement
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
