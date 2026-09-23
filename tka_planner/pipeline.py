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
    "implant_spec_case",
    "component_scales",
    "IMPLANT_MODES",
    "write_implant_spec",
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
    return float(_cached_stl(insert["path"]).vertices[:, 2].min()) * insert["scale"]


def _cached_stl(path) -> Mesh:
    """Library parts are read once per process: re-planning happens on every slider
    movement, and the parts never change underneath it."""
    key = str(path)
    if key not in _STL_CACHE:
        _STL_CACHE[key] = read_stl(key)
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
    key = ("dish", spec["path"])
    if key not in _STL_CACHE:
        _STL_CACHE[key] = dish_geometry(_cached_stl(spec["path"]))
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

    key = ("dims", spec["path"])
    if key not in _STL_CACHE:
        mesh = _cached_stl(spec["path"])
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


def implant_spec_case(
    measurement: CaseMeasurement,
    plan: SurgicalPlan,
    sizing: SizingDecision,
    library=None,
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
    spec = spec or implant_spec_case(measurement, plan, sizing, library)
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
    # patient's size it is slid in its own plane until its box centre is the section's.
    shifts = {}
    for group, record, lib in (("femoral", spec["femoral_component"], lib_f),
                               ("tibial", spec["tibial_component"], lib_t)):
        centre = np.asarray(record["diagnostics"]["section_centre_mm"], dtype=float)
        shifts[group] = np.array([
            *(centre - scales[group][:2] * np.asarray(lib["centre"])), 0.0])

    # The centring above follows the pose, so on its own it would undo the surgeon's
    # slide of a component across its cut. The slide is put back on top of it, in the
    # component's own axes: femoral +X anterior, +Y patient-left; tray +X patient-left,
    # +Y posterior. Lateral is patient-left on a left knee.
    lateral_sign = -1.0 if str(measurement.side).lower().startswith("r") else 1.0
    a = plan.adjustments
    shifts["femoral"] += np.array([a.femoral_shift_ap_mm,
                                   lateral_sign * a.femoral_shift_ml_mm, 0.0])
    shifts["tibial"] += np.array([lateral_sign * a.tibial_shift_ml_mm,
                                  -a.tibial_shift_ap_mm, 0.0])
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
    """
    from .core.fit import section_fit
    from .geom import mesh as gm
    from .scene.build import place, resolve_component_meshes

    if library is None:
        return None
    m = measurement
    specs = resolve_component_meshes(
        library, chart=m.chart, sizing=sizing, side=str(m.side)
    )
    transforms = component_scales(m, plan, sizing, library, mode) or {}
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
            _cached_stl(spec["path"]),
            place(plan.components[name], scales.get(group, spec["scale"]),
                  shifts.get(group)),
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
        if group in scales:
            result["library_sizes"][name]["scale_xyz"] = [
                round(float(v), 6) for v in scales[group]]
            result["library_sizes"][name]["shift_mm"] = [
                round(float(v), 3) for v in shifts[group]]
    result["insert"] = _insert_check(plan, specs, m, transforms)
    return result


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
        _cached_stl(femoral_spec["path"]),
        place(plan.components["femoral_component"],
              scales.get("femoral", femoral_spec["scale"]), shifts.get("femoral")),
    )
    insert_pose = (place(plan.components["tibial_component"],
                         scales.get("tibial", insert_spec["scale"]), shifts.get("tibial"))
                   @ insert_stretch_matrix(display))
    insert = gm.transformed(_cached_stl(insert_spec["path"]), insert_pose)
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
        "implant_spec": (implant_spec if implant_spec is not None
                         else implant_spec_case(m, plan, sizing)),
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


def write_implant_spec(spec: dict, path) -> Path:
    """Write the patient-specific implant specification the CAD model is built from."""
    if spec.get("schema") != "tka-planner/implant-spec":
        raise ValueError("Not an implant specification.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, default=_jsonable) + "\n",
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
