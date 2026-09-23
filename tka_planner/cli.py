"""Command line entry point.

Runs the whole measurement and reporting pipeline in plain Python. Blender is not
imported, not required, and not installed on the machines that run the test suite --
which is the point: the planning decisions are all made here, and Blender's later role
is to render a plan into geometry, not to make any of them.

    tka measure --femur FD1Left.stl --tibia TD1Left.stl --side left --out plan/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core.frames import (
    FrameConstructionError,
    build_femoral_frame,
    build_tibial_frame,
)
from .core.landmarks import (
    load_landmarks,
    overlay_landmarks,
    write_landmark_set,
    write_slicer_template,
)
from .core.landmarks_auto import estimate_landmarks
from .core.meshio import read_stl
from .core.measure import measure_femoral_ml, measure_tibial_plateau
from .core.metrics import compute_all
from .core.planning import KINEMATIC, MECHANICAL, plan_alignment
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
from .core.sizing import load_size_chart, select_discrete_size, solve_parametric_size
from .report.html import render_report, write_report

DEFAULT_SIZE_CHART = Path(__file__).parent.parent / "data" / "SizeChart.csv"


def _jsonable(value):
    """Convert numpy scalars and arrays to plain Python for serialisation.

    Numpy's ``bool_``, ``float64`` and friends are not JSON serialisable, and in a
    numpy-heavy codebase they leak into result dictionaries from comparisons and
    reductions almost anywhere. Handling them centrally is more reliable than casting at
    every site and remembering to keep doing so.
    """
    import numpy as np

    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _measure(args: argparse.Namespace) -> int:
    side = Side.parse(args.side)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    femur = read_stl(args.femur)
    tibia = read_stl(args.tibia)
    case_id = args.case_id or Path(args.femur).parent.name

    print(f"Case {case_id} ({side})")
    print(f"  femur  {femur.n_triangles:>8,} triangles")
    print(f"  tibia  {tibia.n_triangles:>8,} triangles")

    # -- Quality control ----------------------------------------------
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

    # -- Landmarks ----------------------------------------------------
    #
    # The automatic estimate always runs. A landmark file is laid over it, so a rater's
    # picks replace the estimates they cover and everything else -- canal centres, any
    # point skipped -- stays estimated and says so. A file of picks alone could not
    # build a frame, since no rater is asked for the canal centres.
    landmarks = estimate_landmarks(femur, tibia, side, case_id=case_id)
    if args.landmarks:
        picks = load_landmarks(args.landmarks, case_id=case_id, side=side,
                               rater=args.rater, session=args.session)
        landmarks = overlay_landmarks(landmarks, picks)
        print(f"  landmarks from {Path(args.landmarks).name}: "
              f"{len(landmarks.source['picked'])} picked, "
              f"{len(landmarks.source['skipped'])} skipped; "
              f"the rest estimated automatically")
    else:
        print("  landmarks estimated automatically")
        print("    (machine estimates awaiting review, not picked landmarks)")
    written = write_landmark_set(landmarks, output / "landmarks.json")
    print(f"    -> {written.name}")

    qc.extend(check_landmarks_on_mesh(landmarks, {"femur": femur, "tibia": tibia}))
    qc.raise_if_errors()

    # -- Frames and metrics -------------------------------------------
    try:
        femoral_frame = build_femoral_frame(landmarks)
        tibial_frame = build_tibial_frame(landmarks)
    except FrameConstructionError as error:
        print(f"\n[REFUSED] {error}", file=sys.stderr)
        print(
            "\nThe pipeline declined to build a frame rather than produce a plan it "
            "could not justify. Supply the missing landmarks and re-run.",
            file=sys.stderr,
        )
        return 2

    metrics = compute_all(landmarks, femoral_frame, tibial_frame)

    # -- Sizing -------------------------------------------------------
    chart = load_size_chart(args.size_chart)
    femoral_measure = measure_femoral_ml(femur, femoral_frame)
    tibial_measure = measure_tibial_plateau(tibia, tibial_frame)
    measured_ml, measured_ap = femoral_measure.ml_mm, femoral_measure.ap_mm

    sizing = solve_parametric_size(
        chart, measured_ml_mm=measured_ml, measured_ap_mm=measured_ap
    )
    discrete = select_discrete_size(
        chart, measured_ml_mm=measured_ml, measured_ap_mm=measured_ap
    )

    # -- Alignment ----------------------------------------------------
    target = MECHANICAL if args.philosophy == "mechanical" else KINEMATIC
    surgical = plan_alignment(
        landmarks, femoral_frame, tibial_frame,
        target=target,
        femoral_thickness_mm=sizing.femoral_thickness_mm,
        tibial_resection_mm=args.tibial_resection,
        native_slope_deg=metrics["posterior_slope_medial_deg"].value,
        femur_mesh=femur, tibia_mesh=tibia,
    )

    # -- Report -------------------------------------------------------
    counts: dict[str, int] = {}
    for landmark in landmarks:
        counts[landmark.status.value] = counts.get(landmark.status.value, 0) + 1

    document = render_report(
        case_id=case_id,
        side=str(side),
        metrics=metrics,
        femoral_frame=femoral_frame,
        tibial_frame=tibial_frame,
        sizing=sizing,
        comparison_sizing=discrete,
        surgical=surgical,
        measurements={"femur": femoral_measure.to_dict(),
                      "tibia": tibial_measure.to_dict()},
        coverage={"femur": femur_coverage, "tibia": tibia_coverage},
        qc_findings=qc.to_dict(),
        inputs=[
            {"role": "femur mesh", "name": Path(args.femur).name,
             "sha256": femur.sha256},
            {"role": "tibia mesh", "name": Path(args.tibia).name,
             "sha256": tibia.sha256},
            {"role": "size chart", "name": Path(args.size_chart).name,
             "sha256": chart.sha256},
        ],
        landmark_summary=counts,
        tool_version=__version__,
    )
    report_path = write_report(document, output / "report.html")

    summary = {
        "case_id": case_id,
        "side": str(side),
        "tool_version": __version__,
        "inputs": {
            "femur_sha256": femur.sha256,
            "tibia_sha256": tibia.sha256,
            "size_chart_sha256": chart.sha256,
        },
        "scan_coverage": {"femur": femur_coverage, "tibia": tibia_coverage},
        "frames": {"femoral": femoral_frame.to_dict(),
                   "tibial": tibial_frame.to_dict()},
        "metrics": {name: metric.to_dict() for name, metric in metrics.items()},
        "surgical_plan": surgical.to_dict(),
        "sizing": {
            "parametric": sizing.to_dict(),
            "discrete": discrete.to_dict(),
            "femoral_measurement": femoral_measure.to_dict(),
            "tibial_measurement": tibial_measure.to_dict(),
        },
        "quality_control": qc.to_dict(),
        "intended_use": "research_and_demonstration_only__not_a_medical_device",
    }
    plan_path = output / "plan.json"
    plan_path.write_text(
        json.dumps(summary, indent=2, default=_jsonable) + "\n", encoding="utf-8"
    )

    # -- Console summary ----------------------------------------------
    print()
    for name, metric in metrics.items():
        value = f"{metric.value:8.2f}" if metric.value is not None else "       -"
        print(f"  {name:28s} {value} {metric.unit:4s} {metric.quality.value}")
    femoral_cut = surgical.resections["femoral_distal"]
    tibial_cut = surgical.resections["tibial_proximal"]
    print()
    print(f"  valgus cut       {surgical.distal_femoral_valgus_cut_deg:5.1f} deg  "
          f"({surgical.valgus_quality.value})")
    print(f"  posterior slope  {surgical.tibial_slope_deg:5.1f} deg")
    print(f"  femoral resect   medial {femoral_cut.medial_depth_mm:5.1f} / "
          f"lateral {femoral_cut.lateral_depth_mm:5.1f} mm")
    print(f"  tibial resect    medial {tibial_cut.medial_depth_mm:5.1f} / "
          f"lateral {tibial_cut.lateral_depth_mm:5.1f} mm")
    print(f"  plateau AP       {tibial_measure.ap_mm:5.1f} mm  "
          f"(naive bbox would say "
          f"{tibial_measure.diagnostics['naive_proximal_bbox_ap_mm']:.1f})")
    print()
    print(f"  sizing: parameter {sizing.size_parameter:.3f}, "
          f"implant ML {sizing.implant_ml_mm:.1f} mm "
          f"(discrete chart would give {discrete.nearest_discrete_size})")
    if discrete.flags:
        print(f"  discrete chart flags: {', '.join(discrete.flags)}")
    print()
    print(f"  -> {plan_path}")
    print(f"  -> {report_path}")
    return 0


def _landmark_template(args: argparse.Namespace) -> int:
    path = write_slicer_template(args.out, bones=tuple(args.bones))
    print(f"Slicer picking template -> {path}")
    print("  Load it in 3D Slicer beside the case's STL files and place each point in")
    print("  order; use Skip for a point that cannot be identified.")
    return 0


def _landmark_convert(args: argparse.Namespace) -> int:
    """Turn one rater's Slicer file into the native landmark file, checked and in LPS.

    The meshes are optional but should be given: they are what catches a file declared
    in the wrong coordinate system, which otherwise converts cleanly into a mirror image.
    """
    side = Side.parse(args.side)
    landmarks = load_landmarks(args.input, case_id=args.case_id, side=side,
                               rater=args.rater, session=args.session)
    landmarks = landmarks.to_coordinate_system("LPS")

    qc = QCReport()
    meshes = {bone: read_stl(path)
              for bone, path in (("femur", args.femur), ("tibia", args.tibia)) if path}
    if meshes:
        qc.extend(check_landmarks_on_mesh(landmarks, meshes))
    for finding in qc.findings:
        print(f"  {finding}")
    if not qc.ok:
        return 2

    written = write_landmark_set(landmarks, args.out)
    source = landmarks.source
    print(f"{len([lm for lm in landmarks if lm.is_usable])} picked, "
          f"{len(source.get('unplaced_labels', []))} unplaced, "
          f"{len([lm for lm in landmarks if not lm.is_usable])} skipped, "
          f"{len(source.get('unmatched_labels', []))} unrecognised labels")
    if not meshes:
        print("  (no meshes given: the coordinate-system check did not run)")
    print(f"  -> {written}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Start the local application.

    Imported here rather than at module scope so that `tka measure` on a machine that
    never opens a browser does not pay for the server package.
    """
    from .server.__main__ import DEFAULT_STORE
    from .server.__main__ import main as serve_main

    argv = [
        "--store", args.store or str(DEFAULT_STORE),
        "--port", str(args.port),
        "--host", args.host,
    ]
    if args.cases:
        argv += ["--cases", args.cases]
    if args.library:
        argv += ["--library", args.library]
    if args.no_browser:
        argv.append("--no-browser")
    return serve_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tka",
        description=(
            "Open TKA pre-operative planning. Research and demonstration only; "
            "not a medical device."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser(
        "measure", help="measure a case and write a plan and report"
    )
    measure.add_argument("--femur", required=True, help="femur surface STL")
    measure.add_argument("--tibia", required=True, help="tibia surface STL")
    measure.add_argument("--side", required=True, help="left or right")
    measure.add_argument("--out", default="out", help="output directory")
    measure.add_argument("--case-id", default=None, help="anonymised case identifier")
    measure.add_argument(
        "--landmarks", default=None,
        help="landmark file laid over the automatic estimate; omit to use the "
             "estimate alone",
    )
    measure.add_argument("--rater", default=None,
                         help="who picked the landmark file, if it is a Slicer file")
    measure.add_argument("--session", default=None,
                         help="rating session of the landmark file")
    measure.add_argument("--size-chart", default=str(DEFAULT_SIZE_CHART))
    measure.add_argument(
        "--philosophy", default="mechanical", choices=["mechanical", "kinematic"],
        help="alignment philosophy",
    )
    measure.add_argument(
        "--tibial-resection", type=float, default=10.0,
        help="tibial resection depth below the higher plateau, in mm",
    )
    measure.set_defaults(func=_measure)

    # `landmarks` supports the manual-landmark rating study: a picking template for 3D
    # Slicer, and the conversion of each rater's file into the native format.
    landmarks = subparsers.add_parser(
        "landmarks", help="Slicer picking template and landmark file conversion"
    )
    landmark_actions = landmarks.add_subparsers(dest="action", required=True)

    template = landmark_actions.add_parser(
        "template", help="write a Slicer markups file of named, unplaced points"
    )
    template.add_argument("--out", default="landmark_template.mrk.json")
    template.add_argument("--bones", nargs="+", default=["femur", "tibia", "fibula"],
                          choices=["femur", "tibia", "fibula"])
    template.set_defaults(func=_landmark_template)

    convert = landmark_actions.add_parser(
        "convert", help="convert a rater's Slicer file to the native landmark file"
    )
    convert.add_argument("input", help=".mrk.json or .fcsv file from 3D Slicer")
    convert.add_argument("--case-id", required=True)
    convert.add_argument("--side", required=True, help="left or right")
    convert.add_argument("--rater", required=True, help="rater identifier, e.g. R1")
    convert.add_argument("--session", required=True, help="session number")
    convert.add_argument("--femur", default=None,
                         help="femur STL, for the coordinate-system check")
    convert.add_argument("--tibia", default=None,
                         help="tibia STL, for the coordinate-system check")
    convert.add_argument("--out", required=True, help="native landmark file to write")
    convert.set_defaults(func=_landmark_convert)

    # `serve` runs the planner as an application: a local server and a browser viewer.
    # Its arguments are defined by the server package rather than repeated here, so
    # there is one description of them.
    serve = subparsers.add_parser(
        "serve", help="run the planner as a local application in a browser"
    )
    serve.add_argument("--cases", default=None, help="folder holding one case per "
                       "subfolder")
    serve.add_argument("--library", default=None, help="implant library folder")
    serve.add_argument("--store", default=None, help="where cases, plans and "
                       "committed geometry are kept")
    serve.add_argument("--port", type=int, default=8731)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
