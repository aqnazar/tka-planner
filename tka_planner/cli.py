"""Command line entry point.

Runs the whole measurement and reporting pipeline in plain Python. Blender is not
imported, not required, and not installed on the machines that run the test suite --
which is the point: the planning decisions are all made here, and Blender's later role
is to render a plan into geometry, not to make any of them.

    tka measure --femur FD1Left.stl --tibia TD1Left.stl --side left --out plan/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core.frames import FrameConstructionError
from .core.landmarks import load_landmarks, write_landmark_set, write_slicer_template
from .core.meshio import read_stl
from .core.planning import Adjustments
from .core.qc import QCReport, QualityControlError, check_landmarks_on_mesh
from .core.sides import Side
from .pipeline import (
    discrete_sizing,
    measure_case,
    plan_case,
    plan_document,
    render_case_report,
    write_plan,
)
from .report.html import write_report


def _measure(args: argparse.Namespace) -> int:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    # The same pipeline the application runs: quality control, landmarks (the
    # automatic estimate, with any landmark file laid over it), frames, metrics and the
    # sizing measurement.
    try:
        measurement = measure_case(
            args.femur, args.tibia, args.side,
            case_id=args.case_id, landmarks_path=args.landmarks,
            rater=args.rater, session=args.session, size_chart=args.size_chart,
        )
    except QualityControlError as error:
        print(f"\n[REFUSED] {error}", file=sys.stderr)
        return 2
    except FrameConstructionError as error:
        print(f"\n[REFUSED] {error}", file=sys.stderr)
        print(
            "\nThe pipeline declined to build a frame rather than produce a plan it "
            "could not justify. Supply the missing landmarks and re-run.",
            file=sys.stderr,
        )
        return 2

    m = measurement
    print(f"Case {m.case_id} ({m.side})")
    print(f"  femur  {m.femur.n_triangles:>8,} triangles")
    print(f"  tibia  {m.tibia.n_triangles:>8,} triangles")
    source = m.landmarks.source
    if args.landmarks:
        print(f"  landmarks from {Path(args.landmarks).name}: "
              f"{len(source['picked'])} picked, {len(source['skipped'])} skipped; "
              f"the rest estimated automatically")
    else:
        print("  landmarks estimated automatically")
        print("    (machine estimates awaiting review, not picked landmarks)")
    written = write_landmark_set(m.landmarks, output / "landmarks.json")
    print(f"    -> {written.name}")
    for finding in m.qc.findings:
        print(f"  {finding}")

    surgical, sizing = plan_case(
        m,
        philosophy=args.philosophy,
        size_label=args.size,
        adjustments=Adjustments(
            femoral_resection_delta_mm=args.femoral_resection_delta,
            tibial_resection_delta_mm=args.tibial_resection_delta,
        ),
    )
    discrete = discrete_sizing(m)
    controls = {
        "philosophy": args.philosophy,
        "size_override": args.size or "",
        "femoral_resection_delta_mm": args.femoral_resection_delta,
        "tibial_resection_delta_mm": args.tibial_resection_delta,
    }

    report_path = write_report(render_case_report(m, surgical, sizing),
                               output / "report.html")
    plan_path = write_plan(plan_document(m, surgical, sizing, controls=controls),
                           output / "plan.json")

    # -- Console summary ----------------------------------------------
    print()
    for name, metric in m.metrics.items():
        value = f"{metric.value:8.2f}" if metric.value is not None else "       -"
        print(f"  {name:28s} {value} {metric.unit:4s} {metric.quality.value}")
    femoral_cut = surgical.resections["femoral_distal"]
    tibial_cut = surgical.resections["tibial_proximal"]
    diagnostics = surgical.diagnostics
    print()
    print(f"  valgus cut       {surgical.distal_femoral_valgus_cut_deg:5.1f} deg  "
          f"({surgical.valgus_quality.value})")
    print(f"  posterior slope  {surgical.tibial_slope_deg:5.1f} deg")
    print(f"  femoral resect   {diagnostics['femoral_resection_from_datum_mm']:5.1f} mm "
          f"from the distal condyle; medial {femoral_cut.medial_depth_mm:5.1f} / "
          f"lateral {femoral_cut.lateral_depth_mm:5.1f} mm")
    print(f"  tibial resect    {diagnostics['tibial_resection_from_datum_mm']:5.1f} mm "
          f"from the top of the tibia; below the plateaus medial "
          f"{tibial_cut.medial_depth_mm:5.1f} / lateral "
          f"{tibial_cut.lateral_depth_mm:5.1f} mm")
    print(f"  plateau AP       {m.tibial_measure.ap_mm:5.1f} mm  "
          f"(naive bbox would say "
          f"{m.tibial_measure.diagnostics['naive_proximal_bbox_ap_mm']:.1f})")
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
    path = write_slicer_template(args.out, bones=tuple(args.bones),
                                 include_out_of_scan=args.full_length)
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
    measure.add_argument("--size-chart", default=None,
                         help="size chart CSV; defaults to the bundled chart")
    measure.add_argument(
        "--philosophy", default="mechanical", choices=["mechanical", "kinematic"],
        help="alignment philosophy",
    )
    measure.add_argument(
        "--size", default=None,
        help="a chart size, e.g. M2, instead of the size solved from the anatomy",
    )
    # The depths themselves come from the size chart for the solved size; these move
    # the cut from there, as the application's resection controls do.
    measure.add_argument(
        "--femoral-resection-delta", type=float, default=0.0,
        help="bone off the distal femur beyond the chart depth for the size, in mm",
    )
    measure.add_argument(
        "--tibial-resection-delta", type=float, default=0.0,
        help="bone off the proximal tibia beyond the chart depth for the size, in mm",
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
    template.add_argument("--full-length", action="store_true",
                          help="also offer the hip and ankle centres, for scans that "
                               "include them")
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
