"""Each knee, side by side: the Paper 1 script against the current planner, in 3D.

``python -m paper.legacy_view [P009L ...] [--serve]`` writes a viewer to
``<out>/E12/viewer/``: per knee, the bones, the parts the frozen Paper 1 script placed
(read back from its own output files), the parts the planner places for the same knee
-- *like-for-like* (the legacy size and the legacy tibial datum) and its *default* --
and the two main cuts of each, with the numbers that separate them. Everything is in the
knee's own STL frame, in millimetres, so a point picked in the viewer can be checked
against the STL in any other tool.

It reads the legacy runs E12 caches in ``<out>/E12/legacy_runs/`` and does not run the
script itself: run ``python -m paper run E12 --reference`` first. The output holds
patient geometry, so it stays in the gitignored ``paper/out/``; open it with ``--serve``
(a browser will not load the page's modules from ``file://``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .harness import cohort_cases, library_root, measured, out_root
from .legacy_runner import output_dir
from .pose_compare import plane_difference, pose_difference

PAGE = Path(__file__).resolve().parent / "legacy_view_page"
VENDOR = Path(__file__).resolve().parent.parent / "eval" / "demo_app" / "vendor"

BONE_TRIANGLES = 120_000
PART_TRIANGLES = 60_000

COLOURS = {
    "bone": "#e8dcc4",
    "legacy": "#b5179e",
    "like_for_like": "#2a9d4b",
    "default": "#2b6cb0",
}
VARIANTS = {
    "like_for_like": "Planner, like-for-like (legacy size, legacy tibial datum)",
    "default": "Planner, default (own size, 9 mm below the less affected plateau)",
}
LEGACY_PARTS = {
    "femoral_component": ("femur", "Implant_Femoral_{Side}_{size}.stl", "Femoral component"),
    "tibial_component": ("tibia", "Tibial_Plate_{size}.stl", "Tibial tray"),
    "tibial_insert": ("tibia", "Tibial_Insert_{size}.stl", "Insert"),
    "femur_resected": ("femur", "Femur_{Side}_{size}_result.stl", "Femur as cut"),
    "tibia_resected": ("tibia", "Tibia_{Side}_{size}_result.stl", "Tibia as cut"),
}
NEW_PARTS = {
    "femoral_component": ("femoral_component", "Femoral component"),
    "tibial_component": ("tibial_component", "Tibial tray"),
    # The tray and the insert share one CAD origin, so the insert takes the tray's pose.
    "tibial_insert": ("tibial_component", "Insert"),
}
# The compartment landmarks each cut's depth is measured to.
DEPTH_LANDMARKS = {
    "femoral_distal": ("femur.condyle_distal_medial", "femur.condyle_distal_lateral"),
    "tibial_proximal": ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"),
}
METRICS = [
    ("mldfa_deg", "mLDFA"), ("mpta_deg", "MPTA"), ("hka_deviation_deg", "HKA deviation"),
    ("jlca_deg", "JLCA"), ("aldfa_deg", "aLDFA"), ("femoral_ama_deg", "Femoral AMA"),
    ("posterior_slope_medial_deg", "Medial posterior slope"),
    ("condylar_twist_deg", "Condylar twist"),
    ("pco_medial_mm", "PCO medial"), ("pco_lateral_mm", "PCO lateral"),
]


# ---- meshes ---------------------------------------------------------------------------

def _indexed(mesh, target: int):
    """A welded, indexed mesh of at most ``target`` triangles, for drawing only."""
    from tka_planner.geom.resample import decimate, weld

    welded = weld(mesh)
    if welded.n_triangles > target:
        welded = decimate(welded, target)
    return np.asarray(welded.vertices, float), np.asarray(welded.faces, np.int64)


def _write_mesh(folder: Path, name: str, vertices, faces, centre) -> dict:
    """Float32 positions relative to ``centre``, then uint32 indices, in one file."""
    v = (np.asarray(vertices, float) - centre).astype("<f4")
    f = np.asarray(faces).astype("<u4")
    (folder / f"{name}.bin").write_bytes(v.tobytes() + f.tobytes())
    return {"file": f"{name}.bin", "n_vertices": int(len(v)), "n_triangles": int(len(f))}


def _transform(matrix, vertices):
    m = np.asarray(matrix, float)
    return vertices @ m[:3, :3].T + m[:3, 3]


# ---- numbers --------------------------------------------------------------------------

def _removed_side(normal, proximal, bone: str):
    """The cut normal turned to point into the bone the cut removes."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    toward = -1.0 if bone == "femur" else 1.0  # distal femur, proximal tibia
    return n if np.dot(n, proximal) * toward > 0 else -n


def _depths(m, plane: str, point, normal, proximal) -> tuple[float | None, float | None]:
    """Bone removed under each compartment's landmark, measured along the cut normal."""
    bone = "femur" if plane.startswith("femoral") else "tibia"
    n = _removed_side(normal, proximal, bone)
    out = []
    for landmark in DEPTH_LANDMARKS[plane]:
        position = m.landmarks.get(landmark).position_mm
        out.append(None if position is None
                   else float(np.dot(np.asarray(position) - np.asarray(point), n)))
    return out[0], out[1]


def _tilts(normal, frame) -> tuple[float, float]:
    """The cut normal's tilt from the bone's proximal axis: toward lateral, toward
    anterior (degrees). Zero and zero is a cut square to the axis."""
    normal = np.asarray(normal, float)
    if np.dot(normal, frame.z_proximal) < 0:
        normal = -normal
    d = pose_difference(_pose_from(normal), _pose_from(frame.z_proximal),
                        anterior=frame.x_anterior, lateral=frame.lateral,
                        proximal=frame.z_proximal)
    return d["legacy_coronal_deg"], d["legacy_sagittal_deg"]


def _pose_from(normal):
    pose = np.eye(4)
    pose[:3, 2] = np.asarray(normal, float) / np.linalg.norm(normal)
    return pose


def _round(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), digits)


# ---- one knee -------------------------------------------------------------------------

def build_case(case, run: dict, runs: Path, folder: Path) -> dict:
    """Write one knee's meshes and return its description for the page."""
    from tka_planner.core.meshio import read_stl
    from tka_planner.pipeline import discrete_sizing, plan_case
    from tka_planner.scene.build import read_component, resolve_component_meshes

    from .experiments.e12_legacy_baseline import _legacy_flags

    folder.mkdir(parents=True, exist_ok=True)
    side_cap = "Left" if case.side == "left" else "Right"
    log = run.get("log") or {}
    size = log.get("size")
    out = output_dir(run, runs)
    m = measured(case)
    refused = isinstance(m, Exception)

    femur = read_stl(case.femur, compute_hash=False)
    tibia = read_stl(case.tibia, compute_hash=False)
    centre = (np.asarray(m.femoral_frame.origin, float) if not refused
              else np.asarray(femur.vertices, float).min(axis=0))
    layers, planes, notes = [], [], []

    def add(layer_id, group, label, colour, vertices, faces, *, visible=True, opacity=1.0):
        info = _write_mesh(folder, layer_id, vertices, faces, centre)
        layers.append({"id": layer_id, "group": group, "label": label, "colour": colour,
                       "visible": visible, "opacity": opacity, **info})

    for name, mesh in (("femur", femur), ("tibia", tibia)):
        v, f = _indexed(mesh, BONE_TRIANGLES)
        add(f"bone_{name}", "bone", f"{name.capitalize()} (input STL)", COLOURS["bone"], v, f,
            opacity=0.45)

    # ---- the legacy parts, mapped back from the legacy world ------------------------
    poses = run.get("poses") or {}
    bones = {k: np.linalg.inv(np.array(v)) for k, v in poses.get("bones", {}).items()}
    for key, (bone, pattern, label) in LEGACY_PARTS.items():
        path = out / pattern.format(Side=side_cap, size=size)
        if not size or bone not in bones or not path.is_file():
            notes.append(f"Legacy {label.lower()}: no output file.")
            continue
        mesh = read_stl(path, compute_hash=False)
        if mesh.n_triangles < 1000 and key.endswith("resected"):
            notes.append(f"Legacy {label.lower()} is nearly empty "
                         f"({mesh.n_triangles} triangles): the script's cut failed.")
            continue
        v, f = _indexed(mesh, BONE_TRIANGLES if key.endswith("resected") else PART_TRIANGLES)
        resected = key.endswith("resected")
        add(f"legacy_{key}", "legacy", f"Paper 1: {label.lower()}", COLOURS["legacy"],
            _transform(bones[bone], v), f, visible=not resected,
            opacity=0.55 if resected else 0.9)

    legacy_poses = {k: np.array(v) for k, v in poses.get("components", {}).items()}
    back = run.get("back_map") or {}
    legacy_cuts = {}
    for plane, part, bone in (("femoral_distal", "femoral_component", "femur"),
                              ("tibial_proximal", "tibial_component", "tibia")):
        if part in legacy_poses:
            lp = legacy_poses[part]
            # The face the legacy cutting block left, which in the Paper 1 library sits a
            # fraction of a millimetre off the component's origin (E12 measures it).
            offset = back.get(bone, {}).get("cut_offset_mm") or 0.0
            legacy_cuts[plane] = (lp[:3, 3] + offset * lp[:3, 2], lp[:3, 2], offset)

    summary = {
        "case": case.case_id, "side": case.side, "reference": case.reference,
        "legacy_ok": bool(run.get("ok")), "legacy_size": size,
        "flags": _legacy_flags(run, out),
        "planner": "refused" if refused else "accepted",
    }
    tables = []
    if refused:
        summary["planner_reason"] = f"{type(m).__name__}: {str(m).splitlines()[0][:300]}"
        for plane, (point, normal, _) in legacy_cuts.items():
            planes.append({"id": f"legacy_{plane}", "group": "legacy",
                           "label": f"Paper 1 {plane.replace('_', ' ')} cut",
                           "colour": COLOURS["legacy"],
                           "point": (point - centre).tolist(), "normal": list(normal)})
        return _finish(case, folder, centre, None, layers, planes, tables, notes, summary,
                       run)

    frames = {"femoral": m.femoral_frame, "tibial": m.tibial_frame}
    plans = {"like_for_like": plan_case(m, size_label=size, tibial_reference="top_of_tibia"),
             "default": plan_case(m)}
    library = library_root()

    # ---- the planner's parts ---------------------------------------------------------
    for variant, (plan, sizing) in plans.items():
        specs = (resolve_component_meshes(library, chart=m.chart, sizing=sizing,
                                          side=str(m.side)) if library else {})
        for key, (pose_key, label) in NEW_PARTS.items():
            spec = specs.get(key)
            if spec is None:
                notes.append(f"{VARIANTS[variant]}: no library {label.lower()}.")
                continue
            part = read_component(spec)
            v, f = _indexed(part, PART_TRIANGLES)
            v = _transform(plan.components[pose_key], v * spec["scale"])
            scale = "" if abs(spec["scale"] - 1.0) < 1e-9 else f" x{spec['scale']:.3f}"
            name = "Like-for-like" if variant == "like_for_like" else "Default"
            add(f"{variant}_{key}", variant,
                f"{name}: {label.lower()} ({sizing.nearest_discrete_size}{scale})",
                COLOURS[variant], v, f, visible=variant == "like_for_like", opacity=0.9)

    # ---- cut planes ------------------------------------------------------------------
    for plane, frame_key in (("femoral_distal", "femoral"), ("tibial_proximal", "tibial")):
        frame = frames[frame_key]
        title = "Distal femoral cut" if frame_key == "femoral" else "Proximal tibial cut"
        cut_rows, worst = [], 0.0
        entries = []
        if plane in legacy_cuts:
            entries.append(("Paper 1", "legacy", *legacy_cuts[plane][:2]))
        for variant, (plan, _) in plans.items():
            res = plan.resections[plane]
            entries.append(("Like-for-like" if variant == "like_for_like" else "Default",
                            variant, res.point, res.normal))
        for label, group, point, normal in entries:
            medial, lateral = _depths(m, plane, point, normal, frame.z_proximal)
            coronal, sagittal = _tilts(normal, frame)
            if group != "legacy":
                # The same depth as the planner reports it, as a check on this measurement.
                res = plans[group][0].resections[plane]
                for mine, its in ((medial, res.medial_depth_mm), (lateral, res.lateral_depth_mm)):
                    if mine is not None and its is not None:
                        worst = max(worst, abs(mine - its))
            cut_rows.append({
                "source": label,
                "medial depth (mm)": _round(medial), "lateral depth (mm)": _round(lateral),
                "tilt to lateral (deg)": _round(coronal),
                "tilt to anterior (deg)": _round(sagittal),
            })
            planes.append({"id": f"{group}_{plane}", "group": group,
                           "label": f"{label}: {title.lower()}", "colour": COLOURS[group],
                           "point": (np.asarray(point) - centre).tolist(),
                           "normal": list(map(float, normal)),
                           "visible": group in ("legacy", "like_for_like")})
        landmarks = " and ".join(DEPTH_LANDMARKS[plane])
        tables.append({
            "title": title,
            "note": f"Depth: bone removed under {landmarks}, along the cut's normal "
                    f"(negative: the cut passes beyond it); the same measurement gives the "
                    f"planner's own reported depths to {worst:.3f} mm. Tilt: the cut normal "
                    f"from the planner's {frame_key} proximal axis, toward lateral and "
                    f"toward anterior; 0 / 0 is square to the axis. The Paper 1 cut is the "
                    f"face its cutting block left. Planner datum: "
                    f"{plans['default'][0].resections[plane].reference}.",
            "rows": cut_rows,
        })

    diff_rows = []
    for variant, (plan, _) in plans.items():
        for plane, part, frame_key in (("femoral_distal", "femoral_component", "femoral"),
                                       ("tibial_proximal", "tibial_component", "tibial")):
            if part not in legacy_poses:
                continue
            f = frames[frame_key]
            d = pose_difference(legacy_poses[part], plan.components[part],
                                anterior=f.x_anterior, lateral=f.lateral,
                                proximal=f.z_proximal)
            res = plan.resections[plane]
            point, normal, _ = legacy_cuts[plane]
            p = plane_difference(point, normal, res.point, res.normal,
                                 centre=f.origin, proximal=f.z_proximal)
            diff_rows.append({
                "variant": "Like-for-like" if variant == "like_for_like" else "Default",
                "component": "Femoral" if frame_key == "femoral" else "Tibial tray",
                "position (mm)": _round(d["position_mm"]),
                "anterior (mm)": _round(d["anterior_mm"]),
                "lateral (mm)": _round(d["lateral_mm"]),
                "proximal (mm)": _round(d["proximal_mm"]),
                "rotation (deg)": _round(d["rotation_deg"]),
                "coronal (deg)": _round(d["coronal_tilt_deg"]),
                "sagittal (deg)": _round(d["sagittal_tilt_deg"]),
                "axial turn (deg)": _round(d["axial_turn_deg"]),
                "cut angle (deg)": _round(p["angle_deg"]),
                "cut offset at centre (mm)": _round(p["offset_at_centre_mm"]),
            })
    # One row per quantity, one column per component and variant: narrow enough to read.
    columns = [f"{r.pop('component')}, {r.pop('variant').lower()}" for r in diff_rows]
    diff_rows = [{"": key, **{c: r[key] for c, r in zip(columns, diff_rows)}}
                 for key in (diff_rows[0] if diff_rows else {})]
    tables.append({
        "title": "Paper 1 minus planner",
        "note": "Each component's CAD origin and orientation, legacy minus planner, along "
                "the planner's frame of that bone (anterior, lateral, proximal). Axial "
                "turn: about the planner's cut normal, positive counter-clockwise looking "
                "down it. Cut offset: where the two cut planes cross the bone axis through "
                "the frame origin, positive when Paper 1 cuts further proximally.",
        "rows": diff_rows,
    })

    calls = run.get("calls") or {}
    quiet = {(c["bone"], c["mode"]): c for c in calls.get("select_size_quiet", [])}
    primary = (calls.get("select_size") or [{}])[0]
    discrete = discrete_sizing(m)
    tables.append({
        "title": "Sizing",
        "note": "Paper 1 sizes from the world-axis bounding box of the bone trimmed to "
                "150 mm, rounding down on the chart; the planner from the distal block's "
                "extent in its own frame.",
        "rows": [
            {"": "Paper 1", "size": size,
             "femoral ML (mm)": _round(primary.get("measurement_mm"), 1),
             "femoral AP (mm)": _round(quiet.get(("femur", "AP"), {}).get("measurement_mm"), 1),
             "tibial ML (mm)": _round(quiet.get(("tibia", "ML"), {}).get("measurement_mm"), 1),
             "note": "; ".join(x for x in (
                 f"femoral AP suggests {log['cross_check_size']}"
                 if log.get("cross_check_size") and log.get("cross_check_size") != size else "",
                 f"tibia alone sizes {log['tibia_size']}"
                 if log.get("tibia_size") and log.get("tibia_size") != size else "") if x)},
            {"": "Planner, chart", "size": discrete.nearest_discrete_size,
             "femoral ML (mm)": _round(m.femoral_measure.ml_mm, 1),
             "femoral AP (mm)": _round(m.femoral_measure.ap_mm, 1),
             "tibial ML (mm)": _round(getattr(m.tibial_measure, "ml_mm", None), 1),
             "note": m.femoral_measure.method},
            {"": "Planner, parametric", "size": plans["default"][1].nearest_discrete_size,
             "femoral ML (mm)": None, "femoral AP (mm)": None, "tibial ML (mm)": None,
             "note": f"size parameter {plans['default'][1].size_parameter:.2f}"},
        ],
    })

    metric_rows = []
    for key, label in METRICS:
        metric = m.metrics.get(key)
        if metric is None:
            continue
        metric_rows.append({"metric": label,
                            "value": _round(metric.value, 2), "unit": metric.unit,
                            "tier": metric.quality.value})
    tables.append({
        "title": "This knee, as the planner measures it",
        "note": "Paper 1 measures no alignment; these are the planner's.",
        "rows": metric_rows,
    })
    return _finish(case, folder, centre, frames, layers, planes, tables, notes, summary, run)


def _finish(case, folder, centre, frames, layers, planes, tables, notes, summary, run):
    frame_payload = {}
    for key, f in (frames or {}).items():
        frame_payload[key] = {"origin": (np.asarray(f.origin) - centre).tolist(),
                              "anterior": list(map(float, f.x_anterior)),
                              "lateral": list(map(float, f.lateral)),
                              "proximal": list(map(float, f.z_proximal))}
    document = {"summary": summary, "centre_stl_mm": centre.tolist(), "frames": frame_payload,
                "layers": layers, "planes": planes, "tables": tables, "notes": notes,
                "legacy_log": {k: v for k, v in (run.get("log") or {}).items()
                               if not isinstance(v, (dict, list)) or k.endswith("_mm")}}
    (folder / "case.json").write_text(json.dumps(document, indent=1, default=float),
                                      encoding="utf-8")
    return summary


# ---- the whole viewer -------------------------------------------------------------------

def build(ids=None, *, out: Path | None = None) -> Path:
    runs = out_root() / "E12" / "legacy_runs"
    target = out or (out_root() / "E12" / "viewer")
    if not runs.is_dir():
        raise SystemExit(f"No legacy runs in {runs}: run `python -m paper run E12 "
                         f"--reference` first.")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "viewer.js", "style.css"):
        shutil.copy2(PAGE / name, target / name)
    (target / "vendor").mkdir(exist_ok=True)
    for name in ("three.module.js", "OrbitControls.js", "LICENSE-three.txt"):
        shutil.copy2(VENDOR / name, target / "vendor" / name)

    index = []
    for case in cohort_cases(include_reference=True):
        if ids and case.case_id not in ids:
            continue
        result = runs / case.case_id / "legacy_result.json"
        if not result.is_file():
            print(f"{case.case_id}: no legacy run, skipped", flush=True)
            continue
        run = json.loads(result.read_text(encoding="utf-8"))
        print(f"{case.case_id} ...", flush=True)
        index.append(build_case(case, run, runs, target / "cases" / case.case_id))

    previous = target / "cases.json"
    if ids and previous.is_file():
        kept = [c for c in json.loads(previous.read_text(encoding="utf-8"))
                if c["case"] not in {x["case"] for x in index}]
        index = kept + index
    order = {c.case_id: i for i, c in enumerate(cohort_cases(include_reference=True))}
    index.sort(key=lambda c: order.get(c["case"], 99))
    previous.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper.legacy_view",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="knees to (re)build; all when empty")
    parser.add_argument("--serve", action="store_true",
                        help="serve the viewer on 127.0.0.1 after building")
    parser.add_argument("--no-build", action="store_true", help="only serve")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args(argv)
    target = out_root() / "E12" / "viewer"
    if not args.no_build:
        target = build(args.ids or None)
        print(f"viewer: {target}")
    if args.serve:
        import functools
        import http.server

        handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                    directory=str(target))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
