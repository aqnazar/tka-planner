"""E12: the Paper 1 pipeline against the current planner, on identical inputs.

Three questions, one table each:

1. **Does the frozen script still reproduce Paper 1?** (developer question Z-4) Each
   knee is run fresh through ``legacy/TKA_Script_v3.py`` (unmodified, see
   :mod:`paper.legacy_runner`) and compared with the run archived beside the knee from
   the 2026-04-29 Paper 1 batch: chosen size, the bounding-box measurements, and the
   surface distance between the fresh and archived output meshes.
2. **Do the two pipelines size alike?** The legacy round-down chart size from its
   bounding box against the new planner's chart size from its anatomical frame.
3. **Where do they put things?** Component poses and the two cut planes, compared in
   the bone's own STL frame (the legacy world is mapped back through the transforms the
   driver recorded). Two new-planner variants: *like-for-like* -- the legacy size and
   the legacy tibial datum (chart depth below the top of the tibia) -- and the
   planner's *default* -- its own size and the commercial datum, 9 mm below the less
   affected plateau.

The legacy library is ``TKA_LEGACY_LIBRARY`` (+ ``TKA_LEGACY_SIZE_CHART``); the Blender
version ``TKA_LEGACY_BLENDER`` (default 5.1, what Paper 1 used). Fresh legacy runs are
cached under ``<out>/E12/legacy_runs/`` and reused while the script, library, Blender
and inputs are unchanged; set ``TKA_E12_RERUN=1`` to run them again.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..harness import Context, experiment, library_root, measured
from ..legacy_archive import paper1_archive
from ..legacy_runner import (
    BLENDER_DEFAULT,
    BLENDERS,
    FROZEN_SHA256,
    Library,
    back_map,
    check_library,
    default_library,
    output_dir,
    run_knee,
    sha256_bytes,
)
from ..pose_compare import plane_difference, pose_difference

EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "legacy_blend_export.py"

# What the legacy script exports, by role, and the scene object each file came from.
PARTS = {
    "femoral_component": ("Implant_Femoral_{Side}_{size}.stl", "Implant_Femoral_{Side}_{size}"),
    "femur_resected": ("Femur_{Side}_{size}_result.stl", "Femur"),
    "femur_shell": ("Femur_{Side}_{size}_Shell.stl", "Femur.Shell"),
    "tibial_plate": ("Tibial_Plate_{size}.stl", "Tibial_Plate_{size}"),
    "tibial_insert": ("Tibial_Insert_{size}.stl", "Tibial_Insert_{size}"),
    "tibia_resected": ("Tibia_{Side}_{size}_result.stl", "Tibia"),
    "tibia_shell": ("Tibia_{Side}_{size}_Shell.stl", "Tibia.Shell"),
}


def _stl_triangles(path: Path) -> int:
    return max(0, (path.stat().st_size - 84) // 50)


def _cached(case, work: Path, *, library: Library, blender: str) -> dict | None:
    path = work / case.case_id / "legacy_result.json"
    if os.environ.get("TKA_E12_RERUN") or not path.is_file():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    same = (r.get("blender_requested") == blender
            and r.get("script", {}).get("frozen_sha256") == FROZEN_SHA256
            and r.get("library", {}).get("stl_dir") == str(library.stl_dir)
            and r.get("library", {}).get("size_chart") == str(library.chart)
            and r.get("inputs", {}).get("femur_sha256") == sha256_bytes(case.femur.read_bytes())
            and not r.get("plain"))
    return r if same else None


def _legacy_flags(r: dict, out: Path) -> list[str]:
    """Everything in a legacy run a user would have had to notice and correct."""
    log = r.get("log") or {}
    flags = []
    if not r.get("ok"):
        flags.append(f"FAILED: {r.get('error') or 'did not complete'}")
    if log.get("out_of_range"):
        flags.append("size clamped to the chart limit")
    if log.get("cross_check_size") and log.get("cross_check_size") != log.get("size"):
        flags.append(f"femoral AP suggests {log['cross_check_size']}")
    if log.get("tibia_size") and log.get("tibia_size") != log.get("size"):
        flags.append(f"tibia alone sizes {log['tibia_size']}")
    # The script trims each bone to 150 mm with a boolean. A longer segment afterwards
    # means the trim failed and every bounding-box measurement after it is wrong; a
    # shorter one is only a short scan.
    if log.get("femur_height_mm") is not None and log["femur_height_mm"] > 150.5:
        flags.append(f"150 mm trim failed: femur segment {log['femur_height_mm']:.1f} mm, "
                     f"so the sizing bounding box is not the distal femur")
    if log.get("boolean_exact_failed") or log.get("boolean_failed"):
        flags.append(f"boolean fallback {log.get('boolean_exact_failed')}, "
                     f"failed {log.get('boolean_failed')}")
    if log.get("upside_down"):
        flags.append("orientation warning")
    for name in r.get("outputs", []):
        p = out / name
        # Only the bone outputs: a cutting block is legitimately a few hundred triangles.
        if name.startswith(("Femur_", "Tibia_")) and p.is_file() and _stl_triangles(p) < 1000:
            flags.append(f"{name} nearly empty ({_stl_triangles(p)} triangles)")
    return flags


def _archive_meshes(case, run: dict, side_cap: str, scratch: Path, blender_exe: str) -> dict:
    """The archived run's output files by role: its STLs, or exported from its scene."""
    size = run["log"]["size"]
    found, missing = {}, {}
    for role, (filename, obj) in PARTS.items():
        name = filename.format(Side=side_cap, size=size)
        if name in run["stls"]:
            found[role] = (run["stls"][name], "stl")
        else:
            missing[role] = (name, obj.format(Side=side_cap, size=size))
    if missing:
        stamp = run["timestamp"]
        blends = [p for p in sorted(run["folder"].glob(f"tka_*_{stamp}.blend*"))]
        # The scene saved at the end of the run. A later re-save leaves the original in
        # the .blend1 backup, so take whichever copy was written as the run ended.
        candidates = [p for p in blends if abs(p.stat().st_mtime - run["log_end"]) <= 5]
        pick = candidates[0] if candidates else None
        if pick is not None:
            target = scratch / case.case_id / "archive_scene"
            command = [blender_exe, "-b", "--factory-startup", str(pick), "--python",
                       str(EXPORT_SCRIPT), "--", str(target)]
            command += [f"{obj}={name}" for name, obj in missing.values()]
            subprocess.run(command, capture_output=True, text=True, timeout=600)
            for role, (name, _) in missing.items():
                if (target / name).is_file():
                    found[role] = (target / name, f"scene {pick.suffix}")
    return found


@experiment("E12", "Paper 1 pipeline vs the current planner on identical inputs")
def run(ctx: Context) -> None:
    from tka_planner.core.meshio import read_stl
    from tka_planner.pipeline import discrete_sizing, plan_case

    from ..meshdist import surface_distances

    library = default_library()
    blender = os.environ.get("TKA_LEGACY_BLENDER", BLENDER_DEFAULT)
    blender_exe = BLENDERS[blender]
    catalogue = library_root()
    work = ctx.path("legacy_runs")
    ctx.note(f"Legacy script sha256 {FROZEN_SHA256[:16]}..., Blender {blender}, "
             f"library {'TKA_LEGACY_LIBRARY' if os.environ.get('TKA_LEGACY_LIBRARY') else 'catalogue'}"
             f" (size chart sha256 {sha256_bytes(library.chart.read_bytes())[:16]}...).\n")

    run_rows, size_rows, pose_rows, plane_rows, repro_rows = [], [], [], [], []
    for case in ctx.cases:
        side_cap = "Left" if case.side == "left" else "Right"
        r = _cached(case, work, library=library, blender=blender) or run_knee(
            case, out_dir=work, blender=blender, library=library)
        out = output_dir(r, work)
        if r.get("ok") and "poses" in r and any(
                "cut_offset_mm" not in v for v in r.get("back_map", {}).values()):
            # A run cached before the cut-face measurement existed: measure it now.
            r["back_map"] = back_map(r, case, out)
            (work / case.case_id / "legacy_result.json").write_text(
                json.dumps(r, indent=1, default=str), encoding="utf-8")
        log = r.get("log") or {}
        size = log.get("size")
        flags = _legacy_flags(r, out)

        cat = check_library(catalogue, catalogue / "SizeChart.csv", case.side) if catalogue else {}
        cat_missing = (cat.get("missing_files", {}).get(size) if size else None)
        back = r.get("back_map", {})
        run_rows.append({
            "case": case.case_id, "legacy_ok": r.get("ok"),
            "blender": r.get("blender_version"), "seconds": r.get("seconds"),
            "size": size,
            "femur_on_input": back.get("femur", {}).get("fraction_on_input_vertex"),
            "femur_max_um": back.get("femur", {}).get("max_matched_um"),
            "tibia_on_input": back.get("tibia", {}).get("fraction_on_input_vertex"),
            "tibia_max_um": back.get("tibia", {}).get("max_matched_um"),
            "femoral_cut_above_origin_mm": back.get("femur", {}).get("cut_offset_mm"),
            "tibial_cut_above_origin_mm": back.get("tibia", {}).get("cut_offset_mm"),
            "runs_on_catalogue": (None if not size else not cat_missing),
            "catalogue_missing": "; ".join(cat_missing or []),
            "flags": "; ".join(flags),
        })

        # ---- Z-4: does it reproduce the archived Paper 1 run? ----------------
        archive = paper1_archive(case)
        repro = {"case": case.case_id, "archive": archive["id"] if archive else None,
                 "fresh_size": size}
        if archive:
            alog = archive["log"]
            repro.update(
                archive_size=alog["size"], size_same=alog["size"] == size,
                d_femur_ml_mm=(log.get("femur_ml_mm") or np.nan) - (alog["femur_ml_mm"] or np.nan),
                d_femur_ap_mm=(log.get("femur_ap_mm") or np.nan) - (alog["femur_ap_mm"] or np.nan),
                d_tibia_ml_mm=(log.get("tibia_ml_mm") or np.nan) - (alog["tibia_ml_mm"] or np.nan),
                archive_femur_height_mm=alog["femur_height_mm"],
                fresh_femur_height_mm=log.get("femur_height_mm"),
            )
            archive["log_end"] = (archive["folder"] / Path(archive["id"]).name).stat().st_mtime
            meshes = _archive_meshes(case, archive, side_cap, work, blender_exe) if r.get("ok") else {}
            for role, (filename, _) in PARTS.items():
                fresh = out / filename.format(Side=side_cap, size=size)
                if role not in meshes or not fresh.is_file():
                    repro[f"{role}_source"] = "missing"
                    continue
                path, source = meshes[role]
                repro[f"{role}_source"] = source
                if sha256_bytes(path.read_bytes()) == sha256_bytes(fresh.read_bytes()):
                    repro[f"{role}_identical"] = True
                    repro[f"{role}_hausdorff_mm"] = 0.0
                    repro[f"{role}_mean_mm"] = 0.0
                    continue
                # The legacy world is in metres.
                a, b = (SimpleNamespace(vertices=mesh.vertices * 1000.0, faces=mesh.faces)
                        for mesh in (read_stl(path, compute_hash=False),
                                     read_stl(fresh, compute_hash=False)))
                d = surface_distances(a, b, sample=4000 if ctx.quick else 20000,
                                      seed=ctx.seed)
                repro[f"{role}_identical"] = False
                repro[f"{role}_hausdorff_mm"] = d["hausdorff"]
                repro[f"{role}_mean_mm"] = d["mean"]
                repro[f"{role}_triangles_archive_vs_fresh"] = (
                    f"{len(a.faces)} / {len(b.faces)}")
        repro_rows.append(repro)

        # ---- The new planner ---------------------------------------------------
        m = measured(case)
        if isinstance(m, Exception):
            size_rows.append({"case": case.case_id, "legacy_size": size,
                              "new_planner": f"refused ({type(m).__name__})"})
            continue
        discrete = discrete_sizing(m)
        lfl_plan, lfl_sizing = plan_case(m, size_label=size, tibial_reference="top_of_tibia")
        def_plan, def_sizing = plan_case(m)
        calls = r.get("calls") or {}
        quiet = {(c["bone"], c["mode"]): c for c in calls.get("select_size_quiet", [])}
        primary = (calls.get("select_size") or [{}])[0]
        size_rows.append({
            "case": case.case_id,
            "legacy_size": size,
            "legacy_femur_ml_mm": primary.get("measurement_mm"),
            "legacy_femur_ap_mm": quiet.get(("femur", "AP"), {}).get("measurement_mm"),
            "legacy_tibia_ml_mm": quiet.get(("tibia", "ML"), {}).get("measurement_mm"),
            "new_femur_ml_mm": m.femoral_measure.ml_mm,
            "new_femur_ap_mm": m.femoral_measure.ap_mm,
            "new_discrete_size": discrete.nearest_discrete_size,
            "new_parametric_size": def_sizing.nearest_discrete_size,
            "new_parametric_parameter": def_sizing.size_parameter,
            "discrete_agrees": discrete.nearest_discrete_size == size,
            "parametric_agrees": def_sizing.nearest_discrete_size == size,
            "new_planner": "accepted",
        })
        if not r.get("ok") or "poses" not in r:
            continue

        legacy_poses = {k: np.array(v) for k, v in r["poses"]["components"].items()}
        frames = {"femoral": m.femoral_frame, "tibial": m.tibial_frame}
        for variant, plan in (("like_for_like", lfl_plan), ("default", def_plan)):
            for part, frame_key in (("femoral_component", "femoral"),
                                    ("tibial_component", "tibial")):
                f = frames[frame_key]
                d = pose_difference(legacy_poses[part], plan.components[part],
                                    anterior=f.x_anterior, lateral=f.lateral,
                                    proximal=f.z_proximal)
                pose_rows.append({"case": case.case_id, "variant": variant,
                                  "component": part, **d})
            for plane, part, frame_key in (("femoral_distal", "femoral_component", "femoral"),
                                           ("tibial_proximal", "tibial_component", "tibial")):
                f = frames[frame_key]
                res = plan.resections[plane]
                lp = legacy_poses[part]
                # The legacy cut is the face its cutting block actually left, which in
                # the Paper 1 library sits a fraction of a millimetre off the origin.
                offset = (r.get("back_map", {}).get("femur" if frame_key == "femoral"
                                                    else "tibia", {}).get("cut_offset_mm")
                          or 0.0)
                d = plane_difference(lp[:3, 3] + offset * lp[:3, 2], lp[:3, 2],
                                     res.point, res.normal,
                                     centre=f.origin, proximal=f.z_proximal)
                plane_rows.append({"case": case.case_id, "variant": variant,
                                   "plane": plane, **d,
                                   "legacy_cut_offset_from_origin_mm": offset,
                                   "new_valgus_cut_deg": (plan.distal_femoral_valgus_cut_deg
                                                          if frame_key == "femoral" else None),
                                   "new_reference": res.reference})

    ctx.table("legacy_runs", run_rows, digits=4)
    ctx.table("reproduction", repro_rows, digits=3)
    ctx.table("sizes", size_rows, digits=1)
    ctx.table("poses", pose_rows, digits=2)
    ctx.table("planes", plane_rows, digits=2)
    _summarise(ctx, run_rows, repro_rows, size_rows, pose_rows, plane_rows)


def _summarise(ctx, run_rows, repro_rows, size_rows, pose_rows, plane_rows) -> None:
    ok = [r for r in run_rows if r["legacy_ok"]]
    ctx.note(f"Legacy pipeline completed on {len(ok)} of {len(run_rows)} knees.")
    for r in run_rows:
        if r["flags"]:
            ctx.note(f"- {r['case']}: {r['flags']}")
    on_cat = [r["case"] for r in run_rows if r["runs_on_catalogue"] is False]
    if on_cat:
        ctx.note(f"With data/implants as its library the script would stop on: "
                 f"{', '.join(on_cat)} (missing files for the chosen size).")
    with_archive = [r for r in repro_rows if r.get("archive")]
    same = sum(bool(r.get("size_same")) for r in with_archive)
    ctx.note(f"\nReproduction (Z-4): {len(with_archive)} knees have a Paper 1 archive; "
             f"fresh size equals archived size on {same}.")
    for r in with_archive:
        parts = [f"{k[:-len('_hausdorff_mm')]} {r[k]:.3f}" for k in r
                 if k.endswith("_hausdorff_mm")
                 and r[k] is not None]
        ctx.note(f"- {r['case']} vs {r['archive']}: size {r['archive_size']} -> "
                 f"{r['fresh_size']}; Hausdorff mm: {', '.join(parts) or 'no meshes'}")
    accepted = [r for r in size_rows if r.get("new_planner") == "accepted"]
    if accepted:
        ctx.note(f"\nSize agreement with the legacy pipeline: discrete "
                 f"{sum(r['discrete_agrees'] for r in accepted)}/{len(accepted)}, "
                 f"parametric nearest {sum(r['parametric_agrees'] for r in accepted)}/"
                 f"{len(accepted)}.")
    for variant in ("like_for_like", "default"):
        for comp in ("femoral_component", "tibial_component"):
            rows = [r for r in pose_rows if r["variant"] == variant and r["component"] == comp]
            if rows:
                pos = np.array([r["position_mm"] for r in rows])
                rot = np.array([r["rotation_deg"] for r in rows])
                ctx.note(f"{variant} {comp}: position diff median {np.median(pos):.1f} mm "
                         f"(range {pos.min():.1f}-{pos.max():.1f}), rotation median "
                         f"{np.median(rot):.1f} deg (range {rot.min():.1f}-{rot.max():.1f}), "
                         f"n={len(rows)}")
        for plane in ("femoral_distal", "tibial_proximal"):
            rows = [r for r in plane_rows if r["variant"] == variant and r["plane"] == plane]
            if rows:
                ang = np.array([r["angle_deg"] for r in rows])
                off = np.array([r["offset_at_centre_mm"] for r in rows])
                ctx.note(f"{variant} {plane}: angle median {np.median(ang):.1f} deg "
                         f"(range {ang.min():.1f}-{ang.max():.1f}), offset at centre median "
                         f"{np.median(off):+.1f} mm (range {off.min():+.1f} to {off.max():+.1f})")
