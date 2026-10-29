"""E8: the geometry kernel -- does it cut what Blender cuts, what does it repair, and
what does it refuse?

The planner cuts bone with manifold3d. Paper 1's results were cut with Blender's exact
solver, and the port rests on the two agreeing. ``tests/test_kernel_agreement.py``
gates that on four small cases against Blender results committed as STL; this turns
the gate into a reported result and carries it to the cohort's real resections.

Four parts, each its own table:

* ``reference``: the four gate cases, manifold3d against the committed Blender STL.
* ``regenerated``: the four cases re-cut by every Blender found on this machine, to
  show whether the committed reference depends on the Blender version that made it.
* ``cohort_agreement``: each knee's three commit booleans per bone (block-mode shell
  and block, and plane mode's discard box), cut by both solvers from identical input,
  compared on volume, area, bounding box and symmetric Hausdorff distance.
* ``repair`` and ``refusals``: what the kernel's repair does to every cohort bone and
  tool, and which defects it repairs, which it refuses, and what Blender does with the
  same input.

Blender runs in its own process through ``paper/blender_worker.py`` and never in this
one (it is GPL; see the agreement test). Without a Blender installation the Blender
columns are left empty and said to be. Set ``TKA_BLENDER`` to choose the executable;
by default the newest under ``C:/Program Files/Blender Foundation`` is used.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tka_planner.core.meshio import Mesh, read_stl, weld_vertices
from tka_planner.core.qc import assess_mesh_quality
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel
from tka_planner.geom.repair import edge_report, is_manifold, repair, repair_statistics

from ..harness import REPO, Context, experiment, library_root, measured
from ..planned import BONES, bone_mesh, commit_tools, planned_session
from ..surface_distance import hausdorff

WORKER = REPO / "paper" / "blender_worker.py"
REFERENCE = REPO / "tests" / "fixtures" / "kernel_reference"
BLENDER_ROOTS = (Path("C:/Program Files/Blender Foundation"),)
# The agreement gate's own tolerance, for the same verdict on every row.
TOLERANCE_MM = 1e-4
BLENDER_TIMEOUT_S = 3600


# ----------------------------------------------------------------------
# Blender, at arm's length
# ----------------------------------------------------------------------


def blender_executables() -> dict:
    """Every Blender found, by version label, newest last; ``TKA_BLENDER`` first."""
    found = {}
    for root in BLENDER_ROOTS:
        for exe in sorted(root.glob("Blender */blender.exe"), key=_version_key):
            found[exe.parent.name] = exe
    on_path = shutil.which("blender")
    if on_path:
        found.setdefault("blender on PATH", Path(on_path))
    chosen = os.environ.get("TKA_BLENDER")
    if chosen:
        found = {"TKA_BLENDER": Path(chosen), **found}
    return found


def _version_key(path: Path):
    return tuple(int(n) for n in re.findall(r"\d+", path.parent.name)) or (0,)


def primary_blender(found: dict) -> Path | None:
    if "TKA_BLENDER" in found:
        return found["TKA_BLENDER"]
    return list(found.values())[-1] if found else None


def run_blender(exe: Path, jobs: list[dict], folder: Path) -> dict:
    """Run ``jobs`` through the worker in one Blender process; its result record."""
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(jobs), encoding="utf-8")
    t0 = time.perf_counter()
    done = subprocess.run(
        [str(exe), "-b", "--factory-startup", "--python", str(WORKER), "--",
         str(manifest)],
        capture_output=True, text=True, timeout=BLENDER_TIMEOUT_S)
    result_path = manifest.with_suffix(".result.json")
    if not result_path.is_file():
        tail = (done.stderr or done.stdout or "")[-400:]
        raise RuntimeError(f"Blender produced no result (exit {done.returncode}): {tail}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["wall_seconds"] = time.perf_counter() - t0
    return result


def _save(mesh: Mesh, path: Path) -> str:
    np.savez(path, vertices=np.asarray(mesh.vertices, dtype=np.float64),
             faces=np.asarray(mesh.faces, dtype=np.int64))
    return str(path)


def _load(path) -> Mesh:
    with np.load(path) as data:
        return Mesh(vertices=data["vertices"], faces=data["faces"])


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def compare(ours: Mesh, theirs: Mesh) -> dict:
    """manifold3d's result against Blender's, measured on the solid."""
    our_low, our_high = ours.bounds
    their_low, their_high = theirs.bounds
    v_ours, v_theirs = gm.volume(ours), gm.volume(theirs)
    a_ours, a_theirs = gm.surface_area(ours), gm.surface_area(theirs)
    distance = hausdorff(ours, theirs)
    # Blender holds coordinates in single precision, so a vertex it creates can sit up
    # to half a float32 spacing from the exact point. Scanner coordinates run to over a
    # metre, where that spacing is about 1e-4 mm, so the gate's tolerance is reported
    # beside the spacing at the mesh's largest coordinate rather than applied blindly.
    spacing = float(np.spacing(np.float32(np.abs(ours.vertices).max())))
    return {
        "triangles_manifold": int(ours.n_triangles),
        "triangles_blender": int(theirs.n_triangles),
        "volume_mm3": v_ours,
        "volume_rel_diff": (v_ours - v_theirs) / v_theirs if v_theirs else None,
        "area_mm2": a_ours,
        "area_rel_diff": (a_ours - a_theirs) / a_theirs if a_theirs else None,
        "bbox_max_diff_mm": float(max(np.abs(our_low - their_low).max(),
                                      np.abs(our_high - their_high).max())),
        "hausdorff_mm": distance["hausdorff_mm"],
        "hausdorff_p95_mm": max(distance["a_to_b_p95_mm"], distance["b_to_a_p95_mm"]),
        "samples_measured": distance["samples_measured"],
        "max_coordinate_mm": float(np.abs(ours.vertices).max()),
        "float32_spacing_mm": spacing,
        "hausdorff_in_spacings": distance["hausdorff_mm"] / spacing,
        # Welded first: a mesh read back from STL is a triangle soup.
        "blender_output_manifold": is_manifold(weld_vertices(theirs)),
        "manifold_output_manifold": is_manifold(weld_vertices(ours)),
        "agree": distance["hausdorff_mm"] < TOLERANCE_MM,
    }


def float32_agreement(operation: str, target: Mesh, tool: Mesh, theirs: Mesh) -> float:
    """Hausdorff from Blender's result to manifold3d's on the same input rounded to
    single precision, as Blender stores it.

    Separates the two possible causes of a disagreement: if this is much smaller than
    the plain comparison, the difference came from Blender rounding its input, not
    from either solver.
    """
    def rounded(mesh):
        return Mesh(vertices=mesh.vertices.astype(np.float32).astype(np.float64),
                    faces=mesh.faces)

    ours = getattr(ManifoldKernel(), operation)(rounded(target), rounded(tool)).mesh
    return hausdorff(ours, theirs)["hausdorff_mm"]


def _prepared(mesh: Mesh) -> Mesh:
    """The mesh exactly as manifold3d is handed it, so Blender is handed the same."""
    return mesh if is_manifold(mesh) else repair(mesh)[0]


def kernel_verdict(mesh: Mesh) -> tuple[bool, str]:
    """Whether manifold3d's kernel accepts the mesh as a boolean operand, and why not."""
    try:
        clean, _ = ManifoldKernel._prepare(mesh, "target")
        ManifoldKernel._to_manifold(clean, "target")
    except ValueError as refusal:
        return False, str(refusal).split(". ")[0]
    return True, ""


# ----------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------


def reference_cases(ctx: Context, blenders: dict) -> list[dict]:
    from tests.kernel_reference import cases
    from tests.surface_distance import hausdorff as gate_hausdorff

    rows, regenerated = [], []
    kernel = ManifoldKernel()
    gate = list(cases())
    for name, operation, target, tool in gate:
        ours = getattr(kernel, operation)(target, tool).mesh
        committed = read_stl(REFERENCE / f"{name}.stl")
        rows.append({"case": name, "operation": operation, **compare(ours, committed),
                     "gate_hausdorff_mm": gate_hausdorff(ours, committed)})
    ctx.table("reference", rows, digits=9)

    with tempfile.TemporaryDirectory(prefix="tka_e8_ref_") as tmp:
        folder = Path(tmp)
        jobs = []
        for name, operation, target, tool in gate:
            jobs.append({"name": name, "operation": operation,
                         "target": _save(target, folder / f"{name}_target.npz"),
                         "tool": _save(tool, folder / f"{name}_tool.npz"),
                         "out": str(folder / f"{name}_out.npz")})
        for label, exe in blenders.items():
            try:
                result = run_blender(exe, jobs, folder)
            except Exception as error:  # one missing Blender must not stop the rest
                regenerated.append({"blender": label, "error": str(error)[:200]})
                continue
            for (name, operation, target, tool), entry in zip(gate, result["jobs"]):
                row = {"blender": result["blender"], "case": name}
                if "error" in entry:
                    regenerated.append({**row, "error": entry["error"]})
                    continue
                theirs = _load(folder / f"{name}_out.npz")
                written = gm.write_stl(theirs, folder / f"{name}.stl")
                ours = getattr(kernel, operation)(target, tool).mesh
                regenerated.append({
                    **row,
                    "stl_byte_identical": written.read_bytes()
                    == (REFERENCE / f"{name}.stl").read_bytes(),
                    "blender_s": entry["seconds"],
                    # In double precision, before STL rounds Blender's output.
                    **compare(ours, theirs),
                    "hausdorff_float32_inputs_mm": float32_agreement(
                        operation, target, tool, theirs),
                })
    ctx.table("regenerated", regenerated, digits=9)
    return rows + regenerated


def refusal_catalogue(blender: Path | None) -> list[dict]:
    """Known defects through both solvers: what manifold3d repairs, what it refuses,
    and what Blender returns for the same input."""
    cube = gm.box(20.0)
    expected = 20.0 ** 3 - 10.0 * 20.0 * 20.0  # the tool removes the top half
    tool = gm.box(40.0, gm.translation((0.0, 0.0, 20.0)))

    def soup(mesh):
        return Mesh(vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
                    faces=np.arange(mesh.n_triangles * 3).reshape(-1, 3))

    def joined(a, b):
        return Mesh(vertices=np.vstack([a.vertices, b.vertices]),
                    faces=np.vstack([a.faces, b.faces + a.n_vertices]))

    flipped_one = cube.faces.copy()
    flipped_one[0] = flipped_one[0][::-1]
    # Two cubes meeting along one edge share it four ways once welded.
    edge_pair = joined(cube, gm.box(20.0, gm.translation((20.0, 20.0, 0.0))))
    defects = [
        ("clean cube", cube, expected),
        ("triangle soup (unwelded STL)", soup(cube), expected),
        ("degenerate face", Mesh(vertices=cube.vertices,
                                 faces=np.vstack([cube.faces, [[0, 0, 1]]])), expected),
        ("duplicated face", Mesh(vertices=cube.vertices,
                                 faces=np.vstack([cube.faces, cube.faces[:1]])), expected),
        ("hole (one face missing)", Mesh(vertices=cube.vertices, faces=cube.faces[1:]),
         None),
        ("open surface (a disc)", gm.disc(10.0, segments=16), None),
        ("non-manifold edge (cubes on an edge)", soup(edge_pair), None),
        ("one face flipped", Mesh(vertices=cube.vertices, faces=flipped_one), None),
        ("inside out (every face flipped)",
         Mesh(vertices=cube.vertices, faces=cube.faces[:, ::-1]), None),
        ("two overlapping cubes in one mesh",
         joined(cube, gm.box(20.0, gm.translation((10.0, 0.0, 0.0)))), None),
        # The tool reaches only the first cube.
        ("two separate cubes", joined(cube, gm.box(20.0, gm.translation((50.0, 0, 0)))),
         expected + 20.0 ** 3),
    ]
    rows = []
    kernel = ManifoldKernel()
    for name, mesh, volume in defects:
        report = edge_report(mesh)
        stats = repair_statistics(mesh)
        row = {"defect": name, "boundary_edges": report["boundary_edges"],
               "nonmanifold_edges": report["nonmanifold_edges"],
               "repair": "; ".join(stats["actions"]) or "none",
               "components": stats["components"], "expected_mm3": volume}
        try:
            result = kernel.difference(mesh, tool).mesh
            row.update(manifold3d="accepted", manifold3d_mm3=gm.volume(result))
        except ValueError as refusal:
            row.update(manifold3d="refused", reason=str(refusal).split(". ")[0][:140])
        rows.append(row)

    if blender is None:
        return rows
    with tempfile.TemporaryDirectory(prefix="tka_e8_defects_") as tmp:
        folder = Path(tmp)
        jobs = [{"name": str(i), "operation": "difference",
                 "target": _save(mesh, folder / f"{i}_target.npz"),
                 "tool": _save(tool, folder / f"{i}_tool.npz"),
                 "out": str(folder / f"{i}_out.npz")}
                for i, (_, mesh, _) in enumerate(defects)]
        try:
            result = run_blender(blender, jobs, folder)
        except Exception as error:
            for row in rows:
                row["blender"] = f"not run: {str(error)[:80]}"
            return rows
        for row, entry in zip(rows, result["jobs"]):
            if "error" in entry:
                row["blender"] = f"error: {entry['error'][:80]}"
                continue
            out = _load(folder / f"{entry['name']}_out.npz")
            row["blender"] = "returned a result"
            row["blender_mm3"] = gm.volume(out) if out.n_triangles else 0.0
            row["blender_output_manifold"] = is_manifold(weld_vertices(out))
    return rows


def bone_rows(case_id: str, bone: str, mesh: Mesh) -> dict:
    """Repair statistics and quality control for one raw bone mesh."""
    stats = repair_statistics(mesh)
    quality, _ = assess_mesh_quality(mesh)
    accepted, reason = kernel_verdict(mesh)
    stats.pop("actions")
    return {"case": case_id, "mesh": bone, **stats,
            "qc_watertight": quality.watertight,
            "kernel_accepts": accepted, "kernel_reason": reason}


def cohort_agreement(case, session, blender: Path | None,
                     operations: tuple) -> list[dict]:
    """Each commit boolean for one knee, cut by both solvers from identical input."""
    kernel = ManifoldKernel()
    rows, ours, jobs = [], {}, []
    with tempfile.TemporaryDirectory(prefix="tka_e8_cut_") as tmp:
        folder = Path(tmp)
        for bone in BONES:
            target = _prepared(bone_mesh(session, bone))
            target_file = _save(target, folder / f"{bone}_target.npz")
            for name, (operation, tool) in commit_tools(session, bone).items():
                if name not in operations:
                    continue
                key = f"{bone}_{name}"
                tool = _prepared(tool)
                row = {"case": case.case_id, "bone": bone, "boolean": name,
                       "operation": operation, "triangles_in": int(target.n_triangles)}
                t0 = time.perf_counter()
                try:
                    ours[key] = getattr(kernel, operation)(target, tool).mesh
                    row["manifold_s"] = time.perf_counter() - t0
                except ValueError as refusal:
                    row["manifold_refused"] = str(refusal)[:140]
                rows.append(row)
                jobs.append({"name": key, "operation": operation, "target": target_file,
                             "tool": _save(tool, folder / f"{key}_tool.npz"),
                             "out": str(folder / f"{key}_out.npz")})
        if blender is None:
            return rows
        try:
            result = run_blender(blender, jobs, folder)
        except Exception as error:
            for row in rows:
                row["blender_error"] = str(error)[:140]
            return rows
        for row, job, entry in zip(rows, jobs, result["jobs"]):
            key = entry["name"]
            if "error" in entry:
                row["blender_error"] = entry["error"][:140]
                continue
            row["blender_s"] = entry["seconds"]
            row["blender"] = result["blender"]
            if key in ours:
                theirs = _load(folder / f"{key}_out.npz")
                row.update(compare(ours.pop(key), theirs))
                row["hausdorff_float32_inputs_mm"] = float32_agreement(
                    job["operation"], _load(job["target"]), _load(job["tool"]), theirs)
    return rows


@experiment("E8", "Geometry kernel: cross-solver agreement, repair, refusals",
            needs_cohort="optional")
def run(ctx: Context) -> None:
    blenders = blender_executables()
    blender = primary_blender(blenders)
    ctx.record["blenders"] = {label: str(path) for label, path in blenders.items()}
    if blender is None:
        ctx.note("No Blender found: the Blender columns are empty. Set TKA_BLENDER.")

    reference = reference_cases(ctx, blenders)
    gate = [r for r in reference if "blender" not in r]
    ctx.note(f"Reference cases: manifold3d against the committed Blender STL, worst "
             f"Hausdorff {max(r['hausdorff_mm'] for r in gate):.2e} mm, worst volume "
             f"difference {max(abs(r['volume_rel_diff']) for r in gate):.2e} (relative).")
    regenerated = [r for r in reference if "blender" in r]
    for version in sorted({r["blender"] for r in regenerated}):
        mine = [r for r in regenerated if r["blender"] == version]
        identical = sum(bool(r.get("stl_byte_identical")) for r in mine)
        worst = max((r.get("hausdorff_mm", float("nan")) for r in mine), default=None)
        ctx.note(f"- Blender {version}: {identical} of {len(mine)} references regenerate "
                 f"byte-identical; worst double-precision Hausdorff to manifold3d "
                 f"{worst:.2e} mm.")

    refusals = refusal_catalogue(blender)
    ctx.table("refusals", refusals, digits=3)
    refused = [r["defect"] for r in refusals if r["manifold3d"] == "refused"]
    ctx.note(f"Defect catalogue: manifold3d refuses {len(refused)} of {len(refusals)} "
             f"({', '.join(refused)}); Blender returned a result for "
             f"{sum(r.get('blender') == 'returned a result' for r in refusals)}.")
    for r in refusals:
        expected, theirs = r.get("expected_mm3"), r.get("blender_mm3")
        wrong = expected is not None and theirs is not None and abs(theirs - expected) > 1e-6
        if r["manifold3d"] == "refused" or wrong or expected is None:
            ctx.note(f"- {r['defect']}: manifold3d {r['manifold3d']}"
                     f"{'' if r.get('manifold3d_mm3') is None else ' (%.1f mm3)' % r['manifold3d_mm3']}"
                     f"; Blender {'%.1f mm3' % theirs if theirs is not None else r.get('blender')}"
                     f"{', expected %.1f' % expected if expected is not None else ', no defined answer'}"
                     f"{', Blender output not manifold' if r.get('blender_output_manifold') is False else ''}.")

    if not ctx.cases:
        ctx.note("No cohort: the repair and agreement tables need it.")
        return

    library = library_root()
    # The quick run cuts the block only; the full run all three commit booleans.
    operations = ("block",) if ctx.quick else ("shell", "block", "plane")
    repair_rows, agreement = [], []
    for case in ctx.cases:
        m = measured(case)
        planned = not isinstance(m, Exception)
        femur = m.femur if planned else read_stl(case.femur)
        tibia = m.tibia if planned else read_stl(case.tibia)
        repair_rows.append(bone_rows(case.case_id, "Femur", femur))
        repair_rows.append(bone_rows(case.case_id, "Tibia", tibia))
        del femur, tibia
        if not planned:
            ctx.note(f"- {case.case_id}: planner refused ({type(m).__name__}); repair "
                     f"statistics only, no resection to compare.")
            continue
        session = planned_session(case, m, library)
        for bone in BONES:
            for name, (_, tool) in commit_tools(session, bone).items():
                if name != "plane":   # the discard box is a clean box by construction
                    repair_rows.append(bone_rows(case.case_id, f"{bone} {name} tool",
                                                 tool))
        print(f"   {case.case_id}: cutting with both solvers", flush=True)
        agreement += cohort_agreement(case, session, blender, operations)
        del session
        # Written after every knee, so a long run that stops still leaves its rows.
        ctx.table("cohort_agreement", agreement, digits=9)
    ctx.table("repair", repair_rows, digits=0)
    ctx.table("cohort_agreement", agreement, digits=9)

    bones = [r for r in repair_rows if r["mesh"] in BONES]
    ctx.note(f"Cohort bones: {len(bones)}; kernel accepts "
             f"{sum(r['kernel_accepts'] for r in bones)} after repair; vertices merged "
             f"per bone {min(r['vertices_merged'] for r in bones)}-"
             f"{max(r['vertices_merged'] for r in bones)}; degenerate faces removed "
             f"{sum(r['degenerate_faces_removed'] for r in bones)}, duplicates "
             f"{sum(r['duplicate_faces_removed'] for r in bones)}; components per bone "
             f"{min(r['components'] for r in bones)}-{max(r['components'] for r in bones)}.")
    for r in bones:
        if not r["kernel_accepts"]:
            ctx.note(f"- {r['case']} {r['mesh']} refused by the kernel: {r['kernel_reason']}")
    compared = [r for r in agreement if "hausdorff_mm" in r]
    if compared:
        ctx.note(f"Cohort resections compared: {len(compared)}; worst Hausdorff "
                 f"{max(r['hausdorff_mm'] for r in compared):.2e} mm, worst |volume "
                 f"difference| {max(abs(r['volume_rel_diff']) for r in compared):.2e}, "
                 f"worst |area difference| "
                 f"{max(abs(r['area_rel_diff']) for r in compared):.2e} (relative); "
                 f"{sum(r['agree'] for r in compared)} within {TOLERANCE_MM} mm; worst "
                 f"Hausdorff {max(r['hausdorff_in_spacings'] for r in compared):.2f} "
                 f"float32 spacings at the mesh's largest coordinate (Blender stores "
                 f"single precision). With manifold3d given the same input rounded to "
                 f"single precision, the worst Hausdorff is "
                 f"{max(r['hausdorff_float32_inputs_mm'] for r in compared):.2e} mm.")
    failed = [r for r in agreement if "blender_error" in r or "manifold_refused" in r]
    for r in failed:
        blender_did = (" Blender cut it regardless."
                       if "manifold_refused" in r and "blender_s" in r else "")
        ctx.note(f"- {r['case']} {r['bone']} {r['boolean']}: "
                 f"{(r.get('manifold_refused') or r.get('blender_error')).split('. ')[0]}."
                 f"{blender_did}")
