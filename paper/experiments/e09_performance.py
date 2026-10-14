"""E9: how long each stage takes, and whether the plan is the same every time.

**Performance.** Every stage of a knee, timed separately, from reading the STL to
posing one frame of the reduced knee, with the median of several repeats (five in a
full run, two in a quick one). The stages are the pipeline's own functions called in
the order ``measure_case`` and the application call them, so the sum is what a user
waits for; Commit goes through the session exactly as the application's button does,
with the kernel wrapped only to time the shell and the cut apart. Commit time is also
plotted against each bone's triangle count.

A timing is only a benchmark on a quiet machine. The run record carries the CPU's
name, the installed RAM, and how busy the CPU was in the second before the run and
the second after it, and the summary says when that was too busy to publish.

**Determinism.** The plan file's SHA-256 must not change: not between the repeats in
this process (each repeat reads the STL afresh), not in a second process
(``python -m paper.determinism``, run as a subprocess), and not on another operating
system. The last cannot be checked on one machine, so the hashes are written to
``determinism.csv`` for a later Linux run to compare, and the synthetic fixture's hashes
are compared with ``paper/determinism_synthetic.json``, which CI checks on both
operating systems.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tka_planner.core.frames import FrameConstructionError
from tka_planner.core.landmarks_auto import estimate_landmarks
from tka_planner.core.meshio import read_stl
from tka_planner.core.qc import (
    QCReport,
    QualityControlError,
    assess_femur_coverage,
    assess_mesh_quality,
    assess_tibia_coverage,
    check_landmarks_on_mesh,
    check_laterality,
    check_plausible_bone_scale,
)
from tka_planner.core.sides import Side
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel
from tka_planner.pipeline import (
    fit_case,
    implant_spec_case,
    measure_from_landmarks,
    plan_case,
    plan_document,
)

from .. import determinism
from ..harness import REPO, Context, cpu_busy_percent, experiment, library_root, memory_gb
from ..planned import planned_session

STAGES = ("read_stl", "qc", "landmarks", "frames_metrics", "plan", "implant_spec",
          "fit", "plan_file", "scene_build", "repair_femur", "commit_femur",
          "commit_femur_shell", "commit_femur_cut", "repair_tibia", "commit_tibia",
          "commit_tibia_shell", "commit_tibia_cut", "replan", "reduce_frame")
# Above this the machine was not quiet enough for a timing worth publishing.
BUSY_PERCENT = 10.0
FRAMES = 50


class TimedKernel:
    """manifold3d, with each boolean's wall time written down."""

    name = "manifold3d"

    def __init__(self):
        self.inner = ManifoldKernel()
        self.log: list = []

    def difference(self, target, tool):
        return self._timed("difference", target, tool)

    def intersect(self, target, tool):
        return self._timed("intersect", target, tool)

    def _timed(self, operation, target, tool):
        t0 = time.perf_counter()
        result = getattr(self.inner, operation)(target, tool)
        self.log.append((operation, time.perf_counter() - t0))
        return result


def _clock(times: dict, stage: str, function, *args, **kwargs):
    t0 = time.perf_counter()
    value = function(*args, **kwargs)
    times[stage] = time.perf_counter() - t0
    return value


def one_pass(case, library) -> tuple[dict, dict, str]:
    """Every stage once, from the STL files on. Returns the stage times, the hashes,
    and the refusal if the planner refused the knee."""
    times: dict = {}
    side = Side.parse(case.side)

    def read_both():
        return read_stl(case.femur), read_stl(case.tibia)

    femur, tibia = _clock(times, "read_stl", read_both)

    # Quality control exactly as ``measure_case`` runs it.
    def quality_control():
        qc = QCReport()
        for mesh, bone in ((femur, "femur"), (tibia, "tibia")):
            qc.extend(check_plausible_bone_scale(mesh, bone))
            qc.extend(check_laterality(mesh, side))
            qc.extend(assess_mesh_quality(mesh)[1])
        qc.raise_if_errors()
        femur_coverage, femur_report = assess_femur_coverage(femur)
        tibia_coverage, tibia_report = assess_tibia_coverage(tibia)
        qc.extend(femur_report)
        qc.extend(tibia_report)
        return qc, {"femur": femur_coverage, "tibia": tibia_coverage}

    try:
        qc, coverage = _clock(times, "qc", quality_control)
        landmarks = _clock(times, "landmarks", estimate_landmarks, femur, tibia,
                           side, case_id=case.case_id)
        qc.extend(check_landmarks_on_mesh(landmarks, {"femur": femur, "tibia": tibia}))
        qc.raise_if_errors()
        m = _clock(times, "frames_metrics", measure_from_landmarks, landmarks, femur,
                   tibia, femur_path=case.femur, tibia_path=case.tibia,
                   case_id=case.case_id, side=side, qc=qc, coverage=coverage)
    except (QualityControlError, FrameConstructionError) as refusal:
        return times, {}, type(refusal).__name__

    plan, sizing = _clock(times, "plan", plan_case, m, library=library)
    spec = _clock(times, "implant_spec", implant_spec_case, m, plan, sizing, library)
    fit = _clock(times, "fit", fit_case, m, plan, sizing, library,
                 determinism.CONTROLS["implant_mode"])

    def plan_file():
        return determinism.plan_file_bytes(plan_document(
            m, plan, sizing, fit=fit, implant_spec=spec,
            controls=dict(determinism.CONTROLS)))

    hashes = {"plan.json": determinism.sha256(_clock(times, "plan_file", plan_file))}

    session = _clock(times, "scene_build", planned_session, case, m, library)
    for bone in ("Femur", "Tibia"):
        # The repair every boolean of the commit runs on the uncut bone first: timed
        # alone, because it is paid once per boolean, twice per bone in block mode.
        try:
            _clock(times, f"repair_{bone.lower()}", ManifoldKernel._prepare,
                   session.scene.mesh_for(bone), "target")
            kernel = TimedKernel()
            _clock(times, f"commit_{bone.lower()}", session.commit, kernel,
                   bones=(bone,))
        except ValueError:
            # The kernel refused the bone (E8 says why); there is no commit to time.
            hashes[f"geometry/{bone}"] = "refused"
            continue
        for operation, seconds in kernel.log:
            part = "shell" if operation == "intersect" else "cut"
            times[f"commit_{bone.lower()}_{part}"] = seconds
        for name in (bone, f"{bone}.Shell"):
            mesh = session.scene.mesh_for(name)
            if mesh is not None:
                hashes[f"geometry/{name}"] = determinism.mesh_hash(mesh)

    # One control change in Plan mode, then back: what a slider drag costs.
    t0 = time.perf_counter()
    session.replan(tibial_slope_delta_deg=1.0)
    session.replan(tibial_slope_delta_deg=0.0)
    times["replan"] = (time.perf_counter() - t0) / 2.0

    # Reduce mode: one flexion frame, over a sweep, per frame.
    t0 = time.perf_counter()
    for i in range(FRAMES):
        session.set_trial(flexion_deg=120.0 * i / (FRAMES - 1))
    times["reduce_frame"] = (time.perf_counter() - t0) / FRAMES
    return times, hashes, ""


def other_process(case_id: str | None) -> dict:
    """The hash table from a fresh interpreter, through the determinism command."""
    with tempfile.TemporaryDirectory(prefix="tka_e9_") as tmp:
        out = Path(tmp) / "hashes.json"
        which = ["--synthetic"] if case_id is None else ["--case", case_id]
        done = subprocess.run([sys.executable, "-m", "paper.determinism", *which,
                               "--json", str(out)],
                              cwd=REPO, capture_output=True, text=True)
        if done.returncode != 0 or not out.is_file():
            return {"error": (done.stderr or done.stdout)[-300:]}
        return json.loads(out.read_text(encoding="utf-8"))["hashes"]


def _figure(ctx: Context, rows: list[dict]) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "figure skipped: matplotlib is not installed"
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for bone, colour in (("femur", "#1b6ca8"), ("tibia", "#d08c00")):
        points = [(r[f"{bone}_triangles"] / 1e6, r[f"commit_{bone}"]) for r in rows
                  if r.get(f"commit_{bone}") is not None]
        if points:
            ax.scatter(*zip(*points), label=bone.capitalize(), color=colour, s=28)
    ax.set_xlabel("Bone mesh (million triangles)")
    ax.set_ylabel("Commit time, shell + cut (s, median)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"commit_vs_triangles.{suffix}"), dpi=200)
    plt.close(fig)
    return "figure: commit_vs_triangles.png / .svg"


@experiment("E9", "Performance and determinism", needs_cohort="optional")
def run(ctx: Context) -> None:
    library = library_root()
    repeats = 2 if ctx.quick else 5
    ctx.record["cpu_busy_percent_before"] = cpu_busy_percent(1.0)
    ctx.record["ram_available_gb_before"] = memory_gb().get("ram_available_gb")
    ctx.record["repeats"] = repeats

    determinism_rows = []

    # The synthetic fixture: in this process twice, in another, and against the
    # committed table from the paper's machine.
    case, measurement = determinism.synthetic_case()
    first = determinism.hash_table(case, measurement, library)
    case, measurement = determinism.synthetic_case()
    second = determinism.hash_table(case, measurement, library)
    separate = other_process(None)
    committed = (json.loads(determinism.REFERENCE_TABLE.read_text(encoding="utf-8"))
                 if determinism.REFERENCE_TABLE.is_file() else {})
    for name, digest in first.items():
        determinism_rows.append({
            "case": determinism.SYNTHETIC_ID, "item": name, "sha256": digest[:16],
            "in_process": second.get(name) == digest,
            "cross_process": separate.get(name) == digest,
            "vs_committed": (committed.get("hashes", {}).get(name) == digest
                             if committed else None),
        })

    rows = []
    for case in ctx.cases:
        print(f"   {case.case_id}: {repeats} passes", flush=True)
        passes, hashes, refusal = [], [], ""
        for _ in range(repeats):
            times, table, refusal = one_pass(case, library)
            passes.append(times)
            hashes.append(table)
        row = {"case": case.case_id, "refused": refusal or None,
               "kernel_refused": " ".join(
                   bone for bone in ("Femur", "Tibia")
                   if hashes[0].get(f"geometry/{bone}") == "refused") or None}
        femur, tibia = read_stl(case.femur, compute_hash=False), read_stl(
            case.tibia, compute_hash=False)
        row["femur_triangles"], row["tibia_triangles"] = femur.n_triangles, tibia.n_triangles
        del femur, tibia
        for stage in STAGES:
            values = [p[stage] for p in passes if stage in p]
            row[stage] = statistics.median(values) if values else None
        if not refusal:
            row["measure_total"] = sum(row[s] for s in
                                       ("read_stl", "qc", "landmarks", "frames_metrics"))
            separate = other_process(case.case_id)
            for name, digest in hashes[0].items():
                determinism_rows.append({
                    "case": case.case_id, "item": name, "sha256": digest[:16],
                    "in_process": all(h.get(name) == digest for h in hashes[1:]),
                    "cross_process": separate.get(name) == digest,
                    "vs_committed": None,
                })
        rows.append(row)
        ctx.table("stages", rows, digits=4)

    ctx.record["cpu_busy_percent_after"] = cpu_busy_percent(1.0)
    ctx.table("determinism", determinism_rows)
    columns = ["case", "refused", "kernel_refused", "femur_triangles", "tibia_triangles", *STAGES[:4],
               "measure_total", *STAGES[4:]]
    if rows:
        ctx.table("stages", rows, columns, digits=4)

    busy = max(v for v in (ctx.record["cpu_busy_percent_before"],
                           ctx.record["cpu_busy_percent_after"], 0.0) if v is not None)
    ctx.note(f"Machine: {ctx.record.get('cpu')}, {ctx.record.get('ram_total_gb')} GB RAM, "
             f"{ctx.record.get('platform')}; {repeats} repeats, median reported.")
    if busy > BUSY_PERCENT:
        ctx.note(f"NOT PAPER NUMBERS: the CPU was {busy:.0f}% busy with other work "
                 f"around this run (limit {BUSY_PERCENT:.0f}%). Re-run on a quiet machine.")
    if ctx.quick:
        ctx.note("Quick run: two cases, two repeats. Not paper numbers.")

    timed = [r for r in rows if not r["refused"]]
    if timed:
        for stage in ("measure_total", "plan", "implant_spec", "fit", "plan_file",
                      "scene_build", "commit_femur", "commit_tibia", "replan",
                      "reduce_frame"):
            values = [r[stage] for r in timed if r.get(stage) is not None]
            if values:
                ctx.note(f"- {stage}: median {statistics.median(values):.4f} s "
                         f"(range {min(values):.4f}-{max(values):.4f}) over "
                         f"{len(values)} knees")
        shares = [(r[f"repair_{b}"] * 2) / r[f"commit_{b}"] for r in timed
                  for b in ("femur", "tibia")
                  if r.get(f"commit_{b}") and r.get(f"repair_{b}") is not None]
        if shares:
            ctx.note(f"- Each block-mode boolean repairs the uncut bone again (weld and "
                     f"checks); twice that repair is a median "
                     f"{100 * statistics.median(shares):.0f}% of a bone's commit time.")
        ctx.note(_figure(ctx, timed))
    for r in rows:
        if r["refused"]:
            ctx.note(f"- {r['case']} refused ({r['refused']}): timed up to the refusal.")
        if r["kernel_refused"]:
            ctx.note(f"- {r['case']}: the kernel refused to cut {r['kernel_refused']} "
                     f"(see E8); no commit time for it.")

    plan_rows = [r for r in determinism_rows if r["item"] == "plan.json"]
    ctx.note(f"plan.json identical within a process for "
             f"{sum(r['in_process'] for r in plan_rows)} of {len(plan_rows)} cases, "
             f"across processes for {sum(r['cross_process'] for r in plan_rows)}.")
    differing = [r for r in determinism_rows
                 if not (r["in_process"] and r["cross_process"])
                 or r["vs_committed"] is False]
    for r in differing:
        ctx.note(f"- differs: {r['case']} {r['item']} (in process {r['in_process']}, "
                 f"across processes {r['cross_process']}, vs committed "
                 f"{r['vs_committed']})")
    if committed:
        ctx.note(f"Synthetic fixture against {determinism.REFERENCE_TABLE.name} "
                 f"({committed.get('platform')}): "
                 f"{'identical' if all(r['vs_committed'] for r in determinism_rows if r['vs_committed'] is not None) else 'DIFFERENT'}.")
