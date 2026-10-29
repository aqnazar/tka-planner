"""E6c (synthetic part): how well camera-tracked pivoting locates the hip centre.

The plan upgrades a knee-only plan's assumed mechanical axis to a measured one by
pivoting the femur under a tracking camera (WP-C2). Before any phantom is built, this
experiment answers the question the phantom protocol depends on: how large an arc, how
little noise and how many samples are needed for the hip centre to be good to a degree
of mechanical axis -- and how much pelvic motion that budget tolerates.

The simulation, all in millimetres in the tracker frame:

* The hip centre is fixed at a known point (the pelvis may move, below). The femur's
  neutral direction is ``-z``; the knee centre sits ``FEMUR_LENGTH_MM`` from the hip.
* A generic planar four-sphere marker array (about 90 mm across, not a specific
  product) is fixed to the distal femur, its origin 400-450 mm from the hip (drawn per
  repetition) and 60 mm lateral to the femoral axis.
* The leg is circumducted once round a cone of half-angle ``a`` about the neutral
  direction, as a swing without axial twist, with ``N`` samples evenly spaced in
  azimuth from a random start.
* Each sphere position gets isotropic Gaussian noise with the stated 3D RMS (per axis
  RMS / sqrt 3). Poses come from the noisy spheres by Kabsch, as a tracker computes
  them, so the rotational noise is whatever that array size implies; it is reported.
* Pelvic motion: the hip centre translates by up to ``A`` mm in the direction the knee
  is displaced, in step with the circumduction ("in phase"). The opposite phase is run
  at one setting for comparison. The true centre is the pelvis's mean position.

Both formulations in :mod:`tka_planner.core.hip_centre` run on the same samples: the
sphere fit on the four sphere trajectories, the pivot on the Kabsch poses. They run
under the permissive policy so the curves extend below the default refusal thresholds;
the share the default policy would refuse is reported per cell.

Mechanical-axis error is given two ways. ``axis_bound_deg = atan(|e| / L)`` treats the
whole centre error as perpendicular to the axis, the conservative figure the plan asks
for. ``axis_deg`` is the actual angle between the true and the estimated axis, both
from the knee centre at the neutral pose; it is smaller because much of the error of a
small arc lies along the femoral axis, where it changes the femur length instead.
"""

from __future__ import annotations

import time

import numpy as np

from tka_planner.core.hip_centre import (
    DEFAULT_POLICY,
    PERMISSIVE_POLICY,
    HipCentreFitError,
    fit_pivot,
    fit_sphere_pivoting,
    poses_from_marker_points,
)

from ..harness import Context, experiment

FEMUR_LENGTH_MM = 430.0
MARKER_DISTANCE_MM = (400.0, 450.0)
MARKER_LATERAL_MM = 60.0
# Marker-frame sphere positions: a planar array about 90 mm across.
ARRAY_MM = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 80.0],
                     [-30.0, 0.0, 40.0]])
HIP_MM = np.array([20.0, -1500.0, 150.0])   # somewhere in front of the camera

CONES_DEG = (5.0, 10.0, 20.0, 30.0)
NOISE_MM = (0.1, 0.25, 0.5)
SAMPLES = (50, 200, 1000)
PELVIS_MM = (0.0, 1.0, 2.0, 5.0)
REPETITIONS = 200
# The setting the figures and the pelvic comparison are drawn at.
REFERENCE_NOISE_MM = 0.25
REFERENCE_SAMPLES = 200

METHODS = ("sphere", "pivot")
ONE_DEGREE_MM = FEMUR_LENGTH_MM * np.tan(np.radians(1.0))


def _rodrigues(axes: np.ndarray, angle: float) -> np.ndarray:
    """Rotations by one angle about each of ``axes`` (unit, ``(N, 3)``)."""
    k = np.zeros((len(axes), 3, 3))
    k[:, 0, 1], k[:, 0, 2] = -axes[:, 2], axes[:, 1]
    k[:, 1, 0], k[:, 1, 2] = axes[:, 2], -axes[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -axes[:, 1], axes[:, 0]
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def simulate(rng: np.random.Generator, cone_deg: float, noise_mm: float, n: int,
             pelvis_mm: float) -> dict:
    """One pivoting trial: noisy sphere positions, plus the truth they came from."""
    distance = rng.uniform(*MARKER_DISTANCE_MM)
    marker = np.array([MARKER_LATERAL_MM, 0.0,
                       -np.sqrt(distance**2 - MARKER_LATERAL_MM**2)])
    azimuth = rng.uniform(0.0, 2.0 * np.pi) + 2.0 * np.pi * np.arange(n) / n
    swing_axes = np.column_stack([-np.sin(azimuth), np.cos(azimuth), np.zeros(n)])
    rotations = _rodrigues(swing_axes, np.radians(cone_deg))

    # Where the knee goes, as a unit vector across the cone: the pelvis moves with it.
    knee = rotations @ np.array([0.0, 0.0, -1.0])
    across = knee - np.outer(knee[:, 2], [0.0, 0.0, 1.0])
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    hips = HIP_MM + pelvis_mm * across

    translations = hips + rotations @ marker
    spheres = np.einsum("nij,kj->nki", rotations, ARRAY_MM) + translations[:, None, :]
    spheres = spheres + rng.normal(0.0, noise_mm / np.sqrt(3.0), spheres.shape)
    return {"spheres": spheres, "rotations": rotations, "hip": HIP_MM.copy()}


def _errors(centre: np.ndarray, truth: np.ndarray) -> dict:
    error = centre - truth
    transverse = float(np.hypot(error[0], error[1]))
    return {
        "err_mm": float(np.linalg.norm(error)),
        "transverse_mm": transverse,
        "axial_mm": float(abs(error[2])),
        "axis_bound_deg": float(np.degrees(np.arctan(np.linalg.norm(error)
                                                     / FEMUR_LENGTH_MM))),
        # The knee is at hip - L z at the neutral pose; the estimated hip is hip + e.
        "axis_deg": float(np.degrees(np.arctan2(transverse,
                                                FEMUR_LENGTH_MM + error[2]))),
    }


def trial(rng, cone_deg, noise_mm, n, pelvis_mm) -> dict[str, dict]:
    """Both methods on one simulated pivoting; a refusal is recorded, not raised."""
    data = simulate(rng, cone_deg, noise_mm, n, pelvis_mm)
    rotations, translations, fre = poses_from_marker_points(data["spheres"], ARRAY_MM)
    relative = np.einsum("nij,nkj->nik", rotations, data["rotations"])
    rotation_error = np.degrees(np.arccos(np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    shared = {"rot_noise_deg": float(np.sqrt(np.mean(rotation_error**2))),
              "fre_mm": float(np.mean(fre))}
    runs = {
        "sphere": lambda: fit_sphere_pivoting(data["spheres"], policy=PERMISSIVE_POLICY),
        "pivot": lambda: fit_pivot(rotations, translations, policy=PERMISSIVE_POLICY),
    }
    out = {}
    for method, run in runs.items():
        try:
            fit = run()
        except HipCentreFitError as refusal:
            out[method] = {"refused": True, "reason": str(refusal), **shared}
            continue
        out[method] = {
            "refused": False,
            "default_refuses": bool(DEFAULT_POLICY.reasons(fit)),
            "centre_sd_mm": fit.centre_sd_mm,
            "rms_residual_mm": fit.rms_residual_mm,
            "arc_minor_deg": fit.arc_minor_deg,
            **_errors(fit.centre_mm, data["hip"]),
            **shared,
        }
    return out


def _cell_rows(results: dict[str, list[dict]], cell: dict) -> list[dict]:
    rows = []
    for method in METHODS:
        trials = results[method]
        ok = [t for t in trials if not t["refused"]]
        row = {**cell, "method": method, "reps": len(trials),
               "computed": len(ok),
               "default_refused_pct": 100.0 * (len(trials) - sum(
                   not t["default_refuses"] for t in ok)) / max(len(trials), 1)}
        if ok:
            def stat(key, q):
                return float(np.percentile([t[key] for t in ok], q))
            row.update(
                err_med_mm=stat("err_mm", 50), err_p95_mm=stat("err_mm", 95),
                transverse_p95_mm=stat("transverse_mm", 95),
                axial_p95_mm=stat("axial_mm", 95),
                axis_bound_med_deg=stat("axis_bound_deg", 50),
                axis_bound_p95_deg=stat("axis_bound_deg", 95),
                axis_med_deg=stat("axis_deg", 50),
                axis_p95_deg=stat("axis_deg", 95),
                pred_sd_med_mm=stat("centre_sd_mm", 50),
                rms_residual_med_mm=stat("rms_residual_mm", 50),
                arc_minor_med_deg=stat("arc_minor_deg", 50),
                rot_noise_deg=stat("rot_noise_deg", 50),
            )
        rows.append(row)
    return rows


def _sweep(ctx: Context, cones, noises, samples, pelvis, repetitions, *, phase=1.0,
           tag=0) -> list[dict]:
    rows = []
    for i, cone in enumerate(cones):
        for j, noise in enumerate(noises):
            for k, n in enumerate(samples):
                for m, amplitude in enumerate(pelvis):
                    # One stream per cell, so a cell's numbers do not depend on which
                    # other cells were run; both methods see the same samples.
                    rng = np.random.default_rng([ctx.seed, tag, i, j, k, m])
                    results = {method: [] for method in METHODS}
                    for _ in range(repetitions):
                        for method, outcome in trial(rng, cone, noise, n,
                                                     phase * amplitude).items():
                            results[method].append(outcome)
                    cell = {"cone_deg": cone, "noise_mm": noise, "samples": n,
                            "pelvis_mm": phase * amplitude}
                    rows.extend(_cell_rows(results, cell))
    return rows


def _pick(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def minimum_conditions(rows: list[dict], key: str) -> list[dict]:
    """Smallest cone angle at which the 95th percentile of ``key`` is under a degree,
    for each method, noise and sample count, with a still pelvis."""
    found = []
    for method in METHODS:
        for noise in sorted({r["noise_mm"] for r in rows}):
            for n in sorted({r["samples"] for r in rows}):
                cells = sorted(_pick(rows, method=method, noise_mm=noise, samples=n,
                                     pelvis_mm=0.0), key=lambda r: r["cone_deg"])
                passing = [r["cone_deg"] for r in cells
                           if r.get(key) is not None and r[key] < 1.0]
                found.append({"method": method, "noise_mm": noise, "samples": n,
                              "min_cone_deg": min(passing) if passing else None})
    return found


def pelvic_tolerance(rows: list[dict], key: str) -> list[dict]:
    """Largest in-phase pelvic amplitude still under a degree (95th percentile of
    ``key``), per cone angle, at the reference noise and sample count."""
    found = []
    for method in METHODS:
        for cone in sorted({r["cone_deg"] for r in rows}):
            cells = _pick(rows, method=method, cone_deg=cone,
                          noise_mm=REFERENCE_NOISE_MM, samples=REFERENCE_SAMPLES)
            passing = [r["pelvis_mm"] for r in cells
                       if r["pelvis_mm"] >= 0 and r.get(key) is not None and r[key] < 1.0]
            found.append({"method": method, "cone_deg": cone,
                          "max_pelvis_mm": max(passing) if passing else None})
    return found


def _figures(ctx: Context, rows: list[dict], phase_rows: list[dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["figures skipped: matplotlib is not installed"]

    written = []
    colours = {0.1: "#1b6ca8", 0.25: "#d08c00", 0.5: "#b03a2e"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, method in zip(axes, METHODS):
        for noise in NOISE_MM:
            cells = sorted(_pick(rows, method=method, noise_mm=noise,
                                 samples=REFERENCE_SAMPLES, pelvis_mm=0.0),
                           key=lambda r: r["cone_deg"])
            cells = [c for c in cells if "err_med_mm" in c]
            x = [c["cone_deg"] for c in cells]
            ax.plot(x, [c["err_med_mm"] for c in cells], "-o", color=colours[noise],
                    label=f"{noise} mm median")
            ax.plot(x, [c["err_p95_mm"] for c in cells], "--", color=colours[noise],
                    label=f"{noise} mm 95th pct")
        ax.axhline(ONE_DEGREE_MM, color="0.4", lw=0.8)
        ax.text(CONES_DEG[-1], ONE_DEGREE_MM * 1.08, "1 deg of axis (bound)",
                ha="right", va="bottom", fontsize=8, color="0.3")
        ax.set_yscale("log")
        ax.set_xticks(CONES_DEG)
        ax.set_xlabel("cone half-angle (deg)")
        ax.set_title(f"{method} ({REFERENCE_SAMPLES} samples, still pelvis)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("hip centre error (mm)")
    axes[1].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"error_vs_cone.{suffix}"), dpi=150)
        written.append(f"error_vs_cone.{suffix}")
    plt.close(fig)

    cone_colours = dict(zip(CONES_DEG, ("#7b3294", "#1b6ca8", "#008837", "#d08c00")))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, method in zip(axes, METHODS):
        for cone in CONES_DEG:
            cells = sorted(_pick(rows + phase_rows, method=method, cone_deg=cone,
                                 noise_mm=REFERENCE_NOISE_MM, samples=REFERENCE_SAMPLES),
                           key=lambda r: r["pelvis_mm"])
            cells = [c for c in cells if "err_med_mm" in c]
            x = [c["pelvis_mm"] for c in cells]
            ax.plot(x, [c["err_med_mm"] for c in cells], "-o", color=cone_colours[cone],
                    label=f"{cone:g} deg median")
            ax.plot(x, [c["err_p95_mm"] for c in cells], "--",
                    color=cone_colours[cone], label=f"{cone:g} deg 95th pct")
        ax.axhline(ONE_DEGREE_MM, color="0.4", lw=0.8)
        ax.text(0.0, ONE_DEGREE_MM * 1.08, "1 deg of axis (bound)", ha="center",
                va="bottom", fontsize=8, color="0.3")
        ax.set_yscale("log")
        ax.set_xlabel("pelvic motion amplitude (mm; negative = opposite phase)")
        ax.set_title(f"{method} ({REFERENCE_NOISE_MM} mm noise, "
                     f"{REFERENCE_SAMPLES} samples)")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("hip centre error (mm)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"error_vs_pelvis.{suffix}"), dpi=150)
        written.append(f"error_vs_pelvis.{suffix}")
    plt.close(fig)
    return [f"figures: {', '.join(written)}"]


COLUMNS = ["method", "cone_deg", "noise_mm", "samples", "pelvis_mm", "reps", "computed",
           "default_refused_pct", "err_med_mm", "err_p95_mm", "transverse_p95_mm",
           "axial_p95_mm", "axis_bound_med_deg", "axis_bound_p95_deg", "axis_med_deg",
           "axis_p95_deg", "pred_sd_med_mm", "rms_residual_med_mm",
           "arc_minor_med_deg", "rot_noise_deg"]


@experiment("E6c", "Functional hip centre accuracy (synthetic)", needs_cohort=False)
def run(ctx: Context) -> None:
    repetitions = 20 if ctx.quick else REPETITIONS
    t0 = time.perf_counter()
    rows = _sweep(ctx, CONES_DEG, NOISE_MM, SAMPLES, PELVIS_MM, repetitions)
    phase_rows = _sweep(ctx, CONES_DEG, (REFERENCE_NOISE_MM,), (REFERENCE_SAMPLES,),
                        PELVIS_MM[1:], repetitions, phase=-1.0, tag=1)
    seconds = time.perf_counter() - t0

    ctx.table("accuracy", rows, COLUMNS)
    ctx.table("pelvis_opposite_phase", phase_rows, COLUMNS)
    noise_rows = []
    for noise in NOISE_MM:
        cells = [r for r in _pick(rows, method="pivot", noise_mm=noise)
                 if "rot_noise_deg" in r]
        noise_rows.append({"noise_mm": noise,
                           "rot_noise_deg": float(np.median(
                               [c["rot_noise_deg"] for c in cells]))})
    ctx.table("rotational_noise", noise_rows, digits=4)
    bound = minimum_conditions(rows, "axis_bound_p95_deg")
    actual = minimum_conditions(rows, "axis_p95_deg")
    for b, a in zip(bound, actual):
        b["min_cone_deg_actual_axis"] = a["min_cone_deg"]
    ctx.table("minimum_cone", bound,
              ["method", "noise_mm", "samples", "min_cone_deg",
               "min_cone_deg_actual_axis"], digits=2)
    tolerance = pelvic_tolerance(rows, "axis_bound_p95_deg")
    for t, a in zip(tolerance, pelvic_tolerance(rows, "axis_p95_deg")):
        t["max_pelvis_mm_actual_axis"] = a["max_pelvis_mm"]
    ctx.table("pelvic_tolerance", tolerance, digits=1)

    ctx.note(f"{repetitions} seeded repetitions per cell; "
             f"{len(rows) // len(METHODS)} cells in phase plus "
             f"{len(phase_rows) // len(METHODS)} opposite-phase cells; "
             f"{seconds:.0f} s.")
    ctx.note(f"Femur length {FEMUR_LENGTH_MM:g} mm: 1 deg of mechanical axis = "
             f"{ONE_DEGREE_MM:.1f} mm of centre error (conservative bound).")
    ctx.note("Rotational noise from the 4-sphere array (Kabsch): " + ", ".join(
        f"{r['noise_mm']} mm -> {r['rot_noise_deg']:.3f} deg" for r in noise_rows))
    ctx.note("")
    ctx.note("Smallest cone half-angle with the 95th percentile under 1 deg "
             "(bound / actual axis), still pelvis:")
    for row in bound:
        ctx.note(f"- {row['method']}, {row['noise_mm']} mm, {row['samples']} samples: "
                 f"{_deg(row['min_cone_deg'])} / {_deg(row['min_cone_deg_actual_axis'])}")
    ctx.note("")
    ctx.note(f"Largest in-phase pelvic motion with the 95th percentile under 1 deg "
             f"(bound / actual axis), {REFERENCE_NOISE_MM} mm noise, "
             f"{REFERENCE_SAMPLES} samples:")
    for row in tolerance:
        ctx.note(f"- {row['method']}, cone {row['cone_deg']:g} deg: "
                 f"{_mm(row['max_pelvis_mm'])} / {_mm(row['max_pelvis_mm_actual_axis'])}")
    ctx.note("")
    for line in _comparisons(rows, phase_rows):
        ctx.note(line)
    ctx.note("")
    for line in _figures(ctx, rows, phase_rows):
        ctx.note(line)


def _comparisons(rows: list[dict], phase_rows: list[dict]) -> list[str]:
    """The sentences the summary needs, computed rather than written by hand."""
    lines = ["Primary method: the pivot (author, question #13); the sphere fit is reported "
             "beside it. Femur length is a secondary outcome (#14): it can also be entered, "
             "from a long-leg film, and compared with the pivoted length."]
    lines.append(f"In-phase pelvic motion moves the pivot's centre along the femur, not "
                 f"across it -- a femur-length error, which an entered length can detect "
                 f"(95th percentile, {REFERENCE_NOISE_MM} mm, {REFERENCE_SAMPLES} samples; "
                 f"axial | transverse, mm):")
    for cone in CONES_DEG:
        cells = []
        for amplitude in PELVIS_MM:
            [row] = _pick(rows, method="pivot", cone_deg=cone, pelvis_mm=amplitude,
                          noise_mm=REFERENCE_NOISE_MM, samples=REFERENCE_SAMPLES) or [{}]
            if "axial_p95_mm" in row:
                cells.append(f"{amplitude:g} mm: {row['axial_p95_mm']:.1f} | "
                             f"{row['transverse_p95_mm']:.2f}")
        if cells:
            lines.append(f"- cone {cone:g} deg -- " + "; ".join(cells))
    still = [r for r in rows if r["pelvis_mm"] == 0.0 and "err_med_mm" in r]
    pairs = {}
    for r in still:
        pairs.setdefault((r["cone_deg"], r["noise_mm"], r["samples"]), {})[r["method"]] = r
    pivot_wins = [k for k, p in pairs.items()
                  if p["pivot"]["err_med_mm"] < p["sphere"]["err_med_mm"]]
    sphere_wins = sorted(set(pairs) - set(pivot_wins))
    lines.append(f"Still pelvis: the pivot has the lower median centre error in "
                 f"{len(pivot_wins)} of {len(pairs)} cells; the sphere fit in "
                 + (", ".join(f"{c:g} deg/{s} mm/{n}" for c, s, n in sphere_wins)
                    or "none") + ".")
    for method in METHODS:
        worst = max((r for r in still if r["method"] == method),
                    key=lambda r: r["axis_p95_deg"])
        lines.append(f"Largest 95th-percentile actual axis error, {method}, still "
                     f"pelvis: {worst['axis_p95_deg']:.2f} deg "
                     f"({worst['cone_deg']:g} deg cone, {worst['noise_mm']} mm, "
                     f"{worst['samples']} samples); femur-length (axial) error there "
                     f"{worst['axial_p95_mm']:.1f} mm.")
    lines.append("Default policy refusals (share of trials, still pelvis, "
                 f"{REFERENCE_NOISE_MM} mm, {REFERENCE_SAMPLES} samples): " + "; ".join(
                     f"{r['method']} {r['cone_deg']:g} deg {r['default_refused_pct']:.0f}%"
                     for r in _pick(rows, noise_mm=REFERENCE_NOISE_MM,
                                    samples=REFERENCE_SAMPLES, pelvis_mm=0.0)))
    lines.append("Pelvic bias against A / tan(cone), sphere fit, median error in mm "
                 "(in phase | opposite phase | predicted):")
    for cone in CONES_DEG:
        for amplitude in PELVIS_MM[1:]:
            inp = _pick(rows, method="sphere", cone_deg=cone, pelvis_mm=amplitude,
                        noise_mm=REFERENCE_NOISE_MM, samples=REFERENCE_SAMPLES)
            opp = _pick(phase_rows, method="sphere", cone_deg=cone,
                        pelvis_mm=-amplitude)
            if inp and opp and "err_med_mm" in inp[0] and "err_med_mm" in opp[0]:
                lines.append(f"- {cone:g} deg, {amplitude:g} mm: "
                             f"{inp[0]['err_med_mm']:.1f} | {opp[0]['err_med_mm']:.1f} | "
                             f"{amplitude / np.tan(np.radians(cone)):.1f}")
    return lines


def _deg(value) -> str:
    return "none tested" if value is None else f"{value:g} deg"


def _mm(value) -> str:
    return "none tested" if value is None else f"{value:g} mm"
