"""E5: how far landmark picking error moves the metrics and the plan.

Every accepted knee is measured once (``harness.measured``), and its landmarks are then
re-picked many times from a dispersion table by
:func:`tka_planner.validation.montecarlo.propagate`, each draw re-running frames,
metrics, sizing and the plan through the application's own pipeline. The automatic
landmark estimator is not re-run per draw: what is measured is the planner's
sensitivity to where the points are.

**The dispersion is a placeholder.** E3 has not yet measured how far surgeons'
picks scatter, so the main run uses the shipped table of 2 mm per axis for every
picked landmark (``placeholder_uniform_2mm.csv``). The intervals below say how the
planner *would* respond to that much scatter; they are not the planner's uncertainty
until E3's table replaces it (same CSV format, ``landmark_id,sd_mm``).

Variants, at fewer draws: the registry's unsourced ``observer_sd_mm`` values; the
uniform table with the canal centres also moved by 2 mm, since they set both
mechanical axes and are otherwise held fixed; and on one knee, free 3D noise against
the default tangent-plane model.

Outputs: ``intervals`` (per knee and output: nominal, mean, SD, 2.5/97.5 percentiles),
``pooled`` (per output across knees: the 2.5/97.5 percentiles of the deviation from
each knee's nominal, and the median and largest per-knee SD), ``knees`` (draws,
failures, size agreement, which landmarks moved), and the variants' pooled tables.
"""

from __future__ import annotations

import zlib

import numpy as np

from ..harness import _MEASURED, Context, experiment, library_root, measured

N_DRAWS, N_QUICK = 500, 50
N_VARIANT, N_VARIANT_QUICK = 100, 20

# The outputs the summary quotes; every output is in the tables.
HEADLINE = (
    "metric.mldfa_deg", "metric.mpta_deg", "metric.jlca_deg",
    "metric.posterior_slope_medial_deg",
    "bone.femoral_cut_coronal_deg", "bone.tibial_cut_coronal_deg",
    "bone.tibial_cut_sagittal_deg",
    "bone.femoral_depth_medial_mm", "bone.femoral_depth_lateral_mm",
    "bone.tibial_depth_medial_mm", "bone.tibial_depth_lateral_mm",
    "bone.femoral_component_rotation_deg",
    "plan.size_parameter", "plan.insert_thickness_mm",
)


def _seed(base: int, case_id: str, variant: str) -> int:
    """A fixed seed per knee and variant, independent of which knees are run."""
    return (base + zlib.crc32(f"{case_id}/{variant}".encode())) % (2 ** 32)


def _pooled(results) -> list[dict]:
    """Per output, the deviation from nominal pooled over knees, and per-knee SDs."""
    names: list[str] = []
    for result in results:
        names.extend(n for n in result.samples if n not in names)
    rows = []
    for name in names:
        deviations, sds, widths = [], [], []
        for result in results:
            values = result.samples.get(name)
            if values is None or values.size < 2 or name not in result.nominal:
                continue
            deviations.append(values - result.nominal[name])
            sds.append(float(values.std(ddof=1)))
            low, high = np.percentile(values, [2.5, 97.5])
            widths.append(float(high - low))
        if not deviations:
            continue
        pooled = np.concatenate(deviations)
        low, high = np.percentile(pooled, [2.5, 97.5])
        rows.append({
            "output": name,
            "knees": len(deviations),
            "draws": int(pooled.size),
            "deviation_p2_5": float(low),
            "deviation_p97_5": float(high),
            "pooled_sd": float(pooled.std(ddof=1)),
            "median_knee_sd": float(np.median(sds)),
            "max_knee_sd": float(np.max(sds)),
            "median_knee_width_95": float(np.median(widths)),
            "max_knee_width_95": float(np.max(widths)),
        })
    return rows


def _with_canal_centres(dispersion, sd_mm: float = 2.0):
    from tka_planner.validation.montecarlo import Dispersion

    covariances = dict(dispersion.covariances)
    for landmark_id in ("femur.canal_centre_distal", "femur.canal_centre_proximal",
                        "tibia.canal_centre_proximal", "tibia.canal_centre_distal"):
        covariances[landmark_id] = sd_mm ** 2 * np.eye(3)
    return Dispersion(covariances, label=f"{dispersion.label} + canal centres {sd_mm:g} mm",
                      placeholder=True)


def _figure(ctx: Context, results) -> bool:
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [n for n in HEADLINE if any(n in r.samples for r in results)]
    fig, axes = plt.subplots(2, (len(shown) + 1) // 2, figsize=(2.3 * ((len(shown) + 1) // 2), 6))
    for ax, name in zip(axes.flat, shown):
        for i, result in enumerate(results):
            values = result.samples.get(name)
            if values is None or name not in result.nominal:
                continue
            deviation = values - result.nominal[name]
            low, high = np.percentile(deviation, [2.5, 97.5])
            ax.plot([low, high], [i, i], color="#1f5f8b", lw=2)
            ax.plot([0], [i], "|", color="#c8553d")
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([r.case_id for r in results], fontsize=6)
        ax.set_title(name.split(".", 1)[1], fontsize=7)
        ax.axvline(0, color="#888", lw=0.6)
        ax.tick_params(axis="x", labelsize=6)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    fig.suptitle("95% interval of deviation from the nominal plan -- PLACEHOLDER "
                 "dispersion (2 mm), not measured", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"e05_intervals.{suffix}"), dpi=200)
    plt.close(fig)
    return True


@experiment("E5", "Monte Carlo propagation of landmark picking error")
def run(ctx: Context) -> None:
    from tka_planner.validation.montecarlo import placeholder_dispersion, propagate

    n = N_QUICK if ctx.quick else N_DRAWS
    n_variant = N_VARIANT_QUICK if ctx.quick else N_VARIANT
    library = library_root()
    main = placeholder_dispersion("uniform_2mm")
    variants = {
        "registry": placeholder_dispersion("registry"),
        "uniform_with_canals": _with_canal_centres(main),
    }

    results, variant_results, knee_rows, interval_rows = [], {k: [] for k in variants}, [], []
    surface_rows = []
    for case in ctx.cases:
        cached_before = case.case_id in _MEASURED
        m = measured(case)
        if isinstance(m, Exception):
            knee_rows.append({"case": case.case_id, "accepted": False,
                              "refusal": type(m).__name__})
            continue
        result = propagate(m, main, n=n, seed=_seed(ctx.seed, case.case_id, "main"),
                           library=library)
        results.append(result)
        for row in result.summary():
            interval_rows.append({"case": case.case_id, **row})
        knee_rows.append({
            "case": case.case_id, "accepted": True, "draws": n,
            "failed": result.n_failed,
            "failures": " ".join(f"{k}={v}" for k, v in sorted(result.failures.items())),
            "nominal_size": result.nominal_size,
            "size_agreement": result.size_agreement(),
            "perturbed": len(result.perturbed),
            "held_fixed": " ".join(result.held_fixed),
            "moved_freely": " ".join(result.free_moving),
        })
        for name, dispersion in variants.items():
            variant_results[name].append(propagate(
                m, dispersion, n=n_variant, seed=_seed(ctx.seed, case.case_id, name),
                library=library))
        if not surface_rows:
            # The surface model's effect, on the first accepted knee.
            free = propagate(m, main, n=n_variant,
                             seed=_seed(ctx.seed, case.case_id, "free"),
                             surface="free", library=library)
            tangent = {r["output"]: r for r in result.summary()}
            for row in free.summary():
                if row["output"] in tangent:
                    surface_rows.append({"case": case.case_id, "output": row["output"],
                                         "sd_tangent": tangent[row["output"]]["sd"],
                                         "sd_free": row["sd"]})
        if not cached_before:
            _MEASURED.pop(case.case_id, None)  # three million vertices a bone: let go

    ctx.table("knees", knee_rows)
    ctx.table("intervals", interval_rows, digits=4)
    pooled = _pooled(results)
    ctx.table("pooled", pooled, digits=4)
    for name, runs in variant_results.items():
        ctx.table(f"pooled_{name}", _pooled(runs), digits=4)
    if surface_rows:
        ctx.table("surface_model", surface_rows, digits=4)
    drew = _figure(ctx, results)

    # ---- Summary --------------------------------------------------------------
    ctx.note("**PLACEHOLDER DISPERSION.** Every interval here comes from an assumed "
             f"table ({main.label}), not from measured inter-observer data. They become "
             "the planner's uncertainty only when E3's table replaces it.\n")
    accepted = [r for r in knee_rows if r["accepted"]]
    ctx.note(f"{len(accepted)} knees propagated, {n} draws each (seed {ctx.seed} + "
             f"per-knee offset), surface model 'tangent' (surface landmarks move within "
             f"their tangent plane).")
    for row in knee_rows:
        if not row["accepted"]:
            ctx.note(f"- {row['case']} refused at measurement ({row['refusal']}); not propagated.")
    failed = sum(r["failed"] for r in accepted)
    ctx.note(f"Draws whose frames could not be built: {failed} of {n * len(accepted)}"
             + (" (" + "; ".join(f"{r['case']}: {r['failures']}" for r in accepted
                                 if r['failed']) + ")" if failed else "") + ".")
    if accepted:
        held = sorted({h for r in accepted for h in r["held_fixed"].split()})
        ctx.note(f"Held fixed (no row in the table): {', '.join(held) or 'none'}. "
                 "With the canal centres fixed both mechanical axes are fixed, so the "
                 "coronal cut angles barely move; see pooled_uniform_with_canals for the "
                 "axes' own share.")
        agreement = [r["size_agreement"] for r in accepted]
        ctx.note(f"Chart size unchanged from nominal in {np.mean(agreement):.0%} of draws "
                 f"(per knee {min(agreement):.0%} to {max(agreement):.0%}).")
    ctx.note("\nPooled 95% interval of the deviation from each knee's nominal "
             "(main table):")
    by_name = {r["output"]: r for r in pooled}
    canal = {r["output"]: r for r in _pooled(variant_results["uniform_with_canals"])}
    for name in HEADLINE:
        if name in by_name:
            r = by_name[name]
            extra = ""
            if name in canal:
                c = canal[name]
                extra = (f"; with canal centres also moved "
                         f"[{c['deviation_p2_5']:+.2f}, {c['deviation_p97_5']:+.2f}]")
            ctx.note(f"- {name}: [{r['deviation_p2_5']:+.2f}, {r['deviation_p97_5']:+.2f}] "
                     f"(median per-knee SD {r['median_knee_sd']:.2f}, largest "
                     f"{r['max_knee_sd']:.2f}){extra}")
    if surface_rows:
        ctx.note("\nSurface model, first knee (SD tangent vs free 3D): " + "; ".join(
            f"{r['output']} {r['sd_tangent']:.2f} vs {r['sd_free']:.2f}"
            for r in surface_rows if r["output"] in HEADLINE[:8]))
    if not drew:
        ctx.note("Figure skipped: matplotlib is not installed; intervals.csv holds its data.")
