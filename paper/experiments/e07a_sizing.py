"""E7a: what the sizing measurement is taken over -- a bounding box, or the cut.

Paper 1 (the legacy script) sized the implant from the bounding box of the distal
quarter of the femur and of the proximal quarter of the tibia
(``REGIONAL_BBOX_FRACTION = 0.25``), and the planner still drives its femoral size that
way (``measure_femoral_ml``, the same box taken in the femoral frame's axes rather than
the scanner's). A box of the distal quarter is set by the widest thing in it, which is
the epicondyles, above the joint line; the component sits on the distal cut, where the
bone is narrower. The tibial box is set by the tubercle and the eminence.

For every knee this experiment takes both measurements of each sizing dimension:

* **box** -- the legacy regional bounding box, in the bone's anatomical frame;
* **cut** -- the resection cross-section at the plan's own cuts, as the patient-specific
  implant is measured (:func:`tka_planner.pipeline.implant_spec_case`): the femoral
  width at the distal cut, the femoral depth over the component's flange height, and
  the tibial width and depth at the tibial cut.

The tibia also gets the planner's current plateau measurement
(``measure_tibial_plateau``, a section 5 mm below the articular surface), which is
neither: it is a cut, but not the planned one.

Each measurement is turned into the chart size the legacy rule would pick from it
(``select_discrete_size``, round-down, the legacy ``SIZE_ROUND = "D"``), so the table
shows both the difference in millimetres and whether it changes the size.

The plan is the planner's default (mechanical alignment, default controls, the implant
library loaded). Its femoral cut depth follows the size the box picks, so the cut the
section is taken at is itself the box-sized plan's -- the comparison is of where to
measure on one plan, not of two plans.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.geometry import regional_mask
from tka_planner.core.sizing import select_discrete_size
from tka_planner.pipeline import implant_spec_case, plan_case

from ..harness import Context, experiment, library_root, measured
from ..stats import format_p, paired_summary

REGIONAL_BBOX_FRACTION = 0.25   # legacy/TKA_Script_v3.py
SIZE_ROUND = "D"                # legacy default: the largest size not exceeding the bone

# (dimension, chart column it sizes against)
DIMENSIONS = (
    ("femur_ML", "femur_ML"),
    ("femur_AP", "femur_AP"),
    ("tibia_ML", "tibia_ML"),
    ("tibia_AP", "tibia_AP"),
)


def regional_box(mesh, frame, *, end: str) -> dict:
    """The legacy regional bounding box, in the bone's frame: ML along patient-left,
    AP along anterior, over the quarter of the bone at ``end`` of its long axis."""
    region = mesh.vertices[regional_mask(mesh.vertices, frame.z_proximal,
                                         fraction=REGIONAL_BBOX_FRACTION, end=end)]
    local = frame.to_local(region)
    return {"ml": float(np.ptp(local[:, 1])), "ap": float(np.ptp(local[:, 0]))}


def size_for(chart, column: str, value: float | None) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    decision = select_discrete_size(chart, measured_ml_mm=value,
                                    driving_dimension=column, rounding=SIZE_ROUND)
    label = decision.nearest_discrete_size
    return label, chart.labels.index(label)


def measure_knee(case, library) -> dict | Exception:
    """Every measurement of one knee, or the planner's refusal."""
    m = measured(case)
    if isinstance(m, Exception):
        return m
    plan, sizing = plan_case(m, library=library)
    spec = implant_spec_case(m, plan, sizing, library, with_cut_box=False)
    femoral = spec["femoral_component"]["dimensions_mm"]
    tibial = spec["tibial_component"]["dimensions_mm"]
    tibial_box = regional_box(m.tibia, m.tibial_frame, end="high")
    return {
        "chart": m.chart,
        "box": {"femur_ML": m.femoral_measure.ml_mm, "femur_AP": m.femoral_measure.ap_mm,
                "tibia_ML": tibial_box["ml"], "tibia_AP": tibial_box["ap"]},
        "cut": {"femur_ML": femoral["ml"], "femur_AP": femoral["ap_overall"],
                "tibia_ML": tibial["ml"], "tibia_AP": tibial["ap"]},
        "plateau": {"tibia_ML": m.tibial_measure.ml_mm, "tibia_AP": m.tibial_measure.ap_mm},
        "femur_AP_distal_facet": femoral["ap_distal"],
        "femoral_distal_cut_mm": {"medial": plan.resections["femoral_distal"].medial_depth_mm,
                                  "lateral": plan.resections["femoral_distal"].lateral_depth_mm},
        "size_parameter": sizing.size_parameter,
    }


@experiment("E7a", "Sizing measurement: bounding box vs resection cross-section")
def run(ctx: Context) -> None:
    library = library_root()
    if library is None:
        ctx.note("No implant library found (data/implants or TKA_IMPLANT_LIBRARY); "
                 "the plan needs it for the tray thickness. Nothing run.")
        return
    rows, refused = [], []
    for case in ctx.cases:
        result = measure_knee(case, library)
        if isinstance(result, Exception):
            refused.append((case.case_id, type(result).__name__))
            continue
        chart = result["chart"]
        for dimension, column in DIMENSIONS:
            box, cut = result["box"][dimension], result["cut"][dimension]
            plateau = result["plateau"].get(dimension)
            size_box, index_box = size_for(chart, column, box)
            size_cut, index_cut = size_for(chart, column, cut)
            size_plateau, _ = size_for(chart, column, plateau)
            low, high = float(chart.columns[column][0]), float(chart.columns[column][-1])
            row = {
                "case": case.case_id, "side": case.side, "dimension": dimension,
                "box_mm": box, "cut_mm": cut, "box_minus_cut_mm": box - cut,
                "plateau_5mm_mm": plateau,
                "size_box": size_box, "size_cut": size_cut, "size_plateau_5mm": size_plateau,
                "size_steps_box_minus_cut": index_box - index_cut,
                "size_changes": size_box != size_cut,
                "box_outside_chart": not low <= box <= high,
                "cut_outside_chart": not low <= cut <= high,
            }
            if dimension == "femur_AP":
                row["note"] = (f"cut = depth over the flange height; distal facet alone "
                               f"{result['femur_AP_distal_facet']:.1f} mm")
            rows.append(row)

    columns = ["case", "side", "dimension", "box_mm", "cut_mm", "box_minus_cut_mm",
               "plateau_5mm_mm", "size_box", "size_cut", "size_plateau_5mm",
               "size_steps_box_minus_cut", "size_changes", "box_outside_chart",
               "cut_outside_chart", "note"]
    ctx.table("sizing_box_vs_cut", rows, columns, digits=1)

    knees = sorted({r["case"] for r in rows})
    ctx.note(f"{len(knees)} knees measured"
             + (f"; refused: {', '.join(f'{c} ({why})' for c, why in refused)}."
                if refused else "."))
    ctx.note("Box = legacy bounding box of the distal (femur) or proximal (tibia) quarter "
             "of the bone, in its anatomical frame. Cut = section at the planned cut "
             "(femoral AP: depth over the component's flange height). Sizes by the "
             "legacy round-down rule against each dimension's own chart column.\n")
    summary = []
    for dimension, _ in DIMENSIONS:
        sub = [r for r in rows if r["dimension"] == dimension]
        stats = paired_summary([r["box_mm"] for r in sub], [r["cut_mm"] for r in sub])
        changed = sum(r["size_changes"] for r in sub)
        steps = [r["size_steps_box_minus_cut"] for r in sub]
        summary.append({"dimension": dimension, "knees": stats["n"],
                        "box_minus_cut_mean_mm": stats["mean"],
                        "box_minus_cut_sd_mm": stats["sd"],
                        "box_minus_cut_min_mm": stats["min"],
                        "box_minus_cut_max_mm": stats["max"],
                        "wilcoxon_p": stats["wilcoxon_p"],
                        "size_changes": changed,
                        "size_steps_min": min(steps) if steps else None,
                        "size_steps_max": max(steps) if steps else None,
                        "box_outside_chart": sum(r["box_outside_chart"] for r in sub),
                        "cut_outside_chart": sum(r["cut_outside_chart"] for r in sub)})
        if stats["n"]:
            ctx.note(f"- {dimension}: box - cut {stats['mean']:+.1f} mm "
                     f"(SD {stats['sd'] or 0:.1f}, range {stats['min']:+.1f} to "
                     f"{stats['max']:+.1f}; {format_p(stats)}); the chart size changes "
                     f"in {changed} "
                     f"of {stats['n']} knees (box minus cut: {min(steps):+d} to "
                     f"{max(steps):+d} sizes).")
    ctx.table("sizing_summary", summary, digits=2)

    for case_id in ("P009L", "P009R"):
        r = next((x for x in rows if x["case"] == case_id and x["dimension"] == "femur_ML"),
                 None)
        if r:
            ctx.note(f"- {case_id} femoral ML: box {r['box_mm']:.1f} mm "
                     f"({r['size_box']}), at the distal cut {r['cut_mm']:.1f} mm "
                     f"({r['size_cut']}): {r['box_minus_cut_mm']:+.1f} mm.")
    _figure(ctx, rows)


def _figure(ctx: Context, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.note("matplotlib not installed; no figure.")
        return
    knees = sorted({r["case"] for r in rows})
    if not knees:
        return
    x = np.arange(len(knees))
    box_colour, cut_colour, plateau_colour = "#eb6834", "#2a78d6", "#1baf7a"
    titles = {"femur_ML": "Femur, mediolateral", "femur_AP": "Femur, anteroposterior",
              "tibia_ML": "Tibia, mediolateral", "tibia_AP": "Tibia, anteroposterior"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (dimension, column) in zip(axes.flat, DIMENSIONS):
        sub = {r["case"]: r for r in rows if r["dimension"] == dimension}
        box = np.array([sub[k]["box_mm"] if k in sub else np.nan for k in knees])
        cut = np.array([sub[k]["cut_mm"] if k in sub else np.nan for k in knees])
        for i in range(len(knees)):
            ax.plot([x[i], x[i]], [box[i], cut[i]], color="#b9b8b2", lw=1.5, zorder=1)
        ax.scatter(x, box, s=46, color=box_colour, marker="s", zorder=3,
                   edgecolor="white", linewidth=1.5, label="Bounding box (legacy)")
        ax.scatter(x, cut, s=50, color=cut_colour, marker="o", zorder=3,
                   edgecolor="white", linewidth=1.5, label="Section at the planned cut")
        if dimension.startswith("tibia"):
            plateau = np.array([sub[k]["plateau_5mm_mm"] if k in sub else np.nan
                                for k in knees])
            ax.scatter(x, plateau, s=46, color=plateau_colour, marker="^", zorder=3,
                       edgecolor="white", linewidth=1.5,
                       label="Section 5 mm below plateau (current)")
        ax.set_title(titles[dimension], fontsize=11, loc="left")
        ax.set_ylabel("mm")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    _chart_bands(axes)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(knees, rotation=45, ha="right")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"sizing_box_vs_cut.{suffix}"), dpi=200)
    plt.close(fig)


def _chart_bands(axes) -> None:
    """Shade the chart's range for each dimension, so a measurement outside it shows."""
    from tka_planner.core.sizing import load_size_chart
    from tka_planner.pipeline import find_size_chart

    chart = load_size_chart(find_size_chart())
    for ax, (_, column) in zip(axes.flat, DIMENSIONS):
        values = chart.columns[column]
        ax.axhspan(float(values[0]), float(values[-1]), color="#f0efec", zorder=0)
        ax.text(1.0, float(values[-1]), " chart range", transform=ax.get_yaxis_transform(),
                fontsize=8, color="#52514e", va="top", ha="right")
