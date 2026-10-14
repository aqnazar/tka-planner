"""E10: predicted maximum flexion, posterior condylar offset, and the external check.

* **E10a** -- for each knee, mechanical and kinematic plans, the impingement-limited
  maximum flexion under the three trajectory scenarios (S1 native-like, S2 observed CR,
  S3 neutral hinge), the pair that first touches and where, and the envelope across
  scenarios.
* **E10b** -- posterior condylar offset per compartment, before surgery and as planned
  (patient-specific and catalogue implant), the change in mm, and the flexion change
  Bellemans' published slope would predict from it.
* **E10c** -- the model against the clinical literature. The femoral component is moved
  -3 to +3 mm along its own AP axis on each knee (the joint re-seated on the component,
  so only the femur moves; see :func:`tka_planner.pipeline.flexion_model`), and the
  slope of maximum flexion against the planned offset is fitted per knee and scenario,
  to compare with Bellemans et al. 2002 (about 6.1 deg per mm of offset). Separately, a
  pure AP translation of the femur on the tibia is swept on the hinge, and the slope of
  maximum flexion against posterior femoral translation at the limit is compared with
  Banks et al. 2003 (1.4 deg per mm).

Every flexion number here is a geometric limit of the parts under a prescribed motion
-- no soft tissue, no thigh-calf contact -- so it is expected to exceed what patients
reach (Banks: 109 +- 11 deg for CR fixed-bearing knees). The comparisons in E10c are of
slopes, not of absolute angles. Results are reported as they come out, agreeing or not.
"""

from __future__ import annotations

import time

import numpy as np

from ..harness import Context, experiment, library_root, measured

PHILOSOPHIES = ("mechanical", "kinematic")
SCENARIOS = ("S1", "S2", "S3")
OFFSETS_MM = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
QUICK_OFFSETS_MM = (-2.0, 0.0, 2.0)
BANKS_TRANSLATIONS_MM = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0)
BELLEMANS_DEG_PER_MM = 6.1
BANKS_DEG_PER_MM = 1.4


def _accepted(ctx: Context):
    """The knees the planner accepts, with their measurement; refusals are noted."""
    for case in ctx.cases:
        m = measured(case)
        if isinstance(m, Exception):
            ctx.note(f"- {case.case_id} refused ({type(m).__name__}); not planned.")
            continue
        yield case, m


def _library(ctx: Context):
    library = library_root()
    if library is None:
        raise SystemExit("E10 needs the implant library (data/implants or "
                         "TKA_IMPLANT_LIBRARY).")
    return library


def _contact_columns(record: dict) -> dict:
    contacts = record["contacts"]
    if not contacts:
        return {"first_pair": None}
    first = max(contacts, key=lambda c: c["volume_mm3"])
    anterior, left, proximal = first["location_femoral_mm"]
    return {
        "first_pair": first["pair"],
        "region": first["region"],
        "compartment": first["diagnostics"].get("compartment"),
        "contact_anterior_mm": anterior,
        "contact_left_mm": left,
        "contact_proximal_mm": proximal,
        "overlap_mm3": first["volume_mm3"],
        "crop_limited": first["diagnostics"].get("crop_limited"),
        "pairs_at_limit": " ".join(sorted({c["pair"] for c in contacts})),
    }


def _summary_stats(values) -> str:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return "n/a"
    return (f"median {np.median(values):.1f}, range {values.min():.1f} to "
            f"{values.max():.1f} (n = {len(values)})")


def _try_figure():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ----------------------------------------------------------------------
# E10a
# ----------------------------------------------------------------------


@experiment("E10a", "Predicted maximum flexion envelope (S1-S3, MA and KA)")
def run_e10a(ctx: Context) -> None:
    from tka_planner.pipeline import flexion_case, flexion_model, plan_case

    library = _library(ctx)
    rows, envelopes, settings = [], [], None
    for case, m in _accepted(ctx):
        for philosophy in PHILOSOPHIES:
            t0 = time.perf_counter()
            plan, sizing = plan_case(m, philosophy=philosophy, library=library)
            try:
                model = flexion_model(m, plan, sizing, library)
            except ValueError as error:
                ctx.note(f"- {case.case_id} {philosophy}: refused by the solid "
                         f"kernel ({str(error).splitlines()[0][:140]}).")
                break
            if model is None:
                ctx.note(f"- {case.case_id} {philosophy}: the library lacks a part; "
                         "skipped.")
                continue
            result = flexion_case(m, plan, sizing, library, model=model,
                                  trajectories=SCENARIOS)
            settings = result["settings"]
            seconds = round(time.perf_counter() - t0, 2)
            for name, record in result["trajectories"].items():
                rows.append({
                    "case": case.case_id, "side": case.side, "philosophy": philosophy,
                    "trajectory": name, "status": record["parameters"]["status"],
                    "max_flexion_deg": record["max_flexion_deg"],
                    "censored": record["censored"],
                    **_contact_columns(record),
                    "blocked_at_start": " ".join(
                        c["pair"] for c in record["blocked_at_start"]) or None,
                    "never_cleared": " ".join(record["never_cleared"]) or None,
                    "internal_rotation_at_limit_deg":
                        record["at_limit"]["internal_rotation_deg"],
                    "medial_ap_at_limit_mm": record["at_limit"]["medial_ap_mm"],
                    "lateral_ap_at_limit_mm": record["at_limit"]["lateral_ap_mm"],
                    "evaluations": record["search"]["evaluations"],
                })
            env = result["envelope"]
            envelopes.append({
                "case": case.case_id, "side": case.side, "philosophy": philosophy,
                **env,
                "baseline_articular_depth_mm":
                    result["diagnostics"]["baseline_articular_depth_mm"],
                "insert_triangles": result["diagnostics"]["triangles"]["insert"],
                "femur_triangles": result["diagnostics"]["triangles"]["femur"],
                "tibia_triangles": result["diagnostics"]["triangles"]["tibia"],
                "seconds": seconds,
            })
    ctx.table("e10a_max_flexion", rows, digits=2)
    ctx.table("e10a_envelope", envelopes, digits=2)

    ctx.note("Impingement-limited maximum flexion (deg), geometric limit of the parts "
             "under each prescribed trajectory; search 0 to "
             f"{settings['stop_deg'] if settings else '?'} deg, step "
             f"{settings['step_deg'] if settings else '?'} deg, bisected to "
             f"{settings['resolution_deg'] if settings else '?'} deg. Censored values "
             "are lower bounds (nothing touched by the stop angle).")
    for philosophy in PHILOSOPHIES:
        for name in SCENARIOS:
            sel = [r for r in rows if r["philosophy"] == philosophy
                   and r["trajectory"] == name]
            censored = sum(r["censored"] for r in sel)
            pairs = {}
            for r in sel:
                if r["first_pair"]:
                    pairs[r["first_pair"]] = pairs.get(r["first_pair"], 0) + 1
            ctx.note(f"- {philosophy} {name}: "
                     f"{_summary_stats(r['max_flexion_deg'] for r in sel)}; "
                     f"censored {censored}; first pair "
                     + (", ".join(f"{k} {v}" for k, v in sorted(pairs.items()))
                        or "none"))
    # MA against KA, paired by knee.
    for name in SCENARIOS:
        diffs = []
        for case_id in sorted({r["case"] for r in rows}):
            pick = {r["philosophy"]: r for r in rows
                    if r["case"] == case_id and r["trajectory"] == name}
            if len(pick) == 2 and not (pick["mechanical"]["censored"]
                                       or pick["kinematic"]["censored"]):
                diffs.append(pick["kinematic"]["max_flexion_deg"]
                             - pick["mechanical"]["max_flexion_deg"])
        ctx.note(f"- KA minus MA, {name}: {_summary_stats(diffs)} deg "
                 "(uncensored pairs only).")
    spreads = [e["spread_deg"] for e in envelopes if e["spread_deg"] is not None]
    ctx.note(f"- Envelope spread across S1-S3: {_summary_stats(spreads)} deg; "
             f"{sum(e['any_censored'] for e in envelopes)} of {len(envelopes)} "
             "envelopes have a censored end.")
    blocked = [r for r in rows if r["blocked_at_start"]]
    if blocked:
        ctx.note(f"- {len(blocked)} searches had a pair already in contact at "
                 "extension (a plan problem, listed in the table); such a pair counts "
                 "again once the sweep has seen it clear. Never cleared: "
                 f"{sum(bool(r['never_cleared']) for r in rows)} searches.")
        for r in blocked:
            ctx.note(f"  - {r['case']} {r['philosophy']} {r['trajectory']}: "
                     f"{r['blocked_at_start']}")
    cropped = [r for r in rows if r.get("crop_limited")]
    if cropped:
        ctx.note(f"- {len(cropped)} limiting contacts touch the crop box: the bone "
                 "beyond the crop could have touched first.")


# ----------------------------------------------------------------------
# E10b
# ----------------------------------------------------------------------


@experiment("E10b", "Posterior condylar offset as a planning output")
def run_e10b(ctx: Context) -> None:
    from tka_planner.pipeline import fit_case, plan_case

    library = _library(ctx)
    rows = []
    for case, m in _accepted(ctx):
        pre = {c: m.metrics[f"pco_{c}_mm"] for c in ("medial", "lateral")}
        for philosophy in PHILOSOPHIES:
            plan, sizing = plan_case(m, philosophy=philosophy, library=library)
            for mode in ("patient_specific", "catalogue"):
                fit = fit_case(m, plan, sizing, library, mode)
                pco = (fit or {}).get("posterior_condylar_offset") or {}
                row = {"case": case.case_id, "side": case.side,
                       "philosophy": philosophy, "implant_mode": mode,
                       "status": pre["medial"].quality.value}
                if not pco.get("available"):
                    row["note"] = pco.get("reason", "no femoral component")
                    rows.append(row)
                    continue
                for compartment in ("medial", "lateral", "radiographic"):
                    values = pco.get(compartment) or {}
                    for key in ("preoperative_mm", "planned_mm", "change_mm"):
                        row[f"{compartment}_{key}"] = values.get(key)
                change = (pco.get("radiographic") or {}).get("change_mm")
                row["predicted_flexion_change_bellemans_deg"] = (
                    None if change is None else round(BELLEMANS_DEG_PER_MM * change, 2))
                row["cortex_fit_rms_mm"] = pco["cortex_line"]["fit_rms_mm"]
                row["cortex_tangent_offset_mm"] = pco["cortex_line"][
                    "tangent_offset_from_fit_mm"]
                row["shaft_length_mm"] = pco["cortex_line"]["shaft_length_mm"]
                rows.append(row)
    ctx.table("e10b_condylar_offset", rows, digits=2)

    ctx.note("Posterior condylar offset (mm), Bellemans' radiographic definition per "
             "compartment on the mesh; change = planned - preoperative (negative: "
             "offset lost). Predicted flexion change uses Bellemans' published "
             f"{BELLEMANS_DEG_PER_MM} deg/mm on the radiographic (larger-compartment) "
             "change, as a literature-based prediction, not a model output.")
    pre_rows = {r["case"]: r for r in rows if "medial_preoperative_mm" in r}
    for compartment in ("medial", "lateral"):
        ctx.note(f"- Preoperative {compartment}: "
                 + _summary_stats(r[f"{compartment}_preoperative_mm"]
                                  for r in pre_rows.values()))
    for philosophy in PHILOSOPHIES:
        for mode in ("patient_specific", "catalogue"):
            sel = [r for r in rows if r["philosophy"] == philosophy
                   and r["implant_mode"] == mode and "medial_change_mm" in r]
            ctx.note(f"- {philosophy}, {mode}: change medial "
                     f"{_summary_stats(r['medial_change_mm'] for r in sel)}; lateral "
                     f"{_summary_stats(r['lateral_change_mm'] for r in sel)}; "
                     "radiographic "
                     f"{_summary_stats(r['radiographic_change_mm'] for r in sel)}")
    statuses = sorted({r["status"] for r in rows})
    ctx.note(f"- Provenance tier of the offsets: {', '.join(statuses)} (automatic "
             "landmarks on every knee; no mechanical axis is used, so knee-only scans "
             "are not demoted).")


# ----------------------------------------------------------------------
# E10c
# ----------------------------------------------------------------------


def _slope(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - predicted) ** 2).sum()) / total if total > 0 else None
    return float(slope), r2


def _pooled_slope(rows, measure) -> dict:
    """One slope over all knees, each knee's points centred on its own means, so the
    slope is the within-knee effect and differences between knees do not enter it."""
    out = {}
    for name in SCENARIOS:
        xs, ys = [], []
        for case_id in sorted({r["case"] for r in rows}):
            sel = [r for r in rows if r["case"] == case_id and r["trajectory"] == name
                   and not r["censored"]]
            if len(sel) < 3:
                continue
            x = np.array([r[measure] for r in sel], dtype=float)
            y = np.array([r["max_flexion_deg"] for r in sel], dtype=float)
            xs.append(x - x.mean())
            ys.append(y - y.mean())
        out[name] = None
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            if x @ x > 0:
                out[name] = round(float((x @ y) / (x @ x)), 3)
    return out


@experiment("E10c", "External check against Bellemans 2002 and Banks 2003")
def run_e10c(ctx: Context) -> None:
    from tka_planner.core.condylar_offset import planned_condylar_offset
    from tka_planner.core.kinematics import NEUTRAL, with_parameters
    from tka_planner.pipeline import flexion_case, flexion_model, plan_case

    library = _library(ctx)
    offsets = QUICK_OFFSETS_MM if ctx.quick else OFFSETS_MM
    sweep, slopes, banks_rows, banks_slopes = [], [], [], []
    for case, m in _accepted(ctx):
        plan, sizing = plan_case(m, philosophy="mechanical", library=library)
        anterior = np.asarray(plan.components["femoral_component"], dtype=float)[:3, 0]
        anterior = anterior / np.linalg.norm(anterior)
        for offset in offsets:
            try:
                model = flexion_model(m, plan, sizing, library,
                                      femoral_offset_mm=offset)
            except ValueError as error:
                ctx.note(f"- {case.case_id}: refused by the solid kernel "
                         f"({str(error).splitlines()[0][:140]}).")
                model = None
            if model is None:
                break
            component = model.femoral_component.to_mesh().vertices + offset * anterior
            pco = planned_condylar_offset(m.femur.vertices, component, m.landmarks,
                                          m.femoral_frame)
            result = flexion_case(m, plan, sizing, library, model=model,
                                  trajectories=SCENARIOS)
            for name, record in result["trajectories"].items():
                sweep.append({
                    "case": case.case_id, "side": case.side,
                    "component_anterior_shift_mm": offset, "trajectory": name,
                    "pco_medial_mm": pco["medial"]["planned_mm"],
                    "pco_lateral_mm": pco["lateral"]["planned_mm"],
                    "pco_radiographic_mm": pco["radiographic"]["planned_mm"],
                    "max_flexion_deg": record["max_flexion_deg"],
                    "censored": record["censored"],
                    "first_pair": record["first_pair"],
                })
            if offset == 0.0:
                for t_ref in BANKS_TRANSLATIONS_MM:
                    trajectory = with_parameters(
                        NEUTRAL, name=f"AP{t_ref:+.0f}", label="AP translation sweep",
                        ap_translation_mm=t_ref, reference_flexion_deg=109.0)
                    found = flexion_case(m, plan, sizing, library, model=model,
                                         trajectories=(trajectory,))
                    record = found["trajectories"][trajectory.name]
                    theta = record["max_flexion_deg"]
                    banks_rows.append({
                        "case": case.case_id, "side": case.side,
                        "ap_translation_at_109_mm": t_ref,
                        "max_flexion_deg": theta, "censored": record["censored"],
                        "posterior_translation_at_limit_mm":
                            round(-t_ref * theta / 109.0, 3),
                        "first_pair": record["first_pair"],
                    })
        for name in SCENARIOS:
            sel = [r for r in sweep if r["case"] == case.case_id
                   and r["trajectory"] == name and not r["censored"]]
            # An AP shift moves both condyles alike, so one offset is enough.
            slope, r2 = _slope([r["pco_radiographic_mm"] for r in sel],
                               [r["max_flexion_deg"] for r in sel])
            slopes.append({"case": case.case_id, "trajectory": name,
                           "n_points": len(sel), "n_censored": sum(
                               r["censored"] for r in sweep
                               if r["case"] == case.case_id and r["trajectory"] == name),
                           "deg_per_mm": slope, "r2": r2})
        sel = [r for r in banks_rows if r["case"] == case.case_id and not r["censored"]]
        slope, r2 = _slope([r["posterior_translation_at_limit_mm"] for r in sel],
                           [r["max_flexion_deg"] for r in sel])
        banks_slopes.append({"case": case.case_id, "n_points": len(sel),
                             "deg_per_mm_posterior": slope, "r2": r2})
    ctx.table("e10c_offset_sweep", sweep, digits=3)
    ctx.table("e10c_pco_slopes", slopes, digits=3)
    ctx.table("e10c_banks_sweep", banks_rows, digits=3)
    ctx.table("e10c_banks_slopes", banks_slopes, digits=3)

    ctx.note("Slope of modelled maximum flexion against planned posterior condylar "
             f"offset, per knee (component moved {offsets[0]:+.0f} to "
             f"{offsets[-1]:+.0f} mm along its AP axis; censored points excluded). "
             f"Bellemans 2002: about {BELLEMANS_DEG_PER_MM} deg/mm (R^2 = 0.58).")
    for name in SCENARIOS:
        values = [s["deg_per_mm"] for s in slopes if s["trajectory"] == name
                  and s["deg_per_mm"] is not None]
        r2 = [s["r2"] for s in slopes if s["trajectory"] == name
              and s["r2"] is not None]
        ctx.note(f"- {name}: {_summary_stats(values)} deg/mm; per-knee R^2 "
                 f"{_summary_stats(r2)}")
    pooled = _pooled_slope(sweep, "pco_radiographic_mm")
    ctx.note(f"- Pooled within-knee slope (each knee centred on its own means): "
             + ", ".join(f"{k} {v}" for k, v in pooled.items()) + " deg/mm.")
    values = [s["deg_per_mm_posterior"] for s in banks_slopes
              if s["deg_per_mm_posterior"] is not None]
    ctx.note("Slope of modelled maximum flexion against posterior femoral translation "
             "at the limit (hinge plus a pure AP translation, "
             f"{BANKS_TRANSLATIONS_MM[0]:+.0f} to {BANKS_TRANSLATIONS_MM[-1]:+.0f} mm "
             f"at 109 deg): {_summary_stats(values)} deg/mm; Banks 2003: "
             f"{BANKS_DEG_PER_MM} deg/mm (R = 0.64).")

    plt = _try_figure()
    if plt is None:
        ctx.note("(matplotlib not installed: figures skipped; tables written.)")
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for case_id in sorted({r["case"] for r in sweep}):
        sel = [r for r in sweep if r["case"] == case_id and r["trajectory"] == "S3"
               and not r["censored"]]
        axes[0].plot([r["pco_radiographic_mm"] for r in sel],
                     [r["max_flexion_deg"] for r in sel], marker="o", label=case_id)
        sel = [r for r in banks_rows if r["case"] == case_id and not r["censored"]]
        axes[1].plot([r["posterior_translation_at_limit_mm"] for r in sel],
                     [r["max_flexion_deg"] for r in sel], marker="o")
    axes[0].set_xlabel("planned posterior condylar offset (mm)")
    axes[0].set_ylabel("max flexion, S3 (deg)")
    axes[1].set_xlabel("posterior femoral translation at the limit (mm)")
    axes[1].set_ylabel("max flexion (deg)")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(ctx.path("e10c_slopes.png"), dpi=150)
    figure.savefig(ctx.path("e10c_slopes.svg"))
    plt.close(figure)
