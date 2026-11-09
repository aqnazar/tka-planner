"""E10: predicted maximum flexion, posterior condylar offset, and the external check.

* **E10a** -- for each knee, mechanical and kinematic plans, the impingement-limited
  maximum flexion under the three trajectory scenarios (S1 native-like, S2 observed CR,
  S3 neutral hinge), the pair that first touches and where, and the envelope across
  scenarios. S1's pivot is scaled to each knee's condyle-centre spacing (author ruling
  2026-10-26, an assumption; :func:`tka_planner.pipeline.scenarios_case`), and every
  row records the pivot, the spacing and the scaling status.
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
* **E10d** -- why the E10c slope is what it is (author ruling #24): the contact that
  stops flexion seen from the hinge (its distance ``r`` from the flexion axis, against
  the lever-arm prediction of about 57.3/r deg per mm), the offset slope under
  rollback and paradoxical anterior sliding, its dependence on tibial slope, and the
  two ways of changing the offset (component moved with the femur re-seated, against a
  thicker posterior condyle with the bone fixed). The 6.1 deg/mm usually attributed to
  Bellemans 2002 is from secondary citations and is treated as unverified; Massin and
  Gournay 2006 (about 3.3 deg/mm from a geometric model) is the other comparison.

**Stop rule (author ruling #26).** No search stops at a chosen angle: each runs until
the patient's own scanned models make blocking contact, bounded only by the end of the
motion (180 deg, the tibia folded onto the femur), and the bones are the whole scans,
cut only beyond the other bone's reach from the hinge axis. A search that reaches the
bound untouched has no flexion value; it is reported as "no contact in the scanned
models", with the reason measured from the models (how far each bone was scanned, and
whether the gap is closing towards the end of a scan). See
:func:`tka_planner.pipeline.no_contact_reason`.

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
        "contact_hinge_radius_mm": first["diagnostics"].get("hinge_radius_mm"),
        "overlap_mm3": first["volume_mm3"],
        "crop_limited": first["diagnostics"].get("crop_limited"),
        "pairs_at_limit": " ".join(sorted({c["pair"] for c in contacts})),
    }


def _no_contact_columns(record: dict) -> dict:
    """Why a search met nothing (ruling #26), from :func:`no_contact_reason`."""
    reason = record.get("no_contact")
    if not reason:
        return {}
    gaps = [g for g in reason["gaps_mm"].values() if g is not None]
    return {
        "no_contact_reason": reason["reason"],
        "closest_gap_at_bound_mm": min(gaps) if gaps else None,
        "folded_tibia_reach_mm": reason["folded_tibia_reach_from_hinge_mm"],
        "no_contact_text": reason["text"],
    }


def _extent_row(case, extent: dict) -> dict:
    row = {"case": case.case_id, "side": case.side}
    for bone in ("femur", "tibia"):
        for key in ("scanned_from_hinge_mm", "scanned_from_joint_line_mm",
                    "radial_reach_mm", "kept_from_hinge_mm"):
            row[f"{bone}_{key}"] = extent[bone][key]
    row["meeting_reach_mm"] = extent["meeting_reach_mm"]
    return row


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
    rows, envelopes, extents, settings = [], [], [], None
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
                scaling = record["parameters"].get("pivot_scaling") or {}
                rows.append({
                    "case": case.case_id, "side": case.side, "philosophy": philosophy,
                    "trajectory": name, "status": record["parameters"]["status"],
                    # S1's pivot is scaled to the knee (author ruling 2026-10-26).
                    "pivot_ml_mm": record["parameters"]["pivot_ml_mm"],
                    "pivot_scaling": scaling.get("status"),
                    "centre_spacing_mm": scaling.get("centre_spacing_mm"),
                    # Ruling #26: a search that met nothing has no flexion value; the
                    # bound it reached is in searched_to_deg.
                    "max_flexion_deg": record["flexion_limit_deg"],
                    # Shown, never fitted: the usual extreme, as a bound (ruling 2026-10-25).
                    "reported_flexion_deg": record["reported_flexion_deg"],
                    "reported_is_bound": record["reported_is_bound"],
                    "censored": record["censored"],
                    "stop_reason": record["stop_reason"],
                    "searched_to_deg": record["search"]["stop_deg"],
                    **_contact_columns(record),
                    **_no_contact_columns(record),
                    "blocked_at_start": " ".join(
                        c["pair"] for c in record["blocked_at_start"]) or None,
                    "never_cleared": " ".join(record["never_cleared"]) or None,
                    "deepest_articular_overlap_mm": (
                        (record["deepest_articulation"] or {}).get("depth_mm")),
                    "deepest_articular_overlap_at_deg": (
                        (record["deepest_articulation"] or {}).get("flexion_deg")),
                    "internal_rotation_at_limit_deg":
                        record["at_limit"]["internal_rotation_deg"],
                    "medial_ap_at_limit_mm": record["at_limit"]["medial_ap_mm"],
                    "lateral_ap_at_limit_mm": record["at_limit"]["lateral_ap_mm"],
                    "evaluations": record["search"]["evaluations"],
                })
            env = result["envelope"]
            if philosophy == PHILOSOPHIES[0]:
                extents.append(_extent_row(case, result["scan_extent"]))
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
    ctx.table("e10a_scan_extent", extents, digits=1)

    ctx.note("Impingement-limited maximum flexion (deg), geometric limit of the parts "
             "under each prescribed trajectory. Stop rule (author ruling #26): each "
             "search runs until the patient's own scanned models make blocking contact, "
             "bounded only by the end of the motion, "
             f"{settings['stop_deg'] if settings else '?'} deg (tibia folded onto the "
             "femur; past it the motion would carry the tibia through the femur's axis). "
             f"Step {settings['step_deg'] if settings else '?'} deg, bisected to "
             f"{settings['resolution_deg'] if settings else '?'} deg. The bones are the "
             "whole scans, each cut only beyond the other bone's reach from the hinge "
             "axis (e10a_scan_extent). A search that reaches the bound untouched is "
             "reported as 'no contact in the scanned models', with no flexion value, "
             "and the reason measured from the models.")
    pivots = {r["case"]: r for r in rows if r["trajectory"] == "S1"}
    for case_id, r in pivots.items():
        if r["pivot_scaling"] == "assumed":
            ctx.note(f"- {case_id} S1 pivot {r['pivot_ml_mm']:.2f} mm medial of the hinge "
                     f"midpoint: 26.4 x {r['centre_spacing_mm']:.2f} / 52.8, the knee's "
                     "posterior-condyle spacing along the flexion axis (scaling assumed, "
                     "author ruling 2026-10-26).")
        else:
            ctx.note(f"- {case_id} S1 pivot NOT scaled ({r['pivot_scaling']}): the "
                     f"fitted {r['pivot_ml_mm']:.1f} mm is used.")
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
                     f"{_summary_stats(r['max_flexion_deg'] for r in sel)} "
                     "(contact-limited searches only); no contact in the scanned "
                     f"models {censored} (shown as >= 170 deg, never fitted); first pair "
                     + (", ".join(f"{k} {v}" for k, v in sorted(pairs.items()))
                        or "none"))
    for r in rows:
        if r["censored"]:
            ctx.note(f"  - {r['case']} {r['philosophy']} {r['trajectory']}: "
                     f"{r['no_contact_text']}")
    for e in extents:
        ctx.note(f"- {e['case']} scan: femur {e['femur_scanned_from_joint_line_mm']:.0f} "
                 f"mm above its distal condyles ({e['femur_scanned_from_hinge_mm']:.0f} "
                 f"mm above the hinge), tibia {e['tibia_scanned_from_joint_line_mm']:.0f}"
                 f" mm below its highest point ({e['tibia_scanned_from_hinge_mm']:.0f} mm "
                 f"below the hinge); kept in the model to "
                 f"{e['femur_kept_from_hinge_mm']:.0f} and "
                 f"{e['tibia_kept_from_hinge_mm']:.0f} mm from the hinge.")
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
    depths = [r["deepest_articular_overlap_mm"] for r in rows
              if r["deepest_articular_overlap_mm"] is not None]
    ctx.note("- Deepest overlap of the femoral component in the insert's articular "
             "surface over each searched arc (the prescribed motion's error, not "
             f"impingement): {_summary_stats(depths)} mm; over 2 mm in "
             f"{sum(d > 2.0 for d in depths)} of {len(depths)} searches.")
    cropped = [r for r in rows if r.get("crop_limited")]
    if cropped:
        ctx.note(f"- {len(cropped)} limiting contacts touch the end of a shaft as "
                 "scanned (or as kept): bone beyond it could have touched first.")


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
                    "max_flexion_deg": record["flexion_limit_deg"],
                    "censored": record["censored"],
                    "stop_reason": record["stop_reason"],
                    **_contact_columns(record),
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
                        "max_flexion_deg": record["flexion_limit_deg"],
                        "censored": record["censored"],
                        "stop_reason": record["stop_reason"],
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
             f"Bellemans 2002: about {BELLEMANS_DEG_PER_MM} deg/mm, a figure found only "
             "in secondary citations and not verified against the paper (which reports "
             "a correlation, and impingement of the posterior femur on the insert's "
             f"posterior edge in 72% of knees); Massin and Gournay 2006: about "
             f"{MASSIN_DEG_PER_MM:.1f} deg/mm from a geometric model. E10d explains "
             "the modelled slope.")
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


# ----------------------------------------------------------------------
# E10d
# ----------------------------------------------------------------------

DEG_PER_RAD = float(np.degrees(1.0))
E10D_OFFSETS_MM = (-2.0, 0.0, 2.0)
TIBIAL_SLOPE_DELTAS_DEG = (-3.0, 0.0, 3.0, 6.0)
# Femoral AP translation on the tibia at 109 deg (hinge plus a pure translation): -6 is
# 6 mm of rollback, +3 a paradoxical anterior slide.
ROLL_TRANSLATIONS_MM = (-6.0, 3.0)
THICKENINGS_MM = (-2.0, 0.0, 2.0)
MASSIN_DEG_PER_MM = 10.0 / 3.0
BELLEMANS_2005_DEG_PER_SLOPE_DEG = 1.7
MASSIN_DEG_PER_SLOPE_DEG = 1.0


def _thicker_posterior_condyle(model, thicken_mm: float, posterior):
    """The flexion model with the femoral component's posterior condyles ``thicken_mm``
    thicker and the bone left where it is.

    A posterior condyle thickened by a uniform posterior shift of its surface moves the
    centre of its arc, and so the joint it bears, posteriorly by the same amount: in
    flexion the tibia is pushed that far back, along the component's AP axis. The
    bearing passes from the distal to the posterior condyle as the knee flexes, so the
    push is ramped in over the first 90 deg (``min(1, flexion / 90)``); past 90 deg it
    is whole, which is where every limit lies. The component's own solid is not
    re-shaped, so its posterior lip against the insert's is not re-tested: a stated
    approximation, harmless where the limit is bone against bone or insert against bone.
    """
    from dataclasses import fields

    from tka_planner.pipeline import FlexionModel

    class Pushed(FlexionModel):
        def pose(self, trajectory, flexion_deg):
            base = FlexionModel.pose(self, trajectory, flexion_deg)
            shift = np.eye(4)
            ramp = min(1.0, max(0.0, float(flexion_deg) / 90.0))
            shift[:3, 3] = thicken_mm * ramp * np.asarray(posterior, dtype=float)
            return shift @ base

    return Pushed(**{f.name: getattr(model, f.name) for f in fields(model)})


def _limit(model, trajectory):
    from tka_planner.core.kinematics import search_max_flexion

    s = model.settings
    t = model.trajectory(trajectory)      # S1 scaled to this knee
    return search_max_flexion(lambda a: model.contacts(t, a), trajectory=t.name,
                              start_deg=s.start_deg, stop_deg=s.stop_deg,
                              step_deg=s.step_deg, resolution_deg=s.resolution_deg)


def _lever_row(model, limit, anterior_world) -> dict:
    """The contact that stops flexion, seen from the hinge, and the slope the lever
    arm predicts for it.

    For a contact at distance ``r`` from the flexion axis, moving the femoral surface by
    ``d`` mm along the AP axis ``a`` moves the angle at which the tibia reaches it by
    ``d (a.n) / (r (t.n))`` rad, where ``n`` is the contact normal and ``t`` the
    direction the tibial point travels as the knee flexes; when ``a``, ``n`` and ``t``
    line up this is ``d / r``: ``57.3 / r`` deg per mm.
    """
    if limit.censored or not limit.contacts:
        return {"stop_reason": limit.stop_reason}
    contact = max(limit.contacts, key=lambda c: c.volume_mm3)
    d = contact.diagnostics
    basis = model.femoral_frame.matrix[:3, :3]
    origin = model.pivot[:3, 3]
    axis = model.pivot[:3, 0] / np.linalg.norm(model.pivot[:3, 0])
    from_hinge = np.asarray(d["from_hinge_mm"], dtype=float)
    centre = origin + basis @ from_hinge
    radius = float(d["hinge_radius_mm"])
    travel = np.cross(axis, centre - origin)
    travel = (travel / np.linalg.norm(travel)) @ basis
    a = np.asarray(anterior_world, dtype=float) @ basis
    row = {
        "stop_reason": limit.stop_reason,
        "max_flexion_deg": limit.flexion_limit_deg,
        "pair": contact.pair,
        "compartment": d.get("compartment"),
        "contact_anterior_of_hinge_mm": round(float(from_hinge[0]), 2),
        "contact_proximal_of_hinge_mm": round(float(from_hinge[2]), 2),
        "r_mm": radius,
        "slope_57_3_over_r": round(DEG_PER_RAD / radius, 3),
    }
    normal = d.get("thin_direction")
    if normal is not None:
        n = np.asarray(normal, dtype=float)
        along = abs(float(travel @ n))
        row["normal_vs_travel_deg"] = round(float(np.degrees(np.arccos(min(1.0, along)))),
                                            1)
        if along > 1e-3:
            row["slope_with_normal"] = round(
                DEG_PER_RAD * abs(float(a @ n)) / (radius * along), 3)
    return row


def _component_reach(model, anterior_world) -> dict:
    """How close to the flexion axis any femoral contact could lie: the femoral
    component's posterior condyle (its posterior-most point behind the hinge) and the top
    of its posterior condyles (their most proximal point behind the hinge)."""
    vertices = model.femoral_component.to_mesh().vertices
    origin = model.pivot[:3, 3]
    local = (vertices - origin) @ model.femoral_frame.matrix[:3, :3]
    posterior = local[:, 0] < 0.0
    behind = float(-((vertices - origin) @ np.asarray(anterior_world, dtype=float)).min())
    return {
        "posterior_condyle_behind_hinge_mm": round(behind, 2),
        "posterior_condyle_top_above_hinge_mm": round(
            float(local[posterior, 2].max()), 2) if posterior.any() else None,
    }


@experiment("E10d", "Why the modelled offset slope is what it is (ruling #24)")
def run_e10d(ctx: Context) -> None:
    from tka_planner.core.condylar_offset import planned_condylar_offset
    from tka_planner.core.kinematics import NEUTRAL, with_parameters
    from tka_planner.core.planning import Adjustments
    from tka_planner.pipeline import flexion_model, plan_case

    library = _library(ctx)
    offsets = QUICK_OFFSETS_MM if ctx.quick else E10D_OFFSETS_MM
    rolls = {f"AP{t:+.0f}": with_parameters(
        NEUTRAL, name=f"AP{t:+.0f}", label="hinge plus AP translation",
        ap_translation_mm=t, reference_flexion_deg=109.0) for t in ROLL_TRANSLATIONS_MM}
    trajectories = {**{name: name for name in SCENARIOS}, **rolls}
    lever_rows, sweep, slopes, slope_rows, method_rows, reach_rows = [], [], [], [], [], []

    for case, m in _accepted(ctx):
        base_plan, sizing = plan_case(m, philosophy="mechanical", library=library)
        anterior = np.asarray(base_plan.components["femoral_component"],
                              dtype=float)[:3, 0]
        anterior = anterior / np.linalg.norm(anterior)

        # 1. The offset sweep under every trajectory, and the lever at the plan.
        pco_at = {}
        for offset in offsets:
            model = flexion_model(m, base_plan, sizing, library, femoral_offset_mm=offset)
            if model is None:
                break
            component = model.femoral_component.to_mesh().vertices + offset * anterior
            pco_at[offset] = planned_condylar_offset(
                m.femur.vertices, component, m.landmarks,
                m.femoral_frame)["radiographic"]["planned_mm"]
            if offset == 0.0:
                reach_rows.append({"case": case.case_id,
                                   **_component_reach(model, anterior)})
                plan_model = model
            for name, trajectory in trajectories.items():
                limit = _limit(model, trajectory)
                sweep.append({"case": case.case_id, "trajectory": name,
                              "offset_mm": offset, "pco_mm": pco_at[offset],
                              "max_flexion_deg": limit.flexion_limit_deg,
                              "stop_reason": limit.stop_reason,
                              "pair": limit.first_pair})
                if offset == 0.0:
                    lever_rows.append({"case": case.case_id, "trajectory": name,
                                       **_lever_row(model, limit, anterior)})
        pco_per_mm = None
        if len(pco_at) >= 2:
            xs = sorted(pco_at)
            pco_per_mm = (pco_at[xs[0]] - pco_at[xs[-1]]) / (xs[-1] - xs[0])
        for name in trajectories:
            sel = [r for r in sweep if r["case"] == case.case_id
                   and r["trajectory"] == name and r["max_flexion_deg"] is not None]
            slope, r2 = _slope([r["pco_mm"] for r in sel],
                               [r["max_flexion_deg"] for r in sel])
            if len(sel) == 2:
                slope = ((sel[1]["max_flexion_deg"] - sel[0]["max_flexion_deg"])
                         / (sel[1]["pco_mm"] - sel[0]["pco_mm"]))
            slopes.append({"case": case.case_id, "trajectory": name,
                           "n_points": len(sel), "measured_deg_per_mm": slope,
                           "r2": r2})

        # 2. Tibial slope: maximum flexion and the offset slope at each.
        for delta in TIBIAL_SLOPE_DELTAS_DEG:
            plan = base_plan if delta == 0.0 else plan_case(
                m, philosophy="mechanical", library=library,
                adjustments=Adjustments(tibial_slope_delta_deg=delta))[0]
            points = []
            for offset in (offsets[0], 0.0, offsets[-1]):
                if delta == 0.0:
                    found = [r for r in sweep if r["case"] == case.case_id
                             and r["trajectory"] == "S3" and r["offset_mm"] == offset]
                    theta = found[0]["max_flexion_deg"] if found else None
                    pair = found[0]["pair"] if found else None
                else:
                    model = flexion_model(m, plan, sizing, library,
                                          femoral_offset_mm=offset)
                    limit = _limit(model, "S3")
                    theta, pair = limit.flexion_limit_deg, limit.first_pair
                points.append((offset, theta, pair))
            mid = points[1]
            ends = [p for p in (points[0], points[2]) if p[1] is not None]
            offset_slope = None
            if len(ends) == 2 and pco_per_mm:
                offset_slope = -(ends[1][1] - ends[0][1]) / (
                    (ends[1][0] - ends[0][0]) * pco_per_mm)
            slope_rows.append({"case": case.case_id, "tibial_slope_delta_deg": delta,
                               "tibial_slope_deg": round(plan.tibial_slope_deg, 2),
                               "max_flexion_s3_deg": mid[1], "pair": mid[2],
                               "offset_slope_deg_per_mm": offset_slope})

        # 3. How offset is changed: component moved (femur re-seated and re-cut) against
        #    a thicker posterior condyle with the bone fixed.
        for thicken in THICKENINGS_MM:
            for name in ("S3", "S2"):
                pushed = _thicker_posterior_condyle(plan_model, thicken, -anterior)
                limit = _limit(pushed, name)
                method_rows.append({
                    "case": case.case_id, "trajectory": name, "thicken_mm": thicken,
                    "pco_change_mm": thicken, "max_flexion_deg": limit.flexion_limit_deg,
                    "stop_reason": limit.stop_reason, "pair": limit.first_pair})

    ctx.table("e10d_lever", lever_rows, digits=3)
    ctx.table("e10d_offset_sweep", sweep, digits=3)
    ctx.table("e10d_offset_slopes", slopes, digits=3)
    ctx.table("e10d_tibial_slope", slope_rows, digits=3)
    ctx.table("e10d_method", method_rows, digits=3)
    ctx.table("e10d_component_reach", reach_rows, digits=2)

    ctx.note("Why the modelled slope of maximum flexion against posterior condylar "
             "offset is what it is (author ruling #24); mechanical plans, offsets "
             f"{', '.join(f'{o:+.0f}' for o in offsets)} mm. Hypothesis: a contact at "
             "distance r from the flexion axis moves by dPCO/r rad per mm of offset, so "
             "the slope is about 57.3/r deg/mm (with the contact normal n and the "
             "tibial point's direction of travel t: 57.3 |a.n| / (r |t.n|)).")
    for case_id in sorted({r["case"] for r in lever_rows}):
        for row in [r for r in lever_rows if r["case"] == case_id]:
            measured = next((s["measured_deg_per_mm"] for s in slopes
                             if s["case"] == case_id
                             and s["trajectory"] == row["trajectory"]), None)
            if row.get("r_mm") is None:
                ctx.note(f"- {case_id} {row['trajectory']}: no contact in the scanned "
                         "models; no lever.")
                continue
            ratio = (None if measured is None else measured / row["slope_57_3_over_r"])
            ctx.note(
                f"- {case_id} {row['trajectory']}: {row['pair']} ({row['compartment']}) "
                f"at {row['max_flexion_deg']:.1f} deg, "
                f"{row['contact_proximal_of_hinge_mm']:.0f} mm proximal and "
                f"{row['contact_anterior_of_hinge_mm']:+.0f} mm anterior of the hinge, "
                f"r = {row['r_mm']:.1f} mm; 57.3/r = {row['slope_57_3_over_r']:.2f}, "
                f"with the normal {row.get('slope_with_normal', float('nan')):.2f}; "
                "measured "
                + ("n/a" if measured is None else f"{measured:.2f}")
                + " deg/mm"
                + ("" if ratio is None else f" (measured / (57.3/r) = {ratio:.2f})")
                + ".")
    for row in reach_rows:
        behind = row["posterior_condyle_behind_hinge_mm"]
        ctx.note(f"- {row['case']}: the component's posterior condyle lies "
                 f"{behind:.1f} mm behind the hinge, so no contact on the posterior femur "
                 f"can be nearer the axis than that: the steepest slope rigid geometry "
                 f"allows there is about 57.3/{behind:.0f} = {DEG_PER_RAD / behind:.1f} "
                 f"deg/mm. Bellemans' 6.1 deg/mm would need r = "
                 f"{DEG_PER_RAD / BELLEMANS_DEG_PER_MM:.1f} mm; Massin's "
                 f"{MASSIN_DEG_PER_MM:.1f} deg/mm, r = "
                 f"{DEG_PER_RAD / MASSIN_DEG_PER_MM:.1f} mm.")
    for case_id in sorted({r["case"] for r in slope_rows}):
        sel = [r for r in slope_rows if r["case"] == case_id
               and r["max_flexion_s3_deg"] is not None]
        per_degree, _ = _slope([r["tibial_slope_deg"] for r in sel],
                               [r["max_flexion_s3_deg"] for r in sel])
        ctx.note(f"- {case_id} tibial slope (S3): max flexion "
                 + ", ".join(f"{r['tibial_slope_deg']:.1f} deg -> "
                             + ("none" if r["max_flexion_s3_deg"] is None
                                else f"{r['max_flexion_s3_deg']:.1f}")
                             for r in slope_rows if r["case"] == case_id)
                 + "; " + ("n/a" if per_degree is None else f"{per_degree:.2f}")
                 + " deg of flexion per deg of slope (Bellemans 2005 cadaver "
                 f"{BELLEMANS_2005_DEG_PER_SLOPE_DEG}, Massin about "
                 f"{MASSIN_DEG_PER_SLOPE_DEG}); offset slope at each tibial slope "
                 + ", ".join("n/a" if r["offset_slope_deg_per_mm"] is None
                             else f"{r['offset_slope_deg_per_mm']:.2f}"
                             for r in slope_rows if r["case"] == case_id)
                 + " deg/mm.")
    for case_id in sorted({r["case"] for r in slopes}):
        ctx.note(f"- {case_id} offset slope by trajectory (deg/mm): "
                 + ", ".join(f"{s['trajectory']} "
                             + ("n/a" if s["measured_deg_per_mm"] is None
                                else f"{s['measured_deg_per_mm']:.2f}")
                             for s in slopes if s["case"] == case_id)
                 + " (AP-6: 6 mm rollback at 109 deg; AP+3: 3 mm paradoxical anterior "
                 "slide).")
    for case_id in sorted({r["case"] for r in method_rows}):
        for name in ("S3", "S2"):
            sel = [r for r in method_rows if r["case"] == case_id
                   and r["trajectory"] == name and r["max_flexion_deg"] is not None]
            slope, _ = _slope([r["pco_change_mm"] for r in sel],
                              [r["max_flexion_deg"] for r in sel])
            moved = next((s["measured_deg_per_mm"] for s in slopes
                          if s["case"] == case_id and s["trajectory"] == name), None)
            ctx.note(f"- {case_id} {name}: thicker posterior condyle, bone fixed: "
                     + ("n/a" if slope is None else f"{slope:.2f}")
                     + " deg/mm; component moved: "
                     + ("n/a" if moved is None else f"{moved:.2f}") + " deg/mm.")

    # What geometry alone can and cannot produce, from the numbers above.
    paired = []
    for row in lever_rows:
        measured = next((s["measured_deg_per_mm"] for s in slopes
                         if s["case"] == row["case"]
                         and s["trajectory"] == row["trajectory"]), None)
        if row.get("r_mm") is not None and measured is not None:
            paired.append((row["r_mm"], measured * row["r_mm"] / DEG_PER_RAD))
    if paired:
        r = [p[0] for p in paired]
        ratio = [p[1] for p in paired]
        tops = [row["posterior_condyle_top_above_hinge_mm"] for row in reach_rows
                if row["posterior_condyle_top_above_hinge_mm"] is not None]
        behind = [row["posterior_condyle_behind_hinge_mm"] for row in reach_rows]
        ctx.note(
            "Summary. In this model flexion stops where the posterior edge of the insert "
            "(or the posterior tibia) meets uncovered posterior femoral bone "
            f"{min(r):.0f} to {max(r):.0f} mm from the flexion axis"
            + (f", well above the top of the component's posterior condyle (which ends "
               f"{min(tops):.1f} to {max(tops):.1f} mm above the hinge)" if tops else "")
            + ". The lever-arm rule 57.3/r reproduces the measured slopes (measured / "
            f"predicted {_summary_stats(ratio)}): the slope is set by where the contact "
            "is, not by how the offset is changed (moving the component and thickening "
            "the posterior condyle give the same slope), and rollback, which moves the "
            "contact farther out, lowers it. Geometry alone cannot give 6.1 deg/mm: that "
            f"needs r = {DEG_PER_RAD / BELLEMANS_DEG_PER_MM:.1f} mm, inside the posterior "
            f"condyle, which lies {min(behind):.0f} to {max(behind):.0f} mm behind the "
            "axis; even Massin's 3.3 deg/mm (r = "
            f"{DEG_PER_RAD / MASSIN_DEG_PER_MM:.0f} mm) is out of reach of a rigid "
            "hinge about the epicondylar axis. A slope that steep would need a different "
            "mechanism: the offset changing the kinematics themselves (rollback, or the "
            "flexion axis moving with the offset), soft tissue or thigh-calf contact "
            "ending flexion first, or the published figure being a between-group "
            "difference rather than a within-knee slope.")
