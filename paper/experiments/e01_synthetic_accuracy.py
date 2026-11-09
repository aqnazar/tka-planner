"""E1: metric accuracy on synthetic knees whose anatomy is known by construction.

Each knee is built backwards from a stated mLDFA, MPTA, femoral AMA, condylar twist,
posterior slope, femoral and tibial width and side
(:func:`tka_planner.validation.synthetic.synthetic_knee`), then measured and planned
through :func:`~tka_planner.pipeline.measure_from_landmarks` and
:func:`~tka_planner.pipeline.plan_case`, the functions the application calls. The
reported value is compared with the construction.

Two scans of every knee are measured. **Full length** has the femoral head and the
ankle centre, so nothing is assumed and the error is the formulas' own: it should be
zero to rounding. **Knee only** has neither, as in the cohort, so the femoral
mechanical axis is the shaft turned by the population AMA of 6 degrees; its error
against the truth is the cost of that assumption, and its error against what the
declared assumption predicts (the "residual") is again the formulas' own.

Grids
-----
``anatomy``    five levels of each of the five angles, at reference widths: 3125 knees
               per side and scan, reaching mLDFA and MPTA 80-95, AMA 2-11, twist -4 to
               10 and slope -4 to 18 degrees.
``widths``     femoral width 60/75/90 mm by tibial width 60/72/85 mm, crossed with the
               extremes of the angles.
``tibial_axis`` the tibial canal tilted off the mechanical axis (coronal +/-4, sagittal
               +/-6 degrees), which the knee-only planner assumes collinear.
``pose``       the whole knee turned in the scanner (flexion +/-20, abduction +/-15,
               axial +/-30 degrees); every metric should be unchanged. This grid found
               the knee-only femoral axis taking its coronal plane from the scanner
               rather than the femur (mLDFA off by 0.8 degrees at 30 degrees axial),
               since fixed in ``core/frames.py``.

Outputs: ``errors`` (per grid, scan, side and output: mean and max absolute error and
the worst parameter combination), ``provenance`` (the tier of every metric on a
knee-only and a full-length scan, picked and automatic), ``pose`` and ``knees`` (every
knee, every error), and two figures.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..harness import Context, experiment

ANGLES = ("mldfa_deg", "mpta_deg", "ama_deg", "condylar_twist_deg", "posterior_slope_deg")
LEVELS = {
    "mldfa_deg": (80.0, 84.0, 87.5, 91.0, 95.0),
    "mpta_deg": (80.0, 84.0, 87.0, 91.0, 95.0),
    "ama_deg": (2.0, 4.0, 6.0, 8.0, 11.0),
    "condylar_twist_deg": (-4.0, 0.0, 3.0, 7.0, 10.0),
    "posterior_slope_deg": (-4.0, 0.0, 7.0, 12.0, 18.0),
}
QUICK_LEVELS = {k: (v[0], v[2], v[-1]) for k, v in LEVELS.items()}
WIDTHS = {"femoral_width_mm": (60.0, 75.0, 90.0), "tibial_width_mm": (60.0, 72.0, 85.0)}
TIBIAL_TILTS = {"tibial_ama_deg": (-4.0, -2.0, 0.0, 2.0, 4.0),
                "tibial_ama_sagittal_deg": (-6.0, -3.0, 0.0, 3.0, 6.0)}
POSES = {"flexion_deg": (-20.0, 0.0, 20.0), "abduction_deg": (-15.0, 0.0, 15.0),
         "axial_deg": (-30.0, -15.0, 0.0, 15.0, 30.0)}
SCANS = {"knee_only": dict(include_head=False, include_ankle=False),
         "full_length": dict(include_head=True, include_ankle=True)}
SIDES = ("left", "right")

METRICS = ("mldfa_deg", "aldfa_deg", "femoral_ama_deg", "mpta_deg", "hka_deviation_deg",
           "jlca_deg", "condylar_twist_deg", "posterior_slope_medial_deg")
PLAN = ("plan.valgus_cut_deg", "plan.femoral_cut_coronal_deg",
        "plan.tibial_cut_coronal_deg", "plan.tibial_cut_slope_deg",
        "plan.femoral_cut_flexion_deg")
# Metrics are reported rounded to 0.001 degrees, so an error of exactly zero means
# below 0.0005; the plan's angles are not rounded.
EXACT = "none: exact at the reported precision"
# The mechanical target: cuts square to the mechanical axes, 3 degrees of tibial slope.
TARGET_SLOPE_DEG = 3.0


def _knee(side, scan, **params):
    from tka_planner.validation.synthetic import synthetic_knee
    return synthetic_knee(side, **SCANS[scan], **params)


def _measure(landmarks, chart):
    from tka_planner.pipeline import measure_from_landmarks, plan_case
    from tka_planner.validation.synthetic import bone_like

    m = measure_from_landmarks(
        landmarks, bone_like(landmarks, "femur"), bone_like(landmarks, "tibia"),
        femur_path="synthetic_femur", tibia_path="synthetic_tibia",
        case_id=landmarks.case_id, side=landmarks.side, size_chart=chart,
    )
    plan, _ = plan_case(m)
    return m, plan


def _truth_and_model(landmarks, scan):
    """The true value of every output, and what the planner's declared assumptions
    predict it reports. On a full-length scan the two are the same."""
    from tka_planner.core.provenance import POPULATION_FEMORAL_AMA
    from tka_planner.validation.synthetic import expected_metrics

    built = landmarks.source["constructed"]
    truth = dict(expected_metrics(landmarks))
    truth.update({
        "plan.valgus_cut_deg": built["ama_deg"],
        "plan.femoral_cut_coronal_deg": 0.0,
        "plan.tibial_cut_coronal_deg": 0.0,
        "plan.tibial_cut_slope_deg": TARGET_SLOPE_DEG,
        "plan.femoral_cut_flexion_deg": 0.0,
    })
    model = dict(truth)
    if scan == "knee_only":
        # The femoral axis is the shaft turned by the assumed AMA; the true one is the
        # shaft turned by the true AMA. The estimate therefore leans laterally by the
        # difference, which closes mLDFA and tilts both cuts (they share the femoral
        # coronal reference) toward valgus by it.
        excess = built["ama_deg"] - POPULATION_FEMORAL_AMA.value
        model["mldfa_deg"] = truth["mldfa_deg"] - excess
        model["plan.valgus_cut_deg"] = POPULATION_FEMORAL_AMA.value
        model["plan.femoral_cut_coronal_deg"] = excess
        model["plan.tibial_cut_coronal_deg"] = excess
        for name in ("femoral_ama_deg", "hka_deviation_deg"):
            truth.pop(name)
            model.pop(name)
    return truth, model


def _reported(m, plan, landmarks):
    """Every output as reported, with the cuts read against the true axes.

    The constructed mechanical axes are world +Z for both bones and lateral is the
    side's lateral, so cut angles are read in that frame. Signs follow the planner's
    adjustments: + valgus (normal toward lateral), + posterior.
    """
    from tka_planner.core.sides import LPS_PATIENT_LEFT

    out = {name: metric.value for name, metric in m.metrics.items()
           if metric.value is not None}
    lateral = landmarks.side.lateral_direction(LPS_PATIENT_LEFT)
    up, posterior = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])

    def tilt(normal, towards):
        return float(np.degrees(np.arctan2(normal @ towards, normal @ up)))

    femoral = np.asarray(plan.resections["femoral_distal"].normal, dtype=float)
    tibial = np.asarray(plan.resections["tibial_proximal"].normal, dtype=float)
    out.update({
        "plan.valgus_cut_deg": float(plan.distal_femoral_valgus_cut_deg),
        "plan.femoral_cut_coronal_deg": tilt(femoral, lateral),
        "plan.tibial_cut_coronal_deg": tilt(tibial, lateral),
        "plan.tibial_cut_slope_deg": tilt(tibial, posterior),
        "plan.femoral_cut_flexion_deg": tilt(femoral, posterior),
    })
    return out


def _label(params: dict) -> str:
    short = {"mldfa_deg": "mLDFA", "mpta_deg": "MPTA", "ama_deg": "AMA",
             "condylar_twist_deg": "twist", "posterior_slope_deg": "slope",
             "femoral_width_mm": "femW", "tibial_width_mm": "tibW",
             "tibial_ama_deg": "tibAMA", "tibial_ama_sagittal_deg": "tibAMAsag",
             "flexion_deg": "flex", "abduction_deg": "abd", "axial_deg": "axial"}
    return " ".join(f"{short.get(k, k)}={v:g}" for k, v in params.items())


def _grid(levels: dict) -> list[dict]:
    keys = list(levels)
    return [dict(zip(keys, values)) for values in itertools.product(*levels.values())]


def _sweep(grid_name, combos, chart, knees, *, use_model=True):
    """Measure every combination on both sides and both scans; append to ``knees``."""
    for scan in SCANS:
        for side in SIDES:
            for params in combos:
                landmarks = _knee(side, scan, **params)
                m, plan = _measure(landmarks, chart)
                reported = _reported(m, plan, landmarks)
                truth, model = _truth_and_model(landmarks, scan)
                row = {"grid": grid_name, "scan": scan, "side": side, **params}
                for name, true_value in truth.items():
                    value = reported.get(name)
                    row[f"{name}|reported"] = value
                    row[f"{name}|error"] = None if value is None else value - true_value
                    if use_model:
                        row[f"{name}|residual"] = (None if value is None
                                                   else value - model[name])
                knees.append(row)


def _summarise(knees, params_of) -> list[dict]:
    rows = []
    groups: dict = {}
    for knee in knees:
        groups.setdefault((knee["grid"], knee["scan"], knee["side"]), []).append(knee)
    for (grid, scan, side), members in groups.items():
        outputs = sorted({k.split("|")[0] for k in members[0] if k.endswith("|error")},
                         key=lambda n: (n.startswith("plan."), n))
        for name in outputs:
            errors = np.array([np.nan if k.get(f"{name}|error") is None
                               else k[f"{name}|error"] for k in members], dtype=float)
            if np.all(np.isnan(errors)):
                rows.append({"grid": grid, "scan": scan, "side": side, "output": name,
                             "n": 0, "note": "not computable"})
                continue
            worst = int(np.nanargmax(np.abs(errors)))
            row = {"grid": grid, "scan": scan, "side": side, "output": name,
                   "n": int(np.sum(~np.isnan(errors))),
                   "mean_abs_error": float(np.nanmean(np.abs(errors))),
                   "max_abs_error": float(np.abs(errors[worst])),
                   "worst_signed_error": float(errors[worst]),
                   "worst_case": _label(params_of(members[worst]))
                   if errors[worst] != 0 else EXACT}
            residuals = [k.get(f"{name}|residual") for k in members]
            if any(r is not None for r in residuals):
                residuals = np.array([np.nan if r is None else r for r in residuals])
                worst_r = int(np.nanargmax(np.abs(residuals)))
                row.update(mean_abs_residual=float(np.nanmean(np.abs(residuals))),
                           max_abs_residual=float(np.abs(residuals[worst_r])),
                           worst_residual_case=_label(params_of(members[worst_r]))
                           if residuals[worst_r] != 0 else EXACT)
            rows.append(row)
    return rows


def _provenance_rows() -> list[dict]:
    """The tier every metric reaches, by scan and by how the landmarks were obtained.

    "Automatic" marks every landmark as an estimator's and leaves the medial sulcus
    unpicked, which is what the automatic estimator produces on a real knee.
    """
    from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
    from tka_planner.core.landmarks import Landmark, LandmarkSet, LandmarkStatus
    from tka_planner.core.metrics import compute_all

    def automatic(landmarks):
        rebuilt = []
        for lm in landmarks:
            if lm.id == "femur.epicondyle_medial_sulcus":
                rebuilt.append(Landmark(id=lm.id, status=LandmarkStatus.NOT_PICKED,
                                        reason="requires a human pick"))
            elif lm.is_usable:
                rebuilt.append(Landmark(id=lm.id, position_mm=lm.position_mm,
                                        status=LandmarkStatus.ESTIMATED,
                                        origin="synthetic_automatic"))
            else:
                rebuilt.append(lm)
        return LandmarkSet(landmarks.case_id, landmarks.side, landmarks.coordinate_system,
                           rebuilt, source=landmarks.source)

    scenarios = {}
    for scan in SCANS:
        picked = _knee("left", scan)
        scenarios[f"{scan}, picked"] = picked
        scenarios[f"{scan}, automatic"] = automatic(picked)
    tiers = {}
    for label, landmarks in scenarios.items():
        frames = (build_femoral_frame(landmarks), build_tibial_frame(landmarks))
        tiers[label] = (compute_all(landmarks, *frames), frames)

    rows = []
    for name in METRICS:
        row = {"metric": name}
        for label, (metrics, _) in tiers.items():
            metric = metrics[name]
            assumed = ",".join(a.id for a in metric.assumptions)
            row[label] = metric.quality.value + (f" [{assumed}]" if assumed else "")
        rows.append(row)
    for bone, index in (("femoral frame", 0), ("tibial frame", 1)):
        row = {"metric": bone}
        for label, (_, frames) in tiers.items():
            frame = frames[index]
            row[label] = f"{frame.quality.value} ({frame.method})"
        rows.append(row)
    return rows


def _pose_rows(knees_out: list, quick: bool) -> list[dict]:
    """Every metric's change when the whole knee is turned in the scanner."""
    from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
    from tka_planner.core.metrics import compute_all
    from tka_planner.validation.synthetic import posed, scanner_rotation

    poses = _grid({k: (v[0], v[-1]) if quick else v for k, v in POSES.items()})
    anatomies = [{}, {"mldfa_deg": 95.0, "mpta_deg": 80.0, "ama_deg": 11.0,
                      "condylar_twist_deg": 10.0, "posterior_slope_deg": 18.0},
                 {"mldfa_deg": 80.0, "mpta_deg": 95.0, "ama_deg": 2.0,
                  "condylar_twist_deg": -4.0, "posterior_slope_deg": -4.0}]
    rows = []
    for scan in SCANS:
        for side in SIDES:
            worst: dict = {}
            for anatomy in anatomies:
                knee = _knee(side, scan, **anatomy)
                square = compute_all(knee, build_femoral_frame(knee), build_tibial_frame(knee))
                for pose in poses:
                    turned_knee = posed(knee, scanner_rotation(**pose))
                    turned = compute_all(turned_knee, build_femoral_frame(turned_knee),
                                         build_tibial_frame(turned_knee))
                    for name, metric in square.items():
                        if metric.value is None:
                            continue
                        change = turned[name].value - metric.value
                        knees_out.append({"scan": scan, "side": side, "metric": name,
                                          **pose, "change": change})
                        if abs(change) >= abs(worst.get(name, (0.0,))[0]):
                            worst[name] = (change, _label({**anatomy, **pose}))
            for name in METRICS:
                if name in worst:
                    rows.append({"scan": scan, "side": side, "metric": name,
                                 "max_abs_change": abs(worst[name][0]),
                                 "worst_signed_change": worst[name][0],
                                 "worst_case": (worst[name][1] or "reference anatomy")
                                 if worst[name][0] != 0 else EXACT})
    return rows


def _figures(ctx: Context, errors: list[dict], knees: list[dict]) -> bool:
    """Draw the figures; False when matplotlib is not installed (the tables stand)."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anatomy = [r for r in errors if r["grid"] == "anatomy" and r.get("n")]
    names = sorted({r["output"] for r in anatomy}, key=lambda n: (n.startswith("plan."), n))
    series = (("full_length", "max_abs_error", "full length: error", "#1f5f8b"),
              ("knee_only", "max_abs_residual", "knee only: residual", "#4f9d69"),
              ("knee_only", "max_abs_error", "knee only: error vs truth", "#c8553d"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    width = 0.27
    floor = 1e-6
    for i, (scan, key, label, colour) in enumerate(series):
        values = []
        for name in names:
            vals = [r.get(key) for r in anatomy if r["scan"] == scan and r["output"] == name]
            vals = [v for v in vals if v is not None]
            values.append(max(max(vals), floor) if vals else np.nan)
        ax.bar(np.arange(len(names)) + (i - 1) * width, values, width, label=label,
               color=colour)
    ax.set_yscale("log")
    ax.axhline(0.01, color="#888", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([n.replace("plan.", "plan: ").replace("_deg", "") for n in names],
                       rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("max |error| over the anatomy grid (deg)")
    ax.set_title("Anatomy grid, both sides (bars floored at 1e-6)")
    ax.legend(fontsize=8)

    ax = axes[1]
    knee_only = [k for k in knees if k["grid"] == "anatomy" and k["scan"] == "knee_only"
                 and k["side"] == "left"]
    ama = np.array([k["ama_deg"] for k in knee_only])
    for name, colour, label in (("mldfa_deg", "#c8553d", "mLDFA"),
                                ("plan.femoral_cut_coronal_deg", "#1f5f8b",
                                 "femoral cut coronal")):
        err = np.array([k[f"{name}|error"] for k in knee_only])
        levels = sorted(set(ama))
        ax.plot(levels, [np.mean(err[ama == a]) for a in levels], "o-", color=colour,
                label=label)
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(6.0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("true femoral AMA (deg); planner assumes 6")
    ax.set_ylabel("error vs truth, knee-only (deg)")
    ax.set_title("Cost of the assumed AMA")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"e01_accuracy.{suffix}"), dpi=200)
    plt.close(fig)
    return True


@experiment("E1", "Metric accuracy on synthetic knees", needs_cohort=False)
def run(ctx: Context) -> None:
    from tka_planner.pipeline import find_size_chart

    chart = find_size_chart()
    levels = QUICK_LEVELS if ctx.quick else LEVELS
    knees: list[dict] = []

    _sweep("anatomy", _grid(levels), chart, knees)
    extremes = {k: (v[0], v[-1]) for k, v in LEVELS.items()}
    width_levels = {k: (v[0], v[-1]) for k, v in WIDTHS.items()} if ctx.quick else WIDTHS
    width_combos = [dict(**w, **a) for w in _grid(width_levels)
                    for a in (_grid(extremes) if not ctx.quick else [{}])]
    _sweep("widths", width_combos, chart, knees)
    tilts = {k: (v[0], v[2], v[-1]) for k, v in TIBIAL_TILTS.items()} if ctx.quick \
        else TIBIAL_TILTS
    _sweep("tibial_axis", _grid(tilts), chart, knees, use_model=False)

    keys = set(ANGLES) | set(WIDTHS) | set(TIBIAL_TILTS)

    def params_of(knee):
        return {k: knee[k] for k in knee if k in keys}

    errors = _summarise(knees, params_of)
    columns = ["grid", "scan", "side", "output", "n", "mean_abs_error", "max_abs_error",
               "worst_signed_error", "worst_case", "mean_abs_residual", "max_abs_residual",
               "worst_residual_case", "note"]
    ctx.table("errors", errors, columns, digits=6)
    knee_columns = ["grid", "scan", "side", *ANGLES, *WIDTHS, *TIBIAL_TILTS]
    knee_columns += sorted({k for knee in knees for k in knee if "|" in k})
    ctx.table("knees", knees, knee_columns, digits=6)

    provenance = _provenance_rows()
    ctx.table("provenance", provenance)

    pose_changes: list = []
    pose = _pose_rows(pose_changes, ctx.quick)
    ctx.table("pose", pose, digits=6)
    drew = _figures(ctx, errors, knees)

    # ---- Summary --------------------------------------------------------------
    counts = {}
    for knee in knees:
        counts[(knee["grid"], knee["scan"])] = counts.get((knee["grid"], knee["scan"]), 0) + 1
    ctx.note(f"{len(knees)} synthetic knees measured and planned through "
             f"measure_from_landmarks and plan_case ("
             + ", ".join(f"{g}/{s}: {n}" for (g, s), n in sorted(counts.items())) + ").")
    ctx.note("Metric values are reported to 0.001 deg, so a zero error means below 0.0005; "
             "plan angles are unrounded.")
    ctx.note("Ranges: " + "; ".join(f"{k} {v[0]:g} to {v[-1]:g}" for k, v in levels.items())
             + "; widths femur 60-90, tibia 60-85 mm; left and right.")

    def worst(rows, scan, key, grid=("anatomy", "widths")):
        picked = [r for r in rows if r["scan"] == scan and r["grid"] in grid
                  and r.get(key) is not None]
        return max(picked, key=lambda r: r[key]) if picked else None

    full = worst(errors, "full_length", "max_abs_error")
    ctx.note(f"\nFull length (nothing assumed): largest |error| over every metric and plan "
             f"output is {full['max_abs_error']:.2e} ({full['output']}, {full['side']}, "
             f"{full['worst_case']}).")
    residual = worst(errors, "knee_only", "max_abs_residual")
    ctx.note(f"Knee only, against what the declared 6 degree AMA predicts: largest |residual| "
             f"{residual['max_abs_residual']:.3g} ({residual['output']}, {residual['side']}, "
             f"{residual['worst_residual_case']}).")
    ctx.note("Non-zero residuals are second order in the AMA mismatch d = true AMA - 6: the "
             "knee-only femoral frame's transverse plane is tilted by d, so the twist read "
             "in it is off by about twist x (1 - cos d); and the tibial cut's hinge is the "
             "femoral coronal reference, so its slope against the true tibial axis is off "
             "by about 3 x (1 - cos d).")
    for name in ("mldfa_deg", "plan.femoral_cut_coronal_deg", "plan.tibial_cut_coronal_deg"):
        row = next((r for r in errors if r["grid"] == "anatomy" and r["scan"] == "knee_only"
                    and r["output"] == name and r["side"] == "left"), None)
        if row:
            ctx.note(f"Knee only, {name} vs truth: max |error| {row['max_abs_error']:.2f} "
                     f"({row['worst_case']}); this is the AMA assumption, = true AMA - 6.")
    for name in ("mpta_deg", "posterior_slope_medial_deg", "plan.tibial_cut_slope_deg"):
        row = next((r for r in errors if r["grid"] == "tibial_axis" and r["scan"] == "knee_only"
                    and r["output"] == name and r["side"] == "left"), None)
        if row:
            ctx.note(f"Knee only, tibial canal tilted: {name} max |error| "
                     f"{row['max_abs_error']:.2f} ({row['worst_case']}).")
    for scan in SCANS:
        moved = [r for r in pose if r["scan"] == scan]
        if moved:
            top = max(moved, key=lambda r: r["max_abs_change"])
            ctx.note(f"Pose ({scan}): largest metric change {top['max_abs_change']:.2e} "
                     f"({top['metric']}, {top['side']}, {top['worst_case']}).")
    ctx.note("\nProvenance tiers (left knee; right identical):")
    for row in provenance:
        ctx.note(f"- {row['metric']}: " + "; ".join(
            f"{k} = {v}" for k, v in row.items() if k != "metric"))
