"""E7b: how well the implant fits the cut, for each way of shaping it.

One plan per knee (the planner's default), and on it the same two components -- the
femoral component on the distal femoral cut and the tray on the tibial cut -- shaped
four ways:

1. ``chart`` -- the catalogue part at the legacy chart size (round-down from the
   distal-quarter box, :func:`tka_planner.pipeline.discrete_sizing`), unscaled, at the
   plan's pose.
2. ``uniform`` -- the planner's catalogue mode: the nearest catalogue part scaled
   uniformly to the continuous size the box asks for, at the plan's pose. One scale for
   both bones, so the chart's proportions are kept.
3. ``patient_specific`` -- the planner's patient-specific mode
   (:func:`tka_planner.pipeline.component_scales`): each part stretched per axis to the
   width and depth measured on the cuts; the femoral part slid onto the centre of the
   distal section and the tray kept in register under it.

plus two controls, each changing one thing:

2c. ``uniform_cut`` -- one uniform scale for both parts, chosen so the femoral footprint
    is the width of the distal cut, and slid onto the section exactly as (3) is. It
    differs from (3) only in keeping the catalogue's proportions, so (2c) against (3)
    is the value of matching mediolateral and anteroposterior independently, which is
    the paper's claim for the parametric implant.
3c. ``tray_centred`` -- (3) with the tray footprint centred on its own cut. The planner
    keeps the tray in register under the femoral component, so on the tibial cut it
    sits where the femur puts it; the Fusion plate instead moves its outline onto the
    section (``Tray_Offset``). (3c) separates that placement from the shape.

And, where the private Fusion builds are available (``TKA_PATIENT_BUILDS``), a fourth:

4. ``regenerated`` -- the patient's own parts, regenerated in Fusion from the plan,
   posed exactly as the build's ``fit_check`` poses them (the part's CAD frame put on
   the build's pose, unscaled). The builds were made on the plan with the knee's own
   ``plan_controls`` (extra tibial resection for an 8 mm insert), so this row and a
   rerun of (1)-(3c) on that same plan are reported under ``plan = build``, and the
   regenerated footprints are compared with (3) and (3c) directly: the WP-B parity
   question.

Fit is :func:`tka_planner.core.fit.section_fit` throughout: coverage of the cut, and the
largest overhang (component beyond the bone) and underhang (bone left uncovered) in
millimetres, overall and per zone. The zones are the anatomical halves -- medial,
lateral, anterior, posterior -- each the larger of the two quadrants it spans.

Catalogue parts. The library exports a left femoral component only, except at L4, which
also holds a right one; the planner's resolver therefore takes every right knee's
femoral component from L4 and rescales it. L4 is not the same master geometry as S1 to
L3 (its femoral part is the L2 part's width), so this experiment takes the left part of
the wanted size and mirrors it for a right knee instead (L4's left and right parts are
exact mirrors, which justifies it). On left knees this is exactly the planner's own
path, and the run checks that it reproduces :func:`tka_planner.pipeline.fit_case`; the
planner's own numbers for every knee are written beside, so the effect on right knees
is visible.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tka_planner.core.fit import fill_section, section_fit, slice_mesh
from tka_planner.core.geometry import unit
from tka_planner.core.meshio import Mesh, read_stl
from tka_planner.core.planning import Adjustments
from tka_planner.geom import mesh as gm
from tka_planner.pipeline import (
    discrete_sizing,
    fit_case,
    implant_spec_case,
    library_dimensions,
    plan_case,
)
from tka_planner.scene.build import place

from ..harness import Context, experiment, library_root, measured
from ..stats import format_p, paired_summary

STRATEGIES = ("chart", "uniform", "uniform_cut", "patient_specific", "tray_centred",
              "regenerated")
LABELS = {
    "chart": "(1) chart size",
    "uniform": "(2) uniform scale",
    "uniform_cut": "(2c) uniform, cut-sized",
    "patient_specific": "(3) patient-specific",
    "tray_centred": "(3c) patient-specific, tray centred",
    "regenerated": "(4) regenerated CAD",
}
COMPONENTS = (
    # name, group, cut, bone attribute, frame attribute, side of the cut
    ("femoral_component", "femoral", "femoral_distal", "femur", "femoral_frame", -1.0),
    ("tibial_component", "tibial", "tibial_proximal", "tibia", "tibial_frame", 1.0),
)
ZONES = {"medial": ("anteromedial", "posteromedial"),
         "lateral": ("anterolateral", "posterolateral"),
         "anterior": ("anteromedial", "anterolateral"),
         "posterior": ("posteromedial", "posterolateral")}
BUILD_PARTS = {"femoral": "femoral.stl", "tibial": "plate.stl"}
FOOTPRINT_GRID_MM = 0.25

_PARTS: dict = {}


def builds_root() -> Path | None:
    """The private Fusion builds, one folder per knee; ``None`` when not configured."""
    value = os.environ.get("TKA_PATIENT_BUILDS")
    return Path(value) if value and Path(value).is_dir() else None


# ----------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------


def _normalise(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def catalogue_part(library: Path, label: str, group: str, side: str) -> dict:
    """The catalogue part of one size: its mesh in CAD axes and its footprint.

    Femoral parts are exported for the left knee; a right knee gets the mirror image
    across the sagittal plane (CAD Y), with the winding reversed so it stays outward.
    """
    right = str(side).lower().startswith("r")
    key = (str(library), label, group, right and group == "femoral")
    if key in _PARTS:
        return _PARTS[key]
    files = {_normalise(p.name): p for p in (library / label).glob("*.stl")}
    if group == "femoral":
        path = next(p for k, p in files.items()
                    if "implant" in k and "femoral" in k and "left" in k)
    else:
        path = next(p for k, p in files.items()
                    if "tibial" in k and "plate" in k and "shell" not in k)
    mesh = read_stl(path)
    dims = dict(library_dimensions({"path": str(path)}, group))
    mirrored = right and group == "femoral"
    if mirrored:
        mesh = Mesh(vertices=mesh.vertices * np.array([1.0, -1.0, 1.0]),
                    faces=mesh.faces[:, ::-1].copy())
        dims["centre"] = [dims["centre"][0], -dims["centre"][1]]
    _PARTS[key] = {"mesh": mesh, "dims": dims, "size": label, "mirrored": mirrored}
    return _PARTS[key]


def centring_shifts(m, plan, spec: dict, scales: dict, lib_f: dict) -> dict:
    """Slide the femoral footprint onto the section centre and keep the tray in register.

    The same rule as :func:`tka_planner.pipeline.component_scales`, written against a
    given scale so the cut-sized uniform control is placed exactly as (3) is.
    """
    record = spec["femoral_component"]
    centre = np.asarray(record["diagnostics"]["section_centre_mm"], dtype=float)
    femoral = np.array([*(centre - np.asarray(scales["femoral"][:2])
                          * np.asarray(lib_f["centre"])), 0.0])
    lateral_sign = -1.0 if str(m.side).lower().startswith("r") else 1.0
    a = plan.adjustments
    femoral += np.array([a.femoral_shift_ap_mm, lateral_sign * a.femoral_shift_ml_mm, 0.0])
    F = np.asarray(plan.components["femoral_component"], dtype=float)
    T = np.asarray(plan.components["tibial_component"], dtype=float)
    world = F[:3, :2] @ femoral[:2]
    return {"femoral": femoral, "tibial": np.array([*(T[:3, :2].T @ world), 0.0])}


def strategy_setups(m, plan, sizing, spec, library) -> dict:
    """Parts, scales and shifts for strategies 1, 2, 2c and 3 on one plan."""
    chart = m.chart
    side = str(m.side)
    setups = {}

    discrete = discrete_sizing(m)
    label = discrete.nearest_discrete_size
    setups["chart"] = {
        "parts": {g: catalogue_part(library, label, g, side) for g in ("femoral", "tibial")},
        "scales": {"femoral": 1.0, "tibial": 1.0}, "shifts": {},
        "note": f"{label}, unscaled",
    }

    label = sizing.nearest_discrete_size
    scale = sizing.implant_ml_mm / chart.value_at("femur_ML", chart.label_parameter(label))
    setups["uniform"] = {
        "parts": {g: catalogue_part(library, label, g, side) for g in ("femoral", "tibial")},
        "scales": {"femoral": scale, "tibial": scale}, "shifts": {},
        "note": f"{label} x {scale:.4f} (size parameter {sizing.size_parameter:.2f})",
    }

    femoral = spec["femoral_component"]["dimensions_mm"]
    tibial = spec["tibial_component"]["dimensions_mm"]
    parameter = chart.parameter_for("femur_ML", femoral["ml"])
    label = chart.nearest_label(parameter)
    parts = {g: catalogue_part(library, label, g, side) for g in ("femoral", "tibial")}
    scale = femoral["ml"] / parts["femoral"]["dims"]["ml"]
    scales = {"femoral": np.full(3, scale), "tibial": np.full(3, scale)}
    setups["uniform_cut"] = {
        "parts": parts, "scales": scales,
        "shifts": centring_shifts(m, plan, spec, scales, parts["femoral"]["dims"]),
        "note": f"{label} x {scale:.4f} (femoral footprint = distal cut width)",
    }

    label = sizing.nearest_discrete_size
    parts = {g: catalogue_part(library, label, g, side) for g in ("femoral", "tibial")}
    lib_f, lib_t = parts["femoral"]["dims"], parts["tibial"]["dims"]
    size_scale = sizing.implant_ml_mm / chart.value_at("femur_ML",
                                                       chart.label_parameter(label))
    scales = {
        "femoral": np.array([femoral["ap_overall"] / lib_f["ap_overall"],
                             femoral["ml"] / lib_f["ml"], size_scale]),
        "tibial": np.array([tibial["ml"] / lib_t["ml"], tibial["ap"] / lib_t["ap"],
                            size_scale]),
    }
    setups["patient_specific"] = {
        "parts": parts, "scales": scales,
        "shifts": centring_shifts(m, plan, spec, scales, lib_f),
        "note": (f"{label}; femoral x({scales['femoral'][0]:.3f}, "
                 f"{scales['femoral'][1]:.3f}, {scales['femoral'][2]:.3f}), tray "
                 f"x({scales['tibial'][0]:.3f}, {scales['tibial'][1]:.3f}, "
                 f"{scales['tibial'][2]:.3f})"),
    }

    # The planner keeps the tray in register with the femoral component, so the tray is
    # not centred on its own cut. The Fusion plate is: its outline is moved onto the
    # section (Tray_Offset) while the insert stays in register. (3c) is (3) with that
    # one change, the tray footprint's centre put on the tibial section's centre.
    centre = np.asarray(spec["tibial_component"]["diagnostics"]["section_centre_mm"],
                        dtype=float)
    shifts = dict(setups["patient_specific"]["shifts"])
    shifts["tibial"] = np.array([*(centre - scales["tibial"][:2]
                                   * np.asarray(lib_t["centre"])), 0.0])
    setups["tray_centred"] = dict(setups["patient_specific"], shifts=shifts,
                                  note=setups["patient_specific"]["note"]
                                  + "; tray centred on the tibial cut")
    return setups


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------


def posed_parts(m, plan, setup: dict) -> dict:
    """Each group's part posed in the bone's coordinates, with its cut plane."""
    out = {}
    for name, group, cut, *_ in COMPONENTS:
        part = setup["parts"][group]
        pose = setup.get("poses", {}).get(group, plan.components[name])
        matrix = place(pose, setup["scales"][group], setup["shifts"].get(group))
        if setup.get("cut_planes"):
            point, normal = setup["cut_planes"][group]
        else:
            point, normal = plan.resections[cut].point, plan.resections[cut].normal
        out[group] = {"mesh": gm.transformed(part["mesh"], matrix),
                      "point": np.asarray(point, float), "normal": np.asarray(normal, float)}
    return out


def fit_rows(m, posed: dict) -> dict:
    """``section_fit`` of each posed part, as a flat row per component."""
    rows = {}
    for _, group, cut, bone, frame, side_of_cut in COMPONENTS:
        frame = getattr(m, frame)
        fit = section_fit(cut=cut, bone=getattr(m, bone), component=posed[group]["mesh"],
                          point=posed[group]["point"], normal=posed[group]["normal"],
                          component_side=side_of_cut, lateral=frame.lateral,
                          anterior=frame.x_anterior).to_dict()
        row = {
            "coverage_pct": 100.0 * fit["coverage_fraction"],
            "on_bone_pct": 100.0 * fit["component_on_bone_fraction"],
            "overhang_area_mm2": fit["overhang_area_mm2"],
            "max_overhang_mm": fit["max_overhang_mm"],
            "max_underhang_mm": fit["max_underhang_mm"],
            "component_ml_mm": fit["component_extent_mm"]["ml"],
            "component_ap_mm": fit["component_extent_mm"]["ap"],
            "bone_ml_mm": fit["bone_extent_mm"]["ml"],
            "bone_ap_mm": fit["bone_extent_mm"]["ap"],
            "flags": " ".join(fit["flags"]),
        }
        for zone, quadrants in ZONES.items():
            row[f"over_{zone}_mm"] = max(fit["zones"][q]["max_overhang_mm"]
                                         for q in quadrants)
            row[f"under_{zone}_mm"] = max(fit["zones"][q]["max_underhang_mm"]
                                          for q in quadrants)
        rows[group] = row
    return rows


def footprint_agreement(m, a: dict, b: dict) -> dict:
    """How closely two posed parts' footprints on the same cut agree.

    Both are sliced 1 mm off the cut on the component side, as ``section_fit`` slices a
    component, and filled on one grid in the cut plane. Reported per group: the
    difference in mediolateral and anteroposterior extent (b - a), the offset of the
    footprint centres, the intersection over union, and the symmetric Hausdorff
    distance between the two outlines.
    """
    out = {}
    for _, group, _, _, frame, side_of_cut in COMPONENTS:
        frame = getattr(m, frame)
        point, normal = a[group]["point"], unit(a[group]["normal"])
        u = unit(frame.lateral - np.dot(frame.lateral, normal) * normal)
        v = unit(np.cross(normal, u))
        if float(np.dot(v, frame.x_anterior)) < 0:
            v = -v
        level = point + side_of_cut * 1.0 * normal

        def flat(posed):
            rel = slice_mesh(posed["mesh"], level, normal) - point
            return np.stack([rel @ u, rel @ v], axis=-1)

        sa, sb = flat(a[group]), flat(b[group])
        if not len(sa) or not len(sb):
            out[group] = None
            continue
        both = np.concatenate([sa, sb]).reshape(-1, 2)
        low, high = both.min(axis=0) - 1.0, both.max(axis=0) + 1.0
        xs = np.arange(low[0], high[0], FOOTPRINT_GRID_MM) + FOOTPRINT_GRID_MM * 0.5013
        ys = np.arange(low[1], high[1], FOOTPRINT_GRID_MM) + FOOTPRINT_GRID_MM * 0.4987
        ma, mb = fill_section(sa, xs, ys), fill_section(sb, xs, ys)
        gx, gy = np.meshgrid(xs, ys)
        out[group] = {
            "ml_diff_mm": float(np.ptp(gx[mb]) - np.ptp(gx[ma])),
            "ap_diff_mm": float(np.ptp(gy[mb]) - np.ptp(gy[ma])),
            "centre_offset_ml_mm": float(gx[mb].mean() - gx[ma].mean()),
            "centre_offset_ap_mm": float(gy[mb].mean() - gy[ma].mean()),
            "iou": float((ma & mb).sum()) / max(int((ma | mb).sum()), 1),
            "hausdorff_mm": _hausdorff(ma, mb, gx, gy),
        }
    return out


def _edge_points(mask, gx, gy) -> np.ndarray:
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    edge = mask & ~interior
    return np.c_[gx[edge], gy[edge]]


def _directed(p: np.ndarray, q: np.ndarray) -> float:
    worst = 0.0
    for start in range(0, len(p), 2048):
        d = p[start:start + 2048, None, :] - q[None, :, :]
        worst = max(worst, float(np.sqrt((d * d).sum(-1)).min(axis=1).max()))
    return worst


def _hausdorff(ma, mb, gx, gy) -> float:
    pa, pb = _edge_points(ma, gx, gy), _edge_points(mb, gx, gy)
    if not len(pa) or not len(pb):
        return float("nan")
    return max(_directed(pa, pb), _directed(pb, pa))


# ----------------------------------------------------------------------
# The regenerated parts
# ----------------------------------------------------------------------


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    if not a.is_file() or a.stat().st_size != b.stat().st_size:
        return False
    return hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def build_for(case, builds: Path | None) -> dict | None:
    """The knee's Fusion build: its job (poses, controls, parameters) and parts.

    The job is taken from ``jobs_demo.json``, the list the build's ``fit_check`` reads,
    and only for a knee missing from it from the knee's own ``job.json`` (an older build,
    flagged). Read only, by absolute path. The job's bone files are compared with the
    harness's, by path or else by content, to make sure the build is of this knee; the
    paths themselves are never written out.
    """
    if builds is None:
        return None
    folder = builds / case.case_id
    if not all((folder / f).is_file() for f in BUILD_PARTS.values()):
        return None
    demo_path = builds / "jobs_demo.json"
    demo = ({j["name"]: j for j in json.loads(demo_path.read_text(encoding="utf-8"))}
            if demo_path.is_file() else {})
    job = demo.get(case.case_id)
    if job is None:
        if not (folder / "job.json").is_file():
            return None
        job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    bones = job.get("bones")
    same_bones = (None if not bones else
                  len(bones) == 2 and _same_file(Path(bones[0]), case.femur)
                  and _same_file(Path(bones[1]), case.tibia))
    return {
        "job": job,
        "in_fit_check": case.case_id in demo,
        "same_bones": same_bones,
        "parts": {g: {"mesh": read_stl(folder / f), "size": "Fusion", "mirrored": False}
                  for g, f in BUILD_PARTS.items()},
        "controls": job.get("plan_controls") or {},
    }


def build_plan(m, library, build: dict):
    """The plan the build was made on: the knee's recorded controls, rerun today.

    If today's planner puts the tibial cut somewhere else than the build's tray (the
    planner has changed since the build, or a control went unrecorded), the tibial
    resection is adjusted until the cut passes through the tray's origin, and the
    adjustment is returned so it can be reported rather than hidden.
    """
    controls = dict(build["controls"])
    plan, sizing = plan_case(m, library=library, adjustments=Adjustments(**controls))
    tray = np.asarray(build["job"]["plan"]["tray_pose"], float)
    cut = plan.resections["tibial_proximal"]
    offset = float((tray[:3, 3] - cut.point) @ unit(cut.normal))
    inferred = 0.0
    if abs(offset) > 0.01:
        # The tray below today's cut means more bone was taken: a positive delta.
        inferred = -offset
        controls["tibial_resection_delta_mm"] = (
            controls.get("tibial_resection_delta_mm", 0.0) + inferred)
        plan, sizing = plan_case(m, library=library, adjustments=Adjustments(**controls))
    return plan, sizing, inferred


def regenerated_setup(build: dict, plan) -> tuple[dict, dict]:
    """Pose the regenerated parts as the build's fit_check does, and say how that pose
    relates to the plan rerun here: rotation and out-of-plane offset should be nil (the
    build's pose is the plan's, slid within the cut plane)."""
    poses = {"femoral": np.asarray(build["job"]["plan"]["femoral_pose"], float),
             "tibial": np.asarray(build["job"]["plan"]["tray_pose"], float)}
    check = {}
    for name, group, cut, *_ in COMPONENTS:
        P = np.asarray(plan.components[name], float)
        J = poses[group]
        r = plan.resections[cut]
        check[group] = {
            "rotation_diff": float(np.abs(J[:3, :3] - P[:3, :3]).max()),
            "out_of_plane_mm": float((J[:3, 3] - r.point) @ unit(r.normal)),
            "in_plane_slide_mm": [round(float(x), 3)
                                  for x in (P[:3, :2].T @ (J[:3, 3] - P[:3, 3]))],
        }
    setup = {"parts": build["parts"], "scales": {"femoral": 1.0, "tibial": 1.0},
             "shifts": {}, "poses": poses,
             # fit_check's planes: the part's own origin and +Z.
             "cut_planes": {g: (P[:3, 3], P[:3, 2]) for g, P in poses.items()},
             "note": "Fusion build, unscaled, at the build's pose"}
    return setup, check


# ----------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------


def chart_ranges(chart, library: Path) -> dict:
    """The chart's published range per dimension, and the catalogue parts' actual
    footprint range, which is not the same (the library is checked, not assumed)."""
    def parts_range(group, key):
        values = [catalogue_part(library, lbl, group, "left")["dims"][key]
                  for lbl in chart.labels]
        return min(values), max(values)

    def chart_range(column):
        return float(chart.columns[column][0]), float(chart.columns[column][-1])

    return {
        "femoral_ml": (*chart_range("femur_ML"), *parts_range("femoral", "ml")),
        "femoral_ap": (*chart_range("femur_AP"), *parts_range("femoral", "ap_overall")),
        "tibial_ml": (*chart_range("tibia_ML"), *parts_range("tibial", "ml")),
        "tibial_ap": (*chart_range("tibia_AP"), *parts_range("tibial", "ap")),
    }


def library_rows(chart, library: Path) -> list[dict]:
    """The catalogue checked against its own chart: every size's femoral part and tray,
    their bounding boxes and the footprints the fit sees."""
    rows = []
    for label in chart.labels:
        f = catalogue_part(library, label, "femoral", "left")
        t = catalogue_part(library, label, "tibial", "left")
        fe = np.ptp(f["mesh"].vertices, axis=0)
        te = np.ptp(t["mesh"].vertices, axis=0)
        p = chart.label_parameter(label)
        rows.append({
            "size": label,
            "chart_femur_ML": chart.value_at("femur_ML", p),
            "chart_femur_AP": chart.value_at("femur_AP", p),
            "part_bbox_ML": float(fe[1]), "part_bbox_AP": float(fe[0]),
            "part_bbox_SI": float(fe[2]),
            "part_AP_over_ML": float(fe[0] / fe[1]),
            "part_ML_over_AP": float(fe[1] / fe[0]), "part_SI_over_AP": float(fe[2] / fe[0]),
            "footprint_ML": f["dims"]["ml"], "footprint_AP_distal": f["dims"]["ap_distal"],
            "chart_tibia_ML": chart.value_at("tibia_ML", p),
            "chart_tibia_AP": chart.value_at("tibia_AP", p),
            "tray_footprint_ML": t["dims"]["ml"], "tray_footprint_AP": t["dims"]["ap"],
        })
    return rows


@experiment("E7b", "Implant fit: chart size vs uniform scale vs patient-specific")
def run(ctx: Context) -> None:
    library = library_root()
    if library is None:
        ctx.note("No implant library found (data/implants or TKA_IMPLANT_LIBRARY). "
                 "Nothing run.")
        return
    builds = builds_root()
    ctx.note(f"Fusion builds: {'read from TKA_PATIENT_BUILDS' if builds else 'not configured (TKA_PATIENT_BUILDS unset); row (4) skipped'}.")

    fit_table, measures, planner_rows, parity_rows, refused = [], [], [], [], []
    chart = None
    for case in ctx.cases:
        m = measured(case)
        if isinstance(m, Exception):
            refused.append((case.case_id, type(m).__name__))
            continue
        chart = m.chart
        build = build_for(case, builds)
        plans = ["default"] + (["build"] if build is not None else [])
        for plan_name in plans:
            if plan_name == "default":
                plan, sizing = plan_case(m, library=library)
            else:
                plan, sizing, inferred = build_plan(m, library, build)
            spec = implant_spec_case(m, plan, sizing, library, with_cut_box=False)
            setups = strategy_setups(m, plan, sizing, spec, library)
            posed = {}
            if plan_name == "build":
                setups["regenerated"], pose_check = regenerated_setup(build, plan)
            for strategy, setup in setups.items():
                posed[strategy] = posed_parts(m, plan, setup)
                for group, row in fit_rows(m, posed[strategy]).items():
                    fit_table.append({"case": case.case_id, "side": case.side,
                                      "plan": plan_name, "strategy": strategy,
                                      "component": group, "parts": setup["note"], **row})
            femoral = spec["femoral_component"]["dimensions_mm"]
            tibial = spec["tibial_component"]["dimensions_mm"]
            if plan_name == "default":
                uc = setups["uniform_cut"]
                s = float(uc["scales"]["femoral"][0])
                lib_f, lib_t = uc["parts"]["femoral"]["dims"], uc["parts"]["tibial"]["dims"]
                measures.append({
                    "case": case.case_id, "side": case.side,
                    "box_femoral_ml": m.femoral_measure.ml_mm,
                    "cut_femoral_ml": femoral["ml"], "cut_femoral_ap": femoral["ap_overall"],
                    "cut_tibial_ml": tibial["ml"], "cut_tibial_ap": tibial["ap"],
                    "size_parameter_box": sizing.size_parameter,
                    "size_chart": setups["chart"]["parts"]["femoral"]["size"],
                    # What one uniform scale leaves unmatched once the femoral width is
                    # matched: positive = the part is larger than the bone.
                    "uniform_cut_femoral_ap_residual_mm": lib_f["ap_overall"] * s
                    - femoral["ap_overall"],
                    "uniform_cut_tray_ml_residual_mm": lib_t["ml"] * s - tibial["ml"],
                    "uniform_cut_tray_ap_residual_mm": lib_t["ap"] * s - tibial["ap"],
                    "bone_tibial_ap_over_ml": tibial["ap"] / tibial["ml"],
                    "bone_femoral_ap_over_ml": femoral["ap_overall"] / femoral["ml"],
                })
                for mode in ("catalogue", "patient_specific"):
                    fit = fit_case(m, plan, sizing, library, mode)
                    for name, group, *_ in COMPONENTS:
                        planner_rows.append({
                            "case": case.case_id, "side": case.side,
                            "planner_mode": mode, "component": group,
                            "source_size": fit["library_sizes"][name]["source_size"],
                            "coverage_pct": 100.0 * fit[name]["coverage_fraction"],
                            "max_overhang_mm": fit[name]["max_overhang_mm"],
                            "component_ml_mm": fit[name]["component_extent_mm"]["ml"]})
            else:
                agreement = footprint_agreement(m, posed["patient_specific"],
                                                posed["regenerated"])
                centred = footprint_agreement(m, posed["tray_centred"],
                                              posed["regenerated"])
                params = build["job"]
                for group in ("femoral", "tibial"):
                    row = {"case": case.case_id, "side": case.side, "component": group,
                           "in_fit_check": build["in_fit_check"],
                           "same_bone_files": build["same_bones"],
                           "plan_controls": json.dumps(build["controls"]),
                           "inferred_extra_tibial_resection_mm": inferred,
                           "pose_rotation_diff": pose_check[group]["rotation_diff"],
                           "pose_out_of_plane_mm": pose_check[group]["out_of_plane_mm"],
                           "pose_in_plane_slide_mm": str(pose_check[group]["in_plane_slide_mm"])}
                    if group == "femoral":
                        row.update(cad_ml_param_mm=_mm(params["femoral"].get("Femur_ML")),
                                   cut_ml_mm=femoral["ml"])
                    else:
                        row.update(cad_ml_param_mm=_mm(params["plate"].get("Tibial_Plateau_ML")),
                                   cad_ap_param_mm=_mm(params["plate"].get("Tibial_Plateau_AP")),
                                   cut_ml_mm=tibial["ml"], cut_ap_mm=tibial["ap"])
                    row.update({f"regen_minus_ps_{k}": v
                                for k, v in (agreement.get(group) or {}).items()})
                    if group == "tibial":
                        row.update({f"regen_minus_centred_{k}": v
                                    for k, v in (centred.get(group) or {}).items()})
                    parity_rows.append(row)

    if refused:
        ctx.note(f"Refused by the planner: {', '.join(f'{c} ({w})' for c, w in refused)}.")
    if chart is None:
        return
    _write(ctx, chart, library, fit_table, measures, planner_rows, parity_rows)


def _mm(text) -> float | None:
    return float(str(text).split()[0]) if text is not None else None


def _write(ctx, chart, library, fit_table, measures, planner_rows, parity_rows) -> None:
    fit_columns = ["case", "side", "plan", "strategy", "component", "parts",
                   "coverage_pct", "on_bone_pct", "overhang_area_mm2", "max_overhang_mm",
                   "max_underhang_mm",
                   *[f"over_{z}_mm" for z in ZONES], *[f"under_{z}_mm" for z in ZONES],
                   "component_ml_mm", "component_ap_mm", "bone_ml_mm", "bone_ap_mm",
                   "flags"]
    ctx.table("fit", fit_table, fit_columns, digits=2)
    ctx.table("measurements", measures, digits=2)
    ctx.table("planner_fit_case", planner_rows, digits=2)
    ctx.table("library_vs_chart", library_rows(chart, library), digits=3)
    if parity_rows:
        ctx.table("parity_regenerated_vs_patient_specific", parity_rows, digits=3)

    # Knees whose own dimensions fall outside the chart, and outside the parts.
    ranges = chart_ranges(chart, library)
    keys = {"femoral_ml": "cut_femoral_ml", "femoral_ap": "cut_femoral_ap",
            "tibial_ml": "cut_tibial_ml", "tibial_ap": "cut_tibial_ap"}
    outside_rows = []
    for row in measures:
        out = {"case": row["case"]}
        for dim, key in keys.items():
            lo, hi, plo, phi = ranges[dim]
            value = row[key]
            out[dim] = round(value, 1)
            out[f"{dim}_vs_chart"] = ("below" if value < lo else "above" if value > hi
                                      else "in")
            out[f"{dim}_vs_parts"] = ("below" if value < plo else "above" if value > phi
                                      else "in")
        lo, hi, *_ = ranges["femoral_ml"]
        out["box_femoral_ml_vs_chart"] = ("above" if row["box_femoral_ml"] > hi else
                                          "below" if row["box_femoral_ml"] < lo else "in")
        outside_rows.append(out)
    ctx.table("outside_chart", outside_rows)

    knees = len(measures)
    ctx.note(f"\n{knees} knees fitted on the default plan.")
    ctx.note("Chart ranges (published) vs the catalogue parts' footprints: "
             + "; ".join(f"{d} {lo:.0f}-{hi:.0f} vs {plo:.1f}-{phi:.1f} mm"
                         for d, (lo, hi, plo, phi) in ranges.items()) + ".")
    for dim in keys:
        n_chart = sum(r[f"{dim}_vs_chart"] != "in" for r in outside_rows)
        n_parts = sum(r[f"{dim}_vs_parts"] != "in" for r in outside_rows)
        ctx.note(f"- {dim} at the cut: outside the chart in {n_chart} of {knees} knees, "
                 f"outside the parts' range in {n_parts}.")
    any_chart = sum(any(r[f"{d}_vs_chart"] != "in" for d in keys) for r in outside_rows)
    any_parts = sum(any(r[f"{d}_vs_parts"] != "in" for d in keys) for r in outside_rows)
    box_out = sum(r["box_femoral_ml_vs_chart"] != "in" for r in outside_rows)
    ctx.note(f"- Any dimension outside: {any_chart} of {knees} (chart), {any_parts} of "
             f"{knees} (parts). The box femoral ML is above the chart in {box_out}.")

    # Paired statistics, default plan, every strategy against (3).
    default = [r for r in fit_table if r["plan"] == "default"]
    stats_rows = []
    for group in ("femoral", "tibial"):
        for metric in ("coverage_pct", "max_overhang_mm", "max_underhang_mm",
                       "overhang_area_mm2"):
            ref = {r["case"]: r[metric] for r in default
                   if r["strategy"] == "patient_specific" and r["component"] == group}
            for strategy in ("chart", "uniform", "uniform_cut", "tray_centred"):
                other = {r["case"]: r[metric] for r in default
                         if r["strategy"] == strategy and r["component"] == group}
                cases = sorted(ref)
                s = paired_summary([other.get(c) for c in cases], [ref[c] for c in cases])
                stats_rows.append({"plan": "default", "component": group, "metric": metric,
                                   "comparison": f"{strategy} - patient_specific", **s})
    # Paired over the builds fit_check covers (jobs_demo.json); an older build of
    # another rule is in the tables but not in the statistics.
    checked = {r["case"] for r in parity_rows if r["in_fit_check"]}
    build = [r for r in fit_table if r["plan"] == "build" and r["case"] in checked]
    for group in ("femoral", "tibial"):
        for metric in ("coverage_pct", "max_overhang_mm", "max_underhang_mm"):
            ref = {r["case"]: r[metric] for r in build
                   if r["strategy"] == "patient_specific" and r["component"] == group}
            regen = {r["case"]: r[metric] for r in build
                     if r["strategy"] == "regenerated" and r["component"] == group}
            centred = {r["case"]: r[metric] for r in build
                       if r["strategy"] == "tray_centred" and r["component"] == group}
            cases = sorted(ref)
            if cases:
                s = paired_summary([regen.get(c) for c in cases], [ref[c] for c in cases])
                stats_rows.append({"plan": "build", "component": group, "metric": metric,
                                   "comparison": "regenerated - patient_specific", **s})
                if group == "tibial":
                    s = paired_summary([regen.get(c) for c in cases],
                                       [centred.get(c) for c in cases])
                    stats_rows.append({"plan": "build", "component": group,
                                       "metric": metric,
                                       "comparison": "regenerated - tray_centred", **s})
    ctx.table("paired_stats", stats_rows, digits=3)

    ctx.note("\nMeans per strategy, default plan (coverage %, max overhang mm, "
             "max underhang mm):")
    for group in ("femoral", "tibial"):
        for strategy in STRATEGIES[:-1]:
            sub = [r for r in default if r["strategy"] == strategy and r["component"] == group]
            if sub:
                ctx.note(f"- {group:7s} {LABELS[strategy]:26s} "
                         f"{np.mean([r['coverage_pct'] for r in sub]):5.1f} %  "
                         f"{np.mean([r['max_overhang_mm'] for r in sub]):5.2f}  "
                         f"{np.mean([r['max_underhang_mm'] for r in sub]):5.2f}")
    ctx.note("\nPaired against (3), default plan:")
    for s in stats_rows:
        if s["n"] and s["metric"] in ("coverage_pct", "max_overhang_mm"):
            ctx.note(f"- [{s['plan']}] {s['component']} {s['metric']}: {s['comparison']} "
                     f"mean {s['mean']:+.2f} (SD {s['sd'] or 0:.2f}, {s['min']:+.2f} to "
                     f"{s['max']:+.2f}); {format_p(s)}")

    for case_id in ("P009L", "P009R"):
        for r in default:
            if r["case"] == case_id and r["component"] == "femoral" and r["strategy"] in (
                    "uniform", "patient_specific"):
                ctx.note(f"- {case_id} femoral {LABELS[r['strategy']]}: max overhang "
                         f"{r['max_overhang_mm']:.1f} mm (medial {r['over_medial_mm']:.1f},"
                         f" lateral {r['over_lateral_mm']:.1f}); component ML "
                         f"{r['component_ml_mm']:.1f} vs bone {r['bone_ml_mm']:.1f} mm.")

    # Is our part source the planner's own on left knees?
    worst = 0.0
    for r in planner_rows:
        if r["side"] != "left":
            continue
        strategy = "uniform" if r["planner_mode"] == "catalogue" else "patient_specific"
        mine = next(x for x in default if x["case"] == r["case"]
                    and x["strategy"] == strategy and x["component"] == r["component"])
        worst = max(worst, abs(mine["coverage_pct"] - r["coverage_pct"]),
                    abs(mine["max_overhang_mm"] - r["max_overhang_mm"]))
    ctx.note(f"\nLeft knees, this experiment's parts vs the planner's fit_case: largest "
             f"difference {worst:.3g} (coverage % or overhang mm).")
    right_diffs = []
    for r in planner_rows:
        if r["side"] == "right" and r["component"] == "femoral":
            strategy = "uniform" if r["planner_mode"] == "catalogue" else "patient_specific"
            mine = next(x for x in default if x["case"] == r["case"]
                        and x["strategy"] == strategy and x["component"] == "femoral")
            right_diffs.append((r["case"], r["planner_mode"],
                                r["component_ml_mm"] - mine["component_ml_mm"]))
    if right_diffs:
        ctx.note("Right knees, femoral footprint ML, planner (L4-sourced) minus mirrored "
                 "catalogue part: " + ", ".join(f"{c} {mode} {d:+.1f} mm"
                                                 for c, mode, d in right_diffs) + ".")

    if parity_rows:
        ctx.note("\nRegenerated CAD (4) vs patient-specific scaling (3), build plan, "
                 "footprints 1 mm off the cut (regenerated minus (3)):")
        for r in parity_rows:
            if r["component"] == "tibial" and "regen_minus_centred_iou" in r:
                ctx.note(f"- {r['case']} tibial vs (3c), tray centred: ML "
                         f"{r['regen_minus_centred_ml_diff_mm']:+.1f}, AP "
                         f"{r['regen_minus_centred_ap_diff_mm']:+.1f} mm, centre offset "
                         f"({r['regen_minus_centred_centre_offset_ml_mm']:+.1f}, "
                         f"{r['regen_minus_centred_centre_offset_ap_mm']:+.1f}) mm, IoU "
                         f"{r['regen_minus_centred_iou']:.3f}, Hausdorff "
                         f"{r['regen_minus_centred_hausdorff_mm']:.1f} mm.")
            ctx.note(f"- {r['case']} {r['component']}: ML {r.get('regen_minus_ps_ml_diff_mm', float('nan')):+.1f}, "
                     f"AP {r.get('regen_minus_ps_ap_diff_mm', float('nan')):+.1f} mm, IoU "
                     f"{r.get('regen_minus_ps_iou', float('nan')):.3f}, Hausdorff "
                     f"{r.get('regen_minus_ps_hausdorff_mm', float('nan')):.1f} mm; build pose "
                     f"vs plan: rotation {r['pose_rotation_diff']:.1e}, out of plane "
                     f"{r['pose_out_of_plane_mm']:+.3f} mm after "
                     f"{r['inferred_extra_tibial_resection_mm']:+.1f} mm inferred tibial "
                     f"resection; bones match: {r['same_bone_files']}; in fit_check: "
                     f"{r['in_fit_check']}.")
    _figure(ctx, [r for r in fit_table if r["strategy"] != "regenerated"
                  or r["case"] in checked])


def _figure(ctx: Context, fit_table: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.note("matplotlib not installed; no figure.")
        return
    knees = sorted({r["case"] for r in fit_table})
    if not knees:
        return
    style = {  # categorical slots in fixed order, each with its own marker
        "chart": ("#2a78d6", "s"), "uniform": ("#eb6834", "D"),
        "uniform_cut": ("#1baf7a", "v"), "patient_specific": ("#6250d6", "o"),
        "tray_centred": ("#e87ba4", "P"), "regenerated": ("#e34948", "*"),
    }
    x = np.arange(len(knees))
    offsets = dict(zip(STRATEGIES, np.linspace(-0.32, 0.32, len(STRATEGIES))))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for row, group in enumerate(("femoral", "tibial")):
        for col, (metric, label) in enumerate((("coverage_pct", "Coverage of the cut (%)"),
                                               ("max_overhang_mm", "Largest overhang (mm)"))):
            ax = axes[row, col]
            for strategy in STRATEGIES:
                plan = "build" if strategy == "regenerated" else "default"
                values = {r["case"]: r[metric] for r in fit_table
                          if r["plan"] == plan and r["strategy"] == strategy
                          and r["component"] == group}
                if not values:
                    continue
                colour, marker = style[strategy]
                xs = [x[i] + offsets[strategy] for i, k in enumerate(knees) if k in values]
                ys = [values[k] for k in knees if k in values]
                ax.scatter(xs, ys, s=120 if marker == "*" else 42, color=colour,
                           marker=marker, edgecolor="white", linewidth=1.2, zorder=3,
                           label=LABELS[strategy] + (" (build plan)" if plan == "build"
                                                     else ""))
            if metric == "max_overhang_mm":
                ax.axhline(2.0, color="#8a8984", lw=1, ls="--", zorder=1)
                ax.text(1.0, 2.0, "2 mm flag ", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8, color="#52514e")
            ax.set_title(f"{'Femoral component, distal cut' if group == 'femoral' else 'Tibial tray, tibial cut'}",
                         fontsize=10, loc="left")
            ax.set_ylabel(label)
            ax.grid(axis="y", color="#e6e5e0", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(knees, rotation=45, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"fit_by_strategy.{suffix}"), dpi=200)
    plt.close(fig)
