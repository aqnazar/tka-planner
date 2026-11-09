"""E2: metamorphic invariance -- does the plan describe the knee, or the scan of it?

Each accepted knee is changed in ways that leave the anatomy alone, measured again from
nothing (automatic landmarks, frames, metrics, the default mechanical-alignment plan),
and compared with the unchanged run. Every comparison is made after mapping the variant
back into the original's coordinates, so a perfect planner shows zero everywhere.

The variants, by family:

* ``mirror`` -- X flipped about the scanner's sagittal plane, side swapped. A left knee
  becomes a right one; every angle should be identical and every landmark mirrored.
* ``translation`` and ``rigid_<angle>`` -- seeded random rigid motions: a random axis,
  a stated angle (5 to 180 degrees) about the knee centre, then a random 100 mm shift.
  The same motions are applied to every knee.
* ``scanner_<axis>`` -- the leg rotated about the knee centre, about one scanner axis at
  a time, by -30 to +30 degrees: how a patient may lie in the scanner. Signs are
  anatomical, so left and right knees pool: ``x`` positive tips the proximal ends
  anteriorly, ``y`` positive tips them laterally, ``z`` positive turns the leg
  externally (anterior towards lateral).
* ``decimate_50`` / ``decimate_10`` -- vertex clustering to about 50% and 10% of the
  triangles; ``remesh`` -- subdivision then clustering back to the original count on an
  offset grid (:mod:`tka_planner.geom.resample`). Resampled meshes are handed on as a
  triangle soup, as an STL would give them.
* ``noise_<sigma>`` -- Gaussian noise of 0.1, 0.25 and 0.5 mm per axis on every
  (welded) vertex, seeded.

What is reported:

* landmark displacement, mm, per landmark: the variant's landmark mapped back, against
  the original's;
* every metric's change (degrees), and the plan's: resection depths (mm), distal valgus
  cut, slope, cut-normal angles, component rotation about the cut and component
  origin shift, size;
* the rotational references in degrees: the femoral transepicondylar axis (anatomical,
  from the medial prominence -- the sulcus is never estimated) and posterior condylar
  axis, each mapped back and expressed as rotation about the original femur's
  proximal axis, their angle (aTEA against PCA) within each run, and the tibial AP
  (Akagi) axis likewise;
* refusals (quality control or frame) and crashes, per variant.

To say *where* a deviation comes from, every rigid variant is also run with the
original's landmarks carried through the same motion ("oracle landmarks"): frames,
metrics, sizing and plan then see exactly equivariant input, so anything left over is
the downstream stages' own dependence on the scanner axes, and the rest is the landmark
estimators'.

Tolerance. On a knee, the *metrics hold* when every metric moves by at most
``ANGLE_TOL_DEG``, and the *plan holds* when every plan angle does and every depth and
component origin moves by at most ``LENGTH_TOL_MM``. Landmark displacement is reported,
not judged. The mirror and the translations are expected to hold to floating point, and
so are the oracle runs wherever the downstream stages are free of scanner axes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from tka_planner.core.frames import FrameConstructionError
from tka_planner.core.landmarks import LandmarkSet
from tka_planner.core.meshio import Mesh
from tka_planner.core.qc import QCReport, QualityControlError
from tka_planner.geom.resample import as_triangle_soup, decimate, remesh, weld
from tka_planner.pipeline import measure_from_landmarks, measure_meshes, plan_case

from .. import harness
from ..harness import Context, experiment

ANGLE_TOL_DEG = 0.5
LENGTH_TOL_MM = 0.5
RIGID_ANGLES_DEG = (5.0, 15.0, 45.0, 90.0, 180.0)
SCANNER_ANGLES_DEG = (-30.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 30.0)
NOISE_MM = (0.1, 0.25, 0.5)
SHIFT_MM = 100.0

METRICS = ("mldfa_deg", "aldfa_deg", "mpta_deg", "jlca_deg",
           "posterior_slope_medial_deg", "femoral_ama_deg", "hka_deviation_deg",
           "condylar_twist_deg")
PLAN_ANGLES = ("valgus_cut_deg", "tibial_slope_deg", "femoral_cut_normal_deg",
               "tibial_cut_normal_deg", "femoral_component_rotation_deg",
               "tibial_component_rotation_deg")
PLAN_LENGTHS = ("femoral_depth_medial_mm", "femoral_depth_lateral_mm",
                "tibial_depth_medial_mm", "tibial_depth_lateral_mm",
                "femoral_component_origin_mm", "tibial_component_origin_mm")
ROTATION = ("atea_minus_pca_deg", "atea_axis_deg", "pca_axis_deg",
            "tibial_ap_axis_deg")


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One change to a knee that should leave its plan alone.

    ``matrix`` is the motion from original to variant coordinates (4x4, possibly a
    reflection) and is what comparisons undo; ``reshape`` changes the triangles without
    moving the anatomy, for the resampling and noise families.
    """

    family: str
    name: str
    parameter: float
    matrix: np.ndarray | None = None          # rigid or mirror
    swap_side: bool = False
    reshape: Callable[[Mesh, np.random.Generator], Mesh] | None = None

    @property
    def rigid(self) -> bool:
        return self.reshape is None


def _about(point: np.ndarray, rotation: np.ndarray, shift=np.zeros(3)) -> np.ndarray:
    """4x4: rotate about ``point``, then shift."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = point - rotation @ point + shift
    return matrix


def _axis_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _decimator(fraction: float):
    def reshape(mesh: Mesh, rng) -> Mesh:
        return as_triangle_soup(decimate(mesh, int(fraction * mesh.n_triangles)))
    return reshape


def _remesher(mesh: Mesh, rng) -> Mesh:
    return as_triangle_soup(remesh(mesh))


def _noiser(sigma_mm: float):
    def reshape(mesh: Mesh, rng) -> Mesh:
        welded = weld(mesh)
        moved = welded.vertices + rng.normal(scale=sigma_mm, size=welded.vertices.shape)
        return as_triangle_soup(Mesh(vertices=moved, faces=welded.faces))
    return reshape


def variants(knee_centre: np.ndarray, side: str, seed: int, *, quick: bool) -> list:
    """Every variant for one knee. The random motions depend on the seed only, so every
    knee gets the same ones; scanner rotations are signed anatomically."""
    rng = np.random.default_rng(seed)
    lateral = 1.0 if side == "left" else -1.0   # +X is patient-left in LPS
    out = [Variant("mirror", "mirror", 0.0, np.diag([-1.0, 1.0, 1.0, 1.0]),
                   swap_side=True)]
    for k in range(2):
        out.append(Variant("translation", f"translation_{k + 1}", SHIFT_MM,
                           _about(knee_centre, np.eye(3),
                                  SHIFT_MM * _random_unit(rng))))
    for angle in RIGID_ANGLES_DEG:
        rotation = _axis_rotation(_random_unit(rng), angle)
        out.append(Variant(f"rigid_{angle:g}", f"rigid_{angle:g}", angle,
                           _about(knee_centre, rotation,
                                  SHIFT_MM * _random_unit(rng))))
    # Anatomical sense. About X, +angle takes superior (+Z) towards anterior (-Y) on
    # either side. About Y and Z the same world rotation means opposite things on a left
    # and a right knee, so the angle is signed by side: +Y takes superior towards
    # lateral, +Z takes anterior towards lateral (external rotation).
    for axis_name, axis, sense in (("x", np.array([1.0, 0.0, 0.0]), 1.0),
                                   ("y", np.array([0.0, 1.0, 0.0]), lateral),
                                   ("z", np.array([0.0, 0.0, 1.0]), lateral)):
        for angle in SCANNER_ANGLES_DEG:
            out.append(Variant(f"scanner_{axis_name}", f"scanner_{axis_name}_{angle:+g}",
                               angle,
                               _about(knee_centre,
                                      _axis_rotation(axis, sense * angle))))
    out += [
        Variant("decimate", "decimate_50", 0.5, reshape=_decimator(0.5)),
        Variant("decimate", "decimate_10", 0.1, reshape=_decimator(0.1)),
        Variant("remesh", "remesh", 1.0, reshape=_remesher),
    ]
    out += [Variant("noise", f"noise_{sigma:g}", sigma, reshape=_noiser(sigma))
            for sigma in NOISE_MM]
    if quick:
        keep = {"mirror", "translation_1", "rigid_15", "scanner_x_+10", "scanner_z_+10",
                "scanner_z_+30", "decimate_10", "noise_0.25"}
        out = [v for v in out if v.name in keep]
    return out


def _check_sense(variants_: list, side: str) -> None:
    """The anatomical sign convention, asserted rather than trusted."""
    anterior, superior = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    lateral = np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])
    for v in variants_:
        if not v.family.startswith("scanner_") or v.parameter <= 0:
            continue
        r = v.matrix[:3, :3]
        towards = {"x": (superior, anterior), "y": (superior, lateral),
                   "z": (anterior, lateral)}[v.family[-1]]
        assert float(r @ towards[0] @ towards[1]) > 0, v.name


# ----------------------------------------------------------------------
# Measuring a variant
# ----------------------------------------------------------------------


def _moved(mesh: Mesh, matrix: np.ndarray) -> Mesh:
    return Mesh(vertices=mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3],
                faces=mesh.faces, source_path=mesh.source_path, sha256=None)


def _moved_landmarks(landmarks: LandmarkSet, matrix: np.ndarray,
                     side: str) -> LandmarkSet:
    moved = [
        replace(lm, position_mm=matrix[:3, :3] @ lm.position_mm + matrix[:3, 3])
        if lm.position_mm is not None else lm
        for lm in landmarks
    ]
    return LandmarkSet(landmarks.case_id, side, landmarks.coordinate_system, moved,
                       source=dict(landmarks.source))


def _mesh_findings(qc: QCReport) -> QCReport:
    report = QCReport()
    report.findings = [f for f in qc.findings if f.code.startswith("MESH_")]
    return report


def _outcome(run: Callable):
    """``(measurement, plan, sizing, None)`` or ``(None, None, None, failure)``."""
    try:
        measurement = run()
    except (QualityControlError, FrameConstructionError) as refusal:
        return None, None, None, ("refused", type(refusal).__name__,
                                  str(refusal).splitlines()[0][:160])
    except Exception as crash:   # noqa: BLE001 -- a crash is a finding, reported
        return None, None, None, ("crashed", type(crash).__name__, str(crash)[:160])
    try:
        plan, sizing = plan_case(measurement)
    except Exception as crash:   # noqa: BLE001
        return measurement, None, None, ("plan_failed", type(crash).__name__,
                                         str(crash)[:160])
    return measurement, plan, sizing, None


# ----------------------------------------------------------------------
# Comparing with the original
# ----------------------------------------------------------------------


def _unit(v):
    return v / np.linalg.norm(v)


def _in_plane(v, normal):
    return _unit(v - np.dot(v, normal) * normal)


def _azimuth_ml(direction, frame) -> float:
    """Rotation of a mediolateral line about the frame's proximal axis, external
    positive: 0 along the frame's lateral, positive as its lateral end turns
    posterior (which is where external rotation takes it)."""
    d = _in_plane(direction, frame.z_proximal)
    if np.dot(d, frame.lateral) < 0:
        d = -d
    return float(np.degrees(np.arctan2(-np.dot(d, frame.x_anterior),
                                       np.dot(d, frame.lateral))))


def _azimuth_ap(direction, frame) -> float:
    """As :func:`_azimuth_ml` for an anteroposterior line: positive as its anterior
    end turns lateral."""
    d = _in_plane(direction, frame.z_proximal)
    if np.dot(d, frame.x_anterior) < 0:
        d = -d
    return float(np.degrees(np.arctan2(np.dot(d, frame.lateral),
                                       np.dot(d, frame.x_anterior))))


def _wrap(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _rotation_refs(landmarks: LandmarkSet) -> dict:
    """The femoral rotational axes as raw directions, oriented patient-left."""
    def axis(lateral_id, medial_id):
        a, b = landmarks.position(lateral_id), landmarks.position(medial_id)
        if a is None or b is None:
            return None
        d = _unit(a - b)
        return d if d[0] >= 0 else -d      # sign is irrelevant to a line
    return {
        "atea": axis("femur.epicondyle_lateral", "femur.epicondyle_medial_prominence"),
        "pca": axis("femur.condyle_posterior_lateral", "femur.condyle_posterior_medial"),
    }


def _plan_values(plan, sizing) -> dict:
    femoral, tibial = plan.resections["femoral_distal"], plan.resections["tibial_proximal"]
    return {
        "valgus_cut_deg": plan.distal_femoral_valgus_cut_deg,
        "tibial_slope_deg": plan.tibial_slope_deg,
        "femoral_depth_medial_mm": femoral.medial_depth_mm,
        "femoral_depth_lateral_mm": femoral.lateral_depth_mm,
        "tibial_depth_medial_mm": tibial.medial_depth_mm,
        "tibial_depth_lateral_mm": tibial.lateral_depth_mm,
        "femoral_flexion_deg": plan.diagnostics.get("femoral_flexion_deg"),
        "insert_thickness_mm": plan.diagnostics.get("insert_thickness_mm"),
        "size_parameter": sizing.size_parameter,
        "size": sizing.nearest_discrete_size,
    }


def compare(base, base_plan, base_sizing, var, var_plan, var_sizing,
            matrix: np.ndarray) -> dict:
    """Everything about a variant run, mapped back and differenced against the original.

    ``matrix`` takes original to variant coordinates; its inverse brings the variant
    back. Directions use the linear part only, which for the mirror is the reflection
    itself.
    """
    inverse = np.linalg.inv(matrix)
    back_point = lambda p: inverse[:3, :3] @ p + inverse[:3, 3]   # noqa: E731
    back_dir = lambda d: inverse[:3, :3] @ d                       # noqa: E731
    row: dict = {}

    # Landmarks.
    displacement = {}
    status_changed = []
    for lm in base.landmarks:
        other = var.landmarks.get(lm.id)
        if lm.is_usable != other.is_usable:
            status_changed.append(lm.id)
        if lm.is_usable and other.is_usable:
            displacement[lm.id] = float(np.linalg.norm(
                back_point(other.position_mm) - lm.position_mm))
    row["landmarks"] = displacement
    row["landmark_status_changed"] = status_changed

    # Metrics.
    metrics = {}
    for name in METRICS:
        a, b = base.metrics[name].value, var.metrics[name].value
        metrics[name] = None if a is None or b is None else float(b - a)
        if (a is None) != (b is None):
            metrics[name] = float("nan")       # computable in one run only
    row["metrics"] = metrics

    # Rotational references, in the original femoral and tibial frames.
    ff, tf = base.femoral_frame, base.tibial_frame
    base_refs, var_refs = _rotation_refs(base.landmarks), _rotation_refs(var.landmarks)
    rotation = {}
    if all(v is not None for v in (*base_refs.values(), *var_refs.values())):
        base_angle = _azimuth_ml(base_refs["atea"], ff) - _azimuth_ml(base_refs["pca"], ff)
        var_angle = (_azimuth_ml(var_refs["atea"], var.femoral_frame)
                     - _azimuth_ml(var_refs["pca"], var.femoral_frame))
        rotation["atea_minus_pca_original_deg"] = base_angle
        rotation["atea_minus_pca_deg"] = _wrap(var_angle - base_angle)
        for key in ("atea", "pca"):
            rotation[f"{key}_axis_deg"] = _wrap(
                _azimuth_ml(back_dir(var_refs[key]), ff) - _azimuth_ml(base_refs[key], ff))
    rotation["tibial_ap_axis_deg"] = _wrap(
        _azimuth_ap(back_dir(var.tibial_frame.x_anterior), tf)
        - _azimuth_ap(tf.x_anterior, tf))
    rotation["femoral_proximal_axis_deg"] = float(np.degrees(np.arccos(np.clip(
        np.dot(back_dir(var.femoral_frame.z_proximal), ff.z_proximal), -1, 1))))
    rotation["tibial_proximal_axis_deg"] = float(np.degrees(np.arccos(np.clip(
        np.dot(back_dir(var.tibial_frame.z_proximal), tf.z_proximal), -1, 1))))
    row["rotation"] = rotation

    # Plan.
    plan = {}
    if base_plan is not None and var_plan is not None:
        pa, pb = _plan_values(base_plan, base_sizing), _plan_values(var_plan, var_sizing)
        for key in pa:
            if key == "size":
                plan["size_changed"] = pa[key] != pb[key]
            elif pa[key] is not None and pb[key] is not None:
                plan[key] = float(pb[key] - pa[key])
        for cut in ("femoral", "tibial"):
            name = "femoral_distal" if cut == "femoral" else "tibial_proximal"
            na = base_plan.resections[name].normal
            nb = back_dir(var_plan.resections[name].normal)
            plan[f"{cut}_cut_normal_deg"] = float(np.degrees(np.arccos(np.clip(
                np.dot(_unit(na), _unit(nb)), -1, 1))))
        for cut, key, frame in (("femoral", "femoral_component", ff),
                                ("tibial", "tibial_component", tf)):
            pose_a, pose_b = base_plan.components[key], var_plan.components[key]
            # The component's anterior: CAD +X on the femur, -Y on the tray.
            column = (0, 1.0) if cut == "femoral" else (1, -1.0)
            ant_a = column[1] * pose_a[:3, column[0]]
            ant_b = column[1] * back_dir(pose_b[:3, column[0]])
            plan[f"{cut}_component_rotation_deg"] = _wrap(
                _azimuth_ap(ant_b, frame) - _azimuth_ap(ant_a, frame))
            plan[f"{cut}_component_origin_mm"] = float(np.linalg.norm(
                back_point(pose_b[:3, 3]) - pose_a[:3, 3]))
    row["plan"] = plan
    return row


# ----------------------------------------------------------------------
# The experiment
# ----------------------------------------------------------------------


def _flatten(case_id, variant, mode, outcome, compared, seconds) -> dict:
    row = {"case": case_id, "family": variant.family, "variant": variant.name,
           "parameter": variant.parameter, "landmarks": mode,
           "status": "ok" if outcome is None else outcome[0],
           "failure": "" if outcome is None else f"{outcome[1]}: {outcome[2]}",
           "seconds": round(seconds, 2)}
    if compared is None:
        return row
    disp = compared["landmarks"]
    if disp:
        worst = max(disp, key=disp.get)
        row.update(landmark_median_mm=float(np.median(list(disp.values()))),
                   landmark_max_mm=disp[worst], worst_landmark=worst)
    row["landmark_status_changed"] = " ".join(compared["landmark_status_changed"])
    row.update({f"d_{k}": v for k, v in compared["metrics"].items()})
    row.update({f"d_{k}": v for k, v in compared["plan"].items()})
    row.update({f"rot_{k}": v for k, v in compared["rotation"].items()})
    return row


def _holds(row: dict, angle_keys, length_keys=()) -> bool | None:
    """Within tolerance on the named deltas; ``None`` for a run that did not finish.

    Landmark displacement is reported but not judged: under noise a landmark moves by
    about the noise, and what matters is whether the metrics and the plan follow.
    """
    if row["status"] != "ok":
        return None
    angles = [row.get(f"d_{k}") for k in angle_keys]
    lengths = [row.get(f"d_{k}") for k in length_keys]
    if any(a is not None and np.isnan(a) for a in angles):
        return False                       # a metric became (not) computable
    return (all(abs(a) <= ANGLE_TOL_DEG for a in angles if a is not None)
            and all(abs(x) <= LENGTH_TOL_MM for x in lengths if x is not None))


@experiment("E2", "Metamorphic invariance: mirror, rigid motion, scanner rotation, "
            "resampling, noise")
def run(ctx: Context) -> None:
    rows, landmark_rows = [], []
    for index, case in enumerate(ctx.cases):
        base = harness.measured(case)
        if isinstance(base, Exception):
            ctx.note(f"- {case.case_id} refused at baseline ({type(base).__name__}); "
                     f"not varied.")
            continue
        base_plan, base_sizing = plan_case(base)
        centre = (base.femoral_frame.origin + base.tibial_frame.origin) / 2.0
        mesh_quality = _mesh_findings(base.qc)
        knee_variants = variants(centre, case.side, ctx.seed, quick=ctx.quick)
        _check_sense(knee_variants, case.side)
        print(f"{case.case_id}: {len(knee_variants)} variants", flush=True)

        for number, variant in enumerate(knee_variants):
            side = case.side
            if variant.swap_side:
                side = "right" if side == "left" else "left"
            t0 = time.perf_counter()
            if variant.rigid:
                matrix = variant.matrix
                femur, tibia = _moved(base.femur, matrix), _moved(base.tibia, matrix)
                quality = mesh_quality
            else:
                matrix = np.eye(4)
                rng = np.random.default_rng([ctx.seed, index, number])
                femur = variant.reshape(base.femur, rng)
                tibia = variant.reshape(base.tibia, rng)
                quality = None
            measurement, plan, sizing, failure = _outcome(lambda: measure_meshes(
                femur, tibia, side, case_id=case.case_id, mesh_quality=quality))
            compared = (compare(base, base_plan, base_sizing, measurement, plan,
                                sizing, matrix) if measurement is not None else None)
            row = _flatten(case.case_id, variant, "automatic", failure, compared,
                           time.perf_counter() - t0)
            if not variant.rigid:
                row["femur_triangles"] = femur.n_triangles
                row["tibia_triangles"] = tibia.n_triangles
            rows.append(row)
            if compared is not None:
                for landmark, mm in compared["landmarks"].items():
                    landmark_rows.append({"case": case.case_id, "family": variant.family,
                                          "variant": variant.name, "landmark": landmark,
                                          "mm": mm})

            if variant.rigid:
                # Oracle landmarks: the original's, carried through the motion.
                t0 = time.perf_counter()
                oracle = _moved_landmarks(base.landmarks, matrix, side)
                o_measurement, o_plan, o_sizing, o_failure = _outcome(
                    lambda: measure_from_landmarks(
                        oracle, femur, tibia, femur_path="", tibia_path="",
                        case_id=case.case_id, side=side))
                o_compared = (compare(base, base_plan, base_sizing, o_measurement,
                                      o_plan, o_sizing, matrix)
                              if o_measurement is not None else None)
                rows.append(_flatten(case.case_id, variant, "oracle", o_failure,
                                     o_compared, time.perf_counter() - t0))
            del femur, tibia
        # The meshes are 100-200 MB a knee; holding the whole cohort does not fit.
        harness._MEASURED.pop(case.case_id, None)

    for row in rows:
        row["metrics_hold"] = _holds(row, METRICS)
        row["plan_holds"] = _holds(row, PLAN_ANGLES, PLAN_LENGTHS)
    _write(ctx, rows, landmark_rows)


# ----------------------------------------------------------------------
# Tables, figure, summary
# ----------------------------------------------------------------------


def _max_abs(values):
    values = [abs(v) for v in values if v is not None and not np.isnan(v)]
    return max(values) if values else None


def _write(ctx: Context, rows: list, landmark_rows: list) -> None:
    ctx.table("variants", rows, digits=4)

    order = list(dict.fromkeys(r["variant"] for r in rows))
    for mode in ("automatic", "oracle"):
        summary = []
        for name in order:
            group = [r for r in rows if r["variant"] == name and r["landmarks"] == mode]
            if not group:
                continue
            ok = [r for r in group if r["status"] == "ok"]
            entry = {"variant": name, "family": group[0]["family"], "knees": len(group),
                     "refused": sum(r["status"] == "refused" for r in group),
                     "crashed": sum(r["status"] in ("crashed", "plan_failed")
                                    for r in group),
                     "metrics_hold": sum(bool(r["metrics_hold"]) for r in ok),
                     "plan_holds": sum(bool(r["plan_holds"]) for r in ok),
                     "computability_changed": sum(
                         any(isinstance(r.get(f"d_{m}"), float)
                             and np.isnan(r.get(f"d_{m}")) for m in METRICS)
                         for r in ok)}
            for key in (*METRICS, *PLAN_ANGLES, *PLAN_LENGTHS, "size_parameter"):
                entry[key] = _max_abs(r.get(f"d_{key}") for r in ok)
            entry["size_changed"] = sum(bool(r.get("d_size_changed")) for r in ok)
            entry["landmark_max_mm"] = _max_abs(r.get("landmark_max_mm") for r in ok)
            for key in ROTATION:
                entry[f"rot_{key}"] = _max_abs(r.get(f"rot_{key}") for r in ok)
            summary.append(entry)
        ctx.table(f"max_abs_delta_{mode}", summary, digits=3)

    # Landmark displacement per family and landmark.
    families = list(dict.fromkeys(r["family"] for r in landmark_rows))
    landmarks = sorted({r["landmark"] for r in landmark_rows})
    per_landmark = []
    for family in families:
        for landmark in landmarks:
            values = [r["mm"] for r in landmark_rows
                      if r["family"] == family and r["landmark"] == landmark]
            if values:
                per_landmark.append({"family": family, "landmark": landmark,
                                     "n": len(values),
                                     "median_mm": float(np.median(values)),
                                     "max_mm": float(np.max(values))})
    ctx.table("landmark_displacement", per_landmark, digits=2)

    # Rotational reference against scanner tilt.
    tilt = []
    for axis in ("x", "y", "z"):
        for angle in (0.0, *SCANNER_ANGLES_DEG):
            if angle == 0.0:
                tilt.append({"axis": axis, "angle_deg": 0.0, "n": 0})
                continue
            ok = [r for r in rows if r["family"] == f"scanner_{axis}"
                  and r["parameter"] == angle and r["landmarks"] == "automatic"
                  and r["status"] == "ok"]
            entry = {"axis": axis, "angle_deg": angle, "n": len(ok)}
            for key in (*ROTATION, "femoral_proximal_axis_deg"):
                values = [r.get(f"rot_{key}") for r in ok if r.get(f"rot_{key}") is not None]
                entry[f"{key}_median"] = float(np.median(values)) if values else None
                entry[f"{key}_maxabs"] = _max_abs(values)
            for key in ("femoral_component_rotation_deg", "tibial_component_rotation_deg",
                        "mldfa_deg", "mpta_deg", "posterior_slope_medial_deg",
                        "valgus_cut_deg"):
                values = [r.get(f"d_{key}") for r in ok if r.get(f"d_{key}") is not None]
                entry[f"{key}_median"] = float(np.median(values)) if values else None
                entry[f"{key}_maxabs"] = _max_abs(values)
            tilt.append(entry)
    if any(t["n"] for t in tilt):
        ctx.table("rotation_vs_scanner_tilt", tilt, digits=2)
        _figure(ctx, rows)

    _summarise(ctx, rows, per_landmark)


def _figure(ctx: Context, rows: list) -> None:
    """Two rows against scanner rotation: the rotational references and components
    (top), and the metrics and the femoral cut (bottom). Automatic landmarks; median
    over knees with the min-to-max band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("rotation error after mapping back (deg, external +)",
         (("rot_atea_axis_deg", "aTEA axis"), ("rot_pca_axis_deg", "PCA axis"),
          ("rot_tibial_ap_axis_deg", "tibial AP (Akagi) axis"),
          ("d_femoral_component_rotation_deg", "femoral component"),
          ("d_tibial_component_rotation_deg", "tibial component"))),
        ("change from the unrotated run (deg)",
         (("d_mldfa_deg", "mLDFA"), ("d_mpta_deg", "MPTA"),
          ("d_posterior_slope_medial_deg", "medial slope"), ("d_jlca_deg", "JLCA"),
          ("d_femoral_cut_normal_deg", "femoral cut normal"),
          ("rot_femoral_proximal_axis_deg", "femoral axis"))),
    )
    fig, grid = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for (ylabel, series), axes in zip(panels, grid):
        for ax, axis in zip(axes, ("x", "y", "z")):
            for key, label in series:
                xs, med, lo, hi = [0.0], [0.0], [0.0], [0.0]
                for angle in SCANNER_ANGLES_DEG:
                    values = [r.get(key) for r in rows
                              if r["family"] == f"scanner_{axis}"
                              and r["parameter"] == angle
                              and r["landmarks"] == "automatic" and r["status"] == "ok"
                              and r.get(key) is not None]
                    if values:
                        xs.append(angle)
                        med.append(np.median(values))
                        lo.append(np.min(values))
                        hi.append(np.max(values))
                order = np.argsort(xs)
                xs, med, lo, hi = (np.asarray(v)[order] for v in (xs, med, lo, hi))
                line, = ax.plot(xs, med, marker="o", ms=3, label=label)
                ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.12)
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylabel, fontsize=9)
        axes[0].legend(fontsize=8)
    for ax, axis in zip(grid[0], ("x", "y", "z")):
        ax.set_title({"x": "about X: proximal ends anterior (+)",
                      "y": "about Y: proximal ends lateral (+)",
                      "z": "about Z: external rotation (+)"}[axis], fontsize=10)
    for ax in grid[1]:
        ax.set_xlabel("scanner rotation about the knee centre (deg)")
    fig.suptitle("E2: scanner rotation, automatic landmarks "
                 "(median over knees, band = min to max)", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(ctx.path(f"rotation_vs_scanner_tilt.{suffix}"), dpi=150)
    plt.close(fig)


def _summarise(ctx: Context, rows: list, per_landmark: list) -> None:
    auto = [r for r in rows if r["landmarks"] == "automatic"]
    knees = sorted({r["case"] for r in auto})
    ctx.note(f"{len(knees)} knees varied, {len({r['variant'] for r in auto})} variants "
             f"each; tolerance {ANGLE_TOL_DEG} deg and {LENGTH_TOL_MM} mm. Decimation "
             f"and remeshing by vertex clustering (tka_planner.geom.resample).")
    ctx.note("")
    ctx.note("'metrics hold' / 'plan holds': knees within tolerance, with automatic "
             "landmarks and with the original's landmarks carried through the motion "
             "(oracle). 'worst' columns are max abs changes over knees, automatic "
             "landmarks.")
    ctx.note("")
    ctx.note("| variant | ok | refused | crashed | metrics hold auto / oracle | "
             "plan holds auto / oracle | worst metric (deg) | worst femoral cut (deg) | "
             "worst landmark mm (which) |")
    ctx.note("|---|---|---|---|---|---|---|---|---|")

    def share(group, key):
        done = [r for r in group if r["status"] == "ok"]
        return f"{sum(bool(r[key]) for r in done)}/{len(done)}" if group else "-"

    for name in dict.fromkeys(r["variant"] for r in auto):
        group = [r for r in auto if r["variant"] == name]
        oracle = [r for r in rows if r["variant"] == name and r["landmarks"] == "oracle"]
        ok = [r for r in group if r["status"] == "ok"]
        worst_metric = _max_abs(r.get(f"d_{m}") for r in ok for m in METRICS)
        worst_cut = _max_abs(r.get("d_femoral_cut_normal_deg") for r in ok)
        worst = max(ok, key=lambda r: r.get("landmark_max_mm") or 0.0, default=None)
        ctx.note(
            f"| {name} | {len(ok)} | {sum(r['status'] == 'refused' for r in group)} | "
            f"{sum(r['status'] in ('crashed', 'plan_failed') for r in group)} | "
            f"{share(group, 'metrics_hold')} / {share(oracle, 'metrics_hold')} | "
            f"{share(group, 'plan_holds')} / {share(oracle, 'plan_holds')} | "
            + ("" if worst_metric is None else f"{worst_metric:.3f}") + " | "
            + ("" if worst_cut is None else f"{worst_cut:.2f}") + " | "
            + ("" if worst is None or worst.get("landmark_max_mm") is None else
               f"{worst['landmark_max_mm']:.2f} ({worst['worst_landmark']})")
            + " |")
    failures = [r for r in auto if r["status"] != "ok"]
    if failures:
        ctx.note("")
        ctx.note("Refusals and crashes:")
        for r in failures:
            ctx.note(f"- {r['case']} {r['variant']}: {r['status']} -- {r['failure']}")
