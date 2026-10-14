"""Monte Carlo propagation of landmark picking error through the planner.

A plan is a function of perhaps twenty picked points, and every one of them could have
been picked a millimetre or two elsewhere by another surgeon, or by the same surgeon a
week later. How much the plan moves when they do is the question this answers: it
draws many plausible landmark sets around the one in hand, rebuilds the frames, the
metrics and the plan from each, and reports the spread of every output.

Only the landmarks are perturbed. The automatic estimator is not re-run per draw, so
what is measured is the planner's sensitivity to where the points are, not the
estimator's sensitivity to the mesh -- the estimator's own error against a surgeon is a
separate experiment (E3), whose measured dispersion is what this module is built to
take as input.

The dispersion table
--------------------
One row per landmark (CSV, see :func:`read_dispersion`): ``landmark_id,sd_mm``, where
``sd_mm`` is the standard deviation along each axis of an isotropic Gaussian, in
millimetres. Optional columns ``cov_aa, cov_al, cov_ap, cov_ll, cov_lp, cov_pp`` give a
full 3x3 covariance (mm^2) instead, in the landmark's own bone frame -- anterior,
lateral, proximal -- so one table serves left and right knees and knees lying at any
angle in the scanner. A landmark with no row is held fixed, and the result says which.
Lines starting with ``#`` are comments; the first ones label the table, and a table
whose comments say ``PLACEHOLDER`` is reported as one.

If an inter-observer study reports the root-mean-square 3D distance of picks from their
mean, ``r``, the isotropic per-axis SD is ``r / sqrt(3)`` for a point free in space and
``r / sqrt(2)`` for one picked on a surface (see ``surface`` below).

Where a perturbed point may go
------------------------------
A pick made on a bone model cannot leave its surface, so free 3D noise is the wrong
model for surface landmarks: it adds an error along the surface normal that no observer
can make. On the distal condyles that normal *is* the proximal direction, which is the
direction resection depth and the coronal joint line are read along, so free noise
inflates exactly the outputs that matter most. Three models are offered:

``tangent`` (the default)
    The displacement's component along the local surface normal is removed, so a
    surface landmark moves within its tangent plane. Linear in the SD, needs one normal
    per landmark, and exact up to the surface's curvature over a few millimetres.
``project``
    The freely displaced point is snapped to the closest point on the mesh. Exact for
    any curvature, but non-linear, and it lets a point slide around an edge.
``free``
    Isotropic 3D noise, ignoring the surface. Needs no mesh; an upper bound.

Landmarks that are not on the surface -- the canal centres, the head and ankle centres,
anything more than :data:`OFF_SURFACE_MM` from its mesh -- always move freely.

Everything here is numpy and the planner's own pipeline: each draw goes through
:func:`~tka_planner.pipeline.measure_from_landmarks` and
:func:`~tka_planner.pipeline.plan_case`, the same functions the application calls.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..core.frames import AnatomicalFrame, FrameConstructionError
from ..core.landmarks import REGISTRY, Landmark, LandmarkSet, MissingLandmarks
from ..core.meshio import Mesh

__all__ = [
    "Dispersion",
    "read_dispersion",
    "write_dispersion",
    "placeholder_dispersion",
    "PLACEHOLDER_TABLES",
    "SurfaceModel",
    "perturb",
    "PropagationResult",
    "propagate",
    "plan_outputs",
    "SURFACE_MODES",
    "OFF_SURFACE_MM",
]

SURFACE_MODES = ("tangent", "project", "free")

# Landmarks defined as centres rather than surface points. They move freely whatever the
# surface model, because there is no surface for them to stay on.
CENTRE_LANDMARKS = frozenset({
    "femur.head_centre",
    "femur.canal_centre_distal",
    "femur.canal_centre_proximal",
    "tibia.canal_centre_proximal",
    "tibia.canal_centre_distal",
    "tibia.ankle_centre",
    "patella.centre",
})

# A landmark further than this from its bone's surface is treated as off the surface.
# Picks and automatic estimates both land on the mesh, so anything further out is a
# computed point (a midpoint, a centroid) that the surface does not constrain.
OFF_SURFACE_MM = 2.0

_COV_COLUMNS = ("cov_aa", "cov_al", "cov_ap", "cov_ll", "cov_lp", "cov_pp")
_TABLE_DIR = Path(__file__).resolve().parent / "dispersion"
PLACEHOLDER_TABLES = {
    "uniform_2mm": _TABLE_DIR / "placeholder_uniform_2mm.csv",
    "registry": _TABLE_DIR / "placeholder_registry.csv",
}


# ----------------------------------------------------------------------
# The dispersion table
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Dispersion:
    """How far each landmark might be picked from where it is, as a covariance.

    ``covariances`` maps a landmark id to a 3x3 covariance in mm^2, expressed in the
    landmark's bone frame as (anterior, lateral, proximal). An isotropic SD ``s`` is
    ``s**2 * I``. ``label`` is the table's own description, carried into every result so
    a placeholder can never be mistaken for a measurement.
    """

    covariances: dict[str, np.ndarray]
    label: str = ""
    placeholder: bool = False

    def __post_init__(self):
        for landmark_id, cov in self.covariances.items():
            if landmark_id not in REGISTRY:
                raise KeyError(f"Dispersion names an unknown landmark {landmark_id!r}.")
            cov = np.asarray(cov, dtype=float)
            if cov.shape != (3, 3) or not np.allclose(cov, cov.T):
                raise ValueError(f"{landmark_id}: covariance must be a symmetric 3x3.")
            if np.linalg.eigvalsh(cov).min() < -1e-9:
                raise ValueError(f"{landmark_id}: covariance is not positive semidefinite.")

    @classmethod
    def isotropic(cls, sd_mm: dict[str, float], *, label: str = "",
                  placeholder: bool = False) -> "Dispersion":
        return cls({k: (float(v) ** 2) * np.eye(3) for k, v in sd_mm.items()},
                   label=label, placeholder=placeholder)

    def sd_mm(self, landmark_id: str) -> float:
        """The per-axis SD equivalent to this landmark's covariance (its RMS axis SD)."""
        cov = self.covariances.get(landmark_id)
        return 0.0 if cov is None else float(np.sqrt(np.trace(cov) / 3.0))

    def scaled(self, factor: float) -> "Dispersion":
        """The same table with every SD multiplied by ``factor``."""
        return Dispersion({k: (factor ** 2) * np.asarray(v, dtype=float)
                           for k, v in self.covariances.items()},
                          label=f"{self.label} x{factor:g}", placeholder=self.placeholder)


def read_dispersion(path) -> Dispersion:
    """Read a dispersion table: ``landmark_id,sd_mm`` with optional covariance columns.

    Blank ``sd_mm`` and covariance cells, or ``sd_mm`` of zero, hold the landmark fixed,
    which is also what leaving its row out does.
    """
    path = Path(path)
    comments, rows = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        body = []
        for line in handle:
            if line.lstrip().startswith("#"):
                comments.append(line.lstrip()[1:].strip())
            elif line.strip():
                body.append(line)
        rows = list(csv.DictReader(body))

    covariances = {}
    for row in rows:
        landmark_id = (row.get("landmark_id") or "").strip()
        if not landmark_id:
            continue
        if all((row.get(c) or "").strip() for c in _COV_COLUMNS):
            aa, al, ap, ll, lp, pp = (float(row[c]) for c in _COV_COLUMNS)
            covariances[landmark_id] = np.array([[aa, al, ap], [al, ll, lp], [ap, lp, pp]])
        elif (row.get("sd_mm") or "").strip():
            sd = float(row["sd_mm"])
            if sd > 0:
                covariances[landmark_id] = sd * sd * np.eye(3)
    return Dispersion(covariances, label=comments[0] if comments else path.stem,
                      placeholder="PLACEHOLDER" in " ".join(comments).upper())


def write_dispersion(dispersion: Dispersion, path, *, comments: tuple[str, ...] = ()) -> Path:
    """Write a table :func:`read_dispersion` reads back exactly."""
    path = Path(path)
    lines = [f"# {c}" for c in (comments or (dispersion.label,)) if c]
    lines.append("landmark_id,sd_mm," + ",".join(_COV_COLUMNS))
    for landmark_id, cov in dispersion.covariances.items():
        cov = np.asarray(cov, dtype=float)
        if np.allclose(cov, cov[0, 0] * np.eye(3)):
            lines.append(f"{landmark_id},{np.sqrt(cov[0, 0]):.6g}" + "," * len(_COV_COLUMNS))
        else:
            cells = (cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2])
            lines.append(f"{landmark_id},," + ",".join(f"{v:.6g}" for v in cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def placeholder_dispersion(name: str = "uniform_2mm") -> Dispersion:
    """One of the shipped placeholder tables, until E3 measures the real dispersion."""
    if name not in PLACEHOLDER_TABLES:
        raise KeyError(f"No placeholder table {name!r}; have {sorted(PLACEHOLDER_TABLES)}.")
    return read_dispersion(PLACEHOLDER_TABLES[name])


# ----------------------------------------------------------------------
# Drawing perturbed landmark sets
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceModel:
    """Per landmark, how it may move: its surface normal, or free, plus the local mesh.

    Built once per case by :meth:`build`, because finding the surface near each landmark
    is the only part of a draw that touches the mesh.
    """

    mode: str
    normals: dict[str, np.ndarray] = field(default_factory=dict)  # on-surface only
    patches: dict[str, np.ndarray] = field(default_factory=dict)  # (T, 3, 3) for project
    distance_mm: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, landmarks: LandmarkSet, meshes: dict[str, Mesh] | None,
              mode: str = "tangent", *, reach_mm: float = 15.0) -> "SurfaceModel":
        if mode not in SURFACE_MODES:
            raise ValueError(f"surface must be one of {SURFACE_MODES}.")
        if mode == "free":
            return cls(mode)
        if not meshes:
            raise ValueError(f"surface={mode!r} needs the bone meshes; pass "
                             f"surface='free' to perturb without them.")
        normals, patches, distances = {}, {}, {}
        corners_by_bone = {bone: mesh.vertices[mesh.faces] for bone, mesh in meshes.items()}
        centroids_by_bone = {bone: c.mean(axis=1) for bone, c in corners_by_bone.items()}
        for landmark in landmarks:
            if not landmark.is_usable or landmark.id in CENTRE_LANDMARKS:
                continue
            bone = "tibia" if landmark.definition.bone == "fibula" else landmark.definition.bone
            if bone not in corners_by_bone:
                continue
            point = landmark.position_mm
            near = np.linalg.norm(centroids_by_bone[bone] - point, axis=1) <= reach_mm
            if not near.any():
                continue
            patch = corners_by_bone[bone][near]
            closest = closest_point_on_triangles(point[None, :], patch)[0]
            gaps = np.linalg.norm(closest - point, axis=1)
            distances[landmark.id] = float(gaps.min())
            if gaps.min() > OFF_SURFACE_MM:
                continue
            # Area-weighted normal of the faces within 2 mm of the foot point: one
            # triangle's normal is as noisy as the segmentation, a small patch is not.
            local = gaps <= gaps.min() + 2.0
            cross = np.cross(patch[local, 1] - patch[local, 0], patch[local, 2] - patch[local, 0])
            # Faces are oriented inconsistently in a triangle soup, so align before summing.
            reference = cross[int(np.argmax(np.linalg.norm(cross, axis=1)))]
            cross = np.where((cross @ reference)[:, None] < 0, -cross, cross)
            normal = cross.sum(axis=0)
            if np.linalg.norm(normal) < 1e-12:
                continue
            normals[landmark.id] = normal / np.linalg.norm(normal)
            if mode == "project":
                patches[landmark.id] = patch
        return cls(mode, normals, patches, distances)

    def on_surface(self, landmark_id: str) -> bool:
        return landmark_id in self.normals


def _bone_axes(landmark_id: str, frames: dict[str, AnatomicalFrame]) -> np.ndarray:
    """Rows: the landmark's bone frame's anterior, lateral and proximal, in world."""
    bone = REGISTRY[landmark_id].bone
    frame = frames["tibia" if bone in ("tibia", "fibula") else "femur"]
    return np.array([frame.x_anterior, frame.lateral, frame.z_proximal])


def perturb(
    landmarks: LandmarkSet,
    dispersion: Dispersion,
    rng: np.random.Generator,
    *,
    frames: dict[str, AnatomicalFrame],
    surface: SurfaceModel,
) -> LandmarkSet:
    """One plausible re-pick of ``landmarks``.

    Every usable landmark with a row in ``dispersion`` is displaced; the rest keep their
    positions. Landmarks are drawn in a fixed (sorted) order so a seed fixes the draw
    regardless of how the set was assembled. ``frames`` (the nominal femoral and tibial
    frames, keyed ``femur`` and ``tibia``) orient anisotropic covariances.
    """
    moved = []
    for landmark_id in sorted(lm.id for lm in landmarks):
        landmark = landmarks.get(landmark_id)
        position = landmark.position_mm
        cov = dispersion.covariances.get(landmark_id)
        if landmark.is_usable and cov is not None:
            axes = _bone_axes(landmark_id, frames)
            local = rng.multivariate_normal(np.zeros(3), cov, method="eigh")
            delta = local @ axes
            if surface.on_surface(landmark_id):
                if surface.mode == "tangent":
                    normal = surface.normals[landmark_id]
                    delta = delta - float(delta @ normal) * normal
                    position = position + delta
                else:  # project
                    candidate = position + delta
                    feet = closest_point_on_triangles(candidate[None, :],
                                                      surface.patches[landmark_id])[0]
                    position = feet[int(np.argmin(np.linalg.norm(feet - candidate, axis=1)))]
            else:
                position = position + delta
        moved.append(Landmark(
            id=landmark.id, position_mm=position, status=landmark.status,
            origin=landmark.origin, reason=landmark.reason,
            metadata=dict(landmark.metadata),
        ))
    return LandmarkSet(
        case_id=landmarks.case_id, side=landmarks.side,
        coordinate_system=landmarks.coordinate_system, landmarks=moved,
        source={**landmarks.source, "perturbed": dispersion.label},
    )


# ----------------------------------------------------------------------
# What is recorded from each draw
# ----------------------------------------------------------------------


def _angle_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cosine = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _in_plane(vector, normal):
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    vector = np.asarray(vector, dtype=float)
    vector = vector - float(vector @ normal) * normal
    return vector / np.linalg.norm(vector)


def _turn_deg(vector, reference, normal, lateral) -> float:
    """Signed turn of ``vector`` from ``reference`` about ``normal``, + toward lateral."""
    reference = _in_plane(reference, normal)
    vector = _in_plane(vector, normal)
    side = np.cross(np.asarray(normal, dtype=float) / np.linalg.norm(normal), reference)
    if float(side @ _in_plane(lateral, normal)) < 0:
        side = -side
    return float(np.degrees(np.arctan2(vector @ side, vector @ reference)))


def _tilt_deg(normal, frame: AnatomicalFrame, towards: np.ndarray) -> float:
    """Angle of a cut normal from the frame's proximal axis, toward ``towards``."""
    return float(np.degrees(np.arctan2(normal @ towards, normal @ frame.z_proximal)))


def plan_outputs(measurement, plan, sizing, *, reference=None) -> dict[str, float]:
    """The numbers recorded from one draw, as plain floats keyed by name.

    Three groups. The *metrics* and the *plan as reported* are what the planner would
    print for this re-pick. The *plan on the nominal bone* is where this draw's cuts and
    components would actually land, read against the unperturbed landmarks and frames
    in ``reference`` -- a ``(measurement, plan)`` pair -- because the bone does not move
    when a surgeon picks a point elsewhere; only the plan does. Signs follow the
    planner's adjustments: + valgus, + flexion/posterior slope, + external rotation.
    """
    out: dict[str, float] = {}
    for name, metric in measurement.metrics.items():
        if metric.value is not None:
            out[f"metric.{name}"] = float(metric.value)

    femoral = plan.resections["femoral_distal"]
    tibial = plan.resections["tibial_proximal"]
    gaps = plan.diagnostics
    out.update({
        "plan.valgus_cut_deg": float(plan.distal_femoral_valgus_cut_deg),
        "plan.femoral_depth_medial_mm": femoral.medial_depth_mm,
        "plan.femoral_depth_lateral_mm": femoral.lateral_depth_mm,
        "plan.tibial_depth_medial_mm": tibial.medial_depth_mm,
        "plan.tibial_depth_lateral_mm": tibial.lateral_depth_mm,
        "plan.size_parameter": float(sizing.size_parameter),
        "plan.femoral_ml_mm": float(measurement.femoral_measure.ml_mm),
        "plan.femoral_ap_mm": float(measurement.femoral_measure.ap_mm),
        "plan.tibial_ml_mm": float(measurement.tibial_measure.ml_mm),
        "plan.tibial_ap_mm": float(measurement.tibial_measure.ap_mm),
    })
    if "insert_thickness_mm" in gaps:
        out["plan.insert_thickness_mm"] = float(gaps["insert_thickness_mm"])
        out["plan.extension_imbalance_mm"] = float(gaps["extension_imbalance_mm"])

    if reference is None:
        return out
    nominal_m, nominal_plan = reference
    f0, t0 = nominal_m.femoral_frame, nominal_m.tibial_frame
    out["frame.femoral_axis_change_deg"] = _angle_deg(measurement.femoral_frame.z_proximal,
                                                      f0.z_proximal)
    out["frame.tibial_axis_change_deg"] = _angle_deg(measurement.tibial_frame.z_proximal,
                                                     t0.z_proximal)
    out["frame.femoral_rotation_change_deg"] = _turn_deg(
        measurement.femoral_frame.x_anterior, f0.x_anterior, f0.z_proximal, f0.lateral)
    out["frame.tibial_rotation_change_deg"] = _turn_deg(
        measurement.tibial_frame.x_anterior, t0.x_anterior, t0.z_proximal, t0.lateral)

    nominal_lm = nominal_m.landmarks
    for cut, frame, nominal_cut, ids, removes in (
        (femoral, f0, nominal_plan.resections["femoral_distal"],
         ("femur.condyle_distal_medial", "femur.condyle_distal_lateral"), 1.0),
        (tibial, t0, nominal_plan.resections["tibial_proximal"],
         ("tibia.plateau_medial_lowest", "tibia.plateau_lateral_lowest"), -1.0),
    ):
        name = "femoral" if removes > 0 else "tibial"
        normal = np.asarray(cut.normal, dtype=float)
        nominal_normal = np.asarray(nominal_cut.normal, dtype=float)
        out[f"bone.{name}_cut_coronal_deg"] = (
            _tilt_deg(normal, frame, frame.lateral)
            - _tilt_deg(nominal_normal, frame, frame.lateral))
        out[f"bone.{name}_cut_sagittal_deg"] = (
            _tilt_deg(normal, frame, frame.posterior)
            - _tilt_deg(nominal_normal, frame, frame.posterior))
        for compartment, landmark_id in zip(("medial", "lateral"), ids):
            point = nominal_lm.position(landmark_id)
            depth = removes * float((np.asarray(cut.point) - point) @ normal)
            out[f"bone.{name}_depth_{compartment}_mm"] = depth

    poses = plan.components
    nominal_poses = nominal_plan.components
    femoral_normal = np.asarray(nominal_plan.resections["femoral_distal"].normal)
    tibial_normal = np.asarray(nominal_plan.resections["tibial_proximal"].normal)
    # Femoral parts carry anterior on local +X; tibial parts carry posterior on +Y.
    out["bone.femoral_component_rotation_deg"] = _turn_deg(
        poses["femoral_component"][:3, 0], nominal_poses["femoral_component"][:3, 0],
        femoral_normal, f0.lateral)
    out["bone.tibial_component_rotation_deg"] = _turn_deg(
        -poses["tibial_component"][:3, 1], -nominal_poses["tibial_component"][:3, 1],
        tibial_normal, t0.lateral)
    for name, key, normal, frame in (
        ("femoral", "femoral_component", femoral_normal, f0),
        ("tibial", "tibial_component", tibial_normal, t0),
    ):
        shift = poses[key][:3, 3] - nominal_poses[key][:3, 3]
        anterior = _in_plane(frame.x_anterior, normal)
        lateral = _in_plane(frame.lateral, normal)
        out[f"bone.{name}_component_shift_ap_mm"] = float(shift @ anterior)
        out[f"bone.{name}_component_shift_ml_mm"] = float(shift @ lateral)
    return out


# ----------------------------------------------------------------------
# The propagation
# ----------------------------------------------------------------------


@dataclass
class PropagationResult:
    """Every draw's outputs, the nominal plan's, and what failed."""

    case_id: str
    side: str
    n_draws: int
    seed: int
    surface: str
    dispersion_label: str
    placeholder: bool
    nominal: dict[str, float]
    samples: dict[str, np.ndarray]
    sizes: list[str]
    nominal_size: str
    failures: dict[str, int]
    failure_examples: dict[str, str]
    perturbed: tuple[str, ...]
    held_fixed: tuple[str, ...]
    free_moving: tuple[str, ...]

    @property
    def n_failed(self) -> int:
        return int(sum(self.failures.values()))

    def summary(self) -> list[dict]:
        """Per output: nominal, mean, SD, the 2.5th and 97.5th percentiles, and n."""
        rows = []
        for name, values in self.samples.items():
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                continue
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append({
                "output": name,
                "nominal": self.nominal.get(name),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "p2_5": float(low),
                "p97_5": float(high),
                "width_95": float(high - low),
                "n": int(values.size),
            })
        return rows

    def size_agreement(self) -> float:
        """Fraction of successful draws landing on the nominal plan's chart size."""
        if not self.sizes:
            return float("nan")
        return sum(s == self.nominal_size for s in self.sizes) / len(self.sizes)


def propagate(
    measurement,
    dispersion: Dispersion,
    *,
    n: int = 500,
    seed: int = 0,
    surface: str = "tangent",
    library=None,
    philosophy: str = "mechanical",
    tibial_reference: str = "less_affected_plateau",
) -> PropagationResult:
    """Draw ``n`` re-picks of a measured case and re-plan each one.

    ``measurement`` is a :class:`~tka_planner.pipeline.CaseMeasurement`; its landmarks
    are the centre of every draw and its meshes give the surface. Each draw runs the
    same :func:`~tka_planner.pipeline.measure_from_landmarks` and
    :func:`~tka_planner.pipeline.plan_case` as the application. A draw whose frames
    cannot be built -- a re-pick that turns the tibial AP reference backwards, say -- is
    counted by the error's type and left out of the statistics; it is a result, not a
    crash.
    """
    from ..pipeline import discrete_sizing, measure_from_landmarks, plan_case

    rng = np.random.default_rng(seed)
    nominal_plan, nominal_sizing = plan_case(
        measurement, philosophy=philosophy, tibial_reference=tibial_reference,
        library=library)
    nominal = plan_outputs(measurement, nominal_plan, nominal_sizing,
                           reference=(measurement, nominal_plan))
    nominal_size = discrete_sizing(measurement).nearest_discrete_size

    meshes = {"femur": measurement.femur, "tibia": measurement.tibia}
    model = SurfaceModel.build(measurement.landmarks, meshes if surface != "free" else None,
                               surface)
    frames = {"femur": measurement.femoral_frame, "tibia": measurement.tibial_frame}
    usable = sorted(lm.id for lm in measurement.landmarks if lm.is_usable)
    perturbed = tuple(i for i in usable if i in dispersion.covariances)
    held = tuple(i for i in usable if i not in dispersion.covariances)
    free = tuple(i for i in perturbed if not model.on_surface(i))

    samples: dict[str, list[float]] = {}
    sizes: list[str] = []
    failures: dict[str, int] = {}
    examples: dict[str, str] = {}
    for _ in range(n):
        draw = perturb(measurement.landmarks, dispersion, rng, frames=frames,
                       surface=model)
        try:
            m = measure_from_landmarks(
                draw, measurement.femur, measurement.tibia,
                femur_path=measurement.femur_path, tibia_path=measurement.tibia_path,
                case_id=measurement.case_id, side=measurement.side,
                size_chart=measurement.chart_path,
            )
            plan, sizing = plan_case(m, philosophy=philosophy,
                                     tibial_reference=tibial_reference, library=library)
        except (FrameConstructionError, MissingLandmarks, ValueError) as error:
            key = type(error).__name__
            failures[key] = failures.get(key, 0) + 1
            examples.setdefault(key, str(error).splitlines()[-1].strip()[:200])
            continue
        for name, value in plan_outputs(m, plan, sizing,
                                        reference=(measurement, nominal_plan)).items():
            samples.setdefault(name, []).append(value)
        sizes.append(discrete_sizing(m).nearest_discrete_size)

    return PropagationResult(
        case_id=measurement.case_id,
        side=str(measurement.side),
        n_draws=n,
        seed=seed,
        surface=surface,
        dispersion_label=dispersion.label,
        placeholder=dispersion.placeholder,
        nominal=nominal,
        samples={k: np.asarray(v, dtype=float) for k, v in samples.items()},
        sizes=sizes,
        nominal_size=nominal_size,
        failures=failures,
        failure_examples=examples,
        perturbed=perturbed,
        held_fixed=held,
        free_moving=free,
    )


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def closest_point_on_triangles(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Closest point on each triangle to each point: ``(P, 3)`` x ``(T, 3, 3)`` -> ``(P, T, 3)``.

    The barycentric-region method (Ericson, *Real-Time Collision Detection*, 5.1.5),
    vectorised. The same routine the kernel tests use, repeated here because the
    package cannot import from ``tests``.
    """
    a = corners[None, :, 0, :]
    b = corners[None, :, 1, :]
    c = corners[None, :, 2, :]
    p = points[:, None, :]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ptk,ptk->pt", ab, ap)
    d2 = np.einsum("ptk,ptk->pt", ac, ap)
    bp = p - b
    d3 = np.einsum("ptk,ptk->pt", ab, bp)
    d4 = np.einsum("ptk,ptk->pt", ac, bp)
    cp = p - c
    d5 = np.einsum("ptk,ptk->pt", ab, cp)
    d6 = np.einsum("ptk,ptk->pt", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denominator = va + vb + vc
    safe = np.where(np.abs(denominator) < 1e-300, 1.0, denominator)
    closest = a + ab * (vb / safe)[..., None] + ac * (vc / safe)[..., None]

    def pick(condition, value):
        return np.where(condition[..., None], value, closest)

    closest = pick((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, closest.shape))
    closest = pick((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, closest.shape))
    closest = pick((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, closest.shape))
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t_ab = np.where(edge_ab, d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3), 0.0)
    closest = pick(edge_ab, a + ab * t_ab[..., None])
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t_ac = np.where(edge_ac, d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6), 0.0)
    closest = pick(edge_ac, a + ac * t_ac[..., None])
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    span = (d4 - d3) + (d5 - d6)
    t_bc = np.where(edge_bc, (d4 - d3) / np.where(span == 0, 1.0, span), 0.0)
    return pick(edge_bc, b + (c - b) * t_bc[..., None])
