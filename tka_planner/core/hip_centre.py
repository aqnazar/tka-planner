"""The femoral head centre measured by pivoting the femur under a tracking camera.

Every scan in this cohort is knee-only, so the femoral head -- and with it the femoral
mechanical axis -- is outside the CT. The planner then substitutes a population
mechanical-anatomical angle and says so. This module is the way out of that assumption
that does not need more imaging: with a tracked marker fixed to the femur, the surgeon
circumducts the leg about the hip, and the hip centre is the point the motion turns
about.

Two formulations are offered, and they are not interchangeable.

**Sphere fit** (:func:`fit_sphere_pivoting`). A point fixed to the femur stays at a
constant distance from the hip centre, so its trajectory lies on a sphere about it.
This needs only marker *positions*. It starts from the algebraic (Kasa) fit and refines
the true orthogonal distance with Levenberg-Marquardt, because the algebraic fit is
biased on the small caps that a hip arc produces. Several marker spheres on one array
share the centre and keep their own radii. It uses positions only, so its noise is the
tracker's point noise.

**Pivot** (:func:`fit_pivot`). With full rigid-body poses ``(R_i, t_i)`` of the marker,
the hip centre is the one point ``p`` fixed in the marker frame that stays put in the
camera frame: ``R_i p + t_i = c`` for every sample. That is linear in ``(p, c)``, solved
in one least-squares step. It uses the marker's orientation as well as its position;
the price is that rotational noise is multiplied by the lever arm from marker to hip
(about 400 mm).

Both are degenerate for the same motion: rotation about a single axis -- one plane of
flexion, or the leg spun rigidly about one axis -- because every point on that axis is
then a valid centre. The sphere fit on a *single* tracked point is degenerate more
often, whenever that point's path is a circle, which is why the array's spheres are
fitted jointly.

Both return a :class:`HipCentreFit` carrying the centre, the residual, how much of a
sphere the motion covered, and a predicted uncertainty, and both *refuse* -- raise
:class:`HipCentreFitError` -- when the motion cannot determine the centre, rather than
return a point that looks like a measurement and is not. What counts as "cannot" is a
:class:`RefusalPolicy`, so the thresholds are stated and can be argued with.

Frames. Everything here is in the **tracker (camera) frame**. The planner works in the
CT's LPS frame, and the transform between them is a registration that belongs to the
navigation system. :func:`head_centre_landmark` takes that transform as an explicit
argument and does nothing to estimate it.

Numpy only; millimetres; angles in degrees where the name says ``_deg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .landmarks import Landmark, LandmarkSet, LandmarkStatus

__all__ = [
    "SPHERE_METHOD",
    "PIVOT_METHOD",
    "FUNCTIONAL_SOURCE",
    "HipCentreFit",
    "HipCentreFitError",
    "RefusalPolicy",
    "DEFAULT_POLICY",
    "PERMISSIVE_POLICY",
    "fit_sphere_pivoting",
    "fit_pivot",
    "poses_from_marker_points",
    "arc_coverage",
    "head_centre_landmark",
    "functional_head_centre_set",
]

SPHERE_METHOD = "hip_centre.sphere.v1"
PIVOT_METHOD = "hip_centre.pivot.v1"

# The human-readable source a functional head centre carries into the report, so a
# reader can tell it from a head centre fitted to the CT. See developer question L-10.
FUNCTIONAL_SOURCE = "functional, camera"


# ----------------------------------------------------------------------
# Results and refusal
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HipCentreFit:
    """A hip centre in the tracker frame, with what is needed to judge it.

    ``rms_residual_mm`` means different things for the two methods, so ``residual_kind``
    names it: for the sphere fit it is the radial misfit of the marker points, for the
    pivot it is the 3D scatter of the femur-fixed point about the centre. Neither
    reveals pelvic motion that moves in step with the leg: such motion still produces
    a clean sphere, only a larger one. That is why the synthetic harness (experiment
    E6c) measures the bias directly instead of trusting the residual.

    ``arc_minor_deg`` and ``arc_major_deg`` describe how much of the sphere the motion
    covered, as the half-angle of an equivalent circumduction cone in the weakest and
    the strongest direction (see :func:`arc_coverage`). A pure flexion arc has a
    respectable major angle and a minor angle near zero, which is the degenerate case.

    ``centre_sd_mm`` is the predicted one-sigma uncertainty of the centre in its
    worst-determined direction, from the residual and the fit's Jacobian. It is a
    linearised estimate; E6c checks it against the observed error.
    """

    centre_mm: np.ndarray
    method: str
    n_samples: int
    rms_residual_mm: float
    residual_kind: str
    arc_minor_deg: float
    arc_major_deg: float
    arc_max_deg: float
    centre_sd_mm: float
    condition_number: float
    converged: bool
    iterations: int
    radius_mm: float
    pivot_in_marker_mm: np.ndarray | None = None
    centre_covariance: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    frame: str = "tracker"

    def to_dict(self) -> dict:
        record = {
            "method": self.method,
            "frame": self.frame,
            "centre_mm": [round(float(v), 4) for v in self.centre_mm],
            "n_samples": self.n_samples,
            "rms_residual_mm": round(self.rms_residual_mm, 4),
            "residual_kind": self.residual_kind,
            "arc_minor_deg": round(self.arc_minor_deg, 3),
            "arc_major_deg": round(self.arc_major_deg, 3),
            "arc_max_deg": round(self.arc_max_deg, 3),
            "centre_sd_mm": round(self.centre_sd_mm, 4),
            "condition_number": float(f"{self.condition_number:.4g}"),
            "converged": self.converged,
            "iterations": self.iterations,
            "radius_mm": round(self.radius_mm, 3),
        }
        if self.pivot_in_marker_mm is not None:
            record["pivot_in_marker_mm"] = [round(float(v), 4)
                                            for v in self.pivot_in_marker_mm]
        return record


class HipCentreFitError(ValueError):
    """The motion did not determine the hip centre; carries the reasons, and the fit
    when one could be computed, so a caller can show the surgeon what to repeat."""

    def __init__(self, reasons: list[str], fit: HipCentreFit | None = None):
        self.reasons = list(reasons)
        self.fit = fit
        super().__init__("Hip centre refused: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class RefusalPolicy:
    """When a pivoting fit is refused rather than returned.

    The defaults are a parameterised proposal, not a clinical standard, read off the
    synthetic accuracy curves of experiment E6c (a ~425 mm marker lever arm, tracker
    noise 0.1-0.5 mm RMS). Tracker noise alone would allow a 5 degree cone. What sets
    ``min_arc_deg`` is pelvic motion: a hip that moves ``A`` mm with the leg shifts the
    fitted centre by about ``A / tan(cone)``, and the residual does not show it. At
    15 degrees, 2 mm of such motion costs about 7.5 mm, a degree of axis if it all lay
    across the axis. ``max_centre_sd_mm`` refuses a fit whose own predicted
    uncertainty is large (2 mm one-sigma is about 4 mm at the 95th percentile, half a
    degree on a 430 mm femur). ``max_condition`` only catches motion that is degenerate
    outright: rotation about a single axis.
    """

    min_samples: int = 30
    min_arc_deg: float = 15.0
    max_centre_sd_mm: float = 2.0
    max_condition: float = 1e6

    def reasons(self, fit: HipCentreFit) -> list[str]:
        """Every rule the fit breaks, in words a user can act on."""
        found = []
        if fit.n_samples < self.min_samples:
            found.append(f"{fit.n_samples} samples, fewer than {self.min_samples}")
        if not fit.converged:
            found.append("the geometric refinement did not converge")
        if not np.isfinite(fit.condition_number) or (
                fit.condition_number > self.max_condition):
            found.append(
                f"the motion is degenerate (condition number "
                f"{fit.condition_number:.3g} > {self.max_condition:.3g}): move the leg "
                f"in two directions, not about one axis")
        if fit.arc_minor_deg < self.min_arc_deg:
            found.append(
                f"arc too small: equivalent cone half-angle {fit.arc_minor_deg:.1f} deg "
                f"in the weakest direction, below {self.min_arc_deg:.1f} deg")
        if not np.isfinite(fit.centre_sd_mm) or fit.centre_sd_mm > self.max_centre_sd_mm:
            found.append(
                f"predicted centre uncertainty {fit.centre_sd_mm:.2f} mm (1 sd) "
                f"exceeds {self.max_centre_sd_mm:.2f} mm")
        return found

    def check(self, fit: HipCentreFit) -> HipCentreFit:
        reasons = self.reasons(fit)
        if reasons:
            raise HipCentreFitError(reasons, fit)
        return fit


DEFAULT_POLICY = RefusalPolicy()

# For characterising the fit itself (E6c): every result that can be computed is
# returned, so the accuracy curves extend below the default thresholds. Exactly
# degenerate motion is still refused, because no centre exists to return.
PERMISSIVE_POLICY = RefusalPolicy(min_samples=4, min_arc_deg=0.0,
                                  max_centre_sd_mm=float("inf"), max_condition=1e12)


# ----------------------------------------------------------------------
# Arc coverage
# ----------------------------------------------------------------------


def arc_coverage(points: np.ndarray, centre: np.ndarray) -> tuple[float, float, float]:
    """How much of a sphere about ``centre`` the points cover.

    Returns ``(minor_deg, major_deg, max_deg)``. The unit directions from the centre
    are projected onto the plane tangent to their mean direction; the two principal
    spreads ``lambda`` of that projection are converted to the half-angle of a
    circumduction cone with the same spread, ``asin(sqrt(2 lambda))``. A circular
    cone of half-angle ``a`` therefore scores ``(a, a)``; a uniformly filled cap of
    half-angle ``a`` scores about ``a / sqrt(2)`` in both; a single flexion arc scores
    near zero in the minor direction. ``max_deg`` is the largest angle of any sample
    from the mean direction.

    The minor angle is the one that governs the fit: the centre is only as well
    determined as the motion's weakest direction.
    """
    offsets = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(centre, float)
    distances = np.linalg.norm(offsets, axis=1)
    keep = distances > 1e-9
    if keep.sum() < 3:
        return 0.0, 0.0, 0.0
    directions = offsets[keep] / distances[keep, None]
    mean = directions.mean(axis=0)
    if np.linalg.norm(mean) < 1e-9:
        return 90.0, 90.0, 180.0  # directions all round the centre
    mean = mean / np.linalg.norm(mean)
    helper = np.eye(3)[int(np.argmin(np.abs(mean)))]
    e1 = np.cross(mean, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mean, e1)
    tangent = np.column_stack([directions @ e1, directions @ e2])
    spreads = np.linalg.eigvalsh(np.cov(tangent, rowvar=False, bias=True))
    angles = np.degrees(np.arcsin(np.sqrt(np.clip(2.0 * spreads, 0.0, 1.0))))
    max_deg = float(np.degrees(np.arccos(np.clip(directions @ mean, -1.0, 1.0))).max())
    return float(angles[0]), float(angles[1]), max_deg


# ----------------------------------------------------------------------
# Sphere fit
# ----------------------------------------------------------------------


def fit_sphere_pivoting(
    points: np.ndarray,
    *,
    policy: RefusalPolicy = DEFAULT_POLICY,
    max_iterations: int = 100,
    tolerance_mm: float = 1e-9,
) -> HipCentreFit:
    """Hip centre from marker positions alone: one centre, a radius per marker.

    ``points`` is ``(N, 3)`` for one tracked point (for instance the marker array's
    origin) or ``(N, K, 3)`` for the ``K`` spheres of an array, sample by sample. Every
    point is fixed to the femur, so each lies on its own sphere about the same centre;
    fitting them jointly uses all the data without first solving for poses.

    Not :func:`tka_planner.core.geometry.fit_sphere`, which fits one surface and
    returns no uncertainty: here the centre is shared, the refinement is damped
    (Levenberg-Marquardt) because small caps make the centre and radius nearly
    confounded, and the predicted uncertainty is part of the answer.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 2:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"expected (N, 3) or (N, K, 3) points, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("marker points contain NaN or infinite coordinates")
    n_samples, n_markers, _ = points.shape
    if n_samples < 5:
        raise HipCentreFitError([f"{n_samples} samples; a sphere needs at least 5"])

    # Algebraic initialisation, centred and scaled so the conditioning reflects the
    # geometry rather than where the camera happens to be.
    centroids = points.mean(axis=1)
    if float(np.sqrt(np.mean(np.sum((centroids - centroids.mean(axis=0)) ** 2,
                                    axis=1)))) < 1e-6:
        raise HipCentreFitError(["the marker did not move"])
    mean = points.reshape(-1, 3).mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points - mean) ** 2, axis=2))))
    q = (points - mean) / scale
    flat = q.reshape(-1, 3)
    which = np.tile(np.arange(n_markers), n_samples)
    design = np.hstack([2.0 * flat, np.eye(n_markers)[which]])
    target = np.sum(flat**2, axis=1)
    solution, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3 + n_markers:
        raise HipCentreFitError([
            "the marker paths are circles about one axis, so no unique sphere passes "
            "through them (a single plane of motion): move the leg in two directions"])
    centre = mean + scale * solution[:3]
    radii = np.linalg.norm(points - centre, axis=2).mean(axis=0)

    centre, radii, converged, iterations, jacobian, residuals = _levenberg_marquardt(
        points, centre, radii, max_iterations, tolerance_mm)

    dof = residuals.size - (3 + n_markers)
    covariance, condition = _covariance(jacobian, residuals, dof, slice(0, 3))
    minor, major, widest = arc_coverage(centroids, centre)
    fit = HipCentreFit(
        centre_mm=centre,
        method=SPHERE_METHOD,
        n_samples=n_samples,
        rms_residual_mm=float(np.sqrt(np.mean(residuals**2))),
        residual_kind="radial",
        arc_minor_deg=minor,
        arc_major_deg=major,
        arc_max_deg=widest,
        centre_sd_mm=_worst_sd(covariance),
        condition_number=condition,
        converged=converged,
        iterations=iterations,
        radius_mm=float(np.linalg.norm(centroids - centre, axis=1).mean()),
        centre_covariance=covariance,
    )
    return policy.check(fit)


def _levenberg_marquardt(points, centre, radii, max_iterations, tolerance_mm):
    """Minimise the orthogonal distance ``|p_ik - c| - r_k`` over ``(c, r)``."""
    n_markers = points.shape[1]
    which = np.tile(np.arange(n_markers), points.shape[0])
    radius_block = -np.eye(n_markers)[which]

    def evaluate(c, r):
        offsets = (points - c).reshape(-1, 3)
        distances = np.linalg.norm(offsets, axis=1)
        return offsets, distances, distances - r[which]

    offsets, distances, residuals = evaluate(centre, radii)
    cost = float(residuals @ residuals)
    damping = 1e-3
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if np.any(distances < 1e-9):
            break
        jacobian = np.hstack([-offsets / distances[:, None], radius_block])
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ residuals
        while True:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal)),
                                   -gradient)
            trial_centre, trial_radii = centre + step[:3], radii + step[3:]
            t_offsets, t_distances, t_residuals = evaluate(trial_centre, trial_radii)
            trial_cost = float(t_residuals @ t_residuals)
            if trial_cost <= cost or damping > 1e12:
                break
            damping *= 10.0
        if trial_cost <= cost:
            centre, radii = trial_centre, trial_radii
            offsets, distances, residuals, cost = (t_offsets, t_distances,
                                                   t_residuals, trial_cost)
            damping = max(damping / 10.0, 1e-12)
        small = float(np.linalg.norm(step)) < tolerance_mm * (1.0 + float(
            np.linalg.norm(centre)))
        if small or cost == 0.0:
            converged = True
            break
    jacobian = np.hstack([-offsets / np.maximum(distances, 1e-12)[:, None],
                          radius_block])
    return centre, radii, converged, iterations, jacobian, residuals


# ----------------------------------------------------------------------
# Pivot (centre of rotation from full poses)
# ----------------------------------------------------------------------


def fit_pivot(
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    policy: RefusalPolicy = DEFAULT_POLICY,
) -> HipCentreFit:
    """Hip centre from full marker poses: the femur-fixed point that does not move.

    ``rotations`` is ``(N, 3, 3)`` and ``translations`` ``(N, 3)``, each pose mapping
    marker coordinates into the tracker frame. Solves ``R_i p + t_i = c`` for the point
    ``p`` in the marker frame and the centre ``c`` in the tracker frame, in one linear
    least-squares step -- the same formulation as pivot calibration of a tracked
    probe, with the femur as the probe and the hip as its tip.

    It is degenerate only when every rotation shares one axis (a single plane of
    flexion), because any point along that axis is then stationary too.
    """
    rotations = np.asarray(rotations, dtype=float)
    translations = np.asarray(translations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"expected (N, 3, 3) rotations, got {rotations.shape}")
    if translations.shape != (len(rotations), 3):
        raise ValueError(f"expected ({len(rotations)}, 3) translations, "
                         f"got {translations.shape}")
    if not (np.all(np.isfinite(rotations)) and np.all(np.isfinite(translations))):
        raise ValueError("poses contain NaN or infinite values")
    n_samples = len(rotations)
    if n_samples < 3:
        raise HipCentreFitError([f"{n_samples} poses; a pivot needs at least 3"])

    design = np.zeros((3 * n_samples, 6))
    design[:, :3] = rotations.reshape(-1, 3)
    design[:, 3:] = -np.tile(np.eye(3), (n_samples, 1))
    target = -translations.reshape(-1)
    solution, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < 6:
        raise HipCentreFitError([
            "every pose rotates about one axis, so the centre is undetermined along "
            "it: move the leg in two directions, not one plane"])
    pivot, centre = solution[:3], solution[3:]
    residual_vectors = (np.einsum("nij,j->ni", rotations, pivot) + translations
                        - centre)
    covariance, condition = _covariance(design, residual_vectors.reshape(-1),
                                        3 * n_samples - 6, slice(3, 6))
    minor, major, widest = arc_coverage(translations, centre)
    fit = HipCentreFit(
        centre_mm=centre,
        method=PIVOT_METHOD,
        n_samples=n_samples,
        rms_residual_mm=float(np.sqrt(np.mean(np.sum(residual_vectors**2, axis=1)))),
        residual_kind="pivot",
        arc_minor_deg=minor,
        arc_major_deg=major,
        arc_max_deg=widest,
        centre_sd_mm=_worst_sd(covariance),
        condition_number=condition,
        converged=True,
        iterations=1,
        radius_mm=float(np.linalg.norm(pivot)),
        pivot_in_marker_mm=pivot,
        centre_covariance=covariance,
    )
    return policy.check(fit)


def poses_from_marker_points(
    points: np.ndarray, geometry: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid poses of a marker array from its tracked sphere positions (Kabsch).

    ``points`` is ``(N, K, 3)``; ``geometry`` the ``(K, 3)`` sphere positions in the
    marker's own frame, or ``None`` to take the first sample as the reference. Returns
    ``(rotations, translations, fre_mm)``, the last being each sample's RMS fiducial
    registration error. Trackers usually report poses directly; this exists so the
    pivot can be run on raw trajectories, and so the synthetic harness derives pose
    noise from marker noise the way a tracker does.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 3:
        raise ValueError(f"expected (N, K>=3, 3) marker points, got {points.shape}")
    geometry = points[0] if geometry is None else np.asarray(geometry, dtype=float)
    reference = geometry - geometry.mean(axis=0)
    centroids = points.mean(axis=1)
    moved = points - centroids[:, None, :]
    cross = np.einsum("nki,kj->nij", moved, reference)
    u, _, vt = np.linalg.svd(cross)
    sign = np.sign(np.linalg.det(u @ vt))
    fix = np.ones((len(points), 3))
    fix[:, 2] = sign
    rotations = np.einsum("nij,nj,njk->nik", u, fix, vt)
    translations = centroids - np.einsum("nij,j->ni", rotations, geometry.mean(axis=0))
    predicted = np.einsum("nij,kj->nki", rotations, geometry) + translations[:, None, :]
    fre = np.sqrt(np.mean(np.sum((predicted - points) ** 2, axis=2), axis=1))
    return rotations, translations, fre


# ----------------------------------------------------------------------
# Shared statistics
# ----------------------------------------------------------------------


def _covariance(jacobian, residuals, dof, block):
    """Linearised covariance of the parameters in ``block``, and the condition number.

    ``s^2 (J^T J)^-1`` with ``s^2`` the residual variance per degree of freedom. The
    condition number is that of ``J`` itself, the ratio of its extreme singular values.
    """
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if dof <= 0 or not np.isfinite(condition):
        return np.full((3, 3), np.inf), condition
    variance = float(residuals @ residuals) / dof
    inverse = np.linalg.pinv(jacobian.T @ jacobian)
    return variance * inverse[block, block], condition


def _worst_sd(covariance: np.ndarray) -> float:
    if not np.all(np.isfinite(covariance)):
        return float("inf")
    return float(np.sqrt(max(np.linalg.eigvalsh(covariance)[-1], 0.0)))


# ----------------------------------------------------------------------
# Into the plan
# ----------------------------------------------------------------------


def _check_rigid(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"registration must be a 4x4 matrix, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("registration contains NaN or infinite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("registration's last row must be [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError(
            "registration is not rigid (its 3x3 block is not a rotation); a scaled or "
            "sheared transform would silently change the femur length")
    if np.linalg.det(rotation) < 0:
        raise ValueError("registration is a reflection; it would mirror the limb")
    return transform


def head_centre_landmark(
    fit: HipCentreFit,
    tracker_to_lps: np.ndarray,
    *,
    registration: str = "",
    registration_rms_mm: float | None = None,
    policy: RefusalPolicy = DEFAULT_POLICY,
) -> Landmark:
    """The fitted centre as the ``femur.head_centre`` landmark, in the CT's LPS frame.

    ``tracker_to_lps`` is the 4x4 rigid transform from the tracker frame to the CT
    (LPS) frame -- the femur registration done by the navigation system. It is taken,
    never estimated here, and is checked to be rigid: a reflection would mirror the
    limb and a scale would change the femur length, both without any visible symptom.

    The landmark is ``PRESENT`` because it is a measurement on this patient, not an
    estimate, so metrics built on it are ``measured``. What sets it apart from a head
    centre fitted to a CT is recorded where the report can show it: ``origin`` names
    the method, ``metadata["source"]`` is ``"functional, camera"``, and the residual,
    arc coverage, predicted uncertainty and registration travel with it. The predicted
    uncertainty is the fit's alone, in the tracker frame; registration error adds to it
    and is recorded separately when supplied.

    The fit is re-checked against ``policy``, so a centre computed under the permissive
    policy for study cannot enter a plan by accident.
    """
    policy.check(fit)
    transform = _check_rigid(tracker_to_lps)
    position = transform[:3, :3] @ fit.centre_mm + transform[:3, 3]
    metadata = {
        "source": FUNCTIONAL_SOURCE,
        "method": fit.method,
        "rms_residual_mm": round(fit.rms_residual_mm, 4),
        "residual_kind": fit.residual_kind,
        "n_samples": fit.n_samples,
        "arc_minor_deg": round(fit.arc_minor_deg, 2),
        "arc_major_deg": round(fit.arc_major_deg, 2),
        "centre_sd_mm": round(fit.centre_sd_mm, 4),
        "tracker_centre_mm": [round(float(v), 4) for v in fit.centre_mm],
        "registration": registration or "unnamed",
    }
    if registration_rms_mm is not None:
        metadata["registration_rms_mm"] = round(float(registration_rms_mm), 4)
    return Landmark(
        id="femur.head_centre",
        position_mm=position,
        status=LandmarkStatus.PRESENT,
        origin=f"functional_pivot:{fit.method}",
        metadata=metadata,
    )


def functional_head_centre_set(
    fit: HipCentreFit,
    tracker_to_lps: np.ndarray,
    *,
    case_id: str,
    side,
    registration: str = "",
    registration_rms_mm: float | None = None,
    policy: RefusalPolicy = DEFAULT_POLICY,
) -> LandmarkSet:
    """A one-landmark LPS set, ready for
    :func:`~tka_planner.core.landmarks.overlay_landmarks` onto a knee's landmarks,
    where it replaces the ``OUT_OF_SCAN`` head centre of a knee-only scan."""
    landmark = head_centre_landmark(fit, tracker_to_lps, registration=registration,
                                    registration_rms_mm=registration_rms_mm,
                                    policy=policy)
    return LandmarkSet(case_id, side, "LPS", [landmark],
                       source={"kind": "functional_hip_centre",
                               "fit": fit.to_dict(),
                               "registration": registration or "unnamed"})
