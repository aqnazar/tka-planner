"""Geometric primitives for anatomical measurement.

Every routine here is deliberately free of anatomical meaning -- it fits shapes to
points and nothing more -- so that :mod:`tka_planner.core.frames` and
:mod:`tka_planner.core.metrics` can be tested against synthetic geometry whose answer
is known by construction.

Two conventions run throughout:

**Fits report their own residual.** A sphere fitted to a femoral head that is half
outside the scan will still return a centre; the RMS is what reveals it is meaningless.
Callers are expected to check. This is what lets scan-coverage detection be a
measurement rather than an assertion by the operator.

**Directions are canonically signed.** Principal axes and fitted lines are defined only
up to sign -- the covariance of a point cloud cannot distinguish an axis from its
negation. The legacy pipeline hit this as a 180-degree ambiguity in its TEA detection
and patched it by folding angles into a half-open range. Here every direction-returning
function applies :func:`canonical_direction`, so results are reproducible run to run,
and anatomical orientation is imposed afterwards by code that actually knows which way
is proximal.

All lengths are millimetres; all angles are radians unless a name says ``_deg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SphereFit",
    "PlaneFit",
    "LineFit",
    "canonical_direction",
    "unit",
    "fit_sphere",
    "fit_sphere_robust",
    "fit_plane",
    "fit_line",
    "principal_axes",
    "project_to_plane",
    "project_out",
    "orient_towards",
    "angle_between",
    "signed_angle_in_plane",
    "extremal_point",
    "bounding_box",
    "regional_mask",
    "cross_section_centroid",
]


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SphereFit:
    """A fitted sphere. ``rms_mm`` is the residual on the *radial* distance.

    ``surface_spread`` measures how much of the sphere the supporting points actually
    cover, from 0 (a tiny patch) to 1 (the whole sphere); a hemisphere scores about 0.5.
    It matters because a small, nearly flat patch of *any* surface fits some sphere
    beautifully. Low residual alone therefore proves nothing -- coverage is what
    separates a genuine ball from an incidentally curved region.
    """

    centre: np.ndarray
    radius_mm: float
    rms_mm: float
    n_points: int
    converged: bool = True
    surface_spread: float = 0.0
    direction_isotropy: float = 0.0
    n_inliers: int | None = None

    @property
    def inlier_fraction(self) -> float:
        if self.n_inliers is None or self.n_points == 0:
            return 1.0
        return self.n_inliers / self.n_points


@dataclass(frozen=True)
class PlaneFit:
    """A fitted plane. ``rms_mm`` is the residual perpendicular to it."""

    point: np.ndarray
    normal: np.ndarray
    rms_mm: float
    n_points: int


@dataclass(frozen=True)
class LineFit:
    """A fitted line. ``rms_mm`` is the residual perpendicular to it.

    ``extent_mm`` is the spread of the points along the line, which for a diaphyseal
    axis is the lever arm available to the fit. A short extent means an angularly
    uncertain axis even when the RMS looks excellent, so it is reported alongside.
    """

    point: np.ndarray
    direction: np.ndarray
    rms_mm: float
    n_points: int
    extent_mm: float


# ----------------------------------------------------------------------
# Direction handling
# ----------------------------------------------------------------------


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalise, raising on degenerate input rather than returning NaNs."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalise a zero-length vector.")
    return vector / norm


def canonical_direction(vector: np.ndarray) -> np.ndarray:
    """Resolve the sign ambiguity of an undirected axis, reproducibly.

    An axis fitted to a point cloud is a line, not an arrow: ``d`` and ``-d`` describe
    it equally well, and which one a solver returns can depend on input ordering or
    library version. That is a reproducibility hazard for a pipeline whose output is
    meant to be hash-stable.

    The convention chosen is to make the largest-magnitude component positive, with
    ties broken toward the lowest axis index. It carries no anatomical meaning
    whatsoever -- callers that need "proximal" or "anterior" must orient the result
    themselves against a known reference.
    """
    vector = unit(vector)
    dominant = int(np.argmax(np.abs(vector)))
    return -vector if vector[dominant] < 0 else vector


def orient_towards(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip ``direction`` so it points along ``reference``.

    The counterpart to :func:`canonical_direction`: this is how anatomical meaning gets
    imposed on a fitted axis, e.g. orienting a shaft axis proximally using the vector
    from the joint centre to the far end of the segmentation.
    """
    direction = unit(direction)
    return -direction if float(np.dot(direction, reference)) < 0 else direction


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def fit_sphere(
    points: np.ndarray,
    *,
    refine: bool = True,
    max_iterations: int = 50,
    tolerance_mm: float = 1e-9,
) -> SphereFit:
    """Fit a sphere to points.

    Starts from the algebraic (Kasa) fit, which is a single linear solve but biased
    when the points cover only part of the sphere -- exactly the femoral-head case,
    where the articular surface is a partial cap. The Gauss-Newton refinement that
    follows minimises the true geometric residual and removes that bias.

    The returned ``rms_mm`` is the diagnostic that matters. A femoral head genuinely
    present in the scan fits to well under a millimetre; anything else is not a
    femoral head.
    """
    points = _as_points(points, minimum=4, what="sphere fit")

    # Algebraic fit: |p|^2 = 2 p.c - (|c|^2 - r^2), linear in (c, f).
    design = np.hstack([2.0 * points, -np.ones((len(points), 1))])
    target = np.sum(points**2, axis=1)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)

    # Rank deficiency is exactly the degenerate case. If the points lie on a plane
    # ax + by + cz = d then that same relation holds between the design matrix's
    # columns, dropping the rank below four. Infinitely many spheres pass through a
    # circle, so least-squares returns an arbitrary one -- positive radius and all --
    # and only the rank reveals that the answer is meaningless.
    if rank < 4:
        raise ValueError(
            "Sphere fit is underdetermined: the points are coplanar or collinear, "
            "so no unique sphere passes through them."
        )

    centre = solution[:3]
    squared_radius = float(np.dot(centre, centre) - solution[3])
    if squared_radius <= 0:
        raise ValueError("Sphere fit produced a non-positive radius.")
    radius = float(np.sqrt(squared_radius))
    converged = True

    if refine:
        centre, radius, converged = _refine_sphere(
            points, centre, radius, max_iterations, tolerance_mm
        )

    residuals = np.linalg.norm(points - centre, axis=1) - radius
    return SphereFit(
        centre=centre,
        radius_mm=radius,
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        n_points=len(points),
        converged=converged,
        surface_spread=_surface_spread(points, centre),
        direction_isotropy=_direction_isotropy(points, centre),
    )


def fit_sphere_robust(
    points: np.ndarray,
    *,
    keep_fraction: float = 0.5,
    trim_sigma: float = 2.5,
    min_band_mm: float = 0.5,
    max_rounds: int = 12,
) -> SphereFit:
    """Fit a sphere to the dominant spherical structure, ignoring what surrounds it.

    A plain least-squares fit uses every point it is given, so adjacent anatomy drags
    the answer away from the target. Selecting the proximal end of a femur to find the
    head inevitably also catches neck, trochanter and shaft, and their pull is what
    turns a real head into a failed detection.

    The method is least trimmed squares. Each round keeps the ``keep_fraction`` of
    points with the smallest residuals and refits, so the sphere is pulled toward
    whichever structure is genuinely spherical. Trimming by *rank* rather than by a
    residual threshold is what makes this work: under heavy contamination the initial
    residual scale is large, so any threshold derived from it keeps everything and the
    iteration never tightens. A fixed quantile makes progress regardless of scale.

    Once the trimmed set stabilises, the band is widened to ``trim_sigma`` standard
    deviations to readmit points that genuinely belong to the sphere but were cut by the
    aggressive quantile, giving an honest final residual and inlier count.

    Deliberately not RANSAC: no random sampling, so results are bit-reproducible, which
    plan hashing requires.
    """
    points = _as_points(points, minimum=4, what="robust sphere fit")
    if not 0.1 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in [0.1, 1.0].")

    fit = fit_sphere(points, refine=True)
    keep_count = max(int(round(keep_fraction * len(points))), 10)
    previous: np.ndarray | None = None

    for _ in range(max_rounds):
        residuals = np.abs(np.linalg.norm(points - fit.centre, axis=1) - fit.radius_mm)
        # Stable sort so ties break by index and repeated runs agree exactly.
        order = np.argsort(residuals, kind="stable")[:keep_count]
        selected = np.sort(order)

        if previous is not None and np.array_equal(selected, previous):
            break
        previous = selected

        try:
            fit = fit_sphere(points[selected], refine=True)
        except ValueError:
            break  # the trimmed set went degenerate; keep the last good fit

    # Readmit everything consistent with the converged sphere.
    residuals = np.abs(np.linalg.norm(points - fit.centre, axis=1) - fit.radius_mm)
    band = max(trim_sigma * fit.rms_mm, min_band_mm)
    inliers = residuals <= band

    if inliers.sum() >= 4:
        try:
            fit = fit_sphere(points[inliers], refine=True)
        except ValueError:
            pass

    return SphereFit(
        centre=fit.centre,
        radius_mm=fit.radius_mm,
        rms_mm=fit.rms_mm,
        n_points=len(points),
        converged=fit.converged,
        surface_spread=fit.surface_spread,
        direction_isotropy=fit.direction_isotropy,
        n_inliers=int(inliers.sum()),
    )


def _surface_spread(points: np.ndarray, centre: np.ndarray) -> float:
    """How completely the points wrap the fitted centre, on a 0-to-1 scale.

    Computed as ``1 - |mean of unit directions from the centre|``. Directions spread
    over a whole sphere cancel to nearly zero, giving a spread near 1; directions
    confined to a small patch all point the same way, giving a spread near 0. A
    hemisphere lands around 0.5.

    Note this measures wrap-around, not sphericity: a band cut from a cylinder wraps
    completely and scores very high. Use :func:`_direction_isotropy` to tell a ball
    from a tube.
    """
    directions = _unit_directions(points, centre)
    if directions is None:
        return 0.0
    return float(1.0 - np.linalg.norm(directions.mean(axis=0)))


def _direction_isotropy(points: np.ndarray, centre: np.ndarray) -> float:
    """Whether the points surround the centre in three dimensions or only two.

    This is what separates a genuine ball from a band cut out of a tube, and the
    distinction is not academic. A short axial slice of a thick cylinder fits a sphere
    with a sub-millimetre residual at a plausible radius, so residual, radius and
    wrap-around can all look convincing at once. Measured on synthetic shafts, a
    headless femur 360 mm long with an 18 mm shaft radius passed every one of those
    checks.

    The giveaway is that a cylinder band's unit directions lie close to a single great
    circle -- they fan out around the tube's axis but barely vary along it -- whereas a
    spherical cap's directions cover a genuine two-dimensional patch.

    Returns the ratio of the smallest to the largest standard deviation of the unit
    direction vectors: near 0 for a ring or band, approaching 1 for a full sphere. A
    hemispherical cap sits comfortably in between.
    """
    directions = _unit_directions(points, centre)
    if directions is None or len(directions) < 3:
        return 0.0

    spreads = np.sqrt(
        np.linalg.svd(directions - directions.mean(axis=0), compute_uv=False) ** 2
        / max(len(directions) - 1, 1)
    )
    if spreads[0] < 1e-12:
        return 0.0
    return float(spreads[2] / spreads[0])


def _unit_directions(points: np.ndarray, centre: np.ndarray) -> np.ndarray | None:
    offsets = points - centre
    distances = np.linalg.norm(offsets, axis=1)
    valid = distances > 1e-12
    if not np.any(valid):
        return None
    return offsets[valid] / distances[valid, None]


def _refine_sphere(
    points: np.ndarray,
    centre: np.ndarray,
    radius: float,
    max_iterations: int,
    tolerance_mm: float,
) -> tuple[np.ndarray, float, bool]:
    """Gauss-Newton on the geometric residual ``|p - c| - r``."""
    for _ in range(max_iterations):
        offsets = points - centre
        distances = np.linalg.norm(offsets, axis=1)
        if np.any(distances < 1e-12):
            return centre, radius, False  # a point sits on the centre; give up

        directions = offsets / distances[:, None]
        jacobian = np.hstack([-directions, -np.ones((len(points), 1))])
        residuals = distances - radius

        step, *_ = np.linalg.lstsq(jacobian, -residuals, rcond=None)
        centre = centre + step[:3]
        radius = float(radius + step[3])

        if float(np.linalg.norm(step)) < tolerance_mm:
            return centre, radius, True

    return centre, radius, False


def fit_plane(points: np.ndarray) -> PlaneFit:
    """Fit a plane by total least squares; the normal is the least-variance direction."""
    points = _as_points(points, minimum=3, what="plane fit")
    centroid = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - centroid, full_matrices=False)

    normal = canonical_direction(right[2])
    residuals = (points - centroid) @ normal
    return PlaneFit(
        point=centroid,
        normal=normal,
        rms_mm=float(np.sqrt(np.mean(residuals**2))),
        n_points=len(points),
    )


def fit_line(points: np.ndarray) -> LineFit:
    """Fit a line by total least squares; the direction is the greatest-variance one.

    Used for diaphyseal (canal) axes. Check ``extent_mm`` as well as ``rms_mm``: a
    150 mm femur segment offers a far shorter lever arm than a 240 mm one, and the
    angular uncertainty of the axis scales inversely with it even though both may fit
    tightly.
    """
    points = _as_points(points, minimum=2, what="line fit")
    centroid = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - centroid, full_matrices=False)

    direction = canonical_direction(right[0])
    offsets = points - centroid
    along = offsets @ direction
    perpendicular = offsets - np.outer(along, direction)

    return LineFit(
        point=centroid,
        direction=direction,
        rms_mm=float(np.sqrt(np.mean(np.sum(perpendicular**2, axis=1)))),
        n_points=len(points),
        extent_mm=float(along.max() - along.min()),
    )


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal axes and variances, ordered from greatest spread to least.

    Returns ``(axes, variances)`` with ``axes[i]`` the unit direction of the i-th
    component. Each axis is canonically signed, so repeated runs agree.
    """
    points = _as_points(points, minimum=3, what="principal axes")
    centred = points - points.mean(axis=0)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)

    axes = np.array([canonical_direction(row) for row in right])
    variances = singular**2 / max(len(points) - 1, 1)
    return axes, variances


# ----------------------------------------------------------------------
# Projection and angles
# ----------------------------------------------------------------------


def project_to_plane(
    points: np.ndarray, origin: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    """Orthogonally project points onto a plane, staying in 3-D coordinates."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    normal = unit(normal)
    offsets = points - np.asarray(origin, dtype=float)
    return points - np.outer(offsets @ normal, normal)


def project_out(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Remove the component of ``vector`` along ``axis``, returning a unit vector.

    This is how a rotational reference such as the surgical epicondylar axis is brought
    into a frame's transverse plane. The component removed is not noise -- it is the
    axis's out-of-plane tilt -- so metrics that care about pick quality should measure
    it before discarding it.
    """
    axis = unit(axis)
    vector = np.asarray(vector, dtype=float)
    residual = vector - np.dot(vector, axis) * axis
    if float(np.linalg.norm(residual)) < 1e-9:
        raise ValueError(
            "Vector is parallel to the axis; it has no component in the plane."
        )
    return unit(residual)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in radians, numerically safe at 0 and pi."""
    cosine = float(np.dot(unit(a), unit(b)))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def signed_angle_in_plane(
    a: np.ndarray, b: np.ndarray, normal: np.ndarray
) -> float:
    """Signed angle from ``a`` to ``b`` about ``normal``, in radians.

    Positive is counter-clockwise looking down ``normal`` (right-hand rule). Both
    vectors are projected into the plane first.

    Anatomical sign conventions are defined by the caller choosing which normal to pass,
    never by post-hoc negation -- that is how side-dependent sign errors creep in.
    """
    normal = unit(normal)
    a_flat = project_out(a, normal)
    b_flat = project_out(b, normal)
    return float(
        np.arctan2(float(np.dot(np.cross(a_flat, b_flat), normal)),
                   float(np.dot(a_flat, b_flat)))
    )


# ----------------------------------------------------------------------
# Point selection
# ----------------------------------------------------------------------


def extremal_point(
    points: np.ndarray,
    direction: np.ndarray,
    *,
    n_average: int = 10,
) -> np.ndarray:
    """Average the ``n_average`` points furthest along ``direction``.

    A single furthest vertex is a poor landmark: on a segmented mesh it is as likely to
    be a marching-cubes spike or a stray artefact as real anatomy. Averaging a small
    neighbourhood is far more stable, and the legacy pipeline already did this for
    condyle contact with ``CONTACT_AVG_N = 10``.

    ``n_average`` is a recorded plan parameter, not a hidden constant, because it
    materially affects the resulting coordinate.
    """
    points = _as_points(points, minimum=1, what="extremal point")
    if n_average < 1:
        raise ValueError("n_average must be at least 1.")

    projections = points @ unit(direction)
    count = min(n_average, len(points))
    # Stable sort so ties resolve by index and the result is reproducible.
    order = np.argsort(-projections, kind="stable")[:count]
    return points[order].mean(axis=0)


def bounding_box(points: np.ndarray) -> dict:
    """Axis-aligned bounds of a point cloud, in whatever frame the points are given."""
    points = _as_points(points, minimum=1, what="bounding box")
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "min": lo,
        "max": hi,
        "extent": hi - lo,
        "centre": (lo + hi) / 2.0,
    }


def regional_mask(
    points: np.ndarray,
    direction: np.ndarray,
    *,
    fraction: float = 0.25,
    end: str = "low",
) -> np.ndarray:
    """Boolean mask selecting a fraction of the cloud from one end along ``direction``.

    Reproduces the legacy ``get_regional_bbox`` selection so its measurements can be
    replayed for parity, but expressed against an arbitrary direction rather than a
    world axis, so it works in an anatomical frame.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1].")
    if end not in ("low", "high"):
        raise ValueError("end must be 'low' or 'high'.")

    points = _as_points(points, minimum=1, what="regional mask")
    projections = points @ unit(direction)
    low, high = projections.min(), projections.max()
    span = high - low
    if span <= 0:
        return np.ones(len(points), dtype=bool)

    if end == "low":
        return projections <= low + fraction * span
    return projections >= high - fraction * span


def cross_section_centroid(
    points: np.ndarray,
    axis: np.ndarray,
    level: float,
    *,
    slab_thickness_mm: float = 2.0,
    origin: np.ndarray | None = None,
) -> np.ndarray:
    """Centroid of a slab of points cut perpendicular to ``axis``.

    Used to trace the medullary canal: a series of these at different levels defines the
    diaphyseal axis. The slab has thickness because an infinitesimally thin section of a
    triangulated surface would contain almost no vertices.

    ``level`` is measured along ``axis`` from ``origin`` (default: the cloud centroid).
    Note this returns the centroid of the *surface* vertices in the slab -- for a
    roughly circular diaphyseal cortex that approximates the canal centre well.
    """
    points = _as_points(points, minimum=1, what="cross-section centroid")
    axis = unit(axis)
    base = points.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)

    projections = (points - base) @ axis
    half = slab_thickness_mm / 2.0
    in_slab = np.abs(projections - level) <= half

    if not np.any(in_slab):
        raise ValueError(
            f"No points within {slab_thickness_mm} mm of level {level:.1f} mm. "
            f"Available range is {projections.min():.1f} to {projections.max():.1f} mm."
        )
    return points[in_slab].mean(axis=0)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _as_points(points: np.ndarray, *, minimum: int, what: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{what}: expected an (N, 3) array, got {points.shape}.")
    if len(points) < minimum:
        raise ValueError(
            f"{what}: needs at least {minimum} points, got {len(points)}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{what}: input contains NaN or infinite coordinates.")
    return points
