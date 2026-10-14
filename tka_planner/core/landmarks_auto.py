"""Automatic landmark estimation from bone surfaces.

Picking landmarks by hand is the slow step in this pipeline: roughly twenty points per
knee, and the validation cohort needs them twice over for an observer study. This module
produces a complete first pass automatically, so a plan can be generated from nothing
but two STL files.

**Everything it produces is marked** :attr:`~tka_planner.core.landmarks.LandmarkStatus.ESTIMATED`,
never ``PRESENT``. That distinction is the point. An automatic estimate is a starting
position for a human to correct, not a measurement, and a report built on estimates says
so on every line. The same machinery that flags an assumed mechanical axis flags these.

The estimators are geometric and deliberately simple. Each runs in its **bone's own
axes** (:class:`BoneAxes`), found from the surface before any landmark exists: the
diaphyseal axis traced through cross-section centroids, the mediolateral axis from the
posterior condyles, and an origin at the surface's centroid. The scanner's LPS frame
contributes only *signs* -- which end is superior, which side patient-left -- which no
plausibly oriented scan can get wrong. An earlier version searched along the scanner's
axes directly, trusting that the segmentations arrive in LPS. They do, but a leg lies in
the scanner at whatever angle the patient found comfortable, and metamorphic testing
(paper experiment E2) showed what that cost: the tibial AP reference turned 17 degrees
with a 10 degree turn of the leg, a distal condyle jumped 30 mm with a 30 degree tilt,
and moving the knee 100 mm without turning it moved a plateau low point 12.5 mm, because
the articular grid was anchored at the scanner origin. In the bone's own axes a rigid
motion of the scan moves every landmark with the bone, exactly.

A scan oriented too far from LPS for those signs to be trusted is refused upstream by
:func:`tka_planner.core.qc.check_scan_orientation`, which reads the same axes.

Reliability varies by landmark and is stated per estimator below. The condylar and
plateau extremes are dependable, being genuine extrema of the surface. The epicondyles
are fair. The tibial rotational references are the weakest -- the patellar tendon
insertion has no geometric signature a bounding box can find -- and are the ones a human
should correct first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import cross_section_centroid, extremal_point, regional_mask, unit
from .landmarks import Landmark, LandmarkSet, LandmarkStatus
from .meshio import Mesh
from .provenance import POPULATION_FEMORAL_AMA
from .sides import LPS_PATIENT_LEFT, LPS_POSTERIOR, LPS_SUPERIOR, Side

__all__ = [
    "BoneAxes",
    "femoral_axes",
    "tibial_axes",
    "estimate_femoral_landmarks",
    "estimate_tibial_landmarks",
    "estimate_landmarks",
    "condylar_width_mm",
    "distal_window_mm",
]

ANTERIOR = -LPS_POSTERIOR
CONTACT_AVERAGE_N = 10  # matches the legacy CONTACT_AVG_N
PLATEAU_SLAB_MM = 22.0
ENVELOPE_CELL_MM = 2.0
# A face this close to horizontal can be joint surface; anything steeper is the side of
# the bone. cos 55 degrees.
ENVELOPE_MIN_COS = 0.57

# How much of a femur the distal searches look at, in condylar widths. The searches are
# fractions of a length -- the condyles in the bottom fifth, the epicondyles between 15
# and 35 per cent -- and were tuned on knee-only segments, which run 147 to 248 mm here,
# or 1.7 to 2.6 condylar widths. On a full-length femur the same fractions of 450 mm
# reach the shaft: the reference femur's lateral epicondyle was placed 157 mm above the
# joint. So the length they are fractions of is capped at this many condylar widths,
# which leaves every knee-only segment searched exactly as before and searches a longer
# femur as if it had been scanned knee-only. It scales with the bone, not the scan.
DISTAL_WINDOW_CONDYLAR_WIDTHS = 2.75
# The condylar width is the widest caliper of the bottom of the bone, this thick.
CONDYLAR_SLAB_MM = 25.0

# The bone's own axes are refined until a round turns them by less than this. The
# refinement reads discrete point sets, so it ends in a cycle rather than a fixed point,
# of 0.001 to 0.04 degrees on the cohort; it gets there in three to five rounds. Where it
# stops does not depend on how the bone lies, since every round starts from the bone.
AXES_TOLERANCE_DEG = 0.05
AXES_MAX_ROUNDS = 10
# The femur's working axis is its diaphyseal axis turned medially by the population
# mechanical-anatomical angle -- the axis the femoral frame itself falls back to on a
# knee-only scan -- so "distal" here means distal along the mechanical axis, as the
# distal condyles are conventionally read. On this cohort it lies 4.6 to 6.6 degrees from
# the scanner's superior axis, against 7.6 to 11.2 for the diaphyseal axis alone.
FEMORAL_WORKING_AMA_DEG = POPULATION_FEMORAL_AMA.value
# The femoral condylar block, for the anteroposterior reach the orientation gate reads.
REACH_BLOCK_MM = 30.0
# The notch roof is read on a sagittal strip this far either side of the midline, and is
# the centre of the part of that strip within this height of its lowest.
NOTCH_STRIP_MM = 3.0
NOTCH_BAND_MM = 2.0
# The PCL footprint is searched this far either side of the midline between the spines.
PCL_STRIP_MM = 2.0


# ----------------------------------------------------------------------
# The bone's own axes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BoneAxes:
    """A bone's own right-handed axes, found from its surface alone.

    ``rotation`` holds the axes as rows, in scanner coordinates: patient-left, posterior,
    superior -- the LPS order, so a bone lying square in the scanner has the identity and
    the estimators read exactly as they were written for LPS. ``origin`` is the
    area-weighted centroid of the surface, a point fixed to the bone, so that anything
    gridded in these coordinates is gridded on the bone rather than on the scanner.
    ``shaft`` is the diaphyseal axis the canal sections are cut square to: the femur's
    superior axis leans from it by the working mechanical-anatomical angle, the tibia's
    is it. ``None`` on a bone too short to hold a shaft.

    The scanner fixes only signs: superior is the end nearer scanner +Z, patient-left the
    end nearer scanner +X, and posterior completes a right-handed set. Under a mirror
    (x flipped, side swapped) the axes therefore mirror too, and the local coordinates of
    a mirrored bone are the mirror of the original's: the estimators stay exact under it.
    """

    bone: str
    origin: np.ndarray
    rotation: np.ndarray
    method: str
    shaft: np.ndarray | None = None      # the diaphyseal axis, scanner coordinates
    diagnostics: dict = field(default_factory=dict)

    @property
    def patient_left(self) -> np.ndarray:
        return self.rotation[0]

    @property
    def posterior(self) -> np.ndarray:
        return self.rotation[1]

    @property
    def superior(self) -> np.ndarray:
        return self.rotation[2]

    def to_local(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.rotation.T

    def to_world(self, local) -> np.ndarray:
        return np.asarray(local, dtype=float) @ self.rotation + self.origin

    def local_mesh(self, mesh: Mesh) -> Mesh:
        return Mesh(vertices=self.to_local(mesh.vertices), faces=mesh.faces,
                    source_path=mesh.source_path, sha256=mesh.sha256,
                    metadata=dict(mesh.metadata))


def _surface_samples(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Triangle centroids and areas: the surface as an area-weighted cloud.

    Weighting by area rather than counting vertices keeps the principal axes and the
    origin from following the triangulation, which is denser where the surface is curved
    and changes under decimation or remeshing.
    """
    triangles = mesh.vertices[mesh.faces]
    centroids = triangles.mean(axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    if not float(areas.sum()) > 0.0:
        areas = np.ones(len(centroids))
    return centroids, areas


def _weighted_axes(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Principal axes of a weighted cloud, as rows, largest spread first."""
    centre = (points * weights[:, None]).sum(axis=0) / weights.sum()
    spread = points - centre
    covariance = (spread * weights[:, None]).T @ spread / weights.sum()
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, ::-1].T


def _signed(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    vector = unit(vector)
    return -vector if float(np.dot(vector, reference)) < 0.0 else vector


def _perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return unit(vector - float(np.dot(vector, axis)) * axis)


def _rotation(patient_left: np.ndarray, superior: np.ndarray) -> np.ndarray:
    """Rows (patient-left, posterior, superior), right-handed, from two directions."""
    superior = unit(superior)
    patient_left = _perpendicular(patient_left, superior)
    posterior = np.cross(superior, patient_left)
    return np.array([patient_left, posterior, superior])


def _long_axis_guess(centroids: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """The principal axis of the surface nearest the scanner's superior axis.

    Only a starting point for the diaphyseal axis. Choosing among the principal axes,
    rather than taking the largest, keeps a stubby block from starting along its width;
    on every bone of the cohort the two coincide.
    """
    axes = _weighted_axes(centroids, areas)
    pick = int(np.argmax(np.abs(axes @ LPS_SUPERIOR)))
    return _signed(axes[pick], LPS_SUPERIOR)


def _transverse_major_axis(points: np.ndarray, weights: np.ndarray,
                           superior: np.ndarray) -> np.ndarray:
    """The widest direction of a cloud seen along ``superior``, signed patient-left.

    The starting estimate for a mediolateral axis: the condylar block and the plateau are
    both wider than they are deep. Taken from the bone, so a convergence that starts here
    starts at the same place on the bone however the bone lies.
    """
    flat = points - np.outer(points @ superior, superior)
    axes = _weighted_axes(flat, weights)
    return _signed(_perpendicular(axes[0], superior), LPS_PATIENT_LEFT)


def _shaft_axis(points: np.ndarray, superior: np.ndarray,
                bone: str) -> np.ndarray | None:
    """The diaphyseal axis through two canal-section centroids, cut square to ``superior``.

    The same two sections :func:`_canal_centres` places as landmarks. ``None`` when the
    bone is too short to hold a shaft.
    """
    centres = {landmark.id: landmark.position_mm
               for landmark in _canal_centres(_cloud(points), bone, superior)}
    ends = (("femur.canal_centre_distal", "femur.canal_centre_proximal") if bone == "femur"
            else ("tibia.canal_centre_distal", "tibia.canal_centre_proximal"))
    if not all(end in centres for end in ends):
        return None
    span = centres[ends[1]] - centres[ends[0]]
    if float(np.linalg.norm(span)) < 1e-6:
        return None
    return _signed(span, superior)


def _converged_shaft(points: np.ndarray, areas: np.ndarray,
                     bone: str) -> tuple[np.ndarray, bool, int]:
    """The diaphyseal axis, with each round's sections cut square to the last round's axis.

    Starts from the principal axis of the surface nearest the scanner's superior axis.
    Returns ``(axis, found, rounds)``; ``found`` is false, and the axis the principal
    one, when the bone is too short to hold a shaft.
    """
    axis = _long_axis_guess(points, areas)
    rounds = 0
    for rounds in range(1, AXES_MAX_ROUNDS + 1):
        shaft = _shaft_axis(points, axis, bone)
        if shaft is None:
            return axis, False, rounds
        turned = _angle_deg(shaft, axis)
        axis = shaft
        if turned < AXES_TOLERANCE_DEG:
            break
    return axis, True, rounds


def _cloud(points: np.ndarray) -> Mesh:
    """Points as a mesh without triangles, for the helpers that read vertices only."""
    return Mesh(vertices=points, faces=np.zeros((0, 3), dtype=int))


def _turn(vector: np.ndarray, towards: np.ndarray, angle_deg: float) -> np.ndarray:
    """``vector`` turned by ``angle_deg`` in the plane it spans with ``towards``."""
    vector = unit(vector)
    across = _perpendicular(towards, vector)
    angle = np.radians(angle_deg)
    return unit(np.cos(angle) * vector + np.sin(angle) * across)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(unit(a), unit(b))), -1.0, 1.0))))


def _line_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two lines, ignoring which way either points."""
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(unit(a), unit(b)))),
                                              0.0, 1.0))))


def femoral_axes(mesh: Mesh, side: "str | Side") -> BoneAxes:
    """The femur's own axes: its working mechanical axis and its posterior condylar line.

    First the diaphyseal axis, traced through two canal-section centroids, each round's
    sections cut square to the last round's axis until it settles. Superior is that axis
    turned medially by :data:`FEMORAL_WORKING_AMA_DEG` in the plane holding the shaft and
    the mediolateral axis -- the axis the femoral frame falls back to on a knee-only scan.
    Patient-left is the posterior condylar line, found as the femoral estimator always
    found it, in the plane square to that axis. The turn needs the mediolateral axis and
    the mediolateral axis needs the turned axis, so the two are refined together until a
    round turns superior by less than :data:`AXES_TOLERANCE_DEG`.

    Everything is found on the distal window alone (:func:`_distal_part`), and every step
    starts from the bone rather than the scanner, which is what makes the result move
    rigidly with the bone. The search works on triangle centroids, a third as many points
    as an STL's vertices. The diagnostics carry what the orientation gate reads: how far
    the axes lie from the scanner's, and how far the condylar block reaches behind and in
    front of the shaft.
    """
    side = Side.parse(side)
    centroids, areas = _surface_samples(mesh)
    centroids, areas, window = _distal_part(centroids, areas)
    origin = (centroids * areas[:, None]).sum(axis=0) / areas.sum()
    points = centroids - origin

    shaft, found, shaft_rounds = _converged_shaft(points, areas, "femur")
    method = "bone_axes.femur.shaft_ama.v1" if found else "bone_axes.femur.principal.v1"
    superior = shaft
    patient_left = LPS_PATIENT_LEFT
    rounds = 0
    for rounds in range(1, AXES_MAX_ROUNDS + 1):
        rotation = _rotation(patient_left, superior)
        local = points @ rotation.T
        bottom = local[:, 2] <= float(local[:, 2].min()) + 0.20 * window
        start = _transverse_major_axis(local[bottom], areas[bottom],
                                       np.array([0.0, 0.0, 1.0]))
        lateral_local, _ = _converge_posterior_axis(local[bottom], start=start)
        patient_left = lateral_local @ rotation
        if not found:
            break
        updated = _turn(shaft, side.medial_direction(patient_left),
                        FEMORAL_WORKING_AMA_DEG)
        turned = _angle_deg(updated, superior)
        superior = updated
        if turned < AXES_TOLERANCE_DEG:
            break

    axes = BoneAxes("femur", origin, _rotation(patient_left, superior), method,
                    shaft=shaft if found else None)
    axes.diagnostics.update(_axes_diagnostics(axes, shaft_rounds + rounds))
    axes.diagnostics.update(_condylar_reach(axes.to_local(centroids),
                                            axes.rotation @ shaft))
    return axes


def _distal_part(centroids: np.ndarray, areas: np.ndarray):
    """The distal window of a femur's surface, and the window's length.

    The femur's axes are found on the part of it the distal searches read, so that a
    full-length femur gets the axes of its own knee-only scan: where the scan stops
    plays no part, as it plays none in the searches (see
    :data:`DISTAL_WINDOW_CONDYLAR_WIDTHS`). The window is measured along the surface's
    principal axis, which depends on the scan's length, so the two are refined together
    until the part kept stops changing. On a knee-only segment the window is the whole
    segment and everything is kept.
    """
    keep = np.ones(len(centroids), dtype=bool)
    window = 0.0
    for _ in range(AXES_MAX_ROUNDS):
        superior = _long_axis_guess(centroids[keep], areas[keep])
        local = centroids @ _rotation(LPS_PATIENT_LEFT, superior).T
        window = distal_window_mm(_cloud(local))
        updated = local[:, 2] <= float(local[:, 2].min()) + window
        if np.array_equal(updated, keep):
            break
        keep = updated
    return centroids[keep], areas[keep], window


def tibial_axes(mesh: Mesh) -> BoneAxes:
    """The tibia's own axes: its diaphyseal axis and its posterior condylar line.

    Superior is the diaphyseal axis through two canal-section centroids -- the axis the
    tibial frame uses -- each round's sections cut square to the last round's axis until
    it settles. Patient-left is the line touching the backs of the two tibial condyles,
    found on the top ``PLATEAU_SLAB_MM`` exactly as the femur's posterior condylar line
    is found on the condylar block, starting from the plateau's widest direction.

    The tibia's own posterior condyles rather than anything femoral: the tibial frame's
    AP axis is built from the landmarks found in these axes, and a femoral direction here
    would carry the femur's rotation into the tibia's rotational reference, so that
    tibiofemoral rotation measured between the two frames would partly measure itself.
    Nor the plateau's widest direction alone: its outline is asymmetric, and on one
    cohort knee that direction lay 18 degrees from both posterior condylar lines.
    """
    centroids, areas = _surface_samples(mesh)
    origin = (centroids * areas[:, None]).sum(axis=0) / areas.sum()
    points = centroids - origin

    superior, found, rounds = _converged_shaft(points, areas, "tibia")
    method = "bone_axes.tibia.shaft.v1" if found else "bone_axes.tibia.principal.v1"
    rotation = _rotation(LPS_PATIENT_LEFT, superior)
    local = points @ rotation.T
    top = local[:, 2] >= float(local[:, 2].max()) - PLATEAU_SLAB_MM
    start = _transverse_major_axis(local[top], areas[top], np.array([0.0, 0.0, 1.0]))
    lateral_local, _ = _converge_posterior_axis(local[top], start=start)
    axes = BoneAxes("tibia", origin, _rotation(lateral_local @ rotation, superior), method,
                    shaft=superior if found else None)
    axes.diagnostics.update(_axes_diagnostics(axes, rounds))
    return axes


def _axes_diagnostics(axes: BoneAxes, rounds: int) -> dict:
    """How far the bone's axes lie from the scanner's: what the orientation gate reads.

    The tilt is the long axis against scanner superior; the turn is the mediolateral
    axis against scanner left-right, about the bone's own long axis, so that a tilt
    does not count twice.
    """
    turn = 90.0
    if abs(float(np.dot(LPS_PATIENT_LEFT, axes.superior))) < 1.0 - 1e-9:
        turn = _line_angle_deg(axes.patient_left,
                               _perpendicular(LPS_PATIENT_LEFT, axes.superior))
    return {
        "rounds": rounds,
        "superior_tilt_deg": round(_angle_deg(axes.superior, LPS_SUPERIOR), 3),
        "mediolateral_turn_deg": round(turn, 3),
    }


def _condylar_reach(local: np.ndarray, shaft: np.ndarray) -> dict:
    """How far the condylar block reaches behind and in front of the shaft, in mm.

    ``local`` and ``shaft`` are in the femur's own axes. The posterior condyles stand
    well behind the line of the shaft and the trochlea only a little in front of it: 2.2
    to 4.0 times as far on this cohort. A femur facing backwards in the scan reverses
    that, which is the one misorientation the signs taken from the scanner cannot see,
    so the orientation gate reads it.
    """
    z = local[:, 2]
    block = local[z <= float(z.min()) + REACH_BLOCK_MM]
    centres = {landmark.id: landmark.position_mm
               for landmark in _canal_centres(_cloud(local), "femur", shaft)}
    if "femur.canal_centre_distal" not in centres or not len(block):
        return {}
    # The shaft's line, carried down to the middle of the block.
    centre = centres["femur.canal_centre_distal"]
    level = float(block[:, 2].mean())
    shaft = unit(shaft)
    on_line = centre + (level - centre[2]) / shaft[2] * shaft
    return {
        "posterior_reach_mm": round(float(block[:, 1].max() - on_line[1]), 2),
        "anterior_reach_mm": round(float(on_line[1] - block[:, 1].min()), 2),
    }


def _estimated(landmark_id: str, position, method: str, confidence: str) -> Landmark:
    return Landmark(
        id=landmark_id,
        position_mm=np.asarray(position, dtype=float),
        status=LandmarkStatus.ESTIMATED,
        origin=f"auto:{method}",
        metadata={"confidence": confidence},
    )


def _compartments(points: np.ndarray, side: Side, midline_x: float):
    """Split a point cloud into (medial, lateral) using the frame-free LPS ML axis.

    Compartment identity comes from :class:`~tka_planner.core.sides.Side`, not from a
    bounding-box half. The legacy pipeline made the latter choice and so labelled
    compartments correctly on one knee and backwards on the other.
    """
    origin = np.array([midline_x, 0.0, 0.0])
    medial_mask, lateral_mask = side.split_compartments(
        points, origin, LPS_PATIENT_LEFT
    )
    return points[medial_mask], points[lateral_mask]


def _compartment_masks(points: np.ndarray, side: Side, midline_x: float):
    """As :func:`_compartments`, returning the (medial, lateral) masks."""
    origin = np.array([midline_x, 0.0, 0.0])
    return side.split_compartments(points, origin, LPS_PATIENT_LEFT)


def _articular_envelope(mesh: Mesh, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """The proximal surface of the tibia as seen from above, and which of it is interior.

    Faces near the top that are close to horizontal are gathered, and in each
    ``ENVELOPE_CELL_MM`` square of the transverse plane the highest one is kept. That is
    the surface a probe lowered from above would touch: the plateaus, the eminence and the
    rims, but not the cortex of the metaphysis, whose faces are steep, nor anything
    beneath the joint surface, which is always lower in its cell. Face orientation is
    not trusted, so the test is on the absolute normal.

    ``interior`` marks the cells at least two cells in from the edge of the envelope,
    where the rim turns down and the surface stops being a dish.
    """
    triangles = mesh.vertices[mesh.faces]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    cosines = np.zeros(len(normals))
    cosines[valid] = np.abs(normals[valid] @ LPS_SUPERIOR) / lengths[valid]

    keep = (
        (centroids @ LPS_SUPERIOR >= z_max - PLATEAU_SLAB_MM)
        & (cosines >= ENVELOPE_MIN_COS)
    )
    points = centroids[keep]
    if not len(points):
        return np.empty((0, 3)), np.empty(0, dtype=bool)

    cells = np.floor(points[:, :2] / ENVELOPE_CELL_MM).astype(np.int64)
    heights = points @ LPS_SUPERIOR
    # Highest face per cell: sort by cell then height, and take the last of each run.
    order = np.lexsort((heights, cells[:, 1], cells[:, 0]))
    cells, points = cells[order], points[order]
    last = np.ones(len(cells), dtype=bool)
    last[:-1] = np.any(cells[1:] != cells[:-1], axis=1)
    cells, points = cells[last], points[last]

    occupied = {tuple(cell) for cell in cells}
    reach = range(-2, 3)
    interior = np.array([
        all((cx + dx, cy + dy) in occupied for dx in reach for dy in reach)
        for cx, cy in cells
    ], dtype=bool)
    return points, interior


def _central(points: np.ndarray, *, fraction: float) -> np.ndarray:
    """The points in the middle ``fraction`` of their own ML and AP extent.

    The deepest point of a plateau is looked for near its middle. The lateral plateau is
    convex from front to back, so its lowest points are otherwise at the rim.
    """
    if len(points) == 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    for axis in (0, 1):
        low, high = points[:, axis].min(), points[:, axis].max()
        margin = (1.0 - fraction) / 2.0 * (high - low)
        keep &= (points[:, axis] >= low + margin) & (points[:, axis] <= high - margin)
    return points[keep]


def condylar_width_mm(mesh: Mesh) -> float:
    """The widest caliper of the bottom ``CONDYLAR_SLAB_MM`` of a femur, in the transverse plane.

    The widest of 180 directions, so it does not depend on how the femur lies in the
    scanner: on the condylar block that is the mediolateral width. It is the femur's own
    size, measured before any landmark exists, which is what the distal window needs.
    """
    vertices = mesh.vertices
    z = vertices @ LPS_SUPERIOR
    slab = vertices[z <= float(z.min()) + CONDYLAR_SLAB_MM][:, :2]
    angles = np.radians(np.arange(180) * 1.0)
    calipers = slab @ np.stack([np.cos(angles), np.sin(angles)])
    return float(np.max(calipers.max(axis=0) - calipers.min(axis=0)))


def distal_window_mm(mesh: Mesh) -> float:
    """The length of femur the distal searches take their fractions of.

    The whole segment on a knee-only scan, and ``DISTAL_WINDOW_CONDYLAR_WIDTHS``
    condylar widths of it on anything longer. Measured from the most distal point, so it
    is anchored at the joint and the scan's far end plays no part once it is beyond the
    window.
    """
    z = mesh.vertices @ LPS_SUPERIOR
    height = float(z.max() - z.min())
    return min(height, DISTAL_WINDOW_CONDYLAR_WIDTHS * condylar_width_mm(mesh))


def _distal_block(vertices: np.ndarray, window: float) -> np.ndarray:
    """The bottom fifth of the distal window: the condyles and the notch beneath the shaft.

    Written exactly as :func:`~tka_planner.core.geometry.regional_mask` computes its
    fraction, so where the window is the whole segment the block is the same vertices to
    the last bit as before the window existed.
    """
    z = vertices @ LPS_SUPERIOR
    return vertices[z <= float(z.min()) + 0.20 * window]


def estimate_femoral_landmarks(mesh: Mesh, side: Side, *,
                               axes: BoneAxes | None = None) -> list[Landmark]:
    """Estimate femoral landmarks from the distal femur surface, in the femur's own axes.

    ``axes`` are the femur's :func:`femoral_axes`, found here when not given. Every
    search below runs in them, and the landmarks are returned in scanner coordinates.

    Reliability by landmark:

    * distal and posterior condyles -- **good**; true surface extrema, averaged over
      several vertices so a marching-cubes spike cannot define one.
    * intercondylar notch -- **fair**; the roof of the notch on the midline at its lowest,
      which a flat stretch of roof leaves loosely placed from front to back.
    * epicondyles -- **fair**; the most medial and lateral points of the epicondylar
      band. The medial sulcus in particular is a depression rather than an extreme, so
      this systematically finds the prominence beside it instead.
    * canal centres -- **good** where enough shaft is present; the diaphysis is close to
      circular in section.
    """
    side = Side.parse(side)
    axes = femoral_axes(mesh, side) if axes is None else axes
    shaft = LPS_SUPERIOR if axes.shaft is None else axes.rotation @ axes.shaft
    return _in_world(_femoral_landmarks_in_axes(axes.local_mesh(mesh), side, shaft), axes)


def _in_world(landmarks: list[Landmark], axes: BoneAxes) -> list[Landmark]:
    """Landmarks found in a bone's own axes, moved back to scanner coordinates."""
    for landmark in landmarks:
        if landmark.position_mm is not None:
            landmark.position_mm = axes.to_world(landmark.position_mm)
    return landmarks


def _femoral_landmarks_in_axes(mesh: Mesh, side: Side,
                               shaft: np.ndarray = LPS_SUPERIOR) -> list[Landmark]:
    """The femoral searches, on a femur expressed in its own axes.

    In these coordinates +Z is the femur's working mechanical axis and +X its posterior
    condylar line, so "distal", "posterior" and "lateral" below are the femur's own
    directions; the coordinate letters are those of LPS only because the axes are in LPS
    order. ``shaft`` is the diaphyseal axis in these coordinates, which the canal
    sections are cut square to.
    """
    vertices = mesh.vertices
    z = vertices @ LPS_SUPERIOR
    z_min, z_max = float(z.min()), float(z.max())
    # Every distal search is a fraction of this, not of the scan: see
    # DISTAL_WINDOW_CONDYLAR_WIDTHS. On a knee-only segment it is the segment's height.
    window = distal_window_mm(mesh)

    # ---- Stage one: the femur's own mediolateral axis ---------------
    #
    # The posterior condylar line, which the axes were built on. Searching for extremes
    # along a scanner axis finds the wrong points on a femur turned in the scanner: an
    # earlier version took the widest points of the epicondylar band along world X and,
    # on a femur turned 15 degrees, put the rotational reference 26 degrees out. The
    # posterior condyles are genuine extremes of the surface, and the posterior condylar
    # line is the standard surgical rotational reference in any case. Converged again
    # here, from the axes' own mediolateral direction, so the searches below use exactly
    # the line found on these vertices.
    lateral_axis, posterior = _converge_posterior_condylar_axis(mesh, side,
                                                                window=window)

    landmarks: list[Landmark] = []

    # ---- Stage two: posterior condyles, then the notch between them ---
    distal = _distal_block(vertices, window)
    offsets = distal @ lateral_axis
    median = float(np.median(offsets))
    halves = _medial_lateral(distal, offsets >= median, side)
    posterior_condyles = {
        compartment: extremal_point(points, posterior, n_average=CONTACT_AVERAGE_N)
        for compartment, points in halves.items() if len(points) >= CONTACT_AVERAGE_N
    }

    # The notch roof is looked for on the midline between the posterior condyles, and
    # the distal condyles are then split there: at the intercondylar gap rather than at
    # the median of whatever the distal block holds.
    midline = median
    if len(posterior_condyles) == 2:
        middle = 0.5 * (posterior_condyles["medial"] + posterior_condyles["lateral"])
        notch = _notch_roof(distal, float(middle @ lateral_axis), lateral_axis)
        if notch is not None:
            midline = float(notch @ lateral_axis)
            landmarks.append(_estimated("femur.notch_centre", notch, "notch_roof.v2",
                                        "fair"))

    # ---- Stage three: distal condyles, along the femur's own axis ---
    halves = _medial_lateral(distal, offsets >= midline, side)
    for compartment in ("medial", "lateral"):
        points = halves[compartment]
        if len(points) < CONTACT_AVERAGE_N:
            continue
        landmarks.append(_estimated(
            f"femur.condyle_distal_{compartment}",
            extremal_point(points, -LPS_SUPERIOR, n_average=CONTACT_AVERAGE_N),
            "condyle_extreme.v4", "good",
        ))
        if compartment in posterior_condyles:
            landmarks.append(_estimated(
                f"femur.condyle_posterior_{compartment}",
                posterior_condyles[compartment], "condyle_extreme.v4", "good",
            ))

    # ---- Stage four: epicondyles, along the corrected axis ----------
    # 15 to 35 per cent of the distal window above the most distal point: 22 to 87 mm
    # on the knee-only segments. A fraction of the whole scan put a full-length femur's
    # band on the shaft.
    band = vertices[(z >= z_min + 0.15 * window) & (z <= z_min + 0.35 * window)]
    if len(band) >= CONTACT_AVERAGE_N:
        lateral_direction = (
            lateral_axis if side is Side.LEFT else -lateral_axis
        )
        landmarks.append(_estimated(
            "femur.epicondyle_lateral",
            extremal_point(band, lateral_direction, n_average=CONTACT_AVERAGE_N),
            "epicondylar_band.v3", "fair",
        ))
        landmarks.append(_estimated(
            "femur.epicondyle_medial_prominence",
            extremal_point(band, -lateral_direction, n_average=CONTACT_AVERAGE_N),
            "epicondylar_band.v3", "fair",
        ))
        # The medial sulcus is a depression, not a surface extreme, so no extremal
        # search can find it. Left unpicked deliberately: the frame then falls back to
        # the anatomical epicondylar axis and records the substitution, and the condylar
        # twist angle reports itself not computable rather than as a wrong number.
        landmarks.append(Landmark(
            id="femur.epicondyle_medial_sulcus",
            status=LandmarkStatus.NOT_PICKED,
            reason="a depression rather than a surface extreme; requires a human pick",
        ))

    landmarks.extend(_canal_centres(mesh, "femur", shaft))
    return landmarks


def _medial_lateral(points: np.ndarray, patient_left: np.ndarray,
                    side: Side) -> dict[str, np.ndarray]:
    """The two halves of a split, named by compartment: patient-left is lateral on a
    left knee and medial on a right one."""
    left, right = points[patient_left], points[~patient_left]
    if side is Side.LEFT:
        return {"medial": right, "lateral": left}
    return {"medial": left, "lateral": right}


def _notch_roof(distal: np.ndarray, middle: float,
                lateral_axis: np.ndarray) -> np.ndarray | None:
    """The roof of the intercondylar notch on the midline, at its lowest.

    Seen in a sagittal section on the midline, the distal femur is a hook: the trochlear
    groove comes down the front, turns under, and the roof of the notch climbs steeply
    back up behind it to the intercondylar line. The lowest stretch of that outline is
    where the notch roof meets the end of the trochlea -- the top of the notch as an
    anteroposterior view shows it, and where an intramedullary rod enters. Its height is
    sharp, but on most knees it is flat for several millimetres from front to back, so
    the point is the centre of the stretch within ``NOTCH_BAND_MM`` of its lowest rather
    than its single lowest vertex, which noise would slide along the flat.

    Version 1 took the most proximal point within 9 mm of the median of the condylar
    block, which is always the ceiling of the block, 20 to 40 mm above the notch on the
    anterior or posterior cortex; under 0.1 mm of noise it moved 15 to 43 mm.
    """
    strip = distal[np.abs(distal @ lateral_axis - middle) <= NOTCH_STRIP_MM]
    if len(strip) < CONTACT_AVERAGE_N:
        return None
    heights = strip @ LPS_SUPERIOR
    lowest = float(np.sort(heights)[:CONTACT_AVERAGE_N].mean())
    return strip[heights <= lowest + NOTCH_BAND_MM].mean(axis=0)


def _converge_posterior_condylar_axis(
    mesh: Mesh, side: Side, *, window: float | None = None,
    start: np.ndarray | None = None,
):
    """Find the femur's mediolateral axis from its posterior condyles, iteratively.

    Locating the posterior condyles requires knowing which way is posterior, and knowing
    which way is posterior requires the mediolateral axis -- so the two are solved
    together, by :func:`_converge_posterior_axis` on the condylar block.

    Returns ``(lateral_axis, posterior)``, both unit vectors in the transverse plane.
    ``lateral_axis`` points along the posterior condylar line toward the patient's left,
    matching the frame convention; the caller flips it per side where a genuinely lateral
    direction is wanted.

    On the cohort this converges to within a degree of the distal condylar axis, whereas
    the epicondylar-band search it replaced sat 26 degrees away.
    """
    distal = _distal_block(
        mesh.vertices, distal_window_mm(mesh) if window is None else window
    )
    return _converge_posterior_axis(distal, start=start)


def _converge_posterior_axis(
    block: np.ndarray, *, start: np.ndarray | None = None, max_rounds: int = 12,
    tolerance_deg: float = 0.01,
):
    """The line touching the backs of a bone's two condyles, in the transverse plane.

    Each round splits the block about the current mediolateral estimate at its median,
    finds the most posterior points of each half, and takes the line between them as the
    next estimate. It settles in a handful of rounds. ``start`` is where it begins: the
    bone's own widest direction when the caller has one, so that where it settles does
    not depend on how the bone lies; the transverse axis otherwise.
    """
    lateral_axis = (LPS_PATIENT_LEFT if start is None else start).astype(float).copy()
    lateral_axis[2] = 0.0
    lateral_axis /= np.linalg.norm(lateral_axis)
    for _ in range(max_rounds):
        posterior = np.array([-lateral_axis[1], lateral_axis[0], 0.0])
        norm = np.linalg.norm(posterior)
        if norm < 1e-9:
            break
        posterior /= norm
        if float(np.dot(posterior, LPS_POSTERIOR)) < 0:
            posterior = -posterior

        offsets = block @ lateral_axis
        midline = float(np.median(offsets))
        one, other = block[offsets >= midline], block[offsets < midline]
        if min(len(one), len(other)) < CONTACT_AVERAGE_N:
            break

        updated = (
            extremal_point(one, posterior, n_average=2 * CONTACT_AVERAGE_N)
            - extremal_point(other, posterior, n_average=2 * CONTACT_AVERAGE_N)
        )
        updated[2] = 0.0
        norm = np.linalg.norm(updated)
        if norm < 1e-9:
            break
        updated /= norm
        if float(np.dot(updated, lateral_axis)) < 0:
            updated = -updated

        shift = np.degrees(np.arccos(
            np.clip(float(np.dot(updated, lateral_axis)), -1.0, 1.0)
        ))
        lateral_axis = updated
        if shift < tolerance_deg:
            break

    # Report it pointing patient-left, so the sign convention matches the frame.
    if float(np.dot(lateral_axis, LPS_PATIENT_LEFT)) < 0:
        lateral_axis = -lateral_axis

    posterior = np.array([-lateral_axis[1], lateral_axis[0], 0.0])
    posterior /= np.linalg.norm(posterior)
    if float(np.dot(posterior, LPS_POSTERIOR)) < 0:
        posterior = -posterior
    return lateral_axis, posterior


def estimate_tibial_landmarks(mesh: Mesh, side: Side, *,
                              axes: BoneAxes | None = None) -> list[Landmark]:
    """Estimate tibial landmarks from the proximal tibia surface, in the tibia's own axes.

    ``axes`` are the tibia's :func:`tibial_axes`, found here when not given. Every search
    below runs in them -- the compartment split, the articular grid, and the anterior
    and posterior searches for the tubercle, the PCL footprint and the rim -- and the
    landmarks are returned in scanner coordinates.

    Reliability by landmark:

    * plateau low points -- **good**; the deepest part of the middle of each articular
      dish, found on the upward-facing surface only, so the search cannot fall onto the
      sides of the bone.
    * intercondylar tubercles -- **fair**; the most proximal point either side of the
      midline.
    * tibial tubercle -- **fair**; the most anterior prominence below the plateau, which
      does have a real geometric signature.
    * PCL insertion -- **poor**; a soft-tissue footprint with no surface feature to find.
      Estimated as the posterior intercondylar area, and the first landmark a human
      should correct.
    """
    side = Side.parse(side)
    axes = tibial_axes(mesh) if axes is None else axes
    return _in_world(_tibial_landmarks_in_axes(axes.local_mesh(mesh), side), axes)


def _tibial_landmarks_in_axes(mesh: Mesh, side: Side) -> list[Landmark]:
    """The tibial searches, on a tibia expressed in its own axes: +Z its diaphyseal
    axis, +X its posterior condylar line, +Y posterior; the letters are LPS's only
    because the axes are in LPS order."""
    vertices = mesh.vertices
    z = vertices @ LPS_SUPERIOR
    z_max = float(z.max())

    landmarks: list[Landmark] = []

    # A slab at the top holds the articular surface, but the slab also holds the sides
    # of the metaphysis. Its lowest point is therefore always the slab floor on the
    # cortex, whatever the anatomy -- which is what version 1 of the dish estimator
    # returned, 15 mm below the real plateau on case P009. So the plateaus are searched
    # on the articular surface alone: the upward-facing envelope, seen from above.
    plateau = vertices[z >= z_max - PLATEAU_SLAB_MM]
    if len(plateau) < 3 * CONTACT_AVERAGE_N:
        plateau = vertices[
            regional_mask(vertices, LPS_SUPERIOR, fraction=0.15, end="high")
        ]

    midline_x = float((plateau[:, 0].min() + plateau[:, 0].max()) / 2.0)
    medial, lateral = _compartments(plateau, side, midline_x)

    envelope, interior = _articular_envelope(mesh, z_max)
    half_width = float(plateau[:, 0].max() - plateau[:, 0].min()) / 2.0
    off_midline = np.abs(envelope[:, 0] - midline_x) >= 0.2 * half_width
    surface = {}
    for compartment, keep in zip(
        ("medial", "lateral"), _compartment_masks(envelope, side, midline_x)
    ):
        surface[compartment] = envelope[keep & off_midline]
        dish = envelope[keep & off_midline & interior]
        dish = _central(dish, fraction=0.6)
        if len(dish) < CONTACT_AVERAGE_N:
            continue
        landmarks.append(_estimated(
            f"tibia.plateau_{compartment}_lowest",
            extremal_point(dish, -LPS_SUPERIOR, n_average=CONTACT_AVERAGE_N),
            "plateau_dish.v3", "good",
        ))

    # Intercondylar tubercles: the most proximal point on each side of the midline.
    intercondylar = plateau[np.abs(plateau[:, 0] - midline_x) < 16.0]
    if len(intercondylar) >= 2 * CONTACT_AVERAGE_N:
        spine_medial, spine_lateral = _compartments(intercondylar, side, midline_x)
        spines = []
        for compartment, points in (("medial", spine_medial),
                                    ("lateral", spine_lateral)):
            if len(points) >= CONTACT_AVERAGE_N:
                spines.append(extremal_point(points, LPS_SUPERIOR,
                                             n_average=CONTACT_AVERAGE_N))
                landmarks.append(_estimated(
                    f"tibia.spine_{compartment}", spines[-1], "tibial_spine.v2", "fair",
                ))

        # PCL footprint: the posterior intercondylar area just below the joint surface,
        # on the midline -- behind the eminence, between the spines. Version 1 searched
        # the whole 32 mm intercondylar band, whose most posterior points are the backs
        # of the two condyles at its edges, either side of the fossa the ligament
        # inserts into: on every knee it picked one edge or the other, 16 mm off the
        # midline, and a turn of the knee could swap them (32 mm on the reference knee).
        centre = float(np.mean([p[0] for p in spines])) if len(spines) == 2 \
            else midline_x
        posterior_area = plateau[(np.abs(plateau[:, 0] - centre) <= PCL_STRIP_MM)
                                 & (plateau[:, 2] >= z_max - 18.0)]
        if len(posterior_area) >= CONTACT_AVERAGE_N:
            landmarks.append(_estimated(
                "tibia.pcl_insertion_midpoint",
                extremal_point(posterior_area, LPS_POSTERIOR,
                               n_average=CONTACT_AVERAGE_N),
                "posterior_intercondylar.v2", "poor",
            ))

    # Tibial tubercle: the anterior prominence 20-50 mm below the plateau.
    tubercle_slab = vertices[(z <= z_max - 20.0) & (z >= z_max - 50.0)]
    if len(tubercle_slab) >= CONTACT_AVERAGE_N:
        tubercle = extremal_point(tubercle_slab, ANTERIOR,
                                  n_average=CONTACT_AVERAGE_N)
        landmarks.append(_estimated(
            "tibia.tubercle_medial_third",
            tubercle + 6.0 * side.medial_direction(LPS_PATIENT_LEFT),
            "anterior_tubercle.v2", "fair",
        ))
        # Akagi's original reference is the medial border of the tendon insertion,
        # slightly medial to the apex. The offset is nominal, not measured.
        landmarks.append(_estimated(
            "tibia.tubercle_patellar_tendon_medial_border",
            tubercle + 9.0 * side.medial_direction(LPS_PATIENT_LEFT),
            "anterior_tubercle.v2", "poor",
        ))

    # Medial plateau rim, for the native posterior slope. Taken from the articular
    # surface too: the most anterior point of the whole slab is on the metaphyseal
    # flare below the rim, and a slope drawn to it measures the flare.
    # Only the middle of the plateau from side to side counts, since the slope is a
    # property of the dish and not of the intercondylar area beside it, which falls
    # away steeply behind the eminence.
    rim = surface.get("medial", np.empty((0, 3)))
    if len(rim):
        low, high = rim[:, 0].min(), rim[:, 0].max()
        margin = 0.25 * (high - low)
        rim = rim[(rim[:, 0] >= low + margin) & (rim[:, 0] <= high - margin)]
    if len(rim) >= 2 * CONTACT_AVERAGE_N:
        landmarks.append(_estimated(
            "tibia.plateau_medial_anterior",
            extremal_point(rim, ANTERIOR, n_average=CONTACT_AVERAGE_N),
            "plateau_rim.v3", "fair",
        ))
        landmarks.append(_estimated(
            "tibia.plateau_medial_posterior",
            extremal_point(rim, LPS_POSTERIOR, n_average=CONTACT_AVERAGE_N),
            "plateau_rim.v3", "fair",
        ))

    landmarks.extend(_canal_centres(mesh, "tibia"))
    return landmarks


def _canal_centres(
    mesh: Mesh, bone: str, axis: np.ndarray = LPS_SUPERIOR
) -> list[Landmark]:
    """Trace the diaphyseal axis by taking cross-section centroids at two levels.

    The sections are cut square to ``axis``, at fractions of the bone's length along it:
    in the bone's own axes, the diaphyseal axis itself.

    The levels are placed as far apart as the segmentation allows, because the
    separation is the lever arm of the axis fit and angular uncertainty scales inversely
    with it. On the shorter femurs in this cohort that arm is only about 70 mm against a
    more usual 200 mm, so the axis is correspondingly noisier -- which is why the
    separation is recorded in the frame's diagnostics rather than left implicit.
    """
    vertices = mesh.vertices
    axis = unit(axis)
    heights = vertices @ axis
    z_min, z_max = float(heights.min()), float(heights.max())
    height = z_max - z_min
    if height < 60.0:
        return []

    # Stay clear of the articular end, where the section is not the diaphysis.
    if bone == "femur":
        near_level, far_level = z_min + 0.45 * height, z_min + 0.92 * height
        near_id, far_id = "femur.canal_centre_distal", "femur.canal_centre_proximal"
    else:
        near_level, far_level = z_max - 0.45 * height, z_max - 0.92 * height
        near_id, far_id = "tibia.canal_centre_proximal", "tibia.canal_centre_distal"

    landmarks = []
    origin = np.array([0.0, 0.0, 0.0])
    for landmark_id, level in ((near_id, near_level), (far_id, far_level)):
        try:
            centre = cross_section_centroid(
                vertices, axis, level,
                slab_thickness_mm=4.0, origin=origin,
            )
        except ValueError:
            continue
        landmarks.append(_estimated(
            landmark_id, centre, "canal_section.v2", "good",
        ))
    return landmarks


def estimate_landmarks(
    femur: Mesh,
    tibia: Mesh,
    side: "str | Side",
    *,
    case_id: str,
    femur_axes: BoneAxes | None = None,
    tibia_axes: BoneAxes | None = None,
) -> LandmarkSet:
    """Estimate a full landmark set for one knee from its two bone surfaces.

    The result is a complete, immediately usable starting point: every landmark carries
    ``ESTIMATED`` status and a confidence, so a plan built from it runs end to end while
    remaining visibly provisional. Landmarks outside the scan -- the femoral head and the
    ankle centre on a knee-only study -- are recorded as ``OUT_OF_SCAN`` rather than
    omitted, which is what lets the capability model distinguish "not imaged" from
    "not yet picked".

    ``femur_axes`` and ``tibia_axes`` are the bones' own axes when the caller has already
    found them (the orientation gate reads the same ones); otherwise they are found here.
    """
    side = Side.parse(side)
    femur_axes = femoral_axes(femur, side) if femur_axes is None else femur_axes
    tibia_axes = tibial_axes(tibia) if tibia_axes is None else tibia_axes
    landmarks = estimate_femoral_landmarks(femur, side, axes=femur_axes)
    landmarks += estimate_tibial_landmarks(tibia, side, axes=tibia_axes)

    for landmark_id, reason in (
        ("femur.head_centre", "knee-only field of view"),
        ("tibia.ankle_centre", "knee-only field of view"),
    ):
        landmarks.append(
            Landmark(id=landmark_id, status=LandmarkStatus.OUT_OF_SCAN, reason=reason)
        )

    return LandmarkSet(
        case_id=case_id,
        side=side,
        coordinate_system="LPS",
        landmarks=landmarks,
        source={
            "kind": "automatic_estimate",
            "method": "landmarks_auto.v2",
            "note": (
                "Every position here is a machine estimate intended for human review, "
                "not a picked landmark. Confidence is recorded per landmark; the "
                "tibial rotational references are the weakest and should be corrected "
                "first."
            ),
            "femur_sha256": femur.sha256,
            "tibia_sha256": tibia.sha256,
            "bone_axes": {
                bone: {"method": axes.method, **axes.diagnostics}
                for bone, axes in (("femur", femur_axes), ("tibia", tibia_axes))
            },
        },
    )
