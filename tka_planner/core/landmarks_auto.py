"""Automatic landmark estimation from bone surfaces.

Picking landmarks by hand is the slow step in this pipeline: roughly twenty points per
knee, and the validation cohort needs them twice over for an observer study. This module
produces a complete first pass automatically, so a plan can be generated from nothing
but two STL files.

**Everything it produces is marked** :attr:`~tka_planner.core.landmarks.LandmarkStatus.ESTIMATED`,
never ``PRESENT``. That distinction is the point. An automatic estimate is a starting
position for a human to correct, not a measurement, and a report built on estimates says
so on every line. The same machinery that flags an assumed mechanical axis flags these.

The estimators are geometric and deliberately simple. They exploit the fact that the
segmentations already arrive in the LPS patient frame, so anatomical directions are
known before any landmark exists -- superior is +Z, posterior is +Y, patient-left is +X.
This is what the legacy pipeline threw away by recentring each bone on its bounding box
and then spending PCA and a bounding-box heuristic trying to recover it.

Reliability varies by landmark and is stated per estimator below. The condylar and
plateau extremes are dependable, being genuine extrema of the surface. The epicondyles
are fair. The tibial rotational references are the weakest -- the patellar tendon
insertion has no geometric signature a bounding box can find -- and are the ones a human
should correct first.
"""

from __future__ import annotations

import numpy as np

from .geometry import cross_section_centroid, extremal_point, regional_mask
from .landmarks import Landmark, LandmarkSet, LandmarkStatus
from .meshio import Mesh
from .sides import LPS_PATIENT_LEFT, LPS_POSTERIOR, LPS_SUPERIOR, Side

__all__ = [
    "estimate_femoral_landmarks",
    "estimate_tibial_landmarks",
    "estimate_landmarks",
]

ANTERIOR = -LPS_POSTERIOR
CONTACT_AVERAGE_N = 10  # matches the legacy CONTACT_AVG_N
PLATEAU_SLAB_MM = 22.0
ENVELOPE_CELL_MM = 2.0
# A face this close to horizontal can be joint surface; anything steeper is the side of
# the bone. cos 55 degrees.
ENVELOPE_MIN_COS = 0.57


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


def estimate_femoral_landmarks(mesh: Mesh, side: Side) -> list[Landmark]:
    """Estimate femoral landmarks from the distal femur surface.

    Reliability by landmark:

    * distal and posterior condyles -- **good**; true surface extrema, averaged over
      several vertices so a marching-cubes spike cannot define one.
    * intercondylar notch -- **fair**; taken as the roof of the fossa near the midline.
    * epicondyles -- **fair**; the most medial and lateral points of the epicondylar
      band. The medial sulcus in particular is a depression rather than an extreme, so
      this systematically finds the prominence beside it instead.
    * canal centres -- **good** where enough shaft is present; the diaphysis is close to
      circular in section.
    """
    vertices = mesh.vertices
    z = vertices @ LPS_SUPERIOR
    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min

    landmarks: list[Landmark] = []

    # ---- Stage one: find the femur's own mediolateral axis ----------
    #
    # Everything below depends on knowing which way the femur faces, and a femur lies at
    # whatever rotation the scanner found it in. Searching for extremes along the world
    # axes therefore finds the wrong points.
    #
    # An earlier version took the widest points of the epicondylar band along world X.
    # On a femur rotated 15 degrees that lands nowhere near the epicondyles -- it picks
    # whatever part of the condylar circumference happens to be widest in world X -- and
    # it put the rotational reference 26 degrees out. The posterior condyles are a far
    # better anchor: they are genuine extremes of the surface, and the posterior
    # condylar line is the standard surgical rotational reference in any case.
    lateral_axis, posterior = _converge_posterior_condylar_axis(mesh, side)

    landmarks: list[Landmark] = []

    # ---- Stage two: condyles, along the femur's own directions ------
    distal = vertices[regional_mask(vertices, LPS_SUPERIOR, fraction=0.20, end="low")]
    offsets = distal @ lateral_axis
    midline = float(np.median(offsets))
    lateral_half, medial_half = distal[offsets >= midline], distal[offsets < midline]
    if side is Side.RIGHT:
        lateral_half, medial_half = medial_half, lateral_half

    for compartment, points in (("medial", medial_half), ("lateral", lateral_half)):
        if len(points) < CONTACT_AVERAGE_N:
            continue
        landmarks.append(_estimated(
            f"femur.condyle_distal_{compartment}",
            extremal_point(points, -LPS_SUPERIOR, n_average=CONTACT_AVERAGE_N),
            "condyle_extreme.v3", "good",
        ))
        landmarks.append(_estimated(
            f"femur.condyle_posterior_{compartment}",
            extremal_point(points, posterior, n_average=CONTACT_AVERAGE_N),
            "condyle_extreme.v3", "good",
        ))

    # ---- Stage three: epicondyles, along the corrected axis ---------
    band = vertices[(z >= z_min + 0.15 * height) & (z <= z_min + 0.35 * height)]
    if len(band) >= CONTACT_AVERAGE_N:
        lateral_direction = (
            lateral_axis if side is Side.LEFT else -lateral_axis
        )
        landmarks.append(_estimated(
            "femur.epicondyle_lateral",
            extremal_point(band, lateral_direction, n_average=CONTACT_AVERAGE_N),
            "epicondylar_band.v2", "fair",
        ))
        landmarks.append(_estimated(
            "femur.epicondyle_medial_prominence",
            extremal_point(band, -lateral_direction, n_average=CONTACT_AVERAGE_N),
            "epicondylar_band.v2", "fair",
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

    # Intercondylar notch: near the midline of the condylar block, the roof of the
    # fossa is the most proximal surface.
    midline_band = distal[np.abs(offsets - midline) < 9.0]
    if len(midline_band) >= CONTACT_AVERAGE_N:
        landmarks.append(_estimated(
            "femur.notch_centre",
            extremal_point(midline_band, LPS_SUPERIOR, n_average=CONTACT_AVERAGE_N),
            "notch_roof.v1", "fair",
        ))

    landmarks.extend(_canal_centres(mesh, "femur", z_min, z_max))
    return landmarks


def _converge_posterior_condylar_axis(
    mesh: Mesh, side: Side, *, max_rounds: int = 12, tolerance_deg: float = 0.01
):
    """Find the femur's mediolateral axis from its posterior condyles, iteratively.

    Locating the posterior condyles requires knowing which way is posterior, and knowing
    which way is posterior requires the mediolateral axis -- so the two are solved
    together. Starting from the world axis, each round splits the condylar block about
    the current estimate, finds the most posterior point of each half, and takes the line
    between them as the next estimate. It settles in a handful of rounds.

    Returns ``(lateral_axis, posterior)``, both unit vectors in the transverse plane.
    ``lateral_axis`` points along the posterior condylar line toward the patient's left,
    matching the frame convention; the caller flips it per side where a genuinely lateral
    direction is wanted.

    On the cohort this converges to within a degree of the distal condylar axis, whereas
    the epicondylar-band search it replaced sat 26 degrees away.
    """
    vertices = mesh.vertices
    distal = vertices[
        regional_mask(vertices, LPS_SUPERIOR, fraction=0.20, end="low")
    ]

    lateral_axis = LPS_PATIENT_LEFT.astype(float).copy()
    for _ in range(max_rounds):
        posterior = np.array([-lateral_axis[1], lateral_axis[0], 0.0])
        norm = np.linalg.norm(posterior)
        if norm < 1e-9:
            break
        posterior /= norm
        if float(np.dot(posterior, LPS_POSTERIOR)) < 0:
            posterior = -posterior

        offsets = distal @ lateral_axis
        midline = float(np.median(offsets))
        one, other = distal[offsets >= midline], distal[offsets < midline]
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


def estimate_tibial_landmarks(mesh: Mesh, side: Side) -> list[Landmark]:
    """Estimate tibial landmarks from the proximal tibia surface.

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
            "plateau_dish.v2", "good",
        ))

    # Intercondylar tubercles: the most proximal point on each side of the midline.
    intercondylar = plateau[np.abs(plateau[:, 0] - midline_x) < 16.0]
    if len(intercondylar) >= 2 * CONTACT_AVERAGE_N:
        spine_medial, spine_lateral = _compartments(intercondylar, side, midline_x)
        for compartment, points in (("medial", spine_medial),
                                    ("lateral", spine_lateral)):
            if len(points) >= CONTACT_AVERAGE_N:
                landmarks.append(_estimated(
                    f"tibia.spine_{compartment}",
                    extremal_point(points, LPS_SUPERIOR, n_average=CONTACT_AVERAGE_N),
                    "tibial_spine.v1", "fair",
                ))

        # PCL footprint: posterior intercondylar area, just below the joint surface.
        posterior_area = intercondylar[
            intercondylar[:, 2] >= z_max - 18.0
        ]
        if len(posterior_area) >= CONTACT_AVERAGE_N:
            landmarks.append(_estimated(
                "tibia.pcl_insertion_midpoint",
                extremal_point(posterior_area, LPS_POSTERIOR,
                               n_average=CONTACT_AVERAGE_N),
                "posterior_intercondylar.v1", "poor",
            ))

    # Tibial tubercle: the anterior prominence 20-50 mm below the plateau.
    tubercle_slab = vertices[(z <= z_max - 20.0) & (z >= z_max - 50.0)]
    if len(tubercle_slab) >= CONTACT_AVERAGE_N:
        tubercle = extremal_point(tubercle_slab, ANTERIOR,
                                  n_average=CONTACT_AVERAGE_N)
        landmarks.append(_estimated(
            "tibia.tubercle_medial_third",
            tubercle + 6.0 * side.medial_direction(LPS_PATIENT_LEFT),
            "anterior_tubercle.v1", "fair",
        ))
        # Akagi's original reference is the medial border of the tendon insertion,
        # slightly medial to the apex. The offset is nominal, not measured.
        landmarks.append(_estimated(
            "tibia.tubercle_patellar_tendon_medial_border",
            tubercle + 9.0 * side.medial_direction(LPS_PATIENT_LEFT),
            "anterior_tubercle.v1", "poor",
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
            "plateau_rim.v2", "fair",
        ))
        landmarks.append(_estimated(
            "tibia.plateau_medial_posterior",
            extremal_point(rim, LPS_POSTERIOR, n_average=CONTACT_AVERAGE_N),
            "plateau_rim.v2", "fair",
        ))

    landmarks.extend(_canal_centres(mesh, "tibia", float(z.min()), z_max))
    return landmarks


def _canal_centres(
    mesh: Mesh, bone: str, z_min: float, z_max: float
) -> list[Landmark]:
    """Trace the diaphyseal axis by taking cross-section centroids at two levels.

    The levels are placed as far apart as the segmentation allows, because the
    separation is the lever arm of the axis fit and angular uncertainty scales inversely
    with it. On the shorter femurs in this cohort that arm is only about 70 mm against a
    more usual 200 mm, so the axis is correspondingly noisier -- which is why the
    separation is recorded in the frame's diagnostics rather than left implicit.
    """
    vertices = mesh.vertices
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
                vertices, LPS_SUPERIOR, level,
                slab_thickness_mm=4.0, origin=origin,
            )
        except ValueError:
            continue
        landmarks.append(_estimated(
            landmark_id, centre, "canal_section.v1", "good",
        ))
    return landmarks


def estimate_landmarks(
    femur: Mesh,
    tibia: Mesh,
    side: "str | Side",
    *,
    case_id: str,
) -> LandmarkSet:
    """Estimate a full landmark set for one knee from its two bone surfaces.

    The result is a complete, immediately usable starting point: every landmark carries
    ``ESTIMATED`` status and a confidence, so a plan built from it runs end to end while
    remaining visibly provisional. Landmarks outside the scan -- the femoral head and the
    ankle centre on a knee-only study -- are recorded as ``OUT_OF_SCAN`` rather than
    omitted, which is what lets the capability model distinguish "not imaged" from
    "not yet picked".
    """
    side = Side.parse(side)
    landmarks = estimate_femoral_landmarks(femur, side)
    landmarks += estimate_tibial_landmarks(tibia, side)

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
            "method": "landmarks_auto.v1",
            "note": (
                "Every position here is a machine estimate intended for human review, "
                "not a picked landmark. Confidence is recorded per landmark; the "
                "tibial rotational references are the weakest and should be corrected "
                "first."
            ),
            "femur_sha256": femur.sha256,
            "tibia_sha256": tibia.sha256,
        },
    )
