"""The resection table: how much bone each planned cut takes off each compartment.

A plan states its cuts as planes; a surgeon reads them as millimetres of bone off each
condyle and each plateau. This module turns the one into the other by measuring the
patient's bone against the planes, in the layout surgeons expect: distal medial and
lateral, posterior medial and lateral, anterior, the two chamfers, and tibial medial
and lateral, each with the angle of its cut.

**How a thickness is measured.** For each cut, the thickness in a compartment is the
perpendicular distance from the cut plane to the bone point furthest beyond it on the
removed side, within that compartment. Two rules make that well defined:

* *Cuts are taken in order.* The distal cut comes first, then the anterior and
  posterior cuts, then the chamfers. What a cut removes is what is still there when it
  is made, so the posterior cut is measured only above the distal plane, and a chamfer
  only inside the distal, anterior and posterior planes. Without the order a chamfer
  would be credited with the whole posterior condyle.
* *Compartments are halves of the cut*: the medial and lateral halves of the distal
  section, about its centre, as the femoral cut box splits them.

The tibia is the exception, and on purpose. Its plateaus are dishes between a rim and
the intercondylar eminence, so the furthest bone above the cut is a tubercle or the rim,
which no stylus touches. The tibial thickness is read at each plateau's lowest point on
the articular surface, as a stylus reads it, and the furthest bone in each half (outside
the eminence) is carried beside it.

The bone is the femur's surface samples (one per square millimetre, the femoral box's
own) and the tibia's mesh vertices: the furthest point of a surface from a plane is at
a vertex, so vertices are exact for a plane, and samples are within hundredths of a
millimetre on a condyle. The one place a surface measure is not enough is a chamfer:
after the distal and posterior cuts the corner where those two planes meet is cut bone,
not surface, and it is the furthest point from the chamfer whenever the condyle fills
it. Whether it does is decided by casting a ray through the femur from the corner.

Angles are of each cut against the bone's own axes: the distal and tibial cuts in the
coronal plane against the mechanical axis (varus positive: the cut's normal tilted
medially, which leaves the medial side of the cut lower), the posterior cut's external
rotation from the posterior condylar axis found on the same bone samples, and the
anterior and chamfer cuts at the design's angles in the component's sagittal plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import slice_mesh
from .geometry import unit
from .meshio import Mesh

__all__ = [
    "BoxProfile", "ResectionTable", "femoral_resection_rows", "tibial_resection_rows",
    "coronal_angle_deg", "inside_along_z",
]

# The posterior condyles are looked for up to this height above the distal cut, as the
# femoral cut box looks for them.
POSTERIOR_CONDYLE_HEIGHT_MM = 45.0
# On the tibia, bone this close to the centre of the section is the intercondylar
# eminence and belongs to neither plateau: the landmark estimator's intercondylar band.
INTERCONDYLAR_HALF_BAND_MM = 16.0
# Corner rays are cast at this many positions across each condyle.
CORNER_PROBES = 9
_JITTER_MM = np.array([0.6180339887e-3, 0.4142135624e-3])

FEMORAL_CUTS = ("distal", "posterior", "anterior", "posterior_chamfer",
                "anterior_chamfer")
# How each row is named wherever the table is shown.
ROW_LABELS = {
    "distal": "Distal femur",
    "posterior": "Posterior femur",
    "anterior": "Anterior femur",
    "posterior_chamfer": "Posterior chamfer",
    "anterior_chamfer": "Anterior chamfer",
    "tibial": "Tibia",
}
ANGLE_LABELS = {
    "varus_from_mechanical_axis": "varus from mech. axis",
    "valgus_from_anatomical_axis": "valgus from anat. axis",
    "flexion": "flexion",
    "external_rotation_from_posterior_condylar_axis": "ER from PCA",
    "flare_from_component_axis": "flare from component axis",
    "from_anterior_cortex": "from anterior cortex",
    "from_distal_cut": "from distal cut",
    "posterior_slope": "posterior slope",
}


def angle_text(angles: dict) -> str:
    """A row's angles as one line, e.g. ``varus from mech. axis 0.0, flexion 3.0``."""
    return ", ".join(f"{ANGLE_LABELS.get(key, key.replace('_', ' '))} {value:+.1f}"
                     for key, value in angles.items())


@dataclass(frozen=True)
class BoxProfile:
    """The femoral cuts in the component's sagittal plane (local +X anterior, +Z
    proximal, origin on the distal cut), as the femoral cut box places them.

    The posterior chamfer is taken at the same angle as the anterior one: the box design
    states a single chamfer angle.
    """

    posterior_x_mm: float
    anterior_x_at_distal_mm: float
    anterior_flare_deg: float
    flange_height_mm: float
    posterior_chamfer_mm: float
    distal_face_mm: float
    chamfer_deg: float
    shaft_slope_deg: float | None = None

    @classmethod
    def from_box(cls, box: dict, chamfer_deg: float) -> "BoxProfile":
        """From :meth:`FemoralCutBox.to_dict`, with the design's chamfer angle."""
        return cls(
            posterior_x_mm=float(box["posterior_cut_x_mm"]),
            anterior_x_at_distal_mm=float(box["anterior_cut_x_at_distal_mm"]),
            anterior_flare_deg=float(box["anterior_flare_deg"]),
            flange_height_mm=float(box["flange_height_mm"]),
            posterior_chamfer_mm=float(box["posterior_chamfer_mm"]),
            distal_face_mm=float(box["distal_face_mm"]),
            chamfer_deg=float(chamfer_deg),
            shaft_slope_deg=box.get("shaft_slope_deg"),
        )

    def signed(self, x: np.ndarray, z: np.ndarray) -> dict:
        """How far each point lies beyond each cut, outwards from the box, in mm."""
        flare = np.radians(self.anterior_flare_deg)
        c = np.radians(self.chamfer_deg)
        back = self.posterior_x_mm
        front = back + self.posterior_chamfer_mm + self.distal_face_mm
        return {
            "distal": -z,
            "posterior": back - x,
            "anterior": ((x - self.anterior_x_at_distal_mm) * np.cos(flare)
                         - z * np.sin(flare)),
            "posterior_chamfer": (-(x - (back + self.posterior_chamfer_mm)) * np.sin(c)
                                  - z * np.cos(c)),
            "anterior_chamfer": (x - front) * np.sin(c) - z * np.cos(c),
        }

    def corners(self) -> dict:
        """Where each chamfer's cut-bone corner lies: on the distal plane, at the
        posterior and the anterior cut."""
        return {"posterior_chamfer": self.posterior_x_mm,
                "anterior_chamfer": self.anterior_x_at_distal_mm}


@dataclass(frozen=True)
class ResectionTable:
    """Millimetres of bone off each compartment, and the angle of each cut."""

    rows: tuple
    method: str = "resection_table.v1"
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {"method": self.method, "units": "mm, deg", "rows": list(self.rows),
                "notes": list(self.notes)}


def _row(bone, cut, medial, lateral, *, angle=None, definition, points=None,
         extra=None) -> dict:
    def mm(value):
        return None if value is None else round(float(value), 2)

    def where(point):
        return None if point is None else [round(float(c), 2) for c in point]

    row = {
        "bone": bone, "cut": cut,
        "medial_mm": mm(medial), "lateral_mm": mm(lateral),
        "angle_deg": angle or {},
        "definition": definition,
        "points_mm": {"medial": where((points or {}).get("medial")),
                      "lateral": where((points or {}).get("lateral"))},
    }
    if extra:
        row.update(extra)
    return row


def coronal_angle_deg(normal, axis, medial) -> float:
    """The angle of a cut from perpendicular to ``axis``, in the coronal plane.

    Positive is varus: the cut's normal tilted towards ``medial``, which leaves the
    medial side of the cut lower, on the femur and on the tibia alike.
    """
    normal, axis, medial = unit(normal), unit(axis), unit(medial)
    return float(np.degrees(np.arctan2(np.dot(normal, medial), np.dot(normal, axis))))


def femoral_resection_rows(
    femur: Mesh,
    pose: np.ndarray,
    *,
    medial_sign: float,
    samples: np.ndarray,
    profile: BoxProfile | None = None,
    distal_angles: dict | None = None,
) -> list:
    """The femoral rows, measured on the bone in the planned component's frame.

    ``pose`` is the femoral component's pose (local +X anterior, +Y patient-left, +Z
    proximal, origin on the distal cut). ``medial_sign`` is +1 when medial is local +Y.
    ``samples`` are points over the femur's surface in scan coordinates. Without a
    ``profile`` -- no box design, or a box that could not be placed -- only the distal
    row is measured, and the others say why they are missing.
    """
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] / np.linalg.norm(pose[:3, :3], axis=0)[None, :]
    local = (np.asarray(samples, dtype=float) - pose[:3, 3]) @ rotation
    x, y, z = local.T

    band = (z >= 0.0) & (z <= 2.0)
    if not band.any():
        raise ValueError("The distal cut does not meet the femur.")
    centre_y = float(y[band].min() + y[band].max()) / 2.0
    half = float(np.ptp(y[band])) / 2.0
    within = np.abs(y - centre_y) <= half
    sides = {
        "medial": within & (np.sign(y - centre_y) == np.sign(medial_sign)),
        "lateral": within & (np.sign(y - centre_y) == -np.sign(medial_sign)),
    }

    def to_scan(i):
        return pose[:3, 3] + rotation @ local[i]

    def furthest(distance, region):
        out, where = {}, {}
        for side, in_side in sides.items():
            mask = region & in_side
            if mask.any():
                index = int(np.flatnonzero(mask)[np.argmax(distance[mask])])
                out[side], where[side] = float(distance[index]), to_scan(index)
            else:
                out[side], where[side] = 0.0, None
        return out, where

    distal, distal_where = furthest(-z, z < 0.0)
    rows = [_row(
        "femur", "distal", distal["medial"], distal["lateral"],
        angle=distal_angles or {},
        definition="Bone furthest below the distal cut, perpendicular to it, in each "
                   "half of the distal section.",
        points=distal_where,
    )]
    if profile is None:
        missing = "The femoral cut box was not placed, so this cut is not planned."
        rows += [_row("femur", cut, None, None, definition=missing)
                 for cut in FEMORAL_CUTS[1:]]
        return rows

    s = profile.signed(x, z)
    above = z >= 0.0
    inside_ap = above & (s["posterior"] <= 0.0) & (s["anterior"] <= 0.0)
    regions = {
        "posterior": above & (z <= POSTERIOR_CONDYLE_HEIGHT_MM) & (s["posterior"] > 0.0),
        "anterior": above & (z <= profile.flange_height_mm) & (s["anterior"] > 0.0),
        "posterior_chamfer": inside_ap & (s["posterior_chamfer"] > 0.0),
        "anterior_chamfer": inside_ap & (s["anterior_chamfer"] > 0.0),
    }
    measured = {cut: furthest(s[cut], region) for cut, region in regions.items()}

    # The chamfers' corners: cut bone wherever the condyle fills them.
    corner_x = profile.corners()
    probes = _corner_probes(femur, pose, rotation, corner_x, centre_y, half,
                            medial_sign)
    for cut in ("posterior_chamfer", "anterior_chamfer"):
        cx = corner_x[cut]
        corner_value = float(profile.signed(np.array([cx]), np.array([0.0]))[cut][0])
        values, where = measured[cut]
        for side in ("medial", "lateral"):
            filled = probes[cut][side]
            if filled is not None and corner_value > values[side]:
                values[side] = corner_value
                where[side] = pose[:3, 3] + rotation @ np.array([cx, filled, 0.0])

    # External rotation of the posterior cut from the posterior condylar axis, on the
    # same samples: the most posterior point of each condyle above the distal cut.
    condyles = above & (z <= POSTERIOR_CONDYLE_HEIGHT_MM)
    rotation_deg = None
    backs = {}
    for side, in_side in sides.items():
        mask = condyles & in_side
        if mask.any():
            index = int(np.flatnonzero(mask)[np.argmin(x[mask])])
            backs[side] = (x[index], y[index])
    if len(backs) == 2:
        dx = backs["lateral"][0] - backs["medial"][0]
        dy = abs(backs["lateral"][1] - backs["medial"][1])
        if dy > 1e-6:
            rotation_deg = float(np.degrees(np.arctan2(dx, dy)))

    flare = profile.anterior_flare_deg
    anterior_angles = {"flare_from_component_axis": round(flare, 2)}
    if profile.shaft_slope_deg is not None:
        anterior_angles["from_anterior_cortex"] = round(flare - profile.shaft_slope_deg, 2)
    definitions = {
        "posterior": ("Bone furthest behind the posterior cut, above the distal cut and "
                      "up to 45 mm, in each half.",
                      {} if rotation_deg is None else
                      {"external_rotation_from_posterior_condylar_axis":
                       round(rotation_deg, 2)}),
        "anterior": ("Bone furthest in front of the anterior cut, perpendicular to it, "
                     "from the distal cut up to the flange top.", anterior_angles),
        "posterior_chamfer": ("Bone left by the distal and posterior cuts beyond the "
                              "posterior chamfer, perpendicular to it.",
                              {"from_distal_cut": round(profile.chamfer_deg, 2)}),
        "anterior_chamfer": ("Bone left by the distal and anterior cuts beyond the "
                             "anterior chamfer, perpendicular to it.",
                             {"from_distal_cut": round(profile.chamfer_deg, 2)}),
    }
    for cut in FEMORAL_CUTS[1:]:
        values, where = measured[cut]
        definition, angle = definitions[cut]
        rows.append(_row("femur", cut, values["medial"], values["lateral"],
                         angle=angle, definition=definition, points=where))
    return rows


def _corner_probes(femur, pose, rotation, corner_x, centre_y, half, medial_sign):
    """For each chamfer corner and condyle, a probe position where the corner is inside
    the femur, or ``None`` if it is nowhere inside it.

    Probes run across the middle four fifths of each condyle, just above the distal cut.
    The femur is projected into the component's plane once for all of them.
    """
    footprint = _footprint(femur, pose, rotation)
    offsets = np.linspace(0.1, 0.9, CORNER_PROBES) * half
    found = {}
    for cut, cx in corner_x.items():
        found[cut] = {}
        for side, sign in (("medial", np.sign(medial_sign)),
                           ("lateral", -np.sign(medial_sign))):
            ys = centre_y + sign * offsets
            points = np.column_stack([np.full(len(ys), cx), ys, np.full(len(ys), 0.05)])
            inside = inside_along_z(femur, pose, rotation, points, footprint=footprint)
            found[cut][side] = float(ys[np.argmax(inside)]) if inside.any() else None
    return found


def _footprint(mesh: Mesh, pose, rotation):
    """Every triangle's bounding box in a pose's local XY plane."""
    projected = (mesh.vertices - np.asarray(pose, dtype=float)[:3, 3]) @ rotation[:, :2]
    low = np.minimum(np.minimum(projected[mesh.faces[:, 0]], projected[mesh.faces[:, 1]]),
                     projected[mesh.faces[:, 2]])
    high = np.maximum(np.maximum(projected[mesh.faces[:, 0]], projected[mesh.faces[:, 1]]),
                      projected[mesh.faces[:, 2]])
    return low, high


def inside_along_z(mesh: Mesh, pose, rotation, local_points, *,
                   footprint=None) -> np.ndarray:
    """Whether each point, in a pose's local frame, is inside a closed mesh.

    A ray is cast from each point along local +Z and its crossings with the surface
    counted: an odd count is inside. Only the triangles whose local XY footprint can
    hold a probe are transformed, so a million-triangle femur costs tens of
    milliseconds; ``footprint`` (from one projection) saves repeating that projection
    for several sets of probes.
    """
    pose = np.asarray(pose, dtype=float)
    local_points = np.atleast_2d(np.asarray(local_points, dtype=float))
    low, high = footprint if footprint is not None else _footprint(mesh, pose, rotation)
    p_low, p_high = local_points[:, :2].min(axis=0), local_points[:, :2].max(axis=0)
    near = np.all((high >= p_low) & (low <= p_high), axis=1)
    faces = mesh.faces[near]
    inside = np.zeros(len(local_points), dtype=bool)
    if not len(faces):
        return inside
    triangles = (mesh.vertices[faces] - pose[:3, 3]) @ rotation
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0 = b[:, :2] - a[:, :2]
    v1 = c[:, :2] - a[:, :2]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(det) > 1e-12
    safe = np.where(ok, det, 1.0)
    for k, p in enumerate(local_points):
        # A ray through a shared edge or vertex would be counted once per triangle. The
        # probe is nudged by a micron in an irrational direction, which no mesh edge
        # meets by design.
        v2 = p[:2] + _JITTER_MM - a[:, :2]
        s = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / safe
        t = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / safe
        hit = ok & (s >= 0) & (t >= 0) & (s + t <= 1)
        height = a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2])
        inside[k] = bool(int(np.count_nonzero(hit & (height > p[2]))) % 2)
    return inside


def tibial_resection_rows(
    tibia: Mesh,
    point,
    normal,
    *,
    medial,
    anterior,
    plateau_points: dict,
    axis=None,
    slope_deg: float | None = None,
) -> list:
    """The tibial row, measured on the bone against the planned tibial cut.

    The tibia is the one bone where the furthest point is the wrong answer. Its plateaus
    are dishes between a raised rim and the intercondylar eminence, so the bone furthest
    above the cut in either half is a tubercle or the rim, which no stylus touches. The
    tibial thickness is therefore read where the stylus reads it: at each plateau's
    lowest point on the articular surface (``plateau_points``, the plan's plateau
    landmarks, which the automatic estimate finds on the bone's upward-facing surface).

    The highest bone in each plateau, outside a band either side of the centre as wide
    as the landmark estimator's intercondylar band, is carried beside it, so a reader can
    see how much bone stands above the cut in all.

    ``normal`` points proximally, into the bone the cut removes; ``medial`` and
    ``anterior`` are the tibia's anatomical directions and ``axis`` its mechanical axis,
    for the cut's coronal angle.
    """
    point = np.asarray(point, dtype=float)
    normal = unit(np.asarray(normal, dtype=float))
    m = unit(np.asarray(medial, dtype=float) - np.dot(medial, normal) * normal)
    a = unit(np.asarray(anterior, dtype=float) - np.dot(anterior, normal) * normal)

    # The section's extent is read from the outline the cut leaves, not from vertices
    # near it, so a coarse mesh with no vertex near the plane is measured too.
    outline = slice_mesh(tibia, point, normal).reshape(-1, 3)
    if not len(outline):
        raise ValueError("The tibial cut does not meet the tibia.")
    across_outline = (outline - point) @ m
    along_outline = (outline - point) @ a
    centre = float(across_outline.min() + across_outline.max()) / 2.0
    half = float(np.ptp(across_outline)) / 2.0
    ap_low, ap_high = float(along_outline.min()), float(along_outline.max())

    relative = tibia.vertices - point
    height = relative @ normal
    across = relative @ m
    along = relative @ a
    removed = (height > 0.0) & (along >= ap_low) & (along <= ap_high)
    offset = across - centre
    band = INTERCONDYLAR_HALF_BAND_MM
    sides = {"medial": removed & (offset >= band) & (offset <= half),
             "lateral": removed & (offset <= -band) & (offset >= -half)}
    highest = {}
    for side, mask in sides.items():
        highest[side] = float(height[mask].max()) if mask.any() else 0.0

    at = {side: None if p is None else float(np.dot(np.asarray(p, float) - point, normal))
          for side, p in plateau_points.items()}
    angle = {}
    if axis is not None:
        angle["varus_from_mechanical_axis"] = round(
            coronal_angle_deg(normal, axis, medial), 2)
    if slope_deg is not None:
        angle["posterior_slope"] = round(float(slope_deg), 2)
    return [_row(
        "tibia", "tibial", at.get("medial"), at.get("lateral"), angle=angle,
        definition="Bone above the tibial cut at each plateau's lowest point on the "
                   "articular surface, perpendicular to the cut: what a stylus reads.",
        points=plateau_points,
        extra={"highest_mm": {side: round(value, 2) for side, value in highest.items()},
               "highest_definition": (
                   f"Bone furthest above the cut in each half of the section, "
                   f"leaving out {band:.0f} mm either side of its centre for the "
                   f"intercondylar eminence.")},
    )]
