"""The insert: the thickness that closes the joint, and the parts' geometry around it.

The planned construct has no gap. :func:`~tka_planner.core.planning.plan_alignment`
solves the insert from the cut planes -- as thick as closes the tighter compartment in
extension, under the condyles -- and that is the thickness the parametric insert is made
to. This module does the geometry the planes cannot:

* :func:`dish_geometry` finds a library insert's floor and dish bottom, so the viewer can
  stretch the library part to the solved thickness (:func:`stretch_matrix`) until Fusion
  builds the real one;
* :func:`solve_insert` measures the clearance between the posed femoral component and the
  posed insert. After the stretch it should be zero under the condyles; where it is
  negative the parts run into each other -- typically the femoral component against the
  insert's anterior lip when the two components are out of register -- and the plan says
  so.

Both meshes are sampled over their surfaces rather than by vertex, because CAD exports
are made of large triangles whose vertices say little about the surface between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import unit
from .meshio import Mesh

__all__ = [
    "InsertSolution", "surface_samples", "solve_insert", "dish_geometry",
    "stretch_matrix",
]

CELL_MM = 2.0
SAMPLES_PER_MM2 = 8.0


@dataclass(frozen=True)
class InsertSolution:
    """How far the insert's surface must move to meet the femoral component."""

    change_mm: float
    medial_clearance_mm: float
    lateral_clearance_mm: float
    contact_point: np.ndarray
    thickness_at_contact_mm: float
    method: str = "insert.surface_clearance.v1"
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "change_mm": round(self.change_mm, 3),
            "medial_clearance_mm": round(self.medial_clearance_mm, 3),
            "lateral_clearance_mm": round(self.lateral_clearance_mm, 3),
            "thickness_at_contact_mm": round(self.thickness_at_contact_mm, 3),
            "contact_point_mm": [round(float(v), 3) for v in self.contact_point],
            "diagnostics": self.diagnostics,
        }


def surface_samples(mesh: Mesh, per_mm2: float = SAMPLES_PER_MM2,
                    seed: int = 0) -> np.ndarray:
    """Points spread over a mesh's surface in proportion to area. Seeded, so a plan is
    reproducible."""
    triangles = mesh.vertices[mesh.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    counts = np.maximum(1, np.ceil(areas * per_mm2).astype(int))
    index = np.repeat(np.arange(len(triangles)), counts)
    rng = np.random.default_rng(seed)
    r1 = np.sqrt(rng.random(len(index)))
    r2 = rng.random(len(index))
    t = triangles[index]
    return ((1.0 - r1)[:, None] * t[:, 0] + (r1 * (1.0 - r2))[:, None] * t[:, 1]
            + (r1 * r2)[:, None] * t[:, 2])


def solve_insert(
    femoral_component: Mesh,
    insert: Mesh,
    *,
    normal,
    lateral,
    cell_mm: float = CELL_MM,
) -> InsertSolution:
    """The clearance between a posed femoral component and a posed insert.

    ``normal`` is the tibial cut normal, pointing proximally; ``lateral`` names the
    compartments. A positive ``change_mm`` means the insert must be that much thicker to
    close the tighter compartment; negative, that it already reaches past the femoral
    component and must be thinner.
    """
    normal = unit(np.asarray(normal, dtype=float))
    u = unit(np.asarray(lateral, dtype=float) - np.dot(lateral, normal) * normal)
    v = np.cross(normal, u)

    def envelope(points, highest: bool):
        heights = points @ normal
        cells = np.floor(np.stack([points @ u, points @ v], axis=1) / cell_mm)
        cells = cells.astype(np.int64)
        order = np.lexsort((heights if highest else -heights, cells[:, 1], cells[:, 0]))
        cells, points, heights = cells[order], points[order], heights[order]
        last = np.ones(len(cells), dtype=bool)
        last[:-1] = np.any(cells[1:] != cells[:-1], axis=1)
        return {tuple(c): (h, p) for c, h, p in zip(cells[last], heights[last],
                                                     points[last])}

    insert_points = surface_samples(insert)
    top = envelope(insert_points, highest=True)
    bottom = envelope(surface_samples(femoral_component), highest=False)
    shared = sorted(set(top) & set(bottom))
    if not shared:
        raise ValueError("The femoral component does not lie over the insert.")

    clearance = np.array([bottom[c][0] - top[c][0] for c in shared])
    lateral_offset = np.array([top[c][1] @ u for c in shared])
    centre = float(np.median(insert_points @ u))
    is_lateral = lateral_offset >= centre

    tightest = int(np.argmin(clearance))
    contact = top[shared[tightest]][1]
    insert_floor = float((insert.vertices @ normal).min())

    def side_min(mask):
        return float(clearance[mask].min()) if mask.any() else float("nan")

    change = float(clearance[tightest])
    return InsertSolution(
        change_mm=change,
        medial_clearance_mm=side_min(~is_lateral),
        lateral_clearance_mm=side_min(is_lateral),
        contact_point=np.asarray(contact, dtype=float),
        thickness_at_contact_mm=float(contact @ normal) - insert_floor + change,
        diagnostics={"cell_mm": cell_mm, "overlapping_cells": len(shared)},
    )


def dish_geometry(insert: Mesh) -> tuple[float, float]:
    """The floor and the dish bottom of an insert, in its own CAD coordinates.

    The floor is its lowest point, where it sits on the tray. The dish bottom is the
    lowest point of its articular surface in the middle of each compartment -- away from
    the central spine and the rims -- taking the lower of the two. Their difference is
    the polyethylene thickness under the condyle, which is what an insert's thickness
    means.

    CAD axes for tibial parts: +X patient-left, +Y posterior, +Z proximal.
    """
    # 2 mm cells at 8 samples per mm^2 put about 32 samples of each face in a cell, so a
    # cell whose top face happened to receive none -- and would report the floor as its
    # top -- does not occur in practice.
    points = surface_samples(insert, per_mm2=8.0)
    cells = np.floor(points[:, :2] / 2.0).astype(np.int64)
    order = np.lexsort((points[:, 2], cells[:, 1], cells[:, 0]))
    cells, points = cells[order], points[order]
    last = np.ones(len(cells), dtype=bool)
    last[:-1] = np.any(cells[1:] != cells[:-1], axis=1)
    top = points[last]

    floor = float(insert.vertices[:, 2].min())
    half = float(np.abs(insert.vertices[:, 0]).max())
    y_low, y_high = insert.vertices[:, 1].min(), insert.vertices[:, 1].max()
    y_margin = 0.2 * (y_high - y_low)
    middle_ap = (top[:, 1] > y_low + y_margin) & (top[:, 1] < y_high - y_margin)
    bottoms = []
    for sign in (-1.0, 1.0):
        across = sign * top[:, 0]
        in_dish = middle_ap & (across > 0.2 * half) & (across < 0.8 * half)
        if in_dish.any():
            bottoms.append(float(top[in_dish, 2].min()))
    if not bottoms:
        raise ValueError("Could not find the insert's articular dishes.")
    return floor, min(bottoms)


def stretch_matrix(floor_z: float, factor: float) -> np.ndarray:
    """Scale an insert along its own axis about its floor, leaving the floor on the tray.

    Applied in the insert's CAD coordinates, before its pose, so the footprint and the
    seating are untouched and only the height changes.
    """
    matrix = np.eye(4)
    matrix[2, 2] = factor
    matrix[2, 3] = floor_z * (1.0 - factor)
    return matrix
