"""How well a component fits the bone it sits on.

Fit is judged where the component meets the bone: on the resection surface. At each cut
two outlines are compared in the plane of that cut:

* the **bone section**, the outline the saw leaves, found by slicing the bone mesh with
  the cut plane;
* the **component footprint**, found by slicing the posed component just off the cut on
  its own side -- through the tray plate above the tibial cut, and through the distal
  plate of the femoral component below the distal femoral cut -- so the outline is the
  surface that actually seats on the bone.

Both are filled on a fine grid in the plane, and the comparison gives the coverage of the
cut surface, and in each anatomical quadrant the largest **overhang** (component beyond
the bone edge) and **underhang** (bone left uncovered beyond the component edge), each
as a distance in millimetres.

The grid is anatomical: its axes are the bone's lateral and anterior directions laid
into the cut plane, so "anteromedial" means the same thing on a left and a right knee.

The overhang threshold is a convention, not a validated limit. Tibial overhang of more
than about 2 mm is associated with pain in the literature, and it is used here only to
raise a flag; the distances themselves are always reported.

The same grid also gives the fit as a map (:class:`FitField`): every cell of the cut
carries a signed distance -- how far uncovered bone lies beyond the component's edge, or
how far the component reaches beyond the bone's edge -- so the colour map a viewer draws
is the very field the maxima in the table were read from, not a second computation that
could disagree with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import unit
from .meshio import Mesh

__all__ = ["SectionFit", "FitField", "section_fit", "slice_mesh", "fill_section"]

GRID_MM = 0.25
SECTION_OFFSET_MM = 1.0
OVERHANG_FLAG_MM = 2.0
ZONES = ("anteromedial", "anterolateral", "posteromedial", "posterolateral")


@dataclass(frozen=True, eq=False)
class FitField:
    """The fit on one cut as a signed distance at every cell of the section grid.

    ``signed_mm`` is ``(len(ys), len(xs))``. Its sign says which outline is short:

    * **positive** -- bone the component leaves uncovered (underhang), valued at the
      distance to the component's edge;
    * **negative** -- component beyond the bone (overhang), valued at the distance to
      the bone's edge;
    * **zero** -- component seated on bone;
    * **NaN** -- neither bone nor component.

    The grid lies in the cut plane: a cell at ``(xs[i], ys[j])`` is the point
    ``origin + xs[i] * u + ys[j] * v``, with ``u`` lateral and ``v`` anterior laid into
    the plane. ``component_side`` is which side of the plane the component sits on
    (+1 along ``normal``), so a drawing of the map can be lifted off the bone towards
    the component rather than buried in the cut.
    """

    cut: str
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray
    component_side: float
    xs: np.ndarray
    ys: np.ndarray
    signed_mm: np.ndarray

    def points(self, rows=None, columns=None) -> np.ndarray:
        """The 3D position of every cell, or of the given rows and columns, in mm."""
        xs = self.xs if columns is None else self.xs[columns]
        ys = self.ys if rows is None else self.ys[rows]
        gx, gy = np.meshgrid(xs, ys)
        return (self.origin + gx[..., None] * self.u + gy[..., None] * self.v)


@dataclass(frozen=True)
class SectionFit:
    """The fit of one component on one cut."""

    cut: str
    bone_area_mm2: float
    component_area_mm2: float
    coverage_fraction: float
    component_on_bone_fraction: float
    overhang_area_mm2: float
    max_overhang_mm: float
    max_underhang_mm: float
    bone_extent_mm: dict
    component_extent_mm: dict
    zones: dict
    flags: tuple = ()
    method: str = "fit.section.v1"
    diagnostics: dict = field(default_factory=dict)
    # Where each maximum was found, in scan millimetres, so a finding can be shown where
    # it is rather than only named. ``None`` where there is none.
    extreme_points_mm: dict = field(default_factory=dict)
    # The whole grid, kept only when asked for; it is not part of the record.
    grid: FitField | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "cut": self.cut,
            "method": self.method,
            "bone_area_mm2": round(self.bone_area_mm2, 1),
            "component_area_mm2": round(self.component_area_mm2, 1),
            "coverage_fraction": round(self.coverage_fraction, 4),
            "component_on_bone_fraction": round(self.component_on_bone_fraction, 4),
            "overhang_area_mm2": round(self.overhang_area_mm2, 1),
            "max_overhang_mm": round(self.max_overhang_mm, 2),
            "max_underhang_mm": round(self.max_underhang_mm, 2),
            "bone_extent_mm": {k: round(v, 2) for k, v in self.bone_extent_mm.items()},
            "component_extent_mm": {
                k: round(v, 2) for k, v in self.component_extent_mm.items()
            },
            "zones": {
                zone: {k: round(v, 2) for k, v in values.items()}
                for zone, values in self.zones.items()
            },
            "flags": list(self.flags),
            "extreme_points_mm": {
                name: None if point is None else [round(float(c), 2) for c in point]
                for name, point in self.extreme_points_mm.items()
            },
            "diagnostics": self.diagnostics,
        }


def slice_mesh(mesh: Mesh, point, normal) -> np.ndarray:
    """The segments where a plane cuts a mesh, as an ``(N, 2, 3)`` array.

    Works on a triangle soup, so it needs no welded topology. A vertex lying exactly on
    the plane would be counted by both triangles sharing it, so the plane is nudged by a
    distance far below the grid before slicing.
    """
    normal = unit(np.asarray(normal, dtype=float))
    offset = float(np.dot(np.asarray(point, dtype=float), normal)) + 1e-7
    heights = mesh.vertices @ normal - offset
    corner_heights = heights[mesh.faces]
    crosses = (corner_heights.min(axis=1) < 0) & (corner_heights.max(axis=1) > 0)
    if not crosses.any():
        return np.empty((0, 2, 3))

    faces = mesh.faces[crosses]
    d = corner_heights[crosses]
    v = mesh.vertices[faces]
    points = np.full((len(faces), 3, 3), np.nan)
    valid = np.zeros((len(faces), 3), dtype=bool)
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        edge = d[:, i] * d[:, j] < 0
        t = np.where(edge, d[:, i] / np.where(edge, d[:, i] - d[:, j], 1.0), 0.0)
        points[:, k] = v[:, i] + t[:, None] * (v[:, j] - v[:, i])
        valid[:, k] = edge
    keep = valid.sum(axis=1) == 2
    order = np.argsort(~valid[keep], axis=1, kind="stable")[:, :2]
    rows = np.arange(int(keep.sum()))[:, None]
    return points[keep][rows, order]


def fill_section(segments_2d: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fill a set of closed 2D contours on a grid, by the even-odd rule.

    ``segments_2d`` is ``(N, 2, 2)``. Returns a boolean ``(len(ys), len(xs))`` mask. A
    hollow section -- a medullary canal, the notch between two condyles -- comes out
    hollow, which is what a scanline even-odd fill gives for free.
    """
    mask = np.zeros((len(ys), len(xs)), dtype=bool)
    if len(segments_2d) == 0:
        return mask
    a, b = segments_2d[:, 0], segments_2d[:, 1]
    low_y, high_y = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    dy = b[:, 1] - a[:, 1]
    for row, y in enumerate(ys):
        hit = (low_y <= y) & (y < high_y)
        if not hit.any():
            continue
        crossings = np.sort(
            a[hit, 0] + (y - a[hit, 1]) * (b[hit, 0] - a[hit, 0]) / dy[hit]
        )
        for start, end in zip(crossings[0::2], crossings[1::2]):
            mask[row] |= (xs >= start) & (xs < end)
    return mask


def section_fit(
    *,
    cut: str,
    bone: Mesh,
    component: Mesh,
    point,
    normal,
    component_side: float,
    lateral,
    anterior,
    grid_mm: float = GRID_MM,
    offset_mm: float = SECTION_OFFSET_MM,
    keep_grid: bool = False,
) -> SectionFit:
    """Compare the bone's section at a cut with the component's footprint on it.

    ``component`` must already be posed in the bone's coordinates. ``component_side`` is
    +1 when the component lies on the side the normal points to (the tibial tray above
    the tibial cut) and -1 when it lies on the other (the femoral component below the
    distal femoral cut). ``lateral`` and ``anterior`` are the bone's anatomical
    directions; they are laid into the cut plane to name the quadrants.

    ``keep_grid`` keeps the signed distance at every cell as :attr:`SectionFit.grid`,
    for drawing the fit as a colour map. It costs nothing extra: the distances are the
    ones the maxima are read from.
    """
    point = np.asarray(point, dtype=float)
    normal = unit(np.asarray(normal, dtype=float))
    u = unit(np.asarray(lateral, dtype=float) - np.dot(lateral, normal) * normal)
    v = unit(np.cross(normal, u))
    if float(np.dot(v, anterior)) < 0:
        v = -v

    def to_plane(segments):
        rel = segments - point
        return np.stack([rel @ u, rel @ v], axis=-1)

    bone_segments = to_plane(slice_mesh(bone, point, normal))
    footprint_point = point + component_side * offset_mm * normal
    component_segments = to_plane(slice_mesh(component, footprint_point, normal))
    if not len(bone_segments) or not len(component_segments):
        raise ValueError(
            f"The {cut} cut does not pass through both the bone and the component."
        )

    everything = np.concatenate([bone_segments, component_segments]).reshape(-1, 2)
    low = everything.min(axis=0) - 2 * grid_mm
    high = everything.max(axis=0) + 2 * grid_mm
    # Cell centres sit off any round coordinate, so a scanline never runs exactly
    # through a vertex.
    xs = np.arange(low[0], high[0], grid_mm) + grid_mm * 0.5013
    ys = np.arange(low[1], high[1], grid_mm) + grid_mm * 0.4987
    bone_mask = fill_section(bone_segments, xs, ys)
    component_mask = fill_section(component_segments, xs, ys)

    cell = grid_mm * grid_mm
    bone_area = float(bone_mask.sum()) * cell
    component_area = float(component_mask.sum()) * cell
    covered = bone_mask & component_mask
    overhang = component_mask & ~bone_mask
    underhang = bone_mask & ~component_mask

    gx, gy = np.meshgrid(xs, ys)
    centre_x = float(gx[component_mask].mean())
    centre_y = float(gy[component_mask].mean())
    over_distance = _distance_to(overhang, bone_mask, gx, gy)
    under_distance = _distance_to(underhang, component_mask, gx, gy)

    # Quadrants about the component's own centre. u points laterally, so medial is -u.
    lateral_half = gx >= centre_x
    anterior_half = gy >= centre_y
    zone_masks = {
        "anteromedial": anterior_half & ~lateral_half,
        "anterolateral": anterior_half & lateral_half,
        "posteromedial": ~anterior_half & ~lateral_half,
        "posterolateral": ~anterior_half & lateral_half,
    }
    zones = {}
    flags = []
    extremes = {
        "max_overhang": _where_max(over_distance, overhang, gx, gy, point, u, v),
        "max_underhang": _where_max(under_distance, underhang, gx, gy, point, u, v),
    }
    for zone, in_zone in zone_masks.items():
        over = _max_in(over_distance, overhang & in_zone)
        under = _max_in(under_distance, underhang & in_zone)
        zones[zone] = {"max_overhang_mm": over, "max_underhang_mm": under}
        if over > OVERHANG_FLAG_MM:
            flags.append(f"OVERHANG_{zone.upper()}")
            extremes[f"overhang_{zone}"] = _where_max(
                over_distance, overhang & in_zone, gx, gy, point, u, v)

    grid = None
    if keep_grid:
        signed = np.full(bone_mask.shape, np.nan)
        signed[covered] = 0.0
        signed[underhang] = under_distance[underhang]
        signed[overhang] = -over_distance[overhang]
        grid = FitField(cut=cut, origin=point, u=u, v=v, normal=normal,
                        component_side=float(np.sign(component_side) or 1.0),
                        xs=xs, ys=ys, signed_mm=signed)

    return SectionFit(
        cut=cut,
        bone_area_mm2=bone_area,
        component_area_mm2=component_area,
        coverage_fraction=float(covered.sum()) / max(int(bone_mask.sum()), 1),
        component_on_bone_fraction=(
            float(covered.sum()) / max(int(component_mask.sum()), 1)
        ),
        overhang_area_mm2=float(overhang.sum()) * cell,
        max_overhang_mm=_max_in(over_distance, overhang),
        max_underhang_mm=_max_in(under_distance, underhang),
        bone_extent_mm=_extent(bone_mask, gx, gy),
        component_extent_mm=_extent(component_mask, gx, gy),
        zones=zones,
        flags=tuple(flags),
        diagnostics={
            "grid_mm": grid_mm,
            "footprint_offset_mm": offset_mm,
            "overhang_flag_mm": OVERHANG_FLAG_MM,
            "n_bone_segments": int(len(bone_segments)),
            "n_component_segments": int(len(component_segments)),
        },
        extreme_points_mm=extremes,
        grid=grid,
    )


def _where_max(distance, mask, gx, gy, point, u, v):
    """The scan position of the cell where ``distance`` peaks within ``mask``."""
    if not mask.any():
        return None
    masked = np.where(mask, distance, -np.inf)
    row, column = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return point + gx[row, column] * u + gy[row, column] * v


def _extent(mask, gx, gy) -> dict:
    if not mask.any():
        return {"ml": 0.0, "ap": 0.0}
    return {"ml": float(np.ptp(gx[mask])), "ap": float(np.ptp(gy[mask]))}


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Cells of a mask with at least one of their four neighbours outside it."""
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def _distance_to(cells: np.ndarray, target: np.ndarray, gx, gy) -> np.ndarray:
    """For each cell in ``cells``, the distance to the nearest cell of ``target``."""
    distance = np.zeros(cells.shape)
    if not cells.any() or not target.any():
        return distance
    edge = _boundary(target)
    targets = np.column_stack([gx[edge], gy[edge]])
    points = np.column_stack([gx[cells], gy[cells]])
    # Squared distances as |p|^2 + |t|^2 - 2 p.t, so the pairwise work is one matrix
    # product per chunk rather than two broadcast differences: ten times faster on a
    # real cut, and exact to far below the grid at these coordinates.
    centre = targets.mean(axis=0)
    targets, points = targets - centre, points - centre
    target_norms = np.einsum("ij,ij->i", targets, targets)
    nearest = np.empty(len(points))
    for start in range(0, len(points), 4096):
        block = points[start:start + 4096]
        squared = (np.einsum("ij,ij->i", block, block)[:, None] + target_norms[None, :]
                   - 2.0 * block @ targets.T)
        nearest[start:start + 4096] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    distance[cells] = nearest
    return distance


def _max_in(distance: np.ndarray, mask: np.ndarray) -> float:
    return float(distance[mask].max()) if mask.any() else 0.0
