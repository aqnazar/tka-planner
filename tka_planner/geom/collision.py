"""Where two posed solids overlap, through ``manifold3d``.

The flexion search asks one question many times: at this angle, do these two parts
occupy the same space, and if so how much, how deep and where? An exact boolean
intersection answers all three at once -- the overlap is itself a solid, with a volume
and vertices -- so it is used rather than sampled distances, which would need a
tolerance of their own to stand in for "inside".

The bones are the cost. A segmented femur is about a million triangles, so each bone is
cropped once to the region a flexing knee can reach, cut by the plan's cutting block, and
simplified by a stated tolerance before any angle is tried. Booleans on what remains take
milliseconds. The tolerance and the crop are recorded with every result, so a reader can
see what geometry the contact was found on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tka_planner.core.meshio import Mesh

from .kernels.manifold_kernel import ManifoldKernel

__all__ = ["Solid", "Overlap", "overlap"]


@dataclass(frozen=True)
class Overlap:
    """The part two solids share: its volume and its surface (world mm)."""

    volume_mm3: float
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return self.vertices.mean(axis=0)

    def thickness_mm(self, axis, cell_mm: float = 0.5) -> float:
        """How deep the overlap is along ``axis``: the largest extent along it of any
        column of the overlap, ``cell_mm`` wide.

        A condyle sunk into a dish leaves a thin, curved lens whose overall extent along
        the insert's axis is much more than its thickness, because the lens follows the
        slope of the dish. Measured column by column it is the penetration depth. The
        surface is sampled (seeded) rather than read at its vertices, because a boolean's
        triangles are large and its vertices say little about the surface between them.
        On a surface sloping at angle ``a`` a column adds up to ``cell_mm * tan(a)``, so
        the cell is kept small and the sampling dense (32 per mm^2, about eight samples
        per cell on each face of the lens).
        """
        from tka_planner.core.geometry import unit
        from tka_planner.core.insert import surface_samples

        axis = unit(np.asarray(axis, dtype=float))
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array(
            [0.0, 1.0, 0.0])
        u = unit(np.cross(axis, helper))
        v = np.cross(axis, u)
        points = surface_samples(Mesh(vertices=self.vertices, faces=self.faces),
                                 per_mm2=32.0)
        cells = np.floor(np.stack([points @ u, points @ v], axis=1) / cell_mm)
        heights = points @ axis
        _, index = np.unique(cells.astype(np.int64), axis=0, return_inverse=True)
        index = index.ravel()
        top = np.full(index.max() + 1, -np.inf)
        bottom = np.full(index.max() + 1, np.inf)
        np.maximum.at(top, index, heights)
        np.minimum.at(bottom, index, heights)
        return float((top - bottom).max())


class Solid:
    """A closed solid that can be cropped, cut, simplified, moved and intersected."""

    def __init__(self, manifold):
        self._m = manifold

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "Solid":
        """Repair if needed (and only as :mod:`tka_planner.geom.repair` records), then
        hand the mesh to manifold3d, which refuses anything that is not a closed
        surface."""
        clean, _ = ManifoldKernel._prepare(mesh, "solid")
        return cls(ManifoldKernel._to_manifold(clean, "solid"))

    @property
    def n_triangles(self) -> int:
        return int(self._m.num_tri())

    @property
    def volume_mm3(self) -> float:
        return float(self._m.volume())

    def is_empty(self) -> bool:
        return bool(self._m.is_empty())

    def transformed(self, matrix) -> "Solid":
        matrix = np.asarray(matrix, dtype=float)
        return Solid(self._m.transform(matrix[:3, :].tolist()))

    def translated(self, offset) -> "Solid":
        return Solid(self._m.translate([float(v) for v in offset]))

    def cropped(self, frame: np.ndarray, low, high) -> "Solid":
        """Keep only what lies inside a box given in a frame's own coordinates.

        ``frame`` is a 4x4 frame-to-world transform; ``low`` and ``high`` are the box's
        corners in that frame (mm).
        """
        import manifold3d

        low = np.asarray(low, dtype=float)
        size = np.asarray(high, dtype=float) - low
        box = manifold3d.Manifold.cube(size.tolist()).translate(low.tolist())
        box = box.transform(np.asarray(frame, dtype=float)[:3, :].tolist())
        return Solid(self._m ^ box)

    def minus(self, other: "Solid") -> "Solid":
        return Solid(self._m - other._m)

    def simplified(self, tolerance_mm: float) -> "Solid":
        """Fewer triangles, the surface moving by no more than ``tolerance_mm``."""
        if tolerance_mm <= 0.0:
            return self
        return Solid(self._m.simplify(float(tolerance_mm)))

    def min_gap(self, other: "Solid", search_mm: float) -> float | None:
        """The smallest distance between the two surfaces, or ``None`` when it is more
        than ``search_mm`` (0 when they touch or overlap)."""
        gap = float(self._m.min_gap(other._m, float(search_mm)))
        return None if gap >= search_mm else gap

    def to_mesh(self) -> Mesh:
        return ManifoldKernel._from_manifold(self._m, Mesh(
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)))


def overlap(a: Solid, b: Solid) -> Overlap | None:
    """The overlap of two solids, or ``None`` if they share no volume."""
    shared = a._m ^ b._m
    if shared.is_empty():
        return None
    volume = float(shared.volume())
    if volume <= 0.0:
        return None
    surface = shared.to_mesh64()
    vertices = np.array(surface.vert_properties[:, :3], dtype=float, copy=True)
    faces = np.array(surface.tri_verts, dtype=np.int64, copy=True)
    return Overlap(volume_mm3=volume, vertices=vertices, faces=faces)
