"""Distances between large meshes, for comparing legacy outputs.

``tests/surface_distance.py`` measures point-to-surface distance by brute force, every
point against every triangle. That is right for the implant parts (tens of thousands of
triangles) and far too slow for a bone (a million). Here the triangles are bucketed on a
uniform grid, so each point is compared only with the triangles near it, and the answer
is still exact: a triangle that comes within one cell of a point must overlap one of the
27 cells around it. Points with nothing that close are retried on a coarser grid.

numpy only; nothing here is needed by the planner, only by the paper's experiments.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "closest_point_pairwise",
    "nearest_vertex_distance",
    "SurfaceIndex",
    "surface_distances",
]

_OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                     for k in (-1, 0, 1)], dtype=np.int64)


def closest_point_pairwise(p: np.ndarray, a: np.ndarray, b: np.ndarray,
                           c: np.ndarray) -> np.ndarray:
    """Closest point on triangle ``(a[i], b[i], c[i])`` to ``p[i]``, for every ``i``.

    The same barycentric-region method as ``tests/surface_distance.py`` (Ericson,
    section 5.1.5), but pairwise rather than all-against-all.
    """
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denominator = va + vb + vc
    safe = np.where(np.abs(denominator) < 1e-300, 1.0, denominator)
    out = a + ab * (vb / safe)[:, None] + ac * (vc / safe)[:, None]

    def put(mask, value):
        out[mask] = value[mask] if value.ndim == 2 else value

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    put((d6 >= 0) & (d5 <= d6), c)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~((d1 <= 0) & (d2 <= 0)) \
        & ~((d3 >= 0) & (d4 <= d3))
    t = d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3)
    put(edge_ab, a + ab * t[:, None])
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~((d1 <= 0) & (d2 <= 0)) \
        & ~((d6 >= 0) & (d5 <= d6))
    t = d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6)
    put(edge_ac, a + ac * t[:, None])
    span = (d4 - d3) + (d5 - d6)
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) \
        & ~((d3 >= 0) & (d4 <= d3)) & ~((d6 >= 0) & (d5 <= d6))
    t = (d4 - d3) / np.where(span == 0, 1.0, span)
    put(edge_bc, b + (c - b) * t[:, None])
    return out


def _pack(cells: np.ndarray) -> np.ndarray:
    # 21 bits per axis is ample: a 2 m scene at 1 micrometre would still fit.
    shifted = cells + (1 << 20)
    return (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]


def _ranges(sorted_keys: np.ndarray, query_keys: np.ndarray):
    """For each query key, the slice of ``sorted_keys`` equal to it, as pair arrays."""
    left = np.searchsorted(sorted_keys, query_keys, side="left")
    right = np.searchsorted(sorted_keys, query_keys, side="right")
    counts = right - left
    owner = np.repeat(np.arange(len(query_keys)), counts)
    start = np.repeat(left - (np.cumsum(counts) - counts), counts)
    return owner, np.arange(int(counts.sum())) + start


def nearest_vertex_distance(points: np.ndarray, targets: np.ndarray, *,
                            cell: float) -> np.ndarray:
    """Distance from each point to the nearest of ``targets``.

    Exact whenever that distance is at most ``cell``; ``inf`` when no target lies in
    the 27 cells around the point. Used to show that a mesh mapped back through a
    recovered transform lands on its source vertex for vertex.
    """
    points = np.asarray(points, dtype=float)
    targets = np.asarray(targets, dtype=float)
    origin = np.minimum(points.min(axis=0), targets.min(axis=0)) - 2 * cell
    tcell = np.floor((targets - origin) / cell).astype(np.int64)
    keys = _pack(tcell)
    order = np.argsort(keys, kind="stable")
    skeys = keys[order]
    qcell = np.floor((points - origin) / cell).astype(np.int64)
    best = np.full(len(points), np.inf)
    for offset in _OFFSETS:
        owner, index = _ranges(skeys, _pack(qcell + offset))
        if len(owner):
            d = np.linalg.norm(points[owner] - targets[order[index]], axis=1)
            np.minimum.at(best, owner, d)
    return best


class SurfaceIndex:
    """Triangles bucketed on a grid, for exact point-to-surface distance."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, *,
                 cell: float | None = None, big_cells: int = 4):
        corners = np.asarray(vertices, dtype=float)[np.asarray(faces)]
        self.a, self.b, self.c = corners[:, 0], corners[:, 1], corners[:, 2]
        if cell is None:
            edges = np.linalg.norm(self.b - self.a, axis=1)
            cell = max(4.0 * float(np.median(edges)), 1e-9)
        self.cell = cell
        self.origin = corners.reshape(-1, 3).min(axis=0) - 2 * cell
        lo = np.floor((corners.min(axis=1) - self.origin) / cell).astype(np.int64)
        hi = np.floor((corners.max(axis=1) - self.origin) / cell).astype(np.int64)
        span = (hi - lo).max(axis=1)
        # A triangle much larger than a cell (a cutting block's flat face) would land in
        # thousands of cells; those few are checked against every point instead.
        self.big = np.nonzero(span >= big_cells)[0]
        small = np.nonzero(span < big_cells)[0]
        keys, owners = [], []
        for di in range(big_cells):
            for dj in range(big_cells):
                for dk in range(big_cells):
                    step = np.array([di, dj, dk])
                    cells = lo[small] + step
                    inside = np.all(cells <= hi[small], axis=1)
                    keys.append(_pack(cells[inside]))
                    owners.append(small[inside])
        keys = np.concatenate(keys)
        owners = np.concatenate(owners)
        order = np.argsort(keys, kind="stable")
        self.keys, self.triangles = keys[order], owners[order]

    def _candidates(self, points):
        qcell = np.floor((points - self.origin) / self.cell).astype(np.int64)
        owner, tri = [], []
        for offset in _OFFSETS:
            o, index = _ranges(self.keys, _pack(qcell + offset))
            owner.append(o)
            tri.append(self.triangles[index])
        if len(self.big):
            owner.append(np.repeat(np.arange(len(points)), len(self.big)))
            tri.append(np.tile(self.big, len(points)))
        return np.concatenate(owner), np.concatenate(tri)

    def distance(self, points: np.ndarray, *, chunk: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the surface, and whether each value is exact.

        A value is exact when it is at most one cell; beyond that a closer triangle
        could sit outside the 27 cells searched, so the value is only an upper bound
        (and ``inf`` if nothing was near at all).
        """
        points = np.asarray(points, dtype=float)
        out = np.full(len(points), np.inf)
        for start in range(0, len(points), chunk):
            block = points[start:start + chunk]
            owner, tri = self._candidates(block)
            if len(owner) == 0:
                continue
            q = closest_point_pairwise(block[owner], self.a[tri], self.b[tri], self.c[tri])
            d = np.linalg.norm(q - block[owner], axis=1)
            best = np.full(len(block), np.inf)
            np.minimum.at(best, owner, d)
            out[start:start + chunk] = best
        return out, out <= self.cell


def surface_distances(a, b, *, sample: int = 20000, seed: int = 0) -> dict:
    """Symmetric distances between two meshes (anything with vertices and faces), in
    their own units.

    Each mesh's vertices are sampled (up to ``sample``) and measured to the other's
    surface; a value beyond the grid's reach is resolved on a grid 8x coarser and, if
    still out of reach, kept as an upper bound and counted in ``n_bounded``.
    """
    rng = np.random.default_rng(seed)

    def directed(src, dst):
        points = np.asarray(src.vertices, dtype=float)
        if len(points) > sample:
            points = points[rng.choice(len(points), sample, replace=False)]
        index = SurfaceIndex(dst.vertices, dst.faces)
        d, exact = index.distance(points)
        bounded = 0
        if not exact.all():
            coarse = SurfaceIndex(dst.vertices, dst.faces, cell=index.cell * 8)
            d2, exact2 = coarse.distance(points[~exact])
            d[~exact] = np.minimum(d[~exact], d2)
            far = ~exact2
            if far.any():
                # Nothing within eight cells: fall back to the nearest vertex, an upper
                # bound on the surface distance, and say how many such points there were.
                idx = np.nonzero(~exact)[0][far]
                verts = np.asarray(dst.vertices, dtype=float)
                for i in idx:
                    d[i] = min(d[i], float(np.linalg.norm(verts - points[i], axis=1).min()))
                bounded = int(far.sum())
        return d, bounded

    ab, bounded_ab = directed(a, b)
    ba, bounded_ba = directed(b, a)
    both = np.concatenate([ab, ba])
    return {
        "hausdorff": float(max(ab.max(), ba.max())),
        "mean": float(0.5 * (ab.mean() + ba.mean())),
        "p95": float(np.percentile(both, 95)),
        "a_to_b_max": float(ab.max()),
        "b_to_a_max": float(ba.max()),
        "n_samples": int(len(ab) + len(ba)),
        "n_bounded": bounded_ab + bounded_ba,
    }
