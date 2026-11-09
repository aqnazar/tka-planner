"""Point-to-surface distance on bone-sized meshes, exact, through a uniform grid.

``tests/surface_distance.py`` measures every point against every triangle, which is
right for the agreement gate's few thousand triangles and impossible for a resected
bone of a million. This measures the same closest-point distance and only changes
which triangles a point is measured against.

Each triangle is registered in every grid cell its bounding box touches. A point is
measured against the triangles registered in the block of cells around its own, out
to ``reach`` cells; any triangle within ``reach`` cell widths of the point is in that
block, so a nearest distance no larger than ``reach * cell`` is exact. A point whose
nearest candidate is further than that is searched again over a wider block. Nothing
is approximated: the answer is the brute-force answer, found faster.

Two resections of the same bone share almost all of their surface: the solvers differ
only where they cut. :func:`hausdorff` therefore first pairs the samples the other mesh
shares -- a vertex it also has, a triangle it also has -- and takes the distance to the
shared point, an upper bound on the distance to the surface that is zero when the two
agree to the bit, and measures only the rest.

Used by E8 to compare manifold3d's resections with Blender's. The two solvers
triangulate the same surface differently, so the comparison is surface to surface,
never vertex to vertex (see the agreement test).
"""

from __future__ import annotations

import numpy as np

__all__ = ["SurfaceGrid", "closest_points", "point_to_surface", "hausdorff",
           "surface_samples"]

# Above this many (cell, triangle) registrations the cell is doubled, so a mesh with a
# few very long triangles -- a boolean's cut face is full of them -- cannot exhaust
# memory.
_MAX_ENTRIES = 30_000_000
# Point-triangle pairs evaluated at once: about 150 MB of temporaries.
_PAIRS_PER_CHUNK = 400_000


def closest_points(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Closest point on triangle ``i`` to point ``i``, for aligned arrays.

    ``points`` is ``(N, 3)`` and ``corners`` ``(N, 3, 3)``. The barycentric-region
    method of Ericson (*Real-Time Collision Detection*, section 5.1.5), as in
    ``tests/surface_distance.py``, but pairwise rather than all-against-all.
    """
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c

    def dot(u, v):
        return np.einsum("nk,nk->n", u, v)

    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denominator = va + vb + vc
    safe = np.where(np.abs(denominator) < 1e-300, 1.0, denominator)
    out = a + ab * (vb / safe)[:, None] + ac * (vc / safe)[:, None]

    def pick(condition, value):
        out[condition] = value[condition]

    # Vertex regions, then edge regions, each overwriting the last: the same order of
    # precedence as the all-against-all routine.
    pick((d1 <= 0) & (d2 <= 0), a)
    pick((d3 >= 0) & (d4 <= d3), b)
    pick((d6 >= 0) & (d5 <= d6), c)
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    pick(edge_ab, a + ab * (d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3))[:, None])
    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    pick(edge_ac, a + ac * (d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6))[:, None])
    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    span = (d4 - d3) + (d5 - d6)
    pick(edge_bc, b + (c - b) * ((d4 - d3) / np.where(span == 0, 1.0, span))[:, None])
    return out


class SurfaceGrid:
    """A mesh's triangles, bucketed by grid cells, at coarser levels on demand.

    A point near the surface is settled on the finest grid. One that is not -- a cut
    face one solver made and the other did not -- is passed to a grid four times
    coarser, and so on, so its search widens geometrically rather than one ring of
    cells at a time.
    """

    def __init__(self, mesh, *, cell_mm: float | None = None):
        corners = np.asarray(mesh.vertices, dtype=float)[mesh.faces]
        self.corners = corners
        self._low, self._high = corners.min(axis=1), corners.max(axis=1)
        if cell_mm is None and len(corners):
            # About two triangle widths: a few triangles per cell.
            cell_mm = 2.0 * float(np.median((self._high - self._low).max(axis=1)))
        self._base = max(float(cell_mm or 1.0), 1e-3)
        self._levels: list = []

    def _level(self, index: int) -> "_Level":
        while len(self._levels) <= index:
            cell = (self._levels[-1].cell * 4.0 if self._levels else self._base)
            self._levels.append(_Level(self._low, self._high, cell))
        return self._levels[index]

    def distances(self, points: np.ndarray) -> np.ndarray:
        """Exact distance from each point to the nearest point on the surface."""
        points = np.asarray(points, dtype=float)
        out = np.full(len(points), np.inf)
        if len(points) == 0 or len(self.corners) == 0:
            return out
        span = np.ptp(np.vstack([points, self._low, self._high]), axis=0).max()
        pending = np.arange(len(points))
        level = 0
        while len(pending):
            grid = self._level(level)
            if grid.cell > span:
                # One cell spans everything: what is left is measured against all.
                out[pending] = _brute(points[pending], self.corners)
                break
            best = grid.search(points[pending], self.corners)
            # Exact once a point's nearest lies inside the block that was searched.
            done = best <= grid.cell
            out[pending[done]] = best[done]
            pending = pending[~done]
            level += 1
        return out


class _Level:
    """One grid: each triangle registered in every cell its bounding box touches."""

    def __init__(self, low: np.ndarray, high: np.ndarray, cell_mm: float):
        self.origin = low.min(axis=0) - cell_mm
        while True:
            first = np.floor((low - self.origin) / cell_mm).astype(np.int64)
            last = np.floor((high - self.origin) / cell_mm).astype(np.int64)
            span = last - first + 1
            per_triangle = span.prod(axis=1)
            if per_triangle.sum() <= _MAX_ENTRIES:
                break
            cell_mm *= 2.0
        self.cell = float(cell_mm)

        triangle = np.repeat(np.arange(len(low)), per_triangle)
        local = np.arange(len(triangle)) - np.repeat(
            np.cumsum(per_triangle) - per_triangle, per_triangle)
        nx = np.repeat(span[:, 0], per_triangle)
        ny = np.repeat(span[:, 1], per_triangle)
        cells = np.repeat(first, per_triangle, axis=0) + np.stack(
            [local % nx, (local // nx) % ny, local // (nx * ny)], axis=1)

        keys = _key(cells)
        order = np.argsort(keys, kind="stable")
        self._triangles = triangle[order]
        self._keys, self._starts = np.unique(keys[order], return_index=True)
        self._ends = np.append(self._starts[1:], len(order))

    def _lookup(self, cells: np.ndarray):
        """Start and end of each cell's triangle list; empty where none."""
        keys = _key(cells)
        where = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        hit = self._keys[where] == keys
        start = np.where(hit, self._starts[where], 0)
        end = np.where(hit, self._ends[where], 0)
        return start, end

    def search(self, points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Nearest distance over the triangles in the 27 cells around each point."""
        home = np.floor((points - self.origin) / self.cell).astype(np.int64)
        steps = np.arange(-1, 2)
        block = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"),
                         axis=-1).reshape(-1, 3)
        best = np.full(len(points), np.inf)
        for offset in block:
            start, end = self._lookup(home + offset)
            count = end - start
            have = np.flatnonzero(count)
            if len(have) == 0:
                continue
            # Chunk by pairs, so the temporaries stay inside the memory budget.
            total = np.cumsum(count[have])
            cuts = np.searchsorted(total, np.arange(_PAIRS_PER_CHUNK, total[-1],
                                                    _PAIRS_PER_CHUNK)) + 1
            for chunk in np.split(have, np.unique(cuts)):
                if len(chunk) == 0:
                    continue
                n = count[chunk]
                point = np.repeat(chunk, n)
                triangle = self._triangles[
                    np.repeat(start[chunk], n)
                    + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
                distance = np.linalg.norm(
                    closest_points(points[point], corners[triangle])
                    - points[point], axis=1)
                np.minimum.at(best, point, distance)
        return best


def _key(cells: np.ndarray) -> np.ndarray:
    # 21 bits per axis: a million cells along each, far beyond any bone.
    shifted = cells + (1 << 20)
    return (shifted[..., 0] << 42) | (shifted[..., 1] << 21) | shifted[..., 2]


def _brute(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    out = np.full(len(points), np.inf)
    step = max(1, _PAIRS_PER_CHUNK // max(len(corners), 1))
    for start in range(0, len(points), step):
        block = points[start:start + step]
        point = np.repeat(np.arange(len(block)), len(corners))
        triangle = np.tile(np.arange(len(corners)), len(block))
        distance = np.linalg.norm(
            closest_points(block[point], corners[triangle]) - block[point], axis=1)
        out[start:start + step] = distance.reshape(len(block), -1).min(axis=1)
    return out


def surface_samples(mesh) -> np.ndarray:
    """Every vertex a face uses, then every face centroid.

    Vertices alone miss the inside of large triangles, which is exactly where a cut
    face lies; the centroids put a sample on every face.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    return np.vstack([vertices[np.unique(mesh.faces)],
                      vertices[mesh.faces].mean(axis=1)])


def _shared_samples(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Samples of ``a`` that ``b`` shares, and a bound on how far each is from ``b``.

    A vertex of ``a`` that is also a vertex of ``b`` is no further from ``b``'s surface
    than from that vertex; a face of ``a`` whose corners are a face of ``b`` has its
    centroid no further from ``b`` than from that face's centroid. Shared means equal
    once both are rounded to single precision, because Blender stores coordinates in
    single precision and would otherwise share nothing with a double-precision solver.
    The distance to the matched point is kept, not assumed zero, so the bound is a true
    upper bound, and every sample without a match is measured exactly.
    """
    va = np.asarray(a.vertices, dtype=float)
    vb = np.asarray(b.vertices, dtype=float)
    rounded = np.vstack([vb, va]).astype(np.float32)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    index_in_b = np.full(inverse.max() + 1, -1, dtype=np.int64)
    index_in_b[inverse[:len(vb)]] = np.arange(len(vb))
    a_to_b = index_in_b[inverse[len(vb):]]

    used = np.unique(a.faces)
    vertex_match = a_to_b[used]
    vertex_shared = vertex_match >= 0
    vertex_bound = np.zeros(len(used))
    vertex_bound[vertex_shared] = np.linalg.norm(
        va[used[vertex_shared]] - vb[vertex_match[vertex_shared]], axis=1)

    face_shared = np.zeros(len(a.faces), dtype=bool)
    face_bound = np.zeros(len(a.faces))
    mapped = a_to_b[a.faces]
    whole = np.flatnonzero((mapped >= 0).all(axis=1))
    width = int(len(vb)).bit_length()
    if len(whole) and 3 * width <= 63:
        def pack(faces):
            faces = np.sort(faces, axis=1)
            return (faces[:, 0] << (2 * width)) | (faces[:, 1] << width) | faces[:, 2]

        theirs = pack(np.asarray(b.faces, dtype=np.int64))
        order = np.argsort(theirs)
        ours = pack(mapped[whole])
        where = np.minimum(np.searchsorted(theirs[order], ours), len(order) - 1)
        hit = theirs[order][where] == ours
        rows, match = whole[hit], order[where[hit]]
        face_shared[rows] = True
        face_bound[rows] = np.linalg.norm(
            va[a.faces[rows]].mean(axis=1) - vb[b.faces[match]].mean(axis=1), axis=1)
    return (np.concatenate([vertex_shared, face_shared]),
            np.concatenate([vertex_bound, face_bound]))


def point_to_surface(points, mesh, *, cell_mm: float | None = None) -> np.ndarray:
    """Exact distance from each point to the nearest point anywhere on ``mesh``."""
    return SurfaceGrid(mesh, cell_mm=cell_mm).distances(points)


def _directed(a, b) -> tuple[np.ndarray, int]:
    samples = surface_samples(a)
    shared, distance = _shared_samples(a, b)
    measure = ~shared
    if measure.any():
        distance[measure] = point_to_surface(samples[measure], b)
    return distance, int(measure.sum())


def hausdorff(a, b) -> dict:
    """Symmetric Hausdorff distance between two surfaces, sampled at every vertex and
    face centroid of each, in millimetres.

    Also returns both directed maxima and their 95th percentiles, so one stray triangle
    can be told from a surface that disagrees everywhere, and how many samples were
    measured against the other surface rather than bounded by a shared point.
    """
    (ab, measured_ab), (ba, measured_ba) = _directed(a, b), _directed(b, a)
    return {
        "hausdorff_mm": float(max(ab.max(), ba.max())),
        "a_to_b_max_mm": float(ab.max()),
        "b_to_a_max_mm": float(ba.max()),
        "a_to_b_p95_mm": float(np.percentile(ab, 95)),
        "b_to_a_p95_mm": float(np.percentile(ba, 95)),
        "samples": int(len(ab) + len(ba)),
        "samples_measured": measured_ab + measured_ba,
    }
