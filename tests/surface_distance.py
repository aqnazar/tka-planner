"""Distance from points to a triangulated surface.

Comparing two boolean kernels needs a surface-to-surface metric, not a vertex-to-vertex
one. Two solvers that produce the identical solid still triangulate it differently, so a
vertex of one routinely lands in the middle of a face of the other. Measuring
vertex-to-vertex reports that as an error of half a triangle's width, which says
nothing about whether the geometry agrees.

The closest point on a triangle is found by the standard barycentric-region method
(Ericson, *Real-Time Collision Detection*, section 5.1.5), vectorised over all
point-triangle pairs and evaluated in chunks so the pairwise array stays bounded.
"""

from __future__ import annotations

import numpy as np

__all__ = ["closest_point_on_triangles", "point_to_surface", "hausdorff"]


def closest_point_on_triangles(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Closest point on each triangle to each point.

    ``points`` is ``(P, 3)`` and ``corners`` is ``(T, 3, 3)``. Returns ``(P, T, 3)``.
    """
    a = corners[None, :, 0, :]
    b = corners[None, :, 1, :]
    c = corners[None, :, 2, :]
    p = points[:, None, :]

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.einsum("ptk,ptk->pt", ab, ap)
    d2 = np.einsum("ptk,ptk->pt", ac, ap)

    bp = p - b
    d3 = np.einsum("ptk,ptk->pt", ab, bp)
    d4 = np.einsum("ptk,ptk->pt", ac, bp)

    cp = p - c
    d5 = np.einsum("ptk,ptk->pt", ab, cp)
    d6 = np.einsum("ptk,ptk->pt", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # Interior of the face, as the default. Every region below overwrites it.
    denominator = va + vb + vc
    safe = np.where(np.abs(denominator) < 1e-300, 1.0, denominator)
    v = (vb / safe)[..., None]
    w = (vc / safe)[..., None]
    closest = a + ab * v + ac * w

    def pick(condition, value):
        return np.where(condition[..., None], value, closest)

    # Vertex regions.
    closest = pick((d1 <= 0) & (d2 <= 0), np.broadcast_to(a, closest.shape))
    closest = pick((d3 >= 0) & (d4 <= d3), np.broadcast_to(b, closest.shape))
    closest = pick((d6 >= 0) & (d5 <= d6), np.broadcast_to(c, closest.shape))

    # Edge regions.
    edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t_ab = np.where(edge_ab, d1 / np.where(d1 - d3 == 0, 1.0, d1 - d3), 0.0)
    closest = pick(edge_ab, a + ab * t_ab[..., None])

    edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t_ac = np.where(edge_ac, d2 / np.where(d2 - d6 == 0, 1.0, d2 - d6), 0.0)
    closest = pick(edge_ac, a + ac * t_ac[..., None])

    edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    span = (d4 - d3) + (d5 - d6)
    t_bc = np.where(edge_bc, (d4 - d3) / np.where(span == 0, 1.0, span), 0.0)
    closest = pick(edge_bc, b + (c - b) * t_bc[..., None])

    return closest


def point_to_surface(points: np.ndarray, mesh, *, chunk: int = 64) -> np.ndarray:
    """Distance from each point to the nearest point anywhere on ``mesh``'s surface."""
    corners = mesh.vertices[mesh.faces]
    out = np.empty(len(points), dtype=float)

    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        closest = closest_point_on_triangles(block, corners)
        distances = np.linalg.norm(closest - block[:, None, :], axis=2)
        out[start:start + chunk] = distances.min(axis=1)
    return out


def hausdorff(a, b, *, sample: int = 1500, seed: int = 0) -> float:
    """Symmetric sampled Hausdorff distance between two surfaces, in millimetres.

    Sampled at each mesh's vertices, which for these meshes covers the surface densely
    enough to catch any real disagreement, and bounded so the pairwise array stays
    affordable.
    """
    rng = np.random.default_rng(seed)

    def sampled(mesh):
        points = mesh.vertices
        if len(points) > sample:
            points = points[rng.choice(len(points), sample, replace=False)]
        return points

    return max(
        float(point_to_surface(sampled(a), b).max()),
        float(point_to_surface(sampled(b), a).max()),
    )
