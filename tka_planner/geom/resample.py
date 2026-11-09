"""Resampling a surface without changing its shape: decimation and remeshing.

A planner whose answer depends on how finely the bone was triangulated is measuring the
mesh, not the bone. The paper tests that (experiment E2) by measuring the same knee at
half and a tenth of its triangles, and re-triangulated at its own resolution. Those
variants need a resampler that is deterministic, so a result can be reproduced from the
commit alone, and free of dependencies, so it runs wherever the planner does.

The method is **vertex clustering** (Rossignac and Borrel, 1993): space is divided into
cubic cells, every vertex in a cell is replaced by the cell's mean, and triangles whose
corners fall in fewer than three cells disappear. It is the crudest decimator there is --
it ignores curvature and can pinch thin features into non-manifold edges -- which is
the point for a robustness test: a planner that survives it survives any reasonable
export setting. Where the result is non-manifold, quality control says so.

The grid is anchored at the mesh's own bounding-box corner, so resampling a moved copy
gives the moved copy of the resampled mesh, up to floating point.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh

__all__ = [
    "weld",
    "as_triangle_soup",
    "cluster_vertices",
    "decimate",
    "subdivide",
    "remesh",
]


def weld(mesh: Mesh, *, tolerance_mm: float = 1e-4) -> Mesh:
    """Merge coincident vertices, as :func:`~tka_planner.core.meshio.weld_vertices` does.

    The same snapping rule, but the snapped coordinates are hashed to one integer per
    vertex instead of being compared row by row, which is several times faster on the
    million-triangle meshes the resampler is run on.
    """
    keys = np.round(mesh.vertices / tolerance_mm).astype(np.int64)
    _, first, inverse = _unique_rows(keys)
    return Mesh(
        vertices=mesh.vertices[first],
        faces=inverse.reshape(-1)[mesh.faces.reshape(-1)].reshape(mesh.faces.shape),
        source_path=mesh.source_path,
        sha256=None,
        metadata={**mesh.metadata, "welded_tolerance_mm": tolerance_mm},
    )


def as_triangle_soup(mesh: Mesh) -> Mesh:
    """Give every triangle its own three vertices, as :func:`read_stl` does.

    The landmark estimators work on vertex clouds, where an STL soup counts a shared
    corner once per triangle around it. A resampled mesh is returned in the same form,
    so a difference in the result is the resampling and not the bookkeeping.
    """
    vertices = mesh.vertices[mesh.faces.reshape(-1)]
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return Mesh(vertices=vertices, faces=faces, source_path=mesh.source_path,
                sha256=None, metadata=dict(mesh.metadata))


def cluster_vertices(mesh: Mesh, cell_mm: float, *, offset=(0.0, 0.0, 0.0)) -> Mesh:
    """Vertex clustering on a grid of ``cell_mm`` cubes; returns a welded mesh.

    ``offset`` shifts the grid, in millimetres, from the mesh's bounding-box corner.
    Triangles that collapse, and duplicates of a surviving triangle, are dropped; the
    first copy of a duplicate keeps its winding.
    """
    if cell_mm <= 0:
        raise ValueError("cell_mm must be positive.")
    welded = mesh if _is_indexed(mesh) else weld(mesh)
    vertices = welded.vertices
    corner = vertices.min(axis=0) - np.asarray(offset, dtype=float)
    cells = np.floor((vertices - corner) / cell_mm).astype(np.int64)
    _, _, cluster = _unique_rows(cells)
    n_clusters = int(cluster.max()) + 1

    counts = np.bincount(cluster, minlength=n_clusters).astype(float)
    centres = np.column_stack([
        np.bincount(cluster, weights=vertices[:, axis], minlength=n_clusters) / counts
        for axis in range(3)
    ])

    faces = cluster[welded.faces]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2]))
    faces = faces[keep]
    ordered = np.sort(faces, axis=1)
    order = np.lexsort((ordered[:, 2], ordered[:, 1], ordered[:, 0]))
    ordered = ordered[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    faces = faces[np.sort(order[first])]

    # Drop clusters no surviving triangle uses, so the vertex cloud is the surface.
    used = np.unique(faces)
    remap = np.full(n_clusters, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(
        vertices=centres[used],
        faces=remap[faces],
        source_path=mesh.source_path,
        sha256=None,
        metadata={**mesh.metadata, "resample": "vertex_clustering",
                  "cell_mm": float(cell_mm)},
    )


def decimate(mesh: Mesh, target_triangles: int, *, tolerance: float = 0.03,
             max_rounds: int = 24, offset=(0.0, 0.0, 0.0)) -> Mesh:
    """Vertex clustering with the cell size searched to give ``target_triangles``.

    The triangle count falls roughly with the square of the cell size, so the first
    guess comes from the surface area and a bisection on the logarithm of the cell size
    finishes within ``tolerance`` of the target (relative). Deterministic: the same mesh
    and target always give the same cell.
    """
    welded = mesh if _is_indexed(mesh) else weld(mesh)
    if target_triangles >= welded.n_triangles:
        raise ValueError("target_triangles must be below the mesh's triangle count.")
    if target_triangles < 4:
        raise ValueError("target_triangles must be at least 4.")

    triangles = welded.vertices[welded.faces]
    area = 0.5 * float(np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ).sum())
    guess = np.sqrt(2.0 * area / target_triangles)
    low, high = np.log(guess / 8.0), np.log(guess * 8.0)
    best = None
    for _ in range(max_rounds):
        middle = 0.5 * (low + high)
        candidate = cluster_vertices(welded, float(np.exp(middle)), offset=offset)
        error = candidate.n_triangles / target_triangles - 1.0
        if best is None or abs(error) < abs(best[0]):
            best = (error, candidate)
        if abs(error) <= tolerance:
            break
        if error > 0:      # too many triangles: coarser cells
            low = middle
        else:
            high = middle
    return best[1]


def subdivide(mesh: Mesh) -> Mesh:
    """Split every triangle into four at its edge midpoints; the shape is unchanged.

    Returns a welded mesh: each edge midpoint is created once and shared by both
    triangles on the edge.
    """
    welded = mesh if _is_indexed(mesh) else weld(mesh)
    faces = welded.faces
    n_vertices = welded.n_vertices
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    unique, inverse = np.unique(edges[:, 0] * n_vertices + edges[:, 1],
                                return_inverse=True)
    inverse = inverse.reshape(-1)
    a, b = unique // n_vertices, unique % n_vertices
    midpoints = 0.5 * (welded.vertices[a] + welded.vertices[b])
    m = n_vertices + inverse.reshape(3, -1)    # midpoint of edges 01, 12, 20
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m01, m12, m20 = m
    new_faces = np.concatenate([
        np.column_stack([v0, m01, m20]),
        np.column_stack([m01, v1, m12]),
        np.column_stack([m20, m12, v2]),
        np.column_stack([m01, m12, m20]),
    ])
    return Mesh(
        vertices=np.vstack([welded.vertices, midpoints]),
        faces=new_faces,
        source_path=mesh.source_path,
        sha256=None,
        metadata={**mesh.metadata, "resample": "midpoint_subdivision"},
    )


def remesh(mesh: Mesh, *, tolerance: float = 0.03) -> Mesh:
    """Re-triangulate at the mesh's own resolution.

    Subdivision first quadruples the triangles without moving the surface; clustering
    then brings the count back to the original on a grid offset by a third of a cell
    from where it would otherwise fall. Every vertex of the result is a new point -- a
    cell mean -- and the triangle count is the original's, so this is the same surface
    sampled afresh.
    """
    welded = mesh if _is_indexed(mesh) else weld(mesh)
    fine = subdivide(welded)
    target = welded.n_triangles
    triangles = welded.vertices[welded.faces]
    area = 0.5 * float(np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ).sum())
    third = np.sqrt(2.0 * area / target) / 3.0
    result = decimate(fine, target, tolerance=tolerance,
                      offset=(third, third, third))
    return Mesh(vertices=result.vertices, faces=result.faces,
                source_path=mesh.source_path, sha256=None,
                metadata={**result.metadata, "resample": "subdivide_then_cluster"})


def _is_indexed(mesh: Mesh) -> bool:
    """True unless the mesh is a triangle soup straight from an STL."""
    return mesh.n_vertices < 3 * mesh.n_triangles


def _unique_rows(cells: np.ndarray):
    """``np.unique(cells, axis=0)`` with first indices and inverse, for integer triples.

    Each row is packed into one int64 when the grid allows it, which sorts far faster
    than comparing rows; a grid too large to pack falls back to the row comparison.
    """
    shifted = cells - cells.min(axis=0)
    span = shifted.max(axis=0) + 1
    if float(span[0]) * float(span[1]) * float(span[2]) < 2.0 ** 62:
        packed = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]
        unique, first, inverse = np.unique(packed, return_index=True,
                                           return_inverse=True)
    else:
        unique, first, inverse = np.unique(cells, axis=0, return_index=True,
                                           return_inverse=True)
    return unique, first, inverse.reshape(-1)
