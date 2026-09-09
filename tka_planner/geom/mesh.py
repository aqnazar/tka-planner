"""Transforms, primitives and measures over :class:`tka_planner.core.meshio.Mesh`.

The primitives here replace the ``bpy.ops.mesh.primitive_*`` calls the Blender builder
used for cut planes, mechanical axes, landmark spheres, discard boxes and the insert
slab. Each is generated directly as vertices and faces, so the shapes are exactly
reproducible and testable against their analytic volume rather than whatever a given
Blender release happens to produce.

Every function takes and returns millimetres.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tka_planner.core.meshio import Mesh

__all__ = [
    "transformed", "box", "slab", "disc", "cylinder", "uv_sphere",
    "volume", "surface_area", "write_stl", "plane_matrix", "rotation_to",
    "translation",
]


def transformed(mesh: Mesh, matrix) -> Mesh:
    """Apply a 4x4 to a mesh's vertices, leaving its topology alone."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got {matrix.shape}.")

    vertices = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return Mesh(
        vertices=vertices,
        faces=mesh.faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata=dict(mesh.metadata),
    )


def translation(offset) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(offset, dtype=float)
    return matrix


def rotation_to(direction) -> np.ndarray:
    """A 3x3 rotation taking +Z onto ``direction``.

    Uses Rodrigues' formula about the axis between the two vectors. The antiparallel
    case has no such axis, so it is handled separately as a half turn about X.
    """
    target = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(target)
    if norm == 0.0:
        raise ValueError("Direction must not be the zero vector.")
    target = target / norm

    z = np.array([0.0, 0.0, 1.0])
    cosine = float(np.dot(z, target))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(z, target)
    sine = np.linalg.norm(axis)
    axis = axis / sine
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def plane_matrix(point_mm, normal_mm) -> np.ndarray:
    """A 4x4 whose origin is on the plane and whose +Z is the plane normal."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_to(normal_mm)
    matrix[:3, 3] = np.asarray(point_mm, dtype=float)
    return matrix


def _build(vertices, faces, matrix=None) -> Mesh:
    mesh = Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
    )
    return mesh if matrix is None else transformed(mesh, matrix)


def box(size_mm, matrix=None) -> Mesh:
    """An axis-aligned box centred on the origin.

    ``size_mm`` is one edge length, or a triple of them.
    """
    size = np.asarray(
        (size_mm,) * 3 if np.isscalar(size_mm) else size_mm, dtype=float
    )
    half = size / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=float)
    vertices = signs * half

    # Outward-facing winding, which the volume formula and every boolean kernel need.
    faces = np.array([
        [0, 3, 2], [0, 2, 1],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [1, 2, 6], [1, 6, 5],  # +X
        [0, 4, 7], [0, 7, 3],  # -X
    ], dtype=np.int64)
    return _build(vertices, faces, matrix)


def slab(ml_mm: float, ap_mm: float, thickness_mm: float, matrix=None) -> Mesh:
    """A box standing on z = 0, centred in x and y.

    This is the insert spacer. It is built base-first so a thickness change moves only
    the top face, which is what lets the insert be re-stretched in place rather than
    rebuilt.
    """
    block = box((ml_mm, ap_mm, thickness_mm))
    lifted = transformed(block, translation((0.0, 0.0, thickness_mm / 2.0)))
    return lifted if matrix is None else transformed(lifted, matrix)


def disc(radius_mm: float, *, segments: int = 64, matrix=None) -> Mesh:
    """A filled circle in the z = 0 plane, for drawing a cut plane."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.column_stack([
        radius_mm * np.cos(angles),
        radius_mm * np.sin(angles),
        np.zeros(segments),
    ])
    vertices = np.vstack([[0.0, 0.0, 0.0], rim])
    faces = np.array(
        [[0, i + 1, (i + 1) % segments + 1] for i in range(segments)],
        dtype=np.int64,
    )
    return _build(vertices, faces, matrix)


def cylinder(radius_mm: float, length_mm: float, *, segments: int = 16,
             matrix=None) -> Mesh:
    """A closed cylinder centred on the origin, its axis along +Z."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    half = length_mm / 2.0
    ring = np.column_stack([
        radius_mm * np.cos(angles), radius_mm * np.sin(angles), np.zeros(segments)
    ])
    bottom = ring - np.array([0.0, 0.0, half])
    top = ring + np.array([0.0, 0.0, half])
    centre_bottom = np.array([[0.0, 0.0, -half]])
    centre_top = np.array([[0.0, 0.0, half]])
    vertices = np.vstack([bottom, top, centre_bottom, centre_top])

    bottom_centre = 2 * segments
    top_centre = 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([bottom_centre, j, i])
        faces.append([top_centre, segments + i, segments + j])
    return _build(vertices, np.array(faces, dtype=np.int64), matrix)


def uv_sphere(radius_mm: float, *, segments: int = 12, rings: int = 8,
              centre=(0.0, 0.0, 0.0)) -> Mesh:
    """A UV sphere, for a landmark marker."""
    thetas = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    phis = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    grid = []
    for theta in thetas:
        grid.append(np.column_stack([
            radius_mm * np.sin(theta) * np.cos(phis),
            radius_mm * np.sin(theta) * np.sin(phis),
            np.full(segments, radius_mm * np.cos(theta)),
        ]))
    body = np.vstack(grid)
    north = np.array([[0.0, 0.0, radius_mm]])
    south = np.array([[0.0, 0.0, -radius_mm]])
    vertices = np.vstack([body, north, south]) + np.asarray(centre, dtype=float)

    n_rows = len(thetas)
    north_index = n_rows * segments
    south_index = north_index + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([north_index, i, j])
        faces.append([south_index, (n_rows - 1) * segments + j,
                      (n_rows - 1) * segments + i])
    for row in range(n_rows - 1):
        base, nxt = row * segments, (row + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, nxt + i, nxt + j])
            faces.append([base + i, nxt + j, base + j])
    return _build(vertices, np.array(faces, dtype=np.int64))


def volume(mesh: Mesh) -> float:
    """Signed volume by the divergence theorem, in cubic millimetres.

    Positive for outward-facing winding. Used to compare two kernels' output without
    caring how either triangulated it.
    """
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def surface_area(mesh: Mesh) -> float:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def write_stl(mesh: Mesh, path) -> Path:
    """Write a binary STL in millimetres.

    Written directly rather than through Blender's exporter, which is what removes the
    last reason an export path needed a Blender installation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    corners = mesh.vertices[mesh.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0.0)

    records = np.zeros(
        mesh.n_triangles,
        dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2")]),
    )
    records["normal"] = normals
    records["vertices"] = corners

    with path.open("wb") as handle:
        handle.write(b"tka-planner".ljust(80, b"\0"))
        handle.write(struct.pack("<I", mesh.n_triangles))
        handle.write(records.tobytes())
    return path
