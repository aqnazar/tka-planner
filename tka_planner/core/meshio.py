"""Reading surface meshes without Blender.

The whole architecture rests on ``tka_planner.core`` depending on numpy and nothing
else. Without a mesh reader here that claim would be hollow: measurement, sizing and
the Monte Carlo sensitivity study all need vertex coordinates, and routing them through
``bpy`` would drag Blender into the test suite and make a few thousand perturbation
runs impractical.

STL is the interchange format in this pipeline because it is what 3D Slicer exports and
what the implant CAD produces. It is a poor format -- a bare triangle soup with no
vertex sharing, no units and no coordinate system -- so everything it fails to record
must be supplied and checked elsewhere. In particular STL cannot state whether it is in
LPS or RAS, which is why the landmark schema requires the coordinate system explicitly
and why :mod:`tka_planner.core.qc` gates landmarks against the mesh bounds.

Performance matters: the patient meshes here run to 1.27 million triangles, so files are
parsed with a single ``numpy.frombuffer`` over a structured dtype rather than any
per-triangle Python loop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Mesh", "read_stl", "weld_vertices"]


# One binary STL triangle: a normal, three vertices, and a 2-byte attribute word.
_BINARY_TRIANGLE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)
_BINARY_HEADER_BYTES = 84  # 80-byte header + uint32 triangle count


@dataclass(frozen=True)
class Mesh:
    """A triangle surface mesh in millimetres.

    ``vertices`` is ``(N, 3)`` and ``faces`` is ``(M, 3)`` of indices into it. Straight
    from :func:`read_stl` the mesh is an unwelded triangle soup, so ``N == 3 * M`` and
    every triangle owns its own copies; call :func:`weld_vertices` for topology.

    The ``sha256`` and ``n_triangles`` fields exist so a plan can record exactly which
    file it was computed from. Reproducibility claims are only as good as their input
    identification.
    """

    vertices: np.ndarray
    faces: np.ndarray
    source_path: str | None = None
    sha256: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_triangles(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min_xyz, max_xyz)`` of the vertex cloud."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        """Axis-aligned bounding box dimensions."""
        lo, hi = self.bounds
        return hi - lo

    def __repr__(self) -> str:
        extent = np.round(self.extent, 1)
        name = Path(self.source_path).name if self.source_path else "<in-memory>"
        return (
            f"Mesh({name}, {self.n_triangles} triangles, "
            f"extent={extent[0]}x{extent[1]}x{extent[2]} mm)"
        )


def read_stl(path: str | Path, *, compute_hash: bool = True) -> Mesh:
    """Read a binary or ASCII STL file.

    The format is detected from the file's structure, not its opening token: binary STL
    files are permitted to begin with the word ``solid``, so sniffing that prefix
    misidentifies them. The reliable test is whether the triangle count at offset 80
    accounts for the file's exact length.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"STL not found: {path}")

    raw = path.read_bytes()
    if len(raw) < 15:  # shorter than the smallest conceivable ASCII STL
        raise ValueError(f"File is too small to be an STL: {path}")

    if _looks_binary(raw):
        vertices, faces = _parse_binary(raw)
    else:
        vertices, faces = _parse_ascii(raw)

    sha256 = hashlib.sha256(raw).hexdigest() if compute_hash else None
    return Mesh(
        vertices=vertices,
        faces=faces,
        source_path=str(path),
        sha256=sha256,
        metadata={"format": "binary" if _looks_binary(raw) else "ascii",
                  "bytes": len(raw)},
    )


def weld_vertices(
    mesh: Mesh,
    *,
    tolerance_mm: float = 1e-4,
) -> Mesh:
    """Merge coincident vertices so the mesh has real topology.

    STL stores each triangle independently, so a watertight surface still arrives as
    disconnected triangles. Edge-based checks -- non-manifold edges, boundary edges,
    whether the surface is closed -- are meaningless until vertices are welded.

    Merging is done by snapping to a grid of ``tolerance_mm``. The default of 0.1 um
    sits below the precision of the single-precision floats STL stores (about 6e-5 mm at
    a 1000 mm coordinate), so it merges vertices that were written as the same point
    without risking the collapse of genuinely distinct ones.
    """
    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive.")

    keys = np.round(mesh.vertices / tolerance_mm).astype(np.int64)
    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )

    # Keep the original coordinates of the first occurrence rather than the snapped
    # grid position, so welding never moves geometry.
    welded = mesh.vertices[first_index]
    faces = inverse[mesh.faces.reshape(-1)].reshape(mesh.faces.shape)

    return Mesh(
        vertices=welded,
        faces=faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata={**mesh.metadata, "welded_tolerance_mm": tolerance_mm},
    )


# ----------------------------------------------------------------------
# Format handling
# ----------------------------------------------------------------------


def _looks_binary(raw: bytes) -> bool:
    """True when the declared triangle count exactly accounts for the file length."""
    if len(raw) < _BINARY_HEADER_BYTES:
        return False
    declared = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    return len(raw) == _BINARY_HEADER_BYTES + declared * _BINARY_TRIANGLE.itemsize


def _parse_binary(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    n_triangles = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    if n_triangles == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    records = np.frombuffer(raw, dtype=_BINARY_TRIANGLE, count=n_triangles,
                            offset=_BINARY_HEADER_BYTES)
    vertices = records["vertices"].reshape(-1, 3).astype(np.float64)
    faces = np.arange(n_triangles * 3, dtype=np.int64).reshape(n_triangles, 3)
    return vertices, faces


_ASCII_VERTEX = re.compile(
    rb"vertex\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)"
)


def _parse_ascii(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    coordinates = _ASCII_VERTEX.findall(raw)
    if not coordinates:
        raise ValueError(
            "No vertices found. The file is neither a valid binary STL "
            "(its length does not match the declared triangle count) nor a "
            "readable ASCII STL."
        )
    if len(coordinates) % 3 != 0:
        raise ValueError(
            f"ASCII STL has {len(coordinates)} vertices, which is not a whole "
            "number of triangles."
        )

    vertices = np.array(coordinates, dtype=np.float64)
    n_triangles = len(coordinates) // 3
    faces = np.arange(n_triangles * 3, dtype=np.int64).reshape(n_triangles, 3)
    return vertices, faces
