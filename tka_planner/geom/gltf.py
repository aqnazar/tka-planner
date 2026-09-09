"""Encoding a mesh as binary glTF, for the viewer.

This is the one place in the engine where millimetres stop. glTF's convention is
metres, and a viewer that respects it gets a camera whose near and far planes, whose
lighting falloff and whose control damping are all in the units they were tuned for.
So vertices are divided by a thousand here, and the viewer scales node translations by
the same factor when it reads a transform. Nothing else in the project ever sees a
metre.

The encoder writes exactly what our meshes are: one buffer, one mesh, one primitive,
positions and indices, no materials and no normals. Colour is a scene property here and
belongs to the node, not the geometry, and the viewer computes flat normals for free.
Leaving both out roughly halves what crosses the wire for a bone of two million
triangles.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

__all__ = ["write_glb", "write_glb_file", "MM_PER_M"]

MM_PER_M = 1000.0

_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A  # "JSON"
_BIN_CHUNK = 0x004E4942  # "BIN\0"

_FLOAT = 5126
_UNSIGNED_INT = 5125
_TRIANGLES = 4
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def write_glb(mesh, *, name: str = "mesh") -> bytes:
    """Encode a mesh as a self-contained GLB, with vertices converted to metres."""
    positions = np.ascontiguousarray(
        np.asarray(mesh.vertices, dtype=np.float64) / MM_PER_M, dtype=np.float32
    )
    indices = np.ascontiguousarray(
        np.asarray(mesh.faces, dtype=np.int64).reshape(-1), dtype=np.uint32
    )
    if positions.size and indices.size and indices.max() >= positions.shape[0]:
        raise ValueError("A face index points past the end of the vertex array.")

    position_bytes = positions.tobytes()
    index_bytes = indices.tobytes()
    # Every accessor must start on a multiple of its component size. Positions come
    # first and are already four-byte aligned, so only the join needs padding.
    padding = (-len(position_bytes)) % 4
    binary = position_bytes + b"\0" * padding + index_bytes
    index_offset = len(position_bytes) + padding

    low = positions.min(axis=0).tolist() if positions.size else [0.0, 0.0, 0.0]
    high = positions.max(axis=0).tolist() if positions.size else [0.0, 0.0, 0.0]

    document = {
        "asset": {"version": "2.0", "generator": "tka-planner"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": _TRIANGLES,
            }],
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": int(positions.shape[0]),
                "type": "VEC3",
                # Required on POSITION by the specification, and the viewer uses them
                # to frame the camera without walking the vertices itself.
                "min": [float(v) for v in low],
                "max": [float(v) for v in high],
            },
            {
                "bufferView": 1,
                "componentType": _UNSIGNED_INT,
                "count": int(indices.shape[0]),
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(position_bytes),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_bytes),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
    }

    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)   # chunks pad with spaces
    binary += b"\0" * ((-len(binary)) % 4)          # and the binary chunk with zeros

    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    return b"".join([
        struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total),
        struct.pack("<II", len(json_bytes), _JSON_CHUNK), json_bytes,
        struct.pack("<II", len(binary), _BIN_CHUNK), binary,
    ])


def write_glb_file(mesh, path, *, name: str = "mesh") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(write_glb(mesh, name=name))
    return path
