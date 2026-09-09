"""The GLB encoder: the one boundary where millimetres become metres."""

import json
import struct

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.gltf import MM_PER_M, write_glb, write_glb_file


def _parse(blob):
    """Take a GLB apart the way a viewer does, so the test reads the real bytes."""
    magic, version, total = struct.unpack_from("<III", blob, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total == len(blob)

    offset = 12
    chunks = {}
    while offset < len(blob):
        length, kind = struct.unpack_from("<II", blob, offset)
        chunks[kind] = blob[offset + 8: offset + 8 + length]
        offset += 8 + length
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def test_a_box_round_trips_through_the_container():
    mesh = gm.box(40.0)

    document, binary = _parse(write_glb(mesh))

    positions = document["accessors"][0]
    indices = document["accessors"][1]
    assert positions["count"] == mesh.vertices.shape[0]
    assert indices["count"] == mesh.faces.size
    assert document["meshes"][0]["primitives"][0]["mode"] == 4
    assert len(binary) >= positions["count"] * 12 + indices["count"] * 4


def test_vertices_are_written_in_metres():
    """A 40 mm box must measure 0.04 in the file, or the viewer's scale is wrong."""
    mesh = gm.box(40.0)

    document, _ = _parse(write_glb(mesh))

    extent = np.array(document["accessors"][0]["max"]) - np.array(
        document["accessors"][0]["min"]
    )
    assert np.allclose(extent, 40.0 / MM_PER_M)


def test_the_accessor_bounds_match_the_mesh():
    mesh = gm.uv_sphere(13.0, segments=20, rings=14)

    document, _ = _parse(write_glb(mesh))

    low, high = mesh.bounds
    assert np.allclose(document["accessors"][0]["min"], low / MM_PER_M, atol=1e-6)
    assert np.allclose(document["accessors"][0]["max"], high / MM_PER_M, atol=1e-6)


def test_the_geometry_survives_the_encoding():
    mesh = gm.cylinder(6.0, 30.0, segments=24)

    document, binary = _parse(write_glb(mesh))

    view = document["bufferViews"][0]
    positions = np.frombuffer(
        binary, dtype="<f4", count=view["byteLength"] // 4, offset=view["byteOffset"]
    ).reshape(-1, 3)
    assert np.allclose(positions * MM_PER_M, mesh.vertices, atol=1e-3)

    view = document["bufferViews"][1]
    indices = np.frombuffer(
        binary, dtype="<u4", count=view["byteLength"] // 4, offset=view["byteOffset"]
    )
    assert np.array_equal(indices.reshape(-1, 3), mesh.faces)


def test_every_chunk_is_four_byte_aligned():
    """An unaligned chunk fails to load in strict viewers, silently in loose ones."""
    # An odd vertex count is what pushes the positions to a length needing padding.
    mesh = gm.uv_sphere(3.0, segments=7, rings=5)

    blob = write_glb(mesh)
    document, binary = _parse(blob)

    assert len(blob) % 4 == 0
    assert len(binary) % 4 == 0
    assert document["bufferViews"][1]["byteOffset"] % 4 == 0


def test_a_face_index_past_the_vertices_is_refused():
    broken = Mesh(
        vertices=np.zeros((3, 3), dtype=float),
        faces=np.array([[0, 1, 7]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="past the end"):
        write_glb(broken)


def test_writing_to_a_file_creates_the_directory(tmp_path):
    path = write_glb_file(gm.box(5.0), tmp_path / "nested" / "cube.glb")

    assert path.is_file()
    assert path.read_bytes()[:4] == b"glTF"
