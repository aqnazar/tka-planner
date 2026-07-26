"""STL reading.

These tests matter more than their subject suggests: if the reader is wrong, every
measurement downstream is wrong in a way no anatomical test would catch, because the
synthetic fixtures never go through it. So the checks here are structural -- round
trips, format detection, and topology recovered via the Euler characteristic.
"""

import struct

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh, read_stl, weld_vertices


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

# A unit tetrahedron: the smallest closed surface, so Euler's formula applies.
TETRAHEDRON_VERTICES = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
])
TETRAHEDRON_FACES = np.array([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
])


def write_binary_stl(path, vertices, faces, header=b"binary test"):
    """Write a binary STL, the format Slicer and most CAD tools export."""
    with open(path, "wb") as handle:
        handle.write(header.ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(faces)))
        for face in faces:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))  # normal, unused
            for index in face:
                handle.write(struct.pack("<3f", *vertices[index]))
            handle.write(struct.pack("<H", 0))
    return path


def write_ascii_stl(path, vertices, faces):
    lines = ["solid test"]
    for face in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for index in face:
            x, y, z = vertices[index]
            lines.append(f"      vertex {x:.6e} {y:.6e} {z:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines))
    return path


@pytest.fixture
def tetrahedron_stl(tmp_path):
    return write_binary_stl(
        tmp_path / "tetra.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES
    )


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


class TestBinaryReading:
    def test_reads_expected_triangle_count(self, tetrahedron_stl):
        mesh = read_stl(tetrahedron_stl)
        assert mesh.n_triangles == 4
        assert mesh.n_vertices == 12  # unwelded triangle soup

    def test_recovers_coordinates(self, tetrahedron_stl):
        mesh = read_stl(tetrahedron_stl)
        corners = np.unique(np.round(mesh.vertices, 6), axis=0)
        expected = np.unique(np.round(TETRAHEDRON_VERTICES, 6), axis=0)
        assert np.allclose(corners, expected)

    def test_detects_binary_despite_a_solid_header(self, tmp_path):
        """Binary STLs may begin with the word ``solid``.

        Sniffing that prefix -- which plenty of readers do -- misclassifies them and
        yields an empty or corrupt mesh. Detection uses the declared triangle count
        against the file length instead.
        """
        path = write_binary_stl(
            tmp_path / "misleading.stl",
            TETRAHEDRON_VERTICES,
            TETRAHEDRON_FACES,
            header=b"solid this_is_actually_binary",
        )
        mesh = read_stl(path)

        assert mesh.metadata["format"] == "binary"
        assert mesh.n_triangles == 4

    def test_handles_an_empty_mesh(self, tmp_path):
        path = write_binary_stl(tmp_path / "empty.stl", TETRAHEDRON_VERTICES, [])
        mesh = read_stl(path)
        assert mesh.n_triangles == 0


class TestAsciiReading:
    def test_reads_ascii(self, tmp_path):
        path = write_ascii_stl(
            tmp_path / "tetra_ascii.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES
        )
        mesh = read_stl(path)

        assert mesh.metadata["format"] == "ascii"
        assert mesh.n_triangles == 4

    def test_ascii_and_binary_agree(self, tmp_path):
        binary = read_stl(write_binary_stl(
            tmp_path / "b.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES))
        ascii_mesh = read_stl(write_ascii_stl(
            tmp_path / "a.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES))

        assert np.allclose(binary.vertices, ascii_mesh.vertices)

    def test_rejects_a_partial_triangle(self, tmp_path):
        path = tmp_path / "broken.stl"
        path.write_text(
            "solid x\nfacet normal 0 0 0\nouter loop\n"
            "vertex 0 0 0\nvertex 1 0 0\n"  # only two vertices
            "endloop\nendfacet\nendsolid x\n"
        )
        with pytest.raises(ValueError, match="not a whole number of triangles"):
            read_stl(path)


class TestErrorHandling:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="STL not found"):
            read_stl(tmp_path / "absent.stl")

    def test_tiny_file(self, tmp_path):
        path = tmp_path / "tiny.stl"
        path.write_bytes(b"nope")
        with pytest.raises(ValueError, match="too small"):
            read_stl(path)

    def test_unparseable_file(self, tmp_path):
        path = tmp_path / "garbage.stl"
        path.write_bytes(b"x" * 500)
        with pytest.raises(ValueError, match="No vertices found"):
            read_stl(path)


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


class TestProvenance:
    def test_records_a_content_hash(self, tetrahedron_stl):
        """Reproducibility claims are only as good as their input identification."""
        mesh = read_stl(tetrahedron_stl)
        assert mesh.sha256 is not None
        assert len(mesh.sha256) == 64

    def test_hash_tracks_content_not_filename(self, tmp_path):
        first = write_binary_stl(
            tmp_path / "one.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)
        second = write_binary_stl(
            tmp_path / "two.stl", TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)
        moved = write_binary_stl(
            tmp_path / "three.stl", TETRAHEDRON_VERTICES * 2, TETRAHEDRON_FACES)

        assert read_stl(first).sha256 == read_stl(second).sha256
        assert read_stl(first).sha256 != read_stl(moved).sha256

    def test_hashing_can_be_skipped(self, tetrahedron_stl):
        assert read_stl(tetrahedron_stl, compute_hash=False).sha256 is None


# ----------------------------------------------------------------------
# Geometry accessors
# ----------------------------------------------------------------------


class TestMeshProperties:
    def test_bounds_and_extent(self, tetrahedron_stl):
        mesh = read_stl(tetrahedron_stl)
        low, high = mesh.bounds

        assert np.allclose(low, [0, 0, 0])
        assert np.allclose(high, [10, 10, 10])
        assert np.allclose(mesh.extent, [10, 10, 10])

    def test_repr_is_informative(self, tetrahedron_stl):
        text = repr(read_stl(tetrahedron_stl))
        assert "tetra.stl" in text and "4 triangles" in text


# ----------------------------------------------------------------------
# Welding
# ----------------------------------------------------------------------


class TestWelding:
    def test_recovers_shared_vertices(self, tetrahedron_stl):
        welded = weld_vertices(read_stl(tetrahedron_stl))
        assert welded.n_vertices == 4
        assert welded.n_triangles == 4

    def test_satisfies_the_euler_characteristic(self, tetrahedron_stl):
        """For any closed genus-0 triangle mesh, V - E + F = 2 gives V = F/2 + 2.

        This is a strong, independent check: welding too loosely collapses distinct
        vertices and welding too tightly leaves duplicates, and either breaks the
        identity. It also confirms the real patient meshes are watertight.
        """
        welded = weld_vertices(read_stl(tetrahedron_stl))
        assert welded.n_vertices == welded.n_triangles // 2 + 2

    def test_does_not_move_geometry(self, tetrahedron_stl):
        """Welding snaps to a grid to find duplicates but must keep true coordinates."""
        mesh = read_stl(tetrahedron_stl)
        welded = weld_vertices(mesh)

        for vertex in welded.vertices:
            distances = np.linalg.norm(mesh.vertices - vertex, axis=1)
            assert distances.min() < 1e-12

    def test_faces_still_index_the_same_points(self, tetrahedron_stl):
        mesh = read_stl(tetrahedron_stl)
        welded = weld_vertices(mesh)

        original = mesh.vertices[mesh.faces]
        recovered = welded.vertices[welded.faces]
        assert np.allclose(original, recovered)

    def test_preserves_provenance(self, tetrahedron_stl):
        mesh = read_stl(tetrahedron_stl)
        welded = weld_vertices(mesh)

        assert welded.sha256 == mesh.sha256
        assert welded.metadata["welded_tolerance_mm"] == pytest.approx(1e-4)

    def test_rejects_a_non_positive_tolerance(self, tetrahedron_stl):
        with pytest.raises(ValueError, match="must be positive"):
            weld_vertices(read_stl(tetrahedron_stl), tolerance_mm=0.0)

    def test_keeps_genuinely_distinct_nearby_vertices(self, tmp_path):
        """Two points 0.01 mm apart are distinct anatomy, not duplicates."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
            [0.01, 0.0, 0.0],
        ])
        faces = np.array([[0, 1, 2], [3, 1, 2]])
        path = write_binary_stl(tmp_path / "close.stl", vertices, faces)

        welded = weld_vertices(read_stl(path))
        assert welded.n_vertices == 4
