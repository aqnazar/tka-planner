"""Detached specks of surface beyond a bone are set aside before it is measured.

One cohort tibia carried an 80-triangle sliver 37 mm above its plateau. Every tibial
estimator searches below the highest vertex, so the sliver became the top of the tibia,
the spines were placed on it, and the knee was refused at the tibial frame.
"""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh, face_components, weld_vertices
from tka_planner.core.qc import Severity, separate_detached_fragments
from tka_planner.geom import mesh as gm


def soup(*meshes) -> Mesh:
    """Meshes joined as STL delivers them: every triangle owning its own vertices."""
    vertices = np.concatenate([m.vertices[m.faces].reshape(-1, 3) for m in meshes])
    return Mesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3))


def bone():
    return gm.uv_sphere(30.0, segments=48, rings=40)


def speck(centre, size=1.0):
    return gm.box(size, gm.translation(centre))


def test_pieces_are_labelled_largest_first():
    mesh = weld_vertices(soup(speck((0.0, 0.0, 70.0)), bone()))
    labels = face_components(mesh)
    assert labels.max() == 1
    assert np.sum(labels == 1) == 12        # the box
    assert np.sum(labels == 0) == len(bone().faces)


def test_an_unwelded_soup_is_all_separate_triangles():
    labels = face_components(soup(bone()))
    assert len(np.unique(labels)) == len(labels)


def test_a_speck_beyond_the_bone_is_set_aside_and_reported():
    mesh = soup(bone(), speck((5.0, -20.0, 67.0)))
    cleaned, welded, report = separate_detached_fragments(mesh, "tibia")

    assert cleaned.vertices[:, 2].max() == pytest.approx(30.0, abs=1e-6)
    assert len(cleaned.faces) == len(bone().faces)
    assert len(welded.faces) == len(cleaned.faces)
    [finding] = report.findings
    assert finding.code == "MESH_DETACHED_FRAGMENT"
    assert finding.severity is Severity.WARNING
    assert finding.context["n_triangles"] == 12
    assert finding.context["max_outside_mm"] == pytest.approx(37.5, abs=0.1)
    assert cleaned.sha256 == mesh.sha256


def test_what_remains_keeps_its_vertex_order():
    """Searches break ties by vertex index, so removal must not reorder the rest."""
    body = soup(bone())
    mesh = soup(bone(), speck((0.0, 0.0, -80.0)))
    cleaned, _, _ = separate_detached_fragments(mesh, "femur")
    np.testing.assert_array_equal(cleaned.vertices, body.vertices)
    np.testing.assert_array_equal(cleaned.faces, body.faces)


@pytest.mark.parametrize("centre", [(0.0, 0.0, 0.0),      # inside the bone
                                    (0.0, 0.0, 31.0)])    # 1.5 mm proud, within 2 mm
def test_a_speck_within_the_bones_extent_is_left_alone(centre):
    mesh = soup(bone(), speck(centre))
    cleaned, welded, report = separate_detached_fragments(mesh, "femur")
    assert cleaned is mesh
    assert not report.findings


def test_a_large_separate_piece_is_not_this_checks_to_remove():
    """A second bone-sized piece may be anatomy (a fibula, a patella) and is kept."""
    other = gm.uv_sphere(10.0, segments=24, rings=16)
    mesh = soup(bone(), Mesh(vertices=other.vertices + [0.0, 0.0, 100.0],
                             faces=other.faces))
    cleaned, _, report = separate_detached_fragments(mesh, "tibia")
    assert cleaned is mesh
    assert not report.findings


def test_a_clean_mesh_comes_back_untouched():
    mesh = soup(bone())
    welded = weld_vertices(mesh)
    cleaned, same_weld, report = separate_detached_fragments(mesh, "tibia", welded=welded)
    assert cleaned is mesh and same_weld is welded
    assert not report.findings
