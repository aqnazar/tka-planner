"""Decimation and remeshing for the metamorphic tests (paper experiment E2).

The resampler only earns its place if it changes the triangulation and nothing else:
the surface must stay where it was, the count must land where asked, and the same input
must always give the same output, or a result could not be reproduced from the commit.
"""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh, weld_vertices
from tka_planner.geom.mesh import surface_area, transformed, translation, uv_sphere, volume
from tka_planner.geom.resample import (
    as_triangle_soup,
    cluster_vertices,
    decimate,
    remesh,
    subdivide,
    weld,
)
from tka_planner.geom.repair import edge_report

RADIUS = 30.0


@pytest.fixture(scope="module")
def sphere():
    """A fine, closed sphere, read back as an STL would give it: a triangle soup."""
    return as_triangle_soup(uv_sphere(RADIUS, segments=96, rings=64))


def _radii(mesh):
    return np.linalg.norm(mesh.vertices, axis=1)


def test_weld_matches_the_core_weld(sphere):
    ours, core = weld(sphere), weld_vertices(sphere)
    assert np.array_equal(ours.vertices, core.vertices)
    assert np.array_equal(ours.faces, core.faces)


def test_a_soup_has_three_vertices_per_triangle_and_the_same_triangles(sphere):
    welded = weld(sphere)
    soup = as_triangle_soup(welded)
    assert soup.n_vertices == 3 * soup.n_triangles == 3 * welded.n_triangles
    assert np.allclose(soup.vertices[soup.faces], welded.vertices[welded.faces])


def test_subdivision_quadruples_the_triangles_without_moving_the_surface(sphere):
    welded = weld(sphere)
    fine = subdivide(welded)
    assert fine.n_triangles == 4 * welded.n_triangles
    assert surface_area(fine) == pytest.approx(surface_area(welded), rel=1e-12)
    assert volume(fine) == pytest.approx(volume(welded), rel=1e-12)
    report = edge_report(fine)
    assert report["boundary_edges"] == 0 and report["nonmanifold_edges"] == 0


@pytest.mark.parametrize("fraction", [0.5, 0.1])
def test_decimation_lands_on_the_target_count(sphere, fraction):
    target = int(fraction * sphere.n_triangles)
    result = decimate(sphere, target)
    assert result.n_triangles == pytest.approx(target, rel=0.03)


@pytest.mark.parametrize("fraction", [0.5, 0.1])
def test_decimation_keeps_the_surface_in_place(sphere, fraction):
    result = decimate(sphere, int(fraction * sphere.n_triangles))
    # Cell means sit inside a convex surface by at most the cell's sagitta.
    cell = result.metadata["cell_mm"]
    assert np.all(np.abs(_radii(result) - RADIUS) < cell)
    assert volume(result) == pytest.approx(volume(sphere), rel=0.02)


def test_decimation_is_deterministic(sphere):
    one, two = decimate(sphere, sphere.n_triangles // 4), decimate(sphere,
                                                                     sphere.n_triangles // 4)
    assert np.array_equal(one.vertices, two.vertices)
    assert np.array_equal(one.faces, two.faces)


def test_decimating_a_moved_copy_gives_the_moved_decimation(sphere):
    """The grid is anchored to the mesh, not to the scanner origin."""
    move = translation([137.25, -48.5, 912.0])
    moved_then_decimated = decimate(transformed(sphere, move), sphere.n_triangles // 4)
    decimated_then_moved = transformed(decimate(sphere, sphere.n_triangles // 4), move)
    assert np.array_equal(moved_then_decimated.faces, decimated_then_moved.faces)
    assert np.allclose(moved_then_decimated.vertices, decimated_then_moved.vertices,
                       atol=1e-9)


def test_clustering_drops_collapsed_and_duplicate_triangles():
    # Two triangles that collapse onto the same cell triple, and one that collapses
    # to a line.
    vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [0.1, 0.1, 0.0], [10.1, 0.1, 0.0], [0.1, 10.1, 0.0],
                         [0.2, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [3, 4, 5], [0, 3, 6]])
    result = cluster_vertices(Mesh(vertices=vertices, faces=faces), 1.0)
    assert result.n_triangles == 1
    assert result.n_vertices == 3


def test_remeshing_keeps_the_count_and_the_surface_but_moves_the_vertices(sphere):
    welded = weld(sphere)
    result = remesh(sphere)
    assert result.n_triangles == pytest.approx(welded.n_triangles, rel=0.03)
    assert volume(result) == pytest.approx(volume(welded), rel=0.01)
    assert np.all(np.abs(_radii(result) - RADIUS) < result.metadata["cell_mm"])
    # Fresh samples: almost no vertex of the result is a vertex of the original. A few
    # survive where a cell caught a single original vertex, which on a UV sphere
    # happens around the crowded poles.
    original = {tuple(v) for v in np.round(welded.vertices, 6)}
    shared = sum(tuple(v) in original for v in np.round(result.vertices, 6))
    assert shared < 0.05 * result.n_vertices


def test_bad_arguments_are_refused(sphere):
    with pytest.raises(ValueError):
        cluster_vertices(sphere, 0.0)
    with pytest.raises(ValueError):
        decimate(sphere, sphere.n_triangles)
