"""Repair is what stands between a real segmentation and a kernel that wants manifolds."""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom import repair as gr


def test_a_welded_box_is_manifold():
    assert gr.is_manifold(gm.box(10.0)) is True


def test_an_unwelded_triangle_soup_is_not_manifold():
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    assert gr.is_manifold(soup) is False


def test_repair_welds_a_triangle_soup_into_a_manifold():
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    fixed, actions = gr.repair(soup)

    assert gr.is_manifold(fixed) is True
    assert "welded" in actions
    assert gm.volume(fixed) == pytest.approx(1000.0, rel=1e-9)


def test_repair_drops_a_degenerate_face():
    cube = gm.box(10.0)
    faces = np.vstack([cube.faces, [[0, 0, 1]]])
    broken = Mesh(vertices=cube.vertices, faces=faces)
    fixed, actions = gr.repair(broken)

    assert fixed.n_triangles == cube.n_triangles
    assert "dropped 1 degenerate face" in actions


def test_repair_drops_a_duplicated_face():
    cube = gm.box(10.0)
    faces = np.vstack([cube.faces, cube.faces[0:1]])
    broken = Mesh(vertices=cube.vertices, faces=faces)
    fixed, actions = gr.repair(broken)

    assert fixed.n_triangles == cube.n_triangles
    assert "dropped 1 duplicate face" in actions


def test_repair_reports_nothing_for_a_clean_mesh():
    fixed, actions = gr.repair(gm.box(10.0))
    assert actions == ()
    assert fixed.n_triangles == 12


def test_edge_report_counts_a_boundary():
    circle = gm.disc(5.0, segments=8)
    report = gr.edge_report(circle)
    assert report["boundary_edges"] == 8
    assert report["nonmanifold_edges"] == 0
