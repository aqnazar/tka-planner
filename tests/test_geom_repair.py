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


def _soup(mesh):
    return Mesh(
        vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
        faces=np.arange(mesh.n_triangles * 3).reshape(-1, 3),
    )


def _two_boxes():
    a, b = gm.box(10.0), gm.box(10.0, gm.translation((30.0, 0.0, 0.0)))
    return Mesh(vertices=np.vstack([a.vertices, b.vertices]),
                faces=np.vstack([a.faces, b.faces + a.n_vertices]))


def test_one_closed_surface_is_one_component():
    assert gr.connected_components(gm.uv_sphere(5.0, segments=16, rings=8)) == 1


def test_two_separate_boxes_are_two_components():
    assert gr.connected_components(_two_boxes()) == 2


def test_a_soup_counts_every_triangle_as_a_component():
    assert gr.connected_components(_soup(gm.box(10.0))) == 12


def test_components_ignore_vertices_no_face_uses():
    cube = gm.box(10.0)
    padded = Mesh(vertices=np.vstack([cube.vertices, [[99.0, 0.0, 0.0]]]),
                  faces=cube.faces)
    assert gr.connected_components(padded) == 1


def test_components_join_a_long_chain():
    """A strip whose vertex order runs against the hooking order still joins up."""
    n = 2000
    top = np.arange(n)[::-1]
    faces = np.array([[top[i], top[i + 1], n + i] for i in range(n - 1)])
    strip = Mesh(vertices=np.zeros((2 * n, 3)), faces=faces)
    assert gr.connected_components(strip) == 1


def test_repair_statistics_count_a_welded_soup():
    stats = gr.repair_statistics(_soup(gm.box(10.0)))

    assert stats["vertices_in"] == 36
    assert stats["vertices_merged"] == 36 - 8
    assert stats["vertices_out"] == 8
    assert stats["degenerate_faces_removed"] == 0
    assert stats["components"] == 1
    assert stats["accepted"] is True
    assert stats["actions"] == ["welded"]


def test_repair_statistics_count_dropped_faces():
    cube = gm.box(10.0)
    faces = np.vstack([cube.faces, [[0, 0, 1]], cube.faces[0:2]])
    stats = gr.repair_statistics(Mesh(vertices=cube.vertices, faces=faces))

    assert stats["degenerate_faces_removed"] == 1
    assert stats["duplicate_faces_removed"] == 2
    assert stats["triangles_out"] == 12
    assert stats["accepted"] is True


def test_repair_statistics_agree_with_repair():
    cube = gm.box(10.0)
    broken = Mesh(vertices=cube.vertices, faces=np.vstack([cube.faces, [[2, 2, 3]]]))
    fixed, actions = gr.repair(broken)
    stats = gr.repair_statistics(broken)

    assert stats["actions"] == list(actions)
    assert stats["triangles_out"] == fixed.n_triangles
    assert stats["accepted"] == gr.is_manifold(fixed)
