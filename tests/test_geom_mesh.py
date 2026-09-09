"""Mesh transforms, primitives and measures, all in millimetres."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm


def test_box_volume_matches_its_size():
    cube = gm.box(10.0)
    assert gm.volume(cube) == pytest.approx(1000.0, rel=1e-9)


def test_box_surface_area_matches_its_size():
    cube = gm.box(10.0)
    assert gm.surface_area(cube) == pytest.approx(600.0, rel=1e-9)


def test_box_accepts_unequal_sides():
    slab = gm.box((2.0, 3.0, 4.0))
    assert gm.volume(slab) == pytest.approx(24.0, rel=1e-9)


def test_transform_translates_without_changing_volume():
    cube = gm.box(10.0)
    matrix = np.eye(4)
    matrix[:3, 3] = (5.0, -2.0, 7.0)
    moved = gm.transformed(cube, matrix)

    assert gm.volume(moved) == pytest.approx(1000.0, rel=1e-9)
    assert moved.vertices.mean(axis=0) == pytest.approx(
        cube.vertices.mean(axis=0) + matrix[:3, 3]
    )


def test_transform_leaves_faces_alone():
    cube = gm.box(10.0)
    moved = gm.transformed(cube, np.eye(4))
    assert np.array_equal(moved.faces, cube.faces)


def test_rotation_to_takes_z_onto_the_direction():
    direction = np.array([0.0, 1.0, 0.0])
    rotation = gm.rotation_to(direction)
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(direction)


def test_rotation_to_is_orthonormal():
    rotation = gm.rotation_to(np.array([0.3, -0.5, 0.8]))
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_plane_matrix_puts_z_on_the_normal_and_origin_on_the_point():
    point = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 1.0])
    matrix = gm.plane_matrix(point, normal)

    assert matrix[:3, 3] == pytest.approx(point)
    assert matrix[:3, 2] == pytest.approx(normal)


def test_cylinder_volume_approaches_the_analytic_value():
    tube = gm.cylinder(2.0, 10.0, segments=256)
    assert gm.volume(tube) == pytest.approx(np.pi * 4.0 * 10.0, rel=1e-3)


def test_sphere_volume_approaches_the_analytic_value():
    ball = gm.uv_sphere(3.0, segments=128, rings=64)
    assert gm.volume(ball) == pytest.approx(4.0 / 3.0 * np.pi * 27.0, rel=1e-2)


def test_slab_volume_matches_its_footprint_and_thickness():
    block = gm.slab(60.0, 40.0, 9.0)
    assert gm.volume(block) == pytest.approx(60.0 * 40.0 * 9.0, rel=1e-9)


def test_slab_sits_on_z_zero():
    block = gm.slab(60.0, 40.0, 9.0)
    assert block.vertices[:, 2].min() == pytest.approx(0.0)
    assert block.vertices[:, 2].max() == pytest.approx(9.0)


def test_disc_lies_in_its_own_plane():
    circle = gm.disc(55.0, segments=64)
    assert circle.vertices[:, 2] == pytest.approx(np.zeros(circle.n_vertices))


def test_write_stl_round_trips(tmp_path):
    from tka_planner.core.meshio import read_stl

    cube = gm.box(10.0)
    path = gm.write_stl(cube, tmp_path / "cube.stl")
    reloaded = read_stl(path)

    assert gm.volume(reloaded) == pytest.approx(1000.0, rel=1e-5)
