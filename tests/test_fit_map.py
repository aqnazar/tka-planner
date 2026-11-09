"""The fit as a colour map: the grid, the mesh drawn from it, and how it travels."""

import json
import struct

import numpy as np
import pytest

from tests.test_fit import fit, tibia_block, tray
from tka_planner.core.fit import section_fit
from tka_planner.geom.gltf import write_glb
from tka_planner.scene.fitmap import SCALE_MM, STOPS, colours_for, fit_map_mesh, legend
from tka_planner.scene.model import mesh_id


def kept(bone, component):
    return section_fit(cut="tibial_proximal", bone=bone, component=component,
                       point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), component_side=1.0,
                       lateral=(1.0, 0.0, 0.0), anterior=(0.0, -1.0, 0.0),
                       keep_grid=True)


def test_the_map_is_the_field_the_maxima_were_read_from():
    result = kept(tibia_block(), tray(shift=(0.0, -8.0)))
    signed = result.grid.signed_mm

    assert np.nanmax(-signed) == pytest.approx(result.max_overhang_mm)
    assert np.nanmax(signed) == pytest.approx(result.max_underhang_mm)
    # Seated cells read zero, and so many of them that most of the tray is seated.
    assert np.count_nonzero(signed == 0.0) * 0.25 ** 2 == pytest.approx(
        result.coverage_fraction * result.bone_area_mm2, rel=0.01)


def test_the_grid_is_not_kept_unless_asked_for():
    assert fit(tibia_block(), tray()).grid is None


def test_the_worst_overhang_is_found_where_it_is():
    """Slid 8 mm anteriorly (-Y), the tray overhangs the 50 mm block by 3 mm at its
    anterior edge, which lies at y = -28 on the cut."""
    result = kept(tibia_block(), tray(shift=(0.0, -8.0)))
    point = result.extreme_points_mm["max_overhang"]

    assert point[2] == pytest.approx(0.0)
    assert point[1] == pytest.approx(-28.0, abs=0.3)
    assert result.to_dict()["extreme_points_mm"]["overhang_anteromedial"] is not None


def test_the_drawn_map_carries_a_colour_and_a_value_per_vertex():
    result = kept(tibia_block(), tray(shift=(0.0, -8.0)))
    mesh = fit_map_mesh(result.grid)

    values = mesh.metadata["vertex_values_mm"]
    colours = mesh.metadata["vertex_colours"]
    assert len(values) == len(colours) == len(mesh.vertices)
    assert np.isfinite(values).all()
    assert mesh.faces.max() < len(mesh.vertices)
    # Lifted off the cut towards the tray, so it is not buried in the cut face.
    assert mesh.vertices[:, 2] == pytest.approx(0.3)
    assert values.min() == pytest.approx(-result.max_overhang_mm, abs=0.3)


def test_the_colour_scale_is_the_legend():
    for value, colour in STOPS:
        assert colours_for(np.array([value]))[0] == pytest.approx(colour)
    # Beyond the ends the end colour is drawn, as the legend says.
    assert colours_for(np.array([-40.0]))[0] == pytest.approx(STOPS[0][1])
    record = legend(2.0)
    assert record["min_mm"] == -SCALE_MM and record["max_mm"] == SCALE_MM
    assert record["flag_mm"] == -2.0
    assert [stop[0] for stop in record["stops"]] == [value for value, _ in STOPS]


def test_the_map_travels_as_gltf_colours_and_values():
    mesh = fit_map_mesh(kept(tibia_block(), tray(shift=(0.0, -8.0))).grid)
    data = write_glb(mesh)
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + length])
    attributes = document["meshes"][0]["primitives"][0]["attributes"]

    assert set(attributes) == {"POSITION", "COLOR_0", "_VALUE_MM"}
    assert document["accessors"][attributes["COLOR_0"]]["type"] == "VEC3"
    assert document["accessors"][attributes["_VALUE_MM"]]["count"] == len(mesh.vertices)
    offsets = [view["byteOffset"] for view in document["bufferViews"]]
    assert all(offset % 4 == 0 for offset in offsets)


def test_a_recoloured_map_is_a_different_mesh():
    """The viewer caches a mesh for ever under its id, so the colours are in the id."""
    mesh = fit_map_mesh(kept(tibia_block(), tray(shift=(0.0, -8.0))).grid)
    from tka_planner.core.meshio import Mesh

    recoloured = Mesh(mesh.vertices, mesh.faces, metadata={
        **mesh.metadata, "vertex_colours": mesh.metadata["vertex_colours"] * 0.5})
    assert mesh_id(recoloured) != mesh_id(mesh)


def test_an_empty_grid_draws_nothing():
    result = kept(tibia_block(), tray())
    empty = type(result.grid)(**{**result.grid.__dict__,
                                 "signed_mm": np.full_like(result.grid.signed_mm, np.nan)})
    assert fit_map_mesh(empty) is None
