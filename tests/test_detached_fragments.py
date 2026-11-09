"""Detached pieces of a bone mesh: those too small to be anatomy are removed, the rest
reported with an option to remove them.

One cohort tibia carried an 80-triangle sliver 37 mm above its plateau. Every tibial
estimator searches below the highest vertex, so the sliver became the top of the tibia,
the spines were placed on it, and the knee was refused at the tibial frame. One cohort
femur carried two collapsed specks inside it, broken surfaces of zero size that made
the boolean kernel refuse an otherwise clean bone.
"""

import numpy as np
import pytest

from tka_planner.core.islands import (
    DEBRIS_LONGEST_SIDE_MM,
    DEBRIS_VOLUME_CM3,
    find_islands,
    is_debris,
)
from tka_planner.core.meshio import Mesh, face_components, weld_vertices
from tka_planner.core.qc import Severity, clean_islands
from tka_planner.geom import mesh as gm


def soup(*meshes) -> Mesh:
    """Meshes joined as STL delivers them: every triangle owning its own vertices."""
    vertices = np.concatenate([m.vertices[m.faces].reshape(-1, 3) for m in meshes])
    return Mesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3))


def bone():
    return gm.uv_sphere(30.0, segments=48, rings=40)


def speck(centre, size=1.0):
    return gm.box(size, gm.translation(centre))


def shifted(mesh, offset):
    return Mesh(vertices=mesh.vertices + np.asarray(offset, float), faces=mesh.faces)


def test_pieces_are_labelled_largest_first():
    mesh = weld_vertices(soup(speck((0.0, 0.0, 70.0)), bone()))
    labels = face_components(mesh)
    assert labels.max() == 1
    assert np.sum(labels == 1) == 12        # the box
    assert np.sum(labels == 0) == len(bone().faces)


def test_an_unwelded_soup_is_all_separate_triangles():
    labels = face_components(soup(bone()))
    assert len(np.unique(labels)) == len(labels)


def test_a_piece_is_measured_along_its_own_axes():
    """A thin plate turned 45 degrees is still thin: extents are not the scanner's."""
    plate = gm.box(1.0)
    plate = Mesh(vertices=plate.vertices * [20.0, 2.0, 10.0], faces=plate.faces)
    turn = np.array([[np.cos(0.8), -np.sin(0.8), 0], [np.sin(0.8), np.cos(0.8), 0],
                     [0, 0, 1.0]])
    plate = Mesh(vertices=plate.vertices @ turn.T + [0, 0, 80.0], faces=plate.faces)
    _, [piece] = find_islands(weld_vertices(soup(bone(), plate)))
    assert piece.extents_mm == pytest.approx((2.0, 10.0, 20.0), abs=1e-6)
    assert piece.closed
    assert is_debris(piece)         # 0.4 cm3: light, though as long as the limit


def test_a_long_light_sliver_is_debris_by_volume():
    long_sliver = gm.box(1.0)
    long_sliver = Mesh(vertices=long_sliver.vertices * [1.0, 1.0, 40.0],
                       faces=long_sliver.faces)
    _, [piece] = find_islands(weld_vertices(soup(bone(), shifted(long_sliver,
                                                                 (0, 0, 90.0)))))
    assert piece.extents_mm[2] > DEBRIS_LONGEST_SIDE_MM
    assert piece.volume_cm3 < DEBRIS_VOLUME_CM3
    assert is_debris(piece)


def test_the_limits_are_the_authors():
    assert (DEBRIS_LONGEST_SIDE_MM, DEBRIS_VOLUME_CM3) == (20.0, 5.0)


def test_a_compact_closed_piece_under_the_volume_is_debris_whatever_its_length():
    """A 19 mm cube: short on every side, so debris on both counts."""
    cube = gm.box(19.0)
    _, [piece] = find_islands(weld_vertices(soup(bone(), shifted(cube, (0, 0, 90.0)))))
    assert is_debris(piece)


def test_a_thick_piece_over_both_limits_is_not_debris():
    block = gm.box(1.0)
    block = Mesh(vertices=block.vertices * [20.0, 20.0, 25.0], faces=block.faces)
    _, [piece] = find_islands(weld_vertices(soup(bone(), shifted(block, (0, 0, 90.0)))))
    assert piece.volume_cm3 == pytest.approx(10.0, abs=1e-6)
    assert not is_debris(piece)


def test_an_open_sheet_is_judged_by_its_length_alone():
    """Half a cube is open: its signed volume is meaningless, so only length counts."""
    sheet = gm.box(1.0)
    sheet = Mesh(vertices=sheet.vertices * [60.0, 60.0, 1.0], faces=sheet.faces[:-2])
    _, [piece] = find_islands(weld_vertices(soup(bone(), shifted(sheet, (0, 0, 90.0)))))
    assert not piece.closed
    assert piece.extents_mm[2] > DEBRIS_LONGEST_SIDE_MM
    assert not is_debris(piece)


def test_a_cavity_is_judged_by_the_size_of_its_volume_not_its_sign():
    small = gm.uv_sphere(5.0, segments=24, rings=16)      # 0.5 cm3, inward
    small = Mesh(vertices=small.vertices * 4.0, faces=small.faces[:, ::-1])   # 20 mm radius
    _, [piece] = find_islands(weld_vertices(soup(bone(), small)))
    assert piece.volume_cm3 < 0 and abs(piece.volume_cm3) > DEBRIS_VOLUME_CM3
    assert not is_debris(piece)


def test_a_speck_beyond_the_bone_is_removed_with_a_warning():
    mesh = soup(bone(), speck((5.0, -20.0, 67.0)))
    cleaned, welded, report = clean_islands(mesh, "tibia")

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
    cleaned, _, _ = clean_islands(mesh, "femur")
    np.testing.assert_array_equal(cleaned.vertices, body.vertices)
    np.testing.assert_array_equal(cleaned.faces, body.faces)


@pytest.mark.parametrize("centre", [(0.0, 0.0, 0.0),      # inside the bone
                                    (0.0, 0.0, 31.0)])    # 1.5 mm proud, within 2 mm
def test_a_speck_within_the_bones_extent_is_removed_with_a_note(centre):
    mesh = soup(bone(), speck(centre))
    cleaned, _, report = clean_islands(mesh, "femur")
    assert len(cleaned.faces) == len(bone().faces)
    [finding] = report.findings
    assert finding.code == "MESH_DETACHED_FRAGMENT"
    assert finding.severity is Severity.INFO


def test_a_collapsed_speck_of_zero_size_is_removed():
    """The P007R femur: pieces whose every vertex sits at one point."""
    point = np.full((36, 3), [5.0, 2.0, -3.0])
    collapsed = Mesh(vertices=point, faces=np.arange(36).reshape(-1, 3))
    cleaned, _, report = clean_islands(soup(bone(), collapsed), "femur")
    assert len(cleaned.faces) == len(bone().faces)
    assert report.findings[0].severity is Severity.INFO


def test_a_piece_over_the_limits_is_kept_and_reported_where_it_is():
    other = gm.uv_sphere(20.0, segments=24, rings=16)
    mesh = soup(bone(), shifted(other, (0.0, 0.0, 100.0)))
    cleaned, _, report = clean_islands(mesh, "tibia")
    assert cleaned is mesh
    [finding] = report.findings
    assert finding.code == "MESH_LARGE_ISLAND"
    assert finding.severity is Severity.WARNING
    assert finding.context["removable"] is True
    assert finding.context["location_mm"] == pytest.approx([0.0, 0.0, 100.0], abs=0.5)
    assert finding.context["outside_mm"] == pytest.approx(90.0, abs=0.5)


def test_a_large_cavity_inside_the_bone_is_kept_and_called_internal():
    cavity = gm.uv_sphere(15.0, segments=24, rings=16)
    cavity = Mesh(vertices=cavity.vertices, faces=cavity.faces[:, ::-1])   # inward
    mesh = soup(bone(), cavity)
    _, _, report = clean_islands(mesh, "femur")
    [finding] = report.findings
    assert finding.code == "MESH_LARGE_ISLAND"
    assert "inside the bone" in finding.message
    assert finding.context["volume_cm3"] < 0


def test_large_pieces_go_when_asked():
    other = gm.uv_sphere(20.0, segments=24, rings=16)
    mesh = soup(bone(), shifted(other, (0.0, 0.0, 100.0)), speck((0, 0, 0)))
    cleaned, _, report = clean_islands(mesh, "tibia", remove_large=True)
    assert len(cleaned.faces) == len(bone().faces)
    codes = sorted(f.code for f in report.findings)
    assert codes == ["MESH_DETACHED_FRAGMENT", "MESH_LARGE_ISLAND_REMOVED"]


def test_a_clean_mesh_comes_back_untouched():
    mesh = soup(bone())
    welded = weld_vertices(mesh)
    cleaned, same_weld, report = clean_islands(mesh, "tibia", welded=welded)
    assert cleaned is mesh and same_weld is welded
    assert not report.findings
