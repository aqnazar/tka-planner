"""Repair as far as repair can go, refusal beyond it, and one repair per bone.

The author's rule (2026-10-15): repair every bone as far as possible, non-manifold
edges included; refuse with a warning only what cannot be repaired. And repair each bone
once: a commit cut the same uncut bone twice, and repaired it each time.
"""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.core.qc import Severity, check_repairable
from tka_planner.geom import mesh as gm
from tka_planner.geom import repair as gr
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel


def sphere(radius=10.0):
    return gm.uv_sphere(radius, segments=32, rings=24)


def soup(*meshes) -> Mesh:
    vertices = np.concatenate([m.vertices[m.faces].reshape(-1, 3) for m in meshes])
    return Mesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3))


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    monkeypatch.setattr(gr, "_PREPARED", type(gr._PREPARED)())


def test_a_small_hole_is_closed_and_the_volume_kept():
    whole = sphere()
    holed = Mesh(vertices=whole.vertices, faces=np.delete(whole.faces, [100, 101], 0))
    fixed, actions = gr.repair(holed)
    assert gr.kernel_ready(fixed)
    assert any(a.startswith("closed 1 hole") for a in actions)
    assert gm.volume(fixed) == pytest.approx(gm.volume(whole), rel=2e-3)


def test_a_hole_wider_than_the_limit_is_left_open():
    whole = sphere(30.0)
    cap = np.flatnonzero(whole.vertices[whole.faces].mean(axis=1)[:, 2] > 25.0)
    holed = Mesh(vertices=whole.vertices, faces=np.delete(whole.faces, cap, 0))
    fixed, actions = gr.repair(holed)
    assert not gr.kernel_ready(fixed)
    assert any(a.startswith("left 1 opening") for a in actions)


def test_the_box_missing_a_face_is_refused_rather_than_filled():
    """Its open triangle is 14 mm across, wider than the 10 mm limit."""
    cube = gm.box(10.0)
    stats = gr.repair_statistics(Mesh(vertices=cube.vertices, faces=cube.faces[1:]))
    assert stats["accepted"] is False
    assert stats["boundary_edges"] == 3
    assert stats["holes_left_open"] == 1


def test_a_fin_on_a_non_manifold_edge_is_cut_out_and_the_hole_closed():
    whole = sphere()
    a, b = whole.faces[200][:2]
    tip = len(whole.vertices)
    vertices = np.vstack([whole.vertices, whole.vertices[[a, b]].mean(0) * 1.3])
    finned = Mesh(vertices=vertices, faces=np.vstack([whole.faces, [[a, b, tip]]]))
    assert gr.edge_report(finned)["nonmanifold_edges"] == 1

    fixed, actions = gr.repair(finned)
    assert gr.kernel_ready(fixed)
    assert any("non-manifold" in a for a in actions)
    assert gm.volume(fixed) == pytest.approx(gm.volume(whole), rel=5e-3)


def test_faces_wound_the_wrong_way_are_turned_back():
    whole = sphere()
    faces = whole.faces.copy()
    faces[::7] = faces[::7, ::-1]
    fixed, actions = gr.repair(Mesh(vertices=whole.vertices, faces=faces))
    assert gr.kernel_ready(fixed)
    assert any("winding" in a for a in actions)
    assert gm.volume(fixed) == pytest.approx(gm.volume(whole), rel=1e-9)


def test_an_inside_out_bone_is_turned_outward():
    whole = sphere()
    fixed, _ = gr.repair(Mesh(vertices=whole.vertices, faces=whole.faces[:, ::-1]))
    assert gm.volume(fixed) == pytest.approx(gm.volume(whole), rel=1e-9)


def test_collapsed_specks_no_longer_make_the_kernel_refuse_the_bone():
    """P007R's femur: two zero-size broken pieces inside an otherwise clean bone."""
    point = np.full((39, 3), [1.0, 2.0, 3.0])
    collapsed = Mesh(vertices=point, faces=np.array(
        [[0, 1, 2], [0, 1, 3], [0, 1, 4]] + [[i, i + 1, i + 2] for i in range(5, 36, 3)]))
    bone = soup(sphere(), collapsed)
    fixed, actions = gr.repair(bone)
    assert gr.kernel_ready(fixed)
    cut = ManifoldKernel().difference(bone, gm.box(30.0, gm.translation((0, 0, 20.0))))
    assert cut.mesh.vertices[:, 2].max() == pytest.approx(5.0, abs=1e-6)


def test_a_bone_is_prepared_once_however_often_it_is_cut(monkeypatch):
    monkeypatch.setattr(gr, "CACHE_MIN_TRIANGLES", 0)
    calls = []
    real = gr.repair
    monkeypatch.setattr(gr, "repair", lambda m, **k: calls.append(1) or real(m, **k))
    bone = soup(sphere())
    kernel = ManifoldKernel()
    for z in (5.0, 6.0, 7.0):
        kernel.difference(bone, gm.box(30.0, gm.translation((0, 0, 15.0 + z))))
    assert len(calls) == 1


def test_the_cache_tells_meshes_apart_by_content(monkeypatch):
    monkeypatch.setattr(gr, "CACHE_MIN_TRIANGLES", 0)
    one, other = soup(sphere(10.0)), soup(sphere(11.0))
    assert gr.prepare_for_kernel(one)[0] is not gr.prepare_for_kernel(other)[0]
    assert gr.prepare_for_kernel(one)[0] is gr.prepare_for_kernel(one)[0]


def test_quality_control_refuses_what_repair_cannot_close():
    whole = sphere(30.0)
    cap = np.flatnonzero(whole.vertices[whole.faces].mean(axis=1)[:, 2] > 25.0)
    report = check_repairable(
        Mesh(vertices=whole.vertices, faces=np.delete(whole.faces, cap, 0)), "femur")
    [finding] = report.findings
    assert finding.code == "MESH_UNREPAIRABLE"
    assert finding.severity is Severity.ERROR


def test_quality_control_reports_a_repair_it_made():
    whole = sphere()
    report = check_repairable(
        Mesh(vertices=whole.vertices, faces=np.delete(whole.faces, [100], 0)), "tibia")
    [finding] = report.findings
    assert finding.code == "MESH_REPAIRED"
    assert finding.severity is Severity.WARNING


def test_the_weld_made_at_opening_is_what_the_first_commit_cuts(monkeypatch):
    monkeypatch.setattr(gr, "CACHE_MIN_TRIANGLES", 0)
    calls = []
    real = gr.repair
    monkeypatch.setattr(gr, "repair", lambda m, **k: calls.append(1) or real(m, **k))
    bone = soup(sphere())
    welded, _ = real(bone)
    assert gr.remember_welded(bone, welded)
    prepared, actions, ready = gr.prepare_for_kernel(bone)
    assert prepared is welded and ready and actions == ("welded",)
    assert not calls


def test_a_welded_copy_that_is_not_ready_is_not_remembered(monkeypatch):
    monkeypatch.setattr(gr, "CACHE_MIN_TRIANGLES", 0)
    whole = sphere()
    holed = Mesh(vertices=whole.vertices, faces=whole.faces[1:])
    assert not gr.remember_welded(soup(holed), holed)
