"""The pieces E8 and E9 are built from, checked on synthetic geometry.

The experiments themselves run on the cohort, which CI does not have. What they rest on
does not need it: the grid-accelerated surface distance must give the brute-force
answer, the defect catalogue must keep refusing what it refuses, and the determinism
hash table must be stable and in the shape CI compares.
"""

import json

import numpy as np
import pytest

from paper import determinism
from paper.experiments.e08_kernel import compare, kernel_verdict, refusal_catalogue
from paper.experiments.e09_performance import TimedKernel
from paper.harness import cpu_busy_percent, cpu_name, memory_gb
from paper.surface_distance import closest_points, hausdorff, point_to_surface
from tests.surface_distance import closest_point_on_triangles
from tests.surface_distance import point_to_surface as brute_point_to_surface
from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel


def _cut_sphere():
    sphere = gm.uv_sphere(20.0, segments=32, rings=16)
    tool = gm.box(100.0, gm.plane_matrix((0.0, 0.0, 0.0), (0.3, 0.4, 0.87))
                  @ gm.translation((0.0, 0.0, -50.0)))
    return ManifoldKernel().difference(sphere, tool).mesh


# ----------------------------------------------------------------------
# Surface distance
# ----------------------------------------------------------------------


def test_pairwise_closest_points_match_the_all_pairs_routine():
    rng = np.random.default_rng(3)
    corners = rng.normal(size=(200, 3, 3)) * 5.0
    points = rng.normal(size=(200, 3)) * 8.0

    all_pairs = closest_point_on_triangles(points, corners)
    pairwise = closest_points(points, corners)

    assert np.allclose(pairwise, all_pairs[np.arange(200), np.arange(200)], atol=1e-12)


def test_the_grid_gives_the_brute_force_distance():
    """Near and far points alike: a far point must widen its search, not be dropped."""
    rng = np.random.default_rng(4)
    mesh = _cut_sphere()
    points = np.vstack([rng.normal(size=(300, 3)) * 15.0,
                        rng.normal(size=(20, 3)) * 200.0,
                        mesh.vertices[:50] + 1e-3])

    assert np.allclose(point_to_surface(points, mesh),
                       brute_point_to_surface(points, mesh), atol=1e-12)


def test_a_surface_is_at_zero_hausdorff_from_itself():
    mesh = _cut_sphere()
    assert hausdorff(mesh, mesh)["hausdorff_mm"] == 0.0


def test_a_shifted_surface_is_at_its_shift():
    mesh = _cut_sphere()
    moved = Mesh(vertices=mesh.vertices + [0.0, 0.0, 0.01], faces=mesh.faces)

    distance = hausdorff(mesh, moved)["hausdorff_mm"]
    assert distance == pytest.approx(0.01, rel=1e-6)


def test_a_retriangulated_surface_is_the_same_surface():
    """Splitting every face in four changes the triangulation and nothing else."""
    mesh = gm.box(10.0)
    v, f = mesh.vertices, mesh.faces
    mids = (v[f[:, [0, 1, 2]]] + v[f[:, [1, 2, 0]]]) / 2.0
    vertices = np.vstack([v, mids.reshape(-1, 3)])
    n, k = len(v), np.arange(len(f)) * 3
    ab, bc, ca = n + k, n + k + 1, n + k + 2
    faces = np.vstack([np.stack([f[:, 0], ab, ca], 1), np.stack([ab, f[:, 1], bc], 1),
                       np.stack([ca, bc, f[:, 2]], 1), np.stack([ab, bc, ca], 1)])

    assert hausdorff(mesh, Mesh(vertices=vertices, faces=faces))["hausdorff_mm"] < 1e-12


def test_single_precision_rounding_is_bounded_not_hidden():
    """Shared vertices are paired after rounding to float32; the distance kept is the
    real one, so rounding shows up instead of reading as zero."""
    mesh = Mesh(vertices=gm.box(10.0).vertices + 1000.1, faces=gm.box(10.0).faces)
    rounded = Mesh(vertices=mesh.vertices.astype(np.float32).astype(float),
                   faces=mesh.faces)
    result = hausdorff(mesh, rounded)

    assert 0.0 < result["hausdorff_mm"] < 1e-4
    assert result["samples_measured"] == 0


# ----------------------------------------------------------------------
# E8
# ----------------------------------------------------------------------


def test_identical_results_compare_as_agreeing():
    mesh = _cut_sphere()
    row = compare(mesh, mesh)

    assert row["agree"] is True
    assert row["volume_rel_diff"] == 0.0
    assert row["manifold_output_manifold"] is True


def test_the_kernel_refuses_a_hole_and_accepts_a_soup():
    cube = gm.box(10.0)
    soup = Mesh(vertices=cube.vertices[cube.faces].reshape(-1, 3),
                faces=np.arange(36).reshape(-1, 3))

    assert kernel_verdict(soup) == (True, "")
    accepted, reason = kernel_verdict(Mesh(vertices=cube.vertices, faces=cube.faces[1:]))
    assert accepted is False and "not manifold" in reason


def test_the_defect_catalogue_refuses_what_cannot_be_repaired():
    rows = {row["defect"]: row for row in refusal_catalogue(None)}

    for refused in ("hole (one face missing)", "open surface (a disc)",
                    "non-manifold edge (cubes on an edge)", "one face flipped"):
        assert rows[refused]["manifold3d"] == "refused"
    for repaired in ("triangle soup (unwelded STL)", "degenerate face",
                     "duplicated face"):
        assert rows[repaired]["manifold3d"] == "accepted"
        assert rows[repaired]["manifold3d_mm3"] == pytest.approx(
            rows[repaired]["expected_mm3"])


# ----------------------------------------------------------------------
# E9
# ----------------------------------------------------------------------


def test_the_timed_kernel_logs_each_boolean_and_changes_nothing():
    kernel = TimedKernel()
    target, tool = gm.box(10.0), gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.difference(target, tool).mesh
    kernel.intersect(target, tool)

    assert [op for op, _ in kernel.log] == ["difference", "intersect"]
    assert gm.volume(result) == pytest.approx(
        gm.volume(ManifoldKernel().difference(target, tool).mesh))


def test_the_machine_is_described():
    assert cpu_name()
    memory = memory_gb()
    if memory:
        assert 0 < memory["ram_available_gb"] <= memory["ram_total_gb"]
    busy = cpu_busy_percent(0.05)
    assert busy is None or 0.0 <= busy <= 100.0


@pytest.fixture(scope="module")
def synthetic_table():
    case, measurement = determinism.synthetic_case()
    return determinism.hash_table(case, measurement, determinism.library_root())


def test_the_synthetic_plan_hashes_the_same_twice_in_one_process(synthetic_table):
    case, measurement = determinism.synthetic_case()
    again = determinism.hash_table(case, measurement, determinism.library_root())
    assert again == synthetic_table


def test_the_hash_table_has_the_committed_shape(synthetic_table):
    """CI compares against this file, so the names must match even where the hashes
    of another platform may not."""
    committed = json.loads(determinism.REFERENCE_TABLE.read_text(encoding="utf-8"))
    assert list(committed["hashes"]) == list(synthetic_table)
    assert committed["case"] == determinism.SYNTHETIC_ID


def test_the_plan_file_has_no_platform_line_endings():
    case, measurement = determinism.synthetic_case()
    document = determinism.plan_document_for(measurement, determinism.library_root())
    assert b"\r" not in determinism.plan_file_bytes(document)


def test_compare_names_what_differs_and_what_is_missing():
    verdict = dict(determinism.compare({"a": "1", "b": "2", "c": "3"},
                                       {"a": "1", "b": "X", "d": "4"}))
    assert verdict == {"a": True, "b": False, "c": None, "d": None}


def test_the_determinism_command_compares_and_reports(tmp_path, capsys):
    table = tmp_path / "table.json"
    assert determinism.main(["--synthetic", "--no-commit", "--json", str(table)]) == 0
    assert determinism.main(["--synthetic", "--no-commit", "--compare", str(table)]) == 0
    assert "identical" in capsys.readouterr().out

    altered = json.loads(table.read_text(encoding="utf-8"))
    altered["hashes"]["plan.json"] = "0" * 64
    table.write_text(json.dumps(altered), encoding="utf-8")
    assert determinism.main(["--synthetic", "--no-commit", "--compare", str(table)]) == 1
