"""The boolean kernel, and the record it keeps of what it had to do."""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernel import default_kernel
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel


@pytest.fixture
def kernel():
    return ManifoldKernel()


def test_default_kernel_is_manifold3d():
    assert default_kernel().name == "manifold3d"


def test_difference_of_a_box_and_a_half_covering_box(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.difference(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert result.record.backend == "manifold3d"
    assert result.record.operation == "difference"


def test_intersection_of_two_overlapping_boxes(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.intersect(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert result.record.operation == "intersect"


def test_difference_with_a_disjoint_tool_leaves_the_target_alone(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((100.0, 0.0, 0.0)))
    result = kernel.difference(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(1000.0, rel=1e-6)


def test_a_triangle_soup_input_is_repaired_and_the_repair_is_recorded(kernel):
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.difference(soup, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert "welded" in result.record.repaired_target
    assert result.record.repaired_tool == ()


def test_the_record_serialises_for_the_plan_file(kernel):
    result = kernel.difference(gm.box(10.0), gm.box(4.0))
    record = result.record.to_dict()

    assert record["backend"] == "manifold3d"
    assert record["operation"] == "difference"
    assert record["fallback"] is None
    assert isinstance(record["repaired_target"], list)


def test_an_unrepairable_target_raises_with_a_useful_message(kernel):
    open_disc = gm.disc(5.0, segments=8)
    with pytest.raises(ValueError, match="not manifold"):
        kernel.difference(open_disc, gm.box(1.0))


def test_a_cut_preserves_millimetre_precision(kernel):
    """Coordinates run to hundreds of millimetres, so float32 would lose real digits."""
    target = gm.box(10.0, gm.translation((300.0, 0.0, 0.0)))
    tool = gm.box(10.0, gm.translation((305.0, 0.0, 0.0)))
    result = kernel.difference(target, tool)

    assert result.mesh.vertices.dtype == np.float64
    assert result.mesh.vertices[:, 0].min() == pytest.approx(295.0, abs=1e-9)


def test_a_cut_result_can_be_cut_again(kernel):
    """A second commit cuts an already-cut bone, so results must be re-usable inputs.

    manifold3d hands its output back as read-only views onto its own buffers, which
    nanobind then refuses on the way back in. The kernel copies out at the boundary.
    """
    once = kernel.difference(
        gm.uv_sphere(20.0, segments=32, rings=24),
        gm.box(20.0, gm.translation((15.0, 0.0, 0.0))),
    ).mesh
    twice = kernel.difference(
        once, gm.box(20.0, gm.translation((0.0, 15.0, 0.0)))
    ).mesh

    assert gm.volume(twice) < gm.volume(once)


def test_a_cut_result_owns_writable_data(kernel):
    result = kernel.difference(gm.box(10.0), gm.box(4.0)).mesh

    assert result.vertices.flags["WRITEABLE"]
    assert result.faces.flags["WRITEABLE"]
