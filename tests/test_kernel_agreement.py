"""The gate the port rests on: manifold3d must agree with Blender's exact solver.

The Blender side is not computed here. It was computed once inside Blender by
``tests/kernel_reference.py`` and committed as STL under
``tests/fixtures/kernel_reference/``, so this gate runs in an ordinary pytest session
with no Blender installed and no GPL dependency in the environment.

That is deliberate rather than merely convenient. Having ``bpy`` importable in the
working environment makes it far too easy to depend on Blender by accident, which is
the exact thing this port exists to stop. Committing the reference meshes also puts the
gate's evidence in the repository rather than on one machine.

Agreement is measured on the *solid*, never on its triangulation. The two solvers
triangulate the same surface differently, so vertex and triangle counts differ by
design and a vertex-to-vertex comparison would report that difference as an error. See
``tests/surface_distance.py``.

Regenerate the reference only when the cases change:

    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
        --background --factory-startup --python tests/kernel_reference.py
"""

from pathlib import Path

import pytest

from tests.kernel_reference import cases
from tests.surface_distance import hausdorff
from tka_planner.core.meshio import read_stl
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel

REFERENCE = Path(__file__).parent / "fixtures" / "kernel_reference"

# A tenth of a micron. Far below any tolerance that could matter to a resection, and
# far tighter than the 0.05 mm the design document asked for. The two solvers agree
# this closely in practice, so there is no reason to accept less.
TOLERANCE_MM = 1e-4


@pytest.fixture(params=list(cases()), ids=lambda case: case[0])
def agreed(request):
    """One case cut by manifold3d, beside Blender's committed result for it."""
    name, operation, target, tool = request.param
    reference_path = REFERENCE / f"{name}.stl"
    if not reference_path.is_file():
        pytest.skip(
            f"No Blender reference for {name}. Regenerate with "
            f"tests/kernel_reference.py inside Blender."
        )
    ours = getattr(ManifoldKernel(), operation)(target, tool).mesh
    return ours, read_stl(reference_path)


def test_the_two_kernels_agree_on_volume(agreed):
    ours, theirs = agreed
    assert gm.volume(ours) == pytest.approx(gm.volume(theirs), rel=1e-7)


def test_the_two_kernels_agree_on_surface_area(agreed):
    ours, theirs = agreed
    assert gm.surface_area(ours) == pytest.approx(gm.surface_area(theirs), rel=1e-7)


def test_the_two_kernels_agree_on_the_bounding_box(agreed):
    ours, theirs = agreed
    our_low, our_high = ours.bounds
    their_low, their_high = theirs.bounds

    assert our_low == pytest.approx(their_low, abs=TOLERANCE_MM)
    assert our_high == pytest.approx(their_high, abs=TOLERANCE_MM)


def test_the_two_surfaces_are_the_same_surface(agreed):
    """The real gate: no point on either surface is far from the other surface."""
    ours, theirs = agreed
    assert hausdorff(ours, theirs) < TOLERANCE_MM


def test_our_result_is_a_clean_manifold(agreed):
    """Whatever Blender returns, our own output must be usable as the next cut's input."""
    from tka_planner.geom.repair import edge_report

    ours, _ = agreed
    report = edge_report(ours)

    assert report["degenerate_faces"] == 0
    assert report["nonmanifold_edges"] == 0
    assert report["boundary_edges"] == 0


def test_the_notched_box_matches_its_analytic_surface_area(agreed, request):
    """A closed form for one case, so the gate is not purely a comparison.

    Cutting a 23 mm box out through one face of a 40 mm cube removes 23x23 from that
    face and exposes the tool's other five faces inside it.
    """
    if request.node.callspec.id != "box_difference":
        pytest.skip("Only the box case has a closed form.")

    ours, _ = agreed
    expected = 6 * 40.0**2 - 23.0**2 + 23.0**2 + 4 * 13.5 * 23.0
    assert gm.surface_area(ours) == pytest.approx(expected, rel=1e-9)
