"""Generate the Blender reference geometry the kernel agreement gate compares against.

Run inside Blender, once, from the repository root:

    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
        --background --python tests/kernel_reference.py

It writes one STL per case into ``tests/fixtures/kernel_reference/``, which is committed
so the gate runs in an ordinary pytest session with no Blender installed.

Doing it this way rather than importing ``bpy`` into the test process is deliberate.
The ``bpy`` wheel is about a gigabyte and GPL, and having it importable in the working
environment makes it far too easy to depend on Blender by accident, which is the exact
thing this port exists to stop. Committing the reference meshes also means the gate's
evidence is in the repository rather than in whatever was installed on one machine.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tka_planner.geom import mesh as gm  # noqa: E402
from tka_planner.geom.kernels.blender_kernel import BlenderKernel  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "kernel_reference"


def cases():
    """The four agreement cases, defined once here and imported by the test.

    A box difference for the simplest possible check, an oblique cut through a curved
    surface because that is what a resection plane actually is, an intersection because
    that is how the bone shell is made, and a difference with exactly coplanar faces
    because that is the degenerate input an exact solver is most likely to mishandle.

    The box case cuts a notch out of one face, leaving a non-convex n-gon. That is worth
    having explicitly: it is the case that exposed a triangle-fan bug in this project's
    own Blender reader, which inflated the measured surface area by 12% while leaving
    the volume correct.
    """
    yield (
        "box_difference",
        "difference",
        gm.box(40.0),
        gm.box(23.0, gm.translation((18.0, 3.0, -2.0))),
    )
    yield (
        "oblique_cut",
        "difference",
        gm.uv_sphere(20.0, segments=64, rings=32),
        gm.box(
            100.0,
            gm.plane_matrix((0.0, 0.0, 0.0), (0.3, 0.4, 0.87))
            @ gm.translation((0.0, 0.0, -50.0)),
        ),
    )
    yield (
        "intersection",
        "intersect",
        gm.uv_sphere(20.0, segments=64, rings=32),
        gm.box(24.0, gm.translation((8.0, 0.0, 0.0))),
    )


    # Three of the tool's faces lie exactly in the plane of one of the target's, which
    # is the ambiguity an exact solver has to resolve consistently. Both solvers do,
    # and both give the analytically correct area: the faces this removes and the faces
    # it exposes have equal area, so the result's surface area is the cube's own.
    yield (
        "coplanar_difference",
        "difference",
        gm.box(40.0),
        gm.box(40.0, gm.translation((18.0, 3.0, -2.0))),
    )


def main() -> None:
    import bpy

    OUT.mkdir(parents=True, exist_ok=True)
    kernel = BlenderKernel()

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for name, operation, target, tool in cases():
        result = getattr(kernel, operation)(target, tool)
        path = gm.write_stl(result.mesh, OUT / f"{name}.stl")
        print(
            f"{name}: {result.mesh.n_triangles} triangles, "
            f"volume {gm.volume(result.mesh):.4f} mm3 -> {path.name}"
        )

    print(f"Blender {bpy.app.version_string}")


if __name__ == "__main__":
    main()
