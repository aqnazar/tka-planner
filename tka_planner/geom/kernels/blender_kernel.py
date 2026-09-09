"""Blender's exact solver, wrapped as a kernel, for cross-checking only.

Blender is GPL and this project is Apache 2.0, so this backend is never the default,
never imported at runtime, and lives behind the ``blender`` optional extra. It exists
so the agreement test can show that ``manifold3d`` produces the same geometry as the
solver the published results were computed with.

``bpy`` is imported inside the methods rather than at module scope, so importing this
module on a machine with no Blender costs nothing and fails only when it is used.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh

from ..kernel import KernelRecord, KernelResult

__all__ = ["BlenderKernel"]


class BlenderKernel:
    """Difference and intersection through Blender's EXACT boolean modifier."""

    name = "blender-exact"

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "difference", "DIFFERENCE")

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "intersect", "INTERSECT")

    # ------------------------------------------------------------------

    def _operate(self, target: Mesh, tool: Mesh, operation: str,
                 blender_operation: str) -> KernelResult:
        import bpy

        target_object = self._to_object(target, "kernel_target")
        tool_object = self._to_object(tool, "kernel_tool")

        modifier = target_object.modifiers.new(name="Kernel", type="BOOLEAN")
        modifier.operation = blender_operation
        modifier.object = tool_object
        modifier.solver = "EXACT"

        bpy.context.view_layer.objects.active = target_object
        bpy.ops.object.modifier_apply(modifier=modifier.name)

        result = self._from_object(target_object, target)
        for obj in (target_object, tool_object):
            bpy.data.objects.remove(obj, do_unlink=True)

        return KernelResult(
            mesh=result,
            record=KernelRecord(
                backend=self.name, operation=operation, fallback=None,
                notes=("solver=EXACT",),
            ),
        )

    @staticmethod
    def _to_object(mesh: Mesh, name: str):
        """Build a Blender object in millimetres.

        The scene units are irrelevant here because nothing is rendered and nothing is
        exported; the kernel takes millimetres in and gives millimetres out.
        """
        import bpy

        data = bpy.data.meshes.new(name)
        data.from_pydata(
            [tuple(float(c) for c in v) for v in mesh.vertices],
            [],
            [tuple(int(i) for i in f) for f in mesh.faces],
        )
        data.update()
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    @staticmethod
    def _from_object(obj, like: Mesh) -> Mesh:
        """Read the object back as triangles, using Blender's own triangulation.

        A boolean leaves n-gons, and a difference routinely leaves *non-convex* ones:
        cutting a notch out of a face's edge produces a polygon a triangle fan cannot
        represent. Fanning it anyway yields overlapping triangles whose signed areas
        cancel, so the volume still comes out right and the surface is still in the
        right place, while the unsigned surface area is inflated by twice the overlap.
        That is a silent error of the exact kind this project keeps guarding against,
        and it was measured here as a 12% area discrepancy against manifold3d before
        ``calc_loop_triangles`` replaced the fan.
        """
        data = obj.data
        vertices = np.empty(len(data.vertices) * 3, dtype=np.float64)
        data.vertices.foreach_get("co", vertices)

        data.calc_loop_triangles()
        faces = np.empty(len(data.loop_triangles) * 3, dtype=np.int64)
        data.loop_triangles.foreach_get("vertices", faces)

        return Mesh(
            vertices=vertices.reshape(-1, 3),
            faces=faces.reshape(-1, 3),
            source_path=like.source_path,
            metadata={**like.metadata, "kernel": "blender-exact"},
        )
