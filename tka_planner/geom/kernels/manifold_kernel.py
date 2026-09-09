"""The ``manifold3d`` backend.

Chosen as the default for three reasons. It is Apache 2.0, so it does not compromise
this project's licence the way linking Blender would. It is fast enough that a
full-resolution resection is a few seconds rather than tens of them. And it refuses
non-manifold input rather than guessing, which turns a class of silent wrong answers
into a loud one.

That refusal is why :mod:`tka_planner.geom.repair` exists and why every repair is
recorded rather than done quietly.

``Mesh64`` is used throughout rather than the float32 ``Mesh``. Anatomical coordinates
run to several hundred millimetres, where single precision resolves to about 3e-5 mm;
that is below any clinically meaningful tolerance but it is a needless loss on a
round trip, and a planner that publishes its methods should not quietly discard digits.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh

from ..kernel import KernelRecord, KernelResult
from ..repair import edge_report, is_manifold, repair

__all__ = ["ManifoldKernel"]


class ManifoldKernel:
    """Difference and intersection through ``manifold3d``."""

    name = "manifold3d"

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "difference")

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "intersect")

    # ------------------------------------------------------------------

    def _operate(self, target: Mesh, tool: Mesh, operation: str) -> KernelResult:
        clean_target, target_actions = self._prepare(target, "target")
        clean_tool, tool_actions = self._prepare(tool, "tool")

        a = self._to_manifold(clean_target, "target")
        b = self._to_manifold(clean_tool, "tool")
        combined = a - b if operation == "difference" else a ^ b

        return KernelResult(
            mesh=self._from_manifold(combined, target),
            record=KernelRecord(
                backend=self.name,
                operation=operation,
                repaired_target=target_actions,
                repaired_tool=tool_actions,
                fallback=None,
            ),
        )

    @staticmethod
    def _prepare(mesh: Mesh, role: str) -> tuple:
        if is_manifold(mesh):
            return mesh, ()

        fixed, actions = repair(mesh)
        if not is_manifold(fixed):
            report = ", ".join(
                f"{key}={value}" for key, value in edge_report(fixed).items() if value
            )
            raise ValueError(
                f"The {role} mesh is not manifold and repair did not make it one "
                f"({report}). Fix the segmentation rather than forcing the cut: a "
                f"boolean against a broken surface returns broken geometry."
            )
        return fixed, actions

    @staticmethod
    def _to_manifold(mesh: Mesh, role: str):
        import manifold3d

        surface = manifold3d.Mesh64(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint64),
        )
        solid = manifold3d.Manifold(surface)

        # manifold3d reports a bad input by returning an empty solid with a status set,
        # rather than by raising. Left unchecked that surfaces later as a cut that
        # removed everything, which is exactly the silent-wrong-answer failure the
        # repair pass exists to prevent.
        if solid.is_empty() and mesh.n_triangles:
            raise ValueError(
                f"manifold3d rejected the {role} mesh: {solid.status()}. "
                f"It has {mesh.n_triangles} triangles and passed the manifold check, "
                f"so this is an orientation or self-intersection problem."
            )
        return solid

    @staticmethod
    def _from_manifold(solid, like: Mesh) -> Mesh:
        surface = solid.to_mesh64()
        return Mesh(
            vertices=np.asarray(surface.vert_properties, dtype=np.float64)[:, :3],
            faces=np.asarray(surface.tri_verts, dtype=np.int64),
            source_path=like.source_path,
            metadata={**like.metadata, "kernel": "manifold3d"},
        )
