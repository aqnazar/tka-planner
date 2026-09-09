"""One interface for cutting, and the record of how a cut was actually made.

The project already states how every measured number was obtained. Geometry deserves
the same treatment: which solver ran, what it had to repair first, and whether it fell
back. A resected bone with no such record is a claim without a method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tka_planner.core.meshio import Mesh

__all__ = ["KernelRecord", "KernelResult", "MeshKernel", "default_kernel"]


@dataclass(frozen=True)
class KernelRecord:
    """How one boolean was performed."""

    backend: str
    operation: str
    repaired_target: tuple = ()
    repaired_tool: tuple = ()
    fallback: str | None = None
    notes: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "repaired_target": list(self.repaired_target),
            "repaired_tool": list(self.repaired_tool),
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class KernelResult:
    mesh: Mesh
    record: KernelRecord


@runtime_checkable
class MeshKernel(Protocol):
    """Difference and intersection over meshes in millimetres.

    Two operations is the whole surface the planner needs: the discard box and the
    cutting block are subtracted, and the bone shell is an intersection.
    """

    name: str

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult: ...

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult: ...


def default_kernel() -> MeshKernel:
    """The kernel everything uses unless told otherwise.

    Blender is never the default. It is available through
    ``geom.kernels.blender_kernel`` for the cross-check test alone, because it is GPL
    and this project is Apache 2.0.
    """
    from .kernels.manifold_kernel import ManifoldKernel

    return ManifoldKernel()
