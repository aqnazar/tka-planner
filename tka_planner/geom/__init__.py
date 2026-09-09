"""Meshes and boolean geometry, without Blender.

The planning core reads meshes and measures them. This package is what *cuts* them,
which is the one job the pipeline previously had to borrow Blender for. Everything
here works in millimetres, like the core, and unit conversion happens only where a
scene is encoded for a viewer.
"""

from tka_planner.core.meshio import Mesh, read_stl, weld_vertices

__all__ = ["Mesh", "read_stl", "weld_vertices"]
