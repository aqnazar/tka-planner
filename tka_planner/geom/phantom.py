"""The phantom femur as a triangle mesh, for a viewer or an STL.

The parts and their sizes are in :mod:`tka_planner.core.phantom_femur`; this only turns
them into closed surfaces: a tube for the shaft, a tube for the neck and a sphere for the
head. Each is its own closed shell and they are joined without a boolean, so the
result is for looking at, not for cutting.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.core.phantom_femur import PhantomFemur

from . import mesh as gm

__all__ = ["phantom_mesh"]


def _tube(start, end, radius_mm: float, segments: int = 24) -> Mesh:
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    matrix = np.eye(4)
    matrix[:3, :3] = gm.rotation_to(axis / length)
    matrix[:3, 3] = (start + end) / 2.0
    return gm.cylinder(radius_mm, length, segments=segments, matrix=matrix)


def phantom_mesh(phantom: PhantomFemur) -> Mesh:
    """Shaft, neck and head as one mesh (three closed shells)."""
    parts = [
        _tube(phantom.shaft_start, phantom.shaft_end, phantom.shaft_radius_mm),
        _tube(phantom.neck_foot, phantom.head_centre, phantom.neck_radius_mm),
        gm.uv_sphere(phantom.head_diameter_mm / 2.0, segments=32, rings=20,
                     centre=phantom.head_centre),
    ]
    vertices, faces, offset = [], [], 0
    for part in parts:
        vertices.append(part.vertices)
        faces.append(part.faces + offset)
        offset += len(part.vertices)
    return Mesh(vertices=np.vstack(vertices), faces=np.vstack(faces).astype(np.int64),
                metadata={"phantom_femur": phantom.method})
