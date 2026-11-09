"""The femoral component's facet planes, and the bone's extent on each.

In the component's CAD frame (+X anterior, +Y patient-left, +Z proximal, origin on the
distal cut): the five cut planes of the box C, B, D, and the medio-lateral extent of the
bone's cut surface on a facet, bounded by the other four (from ``eval/fit_check.py``).
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.implant_spec import section_mask
from tka_planner.core.meshio import Mesh

from .design import FLARE_DEG

__all__ = ["to_frame", "plane_pose", "femoral_planes_cad", "facet_extents"]


def to_frame(mesh, pose) -> Mesh:
    """A mesh in a pose's own frame (the inverse of placing it there)."""
    P = np.asarray(pose)
    return Mesh(vertices=(mesh.vertices - P[:3, 3]) @ P[:3, :3], faces=mesh.faces)


def plane_pose(point, normal, y=(0, 1, 0)) -> np.ndarray:
    """A frame on the plane: z = normal (the kept bone's side), y = the given in-plane
    direction (ML), x = y x z."""
    z = np.asarray(normal, float)
    z /= np.linalg.norm(z)
    y = np.asarray(y, float)
    y -= (y @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, point
    return P


def femoral_planes_cad(C, B, D) -> dict:
    """The five cut planes, each (point, normal toward the kept bone), in the CAD frame."""
    t = np.tan(np.radians(FLARE_DEG))
    s = np.sqrt(0.5)
    return {"distal": ([0, 0, 0], [0, 0, 1]),
            "posterior": ([-C / 2 - B, 0, 0], [1, 0, 0]),
            "anterior": ([C / 2 + D, 0, D], [-1, 0, t]),
            "anterior chamfer": ([C / 2, 0, 0], [-s, 0, s]),
            "posterior chamfer": ([-C / 2, 0, 0], [s, 0, s])}


def facet_extents(bone_cad, C, B, D, facets=("distal", "posterior chamfer", "posterior"),
                  grid=0.25) -> dict:
    """ML extent [ymin, ymax] of the bone's cut surface on each femoral facet (CAD frame,
    the facet bounded by the other four planes)."""
    planes = femoral_planes_cad(C, B, D)
    out = {}
    for name in facets:
        p, n = planes[name]
        pose = plane_pose(np.array(p, float), n)
        m, xs, ys = section_mask(bone_cad, pose, offset_mm=0.0, grid_mm=grid)
        GX, GY = np.meshgrid(xs, ys)
        pts = pose[:3, 3] + GX[..., None] * pose[:3, 0] + GY[..., None] * pose[:3, 1]
        keep = m.copy()
        for k, (q, nn) in planes.items():
            if k == name:
                continue
            nn = np.asarray(nn, float) / np.linalg.norm(nn)
            keep &= (pts - np.asarray(q, float)) @ nn >= -1.0
        out[name] = [float(GY[keep].min()), float(GY[keep].max())]
    return out
