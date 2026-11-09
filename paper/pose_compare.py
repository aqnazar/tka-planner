"""How far apart two component poses and two cut planes are, in anatomical terms.

Both pipelines place the same CAD parts, so a pose is a rigid map from the part's CAD
frame to the bone's STL frame and two poses of one part can be compared directly: the
translations are the two CAD origins (which the library puts on the cut), the third
columns the two cut normals. The differences are expressed in the new planner's bone
frame -- anterior, lateral, proximal -- because "3 mm anterior" is what a surgeon or a
reviewer can check, and "3 mm along scanner Y" is not.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rotation_angle_deg", "pose_difference", "plane_offset_mm", "plane_difference"]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """The angle of the rotation taking ``r_a`` to ``r_b`` (both 3x3), in degrees."""
    cos = (np.trace(np.asarray(r_a).T @ np.asarray(r_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _signed_angle_deg(u, v, axis) -> float:
    """Signed angle from ``u`` to ``v`` about ``axis`` (right hand), in degrees."""
    axis = _unit(axis)
    u = _unit(u - np.dot(u, axis) * axis)
    v = _unit(v - np.dot(v, axis) * axis)
    return float(np.degrees(np.arctan2(np.dot(np.cross(u, v), axis), np.dot(u, v))))


def _tilts_deg(normal, anterior, lateral, proximal) -> tuple[float, float]:
    """A cut normal's tilt from the bone axis: (toward lateral, toward anterior)."""
    n = _unit(normal)
    return (float(np.degrees(np.arctan2(np.dot(n, lateral), np.dot(n, proximal)))),
            float(np.degrees(np.arctan2(np.dot(n, anterior), np.dot(n, proximal)))))


def pose_difference(legacy: np.ndarray, new: np.ndarray, *, anterior, lateral,
                    proximal) -> dict:
    """``legacy`` minus ``new``, for two 4x4 poses of the same part.

    Positions: the legacy origin's offset from the new one, as a distance and along
    the frame's anterior, lateral and proximal axes. Rotations: the total angle between
    the two orientations, then split into the tilt of the cut normal (coronal: toward
    lateral; sagittal: toward anterior; each also given for both poses, measured from
    the frame's proximal axis) and the turn about the normal (positive turns
    the legacy part's CAD x axis counter-clockwise about the new normal).
    """
    legacy, new = np.asarray(legacy, float), np.asarray(new, float)
    anterior, lateral, proximal = (_unit(anterior), _unit(lateral), _unit(proximal))
    d = legacy[:3, 3] - new[:3, 3]
    nl, nn = legacy[:3, 2], new[:3, 2]
    cor_l, sag_l = _tilts_deg(nl, anterior, lateral, proximal)
    cor_n, sag_n = _tilts_deg(nn, anterior, lateral, proximal)
    return {
        "position_mm": float(np.linalg.norm(d)),
        "anterior_mm": float(np.dot(d, anterior)),
        "lateral_mm": float(np.dot(d, lateral)),
        "proximal_mm": float(np.dot(d, proximal)),
        "rotation_deg": rotation_angle_deg(legacy[:3, :3], new[:3, :3]),
        "normal_angle_deg": float(np.degrees(np.arccos(np.clip(
            np.dot(_unit(nl), _unit(nn)), -1.0, 1.0)))),
        "coronal_tilt_deg": cor_l - cor_n,
        "sagittal_tilt_deg": sag_l - sag_n,
        "legacy_coronal_deg": cor_l,
        "new_coronal_deg": cor_n,
        "legacy_sagittal_deg": sag_l,
        "new_sagittal_deg": sag_n,
        "axial_turn_deg": _signed_angle_deg(new[:3, 0], legacy[:3, 0], nn),
    }


def plane_offset_mm(point_a, normal_a, point_b, normal_b, centre, along) -> float:
    """Distance between two planes where the line through ``centre`` along ``along``
    meets them: positive when plane ``a`` is further along ``along`` than plane ``b``.

    Two cuts that are not parallel have no single separation; measured at the knee
    centre along one bone axis, the number is the extra (or lesser) depth at the place
    the implant sits.
    """
    along = _unit(along)
    centre = np.asarray(centre, float)

    def hit(point, normal):
        normal = _unit(normal)
        denominator = float(np.dot(normal, along))
        if abs(denominator) < 1e-9:
            return float("nan")
        return float(np.dot(normal, np.asarray(point, float) - centre)) / denominator

    return hit(point_a, normal_a) - hit(point_b, normal_b)


def plane_difference(point_l, normal_l, point_n, normal_n, *, centre, proximal) -> dict:
    """Angle between two cut planes and their offset at the knee centre, legacy - new."""
    nl, nn = _unit(normal_l), _unit(normal_n)
    return {
        "angle_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(nl, nn)), -1.0, 1.0)))),
        "offset_at_centre_mm": plane_offset_mm(point_l, nl, point_n, nn, centre, proximal),
    }
