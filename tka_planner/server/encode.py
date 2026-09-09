"""Turning a scene into what the viewer needs, and nothing more.

The wire format is not designed here so much as read off the scene model. A node
already carries a name, a mesh id, a rest transform, a colour, an alpha, a visibility
flag and its tags, and every one of those is something the viewer draws with. The delta
is likewise the model's own :class:`~tka_planner.scene.model.SceneDelta`.

The one thing computed here is the clipping description. Plan mode shows the resection
before it exists as geometry, by clipping the whole bone against the planned half-space,
and the viewer needs the half-space rather than the plane: a plane has two sides, and
which one survives the cut is a clinical fact, not a rendering detail.
"""

from __future__ import annotations

import numpy as np

from tka_planner.scene.resect import BONE_FOR, KEEP

__all__ = ["scene_snapshot", "node_payload", "clip_planes", "matrix_payload"]


def scene_snapshot(scene) -> dict:
    """The whole scene, for a viewer that has just connected."""
    return {
        "nodes": [node_payload(node) for node in scene.nodes.values()],
        "set_poses": {
            set_id: matrix_payload(pose) for set_id, pose in scene.set_poses.items()
        },
        "pivot": matrix_payload(scene.pivot),
        "notes": list(scene.notes),
    }


def node_payload(node) -> dict:
    return {
        "name": node.name,
        "mesh": node.mesh_id,
        "rest": matrix_payload(node.rest),
        "set": node.set_id,
        "colour": [float(channel) for channel in node.colour],
        "alpha": float(node.alpha),
        "visible": bool(node.visible),
        "tags": {key: _plain(value) for key, value in node.tags.items()},
    }


def clip_planes(plan) -> dict:
    """The planned resection as a half-space per bone, in millimetres.

    A point is kept where ``dot(normal, point) + constant >= 0``. The normal is flipped
    for the tibia because the two cuts keep opposite sides: the distal femur keeps the
    bone above its plane and the proximal tibia the bone below. One convention applied
    to both would silently invert a resection, which is the same trap the cutter box
    avoids in :mod:`tka_planner.scene.resect`.
    """
    planes = {}
    for plane_name, resection in plan.resections.items():
        bone = BONE_FOR.get(plane_name)
        if bone is None:
            continue
        normal = np.asarray(resection.normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            continue
        normal = normal / norm
        if KEEP.get(plane_name) == "distal":
            normal = -normal
        point = np.asarray(resection.point, dtype=float)
        planes[bone] = {
            "normal": [float(v) for v in normal],
            "constant_mm": float(-np.dot(normal, point)),
        }
    return planes


def matrix_payload(matrix) -> list:
    """A 4x4 as plain nested lists, row major, which is what JSON can carry."""
    return [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value
