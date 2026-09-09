"""Nodes, sets, and the delta that says what moved.

The structure is two **sets**, femoral and tibial, each holding a bone, its implant
parts and its landmarks. A set carries one pose. Everything inside it carries a **rest
transform** relative to the set origin, fixed when the scene is built and never touched
by posing.

That separation is deliberate and it is the fix for a real bug. The Blender trial rig
posed a parented empty and composed rotations onto whatever the object's current
rotation happened to be, so repeated poses drifted and zeroing the controls did not
return to the start. Here a pose is *assigned*, never accumulated, and the world
transform is always ``set_pose @ rest``. Drift of that kind cannot be expressed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from tka_planner.core.meshio import Mesh

__all__ = ["FEMORAL", "TIBIAL", "Node", "Scene", "SceneDelta"]

FEMORAL = "femoral"
TIBIAL = "tibial"


def _identity() -> np.ndarray:
    return np.eye(4)


@dataclass
class Node:
    """One drawable thing: a mesh instance, or a marker, in a set."""

    name: str
    mesh_id: str | None = None
    rest: np.ndarray = field(default_factory=_identity)
    set_id: str | None = None
    colour: tuple = (0.62, 0.62, 0.62)
    alpha: float = 1.0
    visible: bool = True
    tags: dict = field(default_factory=dict)


@dataclass
class Scene:
    """A mesh table, a node table, one pose per set, and the flexion pivot."""

    meshes: dict = field(default_factory=dict)
    nodes: dict = field(default_factory=dict)
    set_poses: dict = field(
        default_factory=lambda: {FEMORAL: np.eye(4), TIBIAL: np.eye(4)}
    )
    notes: list = field(default_factory=list)
    pivot: np.ndarray = field(default_factory=_identity)

    def add(self, node: Node, mesh: Mesh | None = None) -> Node:
        """Register a node, storing its mesh under a content-addressed id.

        Meshes are shared rather than copied, which is what lets a bone shell reference
        the bone it was taken from without a second copy of two million triangles.
        """
        if mesh is not None:
            mesh_id = _mesh_id(mesh)
            self.meshes.setdefault(mesh_id, mesh)
            node.mesh_id = mesh_id
        self.nodes[node.name] = node
        return node

    def world(self, name: str) -> np.ndarray:
        """The node's world transform: its set's pose composed onto its rest basis."""
        node = self.nodes.get(name)
        if node is None:
            raise KeyError(f"No node named {name!r} in the scene.")
        pose = self.set_poses.get(node.set_id, np.eye(4)) if node.set_id else np.eye(4)
        return pose @ node.rest

    def set_pose(self, set_id: str, matrix) -> None:
        """Assign a set's pose. Assigned, never accumulated."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"A set pose must be 4x4, got {matrix.shape}.")
        self.set_poses[set_id] = matrix

    def nodes_in(self, set_id: str) -> list:
        return [node for node in self.nodes.values() if node.set_id == set_id]

    def mesh_for(self, name: str) -> Mesh | None:
        node = self.nodes[name]
        return self.meshes.get(node.mesh_id) if node.mesh_id else None

    def replace_mesh(self, name: str, mesh: Mesh) -> str:
        """Swap a node's mesh, returning the new id."""
        mesh_id = _mesh_id(mesh)
        self.meshes[mesh_id] = mesh
        self.nodes[name].mesh_id = mesh_id
        return mesh_id


@dataclass
class SceneDelta:
    """What changed. This is also the wire format the viewer receives."""

    poses: dict = field(default_factory=dict)
    meshes: dict = field(default_factory=dict)
    visibility: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def merge(self, other: "SceneDelta") -> "SceneDelta":
        """Combine two deltas into a new one, the later winning where they overlap."""
        return SceneDelta(
            poses={**self.poses, **other.poses},
            meshes={**self.meshes, **other.meshes},
            visibility={**self.visibility, **other.visibility},
            scalars={**self.scalars, **other.scalars},
            notes=[*self.notes, *other.notes],
        )

    def to_dict(self) -> dict:
        return {
            "poses": {
                name: [[float(v) for v in row] for row in np.asarray(matrix)]
                for name, matrix in self.poses.items()
            },
            "meshes": dict(self.meshes),
            "visibility": dict(self.visibility),
            "scalars": dict(self.scalars),
            "notes": list(self.notes),
        }


def _mesh_id(mesh: Mesh) -> str:
    """A content hash, so identical geometry is stored and transferred once."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]
