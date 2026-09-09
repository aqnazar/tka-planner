"""A content-addressed directory of meshes.

The id of a mesh is a hash of its geometry, so the same surface written twice occupies
one file and a bone that a commit did not change keeps the id it already had. That is
what makes re-committing a plan cheap: the store is asked for ids, and the ids it
already holds cost nothing to produce again.

Files are ``npz``, which is numpy's own container. It keeps float64 vertices exactly,
compresses well on the long runs of similar coordinates a segmentation produces, and
needs nothing outside the dependency set. STL would have been the obvious choice and is
the wrong one: it stores float32 triangle soup, so a round trip through it would neither
preserve the geometry nor preserve the id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.scene.model import mesh_id

__all__ = ["MeshStore"]


class MeshStore:
    """Meshes on disk, addressed by content hash."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        """Where a mesh lives.

        Sharded on the first two hex characters. A single flat directory of a few
        thousand files is slow to list on Windows, and a case can hold a dozen meshes
        per committed plan.
        """
        return self.root / digest[:2] / f"{digest}.npz"

    def has(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    def put(self, mesh: Mesh) -> str:
        """Store a mesh and return its id. Storing the same mesh again writes nothing."""
        digest = mesh_id(mesh)
        path = self.path_for(digest)
        if path.is_file():
            return digest

        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and moved into place, so a reader can never open a
        # half-written file: the id promises the content, and a truncated file would
        # break that promise permanently.
        staging = path.with_suffix(".npz.part")
        with staging.open("wb") as handle:
            np.savez_compressed(
                handle,
                vertices=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                faces=np.ascontiguousarray(mesh.faces, dtype=np.int64),
            )
        staging.replace(path)
        return digest

    def get(self, digest: str) -> Mesh:
        """Read a mesh back. Raises ``KeyError`` if the store does not hold it."""
        path = self.path_for(digest)
        if not path.is_file():
            raise KeyError(f"No mesh {digest!r} in {self.root}.")
        with np.load(path) as archive:
            return Mesh(
                vertices=np.array(archive["vertices"], dtype=np.float64),
                faces=np.array(archive["faces"], dtype=np.int64),
                metadata={"mesh_id": digest},
            )

    def size_bytes(self, digest: str) -> int:
        path = self.path_for(digest)
        return path.stat().st_size if path.is_file() else 0
