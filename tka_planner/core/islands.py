"""Detached pieces of a bone mesh, and which of them can be thrown away.

A segmentation exported as one bone is often more than one piece of surface: specks of
collapsed triangles, slivers of a neighbouring bone, the wall of a marrow cavity
exported as a closed shell of its own, occasionally a whole patella or fibular head.
The pieces matter twice. Every landmark search reads the bone's extremes, so a piece
beyond the bone becomes the bone's top or its edge; and the boolean kernel cuts every
piece it is handed, so a broken speck can refuse a cut of an otherwise clean bone.

The rule is the author's (2026-10-25): a piece is debris, and removed, when its longest
side is under 20 mm or its volume is under 5 cm3; anything bigger may be anatomy, and is
kept and reported, with an option to remove it. Volume is the absolute enclosed volume
and counts only for a closed piece (an open shell encloses nothing, so its signed volume
says nothing about its size); an open piece is judged by its longest side alone. This
replaced the first rule (fits inside the smallest adult patella along all three of its
own axes, 14 x 23 x 25 mm), which kept a thin plate or a long sliver however light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meshio import Mesh, face_components

__all__ = ["DEBRIS_LONGEST_SIDE_MM", "DEBRIS_VOLUME_CM3", "Island", "find_islands",
           "is_debris", "keep_faces"]

# A piece is debris when its longest side is under DEBRIS_LONGEST_SIDE_MM or, if it is
# closed, its volume is under DEBRIS_VOLUME_CM3. Both are well below the smallest adult
# patella (extents of 14 x 23 x 25 mm: Muhamed et al., Anat Cell Biol 2017;50:99, mean
# less two SD of the thinnest cohort; Kim et al., Clin Orthop 2009;467:1007, smallest
# height and width among 713 female TKA knees), so no patella, fibular head or other
# piece of anatomy a knee plan needs is lost to either.
DEBRIS_LONGEST_SIDE_MM = 20.0
DEBRIS_VOLUME_CM3 = 5.0


@dataclass(frozen=True)
class Island:
    """One piece of a mesh other than its main body.

    ``extents_mm`` are the piece's lengths along its own principal axes, shortest first.
    ``volume_cm3`` is the signed enclosed volume: negative for a closed shell facing
    inwards, which is how a cavity inside a bone is exported; it means nothing unless
    ``closed``. ``outside_mm`` is how far
    the piece reaches beyond the main body's bounding box (0 inside it).
    """

    label: int
    n_triangles: int
    extents_mm: tuple
    volume_cm3: float
    outside_mm: float
    centre_mm: tuple
    closed: bool = True

    @property
    def inside(self) -> bool:
        return self.outside_mm == 0.0

    def to_dict(self) -> dict:
        return {
            "n_triangles": self.n_triangles,
            "extents_mm": [round(e, 1) for e in self.extents_mm],
            "volume_cm3": round(self.volume_cm3, 3),
            "outside_mm": round(self.outside_mm, 1),
            "centre_mm": [round(c, 1) for c in self.centre_mm],
            "closed": self.closed,
        }


def find_islands(welded: Mesh) -> tuple[np.ndarray, list[Island]]:
    """Each face's piece label (``0`` the main body, the largest) and every other piece.

    The mesh must be welded: on STL triangle soup every triangle is a piece.
    """
    labels = face_components(welded)
    if len(labels) == 0 or labels.max() == 0:
        return labels, []
    vertices = np.asarray(welded.vertices, dtype=np.float64)
    faces = np.asarray(welded.faces)
    main = vertices[np.unique(faces[labels == 0])]
    low, high = main.min(axis=0), main.max(axis=0)
    counts = np.bincount(labels)

    pieces = []
    for label in range(1, len(counts)):
        piece_faces = faces[labels == label]
        points = vertices[np.unique(piece_faces)]
        outside = float(np.max(np.concatenate(
            [low - points.min(axis=0), points.max(axis=0) - high, [0.0]])))
        pieces.append(Island(
            label=label,
            n_triangles=int(counts[label]),
            extents_mm=_principal_extents(points),
            volume_cm3=_signed_volume(vertices, piece_faces) / 1000.0,
            outside_mm=outside,
            centre_mm=tuple(float(c) for c in points.mean(axis=0)),
            closed=_closed(piece_faces),
        ))
    return labels, pieces


def is_debris(island: Island) -> bool:
    """True when the piece is too small to be anatomy: see the module docstring."""
    if island.extents_mm[-1] < DEBRIS_LONGEST_SIDE_MM:
        return True
    return island.closed and abs(island.volume_cm3) < DEBRIS_VOLUME_CM3


def keep_faces(mesh: Mesh, keep: np.ndarray, **metadata) -> Mesh:
    """The mesh without the faces not kept, and without vertices only they used.

    Vertex order is preserved, so every search that breaks ties by index breaks them
    the same way on what remains.
    """
    faces = mesh.faces[keep]
    used = np.unique(faces)
    index = np.full(len(mesh.vertices), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    return Mesh(
        vertices=mesh.vertices[used],
        faces=index[faces],
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata={**mesh.metadata, **metadata},
    )


def _principal_extents(points: np.ndarray) -> tuple:
    """Lengths along the point cloud's own principal axes, shortest first."""
    centred = points - points.mean(axis=0)
    if len(points) < 2 or not np.any(centred):
        return (0.0, 0.0, 0.0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ axes.T
    extents = projected.max(axis=0) - projected.min(axis=0)
    extents = np.concatenate([extents, np.zeros(3 - len(extents))])
    return tuple(float(e) for e in np.sort(extents))


def _closed(faces: np.ndarray) -> bool:
    """Every edge of the piece is shared by exactly two of its faces."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume by the divergence theorem; meaningful only for a closed piece."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
