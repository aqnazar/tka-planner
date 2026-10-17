"""Detached pieces of a bone mesh, and which of them can be thrown away.

A segmentation exported as one bone is often more than one piece of surface: specks of
collapsed triangles, slivers of a neighbouring bone, the wall of a marrow cavity
exported as a closed shell of its own, occasionally a whole patella or fibular head.
The pieces matter twice. Every landmark search reads the bone's extremes, so a piece
beyond the bone becomes the bone's top or its edge; and the boolean kernel cuts every
piece it is handed, so a broken speck can refuse a cut of an otherwise clean bone.

The rule is the author's: a piece smaller than the smallest possible adult patella
cannot be any anatomy a knee plan needs, and is removed; anything larger may be, and is
kept and reported, with an option to remove it. "Smaller" means it fits inside the
smallest patella's extents along every one of its own principal axes, so a thin plate
or a long sliver is not removed merely because it is light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .meshio import Mesh, face_components

__all__ = ["SMALLEST_PATELLA_MM", "Island", "find_islands", "smaller_than_patella",
           "keep_faces"]

# The extents of the smallest adult patella, thinnest first: thickness, height, width.
# Thickness: the smallest published cohort's mean less two SD (16.2 - 2 x 0.9 mm, MRI,
# 70 women; Muhamed et al., Anat Cell Biol 2017;50:99). Height and width: the smallest
# single values reported in 713 female Korean TKA knees (23 and 25 mm; Kim et al.,
# Clin Orthop 2009;467:1007), the width measured after the patellar cut and so an
# underestimate. Iranpour et al. (Clin Orthop 2008;466:1198, 3D CT) report minima of
# 18.4, 24.3 and 36.8 mm.
SMALLEST_PATELLA_MM = (14.0, 23.0, 25.0)


@dataclass(frozen=True)
class Island:
    """One piece of a mesh other than its main body.

    ``extents_mm`` are the piece's lengths along its own principal axes, shortest first.
    ``volume_cm3`` is the signed enclosed volume: negative for a closed shell facing
    inwards, which is how a cavity inside a bone is exported. ``outside_mm`` is how far
    the piece reaches beyond the main body's bounding box (0 inside it).
    """

    label: int
    n_triangles: int
    extents_mm: tuple
    volume_cm3: float
    outside_mm: float
    centre_mm: tuple

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
        ))
    return labels, pieces


def smaller_than_patella(island: Island, smallest=SMALLEST_PATELLA_MM) -> bool:
    """True when the piece fits inside the smallest adult patella along every axis."""
    return all(e <= s for e, s in zip(island.extents_mm, sorted(smallest)))


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


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Enclosed volume by the divergence theorem; meaningful only for a closed piece."""
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
