"""Getting a segmentation into a state a boolean kernel will accept.

``manifold3d`` requires a manifold input and refuses anything else, which is stricter
than Blender's exact solver. That strictness is worth having, because a solver that
silently accepts a broken mesh returns broken geometry, and a cut that looks like a cut
and is not is the failure mode this project keeps guarding against.

Whatever repair does is reported back rather than done quietly, so it can be recorded
in the plan beside the measurement provenance.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh, weld_vertices

__all__ = ["is_manifold", "edge_report", "repair"]


def _edges(faces: np.ndarray) -> np.ndarray:
    """Every face's three edges as sorted vertex pairs."""
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.sort(pairs, axis=1)


def edge_report(mesh: Mesh) -> dict:
    """Count the ways a mesh fails to be a closed manifold."""
    faces = mesh.faces
    degenerate = int((
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    ).sum())

    sorted_faces = np.sort(faces, axis=1)
    _, counts = np.unique(sorted_faces, axis=0, return_counts=True)
    duplicates = int((counts - 1).sum())

    _, edge_counts = np.unique(_edges(faces), axis=0, return_counts=True)
    return {
        "boundary_edges": int((edge_counts == 1).sum()),
        "nonmanifold_edges": int((edge_counts > 2).sum()),
        "degenerate_faces": degenerate,
        "duplicate_faces": duplicates,
    }


def is_manifold(mesh: Mesh) -> bool:
    """True when every edge is shared by exactly two faces and no face is degenerate."""
    report = edge_report(mesh)
    return (
        report["boundary_edges"] == 0
        and report["nonmanifold_edges"] == 0
        and report["degenerate_faces"] == 0
        and report["duplicate_faces"] == 0
    )


def repair(mesh: Mesh) -> tuple[Mesh, tuple[str, ...]]:
    """Weld, then drop degenerate and duplicated faces.

    Deliberately conservative: it never moves a vertex and never fills a hole. Welding
    keeps the first occurrence's coordinates, so geometry is unchanged. A mesh that is
    still not manifold afterwards is reported as such rather than forced, because the
    honest answer to a hole in a segmentation is to say there is a hole.
    """
    actions: list[str] = []
    working = mesh

    if not _is_welded(working) or edge_report(working)["boundary_edges"]:
        welded = weld_vertices(working)
        if welded.n_vertices < working.n_vertices:
            actions.append("welded")
            working = welded

    faces = working.faces
    keep = ~(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    dropped = int((~keep).sum())
    if dropped:
        actions.append(f"dropped {dropped} degenerate face"
                       f"{'s' if dropped != 1 else ''}")
        faces = faces[keep]

    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    duplicated = int((counts - 1).sum())
    if duplicated:
        actions.append(f"dropped {duplicated} duplicate face"
                       f"{'s' if duplicated != 1 else ''}")
        faces = faces[np.sort(first)]

    if not actions:
        return mesh, ()

    return Mesh(
        vertices=working.vertices,
        faces=faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata={**mesh.metadata, "repaired": list(actions)},
    ), tuple(actions)


def _is_welded(mesh: Mesh) -> bool:
    """Cheap test for a triangle soup: as many vertices as face corners."""
    return mesh.n_vertices != mesh.n_triangles * 3
