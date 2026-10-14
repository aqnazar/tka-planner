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

__all__ = ["is_manifold", "edge_report", "repair", "connected_components",
           "repair_statistics"]


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
    fixed, actions, _ = _repair(mesh)
    return fixed, actions


def _repair(mesh: Mesh) -> tuple[Mesh, tuple[str, ...], dict]:
    """:func:`repair`, also returning how many of each thing it changed.

    The counts are kept apart from the action strings so that statistics reported over
    a cohort come from the repair itself rather than from parsing its messages.
    """
    actions: list[str] = []
    changed = {"vertices_merged": 0, "degenerate_faces_removed": 0,
               "duplicate_faces_removed": 0}
    working = mesh

    if not _is_welded(working) or edge_report(working)["boundary_edges"]:
        welded = weld_vertices(working)
        if welded.n_vertices < working.n_vertices:
            actions.append("welded")
            changed["vertices_merged"] = working.n_vertices - welded.n_vertices
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
        changed["degenerate_faces_removed"] = dropped
        faces = faces[keep]

    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    duplicated = int((counts - 1).sum())
    if duplicated:
        actions.append(f"dropped {duplicated} duplicate face"
                       f"{'s' if duplicated != 1 else ''}")
        changed["duplicate_faces_removed"] = duplicated
        faces = faces[np.sort(first)]

    if not actions:
        return mesh, (), changed

    return Mesh(
        vertices=working.vertices,
        faces=faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata={**mesh.metadata, "repaired": list(actions)},
    ), tuple(actions), changed


def connected_components(mesh: Mesh) -> int:
    """How many edge-connected pieces the mesh's faces form.

    Counted on the mesh as given, so a triangle soup counts every triangle as a piece:
    weld first. A bone segmentation should be one piece; a second one is usually a loose
    fragment the segmentation kept, which a boolean cuts along with the bone without
    complaint.

    Union-find by hooking each root onto the smallest root it touches and then
    compressing every path, both vectorised, so a million-triangle bone takes a few
    passes rather than a Python loop over its vertices.
    """
    faces = mesh.faces
    if len(faces) == 0:
        return 0
    parent = np.arange(mesh.n_vertices)
    edges = _edges(faces)
    while True:
        a, b = parent[edges[:, 0]], parent[edges[:, 1]]
        differ = a != b
        if not differ.any():
            break
        # Every label is a root here, and a root only ever hooks onto a smaller one,
        # so no cycle can form.
        np.minimum.at(parent, np.maximum(a, b)[differ], np.minimum(a, b)[differ])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return int(len(np.unique(parent[np.unique(faces)])))


def repair_statistics(mesh: Mesh) -> dict:
    """What preparing a mesh for the kernel does to it, counted.

    Runs the same repair the manifold3d kernel runs before every cut and reports what
    it changed, what it could not fix, and how many pieces are left. ``accepted`` is the
    kernel's own test: a mesh that is not manifold after repair is refused, not forced.
    """
    fixed, actions, changed = _repair(mesh)
    report = edge_report(fixed)
    return {
        "triangles_in": int(mesh.n_triangles),
        "vertices_in": int(mesh.n_vertices),
        **changed,
        "triangles_out": int(fixed.n_triangles),
        "vertices_out": int(fixed.n_vertices),
        "boundary_edges": report["boundary_edges"],
        "nonmanifold_edges": report["nonmanifold_edges"],
        "components": connected_components(fixed),
        "accepted": not any(report.values()),
        "actions": list(actions),
    }


def _is_welded(mesh: Mesh) -> bool:
    """Cheap test for a triangle soup: as many vertices as face corners."""
    return mesh.n_vertices != mesh.n_triangles * 3
