"""Getting a segmentation into a state a boolean kernel will accept.

``manifold3d`` requires a manifold input and refuses anything else, which is stricter
than Blender's exact solver. That strictness is worth having, because a solver that
silently accepts a broken mesh returns broken geometry, and a cut that looks like a cut
and is not is the failure mode this project keeps guarding against.

Whatever repair does is reported back rather than done quietly, so it can be recorded
in the plan beside the measurement provenance.

The author's rule (2026-10-15): repair every bone as far as it can be repaired, and
refuse, with a warning, only what cannot. Repair here therefore goes beyond welding:
pieces too small to be anatomy are removed, the faces meeting at a non-manifold edge are
cut out, faces are turned to one consistent orientation, and holes up to
``HOLE_FILL_MAX_MM`` across are closed. A larger hole is not guessed at; the mesh is
refused, and the refusal says how large it was.

Repairing a bone costs seconds, and every boolean of a commit used to repeat it on the
same uncut bone. :func:`prepare_for_kernel` does it once per bone's content and keeps
the result.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict, deque

import numpy as np

from tka_planner.core.islands import find_islands, keep_faces, smaller_than_patella
from tka_planner.core.meshio import Mesh, weld_vertices

__all__ = ["HOLE_FILL_MAX_MM", "is_manifold", "edge_report", "repair", "kernel_ready",
           "prepare_for_kernel", "connected_components", "repair_statistics"]

# The widest hole closed by a fan of triangles. A segmentation's holes are a few voxels
# where a thin cortex was lost; at this size a patch moves no landmark and no cut
# measurably. A wider opening is something missing from the scan, not a defect of the
# surface, and is refused rather than invented.
HOLE_FILL_MAX_MM = 10.0

# Removing the faces at a non-manifold edge opens a hole, which the same round closes;
# a further round catches anything the first exposed.
REPAIR_ROUNDS = 3

# Only bones are worth keeping prepared: tools are a few thousand triangles and check
# in milliseconds. Four entries hold both bones of a knee and both of the one before.
CACHE_MIN_TRIANGLES = 20_000
CACHE_ENTRIES = 4
_PREPARED: OrderedDict = OrderedDict()
_PREPARED_LOCK = threading.Lock()


def _edges(faces: np.ndarray) -> np.ndarray:
    """Every face's three edges as sorted vertex pairs."""
    pairs = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.sort(pairs, axis=1)


def _directed(faces: np.ndarray) -> np.ndarray:
    """Every face's three edges in winding order, face by face within each third."""
    return np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])


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


def repair(mesh: Mesh, *, max_hole_mm: float = HOLE_FILL_MAX_MM
           ) -> tuple[Mesh, tuple[str, ...]]:
    """Make a mesh a closed, consistently wound manifold, where that can be done honestly.

    In order: weld; drop degenerate and duplicated faces. If that leaves a closed,
    consistently wound manifold, stop there. Otherwise, in up to ``REPAIR_ROUNDS``
    rounds: remove pieces smaller than the smallest adult patella; cut out the faces
    meeting at any non-manifold edge; wind every face consistently, the main body
    outward; close each hole up to ``max_hole_mm`` across with a fan around its
    centroid. No existing vertex is moved (welding keeps each first occurrence), so the
    surface away from a repaired defect is unchanged. A mesh still not manifold
    afterwards is returned as it is, with what was left undone in its actions, for the
    caller to refuse.
    """
    fixed, actions, _ = _repair(mesh, max_hole_mm=max_hole_mm)
    return fixed, actions


def prepare_for_kernel(mesh: Mesh) -> tuple[Mesh, tuple[str, ...]]:
    """The mesh as a boolean kernel should receive it, repaired once per content.

    A commit cuts each bone at least twice (the block shell, then the block), and the
    app re-commits whenever a control moves, always from the same uncut bone. The
    result is kept for the few most recent large meshes, keyed by a hash of their
    vertices and faces, so the cache cannot hand one mesh another mesh's repair.
    """
    key = _fingerprint(mesh)
    with _PREPARED_LOCK:
        if key in _PREPARED:
            _PREPARED.move_to_end(key)
            return _PREPARED[key]
    if is_manifold(mesh) and _consistently_wound(mesh.faces):
        prepared = (mesh, ())
    else:
        prepared = repair(mesh)
    if mesh.n_triangles >= CACHE_MIN_TRIANGLES:
        with _PREPARED_LOCK:
            _PREPARED[key] = prepared
            while len(_PREPARED) > CACHE_ENTRIES:
                _PREPARED.popitem(last=False)
    return prepared


def kernel_ready(mesh: Mesh) -> bool:
    """What the kernel accepts: manifold, and every edge wound once each way."""
    return is_manifold(mesh) and _consistently_wound(mesh.faces)


def _fingerprint(mesh: Mesh) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for array in (mesh.vertices, mesh.faces):
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.data)
    return digest.hexdigest()


def _repair(mesh: Mesh, *, max_hole_mm: float = HOLE_FILL_MAX_MM
            ) -> tuple[Mesh, tuple[str, ...], dict]:
    """:func:`repair`, also returning how many of each thing it changed.

    The counts are kept apart from the action strings so that statistics reported over
    a cohort come from the repair itself rather than from parsing its messages.
    """
    actions: list[str] = []
    changed = {"vertices_merged": 0, "degenerate_faces_removed": 0,
               "duplicate_faces_removed": 0, "islands_removed": 0,
               "island_faces_removed": 0, "nonmanifold_faces_removed": 0,
               "faces_flipped": 0, "holes_filled": 0, "largest_hole_filled_mm": 0.0,
               "holes_left_open": 0, "largest_hole_left_mm": 0.0,
               "tangled_boundary_vertices": 0}
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

    vertices = working.vertices
    if _closed_manifold(faces) and _consistently_wound(faces):
        if not actions:
            return mesh, (), changed
        return _finish(mesh, vertices, faces, actions), tuple(actions), changed

    for _ in range(REPAIR_ROUNDS):
        progress = False
        vertices, faces, n_islands, n_island_faces = _drop_small_islands(
            vertices, faces, mesh)
        if n_islands:
            changed["islands_removed"] += n_islands
            changed["island_faces_removed"] += n_island_faces
            progress = True

        faces, n_cut = _cut_nonmanifold(faces)
        if n_cut:
            changed["nonmanifold_faces_removed"] += n_cut
            progress = True

        faces, n_flipped = _orient(vertices, faces)
        if n_flipped:
            changed["faces_flipped"] += n_flipped
            progress = True

        vertices, faces, filled, left_open, tangled = _fill_holes(
            vertices, faces, max_hole_mm)
        if filled:
            changed["holes_filled"] += len(filled)
            changed["largest_hole_filled_mm"] = max(
                changed["largest_hole_filled_mm"], max(filled))
            progress = True
        changed["holes_left_open"] = len(left_open)
        changed["largest_hole_left_mm"] = max(left_open, default=0.0)
        changed["tangled_boundary_vertices"] = tangled

        if (_closed_manifold(faces) and _consistently_wound(faces)) or not progress:
            break

    n = changed["islands_removed"]
    if n:
        actions.append(f"removed {n} detached piece{'s' if n != 1 else ''} smaller "
                       f"than a patella ({changed['island_faces_removed']} faces)")
    if changed["nonmanifold_faces_removed"]:
        actions.append(f"cut out {changed['nonmanifold_faces_removed']} faces at "
                       f"non-manifold edges")
    if changed["faces_flipped"]:
        actions.append(f"turned {changed['faces_flipped']} faces to a consistent "
                       f"winding")
    n = changed["holes_filled"]
    if n:
        actions.append(f"closed {n} hole{'s' if n != 1 else ''} up to "
                       f"{changed['largest_hole_filled_mm']:.1f} mm across")
    n = changed["holes_left_open"]
    if n:
        actions.append(f"left {n} opening{'s' if n != 1 else ''} wider than "
                       f"{max_hole_mm:g} mm unclosed (largest "
                       f"{changed['largest_hole_left_mm']:.1f} mm)")
    n = changed["tangled_boundary_vertices"]
    if n:
        where = "vertices" if n != 1 else "vertex"
        actions.append(f"left the openings through {n} {where} where surfaces meet at "
                       f"a point unclosed")

    if not actions:
        return mesh, (), changed
    return _finish(mesh, vertices, faces, actions), tuple(actions), changed


def _finish(mesh: Mesh, vertices, faces, actions) -> Mesh:
    return Mesh(
        vertices=vertices,
        faces=faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata={**mesh.metadata, "repaired": list(actions)},
    )


def _closed_manifold(faces: np.ndarray) -> bool:
    _, counts = np.unique(_edges(faces), axis=0, return_counts=True)
    return bool(len(faces)) and bool(np.all(counts == 2))


def _consistently_wound(faces: np.ndarray) -> bool:
    """No directed edge used twice: two faces sharing an edge wind it opposite ways."""
    directed = _directed(faces)
    return len(np.unique(directed, axis=0)) == len(directed)


def _drop_small_islands(vertices, faces, like: Mesh):
    """Remove every piece but the main body that is smaller than a patella."""
    welded = Mesh(vertices=vertices, faces=faces, source_path=like.source_path,
                  sha256=like.sha256, metadata=like.metadata)
    labels, pieces = find_islands(welded)
    small = [p.label for p in pieces if smaller_than_patella(p)]
    if not small:
        return vertices, faces, 0, 0
    drop = np.zeros(len(pieces) + 1, dtype=bool)
    drop[small] = True
    keep = ~drop[labels]
    kept = keep_faces(welded, keep)
    return kept.vertices, kept.faces, len(small), int((~keep).sum())


def _cut_nonmanifold(faces):
    """Remove every face on an edge that three or more faces share.

    A non-manifold edge is a fin, a wall inside the bone or two surfaces glued along a
    line; no rule for which of its faces to keep is principled. All of them go, and
    the hole left behind is closed like any other.
    """
    _, inverse, counts = np.unique(_edges(faces), axis=0, return_inverse=True,
                                   return_counts=True)
    bad = counts[inverse.ravel()] > 2
    if not bad.any():
        return faces, 0
    bad_faces = np.zeros(len(faces), dtype=bool)
    bad_faces[np.tile(np.arange(len(faces)), 3)[bad]] = True
    return faces[~bad_faces], int(bad_faces.sum())


def _orient(vertices, faces):
    """Wind faces so that two faces sharing an edge run it in opposite directions, and
    the main body encloses positive volume.

    A breadth-first walk over the faces, one seed per piece; reached only when the
    winding is inconsistent, which is rare in a segmentation export. Separate pieces
    keep their winding relative to their own seed, so a cavity wall exported facing
    inward stays facing inward.
    """
    n = len(faces)
    if n == 0:
        return faces, 0
    faces = faces.copy()
    flipped = np.zeros(n, dtype=bool)
    if not _consistently_wound(faces):
        directed = _directed(faces)
        _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        owner = np.tile(np.arange(n), 3)
        order = np.argsort(inverse, kind="stable")
        pairs = order[counts[inverse[order]] == 2].reshape(-1, 2)
        a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
        disagree = np.all(directed[pairs[:, 0]] == directed[pairs[:, 1]], axis=1)
        neighbours = [[] for _ in range(n)]
        for f, g, d in zip(a.tolist(), b.tolist(), disagree.tolist()):
            neighbours[f].append((g, d))
            neighbours[g].append((f, d))
        seen = np.zeros(n, dtype=bool)
        for seed in range(n):
            if seen[seed]:
                continue
            seen[seed] = True
            queue = deque([seed])
            while queue:
                face = queue.popleft()
                for other, d in neighbours[face]:
                    if not seen[other]:
                        seen[other] = True
                        flipped[other] = flipped[face] ^ d
                        queue.append(other)
        # Of the two consistent windings of each piece, keep the one that turns fewer
        # faces; the outward test below settles the main body either way.
        labels = _face_labels(vertices, faces)
        for label in np.unique(labels):
            members = labels == label
            if flipped[members].sum() * 2 > members.sum():
                flipped[members] = ~flipped[members]
        faces[flipped] = faces[flipped][:, ::-1]

    main = _face_labels(vertices, faces) == 0
    p = [vertices[faces[main, i]].astype(np.float64) for i in range(3)]
    if np.einsum("ij,ij->i", p[0], np.cross(p[1], p[2])).sum() < 0:
        faces[main] = faces[main][:, ::-1]
        flipped[main] = ~flipped[main]
    return faces, int(flipped.sum())


def _face_labels(vertices, faces) -> np.ndarray:
    from tka_planner.core.meshio import face_components
    return face_components(Mesh(vertices=vertices, faces=faces))


def _fill_holes(vertices, faces, max_hole_mm):
    """Close each boundary loop no wider than ``max_hole_mm`` with a fan of triangles.

    The fan is wound against the loop's existing faces, so the patch continues their
    orientation. A loop that passes through one vertex twice is not traced: it is left
    open, and the mesh refused, which is the honest outcome for a tangle no fan closes.
    Returns the widths of the loops closed and of those left open as too wide, and the
    number of vertices where the boundary is tangled.
    """
    directed = _directed(faces)
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse.ravel()] == 1]
    if len(boundary) == 0:
        return vertices, faces, [], [], 0

    starts, n_out = np.unique(boundary[:, 0], return_counts=True)
    tangled = set(starts[n_out > 1].tolist())
    following = {int(a): int(b) for a, b in boundary.tolist() if int(a) not in tangled}

    filled, left_open, new_faces, new_vertices = [], [], [], []
    visited: set = set()
    for start in list(following):
        if start in visited:
            continue
        loop, v = [], start
        while v in following and v not in visited:
            visited.add(v)
            loop.append(v)
            v = following[v]
        if v != start or len(loop) < 3:
            continue                        # a chain through a tangle, counted below
        points = vertices[loop].astype(np.float64)
        width = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if width > max_hole_mm:
            left_open.append(width)
            continue
        filled.append(width)
        if len(loop) == 3:
            new_faces.append([loop[2], loop[1], loop[0]])
            continue
        centre = len(vertices) + len(new_vertices)
        new_vertices.append(points.mean(axis=0))
        for a, b in zip(loop, loop[1:] + loop[:1]):
            new_faces.append([b, a, centre])
    if not new_faces:
        return vertices, faces, filled, left_open, len(tangled)
    if new_vertices:
        vertices = np.vstack([vertices, np.asarray(new_vertices, dtype=vertices.dtype)])
    faces = np.vstack([faces, np.asarray(new_faces, dtype=faces.dtype)])
    return vertices, faces, filled, left_open, len(tangled)


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
        "accepted": not any(report.values()) and _consistently_wound(fixed.faces),
        "actions": list(actions),
    }


def _is_welded(mesh: Mesh) -> bool:
    """Cheap test for a triangle soup: as many vertices as face corners."""
    return mesh.n_vertices != mesh.n_triangles * 3
