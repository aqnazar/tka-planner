# Blender-Free Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a headless geometry and scene engine so the TKA planner computes and cuts without Blender, ending with the existing add-on running unchanged on top of it.

**Architecture:** Three new packages under `tka_planner/`. `geom/` owns meshes and boolean operations behind a swappable kernel. `scene/` owns a Blender-free scene model, its builder, its updater, resection and motion. `session.py` owns the planning session that today lives inside the add-on's property callbacks. The add-on is archived first, then rewired onto the engine so it only draws.

**Tech Stack:** Python 3.11, numpy, manifold3d, pytest. Blender is optional and only for a cross-check test.

**Spec:** `docs/superpowers/specs/2026-09-09-standalone-application-design.md`

## Global Constraints

- Python floor is 3.10. Do not use syntax newer than that.
- `tka_planner/core/` must keep depending on numpy alone. Do not import `geom`, `scene` or `session` from it.
- `tka_planner/geom/` and `tka_planner/scene/` must never import `bpy`, except inside `geom/kernels/blender_kernel.py`, which imports it lazily inside function bodies.
- All engine code works in millimetres. No unit conversion anywhere in `geom/` or `scene/`.
- Licence is Apache 2.0. `manifold3d` (Apache 2.0) is a runtime dependency. `bpy` (GPL) goes in an optional extra and is never imported at runtime.
- The existing test suite must pass at every commit: `pytest` with no arguments, which excludes tests marked `blender`.
- Reuse `tka_planner.core.meshio.Mesh`. Do not define a second mesh type.
- Rotations compose onto a stored rest basis, never onto an implicit parent. See Task 10.
- Commit after every task. Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Toqp3H79GaUGHWnRdCYVxv
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `archive/blender-addon/` | Frozen copy of the add-on and Blender builder, plus `PROVENANCE.md` |
| `tka_planner/geom/__init__.py` | Re-exports `Mesh`, `read_stl`, kernel types |
| `tka_planner/geom/mesh.py` | Transforms, primitives, STL writing, volume and area |
| `tka_planner/geom/repair.py` | Welding, degenerate and duplicate face removal, manifold test |
| `tka_planner/geom/kernel.py` | `MeshKernel` protocol, `KernelRecord`, `KernelResult`, default kernel selection |
| `tka_planner/geom/kernels/manifold_kernel.py` | `manifold3d` backend |
| `tka_planner/geom/kernels/blender_kernel.py` | Optional Blender backend, cross-check only |
| `tka_planner/scene/model.py` | `Node`, `Scene`, `SceneDelta` |
| `tka_planner/scene/build.py` | `build_scene`, component resolution reused from the add-on |
| `tka_planner/scene/update.py` | `update_scene`, returns a delta |
| `tka_planner/scene/resect.py` | Cutter construction, block cut, shells, `commit_resection` |
| `tka_planner/scene/motion.py` | Flexion axis, flexion pose, trial pose, pivot composition |
| `tka_planner/session.py` | `Controls`, `PlanningSession` |
| `tests/test_geom_mesh.py` … `tests/test_session.py` | One test module per new module |

---

### Task 1: Archive the add-on

**Files:**
- Create: `archive/blender-addon/addon/__init__.py` (copy)
- Create: `archive/blender-addon/blender/` (copy of `build.py`, `io.py`, `__init__.py`)
- Create: `archive/blender-addon/PROVENANCE.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable. `archive/` is excluded from packaging.

- [ ] **Step 1: Commit the working tree so the snapshot is of a real commit**

There are uncommitted changes to `.gitignore`, `README.md`, `docs/METHODS.md`, `tka_planner/addon/__init__.py`, `tka_planner/blender/build.py` and `tka_planner/blender/io.py`. Ask the user to confirm the message, then:

```bash
git add -A
git commit -m "Colour the scene, isolate landmarks, and add a clean viewport mode"
```

- [ ] **Step 2: Copy the add-on and the Blender builder**

```bash
mkdir -p archive/blender-addon
cp -r tka_planner/addon archive/blender-addon/addon
cp -r tka_planner/blender archive/blender-addon/blender
find archive -name __pycache__ -type d -exec rm -rf {} +
```

- [ ] **Step 3: Write the provenance note**

Create `archive/blender-addon/PROVENANCE.md`:

```markdown
# Blender add-on, archived

Taken from commit `<paste the output of: git rev-parse HEAD>` on 2026-09-09.

This is the planning screen as it stood before the standalone application port.
It requires Blender 4.4+ and implements the full feature set listed in section 9
of `docs/superpowers/specs/2026-09-09-standalone-application-design.md`.

## Why it is kept

Two reasons. It is the parity reference the port is diffed against: its scene
output, node poses and mesh volumes are what the headless engine must reproduce.
And it may be revived, because a Blender front end onto the same engine remains a
reasonable thing to want.

## Difference from `legacy/`

`legacy/` holds the frozen script behind the first paper. It is cited, never run,
and never modified. This directory holds working code that is run by the parity
tests and may be brought back.

## Do not edit

Changes belong in `tka_planner/`. Editing this copy destroys its value as a
reference.
```

- [ ] **Step 4: Keep the archive out of the package and the test run**

In `pyproject.toml`, confirm `[tool.setuptools.packages.find]` has `include = ["tka_planner*"]`, which already excludes `archive`. Add to `[tool.pytest.ini_options]`:

```toml
norecursedirs = ["archive", "legacy", ".git", "*.egg-info"]
```

- [ ] **Step 5: Verify nothing changed for the existing suite**

Run: `pytest -q`
Expected: same pass count as before the task, currently 462 passed.

- [ ] **Step 6: Commit**

```bash
git add archive pyproject.toml
git commit -m "Archive the Blender add-on as the port's parity reference"
```

---

### Task 2: Mesh transforms, primitives and measures

**Files:**
- Create: `tka_planner/geom/__init__.py`
- Create: `tka_planner/geom/mesh.py`
- Test: `tests/test_geom_mesh.py`

**Interfaces:**
- Consumes: `tka_planner.core.meshio.Mesh`, `read_stl`, `weld_vertices`.
- Produces:
  - `transformed(mesh: Mesh, matrix: np.ndarray) -> Mesh`
  - `box(size_mm: float | tuple, matrix: np.ndarray | None = None) -> Mesh`
  - `disc(radius_mm: float, *, segments: int = 64, matrix=None) -> Mesh`
  - `cylinder(radius_mm: float, length_mm: float, *, segments: int = 16, matrix=None) -> Mesh`
  - `uv_sphere(radius_mm: float, *, segments: int = 12, rings: int = 8, centre=(0,0,0)) -> Mesh`
  - `slab(ml_mm: float, ap_mm: float, thickness_mm: float, matrix=None) -> Mesh`
  - `volume(mesh: Mesh) -> float`, `surface_area(mesh: Mesh) -> float`
  - `write_stl(mesh: Mesh, path) -> Path`
  - `plane_matrix(point_mm, normal_mm) -> np.ndarray`
  - `rotation_to(direction) -> np.ndarray` (3x3 taking +Z onto `direction`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geom_mesh.py`:

```python
"""Mesh transforms, primitives and measures, all in millimetres."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm


def test_box_volume_matches_its_size():
    cube = gm.box(10.0)
    assert gm.volume(cube) == pytest.approx(1000.0, rel=1e-9)


def test_box_surface_area_matches_its_size():
    cube = gm.box(10.0)
    assert gm.surface_area(cube) == pytest.approx(600.0, rel=1e-9)


def test_box_accepts_unequal_sides():
    slab = gm.box((2.0, 3.0, 4.0))
    assert gm.volume(slab) == pytest.approx(24.0, rel=1e-9)


def test_transform_translates_without_changing_volume():
    cube = gm.box(10.0)
    matrix = np.eye(4)
    matrix[:3, 3] = (5.0, -2.0, 7.0)
    moved = gm.transformed(cube, matrix)

    assert gm.volume(moved) == pytest.approx(1000.0, rel=1e-9)
    assert moved.vertices.mean(axis=0) == pytest.approx(
        cube.vertices.mean(axis=0) + matrix[:3, 3]
    )


def test_transform_leaves_faces_alone():
    cube = gm.box(10.0)
    moved = gm.transformed(cube, np.eye(4))
    assert np.array_equal(moved.faces, cube.faces)


def test_rotation_to_takes_z_onto_the_direction():
    direction = np.array([0.0, 1.0, 0.0])
    rotation = gm.rotation_to(direction)
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(direction)


def test_rotation_to_is_orthonormal():
    rotation = gm.rotation_to(np.array([0.3, -0.5, 0.8]))
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_plane_matrix_puts_z_on_the_normal_and_origin_on_the_point():
    point = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 1.0])
    matrix = gm.plane_matrix(point, normal)

    assert matrix[:3, 3] == pytest.approx(point)
    assert matrix[:3, 2] == pytest.approx(normal)


def test_cylinder_volume_approaches_the_analytic_value():
    tube = gm.cylinder(2.0, 10.0, segments=256)
    assert gm.volume(tube) == pytest.approx(np.pi * 4.0 * 10.0, rel=1e-3)


def test_sphere_volume_approaches_the_analytic_value():
    ball = gm.uv_sphere(3.0, segments=128, rings=64)
    assert gm.volume(ball) == pytest.approx(4.0 / 3.0 * np.pi * 27.0, rel=1e-2)


def test_slab_volume_matches_its_footprint_and_thickness():
    block = gm.slab(60.0, 40.0, 9.0)
    assert gm.volume(block) == pytest.approx(60.0 * 40.0 * 9.0, rel=1e-9)


def test_slab_sits_on_z_zero():
    block = gm.slab(60.0, 40.0, 9.0)
    assert block.vertices[:, 2].min() == pytest.approx(0.0)
    assert block.vertices[:, 2].max() == pytest.approx(9.0)


def test_disc_lies_in_its_own_plane():
    circle = gm.disc(55.0, segments=64)
    assert circle.vertices[:, 2] == pytest.approx(np.zeros(circle.n_vertices))


def test_write_stl_round_trips(tmp_path):
    from tka_planner.core.meshio import read_stl

    cube = gm.box(10.0)
    path = gm.write_stl(cube, tmp_path / "cube.stl")
    reloaded = read_stl(path)

    assert gm.volume(reloaded) == pytest.approx(1000.0, rel=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_geom_mesh.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'tka_planner.geom'`

- [ ] **Step 3: Create the package**

Create `tka_planner/geom/__init__.py`:

```python
"""Meshes and boolean geometry, without Blender.

The planning core reads meshes and measures them. This package is what *cuts* them,
which is the one job the pipeline previously had to borrow Blender for. Everything
here works in millimetres, like the core, and unit conversion happens only where a
scene is encoded for a viewer.
"""

from tka_planner.core.meshio import Mesh, read_stl, weld_vertices

__all__ = ["Mesh", "read_stl", "weld_vertices"]
```

- [ ] **Step 4: Implement the mesh helpers**

Create `tka_planner/geom/mesh.py`:

```python
"""Transforms, primitives and measures over :class:`tka_planner.core.meshio.Mesh`.

The primitives here replace the ``bpy.ops.mesh.primitive_*`` calls the Blender builder
used for cut planes, mechanical axes, landmark spheres, discard boxes and the insert
slab. Each is generated directly as vertices and faces, so the shapes are exactly
reproducible and testable against their analytic volume rather than whatever a given
Blender release happens to produce.

Every function takes and returns millimetres.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tka_planner.core.meshio import Mesh

__all__ = [
    "transformed", "box", "slab", "disc", "cylinder", "uv_sphere",
    "volume", "surface_area", "write_stl", "plane_matrix", "rotation_to",
    "translation",
]


def transformed(mesh: Mesh, matrix) -> Mesh:
    """Apply a 4x4 to a mesh's vertices, leaving its topology alone."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got {matrix.shape}.")

    vertices = mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3]
    return Mesh(
        vertices=vertices,
        faces=mesh.faces,
        source_path=mesh.source_path,
        sha256=mesh.sha256,
        metadata=dict(mesh.metadata),
    )


def translation(offset) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(offset, dtype=float)
    return matrix


def rotation_to(direction) -> np.ndarray:
    """A 3x3 rotation taking +Z onto ``direction``.

    Uses Rodrigues' formula about the axis between the two vectors. The antiparallel
    case has no such axis, so it is handled separately as a half turn about X.
    """
    target = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(target)
    if norm == 0.0:
        raise ValueError("Direction must not be the zero vector.")
    target = target / norm

    z = np.array([0.0, 0.0, 1.0])
    cosine = float(np.dot(z, target))
    if cosine > 1.0 - 1e-12:
        return np.eye(3)
    if cosine < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])

    axis = np.cross(z, target)
    sine = np.linalg.norm(axis)
    axis = axis / sine
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sine * cross + (1.0 - cosine) * (cross @ cross)


def plane_matrix(point_mm, normal_mm) -> np.ndarray:
    """A 4x4 whose origin is on the plane and whose +Z is the plane normal."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_to(normal_mm)
    matrix[:3, 3] = np.asarray(point_mm, dtype=float)
    return matrix


def _build(vertices, faces, matrix=None) -> Mesh:
    mesh = Mesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
    )
    return mesh if matrix is None else transformed(mesh, matrix)


def box(size_mm, matrix=None) -> Mesh:
    """An axis-aligned box centred on the origin.

    ``size_mm`` is one edge length, or a triple of them.
    """
    size = np.asarray(
        (size_mm,) * 3 if np.isscalar(size_mm) else size_mm, dtype=float
    )
    half = size / 2.0
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=float)
    vertices = signs * half

    # Outward-facing winding, which the volume formula and every boolean kernel need.
    faces = np.array([
        [0, 3, 2], [0, 2, 1],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [1, 2, 6], [1, 6, 5],  # +X
        [0, 4, 7], [0, 7, 3],  # -X
    ], dtype=np.int64)
    return _build(vertices, faces, matrix)


def slab(ml_mm: float, ap_mm: float, thickness_mm: float, matrix=None) -> Mesh:
    """A box standing on z = 0, centred in x and y.

    This is the insert spacer. It is built base-first so a thickness change moves only
    the top face, which is what lets the insert be re-stretched in place rather than
    rebuilt.
    """
    block = box((ml_mm, ap_mm, thickness_mm))
    lifted = transformed(block, translation((0.0, 0.0, thickness_mm / 2.0)))
    return lifted if matrix is None else transformed(lifted, matrix)


def disc(radius_mm: float, *, segments: int = 64, matrix=None) -> Mesh:
    """A filled circle in the z = 0 plane, for drawing a cut plane."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rim = np.column_stack([
        radius_mm * np.cos(angles),
        radius_mm * np.sin(angles),
        np.zeros(segments),
    ])
    vertices = np.vstack([[0.0, 0.0, 0.0], rim])
    faces = np.array(
        [[0, i + 1, (i + 1) % segments + 1] for i in range(segments)],
        dtype=np.int64,
    )
    return _build(vertices, faces, matrix)


def cylinder(radius_mm: float, length_mm: float, *, segments: int = 16,
             matrix=None) -> Mesh:
    """A closed cylinder centred on the origin, its axis along +Z."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    half = length_mm / 2.0
    ring = np.column_stack([
        radius_mm * np.cos(angles), radius_mm * np.sin(angles), np.zeros(segments)
    ])
    bottom = ring - np.array([0.0, 0.0, half])
    top = ring + np.array([0.0, 0.0, half])
    centre_bottom = np.array([[0.0, 0.0, -half]])
    centre_top = np.array([[0.0, 0.0, half]])
    vertices = np.vstack([bottom, top, centre_bottom, centre_top])

    bottom_centre = 2 * segments
    top_centre = 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([bottom_centre, j, i])
        faces.append([top_centre, segments + i, segments + j])
    return _build(vertices, np.array(faces, dtype=np.int64), matrix)


def uv_sphere(radius_mm: float, *, segments: int = 12, rings: int = 8,
              centre=(0.0, 0.0, 0.0)) -> Mesh:
    """A UV sphere, for a landmark marker."""
    thetas = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    phis = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    grid = []
    for theta in thetas:
        grid.append(np.column_stack([
            radius_mm * np.sin(theta) * np.cos(phis),
            radius_mm * np.sin(theta) * np.sin(phis),
            np.full(segments, radius_mm * np.cos(theta)),
        ]))
    body = np.vstack(grid)
    north = np.array([[0.0, 0.0, radius_mm]])
    south = np.array([[0.0, 0.0, -radius_mm]])
    vertices = np.vstack([body, north, south]) + np.asarray(centre, dtype=float)

    n_rows = len(thetas)
    north_index = n_rows * segments
    south_index = north_index + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([north_index, i, j])
        faces.append([south_index, (n_rows - 1) * segments + j,
                      (n_rows - 1) * segments + i])
    for row in range(n_rows - 1):
        base, nxt = row * segments, (row + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, nxt + i, nxt + j])
            faces.append([base + i, nxt + j, base + j])
    return _build(vertices, np.array(faces, dtype=np.int64))


def volume(mesh: Mesh) -> float:
    """Signed volume by the divergence theorem, in cubic millimetres.

    Positive for outward-facing winding. Used to compare two kernels' output without
    caring how either triangulated it.
    """
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def surface_area(mesh: Mesh) -> float:
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)


def write_stl(mesh: Mesh, path) -> Path:
    """Write a binary STL in millimetres.

    Written directly rather than through Blender's exporter, which is what removes the
    last reason an export path needed a Blender installation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    corners = mesh.vertices[mesh.faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0.0)

    records = np.zeros(
        mesh.n_triangles,
        dtype=np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                        ("attribute", "<u2")]),
    )
    records["normal"] = normals
    records["vertices"] = corners

    with path.open("wb") as handle:
        handle.write(b"tka-planner".ljust(80, b"\0"))
        handle.write(struct.pack("<I", mesh.n_triangles))
        handle.write(records.tobytes())
    return path
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_geom_mesh.py -q`
Expected: all pass.

- [ ] **Step 6: Confirm the whole suite is still green**

Run: `pytest -q`
Expected: 462 plus the new tests, no failures.

- [ ] **Step 7: Commit**

```bash
git add tka_planner/geom tests/test_geom_mesh.py
git commit -m "Add mesh transforms, primitives and measures without Blender"
```

---

### Task 3: Mesh repair

**Files:**
- Create: `tka_planner/geom/repair.py`
- Test: `tests/test_geom_repair.py`

**Interfaces:**
- Consumes: `Mesh`, `weld_vertices` from `tka_planner.core.meshio`.
- Produces:
  - `is_manifold(mesh: Mesh) -> bool`
  - `edge_report(mesh: Mesh) -> dict` with keys `boundary_edges`, `nonmanifold_edges`, `degenerate_faces`, `duplicate_faces`
  - `repair(mesh: Mesh) -> tuple[Mesh, tuple[str, ...]]` returning the repaired mesh and the list of what was done

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geom_repair.py`:

```python
"""Repair is what stands between a real segmentation and a kernel that wants manifolds."""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom import repair as gr


def test_a_welded_box_is_manifold():
    assert gr.is_manifold(gm.box(10.0)) is True


def test_an_unwelded_triangle_soup_is_not_manifold():
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    assert gr.is_manifold(soup) is False


def test_repair_welds_a_triangle_soup_into_a_manifold():
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    fixed, actions = gr.repair(soup)

    assert gr.is_manifold(fixed) is True
    assert "welded" in actions
    assert gm.volume(fixed) == pytest.approx(1000.0, rel=1e-9)


def test_repair_drops_a_degenerate_face():
    cube = gm.box(10.0)
    faces = np.vstack([cube.faces, [[0, 0, 1]]])
    broken = Mesh(vertices=cube.vertices, faces=faces)
    fixed, actions = gr.repair(broken)

    assert fixed.n_triangles == cube.n_triangles
    assert "dropped 1 degenerate face" in actions


def test_repair_drops_a_duplicated_face():
    cube = gm.box(10.0)
    faces = np.vstack([cube.faces, cube.faces[0:1]])
    broken = Mesh(vertices=cube.vertices, faces=faces)
    fixed, actions = gr.repair(broken)

    assert fixed.n_triangles == cube.n_triangles
    assert "dropped 1 duplicate face" in actions


def test_repair_reports_nothing_for_a_clean_mesh():
    fixed, actions = gr.repair(gm.box(10.0))
    assert actions == ()
    assert fixed.n_triangles == 12


def test_edge_report_counts_a_boundary():
    circle = gm.disc(5.0, segments=8)
    report = gr.edge_report(circle)
    assert report["boundary_edges"] == 8
    assert report["nonmanifold_edges"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_geom_repair.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.geom.repair'`

- [ ] **Step 3: Implement repair**

Create `tka_planner/geom/repair.py`:

```python
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

    if edge_report(working)["boundary_edges"] or not _is_welded(working):
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_geom_repair.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tka_planner/geom/repair.py tests/test_geom_repair.py
git commit -m "Report and repair the ways a segmentation fails to be manifold"
```

---

### Task 4: The kernel interface and the manifold3d backend

**Files:**
- Create: `tka_planner/geom/kernel.py`
- Create: `tka_planner/geom/kernels/__init__.py`
- Create: `tka_planner/geom/kernels/manifold_kernel.py`
- Modify: `pyproject.toml`
- Test: `tests/test_geom_kernel.py`

**Interfaces:**
- Consumes: `Mesh`, `repair`, `is_manifold`.
- Produces:
  - `KernelRecord(backend: str, operation: str, repaired_target: tuple[str, ...], repaired_tool: tuple[str, ...], fallback: str | None, notes: tuple[str, ...])` with `.to_dict()`
  - `KernelResult(mesh: Mesh, record: KernelRecord)`
  - `MeshKernel` protocol with `name: str`, `difference(target, tool) -> KernelResult`, `intersect(target, tool) -> KernelResult`
  - `default_kernel() -> MeshKernel`
  - `ManifoldKernel()` implementing the protocol

- [ ] **Step 1: Add the dependency and confirm the library's API shape**

```bash
pip install "manifold3d>=3.0"
python -c "import manifold3d as m; print(m.__version__); print([n for n in dir(m.Mesh) if not n.startswith('_')]); print([n for n in dir(m.Manifold) if not n.startswith('_')])"
```

Record the printed names. The adapter below is written against the 3.x names
`manifold3d.Mesh(vert_properties=..., tri_verts=...)`, `manifold3d.Manifold(mesh)` and
`Manifold.to_mesh()`. If the probe prints different names, use the printed ones and keep
everything else identical.

In `pyproject.toml`, change the dependency line and add the optional extra:

```toml
dependencies = ["numpy>=1.24", "manifold3d>=3.0"]

[project.optional-dependencies]
dev = ["pytest>=7.4"]
# GPL. Never imported at runtime; used only by the kernel cross-check test.
blender = ["bpy>=4.4"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_geom_kernel.py`:

```python
"""The boolean kernel, and the record it keeps of what it had to do."""

import numpy as np
import pytest

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernel import default_kernel
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel


@pytest.fixture
def kernel():
    return ManifoldKernel()


def test_default_kernel_is_manifold3d():
    assert default_kernel().name == "manifold3d"


def test_difference_of_a_box_and_a_half_covering_box(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.difference(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert result.record.backend == "manifold3d"
    assert result.record.operation == "difference"


def test_intersection_of_two_overlapping_boxes(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.intersect(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert result.record.operation == "intersect"


def test_difference_with_a_disjoint_tool_leaves_the_target_alone(kernel):
    target = gm.box(10.0)
    tool = gm.box(10.0, gm.translation((100.0, 0.0, 0.0)))
    result = kernel.difference(target, tool)

    assert gm.volume(result.mesh) == pytest.approx(1000.0, rel=1e-6)


def test_a_triangle_soup_input_is_repaired_and_the_repair_is_recorded(kernel):
    cube = gm.box(10.0)
    soup = Mesh(
        vertices=cube.vertices[cube.faces].reshape(-1, 3),
        faces=np.arange(cube.n_triangles * 3).reshape(-1, 3),
    )
    tool = gm.box(10.0, gm.translation((5.0, 0.0, 0.0)))
    result = kernel.difference(soup, tool)

    assert gm.volume(result.mesh) == pytest.approx(500.0, rel=1e-6)
    assert "welded" in result.record.repaired_target
    assert result.record.repaired_tool == ()


def test_the_record_serialises_for_the_plan_file(kernel):
    result = kernel.difference(gm.box(10.0), gm.box(4.0))
    record = result.record.to_dict()

    assert record["backend"] == "manifold3d"
    assert record["operation"] == "difference"
    assert record["fallback"] is None
    assert isinstance(record["repaired_target"], list)


def test_an_unrepairable_target_raises_with_a_useful_message(kernel):
    open_disc = gm.disc(5.0, segments=8)
    with pytest.raises(ValueError, match="not manifold"):
        kernel.difference(open_disc, gm.box(1.0))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_geom_kernel.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.geom.kernel'`

- [ ] **Step 4: Implement the interface**

Create `tka_planner/geom/kernel.py`:

```python
"""One interface for cutting, and the record of how a cut was actually made.

The project already states how every measured number was obtained. Geometry deserves
the same treatment: which solver ran, what it had to repair first, and whether it fell
back. A resected bone with no such record is a claim without a method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tka_planner.core.meshio import Mesh

__all__ = ["KernelRecord", "KernelResult", "MeshKernel", "default_kernel"]


@dataclass(frozen=True)
class KernelRecord:
    """How one boolean was performed."""

    backend: str
    operation: str
    repaired_target: tuple[str, ...] = ()
    repaired_tool: tuple[str, ...] = ()
    fallback: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "repaired_target": list(self.repaired_target),
            "repaired_tool": list(self.repaired_tool),
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class KernelResult:
    mesh: Mesh
    record: KernelRecord


@runtime_checkable
class MeshKernel(Protocol):
    """Difference and intersection over meshes in millimetres.

    Two operations is the whole surface the planner needs: the discard box and the
    cutting block are subtracted, and the bone shell is an intersection.
    """

    name: str

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult: ...

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult: ...


def default_kernel() -> MeshKernel:
    """The kernel everything uses unless told otherwise.

    Blender is never the default. It is available through
    ``geom.kernels.blender_kernel`` for the cross-check test alone, because it is GPL
    and this project is Apache 2.0.
    """
    from .kernels.manifold_kernel import ManifoldKernel

    return ManifoldKernel()
```

- [ ] **Step 5: Implement the manifold3d backend**

Create `tka_planner/geom/kernels/__init__.py` containing only a docstring:

```python
"""Boolean backends. Import a specific one; the default comes from ``geom.kernel``."""
```

Create `tka_planner/geom/kernels/manifold_kernel.py`:

```python
"""The ``manifold3d`` backend.

Chosen as the default for three reasons. It is Apache 2.0, so it does not compromise
this project's licence the way linking Blender would. It is fast enough that a
full-resolution resection is a few seconds rather than tens of them. And it refuses
non-manifold input rather than guessing, which turns a class of silent wrong answers
into a loud one.

That refusal is why :mod:`tka_planner.geom.repair` exists and why every repair is
recorded rather than done quietly.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh

from ..kernel import KernelRecord, KernelResult
from ..repair import is_manifold, repair

__all__ = ["ManifoldKernel"]


class ManifoldKernel:
    """Difference and intersection through ``manifold3d``."""

    name = "manifold3d"

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "difference")

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "intersect")

    # ------------------------------------------------------------------

    def _operate(self, target: Mesh, tool: Mesh, operation: str) -> KernelResult:
        import manifold3d

        clean_target, target_actions = self._prepare(target, "target")
        clean_tool, tool_actions = self._prepare(tool, "tool")

        a = self._to_manifold(clean_target)
        b = self._to_manifold(clean_tool)
        combined = a - b if operation == "difference" else a ^ b

        return KernelResult(
            mesh=self._from_manifold(combined, target),
            record=KernelRecord(
                backend=self.name,
                operation=operation,
                repaired_target=target_actions,
                repaired_tool=tool_actions,
                fallback=None,
            ),
        )

    @staticmethod
    def _prepare(mesh: Mesh, role: str) -> tuple[Mesh, tuple[str, ...]]:
        if is_manifold(mesh):
            return mesh, ()

        fixed, actions = repair(mesh)
        if not is_manifold(fixed):
            report = ", ".join(
                f"{k}={v}" for k, v in _report(fixed).items() if v
            )
            raise ValueError(
                f"The {role} mesh is not manifold and repair did not make it one "
                f"({report}). Fix the segmentation rather than forcing the cut: a "
                f"boolean against a broken surface returns broken geometry."
            )
        return fixed, actions

    @staticmethod
    def _to_manifold(mesh: Mesh):
        import manifold3d

        surface = manifold3d.Mesh(
            vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
        return manifold3d.Manifold(surface)

    @staticmethod
    def _from_manifold(solid, like: Mesh) -> Mesh:
        surface = solid.to_mesh()
        return Mesh(
            vertices=np.asarray(surface.vert_properties, dtype=np.float64)[:, :3],
            faces=np.asarray(surface.tri_verts, dtype=np.int64),
            source_path=like.source_path,
            metadata={**like.metadata, "kernel": "manifold3d"},
        )


def _report(mesh: Mesh) -> dict:
    from ..repair import edge_report

    return edge_report(mesh)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_geom_kernel.py -q`
Expected: all pass. If the probe in Step 1 printed different attribute names, fix
`_to_manifold` and `_from_manifold` and rerun.

- [ ] **Step 7: Commit**

```bash
git add tka_planner/geom/kernel.py tka_planner/geom/kernels tests/test_geom_kernel.py pyproject.toml
git commit -m "Cut meshes through a swappable kernel, with manifold3d as the default"
```

---

### Task 5: The Blender cross-check backend and the agreement gate

**Files:**
- Create: `tka_planner/geom/kernels/blender_kernel.py`
- Test: `tests/test_kernel_agreement.py`

**Interfaces:**
- Consumes: `KernelRecord`, `KernelResult`, `Mesh`, `gm.volume`, `gm.surface_area`.
- Produces:
  - `BlenderKernel()` implementing `MeshKernel`, `name = "blender-exact"`
  - `max_vertex_distance(a: Mesh, b: Mesh, *, sample: int = 2000) -> float` in `tests/test_kernel_agreement.py`

**This task is the spec's correctness gate.** If the two kernels disagree beyond
tolerance on the fixtures, stop and report rather than continuing to Task 6.

- [ ] **Step 1: Write the failing agreement test**

Create `tests/test_kernel_agreement.py`:

```python
"""The gate the port rests on: manifold3d must agree with Blender's exact solver.

Marked ``blender`` so it stays out of the default run, like every other test that needs
a Blender installation. Run it deliberately:

    pytest -m blender tests/test_kernel_agreement.py -v
"""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.geom.kernels.manifold_kernel import ManifoldKernel

pytestmark = pytest.mark.blender


def max_vertex_distance(a, b, *, sample: int = 2000) -> float:
    """Largest distance from a sample of ``a``'s vertices to the nearest vertex of ``b``.

    A one-sided sampled Hausdorff distance. Sampling keeps it affordable on a
    million-triangle bone, and one-sidedness is enough here because the test runs it
    both ways.
    """
    rng = np.random.default_rng(0)
    points = a.vertices
    if points.shape[0] > sample:
        points = points[rng.choice(points.shape[0], sample, replace=False)]

    worst = 0.0
    for chunk in np.array_split(points, max(1, len(points) // 200)):
        distances = np.linalg.norm(
            chunk[:, None, :] - b.vertices[None, :, :], axis=2
        )
        worst = max(worst, float(distances.min(axis=1).max()))
    return worst


@pytest.fixture
def kernels():
    from tka_planner.geom.kernels.blender_kernel import BlenderKernel

    return ManifoldKernel(), BlenderKernel()


def _assert_agree(one, other, *, volume_rel=1e-3, distance_mm=0.05):
    assert gm.volume(one) == pytest.approx(gm.volume(other), rel=volume_rel)
    assert gm.surface_area(one) == pytest.approx(
        gm.surface_area(other), rel=volume_rel * 5
    )
    assert max_vertex_distance(one, other) < distance_mm
    assert max_vertex_distance(other, one) < distance_mm


def test_the_two_kernels_agree_on_a_box_difference(kernels):
    manifold, blender = kernels
    target = gm.box(40.0)
    tool = gm.box(40.0, gm.translation((18.0, 3.0, -2.0)))

    _assert_agree(
        manifold.difference(target, tool).mesh,
        blender.difference(target, tool).mesh,
    )


def test_the_two_kernels_agree_on_an_oblique_cut(kernels):
    manifold, blender = kernels
    target = gm.uv_sphere(20.0, segments=64, rings=32)
    tool = gm.box(
        100.0, gm.plane_matrix((0.0, 0.0, 0.0), (0.3, 0.4, 0.87)) @ gm.translation(
            (0.0, 0.0, -50.0)
        )
    )

    _assert_agree(
        manifold.difference(target, tool).mesh,
        blender.difference(target, tool).mesh,
    )


def test_the_two_kernels_agree_on_an_intersection(kernels):
    manifold, blender = kernels
    target = gm.uv_sphere(20.0, segments=64, rings=32)
    tool = gm.box(24.0, gm.translation((8.0, 0.0, 0.0)))

    _assert_agree(
        manifold.intersect(target, tool).mesh,
        blender.intersect(target, tool).mesh,
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest -m blender tests/test_kernel_agreement.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.geom.kernels.blender_kernel'`

- [ ] **Step 3: Implement the Blender backend**

Create `tka_planner/geom/kernels/blender_kernel.py`:

```python
"""Blender's exact solver, wrapped as a kernel, for cross-checking only.

Blender is GPL and this project is Apache 2.0, so this backend is never the default,
never imported at runtime, and lives behind the ``blender`` optional extra. It exists
so the agreement test can show that ``manifold3d`` produces the same geometry as the
solver the published results were computed with.

``bpy`` is imported inside the methods rather than at module scope, so importing this
module on a machine with no Blender costs nothing and fails only when it is used.
"""

from __future__ import annotations

import numpy as np

from tka_planner.core.meshio import Mesh

from ..kernel import KernelRecord, KernelResult

__all__ = ["BlenderKernel"]


class BlenderKernel:
    """Difference and intersection through Blender's EXACT boolean modifier."""

    name = "blender-exact"

    def difference(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "difference", "DIFFERENCE")

    def intersect(self, target: Mesh, tool: Mesh) -> KernelResult:
        return self._operate(target, tool, "intersect", "INTERSECT")

    # ------------------------------------------------------------------

    def _operate(self, target: Mesh, tool: Mesh, operation: str,
                 blender_operation: str) -> KernelResult:
        import bpy

        target_object = self._to_object(target, "kernel_target")
        tool_object = self._to_object(tool, "kernel_tool")

        modifier = target_object.modifiers.new(name="Kernel", type="BOOLEAN")
        modifier.operation = blender_operation
        modifier.object = tool_object
        modifier.solver = "EXACT"

        bpy.context.view_layer.objects.active = target_object
        bpy.ops.object.modifier_apply(modifier=modifier.name)

        result = self._from_object(target_object, target)
        for obj in (target_object, tool_object):
            bpy.data.objects.remove(obj, do_unlink=True)

        return KernelResult(
            mesh=result,
            record=KernelRecord(
                backend=self.name, operation=operation, fallback=None,
                notes=("solver=EXACT",),
            ),
        )

    @staticmethod
    def _to_object(mesh: Mesh, name: str):
        """Build a Blender object in millimetres.

        The scene units are irrelevant here because nothing is rendered and nothing is
        exported; the kernel takes millimetres in and gives millimetres out.
        """
        import bpy

        data = bpy.data.meshes.new(name)
        data.from_pydata(
            [tuple(float(c) for c in v) for v in mesh.vertices],
            [],
            [tuple(int(i) for i in f) for f in mesh.faces],
        )
        data.update()
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    @staticmethod
    def _from_object(obj, like: Mesh) -> Mesh:
        data = obj.data
        vertices = np.empty(len(data.vertices) * 3, dtype=np.float64)
        data.vertices.foreach_get("co", vertices)

        faces = []
        for polygon in data.polygons:
            indices = list(polygon.vertices)
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))

        return Mesh(
            vertices=vertices.reshape(-1, 3),
            faces=np.asarray(faces, dtype=np.int64),
            source_path=like.source_path,
            metadata={**like.metadata, "kernel": "blender-exact"},
        )
```

- [ ] **Step 4: Run the agreement test**

Run: `pytest -m blender tests/test_kernel_agreement.py -v`
Expected: three passes.

If Blender is not installed, run `pip install "bpy>=4.4"` into a **separate** virtual
environment and run the test there. Do not add `bpy` to the working environment, because
its presence makes it too easy to import accidentally.

**Gate:** if any comparison fails, stop. Record the numbers, and report before starting
Task 6. This is the spec's section 4.3 gate and the whole port depends on it.

- [ ] **Step 5: Confirm the default run is unaffected**

Run: `pytest -q`
Expected: the agreement tests are deselected, everything else passes.

- [ ] **Step 6: Commit**

```bash
git add tka_planner/geom/kernels/blender_kernel.py tests/test_kernel_agreement.py
git commit -m "Cross-check manifold3d against Blender's exact solver"
```

---

### Task 6: The scene model

**Files:**
- Create: `tka_planner/scene/__init__.py`
- Create: `tka_planner/scene/model.py`
- Test: `tests/test_scene_model.py`

**Interfaces:**
- Consumes: `Mesh`.
- Produces:
  - `FEMORAL = "femoral"`, `TIBIAL = "tibial"`
  - `Node(name, mesh_id=None, rest=np.eye(4), set_id=None, colour=(0.6,0.6,0.6), alpha=1.0, visible=True, tags=dict)`
  - `Scene(meshes: dict[str, Mesh], nodes: dict[str, Node], set_poses: dict[str, np.ndarray], notes: list[str])` with:
    - `add(node: Node, mesh: Mesh | None = None) -> Node`
    - `world(name: str) -> np.ndarray`
    - `set_pose(set_id: str, matrix) -> None`
    - `nodes_in(set_id: str) -> list[Node]`
  - `SceneDelta(poses: dict[str, list], meshes: dict[str, str], visibility: dict[str, bool], scalars: dict, notes: list)` with `.to_dict()` and `.merge(other)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_model.py`:

```python
"""The scene model, and the rest-basis rule that keeps poses from drifting."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene.model import FEMORAL, TIBIAL, Node, Scene, SceneDelta


@pytest.fixture
def scene():
    built = Scene()
    built.add(Node(name="Femur", set_id=FEMORAL), gm.box(10.0))
    built.add(
        Node(name="Tibia", set_id=TIBIAL, rest=gm.translation((0.0, 0.0, -50.0))),
        gm.box(10.0),
    )
    return built


def test_a_node_with_no_set_pose_is_at_its_rest(scene):
    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -50.0)))


def test_a_set_pose_composes_onto_the_rest_basis(scene):
    scene.set_pose(TIBIAL, gm.translation((0.0, 0.0, 5.0)))
    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -45.0)))


def test_a_set_pose_does_not_reach_another_set(scene):
    scene.set_pose(TIBIAL, gm.translation((0.0, 0.0, 5.0)))
    assert scene.world("Femur") == pytest.approx(np.eye(4))


def test_setting_the_same_pose_twice_is_idempotent(scene):
    pose = gm.plane_matrix((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
    scene.set_pose(TIBIAL, pose)
    once = scene.world("Tibia")
    scene.set_pose(TIBIAL, pose)

    assert scene.world("Tibia") == pytest.approx(once)


def test_returning_a_set_pose_to_identity_returns_the_node_to_rest(scene):
    scene.set_pose(TIBIAL, gm.plane_matrix((9.0, 9.0, 9.0), (1.0, 1.0, 0.0)))
    scene.set_pose(TIBIAL, np.eye(4))

    assert scene.world("Tibia") == pytest.approx(gm.translation((0.0, 0.0, -50.0)))


def test_meshes_are_shared_by_id_not_duplicated(scene):
    cube = gm.box(10.0)
    scene.add(Node(name="Shell", set_id=FEMORAL), cube)
    scene.add(Node(name="ShellCopy", set_id=FEMORAL, mesh_id=scene.nodes["Shell"].mesh_id))

    assert scene.nodes["ShellCopy"].mesh_id == scene.nodes["Shell"].mesh_id
    assert len(scene.meshes) == 3


def test_nodes_in_lists_only_that_set(scene):
    assert [node.name for node in scene.nodes_in(TIBIAL)] == ["Tibia"]


def test_world_of_an_unknown_node_raises(scene):
    with pytest.raises(KeyError, match="Patella"):
        scene.world("Patella")


def test_a_delta_serialises_to_plain_types():
    delta = SceneDelta(
        poses={"TibialSet": np.eye(4)},
        visibility={"Femur": False},
        scalars={"extension_gap_medial_mm": 9.25},
    )
    payload = delta.to_dict()

    assert payload["poses"]["TibialSet"] == [
        [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    ]
    assert payload["visibility"] == {"Femur": False}
    assert payload["scalars"]["extension_gap_medial_mm"] == 9.25


def test_merging_deltas_keeps_the_later_value():
    first = SceneDelta(poses={"A": np.eye(4)}, scalars={"x": 1})
    second = SceneDelta(poses={"A": gm.translation((1.0, 0.0, 0.0))}, scalars={"y": 2})
    merged = first.merge(second)

    assert merged.poses["A"] == pytest.approx(gm.translation((1.0, 0.0, 0.0)))
    assert merged.scalars == {"x": 1, "y": 2}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scene_model.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.scene'`

- [ ] **Step 3: Implement the model**

Create `tka_planner/scene/__init__.py`:

```python
"""A scene, without a renderer.

What the Blender builder did to a ``bpy`` scene, this package does to a plain data
structure: place the bones, the cut planes, the axes, the landmarks, the implants and
the cutting blocks, then move them as the plan changes. Nothing here draws anything, so
the same scene serves a browser, a test, and later a tracker feed.
"""
```

Create `tka_planner/scene/model.py`:

```python
"""Nodes, sets, and the delta that says what moved.

The structure is two **sets**, femoral and tibial, each holding a bone, its implant
parts and its landmarks. A set carries one pose. Everything inside it carries a **rest
transform** relative to the set origin, fixed when the scene is built and never touched
again.

That separation is deliberate and it is the fix for a real bug. The Blender trial rig
posed a parented empty and composed rotations onto whatever the object's current
rotation happened to be, so repeated poses drifted and zeroing the controls did not
return to the start. Here a pose is *assigned*, never accumulated, and the world
transform is always ``set_pose @ rest``. Drift of that kind cannot be expressed.
"""

from __future__ import annotations

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
    colour: tuple[float, float, float] = (0.62, 0.62, 0.62)
    alpha: float = 1.0
    visible: bool = True
    tags: dict = field(default_factory=dict)


@dataclass
class Scene:
    """A mesh table, a node table, and one pose per set."""

    meshes: dict[str, Mesh] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    set_poses: dict[str, np.ndarray] = field(
        default_factory=lambda: {FEMORAL: np.eye(4), TIBIAL: np.eye(4)}
    )
    notes: list[str] = field(default_factory=list)

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

    def nodes_in(self, set_id: str) -> list[Node]:
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
        """Combine two deltas, the later one winning where they overlap."""
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
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scene_model.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tka_planner/scene tests/test_scene_model.py
git commit -m "Model the scene as sets, rest transforms and deltas"
```

---

### Task 7: Motion

Built before the builder, because the builder needs the flexion axis to place the
tibial set's rest basis.

**Files:**
- Create: `tka_planner/scene/motion.py`
- Test: `tests/test_scene_motion.py`

**Interfaces:**
- Consumes: `gm.rotation_to`, `SurgicalPlan`, frames and `LandmarkSet` from `core`.
- Produces:
  - `flexion_axis(plan, femoral_frame, landmarks) -> tuple[np.ndarray, np.ndarray]` returning `(origin_mm, direction)`
  - `pivot_frame(origin_mm, direction) -> np.ndarray` (4x4 whose +X is the flexion axis)
  - `pose_about(pivot: np.ndarray, rotation: np.ndarray, offset=(0,0,0)) -> np.ndarray`
  - `flexion_pose(pivot, *, flexion_deg) -> np.ndarray`
  - `trial_pose(pivot, *, flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0) -> np.ndarray`
  - `rotation_about(axis, degrees) -> np.ndarray` (3x3)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_motion.py`:

```python
"""Flexion and trial reduction as pure matrix composition."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene import motion


@pytest.fixture
def pivot():
    return motion.pivot_frame(
        origin_mm=np.array([0.0, 0.0, 0.0]), direction=np.array([1.0, 0.0, 0.0])
    )


def test_pivot_frame_puts_x_on_the_flexion_axis(pivot):
    assert pivot[:3, 0] == pytest.approx([1.0, 0.0, 0.0])


def test_pivot_frame_is_orthonormal(pivot):
    assert pivot[:3, :3] @ pivot[:3, :3].T == pytest.approx(np.eye(3), abs=1e-12)


def test_zero_flexion_is_the_identity_pose(pivot):
    assert motion.flexion_pose(pivot, flexion_deg=0.0) == pytest.approx(np.eye(4))


def test_flexion_rotates_about_the_axis_through_its_origin(pivot):
    pose = motion.flexion_pose(pivot, flexion_deg=90.0)
    point = np.array([0.0, 0.0, -50.0, 1.0])
    moved = pose @ point

    assert np.linalg.norm(moved[:3]) == pytest.approx(50.0)
    assert moved[0] == pytest.approx(0.0, abs=1e-9)


def test_flexion_leaves_a_point_on_the_axis_alone(pivot):
    pose = motion.flexion_pose(pivot, flexion_deg=75.0)
    assert (pose @ np.array([10.0, 0.0, 0.0, 1.0]))[:3] == pytest.approx(
        [10.0, 0.0, 0.0]
    )


def test_a_pivot_away_from_the_origin_still_holds_its_own_point_fixed():
    offset = motion.pivot_frame(
        origin_mm=np.array([5.0, -7.0, 12.0]), direction=np.array([0.0, 1.0, 0.0])
    )
    pose = motion.flexion_pose(offset, flexion_deg=30.0)
    assert (pose @ np.array([5.0, -7.0, 12.0, 1.0]))[:3] == pytest.approx(
        [5.0, -7.0, 12.0]
    )


def test_all_trial_controls_at_zero_is_the_identity(pivot):
    assert motion.trial_pose(pivot) == pytest.approx(np.eye(4))


def test_returning_every_trial_control_to_zero_returns_to_the_identity(pivot):
    motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0, drawer_ap_mm=3.0)
    assert motion.trial_pose(
        pivot, flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0
    ) == pytest.approx(np.eye(4))


def test_a_trial_pose_is_the_same_however_many_times_it_is_asked_for(pivot):
    once = motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0)
    twice = motion.trial_pose(pivot, flexion_deg=40.0, varus_valgus_deg=6.0)
    assert once == pytest.approx(twice)


def test_the_drawer_translates_along_the_pivot_rest_y_not_the_flexed_one(pivot):
    straight = motion.trial_pose(pivot, drawer_ap_mm=10.0)
    flexed = motion.trial_pose(pivot, flexion_deg=90.0, drawer_ap_mm=10.0)

    rest_y = pivot[:3, 1]
    assert straight[:3, 3] == pytest.approx(rest_y * 10.0)
    assert flexed[:3, 3] - (flexed[:3, :3] @ np.zeros(3)) == pytest.approx(
        straight[:3, 3], abs=1e-9
    )


def test_trial_pose_is_a_rigid_transform(pivot):
    pose = motion.trial_pose(pivot, flexion_deg=33.0, varus_valgus_deg=-4.0,
                             drawer_ap_mm=2.0)
    rotation = pose[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_flexion_pose_agrees_with_trial_pose_at_the_same_angle(pivot):
    assert motion.flexion_pose(pivot, flexion_deg=55.0) == pytest.approx(
        motion.trial_pose(pivot, flexion_deg=55.0)
    )


def test_rotation_about_an_arbitrary_axis_is_orthonormal():
    rotation = motion.rotation_about(np.array([0.3, 0.5, -0.8]), 27.0)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scene_motion.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.scene.motion'`

- [ ] **Step 3: Implement motion**

Create `tka_planner/scene/motion.py`:

```python
"""Flexion and trial reduction, as matrices.

Both are rigid transforms of the tibial set about the flexion axis. The Blender version
achieved them with a parented empty, an animation action and a stack of baked meshes;
none of that was about the motion, all of it was about making Blender's modifier stack
stop re-evaluating during playback. With committed geometry there is nothing to
re-evaluate, so the motion is what it always was: a 4x4.

The composition rule is ``pose = pivot @ R @ pivot^-1``, so a pose is a function of the
control values alone. Setting the same values twice gives the same matrix, and zeroing
them gives the identity. The trial rig's drift bug is unreachable from here.
"""

from __future__ import annotations

import numpy as np

from tka_planner.geom.mesh import rotation_to

__all__ = [
    "flexion_axis", "pivot_frame", "pose_about", "flexion_pose", "trial_pose",
    "rotation_about",
]


def flexion_axis(plan, femoral_frame, landmarks) -> tuple[np.ndarray, np.ndarray]:
    """Where the knee hinges, and about what.

    The transepicondylar axis, through the midpoint of the epicondyles. The femoral
    condyles are close to circular in the sagittal plane and the epicondyles sit near
    the centres of those circles, so the tibia rides around them at a nearly constant
    radius and stays in contact through the arc.

    Ported unchanged from the Blender builder, including its fallback to the femoral
    component origin when neither epicondyle pair is available. Using the posterior
    condyles instead put the axis about two centimetres off the centre of curvature and
    the joint swung apart as it flexed.
    """
    direction = np.asarray(femoral_frame.y_patient_left, dtype=float)

    for pair in (
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_sulcus"),
        ("femur.epicondyle_lateral", "femur.epicondyle_medial_prominence"),
    ):
        if landmarks is not None and landmarks.available(*pair):
            lateral, medial = landmarks.require(*pair)
            midpoint = (np.asarray(lateral) + np.asarray(medial)) / 2.0
            return midpoint, direction

    return np.asarray(plan.components["femoral_component"][:3, 3], dtype=float), direction


def pivot_frame(origin_mm, direction) -> np.ndarray:
    """A 4x4 at the flexion axis whose +X runs along it.

    Flexion turns about local X, a varus or valgus stress about local Y, and the drawer
    slides along local Y. Naming the axes once here is what lets the pose functions
    below stay three lines each.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("The flexion axis direction must not be the zero vector.")

    # `rotation_to` takes +Z onto a direction; rolling it by -90 degrees about Y puts
    # +X there instead, which is the convention the trial controls are written in.
    basis = rotation_to(direction / norm) @ rotation_about(
        np.array([0.0, 1.0, 0.0]), -90.0
    )
    frame = np.eye(4)
    frame[:3, :3] = basis
    frame[:3, 3] = np.asarray(origin_mm, dtype=float)
    return frame


def rotation_about(axis, degrees: float) -> np.ndarray:
    """A 3x3 rotation of ``degrees`` about ``axis``, by Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(float(degrees))
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def pose_about(pivot: np.ndarray, rotation: np.ndarray, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A world pose applying ``rotation`` in the pivot's own frame, about its origin.

    ``offset`` is a translation in the pivot's frame, applied after the rotation, which
    is what the drawer test needs.
    """
    local = np.eye(4)
    local[:3, :3] = rotation
    local[:3, 3] = np.asarray(offset, dtype=float)
    return pivot @ local @ np.linalg.inv(pivot)


def flexion_pose(pivot: np.ndarray, *, flexion_deg: float) -> np.ndarray:
    """The tibial set pose at a point in the flexion arc."""
    return pose_about(pivot, rotation_about(np.array([1.0, 0.0, 0.0]), flexion_deg))


def trial_pose(
    pivot: np.ndarray,
    *,
    flexion_deg: float = 0.0,
    varus_valgus_deg: float = 0.0,
    drawer_ap_mm: float = 0.0,
) -> np.ndarray:
    """The tibial set pose under the three trial controls.

    Flexion turns about the pivot's local X. A varus or valgus stress then turns about
    the tibia's *already flexed* local Y, which is what a real stress exam is relative
    to. The drawer instead slides along the pivot's rest Y, because "anterior" for that
    test means the joint's own anterior rather than wherever flexion left the tibia
    pointing. That distinction is why the offset is applied outside the rotation below.
    """
    rotation = (
        rotation_about(np.array([1.0, 0.0, 0.0]), flexion_deg)
        @ rotation_about(np.array([0.0, 1.0, 0.0]), varus_valgus_deg)
    )
    turned = pose_about(pivot, rotation)

    slide = np.eye(4)
    slide[:3, 3] = pivot[:3, 1] * float(drawer_ap_mm)
    return slide @ turned
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scene_motion.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tka_planner/scene/motion.py tests/test_scene_motion.py
git commit -m "Compose flexion and trial poses about a stored pivot"
```

---

### Task 8: The scene builder

**Files:**
- Create: `tka_planner/scene/build.py`
- Modify: `tka_planner/scene/model.py` (add a `pivot` field to `Scene`, Step 4)
- Test: `tests/test_scene_build.py`

**Interfaces:**
- Consumes: `Scene`, `Node`, `FEMORAL`, `TIBIAL`, `motion.flexion_axis`, `motion.pivot_frame`, `geom.mesh`, `core.meshio.read_stl`.
- Produces:
  - Colour constants: `BONE_COLOUR`, `RESECTED_COLOUR`, `IMPLANT_COLOUR`, `PLANE_COLOUR`, `AXIS_COLOUR`, `BLOCK_COLOUR`, `SHELL_COLOUR`, `LANDMARK_COLOURS`
  - `resolve_component_meshes(library, *, chart, sizing, side) -> dict` (moved verbatim from `blender/build.py:1087`)
  - `build_scene(*, femur_path, tibia_path, plan, femoral_frame, tibial_frame, landmarks=None, components=None, show_planes=True, show_axes=True, show_landmarks=False, insert_thickness_mm=None, insert_footprint_mm=None) -> Scene`
  - `INSERT_SPACER = "TibialInsertSpacer"`

The scene it builds carries the uncut bones and the pivot. Cutting is Task 9.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_build.py`:

```python
"""Building a scene from a plan, with no Blender anywhere."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.core.planning import MECHANICAL, plan_alignment
from tka_planner.geom import mesh as gm
from tka_planner.scene import build as sb
from tka_planner.scene.model import FEMORAL, TIBIAL


@pytest.fixture
def knee(tmp_path):
    """A synthetic knee written to STL, plus its plan and frames."""
    case = synthetic_knee()
    femur_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "femur.stl")
    tibia_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "tibia.stl")

    plan = plan_alignment(
        case.landmarks, case.femoral_frame, case.tibial_frame,
        target=MECHANICAL, femoral_thickness_mm=9.0, tibial_resection_mm=8.0,
        native_slope_deg=5.0,
    )
    return case, plan, femur_path, tibia_path


@pytest.fixture
def scene(knee):
    case, plan, femur_path, tibia_path = knee
    return sb.build_scene(
        femur_path=femur_path, tibia_path=tibia_path, plan=plan,
        femoral_frame=case.femoral_frame, tibial_frame=case.tibial_frame,
        landmarks=case.landmarks, show_landmarks=True,
    )


def test_both_bones_are_present_in_their_own_sets(scene):
    assert scene.nodes["Femur"].set_id == FEMORAL
    assert scene.nodes["Tibia"].set_id == TIBIAL


def test_a_cut_plane_is_built_for_every_resection(knee, scene):
    _, plan, _, _ = knee
    for name in plan.resections:
        assert name in scene.nodes


def test_a_cut_plane_sits_on_its_resection(knee, scene):
    _, plan, _, _ = knee
    resection = plan.resections["femoral_distal"]
    world = scene.world("femoral_distal")

    assert world[:3, 3] == pytest.approx(resection.point)
    assert world[:3, 2] == pytest.approx(
        np.asarray(resection.normal) / np.linalg.norm(resection.normal)
    )


def test_both_mechanical_axes_are_drawn(scene):
    assert "FemoralMechanicalAxis" in scene.nodes
    assert "TibialMechanicalAxis" in scene.nodes


def test_landmarks_become_nodes_tagged_as_landmarks(knee, scene):
    case, _, _, _ = knee
    usable = [landmark for landmark in case.landmarks if landmark.is_usable]

    tagged = [node for node in scene.nodes.values() if node.tags.get("landmark")]
    assert len(tagged) == len(usable)


def test_a_landmark_sits_where_the_landmark_set_says(knee, scene):
    case, _, _, _ = knee
    landmark = next(one for one in case.landmarks if one.is_usable)
    assert scene.world(landmark.id)[:3, 3] == pytest.approx(landmark.position_mm)


def test_landmarks_belong_to_the_set_of_their_bone(knee, scene):
    femoral = [
        node for node in scene.nodes.values()
        if node.tags.get("landmark") and node.name.startswith("femur.")
    ]
    assert femoral and all(node.set_id == FEMORAL for node in femoral)
    tibial = [
        node for node in scene.nodes.values()
        if node.tags.get("landmark") and node.name.startswith("tibia.")
    ]
    assert tibial and all(node.set_id == TIBIAL for node in tibial)


def test_landmarks_can_be_built_hidden(knee):
    case, plan, femur_path, tibia_path = knee
    hidden = sb.build_scene(
        femur_path=femur_path, tibia_path=tibia_path, plan=plan,
        femoral_frame=case.femoral_frame, tibial_frame=case.tibial_frame,
        landmarks=case.landmarks, show_landmarks=False,
    )
    marker = next(
        node for node in hidden.nodes.values() if node.tags.get("landmark")
    )
    assert marker.visible is False


def test_the_pivot_is_recorded_on_the_scene(scene):
    assert scene.notes is not None
    assert scene.pivot.shape == (4, 4)


def test_the_insert_spacer_is_built_at_the_requested_thickness(knee):
    case, plan, femur_path, tibia_path = knee
    scene = sb.build_scene(
        femur_path=femur_path, tibia_path=tibia_path, plan=plan,
        femoral_frame=case.femoral_frame, tibial_frame=case.tibial_frame,
        insert_thickness_mm=9.0, insert_footprint_mm=(60.0, 40.0),
    )
    spacer = scene.mesh_for(sb.INSERT_SPACER)
    assert gm.volume(spacer) == pytest.approx(60.0 * 40.0 * 9.0, rel=1e-6)


def test_no_insert_spacer_when_the_insert_is_off(scene):
    assert sb.INSERT_SPACER not in scene.nodes


def test_building_imports_no_bpy():
    import sys
    assert "bpy" not in sys.modules
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scene_build.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.scene.build'`

- [ ] **Step 3: Move `resolve_component_meshes` across unchanged**

Copy the whole function body from `tka_planner/blender/build.py` lines 1087 to 1213
into the new `tka_planner/scene/build.py`. It touches no `bpy` at all, so it moves
verbatim. Keep every comment.

- [ ] **Step 4: Implement the builder**

Add to `tka_planner/scene/build.py`, above the copied function:

```python
"""Turn a computed plan into a scene, without a renderer.

This is the Blender builder's job with Blender taken out. Everything visible was
decided in :mod:`tka_planner.core` before this module ran; the builder only places it.
That inversion is what kept the planning logic testable, and it is why removing Blender
costs nothing here.

The bones are placed uncut. Cutting is :mod:`tka_planner.scene.resect`, and it happens
once, on commit, rather than on every change of a control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tka_planner.core.meshio import read_stl
from tka_planner.geom import mesh as gm

from . import motion
from .model import FEMORAL, TIBIAL, Node, Scene

__all__ = ["build_scene", "resolve_component_meshes", "INSERT_SPACER"]

INSERT_SPACER = "TibialInsertSpacer"

# One colour per category, carried over from the Blender scene so the two look alike
# while both exist: bone grey, implant blue, cutting-block green, shell red, plane cyan,
# axis orange, landmark yellow. Cut bone keeps its own warmer grey, so a resected
# surface still reads as bone rather than as implant.
BONE_COLOUR = (0.62, 0.62, 0.62)
RESECTED_COLOUR = (0.55, 0.52, 0.50)
IMPLANT_COLOUR = (0.18, 0.38, 0.85)
PLANE_COLOUR = (0.25, 0.80, 0.85)
AXIS_COLOUR = (0.95, 0.55, 0.10)
BLOCK_COLOUR = (0.30, 0.70, 0.35)
SHELL_COLOUR = (0.85, 0.18, 0.18)
LANDMARK_COLOURS = {
    "present": (1.00, 0.85, 0.05),
    "estimated": (0.85, 0.60, 0.05),
    "derived": (1.00, 0.95, 0.45),
}

PLANE_RADIUS_MM = 55.0
AXIS_LENGTH_MM = 150.0
LANDMARK_RADIUS_MM = 2.5


def build_scene(
    *,
    femur_path,
    tibia_path,
    plan,
    femoral_frame,
    tibial_frame,
    landmarks=None,
    components: dict | None = None,
    show_planes: bool = True,
    show_axes: bool = True,
    show_landmarks: bool = False,
    insert_thickness_mm: float | None = None,
    insert_footprint_mm: tuple | None = None,
) -> Scene:
    """Build the full scene from an already-computed plan.

    ``components`` maps a component name to ``{"path": ..., "scale": ..., "group": ...}``
    as :func:`resolve_component_meshes` returns, so the implants appear seated on their
    cuts. Omit it to show the anatomy and the planned cuts alone.
    """
    scene = Scene()

    scene.add(
        Node(name="Femur", set_id=FEMORAL, colour=BONE_COLOUR,
             tags={"bone": True}),
        read_stl(femur_path),
    )
    scene.add(
        Node(name="Tibia", set_id=TIBIAL, colour=BONE_COLOUR,
             tags={"bone": True}),
        read_stl(tibia_path),
    )

    if show_planes:
        for name, resection in plan.resections.items():
            scene.add(
                Node(
                    name=name,
                    set_id=_set_for(name),
                    rest=gm.plane_matrix(resection.point, resection.normal),
                    colour=PLANE_COLOUR,
                    alpha=0.35,
                    tags={"plane": True, "edit_only": True},
                ),
                gm.disc(PLANE_RADIUS_MM),
            )

    if show_axes:
        for label, frame, set_id, length in (
            ("FemoralMechanicalAxis", femoral_frame, FEMORAL, AXIS_LENGTH_MM),
            ("TibialMechanicalAxis", tibial_frame, TIBIAL, -AXIS_LENGTH_MM),
        ):
            direction = np.asarray(frame.z_proximal, dtype=float)
            direction = direction / np.linalg.norm(direction)
            midpoint = np.asarray(frame.origin, dtype=float) + direction * (length / 2.0)
            matrix = gm.plane_matrix(midpoint, direction)
            scene.add(
                Node(name=label, set_id=set_id, rest=matrix, colour=AXIS_COLOUR,
                     tags={"axis": True, "edit_only": True}),
                gm.cylinder(1.2, abs(length)),
            )

    if landmarks is not None:
        marker = gm.uv_sphere(LANDMARK_RADIUS_MM)
        for landmark in landmarks:
            if not landmark.is_usable:
                continue
            scene.add(
                Node(
                    name=landmark.id,
                    set_id=_set_for(landmark.id),
                    rest=gm.translation(landmark.position_mm),
                    colour=LANDMARK_COLOURS.get(
                        landmark.status.value, (0.9, 0.8, 0.2)
                    ),
                    visible=show_landmarks,
                    tags={"landmark": True, "status": landmark.status.value},
                ),
                marker,
            )

    for name, spec in (components or {}).items():
        pose = plan.components.get(f"{spec.get('group', 'femoral')}_component")
        if pose is None or not Path(spec["path"]).is_file():
            scene.notes.append(f"{name}: mesh not found, skipped")
            continue
        group = spec.get("group", "femoral")
        scene.add(
            Node(
                name=name,
                set_id=FEMORAL if group == "femoral" else TIBIAL,
                rest=_seat(pose, spec.get("scale", 1.0)),
                colour=_component_colour(name),
                tags={
                    "component": True,
                    "group": group,
                    "scale": float(spec.get("scale", 1.0)),
                    "source_ml": spec.get("source_ml"),
                },
            ),
            read_stl(spec["path"]),
        )

    if insert_thickness_mm is not None:
        ml_mm, ap_mm = insert_footprint_mm or (60.0, 40.0)
        scene.add(
            Node(
                name=INSERT_SPACER,
                set_id=TIBIAL,
                rest=_seat(plan.components["tibial_component"], 1.0),
                colour=IMPLANT_COLOUR,
                alpha=0.6,
                tags={"insert": True, "footprint_mm": (ml_mm, ap_mm)},
            ),
            gm.slab(ml_mm, ap_mm, insert_thickness_mm),
        )

    origin, direction = motion.flexion_axis(plan, femoral_frame, landmarks)
    scene.pivot = motion.pivot_frame(origin, direction)
    return scene


def _seat(pose, scale_factor: float) -> np.ndarray:
    """Put a component's CAD origin on a cut, at a parametric scale.

    Scaling is about the CAD origin, so the parametric factor never shifts the seating.
    The library gives every part of a bone one shared origin on the cut surface, which
    is what makes a cutting block and its implant coincide exactly under the same pose.
    """
    pose = np.asarray(pose, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = pose[:3, :3] * float(scale_factor)
    matrix[:3, 3] = pose[:3, 3]
    return matrix


def _set_for(name: str) -> str:
    text = str(name).lower()
    if text.startswith("tibia") or text.startswith("fibula") or "tibial" in text:
        return TIBIAL
    return FEMORAL


def _component_colour(name: str):
    if "shell" in name:
        return SHELL_COLOUR
    if "cutting_block" in name:
        return BLOCK_COLOUR
    return IMPLANT_COLOUR
```

Add a `pivot` field to `Scene` in `tka_planner/scene/model.py`, after `notes`:

```python
    pivot: np.ndarray = field(default_factory=_identity)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_scene_build.py tests/test_scene_model.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tka_planner/scene/build.py tka_planner/scene/model.py tests/test_scene_build.py
git commit -m "Build the planning scene without Blender"
```

---

### Task 9: The scene updater

**Files:**
- Create: `tka_planner/scene/update.py`
- Test: `tests/test_scene_update.py`

**Interfaces:**
- Consumes: `Scene`, `SceneDelta`, `build._seat`, `build.INSERT_SPACER`, `gm.plane_matrix`, `gm.slab`.
- Produces:
  - `update_scene(scene, plan, *, insert_thickness_mm=None, implant_ml_mm=None) -> SceneDelta`
  - `set_visibility(scene, predicate, visible) -> SceneDelta`
  - `isolate_landmarks(scene, enabled: bool) -> SceneDelta`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_update.py`:

```python
"""Re-posing a built scene from a re-planned SurgicalPlan."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.core.planning import MECHANICAL, Adjustments, plan_alignment
from tka_planner.geom import mesh as gm
from tka_planner.scene import build as sb
from tka_planner.scene import update as su


@pytest.fixture
def case(tmp_path):
    knee = synthetic_knee()
    femur_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "femur.stl")
    tibia_path = gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "tibia.stl")

    def make(**adjustments):
        return plan_alignment(
            knee.landmarks, knee.femoral_frame, knee.tibial_frame,
            target=MECHANICAL, femoral_thickness_mm=9.0, tibial_resection_mm=8.0,
            native_slope_deg=5.0,
            adjustments=Adjustments(**adjustments) if adjustments else None,
        )

    plan = make()
    scene = sb.build_scene(
        femur_path=femur_path, tibia_path=tibia_path, plan=plan,
        femoral_frame=knee.femoral_frame, tibial_frame=knee.tibial_frame,
        landmarks=knee.landmarks, show_landmarks=True,
        insert_thickness_mm=9.0, insert_footprint_mm=(60.0, 40.0),
    )
    return scene, plan, make


def test_an_unchanged_plan_moves_the_cut_plane_nowhere(case):
    scene, plan, _ = case
    before = scene.world("femoral_distal").copy()
    su.update_scene(scene, plan)

    assert scene.world("femoral_distal") == pytest.approx(before)


def test_a_deeper_tibial_resection_moves_the_tibial_plane(case):
    scene, _, make = case
    before = scene.world("tibial_proximal").copy()
    su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert scene.world("tibial_proximal")[:3, 3] != pytest.approx(before[:3, 3])


def test_the_delta_names_only_what_moved(case):
    scene, _, make = case
    delta = su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert "tibial_proximal" in delta.poses
    assert "Femur" not in delta.poses


def test_the_delta_carries_the_plane_pose_as_a_matrix(case):
    scene, _, make = case
    delta = su.update_scene(scene, make(tibial_resection_delta_mm=3.0))

    assert np.asarray(delta.poses["tibial_proximal"]).shape == (4, 4)


def test_changing_the_implant_size_rescales_without_replacing_the_mesh(case):
    scene, plan, _ = case
    scene.add_component_for_test = None  # no components in this fixture
    delta = su.update_scene(scene, plan, implant_ml_mm=70.0)

    assert delta.meshes == {}


def test_the_insert_slab_is_rebuilt_at_a_new_thickness(case):
    scene, plan, _ = case
    delta = su.update_scene(scene, plan, insert_thickness_mm=12.0)

    assert sb.INSERT_SPACER in delta.meshes
    assert gm.volume(scene.mesh_for(sb.INSERT_SPACER)) == pytest.approx(
        60.0 * 40.0 * 12.0, rel=1e-6
    )


def test_turning_the_insert_off_hides_the_slab(case):
    scene, plan, _ = case
    delta = su.update_scene(scene, plan, insert_thickness_mm=None)

    assert delta.visibility[sb.INSERT_SPACER] is False
    assert scene.nodes[sb.INSERT_SPACER].visible is False


def test_isolate_landmarks_hides_everything_else(case):
    scene, _, _ = case
    su.isolate_landmarks(scene, True)

    assert scene.nodes["Femur"].visible is False
    marker = next(n for n in scene.nodes.values() if n.tags.get("landmark"))
    assert marker.visible is True


def test_isolate_landmarks_restores_what_it_hid(case):
    scene, _, _ = case
    su.isolate_landmarks(scene, True)
    su.isolate_landmarks(scene, False)

    assert scene.nodes["Femur"].visible is True


def test_the_delta_survives_serialisation(case):
    scene, _, make = case
    payload = su.update_scene(scene, make(tibial_slope_delta_deg=2.0)).to_dict()

    assert isinstance(payload["poses"]["tibial_proximal"], list)
    assert len(payload["poses"]["tibial_proximal"]) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scene_update.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.scene.update'`

- [ ] **Step 3: Implement the updater**

Create `tka_planner/scene/update.py`:

```python
"""Re-pose a built scene from a re-planned :class:`SurgicalPlan`.

This is the path every plan control takes. Nothing is imported, created or deleted: the
cut planes, the components and the insert are already in the scene and only their rest
transforms change. Planning itself is a few dozen numpy operations, so the whole round
trip is fast enough to run on every change of a value.

Nothing here cuts anything. That is the Plan-mode guarantee: no boolean runs while a
control is moving, so the update cost is a handful of matrix assignments regardless of
how dense the segmentation is.
"""

from __future__ import annotations

import numpy as np

from tka_planner.geom import mesh as gm

from .build import INSERT_SPACER, _seat
from .model import Scene, SceneDelta

__all__ = ["update_scene", "set_visibility", "isolate_landmarks"]


def update_scene(
    scene: Scene,
    plan,
    *,
    insert_thickness_mm: float | None = None,
    implant_ml_mm: float | None = None,
) -> SceneDelta:
    """Apply a plan to a scene and report what moved."""
    delta = SceneDelta()

    for name, resection in plan.resections.items():
        node = scene.nodes.get(name)
        if node is None:
            continue
        matrix = gm.plane_matrix(resection.point, resection.normal)
        if not np.allclose(matrix, node.rest):
            node.rest = matrix
            delta.poses[name] = matrix

    for node in scene.nodes.values():
        if not node.tags.get("component"):
            continue
        pose = plan.components.get(f"{node.tags['group']}_component")
        if pose is None:
            continue

        source_ml = node.tags.get("source_ml")
        scale = (
            implant_ml_mm / source_ml
            if implant_ml_mm and source_ml
            else node.tags.get("scale", 1.0)
        )
        matrix = _seat(pose, scale)
        if not np.allclose(matrix, node.rest):
            node.rest = matrix
            delta.poses[node.name] = matrix
        node.tags["scale"] = float(scale)

    spacer = scene.nodes.get(INSERT_SPACER)
    if spacer is not None:
        if insert_thickness_mm is None:
            if spacer.visible:
                spacer.visible = False
                delta.visibility[INSERT_SPACER] = False
        else:
            if not spacer.visible:
                spacer.visible = True
                delta.visibility[INSERT_SPACER] = True

            ml_mm, ap_mm = spacer.tags.get("footprint_mm", (60.0, 40.0))
            rebuilt = gm.slab(ml_mm, ap_mm, insert_thickness_mm)
            delta.meshes[INSERT_SPACER] = scene.replace_mesh(INSERT_SPACER, rebuilt)

            matrix = _seat(plan.components["tibial_component"], 1.0)
            if not np.allclose(matrix, spacer.rest):
                spacer.rest = matrix
                delta.poses[INSERT_SPACER] = matrix

    delta.notes.extend(plan.warnings)
    return delta


def set_visibility(scene: Scene, predicate, visible: bool) -> SceneDelta:
    """Show or hide every node matching ``predicate``, reporting only real changes."""
    delta = SceneDelta()
    for node in scene.nodes.values():
        if predicate(node) and node.visible != visible:
            node.visible = visible
            delta.visibility[node.name] = visible
    return delta


def isolate_landmarks(scene: Scene, enabled: bool) -> SceneDelta:
    """Show the landmarks alone, or restore everything.

    What was hidden is remembered on the node rather than inferred on the way back, so
    a node that was already hidden for its own reason stays hidden when isolation ends.
    """
    delta = SceneDelta()
    for node in scene.nodes.values():
        if node.tags.get("landmark"):
            continue
        if enabled:
            if "visible_before_isolate" not in node.tags:
                node.tags["visible_before_isolate"] = node.visible
            if node.visible:
                node.visible = False
                delta.visibility[node.name] = False
        else:
            restored = node.tags.pop("visible_before_isolate", node.visible)
            if node.visible != restored:
                node.visible = restored
                delta.visibility[node.name] = restored
    return delta
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scene_update.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tka_planner/scene/update.py tests/test_scene_update.py
git commit -m "Re-pose the scene from a plan and report what moved"
```

---

### Task 10: Resection and commit

**Files:**
- Create: `tka_planner/scene/resect.py`
- Test: `tests/test_scene_resect.py`

**Interfaces:**
- Consumes: `Scene`, `SceneDelta`, `default_kernel`, `KernelRecord`, `gm.box`, `gm.plane_matrix`.
- Produces:
  - `CUTTER_SIZE_MM = 400.0`
  - `cutter_for(point_mm, normal_mm, *, keep: str) -> Mesh`
  - `CommitResult(scene, records: dict[str, KernelRecord], delta: SceneDelta, notes: tuple[str, ...])` with `.to_dict()`
  - `commit_resection(scene, plan, *, mode="block", kernel=None, bones=("Femur","Tibia"), build_shells=True) -> CommitResult`

`mode` is `"block"`, `"plane"` or `"none"`, matching the add-on's **Resect with**.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_resect.py`:

```python
"""Committing the resection: the one place a boolean runs."""

import numpy as np
import pytest

from tka_planner.geom import mesh as gm
from tka_planner.scene import resect as sr
from tka_planner.scene.model import FEMORAL, TIBIAL, Node, Scene


@pytest.fixture
def scene():
    built = Scene()
    built.add(Node(name="Femur", set_id=FEMORAL), gm.box(100.0))
    built.add(Node(name="Tibia", set_id=TIBIAL), gm.box(100.0))
    return built


def test_a_cutter_keeping_proximal_sits_below_the_plane():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="proximal")
    assert cutter.vertices[:, 2].max() == pytest.approx(0.0, abs=1e-9)


def test_a_cutter_keeping_distal_sits_above_the_plane():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="distal")
    assert cutter.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)


def test_a_cutter_is_large_enough_to_swallow_a_bone():
    cutter = sr.cutter_for((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), keep="proximal")
    assert cutter.extent[0] == pytest.approx(sr.CUTTER_SIZE_MM)


def test_cutting_a_box_in_half_halves_its_volume(scene):
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="plane", bones=("Femur",), build_shells=False
    )
    assert gm.volume(result.scene.mesh_for("Femur")) == pytest.approx(
        500_000.0, rel=1e-4
    )


def test_the_commit_records_the_kernel_for_every_bone_it_cut(scene):
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="plane", bones=("Femur",), build_shells=False
    )
    assert result.records["Femur"].backend == "manifold3d"
    assert result.records["Femur"].operation == "difference"


def test_the_commit_delta_replaces_the_bone_mesh(scene):
    before = scene.nodes["Femur"].mesh_id
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="plane", bones=("Femur",), build_shells=False
    )
    assert "Femur" in result.delta.meshes
    assert result.scene.nodes["Femur"].mesh_id != before


def test_only_the_named_bones_are_cut(scene):
    before = scene.nodes["Tibia"].mesh_id
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="plane", bones=("Femur",), build_shells=False
    )
    assert result.scene.nodes["Tibia"].mesh_id == before


def test_the_commit_serialises_its_geometry_provenance_for_the_plan_file(scene):
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="plane", bones=("Femur",), build_shells=False
    )
    payload = result.to_dict()

    assert payload["resections"]["Femur"]["backend"] == "manifold3d"
    assert payload["resections"]["Femur"]["fallback"] is None


def test_mode_none_cuts_nothing(scene):
    before = scene.nodes["Femur"].mesh_id
    result = sr.commit_resection(scene, _plan_at(z=0.0), mode="none")

    assert result.scene.nodes["Femur"].mesh_id == before
    assert result.records == {}


def test_block_mode_subtracts_the_block_and_builds_the_shell(scene):
    scene.add(
        Node(name="femoral_cutting_block", set_id=FEMORAL,
             tags={"component": True, "group": "femoral"}),
        gm.box(40.0, gm.translation((0.0, 0.0, -10.0))),
    )
    scene.add(
        Node(name="femoral_cutting_block_shell", set_id=FEMORAL,
             tags={"component": True, "group": "femoral", "shell": True}),
        gm.box(40.0, gm.translation((0.0, 0.0, -10.0))),
    )
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="block", bones=("Femur",), build_shells=True
    )

    assert gm.volume(result.scene.mesh_for("Femur")) < 1_000_000.0
    assert "Femur.Shell" in result.scene.nodes
    assert gm.volume(result.scene.mesh_for("Femur.Shell")) == pytest.approx(
        64_000.0, rel=1e-3
    )


def test_the_shell_is_taken_before_the_bone_is_resected(scene):
    """A shell cut from an already-resected femur would miss the surface it mates with."""
    scene.add(
        Node(name="femoral_cutting_block", set_id=FEMORAL,
             tags={"component": True, "group": "femoral"}),
        gm.box(40.0, gm.translation((0.0, 0.0, -10.0))),
    )
    scene.add(
        Node(name="femoral_cutting_block_shell", set_id=FEMORAL,
             tags={"component": True, "group": "femoral", "shell": True}),
        gm.box(40.0, gm.translation((0.0, 0.0, -10.0))),
    )
    result = sr.commit_resection(
        scene, _plan_at(z=0.0), mode="block", bones=("Femur",), build_shells=True
    )
    shell = result.scene.mesh_for("Femur.Shell")

    # The shell reaches below z = 0, which only an unresected bone could supply.
    assert shell.vertices[:, 2].min() < -1.0


def _plan_at(*, z: float):
    """A stand-in plan with one femoral and one tibial resection plane."""
    from tka_planner.core.planning import ResectionPlane, SurgicalPlan
    from tka_planner.core.provenance import Quality

    def plane(name, normal):
        return ResectionPlane(
            name=name, point=np.array([0.0, 0.0, z]),
            normal=np.asarray(normal, dtype=float),
            medial_depth_mm=9.0, lateral_depth_mm=9.0, reference="test",
        )

    return SurgicalPlan(
        philosophy="mechanical", distal_femoral_valgus_cut_deg=5.0,
        valgus_source="test", valgus_quality=Quality.MEASURED,
        tibial_slope_deg=3.0,
        resections={
            "femoral_distal": plane("femoral_distal", (0.0, 0.0, 1.0)),
            "tibial_proximal": plane("tibial_proximal", (0.0, 0.0, 1.0)),
        },
        components={
            "femoral_component": np.eye(4), "tibial_component": np.eye(4),
        },
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scene_resect.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.scene.resect'`

- [ ] **Step 3: Check `Quality`'s member name before running**

Run: `python -c "from tka_planner.core.provenance import Quality; print(list(Quality))"`
Use whichever member the output shows in `_plan_at`. If the enum is named differently,
adjust the import in the test to match.

- [ ] **Step 4: Implement resection**

Create `tka_planner/scene/resect.py`:

```python
"""Committing a plan's resection into geometry.

This is the only module in the engine that runs a boolean, and it runs one only when
asked. That is the whole point of the Plan / Commit / Reduce split: an exact boolean
against a real segmentation costs seconds, which cannot live inside a drag, so it is
moved out of the drag entirely rather than hidden behind a debounce.

Two modes, and they are alternatives rather than a stack. **Block** subtracts the
cutting block itself, which is the instrument that would realise the cut in theatre,
and intersects a copy of the bone with the block's shell to give the patient-specific
mating surface. **Plane** removes everything beyond the planned plane instead, which is
what the plan actually specifies. Stacking them would let the plane swallow the surfaces
the block is shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tka_planner.core.meshio import Mesh
from tka_planner.geom import mesh as gm
from tka_planner.geom.kernel import KernelRecord, MeshKernel, default_kernel

from .model import FEMORAL, TIBIAL, Node, Scene, SceneDelta

__all__ = ["CUTTER_SIZE_MM", "cutter_for", "CommitResult", "commit_resection"]

# Large enough to swallow the discarded side of any bone in the cohort.
CUTTER_SIZE_MM = 400.0

# Which side of each plane is kept. The distal femur is cut from below, so the femur
# keeps the bone above its plane; the proximal tibia is cut from above and keeps the
# bone below. Applying one convention to both silently inverts a resection.
KEEP = {"femoral_distal": "proximal", "tibial_proximal": "distal"}
BONE_FOR = {"femoral_distal": "Femur", "tibial_proximal": "Tibia"}


def cutter_for(point_mm, normal_mm, *, keep: str) -> Mesh:
    """The discard box for a resection plane.

    Its near face lies on the plane and its bulk sits on the side being thrown away, so
    a boolean difference leaves exactly the retained bone.
    """
    if keep not in ("proximal", "distal"):
        raise ValueError("keep must be 'proximal' or 'distal'.")

    normal = np.asarray(normal_mm, dtype=float)
    normal = normal / np.linalg.norm(normal)
    discard = -normal if keep == "proximal" else normal
    centre = np.asarray(point_mm, dtype=float) + discard * (CUTTER_SIZE_MM / 2.0)

    return gm.box(CUTTER_SIZE_MM, gm.translation(centre))


@dataclass
class CommitResult:
    scene: Scene
    records: dict = field(default_factory=dict)
    delta: SceneDelta = field(default_factory=SceneDelta)
    notes: tuple = ()

    def to_dict(self) -> dict:
        """The geometry provenance, shaped to sit in ``plan.json``.

        The plan already records how every number was obtained. This says how the
        geometry was obtained: which solver cut each bone, what it had to repair first,
        and whether it fell back.
        """
        return {
            "resections": {
                name: record.to_dict() for name, record in self.records.items()
            },
            "notes": list(self.notes),
        }


def commit_resection(
    scene: Scene,
    plan,
    *,
    mode: str = "block",
    kernel: MeshKernel | None = None,
    bones: tuple = ("Femur", "Tibia"),
    build_shells: bool = True,
) -> CommitResult:
    """Cut the named bones and return the scene with committed geometry.

    ``bones`` is what makes a re-commit cheap: adjusting the tibial slope re-cuts the
    tibia and leaves an untouched femur alone. The caller decides which bones a change
    reached; see :meth:`tka_planner.session.PlanningSession.bones_affected_by`.
    """
    kernel = kernel or default_kernel()
    result = CommitResult(scene=scene)
    notes: list[str] = []

    if mode == "none":
        return result

    if mode not in ("block", "plane"):
        raise ValueError(f"Unknown resection mode {mode!r}; expected block or plane.")

    for plane_name, resection in plan.resections.items():
        bone_name = BONE_FOR.get(plane_name)
        if bone_name is None or bone_name not in bones:
            continue
        bone = scene.mesh_for(bone_name)
        if bone is None:
            continue

        set_id = FEMORAL if bone_name == "Femur" else TIBIAL
        group = "femoral" if bone_name == "Femur" else "tibial"

        if mode == "block":
            block = _part(scene, f"{group}_cutting_block")
            if block is None:
                notes.append(f"{bone_name}: no cutting block, left whole")
                continue

            # The shell copy is taken *before* the bone is resected, and that ordering
            # is the whole trick: a shell cut from an already-resected femur would be
            # missing the condylar surface it is supposed to mate with.
            if build_shells:
                shell_tool = _part(scene, f"{group}_cutting_block_shell")
                if shell_tool is not None:
                    shell = kernel.intersect(bone, shell_tool)
                    name = f"{bone_name}.Shell"
                    scene.add(
                        Node(name=name, set_id=set_id, colour=(0.85, 0.18, 0.18),
                             tags={"shell": True}),
                        shell.mesh,
                    )
                    result.records[name] = shell.record
                    result.delta.meshes[name] = scene.nodes[name].mesh_id
                else:
                    notes.append(f"{bone_name}: no block shell, shell not built")

            cut = kernel.difference(bone, block)
        else:
            tool = cutter_for(
                resection.point, resection.normal,
                keep=KEEP.get(plane_name, "proximal"),
            )
            cut = kernel.difference(bone, tool)

        result.delta.meshes[bone_name] = scene.replace_mesh(bone_name, cut.mesh)
        result.records[bone_name] = cut.record

    result.notes = tuple(notes)
    result.delta.notes.extend(notes)
    return result


def _part(scene: Scene, name: str) -> Mesh | None:
    """A component's mesh in world space, so it can be used as a boolean tool.

    Components are stored at the origin with their seating in the node's rest
    transform, but a kernel takes plain geometry, so the transform is applied here.
    """
    node = scene.nodes.get(name)
    if node is None or node.mesh_id is None:
        return None
    return gm.transformed(scene.meshes[node.mesh_id], scene.world(name))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_scene_resect.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tka_planner/scene/resect.py tests/test_scene_resect.py
git commit -m "Commit a plan's resection in one place, with a kernel record"
```

---

### Task 11: The planning session

**Files:**
- Create: `tka_planner/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: everything from `core`, `scene`, `geom`.
- Produces:
  - `ADJUSTMENT_FIELDS: tuple[str, ...]` (the thirteen from the add-on)
  - `Controls` dataclass with those thirteen plus `philosophy`, `size_override`, `tibial_resection_mm`, `insert_thickness_mm`, `use_insert`, `resection_mode`, `build_bone_shells`, `show_planes`, `show_axes`, `show_landmarks`, `isolate_landmarks`
  - `TrialControls(flexion_deg, varus_valgus_deg, drawer_ap_mm)`
  - `PlanningSession` with:
    - `open(folder, *, side, library=None) -> PlanningSession` (classmethod)
    - `.controls: Controls`, `.trial: TrialControls`, `.scene: Scene`, `.plan`, `.sizing`
    - `.build() -> SceneDelta`
    - `.replan(**changes) -> SceneDelta`
    - `.reset_adjustments() -> SceneDelta`
    - `.commit(kernel=None, *, bones=None) -> CommitResult`
    - `.set_trial(**changes) -> SceneDelta`
    - `.reset_trial() -> SceneDelta`
    - `.bones_affected_by(changed: tuple[str, ...]) -> tuple[str, ...]`
    - `.report_lines() -> list[str]`
    - `.stale: bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session.py`:

```python
"""The planning session, driven with no Blender and no browser."""

import numpy as np
import pytest

from tests.synthetic import synthetic_knee
from tka_planner.geom import mesh as gm
from tka_planner.session import ADJUSTMENT_FIELDS, PlanningSession


@pytest.fixture
def folder(tmp_path):
    """A patient folder in the naming convention the add-on expects."""
    gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "FD1Left.stl")
    gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "TD1Left.stl")
    return tmp_path


@pytest.fixture
def session(folder, monkeypatch):
    """A session on synthetic anatomy.

    Measurement is stubbed so the test exercises planning, posing and committing rather
    than landmark estimation, which has its own suite. Everything ``_plan`` and
    ``_sizing`` read must be set here, so this list is also the definition of what
    ``_measure`` is responsible for producing.
    """
    from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
    from tka_planner.core.meshio import read_stl

    knee = synthetic_knee()

    def fake_measure(self):
        self.case_id = "CASE_TEST"
        self._femur = read_stl(self.femur_path)
        self._tibia = read_stl(self.tibia_path)
        self._landmarks = knee.landmarks
        self._femoral_frame = knee.femoral_frame
        self._tibial_frame = knee.tibial_frame
        self._metrics = {}
        self._femoral_measure = measure_femoral_ml(self._femur, knee.femoral_frame)
        self._tibial_measure = measure_tibial_plateau(self._tibia, knee.tibial_frame)
        self._native_slope_deg = 5.0

    monkeypatch.setattr(PlanningSession, "_measure", fake_measure)
    opened = PlanningSession.open(folder, side="left")
    opened.build()
    return opened


def test_all_thirteen_adjustments_are_exposed():
    assert len(ADJUSTMENT_FIELDS) == 13
    assert "coronal_correction_deg" in ADJUSTMENT_FIELDS
    assert "tibial_shift_ml_mm" in ADJUSTMENT_FIELDS


def test_a_fresh_session_reports_the_plan_as_computed(session):
    assert session.plan.adjustments.is_identity


def test_every_adjustment_can_be_set_and_produces_a_delta(session):
    for name in ADJUSTMENT_FIELDS:
        delta = session.replan(**{name: 1.5})
        assert delta.poses or delta.meshes or delta.visibility, name
        session.reset_adjustments()


def test_replanning_marks_the_plan_adjusted(session):
    session.replan(tibial_resection_delta_mm=2.0)
    assert not session.plan.adjustments.is_identity


def test_reset_adjustments_returns_to_the_computed_plan(session):
    before = session.scene.world("tibial_proximal").copy()
    session.replan(tibial_resection_delta_mm=2.0)
    session.reset_adjustments()

    assert session.scene.world("tibial_proximal") == pytest.approx(before)


def test_a_femoral_change_affects_only_the_femur(session):
    assert session.bones_affected_by(("femoral_resection_delta_mm",)) == ("Femur",)


def test_a_tibial_change_affects_only_the_tibia(session):
    assert session.bones_affected_by(("tibial_slope_delta_deg",)) == ("Tibia",)


def test_a_coronal_correction_affects_both_bones(session):
    assert set(session.bones_affected_by(("coronal_correction_deg",))) == {
        "Femur", "Tibia"
    }


def test_an_insert_change_affects_no_bone(session):
    assert session.bones_affected_by(("insert_thickness_mm",)) == ()


def test_the_trial_controls_move_the_tibial_set_only(session):
    session.set_trial(flexion_deg=30.0)
    from tka_planner.scene.model import FEMORAL, TIBIAL

    assert not np.allclose(session.scene.set_poses[TIBIAL], np.eye(4))
    assert session.scene.set_poses[FEMORAL] == pytest.approx(np.eye(4))


def test_resetting_the_trial_returns_the_tibial_set_to_identity(session):
    from tka_planner.scene.model import TIBIAL

    session.set_trial(flexion_deg=45.0, varus_valgus_deg=5.0, drawer_ap_mm=3.0)
    session.reset_trial()

    assert session.scene.set_poses[TIBIAL] == pytest.approx(np.eye(4))


def test_the_trial_does_not_change_the_plan(session):
    session.set_trial(flexion_deg=45.0)
    assert session.plan.adjustments.is_identity


def test_a_committed_session_becomes_stale_when_the_plan_changes(session):
    session.commit()
    assert session.stale is False

    session.replan(tibial_resection_delta_mm=1.0)
    assert session.stale is True


def test_committing_again_clears_the_stale_flag(session):
    session.commit()
    session.replan(tibial_resection_delta_mm=1.0)
    session.commit()

    assert session.stale is False


def test_a_commit_records_the_kernel_it_used(session):
    result = session.commit()
    assert all(
        record.backend == "manifold3d" for record in result.records.values()
    )


def test_report_lines_mention_the_correction_and_the_resections(session):
    text = "\n".join(session.report_lines())
    assert "valgus" in text.lower()
    assert "mm" in text


def test_the_session_imports_no_bpy(session):
    import sys
    assert "bpy" not in sys.modules
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_session.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.session'`

- [ ] **Step 3: Implement the session**

Create `tka_planner/session.py`. Port the logic from `archive/blender-addon/addon/__init__.py`:
`_find_bone_files` (line 294), `_adjustments_from` (325), `_sizing_for` (337),
`_plan_from` (360), `_cut_affecting_bones` (130), `_report_lines` (477). Strip every
Blender property access and take the values from `Controls` instead.

```python
"""One planning session, with no user interface attached.

This is the part that did not exist before. The add-on's property callbacks held it:
the cached measurement, the current control values, the re-plan on every change, the
decision about which bones a change actually reaches. All of that was tangled with
Blender's UI lifecycle, which meant it could only be driven by a person holding a
slider.

Pulled out here, the same session is driven by a browser, by a test, or later by a
tracker feed, and none of them need to know about each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from tka_planner.core.frames import build_femoral_frame, build_tibial_frame
from tka_planner.core.landmarks_auto import estimate_landmarks
from tka_planner.core.measure import measure_femoral_ml, measure_tibial_plateau
from tka_planner.core.meshio import read_stl
from tka_planner.core.metrics import compute_all
from tka_planner.core.planning import KINEMATIC, MECHANICAL, Adjustments, plan_alignment
from tka_planner.core.sizing import load_size_chart, solve_parametric_size
from tka_planner.scene import build as scene_build
from tka_planner.scene import motion
from tka_planner.scene import resect as scene_resect
from tka_planner.scene import update as scene_update
from tka_planner.scene.model import TIBIAL, SceneDelta

__all__ = ["ADJUSTMENT_FIELDS", "Controls", "TrialControls", "PlanningSession"]

ADJUSTMENT_FIELDS = (
    "coronal_correction_deg",
    "femoral_resection_delta_mm",
    "femoral_flexion_delta_deg",
    "femoral_varus_delta_deg",
    "femoral_rotation_delta_deg",
    "femoral_shift_ap_mm",
    "femoral_shift_ml_mm",
    "tibial_resection_delta_mm",
    "tibial_slope_delta_deg",
    "tibial_varus_delta_deg",
    "tibial_rotation_delta_deg",
    "tibial_shift_ap_mm",
    "tibial_shift_ml_mm",
)

# Which bones a control's change actually reaches, so a commit re-cuts only what moved.
# Adjusting the tibial slope must not re-solve an untouched femur.
FEMORAL_ONLY = frozenset(name for name in ADJUSTMENT_FIELDS if name.startswith("femoral_"))
TIBIAL_ONLY = frozenset(name for name in ADJUSTMENT_FIELDS if name.startswith("tibial_"))
BOTH_BONES = frozenset({"coronal_correction_deg", "philosophy", "size_override",
                        "tibial_resection_mm", "resection_mode", "build_bone_shells"})
# Controls that move nothing a boolean depends on.
NO_BONES = frozenset({"insert_thickness_mm", "use_insert", "show_planes", "show_axes",
                      "show_landmarks", "isolate_landmarks"})


@dataclass
class Controls:
    """Every plan control, matching the add-on's panel one for one."""

    coronal_correction_deg: float = 0.0
    femoral_resection_delta_mm: float = 0.0
    femoral_flexion_delta_deg: float = 0.0
    femoral_varus_delta_deg: float = 0.0
    femoral_rotation_delta_deg: float = 0.0
    femoral_shift_ap_mm: float = 0.0
    femoral_shift_ml_mm: float = 0.0
    tibial_resection_delta_mm: float = 0.0
    tibial_slope_delta_deg: float = 0.0
    tibial_varus_delta_deg: float = 0.0
    tibial_rotation_delta_deg: float = 0.0
    tibial_shift_ap_mm: float = 0.0
    tibial_shift_ml_mm: float = 0.0

    philosophy: str = "mechanical"
    size_override: str = ""
    tibial_resection_mm: float = 8.0
    insert_thickness_mm: float = 9.0
    use_insert: bool = True
    resection_mode: str = "block"
    build_bone_shells: bool = True
    show_planes: bool = True
    show_axes: bool = True
    show_landmarks: bool = False
    isolate_landmarks: bool = False


@dataclass
class TrialControls:
    """The three Reduce-mode controls. They never touch the plan."""

    flexion_deg: float = 0.0
    varus_valgus_deg: float = 0.0
    drawer_ap_mm: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.flexion_deg == 0.0
            and self.varus_valgus_deg == 0.0
            and self.drawer_ap_mm == 0.0
        )


@dataclass
class PlanningSession:
    """A patient, a plan, a scene, and the controls that move both."""

    femur_path: Path
    tibia_path: Path
    side: str
    library: Path | None = None
    controls: Controls = field(default_factory=Controls)
    trial: TrialControls = field(default_factory=TrialControls)

    scene = None
    plan = None
    sizing = None
    chart = None
    stale: bool = False

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, folder, *, side: str, library=None) -> "PlanningSession":
        """Find the two bone meshes in a patient folder and measure them.

        Accepts the naming the add-on accepted: ``FD1Left.stl`` and ``TD1Left.stl``, or
        a file named for the bone, so a folder exported straight out of 3D Slicer works
        as-is.
        """
        femur_path, tibia_path = find_bone_files(Path(folder), side)
        session = cls(
            femur_path=femur_path, tibia_path=tibia_path, side=side,
            library=Path(library) if library else None,
        )
        session._measure()
        return session

    def _measure(self) -> None:
        """Read the meshes, estimate the landmarks, build the frames and metrics.

        Cached for the life of the session: it is the expensive part, and no control
        changes it. Re-planning reuses this and costs a few dozen numpy operations.

        This is the call sequence `TKA_OT_plan._run` used, lifted out of the operator
        unchanged.
        """
        self.case_id = self.femur_path.parent.name
        self._femur = read_stl(self.femur_path)
        self._tibia = read_stl(self.tibia_path)
        self._landmarks = estimate_landmarks(
            self._femur, self._tibia, self.side, case_id=self.case_id
        )
        self._femoral_frame = build_femoral_frame(self._landmarks)
        self._tibial_frame = build_tibial_frame(self._landmarks)
        self._metrics = compute_all(
            self._landmarks, self._femoral_frame, self._tibial_frame
        )
        self._femoral_measure = measure_femoral_ml(self._femur, self._femoral_frame)
        self._tibial_measure = measure_tibial_plateau(self._tibia, self._tibial_frame)
        self._native_slope_deg = self._metrics[
            "posterior_slope_medial_deg"
        ].value

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def build(self) -> SceneDelta:
        """Compute the plan and build the scene. The equivalent of pressing Plan."""
        self.plan, self.sizing = self._plan()
        components = self._components()
        self.scene = scene_build.build_scene(
            femur_path=self.femur_path,
            tibia_path=self.tibia_path,
            plan=self.plan,
            femoral_frame=self._femoral_frame,
            tibial_frame=self._tibial_frame,
            landmarks=self._landmarks,
            components=components,
            show_planes=self.controls.show_planes,
            show_axes=self.controls.show_axes,
            show_landmarks=self.controls.show_landmarks,
            insert_thickness_mm=(
                self.controls.insert_thickness_mm if self.controls.use_insert else None
            ),
            insert_footprint_mm=self._insert_footprint(),
        )
        self.stale = True
        return SceneDelta(scalars=self._scalars(), notes=list(self.scene.notes))

    def replan(self, **changes) -> SceneDelta:
        """Apply control changes, re-plan, and re-pose the scene.

        No boolean runs here. That is the Plan-mode guarantee, and it is what lets this
        be called on every change of a value rather than on a button press.
        """
        unknown = set(changes) - {f.name for f in fields(Controls)}
        if unknown:
            raise ValueError(f"Unknown control(s): {sorted(unknown)}")

        self.controls = replace(self.controls, **changes)
        self.plan, self.sizing = self._plan()

        delta = scene_update.update_scene(
            self.scene,
            self.plan,
            insert_thickness_mm=(
                self.controls.insert_thickness_mm if self.controls.use_insert else None
            ),
            implant_ml_mm=self.sizing.implant_ml_mm,
        )
        if "isolate_landmarks" in changes:
            delta = delta.merge(
                scene_update.isolate_landmarks(
                    self.scene, self.controls.isolate_landmarks
                )
            )
        if "show_landmarks" in changes:
            delta = delta.merge(
                scene_update.set_visibility(
                    self.scene,
                    lambda node: bool(node.tags.get("landmark")),
                    self.controls.show_landmarks,
                )
            )

        if self.bones_affected_by(tuple(changes)):
            self.stale = True

        delta.scalars.update(self._scalars())
        return delta

    def reset_adjustments(self) -> SceneDelta:
        """Return every plan control to the computed plan."""
        return self.replan(**{name: 0.0 for name in ADJUSTMENT_FIELDS})

    def bones_affected_by(self, changed: tuple) -> tuple:
        """Which bones a set of control changes actually reaches.

        In cutting-block mode a component's own pose moves the block that cuts it, so a
        shift or a rotation is not free. In plane mode those controls move nothing a
        cutter depends on.
        """
        bones: set = set()
        block_mode = self.controls.resection_mode == "block"
        moves_component = {"_shift_", "_rotation_"}

        for name in changed:
            if name in NO_BONES:
                continue
            touches_component = any(token in name for token in moves_component)
            if touches_component and not block_mode:
                continue
            if name in FEMORAL_ONLY:
                bones.add("Femur")
            elif name in TIBIAL_ONLY:
                bones.add("Tibia")
            elif name in BOTH_BONES or name in ADJUSTMENT_FIELDS:
                bones.update({"Femur", "Tibia"})
        return tuple(sorted(bones))

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, kernel=None, *, bones=None):
        """Cut the bones. The only place in the session a boolean runs."""
        result = scene_resect.commit_resection(
            self.scene,
            self.plan,
            mode=self.controls.resection_mode,
            kernel=kernel,
            bones=bones or ("Femur", "Tibia"),
            build_shells=self.controls.build_bone_shells,
        )
        self.stale = False
        return result

    # ------------------------------------------------------------------
    # Reduce
    # ------------------------------------------------------------------

    def set_trial(self, **changes) -> SceneDelta:
        """Pose the implanted construct. Never re-plans, never cuts."""
        unknown = set(changes) - {f.name for f in fields(TrialControls)}
        if unknown:
            raise ValueError(f"Unknown trial control(s): {sorted(unknown)}")

        self.trial = replace(self.trial, **changes)
        pose = motion.trial_pose(
            self.scene.pivot,
            flexion_deg=self.trial.flexion_deg,
            varus_valgus_deg=self.trial.varus_valgus_deg,
            drawer_ap_mm=self.trial.drawer_ap_mm,
        )
        self.scene.set_pose(TIBIAL, pose)
        return SceneDelta(poses={"TibialSet": pose})

    def reset_trial(self) -> SceneDelta:
        return self.set_trial(flexion_deg=0.0, varus_valgus_deg=0.0, drawer_ap_mm=0.0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_lines(self) -> list[str]:
        """The panel's status block: correction, resections, sizing, warnings."""
        plan, sizing = self.plan, self.sizing
        femoral = plan.resections.get("femoral_distal")
        tibial = plan.resections.get("tibial_proximal")

        lines = [
            f"Philosophy: {plan.philosophy}",
            f"Distal femoral valgus cut: "
            f"{plan.distal_femoral_valgus_cut_deg:.1f} deg "
            f"({plan.valgus_source}, {plan.valgus_quality.value})",
            f"Tibial slope: {plan.tibial_slope_deg:.1f} deg",
        ]
        if femoral is not None:
            lines.append(
                f"Femoral resection: {femoral.medial_depth_mm:.1f} mm medial, "
                f"{femoral.lateral_depth_mm:.1f} mm lateral"
            )
        if tibial is not None:
            lines.append(
                f"Tibial resection: {tibial.medial_depth_mm:.1f} mm medial, "
                f"{tibial.lateral_depth_mm:.1f} mm lateral"
            )
        if sizing is not None:
            lines.append(
                f"Size: {sizing.nearest_discrete_size} "
                f"({sizing.implant_ml_mm:.1f} mm ML)"
            )
        lines.extend(f"Warning: {warning}" for warning in plan.warnings)
        return lines

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _plan(self):
        sizing = self._sizing()
        plan = plan_alignment(
            self._landmarks, self._femoral_frame, self._tibial_frame,
            target=(
                MECHANICAL if self.controls.philosophy == "mechanical" else KINEMATIC
            ),
            femoral_thickness_mm=sizing.femoral_thickness_mm,
            tibial_resection_mm=self.controls.tibial_resection_mm,
            native_slope_deg=self._native_slope_deg,
            adjustments=self._adjustments(),
            femur_mesh=self._femur, tibia_mesh=self._tibia,
        )
        return plan, sizing

    def _adjustments(self) -> Adjustments:
        return Adjustments(
            **{name: getattr(self.controls, name) for name in ADJUSTMENT_FIELDS},
            insert_thickness_mm=(
                self.controls.insert_thickness_mm if self.controls.use_insert else None
            ),
        )

    def _sizing(self):
        """The implant size, solved from the anatomy or set by hand.

        An empty or unknown ``size_override`` means solve it from the measurement,
        which is also the right answer if a saved case names a size the chart no longer
        publishes.
        """
        chart = self._size_chart()
        label = self.controls.size_override
        override = chart.label_parameter(label) if label in chart.labels else None
        return solve_parametric_size(
            chart,
            measured_ml_mm=self._femoral_measure.ml_mm,
            measured_ap_mm=self._femoral_measure.ap_mm,
            parameter_override=override,
        )

    def _size_chart(self):
        if self.chart is None:
            self.chart = load_size_chart(find_size_chart())
        return self.chart

    def _components(self) -> dict:
        if self.library is None:
            return {}
        return scene_build.resolve_component_meshes(
            self.library, chart=self._size_chart(), sizing=self.sizing, side=self.side
        )

    def _insert_footprint(self):
        """The tray's own footprint, from the sizing decision rather than a mesh.

        Taken from the resection cross-section the sizing solved, which is where the
        add-on took it from too. A bounding box over the tray mesh would be wrong for
        the same reason bounding-box seating was wrong for the components.
        """
        return (
            self.sizing.diagnostics.get("tibia_ML_mm", 70.0),
            self.sizing.diagnostics.get("tibia_AP_mm", 48.0),
        )

    def _scalars(self) -> dict:
        return {
            "distal_femoral_valgus_cut_deg": float(
                self.plan.distal_femoral_valgus_cut_deg
            ),
            "tibial_slope_deg": float(self.plan.tibial_slope_deg),
            "adjusted": not self.plan.adjustments.is_identity,
            "stale": self.stale,
        }


def find_size_chart() -> Path:
    """Locate SizeChart.csv, whether running from the repository or from an install.

    Ported from the add-on. Running from a checkout the repository copy wins, so edits
    take effect without rebuilding; installed, the bundled copy is used.
    """
    package = Path(__file__).resolve().parent
    candidates = [
        package.parent / "data" / "SizeChart.csv",
        package / "data" / "SizeChart.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "SizeChart.csv not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def find_bone_files(folder: Path, side: str) -> tuple[Path, Path]:
    """The femur and tibia in a patient folder.

    Ported from the add-on: the convention is ``FD1Left.stl`` and ``TD1Left.stl``, and a
    file named for the bone is accepted too, so a folder exported straight out of
    3D Slicer can be used as-is.
    """
    left = side.lower().startswith("l")
    token = "left" if left else "right"

    def find(prefix: str, word: str) -> Path:
        candidates = [
            path for path in folder.glob("*.stl")
            if token in path.stem.lower()
            and (path.stem.lower().startswith(prefix) or word in path.stem.lower())
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No {word} STL for the {token} side in {folder}. Expected something "
                f"like {prefix.upper()}D1{token.title()}.stl or a name containing "
                f"{word!r}."
            )
        return sorted(candidates)[0]

    return find("f", "femur"), find("t", "tibia")
```

- [ ] **Step 4: Confirm the imports resolve**

Every name above was checked against the real modules when this plan was written. Prove
it still holds before running the tests:

```bash
python -c "from tka_planner.session import PlanningSession, ADJUSTMENT_FIELDS; print(len(ADJUSTMENT_FIELDS))"
```

Expected: `13`

If anything fails to import, `archive/blender-addon/addon/__init__.py` lines 827 to 941
are the `TKA_OT_plan` operator, which is the working reference for how these functions
fit together.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_session.py -q`
Expected: all pass.

- [ ] **Step 6: Run the whole suite**

Run: `pytest -q`
Expected: 462 original tests plus the new ones, no failures.

- [ ] **Step 7: Commit**

```bash
git add tka_planner/session.py tests/test_session.py
git commit -m "Extract the planning session from the add-on's property callbacks"
```

---

### Task 12: Point the add-on at the engine

**Files:**
- Modify: `tka_planner/blender/build.py`
- Modify: `tka_planner/addon/__init__.py`
- Create: `tka_planner/blender/render.py`
- Test: `tests/test_addon_parity.py`

**Interfaces:**
- Consumes: `Scene`, `SceneDelta`, `PlanningSession`.
- Produces:
  - `render_scene(scene: Scene) -> dict[str, object]` mapping node name to Blender object
  - `apply_delta(delta: SceneDelta) -> None`

The add-on keeps its panel, its operators and its property definitions. It stops
computing anything and stops cutting anything. Blender becomes a viewer.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_addon_parity.py`:

```python
"""The add-on, driven headlessly, must still produce the scene it always did.

Marked ``blender`` and excluded from the default run. Run it deliberately:

    blender --background --python-expr "import pytest, sys; sys.exit(pytest.main(['-m','blender','tests/test_addon_parity.py','-v']))"
"""

import numpy as np
import pytest

pytestmark = pytest.mark.blender


@pytest.fixture
def rendered(tmp_path):
    from tka_planner.blender import render
    from tka_planner.geom import mesh as gm
    from tka_planner.session import PlanningSession

    gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "FD1Left.stl")
    gm.write_stl(gm.box((70.0, 60.0, 120.0)), tmp_path / "TD1Left.stl")

    session = PlanningSession.open(tmp_path, side="left")
    session.build()
    return session, render.render_scene(session.scene)


def test_every_scene_node_becomes_a_blender_object(rendered):
    session, objects = rendered
    assert set(objects) == set(session.scene.nodes)


def test_an_object_sits_where_the_scene_says_it_does(rendered):
    session, objects = rendered
    world = session.scene.world("femoral_distal")
    matrix = np.array(objects["femoral_distal"].matrix_world)

    # Blender works in metres, the engine in millimetres.
    assert matrix[:3, 3] * 1000.0 == pytest.approx(world[:3, 3], abs=1e-6)


def test_a_replan_moves_the_object_through_a_delta(rendered):
    from tka_planner.blender import render

    session, objects = rendered
    before = np.array(objects["tibial_proximal"].matrix_world)
    render.apply_delta(session.replan(tibial_resection_delta_mm=3.0))

    assert not np.allclose(np.array(objects["tibial_proximal"].matrix_world), before)


def test_a_commit_replaces_the_bone_mesh_in_blender(rendered):
    from tka_planner.blender import render

    session, objects = rendered
    before = len(objects["Femur"].data.vertices)
    result = session.commit()
    render.apply_delta(result.delta)

    assert len(objects["Femur"].data.vertices) != before


def test_a_trial_pose_moves_only_the_tibial_objects(rendered):
    from tka_planner.blender import render

    session, objects = rendered
    femur_before = np.array(objects["Femur"].matrix_world)
    session.set_trial(flexion_deg=40.0)
    # A set pose moves a whole set, so it is applied through `apply_scene` rather than
    # `apply_delta`, which handles per-node poses only.
    render.apply_scene(session.scene)

    assert np.array(objects["Femur"].matrix_world) == pytest.approx(femur_before)
    assert not np.allclose(
        np.array(objects["Tibia"].matrix_world), np.eye(4)
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest -m blender tests/test_addon_parity.py -q`
Expected: `ModuleNotFoundError: No module named 'tka_planner.blender.render'`

- [ ] **Step 3: Write the renderer**

Create `tka_planner/blender/render.py`:

```python
"""Draw an engine scene in Blender, and nothing else.

This replaces everything `build.py` used to decide. The engine computes the plan,
places every node and cuts the bones; this module turns that into Blender objects and
then moves them when a delta arrives. It creates no geometry of its own, runs no
boolean, and holds no plan state.

Unit conversion happens here and only here. The engine works in millimetres throughout;
Blender scenes are built in metres because its viewport clipping behaves badly at a
scale of hundreds.
"""

from __future__ import annotations

import numpy as np

from .io import MM_TO_BU, clear_scene, ensure_collection, move_to_collection, set_material

__all__ = ["render_scene", "apply_delta", "objects"]

_OBJECTS: dict = {}

_COLLECTION_FOR = {
    "bone": "Bones", "plane": "Planning", "axis": "Planning",
    "landmark": "Landmarks", "component": "Implants", "shell": "Implants",
    "insert": "Implants",
}


def render_scene(scene, *, clear: bool = True) -> dict:
    """Create one Blender object per scene node and return them by name."""
    import bpy

    if clear:
        clear_scene()
    _OBJECTS.clear()

    for name, node in scene.nodes.items():
        mesh = scene.meshes.get(node.mesh_id) if node.mesh_id else None
        if mesh is None:
            continue

        data = bpy.data.meshes.new(name)
        data.from_pydata(
            [tuple(float(c) * MM_TO_BU for c in v) for v in mesh.vertices],
            [],
            [tuple(int(i) for i in f) for f in mesh.faces],
        )
        data.update()

        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        move_to_collection(obj, ensure_collection(_collection_for(node)))
        set_material(obj, f"TKA_{name}", node.colour, node.alpha)

        obj.matrix_world = _to_blender(scene.world(name))
        obj.hide_viewport = not node.visible
        obj.hide_render = not node.visible
        _OBJECTS[name] = obj

    return dict(_OBJECTS)


def apply_delta(delta) -> None:
    """Move, re-mesh and re-hide objects from a scene delta.

    A pose change is an assignment, which is why a drag stays responsive: no modifier
    re-evaluates, because there are no modifiers left in this scene.
    """
    import bpy

    for name, matrix in delta.poses.items():
        if name in ("FemoralSet", "TibialSet"):
            continue
        obj = _OBJECTS.get(name)
        if obj is not None:
            obj.matrix_world = _to_blender(np.asarray(matrix))

    for name, visible in delta.visibility.items():
        obj = _OBJECTS.get(name)
        if obj is not None:
            obj.hide_viewport = not visible
            obj.hide_render = not visible

    bpy.context.view_layer.update()


def apply_scene(scene) -> None:
    """Re-sync every object's transform and mesh from the scene.

    Used after a commit or a set-pose change, where a per-node delta would have to
    enumerate a whole set anyway.
    """
    import bpy

    for name, obj in _OBJECTS.items():
        node = scene.nodes.get(name)
        if node is None:
            continue
        obj.matrix_world = _to_blender(scene.world(name))

        mesh = scene.meshes.get(node.mesh_id) if node.mesh_id else None
        if mesh is not None and len(obj.data.vertices) != mesh.n_vertices:
            data = bpy.data.meshes.new(name)
            data.from_pydata(
                [tuple(float(c) * MM_TO_BU for c in v) for v in mesh.vertices],
                [],
                [tuple(int(i) for i in f) for f in mesh.faces],
            )
            data.update()
            old = obj.data
            obj.data = data
            if old.users == 0:
                bpy.data.meshes.remove(old)
    bpy.context.view_layer.update()


def objects() -> dict:
    return dict(_OBJECTS)


def _to_blender(matrix: np.ndarray):
    """A millimetre 4x4 as a Blender metre matrix."""
    import mathutils

    scaled = np.array(matrix, dtype=float, copy=True)
    scaled[:3, 3] *= MM_TO_BU
    return mathutils.Matrix([[float(v) for v in row] for row in scaled])


def _collection_for(node) -> str:
    for tag, collection in _COLLECTION_FOR.items():
        if node.tags.get(tag):
            if tag == "component" and "cutting_block" in node.name:
                return "CuttingBlocks"
            return collection
    return "Planning"
```

- [ ] **Step 4: Rewire the add-on onto the session**

In `tka_planner/addon/__init__.py`:

1. Delete `_restore_cuts_when_idle`, `_cut_affecting_bones`, `_rebake_for_animation`,
   `_defer_cuts`, `RECUT_IDLE_SECONDS`, `_LAST_CHANGE`, `_TIMER_ARMED`,
   `_PENDING_BONES`. The debounce exists only because a boolean ran inside a drag, and
   none does now.
2. Delete `_adjustments_from`, `_sizing_for`, `_plan_from`, `_find_bone_files` and
   `_report_lines`. They now live in `session.py`.
3. Replace the body of `_replan` with:

```python
def _replan(properties, context) -> None:
    """Push the panel's control values into the session and draw the result.

    Silent when there is nothing to update: the properties are also written when a
    blend file loads and when Plan itself runs, and neither should trigger a half-built
    scene. Failures land in the panel's status line rather than raising, because an
    exception inside a property callback leaves Blender's UI in a bad state and says
    nothing useful to the person holding the slider.
    """
    session = _session()
    if session is None or not properties.has_scene:
        return

    try:
        from tka_planner.blender import render
        from tka_planner.session import ADJUSTMENT_FIELDS

        changes = {
            name: getattr(properties, name)
            for name in (*ADJUSTMENT_FIELDS, "philosophy", "size_override",
                         "tibial_resection_mm", "insert_thickness_mm", "use_insert",
                         "isolate_landmarks", "show_landmarks")
        }
        delta = session.replan(**changes)
        render.apply_delta(delta)

        properties.result_lines = "\n".join(session.report_lines())
        properties.status_is_error = False
        properties.status = (
            "Adjusted by hand" if delta.scalars.get("adjusted") else "As planned"
        )
        if session.stale:
            properties.status += " - press Commit to re-cut"
    except Exception as error:
        traceback.print_exc()
        properties.status_is_error = True
        properties.status = f"{type(error).__name__}: {error}"
```

4. Replace the body of `_apply_trial_pose` with:

```python
def _apply_trial_pose(properties, context) -> None:
    """Pose the implanted construct by hand. Never re-plans and never cuts."""
    session = _session()
    if session is None or not properties.has_scene:
        return

    from tka_planner.blender import render

    session.set_trial(
        flexion_deg=properties.trial_flexion_deg,
        varus_valgus_deg=properties.trial_varus_valgus_deg,
        drawer_ap_mm=properties.trial_drawer_ap_mm,
    )
    render.apply_scene(session.scene)
```

5. Replace `TKA_OT_plan.execute` so it opens a `PlanningSession`, calls `build()`, then
   `render.render_scene(session.scene)`, and stores the session in `_SESSION["current"]`.
6. Add a `TKA_OT_commit` operator with `bl_idname = "tka.commit"`, calling
   `session.commit()` then `render.apply_scene(session.scene)`, and add its button to
   the panel directly below **Plan**.
7. Delete the `defer_cuts`, `live_cuts` and `cut_solver` properties and their panel
   rows. Keep `resection_mode` and `build_bone_shells`, which the session reads.

- [ ] **Step 5: Reduce `build.py` to the parts still used**

Everything in `tka_planner/blender/build.py` is now either in the engine or dead. Delete
the file and update `tka_planner/blender/__init__.py` to export from `render` and `io`
instead. The archived copy at `archive/blender-addon/blender/build.py` remains.

- [ ] **Step 6: Run the parity test**

Run the add-on parity test inside Blender:

```bash
blender --background --python-expr "import pytest, sys; sys.exit(pytest.main(['-m','blender','tests/test_addon_parity.py','-v']))"
```

Expected: five passes.

- [ ] **Step 7: Drive the add-on by hand against the parity list**

Install the rebuilt add-on and work through section 9 of the spec, control by control.

```bash
python build_addon.py
```

Confirm each of the thirteen adjustments moves what it should, both resection modes
commit, the shells build, the three trial controls behave and reset, and the reported
numbers match what the archived add-on gives for the same case.

- [ ] **Step 8: Run the whole suite**

Run: `pytest -q`
Expected: all green, Blender tests deselected.

- [ ] **Step 9: Commit**

```bash
git add tka_planner/blender tka_planner/addon tests/test_addon_parity.py
git commit -m "Point the add-on at the engine, leaving Blender as a viewer"
```

---

## Done when

- `pytest -q` is green.
- `pytest -m blender` is green with Blender installed, including the kernel agreement
  gate and the add-on parity test.
- `grep -rn "import bpy" tka_planner/geom tka_planner/scene tka_planner/session.py`
  returns only `geom/kernels/blender_kernel.py`.
- The add-on does everything section 9 of the spec lists, with no Blender boolean
  anywhere in the plan path.

Milestones 6 to 9 of the spec, meaning persistence, the server and the web viewer, get
their own plan, written once this one has landed.
