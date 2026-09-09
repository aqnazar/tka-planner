# Standalone application implementation plan

**Goal:** Make the planner run as an application of its own: a local server, a browser
viewer, and a case store, with Blender required for nothing.

**Architecture:** The engine built in milestones 1-5 is already headless. This plan puts
three things around it. A `store` package persists cases, plans and committed geometry,
keyed so that re-committing a plan already committed costs nothing. A `server` package
exposes the session over HTTP as a set of plain functions with a transport adapter in
front. A `web` viewer renders the scene and drives the same controls the add-on panel
drives.

**Spec:** `docs/superpowers/specs/2026-09-09-standalone-application-design.md`, milestones
6 to 9.

## Global constraints

- Python floor 3.10. `core/` stays numpy-only.
- No `bpy` outside `geom/kernels/blender_kernel.py` and `blender/`.
- Millimetres throughout the engine. Unit conversion to metres happens only in the glTF
  encoder and in the viewer's reading of a node transform.
- Apache 2.0 stays clean: no GPL in the shipped dependency set. Vendored JavaScript must
  be MIT or similar and its licence text must travel with it.
- Cases are identified as `CASE_00N`. The mapping to real identities is never committed
  and never written into the repository.

## Two deviations from the spec, and why

**The spec names FastAPI; this plan uses the standard library.** The shipped dependency
set is numpy and manifold3d. Adding FastAPI pulls in Starlette, Pydantic, `anyio` and
uvicorn to serve one user on one machine, and none of what those buy is load-bearing
here: there is no schema to publish, no concurrency to manage and no third party
calling in. The routing is written as plain functions over a session, with the HTTP
transport as a thin adapter, so putting FastAPI in front later is an adapter file and
not a rewrite.

**The spec names a WebSocket delta channel; this plan returns the delta in the HTTP
response.** Every delta in a local session is caused by a request the viewer itself just
made, so a second channel to deliver it adds a socket, a reconnect policy and an
ordering question in exchange for nothing. The delta format is unchanged and is exactly
what a socket would have carried, so the channel can be added when a second writer, such
as a tracker feed, actually exists to push from.

## File structure

| File | Responsibility |
|---|---|
| `tka_planner/store/meshes.py` | Content-addressed mesh directory |
| `tka_planner/store/db.py` | SQLite: cases, plans, commits |
| `tka_planner/store/keys.py` | The plan hash that keys the commit cache |
| `tka_planner/geom/gltf.py` | GLB encoding, millimetres to metres |
| `tka_planner/server/sessions.py` | Live sessions, and the store behind them |
| `tka_planner/server/api.py` | Routes as plain functions, transport-free |
| `tka_planner/server/encode.py` | Scene and delta to JSON |
| `tka_planner/server/httpd.py` | Standard-library HTTP adapter and static files |
| `tka_planner/server/__main__.py` | `python -m tka_planner.server` |
| `web/index.html` | Page, import map, control panel markup |
| `web/app.js` | Controller: state, requests, panel wiring |
| `web/viewer.js` | three.js scene, camera, clipping, caps, landmarks |
| `web/glb.js` | Minimal GLB reader for our single-mesh payloads |
| `web/style.css` | Panel and layout |
| `web/vendor/` | three.js and OrbitControls, vendored for offline use |

## Tasks

### Task 1: Mesh store

Content-addressed directory. `MeshStore.put(mesh) -> id` writes
`<root>/<first two hex>/<id>.npz` and is a no-op if the file exists. `get(id) -> Mesh`
reads it back. Ids come from the same hash `scene.model` already uses, promoted to a
public `mesh_id`. Gate: a round trip preserves vertices and faces exactly, and putting
the same mesh twice writes one file.

### Task 2: Plan key

`plan_key(session) -> str` over everything a boolean depends on: both source mesh
digests, the side, the resection mode, the shell flag, and every resection plane and
component pose rounded to twelve decimals. Display toggles must not change it. Gate:
moving a slider that moves a cut changes the key; toggling landmark visibility does not.

### Task 3: Case database

SQLite with `cases`, `plans`, `commits` and `meshes`. Every table carries an `owner`
column defaulting to the empty string, so multi-user is a filter rather than a
migration. `Store.cached_commit(key)` returns the committed node-to-mesh map or `None`;
`Store.save_commit(...)` records it. Gate: a commit saved and read back returns the same
mesh ids, and a second session on a cold process finds the cache.

### Task 4: GLB encoder

`write_glb(mesh) -> bytes`: one buffer, one mesh, one primitive, positions and indices,
vertices divided by a thousand. Gate: the header parses, the JSON chunk is valid, and
the accessor bounds match the mesh bounds in metres.

### Task 5: Session manager and API

`SessionManager` opens a `PlanningSession`, holds it under an id, and owns the store.
`Api.dispatch(method, path, body)` routes to plain functions returning
`(status, payload)`. Every mutating route answers with the same envelope: the delta, the
clipping planes, the report lines, the controls, the trial values and the stale flag.
Gate: the whole parity list from spec section 9 drives the API in a test with no browser.

### Task 6: HTTP adapter

`ThreadingHTTPServer` with a handler that serves `/api/...` through the dispatcher and
everything else from `web/`. Binds to localhost only. Gate: a test client walks the
endpoints over a real socket.

### Task 7: Viewer

three.js scene fed by the snapshot. Plan mode clips the bones with the planned planes
and draws a back-face cap so the cut reads as solid. Commit swaps in the cut meshes.
Reduce mode plays the arc the server returns and applies the trial pose. The panel
carries every control in spec section 9. Gate: the parity list, by hand in a browser.

### Task 8: Entry point

`tka serve` starts the server and opens the browser. Gate: the command runs and the page
loads with no network access.

### Task 9: Retire the add-on

Deferred deliberately. The add-on is the only renderer that has been driven by a person,
so it stays until the viewer has been. `archive/` already holds the frozen copy.
