# Standalone application: design

Date: 2026-09-09
Status: approved. Milestones 1 to 8 built; milestone 9 held open deliberately, see
section 11.

Turns the Blender-dependent planner into a standalone application: a local web
application with a Python backend and a browser viewer, with no Blender required to
run it, no feature lost, and explicit seams for tracker integration and a multi-user
case database.

## 1. Decisions

| Decision | Choice |
|---|---|
| Runtime | Local web application. Python backend, browser viewer, launched on localhost. |
| Migration | Strangler port. Extract a headless engine, adapt the add-on onto it, then retire the add-on. |
| Geometry kernel | Pluggable. `manifold3d` is the default. Blender is an optional cross-check backend. |
| Licence | Apache 2.0 preserved. No GPL code in the shipped dependency set. |
| Interaction model | Three phases: Plan, Commit, Reduce. No boolean ever runs inside a drag. |
| Motion primitive | One 4x4 pose per bone set. Flexion, stress, drawer and registration all write it. |
| Expansion targets | Robot and tracker integration; case database and multi-user. |

Explicitly out of scope: DICOM import and in-app segmentation; joints other than the
knee. The design must not foreclose them, but nothing is built for them now.

## 2. Archive

Before any code moves:

1. Commit the current working tree so the snapshot is of a real commit.
2. Copy `tka_planner/addon/` and `tka_planner/blender/` verbatim to
   `archive/blender-addon/`.
3. Write `archive/blender-addon/PROVENANCE.md` naming the source commit, the date, the
   feature set it implemented, and its role as the parity reference for the port.

`archive/` is distinct from `legacy/`. `legacy/` holds the frozen script behind the
first paper and is cited, never run. `archive/` holds a working add-on that may be
revived, and is run by the parity tests during the port.

The live copies under `tka_planner/` keep working throughout the port. They stop being
the product and become the thing the engine is diffed against. They are deleted only
when the parity suite passes against the web application.

## 3. Package layout

Not restructured: `core/`, `report/`, `cli.py`, `validation/`. `tests/` keeps every
existing test unchanged and gains new ones.

| Package | Contents |
|---|---|
| `tka_planner/geom/` | `Mesh` type, kernel protocol, kernel backends, mesh repair, STL and glTF io |
| `tka_planner/scene/` | Scene model, builder, updater, resection, motion |
| `tka_planner/session.py` | Planning session: cached measurement, current controls, replan, commit |
| `tka_planner/server/` | FastAPI application, REST endpoints, WebSocket delta channel |
| `tka_planner/store/` | Case persistence: SQLite plus content-addressed mesh directory |
| `web/` | three.js front end |
| `archive/blender-addon/` | Frozen reference implementation |

`session.py` has no equivalent today. Its logic currently lives in the add-on's property
callbacks, entangled with Blender's UI lifecycle. Extracting it is what allows the same
session to be driven by a browser, by a test, or later by a tracker feed.

The engine works in millimetres throughout. Unit conversion happens only at the glTF
encoding boundary, which removes the millimetre-to-Blender-unit conversion that
`blender/build.py` currently threads through every function.

## 4. Geometry kernel

### 4.1 Interface

`geom.kernel.MeshKernel` is a protocol with two methods:

```python
def difference(self, target: Mesh, tool: Mesh) -> KernelResult: ...
def intersect(self, target: Mesh, tool: Mesh) -> KernelResult: ...
```

`KernelResult` carries the resulting `Mesh` and a record: which backend ran, whether
repair was applied to either input, and whether a fallback was taken. That record is
written into `plan.json` beside the existing measurement provenance. The project already
states how every number was obtained; this extends the same guarantee to geometry.

### 4.2 Backends

`ManifoldKernel` wraps `manifold3d`. It is the default and the only one in the shipped
dependency set. `manifold3d` requires manifold input, and segmentations are not reliably
manifold, so the backend runs a repair pass on failure and records that it did.

`BlenderKernel` imports `bpy` lazily and exists only for the cross-check test. It is
never imported by the server, never a runtime dependency, and lives behind an optional
extra so a normal install does not pull it.

### 4.3 Correctness gate

A test asserts that the two backends agree within tolerance on the committed fixtures,
comparing volume, surface area, bounding box and Hausdorff distance. This is the spike
that must pass before the port is trusted. It is a correctness question, not a latency
one, because the two-mode design removes the boolean from every interactive path.

## 5. Interaction model

### 5.1 Plan mode

The bones sit in the extended reference pose and do not move. The user sets alignment
philosophy, sizing, resection depths, slope, rotation, component position and insert
thickness. The planned resection planes are previewed with a GPU clipping plane and a
stencil cap pass, so the visible cut follows the slider at display refresh with no
geometry work on either side. In cutting-block mode the block also seats on the bone as
the plan changes, so the instrument is visible even though the bone is not yet carved.

No boolean runs in Plan mode.

The clipping-plane preview is not an approximation. A planar resection is a half-space,
and the clipping plane evaluates the same half-space per pixel rather than per triangle.
The only difference from the committed geometry is the triangulation of the cut face.

### 5.2 Commit

One explicit action, with progress indication. It runs the real booleans at full
resolution and produces the committed meshes: resected femur, resected tibia, and the
bone shells if requested. Only bones whose cut actually changed are recomputed, reusing
the `_cut_affecting_bones` logic the add-on already has.

Commit is keyed by the hash of the plan that produced it and cached in the store, so
returning to a plan already committed is instant.

Landmarks are committed with the geometry, in set-local coordinates.

"Commit" is used throughout in preference to "bake". Baking animation frames, meaning
per-frame precomputed geometry, is removed entirely by this design. Committing a
resection is a single geometry operation and is unrelated.

### 5.3 Reduce mode

Operates on committed meshes only. Flexion, varus and valgus stress, AP drawer,
per-compartment gap through the arc, mechanical axis and hip centre checks. Every one is
a rigid transform or a numpy calculation over static geometry. Nothing is recomputed.
Real-time here is structural, not an optimisation.

Editing a plan control while in Reduce marks the construct stale and offers a return to
Plan. The application never shows a pose of geometry that no longer matches the plan.

### 5.4 What this removes

The animation bake, the selective per-bone rebake, the modifier mute-and-settle
debounce, the idle timer, the frame-1 visibility switching rule, and the `EDIT_ONLY`
tagging scheme all become unnecessary. They exist because Blender evaluates its modifier
stack on the thread that draws the viewport. Neither the constraint nor the machinery
survives the move.

### 5.5 Machine requirement

Integrated graphics, four cores, 16GB, no discrete GPU. Both interactive modes are
static-geometry rendering with matrix updates. The only heavy moment is the commit,
which is a deliberate action rather than something hidden inside a drag.

## 6. Scene model

### 6.1 Structure

```
Scene
  FemoralSet   pose: 4x4
    Bone        committed resected femur
    Implants    femoral component, cutting block, cutting block shell
    Landmarks   named points, categorised
  TibialSet    pose: 4x4
    Bone        committed resected tibia
    Implants    tray, insert, cutting block, cutting block shell
    Landmarks   named points, categorised
  Planning      cut planes, mechanical axes  (Plan mode only)
```

A `Scene` holds a mesh table keyed by content hash and a node table. A node carries a
name, a mesh reference or a primitive description, a rest transform, a colour, an alpha,
a visibility flag and its tags.

Every child holds a **rest transform** relative to its set origin, fixed at commit. The
set pose composes onto it. The rest basis is an explicit stored field rather than an
implicit parent relationship, because the trial-rig drift bug came from rotations being
applied without composing onto the rest basis, and a model that stores it explicitly
cannot express that mistake.

### 6.2 Landmarks

Committed in set-local coordinates so they stay glued to their bone through any pose
without recomputation. Rendered as instanced spheres, one draw call per set. Coloured by
the categories the add-on already defines. Visibility toggles per set, independently of
the bone and the implants.

### 6.3 Motion and the registration seam

The set pose is the only motion primitive in Reduce mode. Flexion writes the tibial set
pose relative to the femoral. Stress and drawer compose onto the same pose.

Both sets carry a free pose. Camera-based registration therefore needs no new plumbing:
observed landmark positions plus the same named landmarks in set-local coordinates solve
a rigid transform through Kabsch, and the result is written to the set pose through the
identical call flexion uses. The tracker becomes another writer on an existing channel.

### 6.4 Deltas

`scene.update` returns a delta: poses changed, meshes replaced, visibility changed. The
delta is the WebSocket payload, so the wire format falls out of the model rather than
being designed separately.

## 7. Server and viewer

FastAPI. REST for cases, plans, commits and exports. One WebSocket per session carrying
the delta stream. Committed meshes transfer once as compressed glTF and are addressed by
content hash afterwards; everything subsequent is poses and small scalars.

The viewer is three.js. It owns camera, clipping planes and stencil caps, set poses,
instanced landmarks, and a control panel mirroring the add-on's. It computes nothing
clinical. Every number displayed comes from the server, which preserves the project's
guarantee that the method behind a number is auditable.

## 8. Persistence

SQLite for records, a content-addressed directory for meshes.

A case holds patient meshes, landmark sets, plans with their full provenance, and
committed geometry keyed by plan hash. Multi-user is then a change of database URL plus
an authentication layer, not a rewrite. Nothing in the schema assumes a single user or a
single machine.

Patient identity handling is unchanged. Cases are identified as `CASE_00N` and the
mapping to real identities is never committed.

## 9. Feature parity

The port is complete when every item below works in the web application. This list is
the acceptance criterion, and it is what the parity suite encodes.

**Computation, already Blender-free and unchanged:** deformity metrics with provenance
tiers; continuous parametric sizing; mechanical and kinematic alignment planning;
per-compartment resection depths; QC; `plan.json`, `report.html`, `landmarks.json`; the
`tka` command line.

**Scene:** bone import; cut planes; mechanical axes; landmark display with category
colouring and isolation; implant components, cutting blocks, cutting block shells and
insert seated on their cuts; continuous size rescaling without re-import; clean viewport
mode.

**Resection:** cutting-block mode subtracting the block and intersecting for the
patient-specific shell; cut-plane mode; the two remaining alternatives, never stacked;
bone shells taken before resection.

**Plan adjustments, all thirteen:** `coronal_correction_deg`,
`femoral_resection_delta_mm`, `femoral_flexion_delta_deg`, `femoral_varus_delta_deg`,
`femoral_rotation_delta_deg`, `femoral_shift_ap_mm`, `femoral_shift_ml_mm`,
`tibial_resection_delta_mm`, `tibial_slope_delta_deg`, `tibial_varus_delta_deg`,
`tibial_rotation_delta_deg`, `tibial_shift_ap_mm`, `tibial_shift_ml_mm`. Plus alignment
philosophy, size override, tibial resection, insert thickness and insert on or off.

**Reduce:** `trial_flexion_deg`, `trial_varus_valgus_deg`, `trial_drawer_ap_mm`; the
scripted flexion arc; combination of a scrubbed arc position with a dialled stress;
reset trial pose.

**Session:** patient folder loading with the existing filename conventions; side
selection; implant library resolution including the outward search for incomplete size
folders; reset to plan; clear; the reported metric lines; the warning when per-cut varus
controls break mediolateral agreement.

**Behaviour changes, accepted:** in cutting-block mode the carved bone appears at commit
rather than at the end of each drag. Today it appears at the end of each drag only after
a settle of up to twenty two seconds. The scripted flexion arc plays without a bake.

## 10. Testing

1. The existing 462 core tests keep passing unchanged, at every commit.
2. Kernel agreement: `ManifoldKernel` and `BlenderKernel` agree within tolerance on the
   committed fixtures.
3. Scene parity: for a fixed plan, the headless scene and the add-on scene agree on node
   names, poses and mesh volumes.
4. Session tests: every control in section 9 drives the session headlessly and asserts
   the resulting delta, with no browser and no Blender.
5. Rest-basis regression: composing a set pose twice and resetting returns to identity,
   which is the drift bug encoded as a test.
6. Server tests over the FastAPI test client, including the WebSocket delta stream.

## 11. Milestones

1. Archive the add-on. Commit the working tree first.
2. `geom`: `Mesh`, kernel protocol, `ManifoldKernel`, `BlenderKernel`, repair, and the
   agreement test. Gate: agreement passes on the fixtures.
3. `scene`: model, builder, updater, resection, motion. Gate: scene parity passes.
4. `session.py`: extract from the add-on. Gate: every control in section 9 drives it
   headlessly.
5. Point the live add-on at the engine, so it becomes a renderer only. Gate: the add-on
   still does everything it does today.
6. `store`: SQLite and the content-addressed mesh directory, with the commit cache.
7. `server`: REST and the WebSocket delta channel.
8. `web`: viewer, Plan mode with clipping preview, Commit, Reduce mode, control panel.
   Gate: the full parity list.
9. Retire the live add-on. `archive/` remains.

Milestones 2, 3 and 4 have no user-visible effect and cannot regress the add-on. The
port only becomes visible at milestone 5.

Milestones 1 to 5 are the engine extraction and get the first implementation plan. They
end with the add-on running unchanged on a Blender-free engine, which is a complete and
useful state on its own. Milestones 6 to 9 are the application and get a second plan,
written once the engine exists and its shape is known rather than guessed at now.

### Status

Milestones 1 to 8 are built. Two decisions differ from what section 7 called for, and
both are recorded with their reasoning in `docs/METHODS.md`: the HTTP layer is the
standard library rather than FastAPI, and the delta comes back in the response rather
than over a WebSocket. Neither changes the delta format, and both were taken to keep the
shipped dependency set at numpy and manifold3d.

**Milestone 9 is deliberately not done.** The add-on is the only front end a person has
driven, and the parity list in section 9 is checked headlessly against the API rather
than in a browser. Retiring the add-on before somebody has planned a real case in the
viewer would remove the only working tool on the strength of tests that cannot see. It
should be retired after a first real session, not before, and `archive/` already holds
the frozen copy either way.

Section 12's first open question is answered: the stencil cap pass is not worth its
complexity. A bone is closed, so a second copy drawn inside out behind the same clipping
plane shows the back faces visible through the cut, and in resected-bone colour that
reads as solid. The second question, whether repair changes committed geometry enough to
matter, was answered by the agreement gate: the two solvers agree to within 0.0001 mm on
the fixtures.

Test 3 in section 10 requires a Blender installation and is marked `blender`, so it
stays excluded from the default run like the existing Blender-dependent tests.

## 12. Open questions

None blocking. Two to settle during implementation:

- Whether the stencil cap pass is worth its complexity in cut-plane mode, or whether an
  uncapped preview plus the seated block reads clearly enough. Decide by looking at it.
- Whether `manifold3d` repair changes committed geometry enough to matter clinically on
  any fixture. The agreement test in milestone 2 answers this with numbers.
