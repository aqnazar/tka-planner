# tka-planner

An open, reproducible pre-operative planning pipeline for total knee arthroplasty,
built around a fully parametric implant system.

> **Not a medical device.** Research and demonstration only. See [NOTICE](NOTICE).

Commercial arthroplasty planners — Stryker Mako, Zimmer Biomet ROSA, and their peers —
are closed. You see the output and never the method: which landmarks were used, which
axis definition, what was measured versus assumed. This project publishes all of it.

## What makes it different

**Every number states how it was obtained.** Each metric carries a status of
`measured`, `estimated`, or `not_computable`, together with the versioned method that
produced it and any assumption that was substituted. A knee-only CT cannot yield a true
hip-knee-ankle angle, so the pipeline says so rather than printing a plausible number
derived from a population average. Where an assumption *is* used, it is named, sourced,
and overridable per case.

**Plans are data, not side effects.** A run produces a `plan.json` containing the input
file hashes, the landmarks, the anatomical frames, every metric with its provenance,
and the component poses as 4x4 matrices. Given the same inputs, anyone can re-derive
the same plan. Blender renders a plan into geometry; it does not decide anything.

**The implant is parametric, so planning is too.** The implant is a single master
geometry driven by continuous parameters. Discrete sizes are a legacy convenience, and
a bone that falls between them — or beyond the largest — is not a problem to be clamped.

**Nothing needs Blender, and nothing needs the internet.** Measurement, metrics,
sizing, alignment planning, the scene and the cutting all run in plain Python on numpy
and `manifold3d`. It runs as an application of its own: a local server and a browser
viewer, started with one command, binding the loopback interface only, with the
JavaScript served from this repository rather than a content delivery network. Blender
is one optional front end that draws a scene the engine has already decided; it makes
none of the decisions and can be removed without losing a feature.

**Geometry says how it was made, too.** Every cut records which solver ran, what it had
to repair in the segmentation first, and whether it fell back — the same guarantee the
measurements already carried. `manifold3d` is the default and refuses a non-manifold
input rather than guessing, which turns a class of silent wrong answers into a loud one.
It is checked against Blender's exact solver on committed fixtures, where the two agree
to within 0.0001 mm on volume, surface area, bounding box and Hausdorff distance.

## What it does

From two segmented bone surfaces, with no manual picking required:

- **Measures** the deformity — mLDFA, aLDFA, MPTA, JLCA, condylar twist, native
  posterior slope — each with its provenance tier
- **Sizes** the implant on a continuous parameter, with the tibial dimension taken from
  the resection cross-section rather than a bounding box
- **Plans** the correction: distal femoral valgus cut angle, posterior slope, and
  per-compartment resection depths, under mechanical or kinematic alignment
- **Places** the implants, cutting blocks, shells and insert, each on its own cut
- **Cuts** both bones along the planned planes
- **Flexes** the joint, femur fixed, tibia swinging about the transepicondylar axis, as
  a rigid transform of the committed geometry, so playback needs no bake and stays
  smooth regardless of segmentation density
- **Adjusts** by hand — varus/valgus, both resection depths, slope, femoral flexion,
  component rotation and position, insert thickness and size — with the cuts, the
  components and the bone following as the value changes
- **Trials** the finished construct by hand — flexion, a varus/valgus stress, an AP
  drawer — to check range of motion and impingement once it is cut and implanted,
  instantly and without touching the plan
- **Remembers** what it cut, keyed on the geometry the cut depends on rather than on
  the controls, so returning to a plan already committed costs a disk read instead of a
  second run of the booleans

Methodology, including what each decision replaced and why, is in
[docs/METHODS.md](docs/METHODS.md). The clinical assumptions still awaiting a surgeon's
answer are collected in [docs/CLINICAL_QUESTIONS.md](docs/CLINICAL_QUESTIONS.md), also in
Russian as [docs/CLINICAL_QUESTIONS.ru.md](docs/CLINICAL_QUESTIONS.ru.md).

## Install

```bash
pip install -e ".[dev]"
pytest
```

`pytest` runs the whole suite, cutting included, with no Blender installed. Node is
used by three of the tests to read a glTF file the Python encoder wrote, using the
viewer's own reader; they skip if it is absent.

Blender 4.4+ is needed only to use the add-on, and is tested through 5.1. Two checks
have to be run inside it deliberately, and both gate on their exit code:

```bash
blender --background --factory-startup --python tests/addon_parity.py
blender --background --factory-startup --python tests/kernel_reference.py
```

The first drives the engine through the Blender renderer. The second regenerates the
reference geometry the kernel agreement gate compares against, and is needed only when
those cases change.

## How to operate it

Three ways in. The application is the one to stand in front of. Blender is now optional
and draws a scene it no longer computes; the command line does measurement and reporting
without a screen at all.

### The application

```bash
tka serve --cases path/to/cases --library path/to/implant/library
```

That starts a local server and opens a browser at `http://127.0.0.1:8731/`. Nothing
leaves the machine: the server binds the loopback interface only, the viewer's
JavaScript is served from the repository rather than a content delivery network, and no
request goes anywhere else. It works with the network cable out.

Pick a case, or type the path to a patient folder, and press Open.

**Plan.** The panel carries every control: alignment philosophy, implant size, resection
depths, slope, rotation, component position and insert thickness. The planned cut is
previewed by clipping the bone against the planned half-space, so the cut follows the
slider at the refresh rate of the screen. Nothing is carved yet. The readout on the
right rebuilds on every change, so it cannot lag the plan.

**Commit.** One press cuts the bones for real, at full resolution. On a 1.2 million
triangle segmentation that is around forty seconds with the bone shells on. Only the
bones whose cut actually moved are recomputed, and a cut computed once is stored: coming
back to a plan already committed takes under a second, in this session or in a later
one.

**Reduce.** Flexion, a varus or valgus stress, an anteroposterior drawer, and a scripted
flexion arc, all on the committed geometry. Every one is a rigid transform of the tibial
set, so the response is immediate and stays immediate on integrated graphics. Editing a
plan control here says so and offers the way back to Plan, because the application never
shows a pose of geometry that no longer matches the plan.

`Export plan` writes `plan.json`, `report.html`, `landmarks.json` and the geometry as
STL, including the manual adjustments, which the command line has no way to record.

### The planning screen (Blender)

Build the add-on package, then install it from disk:

```bash
python build_addon.py          # writes tka_planner.zip
```

In Blender: **Edit > Preferences > Add-ons**, open the dropdown at the top right,
choose **Install from Disk...**, select `tka_planner.zip`, and tick **TKA Planner**.

Adding the repository to Blender's script paths does *not* work. Blender scans a script
path for an `addons` subdirectory and expects a module or package whose `__init__`
declares `bl_info`; a repository checkout is neither shape, so nothing appears in the
list however often it is refreshed.

Then press **N** in the 3D viewport and open the **TKA** tab. Set the patient folder,
pick the side, optionally point at the implant library, and press **Plan**.

The bones load, the cut planes and mechanical axes appear, the implants and cutting
blocks seat on the cuts, and the correction angle, resection depths and sizing appear in
the sidebar.

Everything below **Plan** in the panel then adjusts that plan in place, and the scene
follows as the value changes — 15 to 26 updates a second on a full-resolution pair while
a control that moves a cutter is being dragged, and instantly for the few that never do
(below):

| Control | What moves |
|---|---|
| Varus / valgus | Both cuts together, with every component, keeping their shared mediolateral slope |
| Alignment, size, tibial resection | The whole plan; size rescales the implants without re-importing them |
| Femoral: resection, flexion (cut), varus, rotation, AP and ML position | The distal cut and the femoral component and blocks |
| Tibial: resection, slope, varus, rotation, AP and ML position | The proximal cut and the tray, insert and blocks |
| Insert thickness | The insert slab, and the extension gap reported per compartment |

**Resect with** chooses what removes the bone. *Cutting block* subtracts the block
itself and intersects a copy of each bone with the block's shell, giving the
patient-specific mating surface — the same pair of booleans the first pipeline used.
*Cut plane* removes everything beyond the planned plane instead. The two are
alternatives, never stacked: the plane would swallow the surfaces the block is shaping.

Cutting is a separate, explicit step. **Plan** puts the anatomy, the planes, the axes
and the implants on screen and every control moves them at full speed; **Commit cuts**
runs the resection at full resolution, once, when you ask for it.

That split is the whole performance story. An exact boolean against a real segmentation
costs seconds, which cannot live inside a slider drag. The previous version hid it
behind a mute-and-recompute debounce and still paid up to twenty two seconds of stall
every time a drag settled. Taking the boolean out of the drag entirely removes both the
stall and the machinery that managed it, and a control change becomes what it always
should have been: a few dozen numpy operations and a matrix assignment.

A commit re-cuts only the bones whose cut actually changed, so adjusting the tibial
slope leaves an untouched femur alone. Bone shells have their own toggle, as they are an
export deliverable rather than something alignment is judged on.

The panel says **press Commit to cut** whenever the plan has moved since the last one,
so the viewport never claims geometry that no longer matches the plan.

The per-cut varus controls break the mediolateral agreement between the two cuts on
purpose, and the plan warns when they do.

### Trial reduction

Once a plan is cut and implanted, **Trial reduction** poses the finished construct by
hand — Flexion (trial), Varus/valgus stress, AP drawer — the way a surgeon checks range
of motion, ligament balance and impingement intraoperatively. These three are rigid
transforms of the committed bones: no boolean is involved, so they are instant
regardless of resection mode, and they never alter the plan.

A pose is a function of the three values alone, composed onto a stored rest basis rather
than accumulated onto wherever the last one left the construct. Setting the same values
twice gives the same pose and zeroing them returns exactly to extension, which the
previous parented rig did not manage. **Reset trial pose** returns all three to zero.

Distinct from Femoral's **Flexion (cut)**, which changes the femoral cut's sagittal angle
and therefore the plan — the two are named to tell them apart, and each control's tooltip
points at the other.

**Reset to plan** returns every manual control to the computed plan. Adjustments are part
of the plan record, so an adjusted plan still re-derives from its own file.

The folder is expected to contain `FD1Left.stl` and `TD1Left.stl` (or the Right
equivalents). Files named for the bone also work, so a folder exported straight out of
3D Slicer can be used as-is.

### The command line

Same computation, no Blender, useful for batches and for the paper's figures:

```bash
tka measure --femur FD1Left.stl --tibia TD1Left.stl --side left --out plan/
tka measure ... --philosophy kinematic --tibial-resection 8
```

It writes `plan.json` (hash-anchored to the input files), `report.html`
(self-contained), and `landmarks.json`. Pass `--landmarks` to use reviewed landmarks
instead of the automatic estimate.

## Inputs

Bone surfaces as STL, typically segmented in [3D Slicer](https://www.slicer.org/).
Landmarks are optional: without them the pipeline estimates a full set automatically and
marks every point `estimated`, awaiting review. Reviewed landmarks can be supplied as
Slicer markups (`.fcsv` or `.mrk.json`) or in the native landmark format.

Meshes and landmarks must declare their coordinate system. Slicer writes **LPS** by
default (+X patient-left, +Y posterior, +Z superior) and the pipeline works in that
frame directly rather than re-deriving one — the CT already carries an anatomical frame,
including the true spatial relationship between femur and tibia. A landmark file whose
points do not land on the mesh is rejected rather than silently mis-placed.

## Repository layout

| Path | Contents |
|---|---|
| `tka_planner/core/` | numpy only, no `bpy`. Frames, metrics, measurement, sizing, alignment planning. |
| `tka_planner/geom/` | Meshes and booleans. `manifold3d` by default, Blender only as a cross-check. |
| `tka_planner/scene/` | The scene: sets, rest transforms, builder, updater, resection, motion. No `bpy`. |
| `tka_planner/session.py` | One planning session, driveable with no user interface. |
| `tka_planner/panel.py` | The control panel described once, so every front end builds the same one. |
| `tka_planner/store/` | Cases and plans in SQLite, geometry in a content-addressed directory. |
| `tka_planner/server/` | The local application: routes, sessions, and an HTTP adapter. |
| `web/` | The browser viewer. three.js vendored, so it runs with no internet. |
| `tka_planner/blender/` | Blender adapter. Draws a scene the engine built. The only place millimetres become metres. |
| `tka_planner/addon/` | The planning screen: patient in, models and numbers out. |
| `tka_planner/report/` | Single-file HTML reports, no external assets. |
| `tka_planner/validation/` | Cohort runner and landmark sensitivity analysis. |
| `docs/METHODS.md` | Every methodological decision, why it was made, and what it replaced. |
| `archive/blender-addon/` | The add-on before the port, kept as its parity reference. |
| `legacy/` | The frozen script behind the first paper. Do not modify — see its `PROVENANCE.md`. |
| `tests/` | Runs without Blender. |

## Patient data

Cases are identified as `CASE_00N` only. The mapping to real identities is never
committed; see `.gitignore`. Full-resolution meshes stay out of the repository, so the
committed fixtures are coarse meshes and landmark coordinates.

## Citing

This is the second part of published work on a custom parametric TKA implant system.
Citation details to follow publication.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
