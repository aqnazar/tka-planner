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

**The core needs no Blender.** `tka_planner.core` depends only on numpy. Measurement,
metrics, sizing and alignment planning all run in plain Python, which means the
anatomical maths is unit-tested against known-by-construction synthetic geometry, the
sensitivity study runs thousands of Monte Carlo iterations in minutes, and you can audit
the methods without installing anything heavy. Blender renders decisions already made;
it does not make any of them.

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
- **Animates** flexion, femur fixed, tibia swinging about the transepicondylar axis, on
  a baked, modifier-free copy of the construct so playback stays smooth regardless of
  segmentation density
- **Adjusts** by hand — varus/valgus, both resection depths, slope, femoral flexion,
  component rotation and position, insert thickness and size — with the cuts, the
  components and the bone following as the value changes
- **Trials** the finished construct by hand — flexion, a varus/valgus stress, an AP
  drawer — to check range of motion and impingement once it is cut and implanted,
  instantly and without touching the plan

Methodology, including what each decision replaced and why, is in
[docs/METHODS.md](docs/METHODS.md). The clinical assumptions still awaiting a surgeon's
answer are collected in [docs/CLINICAL_QUESTIONS.md](docs/CLINICAL_QUESTIONS.md), also in
Russian as [docs/CLINICAL_QUESTIONS.ru.md](docs/CLINICAL_QUESTIONS.ru.md).

## Install

```bash
pip install -e ".[dev]"
pytest
```

`pytest` runs the full core test suite with no Blender installed. Blender-dependent
tests are marked `blender` and excluded by default.

Geometry production additionally needs Blender 4.4+ on the path. Tested through 5.1,
which replaced Action's flat `fcurves` list with a layered model the add-on detects and
handles either way.

## How to operate it

There are two ways in. The Blender add-on is the one to stand in front of.

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

The resection is **muted while a control is moving and recomputed once you stop**, which
is what keeps a drag responsive. An exact boolean against a real segmentation is not
cheap, measured on Patient_005 at full resolution:

| | dragging | settling after you stop |
|---|---|---|
| Cutting block, with bone shells | 15 fps | 22 s |
| Cutting block, no shells | 22 fps | 17 s |
| Cut plane | 24 fps | 6 s |
| No resection | 26 fps | — |

Two refinements sit on top of that measured baseline. First, a settle only re-solves
whichever bone actually moved — adjusting the tibial slope no longer also re-copies an
untouched femur. Second, a control that never moves a cutter skips the mute-and-recompute
cycle entirely rather than merely shortening it: AP/ML position, in-plane rotation, and
insert thickness change nothing a boolean depends on **in Cut plane mode**, so they are
instant there. In Cutting block mode the same controls are not free, because the cutting
block shares its implant's pose exactly — moving the implant moves the tool that would
cut it, on purpose (the block is described as realising the cut, not decorating it) — so
only insert thickness is free in that mode.

Turn off **Hide cuts while adjusting** to keep the resection live throughout, and expect
the panel to stall for that long on every change. Bone shells have their own toggle, as
they are an export deliverable rather than something alignment is judged on.

The per-cut varus controls break the mediolateral agreement between the two cuts on
purpose, and the plan warns when they do.

### Trial reduction

Once a plan is cut and implanted, **Trial reduction** poses the finished construct by
hand — Flexion (trial), Varus/valgus stress, AP drawer — the way a surgeon checks range
of motion, ligament balance and impingement intraoperatively. These three are rigid
transforms of the same baked, modifier-free bones the flexion animation plays back on:
no boolean is involved, so they are instant regardless of resection mode, and they never
alter the plan. Moving the timeline out of frame 1, or moving any trial control away from
zero, both mean the same thing to the viewport — show the implanted construct rather than
the live editing geometry — so the two switch automatically and can be combined: scrub to
a point in the scripted flexion arc, then dial in a stress there to see how the
construct behaves at that angle. Needs **Animate flexion** on when the plan is built;
**Reset trial pose** returns all three to zero.

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
| `tka_planner/blender/` | Blender adapter and scene builder. The only place millimetres become metres. |
| `tka_planner/addon/` | The planning screen: patient in, models and numbers out. |
| `tka_planner/report/` | Single-file HTML reports, no external assets. |
| `tka_planner/validation/` | Cohort runner and landmark sensitivity analysis. |
| `docs/METHODS.md` | Every methodological decision, why it was made, and what it replaced. |
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
