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
metrics, sizing and reporting all run in plain Python, which means the anatomical maths
is unit-tested against known-by-construction synthetic geometry, the sensitivity study
runs thousands of Monte Carlo iterations in minutes, and you can audit the methods
without installing anything heavy.

## Install

```bash
pip install -e ".[dev]"
pytest
```

`pytest` runs the full core test suite with no Blender installed. Blender-dependent
tests are marked `blender` and excluded by default.

Geometry production additionally needs Blender 4.4+ on the path.

## How to operate it

There are two ways in. The Blender add-on is the one to stand in front of.

### The planning screen (Blender)

1. Open Blender. **Edit > Preferences > File Paths > Scripts**, point it at this
   repository, and restart Blender.
2. **Edit > Preferences > Add-ons**, search "TKA", enable **TKA Planner**.
3. In the 3D viewport press **N** and open the **TKA** tab.
4. Set **Patient folder** to a folder holding one knee's segmented bones, choose the
   side, and press **Plan**.

The bones load, the resection planes and mechanical axes appear in the viewport, and
the correction angle, resection depths and sizing appear in the sidebar beside them.
Switch between mechanical and kinematic alignment and press Plan again to see the cuts
move.

The folder is expected to contain `FD1Left.stl` and `TD1Left.stl` (or the Right
equivalents). Files named for the bone — anything containing "femur" and "tibia" — also
work, so a folder exported straight out of 3D Slicer can be used as-is.

If Blender's Python lacks numpy, install it once:

```
"C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin\python.exe" -m pip install numpy
```

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

Bone surfaces as STL, typically segmented in [3D Slicer](https://www.slicer.org/), plus
a landmark file. Landmarks come from Slicer markups (`.fcsv` or `.mrk.json`) or from the
included Blender picking add-on.

Meshes and landmarks must declare their coordinate system. Slicer writes **LPS** by
default (+X patient-left, +Y posterior, +Z superior) and the pipeline works in that
frame directly rather than re-deriving one — the CT already carries an anatomical frame,
including the true spatial relationship between femur and tibia. A landmark file whose
points do not land on the mesh is rejected rather than silently mis-placed.

## Repository layout

| Path | Contents |
|---|---|
| `tka_planner/core/` | numpy only, no `bpy`. Frames, metrics, sizing, plan schema. |
| `tka_planner/blender/` | Thin Blender adapter. The only place millimetres become metres. |
| `tka_planner/addon/` | Landmark picking add-on. |
| `tka_planner/report/` | Single-file HTML reports, no external assets. |
| `tka_planner/validation/` | Cohort runner and landmark sensitivity analysis. |
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
