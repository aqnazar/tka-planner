# Paper 2 evaluation harness

One command per experiment. Every table and figure in the paper is written by one of
these, from one commit, with the commit and the machine recorded beside it.

```
python -m paper list                 # experiments, and the cohort found
python -m paper run E0               # one experiment
python -m paper run E1 E2 --quick    # smoke run: two cases, fewer samples
python -m paper run all --reference  # everything, plus the full-length reference knee
```

Results go to `paper/out/<id>/` (or `TKA_PAPER_OUT`): tables as CSV with a Markdown
copy, figures, a `summary.md` and a `run.json` with the commit, versions, machine and
seed. The folder is gitignored, because the cohort results are derived from patient
scans.

**Data.** The cohort is read from `TKA_COHORT_DIR` (default `Patient Dataset/` at the
top of the checkout, gitignored). A knee is a folder holding an `FD*` femur and a `TD*`
tibia, plus segmentation exports whose names carry the case id (`..._P009_L...`).
Knees are only ever named by case id and side. The implant catalogue is `data/implants/`
(or `TKA_IMPLANT_LIBRARY`). E7b also reads the private Fusion builds of the
patient-specific parts, read-only and only when `TKA_PATIENT_BUILDS` names their folder
(one `<knee>/` per build with `job.json`, `femoral.stl`, `plate.stl`). Experiments that need no cohort (synthetic knees, the
kernel checks) run from a plain checkout.

**The Paper 1 baseline (E12).** `python -m paper.legacy_runner P009L` runs the frozen
`legacy/TKA_Script_v3.py` on one knee in headless Blender (5.1 by default, the version
Paper 1 used; `--blender 4.4` for the other), changing only its path and side globals
and checking by hash that nothing else changed. The knee's bones are copied to a scratch
folder first, because the script writes `Output/` beside its inputs. It reads the implant
parts from `TKA_LEGACY_LIBRARY` (with `TKA_LEGACY_SIZE_CHART` when the chart lives
elsewhere), else from the catalogue. The catalogue cannot run every knee: it has no
right femoral components outside L4 and no L4 cutting blocks, under the names the script
expects. Paper 1's runs used the twelve-size CAD export with the catalogue's chart.

**Adding an experiment.** A module `paper/experiments/eNN_<topic>.py` with a
`run(ctx)` decorated by `@experiment("ENN", "title", needs_cohort=...)`, imported in
`paper/experiments/__init__.py`. Use `ctx.table(...)` for tables, `ctx.note(...)` for the
summary, `ctx.path(...)` for figures, `ctx.seed` for any randomness, and
`harness.measured(case)` so every experiment shares one measurement per knee and reports
the same refusals.

| Id | Experiment | Data |
|---|---|---|
| E0 | Cohort inventory and acceptance | cohort |
| E2 | Metamorphic invariance: mirror + side swap, random rigid motions, scanner rotation about each axis (-30 to +30 deg), decimation (50%, 10%), remeshing, vertex noise; automatic and oracle (carried) landmarks | cohort |
| E6c | Functional hip centre from camera-tracked pivoting: accuracy vs arc, noise, samples, pelvic motion (synthetic part) | synthetic |
| REG | Regression check: every landmark, frame, metric and the default plan per knee against a frozen baseline (`TKA_REG_FREEZE=1` freezes; baselines in `TKA_PAPER_OUT/baselines/`) | cohort |
| E7a | Sizing measurement: legacy bounding box vs resection cross-section at the planned cuts; chart-size changes | cohort |
| E7b | Implant fit on the cuts: chart size vs uniform scale vs patient-specific (and the Fusion-regenerated parts, with `TKA_PATIENT_BUILDS`) | cohort |
| E1 | Metric accuracy on synthetic knees: anatomy, width, tibial-axis and scanner-pose sweeps; provenance tiers | synthetic |
| E5 | Monte Carlo propagation of landmark picking error (PLACEHOLDER dispersion until E3) | cohort |
| E10a | Impingement-limited maximum flexion per knee, MA and KA plans, trajectories S1-S3, and the envelope | cohort + implant library |
| E10b | Posterior condylar offset per compartment: preoperative, planned (patient-specific and catalogue), change | cohort + implant library |
| E10c | Slope of modelled flexion against offset (component moved -3 to +3 mm) and against posterior femoral translation, against Bellemans 2002 and Banks 2003 | cohort + implant library |
| E12 | Paper 1 pipeline (frozen legacy script, headless Blender) vs the current planner; reproduction of the archived Paper 1 runs | cohort, Blender, legacy library |
| E8 | Geometry kernel: manifold3d against Blender's exact solver (gate cases, every installed Blender, cohort resections), repair statistics, refused defects | cohort optional; Blender optional (`TKA_BLENDER`) |
| E9 | Per-stage time per knee, commit time against triangle count, plan determinism in and across processes | cohort optional; synthetic fixture |
`needs_cohort="optional"` runs on the cohort when it is there and on its synthetic part
alone when it is not.
**Determinism.** `python -m paper.determinism --synthetic` prints the SHA-256 of the
plan file, of each of its sections and of the committed geometry, for the fixture in
`tests/fixtures/determinism/`. `--compare paper/determinism_synthetic.json` checks it
against the table made on the paper's machine; CI runs that on Linux and Windows.
`--case P009L` does the same for a cohort knee. **E9's timings are only paper numbers
from a quiet machine**: its summary says so when the CPU was busy around the run.
