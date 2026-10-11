# eval: the parametric demo and its evaluation scripts

The planner with the implant built from the plan. The demo app is the planner's own
server (live plan controls, Commit, trial reduction, the flexion arc) plus a parametric
step: derive the femoral, plate and insert parameters from the current plan, and build
the parts in Fusion from them.

> This folder is generated. The work happens in a private folder and is copied here by
> `tools/publish_eval.py`, which mirrors it: edits made directly in `eval/` are
> overwritten at the next publish.

## Run the demo

```
cd eval
python demo_server.py --port 8770
```

Then open http://127.0.0.1:8770. It opens on two synthetic knees, `DEMO_SMALL` and
`DEMO_LARGE`, each with its bones and the parts already built for it.

What runs with only this repository:

- the knees, their built parts, the fit checks and the J-curves
- the live plan controls, Commit and the resection
- trial reduction and the flexion arc on the built parts (`kinematics.py`)

What needs more:

| Step | Needs |
|---|---|
| Implant: derive the parameters | An implant library with its `femoral_box.json`. Point `TKA_IMPLANT_LIBRARY` at it, or put it at `data/implants/`, as for the launcher. |
| Build | Fusion, its MCP add-in listening on `127.0.0.1:27182`, and the parametric Fusion documents the build copies (`tray_audit/copies.json` names them). |

## What is here

| Path | Contents |
|---|---|
| `demo_server.py`, `demo_app/` | The demo server and its web app. three.js and IBM Plex vendored, with their licences. |
| `demo_cases.py` | The demo knees: scaling the bones, planning them, deriving the parameters. |
| `demo_cases/<knee>/` | Each demo knee's femur (`FD.stl`), tibia (`TD.stl`) and the scale that made it. |
| `patient_builds/<knee>/` | Each demo knee's job, built parts (STL), build report, fit and J-curves. |
| `kinematics.py`, `jcurve.py`, `fit_check.py` | Flexion on the built parts, the J-curves, the parts' fit on the cut bone. |
| `implant_pipeline.py`, `fusion_driver.py`, `*_edit.py`, `*_block.py`, `insert_*.py` | The Fusion build: parameters into the parametric parts, and the cutting blocks. |
| `eval_stage_b.py`, `review_stage_b.py`, `tray_rule.py`, `troch_*.py` | The femoral cut box and tray rule evaluations behind the paper. |
| `audit_*.py`, `model_audit.py`, `fusion_diff.py`, `*_figure.py`, `*_view.py` | Audits of the Fusion models, and figures. |

Paths inside the JSON files are relative to `eval/`.

## What is not

The cohort knees, the reference knee's own scans, the implant library and the Fusion
documents stay private. The cohort scripts refer to knees by case id only and read
them from a local `Patient Dataset/` that is never committed.

The demo knees are the reference knee scaled independently along its three axes
(`demo_cases/<knee>/scale.json`), so they are anatomically plausible extremes, not
patient scans.
