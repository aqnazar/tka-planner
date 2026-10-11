"""Geometric checks on the Fusion-generated femoral bodies, beyond Fusion's health flags.

Private (patient-derived sizes). Per knee: health, width against the design width
(v15 puts the side faces at +-(Femur_ML - 1 mm)/2), distal thickness against
Distal_cut_depth (9 x ML/82), left/right symmetry, and the flange top against the
planner's flange height (not yet driven: the F / Flange_top item is open).

  python check_bodies.py author_v15 author_v15_scaled [--side-inset 0.5]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
from tka_planner.core.meshio import read_stl  # noqa: E402


def main(models, inset=0.5):
    jobs = {j["name"]: j for j in json.loads((HERE / "v57_driver/jobs.json").read_text())}
    for m in models:
        rep = json.loads((HERE / f"{m}_driver/report.json").read_text())
        print(f"\n== {m}")
        print(f"{'knee':15s} {'ML':>6s} {'width':>6s} {'want':>6s} {'d':>5s} {'distal':>6s} {'want':>5s}"
              f" {'asym':>5s} {'top':>5s} {'plan':>5s}  health")
        bad = 0
        for k in rep["knees"]:
            j, stl = jobs[k["name"]], HERE / f"{m}_driver/{k['name']}.stl"
            warn = sorted({n for n, s, _ in k.get("unhealthy", [])})
            if not k.get("exported") or not stl.exists():
                print(f"{k['name']:15s} not exported {k.get('error', '')} {warn}")
                bad += 1
                continue
            v = read_stl(stl).vertices
            ml = j["femur_ml_mm"]
            width, want = np.ptp(v[:, 1]), ml - 2 * inset
            distal, want_d = -v[:, 2].min(), 9 * ml / 82
            asym = abs(v[:, 1].max() + v[:, 1].min())
            ok = abs(width - want) < 0.05 and abs(distal - want_d) < 0.05 and asym < 0.05
            bad += (not ok) or bool(warn)
            print(f"{k['name']:15s} {ml:6.1f} {width:6.2f} {want:6.2f} {width - want:+5.2f} {distal:6.2f}"
                  f" {want_d:5.2f} {asym:5.2f} {v[:, 2].max():5.1f} {j['flange_height_mm']:5.1f}  "
                  + ("OK " if ok else "BAD ") + (", ".join(warn) if warn else "healthy"))
        print(f"{len(rep['knees']) - bad} of {len(rep['knees'])} pass (health and geometry)")


if __name__ == "__main__":
    a = sys.argv[1:]
    inset = 0.5
    if "--side-inset" in a:
        i = a.index("--side-inset")
        inset, a = float(a[i + 1]), a[:i] + a[i + 2:]
    main(a, inset)
