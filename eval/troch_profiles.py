"""Excess of the anterior profile over the shaft line, per knee (private analysis)."""
import sys
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
from eval_stage_b import cases, LIBRARY
from tka_planner.core import femoral_box as fb
from tka_planner.core.insert import surface_samples
from tka_planner.pipeline import measure_case, plan_case
for name, femur, tibia, side in cases():
    try:
        m = measure_case(femur, tibia, side, case_id=name); plan, _ = plan_case(m, library=LIBRARY)
    except Exception as e:
        print(name, "failed", type(e).__name__); continue
    pose = plan.components["femoral_component"]; R = pose[:3,:3]/np.linalg.norm(pose[:3,0])
    L = (surface_samples(m.femur, per_mm2=1.0) - pose[:3,3]) @ R
    x,y,z = L.T; d=(z>=0)&(z<=2); cy=(y[d].min()+y[d].max())/2; w=np.abs(y-cy)<=np.ptp(y[d])/2
    h = np.arange(0, 141, 1.0); A = fb._anterior_profile(x[w], z[w], h)
    for lo, hi in ((60,100),(80,130)):
        win=(h>=lo)&(h<=hi)&np.isfinite(A); s,c=np.polyfit(h[win],A[win],1); e=A-(c+s*h)
        rms=np.sqrt(np.mean(e[win]**2))
        print(f"{name} win{lo}-{hi} slope {np.degrees(np.arctan(s)):+.1f} rms {rms:.2f} excess@10..100 step10:", " ".join(f"{v:+.1f}" for v in e[10:101:10]))
