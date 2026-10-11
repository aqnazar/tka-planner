"""Candidate trochlea-top rules on the cohort and on noisy synthetic profiles (private)."""
import sys
from pathlib import Path
import numpy as np
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).parent))
from eval_stage_b import cases, LIBRARY
from tka_planner.core import femoral_box as fb
from tka_planner.core.insert import surface_samples
from tka_planner.pipeline import measure_case, plan_case
T = np.tan(np.radians(10.11))
def top(h, e, frac, floor=0.25, run=5, lo=60):
    below = h < lo
    peak = int(np.nanargmax(np.where(below, e, -np.inf)))
    thr = max(floor, frac * e[peak])
    fin = np.flatnonzero(np.isfinite(e) & (h >= h[peak]))
    for k, i in enumerate(fin):
        nxt = fin[k:k + run]
        if len(nxt) == run and np.all(e[nxt] <= thr):
            return h[i]
    return np.nan
def old(h, e, tol=0.25, lo=60):
    i = int(np.flatnonzero(h >= lo)[0])
    while i > 0 and np.isfinite(e[i-1]) and e[i-1] <= tol: i -= 1
    return h[i]
rows = []
for name, femur, tibia, side in cases():
    try:
        m = measure_case(femur, tibia, side, case_id=name); plan, _ = plan_case(m, library=LIBRARY)
    except Exception: continue
    pose = plan.components["femoral_component"]; R = pose[:3,:3]/np.linalg.norm(pose[:3,0])
    L = (surface_samples(m.femur, per_mm2=1.0) - pose[:3,3]) @ R
    x,y,z = L.T; d=(z>=0)&(z<=2); cy=(y[d].min()+y[d].max())/2; w=np.abs(y-cy)<=np.ptp(y[d])/2
    h = np.arange(0, 101, 1.0); A = fb._anterior_profile(x[w], z[w], h)
    win=(h>=60)&np.isfinite(A); s,c=np.polyfit(h[win],A[win],1); e=A-(c+s*h)
    r = [old(h,e)] + [top(h,e,f) for f in (0.05,0.1,0.2)]
    # anterior resection for the 10% rule
    zt = r[2]; i=int(zt); plane = A[i] + (h - zt)*T; ant = np.nanmax((A-plane)[h<=zt])
    print(f"{name:8s} peak {np.nanmax(e[h<60]):4.1f}  old {r[0]:3.0f}  5% {r[1]:3.0f}  10% {r[2]:3.0f}  20% {r[3]:3.0f}   10%: anterior {ant:4.1f} PF {ant-5:+.1f}")
    # noise robustness
    rng = np.random.default_rng(0); outs=[]
    for sd in (0.1, 0.15, 0.3):
        vals=[top(h, e + rng.normal(0, sd, len(e)), 0.1) for _ in range(20)]
        outs.append(f"sd{sd}: {np.nanmin(vals):.0f}-{np.nanmax(vals):.0f}")
    print("          noise (10% rule):", "  ".join(outs))
