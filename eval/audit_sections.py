"""SVG: sagittal sections of the Fusion body at a given ML against the L3 body scaled by ML/82.

Where the two outlines part, a constant in the model is not following Femur_ML.
  python audit_sections.py [model] -> audit_<model>/sections.html
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
from tka_planner.core.meshio import read_stl  # noqa: E402


def section(v, f, y):
    t = v[f]; d = t[:, :, 1] - y
    m = (np.sign(d).max(1) > 0) & (np.sign(d).min(1) < 0)
    segs = []
    for tri, dd in zip(t[m], d[m]):
        pts = [tri[i] + dd[i] / (dd[i] - dd[j]) * (tri[j] - tri[i])
               for i, j in ((0, 1), (1, 2), (2, 0)) if dd[i] * dd[j] < 0]
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)[:, :, [0, 2]]


def panel(model, ml, frac, w=420, h=420):
    out = HERE / f"audit_{model}"
    s = ml / 82
    A, B = read_stl(out / f"ml{ml}.stl"), read_stl(out / "ml82.stl")
    y = frac * (ml - 1) / 2
    SA, SB = section(A.vertices, A.faces, y), section(B.vertices * s, B.faces, y)
    lo, hi = np.array([-45, -15]), np.array([50, 60])
    k = min(w, h) / (hi - lo).max()
    def path(S, col, wd):
        d = " ".join(f"M{(a[0]-lo[0])*k:.1f},{h-(a[1]-lo[1])*k:.1f}L{(b[0]-lo[0])*k:.1f},{h-(b[1]-lo[1])*k:.1f}" for a, b in S)
        return f'<path d="{d}" stroke="{col}" stroke-width="{wd}" fill="none"/>'
    grid = "".join(f'<line x1="{(x-lo[0])*k:.0f}" y1="0" x2="{(x-lo[0])*k:.0f}" y2="{h}" stroke="#eee"/>' for x in range(-40, 51, 10))
    grid += "".join(f'<line x1="0" y1="{h-(z-lo[1])*k:.0f}" x2="{w}" y2="{h-(z-lo[1])*k:.0f}" stroke="#eee"/>' for z in range(-10, 61, 10))
    return (f'<figure><svg width="{w}" height="{h}" style="border:1px solid #ccc">{grid}'
            f'{path(SB, "#1f77b4", 3)}{path(SA, "#d62728", 1.2)}</svg>'
            f'<figcaption>ML {ml}, section y = {y:.1f} mm ({frac:.0%} of half-width). '
            f'blue: L3 x {s:.3f}, red: Fusion at ML {ml}. Grid 10 mm, +X anterior, +Z proximal.</figcaption></figure>')


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "author_v18"
    mls = [int(a) for a in sys.argv[2:]] or [60, 95]
    body = "".join(f"<div>{''.join(panel(model, ml, fr) for fr in (0.0, 0.6, 0.9))}</div>" for ml in mls)
    html = (f"<html><head><style>body{{font:13px sans-serif}} div{{display:flex;gap:12px}} figure{{margin:4px}}"
            f" figcaption{{width:420px}}</style></head><body><h3>{model}: does the body scale with Femur_ML?</h3>{body}</body></html>")
    (HERE / f"audit_{model}/sections.html").write_text(html, encoding="utf-8")
    print("wrote", HERE / f"audit_{model}/sections.html")
