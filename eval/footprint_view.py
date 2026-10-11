"""Footprint of the Fusion-generated femoral bodies: a horizontal section through the
distal slab (Z = -4.5 in the CAD frame), per knee, for two models. Shows the width
against Femur_ML. Private (patient-derived sizes).

  python footprint_view.py v57 author_v12   # writes figures_src/footprint_<models>.svg
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
sys.path.insert(0, str(HERE))

from posterior_view import KNEES, STYLE  # noqa: E402
from review_stage_b import section  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402

Z0 = -4.5


def main(models):
    jobs = {j["name"]: j for j in json.loads((HERE / "v57_driver/jobs.json").read_text())}
    reps = {m: {k["name"]: k for k in json.loads((HERE / f"{m}_driver/report.json").read_text())["knees"]}
            for m in models}
    w, h, pad, scale, cols = 300, 250, 16, 2.6, 3
    rows = -(-len(KNEES) // cols)
    W, H = cols * w + pad, rows * h + 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Segoe UI, sans-serif"><rect width="{W}" height="{H}" fill="#fff"/>',
           f'<text x="{pad}" y="22" font-size="15" font-weight="600">Footprint: horizontal section '
           f'through the distal slab, Z = {Z0} (CAD frame; posterior down, anterior up)</text>']
    x = pad
    for m in models:
        out.append(f'<line x1="{x}" y1="40" x2="{x + 24}" y2="40" stroke="{STYLE[m][0]}" stroke-width="2.5"/>'
                   f'<text x="{x + 30}" y="44" font-size="12">{STYLE[m][1]}</text>')
        x += 250
    out.append(f'<line x1="{x}" y1="40" x2="{x + 24}" y2="40" stroke="#888" stroke-dasharray="4 3"/>'
               f'<text x="{x + 30}" y="44" font-size="12">±Femur_ML/2</text>')
    for i, name in enumerate(KNEES):
        j = jobs[name]
        ox, oy = pad + (i % cols) * w + w / 2, 70 + (i // cols) * h + h / 2 + 20
        g = [f'<g transform="translate({ox},{oy}) scale({scale},{-scale})">']
        half = j["femur_ml_mm"] / 2
        for s in (-1, 1):
            g.append(f'<line x1="{s * half}" y1="-35" x2="{s * half}" y2="35" stroke="#888" '
                     f'stroke-width="{0.8 / scale}" stroke-dasharray="{4 / scale} {3 / scale}"/>')
        notes = []
        for m in models:
            stl, rec = HERE / f"{m}_driver/{name}.stl", reps[m].get(name, {})
            if not stl.exists() or not rec.get("exported"):
                notes.append(f"{m}: not exported")
                continue
            mesh = read_stl(stl)
            # plane Z = Z0: swap Y and Z so section() slices it; segments come back as (X, Y)
            segs = section(mesh.vertices[:, [0, 2, 1]], mesh.faces, Z0)
            d = "".join(f"M{b:.2f} {a:.2f}L{d2:.2f} {c2:.2f}" for a, b, c2, d2 in segs)
            g.append(f'<path d="{d}" stroke="{STYLE[m][0]}" stroke-width="{1.6 / scale}" fill="none"/>')
            ys = np.array([[s[1], s[3]] for s in segs]).ravel() if segs else np.array([0.0])
            notes.append(f"{m}: {ys.min():.1f} to {ys.max():.1f} (should be ±{half:.1f})")
        g.append("</g>")
        out += g
        tx, ty = pad + (i % cols) * w + 6, 70 + (i // cols) * h + 14
        out.append(f'<text x="{tx}" y="{ty}" font-size="13" font-weight="600">{name}  '
                   f'<tspan font-weight="400">ML {j["femur_ml_mm"]:.1f}</tspan></text>')
        for k, n in enumerate(notes):
            out.append(f'<text x="{tx}" y="{ty + 15 + 13 * k}" font-size="10.5" fill="#555">{n}</text>')
    out.append("</svg>")
    dst = HERE / "figures_src"
    dst.mkdir(exist_ok=True)
    f = dst / ("footprint_" + "_".join(models) + ".svg")
    f.write_text("\n".join(out), encoding="utf-8")
    print("wrote", f)


if __name__ == "__main__":
    main(sys.argv[1:] or ["v57", "author_v12"])
