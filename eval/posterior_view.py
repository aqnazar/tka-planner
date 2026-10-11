"""Posterior view of the Fusion-generated femoral bodies: v57 against the author's model.

Private (patient-derived sizes). A coronal section through the posterior condyles, 1.5 mm
behind the inner posterior face, in the CAD frame (Y across, Z up), per knee. The top of
each condyle outline is what the Posterior profile cut shapes.

  python posterior_view.py [model ...]   # default: v57 author_v7; writes figures_src/posterior_view.svg
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
sys.path.insert(0, str(HERE))

from review_stage_b import section  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402

KNEES = ["REF_KJ2_design", "P007L", "P029R", "P009L", "P037R", "P047R"]
STYLE = {"v57": ("#7b3fb5", "v57 working copy"), "author_v7": ("#e07b00", "author's model (sandbox v7)"),
         "author_v12": ("#0f8a6a", "author's model (sandbox v12)"),
         "author_v15": ("#c2185b", "author's model (sandbox v15)"),
         "author_v15_scaled": ("#0f8a6a", "sandbox v15 + constants scaled")}


def coronal(mesh, x0):
    """Segments where the plane X = x0 cuts the mesh, as [[y1, z1, y2, z2], ...]."""
    return section(mesh.vertices[:, [1, 0, 2]], mesh.faces, x0)


def main(models, out_name="posterior_view.svg", distal=False):
    jobs = {j["name"]: j for j in json.loads((HERE / "v57_driver/jobs.json").read_text())}
    reports = {}
    for m in models:
        rep = json.loads((HERE / f"{m}_driver/report.json").read_text())
        reports[m] = {k["name"]: k for k in rep["knees"]}
    w, h, pad = 300, 210, 16
    cols = 3
    rows = -(-len(KNEES) // cols)
    W, H = cols * w + pad, rows * h + 70
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Segoe UI, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#fff"/>',
           (f'<text x="{pad}" y="22" font-size="15" font-weight="600">Coronal section through the middle '
            f'of the distal face, X = 0 (CAD frame, looking from behind)</text>' if distal else
            f'<text x="{pad}" y="22" font-size="15" font-weight="600">Posterior view: coronal '
            f'section 1.5 mm behind the inner posterior face (CAD frame, looking from behind)</text>')]
    x = pad
    for m in models:
        c, label = STYLE[m]
        out.append(f'<line x1="{x}" y1="40" x2="{x + 24}" y2="40" stroke="{c}" stroke-width="2.5"/>'
                   f'<text x="{x + 30}" y="44" font-size="12">{label}</text>')
        x += 250
    out.append(f'<line x1="{x}" y1="40" x2="{x + 24}" y2="40" stroke="#888" stroke-dasharray="4 3"/>'
               f'<text x="{x + 30}" y="44" font-size="12">±ML/2 (the body should reach it)</text>')
    scale = 2.6                                   # px per mm
    for i, name in enumerate(KNEES):
        j = jobs[name]
        p = j["parameters_mm"]
        x0 = 0.0 if distal else -(p["C_Distal_cut_surface"] / 2 + p["B_Posterior_chamfer_projection"]) - 1.5
        ox, oy = pad + (i % cols) * w + w / 2, 70 + (i // cols) * h + h - 40
        g = [f'<g transform="translate({ox},{oy}) scale({scale},{-scale})">']
        half = j["femur_ml_mm"] / 2
        for s in (-1, 1):
            g.append(f'<line x1="{s * half}" y1="-3" x2="{s * half}" y2="40" stroke="#888" '
                     f'stroke-width="{0.8 / scale}" stroke-dasharray="{4 / scale} {3 / scale}"/>')
        g.append(f'<line x1="-55" y1="0" x2="55" y2="0" stroke="#ddd" stroke-width="{1 / scale}"/>')
        notes = []
        for m in models:
            stl = HERE / f"{m}_driver/{name}.stl"
            rec = reports[m].get(name, {})
            if not stl.exists() or not rec.get("exported"):
                notes.append(f"{m}: not exported")
                continue
            segs = coronal(read_stl(stl), x0)
            d = "".join(f"M{a:.2f} {b:.2f}L{c2:.2f} {d2:.2f}" for a, b, c2, d2 in segs)
            g.append(f'<path d="{d}" stroke="{STYLE[m][0]}" stroke-width="{1.6 / scale}" fill="none"/>')
            warn = sorted({n for n, s, _ in rec.get("unhealthy", [])})
            y = read_stl(stl).vertices[:, 1]
            notes.append(f"{m}: width {np.ptp(y):.1f} (ML {j['femur_ml_mm']:.1f}); "
                         + (", ".join(warn) if warn else "healthy"))
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
    (dst / out_name).write_text("\n".join(out), encoding="utf-8")
    print("wrote", dst / out_name)


if __name__ == "__main__":
    args = sys.argv[1:]
    distal = "--distal" in args
    models = [a for a in args if a != "--distal"] or ["v57", "author_v7"]
    main(models, ("distal_view_" if distal else "posterior_view_") + "_".join(models) + ".svg", distal)
