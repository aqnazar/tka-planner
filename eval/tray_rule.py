"""Stage (e) tray rule: the plate's Base Sketch parameters from the tibial cut section.

The tray sits on the shared pose: its origin is the femoral component's, so the
condyle circles are centred on Y = 0, symmetric in X about the origin (Base Sketch:
centres on the TEA line, midpoint at the origin, distance Intercondylar_Distance).
Author rulings 2026-10-09:
  - each condyle diameter is measured on its own compartment
  - no overhang anywhere: the tray stays `INSET` inside the cut outline
  - the anterior edge is the arc tangent to both circles and to the box front (as in v29)

The outline is rebuilt here the way the sketch builds it, then the largest one that fits
is searched for. Private (patient-derived).

  python tray_rule.py   -> tray_driver/tray_params.json, tray_driver/tray_rule.html
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "tray_driver"
INSET = 1.0          # mm, author 2026-10-09
GRID = 0.25          # mm raster for the containment and coverage checks
NOTCH_EDGE = 3.0     # mm of posterior edge kept beside the notch
AP_EXTRA = 4.0       # v29: Tibial_Plateau_AP = condyle diameter + 4 (Condyle = AP - 4)
DESIGN = {"ml": 78.0, "ap": 44.0, "notch_w": 18.0 * 78 / 82, "notch_d": 6.0}


def arc(c, rl, rr, ap):
    """The anterior arc of the Base Sketch: tangent (inside) to the circle at (+c, 0)
    radius rl and the one at (-c, 0) radius rr, and to the line Y = -ap/2.
    Returns (x0, y0, rho); y0 - rho = -ap/2."""
    # Subtracting the two tangency conditions gives x0 = k (rho - m); substituting into
    # the first leaves a quadratic in rho.
    k, m, h = (rl - rr) / (2 * c), (rl + rr) / 2, ap / 2
    A = k * k
    B = -2 * k * (k * m + c) + 2 * (rl - h)
    C = (k * m + c) ** 2 + h * h - rl * rl
    if abs(A) < 1e-12:
        roots = [-C / B] if abs(B) > 1e-12 else []
    else:
        disc = B * B - 4 * A * C
        roots = [] if disc < 0 else [(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)]
    roots = sorted(r for r in roots if r > max(rl, rr))
    if not roots:
        return None
    rho = roots[0]
    return k * (rho - m), rho - h, rho


def notch_edge(c, rl, rr, notch_w):
    """Length of posterior edge left beside the notch, on the shorter side."""
    nx = -(rl - rr) / (2 * c)
    return min(c + rl * nx - notch_w / 2, c - rr * nx - notch_w / 2)


def tray_mask(p, gx, gy):
    """Raster of the tray outline for parameters p (c, rl, rr, ap, notch_w, notch_d)."""
    c, rl, rr, ap = p["c"], p["rl"], p["rr"], p["ap"]
    inside = ((gx - c) ** 2 + gy ** 2 <= rl ** 2) | ((gx + c) ** 2 + gy ** 2 <= rr ** 2)
    a = arc(c, rl, rr, ap)
    # Posterior common tangent of the two circles (the line C9), on the +Y side.
    ang = np.arcsin(np.clip((rl - rr) / (2 * c), -1, 1))
    n = np.array([np.sin(-ang), np.cos(ang)])            # outward normal, +Y-ish
    # Line: n . (x, y) = d, through the tangent point of the +X circle.
    d = np.dot(n, [c, 0]) + rl
    band = (np.abs(gx) <= c + 1e-9) & (n[0] * gx + n[1] * gy <= d)
    if a is not None:
        x0, y0, rho = a
        band &= (gx - x0) ** 2 + (gy - y0) ** 2 <= rho ** 2
    inside |= band
    # The PCL notch: a slot of the given width, from the posterior line `notch_d` deep.
    notch = (np.abs(gx) <= p["notch_w"] / 2) & (n[0] * gx + n[1] * gy >= d - p["notch_d"])
    return inside & ~notch


def polygon_mask(poly, gx, gy):
    x, y = poly[:, 0], poly[:, 1]
    inside = np.zeros(gx.shape, bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cross = ((y[i] > gy) != (y[j] > gy)) & (gx < (x[j] - x[i]) * (gy - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        inside ^= cross
        j = i
    return inside


def dist_to_boundary(poly, q):
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return np.sqrt(((a + t[:, None] * ab - q) ** 2).sum(1)).min()


def erode(mask, r_cells):
    """Binary erosion with a disc, by shifted ANDs (numpy only)."""
    out = mask.copy()
    r = int(np.ceil(r_cells))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r_cells * r_cells:
                out &= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def fit(row, shift=(0.0, 0.0)):
    """`shift`: the plate outline's centre relative to the shared origin (0, 0 = v29)."""
    poly = np.array(row["outline_mm"]) - np.asarray(shift)
    lo, hi = poly.min(0) - 3, poly.max(0) + 3
    xs, ys = np.arange(lo[0], hi[0], GRID), np.arange(lo[1], hi[1], GRID)
    gx, gy = np.meshgrid(xs, ys)
    bone = polygon_mask(poly, gx, gy)
    allowed = erode(bone, INSET / GRID)
    t = row["tibial"]
    # PCL notch: the bone's own notch plus the inset, never smaller than the design's.
    notch_w = max(DESIGN["notch_w"] * row["femur_ml_mm"] / 82, t["pcl_notch_width"] + 2 * INSET)
    notch_d = max(DESIGN["notch_d"], t["pcl_notch_depth"] + INSET)
    best = None
    for c in np.arange(8.0, 32.0, 0.25):
        rl = dist_to_boundary(poly, np.array([c, 0.0])) - INSET
        rr = dist_to_boundary(poly, np.array([-c, 0.0])) - INSET
        if not (polygon_mask(poly, np.array([[c]]), np.array([[0.0]]))[0, 0]
                and polygon_mask(poly, np.array([[-c]]), np.array([[0.0]]))[0, 0]):
            continue
        if rl < 10 or rr < 10:
            continue
        # Keep a stretch of posterior edge beside the PCL notch (the notch-corner fillet
        # needs it; with none the notch cuts into the condyle circles). The edge is the
        # common tangent, slanted when the circles differ: its ends are the tangent points.
        if notch_edge(c, rl, rr, notch_w) < NOTCH_EDGE:
            continue
        for extra in (AP_EXTRA, 3.0, 2.0, 1.0, 0.5):
            ap = 2 * max(rl, rr) + extra
            p = {"c": c, "rl": rl, "rr": rr, "ap": ap, "notch_w": notch_w, "notch_d": notch_d}
            if arc(c, rl, rr, ap) is None:
                continue
            m = tray_mask(p, gx, gy)
            if (m & ~allowed).sum() * GRID ** 2 > 0.5:     # overhang tolerance 0.5 mm^2 (raster)
                continue
            cover = (m & bone).sum() / bone.sum()
            if best is None or cover > best[0]:
                best = (cover, p)
            break
    cover, p = best
    m = tray_mask(p, gx, gy)
    side_plus = "medial" if row["side"] == "right" else "lateral"   # +X is patient-left
    params = {
        "Tibial_Plateau_ML": round(2 * p["c"] + p["rl"] + p["rr"], 3),
        "Tibial_Plateau_AP": round(p["ap"], 3),
        "Left_Condyle_Diameter": round(2 * p["rl"], 3),
        "Right_Condyle_Diameter": round(2 * p["rr"], 3),
        "Intercondylar_Notch_Width": round(p["notch_w"], 3),
        "Internal_Notch_Offset": round(p["notch_d"], 3),
        # Notch-corner fillet: the design's 8 x AP/44, capped by the edge beside the notch.
        "d11": round(min(8 * p["ap"] / 44, notch_edge(p["c"], p["rl"], p["rr"], p["notch_w"]) - 1.0), 3),
    }
    return {
        "name": row["name"], "side": row["side"], "params": params, "c": p["c"],
        "shift_mm": [round(float(v), 2) for v in shift],
        "coverage": round(float(cover), 4),
        "bone_area_mm2": round(float(bone.sum() * GRID ** 2), 1),
        "tray_area_mm2": round(float(m.sum() * GRID ** 2), 1),
        "plus_x_is": side_plus,
        "grid": [float(lo[0]), float(lo[1]), GRID, list(gx.shape)],
        "_mask": m, "_bone": bone, "_allowed": allowed, "_xs": xs, "_ys": ys, "_poly": poly,
    }


def svg(res, w=300):
    xs, ys, poly = res["_xs"], res["_ys"], res["_poly"]
    span = 100.0
    k = w / span
    def X(x): return (x + span / 2) * k
    def Y(y): return (y + span / 2) * k            # +Y posterior drawn downwards
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in poly)
    m = res["_mask"]
    cells = []
    step = 2
    for j in range(0, m.shape[0], step):
        row = m[j]
        i = 0
        while i < len(row):
            if row[i]:
                s = i
                while i < len(row) and row[i]:
                    i += 1
                cells.append(f"M{X(xs[s]):.1f},{Y(ys[j]):.1f}h{(i - s) * GRID * k:.1f}v{step * GRID * k:.1f}h{-(i - s) * GRID * k:.1f}z")
            i += 1
    P = res["params"]
    cap = (f"{res['name']} ({res['side']}): ML {P['Tibial_Plateau_ML']:.1f}, AP {P['Tibial_Plateau_AP']:.1f}, "
           f"condyles +X {P['Left_Condyle_Diameter']:.1f} / -X {P['Right_Condyle_Diameter']:.1f} "
           f"(+X = {res['plus_x_is']}); coverage {res['coverage']:.0%}")
    return (f'<figure><svg width="{w}" height="{w}" style="border:1px solid #ccc;background:#fff">'
            f'<polygon points="{pts}" fill="#eee" stroke="#555" stroke-width="1"/>'
            f'<path d="{" ".join(cells)}" fill="#1f77b4" fill-opacity="0.55"/>'
            f'<line x1="{X(-4)}" y1="{Y(0)}" x2="{X(4)}" y2="{Y(0)}" stroke="#d62728"/>'
            f'<line x1="{X(0)}" y1="{Y(-4)}" x2="{X(0)}" y2="{Y(4)}" stroke="#d62728"/>'
            f'</svg><figcaption>{cap}</figcaption></figure>')


def main(variant=""):
    rows = [r for r in json.loads((OUT / "inputs.json").read_text()) if r.get("femur_ml_mm")]
    if variant == "shifted":
        # The outline centred on the section's bounding-box centre instead of the origin.
        results = [fit(r, r["section_centre_mm"]) for r in rows]
    else:
        results = [fit(r) for r in rows]
    print(f"{'knee':8s} {'side':5s} {'ML':>6s} {'AP':>6s} {'cond+X':>7s} {'cond-X':>7s} {'notch w/d':>10s} "
          f"{'cover':>6s}  bone ML/AP")
    for r, row in zip(results, rows):
        P = r["params"]
        t = row["tibial"]
        print(f"{r['name']:8s} {r['side']:5s} {P['Tibial_Plateau_ML']:6.1f} {P['Tibial_Plateau_AP']:6.1f} "
              f"{P['Left_Condyle_Diameter']:7.1f} {P['Right_Condyle_Diameter']:7.1f} "
              f"{P['Intercondylar_Notch_Width']:5.1f}/{P['Internal_Notch_Offset']:4.1f} {r['coverage']:6.1%}  "
              f"{t['ml']:.1f}/{t['ap']:.1f}  section centre {np.round(row['section_centre_mm'], 1).tolist()}")
    (OUT / f"tray_params{('_' + variant) if variant else ''}.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in results], indent=1), encoding="utf-8")
    html = ("<html><head><style>body{font:12px sans-serif} .g{display:flex;flex-wrap:wrap;gap:8px}"
            " figure{margin:2px;width:300px}</style></head><body>"
            "<h3>Stage (e) tray rule on the cut sections (tray frame, view from above; +X patient-left, "
            "+Y posterior drawn down; red cross = shared origin)</h3>"
            "<p>Grey: tibial cut section. Blue: the v29 outline at the rule's parameters, centred on the "
            "shared pose, 1 mm inside the bone everywhere.</p><div class='g'>"
            + "".join(svg(r) for r in results) + "</div></body></html>")
    name = f"tray_rule{('_' + variant) if variant else ''}.html"
    (OUT / name).write_text(html, encoding="utf-8")
    print("wrote", OUT / name)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")
