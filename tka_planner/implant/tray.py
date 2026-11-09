"""The tray rule: the plate's Base Sketch parameters from the tibial cut section.

The tray sits on the shared pose: its origin is the femoral component's, so the two
condyle circles are centred on Y = 0, symmetric in X about the origin (Base Sketch:
centres on the TEA line, midpoint at the origin). Author rulings 2026-10-09: each
condyle diameter is measured on its own compartment; no overhang anywhere (the tray
stays an inset inside the cut outline); the anterior edge is the arc tangent to both
circles and to the box front. The outline is rebuilt the way the sketch builds it and
the largest that fits is searched for (from ``eval/tray_rule.py``, unchanged).
"""

from __future__ import annotations

import numpy as np

from .design import (
    TRAY_AP_EXTRA_MM,
    TRAY_DESIGN,
    TRAY_DESIGN_FEMUR_ML_MM,
    TRAY_FILLET_MM,
    TRAY_GRID_MM,
    TRAY_INSET_MM,
    TRAY_NOTCH_EDGE_MM,
    TRAY_OVERHANG_TOLERANCE_MM2,
)

__all__ = ["fit", "arc", "notch_edge", "tray_mask", "polygon_mask"]

INSET = TRAY_INSET_MM
GRID = TRAY_GRID_MM


def arc(c, rl, rr, ap):
    """The anterior arc of the Base Sketch: tangent (inside) to the circle at (+c, 0)
    radius rl and the one at (-c, 0) radius rr, and to the line Y = -ap/2.
    Returns (x0, y0, rho); y0 - rho = -ap/2, or None."""
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
    # The PCL notch: a slot of the given width, from the posterior line notch_d deep.
    notch = (np.abs(gx) <= p["notch_w"] / 2) & (n[0] * gx + n[1] * gy >= d - p["notch_d"])
    return inside & ~notch


def polygon_mask(poly, gx, gy):
    """Even-odd point-in-polygon on a grid."""
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


def fit(row, shift=(0.0, 0.0)) -> dict:
    """The largest tray outline 1 mm inside the cut. ``row``: name, side, femur_ml_mm,
    outline_mm (the cut section, tray frame) and tibial (pcl_notch_width, _depth).
    ``shift``: the plate outline's centre relative to the shared origin."""
    poly = np.array(row["outline_mm"]) - np.asarray(shift)
    lo, hi = poly.min(0) - 3, poly.max(0) + 3
    xs, ys = np.arange(lo[0], hi[0], GRID), np.arange(lo[1], hi[1], GRID)
    gx, gy = np.meshgrid(xs, ys)
    bone = polygon_mask(poly, gx, gy)
    allowed = erode(bone, INSET / GRID)
    t = row["tibial"]
    # PCL notch: the bone's own notch plus the inset, never smaller than the design's.
    notch_w = max(TRAY_DESIGN["notch_w"] * row["femur_ml_mm"] / TRAY_DESIGN_FEMUR_ML_MM,
                  t["pcl_notch_width"] + 2 * INSET)
    notch_d = max(TRAY_DESIGN["notch_d"], t["pcl_notch_depth"] + INSET)
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
        # needs it; with none the notch cuts into the condyle circles).
        if notch_edge(c, rl, rr, notch_w) < TRAY_NOTCH_EDGE_MM:
            continue
        for extra in (TRAY_AP_EXTRA_MM, 3.0, 2.0, 1.0, 0.5):
            ap = 2 * max(rl, rr) + extra
            p = {"c": c, "rl": rl, "rr": rr, "ap": ap, "notch_w": notch_w, "notch_d": notch_d}
            if arc(c, rl, rr, ap) is None:
                continue
            m = tray_mask(p, gx, gy)
            if (m & ~allowed).sum() * GRID ** 2 > TRAY_OVERHANG_TOLERANCE_MM2:
                continue
            cover = (m & bone).sum() / bone.sum()
            if best is None or cover > best[0]:
                best = (cover, p)
            break
    if best is None:
        raise ValueError("No tray outline fits inside the tibial cut.")
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
        "d11": round(min(TRAY_FILLET_MM * p["ap"] / TRAY_DESIGN["ap"],
                         notch_edge(p["c"], p["rl"], p["rr"], p["notch_w"]) - 1.0), 3),
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
