"""The natural J-curve and the component's, per condyle, for the demo page.

In the femoral component's CAD frame (+X anterior, +Y patient-left, +Z proximal, origin
on the distal cut, as placed by the plan): the sagittal section of the bone and of the
built component through each condyle's centre, the planned cut box in that plane, and
how much each cut takes off against how much the component puts back (distal, posterior,
anterior). Private (patient-derived).

  python jcurve.py [knee ...]   -> patient_builds/<knee>/jcurve.json
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from fit_check import femoral_planes_cad, plane_pose, to_frame  # noqa: E402
from tka_planner.core.implant_spec import section_mask  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402

BUILDS = HERE / "patient_builds"
WINDOW = (-45.0, 50.0, -20.0, 70.0)     # x0, x1, z0, z1 of the sagittal view (mm)


def loops(mask, xs, ys):
    """Boundary of a filled section as polylines (marching squares, cell-centre grid)."""
    m = np.pad(mask, 1).astype(np.uint8)
    xs = np.r_[xs[0] - (xs[1] - xs[0]), xs, xs[-1] + (xs[1] - xs[0])]
    ys = np.r_[ys[0] - (ys[1] - ys[0]), ys, ys[-1] + (ys[1] - ys[0])]
    segs = []
    # Edges between an inside cell and an outside neighbour, at the half-way line.
    hx, hy = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2
    a = m[:, 1:] != m[:, :-1]                    # vertical edges between columns
    for j, i in zip(*np.nonzero(a)):
        x = xs[i] + hx
        segs.append(((x, ys[j] - hy), (x, ys[j] + hy)))
    b = m[1:, :] != m[:-1, :]                    # horizontal edges between rows
    for j, i in zip(*np.nonzero(b)):
        y = ys[j] + hy
        segs.append(((xs[i] - hx, y), (xs[i] + hx, y)))
    # Chain the unit edges into loops.
    key = lambda p: (round(p[0], 4), round(p[1], 4))
    adj = {}
    for k, (p, q) in enumerate(segs):
        adj.setdefault(key(p), []).append(k)
        adj.setdefault(key(q), []).append(k)
    used = np.zeros(len(segs), bool)
    out = []
    for k0 in range(len(segs)):
        if used[k0]:
            continue
        used[k0] = True
        p, q = segs[k0]
        line = [p, q]
        while True:
            nxt = [k for k in adj[key(line[-1])] if not used[k]]
            if not nxt:
                break
            k = nxt[0]; used[k] = True
            p, q = segs[k]
            line.append(q if key(p) == key(line[-1]) else p)
        if len(line) > 8:
            out.append(np.array(line))
    return out


def smooth(line, n=3):
    """Corner-cut the staircase (Chaikin), keep it closed."""
    pts = line
    for _ in range(n):
        q = 0.75 * pts + 0.25 * np.roll(pts, -1, 0)
        r = 0.25 * pts + 0.75 * np.roll(pts, -1, 0)
        pts = np.empty((2 * len(pts), 2)); pts[0::2], pts[1::2] = q, r
    return pts[:: 2 ** (n - 1)]


def sagittal(mesh, y):
    """Section by the plane y = const, as (x, z) loops in the CAD frame."""
    # Plane frame: normal +Y; local x = CAD X? plane_pose gives x = y_dir x z.
    P = plane_pose(np.array([0.0, y, 0.0]), [0, 1, 0], y=(0, 0, 1))   # local y = CAD Z
    m, xs, ys = section_mask(mesh, P, offset_mm=0.0, grid_mm=0.25)
    out = []
    for line in loops(m, xs, ys):
        loc = smooth(line)
        world = P[:3, 3] + loc[:, :1] * P[:3, 0] + loc[:, 1:] * P[:3, 1]
        out.append(world[:, [0, 2]])               # (x, z)
    return out, (m, xs, ys, P)


def box_polygon(planes, z_top):
    """The cut box in the sagittal plane: a big rectangle clipped by the five kept sides."""
    poly = np.array([[-80, -10], [80, -10], [80, z_top], [-80, z_top]], float)
    for p, n in planes.values():
        p = np.array([p[0], p[2]], float); n = np.array([n[0], n[2]], float)
        keep = (poly - p) @ n >= 0
        out = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            ka, kb = keep[i], keep[(i + 1) % len(poly)]
            if ka:
                out.append(a)
            if ka != kb:
                t = ((p - a) @ n) / ((b - a) @ n)
                out.append(a + t * (b - a))
        poly = np.array(out)
    return poly


def extreme(loops_, axis, sign, band):
    """Furthest point of the loops along axis (0 = x, 1 = z) in direction sign, among
    points whose other coordinate lies in band."""
    best = None
    for L in loops_:
        other = L[:, 1 - axis]
        sel = L[(other >= band[0]) & (other <= band[1])]
        if len(sel):
            v = (sel[:, axis] * sign).max() * sign
            best = v if best is None else (max(best, v) if sign > 0 else min(best, v))
    return best


def circle(pts):
    """Least-squares circle through (x, z) points (Kasa): centre and radius."""
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(1)
    (cx, cz, k), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cz, float(np.sqrt(k + cx ** 2 + cz ** 2))


def radii(loops_, band=4.0):
    """The J-curve as two arcs, fitted on the outer outline: the distal arc on the points
    within `band` of its lowest point (under the distal cut, |x| < 20), the posterior arc
    on those within `band` of its most posterior point (0 < z < 40). The posterior arc's
    centre is the condyle's flexion facet centre. CAD mm."""
    if not loops_:
        return None
    L = max(loops_, key=lambda l: np.ptp(l[:, 0]) * np.ptp(l[:, 1]))
    out = {}
    d = L[np.abs(L[:, 0]) < 20]
    p = L[(L[:, 1] > 0) & (L[:, 1] < 40)]
    for key, sel in (("distal", d[d[:, 1] < d[:, 1].min() + band] if len(d) else d),
                     ("posterior", p[p[:, 0] < p[:, 0].min() + band] if len(p) else p)):
        if len(sel) >= 6:
            cx, cz, r = circle(sel)
            out[key] = {"radius_mm": round(r, 2), "centre": [round(float(cx), 2), round(float(cz), 2)]}
    return out


def run(job, femur=None, comp=None, write=True):
    """`femur`: the femur mesh in the scan frame (read from the job's bones if None);
    `comp`: the built component in its CAD frame (the knee's femoral.stl if None and
    it exists). With no component only the natural curves and the box are given."""
    name = job["name"]
    F = np.array(job["plan"]["femoral_pose"])
    femur = to_frame(femur if femur is not None else read_stl(job["bones"][0]), F)
    if comp is None and (BUILDS / name / "femoral.stl").exists():
        comp = read_stl(BUILDS / name / "femoral.stl")
    fp = {k: float(v.split()[0]) for k, v in job["femoral"].items()}
    C, B, D = fp["C_Distal_cut_surface"], fp["B_Posterior_chamfer_projection"], fp["D_Anterior_chamfer_projection"]
    planes = femoral_planes_cad(C, B, D)
    # Condyle centres: the two halves of the bone's distal cut surface.
    m, xs, ys = section_mask(femur, plane_pose(np.zeros(3), [0, 0, 1]), offset_mm=0.0)
    GX, GY = np.meshgrid(xs, ys)
    keep = m & (GX > -C / 2) & (GX < C / 2)
    side = job["side"]
    out = {"name": name, "side": side, "frame": "femoral CAD: +X anterior, +Z proximal (mm)",
           "window": WINDOW, "params": {"C_Distal_cut_surface": C, "B_Posterior_chamfer_projection": B,
                                       "D_Anterior_chamfer_projection": D, "Femur_ML": fp["Femur_ML"],
                                       "distal_thickness": round(9 * fp["Femur_ML"] / 82, 3)},
           "box": box_polygon(planes, job["femoral_box"]["flange_height_mm"] + 5).round(2).tolist(),
           "condyles": {}}
    for label, sel in (("+Y", GY > 0), ("-Y", GY < 0)):
        y = float(GY[keep & sel].mean())
        # +Y is patient-left: the medial side of a right knee.
        anat = ("medial" if label == "+Y" else "lateral") if side == "right" else ("lateral" if label == "+Y" else "medial")
        bl, _ = sagittal(femur, y)
        cl = sagittal(comp, y)[0] if comp is not None else []
        x0, x1, z0, z1 = WINDOW
        clip = lambda Ls: [L.round(2).tolist() for L in Ls
                           if ((L[:, 0] > x0) & (L[:, 0] < x1) & (L[:, 1] > z0) & (L[:, 1] < z1)).any()]
        # Restoration: bone taken off by each cut vs metal put back, along the cut's normal.
        cuts = {}
        mid = (-C / 2, C / 2)
        bone_d = extreme(bl, 1, -1, mid); comp_d = extreme(cl, 1, -1, mid)
        bone_p = extreme(bl, 0, -1, (0, 25)); comp_p = extreme(cl, 0, -1, (0, 25))
        post_cut = -C / 2 - B
        if bone_d is not None and comp_d is not None:
            cuts["distal"] = {"bone_removed_mm": round(-bone_d, 2), "component_mm": round(-comp_d, 2),
                              "joint_line_change_mm": round(bone_d - comp_d, 2)}
        if bone_p is not None and comp_p is not None:
            cuts["posterior"] = {"bone_removed_mm": round(post_cut - bone_p, 2),
                                 "component_mm": round(post_cut - comp_p, 2),
                                 "offset_change_mm": round(bone_p - comp_p, 2)}
        out["condyles"][anat] = {"y_mm": round(y, 2), "bone": clip(bl), "component": clip(cl), "cuts": cuts,
                                 "arcs": {"bone": radii(bl), "component": radii(cl)}}
    if write:
        (BUILDS / name / "jcurve.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def main(names):
    for job in json.loads((BUILDS / "jobs_demo.json").read_text()):
        if names and job["name"] not in names:
            continue
        r = run(job)
        print(job["name"], {k: (v["y_mm"], v["cuts"]) for k, v in r["condyles"].items()})


if __name__ == "__main__":
    main(sys.argv[1:])
