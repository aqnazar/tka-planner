"""Check a knee's femoral cutting block against its femoral (private: patient-derived).

  - the block's section at several ML levels against the femoral's J-curve cut lines
    (from build.json), continued to Block_Top
  - the femoral's inner (bone-facing) surface against the block: how far the component's
    mating faces lie from the block's faces
  - the bone cut with the block (manifold3d), and what bone is left outside the box

  python block_check.py [knee ...]   -> patient_builds/<knee>/block_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from fit_check import to_frame  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402

BUILDS = HERE / "patient_builds"
TOP = 150.0


def section(mesh, y):
    """Segments (x0, z0, x1, z1) of the mesh cut by the plane Y = y."""
    v = mesh.vertices[mesh.faces]                  # (n, 3, 3)
    d = v[:, :, 1] - y
    segs = []
    for tri, dd in zip(v, d):
        pts = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            da, db = dd[i], dd[(i + 1) % 3]
            if (da < 0) != (db < 0):
                t = da / (da - db)
                p = a + t * (b - a)
                pts.append((p[0], p[2]))
        if len(pts) == 2:
            segs.append((*pts[0], *pts[1]))
    return np.array(segs)


def polyline(lines):
    """The cut lines as one polyline, the posterior and anterior continued to TOP."""
    P, A = lines["posterior"], lines["anterior"]
    pt = (P[0], max(P[1], P[3]))
    a0, a1 = sorted([(A[0], A[1]), (A[2], A[3])], key=lambda p: p[1])
    xa = a1[0] + (a1[0] - a0[0]) * (TOP - a1[1]) / (a1[1] - a0[1])
    pc, ac, ds = lines["posterior chamfer"], lines["anterior chamfer"], lines["distal"]
    pts = [(pt[0], TOP), pt, (pc[2], pc[3]) if pc[3] > pc[1] else (pc[0], pc[1]),
           (min(ds[0], ds[2]), 0.0), (max(ds[0], ds[2]), 0.0), a0, a1, (xa, TOP)]
    return np.array(pts)


def dist_to_polyline(p, poly):
    best = np.full(len(p), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def outside_box(p, lines):
    """How far any point lies on the cut-away side of the box (0 when all inside)."""
    poly = polyline(lines)
    worst = 0.0
    # Each cut line's outward normal points away from the box centre (0, 0, 20).
    for a, b in zip(poly[:-1], poly[1:]):
        t = b - a
        n = np.array([t[1], -t[0]]) / np.linalg.norm(t)
        if (np.array([0.0, 20.0]) - a) @ n > 0:
            n = -n
        # Only points beside this line's span count against it.
        u = ((p - a) @ t) / (t @ t)
        m = (u > 0.02) & (u < 0.98)
        if m.any():
            worst = max(worst, float(((p[m] - a) @ n).max()))
    return worst


def check(name):
    d = BUILDS / name
    rep = json.loads((d / "build.json").read_text())
    job = json.loads((d / "job.json").read_text())
    lines = rep["block"]["finish"]["state"]["cut_lines"]
    poly = polyline(lines)
    block = read_stl(d / "femoral_block.stl")
    out = {"block_version": rep["block"].get("version"), "cut_lines": lines, "sections": {}}
    # The block's inner faces against the J-curve, at ML levels clear of the pegs.
    for y in (-60.0, -10.0, 0.0, 10.0, 60.0):
        s = section(block, y)
        pts = s.reshape(-1, 2)
        inner = pts[(np.abs(pts[:, 0]) < 140) & (pts[:, 1] > -140) & (pts[:, 1] < TOP - 1)]
        dev = dist_to_polyline(inner, poly)
        out["sections"][str(y)] = {"points": int(len(inner)), "max_dev_mm": round(float(dev.max()), 4)}
    # The femoral's bone-facing faces: vertices on the inner side of the component that lie
    # on the box; their distance to the block's box faces.
    fem = read_stl(d / "femoral.stl")
    fv = fem.vertices
    near = dist_to_polyline(fv[:, [0, 2]], poly)
    on_box = near < 0.05
    out["femoral_vertices_on_box"] = int(on_box.sum())
    # Bone cut with the block, in the femoral frame, by the planner's own kernel (what
    # the app's Commit runs).
    from tka_planner.geom.kernel import default_kernel
    F = np.array(job["plan"]["femoral_pose"])
    bone = to_frame(read_stl(job["bones"][0]), F)
    res = default_kernel().difference(bone, block)
    cv = res.mesh.vertices
    rec = res.record.to_dict() if hasattr(res.record, "to_dict") else str(res.record)
    knee = cv[cv[:, 2] < TOP - 5]
    s = lines["posterior"][0]
    out["bone_cut"] = {
        "kernel": rec,
        "lowest_z_mm": round(float(knee[:, 2].min()), 3),
        "below_distal_cut": int((knee[:, 2] < -0.05).sum()),
        "behind_posterior_cut": int((knee[:, 0] < s - 0.05).sum()),
        "outside_box_mm": round(float(outside_box(knee[:, [0, 2]], lines)), 3),
    }
    (d / "block_check.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or ["REF_KJ2", "DEMO_SMALL", "DEMO_LARGE"]:
        print(n, json.dumps(check(n), indent=1))
