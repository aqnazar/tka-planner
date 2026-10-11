"""How the built parts sit on the resected bone: overhang and centring at every cut.

For each demo knee (patient_builds/jobs_demo.json) and every cut plane, the bone's
section on the plane is compared with the part's section just off it, on the part's side.
Reported per plane: the part area outside the bone, the largest ML overhang over the
AP rows (+ = the part sticks out past the bone), and the centre offset part - bone.

  python fit_check.py [knee ...]   -> patient_builds/fit_check.json and a table
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from tka_planner.core.implant_spec import section_mask  # noqa: E402
from tka_planner.core.meshio import Mesh, read_stl  # noqa: E402

BUILDS = HERE / "patient_builds"
FLARE_DEG = 10.1
OFF = 0.5          # mm off the cut, on the part's side


def to_frame(mesh, pose):
    P = np.asarray(pose)
    return Mesh(vertices=(mesh.vertices - P[:3, 3]) @ P[:3, :3], faces=mesh.faces)


def plane_pose(point, normal, y=(0, 1, 0)):
    """A frame on the plane: z = normal (the kept bone's side), y = the given in-plane
    direction (ML), x = y x z."""
    z = np.asarray(normal, float); z /= np.linalg.norm(z)
    y = np.asarray(y, float); y -= (y @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    P = np.eye(4); P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, point
    return P


def on_grid(mask, xs, ys, gx0, gy0, shape, grid):
    """Resample a section mask onto a common grid (nearest cell)."""
    out = np.zeros(shape, bool)
    jj, ii = np.nonzero(mask)
    X = np.round((xs[ii] - gx0) / grid).astype(int)
    Y = np.round((ys[jj] - gy0) / grid).astype(int)
    ok = (X >= 0) & (X < shape[1]) & (Y >= 0) & (Y < shape[0])
    out[Y[ok], X[ok]] = True
    return out


def compare(bone, part, pose, grid=0.25, region=None):
    """Bone section on the plane vs the part's section OFF mm on the part's side.
    `region`: other planes [(point, normal)] whose kept side bounds this facet."""
    try:
        bm, bx, by = section_mask(bone, pose, offset_mm=0.0, grid_mm=grid)
    except ValueError:
        return None
    try:
        pm, px, py = section_mask(part, pose, offset_mm=-OFF, grid_mm=grid)
    except ValueError:
        return None
    lo = np.minimum([bx.min(), by.min()], [px.min(), py.min()])
    hi = np.maximum([bx.max(), by.max()], [px.max(), py.max()])
    shape = (int((hi[1] - lo[1]) / grid) + 2, int((hi[0] - lo[0]) / grid) + 2)
    B = on_grid(bm, bx, by, lo[0], lo[1], shape, grid)
    Pm = on_grid(pm, px, py, lo[0], lo[1], shape, grid)
    if not Pm.any() or not B.any():
        return None
    gy = lo[1] + np.arange(shape[0]) * grid
    gx = lo[0] + np.arange(shape[1]) * grid
    if region:
        GX, GY = np.meshgrid(gx, gy)
        pts = pose[:3, 3] + GX[..., None] * pose[:3, 0] + GY[..., None] * pose[:3, 1]
        keep = np.ones(shape, bool)
        for q, n in region:
            n = np.asarray(n, float) / np.linalg.norm(n)
            keep &= (pts - np.asarray(q, float)) @ n >= -1.0
        B &= keep; Pm &= keep
        if not Pm.any() or not B.any():
            return None
    # ML overhang per AP row: local x is AP, local y is ML -> rows of constant x.
    over = []
    for i in range(shape[1]):
        p, b = Pm[:, i], B[:, i]
        if p.any() and b.any():
            over.append((gy[p].max() - gy[b].max(), gy[b].min() - gy[p].min()))
    over = np.array(over)
    GX, GY = np.meshgrid(gx, gy)
    # Distance of each part cell outside the bone to the nearest bone cell.
    out_pts = np.c_[GX[Pm & ~B], GY[Pm & ~B]]
    bone_pts = np.c_[GX[B], GY[B]][::3]
    dist = np.zeros(0)
    if len(out_pts):
        dist = np.concatenate([np.sqrt(((c[:, None, :] - bone_pts[None]) ** 2).sum(-1)).min(1)
                               for c in np.array_split(out_pts, max(1, len(out_pts) // 400))])
    ext = lambda M: [round(float(GY[M].min()), 1), round(float(GY[M].max()), 1)]
    return {"overhang_max_mm": round(float(dist.max()), 2) if len(dist) else 0.0,
            "overhang_p95_mm": round(float(np.percentile(dist, 95)), 2) if len(dist) else 0.0,
            "ml_extent_part": ext(Pm), "ml_extent_bone": ext(B),
            "outside_mm2": round(float((Pm & ~B).sum() * grid * grid), 1),
            "part_mm2": round(float(Pm.sum() * grid * grid), 1),
            "ml_overhang_plus_y_mm": round(float(over[:, 0].max()), 2) if len(over) else None,
            "ml_overhang_minus_y_mm": round(float(over[:, 1].max()), 2) if len(over) else None,
            "centre_offset_mm": [round(float(GX[Pm].mean() - GX[B].mean()), 2),
                                 round(float(GY[Pm].mean() - GY[B].mean()), 2)]}


def overlap(a, b):
    """Volume shared by two closed meshes (mm^3), by a manifold boolean."""
    import manifold3d
    def M(m):
        q = np.round(m.vertices / 1e-4).astype(np.int64)
        u, inv = np.unique(q, axis=0, return_inverse=True)
        return manifold3d.Manifold(manifold3d.Mesh(vert_properties=(u * 1e-4).astype(np.float32),
                                                   tri_verts=inv.reshape(-1)[m.faces].astype(np.uint32)))
    try:
        return round(float((M(a) ^ M(b)).volume()), 2)
    except Exception as ex:          # noqa: BLE001
        return f"n/a: {ex}"


def femoral_planes_cad(C, B, D):
    t = np.tan(np.radians(FLARE_DEG))
    s = np.sqrt(0.5)
    return {"distal": ([0, 0, 0], [0, 0, 1]),
            "posterior": ([-C / 2 - B, 0, 0], [1, 0, 0]),
            "anterior": ([C / 2 + D, 0, D], [-1, 0, t]),
            "anterior chamfer": ([C / 2, 0, 0], [-s, 0, s]),
            "posterior chamfer": ([-C / 2, 0, 0], [s, 0, s])}


def check(job):
    name = job["name"]
    femur, tibia = (read_stl(p) for p in job["bones"])
    F, T = np.array(job["plan"]["femoral_pose"]), np.array(job["plan"]["tray_pose"])
    out = {"femoral": {}, "plate": None, "insert_vs_plate": None}
    fem = BUILDS / name / "femoral.stl"
    if fem.exists():
        comp = read_stl(fem)
        bone = to_frame(femur, F)
        fp = {k: float(v.split()[0]) for k, v in job["femoral"].items()}
        planes = femoral_planes_cad(fp["C_Distal_cut_surface"], fp["B_Posterior_chamfer_projection"],
                                    fp["D_Anterior_chamfer_projection"])
        for plane, (p, n) in planes.items():
            others = [v for k, v in planes.items() if k != plane]
            out["femoral"][plane] = compare(bone, comp, plane_pose(np.array(p, float), n), region=others)
    pl = BUILDS / name / "plate.stl"
    if pl.exists():
        plate = read_stl(pl)
        bone = to_frame(tibia, T)
        # The tray's frame: +Z proximal, the part above the cut. Bone kept below: z flipped.
        P = plane_pose(np.zeros(3), [0, 0, -1], y=(1, 0, 0))     # local y = tray +X (ML)
        out["plate"] = compare(bone, plate, P)
        ins = BUILDS / name / "insert.stl"
        if ins.exists():
            # Insert footprint just above the plate floor vs the plate's inner floor.
            insert = read_stl(ins)
            Q = plane_pose(np.array([0, 0, 5.0 + OFF]), [0, 0, 1], y=(1, 0, 0))
            a = section_mask(insert, Q, offset_mm=0.0)
            b = section_mask(plate, plane_pose(np.array([0, 0, 2.5]), [0, 0, 1], y=(1, 0, 0)), offset_mm=0.0)
            ca = [float(np.meshgrid(a[1], a[2])[k][a[0]].mean()) for k in (0, 1)]
            cb = [float(np.meshgrid(b[1], b[2])[k][b[0]].mean()) for k in (0, 1)]
            out["insert_vs_plate"] = {"insert_centre": np.round(ca, 2).tolist(), "plate_centre": np.round(cb, 2).tolist(),
                                      "overlap_mm3": overlap(insert, plate),
                                      "insert_floor_footprint_mm2": round(float(a[0].sum() * 0.0625), 1)}
    return out


def main(names):
    jobs = json.loads((BUILDS / "jobs_demo.json").read_text())
    res = {}
    for j in jobs:
        if names and j["name"] not in names:
            continue
        r = res[j["name"]] = check(j)
        print(f"== {j['name']}  Femur_ML {j['femoral']['Femur_ML']}  tray {j['plate']['Tibial_Plateau_ML']} x {j['plate']['Tibial_Plateau_AP']}")
        for plane, v in list(r["femoral"].items()) + [("PLATE on tibial cut", r["plate"])]:
            if v:
                print(f"   {plane:20s} outside {v['outside_mm2']:7.1f} of {v['part_mm2']:7.1f} mm2  "
                      f"overhang max {v['overhang_max_mm']:5.2f} p95 {v['overhang_p95_mm']:5.2f}  "
                      f"ML part {v['ml_extent_part']} bone {v['ml_extent_bone']}  centre part-bone {v['centre_offset_mm']}")
            else:
                print(f"   {plane:20s} (no section)")
        print("   insert vs plate:", r["insert_vs_plate"])
    (BUILDS / "fit_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])


def facet_extents(bone_cad, C, B, D, facets=("distal", "posterior chamfer", "posterior"), grid=0.25):
    """ML extent [ymin, ymax] of the bone's cut surface on each femoral facet (CAD frame,
    the facet bounded by the other four planes)."""
    planes = femoral_planes_cad(C, B, D)
    out = {}
    for name in facets:
        p, n = planes[name]
        pose = plane_pose(np.array(p, float), n)
        m, xs, ys = section_mask(bone_cad, pose, offset_mm=0.0, grid_mm=grid)
        GX, GY = np.meshgrid(xs, ys)
        pts = pose[:3, 3] + GX[..., None] * pose[:3, 0] + GY[..., None] * pose[:3, 1]
        keep = m.copy()
        for k, (q, nn) in planes.items():
            if k == name:
                continue
            nn = np.asarray(nn, float) / np.linalg.norm(nn)
            keep &= (pts - np.asarray(q, float)) @ nn >= -1.0
        out[name] = [float(GY[keep].min()), float(GY[keep].max())]
    return out
