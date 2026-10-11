"""Data for the parameter-pipeline demo page (private: patient-derived; served locally).

For the demo's three knees (demo_cases.py: the reference knee and two scaled
extremes), built by implant_pipeline.py: the femur and tibia (cropped to the knee,
decimated for display), the Fusion-built femoral component, plate and insert in their
planned poses, the bones resected on the plan by a mesh boolean (manifold3d): the femur keeps the
intersection of the five cut half-spaces, the tibia everything below its cut, and every
cut is a closed, capped face; both bones also come intact (cropped and capped the same
way), and a manifest with each part's Fusion parameters,
health and checks. Also the femoral size sweep from the scale audit.

  python demo_build.py            -> demo/  (then: python -m http.server -d demo 8765)
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from tka_planner.core.meshio import Mesh, read_stl  # noqa: E402
from tka_planner.geom.gltf import write_glb  # noqa: E402

DEMO = HERE / "demo"
BUILDS = HERE / "patient_builds"
CROP_MM = 90.0          # bone kept within this distance of the joint (along each bone's axis)
CLUSTER_MM = 1.2        # vertex-clustering cell for display decimation
ANTERIOR_FLARE_DEG = 10.1


def weld(mesh):
    q = np.round(mesh.vertices / 1e-3).astype(np.int64)
    u, inv = np.unique(q, axis=0, return_inverse=True)
    return Mesh(vertices=u * 1e-3, faces=inv.reshape(-1)[mesh.faces])


def decimate(mesh, cell):
    """Vertex clustering: snap to a grid, merge, drop collapsed faces."""
    v = mesh.vertices
    key = np.floor(v / cell).astype(np.int64)
    u, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    centre = np.zeros((len(u), 3))
    np.add.at(centre, inv, v)
    centre /= counts[:, None]
    f = inv[mesh.faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    faces = f[keep]
    return Mesh(vertices=centre, faces=faces)


def manifold(mesh):
    import manifold3d
    q = np.round(mesh.vertices / 1e-4).astype(np.int64)
    u, inv = np.unique(q, axis=0, return_inverse=True)
    m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=(u * 1e-4).astype(np.float32),
                                            tri_verts=inv.reshape(-1)[mesh.faces].astype(np.uint32)))
    if m.status().name != "NoError":
        raise ValueError(f"bone mesh is not a manifold: {m.status()}")
    return m


def trimmed(m, planes):
    """Keep n.x + d >= 0 for every [nx, ny, nz, d]; each cut is capped (closed)."""
    for nx, ny, nz, d in planes:
        m = m.trim_by_plane((nx, ny, nz), -d)
    out = m.to_mesh()
    return Mesh(vertices=np.asarray(out.vert_properties)[:, :3].astype(float),
                faces=np.asarray(out.tri_verts).astype(np.int64))


def crop(mesh, origin, axis, length):
    """Faces whose vertices all lie within `length` of `origin` along +/-axis."""
    h = (mesh.vertices - origin) @ axis
    ok = np.abs(h) <= length
    keep = ok[mesh.faces].all(axis=1)
    used = np.unique(mesh.faces[keep])
    remap = -np.ones(len(mesh.vertices), int)
    remap[used] = np.arange(len(used))
    return Mesh(vertices=mesh.vertices[used], faces=remap[mesh.faces[keep]])


def transform(mesh, pose):
    P = np.asarray(pose)
    return Mesh(vertices=mesh.vertices @ P[:3, :3].T + P[:3, 3], faces=mesh.faces)


def plane_world(pose, point, normal):
    """A plane given in a part's CAD frame -> world [nx, ny, nz, d] with n.x + d >= 0 kept."""
    P = np.asarray(pose)
    n = P[:3, :3] @ np.asarray(normal, float)
    n /= np.linalg.norm(n)
    p = P[:3, :3] @ np.asarray(point, float) + P[:3, 3]
    return [*n.tolist(), float(-n @ p)]


def femoral_planes(pose, C, B, D):
    """The five cuts in the femoral CAD frame (+X anterior, +Z proximal), kept side inward."""
    t = np.tan(np.radians(ANTERIOR_FLARE_DEG))
    s = np.sqrt(0.5)
    return [
        plane_world(pose, [0, 0, 0], [0, 0, 1]),                       # distal: keep z >= 0
        plane_world(pose, [-C / 2 - B, 0, 0], [1, 0, 0]),              # posterior: keep x >= -C/2-B
        plane_world(pose, [C / 2 + D, 0, D], [-1, 0, t]),              # anterior face, 10.1 deg flare
        plane_world(pose, [C / 2, 0, 0], [-s, 0, s]),                  # anterior chamfer
        plane_world(pose, [-C / 2, 0, 0], [s, 0, s]),                  # posterior chamfer
    ]


def glb(mesh, path):
    path.write_bytes(write_glb(mesh, name=path.stem))
    return path.name


def main():
    shutil.rmtree(DEMO, ignore_errors=True)
    (DEMO / "data").mkdir(parents=True)
    (DEMO / "vendor").mkdir()
    for f in ("three.module.js", "OrbitControls.js", "LICENSE-three.txt"):
        shutil.copy(REPO / "web/vendor" / f, DEMO / "vendor" / f)
    shutil.copy(REPO / "web/glb.js", DEMO / "glb.js")
    for f in ("index.html", "demo.js"):
        shutil.copy(HERE / "demo_src" / f, DEMO / f)

    jobs = {j["name"]: j for j in json.loads((BUILDS / "jobs_demo.json").read_text())}
    manifest = {"knees": [], "sweep": []}

    for name, job in jobs.items():
        build = BUILDS / name / "build.json"
        if not build.exists():
            continue
        rep = json.loads(build.read_text())
        d = DEMO / "data" / name
        d.mkdir()
        F, T = np.array(job["plan"]["femoral_pose"]), np.array(job["plan"]["tray_pose"])
        (femur_p, tibia_p), side = job["bones"], job["side"]
        fp = {k: float(v.split()[0]) for k, v in job["femoral"].items()}
        planes = {"femur": femoral_planes(F, fp["C_Distal_cut_surface"], fp["B_Posterior_chamfer_projection"],
                                          fp["D_Anterior_chamfer_projection"]),
                  "tibia": [plane_world(T, [0, 0, 0], [0, 0, -1])]}
        # Crop each bone CROP_MM up (femur) or down (tibia) its axis, capped like the cuts.
        crops = {"femur": [plane_world(F, [0, 0, CROP_MM], [0, 0, -1])],
                 "tibia": [plane_world(T, [0, 0, -CROP_MM], [0, 0, 1])]}
        parts = []
        for bone, path in (("femur", femur_p), ("tibia", tibia_p)):
            m = manifold(read_stl(path))
            whole = decimate(trimmed(m, crops[bone]), CLUSTER_MM)
            cut = decimate(trimmed(m, crops[bone] + planes[bone]), CLUSTER_MM)
            parts.append({"id": bone, "label": bone.capitalize(), "colour": "#e8dcc4", "bone": True,
                          "file": glb(whole, d / f"{bone}.glb"), "resected": glb(cut, d / f"{bone}_resected.glb")})
        for part, pose, colour, label in (("femoral", F, "#b8c4d6", "Femoral component"),
                                          ("plate", T, "#9aa7b8", "Tibial plate"),
                                          ("insert", T, "#f2f2ea", "Insert")):
            stl = BUILDS / name / f"{part}.stl"
            if stl.exists():
                parts.append({"id": part, "label": label, "colour": colour,
                              "file": glb(transform(read_stl(stl), pose), d / f"{part}.glb")})
        # Checks measured on the exported parts.
        checks = {}
        fs = BUILDS / name / "femoral.stl"
        if fs.exists():
            v = read_stl(fs).vertices
            ml = fp["Femur_ML"]
            checks["femoral width (mm)"] = [round(float(np.ptp(v[:, 1])), 2), f"want ML - 1 = {ml - 1:.2f}"]
            checks["femoral distal thickness (mm)"] = [round(float(-v[:, 2].min()), 2), f"want 9 x box/52.1 = {job['plan']['distal_mm']:.2f}"]
        checks["tray coverage of the cut"] = [f"{job['plan']['tray_coverage']:.0%}", "no overhang, 1 mm inset"]
        p = job["plan"]
        checks["insert (poly under the condyle, mm)"] = [round(p["insert_mm"], 2),
                                                        "below 6 mm: +%.1f mm tibial resection would reach 6" % p["extra_tibial_resection_for_min_mm"]
                                                        if p["insert_below_min"] else "ok (>= 6 mm)"]
        checks["femoral-tray tilt in the plan (deg)"] = [p["femoral_tray_tilt_deg"], "the CAD insert assumes 0"]
        notes = []
        ins = rep.get("insert", {})
        if any(h[0] == "Base Sketch" for h in ins.get("health", [])):
            notes.append("Insert outline: still the design outline (its sketch projects plate body edges); "
                         "the rebuild projecting the plate's sketch is in progress.")
        if "Fillet2" in [h[0] for h in ins.get("health", []) if h[1] == 3] or ins.get("suppressed"):
            notes.append("Insert Fillet2 (1 mm round on the anterior top edge) suppressed: its edge is chosen by selection.")
        manifest["knees"].append({
            "name": name, "label": job["label"], "side": side, "parts": parts,
            "anatomy": job["anatomy"],
            "scale": json.loads(Path(femur_p).with_name("scale.json").read_text())["scale_ml_ap_si"]
                     if Path(femur_p).with_name("scale.json").exists() else None, "planes": planes,
            "centre": (T[:3, 3] + 0.5 * (F[:3, 3] - T[:3, 3])).tolist(),
            "fusion": {part: {"folder": rep.get("folder"), "version": rep.get(part, {}).get("version"),
                              "health": rep.get(part, {}).get("health") or [],
                              "parameters": job[part]} for part in ("femoral", "plate", "insert")},
            "plan": {k: v for k, v in p.items() if not k.endswith("_pose")},
            "box": job["femoral_box"],
            "checks": checks, "notes": notes,
        })
        print(name, [x["id"] for x in parts], flush=True)

    sweep = HERE / "audit_author_v19"
    (DEMO / "data/sweep").mkdir()
    for stl in sorted(sweep.glob("ml*.stl"), key=lambda s: int(s.stem[2:])):
        manifest["sweep"].append({"ml": int(stl.stem[2:]), "file": glb(read_stl(stl), DEMO / "data/sweep" / f"{stl.stem}.glb")})
    (DEMO / "data/manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", DEMO)


if __name__ == "__main__":
    main()
