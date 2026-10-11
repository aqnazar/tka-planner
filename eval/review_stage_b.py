"""Stage (b) visual review: sagittal sections of each femur with the cut box drawn on.

Private: it reads the patient dataset. Knees are named by case id only. Writes
``stage_b_review.html`` next to this file, a self-contained page (no network).

Everything is drawn in the femoral component's frame, in millimetres:
+X anterior, +Z proximal, origin where the plan seats the component on the distal cut.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from eval_stage_b import LIBRARY, V57, cases  # noqa: E402
from tka_planner.core import femoral_box as fb  # noqa: E402
from tka_planner.core.meshio import read_stl  # noqa: E402
from tka_planner.pipeline import (  # noqa: E402
    box_design, femoral_cut_box_case, measure_case, plan_case,
)
from tka_planner.scene.build import resolve_component_meshes  # noqa: E402


def local_frame(pose):
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3] / np.linalg.norm(pose[:3, :3], axis=0)[None, :]
    return rotation, pose[:3, 3]


def section(vertices, faces, y0):
    """Segments where the plane y = y0 cuts the mesh, as [[x1, z1, x2, z2], ...]."""
    tri = vertices[faces]                      # (M, 3, 3)
    d = tri[:, :, 1] - y0
    crosses = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    tri, d = tri[crosses], d[crosses]
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        t = np.where(da * db < 0, da / np.where(da == db, 1, da - db), np.nan)
        out.append(tri[:, a] + t[:, None] * (tri[:, b] - tri[:, a]))
    pts = np.stack(out, axis=1)                # (K, 3 edges, 3)
    segs = []
    for row in pts:
        ok = row[np.isfinite(row[:, 0])]
        if len(ok) >= 2:
            segs.append([ok[0, 0], ok[0, 2], ok[1, 0], ok[1, 2]])
    return np.round(np.asarray(segs), 2).tolist()


def box_outline(box, design, condyle_top):
    """The five cuts as one polyline, posterior cut top to flange top."""
    p = box["posterior_cut_x_mm"]
    pc, df, ac = box["posterior_chamfer_mm"], box["distal_face_mm"], box["anterior_chamfer_mm"]
    rise = np.tan(np.radians(design.chamfer_deg))
    flare = np.tan(np.radians(box["anterior_flare_deg"]))
    a0, h = box["anterior_cut_x_at_distal_mm"], box["flange_height_mm"]
    front = p + pc + df
    return [[p, condyle_top], [p, pc * rise], [p + pc, 0.0], [front, 0.0],
            [front + ac / rise, ac], [a0 + h * flare, h]]


def v57_outline():
    """The v57 design's own box, from the measured design values."""
    flare = np.tan(np.radians(10.11))
    a0 = V57["anterior_cut_x_at_distal_mm"]
    return [[-23.9, 45.0], [-23.9, 10.6], [-13.3, 0.0], [13.3, 0.0], [28.2, 15.0],
            [a0 + V57["flange_height_mm"] * flare, V57["flange_height_mm"]]]


ROTATION_LANDMARKS = {
    "PCA medial": "femur.condyle_posterior_medial",
    "PCA lateral": "femur.condyle_posterior_lateral",
    "epicondyle lateral": "femur.epicondyle_lateral",
    "epicondyle medial (prominence)": "femur.epicondyle_medial_prominence",
    "epicondyle medial (sulcus)": "femur.epicondyle_medial_sulcus",
}


def distal_view(samples_local, landmarks, rotation, origin, medial_sign):
    """The condyles seen from distal, anterior up and lateral to the right on both
    knees: (right, up) = (lateral, anterior), in the component frame."""
    lat = -medial_sign
    x, y, z = samples_local.T
    keep = (z >= -15.0) & (z <= 45.0)
    pts = np.c_[y[keep] * lat, x[keep]]
    rng = np.random.default_rng(0)
    if len(pts) > 6000:
        pts = pts[rng.choice(len(pts), 6000, replace=False)]
    marks = {}
    for label, lid in ROTATION_LANDMARKS.items():
        if landmarks.available(lid):
            p = (landmarks.require(lid)[0] - origin) @ rotation
            marks[label] = [round(float(p[1] * lat), 2), round(float(p[0]), 2),
                            round(float(p[2]), 2)]

    def angle(a, b):
        """Rotation of the component's ML axis from the line a->b (medial to lateral),
        positive = external (component anterior turned toward lateral)."""
        if a not in marks or b not in marks:
            return None
        dx, dy = marks[b][0] - marks[a][0], marks[b][1] - marks[a][1]
        return round(float(np.degrees(np.arctan2(dy, dx))), 2)

    return {
        "points": np.round(pts, 1).tolist(),
        "marks": marks,
        "off_pca": angle("PCA medial", "PCA lateral"),
        "off_tea": angle("epicondyle medial (prominence)", "epicondyle lateral"),
        "off_stea": angle("epicondyle medial (sulcus)", "epicondyle lateral"),
    }


DRIVER = HERE / "v57_driver"


def fusion_body(job_name, box, side="right"):
    """The Fusion-generated v57 for this knee, in the component frame (its CAD origin
    sits at the centre of the distal face, cad_origin_x along X), and what Fusion said."""
    report = DRIVER / "report.json"
    rec = {}
    if report.exists():
        rec = next((k for k in json.loads(report.read_text(encoding="utf-8"))["knees"]
                    if k["name"] == job_name), {})
    stl = DRIVER / f"{job_name}.stl"
    status = {"run": bool(rec), "exported": stl.exists() and rec.get("exported", False),
              "warnings": [[n, s, m.split("<")[0][:160]] for n, s, m in rec.get("unhealthy", [])],
              "error": rec.get("error")}
    if not status["exported"]:
        return None, status
    mesh = read_stl(stl)
    v = mesh.vertices + np.array([box["cad_origin_x_mm"], 0.0, 0.0])
    faces = mesh.faces
    # Main Body is the right-knee part (higher flange peak on -Y). Fusion's own left
    # part (Main Body (1), from Mirror2) is made before Fillet8/9 and lacks them, so
    # the left knees get Main Body mirrored across the sagittal plane instead.
    if side.startswith("l"):
        v = v * np.array([1.0, -1.0, 1.0])
        faces = faces[:, ::-1]
        status["mirrored_for_left"] = True
    return (v, faces), status


def body_checks(body, box):
    """Posterior condyle thickness at each condyle's centre plane, and flange top."""
    v = body[0]
    out = {}
    for label, y0 in (("+Y", 20.0), ("-Y", -20.0)):
        s = v[(np.abs(v[:, 1] - y0) < 0.5) & (v[:, 2] > 5.0)]
        out[f"posterior_thickness_{label}"] = round(float(box["posterior_cut_x_mm"]
                                                          - s[:, 0].min()), 2)
    out["flange_top_z"] = round(float(v[:, 2].max()), 2)
    # v57 is mirrored about Y = 0; a body that isn't symmetric came out broken.
    out["y_extent"] = [round(float(v[:, 1].min()), 2), round(float(v[:, 1].max()), 2)]
    out["symmetric"] = abs(out["y_extent"][0] + out["y_extent"][1]) < 1.0
    return out


def knee_record(name, femur_mesh, pose, box, design, implant, side, extra=None,
                landmarks=None, layers=None):
    rotation, origin = local_frame(pose)
    fv = (femur_mesh.vertices - origin) @ rotation
    samples = (fb.femur_samples(femur_mesh) - origin) @ rotation
    x, y, z = samples.T
    distal = (z >= 0.0) & (z <= 2.0)
    centre_y = float(y[distal].min() + y[distal].max()) / 2.0
    width = float(np.ptp(y[distal]))
    within = np.abs(y - centre_y) <= width / 2.0
    medial_sign = 1.0 if side.startswith("r") else -1.0

    heights = np.arange(0.0, 120.5, 1.0)
    front = fb._anterior_profile(x[within], z[within], heights)
    back = -fb._anterior_profile(-x[within], z[within], heights)
    excess = np.array([np.nan if v is None else v for v in box["trochlea_excess_mm"]])
    line = front[:len(excess)] - excess
    slope, intercept = np.polyfit(heights[:len(excess)][np.isfinite(line)],
                                  line[np.isfinite(line)], 1)
    peak = float(np.nanmax(excess[heights[:len(excess)] < fb.SHAFT_WINDOW_MM[0]]))

    planes = {
        "medial condyle": centre_y + medial_sign * width / 4.0,
        "midline (groove/notch)": centre_y,
        "lateral condyle": centre_y - medial_sign * width / 4.0,
    }
    sections = {}
    for label, y0 in planes.items():
        sections[label] = {
            "y": round(y0, 2),
            "bone": section(fv, femur_mesh.faces, y0),
            "implant": section(implant[0], implant[1], y0) if implant else [],
        }
        for key, mesh in (layers or {}).items():
            sections[label][key] = section(mesh[0], mesh[1], y0) if mesh else []

    def clean(a):
        return [None if not np.isfinite(v) else round(float(v), 2) for v in a]

    record = {
        "name": name,
        "side": side,
        "sections": sections,
        "silhouette": {"z": heights.tolist(), "front": clean(front), "back": clean(back)},
        "shaft_line": [round(float(intercept), 3), round(float(slope), 5)],
        "box": {k: v for k, v in box.items()
                if k not in ("trochlea_heights_mm", "trochlea_excess_mm", "frame")},
        "box_outline": np.round(box_outline(box, design, fb.POSTERIOR_CONDYLE_HEIGHT_MM),
                                2).tolist(),
        "excess": clean(excess),
        "threshold": round(max(fb.TROCHLEA_FLOOR_MM, fb.TROCHLEA_FRACTION * peak), 3),
        "peak": round(peak, 2),
        "shaft_window": list(fb.SHAFT_WINDOW_MM),
        "implant_label": implant[2] if implant else None,
    }
    if landmarks is not None:
        record["distal"] = distal_view(samples, landmarks, rotation, origin, medial_sign)
    record.update(extra or {})
    return record


def implant_section_mesh(specs, side):
    spec = specs.get("femoral_component")
    if not spec:
        return None
    mesh = read_stl(spec["path"])
    v = mesh.vertices * spec["scale"]
    source_left = "left" in "".join(c for c in Path(spec["path"]).name.lower() if c.isalpha())
    if source_left != side.startswith("l"):
        v = v * np.array([1.0, -1.0, 1.0])     # mirror across the sagittal plane
    label = (f"library {spec['source_size']} implant x{spec['scale']:.3f}"
             + (" (mirrored)" if source_left != side.startswith("l") else ""))
    return v, mesh.faces, label


def main():
    design = box_design(LIBRARY)
    records, failures = [], []

    # The reference femur at the author's own v57 design pose, with the L3 part (ML 82,
    # the v57 width) at scale 1 and the v57 box from its measured values.
    ref_m = measure_case(REPO / "data/cases/REF_KJ2/FD1Right.stl",
                         REPO / "data/cases/REF_KJ2/TD1Right.stl", "right", case_id="REF_KJ2")
    ref = ref_m.femur
    df_pose = np.load(HERE / "design_poses_kj.npz")["Df"]
    ref_box = fb.femoral_cut_box(ref, df_pose, medial_sign=1.0, design=design).to_dict()
    l3 = next(p for p in (LIBRARY / "L3").glob("*.stl")
              if "implant" in p.name.lower() and "femoral" in p.name.lower())
    l3_mesh = read_stl(l3)
    implant = (l3_mesh.vertices * np.array([1.0, -1.0, 1.0]), l3_mesh.faces,
               "library L3 implant x1.000 (mirrored) = v57 width")
    psi, status = fusion_body("REF_KJ2_design", ref_box)
    v57_body = read_stl(DRIVER / "v57_design.stl")
    extra = {"v57_outline": v57_outline(), "size": "design pose", "fusion": status,
             "v57_body_checks": body_checks((v57_body.vertices, v57_body.faces),
                                            {**ref_box, "posterior_cut_x_mm": -23.85})}
    if psi:
        extra["psi_checks"] = body_checks(psi, ref_box)
    records.append(knee_record("REF_KJ2 @ v57 design pose", ref, df_pose, ref_box, design,
                               implant, "right", extra, landmarks=ref_m.landmarks,
                               layers={"psi": psi,
                                       "v57body": (v57_body.vertices, v57_body.faces)}))
    print("REF_KJ2 design pose done", flush=True)

    for name, femur, tibia, side in cases():
        try:
            m = measure_case(femur, tibia, side, case_id=name)
            plan, sizing = plan_case(m, library=LIBRARY)
            box = femoral_cut_box_case(m, plan, LIBRARY)
        except Exception as error:  # noqa: BLE001 - report every knee
            failures.append({"name": name, "reason": f"{type(error).__name__}: {error}"})
            print(name, "failed", flush=True)
            continue
        if not box or not box.get("available"):
            failures.append({"name": name, "reason": (box or {}).get("reason", "no box")})
            continue
        specs = resolve_component_meshes(LIBRARY, chart=m.chart, sizing=sizing, side=side)
        cut = plan.resections["femoral_distal"]
        psi, status = fusion_body(name, box, side)
        extra = {"size": f"{sizing.nearest_discrete_size} (ML {sizing.implant_ml_mm:.1f})",
                 "distal_mm": [round(cut.medial_depth_mm, 2), round(cut.lateral_depth_mm, 2)],
                 "fusion": status}
        if psi:
            extra["psi_checks"] = body_checks(psi, box)
        records.append(knee_record(name, m.femur, plan.components["femoral_component"],
                                   box, design, implant_section_mesh(specs, side), side,
                                   extra, landmarks=m.landmarks, layers={"psi": psi}))
        print(name, "done", flush=True)

    data = json.dumps({"knees": records, "failures": failures,
                       "design": design.__dict__}, separators=(",", ":"))
    page = (HERE / "stage_b_review_template.html").read_text(encoding="utf-8")
    out = HERE / "stage_b_review.html"
    out.write_text(page.replace("/*DATA*/null", data), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
