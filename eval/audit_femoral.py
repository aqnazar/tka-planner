"""Audit of the sandbox femoral's parameter relations (v18 onwards).

Drives a Driver Tests copy, never the sandbox. Two tests:

  scale        At several Femur_ML values the body must be the L3 body scaled by ML/82,
               apart from what is meant not to scale (the pin length, rounded to 5 mm).
               Any constant left unscaled shows up as a deviation in its region.
  sensitivity  Perturb each constant that does not scale (+10 %) at the smallest and
               largest ML. A constant that changes the body is a design decision; one
               that changes nothing is an extent or a helper.

  python audit_femoral.py scale [--model author_v18]
  python audit_femoral.py sensitivity [--model author_v18]
  python audit_femoral.py report [--model author_v18]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

SCALE_ML = (60, 62, 72, 82, 84, 90, 95)
SENS_ML = (60, 82, 95)
# Constants (mm or dimensionless) that do not follow Femur_ML in v18.
CONSTANTS = ("Notch_Offset_X", "d162", "d222", "d272", "d171", "d218", "d259", "d260",
             "d286", "J_Curve_Offset_FIXED", "J_Curve_Offset_Coefficient")

SCRIPT = r'''
import adsk.core, adsk.fusion, json, struct
LINEAGE = __LINEAGE__
CASES = json.load(open(__CASES__, encoding="utf-8"))
OUT = __OUT__

def write_stl(path, c, t):
    n = len(t) // 3
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<3f", 0, 0, 0))
            for k in range(3):
                j = t[3*i+k]; f.write(struct.pack("<3f", c[3*j]*10, c[3*j+1]*10, c[3*j+2]*10))
            f.write(b"\0\0")

def fresh(app, data_file):
    for d in list(app.documents):
        if d.dataFile and d.dataFile.id == LINEAGE:
            d.close(False)
    doc = app.documents.open(data_file, True)
    return doc, adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))

def state(des):
    out = {}
    for name in ("Main Body", "Main Body (1)"):
        b = des.rootComponent.bRepBodies.itemByName(name)
        if b is None:
            out[name] = None
            continue
        bb = b.boundingBox
        out[name] = {"volume_mm3": round(b.volume * 1000, 3), "faces": b.faces.count,
                     "min": [round(v * 10, 4) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z)],
                     "max": [round(v * 10, 4) for v in (bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}
    tl = des.timeline
    bad = []
    for i in range(tl.count):
        it = tl.item(i)
        if it.healthState != 0:
            try: msg = it.errorOrWarningMessage
            except Exception: msg = ""
            bad.append([it.name, it.healthState, msg.split("<")[0][:160]])
    out["unhealthy"] = bad
    return out

def run(_context: str):
    app = adsk.core.Application.get()
    active = app.activeDocument
    if any(d.isModified for d in app.documents if d.dataFile and d.dataFile.id == LINEAGE):
        raise RuntimeError("An instance of the copy has unsaved changes; not touching it.")
    data_file = app.data.findFileById(LINEAGE)
    results = []
    try:
        for case in CASES:
            doc, des = fresh(app, data_file)
            rec = {"name": case["name"]}
            try:
                des.userParameters.itemByName("Femur_ML").expression = f"{case['ml']} mm"
                for n, e in case.get("set", {}).items():
                    p = des.allParameters.itemByName(n)
                    rec.setdefault("was", {})[n] = p.expression
                    p.expression = e
                des.computeAll()
                rec.update(state(des))
                rec["params"] = {p.name: round(p.value * (10 if p.unit == "mm" else 1), 5)
                                 for p in des.allParameters}
                if case.get("export"):
                    b = des.rootComponent.bRepBodies.itemByName("Main Body")
                    calc = b.meshManager.createMeshCalculator()
                    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
                    m = calc.calculate()
                    write_stl(OUT + "\\" + case["name"] + ".stl", list(m.nodeCoordinatesAsDouble), list(m.nodeIndices))
            except Exception as error:
                rec["error"] = f"{type(error).__name__}: {error}"
            results.append(rec)
    finally:
        doc, des = fresh(app, data_file)
        dirty = [d.name for d in app.documents
                 if d.isValid and d.dataFile and d.dataFile.id == LINEAGE and d.isModified]
        after = {"dirty": dirty, "version": data_file.versionNumber,
                 "latest": data_file.latestVersionNumber}
        if not dirty:
            doc.close(False)
        if active and active.isValid:
            active.activate()
    json.dump({"results": results, "after": after}, open(OUT + "\\_last.json", "w"))
    print(json.dumps({"n": len(results), "after": after}))
'''


def drive(model, cases, tag, variant=""):
    m = MODELS[model]
    out = HERE / f"audit_{model}{'_' + variant if variant else ''}"
    out.mkdir(exist_ok=True)
    allres = {}
    path = out / f"{tag}.json"
    if path.exists():
        allres = json.loads(path.read_text(encoding="utf-8"))
    todo = [c for c in cases if c["name"] not in allres]
    for i in range(0, len(todo), 6):          # several MCP calls: each stays well under the timeout
        chunk = todo[i:i + 6]
        (out / "_cases.json").write_text(json.dumps(chunk), encoding="utf-8")
        script = (SCRIPT.replace("__LINEAGE__", repr(m["lineage"]))
                  .replace("__CASES__", repr(str(out / "_cases.json")))
                  .replace("__OUT__", repr(str(out))))
        print(call_fusion(script, read_only=False).strip().splitlines()[-1])
        got = json.loads((out / "_last.json").read_text())
        for r in got["results"]:
            allres[r["name"]] = r
        path.write_text(json.dumps(allres, indent=1), encoding="utf-8")
        if got["after"]["dirty"] or got["after"]["latest"] != got["after"]["version"]:
            raise SystemExit(f"copy left dirty or saved: {got['after']}")
    return allres


# Fusion's fitted-spline tangent-handle length acts as a weight, not a length: the curve
# scales with its fit points only when the handle lengths stay constant (2026-10-09).
HANDLES_CONSTANT = {"d141": "37 mm", "d142": "24 mm", "d143": "18 mm", "d145": "18 mm",
                    "d129": "16 mm", "d131": "16 mm", "d136": "18 mm", "d137": "20 mm"}
VARIANTS = {"": {}, "handles": HANDLES_CONSTANT}


def scale_cases(variant=""):
    return [{"name": f"ml{ml}", "ml": ml, "export": True, "set": VARIANTS[variant]} for ml in SCALE_ML]


def sens_cases(snapshot):
    cases = []
    for ml in SENS_ML:
        cases.append({"name": f"ml{ml}", "ml": ml})
        for n in CONSTANTS:
            expr = snapshot["params"][n]["expression"]
            cases.append({"name": f"ml{ml}_{n}", "ml": ml, "set": {n: f"( {expr} ) * 1.1"}})
    return cases


def manifold(path, scale=1.0):
    """Welded Manifold of a Fusion STL (unwelded triangles), optionally scaled about the CAD origin."""
    import manifold3d as m3
    from tka_planner.core.meshio import read_stl
    me = read_stl(path)
    q = np.round(me.vertices * scale / 1e-4).astype(np.int64)
    u, inv = np.unique(q, axis=0, return_inverse=True)
    return m3.Manifold(m3.Mesh(vert_properties=(u * 1e-4).astype(np.float32),
                               tri_verts=inv.reshape(-1)[me.faces].astype(np.uint32)))


def pieces(solid, min_volume=0.5):
    out = []
    for p in solid.decompose():
        v = p.volume()
        if v >= min_volume:
            lo, hi = np.array(p.bounding_box()[:3]), np.array(p.bounding_box()[3:])
            out.append((round(v, 1), np.round(lo, 1).tolist(), np.round(hi, 1).tolist()))
    return sorted(out, reverse=True)


def report_scale(model, variant=""):
    from tka_planner.core.meshio import read_stl
    out = HERE / f"audit_{model}{'_' + variant if variant else ''}"
    res = json.loads((out / "scale.json").read_text())
    print(f"{'ML':>4} {'health':8} {'width':>7} {'want':>6} {'distal':>7} {'want':>6} {'pinL':>4}"
          f" {'vol/L3':>7} {'(ML/82)^3':>9} {'not scaled mm3':>14}  pieces > 0.5 mm3 (vol, min, max)")
    for ml in SCALE_ML:
        r = res[f"ml{ml}"]
        if "error" in r:
            print(ml, r["error"]); continue
        s = ml / 82
        v = read_stl(out / f"ml{ml}.stl").vertices
        a, b = manifold(out / f"ml{ml}.stl"), manifold(out / "ml82.stl", s)
        extra, missing = a - b, b - a
        P = r["params"]
        print(f"{ml:4d} {('ok' if not r['unhealthy'] else str(len(r['unhealthy'])) + ' warn'):8} "
              f"{np.ptp(v[:, 1]):7.3f} {ml - 1:6.2f} {-v[:, 2].min():7.3f} {9 * s:6.3f} {P['E_Pin_length']:4.0f} "
              f"{r['Main Body']['volume_mm3'] / res['ml82']['Main Body']['volume_mm3']:7.4f} {s ** 3:9.4f} "
              f"{extra.volume() + missing.volume():14.1f}")
        for tag, solid in (("+", extra), ("-", missing)):
            for pc in pieces(solid):
                print(f"      {tag} {pc}")
        if r["unhealthy"]:
            print("      ", r["unhealthy"])


def report_sens(model):
    out = HERE / f"audit_{model}"
    res = json.loads((out / "sensitivity.json").read_text())
    print(f"{'parameter':28} " + " ".join(f"{'ML'+str(ml)+' dV mm3':>14} {'d bbox':>7}" for ml in SENS_ML))
    for n in CONSTANTS:
        row = []
        for ml in SENS_ML:
            base, r = res[f"ml{ml}"], res.get(f"ml{ml}_{n}")
            if r is None or "error" in r or r["Main Body"] is None:
                row.append(f"{'ERR':>14} {'':>7}"); continue
            dv = r["Main Body"]["volume_mm3"] - base["Main Body"]["volume_mm3"]
            db = max(abs(a - b) for a, b in zip(r["Main Body"]["min"] + r["Main Body"]["max"],
                                                base["Main Body"]["min"] + base["Main Body"]["max"]))
            flag = "!" if r["unhealthy"] else " "
            row.append(f"{dv:13.2f}{flag} {db:7.3f}")
        print(f"{n:28} " + " ".join(row))
    for k, r in res.items():
        if r.get("unhealthy") or r.get("error"):
            print(k, r.get("error", ""), r.get("unhealthy"))


if __name__ == "__main__":
    a = sys.argv[1:]
    model = "author_v18"
    if "--model" in a:
        i = a.index("--model"); model = a[i + 1]; a = a[:i] + a[i + 2:]
    variant = ""
    if "--variant" in a:
        i = a.index("--variant"); variant = a[i + 1]; a = a[:i] + a[i + 2:]
    if a[0] == "scale":
        drive(model, scale_cases(variant), "scale", variant); report_scale(model, variant)
    elif a[0] == "sensitivity":
        snap = json.loads((HERE / "fusion_snapshots/audit_v18.json").read_text(encoding="utf-8"))
        drive(model, sens_cases(snap), "sensitivity"); report_sens(model)
    else:
        report_scale(model); report_sens(model)
