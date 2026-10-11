"""Which sketch geometry does not scale with Femur_ML?

Drives a Driver Tests copy (never the sandbox) to ML 60, 82 and 95, reads every sketch
curve's key points (ends, centres, spline fit and control points) in world mm, and
compares each against its L3 position times ML/82. Fusion profiles and extrude
extents downstream of a curve that fails here cannot scale either.

  python audit_sketches.py [--model author_v18]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

MLS = (60, 82, 95)

SCRIPT = r'''
import adsk.core, adsk.fusion, json
LINEAGE = __LINEAGE__
MLS = __MLS__
OUT = __OUT__

def mm(p): return [p.x * 10, p.y * 10, p.z * 10]

def points(c):
    g = c.worldGeometry
    pts = {}
    if hasattr(g, "startPoint"):
        pts["start"] = mm(g.startPoint); pts["end"] = mm(g.endPoint)
    if hasattr(g, "center"):
        pts["centre"] = mm(g.center)
    if hasattr(g, "radius"):
        pts["r"] = [g.radius * 10, 0, 0]
    fs = adsk.fusion.SketchFittedSpline.cast(c)
    if fs:
        for i, p in enumerate(fs.fitPoints):
            pts[f"fit{i}"] = mm(p.worldGeometry)
    if hasattr(g, "controlPoints"):
        for i, p in enumerate(g.controlPoints):
            pts[f"cp{i}"] = mm(p)
    return pts

def fresh(app, data_file):
    for d in list(app.documents):
        if d.dataFile and d.dataFile.id == LINEAGE:
            d.close(False)
    doc = app.documents.open(data_file, True)
    return doc, adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))

def run(_context: str):
    app = adsk.core.Application.get()
    active = app.activeDocument
    if any(d.isModified for d in app.documents if d.dataFile and d.dataFile.id == LINEAGE):
        raise RuntimeError("copy has unsaved changes")
    data_file = app.data.findFileById(LINEAGE)
    res = {}
    try:
        for ml in MLS:
            doc, des = fresh(app, data_file)
            des.userParameters.itemByName("Femur_ML").expression = f"{ml} mm"
            des.computeAll()
            sk = {}
            for s in des.rootComponent.sketches:
                sk[s.name] = [{"i": i, "type": c.objectType.split("::")[-1],
                               "construction": c.isConstruction, "reference": c.isReference,
                               "pts": points(c)} for i, c in enumerate(s.sketchCurves)]
                sk[s.name + " :: points"] = [{"i": i, "type": "SketchPoint", "construction": False,
                                              "reference": p.isReference,
                                              "pts": {"p": mm(p.worldGeometry)}}
                                             for i, p in enumerate(s.sketchPoints)]
            feats = {}
            for f in des.rootComponent.features.extrudeFeatures:
                rec = {}
                for side, ext in (("one", f.extentOne), ("two", f.extentTwo)):
                    if ext is None: continue
                    d = adsk.fusion.DistanceExtentDefinition.cast(ext)
                    rec[side] = (ext.objectType.split("::")[-1], d.distance.expression if d else "")
                try: rec["start"] = f.startExtent.objectType.split("::")[-1]
                except Exception: pass
                feats[f.name] = rec
            res[str(ml)] = {"sketches": sk, "extrudes": feats}
    finally:
        doc, des = fresh(app, data_file)
        dirty = [d.name for d in app.documents if d.isValid and d.dataFile and d.dataFile.id == LINEAGE and d.isModified]
        if not dirty: doc.close(False)
        if active and active.isValid: active.activate()
    json.dump(res, open(OUT, "w"))
    print(json.dumps({"dirty": dirty}))
'''


def main(model):
    out = HERE / f"audit_{model}"
    out.mkdir(exist_ok=True)
    path = out / "sketches.json"
    if not path.exists():
        script = (SCRIPT.replace("__LINEAGE__", repr(MODELS[model]["lineage"]))
                  .replace("__MLS__", repr(MLS)).replace("__OUT__", repr(str(path))))
        print(call_fusion(script, read_only=False).strip().splitlines()[-1])
    res = json.loads(path.read_text())
    ref = res["82"]["sketches"]
    print(res["82"]["extrudes"])
    for ml in (60, 95):
        s = ml / 82
        print(f"\n== ML {ml}: points off their scaled L3 position by > 0.01 mm")
        for name, curves in res[str(ml)]["sketches"].items():
            bad = []
            for c, c0 in zip(curves, ref[name]):
                for k, p in c["pts"].items():
                    p0 = np.array(c0["pts"].get(k, [np.nan] * 3)) * s
                    d = np.linalg.norm(np.array(p) - p0)
                    if not d < 0.01:
                        bad.append((c["i"], c["type"][6:], "constr" if c["construction"] else
                                    ("ref" if c["reference"] else "geom"), k,
                                    np.round(p0, 2).tolist(), np.round(p, 2).tolist(), round(float(d), 3)))
            if bad:
                print(f"  {name}: {len(bad)}")
                for b in bad[:40]:
                    print("     ", b)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[a.index("--model") + 1] if "--model" in a else "author_v18")
