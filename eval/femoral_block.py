"""The femoral cutting block, derived from each knee's own femoral J-curve (author
2026-10-10).

The old block (Robot TKA / Cutting Block Femoral (L&R) v11) is the femoral's J-curve sketch
with an outer rectangle (150 x 150 mm, extruded 160 mm), sharing the femoral's origin, so the
two land together and the planner only subtracts the block from the bone. Its box was drawn
by hand, so it can't follow a patient femoral; the planner scaled the library block by ML.

Here the block is a design of its own (Driver Tests / Cutting Block Femoral) that holds the
femoral as a reference at the origin:
  Block profile   the femoral's J-curve cut lines (posterior, posterior chamfer, distal,
                  anterior chamfer, anterior), projected and linked; the posterior and
                  anterior cuts continued up to Block_Top; an outer profile Block_Reach
                  anterior, posterior and below the distal cut (larger than the old block's,
                  so no bone is left outside it)
  Block           the profile extruded Block_Width, symmetric about the femoral's XZ plane
  Pegs            the femoral body joined (kept): only its pegs reach inside the box, so the
                  bone gets the peg holes, from the knee's side body.
All three are made again by every build (a swapped femoral can cost the extrusion its
profile), the same rule as the insert outline.
Each knee's copy (Patient Builds/<knee>/) has its reference repointed to that knee's femoral
version, so the cut is that femoral's J-curve; the STL is in the femoral's CAD frame.

The old block also moved every box face 0.4 mm into the bone (OffsetFaces1); not done here.

  python femoral_block.py base        # make and save the design in Driver Tests (once)
  python femoral_block.py base <femoral lineage>   # the same, next to another femoral
                                                   # (the Author Sandbox, 2026-10-10)
  python femoral_block.py check <id>  # read a block design: profile, body, references
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402

COPIES = HERE / "tray_audit" / "copies.json"
NAME = "Cutting Block Femoral"
MSG = ("Claude: femoral cutting block from the femoral's own J-curve (author 2026-10-10): the J-curve "
       "cut lines projected from the femoral reference, cuts continued to Block_Top, outer profile "
       "Block_Reach, extruded Block_Width; femoral pegs joined.")

# Shared with implant_pipeline.py (pasted into its Fusion script).
BLOCK = r'''
BLOCK_PARAMS = (("Block_Top", "150 mm", "How far above the distal cut the anterior and posterior cuts reach"),
                ("Block_Reach", "150 mm", "Outer profile: anterior, posterior and below the distal cut"),
                ("Block_Width", "300 mm", "ML extent, symmetric about the femoral's XZ plane"))
BOX_SKETCH = "J-curve"

def block_params(des):
    up = des.userParameters; added = []
    for n, e, c in BLOCK_PARAMS:
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString(e), "mm", c); added.append(n)
    return added

def femoral_occurrence(des):
    for o in des.rootComponent.occurrences:
        if o.isReferencedComponent and o.component.sketches.itemByName(BOX_SKETCH) is not None:
            return o
    raise RuntimeError("no femoral reference with a J-curve sketch")

def box_lines(occ):
    """The J-curve sketch's five cut lines, by position: the distal on Z = 0, the posterior
    the vertical one, the anterior the one reaching highest, the chamfers the other two."""
    sk = occ.component.sketches.itemByName(BOX_SKETCH)
    lines = [c for c in sk.sketchCurves.sketchLines if not c.isConstruction]
    if len(lines) != 5:
        raise RuntimeError(f"J-curve has {len(lines)} cut lines, expected 5")
    g = lambda l: (l.worldGeometry.startPoint, l.worldGeometry.endPoint)
    out = {}
    out["distal"] = min(lines, key=lambda l: max(abs(p.z) for p in g(l)))
    out["posterior"] = min(lines, key=lambda l: abs(g(l)[0].x - g(l)[1].x) + 10 * (l is out["distal"]))
    out["anterior"] = max(lines, key=lambda l: max(p.z for p in g(l)))
    rest = [l for l in lines if all(l is not v for v in out.values())]
    rest.sort(key=lambda l: sum(p.x for p in g(l)))
    out["posterior chamfer"], out["anterior chamfer"] = rest
    return out

def strip_block(des):
    """Remove Pegs, Block and Block profile: every build makes them again from the knee's
    J-curve (a swapped femoral can lose the extrusion's profile, "Profile 1 missing")."""
    root = des.rootComponent; gone = []
    for coll, name in ((root.features.combineFeatures, "Pegs"), (root.features.extrudeFeatures, "Block"),
                       (root.sketches, "Block profile")):
        f = coll.itemByName(name)
        if f is not None:
            f.deleteMe(); gone.append(name)
    return gone

def make_block(des):
    """Block profile and Block (the extrusion), from the femoral reference's J-curve."""
    root = des.rootComponent
    occ = femoral_occurrence(des)
    sk = root.sketches.add(root.xZConstructionPlane); sk.name = "Block profile"
    proj = {}
    for key, line in box_lines(occ).items():
        proj[key] = adsk.fusion.SketchLine.cast(sk.project(line.createForAssemblyContext(occ)).item(0))
    def top(l):
        a, b = l.startSketchPoint, l.endSketchPoint
        return a if a.worldGeometry.z > b.worldGeometry.z else b
    def bottom(l):
        a, b = l.startSketchPoint, l.endSketchPoint
        return b if a.worldGeometry.z > b.worldGeometry.z else a
    up = des.userParameters
    T, R = up.itemByName("Block_Top").value, up.itemByName("Block_Reach").value
    P = lambda x, z: sk.modelToSketchSpace(adsk.core.Point3D.create(x, 0, z))
    pt, at = top(proj["posterior"]), top(proj["anterior"])
    ab = bottom(proj["anterior"]).worldGeometry; aw = at.worldGeometry
    xa = aw.x + (aw.x - ab.x) * (T - aw.z) / (aw.z - ab.z)
    L = sk.sketchCurves.sketchLines
    l1 = L.addByTwoPoints(pt, P(pt.worldGeometry.x, T))
    l2 = L.addByTwoPoints(l1.endSketchPoint, P(-R, T))
    l3 = L.addByTwoPoints(l2.endSketchPoint, P(-R, -R))
    l4 = L.addByTwoPoints(l3.endSketchPoint, P(R, -R))
    l5 = L.addByTwoPoints(l4.endSketchPoint, P(R, T))
    l6 = L.addByTwoPoints(l5.endSketchPoint, P(xa, T))
    l7 = L.addByTwoPoints(l6.endSketchPoint, at)
    gc = sk.geometricConstraints
    gc.addCollinear(l1, proj["posterior"]); gc.addCollinear(l7, proj["anterior"])
    gc.addHorizontal(l2); gc.addHorizontal(l4); gc.addCollinear(l6, l2)
    gc.addVertical(l3); gc.addVertical(l5)
    D = sk.sketchDimensions; o = sk.originPoint
    H = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    V = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for p, orient, expr, tx in ((l2.startSketchPoint, V, "Block_Top", P(-R / 2, T / 2)),
                                (l3.startSketchPoint, H, "Block_Reach", P(-R / 2, T + 1)),
                                (l4.startSketchPoint, V, "Block_Reach", P(-R - 1, -R / 2)),
                                (l5.startSketchPoint, H, "Block_Reach", P(R / 2, -R - 1))):
        D.addDistanceDimension(o, p, orient, tx).parameter.expression = expr
    prof = max(sk.profiles, key=lambda p: p.areaProperties().area)
    ext = root.features.extrudeFeatures
    ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ei.setSymmetricExtent(adsk.core.ValueInput.createByString("Block_Width"), True)
    f = ext.add(ei); f.name = "Block"; f.bodies.item(0).name = "Block"
    return {"profiles": sk.profiles.count, "fully_constrained": sk.isFullyConstrained,
            "lines": {k: [round(v * 10, 2) for v in (l.worldGeometry.startPoint.x, l.worldGeometry.startPoint.z,
                                                      l.worldGeometry.endPoint.x, l.worldGeometry.endPoint.z)]
                      for k, l in proj.items()}}

def block_pegs(des, side_body):
    """Pegs: the knee's femoral body joined to the block, kept; remade every build."""
    root = des.rootComponent
    old = root.features.combineFeatures.itemByName("Pegs")
    if old is not None:
        old.deleteMe()
    occ = femoral_occurrence(des)
    fb = occ.component.bRepBodies.itemByName(side_body)
    block = root.bRepBodies.itemByName("Block")
    v0 = block.volume
    tools = adsk.core.ObjectCollection.create(); tools.add(fb.createForAssemblyContext(occ))
    ci = root.features.combineFeatures.createInput(block, tools)
    ci.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    ci.isKeepToolBodies = True
    cf = root.features.combineFeatures.add(ci); cf.name = "Pegs"
    try: des.computeAll()
    except Exception: pass
    b = root.bRepBodies.itemByName("Block")
    return {"side_body": side_body, "health": cf.healthState,
            "added_mm3": round((b.volume - v0) * 1000, 1) if b else None}

def block_state(des):
    root = des.rootComponent; out = {}
    b = root.bRepBodies.itemByName("Block")
    if b:
        bb = b.boundingBox
        out["block"] = [round(b.volume * 1000, 1), b.faces.count] + [round(v * 10, 2) for v in (
            bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    sk = root.sketches.itemByName("Block profile")
    if sk:
        out["profile_constrained"] = sk.isFullyConstrained
        out["cut_lines"] = {k: [round(v * 10, 2) for v in (l.worldGeometry.startPoint.x, l.worldGeometry.startPoint.z,
                                                            l.worldGeometry.endPoint.x, l.worldGeometry.endPoint.z)]
                            for k, l in box_lines(femoral_occurrence(des)).items()}
    return out
'''

S = r'''
import adsk.core, adsk.fusion, json, struct, time
MODE = __MODE__
FEMORAL = __FEMORAL__
NAME = __NAME__
MSG = __MSG__
TARGET = __TARGET__
OUTDIR = __OUTDIR__
__BLOCK__

def write_stl(path, body):
    calc = body.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
    m = calc.calculate()
    c, t = list(m.nodeCoordinatesAsDouble), list(m.nodeIndices)
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", len(t) // 3))
        for i in range(len(t) // 3):
            f.write(struct.pack("<3f", 0, 0, 0))
            for k in range(3):
                j = t[3*i+k]; f.write(struct.pack("<3f", c[3*j]*10, c[3*j+1]*10, c[3*j+2]*10))
            f.write(b"\0\0")

def health(des):
    tl = des.timeline
    return [[tl.item(i).name, tl.item(i).healthState, tl.item(i).errorOrWarningMessage.split("<")[0][:150]]
            for i in range(tl.count) if tl.item(i).healthState]

def run(_c):
    app = adsk.core.Application.get(); active = app.activeDocument
    out = {}
    fem = app.data.findFileById(FEMORAL)
    if MODE == "base":
        if TARGET: fem = app.data.findFileById(TARGET)
        folder = fem.parentFolder
        if any(f.name == NAME for f in folder.dataFiles):
            raise RuntimeError(NAME + " exists already in " + folder.name)
        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        des.designType = adsk.fusion.DesignTypes.ParametricDesignType
        try:
            occ = des.rootComponent.occurrences.addByInsert(fem, adsk.core.Matrix3D.create(), True)
            out["femoral"] = [fem.name, fem.versionNumber]
            out["params"] = block_params(des)
            out["make"] = make_block(des)
            out["pegs"] = block_pegs(des, "Main Body")
            occ.isLightBulbOn = False
            des.computeAll()
            out["health"] = health(des); out["state"] = block_state(des)
            write_stl(OUTDIR + "\\base_block.stl", des.rootComponent.bRepBodies.itemByName("Block"))
            if not [h for h in out["health"] if h[1] == 2]:
                doc.saveAs(NAME, folder, MSG, "")
                for _ in range(120):
                    hit = [f for f in folder.dataFiles if f.name == NAME and f.id.startswith("urn:")]
                    if hit: out["id"] = hit[0].id; break
                    time.sleep(1); adsk.doEvents()
        finally:
            doc.close(False)
    else:
        df = app.data.findFileById(TARGET)
        doc = app.documents.open(df, True)
        des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        try:
            out["version"] = df.versionNumber
            out["refs"] = [[r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name] for r in doc.documentReferences]
            out["health"] = health(des); out["state"] = block_state(des)
            out["params"] = [[p.name, p.expression] for p in des.userParameters]
        finally:
            doc.close(False)
    if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def main(mode, target=None):
    sys.path.insert(0, str(HERE))
    from implant_pipeline import SOURCES
    out = HERE / "block_audit"
    out.mkdir(exist_ok=True)
    script = (S.replace("__BLOCK__", BLOCK).replace("__MODE__", repr(mode))
              .replace("__FEMORAL__", repr(SOURCES["femoral"])).replace("__NAME__", repr(NAME))
              .replace("__MSG__", repr(MSG)).replace("__TARGET__", repr(target))
              .replace("__OUTDIR__", repr(str(out))))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (out / f"block_{mode}{'_' + target[-6:] if target else ''}.json").write_text(json.dumps(res, indent=1))
    if mode == "base" and res.get("id"):
        ids = json.loads(COPIES.read_text())
        ids["sandbox_block" if target else "femoral_block"] = {"id": res["id"], "name": NAME, "femoral": res["femoral"]}
        COPIES.write_text(json.dumps(ids, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=1)[:5000])
