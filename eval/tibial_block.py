"""The tibial cutting block, derived from each knee's own plate (author 2026-10-10: "the
same approach" as the femoral block, femoral_block.py).

The old block (Robot TKA / Cutting Block Tibia (L&R) v6) is a 160 x 120 mm slab whose
bottom face is the tibial cut (the shared tibial origin) plus the old plate's keel shape,
drawn by hand; the planner scaled the library block. Here it is a design of its own
(Driver Tests / Cutting Block Tibia) that holds the plate as a reference at the origin:
  Block profile   a square, 2 x Block_Reach, centred on the origin, on the XY plane: the
                  tibial cut, which is the plate's base (checked every build)
  Block           extruded Block_Top up, away from the bone that stays
  Keel            the plate body joined (kept): only its keel and pegs reach below the cut,
                  so the bone gets the holes the knee's plate needs
All three are made again by every build on the knee's plate (as the femoral block). The
plate is modelled on the origin, so the build shifts the block's STL by +Tray_Offset into
the tray frame, as it does the plate's. The planner uses it for the cut and for the shell.

  python tibial_block.py base [plate lineage]    # make and save the design next to the plate
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402

COPIES = HERE / "tray_audit" / "copies.json"
NAME = "Cutting Block Tibia"
MSG = ("Claude: tibial cutting block from the plate (author 2026-10-10): slab 2 x Block_Reach square on the "
       "tibial cut (the plate's base, the shared origin), Block_Top up; the plate body joined for the keel and pegs.")

# Shared with implant_pipeline.py (pasted into its Fusion script).
TBLOCK = r'''
TBLOCK_PARAMS = (("Block_Top", "150 mm", "How far above the tibial cut the block reaches"),
                 ("Block_Reach", "150 mm", "Half the block's width and depth, about the origin"))

def plate_occurrence(des):
    for o in des.rootComponent.occurrences:
        if o.isReferencedComponent and o.component.bRepBodies.itemByName("Body1") is not None \
                and "Plate" in o.component.name:
            return o
    raise RuntimeError("no plate reference")

def plate_base_z(occ):
    """Z of the plate's base: its largest downward planar face (mm)."""
    best = None
    for f in occ.component.bRepBodies.itemByName("Body1").createForAssemblyContext(occ).faces:
        if f.geometry.objectType != adsk.core.Plane.classType(): continue
        ok, n = f.evaluator.getNormalAtPoint(f.pointOnFace)
        if n.z < -0.999 and (best is None or f.area > best.area):
            best = f
    return round(best.pointOnFace.z * 10, 4) if best else None

def tblock_params(des):
    up = des.userParameters; added = []
    for n, e, c in TBLOCK_PARAMS:
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString(e), "mm", c); added.append(n)
    return added

def strip_tblock(des):
    root = des.rootComponent; gone = []
    for coll, name in ((root.features.combineFeatures, "Keel"), (root.features.extrudeFeatures, "Block"),
                       (root.sketches, "Block profile")):
        f = coll.itemByName(name)
        if f is not None:
            f.deleteMe(); gone.append(name)
    return gone

def make_tblock(des):
    root = des.rootComponent
    sk = root.sketches.add(root.xYConstructionPlane); sk.name = "Block profile"
    R = des.userParameters.itemByName("Block_Reach").value
    P = lambda x, y: sk.modelToSketchSpace(adsk.core.Point3D.create(x, y, 0))
    L = sk.sketchCurves.sketchLines
    a = L.addByTwoPoints(P(-R, -R), P(R, -R))
    b = L.addByTwoPoints(a.endSketchPoint, P(R, R))
    c = L.addByTwoPoints(b.endSketchPoint, P(-R, R))
    d = L.addByTwoPoints(c.endSketchPoint, a.startSketchPoint)
    gc = sk.geometricConstraints
    gc.addHorizontal(a); gc.addHorizontal(c); gc.addVertical(b); gc.addVertical(d)
    D = sk.sketchDimensions; o = sk.originPoint
    H = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    V = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for p, orient, tx in ((a.startSketchPoint, H, P(-R / 2, -R - 1)), (a.startSketchPoint, V, P(-R - 1, -R / 2)),
                          (c.startSketchPoint, H, P(R / 2, R + 1)), (c.startSketchPoint, V, P(R + 1, R / 2))):
        D.addDistanceDimension(o, p, orient, tx).parameter.expression = "Block_Reach"
    prof = max(sk.profiles, key=lambda p: p.areaProperties().area)
    ext = root.features.extrudeFeatures
    ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ei.setDistanceExtent(False, adsk.core.ValueInput.createByString("Block_Top"))
    f = ext.add(ei); f.name = "Block"; b = f.bodies.item(0); b.name = "Block"
    if b.boundingBox.maxPoint.z < 0.1:                      # extruded the wrong way: flip
        f.deleteMe()
        ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        ei.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByString("Block_Top")),
                            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        f = ext.add(ei); f.name = "Block"; f.bodies.item(0).name = "Block"
    return {"fully_constrained": sk.isFullyConstrained}

def tblock_keel(des):
    root = des.rootComponent
    occ = plate_occurrence(des)
    block = root.bRepBodies.itemByName("Block"); v0 = block.volume
    tools = adsk.core.ObjectCollection.create()
    tools.add(occ.component.bRepBodies.itemByName("Body1").createForAssemblyContext(occ))
    ci = root.features.combineFeatures.createInput(block, tools)
    ci.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    ci.isKeepToolBodies = True
    cf = root.features.combineFeatures.add(ci); cf.name = "Keel"
    try: des.computeAll()
    except Exception: pass
    b = root.bRepBodies.itemByName("Block")
    return {"health": cf.healthState, "added_mm3": round((b.volume - v0) * 1000, 1) if b else None}

def tblock_state(des):
    root = des.rootComponent; out = {}
    b = root.bRepBodies.itemByName("Block")
    if b:
        bb = b.boundingBox
        out["block"] = [round(b.volume * 1000, 1), b.faces.count] + [round(v * 10, 2) for v in (
            bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    out["plate_base_z_mm"] = plate_base_z(plate_occurrence(des))
    sk = root.sketches.itemByName("Block profile")
    if sk: out["profile_constrained"] = sk.isFullyConstrained
    return out
'''

S = r'''
import adsk.core, adsk.fusion, json, struct, time
PLATE = __PLATE__
NAME = __NAME__
MSG = __MSG__
OUTDIR = __OUTDIR__
__TBLOCK__

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

def run(_c):
    app = adsk.core.Application.get(); active = app.activeDocument
    out = {}
    plate = app.data.findFileById(PLATE)
    folder = plate.parentFolder
    if any(f.name == NAME for f in folder.dataFiles):
        raise RuntimeError(NAME + " exists already in " + folder.name)
    doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    des.designType = adsk.fusion.DesignTypes.ParametricDesignType
    try:
        occ = des.rootComponent.occurrences.addByInsert(plate, adsk.core.Matrix3D.create(), True)
        out["plate"] = [plate.name, plate.versionNumber, folder.name]
        out["params"] = tblock_params(des)
        out["make"] = make_tblock(des)
        out["keel"] = tblock_keel(des)
        occ.isLightBulbOn = False
        des.computeAll()
        tl = des.timeline
        out["health"] = [[tl.item(i).name, tl.item(i).healthState] for i in range(tl.count) if tl.item(i).healthState]
        out["state"] = tblock_state(des)
        write_stl(OUTDIR + "\\base_tibial_block.stl", des.rootComponent.bRepBodies.itemByName("Block"))
        if not [h for h in out["health"] if h[1] == 2] and out["state"]["plate_base_z_mm"] is not None and abs(out["state"]["plate_base_z_mm"]) < 0.001:
            doc.saveAs(NAME, folder, MSG, "")
            for _ in range(120):
                hit = [f for f in folder.dataFiles if f.name == NAME and f.id.startswith("urn:")]
                if hit: out["id"] = hit[0].id; break
                time.sleep(1); adsk.doEvents()
        else:
            out["not_saved"] = True
    finally:
        doc.close(False)
    if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def main(plate=None):
    from implant_pipeline import SOURCES
    out = HERE / "block_audit"
    out.mkdir(exist_ok=True)
    script = (S.replace("__TBLOCK__", TBLOCK).replace("__PLATE__", repr(plate or SOURCES["plate"]))
              .replace("__NAME__", repr(NAME)).replace("__MSG__", repr(MSG)).replace("__OUTDIR__", repr(str(out))))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (out / f"tblock_base{'_sandbox' if plate else ''}.json").write_text(json.dumps(res, indent=1))
    if res.get("id"):
        ids = json.loads(COPIES.read_text())
        ids["sandbox_tibial_block" if plate else "tibial_block"] = {"id": res["id"], "name": NAME, "plate": res["plate"]}
        COPIES.write_text(json.dumps(ids, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[2] if len(sys.argv) > 2 else None), indent=1)[:4000])
