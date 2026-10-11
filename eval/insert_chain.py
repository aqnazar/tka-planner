"""Can the insert be rebuilt on a patient's femoral and plate? (Driver Tests copies only.)

The insert document holds the femoral and plate as xrefs; its articular surface is swept
from the femoral curves and its outline follows the plate wall. Per patient the pipeline
must repoint those xrefs to the patient's saved femoral and plate.

  python insert_chain.py copy-plate     # copy the sandbox plate (v2) into Driver Tests
  python insert_chain.py repoint        # repoint the insert copy, in memory, and report
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

IDS = HERE / "tray_audit/copies.json"

COPY = r'''
import adsk.core, adsk.fusion, json, time
def run(_context: str):
    app = adsk.core.Application.get()
    src = app.data.findFileById("urn:adsk.wipprod:dm.lineage:T8Hb_Ws3Qlaj-oaYqCZTCw")
    folder = app.data.findFileById(__V18__).parentFolder
    new = src.copy(folder)
    print(json.dumps({"id": new.id, "from_version": src.latestVersionNumber}))
'''

REPOINT = r'''
import adsk.core, adsk.fusion, json
INSERT = __INSERT__
FEMORAL = __FEMORAL__
PLATE = __PLATE__
def body_state(des):
    out = {}
    for b in des.rootComponent.bRepBodies:
        bb = b.boundingBox
        out[b.name] = {"vol": round(b.volume * 1000, 3), "faces": b.faces.count,
                       "bb": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}
    return out
def health(des):
    tl = des.timeline
    out = []
    for i in range(tl.count):
        it = tl.item(i)
        if it.healthState:
            try: m = it.errorOrWarningMessage.split("<")[0][:200]
            except Exception: m = ""
            out.append([it.name, it.healthState, m])
    return out
def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(INSERT)
    doc = app.documents.open(df, True)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    out = {"before": body_state(des), "health_before": health(des),
           "occ_before": [(o.name, o.isReferencedComponent) for o in des.rootComponent.occurrences],
           "params": {p.name: p.expression for p in des.userParameters}}
    try:
        fem = app.data.findFileById(FEMORAL); pla = app.data.findFileById(PLATE)
        for o in list(des.rootComponent.occurrences):
            if not o.isReferencedComponent: continue
            n = o.component.name
            target = fem if "Femoral" in n else pla if "Plate" in n else None
            if target: o.replace(target, True)
        des.computeAll()
        out["occ_after"] = [(o.name, o.isReferencedComponent) for o in des.rootComponent.occurrences]
        out["after"] = body_state(des); out["health_after"] = health(des)
        out["refs"] = [(r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name, r.dataFile.id) for r in doc.documentReferences]
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        doc.close(False)
        if active and active.isValid: active.activate()
    print(json.dumps(out))
'''

if __name__ == "__main__":
    ids = json.loads(IDS.read_text())
    if sys.argv[1] == "copy-plate":
        if "plate_v2" in ids:
            print("have", ids["plate_v2"]); sys.exit()
        txt = call_fusion(COPY.replace("__V18__", repr(MODELS["author_v18"]["lineage"])), read_only=False)
        ids["plate_v2"] = json.loads(txt.strip().splitlines()[-1])
        IDS.write_text(json.dumps(ids, indent=1)); print(ids["plate_v2"])
    elif sys.argv[1] == "repoint":
        txt = call_fusion(REPOINT.replace("__INSERT__", repr(ids["insert"]["id"]))
                          .replace("__FEMORAL__", repr(MODELS["author_v19"]["lineage"]))
                          .replace("__PLATE__", repr(ids["plate_v2"]["id"])), read_only=False)
        res = json.loads(txt.strip().splitlines()[-1])
        (HERE / "tray_audit/insert_repoint.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items(): print(k, v)
