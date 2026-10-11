"""Port the Driver Tests designs into the Author Sandbox (author OK, 2026-10-10 22:28).

  femoral  v19 -> + the v20 candidate (Sagittal_Scale drives the J-curve and vertical sizes)
  plate    v2  -> + the keel rule, Tray offset suppressed (plate on the origin)
  insert   v2  -> references on the sandbox femoral and plate as saved here; snap pins,
                  Snap Lock, Top Connection (1), Fillet2 removed; outline from the plate
                  (insert_edit); femoral offset parameters (0 at the design); finishing
                  features (insert_finish)
  block    new -> Cutting Block Femoral on the sandbox femoral (femoral_block.py base)
The TKA Assembly is not touched.

Parameters come over with implant_pipeline's sync (user parameters added; a model parameter
only where it belongs to the same feature). Each part is checked against its Driver Tests
source before saving; nothing is saved with an error.

  python sandbox_port.py plan                   # what each part would change (read only)
  python sandbox_port.py femoral|plate|insert
  python sandbox_port.py dt_insert              # the Driver Tests insert onto the Driver Tests femoral and plate
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_diff import ALIASES  # noqa: E402
from fusion_driver import call_fusion  # noqa: E402
from implant_pipeline import BUILD, SOURCES  # noqa: E402

SANDBOX = {"femoral": ALIASES["sandbox-femoral"], "plate": ALIASES["sandbox-plate"],
           "insert": ALIASES["sandbox-insert"]}
MSG = {
    "femoral": "Claude (author OK 2026-10-10): v20 from Driver Tests: Sagittal_Scale = (C + B + D) / 52.1 mm drives the "
               "J-curve's lengths and radii and the coronal sketches' vertical sizes; Femur_ML drives the widths.",
    "plate": "Claude (author OK 2026-10-10): from Driver Tests: keel rule d31 = min(AP/44*2; (Wedge_Center_Diameter - "
             "Wedge_Thickness)/2 - 0.25 mm); Tray offset suppressed, plate on the origin (the insert's femoral takes "
             "the offset).",
    "insert": "Claude (author OK 2026-10-10): Condylar Profile remade with the femoral's condyle curve projected (linked, so the dish follows the femoral), box above; outline bridge arcs tangent; from Driver Tests / the author's REF_KJ2 edit: references on the "
              "sandbox femoral and plate; snap pins, Snap Lock, Top Connection (1), Fillet2 removed; outline from the "
              "plate with the arc bridge; femoral at -Tray_Offset (0 here); Back, Top, Bottom fillets and slot "
              "clearances by rule.",
    "dt_insert": "Claude: Condylar Profile remade on the femoral's coronal plane with the femoral's condyle curve projected (linked; redefining the old sketch had frozen it at the sandbox femoral v1), box above the curve. References on the Driver Tests femoral (v20 candidate) and plate (v3) instead of the "
                 "sandbox's v1 of each; outline, femoral offset and finishing remade on them.",
}
HELPERS = BUILD[BUILD.index("def source_params"):BUILD.index("def find(app")]

S = r'''
import adsk.core, adsk.fusion, json, time
MODE = __MODE__
SB = __SB__
SRC = __SRC__
MSG = __MSG__
__HELPERS__
__OUTLINE__
__FINISH__

def plan_sync(des, src, skip):
    add, change = [], []
    for n, (e, u, c, user, o) in src.items():
        if n in skip: continue
        p = des.allParameters.itemByName(n)
        if p is None:
            if user: add.append([n, e])
            continue
        if p.expression == e: continue
        if not user and owner(p) != o: continue
        change.append([n, p.expression, e, o])
    return {"add": add, "change": change}

def state(des):
    tl = des.timeline; root = des.rootComponent
    out = {"unhealthy": [[tl.item(i).name, tl.item(i).healthState, tl.item(i).errorOrWarningMessage.split("<")[0][:120]]
                         for i in range(tl.count) if tl.item(i).healthState and not tl.item(i).isSuppressed],
           "suppressed": [tl.item(i).name for i in range(tl.count) if tl.item(i).isSuppressed], "bodies": {}}
    for b in root.bRepBodies:
        bb = b.boundingBox
        out["bodies"][b.name] = [round(b.volume * 1000, 2), b.faces.count] + [round(v * 10, 3) for v in (
            bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    return out

def opened(app, lid, visible):
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == lid:
            if d.isModified: raise RuntimeError(d.name + " is open with unsaved changes")
            d.close(False)
    return app.documents.open(app.data.findFileById(lid), visible)

def save(app, doc, lid, msg):
    v0 = app.data.findFileById(lid).latestVersionNumber
    doc.save(msg)
    for _ in range(240):
        v = app.data.findFileById(lid).latestVersionNumber
        if v > v0: return v
        time.sleep(1); adsk.doEvents()
    return v0

def run(_c):
    app = adsk.core.Application.get(); active = app.activeDocument
    out = {}
    try:
        if MODE == "plan":
            for part in ("femoral", "plate", "insert"):
                src = source_params(app, app.data.findFileById(SRC[part]))
                doc = opened(app, SB[part], False)
                try:
                    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
                    out[part] = {"plan": plan_sync(des, src, {}), "state": state(des),
                                 "timeline": [des.timeline.item(i).name for i in range(des.timeline.count)]}
                finally:
                    doc.close(False)
                d2 = opened(app, SRC[part], False)
                try:
                    des2 = adsk.fusion.Design.cast(d2.products.itemByProductType("DesignProductType"))
                    out[part]["source_state"] = state(des2)
                    out[part]["source_timeline"] = [des2.timeline.item(i).name for i in range(des2.timeline.count)]
                finally:
                    d2.close(False)
        elif MODE in ("femoral", "plate"):
            src = source_params(app, app.data.findFileById(SRC[MODE]))
            doc = opened(app, SB[MODE], True)
            des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
            try:
                out["before"] = state(des)
                out["sync"] = sync(des, src, {})
                if MODE == "plate":
                    out["tray_offset_move"] = plate_at_origin(des)
                compute(des)
                out["after"] = state(des)
                if not [h for h in out["after"]["unhealthy"] if h[1] == 2] and not out["sync"]["sync_failed"]:
                    out["version"] = save(app, doc, SB[MODE], MSG)
                else:
                    out["not_saved"] = True
            finally:
                doc.close(False)
        elif MODE == "dt_insert":
            # The Driver Tests insert on the Driver Tests femoral and plate (it still held
            # the sandbox's v1 of each; the builds repoint it per knee either way).
            fem, pla = app.data.findFileById(SRC["femoral"]), app.data.findFileById(SRC["plate"])
            want = {fem.id: fem.latestVersionNumber, pla.id: pla.latestVersionNumber}
            doc = opened(app, SRC["insert"], True)
            des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
            try:
                out["before"] = state(des)
                out["strip"] = strip_insert(des)
                errs = []
                for o in list(des.rootComponent.occurrences):
                    if not o.isReferencedComponent: continue
                    n = o.component.name
                    t = fem if "Femoral" in n else pla if "Plate" in n else None
                    if t is None: continue
                    try: o.replace(t, True)
                    except Exception as ex: errs.append(str(ex)[:150])
                for _ in range(40):
                    stale = [r for r in doc.documentReferences if r.dataFile.id not in want or r.dataFile.versionNumber != want[r.dataFile.id]]
                    if not stale: break
                    for r in stale:
                        try: r.getLatestVersion()
                        except Exception as ex: errs.append(str(ex)[:120])
                    compute(des); time.sleep(1); adsk.doEvents()
                compute(des)
                out["refs"] = [[r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name] for r in doc.documentReferences]
                out["ref_errors"] = errs
                out["outline"] = rebuild_outline(des)
                out["tray"] = tray_offset(des)
                compute(des)
                out["finish"] = add_finish(des)
                out["after"] = state(des)
                stale = [r for r in doc.documentReferences if r.dataFile.id not in want or r.dataFile.versionNumber != want[r.dataFile.id]]
                if not [h for h in out["after"]["unhealthy"] if h[1] == 2] and not out["finish"]["failed"] and not stale:
                    out["version"] = save(app, doc, SRC["insert"], MSG)
                else:
                    out["not_saved"] = True
            finally:
                doc.close(False)
        elif MODE == "insert":
            src = source_params(app, app.data.findFileById(SRC["insert"]))
            doc = opened(app, SB["insert"], True)
            des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
            try:
                out["before"] = state(des)
                out["strip"] = strip_insert(des)
                out["sync"] = sync(des, src, {})
                want = {SB["femoral"]: app.data.findFileById(SB["femoral"]).latestVersionNumber,
                        SB["plate"]: app.data.findFileById(SB["plate"]).latestVersionNumber}
                errs = []
                for _ in range(40):
                    stale = [r for r in doc.documentReferences
                             if r.dataFile.id in want and r.dataFile.versionNumber != want[r.dataFile.id]]
                    if not stale: break
                    for r in stale:
                        try: r.getLatestVersion()
                        except Exception as ex: errs.append(str(ex)[:120])
                    compute(des); time.sleep(1); adsk.doEvents()
                compute(des)
                out["refs"] = [[r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name]
                               for r in doc.documentReferences]
                out["ref_errors"] = errs
                out["outline"] = rebuild_outline(des)
                out["tray"] = tray_offset(des)
                compute(des)
                out["finish"] = add_finish(des)
                out["after"] = state(des)
                stale = [r for r in doc.documentReferences
                         if r.dataFile.id in want and r.dataFile.versionNumber != want[r.dataFile.id]]
                if not [h for h in out["after"]["unhealthy"] if h[1] == 2] and not out["finish"]["failed"] and not stale:
                    out["version"] = save(app, doc, SB["insert"], MSG)
                else:
                    out["not_saved"] = True
            finally:
                doc.close(False)
    finally:
        if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def main(mode):
    from insert_edit import OUTLINE
    from insert_finish import FINISH
    script = (S.replace("__HELPERS__", HELPERS).replace("__OUTLINE__", OUTLINE).replace("__FINISH__", FINISH)
              .replace("__MODE__", repr(mode)).replace("__SB__", repr(SANDBOX))
              .replace("__SRC__", repr(SOURCES)).replace("__MSG__", repr(MSG.get(mode, ""))))
    res = json.loads(call_fusion(script, read_only=(mode == "plan")).strip().splitlines()[-1])
    (HERE / "block_audit" / f"sandbox_port_{mode}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1)[:6000])
