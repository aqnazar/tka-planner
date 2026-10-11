"""The Author Sandbox TKA Assembly on the ported parts (author 2026-10-10).

  probe   read only: occurrences, transforms, references, pending snapshot
  update  references on the latest sandbox Tibial Plate and Tibial Insert New (which
          carries femoral v20 and plate v3); the two cutting blocks added, hidden, where
          their parts sit (femoral block on the femoral inside the insert, tibial block on
          the plate). Saved only if no occurrence moved and nothing errs.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_diff import ALIASES  # noqa: E402
from fusion_driver import call_fusion  # noqa: E402

IDS = json.loads((HERE / "tray_audit" / "copies.json").read_text())
MSG = ("Claude (author OK 2026-10-10): references on the latest sandbox parts (Tibial Plate, Tibial Insert New "
       "with femoral v20); Cutting Block Femoral and Cutting Block Tibia, hidden, on their parts. "
       "No occurrence moved.")

S = r'''
import adsk.core, adsk.fusion, json, time
MODE = __MODE__
ASM = __ASM__
FBLOCK = __FBLOCK__
TBLOCK = __TBLOCK__
MSG = __MSG__

def mat(m): return [round(v * (10 if i in (3, 7, 11) else 1), 4) for i, v in enumerate(m.asArray())]

def occs(des):
    out = {}
    for o in des.rootComponent.allOccurrences:
        out[o.fullPathName] = {"m": mat(o.transform2), "visible": o.isLightBulbOn,
                               "ref": o.isReferencedComponent}
    return out

def health(des):
    tl = des.timeline
    return [[tl.item(i).name, tl.item(i).healthState] for i in range(tl.count) if tl.item(i).healthState]

def refs(doc):
    return [[r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name, r.dataFile.id] for r in doc.documentReferences]

def moved(a, b):
    out = []
    for k, v in a.items():
        if k in b and max(abs(x - y) for x, y in zip(v["m"], b[k]["m"])) > 1e-3:
            out.append(k)
    return out

def run(_c):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(ASM)
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == ASM:
            if d.isModified: raise RuntimeError("assembly open with unsaved changes")
            d.close(False)
    doc = app.documents.open(df, True)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    out = {"version": df.versionNumber}
    try:
        try: out["pending_snapshot"] = des.snapshots.hasPendingSnapshot
        except Exception as ex: out["pending_snapshot"] = str(ex)[:80]
        out["refs"] = refs(doc); out["health"] = health(des)
        before = occs(des); out["occurrences"] = before
        if MODE == "update":
            errs = []
            for _ in range(30):
                stale = [r for r in doc.documentReferences if r.dataFile.parentFolder.name == "Author Sandbox"
                         and r.dataFile.versionNumber != app.data.findFileById(r.dataFile.id).latestVersionNumber]
                if not stale: break
                for r in stale:
                    try: r.getLatestVersion()
                    except Exception as ex: errs.append(str(ex)[:120])
                time.sleep(1); adsk.doEvents()
            out["ref_errors"] = errs
            after = occs(des)
            out["moved_by_update"] = moved(before, after)
            root = des.rootComponent
            ins = [o for o in root.occurrences if "Tibial Insert New" in o.component.name][0]
            fem = [o for o in ins.childOccurrences if "Femoral" in o.component.name][0]
            pla = [o for o in root.occurrences if o.component.name.startswith("Tibial Plate")][0]
            pla_in = [o for o in ins.childOccurrences if "Plate" in o.component.name]
            out["plate_vs_insert_plate"] = (max(abs(x - y) for x, y in zip(mat(pla.transform2), mat(pla_in[0].transform2)))
                                            if pla_in else None)
            added = []
            have = [o.component.name for o in root.occurrences]
            for lid, where, name in ((FBLOCK, fem, "femoral"), (TBLOCK, pla, "tibial")):
                bdf = app.data.findFileById(lid)
                if any(n.startswith(bdf.name + " ") for n in have):
                    continue                      # added already (a second update)
                o = root.occurrences.addByInsert(bdf, where.transform2, True)
                o.isLightBulbOn = False
                added.append([name, o.name, mat(o.transform2), mat(where.transform2)])
            out["added"] = added
            try: des.computeAll()
            except Exception: pass
            final = occs(des)
            out["moved_final"] = moved(before, final)
            out["refs_after"] = refs(doc); out["health_after"] = health(des)
            try: out["pending_after"] = des.snapshots.hasPendingSnapshot
            except Exception: pass
            if not out["moved_final"] and not [h for h in out["health_after"] if h[1] == 2] and not errs:
                v0 = df.latestVersionNumber
                doc.save(MSG)
                for _ in range(240):
                    v = app.data.findFileById(ASM).latestVersionNumber
                    if v > v0: break
                    time.sleep(1); adsk.doEvents()
                out["saved_version"] = app.data.findFileById(ASM).latestVersionNumber
            else:
                out["not_saved"] = True
    finally:
        doc.close(False)
        if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def main(mode):
    script = (S.replace("__MODE__", repr(mode)).replace("__ASM__", repr(ALIASES["sandbox-assembly"]))
              .replace("__FBLOCK__", repr(IDS["sandbox_block"]["id"]))
              .replace("__TBLOCK__", repr(IDS["sandbox_tibial_block"]["id"])).replace("__MSG__", repr(MSG)))
    res = json.loads(call_fusion(script, read_only=(mode == "probe")).strip().splitlines()[-1])
    (HERE / "block_audit" / f"sandbox_assembly_{mode}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "occurrences"}, indent=1)[:5000])
    print("occurrences:", {k: v["m"][3::4][:3] for k, v in r["occurrences"].items()})
