"""Robustness of the tibial plate (and later the insert) under parameter changes.

Works on Driver Tests copies only. `copy` puts copies of the sandbox plate and insert
there; `plate` sweeps the plate's user parameters over the cohort's range and records
health, the body's extents and volume, and exports each body.

  python tray_audit.py copy
  python tray_audit.py plate
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

OUT = HERE / "tray_audit"
IDS = OUT / "copies.json"
SANDBOX = {"plate": "urn:adsk.wipprod:dm.lineage:T8Hb_Ws3Qlaj-oaYqCZTCw",
           "insert": "urn:adsk.wipprod:dm.lineage:6gUVQt7UQiqGiAfqclNQWw"}

COPY = r'''
import adsk.core, adsk.fusion, json, time
SRC = __SRC__
def run(_context: str):
    app = adsk.core.Application.get()
    folder = app.data.findFileById(__V18__).parentFolder
    out = {}
    for k, lid in SRC.items():
        src = app.data.findFileById(lid)
        new = src.copy(folder)
        out[k] = {"id": new.id, "name": new.name, "from_version": src.latestVersionNumber}
    print(json.dumps(out))
'''

SWEEP = r'''
import adsk.core, adsk.fusion, json, struct
LINEAGE = __L__
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
def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(LINEAGE)
    if any(d.isModified for d in app.documents if d.dataFile and d.dataFile.id == LINEAGE):
        raise RuntimeError("copy has unsaved changes")
    res = []
    for case in CASES:
        doc = app.documents.open(df, False)
        rec = {"name": case["name"], "set": case["set"]}
        try:
            des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
            for n, e in case["set"].items():
                des.allParameters.itemByName(n).expression = e
            des.computeAll()
            tl = des.timeline
            bad = []
            for i in range(tl.count):
                it = tl.item(i)
                if it.healthState != 0:
                    try: msg = it.errorOrWarningMessage.split("<")[0][:160]
                    except Exception: msg = ""
                    bad.append([it.name, it.healthState, msg])
            rec["unhealthy"] = bad
            rec["bodies"] = {}
            for b in des.rootComponent.bRepBodies:
                bb = b.boundingBox
                rec["bodies"][b.name] = {"volume_mm3": round(b.volume * 1000, 3), "faces": b.faces.count,
                    "min": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z)],
                    "max": [round(v * 10, 3) for v in (bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}
                if b.isVisible and case.get("export"):
                    calc = b.meshManager.createMeshCalculator()
                    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.HighQualityTriangleMesh)
                    m = calc.calculate()
                    write_stl(OUT + "\\" + case["name"] + "__" + b.name.replace(" ", "_") + ".stl",
                              list(m.nodeCoordinatesAsDouble), list(m.nodeIndices))
            rec["params"] = {p.name: round(p.value * (10 if p.unit == "mm" else 1), 4) for p in des.allParameters}
        except Exception as error:
            rec["error"] = f"{type(error).__name__}: {error}"
        finally:
            doc.close(False)
        res.append(rec)
    dirty = [d.name for d in app.documents if d.isValid and d.dataFile and d.dataFile.id == LINEAGE and d.isModified]
    if active and active.isValid: active.activate()
    json.dump({"results": res, "dirty": dirty}, open(OUT + "\\_last.json", "w"))
    print(json.dumps({"n": len(res), "dirty": dirty}))
'''


def copy():
    OUT.mkdir(exist_ok=True)
    if IDS.exists():
        print("already copied:", IDS.read_text()); return
    txt = call_fusion(COPY.replace("__SRC__", repr(SANDBOX))
                      .replace("__V18__", repr(MODELS["author_v18"]["lineage"])), read_only=False)
    IDS.write_text(txt.strip().splitlines()[-1], encoding="utf-8")
    print(txt)


def sweep(part, cases, tag):
    OUT.mkdir(exist_ok=True)
    lineage = json.loads(IDS.read_text())[part]["id"]
    path = OUT / f"{tag}.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in cases if c["name"] not in allres]
    for i in range(0, len(todo), 8):
        (OUT / "_cases.json").write_text(json.dumps(todo[i:i + 8]), encoding="utf-8")
        print(call_fusion(SWEEP.replace("__L__", repr(lineage))
                          .replace("__CASES__", repr(str(OUT / "_cases.json")))
                          .replace("__OUT__", repr(str(OUT))), read_only=False).strip().splitlines()[-1])
        got = json.loads((OUT / "_last.json").read_text())
        for r in got["results"]:
            allres[r["name"]] = r
        path.write_text(json.dumps(allres, indent=1), encoding="utf-8")
        if got["dirty"]:
            raise SystemExit(f"copy left dirty: {got['dirty']}")
    return allres


def plate_cases():
    cases = [{"name": "design", "set": {}, "export": True}]
    for ml, ap in ((62, 40), (70, 48), (78, 44), (78, 55), (82, 60), (88, 64), (90, 66)):
        cases.append({"name": f"ml{ml}_ap{ap}", "export": True,
                      "set": {"Tibial_Plateau_ML": f"{ml} mm", "Tibial_Plateau_AP": f"{ap} mm"}})
    # Compartments of different depth (medial deeper by 5 mm).
    for ml, ap in ((78, 55), (88, 64)):
        cases.append({"name": f"ml{ml}_ap{ap}_asym", "export": True,
                      "set": {"Tibial_Plateau_ML": f"{ml} mm", "Tibial_Plateau_AP": f"{ap} mm",
                              "Left_Condyle_Diameter": f"{ap - 4} mm",
                              "Right_Condyle_Diameter": f"{ap - 9} mm"}})
    return cases


def report_plate(res):
    print(f"{'case':18s} {'health':10s} {'X (ML)':>16s} {'Y (AP)':>16s} {'Z':>14s} {'vol':>9s}  notes")
    for k, r in res.items():
        if "error" in r:
            print(k, r["error"]); continue
        b = r["bodies"].get("Body1")
        if not b:
            print(k, "no Body1", list(r["bodies"])); continue
        P = r["params"]
        print(f"{k:18s} {('ok' if not r['unhealthy'] else str(len(r['unhealthy'])) + ' warn'):10s} "
              f"{b['min'][0]:7.2f} {b['max'][0]:7.2f}  {b['min'][1]:7.2f} {b['max'][1]:7.2f}  "
              f"{b['min'][2]:6.2f} {b['max'][2]:6.2f} {b['volume_mm3']:9.1f}  "
              f"want X ±{P['Tibial_Plateau_ML'] / 2:.1f}, AP {P['Tibial_Plateau_AP']:.1f}, cond {P['Left_Condyle_Diameter']:.0f}/{P['Right_Condyle_Diameter']:.0f}")
        for u in r["unhealthy"]:
            print("      ", u)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "copy":
        copy()
    elif cmd == "plate":
        report_plate(sweep("plate", plate_cases(), "plate"))
