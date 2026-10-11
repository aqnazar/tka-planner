"""Tibial plate edits for stage (e), author OK 2026-10-09.

1. Tray_Offset_X / Tray_Offset_Y: a final "Tray offset" move translates the whole plate
   (outline, notch, keel, pins, ledge) off the shared origin. The sketches stay as they
   were: three of them (Insertion Pins, Wedge Profile, Ledge) are tied to the origin.
2. Pins under the condyle centres: Pin_Center_Offset_ML = Intercondylar_Distance/2 + 3 mm
   (22 in the design), not scaled with the AP.
3. Pin_Draft2 = Pin_Length - Pin_Chamfer and Pin_Chamfer = Pin_Diameter * 2.5/6: the draft
   ran to the tip chamfer in the design (6.5 + 2.5 = 9); the rounded pin sizes broke that.
4. The notch-corner fillet (d11) is capped at the posterior edge left beside the notch.
5. The keel's Wedge Fillet radius (d31) scales with the AP like the rest of the keel
   (2 mm at AP 44), capped at (Wedge_Center_Diameter - Wedge_Thickness)/2 - 0.25 mm,
   the room where the wing meets the centre post: a constant 2 mm failed on the small
   demo knee (AP 37) and at AP 40.

  python plate_edit.py test            # on the Driver Tests copy, in memory: checks only
  python plate_edit.py save <lineage>  # apply, check, save (sandbox only on the author's OK)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402

COPY = json.loads((HERE / "tray_audit/copies.json").read_text())["plate"]["id"]
MSG = ("Claude: 'Tray offset' move (Tray_Offset_X/Y) places the whole plate off the shared origin; "
       "pins under the condyle centres (Intercondylar_Distance/2 + 3 mm); "
       "Pin_Draft2 = Pin_Length - Pin_Chamfer and Pin_Chamfer = Pin_Diameter * 2.5/6 (rounded pin sizes made the draft and chamfer fail); notch fillet capped at the edge left beside the notch; keel Wedge Fillet radius scales with the AP (2 mm at AP 44), capped by the wing-to-post room.")

S = r'''
import adsk.core, adsk.fusion, json, time
LINEAGE = __L__
SAVE = __SAVE__
TESTS = json.load(open(__TESTS__, encoding="utf-8"))
MSG = __MSG__

def state(des):
    b = des.rootComponent.bRepBodies.item(0); bb = b.boundingBox
    tl = des.timeline
    bad = [[tl.item(i).name, tl.item(i).healthState] for i in range(tl.count) if tl.item(i).healthState != 0]
    return {"vol": round(b.volume * 1000, 3), "faces": b.faces.count, "unhealthy": bad,
            "bb": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}

def edit(des):
    root = des.rootComponent
    up = des.userParameters
    for n, comment in (("Tray_Offset_X", "Plate centre from the shared origin, +X patient-left (stage e)"),
                       ("Tray_Offset_Y", "Plate centre from the shared origin, +Y posterior (stage e)")):
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString("0 mm"), "mm", comment)
    P = des.allParameters
    P.itemByName("Pin_Draft2").expression = "Pin_Length - Pin_Chamfer"
    # The tip chamfer follows the rounded pin diameter (design 2.5 on 6), not the AP.
    P.itemByName("Pin_Chamfer").expression = "Pin_Diameter * 2.5 / 6"
    # Pins under the condyle centres (design: 19 + 3 = 22), not scaled with the AP.
    P.itemByName("Pin_Center_Offset_ML").expression = "round(( Intercondylar_Distance / 2 + 3 mm ) / 1 mm) * 1 mm"
    # The notch-corner fillet can't exceed the posterior edge left beside the notch.
    P.itemByName("d11").expression = ("min(( Tibial_Plateau_AP / 44 ) * 8; Intercondylar_Distance / 2"
                                      " - Intercondylar_Notch_Width / 2 - 1 mm)")
    # The keel's fillet scales with the keel (round(AP/44 * k) for its other sizes), and
    # stays inside the room between the wing and the centre post (rounded sizes).
    P.itemByName("d31").expression = ("min(( Tibial_Plateau_AP / 44 ) * 2; ( Wedge_Center_Diameter"
                                      " - Wedge_Thickness ) / 2 - 0.25 mm)")
    # The whole plate moves rigidly to its offset (outline, notch, keel, pins, ledge).
    if not any(f.name == "Tray offset" for f in root.features.moveFeatures):
        bodies = adsk.core.ObjectCollection.create()
        for b in root.bRepBodies:
            bodies.add(b)
        mi = root.features.moveFeatures.createInput2(bodies)
        mi.defineAsTranslateXYZ(adsk.core.ValueInput.createByString("Tray_Offset_X"),
                                adsk.core.ValueInput.createByString("Tray_Offset_Y"),
                                adsk.core.ValueInput.createByString("0 mm"), True)
        mf = root.features.moveFeatures.add(mi)
        mf.name = "Tray offset"
    des.computeAll()
    return root.sketches.itemByName("Base Sketch").isFullyConstrained


def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(LINEAGE)
    mine = [d for d in app.documents if d.isValid and d.dataFile and d.dataFile.id == LINEAGE]
    if any(d.isModified for d in mine):
        raise RuntimeError("unsaved changes open on this file")
    opened = not mine
    doc = app.documents.open(df, True) if opened else mine[0]
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    out = {"version_before": df.latestVersionNumber, "before": state(des)}
    out["fully_constrained"] = edit(des)
    out["after"] = state(des)
    same = out["before"]["vol"] == out["after"]["vol"] and out["before"]["bb"] == out["after"]["bb"]
    out["same_at_design"] = same
    results = {}
    if not SAVE:
        doc.close(False)
        for name, sets in TESTS:
            doc = app.documents.open(df, False)
            des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
            try:
                edit(des)
                for n, e in sets.items():
                    des.allParameters.itemByName(n).expression = e
                des.computeAll()
                results[name] = state(des)
            except Exception as ex:
                results[name] = {"vol": 0, "bb": [], "unhealthy": [str(ex)]}
            doc.close(False)
        out["tests"] = results
    else:
        if same and not out["after"]["unhealthy"]:
            doc.save(MSG)
            for _ in range(120):
                if df.latestVersionNumber > out["version_before"]: break
                time.sleep(1); adsk.doEvents()
            out["version_after"] = df.latestVersionNumber
            if opened: doc.close(False)
        else:
            doc.close(False); out["not_saved"] = True
    if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def tests():
    out = [("offset +5,-3", {"Tray_Offset_X": "5 mm", "Tray_Offset_Y": "-3 mm"}),
           ("offset -5.6,+9.5", {"Tray_Offset_X": "-5.6 mm", "Tray_Offset_Y": "9.5 mm"})]
    rule = json.loads((HERE / "tray_driver/tray_params_shifted.json").read_text())
    for r in rule:
        sets = {k: f"{v} mm" for k, v in r["params"].items()}
        sets["Tray_Offset_X"] = f"{r['shift_mm'][0]} mm"
        sets["Tray_Offset_Y"] = f"{r['shift_mm'][1]} mm"
        out.append((r["name"], sets))
    for j in json.loads((HERE / "patient_builds/jobs_demo.json").read_text()):
        out.append((j["name"], {k: v for k, v in j["plate"].items()}))
    for ml, ap in ((62, 40), (78, 55), (88, 64), (90, 66)):
        out.append((f"ml{ml}_ap{ap}", {"Tibial_Plateau_ML": f"{ml} mm", "Tibial_Plateau_AP": f"{ap} mm"}))
    return out


if __name__ == "__main__":
    save = sys.argv[1] == "save"
    lineage = sys.argv[2] if save else COPY
    tests_file = HERE / "tray_audit" / "_plate_tests.json"
    tests_file.write_text(json.dumps([] if save else tests()), encoding="utf-8")
    script = (S.replace("__L__", repr(lineage)).replace("__SAVE__", repr(save))
              .replace("__TESTS__", repr(str(tests_file))).replace("__MSG__", repr(MSG)))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (HERE / "tray_audit" / f"plate_edit_{'save' if save else 'test'}.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "tests"})
    for k, v in res.get("tests", {}).items():
        print(f"  {k:18s} vol {v['vol']:9.1f}  bb {v['bb']}  {v['unhealthy'] or 'healthy'}")
