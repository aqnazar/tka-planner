"""Femoral v19 -> v20 candidate: width and depth driven separately.

v19 scales every length with Femur_ML/82, the J-curve included, while the plan sets the
cut box (C, B, D) from the bone. A femur narrow for its depth (Femur_ML from the bone at
the cuts, 2026-10-09) then gets a J-curve that no longer fits its box, and features fail.

The edit adds one unitless driver,
    Sagittal_Scale = (C + B + D) / 52.1 mm      (the box's depth over the design's: 1 at L3)
and moves every sagittal or vertical length onto it: the J-curve sketch's own lengths and
radii (A, E, F, distal and anterior depths, construction line, R1-R3) and the vertical
dimensions of the coronal sketches (condyle heights d54, d120, d266; flange heights d148,
d149, d257; Posterior_profile_top_offset). Widths stay on Femur_ML. C, B, D keep their
ML/82 defaults; the plan overrides them. At the design (ML 82, box 52.1) nothing changes.

  python femoral_edit.py copy            # copy v19 (Driver Tests) -> Driver Tests "v20 candidate"
  python femoral_edit.py test            # on the candidate, in memory: design, sweep, knees
  python femoral_edit.py save <lineage>  # apply, check unchanged at design, save
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

STATE = HERE / "audit_author_v19" / "v20_candidate.json"
DESIGN_BOX = 26.5 + 10.6 + 15.0
SAG = {
    "R1_Circle_Diameter": "Sagittal_Scale * 33 mm", "R2_Circle_Diameter": "Sagittal_Scale * 71 mm",
    "R3_Circle_Diameter": "Sagittal_Scale * 210 mm",
    "A_Posterior_cut_surface": "Sagittal_Scale * 16 mm", "E_Pin_length": "round(Sagittal_Scale * 16 / 5) * 5 mm",
    "Distal_cut_depth": "Sagittal_Scale * 9 mm", "Anterior_cut_depth": "Sagittal_Scale * 5 mm",
    "Anterior_construction_line": "Sagittal_Scale * 65 mm", "F_Anterior_cut_surface": "Sagittal_Scale * 45 mm",
    "d54": "Sagittal_Scale * 3 mm", "d120": "Sagittal_Scale * 7.3 mm", "d266": "Sagittal_Scale * 5.7 mm",
    "d148": "Sagittal_Scale * 44 mm", "d149": "Sagittal_Scale * 37 mm", "d257": "Sagittal_Scale * 15 mm",
    "Posterior_profile_top_offset": "Sagittal_Scale * 1.901 mm",
}
MSG = ("Claude (v20 candidate): Sagittal_Scale = (C + B + D) / 52.1 mm drives the J-curve's lengths and radii "
       "and the coronal sketches' vertical sizes; widths stay on Femur_ML. Unchanged at the design.")

S = r'''
import adsk.core, adsk.fusion, json, time
LINEAGE = __L__
SAVE = __SAVE__
TESTS = json.load(open(__TESTS__, encoding="utf-8"))
SAG = __SAG__
DESIGN_BOX = __BOX__
MSG = __MSG__

def state(des, side_body="Main Body"):
    root = des.rootComponent
    b = root.bRepBodies.itemByName(side_body)
    tl = des.timeline
    bad = [[tl.item(i).name, tl.item(i).healthState] for i in range(tl.count) if tl.item(i).healthState != 0]
    if b is None: return {"vol": 0, "faces": 0, "unhealthy": bad, "bb": []}
    bb = b.boundingBox
    return {"vol": round(b.volume * 1000, 3), "faces": b.faces.count, "unhealthy": bad,
            "bb": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}

def edit(des):
    up = des.userParameters
    if up.itemByName("Sagittal_Scale") is None:
        up.add("Sagittal_Scale", adsk.core.ValueInput.createByString(
            "( C_Distal_cut_surface + B_Posterior_chamfer_projection + D_Anterior_chamfer_projection ) / %s mm" % DESIGN_BOX),
            "", "Box depth (C + B + D) over the design's 52.1 mm: drives the sagittal and vertical sizes (v20)")
    P = des.allParameters
    for n, e in SAG.items():
        P.itemByName(n).expression = e
    try: des.computeAll()
    except Exception: pass

def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(LINEAGE)
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == LINEAGE:
            if d.isModified: raise RuntimeError("unsaved changes open on this file")
            d.close(False)
    doc = app.documents.open(df, SAVE)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    out = {"version_before": df.latestVersionNumber, "before": state(des)}
    edit(des)
    out["after"] = state(des)
    out["same_at_design"] = out["before"]["vol"] == out["after"]["vol"] and out["before"]["bb"] == out["after"]["bb"]
    if SAVE:
        if out["same_at_design"] and not out["after"]["unhealthy"]:
            doc.save(MSG)
            for _ in range(120):
                if app.data.findFileById(LINEAGE).latestVersionNumber > out["version_before"]: break
                time.sleep(1); adsk.doEvents()
            out["version_after"] = app.data.findFileById(LINEAGE).latestVersionNumber
        else:
            out["not_saved"] = True
        doc.close(False)
    else:
        results = {}
        for name, sets, body in TESTS:
            try:
                for n, e in sets.items():
                    des.allParameters.itemByName(n).expression = e
                try: des.computeAll()
                except Exception: pass
                results[name] = state(des, body)
                results[name]["sag"] = round(des.allParameters.itemByName("Sagittal_Scale").value, 4)
            except Exception as ex:
                results[name] = {"vol": 0, "bb": [], "unhealthy": [str(ex)[:200]]}
        out["tests"] = results
        doc.close(False)
    if active and active.isValid: active.activate()
    print(json.dumps(out))
'''

COPY = r'''
import adsk.core, json, time
SRC = __SRC__
def run(_c):
    app = adsk.core.Application.get()
    src = app.data.findFileById(SRC)
    new = src.copy(src.parentFolder)
    for _ in range(60):
        try:
            f = app.data.findFileById(new.id)
            if f: break
        except Exception: pass
        time.sleep(2); adsk.doEvents()
    print(json.dumps({"id": new.id, "name": new.name, "folder": src.parentFolder.name}))
'''


def tests():
    design = {"C_Distal_cut_surface": 26.5, "B_Posterior_chamfer_projection": 10.6, "D_Anterior_chamfer_projection": 15.0}
    out = []
    for ml in (60, 72, 82, 95):
        sets = {"Femur_ML": f"{ml} mm"}
        sets.update({k: f"{v * ml / 82:.4f} mm" for k, v in design.items()})
        out.append((f"sweep ml{ml}", sets, "Main Body"))
    for j in json.loads((HERE / "patient_builds/jobs_demo.json").read_text()):
        body = "Main Body" if j["side"] == "right" else "Main Body (1)"
        out.append((j["name"], j["femoral"], body))
        # The same box at the old chart width, and at the extremes of width for its depth.
        for ml in (j["femoral_fit"]["chart_femur_ml_mm"], 0.85 * j["femoral_fit"]["femur_ml_mm"]):
            out.append((f"{j['name']} ML {ml:.1f}", dict(j["femoral"], Femur_ML=f"{ml:.4f} mm"), body))
    for j in json.loads((HERE / "v57_driver/jobs.json").read_text()):
        body = "Main Body" if j["side"] == "right" else "Main Body (1)"
        sets = {"Femur_ML": f"{j['femur_ml_mm']} mm"}
        sets.update({k: f"{v} mm" for k, v in j["parameters_mm"].items()})
        out.append((f"cohort {j['name']}", sets, body))
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "copy":
        r = call_fusion(COPY.replace("__SRC__", repr(MODELS["author_v19"]["lineage"])), read_only=False)
        res = json.loads(r.strip().splitlines()[-1])
        STATE.write_text(json.dumps(res, indent=1))
        print(res)
        return
    save = cmd == "save"
    lineage = sys.argv[2] if save else json.loads(STATE.read_text())["id"]
    tf = HERE / "audit_author_v19" / "_v20_tests.json"
    tf.write_text(json.dumps([] if save else tests()), encoding="utf-8")
    script = (S.replace("__L__", repr(lineage)).replace("__SAVE__", repr(save)).replace("__TESTS__", repr(str(tf)))
              .replace("__SAG__", repr(SAG)).replace("__BOX__", repr(f"{DESIGN_BOX:.1f}")).replace("__MSG__", repr(MSG)))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (HERE / "audit_author_v19" / f"v20_{cmd}.json").write_text(json.dumps(res, indent=1))
    print({k: v for k, v in res.items() if k != "tests"})
    for k, v in res.get("tests", {}).items():
        print(f"  {k:22s} sag {v.get('sag', '-')!s:7s} vol {v['vol']:9.1f}  faces {v.get('faces', 0):3d}  bb {v['bb']}  "
              f"{v['unhealthy'] or 'healthy'}")


if __name__ == "__main__":
    main()
