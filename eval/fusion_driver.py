"""Stage (b): build the patient-specific v57 body for every knee in Fusion.

Private: it reads the patient dataset. Knees are named by case id only.

  python fusion_driver.py inputs   # plan every knee, write v57_driver/jobs.json
  python fusion_driver.py run      # drive Fusion, write v57_driver/<knee>.stl + report.json
  python fusion_driver.py run --model author_v7 [knee ...]   # the same on another model

The run works on the *working copy* of the femoral design (Parametric Work) only. For
each knee it sets Femur_ML and the three box lengths, recomputes, checks every timeline
item is healthy, exports Main Body in its own CAD frame, and at the end puts every
parameter it touched back to the expression it read at the start. It never saves.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

OUT = HERE / "v57_driver"
URL = "http://127.0.0.1:27182/mcp"
WORKING_COPY = "urn:adsk.wipprod:dm.lineage:UrLmgsJ_RSSnpPRBRBXOGA"
BOX_PARAMETERS = ("C_Distal_cut_surface", "B_Posterior_chamfer_projection",
                  "D_Anterior_chamfer_projection")
# The J-curve offset sketch's copies of the box lengths (same sketch entities).
OFFSET_COPIES = {"d156": "C_Distal_cut_surface", "d159": "D_Anterior_chamfer_projection",
                 "d168": "B_Posterior_chamfer_projection"}
# The author's K-2 ruling (2026-09-27): link every offset copy to its J-curve original.
# d162 keeps its deliberate 0.2 mm (3D-printing tolerance, ruling D8).
K2_LINKS = dict(OFFSET_COPIES, **{
    "d160": "A_Posterior_cut_surface", "d161": "E_Pin_length",
    "d162": "Distal_cut_depth + 0.2 mm", "d164": "Distal_cut_depth",
    "d165": "Anterior_construction_line",
    "d166": "F_Anterior_cut_surface * ( ( 100 + J_Curve_Offset_Coefficient ) / 100 )",
    "d167": "Anterior_cut_angle"})
# Sandbox v15 dropped d164 (the offset copy of Distal_cut_depth).
K2_LINKS_V15 = {k: v for k, v in K2_LINKS.items() if k != "d164"}
# The author's "scale every constant" instruction and answers 4, 5 and F2/F7 (2026-09-27),
# applied in memory only. For d265 the arc tangent to the side face, which v15 puts at
# (Femur_ML - 1 mm)/2, needs r = (ML - notch)/4 - 0.5 mm (13 at L3); see D265_V15.
SCALE_V15 = {
    "d129": "( Femur_ML / 82 ) * 16", "d131": "( Femur_ML / 82 ) * 16",
    "d266": "( Femur_ML / 82 ) * 5.7", "d257": "( Femur_ML / 82 ) * 15",
    "d264": "( Femur_ML / 82 ) * 1",
    "Notch_Offset_X": "2 mm", "Notch_Fillet_Radius": "Intercondylar_notch_width / 2",
    "Pin_Spacing": "( Femur_ML + Intercondylar_notch_width ) / 2"}
# Not applicable yet: with this d265 the outer posterior arcs flip to the notch side on the
# large knees (the arc end has only a projected body edge to land on) and the condyle tops
# stay 2 mm uncut. Kept as a diagnostic model.
D265_V15 = {"d265": "( Femur_ML - Intercondylar_notch_width ) / 4 - 0.5 mm"}
# Each model is a Fusion file driven the same way. Only the working copy is ever opened
# for v57; author_v7 is a copy of the Author Sandbox femoral v7 in Parametric Work /
# Driver Tests, so the sandbox itself (which the author may have open) is never touched.
MODELS = {
    "v57": {"lineage": WORKING_COPY, "out": "v57_driver", "links": OFFSET_COPIES},
    "author_v7": {"lineage": "urn:adsk.wipprod:dm.lineage:esSXbz2AR8KnkVr26TP9Ew",
                  "out": "author_v7_driver", "links": K2_LINKS},
    "author_v12": {"lineage": "urn:adsk.wipprod:dm.lineage:4wUrgKwXTeqYgAaWuMgS8g",
                   "out": "author_v12_driver", "links": K2_LINKS},
    # Diagnostic: v12 without the Anterior profile clean up cut (suppressed in memory only).
    "author_v12_no16": {"lineage": "urn:adsk.wipprod:dm.lineage:4wUrgKwXTeqYgAaWuMgS8g",
                        "out": "author_v12_no16_driver", "links": K2_LINKS,
                        "suppress": ["Extrude16"]},
    "author_v15": {"lineage": "urn:adsk.wipprod:dm.lineage:JjqVCX86T9aP2IBogkim5g",
                   "out": "author_v15_driver", "links": K2_LINKS_V15},
    "author_v15_scaled": {"lineage": "urn:adsk.wipprod:dm.lineage:JjqVCX86T9aP2IBogkim5g",
                          "out": "author_v15_scaled_driver",
                          "links": dict(K2_LINKS_V15, **SCALE_V15)},
    # Sandbox v16 (2026-09-28): links, scaling, formulas and drafted pins are in the model.
    "author_v16": {"lineage": "urn:adsk.wipprod:dm.lineage:R71-bdWuSneafHoaY9gozA",
                   "out": "author_v16_driver", "links": {}},
    # Sandbox v17: the anterior profile cut's second side is C/2 instead of up to a body vertex.
    "author_v17": {"lineage": "urn:adsk.wipprod:dm.lineage:Exa7cZmdRVqhaUzqkBdAZQ",
                   "out": "author_v17_driver", "links": {}},
    # Sandbox v18: Extrude16 intersects with the (Femur_ML - 1 mm) band.
    "author_v18": {"lineage": "urn:adsk.wipprod:dm.lineage:2VeMgb6wRWyRU0kOzJuu8A",
                   "out": "author_v18_driver", "links": {}},
    # Sandbox v19 (2026-10-09): fitted-spline handle lengths constant, so the flange and
    # condyle outlines scale with Femur_ML.
    "author_v19": {"lineage": "urn:adsk.wipprod:dm.lineage:uSwpl92FRl6MLDFcmfLsMQ",
                   "out": "author_v19_driver", "links": {}},
    # v20 candidate (2026-10-09, Driver Tests copy of v19, femoral_edit.py): Sagittal_Scale =
    # (C + B + D) / 52.1 mm drives the sagittal and vertical sizes; widths stay on Femur_ML.
    "author_v20c": {"lineage": "urn:adsk.wipprod:dm.lineage:B_G-JJBKTOO-sOPXUruLHA",
                    "out": "author_v20c_driver", "links": {}},
    "author_v15_d265": {"lineage": "urn:adsk.wipprod:dm.lineage:JjqVCX86T9aP2IBogkim5g",
                        "out": "author_v15_d265_driver",
                        "links": dict(K2_LINKS_V15, **SCALE_V15, **D265_V15)},
}


def inputs():
    from eval_stage_b import LIBRARY, cases
    from tka_planner.core import femoral_box as fb
    from tka_planner.pipeline import box_design, femoral_cut_box_case, measure_case, plan_case

    design = box_design(LIBRARY)
    jobs = []
    # The reference femur at the author's own design pose: the rule's box there should
    # come back close to v57's own values.
    ref = measure_case(REPO / "data/cases/REF_KJ2/FD1Right.stl",
                       REPO / "data/cases/REF_KJ2/TD1Right.stl", "right", case_id="REF_KJ2")
    df = np.load(HERE / "design_poses_kj.npz")["Df"]
    box = fb.femoral_cut_box(ref.femur, df, medial_sign=1.0, design=design).to_dict()
    jobs.append(job("REF_KJ2_design", "right", df, 82.0, box))
    for name, femur, tibia, side in cases():
        try:
            m = measure_case(femur, tibia, side, case_id=name)
            plan, sizing = plan_case(m, library=LIBRARY)
            box = femoral_cut_box_case(m, plan, LIBRARY)
        except Exception as error:  # noqa: BLE001 - report every knee
            print(name, "failed:", type(error).__name__, error, flush=True)
            continue
        if not box or not box.get("available"):
            print(name, "no box:", (box or {}).get("reason"), flush=True)
            continue
        jobs.append(job(name, side, plan.components["femoral_component"],
                        float(sizing.implant_ml_mm), box))
        print(name, "planned", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    print("wrote", OUT / "jobs.json", len(jobs), "knees")


def job(name, side, pose, femur_ml, box):
    return {
        "name": name,
        "side": side,
        "pose": np.asarray(pose, dtype=float).tolist(),
        "femur_ml_mm": round(femur_ml, 3),
        "parameters_mm": {k: box["cad_parameters"][k] for k in BOX_PARAMETERS},
        "cad_origin_x_mm": box["cad_origin_x_mm"],
        "posterior_cut_x_mm": box["posterior_cut_x_mm"],
        "anterior_cut_x_at_distal_mm": box["anterior_cut_x_at_distal_mm"],
        "flange_height_mm": box["flange_height_mm"],
    }


FUSION_SCRIPT = r'''
import adsk.core, adsk.fusion, json, struct
LINEAGE = __LINEAGE__
JOBS = json.load(open(__JOBS__, encoding="utf-8"))
OUT = __OUT__
BOX = __BOX__
LINKS = __LINKS__
SUPPRESS = __SUPPRESS__

def write_stl(path, c, t):
    n = len(t) // 3
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<3f", 0, 0, 0))
            for k in range(3):
                j = t[3*i+k]; f.write(struct.pack("<3f", c[3*j]*10, c[3*j+1]*10, c[3*j+2]*10))
            f.write(b"\0\0")

def body_state(des):
    b = des.rootComponent.bRepBodies.itemByName("Main Body")
    bb = b.boundingBox
    return b, {"volume_mm3": round(b.volume * 1000, 2),
               "min": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z)],
               "max": [round(v * 10, 3) for v in (bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}

def unhealthy(des):
    """Timeline items that are not healthy: (name, state, message); state 1 = warning."""
    tl = des.timeline
    out = []
    for i in range(tl.count):
        it = tl.item(i)
        if it.healthState != 0:
            try:
                msg = it.errorOrWarningMessage
            except Exception:
                msg = ""
            out.append([it.name, it.healthState, msg])
    return out

def export(b, path):
    calc = b.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
    m = calc.calculate()
    write_stl(path, list(m.nodeCoordinatesAsDouble), list(m.nodeIndices))

def link_offset_copies(des):
    """The J-curve offset sketch (the hidden Offset bodies, whose edges the two sweep
    cuts follow to shape the articular surface) carries its own copies of the box
    lengths. Link each to its J-curve original (spec 3.1, rule L), or the outer
    surface stays at the design while the box moves."""
    linked = {}
    for copy, original in LINKS.items():
        p = des.allParameters.itemByName(copy)
        linked[copy] = [p.expression, original]
        p.expression = original
    return linked

def fresh(app, data_file):
    """The working copy as saved in the cloud: close any open, unsaved copy first."""
    for d in list(app.documents):
        if d.dataFile and d.dataFile.id == LINEAGE:
            d.close(False)
    doc = app.documents.open(data_file, True)
    return doc, adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))

def run(_context: str):
    app = adsk.core.Application.get()
    active = app.activeDocument
    if any(d.isModified for d in app.documents if d.dataFile and d.dataFile.id == LINEAGE):
        raise RuntimeError("An instance of the working copy has unsaved changes; not touching it.")
    was_open = any(d.dataFile and d.dataFile.id == LINEAGE for d in app.documents)
    data_file = app.data.findFileById(LINEAGE)
    report = {"was_open": was_open, "version": data_file.versionNumber, "knees": []}
    try:
        for job in JOBS:
            doc, des = fresh(app, data_file)
            b, design_state = body_state(des)
            if not report.get("design_state"):
                report["design_state"] = design_state
                export(b, OUT + "\\v57_design.stl")
            rec = {"name": job["name"]}
            try:
                tl = des.timeline
                for i in range(tl.count):
                    if tl.item(i).name in SUPPRESS:
                        tl.item(i).isSuppressed = True
                rec["linked"] = link_offset_copies(des)
                des.userParameters.itemByName("Femur_ML").expression = f"{job['femur_ml_mm']:.4f} mm"
                for n, v in job["parameters_mm"].items():
                    des.allParameters.itemByName(n).expression = f"{v:.4f} mm"
                des.computeAll()
                bad = unhealthy(des)
                b, state = body_state(des)
                rec.update({"unhealthy": bad, **state})
                if not any(s == 2 for _, s, _ in bad):
                    export(b, OUT + "\\" + job["name"] + ".stl")
                    rec["exported"] = True
            except Exception as error:
                rec["error"] = f"{type(error).__name__}: {error}"
            report["knees"].append(rec)
    finally:
        doc, des = fresh(app, data_file)
        _, state = body_state(des)
        report["after"] = {"modified": doc.isModified, "version": doc.dataFile.versionNumber,
                           "latest_version": doc.dataFile.latestVersionNumber, **state}
        # Every loaded instance counts, including the copy the working-copy insert and
        # assembly hold as a child: closing that one doesn't unload it.
        dirty = [d.name for d in app.documents
                 if d.isValid and d.dataFile and d.dataFile.id == LINEAGE and d.isModified]
        report["modified_instances"] = dirty
        report["restored"] = (not dirty and state == report.get("design_state", state)
                              and not unhealthy(des))
        report["saved"] = doc.dataFile.latestVersionNumber != report["version"]
        if not was_open and not dirty:
            doc.close(False)
        if active and active.isValid:
            active.activate()
    print(json.dumps(report))
'''


def call_fusion(script, read_only):
    hdr = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def post(body, sid=None):
        h = dict(hdr, **({"Mcp-Session-Id": sid} if sid else {}))
        r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), h),
                                   timeout=900)
        return r.headers.get("Mcp-Session-Id"), r.read().decode("utf-8")

    sid, _ = post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "tka", "version": "0"}}})
    post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    _, out = post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "fusion_mcp_execute",
                              "arguments": {"featureType": "script",
                                            "object": {"readOnly": read_only,
                                                       "script": script}}}}, sid)
    lines = [ln[5:] for ln in out.splitlines() if ln.startswith("data:")] or [out]
    res = json.loads(lines[-1])
    if "error" in res:
        raise RuntimeError(res["error"])
    text = "".join(c.get("text", "") for c in res["result"]["content"])
    try:
        reply = json.loads(text)
    except ValueError:
        return text
    if isinstance(reply, dict):
        if reply.get("error") or reply.get("success") is False:
            raise RuntimeError(f"Fusion: {reply.get('error')} | {reply.get('message', '')[-2000:]}")
        return reply.get("message", text)
    return text


def run(names=None, model="v57"):
    jobs = json.loads((OUT / "jobs.json").read_text(encoding="utf-8"))
    if names:
        jobs = [j for j in jobs if j["name"] in names]
    m = MODELS[model]
    out = HERE / m["out"]
    out.mkdir(exist_ok=True)
    # The MCP server drops scripts with very long lines, so the jobs go by file.
    (out / "run_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    script = (FUSION_SCRIPT.replace("__LINEAGE__", repr(m["lineage"]))
              .replace("__JOBS__", repr(str(out / "run_jobs.json")))
              .replace("__OUT__", repr(str(out)))
              .replace("__BOX__", repr(BOX_PARAMETERS))
              .replace("__LINKS__", repr(m["links"]))
              .replace("__SUPPRESS__", repr(m.get("suppress", []))))
    text = call_fusion(script, read_only=False)
    report = json.loads(text.strip().splitlines()[-1])
    report["model"] = model
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "knees"}))
    for k in report["knees"]:
        if "error" in k:
            print(k["name"], "ERROR", k["error"])
            continue
        print(k["name"], "unhealthy:",
              [(n, s, m[:120]) for n, s, m in k["unhealthy"]] or "none",
              "vol", k["volume_mm3"], "min", k["min"], "max", k["max"])


if __name__ == "__main__":
    if sys.argv[1] == "inputs":
        inputs()
    else:
        args = sys.argv[2:]
        model = "v57"
        if args[:1] == ["--model"]:
            model, args = args[1], args[2:]
        run(args, model)
