"""Patient-specific femoral, tibial plate and insert, built in Fusion from the plan.

Private (patient-derived). For each knee:
  1. jobs: the femoral box (fusion_driver jobs.json), the tray rule (tray_rule.py,
     outline shifted onto the section), and the insert thickness from the plan's poses
  2. Fusion, in `Parametric Work/Patient Builds/<knee>/`:
     femoral  copy of the v20 candidate -> Femur_ML, C, B, D -> save -> export
     plate    copy of the v2 design  -> tray rule -> save -> export; on the origin, its
                                        Tray offset move suppressed (author 2026-10-10)
     insert   copy of the design     -> xrefs repointed to this knee's femoral and plate,
                                        outline from the plate, femoral at -Tray_Offset,
                                        Insert_Thickness, Distal_Cut, finishing features
                                        (insert_finish.py) -> save -> export
     block    copy of the femoral cutting block (femoral_block.py) -> its femoral reference
                                        repointed to this knee's femoral, so the cut is that
                                        femoral's J-curve; pegs from its side body -> save -> export
     tblock   copy of the tibial cutting block (tibial_block.py) -> its plate reference
                                        repointed to this knee's plate; slab on the cut, the
                                        plate joined for the keel -> save -> export (+Tray_Offset)
     `build(..., only=[parts])` builds just those parts on the knee's saved versions of the rest.
     Second runs reuse the knee's files and save a new version.
  3. STLs: the femoral in its CAD frame; the plate and insert shifted by +Tray_Offset
     into the tray frame (the CAD has them on the origin). build.json per knee.

Insert thickness (author 2026-10-09): the vertical distance between the components.
In the insert document the plate's insert floor is at Z = Plate_Thickness (5) and the
femoral origin at Z = 5 + Insert_Thickness + Distal_Cut, so
    Insert_Thickness = h - 5 - Distal_cut_depth,
h being the femoral origin's height above the tibial cut along the tray normal.

  python implant_pipeline.py jobs
  python implant_pipeline.py build [knee ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import MODELS, call_fusion  # noqa: E402

OUT = HERE / "patient_builds"
PLATE_FLOOR_MM = 5.0          # Plate_Thickness: the insert's floor in the insert document
INSERT_MIN_MM = 6.0
DESIGN_BOX_MM = 26.5 + 10.6 + 15.0   # C + B + D of the L3 design           # flagged, not enforced (author 2026-10-09)
IDS = json.loads((HERE / "tray_audit/copies.json").read_text())
SOURCES = {"femoral": MODELS["author_v20c"]["lineage"],     # Driver Tests copies of the
           "plate": IDS["plate_v2"]["id"],                  # sandbox designs
           "insert": IDS["insert"]["id"],
           "block": IDS["femoral_block"]["id"],
           "tblock": IDS["tibial_block"]["id"]}


def make_job(f, r, t):
    """One knee's three parameter sets. f: fusion_driver job (femoral box), r: tibial
    inputs row (tibial_inputs.py), t: tray rule fit (tray_rule.fit, outline shifted)."""
    ml = f["femur_ml_mm"]
    F, T = np.array(r["femoral_pose"]), np.array(r["tray_pose"])
    n = T[:3, 2]
    h = float(n @ (F[:3, 3] - T[:3, 3]))
    # v20c: Distal_cut_depth = 9 mm x Sagittal_Scale, the box depth over the design's.
    box = f["parameters_mm"]
    distal = 9.0 * (box["C_Distal_cut_surface"] + box["B_Posterior_chamfer_projection"]
                    + box["D_Anterior_chamfer_projection"]) / DESIGN_BOX_MM
    thickness = h - PLATE_FLOOR_MM - distal
    plate = {k: f"{v:.4f} mm" for k, v in t["params"].items()}
    plate["Tray_Offset_X"] = f"{t['shift_mm'][0]:.4f} mm"
    plate["Tray_Offset_Y"] = f"{t['shift_mm'][1]:.4f} mm"
    return {
        "name": r["name"], "side": r["side"],
        "femoral": dict({"Femur_ML": f"{ml:.4f} mm"},
                        **{k: f"{v:.4f} mm" for k, v in f["parameters_mm"].items()}),
        "plate": plate,
        "insert": {"Insert_Thickness": f"{thickness:.4f} mm", "Distal_Cut": f"{distal:.4f} mm",
                   "Femur_ML": f"{ml:.4f} mm",
                   "Tibial_Plateau_ML": plate["Tibial_Plateau_ML"],
                   "Tibial_Plateau_AP": plate["Tibial_Plateau_AP"]},
        "plan": {"h_mm": round(h, 3), "insert_mm": round(thickness, 3), "distal_mm": round(distal, 3),
                 "insert_below_min": thickness < INSERT_MIN_MM,
                 "extra_tibial_resection_for_min_mm": round(max(0.0, INSERT_MIN_MM - thickness), 2),
                 "femoral_tray_tilt_deg": round(float(np.degrees(np.arccos(np.clip(F[:3, 2] @ n, -1, 1)))), 2),
                 "tray_coverage": t["coverage"],
                 "femoral_pose": r["femoral_pose"], "tray_pose": r["tray_pose"]},
    }


def print_jobs(out):
    print(f"{'knee':10s} {'ML':>7s} {'h':>6s} {'insert':>6s} {'tilt':>5s}  tray")
    for j in out:
        p = j["plan"]
        print(f"{j['name']:10s} {j['femoral']['Femur_ML']:>10s} {p['h_mm']:6.2f} {p['insert_mm']:6.2f} "
              f"{p['femoral_tray_tilt_deg']:5.2f}  ML {j['plate']['Tibial_Plateau_ML']} AP {j['plate']['Tibial_Plateau_AP']}"
              + ("   insert < 6 mm" if p["insert_below_min"] else ""))


def jobs():
    fem = {j["name"]: j for j in json.loads((HERE / "v57_driver/jobs.json").read_text())}
    tray = {t["name"]: t for t in json.loads((HERE / "tray_driver/tray_params_shifted.json").read_text())}
    rows = [r for r in json.loads((HERE / "tray_driver/inputs.json").read_text()) if r["name"] in tray]
    out = [make_job(fem[r["name"]], r, tray[r["name"]]) for r in rows]
    OUT.mkdir(exist_ok=True)
    (OUT / "jobs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print_jobs(out)


BUILD = r'''
import adsk.core, adsk.fusion, json, struct, time
JOB = json.load(open(__JOB__, encoding="utf-8"))
SOURCES = __SOURCES__
OUT = __OUT__
PARENT_OF = SOURCES["femoral"]

def write_stl(path, body, shift=(0.0, 0.0)):
    """Binary STL in mm, translated by shift (mm, in X and Y)."""
    sx, sy = shift[0] / 10, shift[1] / 10
    calc = body.meshManager.createMeshCalculator()
    calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.VeryHighQualityTriangleMesh)
    m = calc.calculate()
    c, t = list(m.nodeCoordinatesAsDouble), list(m.nodeIndices)
    n = len(t) // 3
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<3f", 0, 0, 0))
            for k in range(3):
                j = t[3*i+k]; f.write(struct.pack("<3f", (c[3*j]+sx)*10, (c[3*j+1]+sy)*10, c[3*j+2]*10))
            f.write(b"\0\0")

def folder_named(parent, name):
    for f in parent.dataFolders:
        if f.name == name: return f
    return parent.dataFolders.add(name)

def file_in(folder, name, source):
    for f in folder.dataFiles:
        if f.name == name: return f, False
    last = None
    for _ in range(60):
        try:
            return source.copy(folder), True
        except Exception as ex:
            last = ex; time.sleep(2); adsk.doEvents()
    raise RuntimeError(f"copy failed: {last}")

def health(des):
    tl = des.timeline; out = []
    for i in range(tl.count):
        it = tl.item(i)
        if it.healthState:
            try: m = it.errorOrWarningMessage.split("<")[0][:200]
            except Exception: m = ""
            out.append([it.name, it.healthState, m])
    return out

def bodies(des):
    out = {}
    for b in des.rootComponent.bRepBodies:
        bb = b.boundingBox
        out[b.name] = {"visible": b.isVisible, "vol": round(b.volume * 1000, 3),
                       "bb": [round(v * 10, 3) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]}
    return out

OPTIONAL = ("Fillet2",)
__OUTLINE__
__FINISH__
__BLOCK__
__TBLOCK__

def source_params(app, df):
    """Every parameter of a source design: name -> (expression, unit, comment, is_user)."""
    doc = app.documents.open(df, False)
    try:
        des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        out = {}
        P = des.allParameters
        for i in range(P.count):
            p = P.item(i)
            out[p.name] = (p.expression, p.unit, getattr(p, "comment", ""),
                           des.userParameters.itemByName(p.name) is not None, owner(p))
        return out
    finally:
        doc.close(False)

def owner(p):
    """The feature or sketch a model parameter belongs to ("" for user parameters)."""
    try:
        return adsk.fusion.ModelParameter.cast(p).createdBy.name
    except Exception:
        return ""

def sync(des, src, skip):
    """Bring a knee's copy up to its source design: add the source's user parameters it
    lacks, then copy every expression the job does not set. Copies made from an older
    design version pick up its later relations this way, as a new version."""
    up = des.userParameters
    added, changed, failed = [], [], []
    pending = [(n, v) for n, v in src.items() if v[3] and up.itemByName(n) is None]
    for _ in range(3):
        left = []
        for n, (e, u, c, _, o) in pending:
            try:
                up.add(n, adsk.core.ValueInput.createByString(e), u, c); added.append(n)
            except Exception:
                left.append((n, (e, u, c, True, o)))
        pending = left
    failed += [n for n, _ in pending]
    for n, (e, u, c, user, o) in src.items():
        if n in skip: continue
        p = des.allParameters.itemByName(n)
        if p is None or p.expression == e: continue
        # A model parameter only where it belongs to the same feature: the insert's
        # finishing features are remade every build, so their d-numbers drift.
        if not user and owner(p) != o: continue
        try:
            p.expression = e; changed.append(n)
        except Exception:
            failed.append(n)
    return {"sync_added": added, "sync_changed": changed, "sync_failed": failed}

def compute(des):
    try:
        des.computeAll()
    except Exception:
        pass        # the timeline's health says what failed

def find(app, file_id, tries=60):
    """findFileById, retried: a file just copied or saved takes a while to resolve."""
    last = None
    for _ in range(tries):
        try:
            f = app.data.findFileById(file_id)
            if f: return f
        except Exception as ex:
            last = ex
        time.sleep(2); adsk.doEvents()
    raise RuntimeError(f"file {file_id} not found: {last}")

def save_and_wait(doc, df, msg):
    app = adsk.core.Application.get()
    before = df.latestVersionNumber
    doc.save(msg)
    for _ in range(240):
        try:
            now = app.data.findFileById(df.id).latestVersionNumber
        except Exception:
            now = before
        if now > before: return now
        time.sleep(1); adsk.doEvents()
    return before

def build_part(app, df, sets, export, msg, prepare=None, strict=True, finish=None, shift=(0.0, 0.0)):
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == df.id:
            if d.isModified: raise RuntimeError(f"{df.name} has unsaved changes open")
            d.close(False)
    doc = app.documents.open(df, True)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    rec = {}
    try:
        if prepare: rec.update(prepare(des, doc) or {})
        rec["was"] = {}
        for n, e in sets.items():
            p = des.allParameters.itemByName(n)
            if p is None:
                rec.setdefault("missing", []).append(n); continue
            rec["was"][n] = p.expression
            p.expression = e
        compute(des)
        # Fillet2 (insert): a 1 mm round on the anterior top edge, chosen by edge. When
        # that edge is gone on a patient's parts it is suppressed, and the report says so.
        tl = des.timeline
        for i in range(tl.count):
            it = tl.item(i)
            if it.name in OPTIONAL and it.healthState == 2:
                it.isSuppressed = True
                rec.setdefault("suppressed", []).append(it.name)
        if rec.get("suppressed"):
            compute(des)
        if finish:
            rec["finish"] = finish(des)
        rec["health"] = health(des); rec["bodies"] = bodies(des)
        # A warning on the femoral or plate has meant a feature that did not build (the
        # body comes out partial), so those are not saved either. The insert keeps its
        # known Base Sketch warning until its outline is rebuilt.
        errors = [h for h in rec["health"] if h[1] == 2 or (strict and h[1] == 1)]
        if errors:
            rec["saved"] = False
        else:
            for name, path in export:
                bs = des.rootComponent.bRepBodies
                # "*": the largest body (the insert's is renamed when its base is rebuilt).
                b = (max(bs, key=lambda x: x.volume) if name == "*" else bs.itemByName(name))
                if b:
                    write_stl(path, b, shift); rec.setdefault("exported", {})[path] = b.name
                    rec["stl_shift_mm"] = list(shift)
                else:
                    rec.setdefault("export_missing", []).append(name)
            rec["version"] = save_and_wait(doc, df, msg)
            rec["saved"] = True
    finally:
        doc.close(False)
    return rec

def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    src = {k: find(app, v) for k, v in SOURCES.items()}
    src_params = {k: source_params(app, f) for k, f in src.items()}
    work = src["femoral"].parentFolder.parentFolder               # Parametric Work
    knee = folder_named(folder_named(work, "Patient Builds"), JOB["name"])
    report = {"name": JOB["name"], "folder": f"Parametric Work/Patient Builds/{JOB['name']}"}
    tag = f"Patient build {JOB['name']}"
    try:
        fem, new_f = file_in(knee, src["femoral"].name, src["femoral"])
        pla, new_p = file_in(knee, src["plate"].name, src["plate"])
        ins, new_i = file_in(knee, src["insert"].name, src["insert"])
        blk, new_b = file_in(knee, src["block"].name, src["block"])
        tbk, new_t = file_in(knee, src["tblock"].name, src["tblock"])
        report["copied"] = {"femoral": new_f, "plate": new_p, "insert": new_i, "block": new_b, "tblock": new_t}
        # Only these parts are built; the others stay at the knee's saved versions.
        ONLY = JOB.get("only")
        try: PREV = json.load(open(OUT + "\\build.json"))
        except Exception: PREV = {}
        def kept(df, part):
            v = find(app, df.id).latestVersionNumber
            old = PREV.get(part) or {}
            return dict(old, kept=True) if old.get("version") == v else {"version": v, "saved": True, "kept": True}
        def wanted(part): return not ONLY or part in ONLY
        side_body = "Main Body" if JOB["side"] == "right" else "Main Body (1)"
        # The plate and insert are modelled on the origin; their STLs go to the tray frame.
        tray = (float(JOB["plate"]["Tray_Offset_X"].split()[0]), float(JOB["plate"]["Tray_Offset_Y"].split()[0]))
        JOB["insert"]["Tray_Offset_X"] = JOB["plate"]["Tray_Offset_X"]
        JOB["insert"]["Tray_Offset_Y"] = JOB["plate"]["Tray_Offset_Y"]
        report["femoral"] = kept(fem, "femoral") if not wanted("femoral") else build_part(app, fem, JOB["femoral"],
                                       [(side_body, OUT + "\\femoral.stl")], tag + ": femoral from the plan",
                                       prepare=lambda des, doc: sync(des, src_params["femoral"], JOB["femoral"]))
        def plate_prep(des, doc):
            r = sync(des, src_params["plate"], JOB["plate"]); r["tray_offset_move"] = plate_at_origin(des)
            return r
        report["plate"] = kept(pla, "plate") if not wanted("plate") else build_part(app, pla, JOB["plate"],
                                     [("Body1", OUT + "\\plate.stl")], tag + ": tray rule, plate on the origin",
                                     prepare=plate_prep, shift=tray)
        if not (report["femoral"].get("saved") and report["plate"].get("saved")):
            raise RuntimeError("femoral or plate not built; insert skipped")
        # The insert must take the versions just saved: a save publishes asynchronously,
        # and a repoint to a lookup made too early takes the previous version.
        def saved(df, version):
            for _ in range(120):
                f = find(app, df.id)
                if f.latestVersionNumber >= version: return f
                time.sleep(1); adsk.doEvents()
            raise RuntimeError(f"{df.name} v{version} not published")
        fem_v = saved(fem, report["femoral"]["version"]); pla_v = saved(pla, report["plate"]["version"])
        def repoint(des, doc):
            # The design's snap pins and top bridge go, and last build's finishing
            # features, before anything recomputes on new parts (insert_finish.py).
            stripped = strip_insert(des)
            synced = sync(des, src_params["insert"], JOB["insert"])
            done, errs = [], []
            # A replace recomputes the insert; if Fillet2 fails then, Fusion rolls the
            # replace back. Suppress it first, try it again once the parts are in.
            tl = des.timeline
            for i in range(tl.count):
                if tl.item(i).name in OPTIONAL:
                    try:
                        tl.item(i).isSuppressed = True
                    except Exception:
                        pass
            for o in list(des.rootComponent.occurrences):
                if not o.isReferencedComponent: continue
                n = o.component.name
                target = fem_v if "Femoral" in n else pla_v if "Plate" in n else None
                if target is None: continue
                try:
                    o.replace(target.latestVersion if hasattr(target, "latestVersion") else target, True)
                except Exception as ex:      # replace recomputes and raises on a failed feature
                    errs.append(str(ex)[:200])
                done.append(n)
            # Hold the references to the exact versions saved above (a replace can be
            # rolled back, and a reference's own latest-version number can be stale).
            want = {fem.id: report["femoral"]["version"], pla.id: report["plate"]["version"]}
            for _ in range(40):
                stale = [r for r in doc.documentReferences
                         if r.dataFile.id in want and r.dataFile.versionNumber != want[r.dataFile.id]]
                if not stale: break
                for r in stale:
                    try: r.getLatestVersion()
                    except Exception as ex: errs.append("getLatestVersion: " + str(ex)[:100])
                compute(des); time.sleep(1); adsk.doEvents()
            compute(des)
            # The insert's outline from this knee's plate (insert_edit.py).
            outline = rebuild_outline(des)
            tray_pos = tray_offset(des)
            compute(des)
            restored = []
            for i in range(tl.count):
                it = tl.item(i)
                if it.name in OPTIONAL:
                    try:
                        it.isSuppressed = False
                        compute(des)
                    except Exception:
                        pass
                    if it.healthState == 2 or it.isSuppressed:
                        it.isSuppressed = True
                        compute(des)
                    else:
                        restored.append(it.name)
            stale = [(r.dataFile.name, r.dataFile.versionNumber) for r in doc.documentReferences
                     if r.dataFile.id in want and r.dataFile.versionNumber != want[r.dataFile.id]]
            if stale:
                raise RuntimeError(f"insert references old versions: {stale}")
            return {**synced, **stripped, "outline": outline, "femoral_offset": tray_pos, "restored": restored,
                    "repointed": done, "replace_errors": errs,
                    "refs": [(r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name) for r in doc.documentReferences]}
        report["insert"] = kept(ins, "insert") if not wanted("insert") else build_part(app, ins, JOB["insert"],
                                      [("*", OUT + "\\insert.stl")], tag + ": insert on this knee's femoral and plate",
                                      prepare=repoint, strict=False, finish=add_finish, shift=tray)
        # The femoral cutting block: its femoral reference on this knee's femoral, so its
        # cut lines are this femoral's J-curve (femoral_block.py). Its features go before
        # the replace (a replace whose recompute fails is rolled back) and are made again.
        def block_repoint(des, doc):
            synced = sync(des, src_params["block"], {})
            stripped = strip_block(des)
            errs = []
            for o in list(des.rootComponent.occurrences):
                if o.isReferencedComponent:
                    try: o.replace(fem_v.latestVersion if hasattr(fem_v, "latestVersion") else fem_v, True)
                    except Exception as ex: errs.append(str(ex)[:200])
            want = report["femoral"]["version"]
            for _ in range(40):
                stale = [r for r in doc.documentReferences if r.dataFile.id == fem.id and r.dataFile.versionNumber != want]
                if not stale: break
                for r in stale:
                    try: r.getLatestVersion()
                    except Exception as ex: errs.append("getLatestVersion: " + str(ex)[:100])
                compute(des); time.sleep(1); adsk.doEvents()
            compute(des)
            refs = [(r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name) for r in doc.documentReferences]
            if [r for r in doc.documentReferences if r.dataFile.id != fem.id or r.dataFile.versionNumber != want]:
                raise RuntimeError(f"block references {refs}, not {fem.name} v{want}")
            made = make_block(des)
            compute(des)
            return {**synced, "stripped": stripped, "made": made, "replace_errors": errs, "refs": refs}
        try:
            report["block"] = kept(blk, "block") if not wanted("block") else build_part(
                app, blk, {}, [("Block", OUT + "\\femoral_block.stl")],
                tag + ": femoral cutting block on this knee's J-curve",
                prepare=block_repoint, finish=lambda des: {"pegs": block_pegs(des, side_body), "state": block_state(des)})
        except Exception as ex:
            report["block"] = {"saved": False, "error": f"{type(ex).__name__}: {ex}"}
        # The tibial cutting block on this knee's plate (tibial_block.py); its STL goes to
        # the tray frame with the plate's.
        def tblock_repoint(des, doc):
            synced = sync(des, src_params["tblock"], {})
            stripped = strip_tblock(des)
            errs = []
            for o in list(des.rootComponent.occurrences):
                if o.isReferencedComponent:
                    try: o.replace(pla_v.latestVersion if hasattr(pla_v, "latestVersion") else pla_v, True)
                    except Exception as ex: errs.append(str(ex)[:200])
            want = report["plate"]["version"]
            for _ in range(40):
                stale = [r for r in doc.documentReferences if r.dataFile.id == pla.id and r.dataFile.versionNumber != want]
                if not stale: break
                for r in stale:
                    try: r.getLatestVersion()
                    except Exception as ex: errs.append("getLatestVersion: " + str(ex)[:100])
                compute(des); time.sleep(1); adsk.doEvents()
            compute(des)
            refs = [(r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name) for r in doc.documentReferences]
            if [r for r in doc.documentReferences if r.dataFile.id != pla.id or r.dataFile.versionNumber != want]:
                raise RuntimeError(f"tibial block references {refs}, not {pla.name} v{want}")
            made = make_tblock(des)
            compute(des)
            base = plate_base_z(plate_occurrence(des))
            if base is None or abs(base) > 0.001:
                raise RuntimeError(f"plate base at Z {base} mm, not on the origin")
            return {**synced, "stripped": stripped, "made": made, "replace_errors": errs, "refs": refs}
        try:
            report["tblock"] = kept(tbk, "tblock") if not wanted("tblock") else build_part(
                app, tbk, {}, [("Block", OUT + "\\tibial_block.stl")],
                tag + ": tibial cutting block on this knee's plate",
                prepare=tblock_repoint, finish=lambda des: {"keel": tblock_keel(des), "state": tblock_state(des)},
                shift=tray)
        except Exception as ex:
            report["tblock"] = {"saved": False, "error": f"{type(ex).__name__}: {ex}"}
    except Exception as ex:
        report["error"] = f"{type(ex).__name__}: {ex}"
    if active and active.isValid: active.activate()
    json.dump(report, open(OUT + "\\build.json", "w"), indent=1)
    print(json.dumps({"name": JOB["name"], "error": report.get("error")}))
'''


def build(names=None, jobs_file=OUT / "jobs.json", only=None):
    all_jobs = json.loads(Path(jobs_file).read_text())
    for job in all_jobs:
        if names and job["name"] not in names:
            continue
        d = OUT / job["name"]
        done = d / "build.json"
        if not names and done.exists() and not json.loads(done.read_text()).get("error"):
            continue            # built already; name the knee to rebuild it
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(dict(job, only=only) if only else job), encoding="utf-8")
        from insert_edit import OUTLINE
        from insert_finish import FINISH
        from femoral_block import BLOCK
        from tibial_block import TBLOCK
        script = (BUILD.replace("__OUTLINE__", OUTLINE).replace("__FINISH__", FINISH).replace("__BLOCK__", BLOCK)
                  .replace("__TBLOCK__", TBLOCK)
                  .replace("__JOB__", repr(str(d / "job.json")))
                  .replace("__SOURCES__", repr(SOURCES)).replace("__OUT__", repr(str(d))))
        print(call_fusion(script, read_only=False).strip().splitlines()[-1], flush=True)
        rep = json.loads((d / "build.json").read_text())
        for part in ("femoral", "plate", "insert", "block", "tblock"):
            r = rep.get(part, {})
            print(f"   {part:8s} saved={r.get('saved')} v{r.get('version')} health={r.get('health') or 'ok'}"
                  f" suppressed={r.get('suppressed', [])} finish={r.get('finish', '')}"
                  f" missing={r.get('missing', [])} {('refs ' + str(r.get('refs'))) if part == 'insert' else ''}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "jobs":
        jobs()
    else:
        build(sys.argv[2:] or None)
