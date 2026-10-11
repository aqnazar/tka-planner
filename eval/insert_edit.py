"""Insert: its outline from the plate it sits in (author OK 2026-10-09: the insert follows
the plate's outline).

The design's Base Sketch projects the plate's inner-wall faces and offsets them; on a
patient plate those references don't resolve and the insert keeps the design outline.
`rebuild_outline` replaces it:
  - sketch "Base Outline" on plane "Ledge level" (XY + Plate_Thickness + 2.5 mm, through
    the plate's wall and ledge, above its floor)
  - the plate body's section there (projectCutEdges): the outer outline, and the pocket's
    inner boundary (the wall's inside, around the ledge and the notch, fillets included),
    already carrying the plate's Tray offset
  - every pocket loop offset inward by Wall_Offset (the design's clearance)
  - the plate's ledge is as tall as its wall, so the pocket is two halves: a rectangle
    over the ledge bridges them; Base Feature takes the halves and the bridge, from the
    plate floor; "Ledge slot" subtracts the plate (keeping it), carving the ledge out
  - the later features are pointed at the new base body; the snap-pin pattern
    (R-Pattern1, it patterned the old base body) is suppressed; the design's top bridge
    (Top Connection (1)) stays, though it may warn: the femoral's Constraint 3 mates to it
It is run on the design once (save) and again by the pipeline after every repoint, so a
patient's insert never depends on references being re-resolved.

  python insert_edit.py test             # in memory: the Driver Tests insert on each demo knee
  python insert_edit.py save <lineage>   # apply to a design file and save
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402

IDS = json.loads((HERE / "tray_audit/copies.json").read_text())
MSG = ("Claude: insert outline from the plate's section at the ledge level (Base Outline, projectCutEdges), "
       "pocket loops offset by Wall_Offset; rebuilt by the pipeline after each repoint. Author OK 2026-10-09.")

# Shared with implant_pipeline.py (pasted into its Fusion script).
OUTLINE = r'''
def rebuild_outline(des):
    V = adsk.core.ValueInput.createByString
    root = des.rootComponent
    log = {}
    up = des.userParameters
    if up.itemByName("Plate_Thickness") is None:
        up.add("Plate_Thickness", V("5 mm"), "mm", "Plate floor height (as the plate)")
    tl = des.timeline
    f = root.features.extrudeFeatures.itemByName("Base Feature")
    f.timelineObject.rollTo(True)
    plane = root.constructionPlanes.itemByName("Ledge level")
    if plane is None:
        pi = root.constructionPlanes.createInput()
        pi.setByOffset(root.xYConstructionPlane, V("Plate_Thickness + 2.5 mm"))
        plane = root.constructionPlanes.add(pi); plane.name = "Ledge level"
    # A fresh sketch every time; the previous one (if any) goes once Base Feature has
    # moved to the new profiles.
    previous = root.sketches.itemByName("Base Outline")
    sk = root.sketches.add(plane)
    occ = [o for o in root.occurrences if "Plate" in o.component.name][0]
    body = occ.component.bRepBodies.item(0).createForAssemblyContext(occ)
    cut = sk.projectCutEdges(body)
    curves = [c for c in cut if hasattr(c, "boundingBox")]
    log["cut_curves"] = len(curves)
    # Group the cut curves into closed loops by shared end points.
    def key(p): return (round(p.x, 4), round(p.y, 4))
    parent = list(range(len(curves)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    ends = {}
    for i, c in enumerate(curves):
        for attr in ("startSketchPoint", "endSketchPoint"):
            try:
                k = key(getattr(c, attr).geometry)
            except Exception:
                continue
            if k in ends: parent[find(i)] = find(ends[k])
            else: ends[k] = i
    groups = {}
    for i in range(len(curves)): groups.setdefault(find(i), []).append(curves[i])
    loops = []
    for g in groups.values():
        xs, ys = [], []
        for c in g:
            b = c.boundingBox; xs += [b.minPoint.x, b.maxPoint.x]; ys += [b.minPoint.y, b.maxPoint.y]
        loops.append({"curves": g, "bb": (min(xs), min(ys), max(xs), max(ys)),
                      "area": (max(xs) - min(xs)) * (max(ys) - min(ys))})
    loops.sort(key=lambda L: -L["area"])
    log["loops_mm"] = [[round(v * 10, 1) for v in L["bb"]] for L in loops]
    # loops[0]: the plate's outer outline. The rest with some size: the pocket's inside.
    pocket = [L for L in loops[1:] if L["area"] > 0.5]       # cm^2 of bounding box
    n_off = 0
    offs = []
    for L in pocket:
        col = adsk.core.ObjectCollection.create()
        for c in L["curves"]: col.add(c)
        x0, y0, x1, y1 = L["bb"]
        # Inward means a smaller loop: try direction points across the loop until the
        # offset comes out smaller (a point outside the loop offsets it outward).
        done = False
        for fx, fy in ((0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7)):
            pt = adsk.core.Point3D.create(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 0)
            try:
                off = sk.offset(col, pt, 0.06)
            except Exception as ex:
                log.setdefault("offset_err", []).append(str(ex)[:80]); continue
            xs = [c.boundingBox.minPoint.x for c in off] + [c.boundingBox.maxPoint.x for c in off]
            ys = [c.boundingBox.minPoint.y for c in off] + [c.boundingBox.maxPoint.y for c in off]
            if max(xs) - min(xs) < x1 - x0:
                done = True; offs.append(((min(xs), min(ys), max(xs), max(ys)), list(off))); break
            for c in off: c.deleteMe()
        if not done:
            continue
        n_off += 1
        dims = [d for d in sk.sketchDimensions if d.objectType.endswith("SketchOffsetCurvesDimension")]
        if dims:
            try: dims[-1].parameter.expression = "Wall_Offset"
            except Exception: pass
    log["offset_loops"] = n_off
    # The plate's ledge is as tall as its wall: the pocket is two halves. Two arcs over
    # the ledge bridge them into one body, each continuing the outline it meets (author,
    # 2026-10-10: anterior on the anterior arc's circle, posterior on the notch arc's);
    # the plate, subtracted below ("Ledge slot"), carves the ledge out of it.
    outer = loops[0]["bb"]
    offs.sort(key=lambda o: o[0][0])
    sel = []
    if len(offs) == 2:
        ((l0, l1, l2, l3), lcur), ((r0, r1, r2, r3), rcur) = offs
        def ledge_side(cur, x):
            # The offset loop's straight side along the ledge.
            best = None
            for c in cur:
                if not c.objectType.endswith("SketchLine"): continue
                a, b = c.startSketchPoint.geometry, c.endSketchPoint.geometry
                if abs(a.x - x) < 1e-3 and abs(b.x - x) < 1e-3 and (best is None or c.length > best.length):
                    best = c
            if best is None: return None
            return sorted((best.startSketchPoint, best.endSketchPoint), key=lambda p: p.geometry.y)
        def neighbour_arc(cur, sp):
            for c in cur:
                if c.objectType.endswith("SketchArc") and any(
                        q.geometry.distanceTo(sp.geometry) < 1e-4 for q in (c.startSketchPoint, c.endSketchPoint)):
                    return c
            return None
        def bridge(p, q, arc):
            if arc is None:
                return sk.sketchCurves.sketchLines.addByTwoPoints(p, q)
            c, R = arc.centerSketchPoint.geometry, arc.radius
            m = adsk.core.Point3D.create((p.geometry.x + q.geometry.x) / 2, (p.geometry.y + q.geometry.y) / 2, 0)
            d = c.vectorTo(m); d.normalize(); d.scaleBy(R)
            mid = adsk.core.Point3D.create(c.x + d.x, c.y + d.y, 0)
            new = sk.sketchCurves.sketchArcs.addByThreePoints(p, mid, q)
            # Tangent to the arc it continues. Fusion has refused this on one build (the
            # arc then solved up to 2 mm off), so the fallback puts it on that arc's own
            # circle, which is the same curve; the log says which held.
            try:
                sk.geometricConstraints.addTangent(new, arc); log.setdefault("bridge_fix", []).append("tangent")
            except Exception:
                if new.isFullyConstrained:          # nothing left to drift (P007R)
                    log.setdefault("bridge_fix", []).append("held")
                    return new
                try:
                    sk.geometricConstraints.addConcentric(new, arc); sk.geometricConstraints.addEqual(new, arc)
                    log.setdefault("bridge_fix", []).append("concentric")
                except Exception:
                    # Already on that circle (both refused as redundant): the radius
                    # alone holds its bulge, which the ends leave free.
                    try:
                        sk.geometricConstraints.addEqual(new, arc); log.setdefault("bridge_fix", []).append("equal")
                    except Exception:
                        # Fusion takes no constraint against an offset curve here; a radius
                        # on the bridge itself (it was built on that circle) pins it.
                        try:
                            c0 = new.centerSketchPoint.geometry
                            d = sk.sketchDimensions.addRadialDimension(new, adsk.core.Point3D.create(c0.x, c0.y + arc.radius, 0))
                            d.parameter.value = arc.radius
                            log.setdefault("bridge_fix", []).append("radius %.2f mm" % (arc.radius * 10))
                        except Exception as ex:
                            log.setdefault("bridge_fix", []).append("none: " + str(ex)[:80])
            return new
        L, Rr = ledge_side(lcur, l2), ledge_side(rcur, r0)
        if L and Rr:
            bridge(L[0], Rr[0], neighbour_arc(lcur, L[0]))       # anterior
            bridge(L[1], Rr[1], neighbour_arc(lcur, L[1]))       # posterior
            rx0, rx1 = l2 - 0.001, r0 + 0.001
            ry0 = min(L[0].geometry.y, Rr[0].geometry.y) - 0.1
            ry1 = max(L[1].geometry.y, Rr[1].geometry.y) + 0.1
            log["bridge"] = "arcs"
        else:
            # No straight side found: a rectangle over the ledge, as before.
            rx0, rx1 = l2 - 0.05, r0 + 0.05
            ry0, ry1 = max(l1, r1), min(l3, r3)
            sk.sketchCurves.sketchLines.addTwoPointRectangle(
                adsk.core.Point3D.create(rx0, ry0, 0), adsk.core.Point3D.create(rx1, ry1, 0))
            log["bridge"] = "rectangle"
        log["bridge_mm"] = [round(v * 10, 2) for v in (rx0, ry0, rx1, ry1)]
        for pr in sk.profiles:
            a = pr.areaProperties(); c = a.centroid; bb = pr.boundingBox
            wide = (bb.maxPoint.x - bb.minPoint.x) > 0.9 * (outer[2] - outer[0])
            if wide:
                continue                                   # the wall ring
            in_rect = rx0 <= c.x <= rx1 and ry0 <= c.y <= ry1
            in_half = any(o[0] <= c.x <= o[2] and o[1] <= c.y <= o[3] for o, _ in offs) and a.area > 2.0
            if in_rect or in_half:
                sel.append(pr)
    if not sel:
        sel = sorted(sk.profiles, key=lambda p: -p.areaProperties().area)[:2]
    halves = sel
    log["halves_mm2"] = [round(p.areaProperties().area * 100, 1) for p in halves]
    log["halves_cx_mm"] = [round(p.areaProperties().centroid.x * 10, 1) for p in halves]
    col = adsk.core.ObjectCollection.create()
    for p in halves: col.add(p)
    f.profile = col
    if previous is not None:
        try: previous.deleteMe()
        except Exception as ex: log["previous_sketch"] = str(ex)[:80]
    sk.name = "Base Outline"
    try:
        f.startExtent = adsk.fusion.OffsetStartDefinition.create(V("-2.5 mm"))
    except Exception as ex:
        log["start_err"] = str(ex)[:100]
    # One body over the whole pocket; the plate itself cuts the slot over its ledge.
    # Made fresh every time, like the sketch: its references to the base body and the
    # plate body go missing when either is remade (the plate's Tray offset suppressed).
    old_slot = root.features.combineFeatures.itemByName("Ledge slot")
    if old_slot is not None:
        try: old_slot.deleteMe()
        except Exception as ex: log["old_slot"] = str(ex)[:100]
    f.timelineObject.rollTo(False)
    tools = adsk.core.ObjectCollection.create(); tools.add(body)
    ci = root.features.combineFeatures.createInput(f.bodies.item(0), tools)
    ci.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    ci.isKeepToolBodies = True
    try:
        cf = root.features.combineFeatures.add(ci); cf.name = "Ledge slot"
    except Exception as ex:
        log["slot_err"] = str(ex)[:150]
    # The later features join to / cut the base body: point them at the new one.
    tl.moveToEnd()
    base = f.bodies.item(0) if f.bodies.count else None
    log["retarget"] = {}
    if base is not None:
        feats = {x.name: x for x in list(root.features.extrudeFeatures) + list(root.features.sweepFeatures)}
        for name in ("Snap Lock (1)", "Top Connection (1)", "Anterior Sweep", "Posterior Sweep"):
            x = feats.get(name)
            if x is None: continue
            try:
                x.timelineObject.rollTo(True)
                b = f.bodies.item(0)
                x.participantBodies = [b]
                log["retarget"][name] = "ok"
            except Exception as ex:
                log["retarget"][name] = str(ex)[:80]
            tl.moveToEnd()
    # The design's top bridge (its sketch projects plate faces) is replaced by the bridge
    # in Base Outline; the snap-pin pattern patterned the old base body. Both suppressed.
    # (Top Connection (1) stays: the femoral's Constraint 3 mates to its face.)
    log["suppressed"] = []
    for i in range(tl.count):
        it = tl.item(i)
        want = it.name == "R-Pattern1"
        if it.name in ("R-Pattern1", "Top Connection (1)", "Constraint 1", "Constraint 2", "Constraint 3")                 and it.isSuppressed != want:
            try: it.isSuppressed = want; log["suppressed"].append([it.name, want])
            except Exception as ex: log["suppressed"].append(it.name + ": " + str(ex)[:60])
    old = root.sketches.itemByName("Base Sketch")
    if old:
        try: old.deleteMe(); log["old_sketch_deleted"] = True
        except Exception as ex: log["old_sketch_deleted"] = str(ex)[:80]
    try: des.computeAll()
    except Exception as ex: log["compute"] = str(ex)[:120]
    log["timeline"] = [tl.item(i).name for i in range(tl.count)]
    return log
'''

S = r'''
import adsk.core, adsk.fusion, json, time
LINEAGE = __L__
SAVE = __SAVE__
TESTS = json.load(open(__TESTS__, encoding="utf-8"))
MSG = __MSG__
__OUTLINE__

def state(des):
    tl = des.timeline
    bad = [[tl.item(i).name, tl.item(i).healthState] for i in range(tl.count) if tl.item(i).healthState]
    out = {"unhealthy": bad, "bodies": {}}
    for b in des.rootComponent.bRepBodies:
        bb = b.boundingBox
        out["bodies"][b.name] = [round(b.volume * 1000, 1)] + [round(v * 10, 2) for v in (bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    return out

def repoint(app, des, fem_id, pla_id):
    root = des.rootComponent
    tl = des.timeline
    for i in range(tl.count):
        if tl.item(i).name == "Fillet2":
            try: tl.item(i).isSuppressed = True
            except Exception: pass
    errs = []
    for o in list(root.occurrences):
        if not o.isReferencedComponent: continue
        n = o.component.name
        target = fem_id if "Femoral" in n else pla_id if "Plate" in n else None
        if not target: continue
        try: o.replace(app.data.findFileById(target), True)
        except Exception as ex: errs.append(str(ex)[:120])
    try: des.computeAll()
    except Exception: pass
    return errs

def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(LINEAGE)
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == LINEAGE:
            if d.isModified: raise RuntimeError("unsaved changes open on this file")
            d.close(False)
    out = {}
    doc = app.documents.open(df, True)
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    out["before"] = state(des)
    try:
        out["log"] = rebuild_outline(des)
        out["after"] = state(des)
        if SAVE:
            if not [h for h in out["after"]["unhealthy"] if h[1] == 2]:
                v0 = df.latestVersionNumber
                doc.save(MSG)
                for _ in range(120):
                    if app.data.findFileById(LINEAGE).latestVersionNumber > v0: break
                    time.sleep(1); adsk.doEvents()
                out["version_after"] = app.data.findFileById(LINEAGE).latestVersionNumber
            else:
                out["not_saved"] = True
        else:
            out["tests"] = {}
            for name, fem, pla, sets in TESTS:
                doc.close(False)
                doc = app.documents.open(df, True)
                des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
                r = {"replace_errors": repoint(app, des, fem, pla)}
                for n, e in sets.items():
                    p = des.allParameters.itemByName(n)
                    if p: p.expression = e
                r["log"] = rebuild_outline(des)
                r.update(state(des))
                out["tests"][name] = r
    finally:
        try: doc.close(False)
        except Exception: pass
        if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def tests():
    files = json.loads((HERE / "patient_builds/demo_files.json").read_text())
    out = []
    for j in json.loads((HERE / "patient_builds/jobs_demo.json").read_text()):
        f = {n: i for n, i, _ in files[j["name"]]}
        out.append((j["name"], f["Knee Prosthesis (Femoral)"], f["Tibial Plate"], dict(j["insert"])))
    return out


if __name__ == "__main__":
    save = sys.argv[1] == "save"
    lineage = sys.argv[2] if save else IDS["insert"]["id"]
    tf = HERE / "tray_audit" / "_insert_tests.json"
    tf.write_text(json.dumps([] if save else tests()), encoding="utf-8")
    script = (S.replace("__OUTLINE__", OUTLINE).replace("__L__", repr(lineage)).replace("__SAVE__", repr(save))
              .replace("__TESTS__", repr(str(tf))).replace("__MSG__", repr(MSG)))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (HERE / "tray_audit" / f"insert_edit_{'save' if save else 'test'}.json").write_text(json.dumps(res, indent=1))
    print("design:", json.dumps(res.get("log")), res.get("after", {}).get("unhealthy"), res.get("version_after"))
    for b, x in res.get("after", {}).get("bodies", {}).items():
        print("    ", b, x)
    for k, v in res.get("tests", {}).items():
        print(k, v["log"], v["replace_errors"], v["unhealthy"])
        for b, x in v["bodies"].items():
            print("    ", b, x)
