"""Insert finishing, from the author's edit of the REF_KJ2 insert (2026-10-10,
Patient Builds/REF_KJ2/Author Edits v5), as rules that hold on any knee.

The author's changes:
  - the plate is dead on the origin: its "Tray offset" move is not used (the author
    rolled it back). The offset still places the tray on the cut, so here the insert's
    femoral is placed at -Tray_Offset instead (Constraint 2 = Tray_Offset_X along the
    femoral's ML plane, Constraint 1 = -Tray_Offset_Y along its AP plane): the outline
    follows the plate, the dishes follow the femoral, and the build shifts the plate and
    insert STLs by +Tray_Offset into the tray frame (author's choice, 2026-10-10)
  - the snap pins and their lock go (Snap Lock sketch and extrude, R-Pattern1), as do
    Top Connection (1) and Fillet2: the Ledge slot (the plate subtracted) does their job
  - the bridge over the ledge is two arcs continuing the outline (insert_edit.OUTLINE)
  - after the sweeps, in this order:
      Back fillet          2 mm   the sharp posterior notch corners (vertical edges where
                                  a face turned to the notch meets the posterior face at
                                  over 10 deg); important. None where the plate's notch
                                  fillet already rounds them (the design size)
      Top fillet           2 mm   the rim: every edge between the articular (sweep)
                                  faces and the side walls
      Bottom fillet        1 mm   the base: every edge of the floor face but the slot's
      Slot clearance       0.5 mm the Ledge slot's lower side walls moved out (2 -> 3 mm wide)
      Slot roof clearance  0.3 mm the slot's roof moved up
    The four after the back fillet may be dropped if they fail (author).
The features are made fresh by every build (edge choices don't survive a new outline).
The dish section (Condylar Profile) is also made fresh every build, on the femoral's coronal
plane: the femoral's condyle curve projected (linked) and the 2 mm box above it. Moving the
old sketch to that plane (redefine) had turned its projection into a frozen copy.

  python insert_finish.py check               # rules vs the author's REF_KJ2 v5, in memory
  python insert_finish.py save <lineage>      # apply to an insert design and save
  python insert_finish.py plate <lineage>     # suppress a plate's Tray offset and save
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_driver import call_fusion  # noqa: E402

AUTHOR_EDIT = "urn:adsk.wipprod:dm.lineage:tsySxxqUQVW4wZqZ5wmOgg"     # REF_KJ2 Author Edits
MSG = ("Claude: insert finishing from the author's REF_KJ2 edit (2026-10-10): snap pins, Snap Lock, "
       "Top Connection (1), Fillet2 removed; outline bridge as arcs; femoral at -Tray_Offset "
       "(Constraint 1/2); Back, Top, Bottom fillets and slot clearances by rule.")
PLATE_MSG = "Claude: Tray offset suppressed, plate on the origin (author 2026-10-10); the offset is placed by the insert's femoral and the build."

# Shared with implant_pipeline.py (pasted into its Fusion script).
FINISH = r'''
LEGACY = ("Snap Lock (1)", "R-Pattern1", "Top Connection (1)", "Fillet2", "Snap Lock")
FINISHING = ("Back fillet", "Top fillet", "Bottom fillet", "Slot clearance", "Slot roof clearance",
             "OffsetFaces1", "OffsetFaces2")
FINISH_PARAMS = (("Back_Fillet_Radius", "2 mm", "Posterior notch corners (author 2026-10-10)"),
                 ("Top_Fillet_Radius", "2 mm", "Rim, articular faces to side walls (author 2026-10-10)"),
                 ("Bottom_Fillet_Radius", "1 mm", "Floor edges but the slot (author 2026-10-10)"),
                 ("Slot_Wall_Clearance", "0.5 mm", "Ledge slot side walls moved out (author 2026-10-10)"),
                 ("Slot_Roof_Clearance", "0.3 mm", "Ledge slot roof moved up (author 2026-10-10)"),
                 ("Tray_Offset_X", "0 mm", "Plate centre from the femoral, +X; the femoral sits at -X here"),
                 ("Tray_Offset_Y", "0 mm", "Plate centre from the femoral, +Y posterior; the femoral sits at -Y here"))
SLOT_HALF_CM = 0.4          # slot faces lie within 4 mm of the ML centre; notch walls are far wider

def _delete_named(des, names):
    tl = des.timeline; gone = []
    for i in reversed(range(tl.count)):
        if i >= tl.count: continue
        it = tl.item(i)
        try: name = it.name
        except Exception: continue
        if name in names:
            try:
                it.entity.deleteMe(); gone.append(name)
            except Exception as ex:
                try:
                    it.isSuppressed = True; gone.append(name + " (suppressed: " + str(ex)[:40] + ")")
                except Exception:
                    gone.append(name + " FAILED " + str(ex)[:60])
    return gone

def strip_insert(des):
    """The design's snap pins and top bridge, and any earlier finishing features."""
    des.timeline.moveToEnd()
    return {"removed": _delete_named(des, FINISHING) + _delete_named(des, LEGACY)}

def finish_params(des):
    up = des.userParameters
    for n, e, c in FINISH_PARAMS:
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString(e), "mm", c)

def tray_offset(des):
    """The femoral at -Tray_Offset from the plate (Constraint 1 moves it +Y, 2 moves it -X)."""
    finish_params(des)
    root = des.rootComponent; out = {}
    for cname, expr in (("Constraint 1", "-Tray_Offset_Y"), ("Constraint 2", "Tray_Offset_X")):
        c = root.assemblyConstraints.itemByName(cname)
        if c is None:
            out[cname] = "missing"; continue
        p = c.geometricRelationships.item(0).offsetOrAngle
        p.expression = expr; out[cname] = p.name + " = " + expr
    out["profile_plane"] = coronal_profile(des)
    return out

def coronal_profile(des):
    """The dish section (Condylar Profile) on the femoral's coronal plane, so it meets
    the J-curve where the femoral's own section does: XZ offset to the femoral's -Y."""
    root = des.rootComponent
    # Leftovers of a remake that could not move the sweeps (named "(new)" or "(old)"),
    # used by nothing, go first.
    used = set()
    for f in root.features.sweepFeatures:
        try: used.add(f.profile.parentSketch.name)
        except Exception: pass
    for s_ in [s_ for s_ in root.sketches if s_.name in ("Condylar Profile (new)", "Condylar Profile (old)")]:
        if s_.name not in used:
            try: s_.deleteMe()
            except Exception: pass
    sk = root.sketches.itemByName("Condylar Profile")
    if sk is None: return "no Condylar Profile"
    # An offset goes along XZ's normal; the plane must sit at y = -Tray_Offset_Y.
    expr = "-Tray_Offset_Y" if root.xZConstructionPlane.geometry.normal.y > 0 else "Tray_Offset_Y"
    plane = root.constructionPlanes.itemByName("Femoral coronal")
    if plane is None:
        sk.timelineObject.rollTo(True)
        pi = root.constructionPlanes.createInput()
        pi.setByOffset(root.xZConstructionPlane, adsk.core.ValueInput.createByString(expr))
        plane = root.constructionPlanes.add(pi); plane.name = "Femoral coronal"
        des.timeline.moveToEnd()
    elif plane.definition.offset.expression != expr:
        plane.definition.offset.expression = expr
    made = remake_profile(des, sk, plane)
    return "y = %.2f mm, %s, box %s" % (plane.geometry.origin.y * 10, made, profile_box_up(des))

def femoral_condyle(des):
    """The femoral's coronal condyle curve (its Condylar profile sketch's spline), in this
    document's context: the curve the dish section is the projection of."""
    for o in des.rootComponent.occurrences:
        if o.isReferencedComponent and "Femoral" in o.component.name:
            sk = o.component.sketches.itemByName("Condylar profile")
            if sk is None: continue
            sp = [c for c in sk.sketchCurves if c.objectType.split("::")[-1] != "SketchLine" and not c.isConstruction
                  and c.objectType.split("::")[-1] not in ("SketchArc", "SketchCircle")]
            if sp: return sp[0].createForAssemblyContext(o)
    return None

def remake_profile(des, old, plane):
    """A new Condylar Profile on the femoral's coronal plane: the femoral's condyle curve
    projected (linked, so the dish follows the femoral on every build), the 2 mm box
    (sides vertical at the curve's ends and equal, top horizontal, 2 mm),
    both sweeps moved onto it, then the old sketch removed. Redefining the old sketch's
    plane instead turned its projection into a frozen copy (2026-10-10: P009L kept the
    design-size dish)."""
    root = des.rootComponent
    curve = femoral_condyle(des)
    if curve is None: return "no femoral condyle curve"
    d_expr = "2 mm"
    try:
        d_expr = [d for d in old.sketchDimensions if d.parameter][0].parameter.expression
    except Exception: pass
    old.timelineObject.rollTo(True)
    sk = root.sketches.add(plane)
    sp = sk.project(curve).item(0)
    a, b = sp.startSketchPoint, sp.endSketchPoint
    up = lambda p, dz: sk.modelToSketchSpace(adsk.core.Point3D.create(p.worldGeometry.x, p.worldGeometry.y, p.worldGeometry.z + dz))
    L = sk.sketchCurves.sketchLines
    l1 = L.addByTwoPoints(up(a, 0), up(a, 0.2))
    l2 = L.addByTwoPoints(l1.endSketchPoint, up(b, 0.2))
    l3 = L.addByTwoPoints(l2.endSketchPoint, up(b, 0))
    gc = sk.geometricConstraints
    errs = []
    # Explicit constraints, not shared points: a point shared with projected geometry
    # loses the join when the curve moves (a new Tray_Offset or Insert_Thickness) and
    # the box is left behind, so the sweeps lose their profile (2026-10-25: DEMO_LARGE
    # insert v17; reproduced on five knees by moving Tray_Offset_X 0.8 mm).
    for end, pt in ((l1.startSketchPoint, a), (l3.endSketchPoint, b)):
        try:
            gc.addCoincident(end, pt)
        except Exception as ex:
            errs.append("coincident " + str(ex)[:60])
    # Sides vertical (the curve's ends are vertical, so they meet it tangent) and equal,
    # top horizontal. Tangent sides with one length and a perpendicular top solved where
    # drawn but not when the linked curve moved with the next femoral (2026-10-11:
    # DEMO_LARGE sides 2.39 against 2 mm, "Failed to solve").
    # The two sides' equal length is not added: with both ends coincident to the curve it is
    # always refused as redundant (the "over constrained" noise of every build, 2026-10-25).
    for f in (lambda: gc.addVertical(l1), lambda: gc.addVertical(l3), lambda: gc.addHorizontal(l2)):
        try: f()
        except Exception as ex: errs.append(str(ex)[:60])
    try:
        sk.sketchDimensions.addDistanceDimension(l1.startSketchPoint, l1.endSketchPoint,
            adsk.fusion.DimensionOrientations.AlignedDimensionOrientation, up(a, 0.1)).parameter.expression = d_expr
    except Exception as ex: errs.append("dim " + str(ex)[:60])
    prof = min(sk.profiles, key=lambda p: p.areaProperties().area) if sk.profiles.count else None
    moved = []
    for f in root.features.sweepFeatures:
        try:
            f.timelineObject.rollTo(True)
            f.profile = prof; moved.append(f.name)
        except Exception as ex:
            errs.append(f.name + ": " + str(ex)[:60])
    des.timeline.moveToEnd()
    name = "Condylar Profile"
    if len(moved) == root.features.sweepFeatures.count:
        old.name = name + " (old)"
        sk.name = name
        try:
            if not old.deleteMe(): errs.append("old sketch not deleted")
        except Exception as ex:
            errs.append("old sketch: " + str(ex)[:60])
    else:
        sk.name = name + " (new)"
    return "profile remade (%d profiles, sweeps %s%s)" % (sk.profiles.count, moved, ("; " + "; ".join(errs)) if errs else "")

def profile_box_up(des):
    """The Condylar Profile's 2 mm box closes above the condyle spline (the author's
    design). Its side isn't constrained, and builds had solved it below (the dish ends
    then cut up to 2 mm low); a box found below is moved up and solved again."""
    sk = des.rootComponent.sketches.itemByName("Condylar Profile")
    sp = [c for c in sk.sketchCurves if c.objectType.split("::")[-1] != "SketchLine"]
    if not sp: return "no spline"
    ev = sp[0].worldGeometry.evaluator; ok, a, b = ev.getParameterExtents()
    ok, p0 = ev.getPointAtParameter(a); ok, p1 = ev.getPointAtParameter(b)
    lines = [l for l in sk.sketchCurves.sketchLines if not l.isConstruction]
    def touches(l):
        return any(q.worldGeometry.distanceTo(p) < 1e-4 for q in (l.startSketchPoint, l.endSketchPoint) for p in (p0, p1))
    far = [l for l in lines if not touches(l)]
    if len(far) != 1: return "? (%d closing lines)" % len(far)
    f = far[0]
    def side():
        zf = (f.startSketchPoint.worldGeometry.z + f.endSketchPoint.worldGeometry.z) / 2
        return zf - (p0.z + p1.z) / 2
    s = side()
    if s > 0: return "above"
    for pt in (f.startSketchPoint, f.endSketchPoint):
        w = pt.worldGeometry
        tgt = sk.modelToSketchSpace(adsk.core.Point3D.create(w.x, w.y, w.z - 2 * s))
        pt.move(pt.geometry.vectorTo(tgt))
    try: des.computeAll()
    except Exception: pass
    return "moved above" if side() > 0 else "still below"

def _normal(f):
    ok, n = f.evaluator.getNormalAtPoint(f.pointOnFace)
    return n

def _planar(f):
    return f.geometry.objectType.endswith("Plane")

def finish_body(des):
    root = des.rootComponent
    return max(root.bRepBodies, key=lambda b: b.volume)

def finish_rules(des):
    """The edges and faces of each finishing feature on the current body."""
    root = des.rootComponent
    body = finish_body(des)
    zmin = body.boundingBox.minPoint.z
    sweep = set()
    for f in root.features.sweepFeatures:
        if f.isSuppressed or f.healthState == 2: continue
        for x in f.faces: sweep.add(x.entityToken)
    def is_wall(f):
        return abs(_normal(f).z) < 0.2
    def is_slot_wall(f):
        n = _normal(f); p = f.pointOnFace
        return _planar(f) and abs(n.x) > 0.99 and abs(p.x) < SLOT_HALF_CM and n.x * p.x < 0
    def is_floor(f):
        return _planar(f) and _normal(f).z < -0.99 and abs(f.pointOnFace.z - zmin) < 1e-3
    back, top, bottom = [], [], []
    for e in body.edges:
        fs = list(e.faces)
        if len(fs) != 2: continue
        a, b = fs
        if e.geometry.objectType.endswith("Line3D"):
            s, t = e.startVertex.geometry, e.endVertex.geometry
            vertical = abs(s.x - t.x) < 1e-4 and abs(s.y - t.y) < 1e-4
            if vertical and s.y > 0 and abs(s.x) > SLOT_HALF_CM:
                # A sharp posterior notch corner: the posterior face against a face turned
                # toward the notch, at more than 10 deg (where the plate's notch fillet
                # already rounds it, the two are tangent and there is nothing to do).
                pt = e.pointOnEdge
                na = a.evaluator.getNormalAtPoint(pt)[1]; nb = b.evaluator.getNormalAtPoint(pt)[1]
                for p_, q_ in ((na, nb), (nb, na)):
                    if p_.y > 0.95 and q_.x * s.x < -0.3 * abs(s.x) and p_.dotProduct(q_) < 0.985:
                        back.append(e); break
        ta, tb = a.entityToken in sweep, b.entityToken in sweep
        if (ta and not tb and is_wall(b)) or (tb and not ta and is_wall(a)):
            top.append(e)
        if (is_floor(a) and not is_slot_wall(b)) or (is_floor(b) and not is_slot_wall(a)):
            if not (is_floor(a) and is_floor(b)):
                bottom.append(e)
    # The slot's lower side walls, from the floor up (its upper step is left as it is).
    walls = [f for f in body.faces if is_slot_wall(f) and abs(f.boundingBox.minPoint.z - zmin) < 1e-3]
    roof = [f for f in body.faces if _planar(f) and _normal(f).z < -0.99 and f.pointOnFace.z > zmin + 0.005
            and abs(f.boundingBox.minPoint.x) < SLOT_HALF_CM and abs(f.boundingBox.maxPoint.x) < SLOT_HALF_CM]
    return {"back": back, "top": top, "bottom": bottom, "walls": walls, "roof": roof}

def _fillet(root, name, edges, radius):
    col = adsk.core.ObjectCollection.create()
    for e in edges: col.add(e)
    fi = root.features.filletFeatures.createInput()
    try:
        fi.edgeSetInputs.addConstantRadiusEdgeSet(col, adsk.core.ValueInput.createByString(radius), True)
    except AttributeError:
        fi.addConstantRadiusEdgeSet(col, adsk.core.ValueInput.createByString(radius), True)
    f = root.features.filletFeatures.add(fi); f.name = name
    return f

def _offset(root, name, faces, dist):
    oi = root.features.offsetFacesFeatures.createInput(list(faces), adsk.core.ValueInput.createByString(dist))
    f = root.features.offsetFacesFeatures.add(oi); f.name = name
    return f

def add_finish(des):
    """Make the finishing features, each from the body as it then is. One that fails is
    removed and reported; the next ones go on."""
    root = des.rootComponent
    finish_params(des)
    des.timeline.moveToEnd()
    try: des.computeAll()
    except Exception: pass
    log = {"made": [], "failed": {}, "counts": {}}
    steps = (("Back fillet", "back", "Back_Fillet_Radius", _fillet),
             ("Top fillet", "top", "Top_Fillet_Radius", _fillet),
             ("Bottom fillet", "bottom", "Bottom_Fillet_Radius", _fillet),
             ("Slot clearance", "walls", "-Slot_Wall_Clearance", _offset),
             ("Slot roof clearance", "roof", "-Slot_Roof_Clearance", _offset))
    for name, key, value, make in steps:
        sel = finish_rules(des)[key]
        log["counts"][name] = len(sel)
        if not sel:
            if key == "back":
                log["none_needed"] = name + ": no sharp posterior notch corner"
            else:
                log["failed"][name] = "nothing selected"
            continue
        # A fillet that won't fit is tried smaller (author 2026-10-10: 1 mm worked on
        # DEMO_LARGE's rim); the radius used is logged when it isn't the parameter.
        tries = [value] + (["min(%s; %s)" % (value, r) for r in ("1.5 mm", "1 mm", "0.5 mm")]
                           if make is _fillet else [])
        errs = []
        for v in tries:
            try:
                f = make(root, name, finish_rules(des)[key] if errs else sel, v)
                try: des.computeAll()
                except Exception: pass
                if f.healthState == 2:
                    errs.append(f.errorOrWarningMessage.split("<")[0][:120]); f.deleteMe()
                    continue
                log["made"].append(name)
                if v != value:
                    log.setdefault("smaller", {})[name] = v
                break
            except Exception as ex:
                errs.append(str(ex)[:150])
        else:
            log["failed"][name] = "error: " + " | ".join(errs)
    return log

def plate_at_origin(des):
    """The plate on the origin: its Tray offset move suppressed (kept, as the author did)."""
    tl = des.timeline; tl.moveToEnd()
    for i in range(tl.count):
        it = tl.item(i)
        if it.name == "Tray offset" and not it.isSuppressed:
            it.isSuppressed = True
            return "suppressed"
    return "already"
'''

S = r'''
import adsk.core, adsk.fusion, json, time
MODE = __MODE__
LINEAGE = __L__
MSG = __MSG__
__OUTLINE__
__FINISH__

def mm(p): return [round(p.x * 10, 2), round(p.y * 10, 2), round(p.z * 10, 2)]

def state(des):
    tl = des.timeline
    bad = [[tl.item(i).name, tl.item(i).healthState, tl.item(i).errorOrWarningMessage.split("<")[0][:100]]
           for i in range(tl.count) if tl.item(i).healthState]
    out = {"unhealthy": bad, "bodies": {}, "timeline": [tl.item(i).name for i in range(tl.count)]}
    for b in des.rootComponent.bRepBodies:
        bb = b.boundingBox
        out["bodies"][b.name] = [round(b.volume * 1000, 2), b.faces.count] + mm(bb.minPoint) + mm(bb.maxPoint)
    for o in des.rootComponent.occurrences:
        out["occ " + o.component.name] = mm(o.transform2.translation)
    return out

def selection(des):
    r = finish_rules(des)
    out = {}
    for k in ("back", "top", "bottom"):
        out[k] = sorted([mm(e.pointOnEdge) for e in r[k]])
    for k in ("walls", "roof"):
        out[k] = sorted([mm(f.pointOnFace) for f in r[k]])
    return out

def author_selection(des):
    root = des.rootComponent; out = {}
    for name, key in (("Back fillet", "back"), ("Top fillet", "top"), ("Bottom fillet", "bottom")):
        f = root.features.filletFeatures.itemByName(name)
        f.timelineObject.rollTo(True)
        out[key] = sorted([mm(e.pointOnEdge) for s in f.edgeSets for e in adsk.fusion.ConstantRadiusFilletEdgeSet.cast(s).edges])
        des.timeline.moveToEnd()
    for name, key in (("OffsetFaces1", "walls"), ("OffsetFaces2", "roof")):
        f = root.features.offsetFacesFeatures.itemByName(name)
        f.timelineObject.rollTo(True)
        out[key] = sorted([mm(x.pointOnFace) for x in f.inputFaces])
        des.timeline.moveToEnd()
    return out

def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    df = app.data.findFileById(LINEAGE)
    for d in list(app.documents):
        if d.isValid and d.dataFile and d.dataFile.id == LINEAGE:
            if d.isModified: raise RuntimeError("unsaved changes open on this file")
            d.close(False)
    out = {}
    if MODE == "check":
        df = [v for v in df.versions if v.versionNumber == 5][0]
    doc = app.documents.open(df, MODE != "check")
    des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
    try:
        out["before"] = state(des)
        if MODE == "check":
            # The author's picks, then this module's on the same body (the author's
            # finishing features removed in memory), then the result of making them.
            out["author"] = author_selection(des)
            strip_insert(des)
            des.computeAll()
            out["mine"] = selection(des)
            out["finish"] = add_finish(des)
            out["after"] = state(des)
        elif MODE == "save":
            out["strip"] = strip_insert(des)
            out["outline"] = rebuild_outline(des)
            out["tray"] = tray_offset(des)
            des.computeAll()
            out["finish"] = add_finish(des)
            out["after"] = state(des)
            if not [h for h in out["after"]["unhealthy"] if h[1] == 2] and not out["finish"]["failed"]:
                v0 = df.latestVersionNumber
                doc.save(MSG)
                for _ in range(120):
                    if app.data.findFileById(LINEAGE).latestVersionNumber > v0: break
                    time.sleep(1); adsk.doEvents()
                out["version_after"] = app.data.findFileById(LINEAGE).latestVersionNumber
            else:
                out["not_saved"] = True
        elif MODE == "plate":
            out["tray_offset"] = plate_at_origin(des)
            des.computeAll()
            out["after"] = state(des)
            if not [h for h in out["after"]["unhealthy"] if h[1] == 2]:
                v0 = df.latestVersionNumber
                doc.save(MSG)
                for _ in range(120):
                    if app.data.findFileById(LINEAGE).latestVersionNumber > v0: break
                    time.sleep(1); adsk.doEvents()
                out["version_after"] = app.data.findFileById(LINEAGE).latestVersionNumber
            else:
                out["not_saved"] = True
    finally:
        try: doc.close(False)
        except Exception: pass
        if active and active.isValid: active.activate()
    print(json.dumps(out))
'''


def main(mode, lineage=None):
    from insert_edit import OUTLINE
    lineage = lineage or AUTHOR_EDIT
    script = (S.replace("__OUTLINE__", OUTLINE).replace("__FINISH__", FINISH)
              .replace("__MODE__", repr(mode)).replace("__L__", repr(lineage))
              .replace("__MSG__", repr(PLATE_MSG if mode == "plate" else MSG)))
    res = json.loads(call_fusion(script, read_only=False).strip().splitlines()[-1])
    (HERE / "tray_audit" / f"insert_finish_{mode}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    res = main(mode, sys.argv[2] if len(sys.argv) > 2 else None)
    if mode == "check":
        for k in ("back", "top", "bottom", "walls", "roof"):
            a, m = res["author"][k], res["mine"][k]
            same = len(a) == len(m) and all(max(abs(u - v) for u, v in zip(p, q)) < 0.05 for p, q in zip(a, m))
            print(f"{k:7s} author {len(a):3d}  mine {len(m):3d}  {'SAME' if same else 'DIFFERENT'}")
            if not same:
                print("   author:", a); print("   mine:  ", m)
        print("finish:", res["finish"])
        print("author body:", res["before"]["bodies"]); print("my body:    ", res["after"]["bodies"])
        print("unhealthy:", res["after"]["unhealthy"])
    else:
        print(json.dumps({k: v for k, v in res.items() if k != "before"}, indent=1)[:4000])
