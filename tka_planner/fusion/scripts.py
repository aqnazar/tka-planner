"""The Fusion-side script blocks the build runs, copied verbatim from eval/.

Each is Python that runs inside Fusion 360 (the adsk API), sent as text through the
MCP add-in. They are the eval scripts' own blocks, unchanged: tests/test_fusion_build.py
checks them against eval/ while eval/ still carries them. No file identifiers are in
them; the build reads those from fusion/local.json.
"""

# ruff: noqa

# insert_edit.OUTLINE: the insert's outline rebuilt from this knee's plate
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
        def neighbour_arc(cur, sp, skip=None):
            # The offset arc that meets sp, from the offset's curves or, when those are
            # gone, from the sketch as it is now: a constraint Fusion cannot solve against
            # the offset makes it remake the whole offset loop ("failed to create offset"),
            # and the offset's curves are then deleted objects (2026-11-08: P047L, "An API
            # Object refers to a deleted Object").
            for c in list(cur) + list(sk.sketchCurves.sketchArcs):
                if c.isValid and c != skip and c.objectType.endswith("SketchArc") and any(
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
            def same(a):
                # The arc it continues, found again when Fusion has remade the offset.
                return a if a.isValid else (neighbour_arc([], p, new) or a)
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
                    arc = same(arc)
                    sk.geometricConstraints.addConcentric(new, arc); sk.geometricConstraints.addEqual(new, arc)
                    log.setdefault("bridge_fix", []).append("concentric")
                except Exception:
                    # Already on that circle (both refused as redundant): the radius
                    # alone holds its bulge, which the ends leave free.
                    try:
                        arc = same(arc)
                        sk.geometricConstraints.addEqual(new, arc); log.setdefault("bridge_fix", []).append("equal")
                    except Exception:
                        # Fusion takes no constraint against an offset curve here; a radius
                        # on the bridge itself (it was built on that circle) pins it.
                        try:
                            arc = same(arc)
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

# insert_finish.FINISH: the insert's finishing features (snap pins, fillets, clearances)
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

def design_profile(kinds):
    """Whether a Condylar Profile, by its constraint kinds, is still the design's own
    (sides held tangent to the condyle curve) and not the box remake_profile draws
    (sides vertical, top horizontal)."""
    box = kinds.count("VerticalConstraint") >= 2 and "HorizontalConstraint" in kinds
    return "TangentConstraint" in kinds or not box

def profile_before_replace(des):
    """The design's own Condylar Profile joins its box to the projected condyle curve by
    shared points. When the next femoral's condyle ends lower than the design's, the
    joins let go, the box opens, both sweeps lose their profile ("Profile 1 missing"),
    and a sweep without a profile refuses a new one (InternalValidationError
    curProfile || curSolidBody), so the remake after the replace cannot mend it
    (2026-11-08: DEMO_SMALL, P007L, P007R on fresh copies). Remade first, on the femoral
    the design was drawn on, the box follows the next femoral by its explicit constraints."""
    sk = des.rootComponent.sketches.itemByName("Condylar Profile")
    if sk is None: return "no Condylar Profile"
    if not design_profile([g.objectType.split("::")[-1] for g in sk.geometricConstraints]):
        return "box already"
    return "remade before the replace: " + coronal_profile(des)

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

# femoral_block.BLOCK: the femoral cutting block on this knee's femoral J-curve
BLOCK = r'''
BLOCK_PARAMS = (("Block_Top", "150 mm", "How far above the distal cut the anterior and posterior cuts reach"),
                ("Block_Reach", "150 mm", "Outer profile: anterior, posterior and below the distal cut"),
                ("Block_Width", "300 mm", "ML extent, symmetric about the femoral's XZ plane"))
BOX_SKETCH = "J-curve"

def block_params(des):
    up = des.userParameters; added = []
    for n, e, c in BLOCK_PARAMS:
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString(e), "mm", c); added.append(n)
    return added

def femoral_occurrence(des):
    for o in des.rootComponent.occurrences:
        if o.isReferencedComponent and o.component.sketches.itemByName(BOX_SKETCH) is not None:
            return o
    raise RuntimeError("no femoral reference with a J-curve sketch")

def box_lines(occ):
    """The J-curve sketch's five cut lines, by position: the distal on Z = 0, the posterior
    the vertical one, the anterior the one reaching highest, the chamfers the other two."""
    sk = occ.component.sketches.itemByName(BOX_SKETCH)
    lines = [c for c in sk.sketchCurves.sketchLines if not c.isConstruction]
    if len(lines) != 5:
        raise RuntimeError(f"J-curve has {len(lines)} cut lines, expected 5")
    g = lambda l: (l.worldGeometry.startPoint, l.worldGeometry.endPoint)
    out = {}
    out["distal"] = min(lines, key=lambda l: max(abs(p.z) for p in g(l)))
    out["posterior"] = min(lines, key=lambda l: abs(g(l)[0].x - g(l)[1].x) + 10 * (l is out["distal"]))
    out["anterior"] = max(lines, key=lambda l: max(p.z for p in g(l)))
    rest = [l for l in lines if all(l is not v for v in out.values())]
    rest.sort(key=lambda l: sum(p.x for p in g(l)))
    out["posterior chamfer"], out["anterior chamfer"] = rest
    return out

def strip_block(des):
    """Remove Pegs, Block and Block profile: every build makes them again from the knee's
    J-curve (a swapped femoral can lose the extrusion's profile, "Profile 1 missing")."""
    root = des.rootComponent; gone = []
    for coll, name in ((root.features.combineFeatures, "Pegs"), (root.features.extrudeFeatures, "Block"),
                       (root.sketches, "Block profile")):
        f = coll.itemByName(name)
        if f is not None:
            f.deleteMe(); gone.append(name)
    return gone

def make_block(des):
    """Block profile and Block (the extrusion), from the femoral reference's J-curve."""
    root = des.rootComponent
    occ = femoral_occurrence(des)
    sk = root.sketches.add(root.xZConstructionPlane); sk.name = "Block profile"
    proj = {}
    for key, line in box_lines(occ).items():
        proj[key] = adsk.fusion.SketchLine.cast(sk.project(line.createForAssemblyContext(occ)).item(0))
    def top(l):
        a, b = l.startSketchPoint, l.endSketchPoint
        return a if a.worldGeometry.z > b.worldGeometry.z else b
    def bottom(l):
        a, b = l.startSketchPoint, l.endSketchPoint
        return b if a.worldGeometry.z > b.worldGeometry.z else a
    up = des.userParameters
    T, R = up.itemByName("Block_Top").value, up.itemByName("Block_Reach").value
    P = lambda x, z: sk.modelToSketchSpace(adsk.core.Point3D.create(x, 0, z))
    pt, at = top(proj["posterior"]), top(proj["anterior"])
    ab = bottom(proj["anterior"]).worldGeometry; aw = at.worldGeometry
    xa = aw.x + (aw.x - ab.x) * (T - aw.z) / (aw.z - ab.z)
    L = sk.sketchCurves.sketchLines
    l1 = L.addByTwoPoints(pt, P(pt.worldGeometry.x, T))
    l2 = L.addByTwoPoints(l1.endSketchPoint, P(-R, T))
    l3 = L.addByTwoPoints(l2.endSketchPoint, P(-R, -R))
    l4 = L.addByTwoPoints(l3.endSketchPoint, P(R, -R))
    l5 = L.addByTwoPoints(l4.endSketchPoint, P(R, T))
    l6 = L.addByTwoPoints(l5.endSketchPoint, P(xa, T))
    l7 = L.addByTwoPoints(l6.endSketchPoint, at)
    gc = sk.geometricConstraints
    gc.addCollinear(l1, proj["posterior"]); gc.addCollinear(l7, proj["anterior"])
    gc.addHorizontal(l2); gc.addHorizontal(l4); gc.addCollinear(l6, l2)
    gc.addVertical(l3); gc.addVertical(l5)
    D = sk.sketchDimensions; o = sk.originPoint
    H = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    V = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for p, orient, expr, tx in ((l2.startSketchPoint, V, "Block_Top", P(-R / 2, T / 2)),
                                (l3.startSketchPoint, H, "Block_Reach", P(-R / 2, T + 1)),
                                (l4.startSketchPoint, V, "Block_Reach", P(-R - 1, -R / 2)),
                                (l5.startSketchPoint, H, "Block_Reach", P(R / 2, -R - 1))):
        D.addDistanceDimension(o, p, orient, tx).parameter.expression = expr
    prof = max(sk.profiles, key=lambda p: p.areaProperties().area)
    ext = root.features.extrudeFeatures
    ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ei.setSymmetricExtent(adsk.core.ValueInput.createByString("Block_Width"), True)
    f = ext.add(ei); f.name = "Block"; f.bodies.item(0).name = "Block"
    return {"profiles": sk.profiles.count, "fully_constrained": sk.isFullyConstrained,
            "lines": {k: [round(v * 10, 2) for v in (l.worldGeometry.startPoint.x, l.worldGeometry.startPoint.z,
                                                      l.worldGeometry.endPoint.x, l.worldGeometry.endPoint.z)]
                      for k, l in proj.items()}}

def block_pegs(des, side_body):
    """Pegs: the knee's femoral body joined to the block, kept; remade every build."""
    root = des.rootComponent
    old = root.features.combineFeatures.itemByName("Pegs")
    if old is not None:
        old.deleteMe()
    occ = femoral_occurrence(des)
    fb = occ.component.bRepBodies.itemByName(side_body)
    block = root.bRepBodies.itemByName("Block")
    v0 = block.volume
    tools = adsk.core.ObjectCollection.create(); tools.add(fb.createForAssemblyContext(occ))
    ci = root.features.combineFeatures.createInput(block, tools)
    ci.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    ci.isKeepToolBodies = True
    cf = root.features.combineFeatures.add(ci); cf.name = "Pegs"
    try: des.computeAll()
    except Exception: pass
    b = root.bRepBodies.itemByName("Block")
    return {"side_body": side_body, "health": cf.healthState,
            "added_mm3": round((b.volume - v0) * 1000, 1) if b else None}

def block_state(des):
    root = des.rootComponent; out = {}
    b = root.bRepBodies.itemByName("Block")
    if b:
        bb = b.boundingBox
        out["block"] = [round(b.volume * 1000, 1), b.faces.count] + [round(v * 10, 2) for v in (
            bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    sk = root.sketches.itemByName("Block profile")
    if sk:
        out["profile_constrained"] = sk.isFullyConstrained
        out["cut_lines"] = {k: [round(v * 10, 2) for v in (l.worldGeometry.startPoint.x, l.worldGeometry.startPoint.z,
                                                            l.worldGeometry.endPoint.x, l.worldGeometry.endPoint.z)]
                            for k, l in box_lines(femoral_occurrence(des)).items()}
    return out
'''

# tibial_block.TBLOCK: the tibial cutting block on this knee's plate
TBLOCK = r'''
TBLOCK_PARAMS = (("Block_Top", "150 mm", "How far above the tibial cut the block reaches"),
                 ("Block_Reach", "150 mm", "Half the block's width and depth, about the origin"))

def plate_occurrence(des):
    for o in des.rootComponent.occurrences:
        if o.isReferencedComponent and o.component.bRepBodies.itemByName("Body1") is not None \
                and "Plate" in o.component.name:
            return o
    raise RuntimeError("no plate reference")

def plate_base_z(occ):
    """Z of the plate's base: its largest downward planar face (mm)."""
    best = None
    for f in occ.component.bRepBodies.itemByName("Body1").createForAssemblyContext(occ).faces:
        if f.geometry.objectType != adsk.core.Plane.classType(): continue
        ok, n = f.evaluator.getNormalAtPoint(f.pointOnFace)
        if n.z < -0.999 and (best is None or f.area > best.area):
            best = f
    return round(best.pointOnFace.z * 10, 4) if best else None

def tblock_params(des):
    up = des.userParameters; added = []
    for n, e, c in TBLOCK_PARAMS:
        if up.itemByName(n) is None:
            up.add(n, adsk.core.ValueInput.createByString(e), "mm", c); added.append(n)
    return added

def strip_tblock(des):
    root = des.rootComponent; gone = []
    for coll, name in ((root.features.combineFeatures, "Keel"), (root.features.extrudeFeatures, "Block"),
                       (root.sketches, "Block profile")):
        f = coll.itemByName(name)
        if f is not None:
            f.deleteMe(); gone.append(name)
    return gone

def make_tblock(des):
    root = des.rootComponent
    sk = root.sketches.add(root.xYConstructionPlane); sk.name = "Block profile"
    R = des.userParameters.itemByName("Block_Reach").value
    P = lambda x, y: sk.modelToSketchSpace(adsk.core.Point3D.create(x, y, 0))
    L = sk.sketchCurves.sketchLines
    a = L.addByTwoPoints(P(-R, -R), P(R, -R))
    b = L.addByTwoPoints(a.endSketchPoint, P(R, R))
    c = L.addByTwoPoints(b.endSketchPoint, P(-R, R))
    d = L.addByTwoPoints(c.endSketchPoint, a.startSketchPoint)
    gc = sk.geometricConstraints
    gc.addHorizontal(a); gc.addHorizontal(c); gc.addVertical(b); gc.addVertical(d)
    D = sk.sketchDimensions; o = sk.originPoint
    H = adsk.fusion.DimensionOrientations.HorizontalDimensionOrientation
    V = adsk.fusion.DimensionOrientations.VerticalDimensionOrientation
    for p, orient, tx in ((a.startSketchPoint, H, P(-R / 2, -R - 1)), (a.startSketchPoint, V, P(-R - 1, -R / 2)),
                          (c.startSketchPoint, H, P(R / 2, R + 1)), (c.startSketchPoint, V, P(R + 1, R / 2))):
        D.addDistanceDimension(o, p, orient, tx).parameter.expression = "Block_Reach"
    prof = max(sk.profiles, key=lambda p: p.areaProperties().area)
    ext = root.features.extrudeFeatures
    ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    ei.setDistanceExtent(False, adsk.core.ValueInput.createByString("Block_Top"))
    f = ext.add(ei); f.name = "Block"; b = f.bodies.item(0); b.name = "Block"
    if b.boundingBox.maxPoint.z < 0.1:                      # extruded the wrong way: flip
        f.deleteMe()
        ei = ext.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        ei.setOneSideExtent(adsk.fusion.DistanceExtentDefinition.create(adsk.core.ValueInput.createByString("Block_Top")),
                            adsk.fusion.ExtentDirections.NegativeExtentDirection)
        f = ext.add(ei); f.name = "Block"; f.bodies.item(0).name = "Block"
    return {"fully_constrained": sk.isFullyConstrained}

def tblock_keel(des):
    root = des.rootComponent
    occ = plate_occurrence(des)
    block = root.bRepBodies.itemByName("Block"); v0 = block.volume
    tools = adsk.core.ObjectCollection.create()
    tools.add(occ.component.bRepBodies.itemByName("Body1").createForAssemblyContext(occ))
    ci = root.features.combineFeatures.createInput(block, tools)
    ci.operation = adsk.fusion.FeatureOperations.JoinFeatureOperation
    ci.isKeepToolBodies = True
    cf = root.features.combineFeatures.add(ci); cf.name = "Keel"
    try: des.computeAll()
    except Exception: pass
    b = root.bRepBodies.itemByName("Block")
    return {"health": cf.healthState, "added_mm3": round((b.volume - v0) * 1000, 1) if b else None}

def tblock_state(des):
    root = des.rootComponent; out = {}
    b = root.bRepBodies.itemByName("Block")
    if b:
        bb = b.boundingBox
        out["block"] = [round(b.volume * 1000, 1), b.faces.count] + [round(v * 10, 2) for v in (
            bb.minPoint.x, bb.minPoint.y, bb.minPoint.z, bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z)]
    out["plate_base_z_mm"] = plate_base_z(plate_occurrence(des))
    sk = root.sketches.itemByName("Block profile")
    if sk: out["profile_constrained"] = sk.isFullyConstrained
    return out
'''

# implant_pipeline.BUILD: the build script as eval/ runs it (the package's BUILD adds the save gate)
BUILD_EVAL = r'''
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
        # A suppressed item (state 3) is a choice, not a fault (plate "Tray offset").
        if it.healthState and not it.isSuppressed:
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
        # A warning on a sweep means it lost its profile and runs on cached geometry, so the
        # part is wrong even though Fusion calls it a warning: never save or export it.
        errors = [h for h in rec["health"] if h[1] == 2 or (strict and h[1] == 1)
                  or (h[1] == 1 and "Sweep" in h[0])]
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
            # The design's own Condylar Profile is remade while its femoral is still the
            # one it was drawn on (insert_finish.profile_before_replace).
            synced["profile_first"] = profile_before_replace(des)
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
