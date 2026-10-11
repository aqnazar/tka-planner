"""Snapshot Fusion documents and diff them: the author's sandbox edits vs the originals.

Private (CAD design data). Read-only in Fusion: it opens the requested versions, reads
them, and closes whatever it opened without saving.

  python fusion_diff.py snap <label> <lineage-or-alias> [version]
  python fusion_diff.py diff <label-a> <label-b>
  python fusion_diff.py history <alias>          # versions and their save comments

Aliases: sandbox-femoral, sandbox-plate, sandbox-insert, sandbox-assembly, sandbox-block,
work-block (Driver Tests cutting block),
work-femoral, work-plate, work-insert, work-assembly, orig-femoral, orig-plate,
orig-insert, orig-assembly.

A snapshot is snapshots/<label>.json (parameters, sketches, features, bodies,
occurrences) plus snapshots/<label>__<body>.stl for every visible body.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyproject.toml").exists())))
from fusion_driver import call_fusion  # noqa: E402

SNAPS = HERE / "fusion_snapshots"
ALIASES = {
    "sandbox-femoral": "urn:adsk.wipprod:dm.lineage:WtyZ2arYROSbM6x0qTXWcw",
    "sandbox-plate": "urn:adsk.wipprod:dm.lineage:T8Hb_Ws3Qlaj-oaYqCZTCw",
    "sandbox-insert": "urn:adsk.wipprod:dm.lineage:6gUVQt7UQiqGiAfqclNQWw",
    "sandbox-assembly": "urn:adsk.wipprod:dm.lineage:5Y8LVtJCQI-CtcSUGy9Eaw",
    "sandbox-block": "urn:adsk.wipprod:dm.lineage:mUWojBOhT_iFIIiY5drmLA",
    "work-block": "urn:adsk.wipprod:dm.lineage:9lT4NUH_SBKAzWwIPChdWQ",
    "work-femoral": "urn:adsk.wipprod:dm.lineage:UrLmgsJ_RSSnpPRBRBXOGA",
    "work-plate": "urn:adsk.wipprod:dm.lineage:J5aaIRrSQlyRnBYsbR8qgw",
    "work-insert": "urn:adsk.wipprod:dm.lineage:n2dMD8onQZ2Y8QSApc3qog",
    "work-assembly": "urn:adsk.wipprod:dm.lineage:oVeVKrZKR4OelfUhyel3lA",
    "orig-femoral": "urn:adsk.wipprod:dm.lineage:Y8blKrpvQuCtIui15PXDAw",
    "orig-plate": "urn:adsk.wipprod:dm.lineage:e1pcsAAhSW-ADKtc73TWPw",
    "orig-insert": "urn:adsk.wipprod:dm.lineage:O_Ya49brSK-SP2OegBYtqQ",
    "orig-assembly": "urn:adsk.wipprod:dm.lineage:CX9OQKzZTUysC8LqRnKSQQ",
    # The author's editable copies of each knee's insert (Patient Builds/<knee>/Author Edits),
    # made 2026-10-10; v1 of each is identical to the build's insert.
    "edit-insert-REF_KJ2": "urn:adsk.wipprod:dm.lineage:tsySxxqUQVW4wZqZ5wmOgg",
    "edit-insert-DEMO_SMALL": "urn:adsk.wipprod:dm.lineage:OWzEWQ8FRL6trLfD6zrtiQ",
    "edit-insert-DEMO_LARGE": "urn:adsk.wipprod:dm.lineage:bG9jNYgwT9SkPeR0zrfrrg",
    "edit-insert-P007L": "urn:adsk.wipprod:dm.lineage:mCNMimlyTN6HKID-pTATQA",
    "edit-insert-P007R": "urn:adsk.wipprod:dm.lineage:4r-XeEg0T9eklA_qg53HRQ",
    "edit-insert-P009R": "urn:adsk.wipprod:dm.lineage:ziXSIXsbRyip1Ws3tuQjKw",
}

SNAP_SCRIPT = r'''
import adsk.core, adsk.fusion, json, struct
JOB = json.load(open(__JOB__, encoding="utf-8"))

def mm3(p): return [round(p.x * 10, 4), round(p.y * 10, 4), round(p.z * 10, 4)]

def write_stl(path, c, t):
    n = len(t) // 3
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<3f", 0, 0, 0))
            for k in range(3):
                j = t[3*i+k]; f.write(struct.pack("<3f", c[3*j]*10, c[3*j+1]*10, c[3*j+2]*10))
            f.write(b"\0\0")

def curve(c):
    g = c.worldGeometry
    rec = {"type": c.objectType.split("::")[-1], "construction": c.isConstruction,
           "reference": c.isReference, "fixed": getattr(c, "isFixed", False),
           "fully": c.isFullyConstrained}
    if hasattr(g, "center") and hasattr(g, "radius"):
        rec["centre"] = mm3(g.center); rec["r"] = round(g.radius * 10, 4)
    if hasattr(g, "startPoint"):
        rec["a"] = mm3(g.startPoint); rec["b"] = mm3(g.endPoint)
    return rec

def snapshot(des, out_base):
    root = des.rootComponent
    snap = {"params": {}, "sketches": {}, "features": [], "bodies": {}, "occurrences": {}}
    for p in des.allParameters:
        mp = adsk.fusion.ModelParameter.cast(p)
        owner = ""
        if mp:
            try: owner = mp.createdBy.name
            except Exception: owner = "?"
        snap["params"][p.name] = {"expression": p.expression, "unit": p.unit,
                                  "value": round(p.value * (10 if p.unit == "mm" else 1), 6),
                                  "kind": "model" if mp else "user", "owner": owner,
                                  "comment": p.comment}
    for comp in [root] + [o.component for o in root.allOccurrences if not o.isReferencedComponent]:
        for sk in comp.sketches:
            key = (comp.name + "/" if comp != root else "") + sk.name
            if key in snap["sketches"]: key += f" #{sk.timelineObject.index if sk.timelineObject else ''}"
            try:
                plane = getattr(sk.referencePlane, "name", "?")
            except Exception:
                plane = "(body face)"
            snap["sketches"][key] = {
                "plane": plane,
                "fully_constrained": sk.isFullyConstrained,
                "dims": {d.parameter.name: d.parameter.expression for d in sk.sketchDimensions},
                "constraints": sorted(k.objectType.split("::")[-1] for k in sk.geometricConstraints),
                "curves": [curve(c) for c in sk.sketchCurves],
            }
    tl = des.timeline
    for i in range(tl.count):
        it = tl.item(i)
        try: msg = it.errorOrWarningMessage if it.healthState else ""
        except Exception: msg = ""
        snap["features"].append({"i": i, "name": it.name,
                                 "type": it.entity.objectType.split("::")[-1] if it.entity else "?",
                                 "health": it.healthState, "suppressed": it.isSuppressed,
                                 "message": msg.split("<")[0][:200]})
    for comp in [root] + [o.component for o in root.allOccurrences if not o.isReferencedComponent]:
        for b in comp.bRepBodies:
            key = (comp.name + "/" if comp != root else "") + b.name
            bb = b.boundingBox
            snap["bodies"][key] = {"visible": b.isVisible, "volume_mm3": round(b.volume * 1000, 3),
                                   "area_mm2": round(b.area * 100, 3), "faces": b.faces.count,
                                   "edges": b.edges.count, "min": mm3(bb.minPoint), "max": mm3(bb.maxPoint)}
            if b.isVisible and comp == root:
                calc = b.meshManager.createMeshCalculator()
                calc.setQuality(adsk.fusion.TriangleMeshQualityOptions.HighQualityTriangleMesh)
                m = calc.calculate()
                safe = "".join(ch if ch.isalnum() else "_" for ch in b.name)
                write_stl(out_base + "__" + safe + ".stl", list(m.nodeCoordinatesAsDouble), list(m.nodeIndices))
    for occ in root.allOccurrences:
        t = occ.transform2
        ref, folder = "", ""
        try:
            if occ.isReferencedComponent and occ.documentReference:
                rdf = occ.documentReference.dataFile
                ref, folder = f"{rdf.name} v{rdf.versionNumber}", rdf.parentFolder.name
        except Exception:
            pass
        snap["occurrences"][occ.fullPathName] = {
            "matrix": [round(t.getCell(r, c) * (10 if c == 3 else 1), 4) for r in range(3) for c in range(4)],
            "referenced": ref, "folder": folder}
    try:
        snap["references"] = [(r.dataFile.name, r.dataFile.versionNumber, r.dataFile.parentFolder.name) for r in des.parentDocument.documentReferences]
    except Exception:
        snap["references"] = []
    return snap

def run(_context: str):
    app = adsk.core.Application.get(); active = app.activeDocument
    lineage = app.data.findFileById(JOB["lineage"])
    if JOB.get("history"):
        print(json.dumps([(v.versionNumber, v.description, str(v.dateModified)) for v in lineage.versions]))
        return
    want = JOB.get("version") or lineage.latestVersionNumber
    df = next(v for v in lineage.versions if v.versionNumber == want)
    doc = next((d for d in app.documents if d.isValid and d.dataFile and d.dataFile.id == lineage.id
                and d.dataFile.versionNumber == want and not d.isModified), None)
    opened = doc is None
    if opened:
        doc = app.documents.open(df, False)
    try:
        des = adsk.fusion.Design.cast(doc.products.itemByProductType("DesignProductType"))
        snap = snapshot(des, JOB["out_base"])
        snap["file"] = {"name": lineage.name, "version": want, "latest": lineage.latestVersionNumber,
                        "folder": lineage.parentFolder.name, "lineage": lineage.id,
                        "description": df.description, "modified_in_memory": doc.isModified}
        json.dump(snap, open(JOB["out_base"] + ".json", "w", encoding="utf-8"), indent=1)
        print("ok", len(snap["params"]), "params", len(snap["sketches"]), "sketches",
              len(snap["features"]), "features", len(snap["bodies"]), "bodies")
    finally:
        if opened:
            doc.close(False)
        if active and active.isValid:
            active.activate()
'''


def fusion(job):
    SNAPS.mkdir(exist_ok=True)
    path = SNAPS / "_job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return call_fusion(SNAP_SCRIPT.replace("__JOB__", repr(str(path))), read_only=False)


def snap(label, target, version=None):
    lineage = ALIASES.get(target, target)
    print(fusion({"lineage": lineage, "version": int(version) if version else None,
                  "out_base": str(SNAPS / label)}).strip())


def history(target):
    for v, desc, date in json.loads(fusion({"lineage": ALIASES.get(target, target),
                                            "history": True}).strip()):
        print(f"v{v:<3d} {date}  {desc}")


def _dict_diff(a, b, what, fmt):
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            out.append(f"+ {what} {k}: {fmt(b[k])}")
        elif k not in b:
            out.append(f"- {what} {k}: {fmt(a[k])}")
        elif a[k] != b[k]:
            out.append(f"~ {what} {k}: {fmt(a[k])}  ->  {fmt(b[k])}")
    return out


def _deviation(pa, pb):
    """Symmetric vertex-to-nearest-vertex distances between two STLs (mm)."""
    from tka_planner.core.meshio import read_stl
    a, b = read_stl(pa).vertices, read_stl(pb).vertices

    def one_way(p, q):
        d = np.empty(len(p))
        for i in range(0, len(p), 2000):
            chunk = p[i:i + 2000]
            d[i:i + 2000] = np.sqrt(((chunk[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(1)
        return d
    rng = np.random.default_rng(0)
    a = a[rng.choice(len(a), min(len(a), 6000), replace=False)]
    b = b[rng.choice(len(b), min(len(b), 6000), replace=False)]
    d = np.concatenate([one_way(a, b), one_way(b, a)])
    return float(np.median(d)), float(np.percentile(d, 95)), float(d.max())


def diff(la, lb):
    A = json.loads((SNAPS / f"{la}.json").read_text(encoding="utf-8"))
    B = json.loads((SNAPS / f"{lb}.json").read_text(encoding="utf-8"))
    lines = [f"# {la} ({A['file']['name']} v{A['file']['version']}, {A['file']['folder']})  vs  "
             f"{lb} ({B['file']['name']} v{B['file']['version']}, {B['file']['folder']})", ""]
    pa = {k: v["expression"] for k, v in A["params"].items()}
    pb = {k: v["expression"] for k, v in B["params"].items()}
    lines += ["## Parameters (expression)"] + (_dict_diff(pa, pb, "param", str) or ["no change"]) + [""]
    lines.append("## Sketches")
    sk = []
    for k in sorted(set(A["sketches"]) | set(B["sketches"])):
        if k not in A["sketches"] or k not in B["sketches"]:
            sk.append(f"{'+' if k in B['sketches'] else '-'} sketch {k}")
            continue
        a, b = A["sketches"][k], B["sketches"][k]
        if a["fully_constrained"] != b["fully_constrained"]:
            sk.append(f"~ sketch {k}: fully constrained {a['fully_constrained']} -> {b['fully_constrained']}")
        sk += [f"  in {k}: {x}" for x in _dict_diff(a["dims"], b["dims"], "dim", str)]
        if a["constraints"] != b["constraints"]:
            from collections import Counter
            ca, cb = Counter(a["constraints"]), Counter(b["constraints"])
            delta = {t: cb[t] - ca[t] for t in set(ca) | set(cb) if cb[t] != ca[t]}
            sk.append(f"~ sketch {k}: constraints {delta}")
        proj_a = sum(c["reference"] for c in a["curves"])
        proj_b = sum(c["reference"] for c in b["curves"])
        if proj_a != proj_b:
            sk.append(f"~ sketch {k}: projected curves {proj_a} -> {proj_b}")
        if len(a["curves"]) != len(b["curves"]):
            sk.append(f"~ sketch {k}: curves {len(a['curves'])} -> {len(b['curves'])}")
        elif a["curves"] != b["curves"]:
            moved = sum(1 for x, y in zip(a["curves"], b["curves"]) if x != y)
            sk.append(f"~ sketch {k}: {moved} curve(s) moved or changed")
    lines += (sk or ["no change"]) + [""]
    lines.append("## Timeline")
    fa = [(f["name"], f["type"], f["suppressed"]) for f in A["features"]]
    fb = [(f["name"], f["type"], f["suppressed"]) for f in B["features"]]
    tl = []
    if fa != fb:
        tl.append("order/content changed:")
        tl += [f"  - {x}" for x in fa if x not in fb] + [f"  + {x}" for x in fb if x not in fa]
        if [x for x in fa if x in fb] != [x for x in fb if x in fa]:
            tl.append("  (features reordered)")
    for f in B["features"]:
        if f["health"]:
            tl.append(f"  {lb}: {f['name']} health {f['health']}: {f['message']}")
    lines += (tl or ["no change, all healthy"]) + [""]
    lines.append("## Bodies")
    bd = _dict_diff(A["bodies"], B["bodies"], "body",
                    lambda v: f"vol {v['volume_mm3']} faces {v['faces']} min {v['min']} max {v['max']}")
    for k in sorted(set(A["bodies"]) & set(B["bodies"])):
        safe = "".join(ch if ch.isalnum() else "_" for ch in k)
        fa_, fb_ = SNAPS / f"{la}__{safe}.stl", SNAPS / f"{lb}__{safe}.stl"
        if A["bodies"][k] != B["bodies"][k] and fa_.exists() and fb_.exists():
            med, p95, mx = _deviation(fa_, fb_)
            bd.append(f"  {k}: surface deviation median {med:.3f}, p95 {p95:.3f}, max {mx:.3f} mm")
    lines += (bd or ["no change"]) + [""]
    lines.append("## Occurrences")
    # Fusion writes the referenced version into occurrence names ("... v2:1"); a new
    # save would otherwise read as a removed and an added part.
    strip = lambda d: {re.sub(r" v\d+(?=:\d+)", "", k): v for k, v in d.items()}
    A["occurrences"], B["occurrences"] = strip(A["occurrences"]), strip(B["occurrences"])
    oc = []
    for k in sorted(set(A["occurrences"]) | set(B["occurrences"])):
        a, b = A["occurrences"].get(k), B["occurrences"].get(k)
        if a is None or b is None:
            oc.append(f"{'+' if b else '-'} {k}")
        elif a["matrix"] != b["matrix"]:
            worst = max(abs(x - y) for x, y in zip(a["matrix"], b["matrix"]))
            oc.append(f"~ {k}: moved (max matrix change {worst})")
    lines += (oc or ["no change"]) + [""]
    text = "\n".join(lines)
    out = SNAPS / f"diff__{la}__{lb}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "snap":
        snap(*sys.argv[2:])
    elif cmd == "diff":
        diff(*sys.argv[2:4])
    elif cmd == "history":
        history(sys.argv[2])
