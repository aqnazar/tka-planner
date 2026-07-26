# TKA Auto Pipeline — Analysis & Action Plan

---

## Current State Summary

The script works end-to-end: it imports bones, measures them, selects sizes, positions components, performs booleans, and exports. The architectural structure (shared `prepare_bone()`, separate femur/tibia pipelines, Z=0 joint line convention) is sound. The issues are in **measurement interpretation**, **centering logic**, **offset handling**, and **data management**. None of these require rearchitecting — they're targeted fixes.

---

## Issue 1: Sizing — Bones Select Too Small, Femur ≠ Tibia

### What's happening

The script measures the **full bounding box** ML and AP of the entire bone, then matches against the size chart. Two problems:

**Problem 1A — Measurement is of the whole bone, not the articular surface.**
The size chart values (e.g. `femur_ML = 72 mm`) represent the **distal condylar width** of the femur and the **proximal plateau width** of the tibia — the dimensions of the articular surfaces that the implant sits on. But `get_bounding_box()` returns the ML/AP extent of the *entire bone*, including the shaft. For the femur this is usually fine (the condyles are the widest part in ML), but in AP the condylar AP is much larger than the shaft AP, and the full-bone AP includes the femoral head region which can skew the number. For the tibia, the proximal plateau is almost always the widest ML section, so this is less of an issue — but any bone with an unusual shaft geometry could throw it off.

**More importantly — the `select_size()` function picks the *nearest* size, which can round down.** If a bone measures 73 mm ML and the chart has M2=72 and M3=74, it picks M2. In orthopaedic practice, **you virtually always round up** (slight overhang is preferable to undercoverage). This is very likely the primary cause of the "too small" result.

**Problem 1B — Femur and tibia select sizes independently.**
The script runs `select_size()` separately for femur (against `femur_ML`) and tibia (against `tibia_ML`). Since the chart values differ between bones (e.g. S4: femur_ML=68, tibia_ML=70), a bone pair can land on different sizes. The user wants a single unified size driven by the femur, with a warning if the tibia measurement disagrees.

### Action plan

1. **Change `select_size()` to round up (next size up) when between sizes**, not nearest. Only round down if the measurement is closer to the smaller size by more than, say, 1 mm tolerance. This is the standard surgical logic: "if in doubt, go bigger."

2. **Use a regional bounding box for measurement instead of the full bone.** For the femur: measure ML and AP of only the distal 20-25% of the bone (the condylar region, where Z < some threshold above z_min). For the tibia: measure ML and AP of only the proximal 20-25% (where Z > some threshold below z_max). This gives the actual articular surface dimensions rather than shaft dimensions. This is a more robust approach than relying on the full bbox.

3. **Unify size selection: femur drives the size, tibia cross-checks.**
   - Run `select_size()` on the femur ML → this is the system size.
   - Measure the tibia ML and run `select_size()` on it for informational purposes only.
   - If tibia would have selected a different size, print a clear warning with both sizes and the measurements. The pipeline still uses the femur-driven size for all components.
   - The `run_tibia()` function should receive `selected` (the femur's size dict) from `femur_results`, not run its own selection.

### Code sketch

```python
def select_size(measurement_m, bone, mode, size_table):
    """Round-up logic: pick the smallest size that is >= the measurement.
       Fall back to nearest if measurement exceeds the largest size."""
    key = f"{bone}_{mode}"
    measure_mm = measurement_m * 1000

    # Sort sizes ascending by the relevant dimension
    sorted_sizes = sorted(size_table, key=lambda s: s[key])

    # Pick the smallest size that covers the bone (>= measurement)
    for s in sorted_sizes:
        if s[key] >= measure_mm:
            return s

    # If measurement exceeds all sizes, return the largest
    return sorted_sizes[-1]
```

---

## Issue 2: Femoral Component Y-Offset (Off-Centre)

### What's happening

The femoral cutting block and implant are placed at `XY = (0, 0)`, which is the **bounding box center of the entire femur**. The script comments explicitly say this is intentional — the rationale being that the bone bbox was centred at (0,0) by `prepare_bone()`.

However, there are two problems:

**Problem 2A — The distal femoral anatomy (where the implant sits) is not centred at the same Y as the whole-bone bbox centre.**
The femoral condyles are **posterior** to the shaft centroid. The bone's bbox `center_y` is an average of the entire shaft + condyles, so it sits anterior to where the implant actually needs to go. This is the source of the -Y shift you're seeing: the implant is placed at the whole-bone Y=0, but it should be placed at the **condylar region Y centroid**.

**Problem 2B — After valgus rotation, the bone centre shifts further.**
The valgus rotation (Step 5) is applied around the X axis with the condyle contact at Z≈0. Because the bone extends far in +Z (toward the femoral head), the rotation causes the proximal end to sweep in Y, which shifts the overall bbox centre in Y. The cutting block, placed before this would matter, is at Y=0, but the bone's effective centre at the distal end has moved.

### Action plan

1. **Measure the distal condylar region bounding box separately and use its XY centre for component placement.** After the bone is prepared and the condyle contact Z is found, isolate the vertices in the distal ~25% of the bone (between `z_min` and `z_min + 0.25 * length`) and compute their bounding box centre in XY. Place the cutting block and implant at *that* XY, not at (0,0).

2. **Alternatively (simpler):** Use `condyle_xy_anat` (which the script already computes) as the XY anchor. The script currently computes this value and prints it but then ignores it for component placement. The `condyle_xy` is the midpoint of the two most-distal condyle vertices, which is a reasonable anatomical centre for the implant. The Y component of this will be posterior to (0,0), which is exactly the correction you need.

3. **Recommended approach:** Compute the distal condylar bounding box centre (option 1) for robustness. The single most-distal vertex per side (option 2) is sensitive to mesh noise — one outlier vertex can skew it. A regional bbox is more stable.

### Code sketch

```python
def get_distal_region_bbox(obj, fraction=0.25):
    """Bounding box of only the distal portion of the bone."""
    verts = get_world_vertices(obj)
    zs = [v.z for v in verts]
    z_min, z_max = min(zs), max(zs)
    z_threshold = z_min + fraction * (z_max - z_min)

    distal_verts = [v for v in verts if v.z <= z_threshold]
    xs = [v.x for v in distal_verts]
    ys = [v.y for v in distal_verts]
    return {
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "ML": max(xs) - min(xs),
        "AP": max(ys) - min(ys),
    }
```

Then in `run_femur()`:
```python
distal_bbox = get_distal_region_bbox(femur, fraction=0.25)
# Use distal_bbox["center_x"], distal_bbox["center_y"] for all component XY
# Use distal_bbox["ML"] and distal_bbox["AP"] for size selection
```

This simultaneously fixes Issue 1 (measurement) and Issue 2 (centering) — the distal region bbox gives you the articular surface dimensions *and* the correct centering point.

---

## Issue 3: Cutting Block ↔ Implant Position Mismatch

### What's happening

The cutting block and implant are imported separately. Each one gets `geometry_to_origin()`, which sets the object origin to the mesh's geometric centroid. Since the cutting block and implant have **completely different shapes**, their centroids are in different positions relative to their shared anatomical reference (the distal cut surface, the mounting plane, etc.).

When you then place both at `XY = (0, 0)` and at some computed Z, each one's centroid is at the right position — but the *anatomical features* (cut surface, mounting surface) are at different offsets from the centroid. This means they don't line up correctly in practice.

**The shell cutting block has the same problem in reverse:** it's placed at the same location as the primary cutting block, but its centroid is different from the primary block's centroid, so the shell geometry doesn't align with the primary block geometry.

### Action plan

**The fundamental fix: use a consistent reference point instead of the geometric centroid.**

Two approaches:

**Option A — Bounding box bottom-centre as origin (recommended):**
Instead of `geometry_to_origin()` (which uses the centroid), set the origin to the **bottom-centre of the bounding box** for all components that share a seating surface. This way, the "bottom" of every component is at Z=0 in local space, and you position them by placing that bottom at the desired world Z. Since the cutting block and implant share the same distal cut plane as their "bottom," they'll align.

```python
def origin_to_bottom_center(obj):
    """Set origin to bottom-center of bounding box."""
    set_active(obj)
    bbox = get_bounding_box(obj)
    # Shift mesh so bottom-center is at origin
    offset_x = -bbox["center_x"]
    offset_y = -bbox["center_y"]
    offset_z = -bbox["z_min"]  # bottom face at Z=0
    for v in obj.data.vertices:
        v.co.x += offset_x
        v.co.y += offset_y
        v.co.z += offset_z
    obj.data.update()
```

**Option B — Compute pairwise offsets from the STLs (more work but exact):**
For each size, measure the offset between the cutting block centroid and the implant centroid in all three axes. Store these as additional size table values. Apply the offset when placing the implant. This is more manual work per size but doesn't change the origin convention.

**Recommendation: Option A.** It's simpler, doesn't require per-size measurements, and makes the positioning logic much more intuitive: "place the bottom of this thing at this Z" instead of "place the centroid at this Z plus this offset from the centroid to the surface."

**For the shell cutting block specifically:** Once both the primary cutting block and the shell cutting block use the same origin convention (bottom-centre), placing the shell at the same location as the primary will automatically align them. Currently the script does `femur_cb_shell.location = femur_cb.location`, which only works if both share the same origin convention. With `origin_to_bottom_center()`, this becomes correct by construction.

---

## Issue 4: Size Table Values — Conceptual Review

### Current values in the CSV

| Column | Stored in | Meaning (per README) | Unit |
|--------|-----------|----------------------|------|
| `femur_cb_z_offset` | CSV + script | Femoral CB centroid → distal cut surface distance | metres, negative |
| `femur_distal_cut` | CSV + script | Distal resection depth (how far CB moves into bone) | metres, positive |
| `tibia_cb_z_offset` | CSV + script | Tibial CB centroid → proximal cut surface distance | metres, positive |
| `tibia_proximal_cut` | CSV + script | Proximal resection depth | metres, positive |
| `femur_implant_z_off` | script only | Femoral implant centroid → mounting surface | metres, 0.0 (TODO) |
| `tibia_plate_z_off` | script only | Tibial plate centroid → top surface | metres, 0.0 (TODO) |
| `tibia_insert_z_off` | script only | Tibial insert centroid → bottom surface | metres, 0.0 (TODO) |

### Conceptual assessment

The **cutting block offsets** (`femur_cb_z_offset`, `tibia_cb_z_offset`) represent "how far is the cut surface from the geometric centroid of this STL." This is inherently fragile because:

- The geometric centroid depends on the mesh topology (denser regions pull it toward them).
- If you re-mesh or modify the cutting block STL, the centroid moves and the offset becomes wrong.
- Every new size requires manual measurement of this offset.

**If you adopt the `origin_to_bottom_center()` approach from Issue 3, these offsets become unnecessary for Z positioning.** The cut surface IS the bottom of the cutting block (or very close to it). You'd position the cutting block by placing its bottom at the condyle contact plane, then shifting by the resection depth. No centroid-to-surface offset needed.

### What the offset values actually mean physically

Let me walk through the femoral cutting block positioning math to make sure the values are conceptually right:

```
Goal: place the CB so its cut surface is at Z = 0 (condyle plane),
      then push it +Z by the resection depth.

Current logic:
  cb_centroid_Z = -femur_cb_z_offset + femur_distal_cut
                = -(-0.009) + 0.0081     [M2 example]
                = 0.009 + 0.0081
                = 0.0171 m = 17.1 mm above Z=0

So the CB centroid sits 17.1 mm above the condyle plane.
The cut surface is at: 0.0171 + (-0.009) = 0.0081 m = 8.1 mm above Z=0.
That 8.1 mm IS the resection depth. ✓ Conceptually correct.
```

And for the tibia:
```
Current logic:
  cb_centroid_Z = resection_z - tibia_cb_z_offset
  resection_z   = plate_top_z - tibia_proximal_cut

  cut_surface_Z = cb_centroid_Z + tibia_cb_z_offset
                = (resection_z - cb_z_offset) + cb_z_offset
                = resection_z  ✓ Conceptually correct.
```

**The math is correct. The values are conceptually right.** The issue is that they're measured relative to the geometric centroid, which is fragile and requires per-size manual measurement.

### The three TODO offsets

`femur_implant_z_off`, `tibia_plate_z_off`, `tibia_insert_z_off` are all `0.0`. This means:

- The femoral implant is placed with its centroid at Z=0 (the condyle plane). But the implant's mounting surface is *above* the centroid, so the implant actually sits too low — its mounting surface is above Z=0 when it should be AT Z=0.
- The tibial plate is placed with its centroid at `plate_top_z`. But the plate's top surface is above the centroid, so the plate top is actually above `plate_top_z`.
- The tibial insert is placed with its centroid at `plate_top_actual`. But its bottom surface is below the centroid, so it doesn't sit flush on the plate.

**These are the 10-30 mm errors you're seeing in component positioning.** The offsets are zero, so every component is misplaced by the distance from its centroid to its relevant surface.

### Action plan

**If you adopt `origin_to_bottom_center()`:** All these offsets become either zero or trivially derivable from the bounding box height. No manual measurement needed. The bottom of each component is at Z=0 in local space, so placing it at the desired Z "just works."

**If you keep the centroid-based approach:** You must measure each offset per size (36 measurements: 3 offsets × 12 sizes). This is the approach described in the README's "Outstanding TODOs" section. It works but is tedious and error-prone.

**Recommendation:** Switch to `origin_to_bottom_center()` and eliminate the centroid-to-surface offsets entirely. Add these columns to the CSV instead:

| New column | Meaning | Used for |
|-----------|---------|----------|
| `femur_cb_height` | Total Z extent of the femoral cutting block | Not strictly needed if using bbox |
| `femur_implant_height` | Total Z extent of the femoral implant | Computing implant_bottom_z |
| `tibia_plate_height` | Total Z extent of the tibial plate | Computing plate positioning |
| `tibia_insert_height` | Total Z extent of the tibial insert | Computing insert positioning |

Actually — **even these aren't needed if you compute the bounding box at runtime after import.** The script already calls `get_bounding_box()` in several places. You can derive heights directly:

```python
implant_height = bbox["z_max"] - bbox["z_min"]  # after origin_to_bottom_center
implant_bottom_z = implant.location.z  # bottom IS at location.z
implant_top_z = implant.location.z + implant_height
```

---

## Issue 5: Hardcoded Values → CSV-Driven (Issue 3 from your list)

### What's happening

The size table is duplicated: once in the CSV file (which is the source of truth) and once hardcoded in the Python script as `SIZE_TABLE`. The script doesn't read the CSV — it has the values copy-pasted in. When you update the CSV, you have to manually update 12 dictionary entries in the script.

Additionally, the script has hardcoded offsets (`femur_cb_z_offset`, `femur_distal_cut`, etc.) that should come from the CSV, plus offsets that aren't in the CSV at all (`femur_implant_z_off`, etc.) that are hardcoded as 0.0.

### Action plan

1. **Read the CSV at runtime.** Replace the hardcoded `SIZE_TABLE` with a function that parses `SizeChart.csv` on startup.

2. **Expand the CSV** to include any values the script needs that aren't there yet. Based on the analysis above, if you switch to `origin_to_bottom_center()`, the CSV only needs:

   - SIZE, Femur ML, Femur AP, Tibia ML, Tibia AP, Patella Ø, Insert thin, Insert thick
   - femur_distal_cut, tibia_proximal_cut (resection depths — these are surgical parameters, not geometric)
   - STL filenames (or generate them from the size label, as `_stl_names()` already does)

   The centroid-to-surface offsets (`femur_cb_z_offset`, `tibia_cb_z_offset`) become unnecessary with the new origin convention.

3. **CSV parsing function:**

```python
import csv

def load_size_table(csv_path):
    """Load size chart from CSV. Returns list of dicts."""
    table = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "size": row["SIZE"],
                "femur_ML": float(row["Femur ML (mm)"]),
                "femur_AP": float(row["Femur AP (mm)"]),
                "tibia_ML": float(row["Tibia ML (mm)"]),
                "tibia_AP": float(row["Tibia AP (mm)"]),
                "femur_distal_cut": float(row["femur_distal_cut (meters)"]),
                "tibia_proximal_cut": float(row["tibia_proximal_cut (meters)"]),
                # Add more columns as the CSV grows
            }
            entry.update(_stl_names(entry["size"]))
            table.append(entry)
    return table
```

---

## Additional Issues Identified

### Issue 6: `apply_transforms()` called excessively

The script calls `apply_transforms()` after almost every operation. While this isn't *wrong*, it makes reasoning about the state harder because after each apply, `location`, `rotation`, and `scale` reset to identity. This means you can't inspect these values in Blender's UI to understand what happened. It also means that if you need to undo a positioning step, you can't — the transform is baked into the mesh.

**Recommendation:** Apply transforms only at two points: (a) immediately after import (to bake the 0.001 scale), and (b) at the very end before export. In between, use `obj.location`, `obj.rotation_euler`, and `obj.matrix_world` to position things, and use `get_world_vertices()` (which already accounts for `matrix_world`) for measurements. This makes the scene inspectable in Blender and reduces the number of irreversible mesh modifications.

### Issue 7: `find_condyle_contact_points()` uses a single vertex

The most-distal vertex from each condyle half is a single mesh vertex. On a moderately tessellated STL, a single vertex can be a noise outlier — a spike in the mesh, a non-manifold artifact, or just a poorly placed triangle. The contact Z is then wrong, and so is the condyle centroid XY.

**Recommendation:** Instead of taking the single most-distal vertex, take the **average of the N most-distal vertices** (e.g., the bottom 1% or bottom 5 vertices by Z) from each half. This gives a much more robust estimate of the condyle surface position.

```python
def find_condyle_contact_points(obj, bbox, n_avg=10):
    verts = get_world_vertices(obj)
    midline = bbox["center_x"]

    medial = sorted([v for v in verts if v.x <= midline], key=lambda v: v.z)[:n_avg]
    lateral = sorted([v for v in verts if v.x > midline], key=lambda v: v.z)[:n_avg]

    medial_z = sum(v.z for v in medial) / len(medial)
    lateral_z = sum(v.z for v in lateral) / len(lateral)
    # ... etc
```

Same applies to `find_plateau_contact_points()` for the tibia.

### Issue 8: No validation of input bone orientation

The script assumes the bone arrives with proximal = +Z, distal = -Z. If the bone is upside down (common with STL exports from different software), everything breaks silently — the "condyles" found will actually be the femoral head, and the size will be completely wrong.

**Recommendation:** Add a simple orientation check after `prepare_bone()`:
- For the femur: the distal end (condyles) should be wider in ML than the mid-shaft. If the bottom of the bone is narrower than the middle, it's likely upside down.
- For the tibia: the proximal end (plateau) should be wider than the distal (ankle). If the top is narrower than the bottom, it's flipped.

Print a clear error and abort if the orientation looks wrong.

### Issue 9: Export doesn't include the cutting blocks

The script exports the boolean-modified bones, the shell, the implant, the plate, and the insert — but not the cutting blocks themselves (they're deleted by `boolean_op(delete_cutter=True)`). For clinical review or 3D printing the surgical guide, the cutting block geometry is often needed as a separate export.

**Recommendation:** Export cutting blocks before performing the boolean, or set `delete_cutter=False` and export them afterward.

---

## Consolidated Action Plan (Priority Order)

| # | Task | Fixes Issue | Effort | Impact |
|---|------|-------------|--------|--------|
| 1 | Switch to `origin_to_bottom_center()` for all components | 3, 4 | Medium | High — eliminates all centroid-to-surface offsets, makes positioning intuitive |
| 2 | Implement `get_distal_region_bbox()` / `get_proximal_region_bbox()` | 1A, 2 | Medium | High — fixes both sizing and centering simultaneously |
| 3 | Change `select_size()` to round-up logic | 1A | Low | High — fixes "too small" directly |
| 4 | Unify size: femur drives, tibia warns | 1B | Low | Medium — ensures matching component sizes |
| 5 | Read size table from CSV at runtime | 5 | Low | Medium — single source of truth, no more copy-paste |
| 6 | Average N vertices instead of single vertex for contact points | 7 | Low | Medium — more robust measurements |
| 7 | Add bone orientation validation | 8 | Low | Medium — prevents silent failures |
| 8 | Reduce `apply_transforms()` calls | 6 | Medium | Low — cleaner state, inspectable scene |
| 9 | Export cutting blocks before boolean | 9 | Low | Low — completeness |

Tasks 1-5 are the core fixes that address your four stated issues. Tasks 6-9 are robustness improvements.

---

## Revised Positioning Logic (After All Fixes)

Here's how the positioning math simplifies with the proposed changes:

### Femoral components

```
1. Import femur, prepare, measure distal region bbox
2. Size selection (round-up, femur-driven)
3. Translate femur so condyle contact → Z=0
4. Apply valgus rotation
5. Compute distal_center_xy from distal region bbox

6. Import femoral cutting block
   → origin_to_bottom_center()
   → location.x = distal_center_xy[0]
   → location.y = distal_center_xy[1]
   → location.z = femur_distal_cut          # bottom (cut surface) at resection depth

7. Import femoral implant
   → origin_to_bottom_center()
   → location.x = distal_center_xy[0]
   → location.y = distal_center_xy[1]
   → location.z = 0.0                        # bottom (mounting surface) at condyle plane
   → implant_bottom_z = location.z            # = 0 (or compute from bbox if needed)
   Wait — actually the implant wraps AROUND the distal femur. Its "bottom" in
   the bounding box sense is the most distal point (the articular surface facing
   the tibia). So with origin_to_bottom_center(), location.z = 0 would place the
   articular surface at Z=0. Depending on your STL geometry, you may want
   origin_to_TOP_center() for the implant, so the mounting surface (top) is at Z=0
   and the articular surface hangs below.

   → Better: use origin_to_bottom_center() universally, and set
     location.z = -(implant_bbox_height) so the TOP is at Z=0.
   → Or: define origin_to_top_center() for implants.
```

The exact convention for the implant depends on your STL orientation. The key principle is: **pick one surface as the reference, put the origin there, and the positioning becomes a single assignment.**

### Tibial components

```
1. plate_top_z = implant_bottom_z - FEMUR_TIBIA_GAP

2. Import tibial plate
   → origin_to_top_center()    # top surface is the reference
   → location.z = plate_top_z  # top surface at plate_top_z

3. Import tibial insert
   → origin_to_bottom_center()
   → location.z = plate_top_z  # bottom sits on top of plate

4. Import tibial cutting block
   → origin_to_top_center()    # cut surface (top) is the reference
   → location.z = plate_top_z - tibia_proximal_cut  # cut surface at resection Z

5. Translate tibia bone so plateau → resection_z
```

This is dramatically simpler than the current centroid + offset approach.

---

## Summary

The script's architecture is good. The core issues are:

1. **Sizing rounds down** → switch to round-up logic
2. **Measures the whole bone** → measure the articular region only
3. **Centres on whole-bone bbox** → centre on distal/proximal region bbox
4. **Centroid-based origins** → switch to surface-based origins (bottom-centre or top-centre)
5. **Hardcoded size table** → read from CSV
6. **Independent femur/tibia sizing** → femur-driven with tibia cross-check

Fixing issues 2+3 simultaneously with `get_distal_region_bbox()` and fixing issues 3+4 simultaneously with `origin_to_bottom_center()` means the actual number of changes is smaller than it looks. The result will be a script where positioning is a one-liner per component with no manual offset measurements needed.
