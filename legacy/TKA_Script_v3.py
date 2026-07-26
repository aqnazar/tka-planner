"""
TKA Automatic Pipeline v3
===========================
Automatically measures femur and tibia bone STLs, selects implant size,
and positions all components based on bounding-box geometry.

KEY CHANGES FROM v2
-------------------
- PCA-based shaft alignment (replaces hardcoded import_z_rotation)
- TEA (transepicondylar axis) detection for femoral ML alignment
- Native STL origins for all components (no more bbox-based origin offsets)
- Two bone import modes: "retained" (CT positions kept) or "independent"
- Batch processing across patient folders
- Step-by-step demo mode with viewport pauses
- All config values in millimetres
- Configurable bone filenames and Scaled_STLs path

HOW TO RUN
----------
Option A — Blender GUI:
    Scripting tab → Open → Run Script  (or Layout tab → Text Editor area)

Option B — Headless:
    blender --background --python tka_auto_pipeline_v3.py

DEMO / PRESENTATION TIPS
-------------------------
To run with a visible viewport:
  1. Open Blender in the Layout tab
  2. Split one area into a Text Editor (Shift+click on area corner → drag)
  3. Open the script in the Text Editor and click ▶ Run Script
  4. Set STEP_BY_STEP_MODE = True for pauses between steps
  5. For dual monitors: Window → New Main Window, put viewport on one and
     text editor on the other

To maximise the 3D viewport at any time: hover over it and press Ctrl+Space.

COORDINATE SYSTEM
-----------------
All units are METRES internally.  STLs are modelled in mm (0.001 scale).

Z = 0 is the femoral condyle contact plane ("joint line").

    +Z  (proximal / superior)
     │  Femur bone       (condyles at Z=0, tilted valgus)
     │  Femoral components (CB, implant — native origins, all at same Z)
     │       ── FEMUR_TIBIA_GAP ──
     │  Tibial components  (plate, insert, CBs — native origins, all at same Z)
     │  Tibia bone         (positioned to match CB resection depth)
    -Z  (distal / inferior)

DIRECTORY STRUCTURE (single patient)
------------------------------------
PATIENT_DIR/
├── Left/                        ← or Right/
│   ├── {FEMUR_STL_FILENAME}
│   └── {TIBIA_STL_FILENAME}
│   └── Output/                  ← created by script
│       ├── tka_pipeline_log_....txt
│       └── *.stl

SCALED_STLS_DIR/ (shared, one copy)
├── SizeChart.csv
├── S1/ ... L4/
"""

import bpy
import os
import sys
import csv
import math
import time
import datetime
import numpy as np
import mathutils

# =============================================================================
# !! SECTION 1: USER CONFIGURATION !!
# All lengths in this section are in MILLIMETRES unless noted otherwise.
# =============================================================================

# --- Paths -------------------------------------------------------------------
# Single-patient mode: set PATIENT_DIR to the folder containing Left/ Right/.
# Batch mode: set BATCH_MODE = True and BATCH_DIR instead.
PATIENT_DIR    = r"C:\Users\aknaz\AQ\SATBAYEV\Surgical Robot\Custom Implant\Scripts\BP\Patient Processing\Patient_001\Left"
SCALED_STLS_DIR = r"C:\Users\aknaz\AQ\SATBAYEV\Surgical Robot\Custom Implant\CAD\Final Exports\Final STLs\Scaled_STLs"
CSV_PATH        = os.path.join(SCALED_STLS_DIR, "SizeChart.csv")

SIDE = "L"   # "L" (left), "R" (right), "B" (both — processes Left/ and Right/ folders)

# --- Bone filenames ----------------------------------------------------------
# {side} is replaced with "Left" or "Right" at runtime.
FEMUR_STL_FILENAME = "FD1{side}.stl"
TIBIA_STL_FILENAME = "TD1{side}.stl"

# --- Bone import mode --------------------------------------------------------
# R = retained:    bones from the same CT, retain relative positions.
#                  Both rotated together using femur's TEA.
# I = independent: bones exported separately, no shared coordinate frame.
#                  Each aligned independently (PCA+TEA for femur,
#                  PCA + manual rotation for tibia).
BONE_IMPORT_MODE = "R"   # "R" (retained) or "I" (independent)

# --- Surgical parameters (mm) -----------------------------------------------
FEMUR_TIBIA_GAP_MM  = 20.0    # visual gap between femoral implant and tibial plate
BONE_CUT_LENGTH_MM  = 150.0   # bone length to keep from articular end (None = no cut)
VALGUS_ANGLE_DEG    = 6.0     # coronal plane correction (rotation around AP/X axis)
SIZE_MODE           = "ML"    # primary dimension for size selection ("ML" or "AP")
SIZE_ROUND          = "D"     # "U" = round up (overhang ok), "D" = round down

# --- Manual size override ----------------------------------------------------
# Set to a size label (e.g. "M2") to skip automatic size selection.
# Set to "" (empty) to use automatic sizing.
MANUAL_SIZE = ""              # e.g. "M2", "L1", or "" for auto

# --- Size comparison mode ----------------------------------------------------
# When "T", runs the pipeline twice: once at the selected size and once at
# the next size down.  Both results are placed side by side (the comparison
# set is shifted in +Y by 200mm) for visual inspection.
COMPARE_SIZE = "F"            # "T" or "F"

# --- Axis convention after PCA+TEA alignment ---------------------------------
#   X = AP (anteroposterior)
#   Y = ML (mediolateral) — TEA aligned to Y
#   Z = SI (superior-inferior, shaft)
#   Valgus rotation → around X (AP), tilts in coronal plane
#   Posterior slope  → around Y (ML), tilts in sagittal plane

# --- Alignment toggles -------------------------------------------------------
USE_PCA = "F"   # "T" = PCA shaft alignment, "F" = skip (bone already upright)
USE_TEA = "F"   # "T" = TEA detection + alignment, "F" = skip (use manual Z rot)
# When USE_PCA or USE_TEA is "F", the bone must already be oriented correctly
# or you must provide manual rotations below.

# Manual bone Z rotation (degrees). Applied when USE_PCA = "F".
# Set to 90 if your bone STLs need the same rotation as v2's import_z_rotation.
BONE_MANUAL_Z_ROT_DEG = 90.0

# --- TEA alignment -----------------------------------------------------------
# The transepicondylar axis (TEA) is detected on the femur and aligned with
# the world Y axis.  A correction angle accounts for the ~3° external rotation
# of the surgical TEA relative to the anatomical TEA.
# NOTE: The script automatically handles L vs R knees — the correction sign
# is flipped internally for right knees.
TEA_CORRECTION_DEG = 3.0      # degrees (always positive — sign handled by code)

# In "I" (independent) mode, the tibia uses a manual Z rotation instead of TEA:
TIBIA_MANUAL_Z_ROT_DEG = 0.0  # degrees (only used in "I" mode)

# --- Component Z rotations (degrees) ----------------------------------------
# Applied to component STLs at import.  These correct for the difference between
# the CAD coordinate frame and the bone coordinate frame after PCA+TEA alignment.
FEMUR_COMPONENT_Z_ROT = 0    # femoral CB, shell, implant
TIBIA_COMPONENT_Z_ROT = 90      # tibial CB, shell, plate, insert

# --- Manual XY offsets (mm) --------------------------------------------------
# Fine-tuning per-patient shifts applied AFTER automatic centring.
# Set to 0 for a new bone, inspect, then dial in.
FEMUR_CB_OFFSET_X_MM       = 0.0
FEMUR_CB_OFFSET_Y_MM       = 0.0
FEMUR_IMPLANT_OFFSET_X_MM  = 0.0
FEMUR_IMPLANT_OFFSET_Y_MM  = 0.0

# Tibial group offset: shifts ALL tibial components AND the tibia bone in X.
# Individual tibial offsets below are applied ON TOP of this.
TIBIA_GROUP_OFFSET_X_MM    = 0.0

# Individual tibial component offsets (on top of group offset):
TIBIA_CB_OFFSET_X_MM       = 0.0
TIBIA_CB_OFFSET_Y_MM       = 0.0
TIBIA_PLATE_OFFSET_X_MM    = 0.0
TIBIA_PLATE_OFFSET_Y_MM    = 0.0
TIBIA_INSERT_OFFSET_X_MM   = 0.0
TIBIA_INSERT_OFFSET_Y_MM   = 0.0

# Tibia BONE-only X offset (on top of group offset, independent of components):
TIBIA_BONE_OFFSET_X_MM     = 0.0

# When FEMUR_IMPLANT_OFFSET_Y is non-zero, the entire tibial group (including
# the tibia bone) shifts by the same amount in Y.

# --- Posterior slope ---------------------------------------------------------
TIBIA_POSTERIOR_SLOPE_DEG = 5.0   # degrees (positive = posterior-down)

# --- Measurement parameters --------------------------------------------------
REGIONAL_BBOX_FRACTION = 0.25   # fraction of bone used for articular-region bbox
CONTACT_AVG_N          = 10     # vertices averaged for contact-point detection

# --- Step-by-step demo mode --------------------------------------------------
STEP_BY_STEP_MODE    = "F"     # "T" = pause at each step, "F" = run continuously
STEP_PAUSE_SECONDS   = 3.0

# =============================================================================
# SECTION 2: BATCH PROCESSING CONFIG
# =============================================================================

BATCH_MODE = "F"               # "T" = batch, "F" = single patient
BATCH_DIR  = r"C:\Users\aknaz\AQ\SATBAYEV\Surgical Robot\Custom Implant\Scripts\BP\Patient Processing"
# Expected structure:
#   BATCH_DIR/
#   ├── Patient_001/
#   │   ├── Left/
#   │   │   ├── FD1Left.stl
#   │   │   └── TD1Left.stl
#   │   └── Right/  (optional)
#   ├── Patient_002/ ...

# =============================================================================
# SECTION 3: INTERNAL CONVERSION (mm → metres)
# =============================================================================

FEMUR_TIBIA_GAP      = FEMUR_TIBIA_GAP_MM / 1000
BONE_CUT_LENGTH      = BONE_CUT_LENGTH_MM / 1000 if BONE_CUT_LENGTH_MM else None

FEMUR_CB_OFFSET_X      = FEMUR_CB_OFFSET_X_MM / 1000
FEMUR_CB_OFFSET_Y      = FEMUR_CB_OFFSET_Y_MM / 1000
FEMUR_IMPLANT_OFFSET_X = FEMUR_IMPLANT_OFFSET_X_MM / 1000
FEMUR_IMPLANT_OFFSET_Y = FEMUR_IMPLANT_OFFSET_Y_MM / 1000
TIBIA_GROUP_OFFSET_X   = TIBIA_GROUP_OFFSET_X_MM / 1000
TIBIA_CB_OFFSET_X      = TIBIA_CB_OFFSET_X_MM / 1000
TIBIA_CB_OFFSET_Y      = TIBIA_CB_OFFSET_Y_MM / 1000
TIBIA_PLATE_OFFSET_X   = TIBIA_PLATE_OFFSET_X_MM / 1000
TIBIA_PLATE_OFFSET_Y   = TIBIA_PLATE_OFFSET_Y_MM / 1000
TIBIA_INSERT_OFFSET_X  = TIBIA_INSERT_OFFSET_X_MM / 1000
TIBIA_INSERT_OFFSET_Y  = TIBIA_INSERT_OFFSET_Y_MM / 1000
TIBIA_BONE_OFFSET_X    = TIBIA_BONE_OFFSET_X_MM / 1000


# =============================================================================
# SECTION 4: SIZE TABLE — loaded from CSV
# =============================================================================

def _stl_names(size):
    return {
        "femur_cb_stl"         : f"CuttingBlockFemoral(L&R)_{size}.stl",
        "femur_cb_shell_stl"   : f"CuttingBlockFemoralShell(L&R)_{size}.stl",
        "tibia_cb_stl"         : f"CuttingBlockTibia(L&R)_{size}.stl",
        "tibia_cb_shell_stl"   : f"CuttingBlockTibiaShell(L&R)_{size}.stl",
        "implant_femoral_left" : f"Implant(Femoral)Left_{size}.stl",
        "implant_femoral_right": f"Implant(Femoral)Right_{size}.stl",
        "tibial_insert_stl"    : f"Tibial_Insert(L&R)_{size}.stl",
        "tibial_plate_stl"     : f"Tibial_Plate(L&R)_{size}.stl",
    }


def _size_folder(size):
    return os.path.join(SCALED_STLS_DIR, size)


def load_size_table(csv_path):
    """Parse SizeChart.csv into a list of dicts.

    Expected columns:
        SIZE, femur_ML_mm, femur_AP_mm, tibia_ML_mm, tibia_AP_mm,
        patella_d_mm, insert_thin_mm, insert_thick_mm,
        femur_distal_cut_m, femur_origin_z_m,
        tibia_proximal_cut_m, tibia_origin_z_m

    All numeric values are POSITIVE.  Offset columns (origin_z) are the Z
    distance from the component STL origin to the cut/mating surface.
    When 0, the STL origin IS at the cut surface (the ideal case).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Size chart CSV not found: {csv_path}")

    table = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "size"     : row["SIZE"].strip(),
                "femur_ML" : float(row["femur_ML_mm"]),
                "femur_AP" : float(row["femur_AP_mm"]),
                "tibia_ML" : float(row["tibia_ML_mm"]),
                "tibia_AP" : float(row["tibia_AP_mm"]),
                "patella_d": float(row["patella_d_mm"]),
                "insert_thin" : float(row["insert_thin_mm"]),
                "insert_thick": float(row["insert_thick_mm"]),
                "femur_distal_cut" : float(row["femur_distal_cut_m"]),
                "femur_origin_z"   : float(row["femur_origin_z_m"]),
                "tibia_proximal_cut": float(row["tibia_proximal_cut_m"]),
                "tibia_origin_z"   : float(row["tibia_origin_z_m"]),
            }
            entry.update(_stl_names(entry["size"]))
            table.append(entry)

    print(f"[OK] Loaded {len(table)} sizes from {csv_path}")
    return table


# =============================================================================
# SECTION 5: LOW-LEVEL HELPERS
# =============================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def import_stl(filepath, name, import_scale=0.001):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"STL not found: {filepath}")
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.ops.wm.stl_import(filepath=filepath)
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=filepath)
    obj       = bpy.context.selected_objects[0]
    obj.name  = name
    obj.scale = (import_scale, import_scale, import_scale)
    return obj


def set_active(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_transforms(obj):
    set_active(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def geometry_to_origin(obj):
    set_active(obj)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')


def duplicate_object(source, new_name):
    set_active(source)
    bpy.ops.object.duplicate(linked=False)
    dup      = bpy.context.selected_objects[0]
    dup.name = new_name
    return dup


def boolean_op(target, cutter, operation='DIFFERENCE', solver='EXACT',
               delete_cutter=True):
    cutter.hide_render   = True
    cutter.hide_viewport = False
    set_active(target)
    mod           = target.modifiers.new(name=f"Bool_{operation}", type='BOOLEAN')
    mod.operation = operation
    mod.object    = cutter
    mod.solver    = solver
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except RuntimeError:
        if solver == 'EXACT':
            print(f"[WARNING] EXACT boolean failed on '{target.name}', retrying FAST…")
            target.modifiers.remove(mod)
            mod           = target.modifiers.new(name=f"Bool_{operation}_FAST", type='BOOLEAN')
            mod.operation = operation
            mod.object    = cutter
            mod.solver    = 'FAST'
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except RuntimeError as e2:
                print(f"[ERROR] FAST boolean also failed: {e2}")
                target.modifiers.remove(mod)
        else:
            print(f"[ERROR] Boolean failed on '{target.name}'.")
    if delete_cutter:
        bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def export_stl(obj, out_dir, out_filename):
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, out_filename)
    set_active(obj)
    try:
        bpy.ops.wm.stl_export(filepath=filepath, export_selected_objects=True)
    except AttributeError:
        bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)
    print(f"  [EXPORT] {filepath}")


def hide_object(obj):
    """Hide from viewport (toggleable via outliner eye icon)."""
    if obj and obj.name in bpy.data.objects:
        obj.hide_set(True)


def step_pause(label):
    """Pause and refresh viewport if step-by-step mode is on."""
    if STEP_BY_STEP_MODE != "T":
        return
    print(f"\n  ▸ STEP: {label}  (pausing {STEP_PAUSE_SECONDS}s...)")
    bpy.context.view_layer.update()
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {'area': area, 'region': region}
                        bpy.ops.view3d.view_all(override)
                        break
    except Exception:
        pass  # headless mode — no viewport to refresh
    time.sleep(STEP_PAUSE_SECONDS)


# =============================================================================
# SECTION 6: COMPONENT IMPORT (native origin mode)
# =============================================================================

def import_component_native(filepath, name, z_rotation_deg=0):
    """Import a component STL keeping its native origin (CAD design origin).

    All components for a given bone side share the same CAD origin, so
    placing them at the same world position aligns them automatically.
    """
    obj = import_stl(filepath, name)
    apply_transforms(obj)   # bake 0.001 scale

    if z_rotation_deg != 0:
        obj.rotation_euler.z = math.radians(z_rotation_deg)
        apply_transforms(obj)

    # NO origin change — the STL's (0,0,0) stays as the object origin
    return obj


# =============================================================================
# SECTION 7: PCA SHAFT ALIGNMENT
# =============================================================================

def pca_align_shaft(obj):
    """Rotate the bone so its longest axis (shaft) aligns with +Z.

    Uses PCA on the vertex cloud.  The eigenvector with the largest
    eigenvalue is the shaft direction.  Returns the quaternion applied.
    """
    apply_transforms(obj)
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    pts   = np.array([(v.x, v.y, v.z) for v in verts])

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov      = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order — column 2 = largest eigenvalue = shaft
    shaft = mathutils.Vector(eigenvectors[:, 2].tolist())

    # Ensure shaft points in +Z (proximal)
    if shaft.z < 0:
        shaft = -shaft

    z_axis = mathutils.Vector((0, 0, 1))
    quat   = shaft.rotation_difference(z_axis)

    cv = mathutils.Vector(centroid.tolist())
    for v in obj.data.vertices:
        v.co = quat @ (v.co - cv) + cv
    obj.data.update()

    shaft_error_deg = math.degrees(shaft.angle(z_axis))
    print(f"  PCA shaft correction: {shaft_error_deg:.1f}° from Z")

    return quat


# =============================================================================
# SECTION 8: TEA DETECTION
# =============================================================================

def detect_tea_angle(obj):
    """Detect the transepicondylar axis (TEA) and return its angle from the Y axis.

    Uses 2D PCA on the epicondylar band vertices (projected to XY) to find
    the direction of maximum spread.

    IMPORTANT: PCA eigenvectors have 180° ambiguity — the same line can be
    reported as θ or θ±180°.  We normalize to [-90°, +90°] since the TEA
    is a line, not a directed vector.  This ensures the rotation to align
    TEA with Y is always the shortest path.

    Returns the angle (radians) that TEA makes with the +Y axis,
    normalized to [-π/2, +π/2].
    """
    apply_transforms(obj)
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    zs = [v.z for v in verts]
    z_min, z_max = min(zs), max(zs)
    z_range = z_max - z_min

    # Epicondylar band: 15-35% above the condyle tips (distal end)
    z_low  = z_min + 0.15 * z_range
    z_high = z_min + 0.35 * z_range
    band = [v for v in verts if z_low <= v.z <= z_high]
    if len(band) < 10:
        print("  [WARNING] Few vertices in epicondylar band, widening to distal 40%")
        band = [v for v in verts if v.z <= z_min + 0.40 * z_range]
    if len(band) < 2:
        print("  [WARNING] Cannot detect TEA — insufficient vertices")
        return 0.0

    # 2D PCA on XY coordinates of the band
    pts = np.array([(v.x, v.y) for v in band])
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Largest eigenvalue's eigenvector = direction of maximum spread = TEA
    tea_dir = eigenvectors[:, 1]  # column 1 = largest (ascending order)

    # Angle from +Y axis (raw, may be off by 180°)
    raw_angle = math.atan2(tea_dir[0], tea_dir[1])  # atan2(x_component, y_component)

    # ── Normalize to [-90°, +90°] to remove 180° ambiguity ──
    # TEA is a LINE, not a vector — both directions are equivalent.
    # We always want the smallest rotation to align with Y.
    angle = raw_angle
    if angle > math.pi / 2:
        angle -= math.pi
    elif angle < -math.pi / 2:
        angle += math.pi

    # Find the actual extreme points along TEA for reporting
    tea_unit = tea_dir / np.linalg.norm(tea_dir)
    projections = centered @ tea_unit
    i_max = np.argmax(projections)
    i_min = np.argmin(projections)
    pt_max = band[i_max]
    pt_min = band[i_min]

    print(f"  TEA detected: {math.degrees(angle):.1f}° from Y axis "
          f"(raw: {math.degrees(raw_angle):.1f}°)")
    print(f"    Endpoint A: ({pt_max.x*1000:.1f}, {pt_max.y*1000:.1f}, "
          f"{pt_max.z*1000:.1f}) mm")
    print(f"    Endpoint B: ({pt_min.x*1000:.1f}, {pt_min.y*1000:.1f}, "
          f"{pt_min.z*1000:.1f}) mm")
    print(f"    Spread:  {(projections.max() - projections.min())*1000:.1f}mm")

    return angle


def apply_z_rotation_around_centroid(obj, angle_rad):
    """Rotate all vertices around Z through the bone's centroid."""
    if abs(angle_rad) < 1e-6:
        return
    apply_transforms(obj)
    verts = [v.co.copy() for v in obj.data.vertices]
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    cv = mathutils.Vector((
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    ))

    rot = mathutils.Matrix.Rotation(angle_rad, 4, 'Z')
    for v in obj.data.vertices:
        v.co = rot @ (v.co - cv) + cv
    obj.data.update()


# =============================================================================
# SECTION 9: BOUNDING BOX AND MEASUREMENT
# =============================================================================

def get_world_vertices(obj):
    mat = obj.matrix_world
    return [mat @ v.co for v in obj.data.vertices]


def get_bounding_box(obj):
    """World-space AABB.  Convention: X = AP, Y = ML, Z = SI."""
    verts = get_world_vertices(obj)
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    return {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "z_min": min(zs), "z_max": max(zs),
        "AP": max(xs) - min(xs),   # X = anteroposterior
        "ML": max(ys) - min(ys),   # Y = mediolateral
        "height": max(zs) - min(zs),
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
        "center_z": (min(zs) + max(zs)) / 2,
    }


def get_regional_bbox(obj, region, fraction=0.25):
    """Regional bbox.  Convention: X = AP, Y = ML."""
    verts = get_world_vertices(obj)
    zs = [v.z for v in verts]
    z_min, z_max = min(zs), max(zs)
    z_range = z_max - z_min
    if region == "distal":
        subset = [v for v in verts if v.z <= z_min + fraction * z_range]
    else:
        subset = [v for v in verts if v.z >= z_max - fraction * z_range]
    if len(subset) < 3:
        raise ValueError(f"Regional bbox ({region}, {fraction}) yielded <3 vertices.")
    xs = [v.x for v in subset]
    ys = [v.y for v in subset]
    return {
        "AP": max(xs) - min(xs),   # X = anteroposterior
        "ML": max(ys) - min(ys),   # Y = mediolateral
        "center_x": (min(xs) + max(xs)) / 2,
        "center_y": (min(ys) + max(ys)) / 2,
    }


def find_condyle_contact(obj, bbox, n=10):
    """Split femur at ML midline (Y = center_y) and find distal contact.
    Convention: Y = ML axis."""
    verts   = get_world_vertices(obj)
    midline = bbox["center_y"]   # ML midline on Y axis
    medial  = sorted([v for v in verts if v.y <= midline], key=lambda v: v.z)[:n]
    lateral = sorted([v for v in verts if v.y >  midline], key=lambda v: v.z)[:n]
    if not medial or not lateral:
        raise ValueError("Cannot split femur at ML midline for condyle detection.")
    med_z = sum(v.z for v in medial)  / len(medial)
    lat_z = sum(v.z for v in lateral) / len(lateral)
    contact_z = min(med_z, lat_z)
    all_pts = medial + lateral
    cx = sum(v.x for v in all_pts) / len(all_pts)
    cy = sum(v.y for v in all_pts) / len(all_pts)
    return med_z, lat_z, contact_z, (cx, cy)


# =============================================================================
# SECTION 10: BONE CUTTING & VALIDATION
# =============================================================================

def cut_bone_to_length(obj, keep_end, length_m):
    apply_transforms(obj)
    bbox = get_bounding_box(obj)
    bone_length = bbox["height"]
    if bone_length <= length_m:
        print(f"  Bone ≤ {length_m*1000:.0f}mm ({bone_length*1000:.1f}mm). No cut needed.")
        return obj
    if keep_end == "distal":
        cut_z = bbox["z_min"] + length_m
        cutter_z = cut_z + 0.5
    else:
        cut_z = bbox["z_max"] - length_m
        cutter_z = cut_z - 0.5
    bpy.ops.mesh.primitive_cube_add(size=2, location=(
        bbox["center_x"], bbox["center_y"], cutter_z))
    cutter = bpy.context.active_object
    cutter.name = "_BoneCutter"
    cutter.scale = ((bbox["ML"] + 0.2) / 2, (bbox["AP"] + 0.2) / 2, 0.5)
    apply_transforms(cutter)
    print(f"  Cutting bone: keeping {keep_end} {length_m*1000:.0f}mm")
    obj = boolean_op(obj, cutter, operation='DIFFERENCE')
    return obj


def validate_bone_orientation(obj, bone_type):
    d = get_regional_bbox(obj, "distal",   0.25)
    p = get_regional_bbox(obj, "proximal", 0.25)
    if bone_type == "femur":
        if d["ML"] < p["ML"] * 0.8:
            print("[WARNING] Femur may be UPSIDE-DOWN.")
    else:
        if p["ML"] < d["ML"] * 0.8:
            print("[WARNING] Tibia may be UPSIDE-DOWN.")


# =============================================================================
# SECTION 11: POSTERIOR SLOPE
# =============================================================================

def apply_posterior_slope(obj, angle_deg, center):
    """Rotate mesh vertices around the Y axis (ML) at an arbitrary centre point.
    Convention: X = AP, Y = ML.  Posterior slope tilts in the sagittal plane
    → rotation around Y (ML)."""
    if abs(angle_deg) < 1e-6:
        return
    angle_rad = math.radians(-angle_deg)   # negate: positive config = posterior-down
    rot = mathutils.Matrix.Rotation(angle_rad, 4, 'Y')
    cv  = mathutils.Vector(center)
    for v in obj.data.vertices:
        v.co = rot @ (v.co - cv) + cv
    obj.data.update()


# =============================================================================
# SECTION 12: BONE PREPARATION
# =============================================================================

def prepare_bone(stl_filepath, name, cut_end=None, cut_length=None):
    """Import bone → bake scale → PCA shaft alignment → XY-centre → cut.

    After this call: shaft along Z, XY-centred, trimmed to working length.
    Z rotation (TEA alignment) is handled EXTERNALLY so both bones can
    share the same rotation when in "retained" mode.
    """
    bone = import_stl(stl_filepath, name=name)
    apply_transforms(bone)

    # PCA: align shaft to +Z (skip if disabled)
    if USE_PCA == "T":
        print(f"  Aligning {name} shaft to Z via PCA...")
        pca_align_shaft(bone)
        apply_transforms(bone)
    else:
        print(f"  PCA disabled — applying manual Z rotation: {BONE_MANUAL_Z_ROT_DEG}°")
        if abs(BONE_MANUAL_Z_ROT_DEG) > 1e-6:
            bone.rotation_euler.z = math.radians(BONE_MANUAL_Z_ROT_DEG)
            apply_transforms(bone)

    # XY-centre
    bbox = get_bounding_box(bone)
    bone.location.x = -bbox["center_x"]
    bone.location.y = -bbox["center_y"]
    apply_transforms(bone)

    # Cut to working length
    if cut_end and cut_length and cut_length > 0:
        bone = cut_bone_to_length(bone, cut_end, cut_length)
        apply_transforms(bone)
        bbox = get_bounding_box(bone)
        bone.location.x = -bbox["center_x"]
        bone.location.y = -bbox["center_y"]
        apply_transforms(bone)

    return bone


# =============================================================================
# SECTION 13: SIZE SELECTION (round-up)
# =============================================================================

def select_size(measurement_m, bone, mode, size_table):
    """Size selection.  SIZE_ROUND = 'U' rounds up, 'D' rounds down."""
    key = f"{bone}_{mode}"
    measure_mm = measurement_m * 1000
    sorted_tab = sorted(size_table, key=lambda s: s[key])

    if SIZE_ROUND == "U":
        # Round up: smallest size that covers the bone (>= measurement)
        for s in sorted_tab:
            if s[key] >= measure_mm:
                overhang = s[key] - measure_mm
                print(f"  {bone.upper()} {mode} = {measure_mm:.1f}mm → "
                      f"Size {s['size']} ({s[key]}mm, +{overhang:.1f}mm) [round-up]")
                return s
    else:
        # Round down: largest size that doesn't exceed the bone (<= measurement)
        for s in reversed(sorted_tab):
            if s[key] <= measure_mm:
                under = measure_mm - s[key]
                print(f"  {bone.upper()} {mode} = {measure_mm:.1f}mm → "
                      f"Size {s['size']} ({s[key]}mm, -{under:.1f}mm) [round-down]")
                return s

    # Fallback: bone exceeds all sizes (round-up) or is below all (round-down)
    fallback = sorted_tab[-1] if SIZE_ROUND == "U" else sorted_tab[0]
    print(f"  [WARNING] {bone.upper()} {mode} = {measure_mm:.1f}mm out of range. "
          f"Using {fallback['size']}.")
    return fallback


def select_size_quiet(measurement_m, bone, mode, size_table):
    """Same logic as select_size but silent — respects SIZE_ROUND."""
    key = f"{bone}_{mode}"
    measure_mm = measurement_m * 1000
    sorted_tab = sorted(size_table, key=lambda s: s[key])
    if SIZE_ROUND == "U":
        for s in sorted_tab:
            if s[key] >= measure_mm:
                return s
    else:
        for s in reversed(sorted_tab):
            if s[key] <= measure_mm:
                return s
    return sorted_tab[-1] if SIZE_ROUND == "U" else sorted_tab[0]


# =============================================================================
# SECTION 14: FEMUR PIPELINE
# =============================================================================

def run_femur(side, size_table, input_dir, output_dir, tea_z_rotation=None,
              size_override=None, y_offset=0.0, do_export=True):
    """Full femur pipeline.  Returns a dict consumed by run_tibia.

    tea_z_rotation : if provided (radians), skip TEA detection and use this.
    size_override  : force a specific size label (e.g. "M1").
    y_offset       : additional Y shift applied to all objects (metres).
    do_export      : if False, skip STL export (used for comparison sets).
    """
    print("\n─── FEMUR PIPELINE ───\n")
    side_cap = side.capitalize()

    # ── Step 1: Prepare bone ─────────────────────────────────────────────
    bone_path = os.path.join(input_dir, FEMUR_STL_FILENAME.format(side=side_cap))
    femur = prepare_bone(bone_path, "Femur",
                         cut_end="distal", cut_length=BONE_CUT_LENGTH)
    validate_bone_orientation(femur, "femur")
    step_pause("Femur imported, PCA-aligned, cut")

    # ── Step 2: TEA alignment ────────────────────────────────────────────
    if tea_z_rotation is not None:
        z_rot = tea_z_rotation
        print(f"  Using pre-computed TEA rotation: {math.degrees(z_rot):.1f}°")
    elif USE_TEA == "T":
        tea_angle = detect_tea_angle(femur)
        # Correction sign depends on side: left = +, right = -
        # (external rotation is toward the lateral side, which flips)
        correction_sign = +1 if side == "left" else -1
        correction = correction_sign * math.radians(TEA_CORRECTION_DEG)
        z_rot = -tea_angle + correction
        print(f"  TEA Z rotation: {math.degrees(z_rot):.1f}° "
              f"(TEA {math.degrees(tea_angle):.1f}° from Y, "
              f"correction {correction_sign:+d}×{TEA_CORRECTION_DEG}°)")
    else:
        z_rot = math.radians(TIBIA_MANUAL_Z_ROT_DEG)
        print(f"  TEA disabled — using manual Z rotation: {TIBIA_MANUAL_Z_ROT_DEG}°")

    apply_z_rotation_around_centroid(femur, z_rot)
    apply_transforms(femur)

    # Re-centre XY after rotation
    bbox = get_bounding_box(femur)
    femur.location.x = -bbox["center_x"]
    femur.location.y = -bbox["center_y"]
    apply_transforms(femur)
    step_pause("Femur TEA-aligned")

    # ── Step 3: Measure articular region ─────────────────────────────────
    region = get_regional_bbox(femur, "distal", REGIONAL_BBOX_FRACTION)
    full   = get_bounding_box(femur)
    ML, AP = region["ML"], region["AP"]
    print(f"  Femur distal region — ML: {ML*1000:.1f}mm  AP: {AP*1000:.1f}mm")
    print(f"  Femur full bbox     — Height: {full['height']*1000:.1f}mm")

    # ── Step 4: Size selection ───────────────────────────────────────────
    primary_dim = ML if SIZE_MODE == "ML" else AP
    if size_override:
        selected = next((s for s in size_table if s["size"] == size_override), None)
        if selected is None:
            raise ValueError(f"size_override '{size_override}' not in size table.")
        size_label = selected["size"]
        print(f"  [OVERRIDE] Size: {size_label}")
    elif MANUAL_SIZE:
        selected = next((s for s in size_table if s["size"] == MANUAL_SIZE), None)
        if selected is None:
            valid = ", ".join(s["size"] for s in size_table)
            raise ValueError(f"MANUAL_SIZE '{MANUAL_SIZE}' not in size table. "
                             f"Valid: {valid}")
        size_label = selected["size"]
        print(f"  [MANUAL] Size override: {size_label}")
    else:
        selected    = select_size(primary_dim, "femur", SIZE_MODE, size_table)
        size_label  = selected["size"]

    other_mode = "AP" if SIZE_MODE == "ML" else "ML"
    other_dim  = AP   if SIZE_MODE == "ML" else ML
    other_sel  = select_size_quiet(other_dim, "femur", other_mode, size_table)
    if other_sel["size"] != size_label:
        print(f"  [WARNING] Femur {other_mode} cross-check → {other_sel['size']}")

    # ── Step 5: Shift femur condyles to Z=0 ──────────────────────────────
    bbox = get_bounding_box(femur)
    _, _, condyle_z, _ = find_condyle_contact(femur, bbox, n=CONTACT_AVG_N)
    femur.location.z = -condyle_z
    apply_transforms(femur)
    print(f"  Condyle contact shifted to Z=0")

    # ── Step 6: Valgus rotation ──────────────────────────────────────────
    # Valgus tilts in the coronal plane → rotation around X (AP axis).
    # Convention: X = AP, Y = ML, Z = SI.
    valgus_rad = math.radians(VALGUS_ANGLE_DEG)
    femur.rotation_euler.x = -valgus_rad if side == "left" else +valgus_rad
    apply_transforms(femur)

    # Re-anchor post-valgus
    bbox_a = get_bounding_box(femur)
    _, _, cz_a, _ = find_condyle_contact(femur, bbox_a, n=CONTACT_AVG_N)
    if abs(cz_a) > 1e-6:
        femur.location.z = -cz_a
        apply_transforms(femur)
        print(f"  Post-valgus re-anchor: {-cz_a*1000:.3f}mm")

    # Component XY anchor from distal region
    distal = get_regional_bbox(femur, "distal", REGIONAL_BBOX_FRACTION)
    comp_x = distal["center_x"]
    comp_y = distal["center_y"]

    # Apply y_offset to both the bone and the component anchor
    if abs(y_offset) > 1e-6:
        femur.location.y = y_offset
        apply_transforms(femur)
        comp_y += y_offset
        print(f"  Y offset applied: {y_offset*1000:.1f}mm")

    print(f"  Component anchor XY: ({comp_x*1000:.2f}, {comp_y*1000:.2f}) mm")
    step_pause("Femur measured, valgus applied")

    # ── Step 7: Femoral components (native origin) ───────────────────────
    size_dir      = _size_folder(size_label)
    cb_path       = os.path.join(size_dir, selected["femur_cb_stl"])
    cb_shell_path = os.path.join(size_dir, selected["femur_cb_shell_stl"])
    implant_key   = f"implant_femoral_{side}"
    implant_path  = os.path.join(size_dir, selected[implant_key])

    # All femoral components share the same CAD origin.
    # Place them at the same world position → they align automatically.
    origin_z  = selected["femur_origin_z"]
    comp_z    = selected["femur_distal_cut"] - origin_z

    femur_cb = import_component_native(
        cb_path, f"CuttingBlock_Femoral_{size_label}", FEMUR_COMPONENT_Z_ROT)
    femur_cb.location = (comp_x + FEMUR_CB_OFFSET_X,
                         comp_y + FEMUR_CB_OFFSET_Y, comp_z)
    apply_transforms(femur_cb)
    if do_export:
        export_stl(femur_cb, output_dir, f"CuttingBlock_Femoral_{size_label}.stl")

    femur_cb_shell = import_component_native(
        cb_shell_path, f"CB_FemoralShell_{size_label}", FEMUR_COMPONENT_Z_ROT)
    femur_cb_shell.location = (comp_x + FEMUR_CB_OFFSET_X,
                               comp_y + FEMUR_CB_OFFSET_Y, comp_z)
    apply_transforms(femur_cb_shell)

    femoral_implant = import_component_native(
        implant_path, f"Implant_Femoral_{side_cap}_{size_label}",
        FEMUR_COMPONENT_Z_ROT)
    femoral_implant.location = (comp_x + FEMUR_IMPLANT_OFFSET_X,
                                comp_y + FEMUR_IMPLANT_OFFSET_Y, comp_z)
    apply_transforms(femoral_implant)

    cb_bbox = get_bounding_box(femur_cb)
    impl_bbox = get_bounding_box(femoral_implant)
    print(f"  Femoral CB   — top: {cb_bbox['z_max']*1000:.2f}mm  "
          f"bottom: {cb_bbox['z_min']*1000:.2f}mm")
    print(f"  Femoral impl — top: {impl_bbox['z_max']*1000:.2f}mm  "
          f"bottom: {impl_bbox['z_min']*1000:.2f}mm")
    step_pause("Femoral components positioned")

    # ── Step 8: Booleans ─────────────────────────────────────────────────
    femur_shell = duplicate_object(femur, "Femur.Shell")
    femur       = boolean_op(femur,       femur_cb,       'DIFFERENCE', delete_cutter=False)
    femur_shell = boolean_op(femur_shell, femur_cb_shell, 'INTERSECT',  delete_cutter=False)

    # ── Step 9: Export and hide ──────────────────────────────────────────
    if do_export:
        export_stl(femur,           output_dir, f"Femur_{side_cap}_{size_label}_result.stl")
        export_stl(femur_shell,     output_dir, f"Femur_{side_cap}_{size_label}_Shell.stl")
        export_stl(femoral_implant, output_dir, f"Implant_Femoral_{side_cap}_{size_label}.stl")

        # Hide cutting blocks and shells (primary run only)
        hide_object(femur_cb)
        hide_object(femur_cb_shell)
        hide_object(femur_shell)

    implant_bottom_z = impl_bbox["z_min"]
    print(f"\n[FEMUR DONE] Size: {size_label}  "
          f"Implant bottom Z: {implant_bottom_z*1000:.2f}mm\n")

    return {
        "implant_bottom_z" : implant_bottom_z,
        "implant_y"        : comp_y + FEMUR_IMPLANT_OFFSET_Y,
        "comp_x"           : comp_x,
        "comp_y"           : comp_y,
        "size_label"       : size_label,
        "selected"         : selected,
        "tea_z_rotation"   : z_rot,
    }


# =============================================================================
# SECTION 15: TIBIA PIPELINE
# =============================================================================

def run_tibia(side, size_table, femur_results, input_dir, output_dir,
              do_export=True):
    """Full tibia pipeline.  Uses femur-driven size and positioning."""
    print("\n─── TIBIA PIPELINE ───\n")
    side_cap = side.capitalize()

    implant_bottom_z = femur_results["implant_bottom_z"]
    femur_implant_y  = femur_results["implant_y"]
    tea_z_rotation   = femur_results["tea_z_rotation"]
    size_label       = femur_results["size_label"]
    selected         = femur_results["selected"]

    # ── Step 1: Prepare bone ─────────────────────────────────────────────
    bone_path = os.path.join(input_dir, TIBIA_STL_FILENAME.format(side=side_cap))
    tibia = prepare_bone(bone_path, "Tibia",
                         cut_end="proximal", cut_length=BONE_CUT_LENGTH)
    validate_bone_orientation(tibia, "tibia")

    # ── Step 2: Rotational alignment ─────────────────────────────────────
    if BONE_IMPORT_MODE == "R":
        # Use the same rotation as the femur
        apply_z_rotation_around_centroid(tibia, tea_z_rotation)
        print(f"  Tibia: using femur's TEA rotation ({math.degrees(tea_z_rotation):.1f}°)")
    else:
        if abs(TIBIA_MANUAL_Z_ROT_DEG) > 1e-6:
            apply_z_rotation_around_centroid(tibia, math.radians(TIBIA_MANUAL_Z_ROT_DEG))
            print(f"  Tibia: manual Z rotation ({TIBIA_MANUAL_Z_ROT_DEG}°)")

    apply_transforms(tibia)
    bbox = get_bounding_box(tibia)
    tibia.location.x = -bbox["center_x"]
    tibia.location.y = -bbox["center_y"]
    apply_transforms(tibia)
    step_pause("Tibia imported and aligned")

    # ── Step 3: Measure proximal region ──────────────────────────────────
    region = get_regional_bbox(tibia, "proximal", REGIONAL_BBOX_FRACTION)
    full   = get_bounding_box(tibia)
    ML, AP = region["ML"], region["AP"]
    print(f"  Tibia proximal region — ML: {ML*1000:.1f}mm  AP: {AP*1000:.1f}mm")

    # Cross-check size
    primary_dim = ML if SIZE_MODE == "ML" else AP
    tibia_sel = select_size_quiet(primary_dim, "tibia", SIZE_MODE, size_table)
    if tibia_sel["size"] != size_label:
        print(f"  [SIZE MISMATCH] Tibia → {tibia_sel['size']} "
              f"(using femur-driven {size_label})")
    else:
        print(f"  Tibia size matches femur: {size_label}")

    # ── Step 4: Component anchor XY ──────────────────────────────────────
    # Y cascades from femoral implant position.
    # X uses tibia's own proximal region + group offset.
    prox_region = get_regional_bbox(tibia, "proximal", REGIONAL_BBOX_FRACTION)
    tib_comp_x  = prox_region["center_x"] + TIBIA_GROUP_OFFSET_X
    tib_comp_y  = femur_implant_y
    print(f"  Tibial anchor XY: ({tib_comp_x*1000:.2f}, {tib_comp_y*1000:.2f}) mm  "
          f"(Y from femoral implant)")

    # ── Step 5: Compute tibial Z references ──────────────────────────────
    plate_top_z    = implant_bottom_z - FEMUR_TIBIA_GAP
    origin_z       = selected["tibia_origin_z"]
    proximal_cut   = selected["tibia_proximal_cut"]
    comp_z         = plate_top_z - origin_z   # where the shared tibial origin goes

    # The cut surface = comp_z + origin_z = plate_top_z (by construction)
    # But if origin_z ≠ 0 and the plate body has thickness, the bone cut surface
    # is determined by the actual geometry. For now:
    bone_cut_z     = plate_top_z   # the mating surface (will refine after plate import)

    print(f"  Plate top Z        : {plate_top_z*1000:.2f}mm")
    print(f"  Component origin Z : {comp_z*1000:.2f}mm")

    # ── Step 6: Position tibial components (native origins) ──────────────
    size_dir    = _size_folder(size_label)
    plate_path  = os.path.join(size_dir, selected["tibial_plate_stl"])
    insert_path = os.path.join(size_dir, selected["tibial_insert_stl"])
    cb_path     = os.path.join(size_dir, selected["tibia_cb_stl"])
    cb_shell_path = os.path.join(size_dir, selected["tibia_cb_shell_stl"])

    # All tibial components share a CAD origin → same world position
    tibial_plate = import_component_native(
        plate_path, f"Tibial_Plate_{size_label}", TIBIA_COMPONENT_Z_ROT)
    tibial_plate.location = (tib_comp_x + TIBIA_PLATE_OFFSET_X,
                             tib_comp_y + TIBIA_PLATE_OFFSET_Y, comp_z)
    apply_transforms(tibial_plate)

    tibial_insert = import_component_native(
        insert_path, f"Tibial_Insert_{size_label}", TIBIA_COMPONENT_Z_ROT)
    tibial_insert.location = (tib_comp_x + TIBIA_INSERT_OFFSET_X,
                              tib_comp_y + TIBIA_INSERT_OFFSET_Y, comp_z)
    apply_transforms(tibial_insert)

    tibia_cb = import_component_native(
        cb_path, f"CuttingBlock_Tibia_{size_label}", TIBIA_COMPONENT_Z_ROT)
    tibia_cb.location = (tib_comp_x + TIBIA_CB_OFFSET_X,
                         tib_comp_y + TIBIA_CB_OFFSET_Y, comp_z)
    apply_transforms(tibia_cb)

    tibia_cb_shell = import_component_native(
        cb_shell_path, f"CB_TibiaShell_{size_label}", TIBIA_COMPONENT_Z_ROT)
    tibia_cb_shell.location = (tib_comp_x + TIBIA_CB_OFFSET_X,
                               tib_comp_y + TIBIA_CB_OFFSET_Y, comp_z)
    apply_transforms(tibia_cb_shell)

    # Print component positions
    pb = get_bounding_box(tibial_plate)
    cb_b = get_bounding_box(tibia_cb)
    ib = get_bounding_box(tibial_insert)
    print(f"  Plate  — top: {pb['z_max']*1000:.2f}mm  bottom: {pb['z_min']*1000:.2f}mm")
    print(f"  Insert — top: {ib['z_max']*1000:.2f}mm  bottom: {ib['z_min']*1000:.2f}mm")
    print(f"  CB     — top: {cb_b['z_max']*1000:.2f}mm  bottom: {cb_b['z_min']*1000:.2f}mm")

    # ── Step 7: Position tibia bone ──────────────────────────────────────
    # The bone's top should be proximal_cut above the bone cut surface.
    # Use the CB's actual geometry to determine cut surface Z:
    # the CB bottom = where bone material below the cut is removed to
    # The plate top ≈ bone cut surface for positioning purposes
    bone_cut_z_actual = plate_top_z   # plate top is the mating surface
    tibia_bone_top_target = bone_cut_z_actual + proximal_cut

    current_bbox = get_bounding_box(tibia)
    bone_shift_x = (tib_comp_x + TIBIA_BONE_OFFSET_X) - prox_region["center_x"]
    bone_shift_y = tib_comp_y - prox_region["center_y"]
    bone_shift_z = tibia_bone_top_target - current_bbox["z_max"]

    tibia.location.x = bone_shift_x
    tibia.location.y = bone_shift_y
    tibia.location.z = bone_shift_z
    apply_transforms(tibia)

    bp = get_bounding_box(tibia)
    print(f"  Tibia bone — top: {bp['z_max']*1000:.2f}mm  "
          f"bottom: {bp['z_min']*1000:.2f}mm")
    print(f"  Bone above cut: {(bp['z_max'] - bone_cut_z_actual)*1000:.2f}mm  "
          f"(target proximal_cut: {proximal_cut*1000:.2f}mm)")
    step_pause("Tibial components and bone positioned")

    # ── Step 8: Posterior slope ───────────────────────────────────────────
    if abs(TIBIA_POSTERIOR_SLOPE_DEG) > 1e-6:
        slope_center = (tib_comp_x, tib_comp_y, bone_cut_z_actual)
        print(f"  Applying {TIBIA_POSTERIOR_SLOPE_DEG:.1f}° posterior slope")
        for comp in [tibial_plate, tibial_insert, tibia_cb, tibia_cb_shell]:
            apply_posterior_slope(comp, TIBIA_POSTERIOR_SLOPE_DEG, slope_center)
        step_pause("Posterior slope applied")

    # ── Step 9: Export CB, then booleans ─────────────────────────────────
    if do_export:
        export_stl(tibia_cb, output_dir, f"CuttingBlock_Tibia_{size_label}.stl")

    tibia_shell = duplicate_object(tibia, "Tibia.Shell")
    tibia       = boolean_op(tibia,       tibia_cb,       'DIFFERENCE', delete_cutter=False)
    tibia_shell = boolean_op(tibia_shell, tibia_cb_shell, 'INTERSECT',  delete_cutter=False)

    # ── Step 10: Export and hide ─────────────────────────────────────────
    if do_export:
        export_stl(tibia,         output_dir, f"Tibia_{side_cap}_{size_label}_result.stl")
        export_stl(tibia_shell,   output_dir, f"Tibia_{side_cap}_{size_label}_Shell.stl")
        export_stl(tibial_plate,  output_dir, f"Tibial_Plate_{size_label}.stl")
        export_stl(tibial_insert, output_dir, f"Tibial_Insert_{size_label}.stl")

        # Hide cutting blocks and shells (primary run only)
        hide_object(tibia_cb)
        hide_object(tibia_cb_shell)
        hide_object(tibia_shell)

    print(f"\n[TIBIA DONE] Size: {size_label}\n")


# =============================================================================
# SECTION 16: LOGGING
# =============================================================================

class _Tee:
    def __init__(self, filepath, original_stdout):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = original_stdout
    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)
        self._file.flush()
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


# =============================================================================
# SECTION 17: ENTRY POINTS
# =============================================================================

def run_single_patient(patient_dir, side, size_table):
    """Process one patient side.  patient_dir contains the bone STLs."""
    output_dir = os.path.join(patient_dir, "Output")
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(output_dir, f"tka_log_{timestamp}.txt")
    original_stdout = sys.stdout
    tee = _Tee(log_path, original_stdout)
    sys.stdout = tee

    try:
        print(f"TKA AUTO PIPELINE v3 — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Patient: {patient_dir}")
        print(f"Side: {side}")
        print(f"Log: {log_path}")
        print(f"\n{'='*60}\n")

        clear_scene()

        # In "R" mode: detect TEA from femur first, pass to both
        tea_z_rot = None
        if BONE_IMPORT_MODE == "R" and USE_TEA == "T":
            print("  [Retained mode] Pre-detecting TEA from femur...")
            side_cap = side.capitalize()
            bone_path = os.path.join(patient_dir,
                                     FEMUR_STL_FILENAME.format(side=side_cap))
            temp_femur = import_stl(bone_path, "_TEA_detect")
            apply_transforms(temp_femur)
            if USE_PCA == "T":
                pca_align_shaft(temp_femur)
                apply_transforms(temp_femur)
            tea_angle = detect_tea_angle(temp_femur)
            correction_sign = +1 if side == "left" else -1
            correction = correction_sign * math.radians(TEA_CORRECTION_DEG)
            tea_z_rot = -tea_angle + correction
            bpy.data.objects.remove(temp_femur, do_unlink=True)
            for mesh in list(bpy.data.meshes):
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            print(f"  TEA pre-computed: {math.degrees(tea_z_rot):.1f}° "
                  f"(correction {correction_sign:+d}×{TEA_CORRECTION_DEG}°)")
            print()

        femur_results = run_femur(side, size_table, patient_dir, output_dir,
                                  tea_z_rotation=tea_z_rot)
        run_tibia(side, size_table, femur_results, patient_dir, output_dir)

        # ── Size comparison mode ─────────────────────────────────────────
        if COMPARE_SIZE == "T":
            _run_comparison(side, size_table, femur_results, tea_z_rot,
                            patient_dir, output_dir)

        # ── Save .blend file ─────────────────────────────────────────────
        blend_path = os.path.join(output_dir, f"tka_{side}_{timestamp}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"  [BLEND] Saved → {blend_path}")

        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE — Log: {log_path}")
        print(f"{'='*60}\n")

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"[OK] Log saved → {log_path}")


def _get_size_down(size_label, size_table):
    """Return the next size label below size_label, or None if smallest."""
    sorted_tab = sorted(size_table, key=lambda s: s["femur_ML"])
    labels = [s["size"] for s in sorted_tab]
    if size_label not in labels:
        return None
    idx = labels.index(size_label)
    if idx == 0:
        return None
    return labels[idx - 1]


def _collect_objects_into(collection_name):
    """Move all scene objects into a new collection."""
    coll = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(coll)
    for obj in list(bpy.context.scene.collection.objects):
        bpy.context.scene.collection.objects.unlink(obj)
        coll.objects.link(obj)
    return coll


def _run_comparison(side, size_table, femur_results, tea_z_rot,
                    patient_dir, output_dir):
    """Run the full pipeline at one size down, shifted +200mm in Y.

    1. Moves existing objects into 'Primary_{size}' collection.
    2. Runs the full pipeline (bones + components + booleans) at the
       comparison size with a Y offset.
    3. Moves new objects into 'Comparison_{size}' collection.
    Both collections are visible and toggleable in the Outliner.
    """
    primary_size = femur_results["size_label"]
    comp_size    = _get_size_down(primary_size, size_table)
    if comp_size is None:
        print(f"\n  [COMPARE] {primary_size} is the smallest size — no comparison.")
        return

    print(f"\n{'='*60}")
    print(f"  SIZE COMPARISON: {primary_size} (primary) vs {comp_size} (comparison)")
    print(f"  Comparison set shifted +200mm in Y")
    print(f"{'='*60}\n")

    # Move primary objects into their own collection
    primary_coll = _collect_objects_into(f"Primary_{primary_size}")
    print(f"  Primary objects → collection 'Primary_{primary_size}'")

    # Run full pipeline at comparison size, shifted +200mm in Y
    y_shift = 0.200  # 200 mm

    comp_femur_results = run_femur(
        side, size_table, patient_dir, output_dir,
        tea_z_rotation=tea_z_rot,
        size_override=comp_size,
        y_offset=y_shift,
        do_export=False,
    )
    run_tibia(
        side, size_table, comp_femur_results, patient_dir, output_dir,
        do_export=False,
    )

    # Move comparison objects into their own collection
    # (they are currently in the scene root, since primary was already moved out)
    comp_coll = _collect_objects_into(f"Comparison_{comp_size}")
    print(f"  Comparison objects → collection 'Comparison_{comp_size}'")
    print(f"\n  Toggle collections in the Outliner to compare sizes.")


def _side_to_full(s):
    """Convert L/R to 'left'/'right'."""
    return {"L": "left", "R": "right"}[s.upper()]


def run():
    """Single-patient entry point.  Handles L, R, or B (both) sides."""
    size_table = load_size_table(CSV_PATH)

    if SIDE.upper() == "B":
        sides = ["L", "R"]
    else:
        sides = [SIDE.upper()]

    for s in sides:
        side_full = _side_to_full(s)
        side_cap  = side_full.capitalize()
        input_dir = os.path.join(PATIENT_DIR, side_cap)
        if not os.path.isdir(input_dir):
            if len(sides) == 1:
                # Try PATIENT_DIR directly (no Left/Right subfolder)
                input_dir = PATIENT_DIR
            else:
                print(f"[SKIP] {side_cap}/ folder not found in {PATIENT_DIR}")
                continue
        run_single_patient(input_dir, side_full, size_table)


def run_batch():
    """Process all patient folders in BATCH_DIR."""
    if not os.path.isdir(BATCH_DIR):
        raise FileNotFoundError(f"Batch directory not found: {BATCH_DIR}")

    size_table = load_size_table(CSV_PATH)
    patient_dirs = sorted([
        d for d in os.listdir(BATCH_DIR)
        if os.path.isdir(os.path.join(BATCH_DIR, d))
    ])

    print(f"\n========== BATCH MODE: {len(patient_dirs)} patients ==========\n")

    for i, folder_name in enumerate(patient_dirs, 1):
        patient_path = os.path.join(BATCH_DIR, folder_name)

        # Process each side subfolder found
        for side_name in ["Left", "Right"]:
            side_path = os.path.join(patient_path, side_name)
            if not os.path.isdir(side_path):
                continue

            side = side_name.lower()
            print(f"\n{'='*60}")
            print(f"  PATIENT {i}/{len(patient_dirs)}: {folder_name} — {side_name}")
            print(f"{'='*60}\n")

            try:
                run_single_patient(side_path, side, size_table)
                print(f"[OK] {folder_name}/{side_name} complete.")
            except Exception as e:
                print(f"[ERROR] {folder_name}/{side_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n========== BATCH COMPLETE ==========\n")


if __name__ == "__main__":
    if BATCH_MODE == "T":
        run_batch()
    else:
        run()


# =============================================================================
# SIZE CHART — COLUMN REFERENCE
# =============================================================================
#
# All values in the CSV are POSITIVE.
#
# femur_distal_cut_m   : resection depth (m).  Bone removed upward from condyle plane.
# femur_origin_z_m     : Z offset from STL origin to cut surface (m).
#                        0 if origin IS at cut surface.
# tibia_proximal_cut_m : resection depth (m).  Bone removed downward from plateau.
# tibia_origin_z_m     : Z offset from STL origin to cut surface (m).
#                        0 if origin IS at cut surface.
#
# With native origins: all femoral components placed at the same (x, y, z).
# All tibial components placed at the same (x, y, z).  Alignment is by
# construction (shared CAD origin).
#
# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#
# Problem: Components misaligned despite native origins
#   The component STLs don't share a CAD origin.  Verify by importing two
#   components into Blender manually — they should overlap.  If not, the
#   STLs were exported with different origins.
#
# Problem: PCA aligns shaft to wrong axis
#   The bone mesh is very asymmetric or has artifacts.  Clean the mesh
#   and retry.  Override with BONE_IMPORT_MODE = "I" and
#   manual rotation.
#
# Problem: TEA detection finds wrong points
#   The epicondylar detection band (15-35% above condyle tips) may not
#   match the bone anatomy.  Adjust the band percentages in detect_tea_angle().
#
# Problem: Step-by-step mode doesn't refresh viewport
#   This only works in Blender GUI mode, not headless (--background).
#
# Problem: Sizing too large or too small
#   Toggle SIZE_ROUND between "U" (up) and "D" (down).  Noisy meshes
#   inflate the bounding box → try "D".  Clean meshes → try "U".
#
# Problem: Components rotated 90° relative to bone
#   Adjust FEMUR_COMPONENT_Z_ROT / TIBIA_COMPONENT_Z_ROT.  After PCA+TEA
#   alignment the axis convention is X=AP, Y=ML.  If your CAD exports use
#   Y=AP, X=ML, add or subtract 90° to the component rotation.
#
# Axis convention (after PCA+TEA alignment):
#   X = AP (anteroposterior)
#   Y = ML (mediolateral, TEA aligned here)
#   Z = SI (superior-inferior, bone shaft)