# TKA Automatic Pipeline v3 — README

## Overview

`tka_auto_pipeline_v3.py` is a Blender Python script that automates the complete pre-surgical geometry preparation workflow for a **Total Knee Arthroplasty (TKA)** custom implant system. Given raw femur and tibia bone STL files (typically from CT segmentation via 3D Slicer), the script:

1. Imports each bone and automatically aligns it using PCA (shaft → Z axis) and TEA detection (mediolateral → Y axis)
2. Trims the bone to a working length from the articular end
3. Measures the articular region's mediolateral (ML) and anteroposterior (AP) dimensions
4. Automatically selects the correct implant size from a 12-size chart (S1 through L4), with manual override available
5. Imports all size-specific components (cutting blocks, shells, implant, tibial plate, tibial insert) using their native CAD origins for automatic alignment
6. Positions all components using anatomically derived geometry with configurable fine-tuning offsets
7. Applies valgus rotation to the femur and posterior slope to the tibial components
8. Performs boolean operations to cut the bone geometry
9. Exports all resulting STL files and a timestamped log file
10. Optionally runs a side-by-side size comparison with one size smaller

The script supports single-patient and batch processing modes, and includes a step-by-step demo mode with viewport pauses for presentations.

---

## What Changed from v1/v2

**Bone alignment (replaces hardcoded rotation):**
- PCA-based shaft alignment detects the bone's longest axis and rotates it to +Z automatically, regardless of the patient's leg orientation during the CT scan
- TEA (transepicondylar axis) detection uses 2D PCA on the epicondylar band to find the ML axis and align it to world Y
- Two import modes: "retained" (bones from the same CT share rotation) and "independent" (each bone aligned separately)

**Component positioning (replaces centroid-based offsets):**
- Native STL origins: all components for a given bone side share the same CAD origin, so placing them at the same world position aligns them automatically
- Eliminated 5 manual offset columns from the CSV (`femur_cb_z_offset`, `femur_implant_z_off`, `tibia_cb_z_offset`, `tibia_plate_z_off`, `tibia_insert_z_off`)
- CSV reduced to 12 columns from the original 14+

**Sizing:**
- Regional bounding-box measurement (distal 25% for femur, proximal 25% for tibia) instead of full-bone bbox
- Round-up or round-down selection via `SIZE_ROUND` config
- Manual size override via `MANUAL_SIZE`
- Femur drives the system size; tibia cross-checks with a warning

**New features:**
- Batch processing across patient folders
- Size comparison mode (full pipeline at two sizes, side by side in collections)
- Step-by-step demo mode with viewport pauses
- All config values in millimetres
- Configurable bone filenames
- Separate shared Scaled_STLs path
- Automatic log file generation
- Cutting blocks and shells hidden (toggleable) after booleans
- Tibial group X offset and bone-only X offset controls

**Axis convention (enforced throughout):**
- X = AP (anteroposterior)
- Y = ML (mediolateral, TEA aligned here)
- Z = SI (superior-inferior, bone shaft)
- Valgus rotation around X (coronal plane)
- Posterior slope around Y (sagittal plane)

---

## Requirements

- **Blender 3.x or 4.x** (tested on both; auto-detects import/export operators)
- **NumPy** — ships with Blender's bundled Python (used for PCA)
- No additional external packages required

---

## How to Run

### Option A — Blender GUI (recommended)

**For development and debugging (Scripting tab):**
1. Open Blender → switch to the **Scripting** workspace
2. Click **Open** and select `tka_auto_pipeline_v3.py`
3. Edit the configuration section at the top of the file
4. Click **▶ Run Script**

**For demos and presentations (Layout tab with split view):**
1. Open Blender in the **Layout** workspace
2. Split one area: hover over an area edge, Shift+click and drag to create a Text Editor
3. Open the script in the Text Editor
4. Set `STEP_BY_STEP_MODE = "T"` for pauses between steps
5. Click **▶ Run Script** — watch the viewport update in real time
6. To maximise the viewport at any time: hover over it and press **Ctrl+Space**
7. For dual monitors: **Window → New Main Window**, put the viewport on one screen and the text editor on the other

### Option B — Headless / batch

```bash
blender --background --python tka_auto_pipeline_v3.py
```

Note: step-by-step mode viewport refreshes are skipped in headless mode.

---

## Directory Structure

### Single patient

```
PATIENT_DIR/
├── Left/
│   ├── FD1Left.stl            ← femur bone
│   ├── TD1Left.stl            ← tibia bone
│   └── Output/                ← created by script
│       ├── tka_log_20260401_143000.txt
│       ├── Femur_Left_M2_result.stl
│       ├── Femur_Left_M2_Shell.stl
│       ├── CuttingBlock_Femoral_M2.stl
│       ├── Implant_Femoral_Left_M2.stl
│       ├── Tibia_Left_M2_result.stl
│       ├── Tibia_Left_M2_Shell.stl
│       ├── CuttingBlock_Tibia_M2.stl
│       ├── Tibial_Plate_M2.stl
│       └── Tibial_Insert_M2.stl
└── Right/                     ← optional, same structure
```

### Shared implant library (separate location)

```
SCALED_STLS_DIR/
├── SizeChart.csv
├── S1/
│   ├── CuttingBlockFemoral(L&R)_S1.stl
│   ├── CuttingBlockFemoralShell(L&R)_S1.stl
│   ├── CuttingBlockTibia(L&R)_S1.stl
│   ├── CuttingBlockTibiaShell(L&R)_S1.stl
│   ├── Implant(Femoral)Left_S1.stl
│   ├── Implant(Femoral)Right_S1.stl
│   ├── Tibial_Insert(L&R)_S1.stl
│   └── Tibial_Plate(L&R)_S1.stl
├── S2/ ... L4/
```

### Batch processing

```
BATCH_DIR/
├── Patient_001/
│   ├── Left/
│   │   ├── FD1Left.stl
│   │   └── TD1Left.stl
│   └── Right/
│       ├── FD1Right.stl
│       └── TD1Right.stl
├── Patient_002/
│   └── Left/
│       ├── FD1Left.stl
│       └── TD1Left.stl
└── ...
```

Each patient side gets its own `Output/` folder with STL results and a log file. The script processes all patients alphabetically, skipping any that fail with an error message.

---

## Configuration Reference

All user-facing settings are in Section 1 at the top of the script. All length values are in **millimetres**; they are converted to metres internally.

### Paths

| Parameter | Description |
|-----------|-------------|
| `PATIENT_DIR` | Root folder for the patient (contains `Left/` and/or `Right/` subfolders) |
| `SCALED_STLS_DIR` | Shared folder containing implant STLs and `SizeChart.csv` |
| `CSV_PATH` | Path to the size chart CSV (defaults to inside `SCALED_STLS_DIR`) |
| `SIDE` | `"left"` or `"right"` — which side to process |

### Bone filenames

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FEMUR_STL_FILENAME` | `"FD1{side}.stl"` | `{side}` is replaced with `"Left"` or `"Right"` |
| `TIBIA_STL_FILENAME` | `"TD1{side}.stl"` | Same replacement |

### Bone import mode

| Value | Description |
|-------|-------------|
| `"R"` (retained) | Both bones from the same CT scan — share the femur's TEA rotation |
| `"I"` (independent) | Bones exported separately — each aligned independently |

In retained mode, TEA is detected from the femur and applied to both bones. In independent mode, the femur uses PCA+TEA and the tibia uses PCA + `TIBIA_MANUAL_Z_ROT_DEG`.

### Surgical parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FEMUR_TIBIA_GAP_MM` | 20.0 | Visual gap between femoral implant bottom and tibial plate top |
| `BONE_CUT_LENGTH_MM` | 150.0 | Length of bone to keep from the articular end (`None` = no cut) |
| `VALGUS_ANGLE_DEG` | 6.0 | Coronal plane correction (rotation around X/AP axis) |
| `SIZE_MODE` | `"ML"` | Primary dimension for size selection (`"ML"` or `"AP"`) |
| `SIZE_ROUND` | `"U"` | `"U"` = round up (overhang preferred), `"D"` = round down |
| `MANUAL_SIZE` | `""` | Force a specific size, e.g. `"M2"` (empty = automatic) |
| `COMPARE_SIZE` | `"F"` | `"T"` = run full pipeline at one size down for side-by-side comparison |

### TEA and rotation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEA_CORRECTION_DEG` | 3.0 | External rotation correction applied after TEA alignment |
| `TIBIA_MANUAL_Z_ROT_DEG` | 0.0 | Manual Z rotation for tibia (only in `"I"` mode) |
| `FEMUR_COMPONENT_Z_ROT` | -90 | Z rotation for femoral component STLs at import |
| `TIBIA_COMPONENT_Z_ROT` | 0 | Z rotation for tibial component STLs at import |

### Manual XY offsets (mm)

Fine-tuning shifts applied after automatic centring. Set all to 0 for a new bone, inspect, then adjust.

**Femoral offsets:**

| Parameter | Description |
|-----------|-------------|
| `FEMUR_CB_OFFSET_X_MM` | Shifts femoral cutting block in X |
| `FEMUR_CB_OFFSET_Y_MM` | Shifts femoral cutting block in Y |
| `FEMUR_IMPLANT_OFFSET_X_MM` | Shifts femoral implant in X |
| `FEMUR_IMPLANT_OFFSET_Y_MM` | Shifts femoral implant in Y (cascades to tibial group Y) |

**Tibial offsets:**

| Parameter | Description |
|-----------|-------------|
| `TIBIA_GROUP_OFFSET_X_MM` | Shifts ALL tibial components AND the tibia bone in X |
| `TIBIA_BONE_OFFSET_X_MM` | Shifts ONLY the tibia bone in X (on top of group offset) |
| `TIBIA_CB_OFFSET_X_MM` | Shifts tibial cutting block in X (on top of group offset) |
| `TIBIA_CB_OFFSET_Y_MM` | Shifts tibial cutting block in Y |
| `TIBIA_PLATE_OFFSET_X_MM` | Shifts tibial plate in X (on top of group offset) |
| `TIBIA_PLATE_OFFSET_Y_MM` | Shifts tibial plate in Y |
| `TIBIA_INSERT_OFFSET_X_MM` | Shifts tibial insert in X (on top of group offset) |
| `TIBIA_INSERT_OFFSET_Y_MM` | Shifts tibial insert in Y |

**Y cascade:** When `FEMUR_IMPLANT_OFFSET_Y_MM` is non-zero, the entire tibial group (all components and the tibia bone) shifts by the same amount in Y, keeping the tibial group directly below the femoral implant. X offsets do not cascade.

### Posterior slope and other

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIBIA_POSTERIOR_SLOPE_DEG` | 5.0 | Posterior slope angle (positive = posterior-down). Applied to plate, insert, and both CBs. The bone is NOT rotated — the angled CB creates the sloped cut. |
| `REGIONAL_BBOX_FRACTION` | 0.25 | Fraction of bone length used for articular-region measurement |
| `CONTACT_AVG_N` | 10 | Number of vertices averaged for condyle/plateau contact detection |

### Step-by-step demo mode

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STEP_BY_STEP_MODE` | `"F"` | `"T"` = pause at each major step, `"F"` = run continuously |
| `STEP_PAUSE_SECONDS` | 3.0 | Duration of each pause in seconds |

### Batch processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_MODE` | `"F"` | `"T"` = process all patient folders in `BATCH_DIR` |
| `BATCH_DIR` | — | Path to folder containing patient subfolders |

---

## Size Chart CSV

The CSV file (`SizeChart.csv`) lives in `SCALED_STLS_DIR` and contains 12 rows (S1 through L4). All values are **positive**.

| Column | Unit | Description |
|--------|------|-------------|
| `SIZE` | — | Size label (S1, S2, ... L4) |
| `femur_ML_mm` | mm | Femur mediolateral dimension |
| `femur_AP_mm` | mm | Femur anteroposterior dimension |
| `tibia_ML_mm` | mm | Tibia mediolateral dimension |
| `tibia_AP_mm` | mm | Tibia anteroposterior dimension |
| `patella_d_mm` | mm | Patellar component diameter |
| `insert_thin_mm` | mm | Thin tibial insert thickness |
| `insert_thick_mm` | mm | Thick tibial insert thickness |
| `femur_distal_cut_m` | m | Distal resection depth |
| `femur_origin_z_m` | m | Z offset from femoral STL origin to cut surface (0 if origin is at the cut surface) |
| `tibia_proximal_cut_m` | m | Proximal resection depth |
| `tibia_origin_z_m` | m | Z offset from tibial STL origin to cut surface (0 if origin is at the cut surface) |

**Native origins:** All femoral component STLs (CB, CB shell, implant) share the same CAD origin. All tibial component STLs (CB, CB shell, plate, insert) share their own CAD origin. When positioned at the same world coordinates, they align automatically. The `origin_z` columns only matter if the CAD origin is not at the cut/mating surface.

---

## Coordinate System

After PCA and TEA alignment, the world axes are:

```
    X = AP (anteroposterior)
    Y = ML (mediolateral, TEA aligned here)
    Z = SI (superior-inferior, bone shaft pointing +Z)
```

Z = 0 is the femoral condyle contact plane ("joint line"). All components are positioned relative to this reference.

```
+Z  (proximal / superior)
 │
 │  Femur bone            (condyles at Z=0, tilted by valgus angle)
 │  Femoral components    (CB, shell, implant — shared native origin)
 │       ── FEMUR_TIBIA_GAP ──
 │  Tibial components     (plate, insert, CBs — shared native origin)
 │  Tibia bone            (positioned so resection depth is correct)
 │
-Z  (distal / inferior)
```

---

## Bone Alignment Pipeline

### PCA shaft alignment

After import and scale baking, the script computes PCA (principal component analysis) on the bone's vertex cloud. The eigenvector with the largest eigenvalue corresponds to the bone shaft. The bone is rotated so this axis aligns with +Z. This replaces the hardcoded `import_z_rotation` from v1/v2 and works for any scan orientation.

### TEA detection

After PCA alignment, the script isolates vertices in the epicondylar height band (15–35% above the condyle tips) and performs 2D PCA on their XY coordinates. The direction of maximum spread is the transepicondylar axis (TEA). The bone is rotated around Z so the TEA aligns with the Y axis, then an additional correction of `TEA_CORRECTION_DEG` (default 3°) is applied for the surgical external rotation.

In **retained mode** (`"R"`), TEA is detected once from the femur and the same rotation is applied to both bones. In **independent mode** (`"I"`), the femur uses TEA and the tibia uses a manual Z rotation.

---

## Femur Pipeline — `run_femur()`

1. **Import and align** — PCA shaft to Z, TEA to Y, XY-centre, cut to working length
2. **Validate orientation** — checks that the distal end is wider than the proximal shaft
3. **Measure** — regional bounding box of the distal 25% gives ML and AP
4. **Size selection** — round-up or round-down from the size chart, or manual override
5. **Condyle contact → Z=0** — shifts the bone so the condyle contact plane is at world Z=0
6. **Valgus rotation** — tilts around X (AP axis) by the valgus angle
7. **Import components** — CB, CB shell, and implant imported with native origins at the same world position (automatic alignment)
8. **Booleans** — DIFFERENCE on the bone, INTERSECT on the shell duplicate
9. **Export** — STL files to the Output folder; cutting blocks and shells hidden from viewport

---

## Tibia Pipeline — `run_tibia()`

1. **Import and align** — same PCA alignment; in retained mode, uses the femur's TEA rotation
2. **Measure** — regional bounding box of the proximal 25%
3. **Size cross-check** — warns if tibia measurement would select a different size
4. **Compute Z references** — plate top Z derived from femoral implant bottom minus gap
5. **Import components** — plate, insert, CB, CB shell imported with native origins at the same world position
6. **Position tibia bone** — top of bone placed at `bone_cut_z + proximal_cut` above the cutting surface, with group and bone-only X offsets applied
7. **Posterior slope** — plate, insert, and both CBs rotated around Y (ML axis) by the slope angle; the bone is left unrotated so the angled CB creates the sloped cut
8. **Booleans** — same as femur
9. **Export** — STL files to Output; cutting blocks and shells hidden

---

## Size Comparison Mode

When `COMPARE_SIZE = "T"`, after the primary pipeline completes:

1. All primary objects are moved into a Blender collection named `Primary_{size}` (e.g. `Primary_L1`)
2. The full pipeline runs again from scratch — fresh bone imports, PCA, TEA, measurements, components, booleans — at one size smaller (e.g. `M4`), with all objects shifted +200mm in Y
3. The comparison objects are moved into `Comparison_{size}` (e.g. `Comparison_M4`)

Both collections are visible in Blender's Outliner. Toggle either collection's eye icon to compare the two sizes. The comparison set includes complete bones with boolean cuts, not just components — it is a full visual replica at the alternate size.

No STL files are exported for the comparison set.

---

## Batch Processing

Set `BATCH_MODE = "T"` and `BATCH_DIR` to a folder containing patient subfolders. Each subfolder should contain `Left/` and/or `Right/` directories with the bone STLs.

The script processes all patients alphabetically. For each patient and side:
- A clean Blender scene is created
- The full pipeline runs
- Output STLs and a log file are written to `{patient_folder}/{side}/Output/`
- If a patient fails, the error is logged and the script continues to the next

---

## Log Files

Every run generates a timestamped text file in the Output folder (e.g. `tka_log_20260401_143000.txt`) that captures all console output: measurements, size selections, component positions, warnings, and export paths. This serves as a protocol record for each patient.

---

## Troubleshooting

**Components misaligned despite native origins**
The component STLs don't share a CAD origin. Verify by importing two components into Blender manually without running the script — they should overlap perfectly. If not, the STLs were exported with different origins in the CAD software.

**PCA aligns shaft to wrong axis**
The bone mesh is very asymmetric or has segmentation artifacts. Clean the mesh in 3D Slicer or Blender and retry. As a workaround, switch to `BONE_IMPORT_MODE = "I"` and use `TIBIA_MANUAL_Z_ROT_DEG` for manual control.

**TEA detection finds wrong points**
The epicondylar detection band (15–35% above condyle tips) may not match the bone anatomy. Adjust the band percentages in `detect_tea_angle()` or use manual offsets to compensate.

**Sizing too large or too small**
Toggle `SIZE_ROUND` between `"U"` (round up) and `"D"` (round down). Noisy meshes inflate the bounding box — try `"D"`. Clean meshes may undersize — try `"U"`. Use `MANUAL_SIZE` to force a specific size.

**Components rotated 90° relative to bone**
Adjust `FEMUR_COMPONENT_Z_ROT` / `TIBIA_COMPONENT_Z_ROT`. After PCA+TEA alignment the convention is X=AP, Y=ML. If your CAD exports use Y=AP, X=ML, add or subtract 90°.

**Boolean result empty or corrupt**
The mesh is non-manifold. The script tries EXACT solver first, then falls back to FAST automatically. If both fail, clean the mesh in Edit Mode (Select > Select All by Trait > Non Manifold, then Mesh > Clean Up > Merge by Distance).

**Hidden objects can't be unhidden**
The script uses `hide_set(True)`, which is toggleable via the eye icon in Blender's Outliner. If the eye icons are greyed out, make sure you are in the correct view layer and that the object's collection is not also hidden.

**Step-by-step mode doesn't refresh viewport**
This only works in Blender GUI mode, not headless (`--background`).

**Tibia bone not moving with tibial group**
The tibia bone's XY position is derived from the tibial component anchor (which includes group offsets and the femoral Y cascade). If the bone appears offset from the components, check `TIBIA_BONE_OFFSET_X_MM` and `TIBIA_GROUP_OFFSET_X_MM`.

---

## Script Structure Reference

| Section | Content |
|---------|---------|
| 1 | User configuration: paths, filenames, surgical parameters, offsets, modes |
| 2 | Batch processing configuration |
| 3 | Internal mm → m conversion |
| 4 | Size table loader (CSV parser) |
| 5 | Low-level helpers: scene management, import, transform, boolean, export, hide, step-pause |
| 6 | Component import (native origin mode) |
| 7 | PCA shaft alignment |
| 8 | TEA detection and Z rotation |
| 9 | Bounding box and measurement utilities |
| 10 | Bone cutting and orientation validation |
| 11 | Posterior slope |
| 12 | Bone preparation (import → PCA → centre → cut) |
| 13 | Size selection (round-up / round-down) |
| 14 | Femur pipeline |
| 15 | Tibia pipeline |
| 16 | Logging (Tee class) |
| 17 | Entry points: single patient, comparison, batch |
