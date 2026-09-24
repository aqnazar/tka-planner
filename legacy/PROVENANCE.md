# Legacy pipeline — provenance record

`TKA_Script_v3.py` is a **byte-identical** copy of the pipeline used to produce the
results in the first published paper on the custom parametric TKA implant system.

```
sha256  8e0bb5feb62e22657b4743f4b4f208b7e6f7d74e66b20ccca76d5a03ec624fee
frozen  2026-07-26
lines   1435
```

## Do not modify this file

Its entire value is being exactly what generated the published figures. Bug fixes,
refactors and improvements belong in `tka_planner/`, never here. If you find a defect
in this script, record it below rather than fixing it — a defect that was present when
the results were produced is part of the provenance.

The new package does **not** reproduce this script's behaviour. An earlier version of
this record said a `tka_planner.core.legacy_estimator` module re-implemented it with
parity tests against archived run logs; no such module or tests exist in the
repository or its history (checked 2026-09-23). Reproducing the published results
therefore means running this script, unmodified, in Blender with the configuration
below. The only shared behaviour is the discrete sizing rule, which
`tka_planner.core.sizing.select_discrete_size` implements with the same round-down
default as `SIZE_ROUND = "D"`.

## Configuration used for the published results

The script carries its configuration as module-level globals in Section 1. The values
as frozen:

| Parameter | Value |
|---|---|
| `BONE_IMPORT_MODE` | `"R"` (retained — both bones share the femur's rotation) |
| `USE_PCA` | `"F"` |
| `USE_TEA` | `"F"` |
| `BONE_MANUAL_Z_ROT_DEG` | `90.0` |
| `SIZE_MODE` | `"ML"` |
| `SIZE_ROUND` | `"D"` (round down) |
| `MANUAL_SIZE` | `""` (automatic) |
| `FEMUR_TIBIA_GAP_MM` | `20.0` |
| `BONE_CUT_LENGTH_MM` | `150.0` |
| `VALGUS_ANGLE_DEG` | `6.0` |
| `TEA_CORRECTION_DEG` | `3.0` |
| `TIBIA_POSTERIOR_SLOPE_DEG` | `5.0` |
| `REGIONAL_BBOX_FRACTION` | `0.25` |
| `CONTACT_AVG_N` | `10` |
| `FEMUR_COMPONENT_Z_ROT` | `0` |
| `TIBIA_COMPONENT_Z_ROT` | `90` |
| all manual XY offsets | `0.0` |

Note that `USE_PCA` and `USE_TEA` are both off: the PCA shaft alignment and TEA
detection were implemented but not trusted in practice, and a fixed 90° Z rotation was
used instead. This is the observation that motivated the landmark-driven redesign.

## Known defects, recorded rather than fixed

These are present in the frozen script and were present when the published results
were generated. They are listed so their effect on those results can be assessed.

1. **`find_condyle_contact` (line 616) mislabels medial and lateral on one side.**
   It splits at the bounding-box midline and calls the lower half "medial"
   unconditionally. After the 90° rotation `+Y` is patient-left, which is lateral on a
   left knee and medial on a right one. *Effect on published results: none.* The
   function immediately takes `min(med_z, lat_z)` and discards the labels, so only the
   contact height is used. The defect matters only for per-compartment measurements,
   which the legacy script never computes.

2. **Femur falls back to a tibial parameter (line 822).** When `USE_TEA == "F"` and no
   rotation is supplied, the femur uses `TIBIA_MANUAL_Z_ROT_DEG`. *Effect: none*, as
   that value is `0.0`.

3. **`cut_bone_to_length` axis inconsistency (line 655).** The cutter's X scale is
   derived from `bbox["ML"]` while the script's stated convention is X = AP.
   *Effect: none*, as both dimensions are inflated by 200 mm and the cutter is
   oversized either way.

4. **Size clamping is silent (lines 761-765).** A bone exceeding the largest chart size
   is clamped with only a printed warning. *Effect: real.* Case 005 measures 86.4 mm
   femoral ML against a largest size of 84.0 mm and was clamped to L4.

5. ~~`SizeChart.csv` column `tibia_proximal_cut_m` is implausible as a resection
   depth.~~ **Not a defect (confirmed by the lead author, 2026-09-23).** The
   column runs 16.3–22.1 mm because it is measured from the **most proximal point of the
   tibia** (usually the intercondylar eminence), not from a plateau; the script places
   the tibia's top that far above the cut (line 1100). It is exactly
   `0.2625 x femur_ML_mm` across all twelve rows. The new planner keeps this datum as its
   "top of tibia" option for the like-for-like comparison, and defaults to the commercial
   convention of 9 mm below the less affected plateau.
