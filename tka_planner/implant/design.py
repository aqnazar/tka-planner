"""The design numbers the implant rules read, in one place.

Each is a number of the parametric Fusion designs the rules drive, as the evaluation
scripts in ``eval/`` publish them (``tray_rule.py``, ``implant_pipeline.py``,
``fit_check.py``, ``demo_cases.py``); none is new here. The box lengths and the
component's own proportions are read from the library's ``femoral_box.json`` as before.
"""

# The femoral component sits this far inside the bone's cut surfaces, each side, and
# the tray this far inside the tibial cut outline (author rulings 2026-10-09).
FEMORAL_INSET_MM = 1.0
TRAY_INSET_MM = 1.0
# The femoral component is Femur_ML - 1 wide (design v19), centred on its origin.
FEMORAL_WIDTH_ALLOWANCE_MM = 1.0

# Tray rule raster and design (plate design v2 / v29 Base Sketch).
TRAY_GRID_MM = 0.25            # raster for the containment and coverage checks
TRAY_NOTCH_EDGE_MM = 3.0       # posterior edge kept beside the PCL notch
TRAY_AP_EXTRA_MM = 4.0         # Tibial_Plateau_AP = condyle diameter + 4
TRAY_DESIGN = {"ml": 78.0, "ap": 44.0, "notch_w": 18.0 * 78 / 82, "notch_d": 6.0}
TRAY_DESIGN_FEMUR_ML_MM = 82.0  # the notch width scales with Femur_ML / 82
TRAY_FILLET_MM = 8.0           # notch-corner fillet d11 = 8 x AP / 44
TRAY_OVERHANG_TOLERANCE_MM2 = 0.5

# The insert document: the plate's insert floor at Plate_Thickness, the femoral origin
# at floor + Insert_Thickness + Distal_Cut (author 2026-10-09).
PLATE_FLOOR_MM = 5.0
# Femoral v20: Distal_cut_depth = 9 mm x Sagittal_Scale, Sagittal_Scale = (C + B + D)
# over the L3 design's box.
DISTAL_CUT_DESIGN_MM = 9.0
DESIGN_BOX_MM = 26.5 + 10.6 + 15.0
# Below this the eval derivation flags the insert (its own report; the planner's
# default minimum is the plan control insert_minimum_mm).
INSERT_FLAG_MM = 6.0

# The anterior cut's flare in the sagittal facet planes of the fit check (fit_check.py).
FLARE_DEG = 10.1

# The femoral parameters the box gives (fusion_driver.BOX_PARAMETERS).
BOX_PARAMETERS = ("C_Distal_cut_surface", "B_Posterior_chamfer_projection",
                  "D_Anterior_chamfer_projection")
