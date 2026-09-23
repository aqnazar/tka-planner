# Building the patient's implant in Fusion

The planner measures the implant each patient needs and writes it to
`implant_spec.json`, beside `plan.json`, from `tka measure` or from the application's
export. These two Fusion scripts put those measurements into the parametric implant
model.

## Once per design: write the mapping

1. In Fusion, open the implant design. Go to Utilities > Scripts and Add-Ins, and under
   Scripts click **+** and add the folder `fusion/TKA_DumpParameters`.
2. Run **TKA_DumpParameters** and save the JSON. It lists every user parameter, and
   every model parameter someone has named, with its current expression.
3. Copy `parameter_map.example.json` next to the design and set each `"fusion"` name to
   the design's own parameter name. Remove any entry the design does not drive.
   `offset_mm` and `scale` adjust a value whose CAD datum differs from the planner's; the
   example keeps the tray 1 mm inside the bone outline.

## Per patient: build the implant

1. Preview what will be set, without Fusion:

   ```bash
   tka cad-params --spec out/implant_spec.json --map parameter_map.json
   ```

2. In Fusion, open the design and run **TKA_ApplyImplantSpec** (added from
   `fusion/TKA_ApplyImplantSpec`). Pick the patient's `implant_spec.json` and the mapping.
   It sets every parameter it can and lists any it could not. If the mapping names an
   `export_stl_to` folder, it then exports the bodies there as STL, next to the
   specification.

The script imports `tka_planner.cad_bridge` from this repository, so it must stay in
`fusion/` inside the checkout. The preview and the script use the same code.

## What the specification contains

All values are in millimetres, in each component's own CAD axes, with the origin where the
plan seats the component:

| Component | Axes | Dimensions |
|---|---|---|
| Femoral component | +X anterior, +Y patient-left, +Z proximal | `ml`, `ap_distal`, `ap_overall`, `medial_condyle_width`, `lateral_condyle_width`, `medial_condyle_ap`, `lateral_condyle_ap`, `notch_width`, `distal_thickness`, `flange_height`, and the distal cut's outline |
| Tibial tray | +X patient-left, +Y posterior, +Z proximal | `ml`, `ap`, `medial_ap`, `lateral_ap`, `pcl_notch_width`, `pcl_notch_depth`, `tray_thickness`, and the tibial cut's outline |
| Insert | as the tray | `thickness_mm`: the polyethylene under the condyle that closes the joint in extension with no gap |

Each outline is 96 points evenly spaced along the cut's boundary. A tray drawn from it
covers the cut exactly, where a scaled catalogue tray cannot follow an asymmetric tibia.
