"""Writes the femoral (v57 working copy) parameter decision list for the author.

Private: CAD design data. Source: live dump of the Parametric Work copy, 2026-09-27.
Outputs, next to the spec:
  ../femoral_parameter_decisions.md   read-through, grouped, with a Decision column
  ../femoral_parameter_decisions.csv  the same rows for Excel (UTF-8 with BOM)
"""

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]

# Kinds: M measured on the bone, D derived from other parameters, C design constant,
# L linked to another parameter, X delete or ignore, ? needs the author.
# Cohort ranges are what the stage (b) rule gives on the 10 knees + reference.
GROUPS = [
    ("A", "Sizing inputs (user parameters)", "", [
        ("Femur_ML", "82 mm", "ML of the component. About 65 dimensions are written (Femur_ML / 82) × value, so it scales almost everything, including the box and the arcs the rules measure separately.", "M: component ML from the bone (stage c).", "Cohort 80.0–93.2. Decide which dimensions keep ML scaling (see K-11)."),
        ("Femur_AP", "74 mm", "Only the depth of the Anterior profile cut extrude along X.", "D: ≥ box AP + margin, or a large constant.", "The name says sizing, but it only sets how far the flange-outline cut reaches."),
        ("External_rotation", "3 deg", "Nothing: no dependents.", "X: delete.", "Your ruling: a construction aid from the upright reference bone."),
    ]),
    ("B", "Cut box: J-curve sketch (XZ, sagittal)", "The five inner cuts. The origin is the centre of the distal face.", [
        ("C_Distal_cut_surface", "(Femur_ML/82) × 26.5", "Distal face length.", "M (stage b): box AP × 0.538.", "Cohort 25.6–31.4. The driver sets it in mm."),
        ("B_Posterior_chamfer_projection", "(Femur_ML/82) × 10.6", "Posterior chamfer run (= rise at 45°). Posterior cut X = −(C/2 + B).", "M (stage b): box AP × 0.214.", "Cohort 10.2–12.5."),
        ("D_Anterior_chamfer_projection", "(Femur_ML/82) × 15", "Anterior chamfer run and rise. The anterior face starts at (C/2 + D, D).", "M (stage b): from the anterior cut position.", "Cohort 14.3–17.6."),
        ("A_Posterior_cut_surface", "(Femur_ML/82) × 16", "Posterior face height (Z from B to B + A). R1 passes through its top point, so A sets where the posterior arc starts.", "D: from R1 and the posterior cut (spec), or M: posterior condyle height above the chamfer.", "Tied to blocker 1 (posterior thickness)."),
        ("F_Anterior_cut_surface", "(Femur_ML/82) × 45", "Length of the anterior face line (reaches Z ≈ 59).", "D: ≥ (flange height − D) / cos 10.11° + margin.", "It must reach past the flange top or the flange is cut short. The cohort flange goes up to 51."),
        ("Distal_cut_depth", "(Femur_ML/82) × 9", "Distal thickness: R2 is tangent to the line this far below the distal face.", "C: 9 mm.", "It sets the distal resection. The insert's Distal_Cut (9) should link to it. ML-scaled today (10.2 at ML 93)."),
        ("Anterior_cut_depth", "(Femur_ML/82) × 5", "Flange thickness: R3 is tangent to the line this far in front of the anterior face.", "C: 5 mm (the rule's flange thickness).", "ML-scaled today (5.7 at ML 93) while the planner assumes 5.0."),
        ("Posterior_construction_line", "(Femur_ML/82) × 9", "Construction line 9 mm behind the posterior face. Nothing is tangent to it.", "X, or C: it could become the posterior-thickness constraint (K-3).", "Appears unused."),
        ("Anterior_construction_line", "(Femur_ML/82) × 65", "Length of the construction line R3 touches (offset of the anterior face).", "C: long enough.", "It only has to be long enough for the tangency."),
        ("Anterior_chamfer_angle", "135 deg", "Actually dimensions the POSTERIOR chamfer.", "C: 45°.", "Names swapped (§6 item 2): rename."),
        ("Posterior_chamfer_angle", "135 deg", "Actually dimensions the ANTERIOR chamfer.", "C: 45°.", "Rename."),
        ("Anterior_cut_angle", "145 deg", "Anterior face 10° beyond vertical (measured flare 10.11°).", "C.", "Angles fixed by your ruling."),
        ("E_Pin_length", "(Femur_ML/82) × 16", "Construction line on the distal face; drives Pin_Length.", "C: 16 mm.", "Belongs to the pins; ML-scaled today."),
    ]),
    ("C", "Articular arcs (user parameters, J-curve)", "R1 passes through the posterior face top and is tangent to R2; R2 is tangent to the distal-thickness line and to R3; R3 is tangent to the flange-thickness line.", [
        ("R1_Circle_Diameter", "(Femur_ML/82) × 33", "Posterior condyle arc.", "M (stage d): fitted to each condyle's posterior sagittal profile.", "BLOCKER 1: with fixed radii the posterior thickness holds only while box AP ≤ 49.4 × ML/82 (REF 3.3 mm, P029R 3.7)."),
        ("R2_Circle_Diameter", "(Femur_ML/82) × 71", "Distal arc.", "M (stage d): distal condyle profile.", ""),
        ("R3_Circle_Diameter", "(Femur_ML/82) × 210", "Trochlear arc joining R2 to the flange.", "M (stage d): trochlear profile.", ""),
        ("J_Curve_Offset_FIXED", "3 (%)", "R1 and R2 of the articulation path are 3 % larger.", "C.", "Articulation clearance against the insert."),
        ("J_Curve_Offset_Coefficient", "30 (%)", "R3 and the anterior lines of the articulation path are 30 % larger.", "C.", ""),
        ("R1_Diameter, R2_Diameter, R3_Diameter", "= R*_Circle_Diameter", "Sketch dimensions.", "L (already linked).", "Nothing to do."),
    ]),
    ("D", "J-curve offset sketch (XZ): path for the two sweep cuts", "Hidden bodies Offset and Posterior Offset are extruded from it; the sweeps that shape the articular surface run along their edges.", [
        ("d156", "(Femur_ML/82) × 26.5", "Copy of C.", "L: = C_Distal_cut_surface.", "Unlinked in the CAD. The driver links it per run; make it permanent (K-2)."),
        ("d168", "(Femur_ML/82) × 10.6", "Copy of B.", "L: = B_Posterior_chamfer_projection.", "Same."),
        ("d159", "(Femur_ML/82) × 15", "Copy of D.", "L: = D_Anterior_chamfer_projection.", "Same."),
        ("d160", "(Femur_ML/82) × 16", "Copy of A.", "L: = A_Posterior_cut_surface.", ""),
        ("d161", "(Femur_ML/82) × 16", "Copy of E_Pin_length.", "L.", ""),
        ("d164", "(Femur_ML/82) × 9", "Copy of Posterior_construction_line.", "L.", ""),
        ("d165", "(Femur_ML/82) × 65", "Copy of Anterior_construction_line.", "L.", ""),
        ("d162", "(Femur_ML/82) × 9.2", "Offset line below the distal face (9.2, not 9).", "L + C: Distal_cut_depth + 0.2?", "Confirm the 0.2 mm is deliberate clearance."),
        ("d163", "(Femur_ML/82) × 6 × 1.30", "Anterior offset line (7.8; the original is 5).", "?: Anterior_cut_depth-based?", "Built on 6, not 5: confirm."),
        ("d166", "(Femur_ML/82) × 45 × 1.30", "Anterior line length.", "L: F_Anterior_cut_surface × 1.3.", ""),
        ("d157, d158, d167", "135 / 135 / 145 deg", "Copies of the three angles.", "L: = originals.", ""),
        ("R1/R2/R3_Offset_Diameter", "R*_Circle_Diameter × (1 + offset %)", "Articulation arcs.", "L (already).", ""),
        ("d187, d188", "H_Pin_base_diameter / 3, / 2", "Pin-related dimensions inside this sketch.", "?", "Why do pin dimensions live in the articulation-path sketch?"),
        ("d171 (Extrude7)", "20 mm", "Width of hidden body Offset; the anterior sweep follows its edge at Y = +5.", "C.", "Confirm the sweep paths at Y +5 (anterior) and −10 (posterior) are intended."),
        ("d218 (Extrude14)", "10 mm", "Width of hidden body Posterior Offset; the posterior sweep follows its edge at Y = −10.", "C.", ""),
    ]),
    ("E", "ML extent", "", [
        ("Half_ML_distance", "(−Femur_ML) / 2", "Symmetric extrude of J-curve base: the component's ML.", "L: follows Femur_ML.", "Check: bone overhang ≤ 2 mm (stage c)."),
    ]),
    ("F", "Intercondylar notch (XY sketch on the distal plane)", "The sketch has 5 projected, linked edges of the J-curve base.", [
        ("Intercondylar_notch_width", "(Femur_ML/82) × 28", "Notch width (Y ±14).", "M (stage c): notch width on the distal-cut section (planner: 26.0 on REF).", ""),
        ("Notch_Offset_X", "(Femur_ML/82) × 2", "Notch front: 2 mm in front of the distal face's anterior edge.", "M or C: notch depth from the bone.", ""),
        ("Offset", "(Femur_ML/82) × 20", "Notch back line 20 mm behind the posterior cut (opens the notch through the back).", "C: large enough to open.", ""),
        ("Notch_fillet_diameters", "(Femur_ML/82) × 20", "Circles rounding the notch's posterior corners.", "D: scale with notch width (spec).", ""),
        ("Fillet_center_offset", "(Femur_ML/82) × 22", "X of those circles (22 behind the origin).", "D.", ""),
        ("Notch_offset_Y", "(Femur_ML/82) × 10", "Circle centres 10 mm outside the notch side.", "D.", ""),
        ("Notch_Fillet_Radius", "(Femur_ML/82) × 10", "Fillet on the notch's two front vertical edges.", "D: scale with notch width.", ""),
    ]),
    ("G", "Condylar profile (coronal, YZ): the section swept to make the articular surface", "A fitted spline through 5 points, symmetric medial/lateral: outer (Y ±45, Z +4), condyle apex (±26, −9), inner (±10, −6). Two projected ML edges.", [
        ("d41", "(Femur_ML/82) × 10", "Half-width of the central groove part (inner points at Y ±10).", "M (stage g): trochlear groove width.", ""),
        ("d47, d48", "(Femur_ML/82) × 16", "Inner point to condyle apex (apex at Y ±26).", "M (stage g): condyle centres from the distal section.", "Medial = lateral today; asymmetry needs a split (approach C)."),
        ("d54", "(Femur_ML/82) × 3", "Apex 3 mm below the inner points (coronal groove depth).", "M (stage g).", ""),
        ("d45", "(Femur_ML/82) × 19", "+Y apex to outer point, horizontal.", "D: from the ML edge.", ""),
        ("d39", "(Femur_ML/82) × 13", "−Y apex to outer point, vertical.", "C or M.", "The two sides are dimensioned differently (d45 horizontal, d39 vertical)."),
        ("d38", "(Femur_ML/82) × 24", "Outer edge line of the tool region (Z +4 to −20).", "C (tool closure).", ""),
        ("d51", "(Femur_ML/82) × 20", "Closing line 20 mm below the origin.", "C.", ""),
        ("d120", "(Femur_ML/82) × 7.3", "Where the projected ML edge meets the profile: 7.3 mm below the outer point.", "D.", ""),
        ("d129", "(Femur_ML/82) × 14", "Vertical 14 at the outer edge.", "?", "Role unclear from the geometry."),
        ("d125, d126", "75 deg", "Spline tangent angle at the inner points.", "C.", ""),
        ("d136", "(Femur_ML/82) × 18", "Tangent handle length at the inner point.", "C.", ""),
        ("d131", "(Femur_ML/82) × 17", "Horizontal tangent handle at the condyle apex.", "C.", ""),
        ("d127, d128", "50 deg", "Tangent angle at the outer points.", "C.", ""),
        ("d130, d137", "(Femur_ML/82) × 20", "Tangent handle lengths at the outer points.", "C.", ""),
    ]),
    ("H", "Anterior profile (flange outline, YZ; cut along +X with a 4° taper)", "A fitted spline, NOT fully constrained. Main Body is the right-knee part: the higher peak is on −Y (lateral on a right knee).", [
        ("d148", "(Femur_ML/82) × 44", "Height of the higher (lateral) flange peak. Flange top on the body ≈ d148 − X × tan 4° (42.6).", "D: = box flange height + taper allowance (stage b rule: flush at the trochlea top).", "BLOCKER 2: the flange top isn't driven by the box (REF 42.9 vs 34; P029R 41.4 vs 28)."),
        ("d147", "(Femur_ML/82) × 16", "Y of the higher (lateral) peak.", "M (stage g): lateral ridge position.", ""),
        ("d149", "(Femur_ML/82) × 37", "Height of the lower (medial) peak.", "M (stage g), or D: d148 − 7.", ""),
        ("d150", "(Femur_ML/82) × 7", "Y of the lower (medial) peak.", "M (stage g).", ""),
        ("d141", "(Femur_ML/82) × 37", "Width between the two bottom points (Z −15).", "C.", ""),
        ("d142", "(Femur_ML/82) × 24", "Side edge at the ML edge (Z −11.7 to 12.3).", "D: follows ML.", ""),
        ("d143, d144", "18 mm / 65 deg", "Tangent handle at the lateral peak.", "C.", ""),
        ("d145, d146", "18 mm / 78 deg", "Tangent handle at the medial peak.", "C.", ""),
        ("Anterior_Profile_Cut_Boundary", "Femur_ML × 2", "Circle enclosing the part (the cut region is circle minus outline).", "D (fine).", ""),
        ("Anterior_profile_cut", "= Femur_AP", "Extrude depth of the outline cut.", "D: ≥ box AP + margin.", ""),
        ("d23", "4 deg", "Taper of the outline cut (bevels the flange edge; lowers its top with X).", "C.", ""),
        ("d189, d190", "0 mm / 0 deg", "Cont of Anterior Profile (misc): extrude to an entity.", "C.", "Failed on P029L/R ('No target body')."),
    ]),
    ("I", "Posterior profile cut (YZ)", "Trims the tops of the posterior condyles and the intercondylar area. 5 projected, linked body edges, including a fixed spline: the most fragile sketch.", [
        ("d124", "(Femur_ML/82) × 12", "Intercondylar top 12 mm below the condyle top (Z 14.4 vs 26.4).", "D: from B + A (the posterior face top).", "Anchored to projected edges today."),
        ("d222", "10 mm", "Tool overhang past the ML edge.", "C.", ""),
    ]),
    ("J", "Pins", "Pins at Y ±25 on the distal face.", [
        ("Pin_Spacing", "(Femur_ML/82) × 50", "Distance between the pins.", "M: condyle centres (spec), or keep ML-scaled.", ""),
        ("Pin_dist_to_center", "Pin_Spacing / 2", "", "D.", ""),
        ("H_Pin_base_diameter", "(Femur_ML/82) × 9", "Pin base Ø.", "C: 9 mm?", "ML-scaled pins change the drill size per patient."),
        ("Pin_End_Profile", "H_Pin_base_diameter × 2/3", "Pin tip Ø.", "D.", ""),
        ("Pin_Length", "= E_Pin_length", "16 mm.", "C.", ""),
        ("Pin_Fillet_Radius", "2 mm", "", "C.", ""),
    ]),
    ("K", "Fillets", "", [
        ("Fillet8 radius (d224)", "1 mm", "11 edges: the notch perimeter (Y ±14) and the posterior profile cut edges.", "C: 1 mm.", "Edge selection breaks: failed on P029L, P047L/R; cached geometry elsewhere (P009 came out broken)."),
        ("Fillet9 radius (d226)", "1 mm", "9 edges: distal-face and notch edges (Z 0 to 2).", "C: 1 mm.", "Same problem."),
        ("Tangency weights (d223, d225, Pin_, Notch_Fillet_Tangency)", "1", "", "C.", ""),
    ]),
    ("L", "Unused or tool-only", "", [
        ("Sketch15 (d204)", "−10 mm", "Offset curves of 3 free splines. Not used by any feature.", "X: delete?", ""),
        ("Sketch18 (d230)", "10 deg", "Two construction lines and a free spline. Not used by any feature.", "X: delete?", ""),
        ("Sweep distances d69, d70, d75, d76", "1 / 0", "Full-path sweeps.", "C.", ""),
        ("Tapers d17, d20, d21, d118, d172, d219", "0 deg", "", "C.", ""),
    ]),
]

CHECKS = [
    ("Every timeline item is healthy: no warnings, no cached geometry.", "Driver records it (Fusion health)."),
    ("Main Body Y extent is ±Femur_ML/2 (symmetric).", "Page flags 'body broken'."),
    ("Inner faces lie on the planner's box: posterior face, both chamfers, distal face, anterior face (≤ 0.05 mm).", "To add: numeric check per face."),
    ("Distal thickness = Distal_cut_depth (body min Z).", "To add."),
    ("Posterior condyle thickness, each side (the rule assumes 5.77).", "Page shows it."),
    ("Flange thickness = Anterior_cut_depth.", "To add."),
    ("Flange top vs the box's flange height.", "Page shows it."),
    ("Notch width and depth vs the bone.", "Stage (c)."),
    ("ML overhang ≤ 2 mm over the cut surfaces.", "Stage (c)."),
    ("Pins at the condyle centres, not breaking through the articular surface.", "To add."),
    ("Left-knee body (Main Body (1), from Mirror2) is the exact mirror of Main Body, fillets included.", "FAILS today: the mirror comes before Fillet8/9 (51 mm³ apart)."),
    ("Insert (xref) regenerates against the new femoral and still articulates.", "Stage (f)."),
]

FIXES = [
    ("K-1", "Mirror order", "Move Mirror2 after Fillet8/9 (or mirror the filleted body), so the left-knee part matches. Decide which body is exported for each side. Until then the page mirrors Main Body for left knees."),
    ("K-2", "Link the offset copies", "Make the links in group D permanent (d156 = C, d168 = B, d159 = D, …)."),
    ("K-3", "Arcs vs box (blocker 1)", "Choose: (a) R1–R3 measured on the bone (stage d, pulled forward); (b) R1–R3 scale with box AP / 49.43 instead of ML; (c) constrain R1 tangent to a line offset from the posterior face by the posterior thickness and let R1's radius follow."),
    ("K-4", "Flange top (blocker 2)", "Fully constrain the Anterior profile spline and drive d148 (and d149) from a new user parameter, Flange_height, with the 4° taper compensated."),
    ("K-5", "Posterior profile cut", "Rebuild its sketch from parameters: no projected body edges, no fixed spline."),
    ("K-6", "Fillet8 / Fillet9", "Make the edge selection survive regeneration (select by face or rule, or rebuild where edge identity persists)."),
    ("K-7", "Other projections", "Intercondylar notch (5) and Condylar profile (2) projected edges hold today; replace them with parameter references for robustness."),
    ("K-8", "Free splines", "Constrain the Anterior profile spline; delete Sketch15 and Sketch18 if unused."),
    ("K-9", "Names", "Swap the chamfer angle names."),
    ("K-10", "Dead parameter", "Delete External_rotation."),
    ("K-11", "What scales with ML", "Decide which dimensions stay ML-scaled and which become constants: thicknesses (9, 5), pins (Ø 9, 16, spacing), fillets, tool sizes."),
    ("K-12", "Insert link", "Link the insert's Distal_Cut to the femoral Distal_cut_depth (§3.3)."),
]


def main():
    md = ["# Femoral component (v57 working copy): parameter decisions",
          "",
          "Private design data. Live dump of `Robot TKA/Parametric Work/Knee Prosthesis (Femoral)` v1, 2026-09-27: "
          "111 parameters, 10 sketches, 27 timeline features. Frame: +X anterior, +Y patient-left, +Z proximal, "
          "origin at the centre of the distal face. Values are at Femur_ML = 82.",
          "",
          "**How to use it:** fill in the Decision column (or the same column in `femoral_parameter_decisions.csv`), "
          "and I'll turn it into formulas in the working copy and rules in the planner.",
          "",
          "**Kinds:** M measured on the bone · D derived from other parameters · C design constant · "
          "L linked to another parameter · X delete or ignore · ? needs you.",
          "",
          "**The three blockers from the driver run:** 1. posterior thickness (C, K-3) · 2. flange top (H, K-4) · "
          "3. CAD robustness (I, K, K-1, K-5, K-6).",
          "",
          "## 1. Parameters: what should drive each one",
          ""]
    rows = []
    for gid, title, note, items in GROUPS:
        md += [f"### {gid}. {title}", ""]
        if note:
            md += [note, ""]
        md += ["| # | Parameter | Current | What it controls | Proposed driver | Note | Decision |",
               "|---|---|---|---|---|---|---|"]
        for i, (name, cur, what, prop, notes) in enumerate(items, 1):
            pid = f"{gid}{i}"
            md.append(f"| {pid} | `{name}` | {cur} | {what} | {prop} | {notes} | |")
            rows.append([pid, title, name, cur, what, prop, notes, ""])
        md.append("")
    md += ["## 2. Checks after every regeneration", "",
           "| # | Check | Status |", "|---|---|---|"]
    for i, (c, s) in enumerate(CHECKS, 1):
        md.append(f"| CH-{i} | {c} | {s} |")
        rows.append([f"CH-{i}", "Check", c, "", "", "", s, ""])
    md += ["", "## 3. Constraints and CAD fixes", "",
           "| # | Item | What | Decision |", "|---|---|---|---|"]
    for k, item, what in FIXES:
        md.append(f"| {k} | {item} | {what} | |")
        rows.append([k, "CAD fix", item, "", what, "", "", ""])
    (OUT / "femoral_parameter_decisions.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    full = OUT / "femoral_parameter_decisions.csv"
    if full.exists():
        # The author fills these in: never overwrite one.
        print(full.name, "exists, left as it is")
        return
    with open(full, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Group", "Parameter / item", "Current", "What it controls",
                    "Proposed driver", "Note", "Decision"])
        w.writerows(rows)
    print(len(rows), "rows")


# ---- The short list: only what needs the author's decision or check -------------
# id -> (priority, what I need from you). Priorities sort the file.
P1 = "1 Blocker: posterior thickness"
P2 = "2 Blocker: flange top"
P3 = "3 Blocker: CAD robustness"
P4 = "4 Stage (b) now"
P5 = "5 Constant or ML-scaled (K-11)"
P6 = "6 Stage (c): ML and notch"
P7 = "7 Stage (d)/(g): later"
P8 = "8 Cleanup"
NEEDED = {
    "C1": (P1, "Pick the K-3 option (a, b or c); R1 is where it bites."),
    "C2": (P1, "Same K-3 choice: does R2 follow the box, the bone, or ML?"),
    "C3": (P1, "Same K-3 choice for R3."),
    "B4": (P1, "Rule for A: derived from R1, measured on the bone, or keep ML-scaled?"),
    "H1": (P2, "Approve a Flange_height parameter driving d148 (K-4)."),
    "H3": (P2, "Medial peak: measured on the bone, or tied to d148 (d148 − 7)?"),
    "H12": (P3, "This extrude fails on P029L/R: is it needed, and what should it end on?"),
    "I1": (P3, "Anchor to B + A instead of projected edges (with K-5)?"),
    "B1": (P4, "Confirm: set in mm by the box rule (0.538 of box AP)."),
    "B2": (P4, "Confirm: set in mm by the box rule (0.214 of box AP)."),
    "B3": (P4, "Confirm: set in mm from the anterior cut position."),
    "B5": (P4, "Approve F derived from the flange height so the face always reaches past it."),
    "B8": (P4, "Delete, or reuse as the posterior-thickness constraint (K-3 option c)?"),
    "A2": (P4, "Femur_AP: derived from box AP + margin, or a large constant?"),
    "H10": (P4, "Follows A2 (it is = Femur_AP): nothing extra unless you want it separate."),
    "D8": (P4, "Is the 0.2 mm (9.2 vs 9) deliberate? Link as Distal_cut_depth + 0.2?"),
    "D9": (P4, "Why 6 (× 1.3) and not 5? Link to Anterior_cut_depth?"),
    "D13": (P4, "Why are pin dimensions in the articulation-path sketch? Keep or move?"),
    "D14": (P4, "Are the sweep paths at Y +5 (anterior) and −10 (posterior) intended? (d218 is the second body.)"),
    "B6": (P5, "Distal thickness: constant 9, or scale with ML (10.2 at ML 93)?"),
    "B7": (P5, "Flange thickness: constant 5 (as the planner assumes), or scale with ML?"),
    "B13": (P5, "Pin length: constant 16, or scale with ML?"),
    "J1": (P5, "Pin spacing: condyle centres from the bone, ML-scaled, or constant?"),
    "J3": (P5, "Pin Ø: constant 9 (instrument-compatible), or scale with ML?"),
    "K1": (P5, "Fillet radius: constant 1 mm?"),
    "K2": (P5, "Fillet radius: constant 1 mm?"),
    "A1": (P6, "Approve the ML rule: widest bone section over the cut faces − margin (start 0 mm, overhang ≤ 2 mm)."),
    "F1": (P6, "Approve the notch-width rule (distal-cut section)."),
    "F2": (P6, "Notch depth: measured from the bone, or constant 2 mm?"),
    "F4": (P6, "Notch corner circles scale with notch width (also F5, F6)?"),
    "F7": (P6, "Notch front fillet scales with notch width?"),
    "G1": (P7, "Groove width: measured (stage g), or keep ML-scaled for now?"),
    "G2": (P7, "Condyle centres: measured; separate medial/lateral if the cohort differs?"),
    "G3": (P7, "Coronal groove depth: measured, or constant?"),
    "G4": (P7, "Outer point from the ML edge: OK?"),
    "G5": (P7, "d39 is vertical on one side while d45 is horizontal on the other: intended?"),
    "G9": (P7, "What does d129 (14 mm, vertical at the outer edge) do?"),
    "H2": (P7, "Lateral ridge position: measured (stage g)?"),
    "H4": (P7, "Medial ridge position: measured (stage g)?"),
    "H6": (P7, "Flange side edge follows ML: OK?"),
    "A3": (P8, "Delete External_rotation (K-10)?"),
    "L1": (P8, "Delete Sketch15 (unused)?"),
    "L2": (P8, "Delete Sketch18 (unused)?"),
}
FIX_NEEDED = {
    "K-1": (P3, "Mirror2, Fillet8, Fillet9", "Move Mirror2 after the fillets? Which body is exported for left knees?"),
    "K-2": (P4, "J-curve offset", "Approve permanent links: d156=C, d168=B, d159=D, d160=A, d161=E_Pin_length, d164, d165, d166=F×1.3, d157/d158/d167=angles."),
    "K-3": (P1, "J-curve (R1, R2, R3)", "Pick (a) arcs measured on the bone, (b) arcs scale with box AP, or (c) R1 tangent to a thickness line with its radius following."),
    "K-4": (P2, "Anterior profile", "Approve: constrain the spline, add Flange_height driving d148 (and d149), with the taper compensated."),
    "K-5": (P3, "Posterior profile cut", "Rebuild from parameters (no projected edges, no fixed spline): you, or me with your OK to save a copy?"),
    "K-6": (P3, "Fillet8, Fillet9", "How to select the edges so they survive regeneration: by face, by rule, or rebuild?"),
    "K-7": (P8, "Intercondylar notch, Condylar profile", "Replace the projected edges with parameter references now, or later?"),
    "K-8": (P2, "Anterior profile, Sketch15, Sketch18", "Constrain the flange spline; delete the unused sketches?"),
    "K-9": (P8, "J-curve", "Swap the chamfer angle names?"),
    "K-11": (P5, "(many)", "The overall policy: which lengths stay ML-scaled (see the P5 rows)."),
    "K-12": (P8, "Tibial Insert New", "Link the insert's Distal_Cut to the femoral Distal_cut_depth?"),
}


def needed_csv():
    import json
    import re
    snap = json.loads((Path(__file__).resolve().parent / "fusion_snapshots"
                       / "base_sandbox_femoral.json").read_text(encoding="utf-8"))
    owner = {k: v["owner"] or "user parameter" for k, v in snap["params"].items()}
    acts_in = {"Femur_ML": "user parameter, drives almost every sketch",
               "Femur_AP": "user parameter, drives Anterior profile cut",
               "External_rotation": "user parameter, drives nothing",
               "R1_Circle_Diameter": "user parameter, drives J-curve + J-curve offset",
               "R2_Circle_Diameter": "user parameter, drives J-curve + J-curve offset",
               "R3_Circle_Diameter": "user parameter, drives J-curve + J-curve offset",
               "Pin_Spacing": "user parameter, drives Pins"}
    owner.update(acts_in)
    rows = []
    for gid, title, note, items in GROUPS:
        for i, (name, cur, what, prop, notes) in enumerate(items, 1):
            pid = f"{gid}{i}"
            if pid not in NEEDED:
                continue
            found = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", name) if t in owner]
            where = ", ".join(sorted({owner[t] for t in found})) or title
            prio, ask = NEEDED[pid]
            rows.append([prio, pid, where, name, cur, what, prop, ask, notes, ""])
    for k, item, what in FIXES:
        if k in FIX_NEEDED:
            prio, where, ask = FIX_NEEDED[k]
            rows.append([prio, k, where, "CAD fix: " + item, "", what, "", ask, "", ""])
    natural = lambda i: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", i)]
    rows.sort(key=lambda r: (r[0], natural(r[1])))
    out = OUT / "femoral_parameter_decisions_needed.csv"
    if out.exists():
        print(out.name, "exists, left as it is (the author may have filled it in)")
        return
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Priority", "ID", "Sketch / feature", "Parameter", "Current",
                    "What it controls", "Proposed driver", "What I need from you",
                    "Note", "Decision"])
        w.writerows(rows)
    print(len(rows), "rows need a decision ->", out.name)


if __name__ == "__main__":
    main()
    needed_csv()
