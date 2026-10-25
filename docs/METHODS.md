# Methods

The decisions behind the numbers, in enough detail to reproduce or challenge them. Each
records what is done, why, and what it replaced — several were arrived at only after the
obvious approach produced output that looked plausible and was wrong.

> Research and demonstration only. Not a medical device.

---

## Coordinate system

Everything works in the **LPS patient frame** the CT segmentation already provides:
`+X` patient-left, `+Y` posterior, `+Z` superior. Millimetres throughout the core;
Blender scenes are built in metres, converted at exactly one boundary
(`tka_planner/blender/io.py`).

Preserving that frame matters more than it sounds. Verified across all twenty meshes in
the cohort: left knees occupy X ∈ [+21, +137] mm and right knees X ∈ [−135, −25], with no
overlap — so the axes already carry anatomical meaning before any landmark exists.
Femur and tibia of the same knee are also already registered to each other, abutting at
the real joint line.

The legacy pipeline discarded both, recentring each bone on its own bounding box, then
spent PCA, a bounding-box heuristic and a fixed 90° rotation reconstructing what it had
thrown away — plus a fixed 20 mm gap standing in for the joint relationship the data
already contained.

## Anatomical frames

Axes are named for meaning, never for their letter: `x_anterior`, `y_patient_left`,
`z_proximal`, right-handed. *Patient-left* is a fixed spatial direction; *medial* and
*lateral* are not, since medial points toward the midline and so flips between knees.
Frames therefore carry a `Side` and delegate every medial/lateral question to it, which
is why no metric contains a side conditional.

Each frame's proximal axis follows an ordered ladder, and the first satisfiable method
wins:

| Bone | Method | Requires | Quality |
|---|---|---|---|
| Femur | `frames.femur.mechanical.v1` | femoral head centre | measured |
| Femur | `frames.femur.ama_assumed.v1` | diaphyseal axis + assumed mechanical-anatomical angle | estimated |
| Tibia | `frames.tibia.mechanical.v1` | ankle centre | measured |
| Tibia | `frames.tibia.anatomical_proxy.v1` | proximal diaphyseal axis | estimated |

The asymmetry is real and worth stating: the tibial mechanical and anatomical axes are
near-collinear, so the tibial fallback is a good estimate. The femoral pair differ by
5–7° with genuine between-patient variation, so the femoral fallback is a population
assumption. **MPTA degrades gracefully on a knee-only scan; mLDFA does not.**

Rotational reference: the surgical epicondylar axis where the medial sulcus has been
picked, otherwise the anatomical axis, with the substitution recorded. The sulcus is a
depression rather than a surface extreme, so no automatic estimator can find it.

**Left and right come from the landmarks' names.** The epicondylar axis is oriented
patient-left by the side (lateral is patient-left on a left knee), and the tibial frame
checks its Akagi line against the spines' labels. The femoral frame used to orient the
epicondylar axis towards scanner +X, which turned the frame 180° on the reference knee
tilted 30° in the scanner; scanner +X is now consulted only when those landmarks are
missing, and a bone lying along it raises `FrameConstructionError`, a refusal, not a bare
`ValueError`.

## Provenance

Every reported value carries a tier:

- **measured** — every input came from the patient's anatomy
- **estimated** — an assumption stood in; it travels with the value and is printed in the report
- **not computable** — the value is `None`, never a plausible guess, and carries what is missing and what would supply it

Quality never improves through composition, and it accounts for **how the landmarks
themselves were obtained**. A value computed from machine-estimated points is not a
measurement however sound the arithmetic, so any metric touching an estimated landmark is
demoted and carries `AUTOMATIC_LANDMARK_ESTIMATE`.

The legacy `VALGUS_ANGLE_DEG = 6.0` is now `POPULATION_FEMORAL_AMA` — same number, same
result, but named, sourced, attached to the metrics it affects, and overridable per
patient.

`TEA_CORRECTION_DEG = 3.0` was deleted. Its docstring conflated two angles: the ~3°
figure in the literature is the condylar twist between the posterior condylar line and
the surgical epicondylar axis, not the difference between the surgical and anatomical
epicondylar axes. With real landmarks it is measured, not assumed.

## Measurement

**Femoral ML** — extent of the distal block along the frame's mediolateral axis (and AP
along its anterior axis), between the 0.1st and 99.9th percentiles of the block's vertices.
Taken in the frame's axes rather than the world's, since a femur lies at whatever
rotation the scanner found it in. Not the bounding box: a box is the distance between
two vertices, so surface noise only widens it. On the cohort, 0.5 mm of vertex noise
added 2.1–3.1 mm to the box and moved the nearest size on 5 of 11 knees; the percentile
extent grew 0.4–0.7 mm and moved none. On clean meshes it reads 0.4–0.8 mm narrower than
the box (`measure.femur.frame_aligned_distal_extent.v2`). No knee's nearest size changes;
the round-down chart size of P029L and P029R, whose boxes were 80.49 and 80.02 mm against
the 80 mm L2 threshold, goes from L2 to L1.

**Tibial AP** — the anteroposterior extent of the **resection cross-section**, cut a few
millimetres below the articular surface. Not a bounding box: a box round the proximal
tibia is measured to the intercondylar eminence above and the tubercle in front, and on
this cohort overstates AP by up to **25.9 mm** (74.5 against a true 48.6 on one case) —
enough on its own to throw sizing out of the chart.

Locating the articular surface excludes the central intercondylar region first, because
the highest point of a proximal tibia belongs to the spines, not the plateau.

## Sizing

The implant is a single master geometry under a uniform scale — every femoral implant
STL shares the same aspect ratio to four decimal places (1 : 1.1788 : 0.8350 at both S1
and L3). The twelve chart sizes are samples of that family, published discretely because
that is how implants are supplied.

Sizing therefore solves for a **continuous size parameter**. It reproduces every chart
row exactly, so legacy parity is provable, and keeps working outside the chart — which
matters, because a real case measures 86.4 mm against a largest chart size of 84.0 and
the legacy pipeline clamped it with only a printed warning.

Not every column is parametric, and the model does not pretend otherwise:

| Behaviour | Columns |
|---|---|
| Exactly linear | `femur_ML`, `femur_AP`, `tibia_AP`, both resection columns |
| Irregular, hand-adjusted | `tibia_ML` (2 and 4 mm steps), `patella_d` |
| Genuinely quantised | `insert_thin`, `insert_thick` — inserts come in fixed thicknesses |

So the chart is interpolated rather than fitted to a law, and quantised columns are
rounded.

**Resection depths are parametric.** `femur_distal_cut` and `tibia_proximal_cut` are
exactly `0.1125 ×` and `0.2625 ×` `femur_ML` across all twelve rows. Each chart size has
its own depth, and a continuous size takes the interpolated depth, so there is no fixed
default. The depth sets how far below its datum a cut sits; the alignment philosophy
sets the cut's angle; the surgeon's resection controls add to the depth.

**The tibial depth follows the commercial convention.** Every technique checked
(Triathlon, ATTUNE, Persona) sets the stylus on the lowest point of the **less affected
plateau** and removes the thinnest tibial construct: 9 mm in Triathlon and ATTUNE (4 mm
base + 5 mm insert), 10 mm in Persona. That is the default here, 9 mm. The alternative
stylus setting, 2 mm below the **more affected** plateau, is also offered. The third
option measures the size chart's `tibia_proximal_cut` from the **top of the tibia**
(usually the eminence), the legacy pipeline's datum, kept for the like-for-like
comparison. Measured that way the cut is 13–17 mm below the plateaus on case P009,
deeper than any commercial default. The surgeon's resection delta moves any of them.

The femoral depth is the component's distal thickness, measured from the more distal
condyle (measured resection): 9.0 mm at L2.

The plan records the reference, the datum point, the depth from it, and the depth
below each plateau.

## The insert and the gap

The plan keeps the bones where they were scanned and closes the construct with **no
gap**. The implant is parametric, so the insert is solved rather than chosen from a
catalogue: in extension, the space between the cut planes is measured normal to the
tibial cut at each compartment. The insert is made exactly as thick as closes the
tighter compartment, after the femoral component's distal thickness and the tray (from
the implant library). The other compartment's gap is the imbalance. With the coronally
parallel cuts of measured resection the bony space is rectangular, so the imbalance is
zero from bone alone; soft tissue is not modelled. The surgeon can choose a thicker or
thinner insert, which tightens or loosens both compartments equally.

Until Fusion builds the insert, the viewer stretches the library insert along its own
axis about its floor, so that its dish (the polyethylene under the condyle) is the solved
thickness. The library's dish is 4.6 mm at S1 and 7.0 mm at L4, on a 3.0–5.0 mm tray.

The gap is seen in **Reduce** mode. A distraction pulls the tibia away from the femur
along its own axis, a rigid pose that does not touch the plan, and it composes with
flexion, varus/valgus stress and the drawer.

**Component register and collision.** The femoral component and the tray are designed in
register: imported together they sit square on each other, and only the height between
them changes. So the femur leads. The tray takes the femoral component's rotation, seen
in its own cut plane, and sits directly under the femoral origin along its cut normal.
The tibial rotation and slides are the surgeon's deliberate departure from register, and
the plan reports it as the femoral component's rotation over the tray and its anterior
and lateral offset (zero by default). Where the tray then sits against the tibia's own
axis and the centre of its cut is reported separately, as the coverage question. With the
library loaded, the posed femoral component and the stretched insert are compared surface
to surface, and a collision is flagged with its location. Placing each component from its
own bone, as the planner did before 2026-09-26, put P009's components 9.8° and 8.9 mm out
of register, with the femoral component 5 mm into the insert's anterior lip.

## Alignment planning

Both cuts are built from **one reference direction and one hinge**, so they share a
mediolateral slope exactly and differ only in the anteroposterior angle.

This is a requirement, not a convenience: the components articulate, so a mediolateral
disagreement is an error the construct absorbs. Deriving each cut from its own estimated
mechanical axis gave a 4.4° disagreement on a real case — error from canal segments only
~82 mm apart on a 174 mm femur, not anatomy. The disagreement is now measured and
reported as a warning.

**Femoral flexion is measured from the femur's own axis.** Zero flexion is a distal cut
square to the femoral frame's proximal axis seen from the side: the mechanical axis where
the femoral head was measured, otherwise the distal diaphyseal axis (the assumed AMA
turns it only in the coronal plane), which is the reference an intramedullary femoral
guide sets. Component flexion and tibial slope are surgical parameters, set explicitly
from the target and each measured on its own bone. The hinge, the coronal and per-cut
varus rotations and the shared-slope check all use the femoral frame's anterior axis.

Until 2026-10-14 the reference's sagittal tilt was stripped against the **scanner's**
anterior axis instead, on the argument that a diaphyseal axis fitted to a supine,
slightly flexed leg leans forwards (~4.3° on P009) only because of how the leg lay. But
that lean is the femur's, and the cut is made on the femur: with the landmarks carried
through a sagittal tilt of the scan, the femoral cut stayed put in the scanner and the
component's flexion *on the femur* moved by exactly the tilt (E2), while the tibial cut
followed its bone. The old construction is kept as
`femoral_sagittal_reference="scanner"` for comparison, and the plan records which was
used.

A positive slope drops the posterior edge of the cut, tilting the plane normal
posteriorly. Inverting this produced cuts sloping forwards.

Slope is measured from the plane **perpendicular to the tibial axis**, the same
reference the native slope metric uses, and the usual clinical convention. It used to be
applied from the scanner's axial plane, so "reproduce the native slope" missed by however
far the tibia leaned in the scanner: 1.8° on case P009. The tibial axis is first brought
into the sagittal plane of the shared reference, which makes it a rotation of that
reference about the shared hinge, so the two cuts still share their mediolateral slope
exactly. The plan records the lean and the slope as it would read against the scanner.

Cut planes are centred on the **actual cross-section the plane makes through the bone**.
The midpoint of two compartment landmarks is about a centimetre off on the tibia, because
the deepest point of each plateau is neither centred nor symmetric.

Resection depth is reported per compartment. The femur is cut from below and the tibia
from above, so removed bone lies on opposite sides of the two planes and one sign
convention applied to both reports the tibial depths negative.

Two philosophies, and no more:

- **mechanical** — perpendicular to the mechanical axis, targeting a neutral limb
- **kinematic** — parallel to the native joint surfaces, restoring the patient's own obliquity

Kinematic is the one a knee-only scan supports without assumption, since it references
only local joint geometry.

## Component placement

The library uses two axis conventions, a quarter turn apart:

| Parts | Local axes |
|---|---|
| Femoral cutting block, its shell, femoral implant | `+X` anterior, `+Y` patient-left, `+Z` proximal |
| Tibial cutting block, its shell, insert, tibial plate | `+X` patient-left, `+Y` posterior, `+Z` proximal |

Both right-handed. Assuming one convention for the whole library leaves the tibial parts
rotated 90°, so each is mapped explicitly and the pose builder asserts right-handedness —
a bad mapping raises rather than silently mirroring a part.

Placement uses the **native CAD origin**, untouched. Every part belonging to a bone
shares one origin sitting on the cut surface, which is exactly why a cutting block and
its implant coincide when given the same pose. Re-originning to the bounding box is wrong
for these shapes — a femoral component wraps the distal femur, so its extremes are the
anterior flange and posterior condyles while the mating surface sits between them; it put
the component 42 mm below its cut and 32 mm to the side.

Femoral component rotation is set off the **posterior condylar axis with 3° external
rotation**, as in theatre — not off the frame's epicondylar axis, which serves the
coronal construction and differs by the condylar twist angle. External means the
component's anterior turns toward the patient's lateral side, so the posterior cut takes
more off the medial condyle. It was once taken as "toward the epicondylar axis", which
agrees only while that axis sits externally of the condylar line. On three knees it sits
internally (twist −4.9°, −0.6° and −2.3°), and the 3° went internal there. It is now
tested on both sides at twists from −6° to +6°.

## The patient-specific implant

The implant is parametric in every dimension, so the planner does not choose an implant:
it measures the one this patient needs, on the planned cuts, and writes it to
`implant_spec.json` for the CAD model (see `fusion/README.md`). All values are in each
component's own CAD axes, with the origin where the plan seats the component.

- **Femoral component**, at the distal cut: overall ML; distal AP; the width and AP of
  each condyle, and the intercondylar notch between them, found as the two lobes across
  the posterior third of the section; the AP depth from the posterior condyles to the
  anterior cortex over the height of the anterior flange; the distal thickness; and the
  outline of the cut.
- **Tibial tray**, at the tibial cut: ML; AP overall and through the middle of each
  compartment (the tibia is asymmetric); the PCL notch, found as the deepest dip in the
  back edge with bone either side of it, wherever it lies; the tray thickness; and the
  outline.
- **Insert**: the solved thickness.

Outlines are traced along the section's boundary and resampled to 96 evenly spaced
points. Casting rays from the centroid fails on the distal femur, whose centroid falls in
the notch, off the bone.

On P009: femoral ML 76.5 mm, AP 74.5 mm, condyles 29.0 and 29.3 mm wide with an 18.3 mm
notch; tray ML 81.3 mm, AP 65.0 mm, medial 50.5 mm and lateral 61.0 mm.

**The femoral cut box.** The femoral component sits on five cuts. The alignment plans
the distal cut, and the other four are placed on the bone in the component's frame
(`core/femoral_box.py`). The design's angles are fixed and only lengths change: the
anterior cut keeps the design's flare from the component axis, the chamfers keep their
angle, and the posterior cut stays parallel to the axis.

- **Posterior cut.** Placed by posterior referencing: the component's posterior-condyle
  thickness comes off the more prominent condyle, and the other condyle loses less,
  depending on the planned rotation.
- **Anterior cut.** Placed flush at the top of the trochlea:
  1. The most anterior bone within the component's width is traced at each height up the
     femur, as the anterior profile.
  2. A line is fitted to that profile over the shaft, 60–100 mm above the distal cut.
  3. The trochlea is what stands proud of that line below the shaft. It fades into the
     shaft smoothly, without an edge. So its top is taken as the first height above its
     highest point where it has fallen to a tenth of that height (or 0.25 mm, whichever
     is more) and stays there for 5 mm. Measured against the trochlea's own height, a
     millimetre-scale ridge isn't lost in a tenth of a millimetre of noise. Requiring
     5 mm means one noisy band can't end it early.

  The flared cut crosses the cortex at that height, and the flange ends there. So the
  cortex above the flange isn't notched, and there's no gap under the flange's tip. The
  deepest bone the cut removes below the flange is the anterior resection. That depth,
  minus the flange's thickness, is reported as the patellofemoral lowering against the
  native trochlea. Both are measured along the component's AP axis. The profile's excess
  over the shaft line is kept in the specification for review. A trochlea that is still
  fading within 5 mm of the shaft window is flagged. The design file is validated when
  it's read, and a malformed one is reported as the reason the box wasn't placed.
- **Chamfers.** The box between the two cuts is divided between the distal face and the
  chamfers in the design's proportions. The component's CAD origin is the centre of the
  distal face.

The design's angles and thicknesses are the implant's own design data. They're read from
`femoral_box.json` in the implant library and aren't part of this repository.

The specification carries the box, the lengths it gives the CAD model, and the resection
table: distal, posterior medial and lateral, anterior, and tibial medial and lateral. A
scan that ends less than 100 mm above the distal cut can't show the shaft, so the box
isn't placed there, and the specification says why.

**Display and fit until the CAD model builds it.** In the default *patient-specific*
mode, the library parts are scaled per axis to the patient's width and depth (keeping the
size's scale along the component axis) and slid in their own plane onto the centre of the
cut. The surgeon's slide is applied on top of that. *Catalogue* mode keeps the library
part at the size's single scale, as the legacy pipeline did. On P009 tibial coverage rises
from 73% to 93% and femoral from 68% to 73%. What remains is shape rather than size: a
scaled symmetric tray cannot follow an asymmetric tibia, and the outline in the
specification is what lets the CAD model do so.

## Component fit

Fit is judged on the cut surface, where the component meets the bone. At each cut the
**bone section** (the bone mesh sliced by the cut plane) is compared with the **component
footprint** (the posed component sliced 1 mm off the cut on its own side: through the tray
plate above the tibial cut, and through the distal plate below the distal femoral cut).
The components come from the implant library at the plan's size, posed exactly as the
viewer poses them.

Both outlines are filled on a 0.25 mm grid in the plane by the even-odd rule, so a
hollow section, such as the intercondylar notch, stays hollow. The grid's axes are the
bone's lateral and anterior directions laid into the cut, so the quadrants are anatomical
on both knees. Reported per cut:

- coverage of the cut surface, and the share of the footprint that is on bone;
- overhang area, and in each quadrant the largest **overhang** (footprint beyond the
  bone edge) and **underhang** (bone beyond the footprint edge), as distances in mm;
- the ML and AP extents of both outlines.

An overhang over 2 mm raises a flag. That limit is a convention from the literature on
tibial overhang, not a validated threshold, and the distances are always reported.

On case P009 the femur is sized from the extent of its distal quarter, 89.6 mm ML (90.2
as a bounding box), which the epicondyles set. The bone at the distal cut is 76.5 mm wide, so the solved
component overhangs the distal cut by up to 12 mm. The legacy pipeline sizes the femur
the same way. The fit check exists to expose exactly this kind of mismatch.

## Manual adjustment

Every control on the planning screen is a field in an `Adjustments` record passed *into*
the planner, never an edit applied to its output. An adjusted plan is therefore still a
pure function of `(landmarks, frames, target, adjustments)`, serialises whole, and
re-derives exactly. Nudging objects in the viewport instead would leave the scene showing
geometry the plan file does not describe — which would quietly cost the project its main
claim.

Signs are anatomical rather than spatial: positive is valgus, deeper, more slope, more
external rotation, more anterior, more lateral, on both knees. A signed rotation about a
world axis means the opposite thing on a right knee, and the mirror-invariance tests
cover each control for exactly that reason.

**Varus/valgus rotates the shared reference, before the two cuts are separated.** Both
cuts descend from that one direction, so they move together and go on sharing a
mediolateral slope exactly. Rotating the two finished cuts by the same angle instead
looks identical and is not: once the sagittal angles are applied their hinges differ, and
the slopes drift apart.

Per-cut varus overrides exist and deliberately do break that agreement — so the plan
measures the disagreement and warns, rather than absorbing it silently.

**Extension gaps** are reported per compartment when an insert thickness is set, measured
normal to the tibial cut, with the femoral component, tray and insert subtracted. Normal
to the *tibial* cut because with posterior slope the two cuts are not parallel and there
is no single separation between them; that is the direction the insert stacks in and a
trial spacer enters. The femoral thickness is subtracted as though perpendicular to the
same direction, exact only for parallel cuts, and wrong by about 0.01 mm at 3° of slope.

**No flexion gap is reported.** It needs the posterior condylar resection, which this
pipeline does not plan.

Re-planning costs about 37 ms on a real 2.5-million-vertex pair, which is what makes the
controls follow a drag. Almost all of it is finding the cut cross-section: the anatomy
maths is 1.7 ms. Projecting the vertices as `vertices @ normal` minus a scalar rather
than `(vertices - seat) @ normal` avoids materialising a full copy of the vertex array
and took that step from 56 ms per cut to 17.

**Re-cutting the bone is a different order of cost, and the resection is muted during a
drag because of it.** An exact boolean takes about 6 s for the plane box and 22 s for the
cutting blocks and their shells, against 1 ms with no resection at all. Those numbers
were missed at first because they were measured headless, where Blender never evaluates
a modifier nobody is looking at — the panel appeared to manage 27 updates a second while
doing none of the work. Muting the booleans on the first change and restoring them once
the controls have been still for 0.6 s keeps the planes, axes and components moving at
15 to 26 fps, and those are what alignment is judged on. The same deferral applies to the
first build, so pressing Plan returns in 3 s rather than 55.

**The settle no longer costs more than the change that caused it.** A drag that only
touches the tibia used to also re-bake the femur, because the deferral was scene-wide
rather than per-control: `set_cuts_visible` mutes and restores every boolean it finds,
with no idea which one the control that just moved actually depends on. That question has
one clean answer without hand-mapping every control to an effect: diff the newly computed
plan's resections and component poses against the last one actually cut. If a bone's
resection plane and pose both come back identical, nothing about it changed, and it is
left alone — no mute, no re-solve, no re-bake. Confirmed against all fourteen adjustment
fields in both resection modes headlessly: fields that only reposition a component (AP/ML
shift, in-plane rotation) diff to no change at all in Cut plane mode, where the plane
cutter tracks the resection alone, and to exactly the one bone whose component moved in
Cutting block mode, where the block's pose *is* the component's pose. Insert thickness
diffs to no change in either mode, since it touches neither.

## Resecting with the cutting block

The block and its shell are booleaned against the bone as the first pipeline did: the
bone minus the block, and a copy of the bone intersected with the block's shell, which
gives the patient-specific mating surface a printed guide would sit on. The copy is taken
**before** the bone is resected — a shell cut from an already-resected femur is missing
the condylar surface it is meant to fit.

The plane box and the block are alternatives rather than a stack. The box removes
everything beyond the plane, so it would swallow the very surfaces the block is shaping
and leave the block boolean with nothing to do; a bone cut by its block is therefore
skipped by the plane pass.

**These booleans are always exact, and the panel's solver choice does not reach them.**
The fast solver does not lose a little precision here, it returns confident nonsense: on
the real cohort it reduced a 409k-vertex femur to 4k and produced a "shell" 227 mm across
— larger than the bone it was intersected with — without raising. A cut that looks like a
cut and is not is precisely the failure this project keeps having to catch by measuring.

## Automatic landmark estimation

A complete first pass from two STL files, so a plan can be produced with no manual
picking. **Everything it emits is marked `ESTIMATED`**, never `PRESENT` — an automatic
estimate is a starting position for a human to correct.

**Every search runs in the bone's own axes** (`landmarks_auto.v2`), found from its
surface before any landmark exists, and the landmarks are mapped back to scanner
coordinates. The axes are right-handed, in LPS order, with the origin at the surface's
area-weighted centroid:

| Bone | Superior | Patient-left |
|---|---|---|
| Femur (`bone_axes.femur.shaft_ama.v1`) | the diaphyseal axis, through two canal sections each cut square to the last round's axis, turned medially by the population AMA (6°) in the plane of the posterior condylar line: the axis the femoral frame falls back to on a knee-only scan | the posterior condylar line, converged in the plane square to that axis |
| Tibia (`bone_axes.tibia.shaft.v1`) | the diaphyseal axis, as for the femur, with no turn: the axis the tibial frame uses | the tibia's own posterior condylar line, on the top 22 mm, converged from the plateau's widest direction |

The femur's axes are found on its distal window only (2.75 condylar widths), so a
full-length femur gets the axes of its own knee-only scan. The tibia's mediolateral axis
is its own rather than the femur's so that the tibial frame's rotational reference,
built from landmarks found in these axes, does not inherit the femur's rotation; and the
posterior condyles rather than the plateau's widest direction, which is asymmetric and
lay 18° from both posterior condylar lines on P009L. The refinements read discrete point
sets and so end in a cycle of 0.001–0.04°, reached in three to five rounds.

**The scanner contributes only signs** — which end is superior, which side patient-left —
and those are protected by the orientation gate (see Quality control). Searching along the
scanner's axes, as version 1 did, measured the scan rather than the knee. Metamorphic
testing (paper experiment E2) showed what it cost: turning the leg 10° in the scanner
turned the tibial AP (Akagi) axis 17°, and 30° turned it 46°; a 30° coronal tilt moved a
distal condyle 30–34 mm; and moving the knee 100 mm *without turning it* moved a plateau
low point up to 12.5 mm and changed the posterior slope by up to 1.7°, because the
articular grid below was anchored at the scanner's origin. In the bone's axes a rigid
motion moves every landmark with the bone exactly (synthetic bones: to 1e-6 mm), and a
mirror mirrors them. Before that, an earlier version still took the extremes of the
epicondylar band along the *world* axes: on a femur rotated ~15° that put the rotational
reference **26° out**.

The plateau low points are searched on the **articular surface only**: faces within 55°
of horizontal, the highest in each 2 mm cell seen from above, away from the edge of that
surface and from the intercondylar strip, in the middle 60% of each compartment. The
first version took the lowest vertex of a 22 mm slab at the top of the bone, and that is
always the slab floor on the cortex. On case P009 both landmarks sat exactly 22 mm below
the top, level with each other, so MPTA came out as 90° by construction. It is 84.4°.

Two estimators were ill-posed whatever the axes and were redefined in v2:

- **Notch centre** (`notch_roof.v2`): the roof of the intercondylar notch on the midline
  between the posterior condyles, at its lowest — where the notch roof meets the end of
  the trochlea, the top of the notch as an AP view shows it and where an intramedullary
  rod enters. The roof is flat there for several millimetres from front to back, so the
  point is the centre of the stretch of a 6 mm sagittal strip within 2 mm of its lowest.
  Version 1 took the most proximal point near the median of the condylar block, which is
  always the block's ceiling, 20–40 mm above the notch on a cortex; under 0.1 mm of noise
  it moved 15–43 mm. The distal condyles are now split at the notch rather than at the
  block's median.
- **PCL footprint** (`posterior_intercondylar.v2`): the most posterior point, within
  18 mm of the top, of a 4 mm strip on the midline between the spines. Version 1 searched
  a 32 mm band, whose most posterior points are the backs of the two condyles at its
  edges: on every knee it picked one edge or the other, 16 mm off the midline, and a turn
  could swap them (32 mm apart on the reference knee).

Confidence is recorded per landmark. Condylar and plateau extremes are good; epicondyles
and the notch fair; the tibial rotational references weakest, since a patellar tendon
insertion has no geometric signature to find. Extremes on flat surfaces are sharp across
the surface and loose along it: a distal condyle's height is well defined, its AP
position is not (it moved 3–16 mm along the condyle between v1 and v2 for a 4–6°
change of axis, with its height unchanged within 0.1 mm).

## Flexion

The femur is held fixed and the tibia swings about the **transepicondylar axis** through
the epicondylar midpoint — the standard approximation, and it works because the condyles
are near-circular in the sagittal plane with the epicondyles near those circle centres,
so the tibia rides round them at a nearly constant radius.

Using the posterior condyles puts the axis on the condylar *surface* rather than at its
centre of curvature. The joint then separates as it flexes: closest approach between the
femoral implant and the insert went from 33 mm to 93 mm over 120°. Through the
epicondyles it holds between 5.3 and 9.1 mm.

**Stated simplifications:** the axis is fixed, the screw-home rotation near extension is
not modelled, and the tibia is treated as a rigid body. Rollback and axial rotation are
the prescribed trajectory's job (next section); with the neutral trajectory the tibia
hinges in one plane, as it always has.

**Playback needs no baked copy, because the geometry is committed before it moves.**
This is worth recording as history, because the machinery it replaced was substantial and
the reason for it was real. The first version parented the tibia and its *unapplied*
resection boolean straight to the flexion pivot. Blender does not cache a modifier's
result across a frame change: it re-solves an unapplied boolean in full on every frame it
evaluates, whether or not the cutter moved. Confirmed with a headless repro before
touching any real code — a dense mesh with a *static* cutter cost the same per frame as
one with a moving cutter, both around 11 s, because the tibia's own transform changing
each frame was already enough to invalidate the cached result. A 120-frame animation at
that cost is not a slow animation, it is a hang.

The answer at the time was to apply the plan once to a plain copy of each bone with the
modifiers baked in and removed, parent only that copy to anything that moved, and hide
the live cuttable originals whenever the construct was posed. It worked: per-frame cost
fell from about 11 s to under a millisecond. But it meant two parallel copies of the
anatomy, a visibility handler to swap between them, an `EDIT_ONLY` tagging scheme, and a
selective rebake to keep the copies in step.

None of that survives the Plan / Commit / Reduce split. A committed resection is real
geometry with no modifier behind it, so there is nothing to re-solve on a frame change
and nothing to bake. Flexion is a rigid transform of the committed bones. The two copies,
the handler, the tags and the rebake are all gone, and the cost is the same sub-millisecond
it was after baking.

**Trial reduction is a function of its three control values.** Flexion, a varus/valgus
stress and an AP drawer, composed about a stored pivot whose basis is anatomical: local X
the transepicondylar axis, local Y anterior. The stress composes onto the *already-flexed*
frame rather than the fixed one, matching how a hands-on exam is actually relative to the
tibia's current position; the drawer slides along the pivot's **rest** Y instead, because
"anterior" for that test means the joint's own anterior rather than wherever flexion left
the tibia pointing.

The pose is computed as `pivot @ R @ pivot⁻¹` and *assigned*, never accumulated. That is
the fix for a real bug, made structural. The parented version overwrote the pivot's
rotation with just the requested turn, silently dropping the fixed anatomical basis the
parenting was built against, so "zero" on every trial control no longer returned to the
actual rest pose. It was caught by an isolated parent-chain repro and fixed by composing
onto the basis. In the current model the mistake cannot be expressed: every node stores a
rest transform, a set pose is assigned on top of it, and the world transform is always
`set_pose @ rest`. Returning the controls to zero returns the construct to extension
exactly, and a test asserts it.

## Flexion trajectory and maximum flexion

**A parameterised trajectory, not one curve (WP-D level D1).** At flexion angle θ the
tibial set is first moved in its own rest frame, then flexed about the hinge above:
`pose(θ) = hinge(θ) @ motion(θ)`. The motion is an **axial rotation** of the tibia about
a line parallel to its long axis through a pivot on the flexion axis, whose
mediolateral position is a parameter (positive medial; positive rotation is internal
tibial rotation, the sense of lateral rollback), plus an **AP translation** of the femur
on the tibia at the pivot (positive anterior, so the paradoxical anterior slide is
representable). Both grow linearly with flexion at the rate set by their value at a
reference angle. A point `s` mm medial of the pivot then moves anteriorly by
`t cos φ + s sin φ`, which is how the per-compartment translations the fluoroscopy
papers report are derived. Composing the motion before the flexion makes the rotation
about the tibia's own axis wherever flexion has carried it — flexion about a femoral
axis, rotation about a tibial one, the Grood–Suntay order.

Three scenarios bracket the motion, because the two studies of CR knees in deep flexion
found that implanted knees do not perform the native rollback and vary between knees by
more than their mean (Bellemans 2002: medial contact 1.97 mm anterior, SD 4.46):

| | Scenario | Parameters | Source |
|---|---|---|---|
| S1 | Native: medial pivot, lateral rollback | pivot 26.4 mm medial (the medial condyle's centre, for centres 52.8 mm apart); knots 20° internal rotation, no translation at 120°, and 33.4°, −4.8 mm at 162°; linear between knots and from extension, held beyond 162° | Fitted to published numbers (below) |
| S2 | Observed CR: paradoxical anterior slide | 2.7° internal rotation and 1.12 mm anterior at the joint centre, at 109° | Bellemans 2002 means (medial 1.97, lateral 0.27 mm anterior, 2.7° internal); 109° is Banks 2003's CR mean maximum flexion standing in for Bellemans' own |
| S3 | Neutral hinge | none | Level D0, unchanged |

**S1 is fitted to the numbers read in the sources** (author ruling 2026-10-25), not to a
table. Pinskerova et al. 2004 (JBJS Br 86:925): in the unloaded living knee the medial
flexion facet's centre does not move from 30° to 120°, the lateral one moves about 15 mm
back from 20° to 120°, and the tibia has turned about 20° internally by 110–120°.
Nakagawa et al. 2000 (JBJS Br 82:1199): from 120° to 162° a further 4 mm back medially and
15 mm laterally. With the pivot on the medial centre and the two centres 2d apart, the
lateral one goes back 2d (sin φ₁₂₀ − sin φ₂₀) between 20° and 120°; 20° at 120° and 15 mm
back give 2d = 52.8 mm, so the pivot lies 26.4 mm medial of the axis midpoint. The earlier
placeholder (20 mm) gave 12–14 mm and left the medial centre 0.5–2 mm out. A straight line
from extension to 20° at 120° adds only about 5 mm of lateral rollback by 162°, against
15 observed, so S1 has a second knot: −4.8 mm of translation (4.0 mm at the medial
centre after the rotation) and 33.4° of rotation at 162°, which puts the lateral centre a
further 15 mm back. That 33.4° is derived (the sources give displacements, not the angle)
and holds only as long as the motion is rigid; beyond 162° nothing was measured and the
values are held. The pivot is for a 52.8 mm centre spacing and is not scaled to each knee,
which matters for knees much wider or narrower than the 80–90 mm femurs it was fitted
for. Iwaki, Pinskerova and Freeman 2000 (JBJS Br 82:1189), the cadaver study the
placeholder named, was not used; Hill et al. 2000 (JBJS Br 82:1196), the living-knee MRI
paper in the same issue, has no numbers in its abstract.

S3 composes to exactly the matrix Reduce mode has always produced; a test asserts
equality, not closeness. The scenario is a trial control in the application and every
`plan.json` records all three parameter sets and which one the plan was previewed with.
Both papers measured contact-point migration of a weight-bearing activity; here it is
applied as rigid translation in a passive preview.

**Maximum flexion is searched, not assumed.** The construct is flexed in 5° steps until
two parts first overlap in a way that blocks it, then bisected to 0.5°. Four pairs are
tested with exact `manifold3d` intersections: the insert against the resected femur,
the femoral component against the insert, the femoral component against the resected
tibia, and the resected tibia against the resected femur. The bones are cropped once to
the reach of a flexing knee (in the femoral frame 60 mm distal to 90 mm proximal of the
hinge; in the tibial frame 100 mm distal to 40 mm proximal), cut by the plan's cutting
blocks exactly as Commit cuts them, and simplified with a 0.2 mm tolerance (a million
triangles to about 15 000). On one knee the limits with no simplification differed by at
most one bisection step (0.3°), and a whole search takes under half a second per
scenario once the bones are prepared (about 25 s per knee, once).

The femoral component is *meant* to touch the insert, and a prescribed motion is not
solved from contact, so a hinge that does not run through the centres of this
component's condyles drives them into the dish or lifts them off it. That is the
motion's error, not impingement, and the two are told apart by a stated rule: overlap
of the component and the insert is **impingement** if any of it lies within 2 mm of the
insert's rim (its anterior or posterior lip or its sides, in the insert's own axes);
otherwise it is the **articulation** and does not stop flexion. The deepest articular
overlap over the arc is reported with every search, as a check on how well the motion
follows the parts. Depth is the overlap's thickness measured column by column, not its
overall extent, which on a sloping dish is several times larger. Both the rim margin
and an optional depth limit (articular overlap deeper than the plan's own seating by
more than a tolerance) are settings. The first cohort run used a 1 mm depth limit and
stopped one knee at 35°, deep in the medial dish: the hinge, not the parts, so the
default no longer counts depth. Any other pair blocks at any overlap above
1 mm³. A pair already in contact at extension is a plan problem, not a flexion limit: it
is reported, and set aside only until the sweep first finds it clear.

The output per plan and scenario is the maximum flexion, the pair that stops it, where
it touched (femoral frame, mm from the knee centre) and the envelope across S1–S3. It is
a **geometric** limit: no soft tissue and no thigh–calf contact, so it is expected to
exceed what patients reach.

## Posterior condylar offset

Bellemans et al. (2002) found it the strongest geometric predictor of the flexion a CR
knee reaches. Their measure is radiographic — the maximal thickness of the posterior
condyle projected posteriorly from the tangent to the posterior cortex of the femoral
shaft, on a true lateral film — and it is measured here on the mesh the same way, per
compartment:

* the **shaft axis** runs between the two femoral canal centres (the anatomical axis, so
  the femoral head is not needed);
* the **sagittal plane** is normal to the transepicondylar axis made perpendicular to the
  shaft axis — the projection a true lateral radiograph makes;
* the **posterior cortex tangent** is a straight line fitted to the posterior silhouette
  of the distal shaft (the most posterior surface point of each 2 mm band, which is the
  outline a film shows), then moved posteriorly until it touches the silhouette without
  crossing it. The shaft used is what a lateral knee film shows: from the distal canal
  centre's level, but no more than 100 mm above the knee centre, for 100 mm or up to the
  proximal canal centre. On the full-length reference femur a tangent over the whole
  canal span followed the femoral bow (6 mm off its own fit) and put the offset near
  60 mm. The fit residual is reported, since a bowed femur makes the tangent ambiguous;
* each compartment's **offset** is the largest distance from that tangent, perpendicular
  to it in the sagittal plane, to any point of the condyle distal of the fitted shaft,
  compartments split at the notch centre. The film's single value is the larger of the
  two.

The **planned** offset is the same distance, from the same tangent, to the placed femoral
component's posterior condyles, shaped and posed as the viewer shows it; the change is
planned minus preoperative. It uses no mechanical axis, so a knee-only scan does not
demote it (every scan here holds 147–248 mm of femur, more shaft than a knee film); it is
`measured` when the canal centres, epicondyles and notch were picked and `estimated` when
they were estimated automatically.

## Quality control

Scan coverage is **detected from the geometry, not declared**. Letting an operator tick
"full-limb scan" would eventually compute an HKA from a knee-only CT.

Femoral head detection requires two independent conditions: bone length ≥ 350 mm, and a
robust proximal sphere fit converging under 1.0 mm RMS at a radius of 18–30 mm.

The thresholds were set by measurement, and the measurements were surprising. Forcing the
fit to run on the real knee-only femurs fits the proximal shaft at radius 15–17 mm with
RMS 0.6–1.5 mm — a short band of diaphysis is simply not obviously non-spherical. Two
apparently sensible criteria turn out not to discriminate at all: wrap-around coverage is
*higher* for a shaft (0.87–0.99) than for a head cap (0.59–0.75), and direction isotropy
overlaps badly. Only the residual separates them, and only with the tighter 1.0 mm bound
— at 1.5 mm a thick-shafted headless femur passed every check. The length gate remains
the primary protection, and it is anatomically airtight: the head-to-condyle distance
*is* the femur's length.

Other gates: a mesh in metres raises rather than computes; a left knee must sit at
positive X in LPS; and landmarks must fall within their bone's bounding box, which
catches an LPS/RAS mismatch decisively since it displaces points by twice their distance
from the midline.

**Detached pieces.** A bone mesh is often more than one piece of surface: collapsed
specks, slivers of a neighbouring bone, a cavity wall exported as a closed shell,
occasionally a whole patella. Every landmark search reads the bone's extremes, so a piece
beyond the bone becomes its top or its edge (one cohort tibia had an 80-triangle sliver
37 mm above the plateau, which became the "top of the tibia"); and the kernel cuts every
piece, so a broken speck refuses the cut of an otherwise clean bone (the P007R femur: six
zero-size pieces, 96 triangles, inside the bone). Every piece but the main body that is debris — longest side
under 20 mm, or, for a closed piece, enclosed volume (absolute, so a cavity wall counts
by its size) under 5 cm³ — is removed: a warning if it lay more than 2 mm outside the
bone, where it would have moved an extreme, a note otherwise. An open piece encloses
nothing, so it is judged by its longest side alone. Both limits lie well below the
smallest adult patella (14 × 23 × 25 mm: Muhamed 2017, mean less two SD of the thinnest
cohort; Kim 2009, smallest height and width in 713 female TKA knees). A larger piece may
be anatomy and is kept, with a warning saying where it
lies (inside the bone's extent, a cavity; outside, a patella, fibula or neighbouring
bone); `--remove-islands`, or the finding's Remove button in the app, removes every piece
but the main body. This replaces the earlier rule (pieces under 1 % of the triangles and
more than 2 mm outside) at the author's ruling (2026-10-15), and the first form of the new
rule (a piece inside the smallest patella's extents along all three of its own axes) at
the author's ruling of 2026-10-25, which kept a thin plate or long sliver however light. On the cohort it changes
one knee's measurement: P007L's femur carried an 8-triangle speck of zero size 0.39 mm
below the bone, which the earlier rule kept (it was within 2 mm) and which had been taken
as the distal medial condyle; measured from the condyle itself, mLDFA and aLDFA move by
0.47° and JLCA by 0.45°. The P007R specks move nothing measurably (femoral AP by
0.0002 mm); the other nine knees are identical.

**Repair before refusal.** A bone that is not a closed surface after that is repaired
when the case is opened, exactly as the kernel will repair it (see *The boolean kernel*),
and refused there (`MESH_UNREPAIRABLE`, an error) only if repair cannot close it — rather
than accepted and then failing at the first cut, as P007R did.

**An entered femur length.** A knee-only scan cannot give the femur's length, so it may
be entered (`--femur-length`, or the field beside the case in the app): the hip-centre to
knee-centre distance from a long-leg film or scanogram. It must lie within 300–650 mm
(`FEMUR_LENGTH_IMPLAUSIBLE` otherwise), is recorded in `plan.json` under
`inputs.entered`, and is never reported as measured. A functional hip centre (camera
pivoting) uses it twice: the fit's predicted centre uncertainty becomes this patient's
mechanical-axis uncertainty, atan(sd / L), instead of one on a nominal 430 mm femur; and
the pivoted length (hip centre to knee centre) is compared with it — more than 10 mm apart
is flagged, because pelvic motion moves the fitted centre without raising the residual,
much of it along the femur, and an independent length is the check that can see it. The
10 mm is a proposal pending the phantom.

**The phantom femur.** With a length entered, the app and `plan.json` (`phantom_femur`)
also carry a schematic of the femur the scan does not hold, as if the whole bone had been
scanned: a head sphere, a neck and a shaft, in the scan's own frame (method
`phantom_femur.v1`, quality `assumed`, never `measured`). It changes no metric: the head
is placed at the entered length along the femoral frame's own mechanical axis. The shaft
axis leans laterally from it by the anatomical-mechanical angle (6.0°; 5.7 ± 1.2, 6.46 ±
1.26, 6.0–6.2 in three measured series), and the neck meets the shaft at 123.2° (3D CT of
247 patients, Yang and Qian 2025), so its length follows from the offset (L sin 6°, 43 mm
at 410 mm) and not from a second source. The head's diameter is the sex-specific mean when
the sex is given (49.65 male, 44.41 female) and otherwise 0.100 of the femur's maximum
length (49/464 and 44/434 in 170 dry femora, Mall 2000), the maximum length being the
entered one plus 40 mm. That 40 mm is this project's own estimate; no published conversion
between hip-to-knee and maximum length was found, and it is to be calibrated on full-femur
scans before the paper prints it. The entered length is also checked against the femoral
width (FL = 187.8 + 3.39 EB, SEE 16.4 mm, 50 dry femora, Abledu 2016) and flagged beyond
two SEE, as a check for a typing error or the wrong length definition; the planner's ML is
not the dry-bone epicondylar breadth, so the flag is informational. The head centre's
error that the priors leave is stated with the phantom: 9 mm mediolateral (the angle's
spread over the length), 6 mm anteroposterior (the offset's lever through the anteversion's
7.5° spread), 5 mm along the axis (the length's own reading), about 12 mm combined at
410 mm, one SD, not counting the error of carrying the shaft axis up from a distal scan,
which is unpublished. We measured it on the one full-length femur we have (REF_KJ2, head
sphere fitted on the mesh, diameter 44.9 mm, hip-to-knee length 418.8 mm): with the scan
cropped to 140, 180, 230 and 300 mm of femur above the knee, the frame's axis fell 3.2,
1.7, 2.0 and 2.4° from the true knee-to-head line, and the phantom's head, placed at the
true length, fell 23, 12, 15 and 17 mm from the fitted head. That is one knee, so it
supports the stated order of magnitude and nothing finer; it is to be repeated on more
full-length femurs before a figure is quoted. On the full-length scan itself the frame's
axis is 6.9° off (the head 51 mm), because an axis taken along a whole bowed diaphysis
is not the one a distal scan gives; that case is not what the phantom is for. The phantom's axis is a population prior and carries no patient
varus or valgus; it cannot support an alignment decision, and the app says so beside it.
Anteversion, the trochanters and the shaft's bow are not drawn. The literature basis, with
the status of every number, is in `PHANTOM_FEMUR_RESEARCH_2026-10-25.md`.

**Orientation gate.** The landmark searches run in each bone's own axes but take their
signs from the scanner, so a knee lying too far from LPS is refused, as an error, before
any landmark is found: a bone's long axis more than **45°** from scanner superior
(`SCAN_ORIENTATION_TILTED`); the femur's posterior condylar line more than **45°** from
scanner left-right, measured about the femur's own axis (`SCAN_ORIENTATION_TURNED`); the
femur below the tibia along its axis (`SCAN_UPSIDE_DOWN`); or the condylar block
reaching more than **5 mm** further in front of the femoral shaft than behind it — a knee
facing backwards, which the line tests cannot see (`SCAN_FACING_BACKWARDS`). 45° is half
of the 90° at which those signs become ambiguous. On the 11 cohort knees the bones' long
axes lie 1–8° from superior, the femur's condylar line 0–15° from left-right, and the
posterior reach exceeds the anterior by 16–31 mm. Before the gate, every knee turned upside down was
planned (errors to 114° in a metric, 119° in the femoral cut) and at 90° seven of
eleven failed inside the planner.
Within it, the estimators are invariant to how the knee lies; the coverage lengths are
measured along the bones' axes too. On the cohort (E2 after the fixes) every accepted
rigid or scanner motion reproduced the original to 0.006 mm in landmarks, 0.001° in
metrics and 0.002° / 0.008 mm in the plan; every knee turned 90° or 180° was refused.
The gate is not a guarantee of the estimators, though: tilted 30° forward about scanner X
(about 38° in all), two knee-only femurs (P007R, P029L) had their long axis estimated
130° from superior, and one knee turned 30° (P047L, already 15° turned) crossed the turn
limit. All three were refused, not planned.

## Verification

`pytest` runs the full suite with no Blender installed.

Synthetic knees are built to *realise* a stated mLDFA, MPTA, mechanical-anatomical angle,
condylar twist and posterior slope, so every metric is checked against a value that was an
**input** rather than against its own output. Three sign errors were caught this way and
would not have been caught otherwise: aLDFA relates to mLDFA by subtraction not addition,
and both the condylar twist and JLCA conventions were inverted.

**Mirror invariance is the flagship test**: reflect a knee through the sagittal plane,
flip its side, and every scalar metric and resection depth must be unchanged. A knee is
not more varus for being a right knee. This is the class of bug the legacy pipeline had,
where compartments were labelled by world position and so were right on one side only.

**Rigid motion is the second.** The same knee moved and turned in the scanner is the
same knee: the automatic landmarks on synthetic bones must move with the bone to 1e-6 mm
under translations, ±30° turns about each axis and an oblique motion; both frames must
turn with landmarks carried through any rotation; and the plan's cuts, depths and
component poses must turn with the knee. Paper experiment E2 runs the same check on the
cohort's meshes.

Blender-side geometry is verified headless and numerically — component offset from its
cut plane, whether parts coincide, whether the joint stays articulated through the arc —
rather than by inspection.

## The boolean kernel

Cutting is the one job the pipeline used to borrow Blender for, and Blender is GPL while
this project is Apache 2.0. The cut now goes through a two-method interface — difference
and intersection over meshes in millimetres — with two backends behind it.

**`manifold3d` is the default**, and the only one in the shipped dependency set. It is
Apache 2.0, it is fast enough that a full-resolution resection is seconds rather than
tens of them, and it *refuses* a non-manifold input rather than guessing. That refusal is
worth more than the speed. A solver that silently accepts a broken mesh returns broken
geometry, and a cut that looks like a cut and is not is the failure this project keeps
guarding against.

Because segmentations are not reliably manifold, a repair pass runs first. Weld, then drop
degenerate and duplicated faces; if that leaves a closed, consistently wound manifold,
nothing more is done (every cohort bone, once the detached pieces are gone). Otherwise, in
up to three rounds: remove debris pieces (*Detached pieces*, under
Quality control); cut out every face on an edge that three or more faces share, since no
rule for which of them to keep is principled; wind every face consistently, the main body
outward and other pieces relative to their own seed, so a cavity wall exported facing
inward stays inward; and close each hole up to **10 mm** across with a fan of triangles
around its centroid, wound against the faces around it. No existing vertex moves. A wider
opening is something missing from the scan rather than a defect of the surface, and a
boundary that passes through one vertex twice (two surfaces meeting at a point) cannot be
closed by a fan; both are left open and the mesh is refused, saying which. This is the
author's rule (2026-10-15): repair every bone as far as it can be repaired and refuse,
with a warning, only what cannot. Whatever repair did is recorded rather than done
quietly, and lands in `plan.json` beside the measurement provenance.

The repair is done **once per bone**. A commit cuts each bone twice (the block's shell,
then the block), and the app re-commits from the same uncut bone after every control
change; each boolean used to repeat the weld and the checks, 92 % of a commit's time. The
prepared mesh and its readiness are now kept per content hash of the mesh (the four most
recent bones), and opening a case hands the kernel the welded copy quality control has
already made. On REF_KJ2 a commit took 11.2 s every time before; the first now takes
0.7 s and a re-commit 0.7 s. P007R, refused at Commit before, commits in 2.1 s and
re-commits in 2.0 s. Opening a case costs 1–4 s more; the geometry is unchanged.
Bone shells are drawn deep orange (they were red, which the fit map uses for overhang).

*A limitation:* neither solver checks that a surface does not pass through itself. A
self-intersecting segmentation is closed and manifold, so it passes the repair and both
manifold3d and Blender's exact solver cut it without complaint, and what they return is
not defined. The cohort's bones have not been tested for it; a segmentation tool's own
export is trusted on this point, and the paper says so.

**Blender's exact solver is kept as a cross-check**, behind an optional extra, never
imported at runtime. Four cases are cut in Blender once and committed as STL, and the
gate asserts the two solvers agree on volume, surface area, bounding box and a symmetric
Hausdorff distance to within 0.0001 mm. Keeping the reference in the repository rather
than the dependency in the environment is deliberate: `bpy` importable in the working
environment makes it far too easy to depend on Blender by accident.

Two measurement decisions are worth stating, because both started as wrong answers.

*Agreement is measured on the solid, never on its triangulation.* Two solvers producing
the identical solid triangulate it differently, so a vertex of one routinely lands in the
middle of a face of the other. A vertex-to-vertex comparison reported that as a 1 mm
error. The metric is point-to-surface, by the standard barycentric-region method.

*A triangle fan cannot read back a boolean's output.* A difference leaves non-convex
n-gons — cutting a notch out of a face's edge produces one — and fanning such a polygon
yields overlapping triangles whose signed areas cancel. The volume stays correct and the
surface stays in the right place while the unsigned surface area is inflated, measured
here at 12%. It presented as the two solvers disagreeing; it was the reader, not either
solver. Blender's own triangulation is used now.

`Mesh64` is used throughout rather than the float32 path. Anatomical coordinates run to
several hundred millimetres, where single precision resolves to about 3e-5 mm. That is
below any clinically meaningful tolerance, but it is a needless loss on a round trip and
a planner that publishes its methods should not quietly discard digits.

## Persistence

Records go in SQLite, geometry goes in a directory addressed by content hash, and the
database refers to geometry by that hash. The split is what lets a hundred-megabyte
femur be stored once and referenced by every plan that did not change it, and it keeps
the database itself small enough to copy.

**A commit is cached on what the boolean reads, not on what the surgeon set.** The key
covers both source mesh digests, the resection mode, the shell flag, and the resection
planes and component poses the plan resolved to; it does not cover the control values.
Two things follow. A display toggle cannot throw away a commit that took forty seconds,
because it changes no plane. And two different routes through the controls that arrive
at the same cut hit the same entry, which keying the inputs would have missed.

**The key is per bone.** Adjusting the tibial slope re-cuts the tibia and leaves the
committed femur untouched. A single key over the whole plan would have thrown both away,
which is the behaviour the add-on had and the reason a slope change used to cost a full
re-cut.

**A recut starts from the bone as it was read.** After a commit the scene holds the cut
femur, so cutting again would subtract the moved block from geometry that is already
missing its condyle: it removes more bone and can never put any back. The uncut mesh
survives in the scene's content-addressed table at no cost, and the commit restores it
before cutting. This was found by a test asserting that two commits either side of a
0.001 mm change give the same volume.

Every row carries an `owner`, empty for a single-user install and never read there. Its
presence is the difference between multi-user being a filter and being a migration.

Meshes are stored as compressed `npz` rather than STL. STL is float32 triangle soup, so
a round trip through it would preserve neither the geometry nor the content hash that
addresses it.

## The application

**The transport is the standard library, not a web framework.** The shipped dependency
set is numpy and `manifold3d`. A local planner serving one surgeon on one machine gains
nothing from an ASGI stack that `ThreadingHTTPServer` does not give it — there is no
schema to publish, no third party calling in, no concurrency to manage. What it loses is
worth avoiding: a clinical tool whose install can fail on a hospital machine because a
transitive dependency moved. The routes are plain functions taking a method, a path and a
body and returning a status and a payload, so a different transport is an adapter file.

The same reasoning retires the WebSocket the design called for. Every delta in a local
session answers a request the viewer itself just made, so a second channel to deliver it
buys a socket, a reconnect policy and an ordering question in exchange for nothing. The
delta format is unchanged and is exactly what a socket would have carried, so the channel
can be added when a second writer — a tracker feed — exists to push from.

Threaded, for one specific reason: a commit runs booleans that take tens of seconds, and
the browser must still be able to fetch the meshes it is already drawing. Bound to the
loopback interface, because a planner holding patient geometry has no business listening
on a network interface, and a default cannot be forgotten.

**Every mutating route answers with the same envelope** — the delta, the clipping
half-spaces, the report lines, both sets of control values and the stale flag. The
viewer therefore has one function for applying a response rather than one per control,
and no route whose answer it has to special-case.

**Millimetres stop at the glTF encoder.** Vertices are divided by a thousand there and
node translations by the same factor in the viewer, which are the two halves of one
boundary. A test asserts the two constants agree, because a mismatch puts every pose a
metre out. Nothing else in the project sees a metre.

## The browser viewer

**Plan mode does not cut. It clips.** A planar resection is a half-space, and a clipping
plane evaluates the same half-space per pixel instead of per triangle, so the previewed
cut follows a slider at the refresh rate of the screen with no geometry work on either
side. The only difference from the committed geometry is the triangulation of the cut
face, which is why the preview is honest rather than an approximation of one.

The design left open whether a stencil cap pass was worth its complexity. It is not.
A bone is a closed surface, so drawing a second copy of it inside out behind the same
clipping plane shows exactly the back faces visible through the cut; in resected-bone
colour that reads as solid. It is a few lines against a multi-pass stencil setup, and the
difference is a cut face that is shaded rather than flat.

**The panel is described once, on the server.** Clinical ranges — the femoral cut may be
flexed between -10 and +15 degrees — are not rendering details. A viewer that wrote its
own copy of them would be a second place for a limit to live, and two places drift. The
schema is served, the panel is built from it, and a test asserts it covers the session's
controls exactly, so a control added to one and forgotten in the other fails the suite
rather than becoming invisible.

**The viewer computes nothing clinical.** Every number displayed comes from the server,
which is what preserves the guarantee that the method behind a number is auditable. A
test greps the viewer for the names of the planning functions; its job is to catch the
moment somebody reimplements a measurement in JavaScript to save a round trip.

**three.js is vendored, its glTF loader is not.** A tool holding patient geometry should
not need the public internet to draw a bone, and a hospital machine frequently cannot
reach it, so the library is committed with its MIT licence rather than fetched from a
content delivery network. Its loader is left out because the server emits one mesh with
one primitive, positions and indices: reading exactly that is forty lines against
vendoring a loader for a format we deliberately do not use. Node reads a file the Python
encoder wrote, using the viewer's own parser, so a disagreement about padding or accessor
offsets fails in the test suite instead of silently in a browser console.
