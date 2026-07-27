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

**Femoral ML** — bounding box of the distal block along the frame's mediolateral axis.
Taken in the frame's axes rather than the world's, since a femur lies at whatever
rotation the scanner found it in.

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

**Resection depth is not sizing.** `femur_distal_cut` and `tibia_proximal_cut` are
exactly `0.1125 ×` and `0.2625 ×` `femur_ML` across all twelve rows, so they describe
component *thickness*, a property of the implant. Conflating the two is why the legacy
pipeline could not plan alignment: choosing a size fixed the resection, leaving no free
variable for an alignment target.

> **Open question.** `tibia_proximal_cut` runs 16.3–22.1 mm where a clinical proximal
> tibial resection is 8–10 mm. It looks like resection *plus* construct height, or a
> distance from a different datum. Needs a CAD cross-check before any resection metric
> derived from it is published. Tracked as question 3.1 in
> [CLINICAL_QUESTIONS.md](CLINICAL_QUESTIONS.md).

## Alignment planning

Both cuts are built from **one reference direction and one hinge**, so they share a
mediolateral slope exactly and differ only in the anteroposterior angle.

This is a requirement, not a convenience: the components articulate, so a mediolateral
disagreement is an error the construct absorbs. Deriving each cut from its own estimated
mechanical axis gave a 4.4° disagreement on a real case — error from canal segments only
~82 mm apart on a 174 mm femur, not anatomy. The disagreement is now measured and
reported as a warning.

The reference's own sagittal tilt is stripped against the **patient** frame before use. A
diaphyseal axis fitted to a supine, slightly flexed leg leans forwards ~4.3°, and that
says how the leg lay in the scanner. Component flexion and tibial slope are surgical
parameters, set explicitly. (Projecting against the *bone* frame's anterior axis removes
nothing — it is perpendicular to that frame's own axis by construction.)

A positive slope drops the posterior edge of the cut, tilting the plane normal
posteriorly. Inverting this produced cuts sloping forwards.

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
coronal construction and differs by the condylar twist angle. The direction of that
rotation is "toward the epicondylar axis", which is what the clinical rule means;
deriving it geometrically sent it the wrong way and left the component 6° short.

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

## Automatic landmark estimation

A complete first pass from two STL files, so a plan can be produced with no manual
picking. **Everything it emits is marked `ESTIMATED`**, never `PRESENT` — an automatic
estimate is a starting position for a human to correct.

The estimator solves for the femur's own mediolateral axis first, iterating on the
posterior condyles: locating them needs a posterior direction, which needs the
mediolateral axis, so the two converge together from the world axis in about three
rounds. Everything downstream then searches along the bone's own directions.

An earlier version took the extremes of the epicondylar band along the *world* axes.
On a femur rotated ~15° that finds whatever part of the condylar circumference is widest
in world X, and it put the rotational reference **26° out**.

Confidence is recorded per landmark. Condylar and plateau extremes are good; epicondyles
fair; the tibial rotational references weakest, since a patellar tendon insertion has no
geometric signature to find.

## Flexion

The femur is held fixed and the tibia swings about the **transepicondylar axis** through
the epicondylar midpoint — the standard approximation, and it works because the condyles
are near-circular in the sagittal plane with the epicondyles near those circle centres,
so the tibia rides round them at a nearly constant radius.

Using the posterior condyles puts the axis on the condylar *surface* rather than at its
centre of curvature. The joint then separates as it flexes: closest approach between the
femoral implant and the insert went from 33 mm to 93 mm over 120°. Through the
epicondyles it holds between 5.3 and 9.1 mm.

**Stated simplifications:** the axis is fixed, so femoral rollback and the screw-home
rotation near extension are not modelled, and the tibia is treated as a rigid body
hinging in one plane.

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

Blender-side geometry is verified headless and numerically — component offset from its
cut plane, whether parts coincide, whether the joint stays articulated through the arc —
rather than by inspection.
