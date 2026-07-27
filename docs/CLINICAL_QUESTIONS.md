# Clinical questions for the surgical team

Every question below marks a place where the planning pipeline currently makes a choice
that a surgeon should be making. Each one names the value or rule the code uses today, so
the answer can be a correction rather than a design exercise.

> Research and demonstration only. Not a medical device. The point of this document is to
> establish which of our assumptions are clinically defensible before any of them are
> published as method.

**How to use this.** Sections 1–3 are blocking: the plan is not clinically meaningful
until they are answered, and several of them change code. Sections 4–7 shape what the
system should become. Section 8 is process. A question marked **[decision]** needs a
surgeon to choose; **[data]** needs something measured, picked or supplied.

---

## Summary — the eight that block us

| # | Question | Why it blocks |
|---|---|---|
| 1.1 | Which alignment philosophy is the default, and is restricted kinematic needed? | Determines what the primary output even is |
| 2.1 | Tibial resection: how much, measured from which plateau? | Hardcoded 10 mm from the less-worn side, unverified |
| 2.3 | Posterior slope target, and does it follow the implant's CR/PS design? | Hardcoded 3°, applied to every case |
| 2.5 | Femoral rotation: fixed 3° off the posterior condylar axis, or measured to the surgical TEA per patient? | Currently fixed; unsafe in valgus knees |
| 3.1 | What does the size chart's `tibia_proximal_cut` column (16.3–22.1 mm) actually measure? | Cannot publish any resection metric derived from it |
| 4.1 | Is a bone-only plan with no soft-tissue balancing clinically useful? | Defines the honest scope of the whole system |
| 5.1 | Should osteophytes be removed from the segmentation before planning? | Osteophytes corrupt every condylar and plateau landmark we take |
| 7.1 | What accuracy would make this credible, and what is the ground truth to test against? | We have no acceptance criterion to validate against |

---

## 1. Alignment philosophy

**1.1 [decision] Which philosophy should be the default, and do we need a third?**
We implement exactly two: *mechanical* (cuts perpendicular to the mechanical axes,
targeting a neutral limb) and *kinematic* (cuts parallel to the native joint surfaces).
Restricted kinematic, adjusted mechanical and functional alignment are deliberately out of
scope. Should restricted kinematic be added, and if so with what boundaries — the usual
form is a residual alignment within ±3° of neutral with aLDFA and MPTA held inside
85–90°?

**1.2 [decision] In mechanical alignment, is the target a neutral HKA of 0°, or a small
residual varus?** We currently aim at 0°. Constitutional varus is a live argument in the
literature and we have no position on it.

**1.3 [decision] At what magnitude of correction should the planner warn or refuse?**
Right now nothing stops the software from planning a large correction on a badly deformed
knee. What coronal correction, in degrees, should raise a flag for the surgeon to review?

**1.4 [decision] Kinematic alignment is defined against the *cartilage* surface, but our
STLs are segmented bone.** We plan against bone extremes. Should we add a cartilage
allowance (a nominal 2 mm is often quoted), and should it differ between the worn and
unworn compartment? See also 5.2.

---

## 2. Resection parameters currently hardcoded

These are all defaults in `tka_planner/core/planning.py`. Every one of them was chosen
because a number was needed, not because it was clinically established.

**2.1 [decision] Tibial resection depth: we use 10 mm, measured from the *higher*
(less-worn) plateau.** Two things to confirm: the magnitude, and the datum. The common
alternatives are a fixed depth from the less-worn side (what we do), or a smaller depth —
typically 2 mm — from the *most-worn* side. Which do you use, and does it change with
deformity severity?

**2.2 [decision] Distal femoral resection is set equal to the femoral component's distal
thickness (9 mm default), seated on the most prominent distal condyle.** Is component
thickness the correct datum, and should the cut reference the less-worn condyle instead,
as on the tibia?

**2.3 [decision] Posterior tibial slope defaults to 3° in mechanical alignment, and
reproduces the patient's measured native slope in kinematic.** Slope is largely dictated
by implant design — a cruciate-retaining knee typically wants more slope than a
posterior-stabilised one. What does *this* implant require, and is the kinematic
behaviour (reproduce native, whatever it measures) acceptable, or should it be capped?

**2.4 [decision] Femoral component flexion defaults to 0°.** Several surgeons plan 3° of
flexion to reduce the risk of anterior femoral notching. Should this be the default, and
should it be automatic when the anterior cortex is at risk? (We do not currently check for
notching at all — see 4.3.)

**2.5 [decision] Femoral component external rotation is fixed at 3° from the posterior
condylar axis.** This is the theatre rule and it is what we implement. But the pipeline
*can* measure the condylar twist angle to the surgical transepicondylar axis per patient
once the medial sulcus is picked. Should rotation be planned patient-specifically against
the surgical TEA instead of the fixed 3°, particularly in valgus knees where the lateral
posterior condyle is hypoplastic and the posterior condylar axis is unreliable?

**2.6 [decision] Tibial component rotation.** Our landmark schema uses the Akagi line —
PCL insertion midpoint to the medial border of the patellar tendon — with a fallback to
the medial third of the tubercle. Which reference do you actually use? The automatic
estimator rates these landmarks as its *weakest*, because a tendon insertion has no
geometric signature on a bone surface, so this may need to be a manual pick in every case.

---

## 3. The implant size chart

**3.1 [data] What does the `tibia_proximal_cut` column measure?** It runs 16.3–22.1 mm
across the twelve chart sizes, where a clinical proximal tibial resection is 8–10 mm. It
scales as exactly `0.2625 × femur_ML`, so it is a property of the implant, not of a
patient. Our reading is that it is resection *plus* construct height, or a distance from a
different datum. We need the CAD definition confirmed before any resection figure derived
from it is published. The same applies to `femur_distal_cut` at `0.1125 × femur_ML`.

**3.2 [decision] Insert thickness selection.** The chart supplies a thin and a thick
insert per size. We have no rule for choosing between them, because the real criterion is
gap balance, which we do not model (4.1). What should the planner do — always report both,
default to the thin one, or leave the field blank until balancing exists?

**3.3 [decision] Sizing between chart sizes.** For the femur, the classic trade-off is
down-sizing to avoid overstuffing the patellofemoral joint versus up-sizing to avoid
notching. For the tibia, maximising coverage versus avoiding overhang. Which way should
the software round, and does it differ per component?

**3.4 [decision] What overhang is acceptable, and is the tolerance different medially and
laterally?** Medial tibial overhang is associated with soft-tissue irritation; we have no
tolerance encoded and produce no overhang measurement at all yet.

**3.5 [decision] One real case in the cohort measures 86.4 mm against a largest chart size
of 84.0 mm.** Because the implant is a single master geometry under a uniform scale, we
can size it continuously and simply produce an 86.4 mm implant. Is a continuously-sized
component acceptable to you clinically, or must every plan snap to a published size?

---

## 4. What we do not model at all

These are the honest gaps. Each is a decision about scope as much as a technical question.

**4.1 [decision] There is no soft-tissue model and no gap balancing whatsoever.** The
plan is purely bony geometry. Flexion and extension gap symmetry, ligament tension and
releases — none of it exists in the system. Is a bone-only plan clinically useful as a
starting point, or does it mislead? If we were to add a minimum viable balancing step,
what would it need to contain to earn its place?

**4.2 [decision] Posterior condylar resection and the flexion gap are not planned.** We
plan the distal femoral cut and the tibial cut. The posterior and chamfer cuts follow from
the component geometry once it is placed, but we do not report posterior condylar
resection depths or posterior condylar offset restoration. Should we?

**4.3 [decision] No anterior notching check.** Nothing in the pipeline detects that the
anterior flange would breach the anterior femoral cortex. This seems like the single most
valuable safety check we could add. Confirm, and tell us the threshold at which it should
warn.

**4.4 [decision] Joint line height is not measured or restored.** We have no metric for
change in joint line position, which matters for mid-flexion stability. Which reference do
you use to judge it?

**4.5 [decision] The patella is out of scope.** A `patella.ridge_apex` landmark is
registered in the schema but nothing uses it. Do you resurface, and if so what would the
planner need to produce — resection depth, residual composite thickness, component size?

**4.6 [decision] PCL retention versus sacrifice.** The pipeline is agnostic and does not
know whether this implant is CR or PS. That choice drives the slope target (2.3), whether
a box cut is needed, and insert geometry. Confirm the design intent.

**4.7 [decision] Bone defects, augments and stems are not modelled.** We assume a primary
knee with intact bone stock. Confirm that revision and severe defect cases are out of
scope for this work.

**4.8 [decision] Flexion contracture and hyperextension are not assessed.** The scan is
supine and static, so we see no information about the extension deficit. Is that a
limitation we simply state, or does it need to be captured some other way?

---

## 5. Imaging, segmentation and data quality

**5.1 [decision] Osteophytes.** Our segmented meshes include them, and the pipeline finds
condylar and plateau landmarks by geometric extremes — so an osteophyte is silently taken
as the joint surface. Should osteophytes be removed during segmentation before the mesh
reaches the planner, and if so, by whom and against what rule? This currently affects
every landmark we take on a degenerate knee, which is to say every real case.

**5.2 [data] Cartilage is not in the segmentation.** Bone-only STLs mean the "joint
surface" we plan against is subchondral bone, and the worn compartment has lost more of it
than the unworn one. How should we account for this — a nominal thickness, a
compartment-dependent allowance, or a statement that plans are bone-referenced and the
surgeon adjusts intra-operatively?

**5.3 [data] Every scan in the cohort is knee-only.** Femur segments run 147–248 mm
against a whole femur of 400–500 mm, so the femoral head and the malleoli are outside the
field of view in all twenty cases. True mechanical axes, HKA and mLDFA therefore **cannot
be measured** — anything reporting them is substituting a population assumption (we use a
6° mechanical-anatomical angle, declared as such). Two questions follow. First: is
kinematic-only planning acceptable for this work, given it is the one philosophy this data
fully supports? Second: can we obtain full-limb imaging — a CT topogram, EOS, or a
long-leg standing radiograph — for even a subset of cases? The code paths for true
mechanical axes already exist and would light up immediately.

**5.4 [decision] Supine CT versus weight-bearing alignment.** Deformity measured supine
differs from deformity under load. If long-leg standing films are available, how should
the two be reconciled?

**5.5 [data] Segmentation protocol.** Who segments, with what threshold and what manual
correction? Different operators produce different bone surfaces, and we have no measure of
how much that moves the plan. A repeat segmentation of the same scan by two operators
would quantify it.

**5.6 [data] Scan acquisition.** Confirm slice thickness, reconstruction kernel and
whether metal artefact is expected in any case. Our QC gates assume a clean bone surface.

---

## 6. Landmarks

**6.1 [data] The medial epicondylar sulcus requires a human pick.** It is a depression
rather than a surface extreme, so no automatic estimator can find it — and without it, the
surgical transepicondylar axis is unavailable and we substitute the anatomical axis,
degrading the provenance of everything downstream. We need a picking protocol, and ideally
the same cases picked independently by two or more surgeons.

**6.2 [data] Inter-observer variability is unquantified.** The sensitivity machinery
exists and runs thousands of Monte Carlo perturbations; what it lacks is a real
measurement of how much surgeons actually disagree when picking the same landmark. Two
surgeons picking a full landmark set on ten cases would let us report a genuine
uncertainty on every metric rather than a synthetic one.

**6.3 [decision] Confirm the landmark definitions themselves.** Our schema defines each
point in prose. Two worth checking explicitly: we take the femoral knee centre as the
**roof of the intercondylar notch**, not the transepicondylar midpoint; and the tibial
plateau references as the **lowest point of each compartment**, which on a worn knee sits
in the defect. Are both correct as you would pick them?

**6.4 [decision] Which landmarks must a surgeon pick, and which may remain automatic?**
Everything the estimator produces is marked `estimated` and awaits review. If a subset can
be trusted automatically, the review burden per case drops substantially.

---

## 7. Validation and acceptance

**7.1 [decision] What accuracy would make this credible?** We need a stated acceptance
criterion — for example ±1° in coronal cut angle and ±2 mm in resection depth — before we
can claim the system is validated rather than merely self-consistent.

**7.2 [data] What is the ground truth to test against?** Three candidates, and we would
like your view on which is worth the effort: a surgeon's manual plan on the same cases; a
commercial planner's output on the same cases; or post-operative CT on cases actually
operated. The first is cheapest and we can start immediately.

**7.3 [data] How many surgeons and how many cases** should the inter-observer and
plan-agreement study cover to be publishable?

**7.4 [decision] What should the report contain, and in what order?** The current HTML
report is our guess at what matters. We would rather rebuild it around what you would
actually read before a case.

**7.5 [decision] How should uncertainty be presented?** Every value carries a provenance
tier — measured, estimated, or not computable — and estimated values name the assumption
substituted. Is that the right way to show it, or is it noise in a clinical report?

---

## 8. Workflow and process

**8.1 [decision] Where does this sit in the clinical workflow?** Who runs it, at what
point before surgery, and how long may it take end to end including landmark review?

**8.2 [decision] What must the surgeon be able to override?** Currently the alignment
philosophy and tibial resection depth are exposed as parameters; everything else is fixed
in code. Which of the values in sections 1–3 should be adjustable per case in the
interface?

**8.3 [decision] Should the plan carry a recorded surgeon approval?** The plan file is
hash-anchored to its input meshes and fully re-derivable. Adding a sign-off — who
approved, when, against which landmark set — is straightforward if it is wanted.

**8.4 [decision] How is the plan meant to reach theatre?** Patient-specific cutting
guides, navigation, or registration to the robot arm? This determines what the plan must
export and what accuracy the transfer step needs to preserve, and it connects this work to
the wider surgical robot programme.

**8.5 [decision] What is needed to move from research to cadaveric or clinical
evaluation?** Ethics approval, institutional route, and who owns that process.

---

## What we would ask for concretely

If the meeting produces nothing else, these four items unblock the most:

1. **Answers to the eight blocking questions** in the summary table.
2. **The CAD definition of the two resection columns** in the size chart (3.1).
3. **Two surgeons picking a full landmark set on ten cases**, to quantify inter-observer
   variability and give us a ground truth for plan agreement (6.2, 7.2).
4. **A decision on full-limb imaging** — even a topogram on a handful of cases converts
   mechanical alignment from an assumption into a measurement (5.3).
