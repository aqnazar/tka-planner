# Manual planning controls — design

Interactive adjustment of a computed plan, in the Blender planning screen, in the manner
of a commercial planner's controls.

## The problem

`tka.plan` is one-shot: it clears the scene, re-reads the STLs, re-estimates landmarks,
re-plans, rebuilds every object, applies boolean cuts and keyframes the flexion arc.
That is seconds of work, so nothing about it can respond to a slider.

A surgeon adjusting varus/valgus expects both cuts and every component to move together
as the value changes. So the work splits by cost:

| Stage | Cost | When it may run |
|---|---|---|
| Read meshes, estimate landmarks, build frames, measure, size | seconds | Press **Plan** |
| `plan_alignment` — pure numpy on a few dozen points | microseconds | Every slider tick |
| Move existing plane and component objects | milliseconds | Every slider tick |
| Import meshes, boolean-cut bone, keyframe flexion | seconds | Press **Plan** |

The design follows that table exactly: a **session cache** holds the expensive results, a
**live path** re-plans and re-poses, and a **rebuild path** does everything.

## Adjustments belong in the core, not the viewport

The temptation is to nudge Blender objects directly. That would break the project's
central claim — that a plan is data anyone can re-derive — because the scene would then
hold geometry the plan file does not describe.

So manual controls become a frozen `Adjustments` dataclass threaded through
`plan_alignment`. The plan stays a pure function of `(landmarks, frames, target,
adjustments)`, it serialises into `plan.json`, and every control is testable with no
Blender present.

```python
@dataclass(frozen=True)
class Adjustments:
    coronal_correction_deg: float = 0.0      # varus/valgus, BOTH cuts together
    femoral_resection_delta_mm: float = 0.0  # + is deeper
    femoral_varus_delta_deg: float = 0.0
    femoral_flexion_delta_deg: float = 0.0
    femoral_rotation_delta_deg: float = 0.0  # + external
    femoral_shift_ap_mm: float = 0.0
    femoral_shift_ml_mm: float = 0.0
    tibial_resection_delta_mm: float = 0.0
    tibial_varus_delta_deg: float = 0.0
    tibial_slope_delta_deg: float = 0.0
    tibial_rotation_delta_deg: float = 0.0
    tibial_shift_ap_mm: float = 0.0
    tibial_shift_ml_mm: float = 0.0
    insert_thickness_mm: float | None = None
```

Where each acts:

- **`coronal_correction_deg`** rotates the *shared reference direction* about the patient
  frame's anteroposterior axis, before either cut is built. Because both cuts descend
  from that one reference, they rotate together and keep their identical mediolateral
  slope — which is the invariant the whole planning module is built to protect. This is
  the varus/valgus control.
- **Per-cut varus deltas** rotate one cut's normal about the same anteroposterior axis,
  *after* the split. They deliberately break the shared slope, so using them raises the
  existing mediolateral-disagreement warning rather than hiding it.
- **Flexion and slope deltas** add to `target.femoral_flexion_deg` and the slope, so they
  travel through the existing `_tilt_about` hinge and inherit its sign handling.
- **Resection deltas** shift the plane point along its own normal, so a positive value
  removes more bone on both compartments equally.
- **Rotation deltas** add to the external rotation fed to `_rotational_reference` on the
  femur, and rotate the tibial component's anterior direction about its cut normal.
- **Shifts** translate the component pose along its own in-plane axes, leaving the cut
  where it is. This is component position, not resection.

`insert_thickness_mm` changes no geometry of the cuts. It sets the construct height used
for the gap calculation and the placeholder slab, which is what "the number and the
spacing" means until the parametric insert model arrives.

## Gaps

Insert thickness is only meaningful next to the gap it fills, so the plan gains a gap
report:

```
extension gap (medial) = distance between the two cut planes at the medial compartment
                       - femoral component distal thickness
                       - tibial tray thickness
                       - insert thickness
```

measured along the femoral cut normal at the medial and lateral landmark positions
projected onto the cuts. Negative means the construct overstuffs the compartment. This
uses only the plan's own planes plus three thicknesses from the size chart, so it costs
nothing and updates live.

The **flexion gap is deliberately absent.** It needs the posterior condylar resection,
which this pipeline does not plan (see `CLINICAL_QUESTIONS.md` 4.2). Reporting a flexion
gap we cannot compute would be exactly the dishonesty the provenance system exists to
prevent.

## The live path in Blender

A module-level `_SESSION` dict, keyed by the resolved patient folder and side, holds the
meshes, landmark set, frames, metrics, measurements and sizing. It is written by **Plan**
and read by every live update. Blender properties cannot hold numpy arrays, so this lives
in the module rather than in the `PropertyGroup`.

Each adjustment property carries an `update=` callback. The callback re-plans from the
cache and calls `update_scene`, which:

1. moves the two cut-plane discs to the new plane poses,
2. sets `matrix_world` on every component object from the new component poses,
3. moves the persistent boolean cutters so the bone cuts follow,
4. resizes and re-seats the insert placeholder,
5. rewrites the readout string.

No object is created or destroyed, so there is nothing to accumulate and nothing to
re-import.

**Bone cuts become live modifiers.** `resect_bone` currently applies the boolean and
deletes the cutter. It gains a `live=True` mode that leaves the modifier unapplied and
keeps the cutter as a hidden object, so moving the cutter re-cuts the bone. On a
segmentation of several hundred thousand triangles the exact solver may not keep up with
a drag, so the panel carries a **Live cuts** toggle and a solver choice; with it off, the
planes and components still move in real time and the bone re-cuts when released.

The tibial cutter is parented to the flexion pivot alongside the tibia, otherwise the cut
would slide through the bone as the joint flexes.

## What comes from the commercial planners, and what does not

Added, because the machinery is already there:

- six-axis component adjustment (depth, varus/valgus, flexion/slope, rotation, AP and ML
  shift) on both components
- a size override that steps through the chart's twelve sizes, or stays on the solved
  continuous size
- live resection depths per compartment, live cut angles, live gaps
- reset to the computed plan

Not added, and each for a stated reason:

- **Gap balancing across flexion** — needs a ligament model, which does not exist
  (`CLINICAL_QUESTIONS.md` 4.1)
- **Implant overhang and bone coverage** — needs the implant footprint intersected with
  the resection cross-section; real new machinery, worth doing next
- **Anterior notching check** — same, and flagged as the most valuable safety addition
- **Limb alignment before and after** — not computable on a knee-only scan, and the
  pipeline already says so rather than guessing

## Testing

Core tests, no Blender:

- an all-zero `Adjustments` reproduces the unadjusted plan exactly
- `coronal_correction_deg` moves both cut normals by the same angle and preserves the
  shared mediolateral slope
- a resection delta changes both compartment depths by exactly that many millimetres
- a slope delta changes the tibial sagittal angle by exactly that many degrees and leaves
  the femoral cut untouched
- shifts move the component pose translation and leave the cut plane's point alone
- gaps decrease by exactly the increase in insert thickness
- mirror invariance survives adjustment: a mirrored knee with mirrored adjustments gives
  mirrored geometry and identical scalars

Blender-side, headless and numerically: after a live update the components still sit on
their planes, and a cut bone still matches its moved plane.
