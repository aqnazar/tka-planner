# Observer studies on screen: Pick and Review

Two modes in the planner's window let a surgeon take part in the validation without a
tracking or navigation system. Everything is done on the screen, on the case the
planner has open, and saved by the server.

```
tka serve --cases path/to/cases --library data/implants --study path/to/study_output
```

`--study` is where picks and verdicts are written; without it they go to a `study`
folder inside the store (`~/.tka-planner/study`). Study output holds patient geometry and
the surgeons' names: it is private, keep it outside the repository.

Both modes ask for the **surgeon's name** and a **round** (1, 2, ...: the same observer's
sessions, at least a week apart for intra-observer variation). The same name means the
same person across sessions: names that differ only in case, spacing or Unicode form
("Dr. Ivanov", " dr.  ivanov ") are one observer. An empty name is refused.

Files are named by a file-safe form of the name, the observer's `<slug>`: the name
normalised (Unicode NFKC, trimmed, single spaces, case folded), ASCII letters and digits
kept, every other run of characters made one `-`, at most 48 characters. When anything
was replaced or cut, the first 8 hex digits of the normalised name's SHA-1 are
appended, so names in Cyrillic or other scripts stay distinct (for example
"Ахметов А.А./KZ" becomes `kz-` and eight hex digits). Every saved record keeps the name
as typed (`observer`) and its normalised form (`observer_key`).

## Pick (landmark picking, experiment E3)

The surgeon places the landmarks on the **whole uncut femur and tibia**, one at a time,
in a fixed order (femur first), each shown with its definition from the landmark
registry:

1. Lateral epicondyle, 2. Medial epicondylar sulcus, ... 19. Fibular head apex.

- **Click** the bone to put the current marker there. **Drag** the current marker to
  slide it over the surface. Orbit, pan and zoom as usual. Only the current marker can
  be grabbed, so a click beside a placed neighbour does not pick that one up.
- **Next / Previous** (or Enter / Backspace), or click a landmark in the list to go to
  it. **Cannot place** records it as skipped, with a reason.
- **Show** draws the current marker's bone only (the default, so the other bone never
  hides the condyles), either bone, or both.
- **Remove marker** / **Reset all** clear what has been placed.
- **Save progress** writes what is done so far. **Finish and save** is enabled once
  every landmark is placed or skipped. Unsaved markers are also kept as a draft in the
  browser and offered back when the same case, observer, mode and round are started
  again.

Two ways to start:

- **No markers** (`blank`): nothing is placed. The automatic estimate is not sent to the
  browser at all, so a blinded pick cannot be anchored on it.
- **Automatic markers** (`adjust`): every landmark the planner estimated starts at its
  automatic position, which stays on screen as a see-through **ghost** joined to the
  marker by a line. The surgeon moves the ones that are wrong; **Reset to automatic**
  puts one back, **Reset all** puts all back, and the ghosts can be hidden. A landmark
  with no estimate on that knee is placed as in `blank`.

Each save writes two files to `<study>/picks/<slug>/`:

- `<case>_<side>_<mode>_r<round>_<UTC>.landmarks.json`: an ordinary landmark file
  (`tka-landmarks/1.0.0`, status `present`, origin `observer:<slug>`; skipped landmarks
  `not_picked` with the reason). It is read by `tka measure --landmarks` like any other.
- `....study.json`: the record: time on each landmark, total time, whether the set is
  complete, each pick's distance to the bone surface, and in `adjust` mode how far each
  marker was moved from the automatic position.

A pick more than 5 mm from its bone's surface is refused as a fault.

## Review (plan verdicts, experiment E13)

The surgeon looks at the plan the planner proposes, turns it in the view (the cuts are
previewed as in Plan mode), chooses **Mechanical** or **Kinematic**, and marks it:

- **OK**, or
- **NOT OK**, with at least one reason (resection depths, coronal alignment, femoral
  flexion, tibial slope, rotation, size, fit on the cut, position, insert thickness, a
  wrong landmark, other) and a severity: *minor, would adjust during surgery* or
  *major, would not use this plan*,

plus an optional comment. The plan's adjustment controls are not shown in Review. If
the plan was adjusted by hand beforehand, the panel says so and the verdict records it.

Each verdict is one line of `<study>/reviews/<slug>.jsonl`, with the case, side,
alignment, verdict, reasons, severity, comment, time taken, every control value and the
report as the surgeon saw it.

## Online (Pick only)

Surgeons can also do the Pick rounds in their own browser, on a static site that holds no
readable geometry (see `online/README.md`). The site runs the same viewer and Pick code. Each
knee was computed beforehand by the planner's own `pick_targets` and is decrypted in the
browser with the study passphrase. Every surgeon gets the study knees plus one group of 20
VSD knees (groups A, B, C, with no overlap), and does both rounds: *No markers*, then
*Automatic markers*, which unlocks only after the first round is finished on that knee.
`python -m online.pull` writes each online save through `save_picks`, so online picks are
the same files as local ones, with `"via": "online"` in their source. Review is not online yet.

