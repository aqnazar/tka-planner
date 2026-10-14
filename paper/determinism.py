"""Is a plan a pure function of its inputs? The hash table that answers it.

The paper states that the same meshes give the same ``plan.json``, byte for byte: run
twice, run in two processes, run on two operating systems. This prints the SHA-256 of
the plan file and of each of its sections, and of the committed geometry, so any two
runs can be compared by eye or by ``--compare``:

    python -m paper.determinism --synthetic                  # what CI runs
    python -m paper.determinism --synthetic --compare paper/determinism_synthetic.json
    python -m paper.determinism --case P009L --json hashes.json   # needs the cohort

``--synthetic`` plans the committed fixture under ``tests/fixtures/determinism/``: two
coarse ellipsoid bones and a landmark file built by ``tests/synthetic.py``. Everything
it reads is a committed file, never generated on the fly, so a difference between
machines is a difference in computation and not in the input. The landmarks are given
rather than estimated, because the automatic estimate needs real bone shape.
``paper/determinism_synthetic.json`` is the table this produced on the paper's machine.

A section hash narrows a mismatch down to the part of the plan that moved; a geometry
hash says whether the committed resection came out the same, triangle for triangle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from .harness import REPO, Case, cohort_cases, library_root

FIXTURE = REPO / "tests" / "fixtures" / "determinism"
REFERENCE_TABLE = REPO / "paper" / "determinism_synthetic.json"
SYNTHETIC_ID = "SYNTHETIC_DETERMINISM"

# The command line's controls at their defaults, so the document is the one
# ``tka measure`` writes.
CONTROLS = {
    "philosophy": "mechanical", "size_override": "",
    "femoral_resection_delta_mm": 0.0, "tibial_resection_delta_mm": 0.0,
    "tibial_reference": "less_affected_plateau", "insert_thickness_delta_mm": 0.0,
    "implant_mode": "patient_specific",
}
SECTIONS = ("inputs", "frames", "metrics", "surgical_plan", "sizing", "implant_spec",
            "fit", "quality_control", "landmarks")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mesh_hash(mesh) -> str:
    """The exact geometry: coordinates as float64 and faces as int64, in order."""
    return sha256(np.ascontiguousarray(mesh.vertices, dtype="<f8").tobytes()
                  + np.ascontiguousarray(mesh.faces, dtype="<i8").tobytes())


# ----------------------------------------------------------------------
# The synthetic fixture
# ----------------------------------------------------------------------


def write_synthetic_fixture(folder: Path = FIXTURE) -> Path:
    """Write the fixture's bones and landmarks. Run once; the files are committed.

    Each bone is a dense ellipsoid over its landmarks, as in ``tests/conftest.py``: the
    measurement takes cross-sections through thin slabs, so it needs a real surface.
    """
    from tests.synthetic import synthetic_knee
    from tka_planner.core.landmarks import write_landmark_set
    from tka_planner.core.meshio import Mesh
    from tka_planner.geom import mesh as gm

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    landmarks = synthetic_knee(case_id=SYNTHETIC_ID)
    unit = gm.uv_sphere(1.0, segments=48, rings=40)
    for prefix, name in (("femur", "FD1Left.stl"), ("tibia", "TD1Left.stl")):
        points = np.array([one.position_mm for one in landmarks
                           if one.is_usable and one.id.startswith(prefix)])
        low, high = points.min(axis=0), points.max(axis=0)
        bone = Mesh(vertices=unit.vertices * ((high - low) / 2.0 + 12.0)
                    + (low + high) / 2.0, faces=unit.faces)
        gm.write_stl(bone, folder / name)
    write_landmark_set(landmarks, folder / "landmarks.json")
    return folder


def synthetic_case(folder: Path = FIXTURE) -> tuple[Case, object]:
    """The fixture as a case and its measurement, from the committed files only."""
    from tka_planner.core.landmarks import load_landmarks
    from tka_planner.core.meshio import read_stl
    from tka_planner.pipeline import measure_from_landmarks

    case = Case(SYNTHETIC_ID, "left", folder / "FD1Left.stl", folder / "TD1Left.stl")
    measurement = measure_from_landmarks(
        load_landmarks(folder / "landmarks.json"),
        read_stl(case.femur), read_stl(case.tibia),
        femur_path=case.femur, tibia_path=case.tibia, case_id=case.case_id,
        side=case.side)
    return case, measurement


# ----------------------------------------------------------------------
# The plan file and its hashes
# ----------------------------------------------------------------------


def plan_document_for(measurement, library) -> dict:
    """The plan document the command line writes for these controls."""
    from tka_planner.pipeline import fit_case, implant_spec_case, plan_case, plan_document

    plan, sizing = plan_case(measurement, library=library)
    spec = implant_spec_case(measurement, plan, sizing, library)
    fit = fit_case(measurement, plan, sizing, library, CONTROLS["implant_mode"])
    return plan_document(measurement, plan, sizing, fit=fit, implant_spec=spec,
                         controls=dict(CONTROLS))


def plan_file_bytes(document: dict) -> bytes:
    """The bytes ``write_plan`` puts on disk, read back from a real file."""
    from tka_planner.pipeline import write_plan

    with tempfile.TemporaryDirectory(prefix="tka_plan_") as tmp:
        return write_plan(document, Path(tmp) / "plan.json").read_bytes()


def section_hashes(document: dict) -> dict:
    from tka_planner.pipeline import _jsonable

    return {f"section/{name}": sha256(json.dumps(document.get(name),
                                                 default=_jsonable).encode("utf-8"))
            for name in SECTIONS}


def geometry_hashes(case: Case, measurement, library) -> dict:
    """Commit in block mode, as the application does, and hash every cut mesh."""
    from .planned import planned_session

    session = planned_session(case, measurement, library)
    session.commit()
    names = ("Femur", "Femur.Shell", "Tibia", "Tibia.Shell")
    return {f"geometry/{name}": mesh_hash(session.scene.mesh_for(name))
            for name in names if session.scene.mesh_for(name) is not None}


def hash_table(case: Case, measurement, library, *, commit: bool = True) -> dict:
    """Every hash for one case, in a fixed order."""
    document = plan_document_for(measurement, library)
    table = {"plan.json": sha256(plan_file_bytes(document)),
             **section_hashes(document)}
    if commit:
        table.update(geometry_hashes(case, measurement, library))
    return table


def environment() -> dict:
    import platform

    from .harness import _version
    return {"platform": platform.platform(), "python": sys.version.split()[0],
            "numpy": _version("numpy"), "manifold3d": _version("manifold3d")}


def compare(ours: dict, reference: dict) -> list[tuple[str, bool | None]]:
    """``(name, same)`` per hash; ``None`` where only one table has it."""
    names = list(dict.fromkeys([*reference, *ours]))
    return [(name, ours[name] == reference[name]
             if name in ours and name in reference else None) for name in names]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper.determinism",
                                     description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--synthetic", action="store_true",
                       help="the committed fixture (no patient data needed)")
    which.add_argument("--case", help="a cohort knee by id, e.g. P009L")
    which.add_argument("--write-fixture", action="store_true",
                       help="regenerate the synthetic fixture files (then commit them)")
    parser.add_argument("--no-commit", action="store_true",
                        help="hash the plan only, without cutting the bones")
    parser.add_argument("--json", help="also write the table to this file")
    parser.add_argument("--compare", help="a table to compare against; exit 1 on any "
                                          "difference")
    args = parser.parse_args(argv)

    if args.write_fixture:
        print(f"wrote {write_synthetic_fixture()}")
        return 0
    library = library_root()
    if args.synthetic:
        case, measurement = synthetic_case()
    else:
        from tka_planner.pipeline import measure_case

        found = {c.case_id: c for c in cohort_cases(include_reference=True)}
        if args.case not in found:
            parser.error(f"no case {args.case} in the cohort (set TKA_COHORT_DIR)")
        case = found[args.case]
        measurement = measure_case(case.femur, case.tibia, case.side,
                                   case_id=case.case_id)

    table = hash_table(case, measurement, library, commit=not args.no_commit)
    record = {"case": case.case_id, "library": bool(library), **environment(),
              "hashes": table}
    for name, digest in table.items():
        print(f"{name:28s} {digest}")
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")

    if not args.compare:
        return 0
    reference = json.loads(Path(args.compare).read_text(encoding="utf-8"))
    print(f"\ncompared with {Path(args.compare).name} "
          f"({reference.get('platform', '?')}, numpy {reference.get('numpy', '?')}, "
          f"manifold3d {reference.get('manifold3d', '?')}):")
    differ = 0
    for name, same in compare(table, reference.get("hashes", {})):
        verdict = {True: "same", False: "DIFFERENT", None: "missing"}[same]
        differ += same is not True
        print(f"  {name:28s} {verdict}")
    print("identical" if not differ else f"{differ} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
