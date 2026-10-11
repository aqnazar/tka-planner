"""Copy the demo and evaluation work from the private folder into the public eval/.

The work lives in docs/Prep for paper/eval/ (gitignored). This mirrors the part of it
that may be public into eval/: the scripts, the demo app, and the two synthetic demo
knees (DEMO_SMALL, DEMO_LARGE: bones and built parts). Patient knees, the reference
knee's own files, logs, Fusion audit output and one-off probes (_*.py) stay behind.
Absolute paths in the knees' JSON become paths relative to eval/. The copy is refused
if anything patient-identifying or machine-specific is left in it.

It also mirrors the implant catalogue, all twelve sizes, into data/implants/, where the
launcher and the demo find it: the implants, cutting blocks and shells, SizeChart.csv and
femoral_box.json. The bones exported with each size (FD1*, TD1*) and print projects
(.3mf, which carry the slicer account) stay behind.

  python tools/publish_eval.py           # mirror into eval/ and data/implants/
  python tools/publish_eval.py --dry-run # list what would change
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "Prep for paper" / "eval"
DST = REPO / "eval"
DEMO = ("DEMO_SMALL", "DEMO_LARGE")
KEEP = {"README.md"}                    # written in eval/ by hand, never mirrored
# The catalogue as exported (TKA_LIBRARY_SOURCE overrides), and where the repo carries it.
LIB_SRC = Path(os.environ.get("TKA_LIBRARY_SOURCE") or REPO.parent / "BP" / "Scaled_STLs")
LIB_DST = REPO / "data" / "implants"
LIB_FILES = {"SizeChart.csv", "femoral_box.json"}

# Copied as they are (glob patterns relative to the private eval/).
FILES = ["*.py", "demo_app/*.html", "demo_app/*.js", "demo_app/*.css",
         "demo_app/fonts/*", "demo_app/vendor/**/*",
         "tray_audit/copies.json"]      # the Fusion documents' ids, read on import by the build
for knee in DEMO:
    FILES += [f"demo_app/data/{knee}/*", f"demo_cases/{knee}/*", f"patient_builds/{knee}/*"]
SKIP = ["_*.py", "*.log", "*.pyc"]

# Rewritten to the demo knees only.
FILTERED = {
    "demo_app/data/manifest.json": lambda m: {"knees": [k for k in m["knees"] if k["name"] in DEMO]},
    "patient_builds/jobs_demo.json": lambda jobs: [j for j in jobs if j["name"] in DEMO],
    "patient_builds/fit_check.json": lambda d: {k: v for k, v in d.items() if k in DEMO},
}

# Nothing that names a patient, the dataset or this machine may leave.
FORBIDDEN = [re.compile(p) for p in (r"Patient[ _]Dataset", r"Patient_0\d\d", r"[A-Za-z]:[\\/]+Users",
                                     r"aknaz", r"Prep for paper")]
CASE_ID = re.compile(r"\bP0\d\d[LR]\b")          # cohort knees: allowed in code, not in data
CODE_MAY_MENTION = {"eval_stage_b.py": [r"Patient[ _]Dataset", r"Patient_0\d\d"],   # where the cohort is read
                    "shoot.py": [r"Prep for paper"],
                    "femoral_parameter_list.py": [r"Prep for paper"]}
TEXT = {".py", ".js", ".json", ".html", ".css", ".md", ".txt"}
GITHUB_LIMIT = 100 * 1024 * 1024


def relativise(value):
    """Absolute paths into the private eval/ become eval/-relative POSIX paths."""
    if isinstance(value, dict):
        return {relativise(k): relativise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relativise(v) for v in value]
    if isinstance(value, str) and len(value) < 1000:
        try:
            return Path(value).resolve().relative_to(SRC.resolve()).as_posix()
        except (ValueError, OSError):
            return value
    return value


def collect():
    """{relative path: bytes} of everything the public eval/ should hold."""
    out = {}
    for pattern in FILES:
        for p in sorted(SRC.glob(pattern)):
            rel = p.relative_to(SRC).as_posix()
            if p.is_file() and not any(fnmatch.fnmatch(p.name, s) for s in SKIP):
                data = p.read_bytes()
                if p.suffix == ".json":
                    data = (json.dumps(relativise(json.loads(data)), indent=1) + "\n").encode()
                out[rel] = data
    for rel, keep in FILTERED.items():
        out[rel] = (json.dumps(relativise(keep(json.loads((SRC / rel).read_text(encoding="utf-8")))),
                               indent=1) + "\n").encode()
    return out


def collect_library():
    """{relative path: bytes} of the catalogue: every size's parts, without its bones."""
    out = {}
    for p in sorted(LIB_SRC.rglob("*")):
        bone = p.name.upper().startswith(("FD1", "TD1"))
        if p.is_file() and (p.name in LIB_FILES or (p.suffix.lower() == ".stl" and not bone)):
            out[p.relative_to(LIB_SRC).as_posix()] = p.read_bytes()
    return out


def check(files):
    problems = []
    for rel, data in files.items():
        if len(data) >= GITHUB_LIMIT:
            problems.append(f"{rel}: {len(data) / 2**20:.0f} MB, over GitHub's 100 MB limit")
        if Path(rel).suffix not in TEXT:
            continue
        text = data.decode("utf-8", errors="replace")
        allowed = CODE_MAY_MENTION.get(rel, [])
        for rx in FORBIDDEN:
            if rx.pattern not in allowed and rx.search(text):
                problems.append(f"{rel}: contains {rx.search(text).group(0)!r}")
        if rel.endswith(".json") and CASE_ID.search(text):
            problems.append(f"{rel}: names cohort knee {CASE_ID.search(text).group(0)}")
    return problems


def mirror(files, dst, dry_run, keep=frozenset()):
    """Make dst hold exactly files (apart from keep); print what changes."""
    existing = {p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file()} if dst.exists() else set()
    existing = {r for r in existing - set(keep) if "__pycache__" not in r}
    added = sorted(r for r in files if r not in existing)
    changed = sorted(r for r in files if r in existing and (dst / r).read_bytes() != files[r])
    removed = sorted(existing - set(files))
    name = dst.relative_to(REPO).as_posix()
    for label, rels in (("add", added), ("update", changed), ("remove", removed)):
        for r in rels:
            print(f"{label:7} {name}/{r}")
    size = sum(len(d) for d in files.values()) / 2**20
    print(f"{name}/: {len(files)} files, {size:.1f} MB: {len(added)} new, {len(changed)} changed, "
          f"{len(removed)} removed\n")
    if dry_run:
        return
    for r in added + changed:
        (dst / r).parent.mkdir(parents=True, exist_ok=True)
        (dst / r).write_bytes(files[r])
    for r in removed:
        (dst / r).unlink()
    for d in sorted((p for p in dst.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    shutil.rmtree(dst / "__pycache__", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for folder in (SRC, LIB_SRC):
        if not folder.exists():
            sys.exit(f"no folder at {folder}")
    if LIB_SRC.resolve() == LIB_DST.resolve():
        sys.exit("TKA_LIBRARY_SOURCE points at data/implants itself; point it at the exported catalogue")
    files, library = collect(), collect_library()
    problems = check(files) + check({f"data/implants/{r}": d for r, d in library.items()})
    if problems:
        print("Refusing to publish:\n  " + "\n  ".join(problems))
        sys.exit(1)
    mirror(files, DST, args.dry_run, keep=KEEP)
    mirror(library, LIB_DST, args.dry_run)


if __name__ == "__main__":
    main()
