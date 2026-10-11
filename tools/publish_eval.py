"""Copy the demo and evaluation work from the private folder into the public eval/.

The work lives in docs/Prep for paper/eval/ (gitignored). This mirrors the part of it
that may be public into eval/: the scripts, the demo app, and the two synthetic demo
knees (DEMO_SMALL, DEMO_LARGE: bones and built parts). Patient knees, the reference
knee's own files, logs, Fusion audit output and one-off probes (_*.py) stay behind.
Absolute paths in the knees' JSON become paths relative to eval/. The copy is refused
if anything patient-identifying or machine-specific is left in it.

  python tools/publish_eval.py           # mirror into eval/
  python tools/publish_eval.py --dry-run # list what would change
"""

import argparse
import fnmatch
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "Prep for paper" / "eval"
DST = REPO / "eval"
DEMO = ("DEMO_SMALL", "DEMO_LARGE")
KEEP = {"README.md"}                    # written in eval/ by hand, never mirrored

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not SRC.exists():
        sys.exit(f"no private eval folder at {SRC}")
    files = collect()
    problems = check(files)
    if problems:
        print("Refusing to publish:\n  " + "\n  ".join(problems))
        sys.exit(1)

    existing = {p.relative_to(DST).as_posix() for p in DST.rglob("*") if p.is_file()} if DST.exists() else set()
    existing -= KEEP
    existing = {r for r in existing if "__pycache__" not in r}
    added = sorted(r for r in files if r not in existing)
    changed = sorted(r for r in files if r in existing and (DST / r).read_bytes() != files[r])
    removed = sorted(existing - set(files))
    for label, rels in (("add", added), ("update", changed), ("remove", removed)):
        for r in rels:
            print(f"{label:7} {r}")
    size = sum(len(d) for d in files.values()) / 2**20
    print(f"\n{len(files)} files, {size:.1f} MB: {len(added)} new, {len(changed)} changed, {len(removed)} removed")
    if args.dry_run:
        return
    for r in added + changed:
        (DST / r).parent.mkdir(parents=True, exist_ok=True)
        (DST / r).write_bytes(files[r])
    for r in removed:
        (DST / r).unlink()
    for d in sorted((p for p in DST.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    shutil.rmtree(DST / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
