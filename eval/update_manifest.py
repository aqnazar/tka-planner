"""Refresh named knees' entries in the demo manifest (demo_build.knee_entry), adding any
knee of jobs_demo.json not in it yet, and keeping the others as they are.

  python update_manifest.py <knee ...>
"""

import json
import sys

from demo_build import DATA, JOBS, knee_entry, vendor


def main(names):
    vendor()
    path = DATA / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    jobs = {j["name"]: j for j in json.loads(JOBS.read_text())}
    entries = {e["name"]: e for e in man["knees"]}
    for name in names:
        entries[name] = knee_entry(jobs[name])
        print(name, [p["id"] for p in entries[name]["parts"]], flush=True)
    man["knees"] = [entries[n] for n in jobs if n in entries]
    path.write_text(json.dumps(man, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
