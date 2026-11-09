"""Bring the online landmark study's saves home, through the planner's own save_picks.

Each save on the site is kept as the surgeon's browser sent it. Here it is written by
:func:`tka_planner.study.save_picks` on a session of that knee, exactly as a local save
is: the same files, the same surface-gap check against the full-resolution bone, the
same distances from the automatic positions. The file names carry the time the surgeon
saved, and the source says ``"via": "online"``. Saves already pulled are skipped (the
ledger is ``<output>/online_pulled.json``), so running it again is harmless.

    python -m online.pull --site https://<site>.netlify.app
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

from online.build_site import STUDY, REPO
from tka_planner.server.sessions import SessionManager
from tka_planner.study import save_picks


def _call(site: str, admin_key: str, body=None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{site.rstrip('/')}/api/list", data=data, method="GET" if data is None else "POST",
        headers={"x-admin-key": admin_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_new(site: str, admin_key: str, done: set) -> list:
    """The saves not yet pulled, a hundred keys per request."""
    keys = [key for key in _call(site, admin_key)["keys"] if key not in done]
    records = []
    for start in range(0, len(keys), 100):
        records += _call(site, admin_key, {"keys": keys[start:start + 100]})["records"]
    return records


def case_folder(case_id: str, roots) -> Path:
    for root in roots:
        folder = Path(root) / case_id
        if folder.is_dir():
            return folder
    raise FileNotFoundError(f"No folder for {case_id} under {', '.join(map(str, roots))}")


def write_saves(records, *, output: Path, roots, manager: SessionManager, log=print) -> list:
    """Write each record through save_picks; return the keys written."""
    ledger_path = output / "online_pulled.json"
    ledger = json.loads(ledger_path.read_text("utf-8")) if ledger_path.is_file() else {}
    written, sessions = [], {}
    for record in records:
        key, value = record["key"], record["value"]
        if key in ledger:
            continue
        knee = (value["case_id"], value["side"])
        if knee not in sessions:
            sessions[knee] = manager.open(case_folder(value["case_id"], roots),
                                          side=value["side"])
        when = dt.datetime.fromisoformat(value["received_utc"].replace("Z", "+00:00"))
        result = save_picks(sessions[knee], output, value["body"], saved_utc=when,
                            via="online")
        ledger[key] = {"landmarks": Path(result["landmarks"]).name,
                       "pulled_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        written.append(key)
        log(f"{value['case_id']} {value['side']} {value['body']['mode']} "
            f"r{value['body']['round']}: {result['placed']} placed, {result['skipped']} skipped")
        output.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2), "utf-8")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True, help="the study site's address")
    parser.add_argument("--secrets", type=Path, default=STUDY / "online_site" / "secrets.json")
    parser.add_argument("--output", type=Path, default=STUDY / "output")
    parser.add_argument("--store", type=Path, default=STUDY / "store")
    args = parser.parse_args(argv)
    admin_key = json.loads(args.secrets.read_text("utf-8"))["admin_key"]
    ledger_path = args.output / "online_pulled.json"
    manager = SessionManager(library=REPO / "data" / "implants", store=args.store,
                             study_dir=args.output)
    roots = (STUDY / "cases", STUDY / "vsd_cases")
    done = set(json.loads(ledger_path.read_text("utf-8"))) if ledger_path.is_file() else set()
    total = len(write_saves(fetch_new(args.site, admin_key, done), output=args.output,
                            roots=roots, manager=manager))
    print(f"{total} new saves written to {args.output}")


if __name__ == "__main__":
    main()
