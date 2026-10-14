"""``python -m paper list`` / ``python -m paper run E1 E2 ...`` / ``python -m paper run all``."""

from __future__ import annotations

import argparse
import sys

from . import experiments  # noqa: F401  (registers every experiment)
from .harness import REGISTRY, cohort_cases, cohort_root, out_root, run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper",
                                     description="Paper 2 evaluation harness.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the experiments and the cohort found")
    run = sub.add_parser("run", help="run experiments by id, or 'all'")
    run.add_argument("ids", nargs="+")
    run.add_argument("--quick", action="store_true",
                     help="smoke run: two cases, fewer samples")
    run.add_argument("--reference", action="store_true",
                     help="also include the full-length reference knee REF_KJ2")
    args = parser.parse_args(argv)

    if args.command == "list":
        for exp in sorted(REGISTRY.values(), key=lambda e: _order(e.experiment_id)):
            print(f"{exp.experiment_id:6} {'cohort' if exp.needs_cohort else '      '} "
                  f"{exp.title}")
        cases = cohort_cases(include_reference=True)
        print(f"\ncohort under {cohort_root()}: "
              f"{', '.join(c.case_id for c in cases) or 'none'}")
        print(f"results under {out_root()}")
        return 0

    ids = sorted(REGISTRY, key=_order) if args.ids == ["all"] else args.ids
    unknown = [i for i in ids if i not in REGISTRY]
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(unknown)}")
    for experiment_id in ids:
        print(f"== {experiment_id}: {REGISTRY[experiment_id].title}")
        out = run_experiment(experiment_id, quick=args.quick,
                             include_reference=args.reference)
        print(f"   -> {out}")
    return 0


def _order(experiment_id: str):
    digits = "".join(ch for ch in experiment_id if ch.isdigit())
    return (int(digits) if digits else 0, experiment_id)


if __name__ == "__main__":
    sys.exit(main())
