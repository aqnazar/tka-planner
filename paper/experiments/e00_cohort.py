"""E0: the cohort as the planner sees it.

For each knee: mesh sizes, bone segment lengths, whether the planner accepts it, the
quality-control findings, and how many landmarks are measured, estimated or missing.
This is the Methods table describing the data, and the denominator for every later
experiment: a knee refused here is reported as refused there too.
"""

from __future__ import annotations

import time

from ..harness import Context, experiment, measured


@experiment("E0", "Cohort inventory and acceptance")
def run(ctx: Context) -> None:
    rows = []
    for case in ctx.cases:
        row = {"case": case.case_id, "side": case.side}
        t0 = time.perf_counter()
        m = measured(case)
        if isinstance(m, Exception):
            refusal = m
            row.update(accepted=False, refusal=type(refusal).__name__,
                       reason=str(refusal).splitlines()[0][:160])
            rows.append(row)
            continue
        row["measure_s"] = round(time.perf_counter() - t0, 2)
        row.update(accepted=True,
                   femur_triangles=m.femur.n_triangles,
                   tibia_triangles=m.tibia.n_triangles,
                   femur_length_mm=float(m.femur.extent[2]),
                   tibia_length_mm=float(m.tibia.extent[2]),
                   qc_warnings=len(m.qc.warnings),
                   qc_codes=" ".join(sorted({f.code for f in m.qc.findings})))
        for status, count in m.landmark_counts().items():
            row[f"landmarks_{status}"] = count
        rows.append(row)
    ctx.table("cohort", rows, digits=1)
    accepted = sum(r["accepted"] for r in rows)
    ctx.note(f"{accepted} of {len(rows)} knees accepted by the planner.")
    for r in rows:
        if not r["accepted"]:
            ctx.note(f"- {r['case']} refused ({r['refusal']}): {r['reason']}")
