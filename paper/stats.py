"""Paired statistics for the paper's per-knee comparisons, in numpy alone.

The cohort is ten knees, so every comparison between two methods is a small paired
sample. The paper reports the differences themselves (mean, SD, range) and, as the test,
the Wilcoxon signed-rank test with its **exact** null distribution: with ten pairs there
are only 1024 sign patterns, so enumerating them is cheaper and more honest than the
normal approximation, which is poor at this size. scipy is not a dependency of the
planner or of the harness, so the test is written out here.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["format_p", "paired_summary", "wilcoxon_signed_rank"]

# Above this many non-zero pairs the 2**n enumeration stops being cheap. The cohort is
# far below it; a larger sample would want the normal approximation instead.
EXACT_LIMIT = 20


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def wilcoxon_signed_rank(differences, *, zero_tolerance: float = 1e-9) -> dict:
    """Exact two-sided Wilcoxon signed-rank test that the differences centre on zero.

    Zero differences are dropped (Wilcoxon's own treatment); tied magnitudes share their
    average rank, and the exact null distribution is enumerated with those same ranks,
    so ties are handled exactly rather than approximately. Returns ``n`` (pairs used),
    ``w_plus`` (sum of the ranks of positive differences) and ``p`` (two-sided), with
    ``p`` ``None`` when no pair differs.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > zero_tolerance]
    n = int(len(d))
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "p": None}
    if n > EXACT_LIMIT:
        raise ValueError(f"{n} pairs: the exact test enumerates 2**n sign patterns; "
                         f"use a normal approximation above {EXACT_LIMIT}.")
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    total = float(ranks.sum())
    # Under the null each rank is positive or negative with equal chance.
    signs = np.array(list(itertools.product((0.0, 1.0), repeat=n)))
    null = signs @ ranks
    extreme = abs(w_plus - total / 2.0)
    p = float(np.mean(np.abs(null - total / 2.0) >= extreme - 1e-9))
    return {"n": n, "w_plus": w_plus, "p": min(p, 1.0)}


def format_p(summary: dict) -> str:
    """``exact Wilcoxon p = 0.002, n = 10`` from a :func:`paired_summary`."""
    if summary.get("wilcoxon_p") is None:
        return "no pair differs, no test"
    return f"exact Wilcoxon p = {summary['wilcoxon_p']:.3g}, n = {summary['wilcoxon_n']}"


def paired_summary(a, b) -> dict:
    """Mean, SD, min and max of ``a - b`` over the pairs both have, and the test.

    ``None`` entries (a knee one method could not measure) are left out of both, so the
    statistics are over complete pairs only, and ``n`` says how many.
    """
    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None
             and np.isfinite(float(x)) and np.isfinite(float(y))]
    if not pairs:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None,
                "wilcoxon_n": 0, "wilcoxon_p": None}
    diff = np.array([x - y for x, y in pairs])
    test = wilcoxon_signed_rank(diff)
    return {
        "n": int(len(diff)),
        "mean": float(diff.mean()),
        "sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
        "min": float(diff.min()),
        "max": float(diff.max()),
        "wilcoxon_n": test["n"],
        "wilcoxon_p": test["p"],
    }
