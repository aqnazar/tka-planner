"""The paper's paired statistics, against null distributions worked out by hand."""

import numpy as np
import pytest

from paper.stats import paired_summary, wilcoxon_signed_rank


def test_all_one_sign_gives_the_smallest_exact_p():
    # Five positive differences: only the all-positive and all-negative patterns are
    # as extreme, 2 of 32.
    result = wilcoxon_signed_rank([0.5, 1.0, 1.5, 2.0, 2.5])
    assert result["n"] == 5
    assert result["w_plus"] == 15.0
    assert result["p"] == pytest.approx(2 / 32)


def test_mixed_signs_match_the_enumerated_distribution():
    # Ranks 1..4, W+ = 1 + 3 + 4 = 8. Of the 16 patterns, W+ <= 2 or >= 8 in six.
    result = wilcoxon_signed_rank([1.0, -2.0, 3.0, 4.0])
    assert result["w_plus"] == 8.0
    assert result["p"] == pytest.approx(6 / 16)


def test_the_test_is_symmetric_in_sign():
    d = np.array([0.3, -1.2, 2.2, 4.1, -0.7, 5.0])
    assert wilcoxon_signed_rank(d)["p"] == pytest.approx(wilcoxon_signed_rank(-d)["p"])


def test_zeros_are_dropped_and_ties_share_ranks():
    result = wilcoxon_signed_rank([0.0, 1.0, 1.0, -1.0])
    assert result["n"] == 3
    # Three tied magnitudes share rank 2; W+ = 4 sits one from the null mean of 3, and
    # every pattern (W+ in 0, 2, 4, 6) is at least that far.
    assert result["w_plus"] == 4.0
    assert result["p"] == pytest.approx(1.0)


def test_no_difference_has_no_p():
    assert wilcoxon_signed_rank([0.0, 0.0])["p"] is None


def test_paired_summary_skips_incomplete_pairs():
    summary = paired_summary([1.0, 2.0, None, 4.0], [0.0, 1.0, 3.0, None])
    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["sd"] == pytest.approx(0.0)
    assert summary["min"] == summary["max"] == pytest.approx(1.0)


def test_paired_summary_of_nothing_is_empty():
    assert paired_summary([None], [1.0])["n"] == 0
