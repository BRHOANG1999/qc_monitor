"""Pre-ictal scoring: rank AUC, the bin-vs-bin AUC matrix + collapse gradient,
and ROC/PR forecasting scores under class imbalance.

Run with: pytest tests/test_preictal_scoring.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal import scoring  # noqa: E402


def test_rank_auc_separation_and_ties():
    rng = np.random.default_rng(0)
    hi = rng.normal(10, 1, 500)
    lo = rng.normal(0, 1, 500)
    assert scoring.rank_auc(hi, lo) > 0.98        # well separated
    assert scoring.rank_auc(lo, hi) < 0.02        # reversed
    assert abs(scoring.rank_auc(lo, lo) - 0.5) < 0.05   # identical -> 0.5
    assert np.isnan(scoring.rank_auc([], lo))     # empty -> NaN


def test_monotonic_ramp_gives_banded_matrix_and_positive_collapse():
    # 6 lead-time bins, nearest first, means ramping DOWN with index (nearest =
    # highest) -> a monotonic pre-ictal ramp toward onset.
    rng = np.random.default_rng(1)
    bins = [rng.normal(10 - k, 1.0, 300) for k in range(6)]
    m = scoring.auc_matrix(bins)
    assert m.shape == (6, 6)
    assert np.allclose(np.diag(m), 0.5)
    # nearer (lower index) beats farther -> upper triangle > 0.5, and grows with
    # distance (banded gradient).
    assert m[0, 5] > m[0, 1] > 0.5
    c = scoring.collapse_gradient(m)
    assert c > 0.2                                # strong positive gradient


def test_flat_bins_collapse_near_zero():
    rng = np.random.default_rng(2)
    bins = [rng.normal(5, 1.0, 300) for _ in range(6)]   # no lead-time trend
    c = scoring.collapse_gradient(scoring.auc_matrix(bins))
    assert abs(c) < 0.06


def test_forecasting_roc_and_pr():
    rng = np.random.default_rng(3)
    near = rng.normal(6, 1, 200)                   # pre-ictal (positive)
    far = rng.normal(0, 1, 2000)                   # interictal (10x, imbalanced)
    roc, pr = scoring.forecasting_scores(near, far)
    assert roc > 0.98
    assert 0.9 < pr <= 1.0                          # sane PR under imbalance
    # No separation -> ROC ~0.5 and PR ~ base rate (positives / total).
    roc0, pr0 = scoring.forecasting_scores(rng.normal(0, 1, 200),
                                           rng.normal(0, 1, 2000))
    assert abs(roc0 - 0.5) < 0.06
    assert pr0 < 0.25                              # near the 200/2200 base rate
