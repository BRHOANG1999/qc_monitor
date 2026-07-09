"""Discriminability scoring: the N x N lead-time-bin AUC matrix + its collapse.

Per CWT scale we ask, for each pair of LEAD-TIME bins (i nearer onset, j
farther): how distinguishable is the coefficient magnitude between them? A rank
AUC (Mann-Whitney) reads structure in log-lead-time -- a banded gradient means a
monotonic ramp toward onset, a block means a discrete state change, ~0.5
everywhere means no signal. The farthest bin reaches interictal, so the
near-vs-far column recovers the standard forecasting AUC (ROC + PR, since the
pre-ictal/interictal base rates are imbalanced).

Bin convention: index 0 = NEAREST to onset, increasing index = farther back
(matches isi.leadtime_bins' ascending lead-time edges).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def rank_auc(a, b) -> float:
    """Rank AUC = P(a > b) (+ 0.5 P(tie)) via Mann-Whitney U. 0.5 = no
    difference, 1 = a strictly above b, 0 = below. NaN if either side empty."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        u, _ = mannwhitneyu(a, b, alternative="greater")
    except ValueError:                      # all-equal within/between -> tie
        return 0.5
    return float(u / (a.size * b.size))


def auc_matrix(bin_values: list) -> np.ndarray:
    """N x N matrix M[i,j] = P(bin_i > bin_j). Diagonal 0.5; NaN where a bin is
    empty. *bin_values*[k] is the pooled |coeff| for lead-time bin k (nearest
    first)."""
    n = len(bin_values)
    m = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            m[i, j] = 0.5 if i == j else rank_auc(bin_values[i], bin_values[j])
    return m


def collapse_gradient(auc: np.ndarray) -> float:
    """Collapse the AUC matrix to a signed monotonicity scalar: the mean over
    all pairs where bin i is NEARER to onset than bin j (i < j) of
    ``AUC[i,j] - 0.5``. > 0 => the coefficient ramps UP toward onset (a pre-ictal
    gradient); ~0 => no lead-time structure; < 0 => ramps down. NaN if no
    scoreable pair."""
    auc = np.asarray(auc, dtype=float)
    n = auc.shape[0]
    vals = [auc[i, j] - 0.5
            for i in range(n) for j in range(i + 1, n)
            if np.isfinite(auc[i, j])]
    return float(np.mean(vals)) if vals else float("nan")


def _average_precision(pos, neg) -> float:
    """PR-AUC (average precision): positives should score higher. Manual, no
    sklearn dependency. NaN if no positives."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0:
        return float("nan")
    if neg.size == 0:
        return 1.0
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="mergesort")     # stable, high score first
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / pos.size
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def forecasting_scores(near_values, far_values) -> tuple[float, float]:
    """(ROC-AUC, PR-AUC) for discriminating the NEAREST-to-onset bin (pre-ictal,
    positive) from the FARTHEST bin (interictal-ish, negative), scoring by
    |coeff|. ROC is the rank AUC (imbalance-insensitive); PR (average precision)
    is reported alongside for the imbalanced base rates."""
    return rank_auc(near_values, far_values), _average_precision(near_values,
                                                                 far_values)
