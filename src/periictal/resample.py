"""Dependence-aware resampling: block bootstrap + block sizing.

Consecutive stimuli within one seizure are autocorrelated, so a trial-level
(iid) bootstrap or permutation understates uncertainty and fabricates a tight
CI (Kunsch 1989; the effective sample size shrinks by ~1+2*sum rho_k -- see
docs/periictal_methods_evidence.md, E1/E2). These helpers size a block from the
integrated autocorrelation time of the series and resample CONTIGUOUS blocks, so
the within-seizure dependence is preserved rather than shuffled away.

Pure + numpy-only so it is unit-testable and shared by forecast.py (per-seizure
AUC forest) and trendtest.py (per-seizure slope forest + circular-shift null).
"""

from __future__ import annotations

import numpy as np

_MAX_LAG = 500          # NASA Rule 2: explicit bound on the autocorr scan.


def autocorr_time(x) -> float:
    """Integrated autocorrelation time ``tau = 1 + 2*sum_{k>=1} rho_k``, summed
    until the first non-positive ``rho_k`` (the standard initial-positive-sequence
    truncation). Returns ``>= 1.0``; 1.0 for white/degenerate input. This is the
    block length a moving-block bootstrap should use."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0.0:
        return 1.0
    tau = 1.0
    max_lag = min(n - 1, _MAX_LAG)
    for k in range(1, max_lag + 1):
        assert k <= _MAX_LAG, "autocorr scan runaway"
        rho = float(np.dot(x[:-k], x[k:]) / var)
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(max(1.0, tau))


def block_length(x) -> int:
    """Moving-block length for series *x*: ``round(autocorr_time)`` clamped to
    ``[1, n//2]`` so there are always at least ~2 blocks."""
    n = int(np.isfinite(np.asarray(x, dtype=float)).sum())
    assert n >= 0, "n must be non-negative"
    if n < 2:
        return 1
    return int(max(1, min(round(autocorr_time(x)), max(1, n // 2))))


def moving_block_resample(x, block: int, rng) -> np.ndarray:
    """One moving-block bootstrap resample of *x*: draw ceil(n/block) contiguous
    blocks of length *block* from random starts (with replacement) and trim to n.
    Preserves within-block autocorrelation."""
    x = np.asarray(x)
    n = x.size
    assert rng is not None, "rng required"
    if n == 0:
        return x.copy()
    block = int(max(1, min(int(block), n)))
    n_blocks = int(np.ceil(n / block))
    hi = n - block + 1
    starts = (rng.integers(0, hi, size=n_blocks) if hi > 0
              else np.zeros(n_blocks, dtype=int))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return x[idx]


def block_index(n: int, block: int, rng) -> np.ndarray:
    """Row indices for a moving-block resample of a length-*n* series -- use to
    resample PAIRED columns (x, y) together so their relationship is preserved."""
    assert n >= 0, "n must be non-negative"
    assert rng is not None, "rng required"
    return moving_block_resample(np.arange(n), block, rng)
