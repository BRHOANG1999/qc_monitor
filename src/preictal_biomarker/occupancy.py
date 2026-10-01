"""State-occupancy analysis + the circular-shift (seizure-time-shuffle) null.

Observed quantity: the fraction of pre-ictal epochs spent in each state, vs a
clean-baseline (>=2 h from any seizure) reference. The honest test keeps the
state time-series fixed and rigidly rotates the *seizure onset times* by a random
offset (wrapping around the record span) many times, re-deriving the pre-ictal
window each time. This preserves the state autocorrelation AND the inter-seizure
spacing; it destroys only the specific seizure->state alignment -- so it answers
"is the pre-ictal occupancy special, or what you'd get aligning random times to
the same data?". With n=7 seizures the null band is wide; the observed bars sit
inside it (the documented NS result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def occupancy_pct(states: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Percent of in-mask, validly-stated epochs in each state (0..k-1)."""
    s = np.asarray(states)[mask]
    s = s[s >= 0]
    if s.size == 0:
        return np.full(k, np.nan)
    return np.array([100.0 * np.mean(s == j) for j in range(k)])


def _preictal_mask_for(t: np.ndarray, onsets: np.ndarray,
                       preictal_sec: float) -> np.ndarray:
    """Epochs whose time-to-next-onset is within (0, preictal_sec]."""
    ons = np.sort(np.asarray(onsets, float))
    if ons.size == 0:
        return np.zeros(t.size, bool)
    idx = np.searchsorted(ons, t, side="left")      # next onset at/after t
    nxt = np.where(idx < ons.size, ons[np.clip(idx, 0, ons.size - 1)], np.inf)
    delta = nxt - t
    return (delta > 0) & (delta <= preictal_sec)


def observed(df: pd.DataFrame, pre_mask: np.ndarray, base_mask: np.ndarray,
             k: int | None = None) -> dict:
    """Observed pre-ictal and baseline occupancy per state."""
    k = int(k or C.K_STATES)
    st = df["state"].to_numpy()
    return {"preictal": occupancy_pct(st, pre_mask, k),
            "baseline": occupancy_pct(st, base_mask, k),
            "n_pre": int(np.sum((st >= 0) & pre_mask)),
            "n_base": int(np.sum((st >= 0) & base_mask))}


def circular_shift_null(df: pd.DataFrame, onsets: np.ndarray, *,
                        k: int | None = None, n_surr: int = 2000,
                        seed: int = 0) -> dict:
    """Circular-shift null for pre-ictal occupancy. Returns observed, the full
    null matrix [n_surr, k], its median / 2.5-97.5 band, and a two-sided p per
    state (fraction of surrogates at least as far from the null median)."""
    k = int(k or C.K_STATES)
    rng = np.random.default_rng(seed)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    lo, hi = np.nanmin(t), np.nanmax(t)
    span = hi - lo
    ons = np.sort(np.asarray(onsets, float))
    obs = occupancy_pct(st, _preictal_mask_for(t, ons, C.PREICTAL_SEC), k)
    null = np.full((n_surr, k), np.nan)
    for i in range(n_surr):                              # bounded loop
        shift = rng.uniform(0, span)
        shifted = lo + ((ons - lo + shift) % span)      # wrap within the span
        m = _preictal_mask_for(t, shifted, C.PREICTAL_SEC)
        null[i] = occupancy_pct(st, m, k)
    med = np.nanmedian(null, axis=0)
    loq = np.nanpercentile(null, 2.5, axis=0)
    hiq = np.nanpercentile(null, 97.5, axis=0)
    p = np.array([_two_sided_p(null[:, j], obs[j], med[j]) for j in range(k)])
    return {"observed": obs, "null": null, "median": med,
            "lo": loq, "hi": hiq, "p": p, "n_surr": int(n_surr)}


def _two_sided_p(null_col: np.ndarray, obs: float, med: float) -> float:
    """Fraction of surrogates at least as far from the null median as observed
    (+1 smoothed). NaN-safe."""
    x = null_col[np.isfinite(null_col)]
    if x.size == 0 or not np.isfinite(obs):
        return np.nan
    ge = np.sum(np.abs(x - med) >= abs(obs - med))
    return float((ge + 1) / (x.size + 1))
