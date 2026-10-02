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


def seizure_prob_by_state(df: pd.DataFrame, onsets, *, horizons_sec=None,
                          k: int | None = None) -> dict:
    """Forward predictive map: P(a seizure onset within H | current state) for
    each state and each horizon H, with the marginal base rate P(sz within H).
    tto = time to the NEXT onset (uncapped). A state is predictive for horizon H
    when P(sz|state) > base rate (lift > 1).

    NOTE: computed over whatever epochs are in df; if df is the peri-ictal matrix
    the base rate is the within-monitoring-window rate (inflated vs the true
    population). Use a full-record df for a population base rate."""
    k = int(k or C.K_STATES)
    horizons_sec = horizons_sec or [30, 60, 300, 600, 1800, 3600]
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    ons = np.sort(np.asarray(onsets, float))
    idx = np.searchsorted(ons, t, side="left")
    nxt = np.where(idx < ons.size, ons[np.clip(idx, 0, ons.size - 1)], np.inf)
    tto = nxt - t                                        # time to next onset (s)
    ok = (st >= 0) & np.isfinite(t)
    P = np.full((k, len(horizons_sec)), np.nan)
    base = np.full(len(horizons_sec), np.nan)
    for h, H in enumerate(horizons_sec):
        hit = (tto > 0) & (tto <= H)
        base[h] = 100.0 * np.mean(hit[ok])
        for s in range(k):
            m = ok & (st == s)
            if m.sum():
                P[s, h] = 100.0 * np.mean(hit[m])
    n_state = np.array([int((ok & (st == s)).sum()) for s in range(k)])
    return {"P": P, "base": base, "horizons_sec": list(horizons_sec),
            "n_state": n_state}


def seizure_prob_null(df: pd.DataFrame, onsets, *, horizons_sec=None,
                      k: int | None = None, n_surr: int = 500, seed: int = 0) -> dict:
    """Shift-null for the per-state forward lift P(sz within H|state)/base: is a
    state's elevated seizure probability more than random onset alignments give?
    Returns observed lift [k,H] and a one-sided p per state×horizon."""
    k = int(k or C.K_STATES)
    hs = horizons_sec or [30, 60, 300, 600, 1800, 3600]
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    lo, hi = np.nanmin(t), np.nanmax(t); span = hi - lo
    ons = np.sort(np.asarray(onsets, float))
    obs = seizure_prob_by_state(df, ons, horizons_sec=hs, k=k)
    obs_lift = obs["P"] / np.where(obs["base"] > 0, obs["base"], np.nan)
    rng = np.random.default_rng(seed)
    nl = np.full((n_surr, k, len(hs)), np.nan)
    for i in range(n_surr):
        sh = lo + ((ons - lo + rng.uniform(0, span)) % span)
        r = seizure_prob_by_state(df, sh, horizons_sec=hs, k=k)
        nl[i] = r["P"] / np.where(r["base"] > 0, r["base"], np.nan)
    p = np.full((k, len(hs)), np.nan)
    for s in range(k):
        for h in range(len(hs)):
            col = nl[:, s, h][np.isfinite(nl[:, s, h])]
            if col.size and np.isfinite(obs_lift[s, h]):
                p[s, h] = (np.sum(col >= obs_lift[s, h]) + 1) / (col.size + 1)
    return {"horizons_sec": hs, "obs_lift": obs_lift, "p": p,
            "n_state": obs["n_state"], "n_surr": int(n_surr)}


_RISK_EDGES_MIN = (1.0, 5.0, 10.0, 30.0, 60.0)       # disjoint bins + 'none' (>60)


def _risk_bin_labels(t: np.ndarray, onsets, edges_min=_RISK_EDGES_MIN):
    """Disjoint time-to-NEXT-onset bin per epoch: (0,e0],(e0,e1],...,(e_{n-2},e_{n-1}],
    and 'none' (>last edge, <=0, or no next onset). Returns (labels[int], n_bins,
    names). Seizure times are FIXED here -- only the states get shifted in the null."""
    ons = np.sort(np.asarray(onsets, float))
    if ons.size == 0:
        return np.full(t.size, len(edges_min), int), len(edges_min) + 1, \
            _risk_bin_names(edges_min)
    idx = np.searchsorted(ons, t, side="left")
    nxt = np.where(idx < ons.size, ons[np.clip(idx, 0, ons.size - 1)], np.inf)
    dmin = (nxt - t) / 60.0
    lab = np.full(t.size, len(edges_min), int)           # default 'none'
    prev = 0.0
    for i, e in enumerate(edges_min):
        lab[(dmin > prev) & (dmin <= e)] = i
        prev = e
    return lab, len(edges_min) + 1, _risk_bin_names(edges_min)


def _risk_bin_names(edges_min=_RISK_EDGES_MIN) -> list:
    names, prev = [], 0.0
    for e in edges_min:
        names.append(f"{int(prev)}-{int(e)}"); prev = e
    return names + ["none"]


def _lift_table(state: np.ndarray, lab: np.ndarray, k: int, nb: int) -> np.ndarray:
    """lift(state, bin) = P(bin | state) / P(bin); NaN where a cell is empty.
    Vectorized (bincount over state*nb+bin) so 1000 surrogates stay cheap."""
    ok = state >= 0
    s, b = state[ok], lab[ok]
    n = s.size
    if n == 0:
        return np.full((k, nb), np.nan)
    cnt = np.bincount(s * nb + b, minlength=k * nb).reshape(k, nb).astype(float)
    ns = cnt.sum(axis=1, keepdims=True)                  # per-state totals
    pb = cnt.sum(axis=0) / n                             # marginal P(bin)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (cnt / np.where(ns > 0, ns, np.nan)) / np.where(pb > 0, pb, np.nan)
    return L


def state_risk_bin_null(df: pd.DataFrame, onsets, *, edges_min=_RISK_EDGES_MIN,
                        k: int | None = None, n_surr: int = 1000,
                        min_shift_sec: float = 7200.0, seed: int = 0) -> dict:
    """Is the state->seizure-risk link real, or an artifact of state autocorrelation?

    Keeps seizure times (and thus each epoch's disjoint time-to-seizure bin) FIXED,
    and circularly shifts the STATE sequence in TIME by a random offset (wrapped,
    >= min_shift_sec so near-seizure epochs decouple), re-pairing shifted states to
    fixed bins. The state x bin lift table is recomputed per shift (n_surr times);
    the 2.5-97.5 percentiles are the chance band and p = fraction of shifts whose
    lift >= observed (one-sided enrichment). If observed sits inside the band, the
    state->risk association has disappeared under the null."""
    k = int(k or C.K_STATES)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    s = df["state"].to_numpy()
    fin = np.isfinite(t)
    t, s = t[fin], s[fin]
    order = np.argsort(t); t, s = t[order], s[order]
    lab, nb, names = _risk_bin_labels(t, onsets, edges_min)
    span = t[-1] - t[0]
    assert span > 2 * min_shift_sec, "record span too short for the 2 h shift floor"
    obs = _lift_table(s, lab, k, nb)
    rng = np.random.default_rng(seed)
    null = np.full((n_surr, k, nb), np.nan)
    for i in range(n_surr):                              # bounded loop
        delta = rng.uniform(min_shift_sec, span - min_shift_sec)
        rot = t[0] + ((t - delta - t[0]) % span)        # wrapped rotated time
        j = np.clip(np.searchsorted(t, rot, side="left"), 0, t.size - 1)
        jm = np.clip(j - 1, 0, t.size - 1)
        pick = np.where(np.abs(t[j] - rot) <= np.abs(rot - t[jm]), j, jm)
        null[i] = _lift_table(s[pick], lab, k, nb)       # states shifted, bins fixed
    lo = np.nanpercentile(null, 2.5, axis=0)
    hi = np.nanpercentile(null, 97.5, axis=0)
    med = np.nanmedian(null, axis=0)
    p = np.full((k, nb), np.nan)
    for si in range(k):
        for j in range(nb):
            col = null[:, si, j][np.isfinite(null[:, si, j])]
            if col.size and np.isfinite(obs[si, j]):
                p[si, j] = (np.sum(col >= obs[si, j]) + 1) / (col.size + 1)
    n_state = np.array([int((s == si).sum()) for si in range(k)])
    n_bin = np.array([int((lab == j).sum()) for j in range(nb)])
    return {"obs": obs, "lo": lo, "hi": hi, "med": med, "p": p, "names": names,
            "n_bins": nb, "n_state": n_state, "n_bin": n_bin, "n_surr": int(n_surr),
            "min_shift_sec": float(min_shift_sec), "edges_min": list(edges_min)}


def _two_sided_p(null_col: np.ndarray, obs: float, med: float) -> float:
    """Fraction of surrogates at least as far from the null median as observed
    (+1 smoothed). NaN-safe."""
    x = null_col[np.isfinite(null_col)]
    if x.size == 0 or not np.isfinite(obs):
        return np.nan
    ge = np.sum(np.abs(x - med) >= abs(obs - med))
    return float((ge + 1) / (x.size + 1))
