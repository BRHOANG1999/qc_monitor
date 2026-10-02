"""Fairer test than cluster occupancy: is pre-ictal SEPARABLE from baseline?

Trains a logistic classifier on the full (standardized) feature set to tell
pre-ictal epochs (<=30 min before a lead onset) from clean-baseline epochs
(>=2 h from any seizure), scored by AUC with **leave-one-seizure-out** so the
pseudoreplicated per-epoch dependence can't inflate it. The observed AUC is
compared to a **circular-shift null** (shift all onsets, relabel, re-classify)
so "separable" means "more separable than aligning random times to the same
data". This searches all feature directions, not just the top PCs that occupancy
sees -- so it can find a pre-ictal shift PCA/occupancy would miss (or show, with
evidence, that a linear classifier can't).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config as C


def _loso_auc(X, grp, base, rng) -> float:
    """Pooled leave-one-seizure-out AUC: per fold, train on other seizures' pre +
    a baseline split, test on the held-out seizure's pre + the disjoint baseline
    split. grp = seizure id per positive row; base = boolean baseline rows."""
    groups = [g for g in np.unique(grp) if g >= 0]
    if len(groups) < 3:
        return np.nan
    bidx = np.flatnonzero(base)
    rng.shuffle(bidx)
    if bidx.size > 30000:                              # cap negatives (AUC-stable)
        bidx = bidx[:30000]
    half = bidx.size // 2
    b_tr, b_te = bidx[:half], bidx[half:]
    ys, ps = [], []
    for g in groups:                                   # bounded loop over seizures
        tr_pos = (grp >= 0) & (grp != g)
        te_pos = grp == g
        Xtr = np.vstack([X[tr_pos], X[b_tr]])
        ytr = np.r_[np.ones(tr_pos.sum()), np.zeros(b_tr.size)]
        if ytr.sum() < 5 or (ytr == 0).sum() < 5:
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        Xte = np.vstack([X[te_pos], X[b_te]])
        yte = np.r_[np.ones(te_pos.sum()), np.zeros(b_te.size)]
        ps.append(clf.predict_proba(sc.transform(Xte))[:, 1])
        ys.append(yte)
    if not ys:
        return np.nan
    y = np.concatenate(ys); p = np.concatenate(ps)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def _labels_for_onsets(t, onsets):
    """(grp, base) for a set of onsets: grp = index of the onset each epoch
    precedes within PREICTAL_SEC (else -1); base = epochs >= BASELINE_MIN_SEC
    from the nearest onset (either side)."""
    ons = np.sort(np.asarray(onsets, float))
    if ons.size == 0:
        return np.full(t.size, -1, int), np.zeros(t.size, bool)
    idx = np.searchsorted(ons, t, side="left")         # next onset at/after t
    nxt = ons[np.clip(idx, 0, ons.size - 1)]
    delta = nxt - t
    grp = np.where((idx < ons.size) & (delta > 0) & (delta <= C.PREICTAL_SEC),
                   idx, -1)
    prev = ons[np.clip(idx - 1, 0, ons.size - 1)]      # nearest onset either side
    dist = np.minimum(np.abs(nxt - t), np.abs(t - prev))
    base = dist >= C.BASELINE_MIN_SEC
    return grp, base


def _baseline_predictors(t, onsets):
    """The two confounds to beat: clock-hour (cyclic sin/cos) and time-since-last
    seizure (log). Clock is absolute (fixed per epoch); tsl is relative to *onsets*."""
    hour = (t % 86400.0) / 3600.0
    ons = np.sort(np.asarray(onsets, float))
    idx = np.searchsorted(ons, t, side="right") - 1
    prev = np.where(idx >= 0, ons[np.clip(idx, 0, ons.size - 1)], t - 7 * 86400.0)
    tsl = np.clip(t - prev, 0, 14 * 86400.0)
    return np.column_stack([np.sin(2 * np.pi * hour / 24),
                            np.cos(2 * np.pi * hour / 24), np.log1p(tsl)])


def incremental_evoked_test(df: pd.DataFrame, onsets, *, features=None,
                            n_surr: int = 300, seed: int = 0) -> dict:
    """Does the evoked feature set add forward predictive value BEYOND clock-hour +
    time-since-last-seizure? Compares LOSO-AUC of a baseline (clock+tsl) model vs
    baseline+evoked, and tests the increment (ΔAUC) against the circular-shift null.
    If ΔAUC is within the null, the evoked response adds nothing over timing."""
    if features is None:
        features = C.FEATURES + [c for c in C.PHFO_FEATURES if c in df.columns]
    features = [f for f in features if f in df.columns]
    Xev = df[features].to_numpy(float)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    ok = np.all(np.isfinite(Xev), axis=1) & np.isfinite(t)
    Xev, t = Xev[ok], t[ok]
    lo, hi = np.nanmin(t), np.nanmax(t); span = hi - lo
    ons = np.sort(np.asarray(onsets, float))

    def aucs(o, sd):
        grp, base = _labels_for_onsets(t, o)
        B = _baseline_predictors(t, o)
        a = _loso_auc(B, grp, base, np.random.default_rng(sd))
        b = _loso_auc(np.hstack([B, Xev]), grp, base, np.random.default_rng(sd))
        return a, b

    a_obs, b_obs = aucs(ons, seed)
    d_obs = (b_obs - a_obs) if (np.isfinite(a_obs) and np.isfinite(b_obs)) else np.nan
    rng = np.random.default_rng(seed)
    nd = np.full(n_surr, np.nan)
    for i in range(n_surr):
        sh = lo + ((ons - lo + rng.uniform(0, span)) % span)
        a_s, b_s = aucs(sh, 1000 + i)
        if np.isfinite(a_s) and np.isfinite(b_s):
            nd[i] = b_s - a_s
    ndv = nd[np.isfinite(nd)]
    p = float((np.sum(ndv >= d_obs) + 1) / (ndv.size + 1)) if (
        ndv.size and np.isfinite(d_obs)) else np.nan
    return {"auc_base": a_obs, "auc_full": b_obs, "delta": d_obs,
            "null_delta": nd, "null_delta_med": float(np.nanmedian(ndv))
            if ndv.size else np.nan, "p": p, "n_features": len(features),
            "n_surr": int(n_surr)}


def classify_null(df: pd.DataFrame, lead_onsets, *, features=None,
                  n_surr: int = 500, seed: int = 0) -> dict:
    """Observed LOSO-AUC (pre vs baseline) + circular-shift null + p-value."""
    if features is None:
        features = C.FEATURES + [c for c in C.PHFO_FEATURES if c in df.columns]
    features = [f for f in features if f in df.columns]
    X = df[features].to_numpy(float)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(t)
    X, t = X[ok], t[ok]
    rng = np.random.default_rng(seed)
    grp, base = _labels_for_onsets(t, lead_onsets)
    auc_obs = _loso_auc(X, grp, base, np.random.default_rng(seed))
    lo, hi = np.nanmin(t), np.nanmax(t); span = hi - lo
    ons = np.sort(np.asarray(lead_onsets, float))
    null = np.full(n_surr, np.nan)
    for i in range(n_surr):                            # bounded loop
        shifted = lo + ((ons - lo + rng.uniform(0, span)) % span)
        g, b = _labels_for_onsets(t, shifted)
        null[i] = _loso_auc(X, g, b, np.random.default_rng(1000 + i))
    nv = null[np.isfinite(null)]
    p = float((np.sum(nv >= auc_obs) + 1) / (nv.size + 1)) if nv.size else np.nan
    return {"auc": auc_obs, "null": null, "null_median": float(np.nanmedian(nv)),
            "null_hi": float(np.nanpercentile(nv, 95)) if nv.size else np.nan,
            "p": p, "n_features": len(features), "n_pre": int((grp >= 0).sum()),
            "n_base": int(base.sum()), "n_surr": int(n_surr)}
