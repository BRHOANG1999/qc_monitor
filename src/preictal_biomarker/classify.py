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
    nxt = np.searchsorted(ons, t, side="left")
    grp = np.full(t.size, -1, dtype=int)
    for i, ti in enumerate(t):
        j = nxt[i]
        if j < ons.size and 0 < ons[j] - ti <= C.PREICTAL_SEC:
            grp[i] = j
    # distance to nearest onset (either side)
    d = np.full(t.size, np.inf)
    for o in ons:
        d = np.minimum(d, np.abs(t - o))
    base = d >= C.BASELINE_MIN_SEC
    return grp, base


def classify_null(df: pd.DataFrame, lead_onsets, *, features=None,
                  n_surr: int = 500, seed: int = 0) -> dict:
    """Observed LOSO-AUC (pre vs baseline) + circular-shift null + p-value."""
    features = list(features or C.FEATURES)
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
