"""Supervised log(time-to-seizure) regression -- the advisor's reframe.

The unsupervised state analysis captured slow drift (the biggest variance), not the
evoked response. Labelling each window by its time-to-next-LEAD-onset forces the
analysis onto seizure-relevant structure: we ask directly whether a HANDFUL of
evoked features predicts log(time-to-seizure) on HELD-OUT seizures, beating a
circular-shift null. Build eyeball-first (``eyeball_trajectories``); only trust the
model (``regression_null``) if a feature visibly bends toward onset, and only trust
significance if it beats the shift null under leave-one-SEIZURE-out.

Stats only (mirrors the classify.py <-> classifier_null_fig split). Adapts
``classify._loso_auc`` (LogisticRegression->Ridge, predict_proba->predict, pooled
roc_auc->Spearman/per-horizon-AUC) and reuses ``periictal.trajectory`` /
``periictal.trendtest`` for the eyeball + per-seizure views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

_MAX_SURR = 100_000                       # NASA Rule 2: explicit loop bound.


# --------------------------------------------------------------------- #
#  Target + helpers
# --------------------------------------------------------------------- #

def _present(features, df) -> list:
    return [f for f in features if f in df.columns]


def _xy(df, features):
    """(X[n,f], t[n]) over rows with all-finite features + finite t_epoch."""
    X = df[features].to_numpy(float)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(t)
    return X[ok], t[ok]


def _pval(null_arr, obs) -> float:
    c = null_arr[np.isfinite(null_arr)]
    if not c.size or not np.isfinite(obs):
        return float("nan")
    return float((np.sum(c >= obs) + 1) / (c.size + 1))


def _log_tto_target(t, lead_onsets, *, cap=None, buffer=None, far_min=None,
                    floor=None):
    """Per epoch: y=log(min(tto_to_next_LEAD, cap)); grp=lead id for pre rows
    (0<tto<=cap) else -1; far=tto>far_min (baseline ceiling, y=log cap); keep=rows
    >= buffer since the previous lead onset (drop the post-ictal window)."""
    cap = C.REG_TARGET_CAP_SEC if cap is None else cap
    buffer = C.REG_POSTICTAL_BUFFER_SEC if buffer is None else buffer
    far_min = C.REG_FAR_MIN_SEC if far_min is None else far_min
    floor = C.REG_MIN_LEADTIME_SEC if floor is None else floor
    assert cap > floor > 0, "need cap > floor > 0"
    ons = np.sort(np.asarray(lead_onsets, float))
    assert ons.size >= 1, "need >= 1 lead onset"
    nxt_i = np.searchsorted(ons, t, side="left")
    nxt = np.where(nxt_i < ons.size, ons[np.clip(nxt_i, 0, ons.size - 1)], np.inf)
    tto = nxt - t
    prv_i = np.searchsorted(ons, t, side="right") - 1
    prev = np.where(prv_i >= 0, ons[np.clip(prv_i, 0, ons.size - 1)], -np.inf)
    tsl = t - prev
    y = np.log(np.clip(np.minimum(tto, cap), floor, None))
    grp = np.where((tto > 0) & (tto <= cap), nxt_i, -1).astype(int)
    return y, grp, (tto > far_min), (tsl >= buffer)


# --------------------------------------------------------------------- #
#  Pooled leave-one-seizure-out regression (adapts classify._loso_auc)
# --------------------------------------------------------------------- #

def _loso_predict(X, y, grp, far, rng, *, alpha=None, neg_cap=None):
    """Pooled held-out (y_true, y_pred) of a Ridge regressor of log-tto. Per fold:
    train on other seizures' pre rows + a far split, test on the held-out seizure's
    pre rows + the DISJOINT far split. Empty arrays if < 3 lead seizures."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    assert X.shape[0] == y.shape[0] == grp.shape[0], "row count mismatch"
    assert X.ndim == 2, "X must be 2-D"
    alpha = C.REG_RIDGE_ALPHA if alpha is None else alpha
    neg_cap = C.REG_NEG_CAP if neg_cap is None else neg_cap
    groups = [g for g in np.unique(grp) if g >= 0]
    if len(groups) < 3:
        return np.empty(0), np.empty(0)
    bidx = np.flatnonzero(far)
    rng.shuffle(bidx)
    if bidx.size > neg_cap:
        bidx = bidx[:neg_cap]
    half = bidx.size // 2
    b_tr, b_te = bidx[:half], bidx[half:]
    ys, ps = [], []
    for g in groups:                                   # bounded by #lead seizures
        tr_pos = (grp >= 0) & (grp != g)
        te_pos = grp == g
        if tr_pos.sum() < 5 or te_pos.sum() < 1 or b_tr.size < 5:
            continue
        Xtr = np.vstack([X[tr_pos], X[b_tr]])
        ytr = np.concatenate([y[tr_pos], y[b_tr]])
        sc = StandardScaler().fit(Xtr)
        clf = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
        Xte = np.vstack([X[te_pos], X[b_te]])
        ps.append(clf.predict(sc.transform(Xte)))
        ys.append(np.concatenate([y[te_pos], y[b_te]]))
    if not ys:
        return np.empty(0), np.empty(0)
    return np.concatenate(ys), np.concatenate(ps)


def _metrics(y_true, y_pred, *, horizons=None, floor=None) -> dict:
    """Spearman rho(true,pred) + per-horizon AUC ('seizure within H') from the same
    pooled held-out predictions. tto recovered as exp(y_true) (capped)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    horizons = C.REG_HORIZONS_SEC if horizons is None else horizons
    out = {"spearman": float("nan"), "horizon_auc": {}}
    if (y_true.size < 3 or np.unique(y_true).size < 2
            or np.unique(y_pred).size < 2):
        out["horizon_auc"] = {float(H): float("nan") for H in horizons}
        return out
    out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
    tto_true = np.exp(y_true)
    score = -y_pred                                     # higher score = sooner
    for H in horizons:
        pos = tto_true <= H
        out["horizon_auc"][float(H)] = (float(roc_auc_score(pos, score))
                                        if pos.any() and (~pos).any()
                                        else float("nan"))
    return out


def _run_once(X, t, onsets, sd) -> tuple:
    """One target build -> LOSO predict -> metrics. Returns (metrics, yt, yp)."""
    y, grp, far, keep = _log_tto_target(t, onsets)
    yt, yp = _loso_predict(X[keep], y[keep], grp[keep], far[keep],
                           np.random.default_rng(sd))
    return _metrics(yt, yp), yt, yp


# --------------------------------------------------------------------- #
#  Observed vs circular-shift null
# --------------------------------------------------------------------- #

def regression_null(df, lead_onsets, *, features=None, n_surr=500, seed=0) -> dict:
    """Held-out LOSO log-tto Spearman + per-horizon AUC, each vs the circular-shift
    null (shift onsets, relabel, re-fit). Returns observed metrics, null arrays,
    p-values, pooled (y_true, y_pred), and pre/far counts."""
    features = _present(features or C.REGRESSION_FEATURES, df)
    assert features, "no regression features present in df"
    assert len(lead_onsets) >= 3, "need >= 3 lead seizures for LOSO"
    assert 1 <= int(n_surr) <= _MAX_SURR, "n_surr out of bounds"
    X, t = _xy(df, features)
    ons = np.sort(np.asarray(lead_onsets, float))
    y, grp, far, keep = _log_tto_target(t, ons)
    yt, yp = _loso_predict(X[keep], y[keep], grp[keep], far[keep],
                           np.random.default_rng(seed))
    obs = _metrics(yt, yp)
    lo, hi = np.nanmin(t), np.nanmax(t); span = float(hi - lo)
    assert span > 0, "degenerate time span"
    rng = np.random.default_rng(seed)
    null_sp = np.full(int(n_surr), np.nan)
    null_h = {float(H): np.full(int(n_surr), np.nan) for H in C.REG_HORIZONS_SEC}
    for i in range(int(n_surr)):
        sh = lo + ((ons - lo + rng.uniform(0, span)) % span)
        m, _, _ = _run_once(X, t, sh, 1000 + i)
        null_sp[i] = m["spearman"]
        for H in null_h:
            null_h[H][i] = m["horizon_auc"].get(H, float("nan"))
    return {"features": features, "spearman": obs["spearman"],
            "p_spearman": _pval(null_sp, obs["spearman"]), "null_spearman": null_sp,
            "horizon_auc": obs["horizon_auc"], "horizons": list(C.REG_HORIZONS_SEC),
            "p_horizon": {H: _pval(null_h[H], obs["horizon_auc"].get(H, float("nan")))
                          for H in null_h}, "null_horizon": null_h,
            "y_true": yt, "y_pred": yp, "n_surr": int(n_surr),
            "n_pre": int(((grp >= 0) & keep).sum()), "n_far": int((far & keep).sum())}


def incremental_evoked_regression(df, lead_onsets, *, features=None, n_surr=300,
                                  seed=0) -> dict:
    """Does the evoked handful add log-tto predictive value BEYOND clock-hour +
    time-since-last-seizure? Delta-Spearman(base -> base+evoked) vs the shift null."""
    from .classify import _baseline_predictors
    features = _present(features or C.REGRESSION_FEATURES, df)
    assert features, "no regression features present in df"
    assert len(lead_onsets) >= 3, "need >= 3 lead seizures for LOSO"
    Xev, t = _xy(df, features)
    ons = np.sort(np.asarray(lead_onsets, float))

    def delta(onsets_, sd):
        y, grp, far, keep = _log_tto_target(t, onsets_)
        B = _baseline_predictors(t, onsets_)
        at, ap = _loso_predict(B[keep], y[keep], grp[keep], far[keep],
                               np.random.default_rng(sd))
        bt, bp = _loso_predict(np.hstack([B, Xev])[keep], y[keep], grp[keep],
                               far[keep], np.random.default_rng(sd))
        sa, sb = _metrics(at, ap)["spearman"], _metrics(bt, bp)["spearman"]
        return sa, sb

    a_obs, b_obs = delta(ons, seed)
    d_obs = (b_obs - a_obs) if (np.isfinite(a_obs) and np.isfinite(b_obs)) else np.nan
    lo, hi = np.nanmin(t), np.nanmax(t); span = float(hi - lo)
    rng = np.random.default_rng(seed)
    nd = np.full(int(n_surr), np.nan)
    for i in range(int(n_surr)):
        sh = lo + ((ons - lo + rng.uniform(0, span)) % span)
        sa, sb = delta(sh, 2000 + i)
        if np.isfinite(sa) and np.isfinite(sb):
            nd[i] = sb - sa
    return {"spearman_base": a_obs, "spearman_full": b_obs, "delta": d_obs,
            "null_delta": nd, "p": _pval(nd, d_obs),
            "null_delta_med": float(np.nanmedian(nd[np.isfinite(nd)]))
            if np.isfinite(nd).any() else float("nan"),
            "n_features": len(features), "n_surr": int(n_surr)}


# --------------------------------------------------------------------- #
#  Per-seizure trend (reuse periictal.trendtest via a small adapter)
# --------------------------------------------------------------------- #

def _trend_frame(df, lead_onsets, *, feats=None, cap=None) -> pd.DataFrame:
    """Adapt the preictal_biomarker df to the columns periictal.trendtest needs:
    signed time_to_onset_sec vs the nearest LEAD onset (+pre/-post), seizure_idx,
    phase in {pre,post}, hour_of_day, t_epoch + feature columns (pre/post rows only)."""
    cap = C.REG_TARGET_CAP_SEC if cap is None else cap
    feats = _present(feats or C.REGRESSION_FEATURES, df)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    ons = np.sort(np.asarray(lead_onsets, float))
    assert ons.size >= 1 and cap > 0, "need onsets and cap > 0"
    nxt_i = np.searchsorted(ons, t, side="left")
    nxt = np.where(nxt_i < ons.size, ons[np.clip(nxt_i, 0, ons.size - 1)], np.inf)
    prv_i = np.clip(nxt_i - 1, 0, ons.size - 1)
    prev = np.where(nxt_i > 0, ons[prv_i], -np.inf)
    d_next, d_prev = nxt - t, t - prev
    pre = (d_next > 0) & (d_next <= cap)
    post = (d_prev > 0) & (d_prev <= cap) & (~pre)
    keep = pre | post
    cols = {"t_epoch": t[keep],
            "time_to_onset_sec": np.where(pre, d_next, -d_prev)[keep],
            "seizure_idx": np.where(pre, nxt_i, prv_i)[keep],
            "phase": np.where(pre, "pre", "post")[keep],
            "hour_of_day": ((t[keep] % 86400.0) / 3600.0)}
    for f in feats:
        cols[f] = pd.to_numeric(df[f], errors="coerce").to_numpy(float)[keep]
    return pd.DataFrame(cols)


def per_seizure_regression(df, lead_onsets, *, feature=None, n_boot=2000,
                           n_surr=1000, seed=0, min_n=5) -> dict:
    """Per-seizure Spearman + block-bootstrap slope forest + circular-shift surrogate
    for one feature vs log lead-time, via periictal.trendtest. The '12 individual
    trajectories, some ramp some not' view."""
    from src.periictal import trendtest as _TT
    feature = C.CSD_PRIMARY if feature is None else feature
    assert feature in df.columns, f"feature '{feature}' missing"
    assert len(lead_onsets) >= 2, "need >= 2 lead seizures"
    frame = _trend_frame(df, lead_onsets, feats=[feature])
    pre = frame[frame["phase"] == "pre"]
    trend = _TT.per_seizure_trend(pre, feature, min_n=min_n)
    forest = _TT.per_seizure_slope_forest(pre, feature, n_boot=n_boot, seed=seed,
                                          min_n=min_n)
    across = _TT.across_seizure_test([r["rho"] for r in trend])
    surro = _TT.circular_shift_surrogate_p(frame, feature, n_surrogates=n_surr,
                                           seed=seed, min_n=min_n)
    return {"feature": feature, "trend": trend, "forest": forest, "across": across,
            "surrogate": surro, "sign_flip_floor": _TT.sign_flip_floor(
                across["n_seizures"]),
            "onsets": np.sort(np.asarray(lead_onsets, float))}


# --------------------------------------------------------------------- #
#  Eyeball: feature vs log lead-time, per-seizure baseline-normalized
# --------------------------------------------------------------------- #

def _baseline_norm(v, sid, tto, cap, *, base_win=7200.0) -> np.ndarray:
    """z-normalize each seizure's feature to its OWN early baseline (the far end,
    tto in [cap-base_win, cap]) so '0' = baseline and a bend toward onset shows."""
    out = np.full(v.size, np.nan)
    for s in np.unique(sid):                           # bounded by #seizures
        m = sid == s
        ref = v[m & (tto >= cap - base_win)]
        ref = ref[np.isfinite(ref)]
        if ref.size >= 3 and np.std(ref) > 0:
            out[m] = (v[m] - np.mean(ref)) / np.std(ref)
        elif ref.size:
            out[m] = v[m] - np.mean(ref)
    return out


def eyeball_trajectories(df, lead_onsets, *, features=None, cap=None,
                         floor=None) -> dict:
    """Per feature, per-seizure baseline-normalized trajectory vs LOG lead-time
    (reuses periictal.trajectory). Returns {traj: {feature: lead_time_trajectory},
    edges, features, n_seizures}. The go/no-go eyeball gate."""
    from src.periictal import trajectory as _TR
    cap = C.REG_TARGET_CAP_SEC if cap is None else cap
    floor = C.REG_MIN_LEADTIME_SEC if floor is None else floor
    features = _present(features or C.REGRESSION_FEATURES, df)
    assert features, "no regression features present in df"
    assert cap > floor > 0, "need cap > floor > 0"
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    ons = np.sort(np.asarray(lead_onsets, float))
    nxt_i = np.searchsorted(ons, t, side="left")
    nxt = np.where(nxt_i < ons.size, ons[np.clip(nxt_i, 0, ons.size - 1)], np.inf)
    tto = nxt - t
    pre = (tto > 0) & (tto <= cap)
    edges = _TR.default_edges(cap, floor)
    tto_c, sid = np.clip(tto[pre], floor, cap), nxt_i[pre]
    out = {}
    for f in features:
        v = pd.to_numeric(df[f], errors="coerce").to_numpy(float)[pre]
        vn = _baseline_norm(v, sid, tto[pre], cap)
        out[f] = _TR.lead_time_trajectory(tto_c, vn, sid, edges, log_centers=True)
    return {"traj": out, "edges": edges, "features": features,
            "n_seizures": int(np.unique(sid).size)}
