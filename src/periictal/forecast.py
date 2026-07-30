"""Preictal-vs-interictal discrimination + forecasting (Chang et al. 2026).

Recreates the paper's metric/modeling strategy on our evoked feature matrix
(``src.periictal.matrix.build_matrix``): label each lead-up stimulus as
preictal (0-30 min before a seizure) or interictal (60-90 min before), then

* build per-class PDF (KDE + histogram) and CDF (ECDF) for any feature,
* score each feature's discrimination with a rank AUC normalised to
  ``max(AUC, 1-AUC)`` and a 500-label-permutation p-value (paper Fig 2),
* fit the paper's prospective multivariable logistic-regression forecaster
  (train epilepsy-phase P, test P+1; ``forecast.py`` Step 3 lives in
  ``logistic_forecast``).

Pure + Dash-free so it is unit-testable; the lens and the PNG export both call
in here. Operates on the ``event x metric`` DataFrame whose meta columns are
``time_to_onset_sec`` (signed, positive = pre-onset), ``phase`` (pre/post),
``seizure_idx``, ``seizure_onset_epoch``, ``hour_of_day``.

Statistical caveat (documented, not hidden): the pooled per-stimulus AUC +
label-permutation p matches the paper exactly, but stimuli within one
seizure's window are autocorrelated, so that p is optimistic (pseudo-
replication). ``per_seizure_values`` / ``paired_seizure_test`` give the honest
seizure-as-unit comparison; the lens shows both.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from src.periictal import config as _cfg
from src.periictal import resample as _res
from src.preictal.scoring import rank_auc
from src.periictal.trendtest import _bh_qvalues

CLASS_PREICTAL = "preictal"
CLASS_INTERICTAL = "interictal"
_MAX_PERM = 100_000
_MAX_SEIZURES = 1_000_000


# --------------------------------------------------------------------- #
#  Labeling
# --------------------------------------------------------------------- #

def _seizure_isi(df) -> dict:
    """Map each ``seizure_onset_epoch`` to its inter-seizure interval (seconds
    since the previous distinct onset). The first seizure maps to +inf."""
    onsets = np.unique(df["seizure_onset_epoch"].to_numpy(dtype=float))
    onsets = onsets[np.isfinite(onsets)]
    isi = {}
    prev = None
    for o in onsets:
        isi[float(o)] = (o - prev) if prev is not None else np.inf
        prev = o
    return isi


def label_classes(df, *,
                  preictal_max_sec: float = _cfg.PREICTAL_MAX_SEC,
                  interictal_lo_sec: float = _cfg.INTERICTAL_LO_SEC,
                  interictal_hi_sec: float = _cfg.INTERICTAL_HI_SEC,
                  min_preictal_sec: float = _cfg.MIN_PREICTAL_SEC,
                  require_clean_preictal: bool = True):
    """Return *df* with an added ``class`` column in
    {preictal, interictal, ""}.

    Preictal = pre-onset rows with ``0 < tto <= preictal_max_sec`` whose
    seizure has ISI >= ``min_preictal_sec`` (a clean 30-min preictal period).
    Interictal = pre-onset rows with ``interictal_lo_sec <= tto <=
    interictal_hi_sec``. Everything else (buffer, post-ictal, out of range) is
    "" (unused). Does not mutate the input.
    """
    assert "time_to_onset_sec" in df.columns, "matrix missing time_to_onset_sec"
    assert "phase" in df.columns, "matrix missing phase"
    tto = df["time_to_onset_sec"].to_numpy(dtype=float)
    is_pre = (df["phase"].to_numpy() == "pre") & np.isfinite(tto)
    cls = np.full(len(df), "", dtype=object)

    pre_mask = is_pre & (tto > 0) & (tto <= preictal_max_sec)
    if require_clean_preictal and "seizure_onset_epoch" in df.columns:
        isi = _seizure_isi(df)
        row_isi = np.array([isi.get(float(o), np.inf)
                            for o in df["seizure_onset_epoch"].to_numpy(dtype=float)])
        pre_mask = pre_mask & (row_isi >= min_preictal_sec)
    cls[pre_mask] = CLASS_PREICTAL

    inter_mask = (is_pre & (tto >= interictal_lo_sec)
                  & (tto <= interictal_hi_sec))
    cls[inter_mask] = CLASS_INTERICTAL

    out = df.copy()
    out["class"] = cls
    return out


def class_arrays(df, feature: str):
    """Finite feature values for (preictal, interictal). Requires a ``class``
    column (call ``label_classes`` first)."""
    assert "class" in df.columns, "call label_classes(df) first"
    assert feature in df.columns, f"feature '{feature}' not in df"
    v = df[feature].to_numpy(dtype=float)
    c = df["class"].to_numpy()
    pre = v[(c == CLASS_PREICTAL) & np.isfinite(v)]
    inter = v[(c == CLASS_INTERICTAL) & np.isfinite(v)]
    return pre, inter


# --------------------------------------------------------------------- #
#  Per-feature discrimination (AUC + permutation p)
# --------------------------------------------------------------------- #

def _auc_from_ranks(ranks_a, na: int, nb: int) -> float:
    """Mann-Whitney AUC = P(a > b) from the pooled *ranks* of group a."""
    u_a = float(np.sum(ranks_a)) - na * (na + 1) / 2.0
    return u_a / (na * nb)


def feature_auc(df, feature: str) -> dict:
    """Rank AUC discriminating preictal (positive) from interictal for
    *feature*, plus the paper's normalised ``auc_norm = max(AUC, 1-AUC)``,
    class counts and direction. NaN when either class is empty."""
    pre, inter = class_arrays(df, feature)
    na, nb = pre.size, inter.size
    if na == 0 or nb == 0:
        return {"feature": feature, "auc": float("nan"),
                "auc_norm": float("nan"), "n_pre": na, "n_inter": nb,
                "direction": "none"}
    auc = rank_auc(pre, inter)          # P(preictal > interictal)
    return {"feature": feature, "auc": float(auc),
            "auc_norm": float(max(auc, 1.0 - auc)), "n_pre": na, "n_inter": nb,
            "direction": "higher" if auc > 0.5 else "lower"}


def permutation_p(df, feature: str, *, n_perm: int = 500, seed: int = 0) -> dict:
    """Paper-style significance: pool the preictal + interictal values, rank
    once, then permute the class labels *n_perm* times and recompute
    ``|AUC-0.5|``. Empirical two-sided p = ``(ge+1)/(n_perm+1)``.

    Caveat: labels are permuted per-STIMULUS, so within-window autocorrelation
    inflates significance (this mirrors the paper; use ``paired_seizure_test``
    for the seizure-as-unit alternative).
    """
    assert 1 <= int(n_perm) <= _MAX_PERM, "n_perm out of bounds"
    pre, inter = class_arrays(df, feature)
    na, nb = pre.size, inter.size
    if na == 0 or nb == 0:
        return {"feature": feature, "p": float("nan"), "n_perm": 0,
                "auc": float("nan"), "auc_norm": float("nan")}
    pooled = np.concatenate([pre, inter])
    ranks = rankdata(pooled)
    obs = _auc_from_ranks(ranks[:na], na, nb)
    obs_stat = abs(obs - 0.5)
    rng = np.random.default_rng(seed)
    n = na + nb
    ge = 0
    for i in range(int(n_perm)):
        assert i < _MAX_PERM, "permutation runaway"
        idx = rng.permutation(n)[:na]
        stat = abs(_auc_from_ranks(ranks[idx], na, nb) - 0.5)
        if stat >= obs_stat - 1e-12:
            ge += 1
    return {"feature": feature, "p": float((ge + 1) / (int(n_perm) + 1)),
            "n_perm": int(n_perm), "auc": float(obs),
            "auc_norm": float(max(obs, 1.0 - obs))}


def scan_features(df, features, *, n_perm: int = 500, seed: int = 0) -> list:
    """Per-feature AUC + permutation p + Benjamini-Hochberg q across features,
    sorted by ``auc_norm`` descending. *features* is the metric column list."""
    assert features is not None, "features required"
    rows: list = []
    for i, f in enumerate(features):
        if f not in df.columns:
            continue
        pp = permutation_p(df, f, n_perm=n_perm, seed=seed + i)
        fa = feature_auc(df, f)
        rows.append({"feature": f, "auc": fa["auc"], "auc_norm": fa["auc_norm"],
                     "direction": fa["direction"], "n_pre": fa["n_pre"],
                     "n_inter": fa["n_inter"], "p": pp["p"], "q": float("nan")})
    q = _bh_qvalues(np.array([r["p"] for r in rows], dtype=float))
    for r, qi in zip(rows, q):
        r["q"] = float(qi)
    rows.sort(key=lambda r: (-r["auc_norm"] if np.isfinite(r["auc_norm"])
                             else np.inf))
    return rows


# --------------------------------------------------------------------- #
#  Honest seizure-as-unit comparison
# --------------------------------------------------------------------- #

def per_seizure_values(df, feature: str):
    """Collapse each seizure to (preictal_mean, interictal_mean) for *feature*
    -- one paired observation per seizure. Only seizures with >=1 finite value
    in BOTH classes are returned. Returns (pre_vals, inter_vals) aligned."""
    assert "class" in df.columns, "call label_classes(df) first"
    assert "seizure_idx" in df.columns, "matrix missing seizure_idx"
    v = df[feature].to_numpy(dtype=float)
    c = df["class"].to_numpy()
    sid = df["seizure_idx"].to_numpy()
    pre_out, inter_out = [], []
    sids = np.unique(sid)
    assert sids.size < _MAX_SEIZURES, "seizure count runaway"
    for s in sids:
        m = sid == s
        pv = v[m & (c == CLASS_PREICTAL)]
        iv = v[m & (c == CLASS_INTERICTAL)]
        pv = pv[np.isfinite(pv)]
        iv = iv[np.isfinite(iv)]
        if pv.size and iv.size:
            pre_out.append(float(np.mean(pv)))
            inter_out.append(float(np.mean(iv)))
    return np.array(pre_out), np.array(inter_out)


def paired_seizure_test(df, feature: str) -> dict:
    """Honest (no pseudo-replication) comparison: per-seizure preictal vs
    interictal means, Wilcoxon signed-rank (sign-test fallback < 6 seizures).
    Returns ``{n_seizures, auc, p, direction}`` where AUC is the fraction of
    seizures with preictal > interictal (a paired effect size)."""
    from scipy.stats import wilcoxon, binomtest
    pre, inter = per_seizure_values(df, feature)
    n = pre.size
    if n == 0:
        return {"feature": feature, "n_seizures": 0, "auc": float("nan"),
                "p": float("nan"), "direction": "none"}
    d = pre - inter
    n_pos, n_neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    auc = float(np.mean(pre > inter))
    direction = "higher" if n_pos > n_neg else "lower" if n_neg > n_pos else "none"
    if n < 6 or np.allclose(d, 0.0):
        m = n_pos + n_neg
        p = (float(binomtest(min(n_pos, n_neg), m, 0.5,
                             alternative="two-sided").pvalue) if m else 1.0)
    else:
        try:
            _stat, p = wilcoxon(d)
            p = float(p)
        except ValueError:
            p = 1.0
    return {"feature": feature, "n_seizures": n, "auc": auc, "p": p,
            "direction": direction}


# --------------------------------------------------------------------- #
#  Within-seizure standardization (fixes the composition artifact in the
#  SCALE-dependent views only; per-seizure AUC is rank-invariant to it -- see
#  docs/periictal_methods_evidence.md, A2 / L3)
# --------------------------------------------------------------------- #

def standardize_to_interictal(df, *, by: str = "seizure_idx", features=None,
                              mode: str = "interictal"):
    """Express each feature in within-*by* reference-SD units, so a between-group
    (e.g. between-seizure) offset can no longer masquerade as a class effect in
    the POOLED AUC / PDF / logistic coefficients.

    mode='interictal' (default): centre+scale each group by that group's
    INTERICTAL-class mean/SD, so interictal sits at ~0 and a preictal shift is in
    interictal-SD units (unbiased -- the SD does not include the between-class
    variance). Requires a 'class' column (call label_classes first).
    mode='grand': centre+scale by the group's overall mean/SD (the biased variant
    kept as a toggle). Does not mutate the input.
    """
    assert by in df.columns, f"matrix missing '{by}'"
    assert mode in ("interictal", "grand"), "mode must be interictal|grand"
    feats = [f for f in (features or _cfg.CHEAP_METRICS) if f in df.columns]
    out = df.copy()
    groups = out[by].to_numpy()
    cls = out["class"].to_numpy() if "class" in out.columns else None
    ref_ok = (mode == "grand") or (cls is not None)
    for f in feats:
        vals = out[f].to_numpy(dtype=float)
        newv = vals.copy()
        for g in np.unique(groups):
            m = groups == g
            ref = m if (mode == "grand" or not ref_ok) else (m & (cls == CLASS_INTERICTAL))
            r = vals[ref]
            r = r[np.isfinite(r)]
            if r.size < 2:
                continue
            sd = float(np.std(r, ddof=1))
            if not np.isfinite(sd) or sd <= 0.0:
                continue
            newv[m] = (vals[m] - float(np.mean(r))) / sd
        out[f] = newv
    return out


# --------------------------------------------------------------------- #
#  Per-seizure AUC forest (the honest headline) with dependence-aware CIs
# --------------------------------------------------------------------- #

def block_bootstrap_auc_ci(pre, inter, *, n_boot: int = 2000, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """Percentile CI for AUC(pre, inter) by moving-block bootstrap of each
    TIME-ORDERED class series (preserves within-class autocorrelation, so the CI
    is not the fake-tight trial-level interval). Returns ``{ci_lo, ci_hi,
    block_pre, block_inter, n_blocks_eff}``. Caller passes time-ordered arrays."""
    pre = np.asarray(pre, dtype=float)
    inter = np.asarray(inter, dtype=float)
    assert pre.ndim == 1 and inter.ndim == 1, "1-D arrays required"
    assert 1 <= int(n_boot) <= 200_000, "n_boot out of bounds"
    nan = {"ci_lo": float("nan"), "ci_hi": float("nan"), "block_pre": 1,
           "block_inter": 1, "n_blocks_eff": float("nan")}
    if pre.size < 2 or inter.size < 2:
        return nan
    bp, bi = _res.block_length(pre), _res.block_length(inter)
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        assert b < 200_000, "bootstrap runaway"
        stats[b] = rank_auc(_res.moving_block_resample(pre, bp, rng),
                            _res.moving_block_resample(inter, bi, rng))
    lo, hi = np.percentile(stats, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    n_eff = pre.size / bp + inter.size / bi
    return {"ci_lo": float(lo), "ci_hi": float(hi), "block_pre": int(bp),
            "block_inter": int(bi), "n_blocks_eff": float(n_eff)}


def per_seizure_auc_forest(df, feature: str, *, n_boot: int = 2000,
                           seed: int = 0, min_n: int = 5) -> list:
    """One AUC per seizure (preictal vs interictal) for *feature*, each with a
    within-seizure block-bootstrap CI and its effective block count -- the plan's
    headline statistic. Rank-invariant to standardization, so valid on raw
    features. Requires a 'class' column. Returns a list sorted by seizure."""
    assert "class" in df.columns, "call label_classes(df) first"
    assert feature in df.columns, f"feature '{feature}' not in df"
    for c in ("seizure_idx", "t_epoch"):
        assert c in df.columns, f"matrix missing '{c}'"
    v = df[feature].to_numpy(dtype=float)
    cls = df["class"].to_numpy()
    sid = df["seizure_idx"].to_numpy()
    t = df["t_epoch"].to_numpy(dtype=float)
    out: list = []
    sids = np.unique(sid)
    assert sids.size < _MAX_SEIZURES, "seizure count runaway"
    for s in sids:
        m = sid == s
        order = np.argsort(t[m])                 # time-order within the seizure
        vm, cm = v[m][order], cls[m][order]
        pre = vm[cm == CLASS_PREICTAL]
        inter = vm[cm == CLASS_INTERICTAL]
        pre, inter = pre[np.isfinite(pre)], inter[np.isfinite(inter)]
        if pre.size < min_n or inter.size < min_n:
            continue
        ci = block_bootstrap_auc_ci(pre, inter, n_boot=n_boot, seed=seed + int(s))
        out.append({"seizure_idx": int(s), "auc": float(rank_auc(pre, inter)),
                    "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                    "n_pre": int(pre.size), "n_inter": int(inter.size),
                    "n_blocks_eff": ci["n_blocks_eff"]})
    return out


# --------------------------------------------------------------------- #
#  PDF / CDF
# --------------------------------------------------------------------- #

def _ecdf(x):
    """Empirical CDF as (sorted_x, cumulative_fraction) step points."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _kde_curve(x, grid):
    """Gaussian-KDE density of *x* on *grid*, or None when undefined."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.unique(x).size < 2:
        return None
    from scipy.stats import gaussian_kde
    try:
        return gaussian_kde(x)(grid)
    except Exception:                     # noqa: BLE001 -- singular cov, etc.
        return None


def pdf_cdf(df, feature: str, *, bins: int = 40, clip_pct: float = 0.5) -> dict:
    """Per-class PDF (density histogram + KDE) and CDF (ECDF) for *feature*,
    plus the AUC. *clip_pct* trims the shared axis to the [clip_pct, 100-
    clip_pct] percentile of the pooled data so a few outliers don't flatten
    the plot. Returns everything the lens / PNG export need."""
    pre, inter = class_arrays(df, feature)
    fa = feature_auc(df, feature)
    out = {"feature": feature, "n_pre": pre.size, "n_inter": inter.size,
           "auc": fa["auc"], "auc_norm": fa["auc_norm"],
           "direction": fa["direction"]}
    pooled = np.concatenate([pre, inter]) if (pre.size or inter.size) else np.array([])
    if pooled.size == 0:
        return {**out, "edges": np.array([]), "centers": np.array([]),
                "pre_hist": np.array([]), "inter_hist": np.array([]),
                "grid": np.array([]), "pre_kde": None, "inter_kde": None,
                "pre_cdf_x": np.array([]), "pre_cdf_y": np.array([]),
                "inter_cdf_x": np.array([]), "inter_cdf_y": np.array([])}
    lo, hi = np.percentile(pooled, [clip_pct, 100 - clip_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(pooled)), float(np.max(pooled)) + 1e-9
    edges = np.linspace(lo, hi, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_hist = (np.histogram(pre, bins=edges, density=True)[0]
                    if pre.size else np.zeros(bins))
        inter_hist = (np.histogram(inter, bins=edges, density=True)[0]
                      if inter.size else np.zeros(bins))
    grid = np.linspace(lo, hi, 256)
    return {**out, "edges": edges, "centers": centers,
            "pre_hist": pre_hist, "inter_hist": inter_hist,
            "grid": grid, "pre_kde": _kde_curve(pre, grid),
            "inter_kde": _kde_curve(inter, grid),
            "pre_cdf_x": _ecdf(pre)[0], "pre_cdf_y": _ecdf(pre)[1],
            "inter_cdf_x": _ecdf(inter)[0], "inter_cdf_y": _ecdf(inter)[1]}


# --------------------------------------------------------------------- #
#  Epileptogenesis phases (for stratified analysis / the forecaster)
# --------------------------------------------------------------------- #

def assign_epi_phase(df, n_phases: int = _cfg.DEFAULT_N_PHASES):
    """Add an ``epi_phase`` column (0..k-1) splitting the animal's seizures into
    *k* consecutive groups of ~equal seizure count (paper's phase stratification
    for feature drift). *k* is clamped to the seizure count. Does not mutate
    the input; returns (df2, k_used)."""
    onsets = np.unique(df["seizure_onset_epoch"].to_numpy(dtype=float))
    onsets = np.sort(onsets[np.isfinite(onsets)])
    n_sz = onsets.size
    k = int(max(1, min(int(n_phases), n_sz)))
    # Rank each seizure onset 0..n_sz-1, then bucket into k equal groups.
    rank_of = {float(o): i for i, o in enumerate(onsets)}
    phase_of = {o: min(k - 1, (rank_of[o] * k) // max(1, n_sz)) for o in rank_of}
    ep = np.array([phase_of.get(float(o), -1)
                   for o in df["seizure_onset_epoch"].to_numpy(dtype=float)])
    out = df.copy()
    out["epi_phase"] = ep
    return out, k


# --------------------------------------------------------------------- #
#  Multivariable logistic-regression forecaster (paper Fig 3 / Fig 5)
# --------------------------------------------------------------------- #

def _xy(df, features):
    """(X, y) for the labeled rows with all *features* finite. y = 1 for
    preictal, 0 for interictal."""
    sub = df[df["class"].isin([CLASS_PREICTAL, CLASS_INTERICTAL])]
    X = sub[list(features)].to_numpy(dtype=float)
    y = (sub["class"].to_numpy() == CLASS_PREICTAL).astype(int)
    ok = np.all(np.isfinite(X), axis=1)
    return X[ok], y[ok]


def _fit_predict_auc(Xtr, ytr, Xte, yte):
    """Standardise + fit logistic regression on train, return (test AUC,
    fpr, tpr) on test. NaN AUC when a split lacks both classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve
    if (np.unique(ytr).size < 2 or np.unique(yte).size < 2
            or Xtr.shape[0] < 5 or Xte.shape[0] < 5):
        return float("nan"), None, None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]
    auc = float(roc_auc_score(yte, proba))
    fpr, tpr, _ = roc_curve(yte, proba)
    return auc, fpr, tpr


def normalized_coefficients(df, features) -> dict:
    """Fit one standardised logistic regression on ALL labeled rows and return
    ``{feature: |coef| / Σ|coef|}`` -- the paper's normalised feature-importance
    (Fig 5C). Empty when the data can't be fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = _xy(df, features)
    if X.shape[0] < 10 or np.unique(y).size < 2:
        return {}
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(X), y)
    coef = np.abs(clf.coef_.ravel())
    tot = coef.sum()
    if tot <= 0:
        return {f: 0.0 for f in features}
    return {f: float(c / tot) for f, c in zip(features, coef)}


def select_best_features(df, metrics, *, k: int = 5, n_phases: int = 4) -> list:
    """The paper's 'best features' selection: rank *metrics* by their MEAN
    per-phase normalised AUC (drift-aware) and take the top *k*."""
    dfp, kp = assign_epi_phase(df, n_phases)
    per_phase_auc = {m: [] for m in metrics}
    for p in range(kp):
        sub = dfp[dfp["epi_phase"] == p]
        for m in metrics:
            if m in sub.columns:
                fa = feature_auc(sub, m)
                if np.isfinite(fa["auc_norm"]):
                    per_phase_auc[m].append(fa["auc_norm"])
    ranked = sorted(metrics,
                    key=lambda m: (np.mean(per_phase_auc[m])
                                   if per_phase_auc[m] else 0.0),
                    reverse=True)
    return ranked[:k]


def prospective_scores(df, features=None, *,
                       n_phases: int = _cfg.DEFAULT_N_PHASES) -> np.ndarray:
    """Per-row PROSPECTIVE predicted preictal-probability of the multivariable
    logistic forecaster -- the paper's combined model output (Fig 3A), usable
    as a metric.

    For each labeled row in phase P>=1 the score is the probability from the
    model TRAINED on phase P-1 (novel data), so it is not overfit. NaN for
    phase 0 (never a test fold) and rows with missing features. Returned
    aligned to *df* rows (positional). *df* must already carry a ``class``
    column (call ``label_classes`` first)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    assert "class" in df.columns, "call label_classes(df) first"
    feats = [f for f in (features or _cfg.PAPER_BEST5) if f in df.columns]
    out = np.full(len(df), np.nan)
    if not feats:
        return out
    dfp, k = assign_epi_phase(df, n_phases)
    ep = dfp["epi_phase"].to_numpy()
    cls = dfp["class"].to_numpy()
    labeled = np.isin(cls, [CLASS_PREICTAL, CLASS_INTERICTAL])
    X = dfp[feats].to_numpy(dtype=float)
    finite = np.all(np.isfinite(X), axis=1)
    for p in range(k - 1):
        tr = labeled & finite & (ep == p)
        te = labeled & finite & (ep == p + 1)
        ytr = (cls[tr] == CLASS_PREICTAL).astype(int)
        if tr.sum() < 5 or te.sum() == 0 or np.unique(ytr).size < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), ytr)
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def logistic_forecast(df, features=None, *, n_phases: int = _cfg.DEFAULT_N_PHASES,
                      select_best: bool = False, metrics=None) -> dict:
    """Prospective multivariable-logistic-regression forecaster (paper Fig 3).

    Split the recording into *n_phases* equal-seizure-count epilepsy phases;
    train a preictal-vs-interictal logistic regression on phase P and test it
    on P+1 (novel data). Returns per-phase test AUCs + ROC curves, the
    normalised feature-importance coefficients, the features used, and the
    phase→AUC linear-trend slope (the paper found AUC rises in later phases).

    *features* defaults to the paper's best-5 (``config.PAPER_BEST5``); with
    ``select_best=True`` the top-5 features are chosen data-drivenly from
    *metrics* by mean per-phase AUC.
    """
    if features is None:
        features = (select_best_features(df, metrics or _cfg.CHEAP_METRICS)
                    if select_best else list(_cfg.PAPER_BEST5))
    features = [f for f in features if f in df.columns]
    assert features, "no usable features for the forecaster"
    dfp, k = assign_epi_phase(df, n_phases)
    phases = []
    for p in range(k - 1):
        tr = dfp[dfp["epi_phase"] == p]
        te = dfp[dfp["epi_phase"] == p + 1]
        Xtr, ytr = _xy(tr, features)
        Xte, yte = _xy(te, features)
        auc, fpr, tpr = _fit_predict_auc(Xtr, ytr, Xte, yte)
        phases.append({"train_phase": p, "test_phase": p + 1, "auc": auc,
                       "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
                       "fpr": (fpr.tolist() if fpr is not None else None),
                       "tpr": (tpr.tolist() if tpr is not None else None)})
    aucs = np.array([ph["auc"] for ph in phases], dtype=float)
    ok = np.isfinite(aucs)
    slope = float("nan")
    if ok.sum() >= 2:
        xs = np.arange(len(aucs))[ok]
        slope = float(np.polyfit(xs, aucs[ok], 1)[0])
    return {"features": features, "n_phases": k, "phases": phases,
            "mean_auc": float(np.nanmean(aucs)) if ok.any() else float("nan"),
            "phase_auc_slope": slope,
            "coefficients": normalized_coefficients(df, features)}
