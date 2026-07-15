"""Per-seizure trend test: the honest way to ask "does an evoked feature change
toward seizure onset?" without pseudo-replicating on the thousands of
autocorrelated stimuli.

The unit of replication is the SEIZURE, not the stimulus. So each seizure is
collapsed to ONE number — the Spearman correlation of the feature with lead-time
over that seizure's pre-onset stimuli — and the test asks whether those
per-seizure trends are consistent across seizures (Wilcoxon signed-rank vs 0, with
a sign-test fallback at the small seizure counts typical here). A POST-onset window
is the built-in positive control (post-ictal depression is real, so the test
should fire there); hour-of-day is the circadian-confound readout; an all-features
scan carries Benjamini-Hochberg q-values so scanning many features doesn't inflate
false positives.

Pure + Dash-free so it can be unit-tested. Operates on the ``event x metric``
matrix (``src.periictal.matrix``): columns ``time_to_onset_sec`` (signed:
positive = pre-onset, negative = post-onset), ``seizure_idx``, ``hour_of_day``,
``phase`` (``pre``/``post``), and the feature columns.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

_MAX_SEIZURES = 1_000_000        # NASA Rule 2: explicit loop bound.
_MAX_PERM = 100_000


def _safe_spearman(a: np.ndarray, b: np.ndarray):
    """Spearman rho, or None when undefined (constant input / < 3 points)."""
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None
    rho, _p = spearmanr(a, b)
    return None if not np.isfinite(rho) else float(rho)


def per_seizure_trend(df, feature: str, *, min_n: int = 5) -> list:
    """One trend statistic PER seizure for *feature* over the rows in *df* (the
    caller passes the pre- or post-onset subset).

    For each ``seizure_idx`` with >= *min_n* finite rows: Spearman
    ``rho(feature, time_to_onset_sec)`` plus ``hour_rho = rho(feature,
    hour_of_day)`` as the circadian-confound readout. Seizures with too few rows
    or zero variance are skipped. Returns a list of
    ``{seizure_idx, rho, n, hour_rho}``."""
    assert feature in df.columns, f"feature '{feature}' not in df"
    for col in ("time_to_onset_sec", "seizure_idx", "hour_of_day"):
        assert col in df.columns, f"matrix missing column '{col}'"
    sids = np.unique(df["seizure_idx"].to_numpy())
    assert sids.size < _MAX_SEIZURES, "seizure count runaway"
    out: list = []
    for sid in sids:
        g = df[df["seizure_idx"] == sid]
        y = g[feature].to_numpy(dtype=float)
        t = g["time_to_onset_sec"].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(t)
        if int(ok.sum()) < min_n:
            continue
        rho = _safe_spearman(y[ok], t[ok])
        if rho is None:
            continue
        h = g["hour_of_day"].to_numpy(dtype=float)
        hok = np.isfinite(y) & np.isfinite(h)
        hr = _safe_spearman(y[hok], h[hok]) if int(hok.sum()) >= min_n else None
        out.append({"seizure_idx": int(sid), "rho": rho, "n": int(ok.sum()),
                    "hour_rho": float(hr) if hr is not None else float("nan")})
    return out


def across_seizure_test(rhos) -> dict:
    """Test whether the per-seizure *rhos* are consistently non-zero: Wilcoxon
    signed-rank vs 0, with a binomial sign-test fallback when < 6 seizures (where
    Wilcoxon is unreliable). Returns ``{stat, p, n_seizures, median_rho,
    direction, n_pos, n_neg}`` (``direction`` = sign of the median rho)."""
    r = np.asarray([x for x in np.asarray(rhos, dtype=float) if np.isfinite(x)],
                   dtype=float)
    assert r.ndim == 1, "rhos must be 1-D"
    assert r.size < _MAX_SEIZURES, "too many seizures"
    n = int(r.size)
    n_pos, n_neg = int(np.sum(r > 0)), int(np.sum(r < 0))
    med = float(np.median(r)) if n else float("nan")
    direction = "negative" if med < 0 else "positive" if med > 0 else "none"
    base = {"n_seizures": n, "median_rho": med, "direction": direction,
            "n_pos": n_pos, "n_neg": n_neg}
    if n == 0:
        return {**base, "stat": float("nan"), "p": float("nan")}
    if n < 6 or np.allclose(r, 0.0):
        m = n_pos + n_neg
        if m == 0:
            return {**base, "stat": 0.0, "p": 1.0}
        p = float(binomtest(min(n_pos, n_neg), m, 0.5,
                            alternative="two-sided").pvalue)
        return {**base, "stat": float(max(n_pos, n_neg)), "p": p}
    try:
        stat, p = wilcoxon(r)
    except ValueError:                                   # all-zero / degenerate
        stat, p = 0.0, 1.0
    return {**base, "stat": float(stat), "p": float(p)}


def positive_control(df, feature: str, *, min_n: int = 5) -> dict:
    """The across-seizure test on the POST-onset window (``phase == 'post'``):
    post-ictal, where an effect is EXPECTED — a sensitivity check. Returns the
    ``across_seizure_test`` dict plus ``available`` (whether any post rows
    exist)."""
    assert "phase" in df.columns, "matrix must carry a 'phase' column"
    assert feature in df.columns, f"feature '{feature}' not in df"
    post = df[df["phase"] == "post"]
    res = across_seizure_test(
        [r["rho"] for r in per_seizure_trend(post, feature, min_n=min_n)])
    res["available"] = bool(len(post) > 0)
    return res


def _bh_qvalues(ps: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for a 1-D array of p-values (NaNs preserved)."""
    ps = np.asarray(ps, dtype=float)
    q = np.full(ps.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(ps))
    m = int(finite.size)
    if m == 0:
        return q
    order = finite[np.argsort(ps[finite])]
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]   # enforce monotonicity
    q[order] = np.minimum(ranked, 1.0)
    return q


def scan_all_features(df, features, *, min_n: int = 5) -> list:
    """Run the pre-onset across-seizure test for every feature in *features*,
    attach BH-FDR *q* across them, the circadian median ``|hour_rho|``, and the
    post-ictal positive-control *p*. Returns a list of dicts sorted by *q*."""
    assert "phase" in df.columns, "matrix must carry a 'phase' column"
    assert features is not None, "features required"
    pre = df[df["phase"] == "pre"]
    rows: list = []
    for f in features:
        if f not in df.columns:
            continue
        per = per_seizure_trend(pre, f, min_n=min_n)
        main = across_seizure_test([r["rho"] for r in per])
        circ = (float(np.nanmedian([abs(r["hour_rho"]) for r in per]))
                if per else float("nan"))
        pc = positive_control(df, f, min_n=min_n)
        rows.append({"feature": f, "p": main["p"], "q": float("nan"),
                     "n_seizures": main["n_seizures"],
                     "median_rho": main["median_rho"],
                     "direction": main["direction"], "circadian": circ,
                     "post_p": pc["p"], "post_available": pc["available"]})
    q = _bh_qvalues(np.array([r["p"] for r in rows], dtype=float))
    for r, qi in zip(rows, q):
        r["q"] = float(qi)
    rows.sort(key=lambda r: (np.inf if not np.isfinite(r["q"]) else r["q"]))
    return rows


def _permute_within(values: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Permute *values* WITHIN each group (breaks the trend, keeps group sizes)."""
    out = values.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = rng.permutation(values[idx])
    return out


def surrogate_p(df, feature: str, *, n_perm: int = 1000, seed: int = 0,
                min_n: int = 5) -> dict:
    """Permutation null for the pre-onset trend: within each seizure permute
    ``time_to_onset_sec`` (breaking the trend), recompute the across-seizure
    statistic ``|median rho|``, empirical two-sided p. Slow — an on-demand
    rigour check, not for the interactive render. Returns
    ``{p, n_perm, observed}``."""
    assert 1 <= int(n_perm) <= _MAX_PERM, "n_perm out of bounds"
    assert feature in df.columns, f"feature '{feature}' not in df"
    pre = df[df["phase"] == "pre"]
    obs = abs(across_seizure_test(
        [r["rho"] for r in per_seizure_trend(pre, feature, min_n=min_n)]
    )["median_rho"])
    if not np.isfinite(obs):
        return {"p": float("nan"), "n_perm": 0, "observed": obs}
    rng = np.random.default_rng(seed)
    sids = pre["seizure_idx"].to_numpy()
    tto = pre["time_to_onset_sec"].to_numpy(dtype=float)
    ge = 0
    for i in range(int(n_perm)):
        assert i < _MAX_PERM, "permutation runaway"
        d2 = pre.assign(time_to_onset_sec=_permute_within(tto, sids, rng))
        stat = abs(across_seizure_test(
            [r["rho"] for r in per_seizure_trend(d2, feature, min_n=min_n)]
        )["median_rho"])
        if np.isfinite(stat) and stat >= obs - 1e-12:
            ge += 1
    return {"p": float((ge + 1) / (int(n_perm) + 1)), "n_perm": int(n_perm),
            "observed": float(obs)}
