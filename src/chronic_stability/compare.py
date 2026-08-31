"""Distribution comparisons + age/stability collinearity for the stability map.

Comparisons are DISTRIBUTIONAL, not point estimates: KS (any difference),
Mann-Whitney U (location / stochastic dominance), Cliff's delta effect size,
and a median difference with a DEPENDENCE-AWARE block-bootstrap CI. Consecutive
stimuli are autocorrelated, so an iid bootstrap would fabricate a tight CI --
we resample contiguous blocks sized from the integrated autocorrelation time
(``src.periictal.resample``), mirroring the peri-ictal forecast code.

Collinearity: configuration age and stability are typically confounded (an older
configuration has drifted further). We DETECT and REPORT this (Spearman rho of
age vs each metric and vs Rₐ, plus the age band the stable window occupies) --
we do NOT adjust it away.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.periictal.forecast import rank_auc
from src.periictal.resample import block_length, moving_block_resample

_MAX_BOOT = 100_000         # NASA Rule 2: explicit bound on the bootstrap loop.


def cliffs_delta(a, b) -> float:
    """Cliff's delta = 2*P(a>b) - 1 in [-1, 1]; 0 = stochastic equality."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.size and b.size, "both groups must be non-empty"
    return float(2.0 * rank_auc(a, b) - 1.0)


def _clean(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def blockboot_median_diff_ci(a, b, *, n_boot: int = 2000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Percentile CI for ``median(a) - median(b)`` via an independent moving-
    block bootstrap of each group (block length from its autocorrelation time)."""
    a, b = _clean(a), _clean(b)
    assert a.size and b.size, "both groups must be non-empty"
    assert 0 < n_boot <= _MAX_BOOT, "n_boot out of range"
    rng = np.random.default_rng(seed)
    ba, bb = block_length(a), block_length(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        assert i < _MAX_BOOT, "bootstrap runaway"
        ra = moving_block_resample(a, ba, rng)
        rb = moving_block_resample(b, bb, rng)
        diffs[i] = np.median(ra) - np.median(rb)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"median_diff": float(np.median(a) - np.median(b)),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "block_a": int(ba), "block_b": int(bb),
            "n_eff_a": float(a.size / ba), "n_eff_b": float(b.size / bb)}


def compare_two_groups(a, b, *, name: str = "", n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Full distributional comparison of two metric samples."""
    a, b = _clean(a), _clean(b)
    assert a.size >= 2 and b.size >= 2, "each group needs >=2 finite values"
    ks = stats.ks_2samp(a, b)
    mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    ci = blockboot_median_diff_ci(a, b, n_boot=n_boot, seed=seed)
    return {
        "name": name, "n_a": int(a.size), "n_b": int(b.size),
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "ks_D": float(ks.statistic), "ks_p": float(ks.pvalue),
        "mwu_U": float(mwu.statistic), "mwu_p": float(mwu.pvalue),
        "cliffs_delta": cliffs_delta(a, b), **ci,
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def compare_polarity(pol_a, pol_b, *, name: str = "") -> dict:
    """Categorical polarity comparison: modal-sign proportions with Wilson CIs
    + a 2x2 Fisher exact test (group x sign)."""
    pa = np.asarray(pol_a, int)
    pb = np.asarray(pol_b, int)
    assert pa.size and pb.size, "both polarity series must be non-empty"
    ka, kb = int((pa == 1).sum()), int((pb == 1).sum())
    na, nb = pa.size, pb.size
    odds, p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])
    return {
        "name": name, "frac_pos_a": ka / na, "frac_pos_b": kb / nb,
        "wilson_a": _wilson_ci(ka, na), "wilson_b": _wilson_ci(kb, nb),
        "fisher_odds": float(odds), "fisher_p": float(p),
    }


def age_confound(epoch, metric, config_start_epoch: float) -> dict:
    """Spearman rho between per-trial configuration AGE and a *metric* -- the
    core collinearity readout. Positive/negative rho means the metric drifts
    monotonically with how old the configuration is."""
    epoch = np.asarray(epoch, float)
    metric = np.asarray(metric, float)
    assert epoch.shape == metric.shape, "epoch/metric length mismatch"
    age_h = (epoch - config_start_epoch) / 3600.0
    m = np.isfinite(age_h) & np.isfinite(metric)
    if m.sum() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(m.sum())}
    rho, p = stats.spearmanr(age_h[m], metric[m])
    return {"rho": float(rho), "p": float(p), "n": int(m.sum()),
            "age_lo_h": float(age_h[m].min()), "age_hi_h": float(age_h[m].max())}


def stable_age_band(stable_epochs, config_epochs, config_start: float) -> dict:
    """The configuration-age band the stable window occupies vs the config's
    full age range -- makes 'stable' visibly coincide (or not) with a
    particular age, so a stable-vs-full difference is read jointly with age."""
    se = np.asarray(stable_epochs, float)
    ce = np.asarray(config_epochs, float)
    assert ce.size, "config must have trials"
    s_age = (se - config_start) / 3600.0
    c_age = (ce - config_start) / 3600.0
    return {"stable_age_lo_h": float(s_age.min()) if se.size else float("nan"),
            "stable_age_hi_h": float(s_age.max()) if se.size else float("nan"),
            "config_age_lo_h": float(c_age.min()),
            "config_age_hi_h": float(c_age.max()),
            "stable_frac_of_config_age":
                float((s_age.max() - s_age.min()) / (c_age.max() - c_age.min()))
                if se.size and c_age.max() > c_age.min() else float("nan")}
