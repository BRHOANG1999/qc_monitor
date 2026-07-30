"""Slow-dynamics estimators on the across-trial feature series (Part B, B1/B2).

THE EIGENVALUE (B1). Fit AR(1) across trials: phi = lag-1 autocorrelation of the
(detrended) feature series; lambda = ln(phi)/dt maps it to the continuous-time
eigenvalue. phi -> 1 (lambda -> 0) as a saddle-node is approached. **Report phi or
lambda, NEVER tau = -1/lambda** -- tau is badly conditioned near phi=1 (phi 0.982
-> tau 110 s, phi 0.998 -> 1000 s), so the seductive number is the unstable one.

DECOUPLING TEST (B2). Under critical slowing the stationary AR(1) variance
sigma^2/(1-phi^2) means variance and phi must rise together. variance up with phi
FLAT => injected noise, not slowing (rules out CSD without ruling out a state
change); both up => consistent with CSD. A clean falsifier on existing data.

Guards baked in: (a) slow trends inflate phi (the rho~0.67 circadian confound), so
phi is DETRENDED by default with a kernel far wider than the AR window (standard
Dakos/Scheffer CSD practice); (b) everything is computed WITHIN gap-free segments
of the TrialSeries (never across a file-boundary hole); (c) statistical power
scales with SEIZURE count, not trial count. Reuses resample.py for phi CIs that
respect within-series autocorrelation.
"""

from __future__ import annotations

import numpy as np

from src.utils.evoked_features import rolling_centered

_EPS = 1e-12


def _ar1(y: np.ndarray) -> float:
    """Biased lag-1 autocorrelation (AR(1) phi) of a 1-D array; NaN if < 3 pts or
    no variance. Matches evoked_features._roll_stat('ar1')."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return float("nan")
    yc = y - y.mean()
    denom = float(np.sum(yc * yc))
    if denom <= 0:
        return float("nan")
    return float(np.sum(yc[:-1] * yc[1:]) / (denom + _EPS))


def lambda_from_phi(phi, dt: float):
    """Continuous-time eigenvalue lambda = ln(phi)/dt for a stable relaxational
    AR(1) (0 < phi < 1). NaN outside (0,1): phi<=0 is not a slow relaxation and
    phi>=1 is the non-stationary boundary. Never returns tau."""
    assert dt > 0, "dt must be > 0"
    phi = np.asarray(phi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where((phi > 0) & (phi < 1), np.log(phi) / dt, np.nan)
    return lam if lam.ndim else float(lam)


def fit_ar1(y, *, dt: float = 1.0, alpha: float = 0.05) -> dict:
    """AR(1) phi of a 1-D series + a large-sample CI ``phi +/- z*sqrt((1-phi^2)/n)``
    (Bartlett's SE for an AR(1) lag-1 coefficient) and ``lambda = ln(phi)/dt``.

    Why NOT a moving-block bootstrap here: phi is a SERIAL-correlation statistic,
    and block joins dilute the correlation, so an MBB CI is downward-biased and
    misses the truth at the natural block length (verified). The analytic AR(1) SE
    is unbiased for this use. The block bootstrap is still the right tool for the
    NON-serial AUC/slope forests (forecast/trendtest). Caller detrends first if
    needed. Reports phi/lambda, never tau."""
    assert 0.0 < alpha < 1.0, "alpha in (0,1)"
    assert dt > 0, "dt must be > 0"
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    phi = _ar1(y)
    n = int(y.size)
    lo = hi = se = float("nan")
    if np.isfinite(phi) and n > 3:
        from scipy.stats import norm
        se = float(np.sqrt(max(0.0, 1.0 - phi * phi) / n))
        z = float(norm.ppf(1.0 - alpha / 2.0))
        lo, hi = phi - z * se, phi + z * se
    return {"phi": phi, "ci_lo": float(lo), "ci_hi": float(hi), "se": se,
            "lambda": lambda_from_phi(phi, dt), "n": n}


def detrend(y: np.ndarray, win: int) -> np.ndarray:
    """High-pass by subtracting a centered rolling mean far wider than the AR
    window -- removes the slow trend that would otherwise inflate phi/variance
    (the rho~0.67 circadian-inflation guard)."""
    if win <= 1:
        return np.asarray(y, dtype=float)
    return np.asarray(y, dtype=float) - rolling_centered(y, win, "mean")


def _sliding(series, win: int, kind: str, do_detrend: bool, detrend_win) -> np.ndarray:
    """Sliding AR(1)/variance over each gap-free segment (NaN across gaps)."""
    out = np.full(series.n, np.nan)
    dwin = int(detrend_win or max(win * 4, win + 1))
    for sl in series.segments():
        y = np.asarray(series.values[sl], dtype=float)
        if do_detrend:
            y = detrend(y, dwin)
        if kind == "ar1":
            out[sl] = rolling_centered(y, win, "ar1")
        else:                                        # variance
            out[sl] = rolling_centered(y, win, "std") ** 2
    return out


def phi_series(series, *, win: int, detrend: bool = True, detrend_win=None):
    """Sliding AR(1) phi across the TrialSeries (within gap-free segments)."""
    assert win >= 3, "win must be >= 3 for AR(1)"
    return _sliding(series, win, "ar1", detrend, detrend_win)


def variance_series(series, *, win: int, detrend: bool = True, detrend_win=None):
    """Sliding across-trial variance across the TrialSeries."""
    assert win >= 2, "win must be >= 2 for variance"
    return _sliding(series, win, "var", detrend, detrend_win)


def decoupling(series, *, win: int, detrend: bool = True, detrend_win=None) -> dict:
    """B2 CSD falsifier: sliding phi and variance + their Spearman coupling over
    finite windows. High positive coupling is consistent with critical slowing
    (both rise together); variance rising with phi flat (low/undefined coupling)
    indicates injected noise instead. Returns the two series + coupling_rho."""
    from scipy.stats import spearmanr
    phi = phi_series(series, win=win, detrend=detrend, detrend_win=detrend_win)
    var = variance_series(series, win=win, detrend=detrend, detrend_win=detrend_win)
    ok = np.isfinite(phi) & np.isfinite(var)
    rho = float("nan")
    if int(ok.sum()) >= 3 and np.unique(phi[ok]).size > 1 and np.unique(var[ok]).size > 1:
        rho = float(spearmanr(phi[ok], var[ok]).statistic)
    return {"phi": phi, "variance": var, "coupling_rho": rho, "n": int(ok.sum())}
