"""Vectorized per-epoch evoked-response features.

A numpy/scipy port of the lab's MATLAB ``FeatureExtractor.m`` /
``RecoveryFitter.m`` so the Chronic Evoked Analyzer can compute the same
~25 measures the toolkit shows, directly from the stored ``evokedData``
traces. The traces in the ``*_evoked.mat`` files are already filtered and
baseline-corrected at extraction, so the math runs on them as-is.

Every function takes ``traces`` shaped ``[epochs x samples]`` (time along
axis 1) and ``time_ms`` shaped ``[samples]`` (ms, t=0 = stim), and returns
one value per epoch (shape ``[epochs]``). ``dt = 1000/fs`` ms per sample.

The cheap features are a single vectorized pass; the expensive ones
(Recovery Tau/Slope, Template Correlation, AC Width, PCA Recon Error,
Exp-Fit A) fit per epoch and are computed only when asked (opt-in).
"""

from __future__ import annotations

import numpy as np

# Spectral integration bands (Hz), matching FeatureExtractor.m.
_LOW_BAND = (1.0, 64.0)
_HIGH_SUM_BAND = (256.0, 1024.0)
_MOMENT_HIGH_BAND = (64.0, 256.0)
_EPS = 1e-10

CHEAP_COLUMNS = [
    "line_length", "log_auc", "peak_amplitude", "trough_amplitude",
    "peak_to_trough", "rms_amplitude", "variance", "peak_latency_ms",
    "trough_latency_ms", "max_slope", "max_slope_time_ms", "early_area",
    "late_area", "early_late_ratio", "autocorrelation", "sum_power_low",
    "freq_moment_low", "sum_power_high", "freq_moment_high",
]
EXPENSIVE_COLUMNS = [
    "recovery_tau", "recovery_slope", "template_correlation",
    "pca_recon_error", "ac_width", "exp_fit_a",
]
ALL_COLUMNS = CHEAP_COLUMNS + EXPENSIVE_COLUMNS

_MAX_EPOCHS = 1_000_000     # NASA Rule 2: explicit per-epoch loop bound.


def _check(traces) -> np.ndarray:
    """Validate + return a float64 ``[epochs x samples]`` view."""
    a = np.asarray(traces, dtype=np.float64)
    assert a.ndim == 2, "traces must be 2-D [epochs x samples]"
    assert a.shape[1] >= 2, "need >=2 samples per epoch"
    return a


# --------------------------------------------------------------------- #
#  Cheap features (vectorized over all epochs at once)
# --------------------------------------------------------------------- #

def line_length(traces) -> np.ndarray:
    a = _check(traces)
    return np.sum(np.abs(np.diff(a, axis=1)), axis=1)


def log_auc(traces, dt: float) -> np.ndarray:
    a = _check(traces)
    assert dt > 0, "dt must be positive"
    return np.log(np.sum(np.abs(a), axis=1) * dt + _EPS)


def peak(traces) -> np.ndarray:
    return np.max(_check(traces), axis=1)


def trough(traces) -> np.ndarray:
    return np.min(_check(traces), axis=1)


def peak_to_trough(traces) -> np.ndarray:
    a = _check(traces)
    return np.max(a, axis=1) - np.min(a, axis=1)


def rms(traces) -> np.ndarray:
    a = _check(traces)
    return np.sqrt(np.mean(a * a, axis=1))


def variance(traces) -> np.ndarray:
    # MATLAB var default normalizes by N-1.
    return np.var(_check(traces), axis=1, ddof=1)


def _latency(a, time_ms, idx) -> np.ndarray:
    t = np.asarray(time_ms, dtype=np.float64)
    assert t.shape[0] == a.shape[1], "time_ms length must match samples"
    return t[idx]


def peak_latency_ms(traces, time_ms) -> np.ndarray:
    a = _check(traces)
    return _latency(a, time_ms, np.argmax(a, axis=1))


def trough_latency_ms(traces, time_ms) -> np.ndarray:
    a = _check(traces)
    return _latency(a, time_ms, np.argmin(a, axis=1))


def max_slope(traces, dt: float) -> np.ndarray:
    a = _check(traces)
    assert dt > 0, "dt must be positive"
    return np.max(np.abs(np.diff(a, axis=1)) / dt, axis=1)


def max_slope_time_ms(traces, time_ms, dt: float) -> np.ndarray:
    a = _check(traces)
    assert dt > 0, "dt must be positive"
    idx = np.argmax(np.abs(np.diff(a, axis=1)), axis=1)
    t = np.asarray(time_ms, dtype=np.float64)
    return t[idx]


def _band_area(a, time_ms, lo_ms, hi_ms) -> np.ndarray:
    t = np.asarray(time_ms, dtype=np.float64)
    assert t.shape[0] == a.shape[1], "time_ms length must match samples"
    mask = (t >= lo_ms) & (t <= hi_ms)
    if not np.any(mask):
        return np.zeros(a.shape[0])
    return np.sum(np.abs(a[:, mask]), axis=1)


def early_area(traces, time_ms) -> np.ndarray:
    return _band_area(_check(traces), time_ms, 0.0, 50.0)


def late_area(traces, time_ms) -> np.ndarray:
    return _band_area(_check(traces), time_ms, 50.0, 200.0)


def early_late_ratio(traces, time_ms) -> np.ndarray:
    a = _check(traces)
    early = _band_area(a, time_ms, 0.0, 50.0)
    late = _band_area(a, time_ms, 50.0, 200.0)
    return early / (late + _EPS)


def autocorrelation(traces) -> np.ndarray:
    """Lag-1 autocorrelation per epoch (Maturana 2020 form)."""
    a = _check(traces)
    y = a - a.mean(axis=1, keepdims=True)
    num = np.sum(y[:, :-1] * y[:, 1:], axis=1)
    den = np.sum(y * y, axis=1) + _EPS
    return num / den


def spectral(traces, fs: float) -> dict:
    """Periodogram band powers + power-weighted freq moments per epoch."""
    a = _check(traces)
    assert fs > 0, "fs must be positive"
    from scipy.signal import periodogram
    f, pxx = periodogram(a, fs=fs, axis=1)
    return {
        "sum_power_low": _band_sum(f, pxx, _LOW_BAND),
        "sum_power_high": _band_sum(f, pxx, _HIGH_SUM_BAND),
        "freq_moment_low": _freq_moment(f, pxx, _LOW_BAND),
        "freq_moment_high": _freq_moment(f, pxx, _MOMENT_HIGH_BAND),
    }


def _band_sum(f, pxx, band) -> np.ndarray:
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.zeros(pxx.shape[0])
    return np.sum(pxx[:, mask], axis=1)


def _freq_moment(f, pxx, band) -> np.ndarray:
    mask = (f >= band[0]) & (f <= band[1])
    if not np.any(mask):
        return np.full(pxx.shape[0], np.nan)
    p = pxx[:, mask]
    fb = f[mask]
    denom = np.sum(p, axis=1) + _EPS
    return (p @ fb) / denom


def compute_cheap(traces, time_ms, fs: float) -> dict:
    """All cheap features as a column->``[epochs]`` dict (one pass)."""
    a = _check(traces)
    dt = 1000.0 / fs
    out = {
        "line_length": line_length(a),
        "log_auc": log_auc(a, dt),
        "peak_amplitude": peak(a),
        "trough_amplitude": trough(a),
        "peak_to_trough": peak_to_trough(a),
        "rms_amplitude": rms(a),
        "variance": variance(a),
        "peak_latency_ms": peak_latency_ms(a, time_ms),
        "trough_latency_ms": trough_latency_ms(a, time_ms),
        "max_slope": max_slope(a, dt),
        "max_slope_time_ms": max_slope_time_ms(a, time_ms, dt),
        "early_area": early_area(a, time_ms),
        "late_area": late_area(a, time_ms),
        "early_late_ratio": early_late_ratio(a, time_ms),
        "autocorrelation": autocorrelation(a),
    }
    out.update(spectral(a, fs))
    return out


# --------------------------------------------------------------------- #
#  Expensive features (per-epoch fits; opt-in)
# --------------------------------------------------------------------- #

def _movmean2d(a, w: int) -> np.ndarray:
    """Centered moving average along axis 1 (edge windows shrink)."""
    n = a.shape[1]
    if w <= 1 or n == 0:
        return a.astype(np.float64, copy=True)
    half = w // 2
    cs = np.cumsum(np.insert(a, 0, 0.0, axis=1), axis=1)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    return (cs[:, hi] - cs[:, lo]) / (hi - lo)


def recovery_tau(traces, time_ms) -> np.ndarray:
    """Exp-decay time constant of the post-peak Hilbert envelope."""
    from scipy.signal import hilbert
    from scipy.optimize import fmin
    a = _check(traces)
    t = np.asarray(time_ms, dtype=np.float64)
    env = _movmean2d(np.abs(hilbert(a, axis=1)), 5)
    pre = t < 0
    win = (t >= 0) & (t <= 50)
    out = np.full(a.shape[0], np.nan)
    assert a.shape[0] < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(a.shape[0]):
        out[i] = _one_tau(env[i], t, pre, win, fmin)
    return out


def _one_tau(env, t, pre, win, fmin) -> float:
    if not np.any(win):
        return np.nan
    pk = np.where(win)[0][np.argmax(env[win])]
    seg = env[pk:]
    trel = t[pk:] - t[pk]
    base = np.mean(env[pre]) if np.any(pre) else np.mean(seg[-max(1, seg.size // 10):])
    y = seg - base
    if y.size < 3 or y[0] <= 0:
        return np.nan
    keep = y > 0.01 * y[0]
    if np.count_nonzero(keep) < 2:
        return np.nan
    p = np.polyfit(trel[keep], np.log(y[keep]), 1)
    if p[0] >= 0:
        return np.nan
    tau0 = -1.0 / p[0]
    obj = lambda q: np.sum((y - q[0] * np.exp(-trel / np.exp(q[1]))) ** 2)
    res = fmin(obj, [y[0], np.log(tau0)], disp=False, maxiter=200)
    tau = float(np.exp(res[1]))
    return tau if 0 < tau <= trel[-1] else np.nan


def recovery_slope(traces, time_ms) -> np.ndarray:
    """Linear slope of each trace from its abs-peak to the end."""
    a = _check(traces)
    t = np.asarray(time_ms, dtype=np.float64)
    pk = np.argmax(np.abs(a), axis=1)
    out = np.full(a.shape[0], np.nan)
    assert a.shape[0] < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(a.shape[0]):
        seg = a[i, pk[i]:]
        if seg.size >= 2:
            out[i] = np.polyfit(t[pk[i]:], seg, 1)[0]
    return out


def template_correlation(traces) -> np.ndarray:
    """Pearson r of each epoch vs the median of the previous 10 epochs."""
    a = _check(traces)
    n = a.shape[0]
    out = np.full(n, np.nan)
    assert n < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(n):
        prev = a[max(0, i - 10):i] if i > 0 else a[:1]
        tmpl = np.median(prev, axis=0)
        if np.std(a[i]) < _EPS or np.std(tmpl) < _EPS:
            continue
        out[i] = np.corrcoef(a[i], tmpl)[0, 1]
    return out


def ac_width(traces) -> np.ndarray:
    """First positive lag where the autocorrelation drops below 0.5."""
    a = _check(traces)
    n = a.shape[0]
    out = np.full(n, np.nan)
    assert n < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(n):
        out[i] = _one_ac_width(a[i])
    return out


def _one_ac_width(x) -> float:
    y = x - x.mean()
    denom = np.sum(y * y)
    if denom < _EPS:
        return np.nan
    # FFT autocorrelation (O(N log N)); np.correlate(full) is O(N^2) and
    # intractable for the ~20k-sample traces.
    n = y.size
    fft = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(fft * np.conj(fft))[:n] / denom
    below = np.where(acf < 0.5)[0]
    if below.size == 0:
        return float(acf.size)
    k = below[0]
    if k == 0:
        return 0.0
    # Linear-interpolate the half-max crossing between lags k-1 and k.
    a1, a2 = acf[k - 1], acf[k]          # a1 >= 0.5 > a2
    return float((k - 1) + (0.5 - a1) / (a2 - a1 + _EPS))


def pca_recon_error(traces) -> np.ndarray:
    """Per-epoch reconstruction error vs a top-3 PCA of the first epochs."""
    a = _check(traces)
    n = a.shape[0]
    base_n = min(20, max(2, n // 4))
    if n < 4:
        return np.full(n, np.nan)
    base = a[:base_n]
    mean = base.mean(axis=0)
    bc = base - mean
    _, _, vt = np.linalg.svd(bc, full_matrices=False)
    ncomp = min(3, base_n - 1)
    comps = vt[:ncomp]
    out = np.empty(n)
    assert n < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(n):
        d = a[i] - mean
        recon = d @ comps.T @ comps
        out[i] = np.linalg.norm(d - recon)
    base_mean = np.mean(out[:base_n]) + _EPS
    return out / base_mean


def exp_fit_a(traces, time_ms) -> np.ndarray:
    """Amplitude A of an exp fit to the pre-peak rising |trace|."""
    a = _check(traces)
    t = np.asarray(time_ms, dtype=np.float64)
    pk = np.argmax(np.abs(a), axis=1)
    out = np.full(a.shape[0], np.nan)
    assert a.shape[0] < _MAX_EPOCHS, "epoch count exceeds bound"
    for i in range(a.shape[0]):
        out[i] = _one_exp_a(np.abs(a[i, :pk[i] + 1]), t[:pk[i] + 1])
    return out


def _one_exp_a(y, t) -> float:
    if y.size < 2:
        return np.nan
    mask = y > 0.05 * np.max(y)
    if np.count_nonzero(mask) < 2:
        return np.nan
    p = np.polyfit(t[mask], np.log(y[mask] + _EPS), 1)
    return float(np.exp(p[1]))


def compute_expensive(traces, time_ms, fs: float) -> dict:
    """All expensive (per-epoch fit) features as a column->array dict."""
    a = _check(traces)
    return {
        "recovery_tau": recovery_tau(a, time_ms),
        "recovery_slope": recovery_slope(a, time_ms),
        "template_correlation": template_correlation(a),
        "pca_recon_error": pca_recon_error(a),
        "ac_width": ac_width(a),
        "exp_fit_a": exp_fit_a(a, time_ms),
    }


def compute_all(traces, time_ms, fs: float, expensive: bool = False) -> dict:
    """Cheap features always; expensive ones only when *expensive*.

    Columns not computed are present as all-NaN arrays so the cache schema
    is uniform regardless of the flag.
    """
    a = _check(traces)
    out = compute_cheap(a, time_ms, fs)
    if expensive:
        out.update(compute_expensive(a, time_ms, fs))
    else:
        nan = np.full(a.shape[0], np.nan)
        for col in EXPENSIVE_COLUMNS:
            out[col] = nan.copy()
    return out


# --------------------------------------------------------------------- #
#  Per-animal rolling series (used by the tab on queried epoch values)
# --------------------------------------------------------------------- #

def rolling_centered(x, win: int, kind: str = "mean") -> np.ndarray:
    """Centered rolling mean / std / cv / ar1 over a 1-D series."""
    a = np.asarray(x, dtype=np.float64)
    assert a.ndim == 1, "rolling input must be 1-D"
    assert win >= 1, "win must be >= 1"
    n = a.size
    half = win // 2
    out = np.full(n, np.nan)
    for i in range(n):
        seg = a[max(0, i - half):min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        out[i] = _roll_stat(seg, kind)
    return out


def _roll_stat(seg, kind: str) -> float:
    if seg.size == 0:
        return np.nan
    if kind == "mean":
        return float(np.mean(seg))
    if kind == "std":
        return float(np.std(seg, ddof=1)) if seg.size > 1 else np.nan
    if kind == "cv":
        m = np.mean(seg)
        return float(np.std(seg, ddof=1) / m) if (seg.size > 1 and m != 0) else np.nan
    if kind == "ar1":
        if seg.size < 3:
            return np.nan
        y = seg - seg.mean()
        return float(np.sum(y[:-1] * y[1:]) / (np.sum(y * y) + _EPS))
    raise ValueError(f"unknown rolling kind: {kind}")
