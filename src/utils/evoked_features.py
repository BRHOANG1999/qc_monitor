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

from dataclasses import dataclass

import numpy as np

# Spectral integration bands (Hz), matching FeatureExtractor.m.
_LOW_BAND = (1.0, 64.0)
_HIGH_SUM_BAND = (256.0, 1024.0)
_MOMENT_HIGH_BAND = (64.0, 256.0)
# The three canonical Chang et al. 2026 bands (Table 1 #14-19). The repo
# historically computed SumPower for the low/high bands and the spectral
# first-moment for the low/mid bands only (4 of the paper's 6); these two
# constants fill in SumPower[mid] and first-moment[vhigh] so all three bands
# carry BOTH statistics.
_MID_BAND = (64.0, 256.0)            # SumPower[64-256]  (paper #16)
_VHIGH_BAND = (256.0, 1024.0)        # first-moment[256-1024] (paper #19)
_EPS = 1e-10

# --- Chang et al. 2026 evoked-morphology features (Table 1 #1-13) ---------
# The 0.5 s evoked response is split at a "transition point" (|dV/dt|~=0,
# between the fast recovery and the slow drift) into a fast component fit
# with an exponential a*exp(b*t) and a slow component fit with a line
# m*t+c (paper Figure 1F). Our evoked epochs are extracted at +/-200 ms in
# most files (some +/-500 ms), so the analysis window is capped at the
# cross-file-safe 200 ms rather than the paper's 500 ms; raise _CHANG_POST_MS
# if the cohort is uniformly +/-500 ms. All choices under the transition
# detector are reconstructions (the paper's exact algorithm is in Supporting
# Information we don't have) and are documented at each step.
_CHANG_GUARD_MS = 1.0                # skip the stim artifact right after t=0
_CHANG_POST_MS = 200.0               # analysis window end (paper used 500)
_CHANG_SMOOTH_MS = 1.0               # derivative/trace smoothing for detection
_CHANG_MIN_SEG = 5                   # min samples per segment to fit
# Per-band lag-1 autocorrelation bands (Table 2 passive #11-13).
_AC_BANDS = {
    "autocorr_low": (1.0, 64.0),
    "autocorr_mid": (64.0, 256.0),
    "autocorr_high": (256.0, 1024.0),
}

# Per-epoch Morlet-wavelet band power. Gamma-and-up only: the evoked
# window is short (<=~200 ms), so lower bands aren't resolvable by a
# faithful CWT. The MATLAB pipeline (run_pipeline.m) computes the same
# columns for the DB path; this Python copy feeds the chronic sidecar.
_WAVELET_BANDS = {
    "wavelet_power_slow_gamma": (30.0, 50.0),
    "wavelet_power_gamma": (50.0, 100.0),
    "wavelet_power_high_gamma": (100.0, 200.0),
}

CHEAP_COLUMNS = [
    "line_length", "log_auc", "peak_amplitude", "trough_amplitude",
    "peak_to_trough", "rms_amplitude", "variance", "peak_latency_ms",
    "trough_latency_ms", "max_slope", "max_slope_time_ms", "early_area",
    "late_area", "early_late_ratio", "autocorrelation", "sum_power_low",
    "freq_moment_low", "sum_power_high", "freq_moment_high",
    "wavelet_power_slow_gamma", "wavelet_power_gamma",
    "wavelet_power_high_gamma",
    # Chang et al. 2026 additions (Table 1 / Table 2), vectorized + cheap.
    "sum_power_mid", "freq_moment_vhigh", "curvature", "skewness",
    "tp_latency_ms", "tp_amplitude",
    "expfit_decay", "expfit_initial", "expfit_rms", "expfit_curvature",
    "expfit_skew", "expfit_area",
    "linfit_slope", "linfit_intercept", "linfit_rms", "linfit_curvature",
    "linfit_skew",
]
EXPENSIVE_COLUMNS = [
    "recovery_tau", "recovery_slope", "template_correlation",
    "pca_recon_error", "ac_width", "exp_fit_a",
    # Per-band lag-1 autocorrelation (Table 2 passive #11-13): 3 filtfilt
    # passes, so opt-in with the other per-epoch/expensive features.
    "autocorr_low", "autocorr_mid", "autocorr_high",
]
ALL_COLUMNS = CHEAP_COLUMNS + EXPENSIVE_COLUMNS

# One-line, code-accurate descriptions of each feature column — the source of
# truth for the "features used" reference shown in the UI (a validation aid).
# KEEP IN SYNC with the functions below; band/window numbers are interpolated
# from the constants above so they cannot silently drift. `y` = the trace over
# an epoch; `dt = 1000/fs` ms per sample; t=0 is the stimulus.
COLUMN_DOCS: dict[str, str] = {
    "line_length": "Σ|Δy| — sum of |sample-to-sample differences| (waveform path "
                   "length / wiggliness).",
    "log_auc": "log(Σ|y|·dt + ε) — log area under the rectified trace.",
    "peak_amplitude": "max(y) — the most positive sample.",
    "trough_amplitude": "min(y) — the most negative sample.",
    "peak_to_trough": "max(y) − min(y) — full peak-to-trough amplitude.",
    "rms_amplitude": "√mean(y²) — root-mean-square amplitude.",
    "variance": "var(y, ddof=1) — sample variance (÷N−1). A critical-slowing "
                "early-warning signal.",
    "peak_latency_ms": "time at argmax(y) — latency of the positive peak (ms "
                       "after stim).",
    "trough_latency_ms": "time at argmin(y) — latency of the negative trough.",
    "max_slope": "max(|Δy|/dt) — steepest instantaneous slope (per ms).",
    "max_slope_time_ms": "time at argmax(|Δy|) — when the steepest slope occurs.",
    "early_area": "Σ|y| over 0–50 ms — rectified area in the early post-stim "
                  "window.",
    "late_area": "Σ|y| over 50–200 ms — rectified area in the late post-stim "
                 "window.",
    "early_late_ratio": "early_area / (late_area + ε) — early-vs-late energy "
                        "balance.",
    "autocorrelation": "lag-1 autocorrelation Σ(yₜ·yₜ₊₁)/Σyₜ² on the "
                       "mean-subtracted trace (Maturana 2020). A critical-slowing "
                       "early-warning signal (AR(1); ≈ exp(−dt/τ)).",
    "sum_power_low": f"Σ periodogram power, {_LOW_BAND[0]:g}–{_LOW_BAND[1]:g} Hz.",
    "freq_moment_low": "power-weighted mean frequency (spectral centroid), "
                       f"{_LOW_BAND[0]:g}–{_LOW_BAND[1]:g} Hz.",
    "sum_power_high": "Σ periodogram power, "
                      f"{_HIGH_SUM_BAND[0]:g}–{_HIGH_SUM_BAND[1]:g} Hz.",
    "freq_moment_high": "power-weighted mean frequency, "
                        f"{_MOMENT_HIGH_BAND[0]:g}–{_MOMENT_HIGH_BAND[1]:g} Hz.",
    "wavelet_power_slow_gamma": "mean Morlet-wavelet power, "
        f"{_WAVELET_BANDS['wavelet_power_slow_gamma'][0]:g}–"
        f"{_WAVELET_BANDS['wavelet_power_slow_gamma'][1]:g} Hz.",
    "wavelet_power_gamma": "mean Morlet-wavelet power, "
        f"{_WAVELET_BANDS['wavelet_power_gamma'][0]:g}–"
        f"{_WAVELET_BANDS['wavelet_power_gamma'][1]:g} Hz.",
    "wavelet_power_high_gamma": "mean Morlet-wavelet power, "
        f"{_WAVELET_BANDS['wavelet_power_high_gamma'][0]:g}–"
        f"{_WAVELET_BANDS['wavelet_power_high_gamma'][1]:g} Hz.",
    # Chang et al. 2026 (Table 1 / Table 2). Fast/slow split at the transition
    # point (smoothed |dy/dt| min after the fast peak) over the post-stim
    # window [1, 200] ms; exp fit on |y−slow_baseline|, linear fit on the slow
    # segment. All reconstructed from the paper's definitions (SI unavailable).
    "sum_power_mid": f"Σ periodogram power, {_MID_BAND[0]:g}–{_MID_BAND[1]:g} Hz "
                     "(paper SumPower[64-256]).",
    "freq_moment_vhigh": "power-weighted mean frequency, "
                         f"{_VHIGH_BAND[0]:g}–{_VHIGH_BAND[1]:g} Hz "
                         "(paper 1st-moment[256-1024]).",
    "curvature": "Σ|dy/dt| over the trace ÷ its peak-to-trough span — scale-free "
                 "wiggliness (paper curvature).",
    "skewness": "Fisher skewness of the trace's sample distribution.",
    "tp_latency_ms": "transition-point latency: time of the smoothed-|dy/dt| "
                     "minimum after the fast peak (fast→slow inflection).",
    "tp_amplitude": "trace amplitude at the transition point.",
    "expfit_decay": "b of a·exp(b·t′) fit to |y−slow_baseline| on the fast "
                    "segment (t′ = ms since the fast peak); b<0 = recovery rate.",
    "expfit_initial": "a of the fast-segment exponential fit — amplitude at the "
                      "peak (paper 'explni'; rises in importance in late phases).",
    "expfit_rms": "RMS of the exponential-fit residuals (goodness).",
    "expfit_curvature": "Σ|dy/dt| over the fast segment ÷ its span.",
    "expfit_skew": "skewness of |dy/dt| over the fast segment.",
    "expfit_area": "mean of the fitted exponential over the fast segment "
                   "(∫f·dt ÷ segment duration).",
    "linfit_slope": "slope m of the line m·t+c fit to the slow segment "
                    "[transition, window end].",
    "linfit_intercept": "intercept c of the slow-segment linear fit.",
    "linfit_rms": "RMS of the slow-segment linear-fit residuals ÷ segment span "
                  "(normalised deviation goodness).",
    "linfit_curvature": "Σ|dy/dt| over the slow segment ÷ its span.",
    "linfit_skew": "skewness of |dy/dt| over the slow segment.",
    # Expensive (per-epoch fits; NOT in the default UMAP set).
    "recovery_tau": "exp-decay time constant of the post-peak Hilbert envelope: "
                    "fit A·exp(−t/τ) from the envelope peak (found in 0–50 ms) to "
                    "the window end (semi-log seed, Nelder-Mead refine). The "
                    "active-probing critical-slowing metric (τ rises toward onset).",
    "recovery_slope": "linear slope of y from its |peak| to the window end.",
    "template_correlation": "Pearson r of each epoch vs the median of the "
                            "previous 10 epochs (waveform stability).",
    "pca_recon_error": "reconstruction error of the epoch against a top-3 PCA "
                       "basis fit on the first epochs (normalised to that baseline).",
    "ac_width": "first lag where the (FFT) autocorrelation drops below 0.5, "
                "linearly interpolated — the AC half-width (broadens under slowing).",
    "exp_fit_a": "amplitude A of an exp fit to the pre-peak rising |y|.",
    "autocorr_low": f"lag-1 autocorrelation of the {_AC_BANDS['autocorr_low'][0]:g}"
                    f"–{_AC_BANDS['autocorr_low'][1]:g} Hz band-passed trace.",
    "autocorr_mid": f"lag-1 autocorrelation of the {_AC_BANDS['autocorr_mid'][0]:g}"
                    f"–{_AC_BANDS['autocorr_mid'][1]:g} Hz band-passed trace.",
    "autocorr_high": f"lag-1 autocorrelation of the {_AC_BANDS['autocorr_high'][0]:g}"
                     f"–{_AC_BANDS['autocorr_high'][1]:g} Hz band-passed trace.",
}

_MAX_EPOCHS = 1_000_000     # NASA Rule 2: explicit per-epoch loop bound.


# --------------------------------------------------------------------- #
#  Configure: optional pre-processing applied before the feature math
# --------------------------------------------------------------------- #
#  The toolkit's evoked traces are already filtered + baseline-corrected at
#  extraction, so the DEFAULT config is a pass-through (features run on the
#  traces as-is, reproducing today's cached columns). The Chronic Evoked
#  "Configure" panel lets the user crop the feature window and apply extra
#  bandpass / notch / smoothing / baseline on top -- mirroring
#  ChronicTabController.openConfigDialog (ChronicTabState.m defaults).

@dataclass
class FeatureConfig:
    window_start_ms: float | None = None   # None = no crop (full trace)
    window_end_ms: float | None = None
    bandpass: bool = False
    bp_low_hz: float = 1.0
    bp_high_hz: float = 100.0
    notch: bool = False
    notch_hz: float = 60.0
    smoothing: bool = False
    smooth_ms: float = 5.0
    baseline: bool = False                 # subtract pre-stim (t<=0) mean

    def is_passthrough(self) -> bool:
        return not (self.bandpass or self.notch or self.smoothing
                    or self.baseline
                    or self.window_start_ms is not None
                    or self.window_end_ms is not None)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureConfig":
        d = d or {}
        f = cls()
        for k in cls.__dataclass_fields__:
            if k in d and d[k] is not None:
                setattr(f, k, d[k])
        return f


def _bandpass(a: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    # Second-order-sections form, NOT transfer-function (b, a). At fs=20 kHz a
    # 4th-order band butter in (b, a) form places poles outside the unit circle
    # for low corners (a documented numerical failure of high-order IIR in
    # (b, a)); filtfilt then blows the output up to ~1e80. SOS is stable there.
    from scipy.signal import butter, sosfiltfilt
    nyq = 0.5 * fs
    lo = max(1e-3, min(lo, nyq * 0.99))
    hi = max(lo + 1e-3, min(hi, nyq * 0.99))
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    # sosfiltfilt needs padlen < signal length; its default (3*(2*n_sections+1))
    # is fine for our ~8k-20k-sample traces but clamp for short windows.
    pad = min(a.shape[1] - 1, 3 * (2 * sos.shape[0] + 1))
    return sosfiltfilt(sos, a, axis=1, padlen=pad)


def _notch(a: np.ndarray, fs: float, f0: float) -> np.ndarray:
    from scipy.signal import iirnotch, filtfilt
    if f0 <= 0 or f0 >= 0.5 * fs:
        return a
    b, c = iirnotch(f0, 30.0, fs)
    pad = min(a.shape[1] - 1, 3 * max(len(b), len(c)))
    return filtfilt(b, c, a, axis=1, padlen=pad)


def _smooth(a: np.ndarray, fs: float, win_ms: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    n = max(1, int(round(win_ms * 1e-3 * fs)))
    return uniform_filter1d(a, size=n, axis=1, mode="nearest")


def preprocess(traces, time_ms, fs: float,
               cfg: "FeatureConfig") -> tuple[np.ndarray, np.ndarray]:
    """Apply the Configure pipeline (filter -> notch -> smooth -> baseline ->
    window crop) and return (processed_traces, processed_time_ms). A
    pass-through config returns the inputs unchanged."""
    a = _check(traces)
    t = np.asarray(time_ms, dtype=np.float64).ravel()
    if cfg is None or cfg.is_passthrough():
        return a, t
    if cfg.bandpass:
        a = _bandpass(a, fs, cfg.bp_low_hz, cfg.bp_high_hz)
    if cfg.notch:
        a = _notch(a, fs, cfg.notch_hz)
    if cfg.smoothing:
        a = _smooth(a, fs, cfg.smooth_ms)
    if cfg.baseline:
        pre = t <= 0.0
        if pre.any():
            a = a - np.nanmean(a[:, pre], axis=1, keepdims=True)
    ws = cfg.window_start_ms if cfg.window_start_ms is not None else t[0]
    we = cfg.window_end_ms if cfg.window_end_ms is not None else t[-1]
    mask = (t >= ws) & (t <= we)
    if mask.sum() >= 2:
        a, t = a[:, mask], t[mask]
    return a, t


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
    from scipy.fft import next_fast_len
    # The raw traces are ~20001 samples -- a bad composite length whose FFT
    # is very slow. Pad to the next fast length (negligible effect on the
    # integrated band powers; this is the dominant warm cost).
    nfft = next_fast_len(a.shape[1])
    f, pxx = periodogram(a, fs=fs, nfft=nfft, axis=1)
    return {
        "sum_power_low": _band_sum(f, pxx, _LOW_BAND),
        "sum_power_mid": _band_sum(f, pxx, _MID_BAND),
        "sum_power_high": _band_sum(f, pxx, _HIGH_SUM_BAND),
        "freq_moment_low": _freq_moment(f, pxx, _LOW_BAND),
        "freq_moment_high": _freq_moment(f, pxx, _MOMENT_HIGH_BAND),
        "freq_moment_vhigh": _freq_moment(f, pxx, _VHIGH_BAND),
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


WAVELET_COLUMNS = list(_WAVELET_BANDS.keys())


def compute_cheap(traces, time_ms, fs: float,
                  include_wavelet: bool = True) -> dict:
    """All cheap features as a column->``[epochs]`` dict (one pass).

    *include_wavelet* False skips the Morlet gamma-band power (the dominant warm
    cost), leaving those columns for the caller to splice from an existing
    sidecar -- used by the incremental sidecar upgrade so a schema/version bump
    doesn't recompute the unchanged wavelet columns."""
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
        "curvature": curvature(a, dt),
        "skewness": skewness(a),
    }
    out.update(spectral(a, fs))
    if include_wavelet:
        out.update(wavelet(a, fs))
    else:
        nan = np.full(a.shape[0], np.nan)
        for col in WAVELET_COLUMNS:
            out[col] = nan.copy()
    out.update(compute_chang(a, time_ms, fs))
    return out


def wavelet(traces, fs: float) -> dict:
    """Per-epoch mean Morlet-wavelet power in the gamma bands
    (``_WAVELET_BANDS``). Thin wrapper over ``wavelet.epoch_wavelet_features``;
    kept import-local so a missing PyWavelets only breaks this feature."""
    a = _check(traces)
    from src.utils.wavelet import epoch_wavelet_features
    return epoch_wavelet_features(a, fs, _WAVELET_BANDS)


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
    out = {
        "recovery_tau": recovery_tau(a, time_ms),
        "recovery_slope": recovery_slope(a, time_ms),
        "template_correlation": template_correlation(a),
        "pca_recon_error": pca_recon_error(a),
        "ac_width": ac_width(a),
        "exp_fit_a": exp_fit_a(a, time_ms),
    }
    out.update(autocorr_bands(a, fs))
    return out


# --------------------------------------------------------------------- #
#  Chang et al. 2026 morphology + spectral features (Table 1 / Table 2)
#
#  These reconstruct the paper's evoked-response feature portfolio. The
#  fast/slow split and the two fits are computed for ALL epochs at once with
#  per-epoch boolean masks and closed-form (log-)linear least squares -- no
#  per-epoch Python loop, no iterative optimiser -- so they stay cheap enough
#  to live in the always-computed set. Every reconstructed choice (window,
#  smoothing, transition rule, baseline, normalisation) is a documented knob.
# --------------------------------------------------------------------- #

def curvature(traces, dt: float) -> np.ndarray:
    """Σ|dy/dt| over the trace, normalised by the peak-to-trough span so it is
    a scale-free "wiggliness" (Table 2 #2 / the per-segment curvature base)."""
    a = _check(traces)
    assert dt > 0, "dt must be positive"
    path = np.sum(np.abs(np.diff(a, axis=1)), axis=1) / dt
    span = np.max(a, axis=1) - np.min(a, axis=1)
    return path / (span + _EPS)


def skewness(traces) -> np.ndarray:
    """Fisher skewness of each trace's sample distribution (Table 2 #3)."""
    a = _check(traces)
    y = a - a.mean(axis=1, keepdims=True)
    m2 = np.mean(y * y, axis=1)
    m3 = np.mean(y * y * y, axis=1)
    sd = np.sqrt(m2)
    return np.where(sd > _EPS, m3 / (sd ** 3), 0.0)


def _masked_count(mask) -> np.ndarray:
    return mask.sum(axis=1).astype(np.float64)


def _masked_linfit(x_row, y, mask):
    """Per-epoch OLS ``y ~ slope*x + intercept`` over *mask*.

    *x_row* is the shared 1-D abscissa (samples,), *y* and *mask* are
    ``[epochs x samples]``. Returns ``(slope, intercept, n)`` with NaN slope/
    intercept where an epoch has < 2 masked points or a degenerate spread.
    Pure closed-form normal equations -- vectorised over all epochs.
    """
    m = mask.astype(np.float64)
    n = m.sum(axis=1)
    X = np.broadcast_to(np.asarray(x_row, dtype=np.float64), y.shape)
    sx = np.sum(m * X, axis=1)
    sy = np.sum(m * y, axis=1)
    sxx = np.sum(m * X * X, axis=1)
    sxy = np.sum(m * X * y, axis=1)
    denom = n * sxx - sx * sx
    ok = (n >= 2) & (np.abs(denom) > _EPS)
    slope = np.where(ok, (n * sxy - sx * sy) / np.where(ok, denom, 1.0), np.nan)
    intercept = np.where(ok, (sy - slope * sx) / np.where(n > 0, n, 1.0), np.nan)
    return slope, intercept, n


def _masked_rms(x_row, y, mask, slope, intercept):
    """RMS of residuals of a per-epoch linear model over *mask*."""
    m = mask.astype(np.float64)
    n = m.sum(axis=1)
    X = np.broadcast_to(np.asarray(x_row, dtype=np.float64), y.shape)
    pred = slope[:, None] * X + intercept[:, None]
    resid = (y - pred) * m
    sse = np.sum(resid * resid, axis=1)
    return np.where(n > 0, np.sqrt(sse / np.where(n > 0, n, 1.0)), np.nan)


def _masked_skew(v, mask):
    """Fisher skewness of *v* over *mask*, per epoch (NaN when < 3 points)."""
    m = mask.astype(np.float64)
    n = m.sum(axis=1)
    mean = np.sum(m * v, axis=1) / np.where(n > 0, n, 1.0)
    d = (v - mean[:, None]) * m
    m2 = np.sum(d * d, axis=1) / np.where(n > 0, n, 1.0)
    m3 = np.sum(d * d * d, axis=1) / np.where(n > 0, n, 1.0)
    sd = np.sqrt(m2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = m3 / (sd ** 3)
    return np.where((n >= 3) & (sd > _EPS), sk, np.nan)


def _transition_indices(a, t, fs):
    """Locate, per epoch, the fast-component peak and the fast->slow
    transition point (smoothed |dy/dt| minimum after the peak).

    Returns ``(peak_idx, trans_idx, post_mask, valid)`` where *post_mask* is
    the ``[guard, post]`` analysis window and *valid* flags epochs with enough
    room for both segments. All within the post-stim window only."""
    n_ep, n_s = a.shape
    post = (t >= _CHANG_GUARD_MS) & (t <= _CHANG_POST_MS)
    idx = np.arange(n_s)
    if post.sum() < 2 * _CHANG_MIN_SEG + 1:
        z = np.zeros(n_ep, dtype=int)
        return z, z, post, np.zeros(n_ep, dtype=bool)
    lo = int(np.argmax(post))                 # first in-window sample
    hi = int(n_s - np.argmax(post[::-1]))     # one past last in-window sample
    # Fast-component peak = max |y| inside the window.
    absa = np.abs(a)
    absa_win = np.where(post[None, :], absa, -np.inf)
    peak_idx = np.argmax(absa_win, axis=1)
    # Smoothed |dy/dt| for a stable transition minimum.
    w = max(1, int(round(_CHANG_SMOOTH_MS * 1e-3 * fs)))
    dabs = np.abs(np.diff(_movmean2d(a, w), axis=1))       # [ep, n_s-1]
    dabs = np.concatenate([dabs, dabs[:, -1:]], axis=1)     # pad to n_s
    # Search the transition strictly after the peak, within the window,
    # leaving room for a fittable slow segment at the end.
    trans_hi = hi - _CHANG_MIN_SEG
    after_peak = idx[None, :] >= (peak_idx[:, None] + _CHANG_MIN_SEG)
    in_search = post[None, :] & after_peak & (idx[None, :] < trans_hi)
    cand = np.where(in_search, dabs, np.inf)
    trans_idx = np.argmin(cand, axis=1)
    valid = (in_search.any(axis=1)
             & (peak_idx >= lo + _CHANG_MIN_SEG - _CHANG_MIN_SEG)
             & (trans_idx - peak_idx >= _CHANG_MIN_SEG)
             & (hi - trans_idx >= _CHANG_MIN_SEG))
    return peak_idx, trans_idx, post, valid


def compute_chang(traces, time_ms, fs: float) -> dict:
    """Table 1 #1-13: transition point + exponential (fast) + linear (slow)
    component features, vectorised over all epochs. Invalid epochs (window
    too short / no clean transition) get NaN across the group."""
    a = _check(traces)
    t = np.asarray(time_ms, dtype=np.float64)
    assert t.shape[0] == a.shape[1], "time_ms length must match samples"
    dt = 1000.0 / fs
    n_ep, n_s = a.shape
    idx = np.arange(n_s)
    peak_idx, trans_idx, post, valid = _transition_indices(a, t, fs)

    # Segment masks (within the post-stim window).
    hi = int(n_s - np.argmax(post[::-1])) if post.any() else n_s
    fast = (idx[None, :] >= peak_idx[:, None]) & (idx[None, :] <= trans_idx[:, None])
    slow = (idx[None, :] >= trans_idx[:, None]) & (idx[None, :] < hi) & post[None, :]
    fast = fast & valid[:, None]
    slow = slow & valid[:, None]

    # Slow-component baseline = level at/after the transition (median of slow
    # segment) -- the fast component decays toward this.
    slow_f = slow.astype(np.float64)
    slow_n = np.sum(slow_f, axis=1)
    base = np.sum(slow_f * a, axis=1) / np.where(slow_n > 0, slow_n, 1.0)

    # --- Fast component: exponential a*exp(b*t') on |y - baseline| --------- #
    trel = t[None, :] - t[peak_idx][:, None]        # ms since the peak
    dev = np.abs(a - base[:, None])
    pos = dev > _EPS
    exp_mask = fast & pos
    with np.errstate(invalid="ignore", divide="ignore"):
        logdev = np.log(np.where(exp_mask, dev, 1.0))
    # trel varies per epoch (measured from each epoch's peak), so fit with the
    # 2-D-abscissa masked least squares.
    b, ln_a = _linfit2d(trel, logdev, exp_mask)
    expfit_decay = b
    expfit_initial = np.exp(np.clip(ln_a, -50.0, 50.0))
    # Goodness: RMS of residuals in linear (dev) space. Clip the exponent so a
    # pathological (positive-b) fit can't overflow -> inf.
    pred_dev = np.exp(np.clip(ln_a[:, None] + b[:, None] * trel, -50.0, 50.0))
    r = (dev - pred_dev) * exp_mask
    fe_n = np.sum(exp_mask, axis=1)
    expfit_rms = np.where(fe_n > 0, np.sqrt(np.sum(r * r, axis=1)
                                            / np.where(fe_n > 0, fe_n, 1.0)), np.nan)
    # Curvature over the fast segment (Σ|dy| / span), + skewness of |dy|.
    dabs_full = np.abs(np.diff(a, axis=1))
    dabs_full = np.concatenate([dabs_full, dabs_full[:, -1:]], axis=1) / dt
    fspan = _masked_span(a, fast)
    expfit_curvature = (np.sum(dabs_full * fast, axis=1)
                        / (fspan + _EPS))
    expfit_skew = _masked_skew(dabs_full, fast)
    # Area under the fit, normalised by segment duration.
    fdur = np.sum(fast, axis=1) * dt
    expfit_area = (np.sum(pred_dev * exp_mask, axis=1) * dt
                   / (fdur + _EPS))

    # --- Slow component: line m*t + c on y over [transition, end] ---------- #
    linfit_slope, linfit_intercept, ln_n = _masked_linfit(t, a, slow)
    linfit_rms = _masked_rms(t, a, slow, linfit_slope, linfit_intercept)
    sspan = _masked_span(a, slow)
    linfit_rms = linfit_rms / (sspan + _EPS)        # normalised deviation
    linfit_curvature = np.sum(dabs_full * slow, axis=1) / (sspan + _EPS)
    linfit_skew = _masked_skew(dabs_full, slow)

    nan = np.full(n_ep, np.nan)
    out = {
        "tp_latency_ms": np.where(valid, t[trans_idx], np.nan),
        "tp_amplitude": np.where(valid, a[np.arange(n_ep), trans_idx], np.nan),
        "expfit_decay": np.where(valid, expfit_decay, np.nan),
        "expfit_initial": np.where(valid, expfit_initial, np.nan),
        "expfit_rms": np.where(valid, expfit_rms, np.nan),
        "expfit_curvature": np.where(valid, expfit_curvature, np.nan),
        "expfit_skew": np.where(valid, expfit_skew, np.nan),
        "expfit_area": np.where(valid, expfit_area, np.nan),
        "linfit_slope": np.where(valid, linfit_slope, np.nan),
        "linfit_intercept": np.where(valid, linfit_intercept, np.nan),
        "linfit_rms": np.where(valid, linfit_rms, np.nan),
        "linfit_curvature": np.where(valid, linfit_curvature, np.nan),
        "linfit_skew": np.where(valid, linfit_skew, np.nan),
    }
    return out


def _linfit2d(x2d, y, mask):
    """Per-epoch OLS with a per-epoch (2-D) abscissa *x2d*. Returns
    ``(slope, intercept)``; NaN where < 2 masked points."""
    m = mask.astype(np.float64)
    n = m.sum(axis=1)
    sx = np.sum(m * x2d, axis=1)
    sy = np.sum(m * y, axis=1)
    sxx = np.sum(m * x2d * x2d, axis=1)
    sxy = np.sum(m * x2d * y, axis=1)
    denom = n * sxx - sx * sx
    ok = (n >= 2) & (np.abs(denom) > _EPS)
    slope = np.where(ok, (n * sxy - sx * sy) / np.where(ok, denom, 1.0), np.nan)
    intercept = np.where(ok, (sy - slope * sx) / np.where(n > 0, n, 1.0), np.nan)
    return slope, intercept


def _masked_span(a, mask):
    """Per-epoch (max - min) of *a* over *mask* (0 where empty)."""
    big = np.where(mask, a, -np.inf)
    small = np.where(mask, a, np.inf)
    mx = np.max(big, axis=1)
    mn = np.min(small, axis=1)
    span = mx - mn
    return np.where(np.isfinite(span), span, 0.0)


def autocorr_bands(traces, fs: float) -> dict:
    """Lag-1 autocorrelation of the band-limited trace, per band
    (Table 2 #11-13). Band-passes each epoch then applies the Maturana lag-1
    form. Kept in the expensive set -- 3 filtfilt passes are the cost."""
    a = _check(traces)
    out = {}
    for key, band in _AC_BANDS.items():
        lo, hi = band
        try:
            filt = _bandpass(a, fs, lo, min(hi, 0.49 * fs))
        except Exception:            # noqa: BLE001 -- degrade to NaN, never crash warm
            out[key] = np.full(a.shape[0], np.nan)
            continue
        y = filt - filt.mean(axis=1, keepdims=True)
        num = np.sum(y[:, :-1] * y[:, 1:], axis=1)
        den = np.sum(y * y, axis=1) + _EPS
        out[key] = num / den
    return out


def compute_all(traces, time_ms, fs: float, expensive: bool = False,
                cfg: "FeatureConfig | None" = None,
                include_wavelet: bool = True) -> dict:
    """Cheap features always; expensive ones only when *expensive*.

    When *cfg* is given (and not pass-through) the traces are pre-processed
    (Configure: filter/notch/smooth/baseline/window-crop) before the feature
    math; the default cfg=None reproduces today's behavior exactly. Columns
    not computed are present as all-NaN arrays so the schema stays uniform.
    *include_wavelet* False leaves the wavelet columns NaN (incremental upgrade).
    """
    a, t = preprocess(traces, time_ms, fs, cfg or FeatureConfig())
    out = compute_cheap(a, t, fs, include_wavelet=include_wavelet)
    if expensive:
        out.update(compute_expensive(a, t, fs))
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
