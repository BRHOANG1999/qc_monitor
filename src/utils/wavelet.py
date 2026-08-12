"""Morlet continuous-wavelet (CWT) time-frequency analysis.

Welch PSD (``filters.compute_psd``) averages power over ~2 s windows, which
smears the *transient* time-frequency structure this lab cares about --
evoked responses and seizures are short, non-stationary events. A complex
Morlet CWT resolves time AND frequency together, so it backs both the
scalogram views and the transient-aware "wavelet band power" metrics.

Conventions mirror ``src/utils/filters.py`` / ``hilbert_envelope.py``:
float64, NaN/inf sanitised to 0 before any transform, linear power
(signal-units squared), and the trapezoid alias for NumPy 2.x.

PERFORMANCE: raw LFP is 20 kHz (~72M samples/hour). NEVER hand a full
channel to ``scalogram`` -- pass an already-windowed segment (a zoomed
view, a stim epoch, or a bounded slice). Every entry point decimates to a
working fs (~2 kHz) before the CWT and asserts an input-length ceiling.
``pywt`` is imported lazily inside the functions (like ``h5py`` elsewhere)
so importing this module never forces the dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import decimate

# NumPy 2.x renamed trapz -> trapezoid (mirrors filters.py / spectral.py).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Complex Morlet: bandwidth 1.5, centre frequency 1.0 (good time/freq
# trade-off for LFP). Do not retune without lab sign-off.
_CMOR = "cmor1.5-1.0"
_DEFAULT_WORKING_FS = 2000.0
_MAX_CWT_SAMPLES = 5_000_000      # NASA Rule 2 guard against a full-channel CWT
_MAX_DECIM_STEPS = 8              # bounded decimation loop

# Sibling gamma bands (slow gamma lives in filters.SLOW_GAMMA_BAND=(30,50)).
GAMMA_BAND = (50.0, 100.0)
HIGH_GAMMA_BAND = (100.0, 200.0)


def _decimate_to(signal: np.ndarray, fs: float,
                 target_fs: float) -> tuple[np.ndarray, float]:
    """Anti-aliased decimate to <= ``target_fs`` (FIR, zero-phase), in
    bounded integer steps (<=13/step per scipy guidance). No-op when the
    signal is already at/near the target. NaN/inf are zeroed first because
    the FIR filter would otherwise smear them across the trace."""
    x = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    cur_fs = float(fs)
    if cur_fs <= target_fs * 1.1 or x.size < 64:
        return x, cur_fs
    steps = 0
    while cur_fs > target_fs * 1.1 and steps < _MAX_DECIM_STEPS:
        steps += 1
        q = int(min(13, max(2, np.floor(cur_fs / target_fs))))
        if x.size // q < 16:
            break
        x = decimate(x, q, ftype="fir", zero_phase=True)
        cur_fs /= q
    return np.asarray(x, dtype=np.float64), cur_fs


def _freqs_to_scales(freqs: np.ndarray, work_fs: float) -> np.ndarray:
    """CWT scales for the requested Hz, via ``scale = fc * fs / f`` where
    fc is the Morlet centre frequency (1.0 for cmor1.5-1.0)."""
    import pywt
    fc = float(pywt.central_frequency(_CMOR))
    return fc * work_fs / np.asarray(freqs, dtype=np.float64)


def scalogram(signal: np.ndarray, fs: float, fmin: float = 2.0,
              fmax: float | None = None, n_freqs: int = 64,
              target_fs: float = _DEFAULT_WORKING_FS):
    """Morlet scalogram of a 1-D segment.

    Returns ``(freqs, times, power)`` where ``freqs`` is ascending Hz
    (log-spaced), ``times`` is seconds, and ``power = |CWT|**2`` shaped
    ``[n_freqs x n_time]`` (linear). Decimates to <= ``target_fs`` first.
    The caller MUST pass a bounded segment, not a full channel.
    """
    import pywt
    x = np.asarray(signal, dtype=np.float64).ravel()
    assert x.ndim == 1, "scalogram needs a 1-D segment"
    x, work_fs = _decimate_to(x, fs, target_fs)
    # Guard the DECIMATED size. Asserting on the RAW size (before decimation)
    # made continuous callers crash on any recording > ~250 s @ 20 kHz even
    # though the decimated CWT was perfectly feasible; a bounded epoch stays
    # tiny either way, so small-window callers are unaffected.
    assert x.size <= _MAX_CWT_SAMPLES, \
        "segment too long even after decimation; window/slice first"
    if x.size < 8:
        return np.array([]), np.array([]), np.zeros((0, 0))
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)
    hi_cap = 0.45 * work_fs
    fmax = hi_cap if fmax is None else min(float(fmax), hi_cap)
    fmin = max(float(fmin), work_fs / x.size)   # >=1 cycle in the window
    if fmax <= fmin:
        fmax = min(hi_cap, fmin * 1.5)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), int(n_freqs))
    scales = _freqs_to_scales(freqs, work_fs)
    coef, _ = pywt.cwt(x, scales, _CMOR, sampling_period=1.0 / work_fs)
    power = np.nan_to_num(np.abs(coef) ** 2)    # rows align to ascending freqs
    times = np.arange(x.size, dtype=np.float64) / work_fs
    return freqs, times, power


def wavelet_band_power(signal: np.ndarray, fs: float, lo: float, hi: float,
                       n_freqs: int = 24,
                       target_fs: float = _DEFAULT_WORKING_FS):
    """Time-resolved wavelet band power in ``[lo, hi]`` Hz: the scalogram
    integrated over the band per time sample. Returns ``(power_t,
    work_fs)`` (``power_t`` at the decimated rate)."""
    assert hi > lo > 0, "need 0 < lo < hi"
    freqs, times, power = scalogram(signal, fs, fmin=lo, fmax=hi,
                                    n_freqs=n_freqs, target_fs=target_fs)
    if freqs.size == 0 or times.size < 2:
        return np.zeros(0), float(fs)
    work_fs = 1.0 / float(times[1] - times[0])
    if freqs.size < 2:
        return np.nan_to_num(power[0]), work_fs
    band = _trapezoid(power, freqs, axis=0)     # integrate over Hz, per time
    return np.nan_to_num(band), work_fs


def epoch_wavelet_band_power(signal: np.ndarray, fs: float, stim_times,
                             lo: float, hi: float,
                             win: tuple[float, float],
                             max_epochs: int = 5000):
    """Per-stim wavelet band power over ``[onset+win0, onset+win1]`` s
    (win0 may be negative). Time-mean of the band power per epoch. Mirrors
    ``filters.epoch_band_power``; returns ``(times, powers)`` for the
    in-bounds epochs."""
    assert win[1] > win[0], "win end must exceed start"
    assert hi > lo > 0, "need 0 < lo < hi"
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[0]
    out_t: list[float] = []
    out_v: list[float] = []
    for s in np.asarray(stim_times, dtype=np.float64)[:max_epochs]:
        i0 = int(round((float(s) + win[0]) * fs))
        i1 = int(round((float(s) + win[1]) * fs))
        if i0 < 0 or i1 > n or (i1 - i0) < 8:
            continue
        band, _ = wavelet_band_power(signal[i0:i1], fs, lo, hi)
        if band.size == 0:
            continue
        out_t.append(float(s))
        out_v.append(float(np.mean(band)))
    return np.asarray(out_t), np.asarray(out_v)


def epoch_wavelet_features(traces, fs: float, bands: dict) -> dict:
    """Per-epoch mean wavelet power per band for an ``[epochs x samples]``
    trace block (same contract as ``evoked_features._check``). ``bands`` is
    ``{column_name: (lo, hi)}``. One CWT per epoch over the full requested
    span; each band value is the time-and-band mean of ``|CWT|**2``.
    Non-finite epochs -> NaN."""
    a = np.asarray(traces, dtype=np.float64)
    assert a.ndim == 2, "traces must be 2-D [epochs x samples]"
    assert bands, "bands must be non-empty"
    n_ep = a.shape[0]
    out = {col: np.full(n_ep, np.nan) for col in bands}
    lo_all = min(lo for lo, _ in bands.values())
    hi_all = max(hi for _, hi in bands.values())
    n_ep = min(n_ep, _MAX_CWT_SAMPLES)          # bounded loop
    for ei in range(n_ep):
        freqs, _t, power = scalogram(a[ei], fs, fmin=lo_all, fmax=hi_all,
                                     n_freqs=48)
        if freqs.size == 0:
            continue
        for col, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            out[col][ei] = (float(np.mean(power[mask]))
                            if mask.any() else 0.0)
    return out
