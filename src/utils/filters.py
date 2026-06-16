"""Shared LFP filter / smoothing / PSD helpers for the dashboard.

The dashboard's LFP-bearing tabs (LFP Browser, Video Review, Evoked
Waveforms, Session Compare) all need the same handful of operations:

* zero-phase Butterworth highpass / lowpass
* notch (50 / 60 Hz, Q=30)
* Gaussian smoothing
* Welch PSD

Centralising them here keeps the Tier 1 batch spectral analyzer at
``src/analyzers/spectral.py`` and the live dashboard agreeing on the
same parameters (nperseg, window, overlap).

A tiny LRU cache (``get_filtered``) sits in front of ``apply_filter``
so the LFP Browser's zoom callbacks don't re-filter on every relayout
event.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import RLock

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, welch

logger = logging.getLogger("qc_monitor.utils.filters")

SUPPORTED_NOTCH: tuple[int | None, ...] = (None, 50, 60)
DEFAULT_BUTTER_ORDER = 4
DEFAULT_NOTCH_Q = 30.0

# Slow gamma band (Hz). A 30-50 sub-slice of the lab's wide 30-100 "gamma".
SLOW_GAMMA_BAND: tuple[float, float] = (30.0, 50.0)

# np.trapz was deprecated in NumPy 2.0 / removed in 2.2 -- prefer the new
# name, fall back for older NumPy (mirrors src/analyzers/spectral.py).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def band_power(freqs: np.ndarray, psd: np.ndarray,
                lo: float, hi: float) -> float:
    """Integrated power in ``[lo, hi]`` Hz from a PSD (V**2).

    Trapezoidal integral of *psd* over the ``lo <= f <= hi`` mask, matching
    the linear-power convention the batch spectral analyzer uses. Pair with
    ``compute_psd``. Returns 0.0 when the band is empty / inputs are bad.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd, dtype=np.float64)
    if freqs.size == 0 or psd.size != freqs.size or hi <= lo:
        return 0.0
    mask = (freqs >= lo) & (freqs <= hi) & np.isfinite(psd)
    if mask.sum() < 2:
        return float(psd[mask].sum()) if mask.any() else 0.0
    return float(_trapezoid(psd[mask], freqs[mask]))

# ----- LRU cache --------------------------------------------------- #
# Key shape: (file_path, hp, lp, notch, smooth_ms) -- per chunk + per
# settings combination. Capping at 4 keeps RAM bounded under typical
# usage (a few open files, a couple of filter settings each).
_FILTER_CACHE_MAX = 4
_filter_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_filter_lock = RLock()


# ====================================================================== #
#  Public API
# ====================================================================== #

def filter_settings_key(hp, lp, notch, smooth_ms) -> tuple:
    """Canonicalize a filter-settings tuple for use as a cache key."""
    return (
        None if not hp else float(hp),
        None if not lp else float(lp),
        None if notch in (None, 0, "Off", "off") else int(notch),
        0.0 if not smooth_ms else float(smooth_ms),
    )


def apply_filter(signal: np.ndarray, fs: float, *,
                  highpass: float | None = None,
                  lowpass: float | None = None,
                  notch: int | None = None,
                  smoothing_ms: float = 0.0) -> np.ndarray:
    """Cascade: highpass -> lowpass -> notch -> Gaussian smoothing.

    Accepts a 1-D or 2-D signal (samples x channels). Returns a fresh
    float32 array; the input is never modified. Stages that aren't
    requested (None / 0) are skipped.

    Edge effects: ``sosfiltfilt`` is zero-phase (forward+backward) so
    the first/last ~ 3*order/cutoff seconds carry transient ringing.
    Document at the callsite if you need to mask those edges.

    NaN handling: short gaps (<= 50 ms) are linearly interpolated
    before filtering and the NaN mask reapplied after; longer gaps
    stay NaN so stim-blank windows survive.
    """
    assert isinstance(signal, np.ndarray), "signal must be a numpy array"
    assert fs > 0, "fs must be positive"
    assert notch in SUPPORTED_NOTCH, f"notch must be one of {SUPPORTED_NOTCH}"

    if signal.size == 0:
        return signal.astype(np.float32, copy=False)

    nyq = fs / 2.0
    is_1d = signal.ndim == 1
    work = np.asarray(signal, dtype=np.float32)
    if is_1d:
        work = work[:, None]

    # NaN handling. sosfiltfilt propagates NaN through the entire
    # array (forward + backward passes both touch each sample), so we
    # have to feed the filter a NaN-free array. Strategy:
    #   1. Capture the mask of LONG gaps (>50 ms) before touching anything.
    #   2. Linearly interpolate EVERY NaN gap so the filter sees a
    #      contiguous signal.
    #   3. After filtering, restore the long-gap NaNs so stim-blank
    #      windows still render as breaks.
    long_gap_mask = _long_gap_mask(work, fs, max_gap_ms=50.0)
    had_nans = bool(np.isnan(work).any())
    if had_nans:
        work = _interpolate_short_gaps(work, fs,
                                         max_gap_ms=float("inf"))

    # Highpass
    if highpass and highpass > 0:
        wn = highpass / nyq
        if not (0 < wn < 1):
            logger.warning("Highpass %s Hz outside Nyquist for fs=%s; "
                            "skipping", highpass, fs)
        else:
            sos = butter(DEFAULT_BUTTER_ORDER, wn, btype="highpass",
                          output="sos")
            work = sosfiltfilt(sos, work, axis=0).astype(np.float32,
                                                          copy=False)

    # Lowpass
    if lowpass and lowpass > 0:
        wn = lowpass / nyq
        if not (0 < wn < 1):
            logger.warning("Lowpass %s Hz outside Nyquist for fs=%s; "
                            "skipping", lowpass, fs)
        else:
            sos = butter(DEFAULT_BUTTER_ORDER, wn, btype="lowpass",
                          output="sos")
            work = sosfiltfilt(sos, work, axis=0).astype(np.float32,
                                                          copy=False)

    # Notch
    if notch:
        if notch >= nyq - 5:
            logger.warning("Notch %s Hz too close to Nyquist (fs=%s); "
                            "skipping", notch, fs)
        else:
            b, a = iirnotch(w0=notch / nyq, Q=DEFAULT_NOTCH_Q)
            sos = tf2sos(b, a)
            work = sosfiltfilt(sos, work, axis=0).astype(np.float32,
                                                          copy=False)

    # Gaussian smoothing (in samples)
    if smoothing_ms and smoothing_ms > 0:
        # sigma in samples: smoothing_ms / 1000 * fs / 2 gives a window
        # whose FWHM ~ smoothing_ms. Cap sigma so users don't accidentally
        # nuke the entire signal.
        sigma_samples = max(0.5, smoothing_ms * fs / 2000.0)
        sigma_samples = min(sigma_samples, work.shape[0] / 4)
        work = gaussian_filter1d(work, sigma=sigma_samples, axis=0,
                                  mode="nearest").astype(np.float32,
                                                          copy=False)

    # Restore long-gap NaNs so stim-blank regions render as gaps.
    if long_gap_mask is not None and long_gap_mask.any():
        work[long_gap_mask] = np.nan

    return work[:, 0] if is_1d else work


def compute_psd(signal: np.ndarray, fs: float,
                 nperseg: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with the same parameters Tier 1's spectral analyzer uses.

    Auto-sizes nperseg to ``min(8192, len // 8)`` rounded down to the
    nearest power of two for FFT-friendly sizes; 50% overlap; Hanning
    window (scipy default).
    """
    assert isinstance(signal, np.ndarray), "signal must be a numpy array"
    assert fs > 0, "fs must be positive"

    sig = signal[np.isfinite(signal)] if signal.ndim == 1 else signal
    if sig.size < 64:
        return np.array([]), np.array([])

    n = sig.shape[0] if sig.ndim == 1 else sig.shape[0]
    if nperseg is None:
        nperseg = min(8192, max(256, n // 8))
        nperseg = 1 << int(np.floor(np.log2(nperseg)))
    nperseg = min(nperseg, n)

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg,
                        noverlap=nperseg // 2, axis=0)
    return freqs, psd


def get_filtered(cache_key: str, signal: np.ndarray, fs: float,
                  *, highpass: float | None = None,
                  lowpass: float | None = None,
                  notch: int | None = None,
                  smoothing_ms: float = 0.0) -> np.ndarray:
    """LRU-cached ``apply_filter`` keyed by *cache_key* + settings.

    *cache_key* is typically the file path (so different files don't
    collide). The cached array is returned by reference -- callers
    must not mutate it. Capped at 4 entries (LRU on insert).
    """
    settings = filter_settings_key(highpass, lowpass, notch, smoothing_ms)
    key = (cache_key,) + settings

    with _filter_lock:
        hit = _filter_cache.pop(key, None)
        if hit is not None:
            _filter_cache[key] = hit       # MRU bump
            logger.debug("Filter cache HIT %s", key)
            return hit

    # Cache miss -- compute outside the lock so concurrent callers
    # for different keys don't serialize.
    logger.debug("Filter cache MISS %s", key)
    out = apply_filter(signal, fs,
                        highpass=settings[0], lowpass=settings[1],
                        notch=settings[2], smoothing_ms=settings[3])
    with _filter_lock:
        _filter_cache[key] = out
        while len(_filter_cache) > _FILTER_CACHE_MAX:
            _filter_cache.popitem(last=False)
    return out


def evict_filtered(cache_key: str | None = None) -> int:
    """Drop cached entries. With *cache_key*, only that file's entries
    go; without, the whole cache. Returns the number evicted."""
    with _filter_lock:
        if cache_key is None:
            n = len(_filter_cache)
            _filter_cache.clear()
            return n
        keys = [k for k in _filter_cache if k[0] == cache_key]
        for k in keys:
            _filter_cache.pop(k, None)
        return len(keys)


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _long_gap_mask(arr: np.ndarray, fs: float,
                    max_gap_ms: float) -> np.ndarray | None:
    """Return a boolean mask of NaN gaps STRICTLY LONGER than max_gap_ms.

    Short gaps don't appear in the mask -- they'll be silently
    interpolated and stay as real filtered values. Long gaps (e.g.
    stim-blank windows) appear so we can restore them as NaN after
    filtering.
    """
    if not np.isnan(arr).any():
        return None
    max_gap = int(max_gap_ms * fs / 1000.0)
    mask = np.zeros_like(arr, dtype=bool)
    is_1d = arr.ndim == 1
    cols = arr[:, None] if is_1d else arr
    mcols = mask[:, None] if is_1d else mask
    for ch in range(cols.shape[1]):
        col = cols[:, ch]
        nan = np.isnan(col)
        if not nan.any():
            continue
        edges = np.diff(np.concatenate(([0], nan.view(np.int8), [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) > max_gap:
                mcols[s:e, ch] = True
    return mcols[:, 0] if is_1d else mcols


def _interpolate_short_gaps(arr: np.ndarray, fs: float,
                             max_gap_ms: float) -> np.ndarray:
    """Linearly interpolate NaN runs <= max_gap_ms; longer runs untouched.

    Pass ``max_gap_ms = float('inf')`` to interpolate EVERY gap (used
    by ``apply_filter`` because ``sosfiltfilt`` propagates NaN through
    the entire array; we restore long gaps to NaN after filtering).

    Edges (NaN at start or end of the array) get forward/backward-
    filled from the nearest finite value rather than left alone --
    leaving them NaN would crash sosfiltfilt the same way.
    """
    is_inf = max_gap_ms == float("inf")
    max_gap = int(max_gap_ms * fs / 1000.0) if not is_inf else 0
    out = arr.copy()
    is_1d = out.ndim == 1
    cols = out[:, None] if is_1d else out
    for ch in range(cols.shape[1]):
        col = cols[:, ch]
        nan = np.isnan(col)
        if not nan.any():
            continue
        edges = np.diff(np.concatenate(([0], nan.view(np.int8), [0])))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        for s, e in zip(starts, ends):
            gap_len = e - s
            if not is_inf and gap_len > max_gap:
                continue
            if s == 0 and e == col.size:
                # All-NaN channel: nothing useful we can do.
                continue
            if s == 0:
                col[s:e] = col[e]
            elif e == col.size:
                col[s:e] = col[s - 1]
            else:
                col[s:e] = np.linspace(col[s - 1], col[e],
                                        gap_len + 2)[1:-1]
    return out if not is_1d else cols[:, 0]
