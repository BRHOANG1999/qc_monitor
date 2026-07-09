"""Morlet CWT over a decimated feature trajectory.

The CWT scale axis IS the log-spaced time-scale tiling -- we don't hand-define
scale edges on top of it. ``pywt`` maps scale <-> pseudo-frequency. We persist
per-scale scalar summaries only (never full scalograms). ``scipy.signal.cwt`` is
deprecated, so this uses ``pywt`` (complex Morlet).
"""

from __future__ import annotations

import numpy as np
import pywt


def make_scales(n: int, dt: float, wavelet: str = "cmor1.5-1.0",
                 scales_per_octave: int = 4, min_leadtime_sec: float = 1.0,
                 max_scales: int = 64) -> np.ndarray:
    """Log/dyadic CWT scales for a length-*n* trajectory sampled every *dt* s.

    Smallest scale ~ the pseudo-period ``min_leadtime_sec``; largest ~ a quarter
    of the record (so at least ~4 cycles fit -- longer scales aren't
    estimable). Returns ascending scales (float array, >= 1 entry)."""
    assert n >= 2 and dt > 0 and scales_per_octave >= 1
    # scale for a target pseudo-frequency f:  scale = f_c / (f * dt), with
    # f_c = central frequency of the wavelet (pywt.central_frequency).
    fc = pywt.central_frequency(wavelet)
    f_hi = 1.0 / max(min_leadtime_sec, 2 * dt)         # fastest resolvable
    f_lo = 4.0 / (n * dt)                               # ~4 cycles in the record
    if f_lo >= f_hi:
        f_lo = f_hi / 2.0
    n_oct = max(1.0, np.log2(f_hi / f_lo))
    k = min(max_scales, int(np.ceil(n_oct * scales_per_octave)) + 1)
    freqs = f_hi * 2.0 ** (-np.arange(k) / scales_per_octave)   # hi -> lo
    scales = fc / (freqs * dt)
    return np.sort(scales)


def morlet_cwt(trajectory: np.ndarray, dt: float,
                wavelet: str = "cmor1.5-1.0", scales_per_octave: int = 4,
                min_leadtime_sec: float = 1.0):
    """Complex Morlet CWT of a 1-D feature trajectory. Returns
    ``(coeffs[n_scales, n], scales, pseudo_freqs_hz)``."""
    x = np.asarray(trajectory, dtype=float)
    assert x.ndim == 1 and x.size >= 2, "trajectory must be 1-D, len >= 2"
    scales = make_scales(x.size, dt, wavelet, scales_per_octave,
                          min_leadtime_sec)
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=dt)
    return coeffs, scales, freqs


def scale_summaries(coeffs: np.ndarray, scales: np.ndarray,
                     freqs: np.ndarray) -> list[dict]:
    """Per-scale scalar summaries of the coefficient magnitude |W| -- the
    Stage-1 persisted deliverable (mean / std / max per scale)."""
    mag = np.abs(np.asarray(coeffs))
    out: list[dict] = []
    for i in range(mag.shape[0]):
        row = mag[i]
        out.append({
            "scale_index": int(i),
            "scale": float(scales[i]),
            "pseudo_freq_hz": float(freqs[i]),
            "coeff_mean": float(np.mean(row)),
            "coeff_std": float(np.std(row)),
            "coeff_max": float(np.max(row)),
        })
    return out
