"""Infraslow spectral analysis of the across-trial feature series (Part B, B3).

Does an infraslow mode (0.001-0.01 Hz = 100-1000 s periods) exist in the feature
trial series at all, and does its band power change approaching onset? This is the
one analysis that directly tests for an infraslow mode -- and the one nobody has
run on this data. A tau~200 s relaxation sits at ~1/(2*pi*tau) ~ 0.0008 Hz, in
band; a slow oscillatory drive shows a line here.

Welch PSD per gap-free SEGMENT (uniform within a segment at ~dt), band power
integrated over [f_lo, f_hi], length-weighted across segments. Pure + scipy-only.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_LO = 0.001
_DEFAULT_HI = 0.01


def _trapz(y, x) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def welch_band_power(values, dt: float, f_lo: float = _DEFAULT_LO,
                     f_hi: float = _DEFAULT_HI, *, nperseg=None):
    """(band_power, total_power) of a uniformly-sampled 1-D series via Welch PSD,
    band_power = integral of the PSD over [f_lo, f_hi]. NaN when too short."""
    from scipy.signal import welch
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    assert f_hi > f_lo >= 0, "need f_hi > f_lo >= 0"
    if n < 16 or not (dt > 0):
        return float("nan"), float("nan")
    nps = int(nperseg or min(n, max(256, n // 4)))
    nps = max(16, min(nps, n))
    f, pxx = welch(y, fs=1.0 / dt, nperseg=nps, detrend="linear")
    band = (f >= f_lo) & (f <= f_hi)
    bp = _trapz(pxx[band], f[band]) if int(band.sum()) >= 2 else 0.0
    tot = _trapz(pxx, f) if f.size >= 2 else float("nan")
    return float(bp), float(tot)


def series_band_power(series, f_lo: float = _DEFAULT_LO, f_hi: float = _DEFAULT_HI,
                      *, nperseg=None, min_seg: int = 64) -> dict:
    """Length-weighted infraslow band power over the TrialSeries' gap-free
    segments (uses the measured dt), plus the in-band FRACTION of total power --
    the scale-free readout of 'is there an infraslow mode'. Returns a dict with
    band_power, total_power, band_frac, dt, n_seg, and the band edges."""
    dt = float(series.dt_med)
    empty = {"band_power": float("nan"), "total_power": float("nan"),
             "band_frac": float("nan"), "dt": dt, "n_seg": 0,
             "f_lo": f_lo, "f_hi": f_hi}
    if not np.isfinite(dt) or dt <= 0:
        return empty
    bps, tots, wts = [], [], []
    for sl in series.segments():
        if (sl.stop - sl.start) < min_seg:
            continue
        bp, tot = welch_band_power(series.values[sl], dt, f_lo, f_hi,
                                   nperseg=nperseg)
        if np.isfinite(bp) and np.isfinite(tot) and tot > 0:
            bps.append(bp)
            tots.append(tot)
            wts.append(sl.stop - sl.start)
    if not bps:
        return empty
    w = np.asarray(wts, dtype=float)
    bp = float(np.average(bps, weights=w))
    tot = float(np.average(tots, weights=w))
    return {"band_power": bp, "total_power": tot,
            "band_frac": (bp / tot) if tot > 0 else float("nan"),
            "dt": dt, "n_seg": len(bps), "f_lo": f_lo, "f_hi": f_hi}
