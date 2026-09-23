"""Pure power-spectrum comparison for the riding event.

Given two groups of post-stim windows — event-carrying and clean — this computes
their averaged Welch PSDs and locates the event's excess-power band and its
fundamental + harmonics, while flagging mains line-noise (50/60/120/180 Hz) so a
line-noise peak is never mistaken for the event's own rhythm. Averaging the PSD
across the many epochs in a group (Bartlett-style) denoises a single short window.

No I/O, no plotting; ``render`` draws these arrays.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-20


def line_noise_freqs(fs: float, base=(50.0, 60.0), n_harm: int = 4) -> list:
    """Mains line-noise fundamentals + harmonics up to Nyquist (for annotation /
    exclusion). Both 50 and 60 Hz families by default (unknown mains)."""
    nyq = 0.5 * float(fs)
    out = []
    for b in base:
        for h in range(1, int(n_harm) + 1):
            f = b * h
            if 0 < f < nyq:
                out.append(round(f, 3))
    return sorted(set(out))


def welch_psd(traces, fs: float, *, time_ms=None, win_ms=None,
              nperseg: int | None = None) -> tuple:
    """Epoch-averaged Welch PSD of a group of windows.

    Crops each epoch to ``win_ms`` (ms, needs *time_ms*) when given, then
    averages the per-epoch Welch PSDs. Returns ``(freqs, psd_mean, n_epochs)``;
    empty arrays + 0 when there is nothing usable."""
    from scipy.signal import welch
    a = np.asarray(traces, dtype=np.float64)
    assert a.ndim == 2, "traces must be [epochs x samples]"
    assert fs and fs > 0, "fs must be positive"
    if time_ms is not None and win_ms is not None:
        t = np.asarray(time_ms, dtype=np.float64)
        m = (t >= float(win_ms[0])) & (t <= float(win_ms[1]))
        if m.sum() >= 8:
            a = a[:, m]
    if a.shape[0] < 1 or a.shape[1] < 8:
        return np.empty(0), np.empty(0), 0
    nps = int(nperseg or min(a.shape[1], 4096))
    nps = max(16, min(nps, a.shape[1]))
    f, pxx = welch(a, fs=float(fs), nperseg=nps, noverlap=nps // 2, axis=1)
    with np.errstate(invalid="ignore"):
        psd = np.nanmean(pxx, axis=0)
    return f, psd, int(a.shape[0])


def excess_db(psd_event, psd_clean) -> np.ndarray:
    """10·log10(event / clean) per frequency bin — how much extra power the event
    group carries. NaN-safe; 0 where either side is non-finite."""
    e = np.asarray(psd_event, dtype=np.float64)
    c = np.asarray(psd_clean, dtype=np.float64)
    assert e.shape == c.shape, "PSDs must share the frequency grid"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (e + _EPS) / (c + _EPS)
        out = 10.0 * np.log10(ratio)
    return np.where(np.isfinite(out), out, 0.0)


def _near_line(freqs, line_hz, notch_bw: float) -> np.ndarray:
    """Boolean mask of bins within ``notch_bw`` Hz of any line-noise frequency."""
    f = np.asarray(freqs, dtype=np.float64)
    mask = np.zeros(f.shape[0], dtype=bool)
    for ln in line_hz:
        mask |= np.abs(f - float(ln)) <= float(notch_bw)
    return mask


def find_fundamental(freqs, exc_db, *, fmin: float = 4.0, fmax: float = 2000.0,
                     line_hz=None, notch_bw: float = 2.0,
                     harm_tol_hz: float = 5.0) -> dict:
    """Locate the event's fundamental = the largest excess-power peak in
    ``[fmin, fmax]`` that is NOT within ``notch_bw`` of a mains line, then report
    which integer harmonics also show excess. Returns
    ``{fundamental_hz, peak_db, harmonics:[{n,hz,db}], n_harmonics}`` (NaN
    fundamental when nothing qualifies)."""
    f = np.asarray(freqs, dtype=np.float64)
    d = np.asarray(exc_db, dtype=np.float64)
    assert f.shape == d.shape, "freqs / exc_db length mismatch"
    line_hz = line_hz or []
    band = (f >= float(fmin)) & (f <= float(fmax)) & np.isfinite(d)
    band &= ~_near_line(f, line_hz, notch_bw)
    if not band.any() or np.nanmax(np.where(band, d, -np.inf)) <= 0:
        return {"fundamental_hz": float("nan"), "peak_db": float("nan"),
                "harmonics": [], "n_harmonics": 0}
    idx = int(np.argmax(np.where(band, d, -np.inf)))
    f0 = float(f[idx])
    harmonics = []
    for n in range(1, 7):
        target = f0 * n
        if target > f[-1]:
            break
        j = int(np.argmin(np.abs(f - target)))
        if abs(f[j] - target) <= harm_tol_hz and d[j] > 0 and not _near_line(
                f[j:j + 1], line_hz, notch_bw)[0]:
            harmonics.append({"n": n, "hz": float(f[j]), "db": float(d[j])})
    return {"fundamental_hz": f0, "peak_db": float(d[idx]),
            "harmonics": harmonics, "n_harmonics": len(harmonics)}


def excess_band(freqs, exc_db, *, thresh_db: float = 3.0, fmin: float = 4.0,
                fmax: float = 8000.0) -> tuple:
    """Contiguous frequency band around the peak excess where the event exceeds
    the clean group by >= ``thresh_db``, searched within ``[fmin, fmax]`` (so DC /
    baseline leakage below *fmin* can't stretch the band to 0 Hz). Returns
    ``(lo_hz, hi_hz)`` or ``(nan, nan)`` when the event never clears the
    threshold."""
    f = np.asarray(freqs, dtype=np.float64)
    d = np.asarray(exc_db, dtype=np.float64)
    assert f.shape == d.shape, "freqs / exc_db length mismatch"
    inband = (f >= float(fmin)) & (f <= float(fmax))
    above = np.isfinite(d) & (d >= float(thresh_db)) & inband
    if not above.any():
        return float("nan"), float("nan")
    peak = int(np.argmax(np.where(above, d, -np.inf)))
    lo = peak
    while lo > 0 and above[lo - 1]:              # bounded by array length
        lo -= 1
    hi = peak
    while hi < f.shape[0] - 1 and above[hi + 1]:
        hi += 1
    return float(f[lo]), float(f[hi])
