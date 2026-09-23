"""Pure numpy/scipy core for riding-event analysis.

Every function here is deterministic and side-effect free (no I/O, no plotting),
so the plant-and-recover unit tests exercise the whole detection chain on
synthetic signals. Shapes follow the evoked convention: ``traces`` is
``[epochs x samples]`` with time along axis 1; a continuous LFP ``signal`` is
1-D ``[samples]``. All sample rates are the native ~20 kHz.

The two reuse anchors from elsewhere in the repo are kept at arm's length so this
module stays low-level: ``evoked_features._bandpass`` (SOS Butterworth, stable at
20 kHz), ``hilbert_envelope.band_envelope`` (FFT band envelope) and
``peakseek.peakseek`` (the lab's min-distance peak detector). The matched filter
and the rising-edge alignment (the one primitive with no prior Python form) are
implemented here.
"""

from __future__ import annotations

import numpy as np

from src.utils import evoked_features as _ef
from src.utils.hilbert_envelope import band_envelope
from src.utils.peakseek import peakseek

_EPS = 1e-12
_MAX_EPOCHS = 2_000_000          # NASA Rule 2: explicit per-epoch loop bound.
_MAX_SNIPPETS = 2_000_000


# --------------------------------------------------------------------- #
#  Robust template + residual (Prong A core)
# --------------------------------------------------------------------- #

def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (raw, not scaled), NaN-safe. 0 when degenerate."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.median(np.abs(a - np.median(a))))


def robust_template(traces, *, iters: int = 1, k: float = 4.0,
                    win_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-recording *average evoked response* as an element-wise MEDIAN across
    epochs, refined by dropping the epochs that carry the riding event so they
    don't bias the template.

    Returns ``(template[samples], keep_mask[epochs])``. The median is already
    robust to a minority of event epochs; each of *iters* refit passes then
    recomputes the median over only the epochs whose whole-trace residual RMS is
    within ``median + k*MAD`` of the cohort (``win_mask`` restricts the RMS to a
    sample window, e.g. post-stim). Idempotent once the kept set stabilises.
    """
    a = np.asarray(traces, dtype=np.float64)
    assert a.ndim == 2 and a.shape[0] >= 1, "traces must be [epochs x samples]"
    assert iters >= 0 and k > 0, "iters >= 0 and k > 0"
    cols = (np.asarray(win_mask, dtype=bool) if win_mask is not None
            else np.ones(a.shape[1], dtype=bool))
    assert cols.shape[0] == a.shape[1], "win_mask length must match samples"
    template = np.median(a, axis=0)
    keep = np.ones(a.shape[0], dtype=bool)
    n_iter = int(min(iters, 10))                 # NASA Rule 2: bounded refit
    for _ in range(n_iter):
        resid = a[:, cols] - template[cols][None, :]
        rms = np.sqrt(np.mean(resid * resid, axis=1))
        thr = np.median(rms) + k * _mad(rms)
        new_keep = rms <= thr
        if new_keep.sum() < 1 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 1 else keep
            break
        keep = new_keep
        template = np.median(a[keep], axis=0)
    return template, keep


def residuals(traces, template) -> np.ndarray:
    """Per-epoch residual ``epoch - template`` (the stereotyped evoked waveform
    cancels; the riding event and noise remain)."""
    a = np.asarray(traces, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    assert a.ndim == 2, "traces must be 2-D"
    assert t.shape[0] == a.shape[1], "template length must match samples"
    return a - t[None, :]


def _window_mask(time_ms, win_ms) -> np.ndarray:
    t = np.asarray(time_ms, dtype=np.float64)
    lo, hi = float(win_ms[0]), float(win_ms[1])
    assert hi > lo, "win_ms must be increasing"
    return (t >= lo) & (t <= hi)


def event_energy(resid, time_ms, *, win_ms=(2.0, 100.0), fs: float | None = None,
                 band=None) -> np.ndarray:
    """Per-epoch event energy = RMS of the residual over the post-stim window
    ``win_ms`` (ms). Broadband by default: after the template is removed a clean
    epoch's residual is noise while an event epoch's is the (large) event, so the
    raw residual RMS already separates them without assuming the event's band.
    Pass ``band=(lo,hi)`` + ``fs`` to band-limit first (used once the spectral
    figure has revealed the event band)."""
    r = np.asarray(resid, dtype=np.float64)
    assert r.ndim == 2, "resid must be 2-D [epochs x samples]"
    mask = _window_mask(time_ms, win_ms)
    if mask.sum() < 2:                            # window fell outside the epoch
        mask = np.ones(r.shape[1], dtype=bool)
    if band is not None:
        assert fs and fs > 0, "band filtering needs fs"
        r = _ef._bandpass(r, float(fs), float(band[0]), float(band[1]))
    w = r[:, mask]
    return np.sqrt(np.mean(w * w, axis=1))


def flag_events(energy, *, k: float = 4.0) -> np.ndarray:
    """Boolean event mask: epochs whose energy exceeds ``median + k*MAD`` (robust
    to the events themselves). Returns all-False when nothing stands out."""
    e = np.asarray(energy, dtype=np.float64)
    assert e.ndim == 1, "energy must be 1-D"
    assert k > 0, "k must be positive"
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return np.zeros(e.shape[0], dtype=bool)
    thr = float(np.median(finite)) + k * _mad(finite)
    return np.isfinite(e) & (e > thr)


# --------------------------------------------------------------------- #
#  Candidate detection + rising-edge alignment + matched filter (Prong B)
# --------------------------------------------------------------------- #

def detect_candidates(signal, fs: float, *, band=(20.0, 200.0),
                      min_dist_sec: float = 0.05, k: float = 4.0,
                      smooth_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray,
                                                       float]:
    """Candidate event sample indices on a continuous LFP *signal*.

    RAW band-limited amplitude envelope (``band_envelope(smooth=False)``) with a
    short ``smooth_ms`` moving average -> robust height threshold
    (``median + k*MAD``) -> ``peakseek`` with a min-distance refractory. Returns
    ``(locs[samples], envelope, threshold)``. (The BHZ ``smooth=True`` low-pass is
    ~0.2 Hz -- tuned for tens-of-seconds seizures -- and erases the short riding
    event, so it is deliberately not used here.)"""
    x = np.asarray(signal, dtype=np.float64)
    assert x.ndim == 1, "signal must be 1-D"
    assert fs and fs > 0, "fs must be positive"
    env = band_envelope(x, float(fs), float(band[0]), float(band[1]), smooth=False)
    if smooth_ms and smooth_ms > 0:
        from scipy.ndimage import uniform_filter1d
        w = max(1, int(round(float(smooth_ms) * 1e-3 * float(fs))))
        env = uniform_filter1d(env, size=w, mode="nearest")
    finite = env[np.isfinite(env)]
    if finite.size == 0:
        return np.empty(0, dtype=np.int64), env, float("nan")
    thr = float(np.median(finite)) + k * _mad(finite)
    dist = max(1, int(round(float(min_dist_sec) * float(fs))))
    locs, _pks = peakseek(env, dist, minpeakh=thr)
    return locs.astype(np.int64), env, thr


def snippets_around(signal, locs, *, pre: int, post: int) -> tuple[np.ndarray,
                                                                   np.ndarray]:
    """Fixed-width ``[pre, post)`` sample windows centred on each of *locs*.

    Returns ``(snips[M x (pre+post)], kept_locs[M])`` dropping any window that
    would run off either edge (so no zero-padding contaminates a template)."""
    x = np.asarray(signal, dtype=np.float64)
    loc = np.asarray(locs, dtype=np.int64)
    assert x.ndim == 1, "signal must be 1-D"
    assert pre >= 0 and post >= 1, "need pre >= 0, post >= 1"
    n = x.shape[0]
    ok = (loc - pre >= 0) & (loc + post <= n)
    kept = loc[ok]
    if kept.size == 0:
        return np.empty((0, pre + post)), kept
    idx = kept[:, None] + np.arange(-pre, post)[None, :]
    return x[idx], kept


def rising_edge_align(snippets, fs: float, *, search_ms: float = 10.0,
                      pre_ms: float = 5.0, post_ms: float = 25.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Align candidate *snippets* by their RISING EDGE.

    Per snippet the fiducial is the steepest positive slope (max ``dV/dt``) inside
    the first ``search_ms`` — the event's onset — then a fixed ``[-pre_ms,
    +post_ms]`` window is re-extracted around it. Snippets whose window falls off
    an edge are dropped. Returns ``(aligned[M x W], fiducials[M])`` (fiducials in
    original-snippet samples). This is the one primitive with no prior Python
    form (ported from ``batchEvokedWorkerFcn.m`` threshold-then-first-peak)."""
    s = np.asarray(snippets, dtype=np.float64)
    assert s.ndim == 2 and s.shape[0] >= 1, "snippets must be [M x L]"
    assert fs and fs > 0, "fs must be positive"
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    sw = max(2, int(round(search_ms * 1e-3 * fs)))
    L = s.shape[1]
    sw = min(sw, L - 1)
    dv = np.diff(s[:, :sw + 1], axis=1)          # slope over the search region
    fid = np.argmax(dv, axis=1) + 1              # rising-edge sample per snippet
    out, keep_fid = [], []
    assert s.shape[0] < _MAX_SNIPPETS, "snippet count exceeds bound"
    for i in range(s.shape[0]):
        f = int(fid[i])
        if f - pre >= 0 and f + post <= L:
            out.append(s[i, f - pre:f + post])
            keep_fid.append(f)
    if not out:
        return np.empty((0, pre + post)), np.empty(0, dtype=np.int64)
    return np.asarray(out), np.asarray(keep_fid, dtype=np.int64)


def matched_filter_series(signal, template) -> np.ndarray:
    """Normalized cross-correlation (per-lag Pearson r) of *template* against a
    continuous *signal*, computed in O(n log n) via an FFT convolution for the
    numerator and cumulative sums for the per-window normalisation. Returns
    ``r[n-W+1]`` in ``[-1, 1]`` (r[i] aligns the template start at sample i)."""
    from scipy.signal import fftconvolve
    x = np.asarray(signal, dtype=np.float64)
    t = np.asarray(template, dtype=np.float64)
    assert x.ndim == 1 and t.ndim == 1, "signal and template must be 1-D"
    w = t.shape[0]
    assert 2 <= w <= x.shape[0], "need 2 <= len(template) <= len(signal)"
    t0 = t - t.mean()
    tnorm = float(np.sqrt(np.sum(t0 * t0)))
    if tnorm <= _EPS:
        return np.zeros(x.shape[0] - w + 1)
    num = fftconvolve(x, t0[::-1], mode="valid")            # Σ x_win · t0
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csq = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = csum[w:] - csum[:-w]
    s2 = csq[w:] - csq[:-w]
    local_var = np.maximum(s2 - s1 * s1 / w, 0.0)           # Σ(x_win - mean)²
    denom = np.sqrt(local_var) * tnorm
    r = np.where(denom > _EPS, num / np.where(denom > _EPS, denom, 1.0), 0.0)
    return np.clip(r, -1.0, 1.0)


def matched_filter_detect(signal, template, fs: float, *, thresh: float = 0.7,
                          refractory_sec: float = 0.05) -> tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
    """Detect events where the matched-filter correlation crosses *thresh*.

    Returns ``(locs[samples], scores, r_series)`` — ``locs`` are the template-
    start samples of accepted detections (min-distance ``refractory_sec`` apart,
    keeping the higher r in a tie, via ``peakseek``)."""
    r = matched_filter_series(signal, template)
    assert fs and fs > 0, "fs must be positive"
    if r.size < 3:
        return np.empty(0, dtype=np.int64), np.empty(0), r
    dist = max(1, int(round(float(refractory_sec) * float(fs))))
    locs, pks = peakseek(r, dist, minpeakh=float(thresh))
    return locs.astype(np.int64), pks, r
