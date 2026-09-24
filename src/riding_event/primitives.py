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
_MAX_STIMS = 5_000_000

# Stim-artifact blank window (ms around each stimulus): the sharp deflection is
# removed BEFORE any filter/envelope/spectrum/matched-filter so its broadband
# energy and filter ringing can't leak into the post-stim analysis. Narrow so the
# riding event (which starts a few ms after the stim) is preserved. Matches the
# toolkit's -1..2 ms artifact window (peri_stim_artifact._ARTIFACT_MS).
DEFAULT_ARTIFACT_MS = (1.0, 2.0)


# --------------------------------------------------------------------- #
#  Stim-artifact blanking (run FIRST, before any other analysis)
# --------------------------------------------------------------------- #

def blank_artifact_epochs(traces, time_ms, *, pre_ms: float = 1.0,
                          post_ms: float = 2.0) -> np.ndarray:
    """Linearly interpolate across the stim artifact ``[-pre_ms, +post_ms]`` (t=0
    = stim) in every epoch, so the artifact never enters the template, residual,
    spectra or any filtered feature. Returns a new ``[epochs x samples]`` array;
    the post-stim event (starting a few ms out) is preserved. Falls back to NaN
    when the window touches an epoch edge (no anchor to interpolate from)."""
    a = np.array(traces, dtype=np.float64, copy=True)
    t = np.asarray(time_ms, dtype=np.float64)
    assert a.ndim == 2 and t.size == a.shape[1], "traces/time_ms shape mismatch"
    assert pre_ms >= 0 and post_ms >= 0, "pre_ms/post_ms must be >= 0"
    m = (t >= -abs(pre_ms)) & (t <= abs(post_ms))
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return a
    i0, i1 = int(idx[0]), int(idx[-1])
    lo, hi = i0 - 1, i1 + 1
    if lo < 0 or hi >= t.size or t[hi] == t[lo]:
        a[:, i0:i1 + 1] = np.nan                  # edge: can't anchor an interp
        return a
    frac = (t[i0:i1 + 1] - t[lo]) / (t[hi] - t[lo])
    a[:, i0:i1 + 1] = (a[:, lo][:, None] * (1.0 - frac)[None, :]
                       + a[:, hi][:, None] * frac[None, :])
    return a


def blank_artifact_continuous(signal, stim_samples, *, pre: int,
                              post: int) -> np.ndarray:
    """Linearly interpolate across the stim artifact (``[s-pre, s+post)`` samples)
    at every stimulus *s* in a continuous 1-D *signal*, so the envelope, matched
    filter and thresholds see no stim transients. Returns a new array; the
    stim-evoked RESPONSE after the narrow blank is left intact (it is excluded
    from detections separately, by the wider stim-time gate)."""
    x = np.array(signal, dtype=np.float64, copy=True)
    ss = np.asarray(stim_samples, dtype=np.int64)
    assert x.ndim == 1, "signal must be 1-D"
    assert pre >= 0 and post >= 1, "need pre >= 0, post >= 1"
    n = x.shape[0]
    cnt = 0
    for s in ss:
        assert cnt < _MAX_STIMS, "stim count exceeds bound"
        cnt += 1
        lo, hi = max(0, int(s) - pre), min(n, int(s) + post)
        a, b = lo - 1, hi
        if hi <= lo:
            continue
        if a >= 0 and b < n:
            x[lo:hi] = np.interp(np.arange(lo, hi), [a, b], [x[a], x[b]])
        else:
            x[lo:hi] = np.nan
    return x


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


def flag_events(energy, *, k: float = 4.0, noise_k: float | None = None,
                min_pct: float | None = None) -> np.ndarray:
    """Boolean event mask: epochs above the noise floor ``median + k*MAD`` and
    below the ``noise_k`` ceiling (glitch rejection). When *min_pct* is given the
    epoch must ALSO be at/above that percentile of the score — a selectivity cap
    for a HEAVY-TAILED score (the ripple energy has a long tail, so MAD alone
    flags far too many; the percentile isolates the prominent tail while the
    noise-floor term still keeps a genuinely clean recording near-empty).
    ``noise_k``/``min_pct`` None = that bound off. All-False when nothing stands out."""
    e = np.asarray(energy, dtype=np.float64)
    assert e.ndim == 1, "energy must be 1-D"
    assert k > 0, "k must be positive"
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return np.zeros(e.shape[0], dtype=bool)
    med, mad = float(np.median(finite)), _mad(finite)
    mask = np.isfinite(e) & (e > med + k * mad)
    if min_pct is not None:
        mask &= e >= float(np.percentile(finite, float(min_pct)))
    if noise_k:
        mask &= e <= med + float(noise_k) * mad
    return mask


# --------------------------------------------------------------------- #
#  Candidate detection + rising-edge alignment + matched filter (Prong B)
# --------------------------------------------------------------------- #

def detect_candidates(signal, fs: float, *, band=(20.0, 200.0),
                      min_dist_sec: float = 0.05, k: float = 4.0,
                      noise_k: float | None = None,
                      smooth_ms: float = 5.0) -> tuple[np.ndarray, np.ndarray,
                                                       float, float]:
    """Candidate event sample indices on a continuous LFP *signal*.

    RAW band-limited amplitude envelope (``band_envelope(smooth=False)``) with a
    short ``smooth_ms`` moving average -> robust band ``median + k*MAD`` (lower)
    to ``median + noise_k*MAD`` (upper) -> ``peakseek`` with a min-distance
    refractory. The upper ``noise_k`` bound rejects implausibly large transients
    (glitches / electrical artifacts) that dwarf a real event -- the physiological
    riding event is a bounded-amplitude oscillation, a lone giant spike is noise.
    Returns ``(locs[samples], envelope, thr_lo, thr_hi)`` (``thr_hi=inf`` when no
    ceiling). (The BHZ ``smooth=True`` low-pass is ~0.2 Hz, tuned for
    tens-of-seconds seizures, and erases the short event, so it is not used.)"""
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
        return np.empty(0, dtype=np.int64), env, float("nan"), float("inf")
    med, mad = float(np.median(finite)), _mad(finite)
    thr = med + k * mad
    thr_hi = med + float(noise_k) * mad if noise_k else float("inf")
    dist = max(1, int(round(float(min_dist_sec) * float(fs))))
    locs, _pks = peakseek(env, dist, minpeakh=thr)
    if np.isfinite(thr_hi) and locs.size:
        locs = locs[env[locs] <= thr_hi]          # drop giant-transient noise
    return locs.astype(np.int64), env, thr, thr_hi


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


# pHFO gate defaults. HF-SNR (event HF-RMS / baseline HF-RMS) is the primary
# criterion -- it catches SUSTAINED bursts (prominence misses them: a long packet
# keeps the median envelope high -> low prom) and slow-wave-RIDING packets (frac
# misses them: the leftover slow component dilutes the HF fraction). OR'd with
# prominence (+relaxed frac) for brief sharp packets. Validated on the labelled set:
# LFDs sit at SNR~1 & prom~3; pHFOs at SNR>=2.5 or prom>6.
DEFAULT_PHFO_SNR = 2.0           # min HF-band RMS(event) / RMS(baseline)
DEFAULT_PHFO_PROM = 6.0          # min HF-envelope peak/median (brief-packet path)
DEFAULT_PHFO_FRAC = 0.10         # min HF-band energy fraction (with the prom path)
_PHFO_EVT_MS = (2.0, 120.0)      # SNR event window, relative to the reference
_PHFO_BASE_MS = (-120.0, -20.0)  # SNR baseline window (pre-reference)
_PHFO_GATE_MS = (2.0, 100.0)     # window frac/prom are measured over (epochs path)


def phfo_gate(frac, prom, snr, *, snr_min: float = DEFAULT_PHFO_SNR,
              prom_min: float = DEFAULT_PHFO_PROM,
              frac_min: float = DEFAULT_PHFO_FRAC) -> np.ndarray:
    """The pHFO DETECTION decision (boolean mask): HF-SNR >= *snr_min* OR
    (prominence >= *prom_min* AND HF-fraction >= *frac_min*). The SNR term catches
    sustained / slow-wave-riding bursts; the prom+frac term catches brief sharp
    packets. Missing SNR (NaN, e.g. continuous windows with no baseline) falls back
    to the prom+frac path."""
    fr = np.asarray(frac, dtype=np.float64)
    pr = np.asarray(prom, dtype=np.float64)
    sn = np.asarray(snr, dtype=np.float64)
    return ((np.isfinite(sn) & (sn >= float(snr_min)))
            | (np.isfinite(pr) & (pr >= float(prom_min))
               & np.isfinite(fr) & (fr >= float(frac_min))))


def _hf_snr(x2d, fs: float, band, ev_mask, base_mask) -> np.ndarray:
    """HF-band RMS over *ev_mask* columns / HF-band RMS over *base_mask* columns,
    per row of the 2-D window matrix *x2d*. NaN when a mask is empty."""
    lo = max(1.0, float(band[0]))
    hi = min(float(band[1]), 0.49 * float(fs))
    n = x2d.shape[0]
    if hi <= lo or n == 0 or not ev_mask.any() or not base_mask.any():
        return np.full(n, np.nan)
    hf = _ef._bandpass(np.asarray(x2d, dtype=np.float64), float(fs), lo, hi)
    ev = np.sqrt(np.mean(hf[:, ev_mask] ** 2, axis=1))
    ba = np.sqrt(np.mean(hf[:, base_mask] ** 2, axis=1))
    return ev / (ba + _EPS)


def phfo_metrics(signal, locs, fs: float, band, *, pre_ms: float = 10.0,
                 post_ms: float = 60.0, evt_ms=_PHFO_EVT_MS,
                 base_ms=_PHFO_BASE_MS) -> tuple:
    """Per-detection pHFO metrics ``(hf_frac, prominence, hf_snr)``:

    - ``hf_frac`` = HF-band RMS / broadband RMS over ``[-pre_ms, +post_ms]``.
    - ``prominence`` = peak HF envelope / median HF envelope (brief-packet cue).
    - ``hf_snr`` = HF-band RMS in ``evt_ms`` / HF-band RMS in ``base_ms`` (both
      relative to the detection *loc*) -- the robust cue that also catches SUSTAINED
      bursts and slow-wave-riding packets. See ``phfo_gate`` for the decision.

    NaN for windows that run off an edge. Vectorised over detections."""
    from scipy.signal import hilbert
    x = np.asarray(signal, dtype=np.float64)
    loc = np.asarray(locs, dtype=np.int64)
    assert x.ndim == 1 and fs and fs > 0, "signal 1-D, fs > 0"
    assert loc.ndim == 1, "locs must be 1-D"
    pre = max(1, int(round(pre_ms * 1e-3 * fs)))
    post = max(1, int(round(post_ms * 1e-3 * fs)))
    frac = np.full(loc.size, np.nan)
    prom = np.full(loc.size, np.nan)
    snr = np.full(loc.size, np.nan)
    lo = max(1.0, float(band[0]))
    hi = min(float(band[1]), 0.49 * float(fs))
    if hi <= lo:
        return frac, prom, snr
    ok = (loc - pre >= 0) & (loc + post <= x.size)
    if ok.any():
        idx = (loc[ok] - pre)[:, None] + np.arange(pre + post)[None, :]
        win = x[idx]
        hf = _ef._bandpass(win, float(fs), lo, hi)
        env = np.abs(hilbert(hf, axis=1))
        hf_rms = np.sqrt(np.mean(hf * hf, axis=1))
        tot = np.sqrt(np.mean((win - win.mean(axis=1, keepdims=True)) ** 2, axis=1))
        frac[ok] = hf_rms / (tot + _EPS)
        prom[ok] = np.max(env, axis=1) / (np.median(env, axis=1) + _EPS)
    # HF-SNR: event vs a local pre-baseline window (both relative to loc)
    b0 = int(round(base_ms[0] * 1e-3 * fs))
    e1 = int(round(evt_ms[1] * 1e-3 * fs))
    span = e1 - b0
    ok2 = (loc + b0 >= 0) & (loc + b0 + span <= x.size)
    if ok2.any() and span > 4:
        idx2 = (loc[ok2] + b0)[:, None] + np.arange(span)[None, :]
        tw = (np.arange(span) + b0) / float(fs) * 1000.0   # ms relative to loc
        evm = (tw >= evt_ms[0]) & (tw <= evt_ms[1])
        bam = (tw >= base_ms[0]) & (tw <= base_ms[1])
        snr[ok2] = _hf_snr(x[idx2], fs, band, evm, bam)
    return frac, prom, snr


def phfo_metrics_epochs(traces, time_ms, fs: float, band, *,
                        gate_ms=_PHFO_GATE_MS, evt_ms=_PHFO_EVT_MS,
                        base_ms=_PHFO_BASE_MS) -> tuple:
    """Per-epoch pHFO metrics ``(hf_frac, prominence, hf_snr)`` on FULL epochs
    ``[epochs x T]`` with ``time_ms`` (t=0 = stim): frac/prom over *gate_ms*, and
    hf_snr = HF-RMS in the post-stim *evt_ms* window / HF-RMS in the PRE-stim
    *base_ms* window (the stim-locked baseline). Pass the RESIDUAL (evoked response
    subtracted) so only the riding packet remains. Same decision via ``phfo_gate``."""
    from scipy.signal import hilbert
    x = np.asarray(traces, dtype=np.float64)
    t = np.asarray(time_ms, dtype=np.float64)
    assert x.ndim == 2, "traces must be [epochs x T]"
    assert fs and fs > 0, "fs must be positive"
    e = x.shape[0]
    frac = np.full(e, np.nan)
    prom = np.full(e, np.nan)
    lo = max(1.0, float(band[0]))
    hi = min(float(band[1]), 0.49 * float(fs))
    gm = (t >= gate_ms[0]) & (t <= gate_ms[1])
    if hi <= lo or e == 0 or not gm.any():
        return frac, prom, np.full(e, np.nan)
    g = x[:, gm]
    hf = _ef._bandpass(g, float(fs), lo, hi)
    env = np.abs(hilbert(hf, axis=1))
    hf_rms = np.sqrt(np.mean(hf * hf, axis=1))
    tot = np.sqrt(np.mean((g - g.mean(axis=1, keepdims=True)) ** 2, axis=1))
    frac = hf_rms / (tot + _EPS)
    prom = np.max(env, axis=1) / (np.median(env, axis=1) + _EPS)
    evm = (t >= evt_ms[0]) & (t <= evt_ms[1])
    bam = (t >= base_ms[0]) & (t <= base_ms[1])
    snr = _hf_snr(x, fs, band, evm, bam)
    return frac, prom, snr


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
