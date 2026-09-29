"""Stage 1 preprocessing: turn a windowed trace matrix into SHAPE space.

Input ``X`` is already baseline-subtracted and cropped to the analysis window
(``data.build_dataset`` does this via the shared readers, ``synth.generate`` produces
it directly). Here we:

  * optionally align each trace to its first deflection instead of to stim onset
    (a latency-robust view; off by default),
  * normalize each trace to unit L2 (default) or unit peak-abs, KEEPING the norm
    divisor as the per-trial ``gain`` scalar (there is no per-trial gain in the .mat,
    so the norm divisor is the natural amplitude-scaling estimate),
  * QC-drop trials with no evoked response: correlation with the grand-mean shape
    below a threshold. The dropped COUNT is always returned (never a silent discard).

Pure numpy so it is unit-testable against ``synth``. Sign is preserved throughout
(a trough-first and a peak-first response are different shapes, not the same one).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _shift_rows(X: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each row of *X* by its integer sample count with edge fill (no wrap).
    Positive shift delays the trace. Vectorized-enough for our N."""
    n, T = X.shape
    out = np.empty_like(X)
    for i in range(n):
        s = int(shifts[i])
        if s == 0:
            out[i] = X[i]
        elif s > 0:
            s = min(s, T)
            out[i, :s] = X[i, 0]
            out[i, s:] = X[i, :T - s]
        else:
            s = min(-s, T)
            out[i, T - s:] = X[i, -1]
            out[i, :T - s] = X[i, s:]
    return out


def first_deflection_index(X: np.ndarray, thresh_frac: float = 0.2) -> np.ndarray:
    """Per-trace index of the first sample whose |amplitude| exceeds
    ``thresh_frac`` of that trace's max |amplitude|. Falls back to 0 for flat rows."""
    assert 0.0 < thresh_frac < 1.0, "thresh_frac must be in (0, 1)"
    peak = np.nanmax(np.abs(X), axis=1)
    peak[peak == 0] = 1.0
    thr = thresh_frac * peak
    over = np.abs(X) >= thr[:, None]
    idx = np.argmax(over, axis=1)          # first True, or 0 if none
    idx[~over.any(axis=1)] = 0
    return idx.astype(int)


def align_first_deflection(X: np.ndarray, thresh_frac: float = 0.2
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Align every trace so its first deflection sits at the MEDIAN first-deflection
    index. Returns ``(X_aligned, applied_shifts)``. This absorbs pure latency jitter
    so downstream clustering sees morphology, not onset time."""
    assert X.ndim == 2, "X must be 2-D [n, T]"
    idx = first_deflection_index(X, thresh_frac)
    target = int(np.median(idx))
    shifts = target - idx
    return _shift_rows(X, shifts), shifts


def normalize(X: np.ndarray, mode: str = "l2") -> tuple[np.ndarray, np.ndarray]:
    """Normalize each trace so clustering sees SHAPE, not size. ``mode='l2'`` divides
    by the L2 norm (weights the whole waveform), ``mode='peak'`` by max |amplitude|
    (weights the largest deflection). Returns ``(Xn, gain)`` where ``gain`` is the
    per-trial divisor kept as the amplitude-scaling scalar. Flat rows stay zeros with
    gain 1.0 (sign preserved for non-flat rows)."""
    assert mode in ("l2", "peak"), "mode must be 'l2' or 'peak'"
    assert X.ndim == 2, "X must be 2-D [n, T]"
    if mode == "l2":
        gain = np.sqrt(np.nansum(X * X, axis=1))
    else:
        gain = np.nanmax(np.abs(X), axis=1)
    gain = np.asarray(gain, dtype=np.float64)
    safe = gain.copy()
    safe[safe == 0] = 1.0
    return X / safe[:, None], gain


def grand_mean_correlation(Xn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation of each normalized trace with the grand-mean normalized
    shape. Returns ``(corr[n], grand_mean[T])``. Reported as a diagnostic only, NOT the
    QC drop rule: a genuinely distinct minority shape correlates poorly with the blurred
    grand mean, so dropping on it would preferentially delete the rarer shapes and bias
    the very structure this package is trying to find."""
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    gm = np.nanmean(Xn, axis=0)
    gm_c = gm - np.nanmean(gm)
    gm_norm = np.sqrt(np.nansum(gm_c * gm_c))
    if gm_norm == 0:
        return np.zeros(Xn.shape[0], dtype=np.float64), gm
    Xc = Xn - np.nanmean(Xn, axis=1, keepdims=True)
    num = np.nansum(Xc * gm_c[None, :], axis=1)
    den = np.sqrt(np.nansum(Xc * Xc, axis=1)) * gm_norm
    den[den == 0] = np.nan
    corr = num / den
    return np.nan_to_num(corr, nan=0.0), gm


def smoothness(Xn: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of each trace. A real evoked LFP response is smooth
    (autocorr near 1); a no-response trial is noise (autocorr near 0). This is the QC
    drop signal: it is SHAPE-AGNOSTIC, so unlike a grand-mean correlation it does not
    prefer the majority shape."""
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    Xc = Xn - np.nanmean(Xn, axis=1, keepdims=True)
    num = np.nansum(Xc[:, :-1] * Xc[:, 1:], axis=1)
    den = np.nansum(Xc * Xc, axis=1)
    den = np.where(den == 0, np.nan, den)
    ac = num / den
    return np.nan_to_num(ac, nan=0.0)


def preprocess(X: np.ndarray, *, mode: str = "l2", align: bool = False,
               qc_min_smoothness: float = 0.3, thresh_frac: float = 0.2,
               log: Callable[[str], None] | None = None) -> dict:
    """Full Stage-1 pass. Returns a dict with the SHAPE matrix and everything a
    caller needs to keep meta aligned and report what happened:

    ``{"Xn", "gain", "keep", "corr", "smooth", "grand_mean", "shifts", "n_in",
    "n_kept", "n_dropped"}``. ``keep`` is a boolean mask over the INPUT rows (apply it
    to both ``Xn`` and the meta frame). Trials are dropped on SMOOTHNESS (lag-1
    autocorrelation) so no-response noise trials go but distinct minority shapes stay.
    Nothing is dropped silently: ``n_dropped`` is always set and ``log`` (if given) is
    called with the count.
    """
    assert X.ndim == 2, "X must be 2-D [n, T]"
    n_in = int(X.shape[0])
    shifts = np.zeros(n_in, dtype=int)
    Xw = X
    if align:
        Xw, shifts = align_first_deflection(X, thresh_frac)
    Xn, gain = normalize(Xw, mode)
    corr, gm = grand_mean_correlation(Xn)
    smooth = smoothness(Xn)
    keep = smooth >= qc_min_smoothness
    n_kept = int(keep.sum())
    n_dropped = n_in - n_kept
    if log is not None:
        log(f"Stage 1 QC: kept {n_kept}/{n_in} trials "
            f"(dropped {n_dropped} with lag-1 autocorr < {qc_min_smoothness:.2f})")
    return {"Xn": Xn[keep], "gain": gain[keep], "keep": keep,
            "corr": corr, "smooth": smooth, "grand_mean": gm, "shifts": shifts,
            "n_in": n_in, "n_kept": n_kept, "n_dropped": n_dropped}
