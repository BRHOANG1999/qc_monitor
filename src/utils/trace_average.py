"""Average a set of per-file waveform traces onto a common time grid.

The weekly/daily stim figures need a daily / weekly *average* stimulus waveform
from many per-file mean traces. Traces of one channel at one config share a time
grid in practice, but nothing guarantees it, so this handles both: a fast
equal-length ``nanmean`` path and an ``np.interp`` fallback onto a reference grid
when lengths differ. Pure (numpy only) so it is trivially unit-testable and
carries no matplotlib/Dash import weight.

*Alignment*: pass ``align="rising_edge"`` to align each trace by the stimulus
artifact's RISING EDGE (the steepest positive slope near t=0) before averaging,
instead of trusting the recorded t=0. Per-recording onset jitter otherwise blurs
the average (a smeared peak); aligning on the sharp rising edge keeps it crisp.
"""

from __future__ import annotations

import warnings

import numpy as np

_MAX_TRACES = 100_000       # NASA Rule 2 bound

# Window (ms) around the stimulus onset in which the rising edge is sought.
_EDGE_WIN = (-0.5, 0.5)


def _rising_edge_time(tm: np.ndarray, v: np.ndarray, win=_EDGE_WIN):
    """Time (ms) of the stim-artifact rising edge: the sample of maximum POSITIVE
    slope within *win* around t=0 (the onset of the fast phase). None when the
    window catches too few samples."""
    mask = (tm >= win[0]) & (tm <= win[1])
    idx = np.where(mask)[0]
    if idx.size < 3:
        return None
    dv = np.diff(v[idx])
    if dv.size == 0:
        return None
    return float(tm[idx[int(np.argmax(dv))]])


def _stack(traces: list[dict], value_key: str, time_key: str, align=None):
    """Stack traces onto one time grid -> ``(ref_t, stack)`` or ``(None, None)``.
    Equal-length traces stack directly; otherwise (and always when *align* shifts
    the grids) each is interpolated onto a reference axis. ``align='rising_edge'``
    re-references every trace's time axis so its rising edge sits at the MEDIAN
    rising-edge time, then interpolates onto the common grid."""
    assert isinstance(traces, list), "traces must be a list"
    assert len(traces) < _MAX_TRACES, "trace count runaway"
    usable = []
    for t in traces:
        tm = t.get(time_key)
        v = t.get(value_key)
        if tm is None or v is None:
            continue
        tm = np.asarray(tm, dtype=float)
        v = np.asarray(v, dtype=float)
        if tm.size >= 3 and tm.size == v.size:
            usable.append((tm, v))
    if not usable:
        return None, None
    if align == "rising_edge":
        edges = [_rising_edge_time(tm, v) for tm, v in usable]
        valid = [e for e in edges if e is not None]
        if valid:
            ref_edge = float(np.median(valid))
            usable = [((tm - (e - ref_edge)) if e is not None else tm, v)
                      for (tm, v), e in zip(usable, edges)]
        # Shifted axes no longer share a grid -> interpolate onto the longest.
        ref_t = max((tm for tm, _ in usable), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v in usable])
        return ref_t, stack
    if len({tm.size for tm, _ in usable}) == 1:
        ref_t = usable[0][0]
        stack = np.vstack([v for _, v in usable])
    else:
        ref_t = max((tm for tm, _ in usable), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v in usable])
    return ref_t, stack


def _mean_sem(stack: np.ndarray):
    """(mean, sem) over axis 0. SEM = std(ddof=1)/sqrt(n), 0 where n<2."""
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        n = np.sum(np.isfinite(stack), axis=0)
        std = np.nanstd(stack, axis=0, ddof=1)
        sem = np.where(n >= 2, std / np.sqrt(np.maximum(n, 1)), 0.0)
    return mean, np.nan_to_num(sem)


def average_traces(traces: list[dict], *, value_key: str = "stim_trace",
                   time_key: str = "time_ms", align=None
                   ) -> tuple[list | None, list | None]:
    """Mean of ``traces[i][value_key]`` over a shared ``time_key`` grid. Returns
    ``(time_ms, mean)`` as plain lists, or ``(None, None)`` when nothing usable.
    *align* -> :func:`_stack` (e.g. ``'rising_edge'``)."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align)
    if ref_t is None:
        return None, None
    with np.errstate(invalid="ignore"):
        y = np.nanmean(stack, axis=0)
    return ref_t.tolist(), y.tolist()


def average_with_sem(traces: list[dict], *, value_key: str = "stim_trace",
                     time_key: str = "time_ms", align=None):
    """Like :func:`average_traces` but also returns the per-sample standard error
    of the mean: ``(time_ms, mean, sem)`` (or ``(None, None, None)``). *align* ->
    :func:`_stack`. SEM is the convention for evoked traces; caller shades
    mean +/- sem."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align)
    if ref_t is None:
        return None, None, None
    mean, sem = _mean_sem(stack)
    return ref_t.tolist(), mean.tolist(), sem.tolist()


def align_average_with_traces(traces: list[dict], *, value_key: str = "stim_trace",
                              time_key: str = "time_ms", align=None):
    """``(time_ms, mean, sem, aligned_traces)`` -- the mean +/- SEM AND every
    constituent trace resampled onto the SAME (optionally rising-edge-aligned)
    grid, so a figure can overlay the individual recordings transparently under
    the bold mean. ``(None, None, None, [])`` when nothing usable."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align)
    if ref_t is None:
        return None, None, None, []
    mean, sem = _mean_sem(stack)
    return (ref_t.tolist(), mean.tolist(), sem.tolist(),
            [row.tolist() for row in stack])
