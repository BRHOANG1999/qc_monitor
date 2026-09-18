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
    """Time (ms) of the stim-artifact ONSET: the steepest POSITIVE slope within
    *win*, searched only UP TO the artifact's dominant excursion (max |v|).

    The artifact is BIPHASIC -- a sharp onset rise, then (after the trough) an
    equally steep RECOVERY rise. A plain argmax of the slope over the whole
    window latches onto whichever is steeper, so a chunk of epochs (~12% in the
    measured recording) lock onto the recovery edge and end up aligned ~0.2 ms
    off. Capping the slope search at the dominant excursion pins every epoch to
    the onset, which always precedes the trough/recovery. None when the window
    catches too few samples."""
    mask = (tm >= win[0]) & (tm <= win[1])
    idx = np.where(mask)[0]
    if idx.size < 3:
        return None
    vv = np.asarray(v, dtype=float)[idx]
    dom = int(np.argmax(np.abs(vv)))           # the artifact's main excursion
    dv = np.diff(vv[:max(dom, 2) + 1])         # slope only up to that excursion
    if dv.size == 0:
        return None
    return float(tm[idx[int(np.argmax(dv))]])


def _stack(traces: list[dict], value_key: str, time_key: str, align=None,
           align_key=None):
    """Stack traces onto one time grid -> ``(ref_t, stack)`` or ``(None, None)``.
    Equal-length traces stack directly; otherwise (and always when *align* shifts
    the grids) each is interpolated onto a reference axis. ``align='rising_edge'``
    re-references every trace's time axis so its rising edge sits at the MEDIAN
    rising-edge time, then interpolates onto the common grid.

    *align_key* (default = *value_key*) selects the REFERENCE channel whose rising
    edge sets each trace's shift, while ``value_key`` is what gets stacked. So a
    caller can align the LFP (``value_key='evoked_trace'``) by the stim command
    (``align_key='stim_trace'``) -- a sharp, unambiguous edge -- instead of the
    LFP's own noisy edge. Per trace the align channel falls back to the value
    channel when it is missing or a different length."""
    assert isinstance(traces, list), "traces must be a list"
    assert len(traces) < _MAX_TRACES, "trace count runaway"
    ak = align_key or value_key
    usable = []                                    # (tm, v, va)
    for t in traces:
        tm = t.get(time_key)
        v = t.get(value_key)
        if tm is None or v is None:
            continue
        tm = np.asarray(tm, dtype=float)
        v = np.asarray(v, dtype=float)
        if tm.size < 3 or tm.size != v.size:
            continue
        va = t.get(ak)
        va = np.asarray(va, dtype=float) if va is not None else None
        if va is None or va.size != tm.size:       # missing/ragged -> use value
            va = v
        usable.append((tm, v, va))
    if not usable:
        return None, None
    if align == "rising_edge":
        edges = [_rising_edge_time(tm, va) for tm, _v, va in usable]  # on the ref
        valid = [e for e in edges if e is not None]
        if valid:
            ref_edge = float(np.median(valid))
            shifted = [((tm - (e - ref_edge)) if e is not None else tm, v)
                       for (tm, v, _va), e in zip(usable, edges)]
        else:
            shifted = [(tm, v) for tm, v, _va in usable]
        # Shifted axes no longer share a grid -> interpolate onto the longest.
        ref_t = max((tm for tm, _ in shifted), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v in shifted])
        return ref_t, stack
    if len({tm.size for tm, _v, _va in usable}) == 1:
        ref_t = usable[0][0]
        stack = np.vstack([v for _tm, v, _va in usable])
    else:
        ref_t = max((tm for tm, _v, _va in usable), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v, _va in usable])
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
                   time_key: str = "time_ms", align=None, align_key=None
                   ) -> tuple[list | None, list | None]:
    """Mean of ``traces[i][value_key]`` over a shared ``time_key`` grid. Returns
    ``(time_ms, mean)`` as plain lists, or ``(None, None)`` when nothing usable.
    *align* -> :func:`_stack` (e.g. ``'rising_edge'``); *align_key* selects the
    reference channel for the edge (default: *value_key*)."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align,
                          align_key=align_key)
    if ref_t is None:
        return None, None
    with np.errstate(invalid="ignore"):
        y = np.nanmean(stack, axis=0)
    return ref_t.tolist(), y.tolist()


def average_with_sem(traces: list[dict], *, value_key: str = "stim_trace",
                     time_key: str = "time_ms", align=None, align_key=None):
    """Like :func:`average_traces` but also returns the per-sample standard error
    of the mean: ``(time_ms, mean, sem)`` (or ``(None, None, None)``). *align* /
    *align_key* -> :func:`_stack`. SEM is the convention for evoked traces; caller
    shades mean +/- sem."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align,
                          align_key=align_key)
    if ref_t is None:
        return None, None, None
    mean, sem = _mean_sem(stack)
    return ref_t.tolist(), mean.tolist(), sem.tolist()


def align_average_with_traces(traces: list[dict], *, value_key: str = "stim_trace",
                              time_key: str = "time_ms", align=None,
                              align_key=None):
    """``(time_ms, mean, sem, aligned_traces)`` -- the mean +/- SEM AND every
    constituent trace resampled onto the SAME (optionally rising-edge-aligned)
    grid, so a figure can overlay the individual recordings transparently under
    the bold mean. ``(None, None, None, [])`` when nothing usable. *align_key*
    selects the reference channel for the edge (default: *value_key*) -- e.g.
    align the LFP by the stim command."""
    ref_t, stack = _stack(traces, value_key, time_key, align=align,
                          align_key=align_key)
    if ref_t is None:
        return None, None, None, []
    mean, sem = _mean_sem(stack)
    return (ref_t.tolist(), mean.tolist(), sem.tolist(),
            [row.tolist() for row in stack])
