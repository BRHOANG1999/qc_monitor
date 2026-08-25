"""Average a set of per-file waveform traces onto a common time grid.

The weekly stim figures need a daily / weekly *average* stimulus waveform from
many per-file mean traces. Traces of one channel at one config share a time grid
in practice, but nothing guarantees it, so this handles both: a fast
equal-length ``nanmean`` path and an ``np.interp`` fallback onto a reference grid
when lengths differ. Pure (numpy only) so it is trivially unit-testable and
carries no matplotlib/Dash import weight.
"""

from __future__ import annotations

import warnings

import numpy as np

_MAX_TRACES = 100_000       # NASA Rule 2 bound


def _stack(traces: list[dict], value_key: str, time_key: str):
    """Stack traces onto one time grid -> ``(ref_t, stack)`` or ``(None, None)``.
    Equal-length traces stack directly; otherwise each is interpolated onto the
    longest trace's (ascending) axis."""
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
    if len({tm.size for tm, _ in usable}) == 1:
        ref_t = usable[0][0]
        stack = np.vstack([v for _, v in usable])
    else:
        ref_t = max((tm for tm, _ in usable), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v in usable])
    return ref_t, stack


def average_traces(traces: list[dict], *, value_key: str = "stim_trace",
                   time_key: str = "time_ms") -> tuple[list | None, list | None]:
    """Mean of ``traces[i][value_key]`` over a shared ``time_key`` grid. Returns
    ``(time_ms, mean)`` as plain lists, or ``(None, None)`` when nothing usable."""
    ref_t, stack = _stack(traces, value_key, time_key)
    if ref_t is None:
        return None, None
    with np.errstate(invalid="ignore"):
        y = np.nanmean(stack, axis=0)
    return ref_t.tolist(), y.tolist()


def average_with_sem(traces: list[dict], *, value_key: str = "stim_trace",
                     time_key: str = "time_ms"):
    """Like :func:`average_traces` but also returns the per-sample standard error
    of the mean: ``(time_ms, mean, sem)`` (or ``(None, None, None)``). SEM =
    ``std(ddof=1) / sqrt(n_valid)``; 0 where a sample has < 2 finite traces so
    the band collapses instead of NaN-ing. SEM is the convention for evoked
    traces; the caller shades mean +/- sem."""
    ref_t, stack = _stack(traces, value_key, time_key)
    if ref_t is None:
        return None, None, None
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # ddof=1, n=1
        mean = np.nanmean(stack, axis=0)
        n = np.sum(np.isfinite(stack), axis=0)
        std = np.nanstd(stack, axis=0, ddof=1)
        sem = np.where(n >= 2, std / np.sqrt(np.maximum(n, 1)), 0.0)
    return ref_t.tolist(), mean.tolist(), np.nan_to_num(sem).tolist()
