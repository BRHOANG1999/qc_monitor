"""Average a set of per-file waveform traces onto a common time grid.

The weekly stim figures need a daily / weekly *average* stimulus waveform from
many per-file mean traces. Traces of one channel at one config share a time grid
in practice, but nothing guarantees it, so this handles both: a fast
equal-length ``nanmean`` path and an ``np.interp`` fallback onto a reference grid
when lengths differ. Pure (numpy only) so it is trivially unit-testable and
carries no matplotlib/Dash import weight.
"""

from __future__ import annotations

import numpy as np

_MAX_TRACES = 100_000       # NASA Rule 2 bound


def average_traces(traces: list[dict], *, value_key: str = "stim_trace",
                   time_key: str = "time_ms") -> tuple[list | None, list | None]:
    """Mean of ``traces[i][value_key]`` over a shared ``time_key`` grid.

    Each trace is a dict with a time axis (``time_key``) and a value array
    (``value_key``) of equal length. Returns ``(time_ms, y)`` as plain lists, or
    ``(None, None)`` when nothing is usable. When lengths differ, every trace is
    linearly interpolated onto the longest trace's grid before averaging;
    equal-length traces take the cheap stack-and-``nanmean`` path. NaNs are
    ignored per sample."""
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
    lengths = {tm.size for tm, _ in usable}
    if len(lengths) == 1:
        ref_t = usable[0][0]
        stack = np.vstack([v for _, v in usable])
    else:
        # Reference grid = the longest trace's axis (most resolution); interp the
        # rest onto it. Requires each source axis be ascending, which time_ms is.
        ref_t = max((tm for tm, _ in usable), key=lambda a: a.size)
        stack = np.vstack([np.interp(ref_t, tm, v) for tm, v in usable])
    with np.errstate(invalid="ignore"):
        y = np.nanmean(stack, axis=0)
    return ref_t.tolist(), y.tolist()
