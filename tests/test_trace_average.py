"""trace_average: rising-edge alignment + the mean-with-constituent-traces helper
that backs the transparent-overlay stim figures.

Run: pytest tests/test_trace_average.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.trace_average import (            # noqa: E402
    average_with_sem, align_average_with_traces)

_T = [round(-0.5 + 0.05 * i, 3) for i in range(30)]     # -0.5 .. 0.95 ms


def _step(edge_t: float) -> dict:
    """A 0->5 step whose rising edge is at *edge_t* (value 'v')."""
    return {"time_ms": _T, "v": [5.0 if x >= edge_t else 0.0 for x in _T]}


def test_align_average_with_traces_shapes():
    t = [0.0, 0.1, 0.2, 0.3, 0.4]
    traces = [{"time_ms": t, "v": [0, 1, 2, 1, 0]},
              {"time_ms": t, "v": [0, 1, 2, 1, 0]},
              {"time_ms": t, "v": [0, 3, 4, 3, 0]}]
    tm, mean, sem, tr = align_average_with_traces(traces, value_key="v")
    assert len(tm) == 5 and len(mean) == 5 and len(sem) == 5
    assert len(tr) == 3 and all(len(x) == 5 for x in tr)   # constituent traces
    assert mean[2] == 8 / 3                                  # (2+2+4)/3


def test_empty_returns_nones():
    assert align_average_with_traces([], value_key="v") == (None, None, None, [])


def test_rising_edge_alignment_sharpens_the_mean():
    # Two identical steps offset by 3 samples in time. WITHOUT alignment the mean
    # is a 2-step staircase (edge smeared); WITH rising-edge alignment both edges
    # coincide so the mean's edge is as steep as an individual step.
    a, b = _step(0.0), _step(0.15)
    _t, mu, _s = average_with_sem([a, b], value_key="v")
    _t, ma, _s = average_with_sem([a, b], value_key="v", align="rising_edge")
    assert max(abs(np.diff(mu))) < max(abs(np.diff(ma)))    # aligned is sharper


def test_alignment_noop_when_already_aligned():
    # Same edge in both -> alignment must not change the mean.
    a, b = _step(0.1), _step(0.1)
    _t, mu, _s = average_with_sem([a, b], value_key="v")
    _t, ma, _s = average_with_sem([a, b], value_key="v", align="rising_edge")
    assert np.allclose(mu, ma, atol=1e-9)
