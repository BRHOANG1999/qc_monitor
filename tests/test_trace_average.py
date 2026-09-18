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


def _two_ch(stim_edge: float, ev_edge: float) -> dict:
    """A trace carrying a 'stim' step (edge at *stim_edge*) and an 'evoked' step
    (edge at *ev_edge*) on the shared grid."""
    return {"time_ms": _T,
            "stim": [5.0 if x >= stim_edge else 0.0 for x in _T],
            "evoked": [5.0 if x >= ev_edge else 0.0 for x in _T]}


def test_align_key_uses_reference_channel():
    # Both evoked edges coincide (0.1); the stim edges differ (0.0 vs 0.2).
    a, b = _two_ch(0.0, 0.1), _two_ch(0.2, 0.1)
    # Aligning ON THE EVOKED (its own edges already coincide) -> sharp mean.
    _t, m_ev = average_with_sem([a, b], value_key="evoked", align="rising_edge")[:2]
    # Aligning ON THE STIM (edges differ -> shifts the evoked apart) -> blurred.
    _t, m_stim = average_with_sem([a, b], value_key="evoked", align_key="stim",
                                  align="rising_edge")[:2]
    assert max(abs(np.diff(m_ev))) > max(abs(np.diff(m_stim)))   # value edge sharper


def test_align_key_none_matches_value_key():
    a, b = _two_ch(0.0, 0.1), _two_ch(0.2, 0.15)
    _t, m0, _s = average_with_sem([a, b], value_key="evoked", align="rising_edge")
    _t, m1, _s = average_with_sem([a, b], value_key="evoked", align_key="evoked",
                                  align="rising_edge")
    assert np.allclose(m0, m1)                                   # default == self-ref


def test_align_key_missing_falls_back_to_value():
    # A trace with no 'stim' channel must not crash; it aligns on 'evoked'.
    a = _two_ch(0.0, 0.1)
    b = {"time_ms": _T, "evoked": [5.0 if x >= 0.1 else 0.0 for x in _T]}  # no stim
    _t, m, _s = average_with_sem([a, b], value_key="evoked", align_key="stim",
                                 align="rising_edge")
    assert m is not None and len(m) == len(_T)


def test_rising_edge_picks_onset_not_recovery():
    # A BIPHASIC stim artifact: onset rise (+5), then after the trough a STEEPER
    # recovery rise (+8). A plain argmax of the slope would lock onto the recovery
    # -- the bug that shifted ~12% of epochs ~0.2 ms off. The detector must return
    # the ONSET, which precedes the dominant (trough) excursion.
    from src.utils.trace_average import _rising_edge_time
    i0 = _T.index(0.0)
    v = [0.0] * len(_T)
    v[i0] = 5.0            # onset peak (rise 0->5 from the prior sample)
    v[i0 + 1] = 1.0
    v[i0 + 2] = -8.0       # trough = the dominant |v| excursion
    v[i0 + 3] = 0.0        # recovery (rise -8->0, steeper than the onset)
    edge = _rising_edge_time(np.array(_T, float), np.array(v, float))
    assert edge is not None
    assert edge < 0.05     # onset foot (~ -0.05), NOT the +0.10 recovery foot
