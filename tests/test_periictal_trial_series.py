"""Across-trial feature-series substrate (Part B, B0).

Covers the pure builders: signed nearest-onset lead time, gap flagging from the
measured dt (no hardcoded rate), and segmentation at gaps (so AR/spectral fits
never straddle a file-boundary hole).

Run: pytest tests/test_periictal_trial_series.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import trial_series as ts       # noqa: E402


def test_nearest_onset_delta_sign_and_choice():
    secs = np.array([50.0, 120.0, 250.0])
    onsets = np.array([100.0, 300.0])
    tto = ts.nearest_onset_delta(secs, onsets)
    # 50 -> nearest onset 100 (upcoming, +50); 120 -> nearest 100 (past, -20);
    # 250 -> nearest 300 (upcoming, +50)
    assert np.allclose(tto, [50.0, -20.0, 50.0])


def test_nearest_onset_delta_no_onsets_is_nan():
    tto = ts.nearest_onset_delta(np.array([1.0, 2.0]), np.array([]))
    assert np.all(np.isnan(tto))


def test_gap_flagging_uses_measured_dt():
    secs = np.array([0.0, 2.0, 4.0, 6.0, 1000.0, 1002.0])
    vals = np.arange(6.0)
    s = ts.assemble_trial_series("A", "feat", "ch", secs, vals, np.array([500.0]))
    assert s.dt_med == 2.0
    # only the 994 s jump is a gap (threshold = max(120, 5*2))
    assert list(s.gap_after) == [False, False, False, True, False, False]


def test_segments_split_at_gaps():
    secs = np.array([0.0, 2.0, 4.0, 6.0, 1000.0, 1002.0])
    vals = np.arange(6.0)
    s = ts.assemble_trial_series("A", "feat", "ch", secs, vals, np.array([]))
    segs = s.segments()
    assert [(sl.start, sl.stop) for sl in segs] == [(0, 4), (4, 6)]
    assert s.n == 6


def test_time_to_onset_populated_and_signed():
    secs = np.linspace(0.0, 100.0, 11)
    s = ts.assemble_trial_series("A", "feat", "ch", secs, secs.copy(),
                                 np.array([60.0]))
    # before onset -> positive, after -> negative, monotone decreasing
    assert s.time_to_onset[0] > 0 and s.time_to_onset[-1] < 0
    assert np.all(np.diff(s.time_to_onset) < 0)
