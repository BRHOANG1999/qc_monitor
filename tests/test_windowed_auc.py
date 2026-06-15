"""Tests for windowed_auc -- the sustained-vs-transient discriminator.

Run with: pytest tests/test_windowed_auc.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.hilbert_envelope import windowed_auc  # noqa: E402


def test_same_length_and_nonnegative():
    env = np.abs(np.random.randn(500))
    auc = windowed_auc(env, 100.0, 1.0)
    assert auc.shape == env.shape
    assert np.all(auc >= 0)


def test_empty():
    assert windowed_auc(np.zeros(0), 100.0, 1.0).shape == (0,)


def test_sustained_hump_beats_tall_transient_spike():
    fs, n = 100.0, 2000
    env = np.zeros(n)
    env[500:1000] = 0.02       # 5 s sustained low hump
    env[1500] = 0.5            # one tall but momentary spike
    auc = windowed_auc(env, fs, window_sec=2.0)
    # The low sustained hump out-scores the 25x-taller lone spike.
    assert auc[750] > auc[1500]


def test_window_scales_area():
    fs, n = 100.0, 1000
    env = np.ones(n) * 0.1     # flat
    a1 = windowed_auc(env, fs, 1.0)
    a2 = windowed_auc(env, fs, 2.0)
    # A wider window integrates more area on a flat signal (away
    # from the edges).
    assert a2[500] > a1[500]


def test_auc_event_count_hump_vs_spike():
    from src.utils.mass_analyze import auc_event_count
    fs, n = 100.0, 3000
    hump = np.zeros(n)
    hump[1000:1500] = 0.02         # 5 s sustained hump
    spike = np.zeros(n)
    spike[2000] = 0.5              # one tall but momentary spike
    # A threshold between the two integrated areas keeps the hump
    # and drops the spike.
    assert auc_event_count(hump, fs, 5.0, 0.03) == 1
    assert auc_event_count(spike, fs, 5.0, 0.03) == 0


def test_auc_event_count_empty():
    from src.utils.mass_analyze import auc_event_count
    assert auc_event_count(np.zeros(0), 100.0, 5.0, 0.03) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
