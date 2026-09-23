"""Riding-event-vs-onset aggregation: event rate and archetype prevalence binned
by time-to-seizure-onset. Pure helpers only (no network / no store).

Run with: pytest tests/test_riding_event_onset.py -q
"""
from __future__ import annotations
import os, sys
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.riding_event_onset import rate_by_tto, archetype_by_tto  # noqa: E402


def test_rate_by_tto_bins_and_baseline():
    # pre-ictal epochs at 60s (flagged) and 400s (not); interictal (NaN): 1 of 2 flagged
    tto = np.array([60., 400., np.nan, np.nan])
    flag = np.array([1, 0, 1, 0], dtype=bool)
    cen, rate, n, base = rate_by_tto(tto, flag, window_sec=600., n_bins=2)
    assert base == 0.5                                  # interictal baseline
    assert list(n) == [1, 1]
    assert rate[0] == 1.0 and rate[1] == 0.0            # near-onset bin hotter
    assert cen[0] < cen[1]                              # minutes, ascending


def test_rate_by_tto_empty_bins_are_nan():
    tto = np.array([np.nan, np.nan])
    cen, rate, n, base = rate_by_tto(tto, np.array([1, 0], bool), 600., n_bins=3)
    assert np.isnan(rate).all() and (n == 0).all()
    assert base == 0.5


def test_archetype_by_tto_fractions_sum_to_one():
    tto = np.array([50., 55., 500., np.nan])
    dom = np.array([0, 1, 1, 0])
    cen, frac, n = archetype_by_tto(tto, dom, k=2, window_sec=600., n_bins=2)
    assert frac.shape == (2, 2)
    # near-onset bin has one of each archetype -> 50/50
    col0 = frac[:, 0]
    assert np.isfinite(col0).all() and abs(col0.sum() - 1.0) < 1e-9
    assert abs(col0[0] - 0.5) < 1e-9
