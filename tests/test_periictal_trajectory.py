"""Peri-ictal lead-time trajectory: median (+ IQR) per dyadic log-lead-time bin,
per-seizure medians, low-n flagging, and monotone-ramp sanity. This is the
count-agnostic view that fixes the near-onset-sparse / far-onset-abundant
imbalance (one summary per bin = equal weight per time-scale).

Run with: pytest tests/test_periictal_trajectory.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal.trajectory import (default_edges,  # noqa: E402
                                      lead_time_trajectory)


def test_bins_median_iqr_and_counts():
    edges = np.array([2., 4., 8., 16.])
    tto = np.array([3, 3.5, 5, 6, 7, 10, 12, 14, 15.])   # bins: 2, 3, 4 points
    vals = np.array([1, 3, 10, 20, 30, 100, 100, 100, 100.])
    sz = np.zeros(len(tto), int)
    tr = lead_time_trajectory(tto, vals, sz, edges, min_n=3)
    assert list(tr["n"]) == [2, 3, 4]
    assert tr["median"][1] == 20                          # median of 10,20,30
    assert tr["p25"][1] == 15 and tr["p75"][1] == 25
    assert tr["low_n"][0] and not tr["low_n"][1]          # bin0 n=2 < min_n 3
    assert np.allclose(tr["centers"], np.sqrt(edges[:-1] * edges[1:]))


def test_per_seizure_medians_are_separate():
    edges = np.array([2., 4., 8.])
    tto = np.array([3, 3, 5, 5.])
    vals = np.array([1, 2, 10, 20.])
    sz = np.array([0, 1, 0, 1])
    tr = lead_time_trajectory(tto, vals, sz, edges)
    assert set(tr["per_seizure"]) == {0, 1}
    assert tr["per_seizure"][0][0] == 1 and tr["per_seizure"][1][0] == 2


def test_monotone_ramp_shows_monotone_trajectory():
    # Inject a ramp that RISES toward onset (larger value at smaller lead-time);
    # the binned median must decrease from near-onset (centers[0]) to far.
    edges = default_edges(3600.0)
    rng = np.random.default_rng(0)
    tto = rng.uniform(2, 3600, 5000)
    vals = -np.log10(tto) + rng.normal(0, 0.01, tto.size)
    sz = rng.integers(0, 4, tto.size)
    tr = lead_time_trajectory(tto, vals, sz, edges)
    med = tr["median"][np.isfinite(tr["median"])]
    assert med[0] > med[-1]                                # near-onset > far


def test_nan_values_are_dropped_not_counted():
    edges = np.array([2., 4., 8.])
    tr = lead_time_trajectory(np.array([5., 5.]), np.array([np.nan, 4.0]),
                              np.array([0, 0]), edges)
    assert tr["n"][1] == 1 and tr["median"][1] == 4.0


def test_length_mismatch_guard():
    import pytest
    with pytest.raises(AssertionError):
        lead_time_trajectory(np.array([3., 3.]), np.array([1.]),
                             np.array([0, 0]), np.array([2., 4.]))


def test_default_edges_are_dyadic_ascending():
    e = default_edges(21600.0, 2.0)
    assert e[0] == 2.0 and e[-1] == 21600.0
    assert np.all(np.diff(e) > 0)                          # strictly ascending


def test_linear_edges_are_equal_width_from_zero():
    from src.periictal.trajectory import linear_edges
    e = linear_edges(7200.0, 12)
    assert e[0] == 0.0 and e[-1] == 7200.0 and len(e) == 13
    widths = np.diff(e)
    assert np.allclose(widths, widths[0])                  # equal-width bins
    assert np.allclose(widths[0], 600.0)                   # 7200/12 = 10 min each


def test_arithmetic_centers_when_not_log():
    # linear ladder from 0 -> geometric centre of the first bin would be 0
    # (sqrt(0*x)); arithmetic centres avoid that and sit mid-bin.
    from src.periictal.trajectory import linear_edges
    e = linear_edges(1200.0, 4)                            # 0,300,600,900,1200
    tto = np.array([150., 450., 750., 1050.])
    tr = lead_time_trajectory(tto, np.array([1., 2, 3, 4]),
                              np.array([0, 0, 1, 1]), e, log_centers=False)
    assert np.allclose(tr["centers"], [150., 450., 750., 1050.])   # mid-bin
    # geometric (default) would put the first centre at 0 — the linear bug we fix
    tr_log = lead_time_trajectory(tto, np.array([1., 2, 3, 4]),
                                  np.array([0, 0, 1, 1]), e)
    assert tr_log["centers"][0] == 0.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
