"""Null-onset control pure helpers: per-seizure fold (own time axis, horizon-
independent) + bootstrap median CI. No store/network.

Run: pytest tests/test_onset_null.py -q
"""
from __future__ import annotations
import os, sys
import numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.riding_event.onset_null import _fold_per_seizure, _boot_median_ci  # noqa: E402


def _traj(centers_min, rate):
    return {"centers_min": np.asarray(centers_min, float),
            "mf_rate": np.asarray(rate, float)}


def test_fold_uses_each_trajectory_own_axis():
    # two trajectories with DIFFERENT bin counts both yield a fold
    c1 = np.arange(-360, 60, 60.0)   # 7 bins, hourly, -6..0h
    t1 = _traj(c1, [1, 1, 1, 1, 2, 2, 2])   # baseline(-6..-4)=1, last-hour(-1..0)=2 -> fold 2
    c2 = np.arange(-360, 60, 30.0)   # 14 bins, half-hourly (different length)
    t2 = _traj(c2, np.concatenate([np.full(4, 1.0), np.full(10, 3.0)]))
    f = _fold_per_seizure([t1, t2], "mf_rate", baseline_h=(6, 4), last_h=1)
    assert f.size == 2                            # both counted despite length diff
    assert abs(f[0] - 2.0) < 1e-9


def test_fold_drops_uncovered_baseline():
    c = np.arange(-180, 60, 60.0)                 # only -3..0h -> no -6..-4 baseline
    f = _fold_per_seizure([_traj(c, [1, 1, 2, 2])], "mf_rate", baseline_h=(6, 4))
    assert f.size == 0                            # excluded (no baseline coverage)


def test_boot_median_ci():
    lo, hi = _boot_median_ci(np.array([1.0, 1.2, 1.4, 1.6, 2.0]), seed=0)
    assert lo <= np.median([1.0, 1.2, 1.4, 1.6, 2.0]) <= hi
    assert _boot_median_ci(np.array([1.0, 2.0])) == (float("nan"),) * 2 or \
        np.isnan(_boot_median_ci(np.array([1.0, 2.0]))[0])   # <3 -> nan
