"""Unit tests for src/chronic_stability/ra_flatrun.py."""

import numpy as np
import pytest

from src.chronic_stability import ra_flatrun as fr


def _series():
    """Flat plateau (0..100 h), then a drifting ramp (100..200 h). Hourly."""
    t = np.arange(0.0, 200.0, 1.0)
    r = np.where(t < 100.0, 0.13, 0.13 + 0.01 * (t - 100.0))
    rng = np.random.default_rng(0)
    r = r + rng.normal(0, 0.004, r.size)     # tiny wiggle on the plateau
    return t, r


def test_detects_flat_plateau_not_the_ramp():
    t, r = _series()
    run = fr.detect_longest_flat_run(t, r, t * 3600.0)
    assert run["found"]
    # the flat run must live in the first (plateau) half, not the ramp
    assert t[run["idx_start"]] < 20.0
    assert t[run["idx_end"]] <= 105.0
    assert run["duration_h"] > 80.0


def test_global_scale_not_local_median():
    """A low-magnitude flat window must not be penalised: halving the plateau
    level must not shrink the detected run."""
    t, r = _series()
    run_hi = fr.detect_longest_flat_run(t, r, t * 3600.0)
    r2 = r.copy()
    r2[t < 100.0] *= 0.25                      # much smaller magnitude, same flat
    run_lo = fr.detect_longest_flat_run(t, r2, t * 3600.0)
    assert run_lo["found"]
    assert run_lo["duration_h"] >= 0.8 * run_hi["duration_h"]


def test_large_gap_breaks_run():
    t = np.concatenate([np.arange(0, 50.0), np.arange(60.0, 110.0)])  # 10h gap
    r = np.full(t.size, 0.13) + np.random.default_rng(1).normal(0, 0.003, t.size)
    run = fr.detect_longest_flat_run(t, r, t * 3600.0, gap_tol_h=6.0)
    assert run["found"]
    # the run cannot straddle the 10 h gap at t=50->60
    assert not (t[run["idx_start"]] < 50.0 and t[run["idx_end"]] > 60.0)


def test_lone_spike_tolerated_adjacent_spikes_break():
    t = np.arange(0.0, 120.0)
    base = np.full(t.size, 0.13) + np.random.default_rng(2).normal(0, 0.002, t.size)
    lone = base.copy()
    lone[60] += 0.6                              # single spike inside the plateau
    run_lone = fr.detect_longest_flat_run(t, lone, t * 3600.0)
    assert run_lone["found"] and run_lone["duration_h"] > 100.0
    assert 60 in run_lone["flagged_spikes_in_run"]
    adj = base.copy()
    adj[60] += 0.6
    adj[61] += 0.6                               # two adjacent spikes -> break
    run_adj = fr.detect_longest_flat_run(t, adj, t * 3600.0)
    assert run_adj["duration_h"] < run_lone["duration_h"]


def test_validation_iou_and_deltas():
    t, r = _series()
    run = fr.detect_longest_flat_run(t, r, t * 3600.0)
    v = fr.validate_against_window(run, t * 3600.0, 0.0, 100.0 * 3600.0)
    assert 0.0 <= v["iou"] <= 1.0
    assert v["pct_det_covered"] > 90.0
    assert "verdict_pass" in v


def test_robust_scale_floored_off_zero():
    assert fr.robust_scale(np.zeros(10)) > 0.0
