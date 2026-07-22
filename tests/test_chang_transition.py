"""Transition-point detector + the v4 feature definitions.

Regression cover for the reconstruction defects that a figure of the fits drawn
on real traces exposed:

  * the transition was the GLOBAL minimum of |dy/dt| after the peak, which is
    biased to the settled tail by construction -- on the trial-averaged trace
    (the cleanest signal available) it landed at 196.55 ms of a 200 ms window,
    giving a 189 ms "fast" segment and a 3.5 ms "slow" sliver;
  * the peak was argmax|y| over the whole window, so 50% of real epochs
    "peaked" after 50 ms and 34% after 100 ms;
  * curvature summed |dy/dt| without dividing by sample count, so it tracked
    segment length (rho = +0.78 / +0.82 on real data);
  * a positive decay constant -- a growing exponential -- was accepted as a
    "recovery rate" in 8.5% of epochs.

None of this was caught because nothing tested the detector on a trace with a
KNOWN breakpoint.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.evoked_features as ef  # noqa: E402

FS = 20000.0
T = np.linspace(-200.0, 200.0, 8001)          # ms, t=0 = stim
_RES = T[1] - T[0]


def _exp_then_flat(A=5.0, B=-0.08, base=0.5, t0=1.0, brk=60.0):
    y = np.full(T.size, base, dtype=float)
    seg = (T >= t0) & (T < brk)
    y[seg] += A * np.exp(B * (T[seg] - t0))
    return y[None, :]


def _exp_then_line(A=3.0, B=-0.15, m=0.01, c=0.3, t0=1.0, brk=40.0):
    y = m * T + c
    seg = (T >= t0) & (T < brk)
    y[seg] += A * np.exp(B * (T[seg] - t0))
    return y[None, :]


# ------------------------------------------------- breakpoint recovery --- #

def test_finds_known_breakpoint_exp_then_flat():
    out = ef.compute_chang(_exp_then_flat(brk=60.0), T, FS)
    assert out["tp_latency_ms"][0] == pytest.approx(60.0, abs=2.0)


def test_finds_known_breakpoint_exp_then_sloped_line():
    """The slow component's asymptote is a LINE. Detrending with a constant
    instead collapsed the boundary to the start of the window."""
    out = ef.compute_chang(_exp_then_line(brk=40.0), T, FS)
    assert out["tp_latency_ms"][0] == pytest.approx(40.0, abs=2.0)


def test_breakpoint_tracks_where_the_break_actually_is():
    seen = [ef.compute_chang(_exp_then_flat(brk=b), T, FS)["tp_latency_ms"][0]
            for b in (30.0, 60.0, 90.0)]
    assert seen == sorted(seen), seen
    for got, want in zip(seen, (30.0, 60.0, 90.0)):
        assert got == pytest.approx(want, abs=3.0)


# ------------------------------------------------------- the tail bias --- #

def test_transition_is_not_pushed_to_the_window_end():
    """THE incident: a trace that settles early must not split at the tail."""
    tp = ef.compute_chang(_exp_then_flat(brk=40.0), T, FS)["tp_latency_ms"][0]
    assert tp < 150.0, f"transition {tp} ms is back in the settled tail"


def test_a_settled_trace_leaves_a_substantial_slow_segment():
    _, trans, post, valid = ef._transition_indices(_exp_then_flat(brk=40.0),
                                                   T, FS)
    hi = int(T.size - np.argmax(post[::-1]))
    assert bool(valid[0])
    assert (hi - trans[0]) * _RES > 50.0, "slow segment collapsed to a sliver"


# ------------------------------------------------------ peak detection --- #

def test_peak_search_ignores_a_larger_late_hump():
    """argmax over the whole window let a slow late hump win the 'fast' peak."""
    y = _exp_then_flat(A=1.0, brk=30.0)[0]
    late = (T >= 120.0) & (T <= 160.0)
    y[late] += 10.0                              # far larger than the response
    pk, _, _, _ = ef._transition_indices(y[None, :], T, FS)
    assert T[pk[0]] <= ef._CHANG_PEAK_MAX_MS


def test_peak_is_refined_on_the_raw_trace():
    """Smoothing delays an abrupt onset by ~half a window, biasing `a` low."""
    out = ef.compute_chang(_exp_then_flat(A=5.0, B=-0.08), T, FS)
    assert out["expfit_initial"][0] == pytest.approx(5.0, abs=1e-2)


# ------------------------------------------- sample-rate independence --- #

def test_min_segment_is_a_duration_not_a_sample_count():
    """Was a flat 5 samples = 0.25 ms at 20 kHz, so it validated pure noise.
    (A small absolute floor still applies at very low rates.)"""
    lo = ef._min_seg_samples(5000.0)
    hi = ef._min_seg_samples(20000.0)
    assert hi == pytest.approx(4 * lo, rel=0.2), (lo, hi)
    for fs in (5000.0, 20000.0):
        assert ef._min_seg_samples(fs) * (1000.0 / fs) >= 0.9


# --------------------------------------------------- failed-fit reject --- #

def test_growing_exponential_is_rejected():
    """b > 0 is a failed fit, not a recovery rate (the column says 'b < 0')."""
    y = np.full(T.size, 0.5)
    seg = (T >= 1.0) & (T < 60.0)
    y[seg] += 0.05 * np.exp(+0.05 * (T[seg] - 1.0))     # GROWS
    out = ef.compute_chang(y[None, :], T, FS)
    if np.isfinite(out["expfit_decay"][0]):
        assert out["expfit_decay"][0] < 0.0
    for k in ("expfit_decay", "expfit_initial", "expfit_rms",
              "expfit_curvature", "expfit_area"):
        assert np.isnan(out[k][0]) or out["expfit_decay"][0] < 0


def test_rejecting_the_exponential_keeps_the_linear_group():
    """A failed exp fit must not void the transition or the slow-segment line,
    which are computed independently."""
    y = np.full(T.size, 0.5)
    seg = (T >= 1.0) & (T < 60.0)
    y[seg] += 0.05 * np.exp(+0.05 * (T[seg] - 1.0))
    out = ef.compute_chang(y[None, :], T, FS)
    if np.isnan(out["expfit_decay"][0]):
        assert np.isfinite(out["tp_latency_ms"][0])
        assert np.isfinite(out["linfit_slope"][0])


# ------------------------------------------- curvature length-invariance --- #

def test_curvature_does_not_track_segment_length():
    """Same waveform SHAPE, different break -> curvature must stay comparable.
    The Sigma form grew with sample count (rho +0.78 with length on real data)."""
    vals = [ef.compute_chang(_exp_then_flat(brk=b), T, FS)["linfit_curvature"][0]
            for b in (30.0, 60.0, 90.0)]
    vals = [v for v in vals if np.isfinite(v)]
    assert len(vals) >= 2
    assert max(vals) / max(min(vals), 1e-12) < 10.0, vals


# ---------------------------------------------------------- baseline --- #

def test_slow_baseline_is_a_median_not_a_mean():
    """The comment always said median; the code took a mean, which one
    artifact sample drags off the settled level."""
    y = _exp_then_flat(brk=40.0)
    y[0, (T >= 100.0) & (T <= 104.0)] += 5.0    # ~2.5% of the slow segment
    out = ef.compute_chang(y, T, FS)
    # With a MEAN baseline these samples drag `base` from 0.5 to ~0.63, giving
    # a ~= 4.87. The median is unmoved, so `a` is still exactly 5.
    assert out["expfit_initial"][0] == pytest.approx(5.0, rel=1e-3)


def test_split_is_sensitive_to_outliers_in_the_slow_segment():
    """KNOWN LIMITATION, pinned so a change is deliberate.

    The split scores each side by unexplained VARIANCE, which rewards a segment
    that contains high-variance content: a spike in the slow segment makes a
    late split look bad (the line cannot explain the spike, and the segment's
    own variance is small), so the boundary collapses early. Feature values
    still recover, but tp_latency_ms is not trustworthy on spiky epochs.
    """
    y = _exp_then_flat(brk=40.0)
    y[0, (T >= 100.0) & (T <= 102.0)] += 1.0
    tp = ef.compute_chang(y, T, FS)["tp_latency_ms"][0]
    assert tp < 40.0, ("outlier sensitivity has changed -- if this is now "
                       f"robust (tp={tp}), update the limitation note")


# ------------------------------------------------------------ shapes --- #

def test_all_epochs_processed_and_shapes_match():
    stack = np.vstack([_exp_then_flat(brk=40.0), _exp_then_line(brk=40.0),
                       _exp_then_flat(brk=80.0)])
    out = ef.compute_chang(stack, T, FS)
    for k, v in out.items():
        assert v.shape == (3,), (k, v.shape)


def test_short_window_returns_nan_without_raising():
    t = np.linspace(-1.0, 1.2, 40)
    out = ef.compute_chang(np.zeros((2, 40)), t, FS)
    assert np.isnan(out["tp_latency_ms"]).all()


# ------------------------------------------- sliding trial average (v4) --- #

def test_trial_average_preserves_one_row_per_stimulus():
    """Row count must be unchanged: each row still carries its own stimulus
    time, and the peri-ictal join to seizure onsets depends on that 1:1 map."""
    x = np.random.default_rng(0).normal(size=(17, 400))
    assert ef.trial_moving_average(x, 5).shape == x.shape


def test_trial_average_is_centred_and_shrinks_at_the_edges():
    x = np.arange(7, dtype=float)[:, None] * np.ones((1, 3))
    got = ef.trial_moving_average(x, 5)[:, 0]
    assert got[3] == pytest.approx(3.0)          # centred: mean(1..5)
    assert got[0] == pytest.approx(1.0)          # shrunk: mean(0,1,2)
    assert got[-1] == pytest.approx(5.0)         # shrunk: mean(4,5,6)


def test_trial_average_reduces_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 50))
    raw = float(np.std(x))
    avg = float(np.std(ef.trial_moving_average(x, 5)))
    assert avg < raw / 1.8, (raw, avg)           # ~sqrt(5) ≈ 2.24


def test_trial_average_is_a_no_op_when_disabled():
    x = np.random.default_rng(2).normal(size=(6, 20))
    assert np.allclose(ef.trial_moving_average(x, 1), x)


def test_trial_average_handles_fewer_trials_than_the_window():
    x = np.random.default_rng(3).normal(size=(2, 20))
    out = ef.trial_moving_average(x, 5)
    assert out.shape == x.shape
    assert np.allclose(out[0], x.mean(axis=0))   # window clamps to what exists


def test_trial_average_does_not_mutate_the_input():
    x = np.random.default_rng(4).normal(size=(9, 30))
    before = x.copy()
    ef.trial_moving_average(x, 5)
    assert np.array_equal(x, before)
