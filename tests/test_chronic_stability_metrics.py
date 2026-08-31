"""Unit tests for src/chronic_stability/metrics.py (numeric parts)."""

import numpy as np

from src.chronic_stability import metrics as M


def _win_ms(n=61, lo=-1.0, hi=2.0):
    return np.linspace(lo, hi, n)


def _biphasic(win_ms, sign=+1, amp=10.0):
    """A trace whose FIRST post-stim extremum is a peak (sign +1) or trough."""
    y = np.zeros_like(win_ms)
    post = win_ms > 0
    # first bump near t=+0.2 ms, opposite rebound later near t=+1.0 ms
    y += sign * amp * np.exp(-((win_ms - 0.2) ** 2) / (2 * 0.05 ** 2))
    y -= sign * 0.5 * amp * np.exp(-((win_ms - 1.0) ** 2) / (2 * 0.1 ** 2))
    y[~post] = 0.0
    return y


def test_polarity_sign_of_first_extremum():
    w = _win_ms()
    pos = np.vstack([_biphasic(w, +1) for _ in range(5)])
    neg = np.vstack([_biphasic(w, -1) for _ in range(5)])
    assert np.all(M.stim_polarity(pos, w) == 1)
    assert np.all(M.stim_polarity(neg, w) == -1)


def test_amplitude_peak_to_trough():
    w = _win_ms()
    seg = _biphasic(w, +1, amp=10.0)[None, :]
    amp = M.per_trial_metrics(seg, w, np.median(seg, axis=0))["amplitude"]
    assert amp[0] > 10.0            # peak (~10) minus rebound trough (<0)


def test_template_corr_and_nrmse_identity():
    w = _win_ms()
    T = _biphasic(w, +1)
    seg = np.vstack([T, T])         # identical to template
    assert np.allclose(M.template_correlation(seg, T), 1.0, atol=1e-6)
    assert np.allclose(M.normalized_rmse(seg, T), 0.0, atol=1e-9)


def test_nrmse_grows_with_deviation():
    w = _win_ms()
    T = _biphasic(w, +1)
    noisy = T + np.random.default_rng(0).normal(0, 1.0, T.size)
    n_id = M.normalized_rmse(T[None, :], T)[0]
    n_noisy = M.normalized_rmse(noisy[None, :], T)[0]
    assert n_noisy > n_id


def test_switch_count():
    assert M.switch_count(np.array([1, 1, -1, -1, 1])) == 2
    assert M.switch_count(np.array([1])) == 0


def test_build_template_requires_enough_trials():
    w = _win_ms()
    try:
        M.build_template(np.zeros((10, w.size)))
        assert False, "should have asserted on too-few trials"
    except AssertionError:
        pass
