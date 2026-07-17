"""Chang et al. 2026 evoked-feature reconstruction (Table 1 / Table 2).

These lock in the closed-form fit recovery on synthetic traces with known
morphology, the spectral additions, and the schema/doc invariants. The exact
transition/fit algorithm is a documented reconstruction (paper SI unavailable),
so the tests assert on parameter recovery, not byte-exact paper values.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.evoked_features as ef  # noqa: E402


FS = 20000.0
T = np.linspace(-200.0, 200.0, 8001)     # ms, t=0 = stim


def _flat_slow_exp_fast(A, B, base, tpeak=1.0, ttrans=60.0):
    """base (flat slow) + A*exp(B*(t-tpeak)) on [tpeak, ttrans)."""
    y = np.full(T.size, base, dtype=float)
    seg = (T >= tpeak) & (T < ttrans)
    y[seg] += A * np.exp(B * (T[seg] - tpeak))
    return y[None, :]


def test_exp_fast_fit_recovers_a_and_b():
    out = ef.compute_chang(_flat_slow_exp_fast(5.0, -0.08, 0.5), T, FS)
    assert np.isclose(out["expfit_initial"][0], 5.0, atol=1e-3)
    assert np.isclose(out["expfit_decay"][0], -0.08, atol=1e-3)
    # Flat slow segment -> ~zero slope, intercept == baseline.
    assert abs(out["linfit_slope"][0]) < 1e-4
    assert np.isclose(out["linfit_intercept"][0], 0.5, atol=1e-3)
    # Transition sits at the fast->flat boundary.
    assert 55.0 <= out["tp_latency_ms"][0] <= 65.0


def test_linear_slow_fit_recovers_slope_intercept():
    m, c = 0.01, 0.3
    y = m * T + c
    seg = (T >= 1.0) & (T < 40.0)
    y[seg] += 3.0 * np.exp(-0.15 * (T[seg] - 1.0))
    out = ef.compute_chang(y[None, :], T, FS)
    assert np.isclose(out["linfit_slope"][0], m, atol=2e-3)
    assert np.isclose(out["linfit_intercept"][0], c, atol=2e-2)


def test_spectral_adds_all_three_bands():
    # Pure 128 Hz tone -> power concentrated in the 64-256 (mid) band.
    y = np.sin(2 * np.pi * 128.0 * (T * 1e-3))[None, :]
    s = ef.spectral(y, FS)
    for k in ("sum_power_low", "sum_power_mid", "sum_power_high",
              "freq_moment_low", "freq_moment_high", "freq_moment_vhigh"):
        assert k in s
    assert s["sum_power_mid"][0] > s["sum_power_low"][0]
    assert s["sum_power_mid"][0] > s["sum_power_high"][0]
    # Mid-band centroid lands near 128 Hz.
    assert 100.0 <= s["freq_moment_high"][0] <= 160.0


def test_curvature_and_skewness_signs():
    # Right-skewed: a positive spike on a flat trace.
    y = np.zeros(T.size); y[4000] = 10.0
    assert ef.skewness(y[None, :])[0] > 1.0
    # A wigglier trace has larger curvature than a smooth ramp.
    ramp = np.linspace(0, 1, T.size)[None, :]
    noisy = ramp + 0.2 * np.sin(2 * np.pi * 200 * T * 1e-3)[None, :]
    dt = 1000.0 / FS
    assert ef.curvature(noisy, dt)[0] > ef.curvature(ramp, dt)[0]


def test_short_window_yields_nan_not_crash():
    # Too few post-stim samples for two segments -> all-NaN, no exception.
    t = np.linspace(-1.0, 1.0, 41)
    out = ef.compute_chang(np.random.randn(3, 41), t, FS)
    for k in ("tp_latency_ms", "expfit_decay", "linfit_slope"):
        assert np.all(np.isnan(out[k]))


def test_schema_and_docs_complete():
    t = np.linspace(-200, 200, 8001)
    tr = np.random.RandomState(1).randn(8, 8001) * 0.5
    for expensive in (False, True):
        d = ef.compute_all(tr, t, FS, expensive=expensive)
        assert set(d.keys()) == set(ef.ALL_COLUMNS)
        assert all(len(v) == 8 for v in d.values())
    assert set(ef.ALL_COLUMNS) <= set(ef.COLUMN_DOCS.keys())


def test_new_columns_present_in_schema():
    for c in ("sum_power_mid", "freq_moment_vhigh", "curvature", "skewness",
              "tp_latency_ms", "tp_amplitude", "expfit_decay", "expfit_initial",
              "expfit_rms", "expfit_curvature", "expfit_skew", "expfit_area",
              "linfit_slope", "linfit_intercept", "linfit_rms",
              "linfit_curvature", "linfit_skew"):
        assert c in ef.CHEAP_COLUMNS
    for c in ("autocorr_low", "autocorr_mid", "autocorr_high"):
        assert c in ef.EXPENSIVE_COLUMNS
