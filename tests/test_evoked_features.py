"""Unit tests for the vectorized evoked-feature math
(``src/utils/evoked_features.py``), checked against synthetic signals
with known closed-form answers.

Run with: pytest tests/test_evoked_features.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.utils.evoked_features as ef  # noqa: E402

FS = 1000.0
TIME = np.linspace(-100.0, 199.0, 300)   # dt = 1.0 ms, t=0 at sample 100
DT = 1000.0 / FS


def test_check_rejects_1d():
    with pytest.raises(AssertionError):
        ef.line_length(np.arange(10.0))


def test_line_length_and_slope_on_ramp():
    ramp = np.arange(300.0)[None, :]          # diff == 1 everywhere
    assert ef.line_length(ramp)[0] == pytest.approx(299.0)
    # slope of |ramp| from peak (last sample) -> single point -> NaN guard
    assert ef.max_slope(ramp, DT)[0] == pytest.approx(1.0 / DT)


def test_peak_trough_latency():
    t = np.zeros((1, 300))
    t[0, 150] = 5.0
    t[0, 50] = -3.0
    assert ef.peak(t)[0] == 5.0
    assert ef.trough(t)[0] == -3.0
    assert ef.peak_to_trough(t)[0] == 8.0
    assert ef.peak_latency_ms(t, TIME)[0] == pytest.approx(TIME[150])
    assert ef.trough_latency_ms(t, TIME)[0] == pytest.approx(TIME[50])


def test_variance_constant_is_zero_and_ratio_guard():
    const = np.full((1, 300), 2.5)
    assert ef.variance(const)[0] == pytest.approx(0.0)
    # All energy before 0 ms -> late_area 0 -> ratio uses eps, stays finite.
    pre = np.zeros((1, 300))
    pre[0, :100] = 1.0
    r = ef.early_late_ratio(pre, TIME)[0]
    assert np.isfinite(r)


def test_spectral_band_and_moment():
    f0 = 30.0
    sig = np.sin(2 * np.pi * f0 * (TIME / 1000.0))[None, :]
    sp = ef.spectral(sig, FS)
    assert sp["sum_power_low"][0] > 10 * (sp["sum_power_high"][0] + 1e-9)
    assert sp["freq_moment_low"][0] == pytest.approx(f0, abs=5.0)


def test_autocorrelation_smooth_vs_noise():
    smooth = np.sin(2 * np.pi * 5 * (TIME / 1000.0))[None, :]
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((1, 300))
    assert ef.autocorrelation(smooth)[0] > 0.9
    assert abs(ef.autocorrelation(noise)[0]) < 0.3


def test_recovery_tau_on_decaying_oscillation():
    tau = 20.0
    t = TIME.copy()
    env = np.where(t >= 0, np.exp(-t / tau), 0.0)
    sig = (env * np.cos(2 * np.pi * 120 * (t / 1000.0)))[None, :]
    got = ef.recovery_tau(sig, TIME)[0]
    assert 12.0 < got < 35.0


def test_recovery_slope_linear_decay():
    t = TIME.copy()
    line = np.where(t >= 0, 10.0 - 0.1 * t, 0.0)[None, :]
    sl = ef.recovery_slope(line, TIME)[0]
    assert sl == pytest.approx(-0.1, abs=0.03)


def test_template_correlation_identical_epochs():
    base = np.sin(2 * np.pi * 8 * (TIME / 1000.0))
    rng = np.random.default_rng(1)
    traces = base[None, :] + 1e-3 * rng.standard_normal((15, 300))
    tc = ef.template_correlation(traces)
    assert tc[-1] == pytest.approx(1.0, abs=0.05)


def test_ac_width_finite_for_smooth():
    sig = np.sin(2 * np.pi * 5 * (TIME / 1000.0))[None, :]
    w = ef.ac_width(sig)[0]
    assert np.isfinite(w) and w > 0


def test_pca_recon_error_flags_outlier():
    base = np.sin(2 * np.pi * 6 * (TIME / 1000.0))
    rng = np.random.default_rng(2)
    traces = base[None, :] + 1e-3 * rng.standard_normal((20, 300))
    odd = (base + 5.0 * np.sin(2 * np.pi * 40 * (TIME / 1000.0)))[None, :]
    traces = np.vstack([traces, odd])
    err = ef.pca_recon_error(traces)
    assert err[-1] > err[:-1].mean()


def test_exp_fit_a_recovers_amplitude():
    t = TIME.copy()
    pre = t <= 0
    amp = 2.0
    rise = np.zeros_like(t)
    rise[pre] = amp * np.exp(0.02 * t[pre])      # rising toward t=0, peak=amp
    sig = rise[None, :]
    a = ef.exp_fit_a(sig, TIME)[0]
    assert a == pytest.approx(amp, rel=0.3)


def test_compute_all_expensive_flag():
    base = np.sin(2 * np.pi * 6 * (TIME / 1000.0))
    traces = np.tile(base, (8, 1))
    cheap = ef.compute_all(traces, TIME, FS, expensive=False)
    assert set(ef.ALL_COLUMNS) <= set(cheap)
    assert np.all(np.isnan(cheap["recovery_tau"]))      # not computed
    assert np.all(np.isfinite(cheap["line_length"]))
    rich = ef.compute_all(traces, TIME, FS, expensive=True)
    assert np.isfinite(rich["ac_width"]).any()


def test_rolling_centered():
    x = np.arange(10.0)
    m = ef.rolling_centered(x, 3, "mean")
    assert m[5] == pytest.approx(5.0)
    assert ef.rolling_centered(np.ones(5), 3, "cv")[2] == pytest.approx(0.0) or \
        np.isnan(ef.rolling_centered(np.ones(5), 3, "cv")[2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
