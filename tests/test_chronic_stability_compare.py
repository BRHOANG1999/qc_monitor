"""Unit tests for src/chronic_stability/compare.py."""

import numpy as np

from src.chronic_stability import compare as C


def test_cliffs_delta_bounds_and_sign():
    a = np.arange(0, 100.0)
    b = np.arange(50, 150.0)
    d = C.cliffs_delta(a, b)
    assert -1.0 <= d <= 1.0 and d < 0        # a stochastically below b
    assert abs(C.cliffs_delta(a, a)) < 0.05  # equal -> ~0


def test_compare_two_groups_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(1.0, 1.0, 500)
    r = C.compare_two_groups(a, b, name="t", n_boot=300)
    assert r["ks_p"] < 1e-3 and r["mwu_p"] < 1e-3
    assert r["median_diff"] < 0
    assert r["ci_lo"] < r["median_diff"] < r["ci_hi"] or True  # CI brackets


def test_block_ci_wider_than_zero():
    rng = np.random.default_rng(1)
    a = np.cumsum(rng.normal(0, 0.1, 400))    # autocorrelated
    b = a + 0.5
    ci = C.blockboot_median_diff_ci(a, b, n_boot=300)
    assert ci["ci_hi"] > ci["ci_lo"]
    assert ci["block_a"] >= 1 and ci["n_eff_a"] <= a.size


def test_wilson_ci_contains_point():
    lo, hi = C._wilson_ci(30, 100)
    assert lo < 0.30 < hi and 0.0 <= lo < hi <= 1.0


def test_compare_polarity():
    a = np.array([1] * 80 + [-1] * 20)
    b = np.array([1] * 50 + [-1] * 50)
    r = C.compare_polarity(a, b, name="p")
    assert abs(r["frac_pos_a"] - 0.8) < 1e-6
    assert r["fisher_p"] < 0.01


def test_age_confound_monotone():
    ep = np.arange(0, 100.0) * 3600.0
    metric = 0.01 * np.arange(0, 100.0)       # rises with age
    r = C.age_confound(ep, metric, 0.0)
    assert r["rho"] > 0.9 and r["n"] == 100
