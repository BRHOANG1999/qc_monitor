"""Slow-dynamics estimators (Part B, B1/B2).

B1: the across-trial AR(1) eigenvalue must recover a known phi and map it to
lambda (never tau). B2: the variance/phi decoupling test must tell critical
slowing (both rise, coupled) from injected noise (variance rises, phi flat).

Run: pytest tests/test_periictal_slow_dynamics.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import slow_dynamics as sd       # noqa: E402
from src.periictal import trial_series as ts        # noqa: E402


def _ar1_process(n, phi, sigma, rng, x0=0.0):
    phi = np.broadcast_to(np.asarray(phi, dtype=float), (n,))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = phi[i] * x[i - 1] + sigma[i] * rng.standard_normal()
    return x


def _series(values, dt=2.0, onsets=None):
    n = len(values)
    secs = np.arange(n, dtype=float) * dt
    return ts.assemble_trial_series("A", "feat", "ch", secs,
                                    np.asarray(values, dtype=float),
                                    np.array([] if onsets is None else onsets))


# --------------------------------------------------- lambda mapping --- #

def test_lambda_from_phi_and_no_tau():
    lam = sd.lambda_from_phi(np.array([0.5, 0.99, 1.0, -0.2, 0.0]), 2.0)
    assert np.isclose(lam[0], np.log(0.5) / 2.0)
    assert np.isclose(lam[1], np.log(0.99) / 2.0)
    assert np.isnan(lam[2]) and np.isnan(lam[3]) and np.isnan(lam[4])  # boundary
    # the API reports phi/lambda, never tau
    out = sd.fit_ar1(_ar1_process(500, 0.8, 1.0, np.random.default_rng(0)),
                     dt=2.0)
    assert "tau" not in out and "lambda" in out and "phi" in out


# ------------------------------------------------ eigenvalue recovery --- #

def test_fit_ar1_recovers_known_phi_with_covering_ci():
    x = _ar1_process(6000, 0.9, 1.0, np.random.default_rng(1))
    out = sd.fit_ar1(x, dt=2.0)
    assert abs(out["phi"] - 0.9) < 0.05, out["phi"]
    assert out["ci_lo"] <= out["phi"] <= out["ci_hi"]    # symmetric analytic CI
    assert out["ci_lo"] < 0.9 < out["ci_hi"]              # CI covers the truth
    assert np.isclose(out["lambda"], np.log(out["phi"]) / 2.0)


def test_phi_series_rises_as_phi_ramps_toward_one():
    rng = np.random.default_rng(2)
    x = _ar1_process(6000, np.linspace(0.75, 0.98, 6000), 1.0, rng)
    phi = sd.phi_series(_series(x), win=300)
    q = phi[np.isfinite(phi)]
    head, tail = q[:len(q) // 4], q[-len(q) // 4:]
    assert np.nanmean(tail) > np.nanmean(head) + 0.05     # phi climbs


# ---------------------------------------------- variance/phi decoupling --- #

def test_decoupling_separates_slowing_from_noise_injection():
    rng = np.random.default_rng(3)
    n = 6000
    slowing = _ar1_process(n, np.linspace(0.80, 0.985, n), 1.0, rng)   # phi rises
    noise = _ar1_process(n, 0.85, np.linspace(1.0, 4.0, n), rng)       # sigma rises
    d_slow = sd.decoupling(_series(slowing), win=300)
    d_noise = sd.decoupling(_series(noise), win=300)
    # critical slowing: phi and variance rise together -> strong positive coupling
    assert d_slow["coupling_rho"] > 0.5, d_slow["coupling_rho"]
    # injected noise: variance climbs but phi is flat -> far weaker coupling
    assert d_slow["coupling_rho"] > d_noise["coupling_rho"] + 0.3
    # sanity: variance actually rose in the noise case
    v = d_noise["variance"][np.isfinite(d_noise["variance"])]
    assert np.nanmean(v[-len(v) // 4:]) > np.nanmean(v[:len(v) // 4])


def test_segments_isolate_gaps_no_ar_across_holes():
    # two AR(1) runs separated by a big time gap must be fit independently
    rng = np.random.default_rng(4)
    a = _ar1_process(1000, 0.9, 1.0, rng)
    b = _ar1_process(1000, 0.9, 1.0, rng)
    secs = np.concatenate([np.arange(1000) * 2.0,
                           np.arange(1000) * 2.0 + 1000 * 2.0 + 10000.0])
    s = ts.assemble_trial_series("A", "feat", "ch", secs,
                                 np.concatenate([a, b]), np.array([]))
    assert len(s.segments()) == 2                        # the gap split the series
    phi = sd.phi_series(s, win=200)
    assert np.isfinite(phi).sum() > 1500                 # both segments estimated
