"""Slow-dynamics summary orchestrator (Part B UI feed).

summarize() must surface phi rising toward onset (phi_near > phi_far) and stay
empty-safe on short input.

Run: pytest tests/test_periictal_slow_report.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import slow_report as sr         # noqa: E402
from src.periictal import trial_series as ts        # noqa: E402


def _ar1_ramp(n, phi0, phi1, rng):
    phi = np.linspace(phi0, phi1, n)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi[i] * x[i - 1] + rng.standard_normal()
    return x


def test_summary_shows_phi_rising_toward_onset():
    rng = np.random.default_rng(0)
    n, dt = 6000, 2.0
    vals = _ar1_ramp(n, 0.75, 0.98, rng)          # phi climbs over the series
    secs = np.arange(n, dtype=float) * dt
    onset = secs[-1] + dt                          # onset just after the last stim
    s = ts.assemble_trial_series("A", "feat", "ch", secs, vals,
                                 np.array([onset]))
    out = sr.summarize(s, win=300, cap_h=6.0)
    assert out["insufficient"] is False
    # near-onset (tail, high phi) exceeds far (head, low phi)
    assert out["phi_near"] > out["phi_far"] + 0.05, (out["phi_near"], out["phi_far"])
    # lambda is reported (not tau); near lambda closer to 0 than far
    assert "tau" not in out
    assert out["lambda_near"] > out["lambda_far"]   # closer to 0 (less negative)
    assert np.isfinite(out["band_frac"])
    assert len(out["phi_leadtime"]["centers_h"]) == 8


def test_summary_is_empty_safe_on_short_series():
    secs = np.arange(10, dtype=float) * 2.0
    s = ts.assemble_trial_series("A", "feat", "ch", secs, np.arange(10.0),
                                 np.array([100.0]))
    out = sr.summarize(s, win=300)
    assert out["insufficient"] is True
