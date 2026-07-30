"""Infraslow spectral analysis of the feature series (Part B, B3).

A planted 300 s oscillation (~0.0033 Hz, in the 0.001-0.01 Hz band) must show
elevated in-band power/fraction vs a white series, and vs an out-of-band probe.

Run: pytest tests/test_periictal_spectral.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import spectral as sp           # noqa: E402
from src.periictal import trial_series as ts       # noqa: E402


def _series(values, dt=2.0):
    n = len(values)
    secs = np.arange(n, dtype=float) * dt
    return ts.assemble_trial_series("A", "feat", "ch", secs,
                                    np.asarray(values, float), np.array([]))


def test_planted_infraslow_line_shows_in_band_power():
    rng = np.random.default_rng(0)
    n, dt = 8000, 2.0
    t = np.arange(n) * dt
    osc = np.sin(2 * np.pi * t / 300.0) + 0.3 * rng.standard_normal(n)   # ~0.0033 Hz
    white = rng.standard_normal(n)
    bf_osc = sp.series_band_power(_series(osc, dt))["band_frac"]
    bf_white = sp.series_band_power(_series(white, dt))["band_frac"]
    assert bf_osc > 3 * bf_white, (bf_osc, bf_white)      # the line dominates in-band
    assert bf_osc > 0.3                                   # much power sits in-band


def test_out_of_band_probe_sees_little():
    rng = np.random.default_rng(1)
    n, dt = 8000, 2.0
    t = np.arange(n) * dt
    osc = np.sin(2 * np.pi * t / 300.0) + 0.3 * rng.standard_normal(n)
    inb = sp.welch_band_power(osc, dt, 0.001, 0.01)[0]
    outb = sp.welch_band_power(osc, dt, 0.05, 0.10)[0]    # away from the 0.0033 Hz line
    assert inb > 10 * outb, (inb, outb)


def test_degenerate_series_is_safe():
    out = sp.series_band_power(_series(np.zeros(50)))
    assert out["n_seg"] == 0 or not np.isfinite(out["band_frac"])
