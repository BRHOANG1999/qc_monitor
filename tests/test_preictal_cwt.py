"""Pre-ictal Morlet CWT: power concentrates at an injected time-scale, and the
scale<->pseudo-frequency mapping is monotonic.

Run with: pytest tests/test_preictal_cwt.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal import cwt  # noqa: E402


def test_scales_monotonic_freq():
    scales = cwt.make_scales(n=512, dt=1.0, scales_per_octave=4)
    assert len(scales) >= 2
    assert list(scales) == sorted(scales)            # ascending scale
    # pseudo-freq falls as scale rises.
    _c, sc, fr = cwt.morlet_cwt(np.random.randn(512), dt=1.0)
    order = np.argsort(sc)
    assert np.all(np.diff(fr[order]) <= 1e-9)        # freq decreasing with scale


def test_cwt_power_concentrates_at_injected_scale():
    n, dt = 1024, 1.0                                # 1 Hz trajectory cadence
    t = np.arange(n) * dt
    period = 40.0                                    # a 40 s oscillation
    x = np.sin(2 * np.pi * t / period)
    coeffs, scales, freqs = cwt.morlet_cwt(x, dt, min_leadtime_sec=2.0)
    summ = cwt.scale_summaries(coeffs, scales, freqs)
    best = max(summ, key=lambda s: s["coeff_mean"])
    # The winning scale's pseudo-frequency should match 1/period within 30%.
    assert abs(best["pseudo_freq_hz"] - 1.0 / period) < 0.3 / period


def test_cwt_discriminates_two_timescales():
    # A slower oscillation must peak at a LARGER scale (lower pseudo-freq) than
    # a faster one -- the property the whole engine depends on.
    n, dt = 1024, 1.0
    t = np.arange(n) * dt

    def peak_scale(period):
        c, sc, fr = cwt.morlet_cwt(np.sin(2 * np.pi * t / period), dt,
                                   min_leadtime_sec=2.0)
        s = cwt.scale_summaries(c, sc, fr)
        return max(s, key=lambda r: r["coeff_mean"])

    fast, slow = peak_scale(30.0), peak_scale(120.0)
    assert slow["scale"] > fast["scale"]
    assert slow["pseudo_freq_hz"] < fast["pseudo_freq_hz"]


def test_scale_summaries_shape():
    coeffs, scales, freqs = cwt.morlet_cwt(np.random.randn(256), dt=2.0)
    summ = cwt.scale_summaries(coeffs, scales, freqs)
    assert len(summ) == len(scales)
    for i, s in enumerate(summ):
        assert s["scale_index"] == i
        assert s["coeff_max"] >= s["coeff_mean"] >= 0.0
