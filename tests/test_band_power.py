"""Slow-gamma band helpers: band_power integral + band_envelope
selectivity, and that the 20-200 Hz wrapper is unchanged by the refactor.

Run with: pytest tests/test_band_power.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.filters import (  # noqa: E402
    band_power, compute_psd, SLOW_GAMMA_BAND)
from src.utils.hilbert_envelope import (  # noqa: E402
    band_envelope, hilbert_envelope_20_200)

_FS = 1000.0
_LO, _HI = SLOW_GAMMA_BAND


def _tone(freq, n=8000, fs=_FS, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t)


# ---- band_power -------------------------------------------------- #

def test_band_power_concentrates_in_band():
    f40 = _tone(40.0)
    freqs, psd = compute_psd(f40, _FS)
    in_band = band_power(freqs, psd, _LO, _HI)
    out_band = band_power(freqs, psd, 60.0, 200.0)
    # A 40 Hz tone's power lives in 30-50, not 60-200.
    assert in_band > 0
    assert in_band > 50 * max(out_band, 1e-12)


def test_band_power_zero_for_out_of_band_tone():
    f100 = _tone(100.0)
    freqs, psd = compute_psd(f100, _FS)
    in_band = band_power(freqs, psd, _LO, _HI)
    total = band_power(freqs, psd, 1.0, 200.0)
    # Almost none of a 100 Hz tone's power falls in 30-50 Hz.
    assert in_band < 0.01 * total


def test_band_power_guards():
    assert band_power(np.array([]), np.array([]), 30, 50) == 0.0
    # Mismatched lengths / inverted band -> 0.
    assert band_power(np.array([1.0, 2.0]), np.array([1.0]), 30, 50) == 0.0
    assert band_power(np.array([30.0, 40.0]),
                      np.array([1.0, 1.0]), 50, 30) == 0.0


# ---- band_envelope ----------------------------------------------- #

def test_band_envelope_passes_in_band_tone():
    env_in = band_envelope(_tone(40.0), _FS, _LO, _HI, smooth=False)
    env_lo = band_envelope(_tone(5.0), _FS, _LO, _HI, smooth=False)
    env_hi = band_envelope(_tone(100.0), _FS, _LO, _HI, smooth=False)
    # 40 Hz survives the 30-50 mask; 5 Hz and 100 Hz are killed.
    assert np.mean(env_in) > 0.3
    assert np.mean(env_lo) < 0.05 * np.mean(env_in)
    assert np.mean(env_hi) < 0.05 * np.mean(env_in)


def test_band_envelope_length_and_finite():
    sig = _tone(40.0, n=5000)
    env = band_envelope(sig, _FS, _LO, _HI)
    assert env.shape == sig.shape
    assert np.all(np.isfinite(env))
    # NaNs are tolerated (zeroed), still finite + same length.
    sig2 = sig.copy()
    sig2[100:150] = np.nan
    env2 = band_envelope(sig2, _FS, _LO, _HI)
    assert env2.shape == sig2.shape and np.all(np.isfinite(env2))


def test_20_200_wrapper_matches_direct_band_call():
    sig = _tone(120.0, n=6000) + 0.3 * _tone(40.0, n=6000)
    a = hilbert_envelope_20_200(sig, _FS)
    b = band_envelope(sig, _FS, 20.0, 200.0, smooth=True)
    assert np.allclose(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
