"""Tests for src/utils/wavelet.py (Morlet CWT scalogram + band power).

Run: pytest tests/test_wavelet.py -q
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import wavelet as wav  # noqa: E402
from src.utils import filters as flt  # noqa: E402


def _tone(freq, fs, dur, t0=0.0):
    t = np.arange(int(fs * dur)) / fs + t0
    return np.sin(2 * np.pi * freq * t)


def test_scalogram_peaks_at_tone_frequency():
    fs = 2000.0
    x = _tone(40.0, fs, 2.0)
    freqs, times, power = wav.scalogram(x, fs, fmin=5, fmax=200, n_freqs=64)
    assert freqs.size and times.size and power.shape == (freqs.size, times.size)
    peak_f = freqs[int(np.argmax(power.mean(axis=1)))]
    assert 32 < peak_f < 50, f"peak {peak_f} Hz not near 40"
    # freqs ascending
    assert np.all(np.diff(freqs) > 0)


def test_band_power_localizes_in_time():
    # Tone only in the SECOND half -> band power low then high.
    fs = 2000.0
    silence = np.zeros(int(fs * 1.0))
    tone = _tone(40.0, fs, 1.0)
    x = np.concatenate([silence, tone])
    band, work_fs = wav.wavelet_band_power(x, fs, 30, 50)
    assert band.size > 4 and work_fs > 0
    h = band.size // 2
    assert band[h + h // 2] > 5 * (band[h // 2] + 1e-9)


def test_epoch_wavelet_band_power_tracks_filters_band_power():
    # On band-limited epochs, wavelet band power should correlate with the
    # Welch-PSD epoch band power (same physical quantity, different method).
    fs = 2000.0
    rng = np.random.default_rng(0)
    stim = np.arange(1.0, 9.0, 1.0)
    sig = np.zeros(int(fs * 10))
    amps = rng.uniform(0.5, 3.0, size=stim.size)
    for s, a in zip(stim, amps):
        i0 = int(s * fs)
        seg = a * _tone(40.0, fs, 0.2)
        sig[i0:i0 + seg.size] += seg
    win = (0.0, 0.2)
    _t, wpow = wav.epoch_wavelet_band_power(sig, fs, stim, 30, 50, win)
    _t2, ppow = flt.epoch_band_power(sig, fs, stim, 30, 50, win)
    assert wpow.size == stim.size and ppow.size == stim.size
    r = np.corrcoef(wpow, ppow)[0, 1]
    assert r > 0.9, f"wavelet vs welch band power corr {r:.2f} too low"


def test_epoch_wavelet_features_shape_and_bands():
    fs = 2000.0
    traces = np.vstack([_tone(40.0, fs, 0.2), _tone(80.0, fs, 0.2),
                        np.zeros(int(fs * 0.2))])
    bands = {"slow_gamma": (30, 50), "gamma": (50, 100)}
    out = wav.epoch_wavelet_features(traces, fs, bands)
    assert set(out) == set(bands)
    assert out["slow_gamma"].shape == (3,)
    # epoch 0 (40 Hz) heavier in slow_gamma; epoch 1 (80 Hz) heavier in gamma
    assert out["slow_gamma"][0] > out["gamma"][0]
    assert out["gamma"][1] > out["slow_gamma"][1]


def test_edge_cases_dont_raise():
    fs = 2000.0
    assert wav.scalogram(np.zeros(4), fs)[0].size == 0          # too short
    band, _ = wav.wavelet_band_power(np.zeros(500), fs, 30, 50)  # flat
    assert np.all(band == 0) or band.size == 0
    nan = np.full(500, np.nan)
    f, _t, p = wav.scalogram(nan, fs, fmin=30, fmax=50)
    assert np.all(np.isfinite(p))                                # sanitized
    with pytest.raises(AssertionError):
        wav.wavelet_band_power(np.zeros(500), fs, 50, 30)        # hi<=lo


def test_decimation_makes_20khz_tractable():
    # A 20 kHz x 60 s segment must complete quickly (decimation engaged).
    fs = 20000.0
    x = _tone(80.0, fs, 60.0)
    t0 = time.perf_counter()
    freqs, times, power = wav.scalogram(x, fs, fmin=10, fmax=200, n_freqs=32)
    dt = time.perf_counter() - t0
    assert freqs.size and power.size
    # decimated to ~2 kHz -> ~120k time cols, not 1.2M
    assert times.size < 200_000
    assert dt < 20.0, f"scalogram too slow ({dt:.1f}s) -- decimation off?"
