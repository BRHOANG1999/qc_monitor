"""Smoke tests for src/utils/filters.py.

Validates the filter cascade (HP/LP/notch/smoothing) against synthetic
signals where we know the expected outcome by construction, the PSD
helper on white noise + a pure tone, and the LRU cache's reuse
guarantee. No mocks -- everything runs in <1 s.

Run with: pytest tests/test_filters.py -q
Or directly: python -m tests.test_filters
"""

from __future__ import annotations

import os
import sys

import numpy as np


# Make sure src.* imports work when invoking the file directly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.filters import (  # noqa: E402
    apply_filter, compute_psd, evict_filtered, filter_settings_key,
    get_filtered,
)


def _tone(freq_hz: float, fs: float, dur_s: float = 5.0) -> np.ndarray:
    t = np.arange(int(dur_s * fs)) / fs
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ===================================================================== #
#  apply_filter
# ===================================================================== #

def test_highpass_attenuates_low_freq():
    """HP=50 should kill a 5 Hz tone by >60 dB (>1000x amplitude)."""
    fs = 10000.0
    sig = _tone(5, fs)
    out = apply_filter(sig, fs, highpass=50)
    assert out.std() < sig.std() / 1000, (
        f"HP didn't attenuate enough: orig {sig.std():.3f} -> {out.std():.3f}")


def test_lowpass_attenuates_high_freq():
    """LP=100 should kill a 1 kHz tone."""
    fs = 10000.0
    sig = _tone(1000, fs)
    out = apply_filter(sig, fs, lowpass=100)
    assert out.std() < sig.std() / 100, (
        f"LP didn't attenuate enough: orig {sig.std():.3f} -> {out.std():.3f}")


def test_notch_kills_60_but_passes_50():
    """Notch=60 attenuates 60 Hz, leaves 50 Hz alone."""
    fs = 10000.0
    out_60 = apply_filter(_tone(60, fs), fs, notch=60)
    out_50 = apply_filter(_tone(50, fs), fs, notch=60)
    assert out_60.std() < 0.5 * _tone(60, fs).std(), (
        f"Notch 60 left too much 60 Hz: {out_60.std():.3f}")
    assert out_50.std() > 0.8 * _tone(50, fs).std(), (
        f"Notch 60 unexpectedly attenuated 50 Hz: {out_50.std():.3f}")


def test_smoothing_attenuates_high_freq():
    """Gaussian smoothing reduces amplitude of a high-freq tone."""
    fs = 10000.0
    sig = _tone(1000, fs)
    out = apply_filter(sig, fs, smoothing_ms=10)
    assert out.std() < sig.std() / 2


def test_nan_handling_short_gap_interpolated_long_gap_kept():
    """Short NaN gaps get interpolated (no NaN in filter output),
    long ones survive as NaN so stim-blank windows show as breaks."""
    fs = 10000.0
    sig = _tone(5, fs)
    sig[100:110] = np.nan          # 1 ms gap
    sig[20000:21000] = np.nan      # 100 ms gap
    out = apply_filter(sig, fs, highpass=1)
    assert not np.isnan(out[100:110]).any(), "Short gap should be filled"
    assert np.isnan(out[20000:21000]).all(), "Long gap should remain NaN"
    # Region adjacent to long gap should be clean (sosfiltfilt no
    # longer propagates NaN because we pre-interpolated):
    assert np.isfinite(out[19500])


def test_apply_filter_2d_preserves_shape():
    fs = 10000.0
    sig2d = np.column_stack([_tone(5, fs), _tone(1000, fs)])
    out = apply_filter(sig2d, fs, lowpass=200)
    assert out.shape == sig2d.shape


# ===================================================================== #
#  compute_psd
# ===================================================================== #

def test_psd_flat_for_white_noise():
    fs = 10000.0
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(10 * fs)).astype(np.float32)
    freqs, psd = compute_psd(noise, fs)
    assert len(freqs) > 10
    # Roughly flat: ratio max/min within an order of magnitude
    body = psd[(freqs > 100) & (freqs < 4000)]
    assert body.max() / body.min() < 20


def test_psd_peak_at_known_tone():
    fs = 10000.0
    rng = np.random.default_rng(0)
    sig = (_tone(1000, fs)
           + 0.05 * rng.standard_normal(int(5 * fs)).astype(np.float32))
    freqs, psd = compute_psd(sig.astype(np.float32), fs)
    peak_hz = float(freqs[int(np.argmax(psd))])
    assert 950 < peak_hz < 1050, f"Peak at {peak_hz} Hz, expected ~1000"


# ===================================================================== #
#  Cache
# ===================================================================== #

def test_filter_cache_reuses_array():
    evict_filtered()
    fs = 10000.0
    sig = _tone(5, fs)
    a = get_filtered("k", sig, fs, highpass=50)
    b = get_filtered("k", sig, fs, highpass=50)
    assert a is b, "second call should return cached object by reference"


def test_filter_settings_key_canonicalises():
    k1 = filter_settings_key(0, 0, 0, 0)
    k2 = filter_settings_key(None, None, None, None)
    assert k1 == k2 == (None, None, None, 0.0)
    k3 = filter_settings_key(1, 2, 60, 5)
    assert k3 == (1.0, 2.0, 60, 5.0)


def test_evict_targets_one_file():
    evict_filtered()
    fs = 10000.0
    sig = _tone(5, fs)
    get_filtered("file_a", sig, fs, lowpass=100)
    get_filtered("file_b", sig, fs, lowpass=100)
    n = evict_filtered("file_a")
    assert n == 1
    # file_b survives
    b = get_filtered("file_b", sig, fs, lowpass=100)
    # If we'd evicted file_b too, get_filtered would recompute; we
    # can't easily detect that from outside, so the rowcount above
    # is the assertion that matters.
    assert b is not None


# ===================================================================== #
#  Manual entry point
# ===================================================================== #

if __name__ == "__main__":
    tests = [v for k, v in globals().items()
             if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
