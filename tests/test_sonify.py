"""LFP-to-sound synthesis: a channel segment becomes a short, audible, valid WAV
data URI (time-compressed so low-frequency LFP shifts into the audible band).

Run with: pytest tests/test_sonify.py -q
"""

from __future__ import annotations

import base64
import io
import os
import sys
import wave

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.sonify import to_wav_datauri, compression_ratio  # noqa: E402


def _decode(uri):
    assert uri.startswith("data:audio/wav;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.getnchannels(), w.getframerate(), w.getnframes()


def test_produces_valid_wav_of_requested_length():
    fs = 1000.0
    y = np.sin(2 * np.pi * 10.0 * np.arange(int(60 * fs)) / fs)   # 60 s @ 10 Hz
    uri = to_wav_datauri(y, fs, out_seconds=6.0)
    ch, sr, nframes = _decode(uri)
    assert ch == 1 and sr == 44100
    assert abs(nframes / sr - 6.0) < 0.05                          # ~6 s clip


def test_compression_ratio_is_the_frequency_upshift():
    # 60 s -> 6 s means 10x faster, so 10 Hz content lands at 100 Hz (audible)
    assert abs(compression_ratio(60_000, 1000.0, 6.0) - 10.0) < 1e-9


def test_out_seconds_clamped_and_degenerate_is_empty():
    fs = 1000.0
    y = np.random.default_rng(0).normal(size=5000)
    _c, _sr, n_short = _decode(to_wav_datauri(y, fs, out_seconds=0.01))  # -> 0.25 s
    assert abs(n_short / 44100 - 0.25) < 0.02
    assert to_wav_datauri(np.array([1.0]), fs) == ""                # too short
    assert to_wav_datauri(np.array([1.0, 2.0]), 0.0) == ""          # fs<=0


def test_normalized_int16_range():
    fs = 500.0
    y = 1e-6 * np.sin(2 * np.pi * 5 * np.arange(5000) / fs)         # tiny amplitude
    uri = to_wav_datauri(y, fs, out_seconds=2.0)
    raw = base64.b64decode(uri.split(",", 1)[1])
    with wave.open(io.BytesIO(raw), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert pcm.max() > 20000 and pcm.min() < -20000                # normalized loud
    assert pcm.max() <= 32767 and pcm.min() >= -32768
