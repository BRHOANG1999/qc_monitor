"""Turn one LFP channel into audible sound.

LFP is low-frequency and long, so it is not directly audible in real time. We
TIME-COMPRESS the selected segment into a short clip: the segment (duration
``in_sec = len(y)/fs``) is resampled to ``out_seconds`` at an audio sample rate,
which shifts every frequency up by ``in_sec/out_seconds`` (e.g. a 60 s window
played in 6 s lifts 10 Hz content to 100 Hz). The clip is normalized, de-meaned,
faded in/out to avoid clicks, and returned as a base64 ``data:audio/wav`` URI that
an HTML ``<audio>`` element can play directly. Pure + dependency-light (numpy +
stdlib ``wave``) so it is unit-testable off the dashboard.
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np

_SR = 44100                 # output audio sample rate (Hz)
_MAX_SECONDS = 60.0         # clip length cap (a longer clip is just tedious)


def compression_ratio(in_samples: int, fs: float, out_seconds: float) -> float:
    """How much faster than real time the clip plays (= frequency up-shift factor)."""
    in_sec = in_samples / float(fs) if fs else 0.0
    return in_sec / out_seconds if out_seconds > 0 else 0.0


def to_wav_datauri(y, fs: float, out_seconds: float = 8.0, *,
                   sr: int = _SR, fade_ms: float = 12.0) -> str:
    """Time-compress channel signal *y* (sampled at *fs* Hz) into an audible mono
    WAV and return a ``data:audio/wav;base64,...`` URI. Empty/degenerate input →
    ``""``. *out_seconds* is clamped to (0, 60]."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size < 2 or fs <= 0:
        return ""
    out_seconds = float(min(max(out_seconds, 0.25), _MAX_SECONDS))
    n_out = max(2, int(round(out_seconds * sr)))
    # linear resample onto the output timeline (content shifts up in frequency)
    yr = np.interp(np.linspace(0.0, 1.0, n_out),
                   np.linspace(0.0, 1.0, y.size), y)
    yr = yr - np.mean(yr)
    peak = float(np.max(np.abs(yr)))
    if peak > 0:
        yr = yr / peak
    f = int(round(fade_ms / 1000.0 * sr))
    if f > 0 and 2 * f < n_out:                     # click-free ends
        ramp = np.linspace(0.0, 1.0, f)
        yr[:f] *= ramp
        yr[-f:] *= ramp[::-1]
    pcm = np.clip(np.round(yr * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
