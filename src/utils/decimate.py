"""Min/max envelope decimation for LFP/EEG time-series rendering.

A 150 us biphasic stim pulse at 20 kHz spans only ~3 samples. Every-Nth
subsampling drops most pulses; min/max envelope per bin always carries
the peak of any sub-bin transient into the rendered figure. The same
helper drives the LFP Browser and the Video Review LFP trace so the
two share a single decimation policy.
"""

from __future__ import annotations

import numpy as np


_RAW_CAP = 200_000      # visible_samples below this -> render raw
_TARGET_BINS = 60_000   # envelope target when decimating


def envelope(y: np.ndarray, fs: float, target_bins: int,
             t_start: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray, int]:
    """Min/max envelope of a 1-D series.

    Returns (x, y_env, decim). When decim == 1 the returned arrays are
    the raw samples (and their absolute time stamps); otherwise the
    arrays interleave (bin_left, bin_right) x-coords with (min, max)
    y-values, so Plotly draws one vertical line per bin.

    Bin times are absolute (offset by *t_start* seconds) so x-coords
    stay coordinate-consistent across zoom levels -- click handlers and
    overlay shapes keep working.
    """
    assert y.ndim == 1, "envelope expects a 1-D series"
    assert fs > 0, "fs must be positive"
    assert target_bins >= 1, "target_bins must be >= 1"

    n = y.shape[0]
    if n == 0:
        return np.empty(0), np.empty(0), 1

    decim = max(1, n // target_bins) if n > target_bins else 1
    if decim == 1:
        x = t_start + np.arange(n) / fs
        return x, y, 1

    n_bins = n // decim
    trimmed = y[: n_bins * decim].reshape(n_bins, decim)
    mins = trimmed.min(axis=1)
    maxs = trimmed.max(axis=1)

    bin_left_t = t_start + (np.arange(n_bins) * decim) / fs
    bin_right_t = t_start + ((np.arange(n_bins) + 1) * decim - 1) / fs

    x_out = np.empty(n_bins * 2)
    y_out = np.empty(n_bins * 2)
    x_out[0::2] = bin_left_t
    x_out[1::2] = bin_right_t
    y_out[0::2] = mins
    y_out[1::2] = maxs
    return x_out, y_out, decim


def envelope_channel(signal: np.ndarray, ch: int, fs: float,
                     lo: int, hi: int, target_bins: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Envelope of one channel over the half-open sample window [lo, hi)."""
    assert signal.ndim == 2, "signal must be 2-D [samples, channels]"
    assert 0 <= ch < signal.shape[1], "channel out of range"
    assert 0 <= lo <= hi <= signal.shape[0], "window out of range"
    return envelope(signal[lo:hi, ch], fs, target_bins, t_start=lo / fs)


def window_slice(n_samples: int, fs: float,
                 x0: float | None, x1: float | None
                 ) -> tuple[int, int]:
    """Clamp seconds [x0, x1] to a valid half-open sample window."""
    assert n_samples >= 0, "n_samples must be non-negative"
    assert fs > 0, "fs must be positive"

    if x0 is None:
        lo = 0
    else:
        lo = int(np.floor(float(x0) * fs))
    if x1 is None:
        hi = n_samples
    else:
        hi = int(np.ceil(float(x1) * fs))
    lo = max(0, min(lo, n_samples))
    hi = max(lo, min(hi, n_samples))
    return lo, hi


def choose_target_bins(visible_samples: int,
                       raw_cap: int = _RAW_CAP,
                       target_bins: int = _TARGET_BINS) -> int:
    """Decimation policy. Returns 0 when the visible window is small
    enough to render raw (caller should request raw samples), otherwise
    returns the target bin count for the envelope."""
    assert visible_samples >= 0, "visible_samples must be non-negative"
    if visible_samples <= raw_cap:
        return 0
    return target_bins


def parse_relayout(rd: dict) -> tuple[float | None, float | None, bool]:
    """Extract the new x-range from a Plotly relayoutData payload.

    Returns (x0, x1, is_reset). x0/x1 are None for events that aren't a
    pan/zoom (title edits, autosize, hover). is_reset is True when the
    user double-clicked to autoscale the x-axis.
    """
    assert isinstance(rd, dict), "relayoutData must be a dict"
    if rd.get("xaxis.autorange") is True:
        return (None, None, True)
    x0 = rd.get("xaxis.range[0]")
    x1 = rd.get("xaxis.range[1]")
    if x0 is None and x1 is None:
        rng = rd.get("xaxis.range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            x0, x1 = rng[0], rng[1]
    return (x0, x1, False)
