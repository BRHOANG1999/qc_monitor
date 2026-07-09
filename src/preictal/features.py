"""Base-feature trajectory builders (Stage 1: the headline ``line_length``).

Each wraps an EXISTING analyzer so "which feature carries the pre-ictal signal"
is answered by running the same CWT+scoring over each. Every feature fn maps a
1-D signal window (native fs) to one scalar; the trajectory builder calls it per
decimated window.
"""

from __future__ import annotations

import numpy as np

from src.preictal.registry import register_feature


@register_feature("line_length")
def line_length(window: np.ndarray, fs: float) -> float:
    """Line length (Sum |Delta|) of the 20-200 Hz Hilbert envelope over the
    window -- the lab's BHZ front-end (``hilbert_envelope_20_200``), the
    headline pre-ictal feature. Falls back to the raw signal's line length when
    the window is too short to band-filter."""
    w = np.asarray(window, dtype=float)
    if w.size < 2:
        return 0.0
    # 20-200 Hz needs fs > 400 (Nyquist > 200) and enough samples for the FFT
    # bandpass; otherwise the raw line length is the honest fallback.
    if fs > 400.0 and w.size >= 64:
        try:
            from src.utils.hilbert_envelope import hilbert_envelope_20_200
            env = hilbert_envelope_20_200(w, float(fs))
            return float(np.sum(np.abs(np.diff(env))))
        except Exception:  # noqa: BLE001 -- never let one window kill the run
            pass
    return float(np.sum(np.abs(np.diff(w))))
