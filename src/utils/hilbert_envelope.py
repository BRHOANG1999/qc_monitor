"""1:1 Python port of ``tay_preprocess.m`` from BHZ_DETECTOR.

The MATLAB original (``C:\\Users\\hoang392\\Downloads\\src\\
tay_preprocess.m``) computes the instantaneous amplitude of an
LFP trace in the 20-200 Hz band, then smooths it with a slow
Butterworth low-pass. It's the lab's "brain feature over time"
default -- a continuous envelope per sample.

Magic numbers preserved as-is from the MATLAB source so the
Python and MATLAB outputs match within float-precision:

  * HP / LP cutoffs: 20 Hz / 200 Hz
  * Bandpass: FFT zero-mask (NOT a butter-bandpass) -- one-shot
    multiplication of the Fourier transform.
  * Envelope: ``abs(ifft(masked))`` -- the magnitude of the
    inverse-transformed band-limited signal. MATLAB comments
    it as "instantaneous amplitude via Hilbert" because the
    FFT zero-mask + abs(ifft) is equivalent to taking the
    magnitude of the analytic signal in that band.
  * Smoothing: ``butter(2, 0.1/500, 'low')`` + ``filtfilt``.
    The 0.1/500 normalized cutoff is fixed regardless of fs --
    that's how the MATLAB code works and we don't second-guess
    it here.

Returns the smoothed envelope with the same length as the
input. The dashboard then runs this through the existing
``envelope()`` decimator from ``src/utils/decimate.py`` before
plotting.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, filtfilt

logger = logging.getLogger("qc_monitor.utils.hilbert_envelope")

# MATLAB constants -- do not "tune" without lab sign-off; the
# downstream MATLAB pipeline reads files keyed on these.
_HPASS_HZ: float = 20.0
_LPASS_HZ: float = 200.0
_SMOOTH_NORM_CUTOFF: float = 0.1 / 500.0  # tay_preprocess.m:15
_SMOOTH_ORDER: int = 2


def hilbert_envelope_20_200(signal: np.ndarray,
                              fs: float) -> np.ndarray:
    """Return the smoothed 20-200 Hz instantaneous amplitude.

    Drops NaNs in the input to 0 before the FFT (the lab's
    stim-blank fills produce NaN runs; transforming them
    propagates NaN everywhere). The returned array has the
    same shape and dtype-family (float64) as the input.
    """
    assert isinstance(signal, np.ndarray), "signal must be ndarray"
    assert signal.ndim == 1, "signal must be 1-D"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"

    n = signal.shape[0]
    if n < 4:
        # Below the filtfilt minimum; return zeros same length.
        return np.zeros(n, dtype=np.float64)

    x = signal.astype(np.float64, copy=True)
    # Replace NaN / inf with 0 so the FFT stays finite. The
    # MATLAB tay_preprocess crashes on non-finite input
    # (createerror=5/0); we'd rather be defensive.
    x[~np.isfinite(x)] = 0.0

    fdata = np.fft.fft(x)
    hp_bin = int(np.floor(n * _HPASS_HZ / fs))
    lp_bin = int(np.floor(n * _LPASS_HZ / fs))
    # MATLAB indexing: fdata(1:floor(N*HP/fs))=0 zeros bins
    # 1..hp_bin inclusive. Python equivalent is [:hp_bin].
    fdata[:hp_bin] = 0
    if lp_bin < n:
        fdata[lp_bin:] = 0

    envelope_data = np.abs(np.fft.ifft(fdata))

    b, a = butter(_SMOOTH_ORDER, _SMOOTH_NORM_CUTOFF, btype="low")
    # filtfilt requires len(x) > 3 * max(len(a), len(b)); we
    # already guarded n < 4 above. For very short signals the
    # default padlen can still exceed; clamp.
    pad = min(3 * max(len(a), len(b)), n - 1)
    return filtfilt(b, a, envelope_data, padlen=pad)
