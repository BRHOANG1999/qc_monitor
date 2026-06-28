"""Tier 1: Wavelet band power — per-channel Morlet-CWT band power.

Sibling of ``analyzers/spectral.py``. Welch (in spectral.py) averages power
over 2 s windows; this captures the same bands via a Morlet CWT, which is
sensitive to the SHORT transient bursts (seizure/evoked) that Welch smooths
out. Gamma-and-up only -- the lows are already covered by the Welch bands.

A full-recording CWT is the heaviest Tier-1 cost, so this is gated behind
``config["wavelet_qc"]["enabled"]`` in the dispatcher and bounds the input
to a fixed-duration centre window before the (decimating) transform.
"""

import numpy as np

from src.utils.mat_loader import ChunkData
from src.utils import wavelet as _wav

DEFAULT_WAVELET_BANDS = {
    "slow_gamma": (30.0, 50.0),
    "gamma": (50.0, 100.0),
    "high_gamma": (100.0, 200.0),
}

# Bound the per-channel CWT input so a 1-hour 20 kHz file stays tractable:
# analyse a centre window of at most this many seconds.
_MAX_ANALYZE_SEC = 120.0


def _centre_window(sig: np.ndarray, fs: float) -> np.ndarray:
    """Centre slice of at most ``_MAX_ANALYZE_SEC`` seconds."""
    cap = int(_MAX_ANALYZE_SEC * fs)
    if sig.size <= cap:
        return sig
    start = (sig.size - cap) // 2
    return sig[start:start + cap]


def analyze_wavelet(chunk: ChunkData, bands: dict = None,
                    target_fs: float = 2000.0) -> list[dict]:
    """Per-channel wavelet band power (mean over time of the band power).
    Returns one dict per channel, keys ``f"{band}_wavelet_power"`` --
    parallel to ``analyze_spectral``'s ``f"{band}_power"``."""
    assert chunk is not None, "chunk required"
    if bands is None:
        bands = DEFAULT_WAVELET_BANDS
    results = []
    for ch in range(chunk.num_channels):
        sig = chunk.signal[:, ch]
        sig = sig[np.isfinite(sig)]
        if len(sig) < 256:
            results.append(_empty_wavelet(bands))
            continue
        seg = _centre_window(np.asarray(sig, dtype=np.float64), chunk.fs)
        metrics = {}
        for name, (lo, hi) in bands.items():
            power_t, _work_fs = _wav.wavelet_band_power(seg, chunk.fs, lo, hi)
            metrics[f"{name}_wavelet_power"] = (
                float(np.mean(power_t)) if power_t.size else 0.0)
        results.append(metrics)
    return results


def _empty_wavelet(bands: dict) -> dict:
    return {f"{name}_wavelet_power": 0.0 for name in bands}
