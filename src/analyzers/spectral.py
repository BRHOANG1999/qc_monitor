"""Tier 1: Spectral analysis — PSD, band powers, line noise (Python/scipy)."""

import numpy as np
from scipy.signal import welch
from src.utils.mat_loader import ChunkData

DEFAULT_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
    "high_gamma": (100, 200),
}


def analyze_spectral(chunk: ChunkData,
                     bands: dict = None,
                     line_noise_freq: float = 60.0) -> list[dict]:
    """Per-channel spectral metrics. Returns list of dicts (one per channel)."""
    if bands is None:
        bands = DEFAULT_BANDS

    results = []
    for ch in range(chunk.num_channels):
        sig = chunk.signal[:, ch]
        sig = sig[np.isfinite(sig)]

        if len(sig) < 256:
            results.append(_empty_spectral(bands))
            continue

        # Welch PSD with 2-second windows
        nperseg = min(int(2 * chunk.fs), len(sig))
        freqs, psd = welch(sig, fs=chunk.fs, nperseg=nperseg, noverlap=nperseg // 2)

        # Band powers
        metrics = {}
        total_power = float(np.trapz(psd, freqs))
        metrics["total_power"] = total_power

        for band_name, (flo, fhi) in bands.items():
            mask = (freqs >= flo) & (freqs <= fhi)
            band_power = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
            metrics[f"{band_name}_power"] = band_power

        # Line noise (60Hz ± 2Hz)
        noise_mask = (freqs >= line_noise_freq - 2) & (freqs <= line_noise_freq + 2)
        line_noise_power = float(np.trapz(psd[noise_mask], freqs[noise_mask])) if np.any(noise_mask) else 0.0
        metrics["line_noise_power"] = line_noise_power
        metrics["line_noise_ratio"] = line_noise_power / total_power if total_power > 0 else 0.0

        results.append(metrics)

    return results


def _empty_spectral(bands: dict) -> dict:
    metrics = {"total_power": 0.0, "line_noise_power": 0.0, "line_noise_ratio": 0.0}
    for band_name in bands:
        metrics[f"{band_name}_power"] = 0.0
    return metrics
