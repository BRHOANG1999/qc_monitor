"""Tier 1: Artifact detection via power envelope (Python/scipy)."""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from src.utils.mat_loader import ChunkData


def analyze_artifact(chunk: ChunkData,
                     method: str = "fixed",
                     threshold: float = 500.0,
                     mad_k: float = 4.0,
                     merge_gap_sec: float = 2.0) -> list[dict]:
    """Per-channel artifact percentage. Returns list of dicts (one per channel)."""
    results = []

    for ch in range(chunk.num_channels):
        sig = chunk.signal[:, ch].copy()
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        if len(sig) < 256:
            results.append({"artifact_pct": 0.0})
            continue

        # Bandpass 20-200 Hz
        nyq = chunk.fs / 2
        lo = min(20.0 / nyq, 0.95)
        hi = min(200.0 / nyq, 0.95)
        if lo >= hi:
            results.append({"artifact_pct": 0.0})
            continue

        try:
            b, a = butter(2, [lo, hi], btype="band")
            filtered = filtfilt(b, a, sig)
        except ValueError:
            results.append({"artifact_pct": 0.0})
            continue

        # Power envelope via Hilbert transform
        analytic = hilbert(filtered)
        envelope = np.abs(analytic) ** 2

        # Smooth envelope
        smooth_samples = max(1, int(0.05 * chunk.fs))
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope = np.convolve(envelope, kernel, mode="same")

        # Threshold
        if method == "mad":
            median_env = np.median(envelope)
            mad = np.median(np.abs(envelope - median_env))
            thresh = median_env + mad_k * 1.4826 * mad
        else:
            thresh = threshold

        artifact_mask = envelope > thresh

        # Morphological closing: fill gaps shorter than merge_gap_sec
        gap_samples = int(merge_gap_sec * chunk.fs)
        if gap_samples > 1:
            artifact_mask = _close_gaps(artifact_mask, gap_samples)

        artifact_pct = 100.0 * np.sum(artifact_mask) / len(artifact_mask)
        results.append({"artifact_pct": float(artifact_pct)})

    return results


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than max_gap between True regions."""
    result = mask.copy()
    in_gap = False
    gap_start = 0

    for i in range(len(result)):
        if result[i]:
            if in_gap and (i - gap_start) <= max_gap:
                result[gap_start:i] = True
            in_gap = False
        else:
            if not in_gap:
                gap_start = i
                in_gap = True

    return result
