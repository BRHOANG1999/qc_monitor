"""Tier 1: Basic signal integrity QC (Python/numpy)."""

import numpy as np
from src.utils.mat_loader import ChunkData


def analyze_basic_qc(chunk: ChunkData, clipping_voltage: float = 10.0,
                     flatline_std_threshold: float = 1e-6) -> list[dict]:
    """Per-channel basic QC metrics. Returns list of dicts (one per channel)."""
    results = []
    for ch in range(chunk.num_channels):
        sig = chunk.signal[:, ch]

        has_nan = int(np.any(np.isnan(sig)))
        has_inf = int(np.any(np.isinf(sig)))

        # Clean for subsequent analysis
        clean = sig[np.isfinite(sig)] if (has_nan or has_inf) else sig

        if len(clean) == 0:
            results.append({
                "has_nan": has_nan, "has_inf": has_inf, "has_clipping": 0,
                "flat_line_pct": 100.0, "dc_offset": 0.0, "rms_amplitude": 0.0,
            })
            continue

        has_clipping = int(np.any(np.abs(clean) >= clipping_voltage * 0.99))
        dc_offset = float(np.mean(clean))
        rms = float(np.sqrt(np.mean(clean ** 2)))

        # Flat-line: compute std over 1-second windows
        win_samples = int(chunk.fs)
        n_windows = max(1, len(clean) // win_samples)
        flat_count = 0
        for i in range(n_windows):
            window = clean[i * win_samples:(i + 1) * win_samples]
            if np.std(window) < flatline_std_threshold:
                flat_count += 1
        flat_line_pct = 100.0 * flat_count / n_windows

        results.append({
            "has_nan": has_nan,
            "has_inf": has_inf,
            "has_clipping": has_clipping,
            "flat_line_pct": float(flat_line_pct),
            "dc_offset": dc_offset,
            "rms_amplitude": rms,
        })

    return results
