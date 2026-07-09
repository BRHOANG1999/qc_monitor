"""Feature-trajectory construction over a seizure's pre-ictal lead-up.

Builds a DECIMATED feature trajectory over ``[onset - ceiling, onset]`` -- one
scalar per ``step_sec`` window of the (channel) signal, concatenated across
chunk-file boundaries -- then robust-z normalizes it against the animal's own
scale so cross-animal pooling detects pre-ictal structure, not animal identity.
"""

from __future__ import annotations

import numpy as np

from src.preictal.isi import parse_chunk_datetime
from src.preictal.types import FeatureSpec


def robust_z(x: np.ndarray, baseline: np.ndarray | None = None) -> np.ndarray:
    """Median/MAD standardization. *baseline* (defaults to *x*) sets the
    center/scale -- pass an interictal baseline to normalize against it. MAD of
    0 falls back to 1 (a flat trajectory maps to zeros)."""
    x = np.asarray(x, dtype=float)
    b = np.asarray(baseline if baseline is not None else x, dtype=float)
    if b.size == 0:
        return x
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med)))
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (x - med) / scale


def _decimate_to(sig: np.ndarray, fs: float,
                  target_fs: float) -> tuple[np.ndarray, float]:
    """Anti-alias decimate *sig* toward *target_fs*. Integer factor only (safe
    + cheap); returns (decimated, effective_fs). No-op when fs <= target."""
    q = int(fs // target_fs) if target_fs > 0 else 1
    if q <= 1 or sig.size < 30:
        return sig, fs
    try:
        from scipy.signal import decimate
        # decimate in steps of <=13 to keep the FIR filter well-conditioned.
        out, eff = sig, fs
        it = 0
        while q > 1 and out.size > 30 and it < 8:
            it += 1
            step = min(q, 10)
            out = decimate(out, step, ftype="fir", zero_phase=True)
            eff /= step
            q //= step
        return np.asarray(out, dtype=float), eff
    except Exception:  # noqa: BLE001 -- fall back to raw on any DSP hiccup
        return sig, fs


def feature_trajectory(signal: np.ndarray, fs: float, feature: FeatureSpec,
                        step_sec: float, target_fs: float = 500.0) -> np.ndarray:
    """Decimate *signal* toward *target_fs*, cut it into non-overlapping
    ``step_sec`` windows, and apply ``feature.fn(window, fs)`` per window ->
    the 1-D feature trajectory (cadence 1/step_sec Hz)."""
    assert step_sec > 0, "step_sec must be > 0"
    sig = np.asarray(signal, dtype=float)
    sig, eff_fs = _decimate_to(sig, float(fs), float(target_fs))
    win = int(round(step_sec * eff_fs))
    if win < 2 or sig.size < win:
        return np.array([], dtype=float)
    n = sig.size // win
    traj = np.empty(n, dtype=float)
    for i in range(n):
        w = sig[i * win:(i + 1) * win]
        traj[i] = float(feature.fn(w, eff_fs, **feature.params))
    return traj


def gather_leadup_signal(store, seizure, ceiling_sec: float,
                          channel_index: int, max_gap_sec: float = 120.0):
    """Concatenate the channel signal over ``[onset - ceiling, onset]`` across
    every chunk file of the seizure's session that overlaps it. Returns
    ``(signal_1d, fs)`` or ``(None, None)`` when nothing is available or a gap
    larger than *max_gap_sec* leaves the lead-up too incomplete to trust."""
    onset = seizure.onset_epoch
    start = onset - float(ceiling_sec)
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT file_path, chunk_datetime, duration_sec
               FROM processed_files
               WHERE session_dir = ? AND chunk_datetime IS NOT NULL
               ORDER BY chunk_datetime ASC""",
            (seizure.session_dir,)).fetchall()
    from src.utils.chunk_cache import get_chunk
    segments: list[np.ndarray] = []
    fs_out: float | None = None
    covered_until: float | None = None
    for r in rows:
        dt = parse_chunk_datetime(r["chunk_datetime"])
        if dt is None:
            continue
        f_start = dt.timestamp()
        f_end = f_start + float(r["duration_sec"] or 0.0)
        ov0, ov1 = max(start, f_start), min(onset, f_end)
        if ov1 <= ov0:
            continue
        if covered_until is not None and (ov0 - covered_until) > max_gap_sec:
            return None, None                      # lead-up too gappy -> skip
        try:
            chunk = get_chunk(r["file_path"])
        except Exception:  # noqa: BLE001
            continue
        if channel_index < 0 or channel_index >= chunk.signal.shape[1]:
            continue
        fs = float(chunk.fs)
        fs_out = fs_out or fs
        s0 = max(0, int(round((ov0 - f_start) * fs)))
        s1 = min(chunk.signal.shape[0], int(round((ov1 - f_start) * fs)))
        if s1 > s0:
            segments.append(np.asarray(chunk.signal[s0:s1, channel_index],
                                       dtype=float))
            covered_until = ov1
    if not segments:
        return None, None
    return np.concatenate(segments), fs_out
