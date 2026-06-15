"""Stim-artifact blanking, shared by the Video Review trace and the
Mass Analyze scan so both compute the Hilbert envelope on the SAME
signal.

The lab blanks a short window around each stim pulse (NaN-fill) before
any 20-200 Hz envelope work; ``hilbert_envelope_20_200`` turns NaN into
0, so the artifact contributes nothing. If the scan skips this step,
stim pulses leak band-limited energy that inflates the envelope (and the
sliding-window AUC), producing false positives the displayed trace never
shows.

This module is dashboard-independent on purpose: ``mass_analyze.py`` is a
util and must not import ``src/dashboard``. The pure-numpy core
(:func:`blank_series_with_stim_times`) plus the two DB lookups live here
so there is exactly one blanking implementation.
"""

from __future__ import annotations

import json

import numpy as np

# Canonical defaults -- match the dashboard's
# feature_analysis.stim_artifact_start_ms / _end_ms config keys.
BLANK_PRE_MS: float = -5.0
BLANK_POST_MS: float = 15.0


def blank_series_with_stim_times(series: np.ndarray, fs: float,
                                   stim_times: np.ndarray | None,
                                   blank_pre_ms: float = BLANK_PRE_MS,
                                   blank_post_ms: float = BLANK_POST_MS,
                                   ) -> np.ndarray:
    """Return a copy of *series* with a window around each stim pulse
    NaN-blanked.

    The window is ``[center + pre, center + post)`` in samples, where
    ``center = round(t_sec * fs)`` and pre/post come from the ms params
    (pre is normally negative). Bounds are clipped to the series. NaNs
    are downstream-safe: ``hilbert_envelope_20_200`` zeroes them before
    the FFT.

    NASA Rule 5: asserts at the boundary.
    """
    assert isinstance(series, np.ndarray), "series must be ndarray"
    assert series.ndim == 1, "series must be 1-D"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    out = series.astype(np.float32, copy=True)
    if stim_times is None or len(stim_times) == 0:
        return out
    pre = int(round(blank_pre_ms * 1e-3 * fs))
    post = int(round(blank_post_ms * 1e-3 * fs))
    n = out.shape[0]
    max_iter = len(stim_times) + 1
    for i, t_sec in enumerate(stim_times):
        assert i < max_iter, "stim blanking runaway"
        center = int(round(float(t_sec) * fs))
        lo = max(0, center + pre)
        hi = min(n, center + post)
        if hi > lo:
            out[lo:hi] = np.nan
    return out


def stim_times_for_file(store, file_id: int) -> np.ndarray:
    """Stim onset times (sec) from the MATLAB pipeline.

    Reads ``evoked_features.epoch_time_sec`` -- one row per detected
    stim event. Empty array for files that aren't MATLAB-processed.
    """
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT epoch_time_sec
               FROM evoked_features
               WHERE file_id = ? AND epoch_time_sec IS NOT NULL
               ORDER BY epoch_time_sec""",
            (file_id,),
        ).fetchall()
    return np.asarray([r["epoch_time_sec"] for r in rows],
                       dtype=np.float64)


def stim_copy_channels(store, session_dir: str | None) -> set[int]:
    """Channel indices flagged as stim-copy in the session config.

    These record the stimulator output itself, so blanking them would
    erase the only signal worth seeing -- callers skip blanking on them.
    """
    if not session_dir:
        return set()
    cfg = store.get_session_config(session_dir) or {}
    raw = cfg.get("stim_copy_channels")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, list):
        return {int(c) for c in raw}
    return set()
