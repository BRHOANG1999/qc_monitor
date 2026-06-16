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
import logging
from collections import OrderedDict
from threading import Lock

import numpy as np

logger = logging.getLogger("qc_monitor.utils.stim_blank")

# Canonical defaults -- match the dashboard's
# feature_analysis.stim_artifact_start_ms / _end_ms config keys.
BLANK_PRE_MS: float = -5.0
BLANK_POST_MS: float = 15.0

# stimCopy onset-detection params (fallback when evoked_features is
# empty). The stimCopy channel records the stimulator output directly,
# so pulses are large, brief deflections over a near-zero baseline.
_COPY_K_SIGMA: float = 8.0       # threshold = median + k * robust sigma
_COPY_REFRACTORY_SEC: float = 0.003
_COPY_MIN_EVENTS: int = 3        # fewer -> treat as "no stim"

# Small per-file cache so repeated trace renders of one file don't
# re-detect onsets on the (large) stimCopy channel.
_copy_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
_copy_cache_lock = Lock()
_COPY_CACHE_MAX = 16


def detect_stim_onsets(stimcopy: np.ndarray, fs: float,
                         *,
                         k_sigma: float = _COPY_K_SIGMA,
                         refractory_sec: float = _COPY_REFRACTORY_SEC,
                         min_events: int = _COPY_MIN_EVENTS,
                         ) -> np.ndarray:
    """Detect stim pulse onset times (sec) from a stimCopy channel.

    Robust threshold (median + k * MAD-sigma) on the rectified signal,
    take rising edges, merge anything within *refractory_sec* into one
    onset. Returns an empty array when fewer than *min_events* are found
    or the channel has no large deflections -- so a true no-stim
    baseline blanks nothing.
    """
    assert isinstance(stimcopy, np.ndarray), "stimcopy must be ndarray"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    x = np.abs(np.nan_to_num(stimcopy.astype(np.float64, copy=False)))
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad if mad > 0 else float(x.std() or 1.0)
    thr = med + k_sigma * sigma
    # Require genuinely large deflections (else it's just noise).
    if float(x.max()) < med + 10.0 * sigma:
        return np.zeros(0, dtype=np.float64)
    above = x > thr
    if not above.any():
        return np.zeros(0, dtype=np.float64)
    # Rising edges: above[i] and not above[i-1].
    prev = np.concatenate(([False], above[:-1]))
    edges = np.flatnonzero(above & ~prev)
    refr = max(1, int(round(refractory_sec * fs)))
    onsets: list[int] = []
    last = -refr - 1
    max_iter = edges.shape[0] + 1
    for i, e in enumerate(edges):
        assert i < max_iter, "edge merge runaway"
        if int(e) - last >= refr:
            onsets.append(int(e))
            last = int(e)
    if len(onsets) < min_events:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(onsets, dtype=np.float64) / float(fs)


def _stim_times_from_copy(store, file_id: int,
                            session_dir: str | None) -> np.ndarray:
    """Fallback: detect stim onsets off the session's stimCopy channel
    when the MATLAB catalogue (evoked_features) has none. Reuses the
    already-cached chunk so it adds no extra SMB read in the scan/trace
    paths. Cached per file_id."""
    with _copy_cache_lock:
        hit = _copy_cache.get(file_id)
    if hit is not None:
        return hit
    result = np.zeros(0, dtype=np.float64)
    copies = stim_copy_channels(store, session_dir)
    if copies:
        row = store.file_row(file_id)
        fp = (row or {}).get("file_path")
        if fp:
            try:
                from src.utils.chunk_cache import get_chunk
                chunk = get_chunk(fp)
                sig = chunk.signal
                if sig.ndim == 2:
                    for ci in sorted(copies):
                        if ci < sig.shape[1]:
                            result = detect_stim_onsets(
                                sig[:, ci], float(chunk.fs))
                            if result.size:
                                break
            except Exception as e:
                logger.warning(
                    "stimCopy onset detect failed file=%s: %s",
                    file_id, e)
    with _copy_cache_lock:
        _copy_cache[file_id] = result
        while len(_copy_cache) > _COPY_CACHE_MAX:
            _copy_cache.popitem(last=False)
    return result


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
    """Stim onset times (sec) for *file_id*.

    Primary source is the MATLAB catalogue
    (``evoked_features.epoch_time_sec`` -- one row per detected stim
    event). When that's empty (the evoked pipeline hasn't run for this
    file), fall back to detecting pulses directly off the session's
    stimCopy channel, so blanking still works. Returns an empty array
    only when neither source finds stim.
    """
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT epoch_time_sec
               FROM evoked_features
               WHERE file_id = ? AND epoch_time_sec IS NOT NULL
               ORDER BY epoch_time_sec""",
            (file_id,),
        ).fetchall()
        if not rows:
            sd_row = conn.execute(
                "SELECT session_dir FROM processed_files WHERE id=?",
                (file_id,)).fetchone()
        else:
            sd_row = None
    if rows:
        return np.asarray([r["epoch_time_sec"] for r in rows],
                           dtype=np.float64)
    session_dir = sd_row["session_dir"] if sd_row else None
    return _stim_times_from_copy(store, file_id, session_dir)


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
