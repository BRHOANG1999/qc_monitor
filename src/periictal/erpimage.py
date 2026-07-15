"""Evoked ERP-image: the trials leading into one seizure, stacked as a heatmap.

Columns = individual evoked-response trials ordered by time (rightmost = closest
to onset); rows = samples of the post-stim response window (ms); colour = signed
amplitude. A sliding trial-average (configurable window + overlap) denoises the
single trials and reveals how the evoked waveform evolves approaching the
seizure. This is the ONLY peri-ictal view that reads the raw ``evokedData``
traces (the feature sidecars store scalars, not traces), so the gather is heavy
(~15 s per recording file) and belongs on a background thread.

Reuses ``evoked_output.read_file_evoked`` (traces ``[epochs x samples]``,
``time_ms`` in ms with t=0 = stim) and ``evoked_features.preprocess`` (the same
window-crop + SOS-fixed filtering as the feature variants). The two array
reducers are pure + unit-tested; ``gather_leadup_trials`` is the trace-reading
integration path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.evoked_features import FeatureConfig, preprocess
from src.utils.evoked_output import (animals_in_filename, list_evoked_files,
                                     parse_recording_dt, read_file_evoked)

_MAX_FILES = 2_000_000        # NASA Rule 2.
_MAX_TRIALS = 2_000_000


def sliding_trial_average(trials: np.ndarray, tto: np.ndarray, n: int,
                          step: int):
    """Moving average across ordered trials: each output column is the mean of a
    window of *n* consecutive trials, advanced by *step* (overlap = n - step).

    *trials* is ``[n_trials x n_samples]`` ordered by DESCENDING time-to-onset
    (rightmost = nearest onset). Returns ``(Z[n_samples x n_cols], col_tto)``
    where ``Z`` is samples-by-column (ready for a heatmap) and ``col_tto`` is the
    median time-to-onset of each window. Denoises and bounds the column count."""
    trials = np.asarray(trials, dtype=np.float64)
    tto = np.asarray(tto, dtype=np.float64)
    assert trials.ndim == 2, "trials must be [n_trials x n_samples]"
    assert trials.shape[0] == tto.shape[0], "trials/tto length mismatch"
    nt = trials.shape[0]
    n = int(max(1, min(n, nt)))
    step = int(max(1, step))
    starts = list(range(0, max(1, nt - n + 1), step))
    if starts and starts[-1] != nt - n and nt - n > 0:
        starts.append(nt - n)                       # always include the last window
    cols, col_tto = [], []
    for s in starts:
        assert len(cols) < _MAX_TRIALS, "sliding-average runaway"
        seg = trials[s:s + n]
        cols.append(np.nanmean(seg, axis=0))
        col_tto.append(float(np.nanmedian(tto[s:s + n])))
    z = np.array(cols).T if cols else np.empty((trials.shape[1], 0))
    return z, np.array(col_tto)


def decimate_rows(z: np.ndarray, row_ms: np.ndarray, max_rows: int = 300):
    """Block-MEAN the sample rows to <= *max_rows* (a heatmap cell must be one
    averaged amplitude, not a min/max envelope). Returns ``(z2, row_ms2)``."""
    z = np.asarray(z, dtype=np.float64)
    row_ms = np.asarray(row_ms, dtype=np.float64)
    assert z.ndim == 2, "z must be 2-D [rows x cols]"
    nr = z.shape[0]
    if nr <= int(max_rows) or nr == 0:
        return z, row_ms
    decim = nr // int(max_rows)
    nb = nr // decim
    trim = nb * decim
    z2 = z[:trim].reshape(nb, decim, z.shape[1]).mean(axis=1)
    r2 = row_ms[:trim].reshape(nb, decim).mean(axis=1)
    return z2, r2


def _channel_traces(rec, animal: str):
    """(traces, time_ms, stim_times) for the *animal* channel of one file's
    per-channel dict, or (None, None, None)."""
    for ch, d in rec.items():
        if split_animal_electrode(ch)[0] != animal or not is_animal_channel(ch):
            continue
        tr, tms = d.get("traces"), d.get("time_ms")
        if tr is not None and tms is not None and len(tr) >= 1 and tms.size >= 2:
            return np.asarray(tr, float), np.asarray(tms, float), d.get("times") or []
    return None, None, None


def gather_leadup_trials(evoked_dir: str, animal: str, onset_epoch: float,
                         lookback_sec: float, cfg: FeatureConfig | None = None,
                         progress=None) -> dict:
    """Read the *animal*'s evoked trials in ``[onset - lookback, onset]`` from the
    raw ``*_evoked.mat`` traces, windowed by *cfg*, ordered by descending
    time-to-onset (nearest onset last). Returns ``{"trials": [n_trials x
    n_samples], "row_ms": [n_samples], "tto": [n_trials]}`` (empty arrays when
    nothing is found). Heavy: reads trace matrices; call off the render thread."""
    assert evoked_dir and animal, "evoked_dir and animal required"
    assert lookback_sec > 0, "lookback_sec must be > 0"
    t_lo, t_hi = onset_epoch - lookback_sec, onset_epoch
    files = _candidate_files(evoked_dir, animal, t_lo, t_hi)
    rows_ms = None
    trial_rows, trial_tto = [], []
    for i, fp in enumerate(files):
        if progress:
            progress(i, len(files), fp)
        rec = read_file_evoked(fp, only_animals=[animal])
        tr, tms, times = _channel_traces(rec, animal)
        if tr is None:
            continue
        rec_dt = parse_recording_dt(fp) or datetime.min
        base = rec_dt.timestamp()
        fs = 1000.0 / float(np.mean(np.diff(tms)))
        proc, t_ms = preprocess(tr, tms, fs, cfg or FeatureConfig())
        rows_ms = t_ms if rows_ms is None else rows_ms
        for j in range(proc.shape[0]):
            st = float(times[j]) if j < len(times) else None
            if st is None:
                continue
            abst = base + st
            if t_lo <= abst <= t_hi:
                trial_rows.append(proc[j])
                trial_tto.append(onset_epoch - abst)
    return _order_by_onset(trial_rows, trial_tto, rows_ms)


def _candidate_files(evoked_dir, animal, t_lo, t_hi) -> list:
    """Animal files whose recording start is plausibly within the window (a
    recording is a short chunk; a generous 2 h guard before t_lo covers the file
    that CONTAINS t_lo), oldest first."""
    out = []
    for i, fp in enumerate(list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "evoked file scan runaway"
        if animal not in animals_in_filename(fp):
            continue
        dt = parse_recording_dt(fp)
        if dt is None:
            continue
        ts = dt.timestamp()
        if ts <= t_hi and ts >= t_lo - 7200.0:
            out.append((ts, fp))
    return [fp for _ts, fp in sorted(out)]


def _order_by_onset(trial_rows, trial_tto, rows_ms) -> dict:
    """Sort trials by DESCENDING time-to-onset (nearest onset last = rightmost)."""
    if not trial_rows or rows_ms is None:
        return {"trials": np.empty((0, 0)), "row_ms": np.empty(0),
                "tto": np.empty(0)}
    tto = np.array(trial_tto, dtype=np.float64)
    order = np.argsort(-tto)                         # far first, near-onset last
    trials = np.array(trial_rows, dtype=np.float64)[order]
    return {"trials": trials, "row_ms": np.asarray(rows_ms, float),
            "tto": tto[order]}
