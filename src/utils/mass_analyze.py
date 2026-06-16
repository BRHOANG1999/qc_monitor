"""PI Mass Analyze: bulk Hilbert/peakseek pre-screen.

The PI picks an animal + a cutoff threshold on the 20-200 Hz
Hilbert envelope. The worker walks every queue-eligible file
for that animal, computes the envelope (or reads the cached
peak count), and partitions files into:

* **zero peaks above threshold** -- on the lab's own BHZ
  detection rule there's nothing behavioural to score. These
  get a bulk ``status='pending_pi_review'`` + ``markers=[]``
  write on confirm, and the PI clears them in the normal
  Event Verification list with one click.
* **≥1 peak above threshold** -- stay in the undergrad queue
  for normal scoring.

Module owns:
* ``FilePeakResult`` dataclass + the cache-aside
  ``get_or_compute_peak_count``.
* ``pending_files_for_animal`` (queue-eligible files for one
  animal).
* ``scan_for_animal`` (the worker's per-job entry point;
  writes live progress to ``mass_analyze_job``).
* ``commit_threshold`` (flips zero-peak files to
  ``pending_pi_review`` via ``store.mark_review``).
* ``start_worker`` (idempotent daemon mirror of
  ``assignments.start_warmer``).

Reuses:
* ``hilbert_envelope_20_200`` -- the FFT bandpass + butter
  smoothing port.
* ``peakseek`` -- the local-maxima + height + min-distance
  pruner.
* ``chunk_cache.get_chunk`` -- so adjacent scans of the same
  .mat don't re-read off the SMB share.
* ``store.mark_review`` -- single-row writer the commit step
  calls in a loop (already handles the ``pending_pi_review``
  status post commit `b682621`).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.chunk_cache import get_chunk
from src.utils.hilbert_envelope import (
    hilbert_envelope_20_200,
    windowed_auc,
)
from src.utils.peakseek import peakseek
from src.utils import stim_blank

logger = logging.getLogger("qc_monitor.utils.mass_analyze")


# BHZ minimum peak distance, in seconds. Matches the
# behavioral_seizure_detection.m:237 hardcoded targetfs*30*5.
DEFAULT_MIN_PEAK_DIST_SEC: float = 150.0

# Stim-blank window for the scan -- captured from config at
# start_worker so the scan blanks exactly like the Video Review trace.
# Defaults match the dashboard's feature_analysis.* defaults.
_BLANK_PRE_MS: float = stim_blank.BLANK_PRE_MS
_BLANK_POST_MS: float = stim_blank.BLANK_POST_MS

# Per-file work (SMB read + FFT) is independent across files, so the
# scan/benchmark loops run on a bounded thread pool. numpy releases the
# GIL during the FFT/filtfilt and the SMB reads overlap, so threads give
# real speedup without the per-process chunk-memory blowup of processes.
# Bounded (not os.cpu_count) so we don't thrash the SMB share. Captured
# from config at start_worker.
_SCAN_WORKERS: int = 4


def _run_file_pool(files: list, work_fn, cancel_fn, on_result,
                     *, workers: int) -> bool:
    """Map *work_fn* over *files* on a bounded thread pool.

    ``on_result(file, result)`` is invoked in THIS (draining) thread as
    each file completes, so tallies + the job-row progress write stay
    single-threaded and race-free. ``cancel_fn() -> bool`` is checked
    between completions; on cancel, pending futures are dropped and the
    function returns True (running ones finish in the background -- they
    only warm the cache). ``work_fn`` must catch its own per-file errors
    and return a sentinel rather than raising.
    """
    n = len(files)
    if n == 0:
        return False
    ex = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    futs = {ex.submit(work_fn, f): f for f in files}
    cancelled = False
    try:
        max_iter = n + 1
        for i, fut in enumerate(as_completed(futs)):
            assert i < max_iter, "pool drain runaway"
            if cancel_fn():
                cancelled = True
                break
            on_result(futs[fut], fut.result())
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return cancelled


def _blank_channel_series(store, file_id: int, channel: int,
                            series, fs: float,
                            session_dir: str | None):
    """Stim-blank one channel's series the same way the trace does.

    Skips blanking on stim-copy channels (they record the stimulator
    output). Reuses src/utils/stim_blank so the scan and the displayed
    Hilbert/AUC trace operate on an identical signal.
    """
    if channel in stim_blank.stim_copy_channels(store, session_dir):
        return series
    times = stim_blank.stim_times_for_file(store, file_id)
    return stim_blank.blank_series_with_stim_times(
        series, fs, times, _BLANK_PRE_MS, _BLANK_POST_MS)


@dataclass(frozen=True)
class FilePeakResult:
    """One peakseek result for one (file, channel, cutoff)."""
    file_id: int
    channel: int
    cutoff: float
    n_peaks: int
    peak_times: list[float]


@dataclass(frozen=True)
class AucResult:
    """One AUC-screen peakseek result for one
    (file, channel, auc_threshold, window_sec)."""
    file_id: int
    channel: int
    auc_threshold: float
    window_sec: float
    n_events: int


# --------------------------------------------------------------- #
# Channel picker (shared with the PI detail panel heuristic)
# --------------------------------------------------------------- #

def first_animal_channel_index(store, session_dir: str) -> int:
    """Pick the first ``is_animal_channel`` from session_config.

    Matches the PI detail-panel default in
    ``event_verification.py``. Falls back to channel 0 when
    nothing animal-shaped is found (defensive; we'd rather
    scan a wrong channel than skip the file silently).
    """
    if not session_dir:
        return 0
    names = store._channel_names_for_session(session_dir)
    max_iter = 64  # NASA Rule 2
    for i, name in enumerate(names):
        assert i < max_iter, "channel scan runaway"
        if isinstance(name, str) and is_animal_channel(name):
            return i
    return 0


def animal_channel_index(store, session_dir: str, animal_id: str,
                           electrode: int = 0) -> int:
    """Channel index of *animal_id*'s *electrode*-th contact.

    Cages hold two animals per session, so the scan must NOT just grab
    the first animal channel (that may belong to the other animal).
    Reuses ``store.electrodes_for_animal_in_session`` -- the same
    ordered, animal-filtered electrode list the Video Review channel
    picker uses -- so the scan channel matches the trace's default.
    *electrode* is an ordinal (0 = first contact), clamped to what the
    animal has. Falls back to ``first_animal_channel_index`` when no
    channel parses to this animal.
    """
    if session_dir and animal_id:
        elecs = store.electrodes_for_animal_in_session(
            session_dir, animal_id)
        if elecs:
            e = max(0, min(int(electrode or 0), len(elecs) - 1))
            return int(elecs[e]["channel_index"])
    return first_animal_channel_index(store, session_dir)


# --------------------------------------------------------------- #
# Cache-aside peakseek
# --------------------------------------------------------------- #

def get_or_compute_peak_count(store, file_id: int,
                                 channel: int, cutoff: float,
                                 *,
                                 min_peak_dist_sec: float =
                                     DEFAULT_MIN_PEAK_DIST_SEC,
                                 ) -> FilePeakResult:
    """Read ``envelope_peak_cache``; on miss, load + Hilbert +
    peakseek + write cache. NASA Rule 5: asserts at boundaries.
    """
    assert isinstance(file_id, int), "file_id must be int"
    assert isinstance(channel, int) and channel >= 0, \
        "channel must be >= 0"
    assert isinstance(cutoff, (int, float)) and cutoff > 0, \
        "cutoff must be > 0"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT n_peaks, peak_times_json
               FROM envelope_peak_cache
               WHERE file_id = ? AND channel = ?
                 AND cutoff = ? AND min_peak_dist_sec = ?""",
            (file_id, channel, float(cutoff),
             float(min_peak_dist_sec)),
        ).fetchone()
    if row is not None:
        try:
            times = list(json.loads(row["peak_times_json"]
                                       or "[]"))
        except json.JSONDecodeError:
            times = []
        return FilePeakResult(
            file_id=file_id, channel=channel,
            cutoff=float(cutoff),
            n_peaks=int(row["n_peaks"]), peak_times=times,
        )
    # Cache miss: compute.
    result = _compute_peak_count_uncached(
        store, file_id, channel, cutoff, min_peak_dist_sec)
    _write_cache(store, result, min_peak_dist_sec)
    return result


def _compute_peak_count_uncached(store, file_id: int,
                                    channel: int,
                                    cutoff: float,
                                    min_peak_dist_sec: float
                                    ) -> FilePeakResult:
    file_row = store.file_row(file_id)
    if not file_row or not file_row.get("file_path"):
        return FilePeakResult(
            file_id=file_id, channel=channel,
            cutoff=float(cutoff), n_peaks=0,
            peak_times=[])
    chunk = get_chunk(file_row["file_path"])
    fs = float(chunk.fs)
    if chunk.signal.ndim != 2 or chunk.signal.shape[1] <= channel:
        logger.debug(
            "mass_analyze: ch=%d out of range for file %s "
            "(shape=%s)", channel, file_id, chunk.signal.shape)
        return FilePeakResult(
            file_id=file_id, channel=channel,
            cutoff=float(cutoff), n_peaks=0,
            peak_times=[])
    series = chunk.signal[:, channel].astype(
        np.float32, copy=False)
    series = _blank_channel_series(
        store, file_id, channel, series, fs,
        file_row.get("session_dir"))
    env = hilbert_envelope_20_200(series, fs)
    min_peak_dist = max(1, int(round(fs * min_peak_dist_sec)))
    locs, _heights = peakseek(
        env, minpeakdist=min_peak_dist, minpeakh=float(cutoff),
    )
    times = (locs.astype(np.float64) / fs).tolist()
    return FilePeakResult(
        file_id=file_id, channel=channel,
        cutoff=float(cutoff), n_peaks=len(times),
        peak_times=times,
    )


def _write_cache(store, result: FilePeakResult,
                   min_peak_dist_sec: float) -> None:
    now = datetime.now().isoformat()
    with store.connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO envelope_peak_cache
               (file_id, channel, cutoff, min_peak_dist_sec,
                n_peaks, peak_times_json, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (result.file_id, result.channel,
             float(result.cutoff), float(min_peak_dist_sec),
             int(result.n_peaks),
             json.dumps(result.peak_times), now),
        )
        conn.commit()


# --------------------------------------------------------------- #
# File enumeration
# --------------------------------------------------------------- #

def pending_files_for_animal(store, animal_id: str
                                ) -> list[dict]:
    """Queue-eligible files for *animal_id*.

    "Eligible" = has a session_config, the animal id appears
    in its channel_names, AND either has no review_state row
    or its latest status is ``pi_flagged``. Files already in
    pending_pi_review / pi_approved / no_events / has_events /
    abandoned are excluded -- mass-analyze doesn't touch
    things that already have a state.
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT pf.id AS file_id,
                      pf.session_dir, pf.file_path,
                      pf.duration_sec, pf.sampling_rate,
                      pf.chunk_datetime
               FROM processed_files pf
               JOIN session_config sc
                 ON sc.session_dir = pf.session_dir
               WHERE sc.channel_names LIKE ?
                 AND NOT EXISTS (
                   SELECT 1 FROM review_state rs
                   WHERE rs.file_id = pf.id
                     AND rs.status IN (
                         'claimed', 'no_events', 'has_events',
                         'abandoned', 'pending_pi_review',
                         'pi_approved'
                     )
                 )
                 -- Don't pre-screen a file someone is actively
                 -- reviewing (claimed within the TTL); auto-clearing
                 -- it would race the human's submission.
                 AND NOT EXISTS (
                   SELECT 1 FROM file_claim fc
                   WHERE fc.file_id = pf.id
                     AND fc.claimed_at >= ?
                 )
               ORDER BY pf.chunk_datetime ASC""",
            (f'%"{animal_id}%', store.claim_cutoff_iso()),
        ).fetchall()
    # Post-filter: the LIKE is loose ("BCH062" would also match
    # "BCH0620"); use the parser to be precise.
    out: list[dict] = []
    max_iter = 4096
    for i, r in enumerate(rows):
        assert i < max_iter, "row scan runaway"
        names = store._channel_names_for_session(r["session_dir"])
        match = False
        for n in names:
            if not isinstance(n, str):
                continue
            if not is_animal_channel(n):
                continue
            a, _ = split_animal_electrode(n)
            if a == animal_id:
                match = True
                break
        if match:
            out.append(dict(r))
    return out


def auc_event_count(env: np.ndarray, fs: float,
                      window_sec: float,
                      auc_threshold: float) -> int:
    """Count sustained-elevation events in *env* via the
    sliding-window AUC.

    A real behavioural seizure is a *prolonged* envelope hump; a
    thin noise spike -- however tall -- integrates to almost
    nothing over the window. We threshold ``windowed_auc`` and
    count its peaks, spacing them one window apart so a single
    plateau is counted once.

    NASA Rule 5: asserts at the boundary.
    """
    assert isinstance(env, np.ndarray), "env must be ndarray"
    assert env.ndim == 1, "env must be 1-D"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    assert window_sec > 0, "window_sec > 0"
    assert auc_threshold > 0, "auc_threshold > 0"
    if env.shape[0] == 0:
        return 0
    return _count_auc_peaks(
        env, fs, window_sec, auc_threshold,
        min_peak_dist_sec=window_sec)


def _count_auc_peaks(env: np.ndarray, fs: float,
                       window_sec: float, auc_threshold: float,
                       *, min_peak_dist_sec: float) -> int:
    """Shared body: peakseek on the sliding-window AUC trace."""
    if env.shape[0] == 0:
        return 0
    auc = windowed_auc(env, fs, window_sec)
    min_dist = max(1, int(round(float(min_peak_dist_sec) * float(fs))))
    locs, _heights = peakseek(
        auc, minpeakdist=min_dist, minpeakh=float(auc_threshold))
    return int(len(locs))


# --------------------------------------------------------------- #
# Cache-aside AUC screen (sibling of get_or_compute_peak_count)
# --------------------------------------------------------------- #

def get_or_compute_auc_count(store, file_id: int, channel: int,
                               auc_threshold: float,
                               window_sec: float,
                               *,
                               min_peak_dist_sec: float =
                                   DEFAULT_MIN_PEAK_DIST_SEC,
                               ) -> AucResult:
    """Read ``envelope_auc_cache``; on miss compute envelope ->
    windowed_auc -> peakseek and write the cache. Mirrors
    ``get_or_compute_peak_count``."""
    assert isinstance(file_id, int), "file_id must be int"
    assert isinstance(channel, int) and channel >= 0, \
        "channel must be >= 0"
    assert isinstance(auc_threshold, (int, float)) \
        and auc_threshold > 0, "auc_threshold must be > 0"
    assert isinstance(window_sec, (int, float)) and window_sec > 0, \
        "window_sec must be > 0"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT n_events FROM envelope_auc_cache
               WHERE file_id = ? AND channel = ?
                 AND auc_threshold = ? AND window_sec = ?
                 AND min_peak_dist_sec = ?""",
            (file_id, channel, float(auc_threshold),
             float(window_sec), float(min_peak_dist_sec)),
        ).fetchone()
    if row is not None:
        return AucResult(
            file_id=file_id, channel=channel,
            auc_threshold=float(auc_threshold),
            window_sec=float(window_sec),
            n_events=int(row["n_events"]))
    result = _compute_auc_count_uncached(
        store, file_id, channel, auc_threshold, window_sec,
        min_peak_dist_sec)
    _write_auc_cache(store, result, min_peak_dist_sec)
    return result


def _compute_auc_count_uncached(store, file_id: int, channel: int,
                                  auc_threshold: float,
                                  window_sec: float,
                                  min_peak_dist_sec: float
                                  ) -> AucResult:
    empty = AucResult(
        file_id=file_id, channel=channel,
        auc_threshold=float(auc_threshold),
        window_sec=float(window_sec), n_events=0)
    file_row = store.file_row(file_id)
    if not file_row or not file_row.get("file_path"):
        return empty
    chunk = get_chunk(file_row["file_path"])
    fs = float(chunk.fs)
    if chunk.signal.ndim != 2 or chunk.signal.shape[1] <= channel:
        logger.debug(
            "auc screen: ch=%d out of range for file %s (shape=%s)",
            channel, file_id, chunk.signal.shape)
        return empty
    series = chunk.signal[:, channel].astype(np.float32, copy=False)
    series = _blank_channel_series(
        store, file_id, channel, series, fs,
        file_row.get("session_dir"))
    env = hilbert_envelope_20_200(series, fs)
    n = _count_auc_peaks(
        env, fs, window_sec, auc_threshold,
        min_peak_dist_sec=min_peak_dist_sec)
    return AucResult(
        file_id=file_id, channel=channel,
        auc_threshold=float(auc_threshold),
        window_sec=float(window_sec), n_events=n)


def _write_auc_cache(store, result: AucResult,
                       min_peak_dist_sec: float) -> None:
    now = datetime.now().isoformat()
    with store.connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO envelope_auc_cache
               (file_id, channel, auc_threshold, window_sec,
                min_peak_dist_sec, n_events, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (result.file_id, result.channel,
             float(result.auc_threshold), float(result.window_sec),
             float(min_peak_dist_sec), int(result.n_events), now),
        )
        conn.commit()


# --------------------------------------------------------------- #
# Two independent screens -> Pool 1 / Pool 2 / Pool 3
# --------------------------------------------------------------- #

def pool_files(store, animal_id: str, peak_cutoff: float,
                 auc_threshold: float | None = None,
                 window_sec: float | None = None,
                 *,
                 electrode: int = 0,
                 min_peak_dist_sec: float =
                     DEFAULT_MIN_PEAK_DIST_SEC,
                 ) -> dict:
    """Run the screen(s) over *animal_id*'s pending files.

    Reads the caches only (``envelope_peak_cache`` +
    ``envelope_auc_cache``), no compute, so call this AFTER a scan
    has populated them. Returns three pools::

        {"pool1": [file_id, ...],          # envelope-screen positives
         "pool2": [file_id, ...],          # AUC-screen positives
         "pool3": [{"file_id": id,         # flagged by exactly one
                    "only": "envelope"|"auc"}, ...]}

    pool1 / pool2 are independent (a file may be in both or
    neither); pool3 is their symmetric difference, tagged with
    which screen caught it. When *auc_threshold* / *window_sec* are
    omitted, only the envelope screen runs (pool2 / pool3 empty).
    All keep ``pending_files_for_animal``'s chronological order.
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    assert isinstance(peak_cutoff, (int, float)) and peak_cutoff > 0, \
        "peak_cutoff > 0"
    run_auc = bool(auc_threshold) and bool(window_sec)
    files = pending_files_for_animal(store, animal_id)
    pool1: list[int] = []
    pool2: list[int] = []
    pool3: list[dict] = []
    max_iter = len(files) + 1
    for i, f in enumerate(files):
        assert i < max_iter, "pool split runaway"
        fid = int(f["file_id"])
        ch = animal_channel_index(
            store, f["session_dir"], animal_id, electrode)
        with store.connection() as conn:
            prow = conn.execute(
                """SELECT n_peaks FROM envelope_peak_cache
                   WHERE file_id=? AND channel=? AND cutoff=?
                     AND min_peak_dist_sec=?""",
                (fid, ch, float(peak_cutoff),
                 float(min_peak_dist_sec)),
            ).fetchone()
            arow = None
            if run_auc:
                arow = conn.execute(
                    """SELECT n_events FROM envelope_auc_cache
                       WHERE file_id=? AND channel=?
                         AND auc_threshold=? AND window_sec=?
                         AND min_peak_dist_sec=?""",
                    (fid, ch, float(auc_threshold), float(window_sec),
                     float(min_peak_dist_sec)),
                ).fetchone()
        if prow is None:
            continue  # envelope screen not scanned yet
        env_pos = int(prow["n_peaks"]) > 0
        if env_pos:
            pool1.append(fid)
        if not run_auc:
            continue
        if arow is None:
            continue  # AUC screen not scanned yet
        auc_pos = int(arow["n_events"]) > 0
        if auc_pos:
            pool2.append(fid)
        if env_pos != auc_pos:
            pool3.append({"file_id": fid,
                           "only": "envelope" if env_pos else "auc"})
    return {"pool1": pool1, "pool2": pool2, "pool3": pool3}


def count_pending_for_animal(store, animal_id: str) -> int:
    """Cheap count of queue-eligible files for *animal_id*.

    Same filter as ``pending_files_for_animal`` but stops at
    ``COUNT(DISTINCT pf.id)`` so the UI scope counter doesn't pay
    the post-filter cost just to learn the count. The post-filter
    in ``pending_files_for_animal`` (``split_animal_electrode``)
    can only trim the SQL result, never expand it -- the count
    here is therefore an upper bound. In practice the LIKE pattern
    is tight enough that the two agree on every animal id format
    the lab uses; if the counter ever overcounts it's a small UX
    bug, not a correctness issue (the actual scan still walks the
    precise list).
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT COUNT(DISTINCT pf.id) AS n
               FROM processed_files pf
               JOIN session_config sc
                 ON sc.session_dir = pf.session_dir
               WHERE sc.channel_names LIKE ?
                 AND NOT EXISTS (
                   SELECT 1 FROM review_state rs
                   WHERE rs.file_id = pf.id
                     AND rs.status IN (
                         'claimed', 'no_events', 'has_events',
                         'abandoned', 'pending_pi_review',
                         'pi_approved'
                     )
                 )
                 AND NOT EXISTS (
                   SELECT 1 FROM file_claim fc
                   WHERE fc.file_id = pf.id
                     AND fc.claimed_at >= ?
                 )""",
            (f'%"{animal_id}%', store.claim_cutoff_iso()),
        ).fetchone()
    return int(row["n"]) if row else 0


# --------------------------------------------------------------- #
# Ground truth + labeled-file enumeration (screen benchmark)
# --------------------------------------------------------------- #

# Statuses that carry a human verdict we can score against.
_LABELED_STATUSES: tuple[str, ...] = (
    "has_events", "no_events", "pi_approved")


def file_ground_truth(status: str,
                        markers_json: str | None) -> str | None:
    """Map a latest review_state row to a per-file label.

    Returns ``'pos'`` (truly has >=1 event), ``'neg'`` (truly
    event-free), or ``None`` (no usable human verdict). The PI-
    rollout migration folded legacy has_events/no_events into
    ``pi_approved``, so that status is scored by whether its
    marker list is non-empty. ``pending_pi_review`` is excluded:
    a chunk of those are Mass-Analyze auto-clears created by the
    envelope screen itself, so counting them would be circular.
    """
    if status == "has_events":
        return "pos"
    if status == "no_events":
        return "neg"
    if status == "pi_approved":
        return "pos" if _markers_nonempty(markers_json) else "neg"
    return None


def _markers_nonempty(markers_json: str | None) -> bool:
    if not markers_json:
        return False
    try:
        data = json.loads(markers_json)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(data, list) and len(data) > 0


def labeled_files_for_animal(store, animal_id: str) -> list[dict]:
    """Human-reviewed files for *animal_id*, with a ground-truth
    label. The inverse of ``pending_files_for_animal``: keep only
    files whose latest review_state is a labeled status, then
    derive ``truth`` via :func:`file_ground_truth`. Rows with no
    usable verdict are dropped.

    Each dict: ``{file_id, session_dir, truth}``.
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    with store.connection() as conn:
        rows = conn.execute(
            """SELECT pf.id AS file_id, pf.session_dir,
                      rs.status, rs.markers_json
               FROM processed_files pf
               JOIN session_config sc
                 ON sc.session_dir = pf.session_dir
               JOIN review_state rs ON rs.file_id = pf.id
               WHERE sc.channel_names LIKE ?
                 AND rs.status IN ('has_events', 'no_events',
                                    'pi_approved')
                 -- latest row per file only
                 AND rs.updated_at = (
                   SELECT MAX(rs2.updated_at) FROM review_state rs2
                   WHERE rs2.file_id = pf.id)
               ORDER BY pf.chunk_datetime ASC""",
            (f'%"{animal_id}%',),
        ).fetchall()
    out: list[dict] = []
    max_iter = 8192
    for i, r in enumerate(rows):
        assert i < max_iter, "labeled scan runaway"
        # Precise animal match (the LIKE is loose).
        names = store._channel_names_for_session(r["session_dir"])
        match = False
        for n in names:
            if not isinstance(n, str) or not is_animal_channel(n):
                continue
            a, _ = split_animal_electrode(n)
            if a == animal_id:
                match = True
                break
        if not match:
            continue
        truth = file_ground_truth(r["status"], r["markers_json"])
        if truth is None:
            continue
        out.append({"file_id": int(r["file_id"]),
                     "session_dir": r["session_dir"],
                     "truth": truth})
    return out


def screen_file(store, file_id: int, channel: int,
                  peak_cutoff: float, auc_threshold: float,
                  window_sec: float,
                  *,
                  min_peak_dist_sec: float =
                      DEFAULT_MIN_PEAK_DIST_SEC,
                  ) -> tuple[bool, bool]:
    """Run both screens on one file (cache-aside). Returns
    ``(env_pos, auc_pos)`` -- whether each screen flagged >=1
    detection."""
    assert isinstance(file_id, int), "file_id must be int"
    peak = get_or_compute_peak_count(
        store, file_id, channel, peak_cutoff,
        min_peak_dist_sec=min_peak_dist_sec)
    auc = get_or_compute_auc_count(
        store, file_id, channel, auc_threshold, window_sec,
        min_peak_dist_sec=min_peak_dist_sec)
    return (peak.n_peaks > 0, auc.n_events > 0)


# --------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------- #

def create_job(store, pi_email: str, animal_id: str,
                 cutoff: float,
                 auc_threshold: float | None = None,
                 auc_window_sec: float | None = None,
                 electrode: int = 0) -> int:
    """Insert a 'pending' job; returns the row id. The daemon
    worker picks it up. ``auc_threshold`` / ``auc_window_sec``
    drive the second (AUC) screen so the scan populates both
    pools; omit them to scan the envelope screen only.
    ``electrode`` selects which of the animal's contacts to scan
    (0 = first)."""
    assert pi_email, "pi_email required"
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    assert isinstance(cutoff, (int, float)) and cutoff > 0, \
        "cutoff > 0"
    now = datetime.now().isoformat()
    with store.connection() as conn:
        cur = conn.execute(
            """INSERT INTO mass_analyze_job
               (pi_email, animal_id, cutoff, auc_threshold,
                auc_window_sec, electrode, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (pi_email.lower(), animal_id, float(cutoff),
             float(auc_threshold) if auc_threshold else None,
             float(auc_window_sec) if auc_window_sec else None,
             int(electrode or 0), now),
        )
        conn.commit()
        job_id = int(cur.lastrowid)
    _wake.set()
    return job_id


def get_job(store, job_id: int) -> dict | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM mass_analyze_job WHERE id = ?",
            (int(job_id),),
        ).fetchone()
    return dict(row) if row else None


def active_job_for_animal(store, animal_id: str) -> dict | None:
    """Most-recent still-active (pending/running) scan for
    *animal_id*, or None.

    The worker runs server-side in a daemon thread, so a scan
    started before a tab reload keeps going. This lets the UI
    reattach its progress poll on reload instead of showing a
    blank panel while work is silently happening in the
    background. Only pending/running jobs qualify -- a finished
    'done' job isn't surfaced here (it would re-show a stale
    Confirm button); re-running the scan is instant off the
    cache if the user needs the Confirm step back.
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT * FROM mass_analyze_job
               WHERE animal_id = ?
                 AND status IN ('pending', 'running')
               ORDER BY created_at DESC LIMIT 1""",
            (animal_id,),
        ).fetchone()
    return dict(row) if row else None


def last_done_job_for_animal(store, animal_id: str) -> dict | None:
    """Most-recent COMPLETED scan for *animal_id*, or None.

    Lets the UI restore the pool browser after a tab reload: the scan's
    cache rows persist, so re-running ``pool_files`` at this job's
    thresholds rebuilds the exact same pools with no recompute.
    """
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    with store.connection() as conn:
        row = conn.execute(
            """SELECT * FROM mass_analyze_job
               WHERE animal_id = ? AND status = 'done'
               ORDER BY finished_at DESC, created_at DESC LIMIT 1""",
            (animal_id,),
        ).fetchone()
    return dict(row) if row else None


def cancel_job(store, job_id: int) -> bool:
    """Flip a pending/running job to 'cancelled'. The worker
    checks this between files and bails cleanly."""
    with store.connection() as conn:
        cur = conn.execute(
            """UPDATE mass_analyze_job
               SET status='cancelled',
                   finished_at=datetime('now')
               WHERE id = ?
                 AND status IN ('pending', 'running')""",
            (int(job_id),),
        )
        conn.commit()
        return cur.rowcount > 0


def scan_for_animal(store, job_id: int,
                      *,
                      min_peak_dist_sec: float =
                          DEFAULT_MIN_PEAK_DIST_SEC,
                      ) -> dict:
    """The worker's per-job entry. Updates the job row's
    ``scanned_files`` / ``n_zero_peaks`` / ``n_with_peaks``
    after EACH file so the UI poll sees live progress.

    Returns the final summary dict (mirrors the final job row).
    """
    job = get_job(store, job_id)
    if not job:
        return {}
    animal_id = job["animal_id"]
    cutoff = float(job["cutoff"])
    # The AUC screen runs alongside the envelope screen when the
    # job carries an AUC threshold + window. Its only job here is
    # to warm envelope_auc_cache so pool_files can read both
    # screens; the progress counters stay envelope-based.
    auc_threshold = job.get("auc_threshold")
    auc_window = job.get("auc_window_sec")
    run_auc = bool(auc_threshold) and bool(auc_window)
    electrode = int(job.get("electrode") or 0)
    # Flip to running.
    now = datetime.now().isoformat()
    with store.connection() as conn:
        conn.execute(
            """UPDATE mass_analyze_job
               SET status='running', started_at=?
               WHERE id=?""",
            (now, job_id),
        )
        conn.commit()
    files = pending_files_for_animal(store, animal_id)
    total = len(files)
    with store.connection() as conn:
        conn.execute(
            """UPDATE mass_analyze_job
               SET total_files=? WHERE id=?""",
            (total, job_id),
        )
        conn.commit()
    n_zero = 0
    n_with = 0
    n_auc_pos = 0      # Pool 2 (AUC-screen positives)
    n_disagree = 0     # Pool 3 (flagged by exactly one screen)
    done = 0

    def _work(f):
        channel = animal_channel_index(
            store, f["session_dir"], animal_id, electrode)
        npk = None
        auc_pos = None
        try:
            npk = get_or_compute_peak_count(
                store, int(f["file_id"]), channel, cutoff,
                min_peak_dist_sec=min_peak_dist_sec).n_peaks
        except Exception as e:
            logger.warning("mass_analyze file=%s ch=%s failed: %s",
                            f["file_id"], channel, e)
        if run_auc:
            try:
                auc_pos = get_or_compute_auc_count(
                    store, int(f["file_id"]), channel,
                    float(auc_threshold), float(auc_window),
                    min_peak_dist_sec=min_peak_dist_sec).n_events > 0
            except Exception as e:
                logger.warning("auc screen file=%s ch=%s failed: %s",
                                f["file_id"], channel, e)
        return (npk, auc_pos)

    def _record(_f, res):
        nonlocal n_zero, n_with, n_auc_pos, n_disagree, done
        npk, auc_pos = res
        # Pool 1 (envelope): None (scan error -> safer 'with') or >0.
        env_pos = (npk is None) or (npk > 0)
        if npk == 0:
            n_zero += 1
        else:
            n_with += 1
        # Pool 2 (AUC) + Pool 3 (disagreement), only when AUC ran.
        if auc_pos:
            n_auc_pos += 1
        if auc_pos is not None and env_pos != bool(auc_pos):
            n_disagree += 1
        done += 1
        with store.connection() as conn:
            conn.execute(
                """UPDATE mass_analyze_job SET scanned_files=?,
                   n_zero_peaks=?, n_with_peaks=?,
                   n_auc_pos=?, n_disagree=? WHERE id=?""",
                (done, n_zero, n_with, n_auc_pos, n_disagree, job_id))
            conn.commit()

    cancelled = _run_file_pool(
        files, _work,
        lambda: _job_status(store, job_id) == "cancelled",
        _record, workers=_SCAN_WORKERS)
    if cancelled:
        logger.info("mass_analyze job %s cancelled at %d/%d",
                     job_id, done, total)
        return {"status": "cancelled", "scanned_files": done,
                 "total_files": total, "n_zero_peaks": n_zero,
                 "n_with_peaks": n_with}
    # Done.
    with store.connection() as conn:
        conn.execute(
            """UPDATE mass_analyze_job
               SET status='done', finished_at=?
               WHERE id=?""",
            (datetime.now().isoformat(), job_id),
        )
        conn.commit()
    return {"status": "done",
             "scanned_files": total,
             "total_files": total,
             "n_zero_peaks": n_zero,
             "n_with_peaks": n_with}


def _job_status(store, job_id: int) -> str:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT status FROM mass_analyze_job WHERE id=?",
            (int(job_id),),
        ).fetchone()
    return row["status"] if row else ""


# --------------------------------------------------------------- #
# Commit: turn zero-peak files into pending_pi_review submissions
# --------------------------------------------------------------- #

def commit_threshold(store, animal_id: str, cutoff: float,
                       pi_email: str,
                       *,
                       electrode: int = 0,
                       min_peak_dist_sec: float =
                           DEFAULT_MIN_PEAK_DIST_SEC,
                       ) -> dict:
    """Mark every file in *animal_id*'s pending pool whose
    cached peak count at *cutoff* is zero as
    ``pending_pi_review`` with ``markers=[]``.

    Returns ``{n_cleared, n_skipped, file_ids_cleared,
    file_ids_skipped}``. Skipped files either had >=1 peak,
    weren't in the cache (the scan missed them), or already
    advanced to a finalised status since the scan.
    """
    assert isinstance(animal_id, str) and animal_id
    assert isinstance(cutoff, (int, float)) and cutoff > 0
    assert pi_email, "pi_email required"
    files = pending_files_for_animal(store, animal_id)
    cleared: list[int] = []
    skipped: list[int] = []
    max_iter = len(files) + 1
    for i, f in enumerate(files):
        assert i < max_iter, "commit loop runaway"
        fid = int(f["file_id"])
        ch = animal_channel_index(
            store, f["session_dir"], animal_id, electrode)
        with store.connection() as conn:
            row = conn.execute(
                """SELECT n_peaks FROM envelope_peak_cache
                   WHERE file_id=? AND channel=? AND cutoff=?
                     AND min_peak_dist_sec=?""",
                (fid, ch, float(cutoff),
                 float(min_peak_dist_sec)),
            ).fetchone()
        if row is None or int(row["n_peaks"]) > 0:
            skipped.append(fid)
            continue
        try:
            store.mark_review(
                fid, pi_email, "pending_pi_review",
                markers=[],
                note=(f"Mass Analyze auto-clear at "
                       f"cutoff={cutoff:g}"),
            )
            store.insert_review_event(
                fid, pi_email, "mass_analyze_clear",
                {"cutoff": float(cutoff),
                 "channel": ch,
                 "animal_id": animal_id,
                 "min_peak_dist_sec":
                     float(min_peak_dist_sec)},
            )
            cleared.append(fid)
        except Exception as e:
            logger.warning(
                "commit_threshold mark_review failed "
                "file=%s: %s", fid, e)
            skipped.append(fid)
    return {
        "n_cleared": len(cleared),
        "n_skipped": len(skipped),
        "file_ids_cleared": cleared,
        "file_ids_skipped": skipped,
    }


# --------------------------------------------------------------- #
# Screen-comparison benchmark (two screens vs human labels)
# --------------------------------------------------------------- #

def create_screen_eval_job(store, pi_email: str, animal_id: str,
                             peak_cutoff: float,
                             auc_threshold: float,
                             auc_window_sec: float,
                             electrode: int = 0) -> int:
    """Insert a 'pending' benchmark job; returns the row id."""
    assert pi_email, "pi_email required"
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    assert peak_cutoff > 0 and auc_threshold > 0 \
        and auc_window_sec > 0, "thresholds/window must be > 0"
    now = datetime.now().isoformat()
    with store.connection() as conn:
        cur = conn.execute(
            """INSERT INTO screen_eval_job
               (pi_email, animal_id, peak_cutoff, auc_threshold,
                auc_window_sec, electrode, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (pi_email.lower(), animal_id, float(peak_cutoff),
             float(auc_threshold), float(auc_window_sec),
             int(electrode or 0), now),
        )
        conn.commit()
        job_id = int(cur.lastrowid)
    _wake.set()
    return job_id


def get_screen_eval_job(store, job_id: int) -> dict | None:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT * FROM screen_eval_job WHERE id = ?",
            (int(job_id),),
        ).fetchone()
    return dict(row) if row else None


def cancel_screen_eval_job(store, job_id: int) -> bool:
    with store.connection() as conn:
        cur = conn.execute(
            """UPDATE screen_eval_job
               SET status='cancelled',
                   finished_at=datetime('now')
               WHERE id = ? AND status IN ('pending', 'running')""",
            (int(job_id),),
        )
        conn.commit()
        return cur.rowcount > 0


def _screen_eval_status(store, job_id: int) -> str:
    with store.connection() as conn:
        row = conn.execute(
            "SELECT status FROM screen_eval_job WHERE id=?",
            (int(job_id),),
        ).fetchone()
    return row["status"] if row else ""


def run_screen_benchmark(store, job_id: int,
                           *,
                           min_peak_dist_sec: float =
                               DEFAULT_MIN_PEAK_DIST_SEC,
                           ) -> dict:
    """Worker entry: run both screens over the animal's labeled
    files and accumulate two confusion matrices + the
    disagreement breakdown. Mirrors ``scan_for_animal``.
    """
    job = get_screen_eval_job(store, job_id)
    if not job:
        return {}
    animal_id = job["animal_id"]
    peak_cutoff = float(job["peak_cutoff"])
    auc_threshold = float(job["auc_threshold"])
    auc_window = float(job["auc_window_sec"])
    electrode = int(job.get("electrode") or 0)
    with store.connection() as conn:
        conn.execute(
            """UPDATE screen_eval_job
               SET status='running', started_at=? WHERE id=?""",
            (datetime.now().isoformat(), job_id),
        )
        conn.commit()
    files = labeled_files_for_animal(store, animal_id)
    total = len(files)
    with store.connection() as conn:
        conn.execute(
            "UPDATE screen_eval_job SET total_files=? WHERE id=?",
            (total, job_id),
        )
        conn.commit()
    # 8 confusion counters + 4 disagreement counters.
    c = {k: 0 for k in (
        "p1_tp", "p1_fp", "p1_tn", "p1_fn",
        "p2_tp", "p2_fp", "p2_tn", "p2_fn",
        "env_only_true", "env_only_false",
        "auc_only_true", "auc_only_false")}
    done = 0

    def _work(f):
        channel = animal_channel_index(
            store, f["session_dir"], animal_id, electrode)
        try:
            env_pos, auc_pos = screen_file(
                store, int(f["file_id"]), channel,
                peak_cutoff, auc_threshold, auc_window,
                min_peak_dist_sec=min_peak_dist_sec)
        except Exception as e:
            logger.warning("benchmark file=%s ch=%s failed: %s",
                            f["file_id"], channel, e)
            return None
        return (env_pos, auc_pos, f["truth"] == "pos")

    def _record(_f, res):
        nonlocal done
        if res is None:
            return
        env_pos, auc_pos, truth_pos = res
        _tally(c, "p1", env_pos, truth_pos)
        _tally(c, "p2", auc_pos, truth_pos)
        if env_pos != auc_pos:
            who = "env" if env_pos else "auc"
            c[f"{who}_only_{'true' if truth_pos else 'false'}"] += 1
        done += 1
        with store.connection() as conn:
            conn.execute(
                """UPDATE screen_eval_job SET scanned_files=?,
                   p1_tp=?, p1_fp=?, p1_tn=?, p1_fn=?,
                   p2_tp=?, p2_fp=?, p2_tn=?, p2_fn=?,
                   env_only_true=?, env_only_false=?,
                   auc_only_true=?, auc_only_false=?
                   WHERE id=?""",
                (done, c["p1_tp"], c["p1_fp"], c["p1_tn"],
                 c["p1_fn"], c["p2_tp"], c["p2_fp"], c["p2_tn"],
                 c["p2_fn"], c["env_only_true"],
                 c["env_only_false"], c["auc_only_true"],
                 c["auc_only_false"], job_id),
            )
            conn.commit()

    cancelled = _run_file_pool(
        files, _work,
        lambda: _screen_eval_status(store, job_id) == "cancelled",
        _record, workers=_SCAN_WORKERS)
    if cancelled:
        return {"status": "cancelled", **c}
    with store.connection() as conn:
        conn.execute(
            """UPDATE screen_eval_job
               SET status='done', finished_at=? WHERE id=?""",
            (datetime.now().isoformat(), job_id),
        )
        conn.commit()
    return {"status": "done", "scanned_files": total,
             "total_files": total, **c}


def _tally(c: dict, prefix: str, pred_pos: bool,
             truth_pos: bool) -> None:
    """Increment the right confusion counter for one prediction."""
    if truth_pos:
        c[f"{prefix}_tp" if pred_pos else f"{prefix}_fn"] += 1
    else:
        c[f"{prefix}_fp" if pred_pos else f"{prefix}_tn"] += 1


# --------------------------------------------------------------- #
# Daemon worker
# --------------------------------------------------------------- #

_worker_started: bool = False
_worker_lock = threading.Lock()
_wake = threading.Event()


def start_worker(store, config: dict) -> None:
    """Spawn the daemon. Idempotent. Mirrors
    ``assignments.start_warmer`` / ``event_clip.start_worker``."""
    assert config is None or isinstance(config, dict)
    cfg = (config or {}).get("mass_analyze", {}) or {}
    # Capture the stim-blank window so the scan blanks exactly like the
    # Video Review trace (which reads the same config keys).
    fa = (config or {}).get("feature_analysis", {}) or {}
    global _BLANK_PRE_MS, _BLANK_POST_MS, _SCAN_WORKERS
    _BLANK_PRE_MS = float(
        fa.get("stim_artifact_start_ms", stim_blank.BLANK_PRE_MS))
    _BLANK_POST_MS = float(
        fa.get("stim_artifact_end_ms", stim_blank.BLANK_POST_MS))
    try:
        _SCAN_WORKERS = max(1, int(cfg.get("scan_workers", 4)))
    except (TypeError, ValueError):
        _SCAN_WORKERS = 4
    if cfg.get("enabled") is False:  # explicit disable only
        logger.info("mass_analyze worker disabled in config")
        return
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(
        target=_worker_loop, args=(store,),
        daemon=True, name="qc-mass-analyze-worker",
    )
    t.start()
    logger.info("mass_analyze worker started")


def _worker_loop(store) -> None:
    interval = 2.0
    max_iter = 10 ** 9
    i = 0
    while True:
        assert i < max_iter, "worker loop runaway"
        i += 1
        try:
            with store.connection() as conn:
                row = conn.execute(
                    """SELECT id FROM mass_analyze_job
                       WHERE status='pending'
                       ORDER BY created_at ASC LIMIT 1"""
                ).fetchone()
            if row is not None:
                scan_for_animal(store, int(row["id"]))
                continue
            # No scan pending -- look for a benchmark job.
            with store.connection() as conn:
                brow = conn.execute(
                    """SELECT id FROM screen_eval_job
                       WHERE status='pending'
                       ORDER BY created_at ASC LIMIT 1"""
                ).fetchone()
            if brow is not None:
                run_screen_benchmark(store, int(brow["id"]))
                continue
            _wake.wait(timeout=interval)
            _wake.clear()
        except Exception:
            logger.exception(
                "mass_analyze worker iter failed")
            time.sleep(interval)
