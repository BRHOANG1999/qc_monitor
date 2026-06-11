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
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.chunk_cache import get_chunk
from src.utils.hilbert_envelope import hilbert_envelope_20_200
from src.utils.peakseek import peakseek

logger = logging.getLogger("qc_monitor.utils.mass_analyze")


# BHZ minimum peak distance, in seconds. Matches the
# behavioral_seizure_detection.m:237 hardcoded targetfs*30*5.
DEFAULT_MIN_PEAK_DIST_SEC: float = 150.0


@dataclass(frozen=True)
class FilePeakResult:
    """One peakseek result for one (file, channel, cutoff)."""
    file_id: int
    channel: int
    cutoff: float
    n_peaks: int
    peak_times: list[float]


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
               ORDER BY pf.chunk_datetime ASC""",
            (f'%"{animal_id}%',),  # animal prefix match
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
                 )""",
            (f'%"{animal_id}%',),
        ).fetchone()
    return int(row["n"]) if row else 0


# --------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------- #

def create_job(store, pi_email: str, animal_id: str,
                 cutoff: float) -> int:
    """Insert a 'pending' job; returns the row id. The daemon
    worker picks it up."""
    assert pi_email, "pi_email required"
    assert isinstance(animal_id, str) and animal_id, \
        "animal_id required"
    assert isinstance(cutoff, (int, float)) and cutoff > 0, \
        "cutoff > 0"
    now = datetime.now().isoformat()
    with store.connection() as conn:
        cur = conn.execute(
            """INSERT INTO mass_analyze_job
               (pi_email, animal_id, cutoff, status,
                created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (pi_email.lower(), animal_id, float(cutoff), now),
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
    max_iter = total + 1
    for i, f in enumerate(files):
        assert i < max_iter, "scan loop runaway"
        # Cooperative cancel check between files.
        cur_status = _job_status(store, job_id)
        if cur_status == "cancelled":
            logger.info("mass_analyze job %s cancelled at "
                         "%d/%d", job_id, i, total)
            return {"status": "cancelled",
                     "scanned_files": i,
                     "total_files": total,
                     "n_zero_peaks": n_zero,
                     "n_with_peaks": n_with}
        channel = first_animal_channel_index(
            store, f["session_dir"])
        try:
            result = get_or_compute_peak_count(
                store, int(f["file_id"]), channel, cutoff,
                min_peak_dist_sec=min_peak_dist_sec,
            )
        except Exception as e:
            logger.warning(
                "mass_analyze file=%s ch=%s failed: %s",
                f["file_id"], channel, e)
            # Treat scan failure as "with peaks" -- safer
            # (file stays in queue for normal review).
            n_with += 1
        else:
            if result.n_peaks == 0:
                n_zero += 1
            else:
                n_with += 1
        # Live progress update.
        with store.connection() as conn:
            conn.execute(
                """UPDATE mass_analyze_job
                   SET scanned_files=?,
                       n_zero_peaks=?, n_with_peaks=?
                   WHERE id=?""",
                (i + 1, n_zero, n_with, job_id),
            )
            conn.commit()
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
        ch = first_animal_channel_index(
            store, f["session_dir"])
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
            if row is None:
                _wake.wait(timeout=interval)
                _wake.clear()
                continue
            scan_for_animal(store, int(row["id"]))
        except Exception:
            logger.exception(
                "mass_analyze worker iter failed")
            time.sleep(interval)
