"""ffmpeg event-clip extractor + LFP numpy stitcher.

The PI verification flow needs the 5-min-before-EO to
5-min-after-BB window for every flagged event. When the
window crosses a recording boundary, signals stitch from the
adjacent file via ``store.get_adjacent_files``.

Two extraction paths:
* **LFP** (in-process numpy): cheap. Load each segment via
  ``chunk_cache.get_chunk``, slice by sample index, concatenate
  with NaN bridges at boundaries so the Plotly trace shows the
  gap visually.
* **Video** (ffmpeg subprocess + cache): expensive. Concat-
  demuxer with ``-c copy`` so the stream is not re-encoded.
  Falls back to H.264 re-encode if stream-copy fails (keyframe
  alignment can break the copy on some MP4 sources).

Background worker:
* Module-level daemon thread (same pattern as
  ``src/utils/assignments.py:start_warmer``) polls the
  ``event_clip_job`` table for status='pending' rows and
  processes them.
* Cache hits (status='done' rows whose cache_path still
  exists) skip the worker entirely.
* LRU eviction over the cache directory enforces
  ``config.event_clip.cache_max_gb``.

Public API:
  ``ClipSpec``                  -- the request payload.
  ``spec_hash(spec, segments)`` -- cache key.
  ``resolve_clip_segments``     -- walks adjacent files.
  ``extract_lfp_clip``          -- returns (t, signal, fs).
  ``submit_video_clip_job``     -- queues / returns cached.
  ``poll_video_clip_status``    -- for UI status polling.
  ``start_worker``              -- idempotent daemon launcher.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.chunk_cache import get_chunk

logger = logging.getLogger("qc_monitor.utils.event_clip")


# --------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------- #

@dataclass(frozen=True)
class ClipSpec:
    """A request to extract one event window."""
    file_id: int
    eo_sec: float
    bb_sec: float
    pre_sec: float = 300.0   # 5 min before EO
    post_sec: float = 300.0  # 5 min after BB


@dataclass(frozen=True)
class Segment:
    """One contiguous slice of one file's signal / video."""
    file_id: int
    file_path: str
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


# --------------------------------------------------------------- #
# Cache-key + segment resolution
# --------------------------------------------------------------- #

def spec_hash(spec: ClipSpec, segments: list[Segment]) -> str:
    """SHA-256 of the normalised spec + segment list. Used as
    the row's ``spec_hash`` in ``event_clip_job`` and as the
    cache filename stem."""
    assert isinstance(spec, ClipSpec), "spec required"
    seg_repr = [
        (s.file_path, round(s.start_sec, 3),
         round(s.end_sec, 3))
        for s in segments
    ]
    blob = json.dumps({
        "file_id": int(spec.file_id),
        "eo": round(float(spec.eo_sec), 3),
        "bb": round(float(spec.bb_sec), 3),
        "pre": round(float(spec.pre_sec), 3),
        "post": round(float(spec.post_sec), 3),
        "segs": seg_repr,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def resolve_clip_segments(spec: ClipSpec, store
                            ) -> list[Segment]:
    """Walk prev/next files until the EO-pre to BB+post window
    is covered. Returns 1-3 ordered Segments.

    Falls back to "clip to file boundary" with a single
    Segment when the adjacent file doesn't exist (e.g. the
    previous chunk hasn't been ingested yet).
    """
    assert isinstance(spec, ClipSpec), "spec required"
    file = store.file_row(spec.file_id)
    if not file:
        return []
    dur = float(file["duration_sec"] or 0.0)
    clip_start = float(spec.eo_sec) - float(spec.pre_sec)
    clip_end = float(spec.bb_sec) + float(spec.post_sec)
    segments: list[Segment] = []
    # Pre-window from the previous file.
    if clip_start < 0:
        prev = store.get_adjacent_files(spec.file_id, "prev")
        if prev and prev.get("file_path"):
            prev_dur = float(prev.get("duration_sec") or 0.0)
            tail_sec = -clip_start
            segments.append(Segment(
                file_id=int(prev["id"]),
                file_path=str(prev["file_path"]),
                start_sec=max(0.0, prev_dur - tail_sec),
                end_sec=prev_dur,
            ))
    # Main file segment, clamped to the file's bounds.
    segments.append(Segment(
        file_id=int(file["id"]),
        file_path=str(file["file_path"]),
        start_sec=max(0.0, clip_start),
        end_sec=min(dur, clip_end),
    ))
    # Post-window from the next file.
    if clip_end > dur:
        nxt = store.get_adjacent_files(spec.file_id, "next")
        if nxt and nxt.get("file_path"):
            nxt_dur = float(nxt.get("duration_sec") or 0.0)
            head_sec = clip_end - dur
            segments.append(Segment(
                file_id=int(nxt["id"]),
                file_path=str(nxt["file_path"]),
                start_sec=0.0,
                end_sec=min(nxt_dur, head_sec),
            ))
    return segments


# --------------------------------------------------------------- #
# LFP path (in-process numpy)
# --------------------------------------------------------------- #

def extract_lfp_clip(spec: ClipSpec, channel: int, store
                      ) -> tuple[np.ndarray, np.ndarray, float]:
    """Stitch the LFP for *channel* across all clip segments.

    Returns ``(t, signal, fs)`` where ``t`` starts at 0 and
    ``signal`` has ``np.nan`` bridges of 1 sample between
    segments so the Plotly trace shows the boundary explicitly.
    """
    assert isinstance(spec, ClipSpec), "spec required"
    assert isinstance(channel, int), "channel must be int"
    segs = resolve_clip_segments(spec, store)
    if not segs:
        return (np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32), 0.0)
    pieces: list[np.ndarray] = []
    fs_seen: float | None = None
    max_iter = 8  # NASA Rule 2; segments cap is 3
    for i, seg in enumerate(segs):
        assert i < max_iter, "segment loop runaway"
        chunk = get_chunk(seg.file_path)
        fs = float(chunk.fs)
        if fs_seen is None:
            fs_seen = fs
        elif abs(fs - fs_seen) > 1e-3:
            # Adjacent files at different fs: drop the segment
            # rather than mangle the time axis.
            logger.warning(
                "event_clip: fs mismatch %s vs %s; "
                "dropping segment %s", fs, fs_seen, seg.file_path)
            continue
        sig = chunk.signal
        if sig.ndim != 2 or sig.shape[1] <= channel:
            continue
        lo = max(0, int(round(seg.start_sec * fs)))
        hi = min(sig.shape[0], int(round(seg.end_sec * fs)))
        if hi > lo:
            pieces.append(sig[lo:hi, channel]
                            .astype(np.float32, copy=True))
            if i < len(segs) - 1:
                pieces.append(np.array([np.nan],
                                         dtype=np.float32))
    if not pieces or fs_seen is None:
        return (np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float32), 0.0)
    signal = np.concatenate(pieces)
    t = np.arange(len(signal), dtype=np.float64) / float(fs_seen)
    return (t, signal, float(fs_seen))


# --------------------------------------------------------------- #
# Video path (ffmpeg + cache)
# --------------------------------------------------------------- #

_worker_started: bool = False
_worker_lock = threading.Lock()
_worker_wake = threading.Event()


def _ffmpeg_bin(config: dict) -> str:
    return ((config or {}).get("event_clip", {}) or {}
              ).get("ffmpeg_bin", "ffmpeg")


def _cache_dir(config: dict) -> Path:
    raw = ((config or {}).get("event_clip", {}) or {}
             ).get("cache_dir", "secrets/event_clip_cache")
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_max_gb(config: dict) -> float:
    return float(((config or {}).get("event_clip", {}) or {}
                     ).get("cache_max_gb", 5))


def _fallback_to_reencode(config: dict) -> bool:
    return bool(((config or {}).get("event_clip", {}) or {}
                    ).get("fallback_to_reencode", True))


def submit_video_clip_job(spec: ClipSpec, store,
                            config: dict) -> str:
    """Cache-hit or insert a pending job. Returns ``spec_hash``.

    The PI tab polls ``poll_video_clip_status(spec_hash)``
    until status == 'done' (cache_path is then ready to be
    served by ``/media/clip/<spec_hash>``).
    """
    segs = resolve_clip_segments(spec, store)
    h = spec_hash(spec, segs)
    cache_dir = _cache_dir(config)
    cache_path = cache_dir / f"{h}.mp4"
    conn = store._connect()
    try:
        existing = conn.execute(
            "SELECT status, cache_path FROM event_clip_job "
            "WHERE spec_hash = ?", (h,),
        ).fetchone()
        if existing:
            st = existing["status"]
            if st == "done" and cache_path.exists():
                return h
            if st in ("pending", "running"):
                return h
            # Failed or stale -- requeue.
            conn.execute(
                "DELETE FROM event_clip_job WHERE spec_hash = ?",
                (h,))
        now = datetime.now().isoformat()
        conn.execute(
            """INSERT INTO event_clip_job
               (spec_hash, file_id, eo_sec, bb_sec, status,
                cache_path, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (h, int(spec.file_id), float(spec.eo_sec),
             float(spec.bb_sec), str(cache_path), now),
        )
        conn.commit()
    finally:
        conn.close()
    _worker_wake.set()
    return h


def poll_video_clip_status(spec_hash_str: str,
                              store) -> dict:
    """Return ``{status, cache_path, error}`` for a job."""
    assert spec_hash_str, "spec_hash required"
    conn = store._connect()
    try:
        row = conn.execute(
            """SELECT status, cache_path, error
               FROM event_clip_job WHERE spec_hash = ?""",
            (spec_hash_str,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return {"status": "missing", "cache_path": None,
                "error": None}
    return {"status": row["status"],
             "cache_path": row["cache_path"],
             "error": row["error"]}


def _run_ffmpeg(cmd: list[str], timeout_sec: float) -> None:
    """Subprocess wrapper with explicit timeout + error log."""
    assert isinstance(cmd, list) and cmd, "cmd required"
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout_sec,
    )
    if result.returncode != 0:
        snippet = (result.stderr or "")[-400:]
        raise RuntimeError(
            f"ffmpeg exit {result.returncode}: {snippet}")


def _resolve_video_for_segment(seg: Segment) -> str:
    """Map a segment's .mat path to its first companion
    video. The reviewer's focused camera could be cam N>1, but
    for PI clips we standardize on cam 1 (the master that the
    Video Review tab's controls drive)."""
    from src.utils.video import companion_video_paths
    paths = companion_video_paths(seg.file_path)
    if not paths:
        raise FileNotFoundError(
            f"no companion video for {seg.file_path}")
    return paths[0]


def _process_one_job(row: dict, store, config: dict) -> None:
    spec_h = row["spec_hash"]
    cache_path = Path(row["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    conn = store._connect()
    try:
        conn.execute(
            "UPDATE event_clip_job SET status='running', "
            "started_at=? WHERE spec_hash = ?",
            (datetime.now().isoformat(), spec_h),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        # Rebuild the segments from the original spec so we
        # don't trust stale cached state.
        spec = ClipSpec(
            file_id=int(row["file_id"]),
            eo_sec=float(row["eo_sec"]),
            bb_sec=float(row["bb_sec"]),
        )
        segs = resolve_clip_segments(spec, store)
        _ffmpeg_concat(segs, cache_path, config)
        conn = store._connect()
        try:
            conn.execute(
                "UPDATE event_clip_job SET status='done', "
                "finished_at=? WHERE spec_hash = ?",
                (datetime.now().isoformat(), spec_h),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("event_clip: produced %s (%d segments)",
                     cache_path, len(segs))
    except Exception as e:
        logger.warning("event_clip job %s failed: %s",
                        spec_h, e)
        conn = store._connect()
        try:
            conn.execute(
                "UPDATE event_clip_job SET status='failed', "
                "error=?, finished_at=? WHERE spec_hash = ?",
                (str(e), datetime.now().isoformat(), spec_h),
            )
            conn.commit()
        finally:
            conn.close()


def _ffmpeg_concat(segs: list[Segment], out_path: Path,
                     config: dict) -> None:
    """Materialise the segments as a single MP4. Tries
    stream-copy first; falls back to re-encode if enabled."""
    assert segs, "segments required"
    ffmpeg = _ffmpeg_bin(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    if len(segs) == 1:
        seg = segs[0]
        src = _resolve_video_for_segment(seg)
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{seg.start_sec:.3f}",
            "-i", src,
            "-t", f"{seg.duration:.3f}",
            "-c", "copy", str(out_path),
        ]
        try:
            _run_ffmpeg(cmd, timeout_sec=300.0)
            return
        except Exception as e:
            if not _fallback_to_reencode(config):
                raise
            logger.info(
                "event_clip: stream-copy failed (%s); "
                "re-encoding", e)
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{seg.start_sec:.3f}",
            "-i", src,
            "-t", f"{seg.duration:.3f}",
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac", str(out_path),
        ]
        _run_ffmpeg(cmd, timeout_sec=900.0)
        return
    # Multi-segment: write per-segment .ts files then concat.
    tmp_dir = out_path.parent / f".tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        listing = tmp_dir / "concat.txt"
        ts_paths: list[Path] = []
        max_iter = 8
        for i, seg in enumerate(segs):
            assert i < max_iter, "segment loop runaway"
            ts = tmp_dir / f"seg{i}.ts"
            ts_paths.append(ts)
            src = _resolve_video_for_segment(seg)
            cmd = [
                ffmpeg, "-y", "-loglevel", "error",
                "-ss", f"{seg.start_sec:.3f}",
                "-i", src,
                "-t", f"{seg.duration:.3f}",
                "-c", "copy",
                "-bsf:v", "h264_mp4toannexb",
                "-f", "mpegts", str(ts),
            ]
            _run_ffmpeg(cmd, timeout_sec=300.0)
        with listing.open("w", encoding="utf-8") as f:
            for ts in ts_paths:
                f.write(f"file '{ts.as_posix()}'\n")
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(listing),
            "-c", "copy", str(out_path),
        ]
        _run_ffmpeg(cmd, timeout_sec=300.0)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


# --------------------------------------------------------------- #
# Daemon worker + LRU eviction
# --------------------------------------------------------------- #

def _evict_lru(cache_dir: Path, max_gb: float) -> None:
    """Drop the oldest cache files until total size <= max_gb."""
    if max_gb <= 0:
        return
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    files = []
    try:
        for p in cache_dir.glob("*.mp4"):
            try:
                files.append((p.stat().st_mtime, p.stat().st_size, p))
            except OSError:
                continue
    except OSError:
        return
    total = sum(s for _, s, _ in files)
    if total <= max_bytes:
        return
    files.sort(key=lambda t: t[0])  # oldest first
    for _mtime, size, p in files:
        if total <= max_bytes:
            break
        try:
            p.unlink()
            total -= size
        except OSError:
            pass


def _worker_loop(store, config: dict) -> None:
    """Pop pending jobs from event_clip_job; LRU-evict after
    each batch."""
    interval = 2.0  # seconds between idle polls
    max_iter = 10 ** 9
    i = 0
    cache_dir = _cache_dir(config)
    max_gb = _cache_max_gb(config)
    while True:
        assert i < max_iter, "worker loop runaway"
        i += 1
        try:
            conn = store._connect()
            try:
                row = conn.execute(
                    "SELECT spec_hash, file_id, eo_sec, "
                    "bb_sec, cache_path "
                    "FROM event_clip_job "
                    "WHERE status = 'pending' "
                    "ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row is None:
                _worker_wake.wait(timeout=interval)
                _worker_wake.clear()
                continue
            _process_one_job(dict(row), store, config)
            _evict_lru(cache_dir, max_gb)
        except Exception:
            logger.exception("event_clip worker iter failed")
            time.sleep(interval)


def start_worker(store, config: dict) -> None:
    """Spawn the background extractor thread. Idempotent."""
    assert config is None or isinstance(config, dict), \
        "config dict required"
    cfg = (config or {}).get("event_clip", {}) or {}
    if not cfg.get("enabled", True):
        logger.info("event_clip worker disabled in config")
        return
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(
        target=_worker_loop, args=(store, config),
        daemon=True, name="qc-event-clip-worker",
    )
    t.start()
    logger.info("event_clip worker started "
                 "(cache=%s, cap=%.1f GB)",
                 _cache_dir(config), _cache_max_gb(config))


def probe_ffmpeg(config: dict) -> bool:
    """Quick startup probe so the PI tab can warn if ffmpeg
    isn't on PATH."""
    try:
        result = subprocess.run(
            [_ffmpeg_bin(config), "-version"],
            capture_output=True, text=True, timeout=5.0,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
