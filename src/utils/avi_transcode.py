"""On-demand .avi -> mp4 transcoding for the browser <video> element.

The lab's behavioral recordings ship as ``.avi`` (often multi-camera
``_v1.avi`` / ``_v2.avi``), which HTML5 ``<video>`` can't play. This module
makes a browser-playable ``.mp4`` once, in the background, and caches it
content-addressed by (source path, mtime, size) so later opens are instant.

Fast path: when the .avi already holds an H.264/HEVC stream we stream-copy it
into mp4 (`-c:v copy`, seconds). Otherwise we re-encode to H.264. Audio is
dropped (`-an`) -- behavioral cameras are silent and it avoids codec issues.

No DB table: state is an in-process running-set + a content-addressed cache
file, mirroring the chronic/import background-warm pattern. The media route
kicks ``ensure_async`` and serves the cache file once ``status`` is "ready".
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger("qc_monitor.utils.avi_transcode")

# Browser-playable video codecs -> stream-copy (no re-encode) when present.
_COPY_OK = frozenset({"h264", "hevc"})

_RUNNING: set[str] = set()
_ERRORS: dict[str, str] = {}
_LOCK = threading.Lock()


# ---- config accessors (reuse event_clip's ffmpeg, own cache dir) ---- #

def _cfg(config: dict) -> dict:
    return (config or {}).get("video_transcode", {}) or {}


def _ffmpeg_bin(config: dict) -> str:
    return ((config or {}).get("event_clip", {}) or {}).get(
        "ffmpeg_bin", "ffmpeg")


def _ffprobe_bin(config: dict) -> str:
    ff = _ffmpeg_bin(config)
    # Derive ffprobe alongside ffmpeg (same dir / same naming convention).
    head, tail = os.path.split(ff)
    probe = tail.replace("ffmpeg", "ffprobe") or "ffprobe"
    return os.path.join(head, probe) if head else probe


def cache_dir(config: dict) -> Path:
    raw = _cfg(config).get("cache_dir", "secrets/avi_mp4_cache")
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_max_gb(config: dict) -> float:
    return float(_cfg(config).get("cache_max_gb", 10))


def cache_path(avi_path: str, config: dict) -> Path:
    """Content-addressed mp4 cache path for *avi_path* (keyed by path + mtime
    + size so a re-recorded file re-transcodes)."""
    assert avi_path, "avi_path required"
    try:
        st = os.stat(avi_path)
        sig = f"{avi_path}|{int(st.st_mtime)}|{st.st_size}"
    except OSError:
        sig = avi_path
    h = hashlib.sha1(sig.encode("utf-8", "replace")).hexdigest()[:20]
    return cache_dir(config) / f"{h}.mp4"


def status(avi_path: str, config: dict) -> str:
    """'ready' (cached mp4 exists), 'running' (transcoding now), 'error'
    (last attempt failed), or 'absent'."""
    cp = cache_path(avi_path, config)
    if cp.exists() and cp.stat().st_size > 0:
        return "ready"
    with _LOCK:
        if str(cp) in _RUNNING:
            return "running"
        if str(cp) in _ERRORS:
            return "error"
    return "absent"


def ready_path(avi_path: str, config: dict) -> str | None:
    cp = cache_path(avi_path, config)
    return str(cp) if (cp.exists() and cp.stat().st_size > 0) else None


def ensure_async(avi_path: str, config: dict) -> str | None:
    """Return the cached mp4 path if ready; otherwise kick a background
    transcode (coalesced) and return None. Non-blocking."""
    if not avi_path or not os.path.isfile(avi_path):
        return None
    cp = cache_path(avi_path, config)
    if cp.exists() and cp.stat().st_size > 0:
        return str(cp)
    with _LOCK:
        if str(cp) in _RUNNING:
            return None
        _RUNNING.add(str(cp))
        _ERRORS.pop(str(cp), None)
    t = threading.Thread(target=_worker, args=(avi_path, cp, config),
                          daemon=True)
    t.start()
    return None


def _worker(avi_path: str, cp: Path, config: dict) -> None:
    try:
        _transcode(avi_path, cp, config)
    except Exception as e:  # noqa: BLE001 -- transcode must never crash a thread
        logger.warning("avi transcode failed %s: %s", avi_path, e)
        with _LOCK:
            _ERRORS[str(cp)] = str(e)
    finally:
        with _LOCK:
            _RUNNING.discard(str(cp))
        try:
            _evict_lru(cache_dir(config), _cache_max_gb(config))
        except Exception:  # noqa: BLE001
            pass


def _probe_codec(avi_path: str, config: dict) -> str:
    """First video stream codec name (lowercased), or "" on any failure."""
    try:
        out = subprocess.run(
            [_ffprobe_bin(config), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of",
             "default=nw=1:nk=1", avi_path],
            capture_output=True, text=True, timeout=60.0)
        return (out.stdout or "").strip().splitlines()[0].lower() \
            if out.stdout.strip() else ""
    except Exception as e:  # noqa: BLE001
        logger.debug("ffprobe failed %s: %s", avi_path, e)
        return ""


def _transcode(avi_path: str, cp: Path, config: dict) -> None:
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.mp4")
    ffmpeg = _ffmpeg_bin(config)
    codec = _probe_codec(avi_path, config)
    base = [ffmpeg, "-y", "-loglevel", "error", "-i", avi_path, "-an",
            "-movflags", "+faststart"]
    if codec in _COPY_OK:
        cmd = base[:6] + ["-c:v", "copy", "-an", "-movflags",
                          "+faststart", str(tmp)]
        timeout = 300.0
    else:
        cmd = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                      str(tmp)]
        timeout = float(_cfg(config).get("reencode_timeout_sec", 1800))
    logger.info("transcoding %s (codec=%s) -> %s", avi_path, codec or "?", cp)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        # Stream-copy can fail on an odd container; retry with a real encode.
        if codec in _COPY_OK:
            cmd = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf",
                          "23", str(tmp)]
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=float(_cfg(config).get(
                                     "reencode_timeout_sec", 1800)))
        if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            snippet = (res.stderr or "")[-300:]
            raise RuntimeError(f"ffmpeg exit {res.returncode}: {snippet}")
    os.replace(tmp, cp)


def _evict_lru(cdir: Path, max_gb: float) -> None:
    """Drop oldest-accessed cache files once the dir exceeds *max_gb*."""
    files = [(p.stat().st_atime, p.stat().st_size, p)
             for p in cdir.glob("*.mp4")]
    total = sum(s for _a, s, _p in files)
    cap = max_gb * (1024 ** 3)
    if total <= cap:
        return
    for _atime, size, p in sorted(files):
        try:
            p.unlink()
        except OSError:
            continue
        total -= size
        if total <= cap:
            break
