"""Opt-in memory-leak tracer.

When a long-lived process (the dashboard service) grows without bound, the cause
is almost always a small number of allocation SITES that keep adding references
the process never releases. ``tracemalloc`` records the traceback of every
allocation; comparing two snapshots shows exactly which ``file:line`` grew, in
bytes -- so a leak names its own source within a couple of snapshots.

Off unless ``debug.memtrace.enabled`` is true in config (or env ``QC_MEMTRACE=1``):
tracemalloc roughly doubles allocation cost, so it is a diagnostic switch, not a
default. Enable it, restart the service, wait a few intervals, then read the log:
the leaking line is the one with a steady positive ``size_diff`` every snapshot.

Usage (wired in ``dashboard/app.py`` boot):
    from src.utils import memtrace
    memtrace.start(config)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import tracemalloc

logger = logging.getLogger("qc_monitor.memtrace")

_thread: threading.Thread | None = None
_lock = threading.Lock()

DEFAULT_INTERVAL_SEC = 120.0
DEFAULT_TOP = 20
DEFAULT_FRAMES = 12
_MIN_INTERVAL_SEC = 15.0


def _rss_gb() -> float:
    """Process RSS in GB, or -1.0 if it cannot be read (psutil optional)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:                                  # noqa: BLE001
        return -1.0


def _settings(config: dict) -> tuple[bool, float, int, int, str]:
    d = ((config or {}).get("debug", {}) or {}).get("memtrace", {}) or {}
    enabled = bool(d.get("enabled", False)) or os.environ.get(
        "QC_MEMTRACE", "") in ("1", "true", "True")
    try:
        interval = max(_MIN_INTERVAL_SEC,
                       float(d.get("interval_sec", DEFAULT_INTERVAL_SEC)))
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_SEC
    try:
        top = max(1, int(d.get("top", DEFAULT_TOP)))
    except (TypeError, ValueError):
        top = DEFAULT_TOP
    try:
        frames = max(1, int(d.get("frames", DEFAULT_FRAMES)))
    except (TypeError, ValueError):
        frames = DEFAULT_FRAMES
    logpath = d.get("logfile") or os.path.join(os.getcwd(), "memtrace.log")
    return enabled, interval, top, frames, logpath


def _format_stat(stat) -> str:
    """One growth line: the site, its delta since the last snapshot, and its
    running total. ``stat.traceback`` renders the deepest frame first."""
    frame = stat.traceback[0] if stat.traceback else None
    where = f"{frame.filename}:{frame.lineno}" if frame else "?"
    return (f"  {stat.size_diff / 1e6:+9.1f} MB  {stat.count_diff:+8d} obj  "
            f"(total {stat.size / 1e6:8.1f} MB)  {where}")


def _loop(interval: float, top: int, logpath: str) -> None:
    prev = None
    it = 0
    max_iter = 10 ** 9                                 # NASA rule 2 bound
    while True:
        assert it < max_iter, "memtrace loop runaway"
        it += 1
        time.sleep(interval)
        try:
            snap = tracemalloc.take_snapshot()
        except Exception as e:                         # noqa: BLE001
            logger.warning("memtrace snapshot failed: %s", e)
            continue
        header = (f"=== memtrace {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                  f"RSS={_rss_gb():.2f} GB ===")
        lines = [header]
        if prev is not None:
            # Rank sites by GROWTH since the previous snapshot: the leak is the
            # line that keeps growing every interval.
            for stat in snap.compare_to(prev, "lineno")[:top]:
                if stat.size_diff <= 0:
                    break                              # only the growers
                lines.append(_format_stat(stat))
        else:
            lines.append("  (baseline snapshot; growth shown from next tick)")
        prev = snap
        try:
            with open(logpath, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as e:
            logger.warning("memtrace log write failed: %s", e)


def start(config: dict) -> bool:
    """Start the tracer if enabled. Idempotent; returns True when it launches."""
    enabled, interval, top, frames, logpath = _settings(config)
    if not enabled:
        return False
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        if not tracemalloc.is_tracing():
            tracemalloc.start(frames)
        _thread = threading.Thread(target=_loop, args=(interval, top, logpath),
                                   daemon=True, name="qc-memtrace")
        _thread.start()
    logger.info("memtrace started (every %.0fs, top %d, %d frames -> %s)",
                interval, top, frames, logpath)
    return True


def is_running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()
