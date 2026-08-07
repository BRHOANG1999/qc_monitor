"""In-process performance ledger: what is eating computation.

Every Dash callback (timed by a Flask after_request hook), every tab render, and
every instrumented Overview stage records its wall-time here under a label. The
Performance tab reads ``top()`` to rank the costliest work so a "everything is
slow" complaint becomes "these three labels dominate".

Aggregates only (count / total / max / last / avg) -- no per-call history, so
memory is bounded by the number of DISTINCT labels (capped). Thread-safe: the
after_request hook runs on many server threads at once. Best-effort: recording
never raises into a request.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

logger = logging.getLogger("qc_monitor.dashboard.perf")

_LOCK = threading.Lock()
_STATS: dict[str, dict] = {}      # label -> {n,total,max,last}
_MAX_LABELS = 2000                # bound distinct labels (runaway backstop)
_started_at = time.time()

# Persistence: the ledger is RAM-only by default, so a daemon restart wipes the
# "which processes are slow" history -- exactly what we want to keep across
# restarts. start_persistence() loads a prior ledger on boot and flushes it to a
# JSON file on a timer. We persist to a FILE, never the DB: writing high-rate
# perf aggregates into the same contended SQLite would add to the write-lock
# pressure that is itself a top cause of dashboard slowness.
_PERSIST_PATH: str | None = None
_PERSIST_SINCE: float = _started_at   # epoch the accumulated totals span from
_persist_thread = None


def record(label: str, ms: float) -> None:
    """Add one timing sample (milliseconds) under *label*."""
    try:
        if not label:
            return
        with _LOCK:
            s = _STATS.get(label)
            if s is None:
                if len(_STATS) >= _MAX_LABELS:
                    return
                s = {"n": 0, "total": 0.0, "max": 0.0, "last": 0.0}
                _STATS[label] = s
            s["n"] += 1
            s["total"] += ms
            if ms > s["max"]:
                s["max"] = ms
            s["last"] = ms
    except Exception:
        pass


class Timer:
    """Context manager that records its wall-time under *label* on exit."""

    __slots__ = ("label", "_t0")

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        record(self.label, (time.perf_counter() - self._t0) * 1000.0)
        return False


def top(n: int = 40, by: str = "total") -> list[dict]:
    """Rows sorted by *by* ('total' | 'max' | 'avg' | 'last' | 'n'), richest
    first. Each row: label, n, total_ms, avg_ms, max_ms, last_ms."""
    with _LOCK:
        rows = [
            {"label": k, "n": v["n"], "total_ms": v["total"],
             "avg_ms": (v["total"] / v["n"] if v["n"] else 0.0),
             "max_ms": v["max"], "last_ms": v["last"]}
            for k, v in _STATS.items()
        ]
    key = {"total": "total_ms", "max": "max_ms", "avg": "avg_ms",
           "last": "last_ms", "n": "n"}.get(by, "total_ms")
    rows.sort(key=lambda r: r.get(key, 0), reverse=True)
    return rows[:n]


def reset() -> None:
    """Clear all counters (the 'zero it and reproduce' button). Also resets the
    accumulation window so avg/total read from now, and flushes the cleared
    ledger so a restart doesn't resurrect the old totals."""
    global _PERSIST_SINCE
    with _LOCK:
        _STATS.clear()
        _PERSIST_SINCE = time.time()
    if _PERSIST_PATH:
        dump_to(_PERSIST_PATH)


def uptime_sec() -> float:
    return max(0.0, time.time() - _started_at)


def accumulating_since() -> float:
    """Epoch the current totals span from (survives restarts via the ledger
    file; moved forward only by reset())."""
    return _PERSIST_SINCE


# ---------------------------------------------------------------------- #
#  Persistence -- accumulate the ledger across daemon restarts (JSON file).
# ---------------------------------------------------------------------- #

def dump_to(path: str) -> bool:
    """Atomically write the ledger to *path* as JSON. Best-effort: returns False
    on any error (a perf flush must never crash the daemon). Writes to a temp
    file then replaces, so a crash mid-write can't corrupt the ledger."""
    try:
        with _LOCK:
            payload = {"since": _PERSIST_SINCE, "saved_at": time.time(),
                       "stats": {k: dict(v) for k, v in _STATS.items()}}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("perf ledger dump failed (%s): %s", path, e)
        return False


def load_from(path: str) -> bool:
    """Merge a previously-dumped ledger into the in-memory stats (adds n/total,
    maxes max, keeps the earliest 'since'). Best-effort. Called once on boot so
    the 'which processes are slow' history survives restarts."""
    global _PERSIST_SINCE
    try:
        if not os.path.exists(path):
            return False
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:  # noqa: BLE001
        logger.debug("perf ledger load failed (%s): %s", path, e)
        return False
    saved = payload.get("stats", {}) or {}
    with _LOCK:
        since = payload.get("since")
        if isinstance(since, (int, float)):
            _PERSIST_SINCE = min(_PERSIST_SINCE, float(since))
        for label, v in saved.items():
            if len(_STATS) >= _MAX_LABELS:
                break
            s = _STATS.get(label)
            if s is None:
                s = {"n": 0, "total": 0.0, "max": 0.0, "last": 0.0}
                _STATS[label] = s
            s["n"] += int(v.get("n", 0))
            s["total"] += float(v.get("total", 0.0))
            s["max"] = max(s["max"], float(v.get("max", 0.0)))
            # 'last' is a point-in-time value; keep the loaded one until a live
            # sample overwrites it (so a restarted process still shows a value).
            if not s.get("last"):
                s["last"] = float(v.get("last", 0.0))
    logger.info("perf ledger loaded from %s (%d labels)", path, len(saved))
    return True


def start_persistence(path: str, interval_sec: float = 60.0) -> None:
    """Load the prior ledger, then flush the live ledger to *path* every
    *interval_sec* on a daemon thread. Idempotent: a second call is a no-op so
    reloads don't spawn duplicate flushers. Call once at boot."""
    global _PERSIST_PATH, _persist_thread
    if _persist_thread is not None and _persist_thread.is_alive():
        return
    _PERSIST_PATH = path
    load_from(path)

    def _flush_loop():
        # Fixed cadence; a flush is a tiny local JSON write (no DB, no lock held
        # across the write). max_iter guard per NASA rule 2.
        it = 0
        while it < 10_000_000:
            it += 1
            time.sleep(max(5.0, interval_sec))
            dump_to(path)

    th = threading.Thread(target=_flush_loop, daemon=True, name="perf-ledger")
    _persist_thread = th
    th.start()
