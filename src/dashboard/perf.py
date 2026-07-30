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

import threading
import time

_LOCK = threading.Lock()
_STATS: dict[str, dict] = {}      # label -> {n,total,max,last}
_MAX_LABELS = 2000                # bound distinct labels (runaway backstop)
_started_at = time.time()


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
    """Clear all counters (the 'zero it and reproduce' button)."""
    with _LOCK:
        _STATS.clear()


def uptime_sec() -> float:
    return max(0.0, time.time() - _started_at)
