"""Background self-healer for stale recording paths.

External data drives get remounted under different letters (an old data drive
returns as ``H:\\``, ``J:\\`` ...). ``processed_files.file_path`` / ``session_dir``
(and the two ``evoked_output_path`` columns) are stored as raw absolute paths,
so every re-letter leaves thousands of rows pointing at a drive that no longer
exists -- the dashboard and reports can't open those recordings.

This worker fixes that the boring way: every *interval* it walks a BOUNDED batch
of the DB and, for each row whose stored path no longer resolves, repoints it to
wherever the file currently lives on a mounted drive -- the cheap per-drive stat
(``candidate_paths``) first, a recursive ``EEGLocator`` walk only for stragglers.
It RE-LINKS known files only; it never re-ingests or reprocesses anything.

Runs off the watch loop (a daemon thread, mirroring ``sidecar_warm``) so a
slow/half-down UNC stat can never stall file registration or processing. Bounded
everywhere -- ``rows_per_pass`` cursor batch, a whole-pass ``wall_clock_sec``
budget, and a per-file ``max_walk_sec`` walk cap -- and it never lets its loop
die. The cursor advances each pass and wraps to the table start, so the whole
corpus is re-checked continuously; a fresh boot starts a pass at row 0.
"""

from __future__ import annotations

import logging
import threading
import time

from src.maintenance.relocate_paths import run_heal_pass

logger = logging.getLogger("qc_monitor.path_heal")

DEFAULT_INTERVAL_SEC = 3600.0
DEFAULT_BOOT_DELAY_SEC = 120.0
DEFAULT_ROWS_PER_PASS = 2000
DEFAULT_WALL_CLOCK_SEC = 120.0
DEFAULT_MAX_WALK_SEC = 20.0
_MIN_INTERVAL_SEC = 60.0

_thread: threading.Thread | None = None
_lock = threading.Lock()

# Live status (in-process) a Workers view can poll to tell "slow" from "frozen".
_status: dict = {"phase": "starting", "pos": None, "cursor": 0,
                 "relocated_total": 0, "unresolved_last": 0, "next_at": None}
_status_lock = threading.Lock()


def _set_status(**kw) -> None:
    with _status_lock:
        _status.update(kw)


def status() -> dict:
    with _status_lock:
        return dict(_status)


def _cfg(config: dict) -> dict:
    return (config or {}).get("path_healing", {}) or {}


def _fnum(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _settings(config: dict) -> dict:
    c = _cfg(config)
    return {
        "enabled": bool(c.get("enabled", True)),
        "interval_sec": max(_MIN_INTERVAL_SEC,
                            _fnum(c.get("interval_sec"), DEFAULT_INTERVAL_SEC)),
        "boot_delay_sec": max(0.0, _fnum(c.get("boot_delay_sec"),
                                         DEFAULT_BOOT_DELAY_SEC)),
        "rows_per_pass": max(1, int(_fnum(c.get("rows_per_pass"),
                                          DEFAULT_ROWS_PER_PASS))),
        "wall_clock_sec": max(1.0, _fnum(c.get("wall_clock_sec"),
                                         DEFAULT_WALL_CLOCK_SEC)),
        "max_walk_sec": max(0.0, _fnum(c.get("max_walk_sec"),
                                       DEFAULT_MAX_WALK_SEC)),
        "use_locator": bool(c.get("use_locator", True)),
        "include_evoked": bool(c.get("include_evoked", True)),
    }


def _progress(done: int, total: int, s) -> None:
    _set_status(phase="scanning", pos=(done, total), relocated_total=s.relocated)


def _run(store, config: dict) -> None:
    st = _settings(config)
    logger.info("path-heal worker started (batch=%d rows every %.0fs, "
                "wall_clock=%.0fs, max_walk=%.0fs, use_locator=%s, "
                "include_evoked=%s)", st["rows_per_pass"], st["interval_sec"],
                st["wall_clock_sec"], st["max_walk_sec"], st["use_locator"],
                st["include_evoked"])
    if st["boot_delay_sec"]:
        _set_status(phase="settling", next_at=time.time() + st["boot_delay_sec"])
        time.sleep(st["boot_delay_sec"])
    after_id = 0
    while True:
        try:
            summary, after_id = run_heal_pass(
                store, config, apply=True, after_id=after_id,
                rows_per_pass=st["rows_per_pass"],
                use_locator=st["use_locator"],
                include_evoked=st["include_evoked"],
                max_walk_sec=st["max_walk_sec"],
                wall_clock_sec=st["wall_clock_sec"], progress=_progress)
            if (summary.relocated or summary.collided or summary.unresolved
                    or summary.evoked_relocated):
                logger.info(
                    "path-heal: scanned=%d already_ok=%d relocated=%d "
                    "collided=%d unresolved=%d evoked_relocated=%d in %.0fs "
                    "(cursor->%d)", summary.scanned, summary.already_ok,
                    summary.relocated, summary.collided, summary.unresolved,
                    summary.evoked_relocated, summary.elapsed_sec, after_id)
            _set_status(phase="idle" if after_id == 0 else "sleeping",
                        pos=None, cursor=after_id,
                        unresolved_last=summary.unresolved,
                        next_at=time.time() + st["interval_sec"])
        except Exception as e:              # noqa: BLE001 -- never kill the loop
            logger.warning("path-heal loop error: %s", e)
            _set_status(phase="error", next_at=time.time() + st["interval_sec"])
        time.sleep(st["interval_sec"])


def start_worker(store, config: dict) -> bool:
    """Start the path-heal worker unless disabled or already running. Idempotent."""
    global _thread
    if not _settings(config)["enabled"]:
        logger.info("path-heal worker disabled by config")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _thread = threading.Thread(target=_run, args=(store, config),
                                   daemon=True, name="path-heal")
        _thread.start()
        return True


def is_running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()
