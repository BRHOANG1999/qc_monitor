"""Periodic flush of "Needs more onsets" events into the BHZ day CSVs.

Onsets reach `needs_scoring` by two different routes, and only ONE of them
used to export:

* the reviewer presses Submit -> ``video._save_review`` routes incomplete work
  to ``needs_scoring`` and exports the onsets inline; but
* the 4-second AUTOSAVE (``video._autosave_draft`` -> ``store.upsert_scoring_draft``)
  writes a ``needs_scoring`` row directly, bypassing that path entirely.

So a reviewer who drops onsets and navigates away without submitting left the
onsets in the DB only. Those files also never reach the PI approval gate (they
are incomplete by definition), so nothing else would ever export them.

This worker closes that gap: every *interval* it re-runs the same batch export
the PI's "Flush all Needs-more-onsets -> CSV" button uses. It is safe to run
repeatedly -- ``bhz_csv.write_event_rows`` dedups by ``(filename, EventEO)``
and does not touch the file at all when there is nothing new, so a steady state
is a cheap DB read. It also self-heals transient failures: a day CSV that is
momentarily locked (someone has it open in Excel) simply exports on the next
pass.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("qc_monitor.needs_scoring_flush")

DEFAULT_INTERVAL_SEC = 600.0
_MIN_INTERVAL_SEC = 60.0

_thread: threading.Thread | None = None
_lock = threading.Lock()


def _settings(config: dict) -> tuple[bool, float]:
    b = (config or {}).get("bhz_csv", {}) or {}
    if not b.get("enabled"):
        return False, DEFAULT_INTERVAL_SEC          # CSV export off entirely
    enabled = bool(b.get("auto_flush_enabled", True))
    try:
        interval = float(b.get("auto_flush_interval_sec", DEFAULT_INTERVAL_SEC))
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_SEC
    return enabled, max(_MIN_INTERVAL_SEC, interval)


def flush_once(store, config: dict) -> dict:
    """Run one export pass. Returns the summary dict; never raises."""
    try:
        # Lazy import: this pulls in the dashboard tab module, and the worker
        # may start before/independently of the dashboard.
        from src.dashboard.tabs.video import _export_all_needs_scoring
        return _export_all_needs_scoring(store, config)
    except Exception as e:                  # noqa: BLE001 -- worker must not die
        logger.warning("needs_scoring auto-flush failed: %s", e)
        return {"animals": 0, "files": 0, "rows": 0, "skipped": 0, "errors": 1}


def _run(store, config: dict) -> None:
    enabled, interval = _settings(config)
    logger.info("needs_scoring auto-flush worker started (every %.0f s)",
                interval)
    while True:
        try:
            s = flush_once(store, config)
            # Only log when something actually happened -- a steady state is
            # the normal case and would otherwise spam the log every pass.
            if s.get("rows") or s.get("errors"):
                logger.info("needs_scoring auto-flush: %d row(s) written over "
                            "%d file(s)/%d animal(s), %d skipped, %d error(s)",
                            s.get("rows", 0), s.get("files", 0),
                            s.get("animals", 0), s.get("skipped", 0),
                            s.get("errors", 0))
        except Exception as e:              # noqa: BLE001 -- never kill the loop
            logger.warning("needs_scoring auto-flush loop error: %s", e)
        time.sleep(interval)


def start_worker(store, config: dict) -> bool:
    """Start the flush worker unless disabled or already running. Idempotent."""
    global _thread
    enabled, _interval = _settings(config)
    if not enabled:
        logger.info("needs_scoring auto-flush disabled by config")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _thread = threading.Thread(target=_run, args=(store, config),
                                   daemon=True, name="needs-scoring-flush")
        _thread.start()
        return True


def is_running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()
