"""Background builder for evoked feature sidecars.

Sidecars used to be created only on demand -- when someone opened the Chronic
Evoked tab for an animal. New recordings arrive continuously, so the backlog
only grew (the coverage monitor reported 391 recordings with no sidecar), and
any view that needs features either showed nothing or tried to compute them
inline and appeared to hang.

This worker fills that gap the boring way: every *interval* it computes a small
BATCH of missing sidecars, one file at a time, then sleeps. Deliberately slow
and low-priority --

* computing one sidecar reads a whole multi-GB recording off the SMB share, so
  a flat-out catch-up would starve the dashboard and the analysis sweeps;
* the work is naturally self-limiting: it shrinks as the backlog closes and
  becomes a cheap no-op scan once coverage is complete.

It builds MISSING sidecars first (those are pure coverage gaps). Upgrading an
older-but-readable sidecar to the current schema is optional and off by default,
because those already return usable rows -- just without the newest feature
columns.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("qc_monitor.sidecar_warm")

DEFAULT_INTERVAL_SEC = 300.0
DEFAULT_BATCH = 4
_MIN_INTERVAL_SEC = 30.0

_thread: threading.Thread | None = None
_lock = threading.Lock()


def _cfg(config: dict) -> dict:
    return ((config or {}).get("chronic_evoked", {}) or {}).get(
        "sidecar_warm", {}) or {}


def _settings(config: dict) -> tuple[bool, float, int, bool]:
    c = _cfg(config)
    enabled = bool(c.get("enabled", True))
    try:
        interval = float(c.get("interval_sec", DEFAULT_INTERVAL_SEC))
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_SEC
    try:
        batch = int(c.get("batch", DEFAULT_BATCH))
    except (TypeError, ValueError):
        batch = DEFAULT_BATCH
    upgrade = bool(c.get("upgrade_outdated", False))
    return enabled, max(_MIN_INTERVAL_SEC, interval), max(1, batch), upgrade


def find_pending(config: dict, limit: int = 50, upgrade_outdated: bool = False):
    """Up to *limit* (mat_path, animal) pairs needing a sidecar, NEWEST first
    (a fresh recording is the one someone is most likely to open). Returns
    missing sidecars; with *upgrade_outdated* also older-schema ones."""
    from src.utils.evoked_output import (animals_in_filename,
                                         feature_sidecar_path,
                                         list_evoked_files, sidecar_is_current)
    import os
    evoked_dir = ((config or {}).get("chronic_evoked", {}) or {}).get(
        "evoked_output_dir", "")
    if not evoked_dir:
        return []
    out = []
    for i, fp in enumerate(reversed(list_evoked_files(evoked_dir))):
        assert i < 1_000_000, "evoked scan runaway"
        if len(out) >= limit:
            break
        for a in animals_in_filename(fp):
            if len(out) >= limit:
                break
            exists = os.path.exists(feature_sidecar_path(fp, a))
            if not exists:
                out.append((fp, a))
            elif upgrade_outdated and not sidecar_is_current(fp, a):
                out.append((fp, a))
    return out


def warm_batch(config: dict, batch: int = DEFAULT_BATCH,
               upgrade_outdated: bool = False) -> dict:
    """Build up to *batch* pending sidecars. Returns a summary; never raises."""
    from src.utils.evoked_output import read_or_compute_sidecar
    built = failed = 0
    pending = find_pending(config, limit=batch,
                           upgrade_outdated=upgrade_outdated)
    for fp, animal in pending:
        try:
            rows = read_or_compute_sidecar(fp, animal)
            if rows:
                built += 1
            else:
                failed += 1
        except Exception as e:              # noqa: BLE001 -- one bad file
            failed += 1
            logger.debug("sidecar warm failed for %s/%s: %s", fp, animal, e)
    return {"built": built, "failed": failed, "pending_seen": len(pending)}


def _run(store, config: dict) -> None:
    enabled, interval, batch, upgrade = _settings(config)
    logger.info("sidecar warm worker started (batch=%d every %.0f s, "
                "upgrade_outdated=%s)", batch, interval, upgrade)
    while True:
        try:
            s = warm_batch(config, batch=batch, upgrade_outdated=upgrade)
            if s["built"] or s["failed"]:
                logger.info("sidecar warm: built %d, failed %d "
                            "(%d pending this pass)",
                            s["built"], s["failed"], s["pending_seen"])
        except Exception as e:              # noqa: BLE001 -- never kill the loop
            logger.warning("sidecar warm loop error: %s", e)
        time.sleep(interval)


def start_worker(store, config: dict) -> bool:
    """Start the warm worker unless disabled or already running. Idempotent."""
    global _thread
    enabled, _i, _b, _u = _settings(config)
    if not enabled:
        logger.info("sidecar warm worker disabled by config")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _thread = threading.Thread(target=_run, args=(store, config),
                                   daemon=True, name="sidecar-warm")
        _thread.start()
        return True


def is_running() -> bool:
    with _lock:
        return _thread is not None and _thread.is_alive()
