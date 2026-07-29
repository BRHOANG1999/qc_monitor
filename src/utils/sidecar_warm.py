"""Background builder for evoked feature sidecars.

Sidecars used to be created only on demand -- when someone opened the Chronic
Evoked tab for an animal. New recordings arrive continuously, so the backlog
only grew (the coverage monitor reported 391 recordings with no sidecar), and
any view that needs features either showed nothing or tried to compute them
inline and appeared to hang.

This worker fills that gap the boring way: every *interval* it computes a small
BATCH of missing sidecars, one file at a time, then sleeps. Low-priority and
self-limiting -- it shrinks as the backlog closes and becomes a cheap no-op scan
once coverage is complete.

The cadence auto-tunes to WHERE ``evoked_output_dir`` lives (``_path_is_local``):

* on a network share, one sidecar reads a multi-GB recording over SMB, so a
  flat-out catch-up would starve the dashboard and the analysis sweeps -- keep
  the gentle default (a small batch every few minutes);
* on a LOCAL fixed disk those reads are fast and private, so the multi-minute
  idle between batches is pure wasted wall-clock -- run near-continuously
  instead. Still SERIAL (one recording in memory at a time), so the memory
  profile is unchanged; only the idle shrinks.

An explicit ``interval_sec`` / ``batch`` in config always overrides the auto
choice.

It builds MISSING sidecars first (those are pure coverage gaps). Upgrading an
older-but-readable sidecar to the current schema is optional and off by default,
because those already return usable rows -- just without the newest feature
columns.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("qc_monitor.sidecar_warm")

# Gentle cadence for a network share (a multi-GB SMB read per sidecar).
DEFAULT_INTERVAL_SEC = 300.0
DEFAULT_BATCH = 4
# Brisk cadence for a local fixed disk: a short breather between batches instead
# of the multi-minute share-recovery idle. Still one recording in memory at a
# time -- only the idle changes, not the peak footprint.
LOCAL_INTERVAL_SEC = 60.0
LOCAL_BATCH = 20
_MIN_INTERVAL_SEC = 30.0
_DRIVE_FIXED = 3                    # Windows GetDriveType: local fixed disk

_thread: threading.Thread | None = None
_lock = threading.Lock()

# Live status the dashboard polls so a reviewer can tell "slow" from "frozen".
# In-process (the warmer runs in the dashboard's process) -> no DB / share cost.
_status: dict = {"phase": "starting", "file": None, "animal": None,
                 "started_at": None, "next_at": None, "pos": None,
                 "built": 0, "failed": 0}
_status_lock = threading.Lock()


def _set_status(**kw) -> None:
    with _status_lock:
        _status.update(kw)


def _incr(key: str, n: int = 1) -> None:
    with _status_lock:
        _status[key] = int(_status.get(key, 0)) + n


def status() -> dict:
    """A snapshot of what the warmer is doing right now (dashboard poll)."""
    with _status_lock:
        return dict(_status)


def _cfg(config: dict) -> dict:
    return ((config or {}).get("chronic_evoked", {}) or {}).get(
        "sidecar_warm", {}) or {}


def _evoked_dir(config: dict) -> str:
    return ((config or {}).get("chronic_evoked", {}) or {}).get(
        "evoked_output_dir", "") or ""


def _path_is_local(path: str) -> bool:
    """True only when *path* sits on a local FIXED disk (fast, private I/O).

    A UNC path, a mapped network drive, or anything we cannot positively
    classify counts as REMOTE -- the gentle SMB-friendly cadence stays the
    default whenever we are unsure, so this never accelerates a real share read
    by mistake. Windows-only detection via ``GetDriveTypeW``; degrades to False
    (remote / gentle) on any error or non-Windows host."""
    raw = (path or "").strip()
    if not raw or raw.startswith("\\\\") or raw.startswith("//"):
        return False                                  # empty or UNC -> network
    p = os.path.abspath(raw)
    if p.startswith("\\\\"):                           # abspath'd UNC
        return False
    drive = os.path.splitdrive(p)[0]                   # e.g. "D:"
    if not drive:
        return False
    try:
        import ctypes
        return int(ctypes.windll.kernel32.GetDriveTypeW(
            drive + "\\")) == _DRIVE_FIXED
    except Exception:                                  # noqa: BLE001
        return False


def _default_cadence(config: dict) -> tuple[str, float, int]:
    """(mode, interval, batch) chosen from where the evoked dir lives."""
    if _path_is_local(_evoked_dir(config)):
        return "local", LOCAL_INTERVAL_SEC, LOCAL_BATCH
    return "remote", DEFAULT_INTERVAL_SEC, DEFAULT_BATCH


def _settings(config: dict) -> tuple[bool, float, int, bool]:
    c = _cfg(config)
    _mode, d_interval, d_batch = _default_cadence(config)
    enabled = bool(c.get("enabled", True))
    try:
        interval = float(c.get("interval_sec", d_interval))
    except (TypeError, ValueError):
        interval = d_interval
    try:
        batch = int(c.get("batch", d_batch))
    except (TypeError, ValueError):
        batch = d_batch
    upgrade = bool(c.get("upgrade_outdated", False))
    return enabled, max(_MIN_INTERVAL_SEC, interval), max(1, batch), upgrade


def find_pending(config: dict, limit: int = 50, upgrade_outdated: bool = False):
    """Up to *limit* (mat_path, animal) pairs needing a sidecar, NEWEST first
    (a fresh recording is the one someone is most likely to open). Returns
    missing sidecars; with *upgrade_outdated* also older-schema ones."""
    from src.utils.evoked_output import (animals_in_filename,
                                         feature_sidecar_path,
                                         list_evoked_files, read_feature_sidecar,
                                         sidecar_is_current)
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
            if not os.path.exists(feature_sidecar_path(fp, a)):
                out.append((fp, a))              # missing
            elif read_feature_sidecar(fp, a) is None:
                # Exists but the BUILD would REJECT it -- a non-additive schema
                # bump (v5) makes every old sidecar unreadable, so it is
                # effectively missing and MUST be rebuilt regardless of the
                # 'upgrade_outdated' flag (which only gates the optional refresh
                # of a still-READABLE older-but-additive sidecar). Without this
                # the corpus never re-warms after a non-additive bump and every
                # build stays empty.
                out.append((fp, a))
            elif upgrade_outdated and not sidecar_is_current(fp, a):
                out.append((fp, a))
    return out


def warm_batch(config: dict, batch: int = DEFAULT_BATCH,
               upgrade_outdated: bool = False) -> dict:
    """Build up to *batch* pending sidecars. Returns a summary; never raises.
    Publishes per-file live status (``status()``) so the dashboard can show
    active progress."""
    import os
    from src.utils.evoked_output import read_or_compute_sidecar
    built = failed = 0
    pending = find_pending(config, limit=batch,
                           upgrade_outdated=upgrade_outdated)
    for i, (fp, animal) in enumerate(pending):
        _set_status(phase="warming", file=os.path.basename(fp), animal=animal,
                    started_at=time.time(), pos=(i + 1, len(pending)))
        try:
            rows = read_or_compute_sidecar(fp, animal)
            if rows:
                built += 1
                _incr("built")
            else:
                failed += 1
                _incr("failed")
        except Exception as e:              # noqa: BLE001 -- one bad file
            failed += 1
            _incr("failed")
            logger.debug("sidecar warm failed for %s/%s: %s", fp, animal, e)
    return {"built": built, "failed": failed, "pending_seen": len(pending)}


def _run(store, config: dict) -> None:
    enabled, interval, batch, upgrade = _settings(config)
    mode, _di, _db = _default_cadence(config)
    logger.info("sidecar warm worker started (%s dir: batch=%d every %.0f s, "
                "upgrade_outdated=%s)", mode, batch, interval, upgrade)
    while True:
        try:
            s = warm_batch(config, batch=batch, upgrade_outdated=upgrade)
            if s["built"] or s["failed"]:
                logger.info("sidecar warm: built %d, failed %d "
                            "(%d pending this pass)",
                            s["built"], s["failed"], s["pending_seen"])
            # Sleeping with a next-batch time so the dashboard shows a countdown
            # (idle between 5-min batches must not read as "frozen").
            _set_status(phase="idle" if s["pending_seen"] == 0 else "sleeping",
                        file=None, animal=None, pos=None,
                        next_at=time.time() + interval)
        except Exception as e:              # noqa: BLE001 -- never kill the loop
            logger.warning("sidecar warm loop error: %s", e)
            _set_status(phase="error", next_at=time.time() + interval)
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
