"""Background worker for the pre-ictal sweep (mirrors mass_analyze's worker).

A single daemon thread drains the ``preictal_job`` queue FIFO, running the heavy
CWT sweep off the main loop. Cooperative cancel: the engine polls the job's
status between seizures. Stuck 'running' rows are reaped by Store._init_db on
the next restart.
"""

from __future__ import annotations

import logging
import threading

from src.preictal import engine

logger = logging.getLogger(__name__)

_worker_started = False
_worker_lock = threading.Lock()
_wake = threading.Event()


def wake() -> None:
    """Nudge the worker to check the queue now (call after enqueueing)."""
    _wake.set()


def start_worker(store, config: dict) -> None:
    """Idempotently launch the daemon thread. No-op when preictal (or its
    worker) is disabled in config."""
    global _worker_started
    pcfg = (config or {}).get("preictal", {}) or {}
    if not pcfg.get("enabled", False):
        return
    if not (pcfg.get("worker", {}) or {}).get("enabled", True):
        return
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker_loop, args=(store, config),
                     name="qc-preictal-worker", daemon=True).start()
    logger.info("preictal worker started")


def _worker_loop(store, config: dict) -> None:
    poll = float(((config or {}).get("preictal", {}).get("worker", {}) or {})
                 .get("poll_interval_sec", 5.0))
    max_iter = 10 ** 9
    i = 0
    while True:
        assert i < max_iter, "preictal worker loop runaway"
        i += 1
        try:
            job = store.claim_next_preictal_job()
            if job is None:
                _wake.wait(timeout=poll)
                _wake.clear()
                continue
            _run_one(store, config, job)
        except Exception:  # noqa: BLE001 -- worker must survive a bad iteration
            logger.exception("preictal worker iteration failed")
            _wake.wait(timeout=poll)
            _wake.clear()


def _run_one(store, config: dict, job: dict) -> None:
    jid = int(job["id"])

    def _cancelled(_jid=jid) -> bool:
        return store.preictal_job_status(_jid) == "cancelled"

    try:
        run_id = engine.run_sweep(
            store, config, scope=job["scope"],
            period_start=job.get("period_start"),
            period_end=job.get("period_end"),
            animals=job.get("animals"), cancel_fn=_cancelled)
        store.finish_preictal_job(jid, "done", run_id=run_id)
        logger.info("preictal job %s -> run %s done", jid, run_id)
    except engine.Cancelled:
        store.finish_preictal_job(jid, "cancelled")
        logger.info("preictal job %s cancelled", jid)
    except Exception as e:  # noqa: BLE001
        logger.exception("preictal job %s failed", jid)
        store.finish_preictal_job(jid, "failed", error=str(e))
