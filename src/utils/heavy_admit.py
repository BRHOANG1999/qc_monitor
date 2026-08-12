"""Process-wide admission gate for heavy background dashboard compute.

The peri-ictal tab builds its matrix / ERP / slow-dynamics / sliding-AUC views on
background daemon threads inside the single dashboard process. Nothing caps how
many DISTINCT builds run at once: N users with N different selections spawn N
build threads, and each (post the evoked off-proc split) also forks a ~600 MB
reader child. Unbounded, they jointly saturate every core and pressure memory --
starving each other and any interactive user.

This gate bounds how many heavy builds RUN concurrently. A build that can't get a
slot parks on the semaphore (cheap for a waiting thread) and, if given a progress
callback, publishes a "waiting…" status so its dcc.Interval poll still shows life
-- the loading UX is preserved. It caps CONCURRENCY, not the number of queued
threads: only <= N do real work at once, however many selections are in flight.

Interactive Video Review renders are deliberately NOT gated here -- they are
per-interaction (bounded by user pace + the waitress pool) and mostly release the
GIL, so making them queue behind a background build would only add latency. The
gate targets the unbounded background builds, which are the real blowup.

Size: ``min(cpu-1, 3)`` by default; override with ``QC_HEAVY_COMPUTE_SLOTS`` or
``configure(n)`` at startup.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger("qc_monitor.utils.heavy_admit")


def _default_slots() -> int:
    env = os.environ.get("QC_HEAVY_COMPUTE_SLOTS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            logger.warning("ignoring non-integer QC_HEAVY_COMPUTE_SLOTS=%r", env)
    cpu = os.cpu_count() or 2
    return max(1, min(cpu - 1, 3))


_SLOTS = _default_slots()
_sem = threading.BoundedSemaphore(_SLOTS)
_cfg_lock = threading.Lock()


def limit() -> int:
    """Current number of heavy-compute slots."""
    return _SLOTS


def configure(n: int) -> None:
    """Resize the gate. Call once at startup before any ``admit()`` -- resizing
    while builds hold slots would desync the count."""
    global _SLOTS, _sem
    n = max(1, int(n))
    with _cfg_lock:
        _SLOTS = n
        _sem = threading.BoundedSemaphore(n)
    logger.info("heavy-compute gate sized to %d slot(s)", n)


@contextlib.contextmanager
def admit(label: str = "", progress=None):
    """Hold one heavy-compute slot for the duration of the ``with`` body.

    Tries non-blocking first, so the uncontended case never invokes *progress*;
    on contention it publishes a "waiting…" status (when *progress* is given) and
    parks until a slot frees. The captured semaphore is released even if resized
    meanwhile, so the acquire/release always pair on the same object."""
    sem = _sem                                  # capture: release the one we took
    if not sem.acquire(blocking=False):
        if progress is not None:
            try:
                progress("waiting for a compute slot…")
            except Exception:  # noqa: BLE001 -- progress must never break admission
                pass
        logger.info("heavy-compute gate full (%d slots); '%s' waiting",
                    _SLOTS, label)
        sem.acquire()
    try:
        yield
    finally:
        try:
            sem.release()
        except (ValueError, RuntimeError):      # over-release: defensive no-op
            pass
