"""Bound a blocking call to a wall-clock deadline.

Runs a (usually I/O-blocking) function on a throwaway daemon thread and waits at
most ``timeout_sec`` for it. If it finishes, returns its result (or re-raises the
exception it raised). If it doesn't, returns ``default`` and ABANDONS the worker
thread -- it keeps running as a daemon and dies with the process.

This is the portable way to stop a hung network/file read from wedging a request
thread indefinitely: Windows has no ``SIGALRM``, and a blocked C-level read
(``os.stat``, ``h5py.File``) can't be interrupted in-thread anyway. The Overview
evoked-thumbnail fallback read a session ``.mat`` off a network share with no
timeout, *under a single-flight lock* -- one hung share read froze the whole page
(every later refresh tick bailed on the never-released lock).

Cost: a truly-hung call leaks one daemon thread until the process exits. That's
acceptable because it only happens on a real infra stall and is bounded by how
often the caller runs (the callers here are single-flighted and TTL-cached).
"""

from __future__ import annotations

import threading


class _Holder:
    """Mutable cross-thread result slot (no lock needed: the worker only writes,
    the caller only reads after join(), and ``done`` is set last)."""

    __slots__ = ("value", "error", "done")

    def __init__(self):
        self.value = None
        self.error = None
        self.done = False


def call_with_deadline(fn, timeout_sec, default=None):
    """Return ``fn()`` if it completes within *timeout_sec*, else *default*.

    Re-raises any exception ``fn()`` raised, but ONLY when ``fn`` actually
    finished; on a timeout the worker thread is abandoned (daemon) and *default*
    is returned so the caller is never blocked longer than *timeout_sec*.
    """
    assert callable(fn), "fn must be callable"
    assert timeout_sec and timeout_sec > 0, "timeout_sec must be > 0"
    holder = _Holder()

    def _run():
        try:
            holder.value = fn()
        except Exception as e:              # noqa: BLE001 -- surface to caller
            holder.error = e
        finally:
            holder.done = True              # set LAST so the caller's read is safe

    t = threading.Thread(target=_run, name="deadline-call", daemon=True)
    t.start()
    t.join(timeout_sec)
    if not holder.done:
        return default                      # timed out -> abandon the worker
    if holder.error is not None:
        raise holder.error
    return holder.value
