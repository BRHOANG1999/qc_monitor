"""Cross-thread 'what is the pipeline doing right now' progress for the nav bar.

A slow tab (Overview builds ~12 DB-backed cards synchronously) blocks its server
thread with no feedback, so the page reads as frozen. The builder publishes its
current stage here; the nav status-poll callback -- which runs on ANOTHER server
thread (the dashboard is served multi-threaded: a waitress pool or Flask
``threaded=True``) -- reads it and shows it under the header pill.

The publishing side uses a thread-local for "who am I building for", so builders
deep in the call tree call ``stage(text)`` without threading a user id through
every function. The shared ``_PROGRESS`` map (user -> stage text) is read by the
poll thread under a lock. Best-effort telemetry ONLY: every function swallows its
own errors so instrumentation can never break a render.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_PROGRESS: dict[str, str] = {}          # user_email -> current stage text
_TL = threading.local()                 # per-thread: which user this build is for


def begin(user: str | None) -> None:
    """Mark the calling thread as building for *user*; clears any stale stage."""
    try:
        _TL.user = user or ""
        stage("")
    except Exception:                   # never raise into a render
        pass


def stage(text: str) -> None:
    """Publish the current pipeline stage for this thread's user."""
    try:
        user = getattr(_TL, "user", "")
        if not user:
            return
        with _LOCK:
            _PROGRESS[user] = text or ""
    except Exception:
        pass


def end() -> None:
    """Clear this thread's user progress (call in a finally after the build)."""
    try:
        user = getattr(_TL, "user", "")
        if user:
            with _LOCK:
                _PROGRESS.pop(user, None)
        _TL.user = ""
    except Exception:
        pass


def get(user: str | None) -> str:
    """Current stage text for *user*, or '' when idle/unknown."""
    try:
        if not user:
            return ""
        with _LOCK:
            return _PROGRESS.get(user, "")
    except Exception:
        return ""
