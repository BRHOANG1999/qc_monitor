"""Lightweight per-user activity tracking.

Call ``track(store, area, action, target, detail)`` from any callback to record
a meaningful user action into the ``user_activity`` table. Verbs already
audited in ``review_event_log`` / ``training_attempt`` are surfaced by the
User Activity tab's unified feed, so only the *gaps* (analysis runs, file
browser, CSV finalize, auto-filter, training PI actions, tab navigation) call
this. Best-effort: identity comes from the request's ``g.user`` and a failure
never propagates into the callback.
"""

from __future__ import annotations

import logging
import threading

from src.dashboard.auth import current_user_email

logger = logging.getLogger("qc_monitor.dashboard.activity")


def track(store, area: str, action: str,
          target: str | None = None, detail: dict | None = None) -> None:
    """Record ``(current user, area, action, target, detail)``. No-op when
    there's no signed-in user; swallows all errors."""
    try:
        email = current_user_email()      # must read request context here
    except Exception:
        return
    if not email:
        return

    # Fire-and-forget: the user_activity INSERT can block up to ~30s on the
    # SQLite write lock during a background write storm. render_tab calls this
    # on EVERY tab switch, so a synchronous write stalled every render for tens
    # of seconds (observed: tab-content renders at 100-300s). Do the write on a
    # daemon thread so the request thread returns immediately; the write still
    # lands (or is swallowed) whenever it gets the lock.
    def _write():
        try:
            store.log_user_activity(email, area, action, target, detail)
        except Exception as e:  # noqa: BLE001 -- tracking must never matter
            logger.debug("activity.track failed (%s/%s): %s", area, action, e)

    try:
        threading.Thread(target=_write, name="activity-track",
                         daemon=True).start()
    except Exception as e:  # noqa: BLE001
        logger.debug("activity.track spawn failed: %s", e)
