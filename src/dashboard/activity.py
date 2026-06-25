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

from src.dashboard.auth import current_user_email

logger = logging.getLogger("qc_monitor.dashboard.activity")


def track(store, area: str, action: str,
          target: str | None = None, detail: dict | None = None) -> None:
    """Record ``(current user, area, action, target, detail)``. No-op when
    there's no signed-in user; swallows all errors."""
    try:
        email = current_user_email()
        if not email:
            return
        store.log_user_activity(email, area, action, target, detail)
    except Exception as e:  # noqa: BLE001 -- tracking must never break a tab
        logger.debug("activity.track failed (%s/%s): %s", area, action, e)
