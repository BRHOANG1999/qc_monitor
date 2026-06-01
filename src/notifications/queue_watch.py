"""Queue-stuck alarm.

Detects a stalled processing queue: no files moved to ``done`` in the
last hour AND pending count above the configured floor. Sends a
single warning per stuck episode and a single "all clear" on
recovery (so a brief stall doesn't flood the inbox).

Stale state is owned by the scheduler; this module is the dumb side
that just computes the boolean condition and assembles the email.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store

logger = logging.getLogger("qc_monitor.notifications.queue_watch")


def _pending_count(store: Store) -> int:
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM processed_files "
            "WHERE status = 'pending'"
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()


def _last_done_at(store: Store) -> datetime | None:
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT MAX(processed_at) AS ts FROM processed_files "
            "WHERE status = 'done'"
        ).fetchone()
        if not row or not row["ts"]:
            return None
        try:
            return datetime.fromisoformat(row["ts"])
        except ValueError:
            return None
    finally:
        conn.close()


def evaluate(store: Store, config: dict, now: datetime | None = None
             ) -> dict:
    """Return whether the queue is currently stuck and the inputs why.

    The scheduler compares this against its last-fire state and
    decides whether to actually send an email.
    """
    assert isinstance(store, Store), "store must be a Store"
    cfg = (config.get("notifications", {}) or {}).get("queue_watch", {}) or {}
    if not cfg.get("enabled", False):
        return {"stuck": False, "reason": "disabled"}
    min_pending = int(cfg.get("min_pending", 100))
    stall_minutes = int(cfg.get("stall_minutes", 60))
    assert min_pending >= 0, "min_pending must be >= 0"
    assert stall_minutes > 0, "stall_minutes must be positive"

    now = now or datetime.now()
    pending = _pending_count(store)
    last_done = _last_done_at(store)
    minutes_since_done = (
        (now - last_done).total_seconds() / 60.0
        if last_done else float("inf")
    )
    stuck = (pending >= min_pending
              and minutes_since_done >= stall_minutes)
    return {
        "stuck": stuck,
        "pending": pending,
        "minutes_since_done": minutes_since_done,
        "min_pending": min_pending,
        "stall_minutes": stall_minutes,
        "last_done_at": last_done.isoformat() if last_done else None,
    }


def send_stuck_alert(eval_result: dict, config: dict,
                      emailer: EmailAlerter) -> bool:
    cfg = (config.get("notifications", {}) or {}).get("queue_watch", {}) or {}
    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or [])
    )
    if not recipients:
        return False
    pending = eval_result.get("pending", 0)
    msd = eval_result.get("minutes_since_done")
    msd_str = f"{msd:.0f} min" if msd != float("inf") else "ever"
    subject = (f"QC queue stuck: {pending} pending, "
                f"no progress for {msd_str}")
    body = (
        "The QC processing queue appears stalled.\n\n"
        f"  Pending files: {pending}\n"
        f"  Minutes since last completion: {msd_str}\n"
        f"  Configured floor (min_pending): "
        f"{eval_result.get('min_pending')}\n"
        f"  Stall threshold (minutes): "
        f"{eval_result.get('stall_minutes')}\n"
        f"  Last 'done' at: "
        f"{eval_result.get('last_done_at') or '(never)'}\n\n"
        "Check the dispatcher service (NSSM) and the network share "
        "mount before this falls behind further."
    )
    ok = emailer.send(
        subject=subject, body=body, severity="warning",
        recipients=recipients,
    )
    return bool(ok)


def send_clear(eval_result: dict, config: dict,
                emailer: EmailAlerter) -> bool:
    cfg = (config.get("notifications", {}) or {}).get("queue_watch", {}) or {}
    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or [])
    )
    if not recipients:
        return False
    pending = eval_result.get("pending", 0)
    msd = eval_result.get("minutes_since_done")
    msd_str = f"{msd:.0f} min" if msd != float("inf") else "?"
    subject = "QC queue recovered"
    body = (
        "The processing queue is moving again.\n\n"
        f"  Pending files: {pending}\n"
        f"  Minutes since last completion: {msd_str}\n"
    )
    ok = emailer.send(
        subject=subject, body=body, severity="info",
        recipients=recipients,
    )
    return bool(ok)
