"""Weekly video QC roll-up.

Once a week (default Monday 8 am), summarise the past 7 days of
Tier 3 video QC: counts by status, top-5 worst clips by pct_bad,
and a short tail of the worst-reason breakdown. Tier 3 is new so a
weekly cadence keeps inbox noise low while still surfacing camera
trouble before it becomes systemic.

Recipients default to ``alerting.smtp.recipients``; override with
``notifications.video_weekly.recipients``.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from html import escape

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store

logger = logging.getLogger("qc_monitor.notifications.video_weekly")


def _format_text(today: date, rows: list[dict]) -> str:
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_warn = sum(1 for r in rows if r.get("status") == "warning")
    n_crit = sum(1 for r in rows if r.get("status") == "critical")
    lines = [
        f"Weekly video QC -- {today.isoformat()}",
        f"  (rolling 7-day window, {len(rows)} videos analyzed)",
        "",
        f"  OK:       {n_ok}",
        f"  Warning:  {n_warn}",
        f"  Critical: {n_crit}",
        "",
    ]
    worst = sorted(
        [r for r in rows
         if r.get("pct_bad") is not None
         and r.get("status") in ("warning", "critical")],
        key=lambda r: r.get("pct_bad", 0), reverse=True,
    )[:5]
    if worst:
        lines.append("Worst clips (top 5 by pct_bad):")
        for r in worst:
            name = os.path.basename(r.get("video_path") or "?")
            reason = r.get("first_bad_reason") or "?"
            lines.append(
                f"  {r.get('pct_bad', 0):.1f}%  {name}  ({reason})"
            )
    return "\n".join(lines).rstrip()


def _format_html(today: date, rows: list[dict]) -> str:
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_warn = sum(1 for r in rows if r.get("status") == "warning")
    n_crit = sum(1 for r in rows if r.get("status") == "critical")
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;'
        'color:#1a1a2e">',
        f'<h2>Weekly video QC -- {today.isoformat()}</h2>',
        f'<p style="color:#6c6c80">Rolling 7-day window, '
        f'{len(rows)} videos analyzed.</p>',
        '<table cellpadding="6" style="border-collapse:collapse">',
        f'<tr><td>OK</td><td style="color:#1aa260">{n_ok}</td></tr>',
        f'<tr><td>Warning</td><td style="color:#b8860b">{n_warn}</td></tr>',
        f'<tr><td>Critical</td><td style="color:#c0392b">{n_crit}</td></tr>',
        '</table>',
    ]
    worst = sorted(
        [r for r in rows
         if r.get("pct_bad") is not None
         and r.get("status") in ("warning", "critical")],
        key=lambda r: r.get("pct_bad", 0), reverse=True,
    )[:5]
    if worst:
        parts.append('<h3 style="color:#5e7ce2">Worst clips '
                     '(top 5 by pct_bad)</h3><ul>')
        for r in worst:
            name = os.path.basename(r.get("video_path") or "?")
            reason = r.get("first_bad_reason") or "?"
            parts.append(
                f"<li><b>{r.get('pct_bad', 0):.1f}%</b> &mdash; "
                f"{escape(name)} ({escape(str(reason))})</li>"
            )
        parts.append('</ul>')
    parts.append(
        '<hr><small style="color:#6c6c80">QC Monitor video weekly digest. '
        'Edit recipients in config.yaml -> notifications.video_weekly.</small>'
        '</body></html>'
    )
    return "".join(parts)


def send_video_weekly(today: date, config: dict, store: Store,
                       emailer: EmailAlerter) -> dict:
    """Build the rolling-7d video QC summary and dispatch it."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    cfg = (config.get("notifications", {}) or {}).get("video_weekly", {}) or {}
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}

    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or [])
    )
    if not recipients:
        return {"sent": False, "reason": "no recipients"}

    rows = store.get_recent_video_qc(hours=168)
    if not rows:
        return {"sent": False, "reason": "no video QC data this week"}

    subject = f"QC weekly video roll-up -- {today.isoformat()}"
    text_body = _format_text(today, rows)
    html_body = _format_html(today, rows)
    ok = emailer.send(
        subject=subject, body=text_body, body_html=html_body,
        recipients=recipients, subject_prefix=False,
    )
    return {
        "sent": bool(ok),
        "recipients": len(recipients),
        "videos_analyzed": len(rows),
    }
