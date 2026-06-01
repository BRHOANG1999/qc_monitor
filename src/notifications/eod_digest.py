"""End-of-day QC summary email.

Once per day (default 6 pm local), summarise the day's processing:
files processed, errors, the top-3 worst chunks by artifact_pct, the
top-3 worst by line_noise_ratio, and per-session done/error split.
Sent to ``alerting.smtp.recipients`` (or an override list under
``notifications.eod_digest.recipients``).

Idempotent across restarts via the shared notification-state JSON
file; the scheduler is the single owner of state I/O.
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from html import escape

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store

logger = logging.getLogger("qc_monitor.notifications.eod_digest")


def _today_iso_bounds(today: date) -> tuple[str, str]:
    """Return ISO datetime strings for today's [00:00, 24:00)."""
    start = datetime.combine(today, datetime.min.time()).isoformat()
    end = datetime.combine(today, datetime.max.time()).isoformat()
    return start, end


def _collect_eod_stats(store: Store, today: date) -> dict:
    """Pull every piece of data the digest needs in one shot.

    Returns a flat dict so the formatters can stay dumb.
    """
    assert isinstance(store, Store), "store must be a Store"
    start, end = _today_iso_bounds(today)
    conn = store._connect()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN status='done'    THEN 1 ELSE 0 END) AS done,
                 SUM(CASE WHEN status='error'   THEN 1 ELSE 0 END) AS errors,
                 SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending
               FROM processed_files
               WHERE processed_at BETWEEN ? AND ?""",
            (start, end),
        ).fetchone()
        totals = dict(row) if row else {"total": 0, "done": 0,
                                          "errors": 0, "pending": 0}

        # Top-3 worst chunks by artifact_pct (today only).
        worst_artifact = [dict(r) for r in conn.execute(
            """SELECT pf.session_name, pf.file_path,
                       cq.channel_name, cq.artifact_pct
               FROM chunk_qc cq
               JOIN processed_files pf ON pf.id = cq.file_id
               WHERE pf.processed_at BETWEEN ? AND ?
                 AND cq.artifact_pct IS NOT NULL
               ORDER BY cq.artifact_pct DESC
               LIMIT 3""",
            (start, end),
        ).fetchall()]

        worst_line_noise = [dict(r) for r in conn.execute(
            """SELECT pf.session_name, pf.file_path,
                       cq.channel_name, cq.line_noise_ratio
               FROM chunk_qc cq
               JOIN processed_files pf ON pf.id = cq.file_id
               WHERE pf.processed_at BETWEEN ? AND ?
                 AND cq.line_noise_ratio IS NOT NULL
               ORDER BY cq.line_noise_ratio DESC
               LIMIT 3""",
            (start, end),
        ).fetchall()]

        sessions_touched = [dict(r) for r in conn.execute(
            """SELECT session_name, session_dir,
                      COUNT(*) AS n_files,
                      SUM(CASE WHEN status='done'  THEN 1 ELSE 0 END) AS done,
                      SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
               FROM processed_files
               WHERE processed_at BETWEEN ? AND ?
               GROUP BY session_dir
               ORDER BY n_files DESC""",
            (start, end),
        ).fetchall()]
    finally:
        conn.close()

    return {
        "totals": totals,
        "worst_artifact": worst_artifact,
        "worst_line_noise": worst_line_noise,
        "sessions": sessions_touched,
    }


def _format_text(today: date, stats: dict) -> str:
    t = stats["totals"]
    lines = [
        f"QC end-of-day summary -- {today.isoformat()}",
        "",
        f"  Files processed today: {t.get('total', 0)}",
        f"    done:    {t.get('done', 0)}",
        f"    errors:  {t.get('errors', 0)}",
        f"    pending: {t.get('pending', 0)}",
        "",
    ]
    if stats["worst_artifact"]:
        lines.append("Worst artifact_pct (top 3):")
        for r in stats["worst_artifact"]:
            lines.append(
                f"  {r.get('artifact_pct', 0):.1f}%  "
                f"{r.get('channel_name', '?')}  "
                f"{r.get('session_name', '?')}"
            )
        lines.append("")
    if stats["worst_line_noise"]:
        lines.append("Worst line_noise_ratio (top 3):")
        for r in stats["worst_line_noise"]:
            lines.append(
                f"  {r.get('line_noise_ratio', 0):.3f}  "
                f"{r.get('channel_name', '?')}  "
                f"{r.get('session_name', '?')}"
            )
        lines.append("")
    if stats["sessions"]:
        lines.append("Sessions touched today:")
        for s in stats["sessions"]:
            lines.append(
                f"  {s.get('session_name', '?')}: "
                f"{s.get('done', 0)} done / {s.get('errors', 0)} err "
                f"({s.get('n_files', 0)} files)"
            )
    return "\n".join(lines).rstrip()


def _format_html(today: date, stats: dict) -> str:
    t = stats["totals"]
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;'
        'color:#1a1a2e">',
        f'<h2>QC end-of-day summary -- {today.isoformat()}</h2>',
        '<table cellpadding="6" style="border-collapse:collapse">',
        f'<tr><td><b>Files processed</b></td><td>{t.get("total", 0)}</td></tr>',
        f'<tr><td>Done</td><td style="color:#1aa260">{t.get("done", 0)}</td></tr>',
        f'<tr><td>Errors</td><td style="color:#c0392b">{t.get("errors", 0)}</td></tr>',
        f'<tr><td>Pending</td><td style="color:#b8860b">{t.get("pending", 0)}</td></tr>',
        '</table>',
    ]
    if stats["worst_artifact"]:
        parts.append('<h3 style="color:#5e7ce2">Worst artifact_pct</h3><ul>')
        for r in stats["worst_artifact"]:
            parts.append(
                f"<li><b>{r.get('artifact_pct', 0):.1f}%</b> "
                f"&mdash; {escape(str(r.get('channel_name', '?')))} "
                f"in {escape(str(r.get('session_name', '?')))}</li>"
            )
        parts.append('</ul>')
    if stats["worst_line_noise"]:
        parts.append('<h3 style="color:#5e7ce2">Worst line-noise ratio</h3><ul>')
        for r in stats["worst_line_noise"]:
            parts.append(
                f"<li><b>{r.get('line_noise_ratio', 0):.3f}</b> "
                f"&mdash; {escape(str(r.get('channel_name', '?')))} "
                f"in {escape(str(r.get('session_name', '?')))}</li>"
            )
        parts.append('</ul>')
    if stats["sessions"]:
        parts.append('<h3 style="color:#5e7ce2">Sessions touched today</h3><ul>')
        for s in stats["sessions"]:
            parts.append(
                f"<li><b>{escape(str(s.get('session_name', '?')))}</b>: "
                f"{s.get('done', 0)} done / {s.get('errors', 0)} err "
                f"({s.get('n_files', 0)} files)</li>"
            )
        parts.append('</ul>')
    parts.append(
        '<hr><small style="color:#6c6c80">QC Monitor end-of-day digest. '
        'Edit recipients in config.yaml -> notifications.eod_digest.</small>'
        '</body></html>'
    )
    return "".join(parts)


def send_eod_digest(today: date, config: dict, store: Store,
                     emailer: EmailAlerter) -> dict:
    """Build today's EOD summary and dispatch it via *emailer*."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    cfg = (config.get("notifications", {}) or {}).get("eod_digest", {}) or {}
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}

    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or [])
    )
    if not recipients:
        return {"sent": False, "reason": "no recipients"}

    stats = _collect_eod_stats(store, today)
    text_body = _format_text(today, stats)
    html_body = _format_html(today, stats)
    subject = f"QC end-of-day summary -- {today.isoformat()}"

    ok = emailer.send(
        subject=subject, body=text_body, body_html=html_body,
        recipients=recipients, subject_prefix=False,
    )
    return {
        "sent": bool(ok),
        "recipients": len(recipients),
        "files_today": stats["totals"].get("total", 0),
        "errors_today": stats["totals"].get("errors", 0),
    }
