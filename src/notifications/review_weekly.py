"""Weekly PI review-status digest.

Once a week (default Friday 5 pm), emails every address in
``review_queue.pi_emails`` a summary of:

  * Per-animal backlog (count + oldest unreviewed age + assignees).
  * Per-user throughput over the last 7 days.
  * Animals with zero active assignees (and yet a backlog).
  * Any animal whose oldest unreviewed file > crit_age_days.

Mirrors the patterns in ``video_weekly.py`` / ``evoked_weekly.py``
so the existing DigestScheduler picks it up uniformly.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from html import escape

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store
from src.utils import assignments as _assignments

logger = logging.getLogger("qc_monitor.notifications.review_weekly")


def _fmt_age(chunk_dt_str: str | None) -> str:
    if not chunk_dt_str:
        return "?"
    try:
        dt = datetime.strptime(chunk_dt_str,
                                 "%Y_%m_%d__%H_%M_%S")
    except ValueError:
        return chunk_dt_str
    age_days = (datetime.now() - dt).total_seconds() / 86400.0
    if age_days < 1:
        return f"{age_days * 24:.0f} h"
    return f"{age_days:.1f} d"


def _gather(today: date, config: dict, store: Store) -> dict:
    cfg = (config or {}).get("review_queue", {}) or {}
    queue_cfg = cfg.get("queue", {}) or {}
    crit_days = int(queue_cfg.get("crit_age_days", 7))

    floor = store.review_backlog_floor()
    backlog = store.review_backlog_summary(since_iso=floor)
    throughput = store.review_user_throughput(days=7)
    assignments = _assignments.load_assignments(config)
    by_animal: dict[str, list[str]] = {}
    for a in assignments:
        by_animal.setdefault(a.animal_id, []).append(a.user_email)

    enriched_backlog = []
    crit = []
    orphaned = []
    for row in backlog:
        animal = row["animal_id"]
        oldest = row.get("oldest_chunk_datetime") or ""
        try:
            dt = datetime.strptime(oldest,
                                     "%Y_%m_%d__%H_%M_%S")
            age_days = (datetime.now() - dt).total_seconds() / 86400
        except ValueError:
            age_days = 0
        item = {
            **row,
            "age_label": _fmt_age(oldest),
            "age_days": age_days,
            "assignees": by_animal.get(animal, []),
        }
        enriched_backlog.append(item)
        if age_days >= crit_days:
            crit.append(item)
        if not by_animal.get(animal):
            orphaned.append(item)

    return {
        "backlog": enriched_backlog,
        "crit": crit,
        "orphaned": orphaned,
        "throughput": throughput,
        "crit_days": crit_days,
        "total_unreviewed": sum(r["n_unreviewed"]
                                  for r in enriched_backlog),
    }


def _format_text(today: date, data: dict) -> str:
    lines = [
        f"Review status weekly -- {today.isoformat()}",
        f"  Total unreviewed: {data['total_unreviewed']}",
        f"  Animals with backlog: {len(data['backlog'])}",
        f"  Critical (>= {data['crit_days']} days old): "
        f"{len(data['crit'])}",
        f"  Orphaned (no assignee): {len(data['orphaned'])}",
        "",
        "Top 10 animals by backlog:",
    ]
    for r in data["backlog"][:10]:
        assignees = (", ".join(r["assignees"])
                      if r["assignees"] else "(no one)")
        lines.append(
            f"  {r['animal_id']:<14} "
            f"{r['n_unreviewed']:>5} files  "
            f"oldest {r['age_label']:<8}  "
            f"-> {assignees}"
        )
    lines.append("")
    if data["crit"]:
        lines.append("Critical-age animals:")
        for r in data["crit"]:
            lines.append(
                f"  {r['animal_id']} -- oldest {r['age_label']}"
            )
        lines.append("")
    if data["orphaned"]:
        lines.append("Orphaned animals (no active assignee):")
        for r in data["orphaned"][:15]:
            lines.append(
                f"  {r['animal_id']} -- "
                f"{r['n_unreviewed']} files waiting"
            )
        lines.append("")
    if data["throughput"]:
        lines.append("Reviewer throughput (last 7 days):")
        for r in data["throughput"]:
            lines.append(
                f"  {r['user_email']:<32} "
                f"{r['n_total']:>3} files  "
                f"(no-events {r['n_no_events']}, "
                f"with-events {r['n_has_events']})"
            )
    return "\n".join(lines).rstrip()


def _format_html(today: date, data: dict) -> str:
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;'
        'color:#1a1a2e">',
        f'<h2>Review status weekly -- {today.isoformat()}</h2>',
        '<table cellpadding="6" style="border-collapse:collapse">',
        f'<tr><td><b>Total unreviewed</b></td>'
        f'<td>{data["total_unreviewed"]}</td></tr>',
        f'<tr><td>Animals with backlog</td>'
        f'<td>{len(data["backlog"])}</td></tr>',
        f'<tr><td>Critical (>= {data["crit_days"]} days)</td>'
        f'<td style="color:#c0392b">{len(data["crit"])}</td></tr>',
        f'<tr><td>Orphaned (no assignee)</td>'
        f'<td style="color:#b8860b">{len(data["orphaned"])}</td></tr>',
        '</table>',
    ]
    if data["backlog"]:
        parts.append('<h3 style="color:#5e7ce2">'
                      'Top 10 by backlog</h3><ul>')
        for r in data["backlog"][:10]:
            assignees = (", ".join(escape(a)
                                     for a in r["assignees"])
                         or "(no one)")
            parts.append(
                f"<li><b>{escape(r['animal_id'])}</b> &mdash; "
                f"{r['n_unreviewed']} files, oldest "
                f"{escape(r['age_label'])} &rarr; {assignees}"
                f"</li>"
            )
        parts.append('</ul>')
    if data["crit"]:
        parts.append('<h3 style="color:#c0392b">Critical age</h3>'
                      '<ul>')
        for r in data["crit"]:
            parts.append(
                f"<li>{escape(r['animal_id'])} &mdash; "
                f"oldest {escape(r['age_label'])}</li>"
            )
        parts.append('</ul>')
    if data["orphaned"]:
        parts.append('<h3 style="color:#b8860b">'
                      'Orphaned animals</h3><ul>')
        for r in data["orphaned"][:15]:
            parts.append(
                f"<li>{escape(r['animal_id'])} &mdash; "
                f"{r['n_unreviewed']} files waiting</li>"
            )
        parts.append('</ul>')
    if data["throughput"]:
        parts.append('<h3 style="color:#5e7ce2">'
                      'Reviewer throughput (last 7 d)</h3>'
                      '<table cellpadding="6" '
                      'style="border-collapse:collapse;'
                      'border:1px solid #ddd">'
                      '<tr style="background:#f4f6fb">'
                      '<th>Reviewer</th><th>Total</th>'
                      '<th>No events</th><th>With events</th>'
                      '</tr>')
        for r in data["throughput"]:
            parts.append(
                f'<tr><td>{escape(r["user_email"])}</td>'
                f'<td>{r["n_total"]}</td>'
                f'<td>{r["n_no_events"]}</td>'
                f'<td>{r["n_has_events"]}</td></tr>'
            )
        parts.append('</table>')
    parts.append(
        '<hr><small style="color:#6c6c80">QC Monitor review-status '
        'weekly digest. Edit recipients in '
        'config.yaml -> review_queue.pi_emails.</small>'
        '</body></html>'
    )
    return "".join(parts)


def send_review_weekly(today: date, config: dict, store: Store,
                        emailer: EmailAlerter) -> dict:
    """Build and dispatch the weekly review-status summary."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    cfg = (config or {}).get("review_queue", {}) or {}
    wd = cfg.get("weekly_digest", {}) or {}
    if not wd.get("enabled", False):
        return {"sent": False, "reason": "disabled"}
    recipients = list(cfg.get("pi_emails", []) or [])
    if not recipients:
        return {"sent": False, "reason": "no PI recipients"}
    data = _gather(today, config, store)
    if data["total_unreviewed"] == 0 and not data["throughput"]:
        return {"sent": False, "reason": "nothing to report"}
    subject = (f"QC review status -- {today.isoformat()} "
                f"({data['total_unreviewed']} unreviewed)")
    text = _format_text(today, data)
    html = _format_html(today, data)
    ok = emailer.send(
        subject=subject, body=text, body_html=html,
        recipients=recipients, subject_prefix=False,
    )
    return {
        "sent": bool(ok),
        "recipients": len(recipients),
        "total_unreviewed": data["total_unreviewed"],
        "n_critical": len(data["crit"]),
        "n_orphaned": len(data["orphaned"]),
    }
