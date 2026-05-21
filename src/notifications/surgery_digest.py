"""Daily surgery digest -- summarise today's pre-op / surgery / post-op
tasks and send by email (full HTML) and SMS-via-email-gateway (short
plaintext). Reuses the surgery-tab's data path so the digest sees the
same view the dashboard does.
"""

from __future__ import annotations

import logging
from datetime import date
from html import escape

from src.dashboard.tabs import surgeries as surg
from src.alerting.email_alert import EmailAlerter

logger = logging.getLogger("qc_monitor.notifications.digest")


def _build_today(today: date, config: dict) -> tuple[dict, list[dict], list[dict]]:
    """Run the surgery-tab data path for *today*.

    Returns ``(grouped, upcoming, sheets_cfg)`` where *grouped* maps the
    five step labels to task dicts (same shape rendered in the tab) and
    *upcoming* is the next-7-days list.
    """
    cfg = surg._surgeries_cfg(config)
    sheets_cfg = cfg.get("sheets", []) or []
    aliases = surg._aliases(config)
    implant_kw = surg._implant_kw(config)
    sa_file = surg._resolve_sa_path(
        config, cfg.get("service_account_file", ""),
    )
    # Force a fresh pull regardless of cache age -- the digest fires
    # once per day, so the cost is irrelevant.
    ttl_sec = 0.0

    grouped: dict[str, list[dict]] = {g: [] for g in surg._GROUP_ORDER}
    upcoming: list[dict] = []

    for s in sheets_cfg:
        label = s.get("label") or s.get("url") or s.get("sheet_id", "?")
        header_row = int(s.get("header_row", 1))
        date_cols_cfg = s.get("date_columns") or None
        df = None
        if s.get("sheet_id") and s.get("tab_name"):
            if not sa_file:
                continue
            df = surg._load_sheet_via_api(
                s["sheet_id"], s["tab_name"], sa_file, ttl_sec,
                header_row=header_row,
            )
        elif s.get("url"):
            df = surg._load_sheet(
                s["url"], ttl_sec, header_row=header_row,
            )
        else:
            continue

        tasks, _drop = surg._tasks_from_frame(
            df, label, today, aliases, implant_kw,
            date_columns_cfg=date_cols_cfg,
        )
        for t in tasks:
            grouped.setdefault(t["step"], []).append(t)
        upcoming.extend(surg._upcoming(
            df, label, today, aliases, date_columns_cfg=date_cols_cfg,
        ))

    upcoming.sort(key=lambda x: x["in_days"])
    return grouped, upcoming, sheets_cfg


# ===================================================================== #
#  Formatting
# ===================================================================== #

def _format_text(today: date, grouped: dict, upcoming: list[dict]) -> str:
    """Plain-text digest body, used for both email fallback and SMS."""
    lines = [f"Surgery digest -- {today.isoformat()}", ""]
    any_today = False
    for group in surg._GROUP_ORDER:
        bucket = grouped.get(group, [])
        if not bucket:
            continue
        any_today = True
        lines.append(f"-- {group} --")
        for t in bucket:
            note = f" -- {t['notes']}" if t.get("notes") else ""
            lines.append(
                f"  {t['animal']} ({t['type']}, op {t['surgery_date']}){note}"
            )
        lines.append("")
    if not any_today:
        lines.append("No animals due for any step today.")
        lines.append("")
    if upcoming:
        lines.append("Upcoming (next 7 days):")
        for u in upcoming[:10]:
            lines.append(
                f"  +{u['in_days']}d  {u['animal']} ({u['type']}, "
                f"{u['surgery_date']})"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_html(today: date, grouped: dict, upcoming: list[dict]) -> str:
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;'
        'color:#1a1a2e">',
        f'<h2>Surgery digest -- {today.isoformat()}</h2>',
    ]
    any_today = False
    for group in surg._GROUP_ORDER:
        bucket = grouped.get(group, [])
        if not bucket:
            continue
        any_today = True
        parts.append(
            f'<h3 style="color:#5e7ce2;margin-top:18px">{escape(group)}</h3>'
        )
        parts.append("<ul>")
        for t in bucket:
            note = (f" -- {escape(t['notes'])}"
                    if t.get("notes") else "")
            parts.append(
                f"<li><b>{escape(t['animal'])}</b> "
                f"({escape(t['type'])}, op {escape(t['surgery_date'])})"
                f"{note}</li>"
            )
        parts.append("</ul>")
    if not any_today:
        parts.append(
            '<p style="color:#6c6c80"><em>No animals due for any step '
            'today.</em></p>'
        )
    if upcoming:
        parts.append('<h3 style="color:#5e7ce2;margin-top:18px">'
                     'Upcoming (next 7 days)</h3>')
        parts.append("<ul>")
        for u in upcoming[:10]:
            parts.append(
                f"<li>+{u['in_days']}d &mdash; <b>{escape(u['animal'])}"
                f"</b> ({escape(u['type'])}, "
                f"{escape(u['surgery_date'])})</li>"
            )
        parts.append("</ul>")
    parts.append('<hr><small style="color:#6c6c80">QC Monitor surgery '
                 'digest. Edit recipients in config.yaml -> '
                 'surgery_digest.</small>')
    parts.append("</body></html>")
    return "".join(parts)


def _format_sms(today: date, grouped: dict, upcoming: list[dict]) -> str:
    """Sub-160-char summary suitable for an SMS gateway."""
    counts = [(g, len(grouped.get(g, []))) for g in surg._GROUP_ORDER]
    counts = [(g, n) for g, n in counts if n > 0]
    if not counts:
        return f"QC: {today.isoformat()} -- no surgery tasks today."

    bits = [f"QC {today.isoformat()}:"]
    for g, n in counts:
        # Compact tag: Pre, Surg, P1, P2, P3
        if g == "Pre-op (Motrin)":
            bits.append(f"Pre={n}")
        elif g == "Surgery day":
            bits.append(f"Surg={n}")
        else:
            bits.append(f"{g.split()[-2][0]}{g.split()[-1]}={n}")
    # Add per-animal hint if room
    head = " ".join(bits)
    animals = []
    for g, _ in counts:
        for t in grouped[g][:3]:
            animals.append(f"{t['animal']}({g.split()[0][:3]})")
    tail = " " + ",".join(animals)
    msg = head + tail
    return msg[:155]  # leave a few chars of headroom


# ===================================================================== #
#  Send
# ===================================================================== #

def send_today_digest(today: date, config: dict,
                      emailer: EmailAlerter) -> dict:
    """Build today's digest and dispatch it via *emailer*.

    Returns a small dict suitable for logging:
    ``{"sent": bool, "email_recipients": int, "sms_recipients": int,
       "tasks_today": int, "upcoming_n": int}``.
    """
    cfg = config.get("surgery_digest", {}) or {}
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}

    email_to = list(cfg.get("email_recipients", []) or [])
    sms_to = list(cfg.get("sms_recipients", []) or [])
    if not email_to and not sms_to:
        return {"sent": False, "reason": "no recipients"}

    grouped, upcoming, _sheets_cfg = _build_today(today, config)
    n_tasks = sum(len(v) for v in grouped.values())

    text_body = _format_text(today, grouped, upcoming)
    html_body = _format_html(today, grouped, upcoming)
    subject = f"Surgery digest -- {today.isoformat()} ({n_tasks} task"
    subject += f"{'s' if n_tasks != 1 else ''})"

    ok_email = True
    if email_to:
        ok_email = emailer.send(
            subject=subject, body=text_body, body_html=html_body,
            recipients=email_to, subject_prefix=False,
        )

    ok_sms = True
    if sms_to:
        sms_text = _format_sms(today, grouped, upcoming)
        # SMS carriers strip HTML; send plaintext only.
        ok_sms = emailer.send(
            subject=f"QC {today.isoformat()}",
            body=sms_text, body_html=f"<pre>{escape(sms_text)}</pre>",
            recipients=sms_to, subject_prefix=False,
        )

    return {
        "sent": bool(ok_email and ok_sms),
        "email_recipients": len(email_to),
        "sms_recipients": len(sms_to),
        "tasks_today": n_tasks,
        "upcoming_n": len(upcoming),
    }
