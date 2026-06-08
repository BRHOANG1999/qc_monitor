"""PI review-status tab.

Shows two tables:

1. Per-animal backlog — n_unreviewed + oldest_chunk_datetime + the
   current assignee(s) pulled from the Reviewer Assignments sheet.
2. Per-user throughput over the last 7 days — counts of
   ``no_events`` vs ``has_events`` finalisations + when they
   last finished anything.

Gated on ``config.review_queue.pi_emails`` so only PIs see it.
Everyone else gets a friendly "not authorised" card.
"""

from __future__ import annotations

import logging
from datetime import datetime

from dash import dash_table, html, dcc

from src.db.store import Store
from src.utils import assignments as _assignments
from src.dashboard.auth import current_user_email

logger = logging.getLogger("qc_monitor.dashboard.review_status")

DARK_TABLE_STYLE = dict(
    style_cell={
        "backgroundColor": "#13131f", "color": "#f0f0f5",
        "border": "1px solid rgba(255,255,255,0.05)",
        "padding": "8px 10px", "fontSize": "12px",
        "textAlign": "left",
        "fontFamily": "ui-monospace, SF Mono, monospace",
    },
    style_header={
        "backgroundColor": "#0a0a14", "color": "#a0a0b0",
        "fontWeight": "600", "fontSize": "11px",
        "textTransform": "uppercase", "letterSpacing": "0.4px",
    },
    style_data={"whiteSpace": "normal"},
)


def _is_pi(config: dict, email: str | None) -> bool:
    if not email:
        return False
    pi_emails = (((config or {}).get("review_queue", {}) or {})
                  .get("pi_emails", []) or [])
    return email.lower() in {e.lower() for e in pi_emails}


def _fmt_age(chunk_dt_str: str | None) -> str:
    if not chunk_dt_str:
        return "—"
    try:
        dt = datetime.strptime(chunk_dt_str,
                                 "%Y_%m_%d__%H_%M_%S")
    except ValueError:
        return chunk_dt_str
    age_days = (datetime.now() - dt).total_seconds() / 86400.0
    if age_days < 1:
        return f"{age_days * 24:.0f} h ago"
    return f"{age_days:.1f} d ago"


def layout(store: Store, config: dict | None = None):
    email = current_user_email()
    if not _is_pi(config or {}, email):
        return html.Div([
            html.Div("🔒  Not authorised",
                     style={"fontSize": "18px",
                            "fontWeight": "600",
                            "color": "#f0f0f5",
                            "marginBottom": "8px"}),
            html.Div(
                "This tab is for PIs only. Ask the lab lead to "
                "add your email to "
                "config.review_queue.pi_emails to access it.",
                style={"color": "#a0a0b0", "fontSize": "13px",
                        "maxWidth": "520px"}),
        ], style={"padding": "40px 24px"})

    queue_cfg = (((config or {}).get("review_queue", {}) or {})
                  .get("queue", {}) or {})
    warn_days = int(queue_cfg.get("warn_age_days", 3))
    crit_days = int(queue_cfg.get("crit_age_days", 7))

    # --- Per-animal backlog -------------------------------------- #
    floor = store.review_backlog_floor()
    backlog = store.review_backlog_summary(since_iso=floor)
    assignments = _assignments.load_assignments(config or {})
    by_animal: dict[str, list[str]] = {}
    for a in assignments:
        by_animal.setdefault(a.animal_id, []).append(a.user_email)

    backlog_rows = []
    backlog_alerts = 0
    for row in backlog:
        animal = row["animal_id"]
        oldest = row.get("oldest_chunk_datetime") or ""
        age_label = _fmt_age(oldest)
        try:
            dt = datetime.strptime(oldest,
                                     "%Y_%m_%d__%H_%M_%S")
            age_days = (datetime.now() - dt).total_seconds() / 86400
        except ValueError:
            age_days = 0
        flag = ("🔴 crit" if age_days >= crit_days
                else "🟠 warn" if age_days >= warn_days
                else "")
        if flag:
            backlog_alerts += 1
        assignees = by_animal.get(animal) or []
        backlog_rows.append({
            "animal": animal,
            "n_unreviewed": row["n_unreviewed"],
            "oldest": age_label,
            "flag": flag,
            "assignees": ", ".join(assignees) if assignees
                         else "— unassigned —",
        })

    # --- Per-user throughput (last 7 days) ----------------------- #
    throughput = store.review_user_throughput(days=7)
    user_rows = []
    for r in throughput:
        user_rows.append({
            "user_email": r["user_email"],
            "n_total": r["n_total"],
            "n_no_events": r["n_no_events"],
            "n_has_events": r["n_has_events"],
            "last_finalised_at":
                (r["last_finalised_at"] or "")[:16].replace("T",
                                                              " "),
            "animals_assigned": ", ".join(
                _assignments.animals_for_user(config or {},
                                                r["user_email"])) or "—",
        })

    return html.Div([
        html.H3("Review status", style={"color": "white",
                                          "marginTop": "0"}),
        html.Div([
            html.Span(f"📊  {len(backlog_rows)} animals with "
                       f"unreviewed files  ·  ",
                       style={"color": "#cfd0d6", "fontSize": "13px"}),
            html.Span(f"⚠ {backlog_alerts} need attention",
                       style={"color": "#FFA15A"
                              if backlog_alerts else "#888",
                              "fontSize": "13px",
                              "fontWeight": "600"}),
        ], style={"marginBottom": "16px"}),

        html.H4("Per-animal backlog", style={"color": "#cfd0d6",
                                                "marginBottom": "8px",
                                                "fontSize": "14px"}),
        dash_table.DataTable(
            data=backlog_rows,
            columns=[
                {"name": "Animal", "id": "animal"},
                {"name": "Unreviewed", "id": "n_unreviewed"},
                {"name": "Oldest", "id": "oldest"},
                {"name": "Flag", "id": "flag"},
                {"name": "Assigned to", "id": "assignees"},
            ],
            page_size=25,
            sort_action="native",
            **DARK_TABLE_STYLE,
            style_data_conditional=[
                {"if": {"filter_query": '{flag} = "🔴 crit"'},
                 "backgroundColor": "rgba(239, 85, 59, 0.08)"},
                {"if": {"filter_query": '{flag} = "🟠 warn"'},
                 "backgroundColor": "rgba(255, 161, 90, 0.08)"},
            ],
            export_format="csv",
        ),

        html.H4("Per-user throughput (last 7 days)",
                style={"color": "#cfd0d6",
                        "marginTop": "24px",
                        "marginBottom": "8px",
                        "fontSize": "14px"}),
        dash_table.DataTable(
            data=user_rows,
            columns=[
                {"name": "Reviewer", "id": "user_email"},
                {"name": "Total finalised", "id": "n_total"},
                {"name": "No events", "id": "n_no_events"},
                {"name": "With events", "id": "n_has_events"},
                {"name": "Last finished at",
                    "id": "last_finalised_at"},
                {"name": "Animals assigned",
                    "id": "animals_assigned"},
            ],
            page_size=25,
            sort_action="native",
            **DARK_TABLE_STYLE,
            export_format="csv",
        ),

        html.Div([
            html.A("Open Reviewer Assignments sheet",
                    href=("https://docs.google.com/spreadsheets/d/"
                           f"{((config or {}).get('review_queue', {}).get('assignments', {}).get('sheet_id') or '')}"
                           f"/edit"),
                    target="_blank",
                    style={"color": "#5e7ce2",
                            "fontSize": "12px",
                            "textDecoration": "none",
                            "marginTop": "20px",
                            "display": "inline-block"}),
        ]),
    ], style={"padding": "20px 24px"})
