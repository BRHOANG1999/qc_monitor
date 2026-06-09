"""Alerts tab -- read-only 7-day alert history.

Tiny tab: one table, no callbacks, no filters beyond what dash_table
gives for free. Lifted out of src/dashboard/app.py as the first
proof-of-concept for the per-tab pattern documented in tabs/video.py
and tabs/surgeries.py (``layout(store)`` plus a no-op
``register_callbacks`` so create_app can wire it the same way as the
others).

Why this tab first: it has no state besides the live ``alerts`` table
read, no Input/Output bindings, and pulls in the smallest dependency
surface (dash_table + the two shared table styles). If the move
breaks for some reason the regression is contained to one click.
"""

from __future__ import annotations

from dash import dash_table, html

from src.dashboard.components import DARK_TABLE_STYLE, ZEBRA_STRIPE
from src.db.store import Store


def layout(store: Store):
    """Build the alerts tab layout. No callbacks back into Dash --
    the table is fully rendered server-side from a single
    ``store.get_recent_alerts`` call."""
    alerts = store.get_recent_alerts(hours=168)  # 7 days
    return html.Div([
        html.H3(f"Alert History (last 7 days) -- {len(alerts)} alerts",
                style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                   "type": a["alert_type"], "message": a["message"],
                   "session": a.get("session_dir", "")}
                  for a in alerts],
            columns=[{"name": c, "id": c}
                     for c in ["time", "severity", "type",
                                "message", "session"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{severity} = critical"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{severity} = warning"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{severity} = info"},
                 "backgroundColor": "#112233", "color": "#6bb5ff"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ) if alerts else html.P("No alerts in the last 7 days",
                                  style={"color": "#888"}),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """No callbacks on this tab. Kept for symmetry with the other
    tab modules so create_app can call register_callbacks without a
    special case."""
    return
