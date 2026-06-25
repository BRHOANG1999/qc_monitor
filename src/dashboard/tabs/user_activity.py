"""User Activity tab (PI / mentor only).

Two views over a unified per-user feed (``store.unified_user_feed`` =
``user_activity`` ∪ ``review_event_log`` ∪ ``training_attempt``):

* **Active now** — who's been seen in the last few minutes + their latest
  action (presence reuses ``users.last_seen_at``).
* **Activity feed** — a filterable who-did-what-when table (by user / area /
  time range), exportable to CSV.

Gated to PIs (``config.review_queue.pi_emails``) and mentors
(``users.role == 'mentor'``) since it surfaces every user's behaviour.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from dash import Input, Output, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.auth import current_user_email
from src.dashboard.components import DARK_TABLE_STYLE, ZEBRA_STRIPE
from src.dashboard.tabs.review_status import _is_pi

logger = logging.getLogger("qc_monitor.dashboard.user_activity")

_HOURS_OPTIONS = [
    {"label": "Last 1 h", "value": 1},
    {"label": "Last 24 h", "value": 24},
    {"label": "Last 7 d", "value": 168},
    {"label": "All", "value": 0},
]
_AREA_OPTIONS = [
    {"label": "All areas", "value": ""},
    {"label": "Navigation", "value": "nav"},
    {"label": "Video review", "value": "review"},
    {"label": "Video analysis", "value": "video"},
    {"label": "Training", "value": "training"},
    {"label": "Chronic Evoked", "value": "chronic_evoked"},
    {"label": "Evoked Waveforms", "value": "waveforms"},
    {"label": "LFP Browser", "value": "lfp_browser"},
    {"label": "Event Verification", "value": "event_verification"},
    {"label": "File browser", "value": "file_browser"},
]
_PRESENCE_MIN = 15


def _can_view(store: Store, config: dict | None, email: str | None) -> bool:
    if not email:
        return False
    if _is_pi(config or {}, email):
        return True
    try:
        return store.get_user_role(email) == "mentor"
    except Exception:  # noqa: BLE001
        return False


def _fmt_when(iso: str | None) -> str:
    return (iso or "")[:19].replace("T", " ")


def _ago(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return iso[:16]
    secs = (datetime.now() - dt).total_seconds()
    if secs < 90:
        return f"{secs:.0f} s ago"
    if secs < 5400:
        return f"{secs / 60:.0f} m ago"
    return f"{secs / 3600:.1f} h ago"


def _doing(row: dict) -> str:
    parts = [row.get("last_area") or "", row.get("last_action") or ""]
    label = " · ".join(p for p in parts if p) or "—"
    tgt = row.get("last_target")
    return f"{label}  ({tgt})" if tgt else label


def _detail_str(detail) -> str:
    if not detail:
        return ""
    try:
        d = json.loads(detail) if isinstance(detail, str) else detail
    except (json.JSONDecodeError, TypeError):
        return str(detail)[:80]
    if isinstance(d, dict):
        return ", ".join(f"{k}={v}" for k, v in list(d.items())[:4])[:120]
    return str(d)[:120]


def _active_rows(store: Store) -> list[dict]:
    out = []
    for u in store.active_users(window_min=_PRESENCE_MIN):
        out.append({
            "user": u["email"],
            "role": u.get("role") or "reviewer",
            "doing": _doing(u),
            "last_seen": _ago(u.get("last_seen_at")),
        })
    return out


def _feed_rows(store: Store, user: str | None, area: str | None,
               hours: int) -> list[dict]:
    since = None
    if hours:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
    rows = store.unified_user_feed(
        user_email=(user or None), area=(area or None),
        since_iso=since, limit=1000)
    return [{
        "when": _fmt_when(r.get("at")),
        "user": r.get("user_email"),
        "area": r.get("area"),
        "action": r.get("action"),
        "target": r.get("target") or "",
        "detail": _detail_str(r.get("detail")),
    } for r in rows]


def layout(store: Store, config: dict | None = None):
    email = current_user_email()
    if not _can_view(store, config, email):
        return html.Div([
            html.Div("🔒  Not authorised",
                     style={"fontSize": "18px", "fontWeight": "600",
                            "color": "#f0f0f5", "marginBottom": "8px"}),
            html.Div("This tab is for PIs / mentors only.",
                     style={"color": "#a0a0b0", "fontSize": "13px"}),
        ], style={"padding": "40px 24px"})

    dd = {"width": "220px"}
    return html.Div([
        html.H3("User activity", style={"color": "white", "marginTop": "0"}),
        dcc.Interval(id="user-activity-tick", interval=15000),

        html.H4(f"Active now (last {_PRESENCE_MIN} min)",
                style={"color": "#cfd0d6", "fontSize": "14px",
                       "marginBottom": "8px"}),
        dash_table.DataTable(
            id="user-activity-active", data=_active_rows(store),
            columns=[{"name": "User", "id": "user"},
                     {"name": "Role", "id": "role"},
                     {"name": "Doing", "id": "doing"},
                     {"name": "Last seen", "id": "last_seen"}],
            page_size=12, **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE]),

        html.H4("Activity feed", style={"color": "#cfd0d6",
                                        "fontSize": "14px",
                                        "marginTop": "22px",
                                        "marginBottom": "8px"}),
        html.Div([
            html.Div([
                html.Label("User", style={"color": "#a0a0b0",
                                          "fontSize": "12px"}),
                dcc.Dropdown(id="user-activity-user", options=[],
                             placeholder="All users", style=dd,
                             className="dark-dropdown"),
            ]),
            html.Div([
                html.Label("Area", style={"color": "#a0a0b0",
                                          "fontSize": "12px"}),
                dcc.Dropdown(id="user-activity-area", options=_AREA_OPTIONS,
                             value="", clearable=False, style=dd,
                             className="dark-dropdown"),
            ]),
            html.Div([
                html.Label("Range", style={"color": "#a0a0b0",
                                           "fontSize": "12px"}),
                dcc.Dropdown(id="user-activity-hours", options=_HOURS_OPTIONS,
                             value=24, clearable=False,
                             style={"width": "140px"},
                             className="dark-dropdown"),
            ]),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                  "marginBottom": "10px"}),
        dash_table.DataTable(
            id="user-activity-feed",
            data=_feed_rows(store, None, "", 24),
            columns=[{"name": "When", "id": "when"},
                     {"name": "User", "id": "user"},
                     {"name": "Area", "id": "area"},
                     {"name": "Action", "id": "action"},
                     {"name": "Target", "id": "target"},
                     {"name": "Detail", "id": "detail"}],
            page_size=25, sort_action="native", filter_action="native",
            export_format="csv", **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE]),
    ], style={"padding": "20px 24px"})


def register_callbacks(app, store: Store, config: dict) -> None:
    """Refresh the active-now table (interval + global refresh) and the feed
    (filters + global refresh). Both PI/mentor-gated."""

    @app.callback(
        Output("user-activity-active", "data"),
        Output("user-activity-user", "options"),
        Input("user-activity-tick", "n_intervals"),
        Input("refresh-trigger", "data"),
    )
    def _refresh_active(_t, _r):
        email = (current_user_email() or "").lower()
        if not _can_view(store, config, email):
            return [], []
        opts = [{"label": u, "value": u}
                for u in store.distinct_activity_users()]
        return _active_rows(store), opts

    @app.callback(
        Output("user-activity-feed", "data"),
        Input("user-activity-user", "value"),
        Input("user-activity-area", "value"),
        Input("user-activity-hours", "value"),
        Input("refresh-trigger", "data"),
    )
    def _refresh_feed(user, area, hours, _r):
        email = (current_user_email() or "").lower()
        if not _can_view(store, config, email):
            return []
        return _feed_rows(store, user, area, int(hours or 0))
