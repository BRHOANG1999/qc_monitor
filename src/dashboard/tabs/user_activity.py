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
import os
from datetime import datetime, timedelta

from dash import (ALL, Input, Output, callback_context, dash_table, dcc,
                  html, no_update)

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
    {"label": "Auto-screen (seizure threshold)", "value": "auto-screen"},
]
_PRESENCE_MIN = 15

# Interactive plot config for the training-review LFP/Hilbert plots so the PI
# can zoom + pan them (scroll to zoom, drag to box-zoom, double-click to
# reset) -- mirrors the Video Review LFP trace config.
_INTERACTIVE_PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "scrollZoom": True,
}


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
        # Auto-filter sweep rows carry a nested per_animal breakdown; render it
        # readably (per animal: cleared✓ / flagged⚑ / error✗) instead of the
        # generic k=v truncation, so "why files remain" is visible at a glance.
        if "per_animal" in d:
            return _auto_filter_detail(d)
        return ", ".join(f"{k}={v}" for k, v in list(d.items())[:4])[:120]
    return str(d)[:120]


def _auto_filter_detail(d: dict) -> str:
    """Human summary of an auto_filter_sweep detail payload."""
    head = (f"{d.get('cleared', 0)} cleared · {d.get('flagged', 0)} flagged "
            f"({d.get('new_flags', 0)} new) · {d.get('error', 0)} error")
    per = d.get("per_animal") or {}
    parts = []
    for aid, s in list(per.items())[:8]:
        parts.append(f"{aid}: {s.get('cleared', 0)}✓/"
                     f"{s.get('flagged', 0)}⚑/{s.get('error', 0)}✗")
    tail = ("  ·  " + "  ".join(parts)) if parts else ""
    return (head + tail)[:200]


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


# --------------------------------------------------------------------- #
#  Training review: visit the LFPs a student scored, with their answer
# --------------------------------------------------------------------- #

_REVIEW_BTN_STYLE = {
    "background": "#0a84ff", "color": "white", "border": "none",
    "borderRadius": "5px", "padding": "3px 12px", "cursor": "pointer",
    "fontSize": "12px", "fontWeight": "600"}


def _pct_color(pct: float) -> str:
    return "#30d158" if pct >= 85 else "#ff9f0a" if pct >= 60 else "#ff453a"


def _train_attempt_list(store: Store, student: str | None):
    """A row per training example the student scored (newest first), each with
    a Review button that opens the LFP + their marks."""
    if not student:
        return html.Div("Pick a student to list the recordings they scored.",
                        style={"color": "#a0a0b0", "fontSize": "13px",
                               "padding": "8px 2px"})
    attempts = store.training_attempts_for_student(student, limit=200)
    if not attempts:
        return html.Div("No training attempts for this student yet.",
                        style={"color": "#a0a0b0", "fontSize": "13px",
                               "padding": "8px 2px"})
    rows = []
    for a in attempts:
        fname = (os.path.basename(a["file_path"]) if a.get("file_path")
                 else f"file #{a.get('file_id')}")
        pct = round((a.get("agreement") or 0) * 100)
        rows.append(html.Div([
            html.Span(_fmt_when(a.get("created_at")),
                      style={"width": "150px", "color": "#a0a0b0",
                             "fontSize": "12px", "flexShrink": "0"}),
            html.Span(f"Stage {a.get('stage')}",
                      style={"width": "64px", "color": "#cfd0d6",
                             "fontSize": "12px", "flexShrink": "0"}),
            html.Span(fname, title=fname,
                      style={"flex": "1", "minWidth": "0", "color": "#f0f0f5",
                             "fontSize": "12px", "overflow": "hidden",
                             "textOverflow": "ellipsis",
                             "whiteSpace": "nowrap"}),
            html.Span(f"{pct}%",
                      style={"width": "48px", "textAlign": "right",
                             "color": _pct_color(pct), "fontWeight": "700",
                             "fontSize": "13px", "flexShrink": "0"}),
            html.Button("Review",
                        id={"type": "ua-train-review-btn",
                            "attempt": int(a["attempt_id"])},
                        n_clicks=0, style=_REVIEW_BTN_STYLE),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                  "padding": "6px 8px",
                  "borderBottom": "1px solid #2a2a3a"}))
    return html.Div(rows, style={"maxHeight": "320px", "overflowY": "auto",
                                  "border": "1px solid #2a2a3a",
                                  "borderRadius": "6px"})


def _onset_seconds(events) -> list[float]:
    """EO (eye-open onset) seconds from an events list, skipping blanks."""
    out = []
    for e in (events or []):
        t = (e or {}).get("EO_sec")
        if isinstance(t, (int, float)):
            out.append(float(t))
    return out


def _overlay_onsets(fig, student_events, validated_events) -> None:
    """Mark the student's onsets (orange dashed) vs the validated onsets
    (green) on the LFP figure so the PI sees where the student was off."""
    try:
        for t in _onset_seconds(validated_events):
            fig.add_vline(x=t, line={"color": "#30d158", "width": 1.5})
        for t in _onset_seconds(student_events):
            fig.add_vline(x=t, line={"color": "#ff9f0a", "width": 1.5,
                                     "dash": "dash"})
    except Exception:  # noqa: BLE001 -- overlay is best-effort
        pass


def _train_review_panel(store: Store, attempt_id: int):
    """LFP the student saw + their marks (orange) vs validated (green) +
    the agreement breakdown, for one attempt."""
    a = store.get_training_attempt(int(attempt_id))
    if not a:
        return html.Div("Attempt not found.",
                        style={"color": "#a0a0b0", "padding": "8px"})
    # Lazy import: the training module pulls heavy LFP-rendering deps.
    from src.dashboard.tabs.training import (
        _build_figures, _first_animal_channel, _channel_for_animal,
        _events_table)
    file_id = a.get("file_id")
    session_dir = a.get("session_dir")
    file_path = a.get("file_path")
    # Same animal the student saw in Training: the seizure animal's channel
    # (not the default recording electrode), resolved with the filename
    # fallback so historical CSV examples (no session_config) work too.
    gt_animal = store.validated_seizure_animal_for_file(int(file_id))
    gt_channel = _channel_for_animal(store, session_dir, gt_animal, file_path)
    if gt_channel is not None:
        channel = gt_channel
        validated = store.validated_events_for_file(int(file_id),
                                                    animal_id=gt_animal)
    else:
        channel = _first_animal_channel(store, session_dir, file_path)
        validated = store.validated_events_for_file(int(file_id))
    lfp, hil, _dur = _build_figures(store, int(file_id), channel, "hilbert")
    try:
        answer = json.loads(a.get("submitted_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        answer = {}
    try:
        breakdown = json.loads(a.get("breakdown_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        breakdown = {}
    student_events = answer.get("events") or []
    _overlay_onsets(lfp, student_events, validated)

    pct = round((a.get("agreement") or 0) * 100)
    fname = (os.path.basename(a["file_path"]) if a.get("file_path")
             else f"file #{file_id}")
    who = f"{gt_animal} · Ch{channel}" if gt_animal else f"Ch{channel}"
    header = html.Div([
        html.Span(f"{pct}%", style={"fontSize": "24px", "fontWeight": "800",
                                     "color": _pct_color(pct)}),
        html.Span(f"  {a.get('student_email')}  ·  Stage {a.get('stage')}"
                  f"  ·  {who}  ·  {fname}",
                  style={"color": "#cfd0d6", "fontSize": "13px"}),
    ], style={"marginBottom": "6px"})
    legend = html.Div(
        "Onsets:  orange dashed = student   ·   green = validated",
        style={"color": "#a0a0b0", "fontSize": "11px",
               "marginBottom": "6px"})
    csv_info = _csv_source_block(store.external_example_for_file(int(file_id)))

    if int(a.get("stage") or 1) == 1:
        you = ("events present" if answer.get("events_present")
               else "no events")
        val = "events present" if validated else "no events"
        compare = html.Div(f"Student said: {you}   ·   Validated: {val}",
                           style={"color": "#cfd0d6", "fontSize": "12px",
                                  "marginTop": "8px"})
    else:
        b = breakdown or {}
        compare = html.Div([
            html.Div(f"matched {b.get('matched', 0)} · missed "
                     f"{b.get('missed', 0)} · extra {b.get('false_pos', 0)} "
                     f"(of {b.get('validated', 0)} validated)",
                     style={"color": "#cfd0d6", "fontSize": "12px",
                            "marginTop": "8px"}),
            html.Div([
                _events_table(student_events, "Student's answer", "#0a84ff"),
                _events_table(validated, "Validated answer", "#30d158"),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                      "marginTop": "8px"}),
        ])
    children = [header, legend]
    if csv_info is not None:
        children.append(csv_info)
    children += [
        dcc.Graph(figure=lfp, config=_INTERACTIVE_PLOT_CONFIG),
        dcc.Graph(figure=hil, config=_INTERACTIVE_PLOT_CONFIG),
        compare,
    ]
    return html.Div(children, style={
        "marginTop": "10px", "padding": "12px",
        "border": "1px solid #2a2a3a", "borderRadius": "8px"})


def _csv_source_block(ext: dict | None):
    """The CSV provenance for a historical validated example (source CSV,
    recording file/folder, animal, validated label). None for live PI-
    approved examples (no CSV behind them)."""
    if not ext:
        return None
    rep = ext.get("rep_type") or "?"
    rac = ext.get("rep_racine")
    fields = [
        ("Source CSV", ext.get("source_csv")),
        ("Recording", ext.get("filename")),
        ("Folder", ext.get("folder")),
        ("Animal", ext.get("animal")),
        ("Validated", f"{rep} · Racine {rac}" if rac is not None else None),
        ("Sampling", f"{ext.get('fs'):.0f} Hz" if ext.get("fs") else None),
    ]
    rows = [
        html.Div([
            html.Span(f"{k}: ", style={"color": "#8a8a9a"}),
            html.Span(str(v), style={"color": "#d8d8e0",
                                     "wordBreak": "break-all"}),
        ], style={"fontSize": "11px", "marginBottom": "2px"})
        for k, v in fields if v
    ]
    return html.Div([
        html.Div("CSV source (historical validated example)",
                 style={"color": "#a0a0b0", "fontSize": "11px",
                        "fontWeight": "600", "marginBottom": "4px"}),
        *rows,
    ], style={"marginBottom": "8px", "padding": "8px 10px",
              "background": "#1b1b27", "border": "1px solid #2a2a3a",
              "borderRadius": "6px"})


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

        html.H4("Training review — visit what a student scored",
                style={"color": "#cfd0d6", "fontSize": "14px",
                       "marginTop": "26px", "marginBottom": "8px"}),
        html.Div("Open any recording an undergrad scored in Training, with "
                 "their onsets/marks overlaid on the LFP against the "
                 "validated answer.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "10px"}),
        html.Div([
            html.Label("Student", style={"color": "#a0a0b0",
                                         "fontSize": "12px"}),
            dcc.Dropdown(id="ua-train-student", options=[],
                         placeholder="Pick a student", style=dd,
                         className="dark-dropdown"),
        ], style={"marginBottom": "10px"}),
        html.Div(id="ua-train-attempts"),
        dcc.Loading(html.Div(id="ua-train-review"), type="default"),
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

    # ---- Training review ---- #
    @app.callback(
        Output("ua-train-student", "options"),
        Input("user-activity-tick", "n_intervals"),
        Input("refresh-trigger", "data"),
    )
    def _train_students(_t, _r):
        email = (current_user_email() or "").lower()
        if not _can_view(store, config, email):
            return []
        return [{"label": s, "value": s}
                for s in store.distinct_training_students()]

    @app.callback(
        Output("ua-train-attempts", "children"),
        Output("ua-train-review", "children"),
        Input("ua-train-student", "value"),
    )
    def _train_attempts(student):
        email = (current_user_email() or "").lower()
        if not _can_view(store, config, email):
            return no_update, no_update
        # New student -> fresh list, clear any open review panel.
        return _train_attempt_list(store, student), []

    @app.callback(
        Output("ua-train-review", "children", allow_duplicate=True),
        Input({"type": "ua-train-review-btn", "attempt": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def _train_review(_clicks):
        email = (current_user_email() or "").lower()
        if not _can_view(store, config, email):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict) or trig.get("attempt") is None:
            return no_update
        # Ignore the render-time n_clicks=0 firings; only act on a real click.
        if not any(callback_context.triggered and t["value"]
                   for t in callback_context.triggered):
            return no_update
        try:
            return _train_review_panel(store, int(trig["attempt"]))
        except Exception as e:  # noqa: BLE001 -- surface, never crash the tab
            logger.warning("training review panel failed: %s", e)
            return html.Div(f"Couldn't load this recording: {e}",
                            style={"color": "#ff9f0a", "padding": "8px"})
