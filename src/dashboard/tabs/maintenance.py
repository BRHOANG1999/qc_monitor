"""Maintenance tab -- rig hygiene at a glance.

Reads each rig x task tab from the Maintenance Tracker Google Sheet,
parses the latest Timestamp per tab, and reports OK / overdue against
a configurable cadence (weekly cage cleaning, Mon/Wed/Fri battery
swap, etc.). Same auth + TTL cache as the Surgeries tab.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd
from dash import Input, Output, callback_context, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path, _last_fetched,
    _find_column, _safe_date, _normalize_text,
)

logger = logging.getLogger("qc_monitor.dashboard.maintenance")

# Day-of-week constants (pd / datetime: Monday=0..Sunday=6)
_MWF_DAYS = (0, 2, 4)  # Mon, Wed, Fri

# Cadence strings supported by is_overdue / next_due
_CADENCE_WEEKLY = "weekly"
_CADENCE_MWF = "mwf"
_CADENCE_DAILY = "daily"
_CADENCE_EVERY_PREFIX = "every:"


# ===================================================================== #
#  Cadence math
# ===================================================================== #

def _most_recent_mwf_midnight(now: datetime) -> datetime:
    """Return midnight of the most recent Mon/Wed/Fri at or before *now*."""
    assert isinstance(now, datetime), "now must be a datetime"
    d = now.date()
    for back in range(0, 7):
        candidate = d - timedelta(days=back)
        if candidate.weekday() in _MWF_DAYS:
            return datetime.combine(candidate, time.min)
    # Unreachable -- MWF appears 3x per week, 7 days covers it.
    return datetime.combine(d, time.min)


def is_overdue(last: datetime | None, cadence: str, now: datetime,
               grace_hours: float) -> tuple[bool, datetime | None, str]:
    """Compute whether a task is overdue.

    Returns (overdue, last_due_at, reason).
      - overdue: True when *last* is too far behind the schedule.
      - last_due_at: the most recent scheduled occurrence at/before now
        (None for `weekly` since there's no fixed anchor day).
      - reason: short human label ("never", "ok", "due today",
        "overdue").
    """
    assert isinstance(now, datetime), "now must be a datetime"
    assert isinstance(grace_hours, (int, float)), "grace_hours numeric"

    if last is None:
        return True, None, "never"

    grace = timedelta(hours=float(grace_hours))
    cadence = (cadence or "").strip().lower()

    if cadence == _CADENCE_WEEKLY:
        gap = now - last
        if gap <= timedelta(days=7) + grace:
            return False, None, "ok"
        return True, None, "overdue"

    if cadence == _CADENCE_MWF:
        due = _most_recent_mwf_midnight(now)
        if last >= due - grace:
            # Already done on/after the latest scheduled day.
            return False, due, "ok"
        # Last is before the most recent scheduled day -> overdue.
        return True, due, "overdue"

    if cadence == _CADENCE_DAILY:
        if (now - last) <= timedelta(hours=24) + grace:
            return False, None, "ok"
        return True, None, "overdue"

    if cadence.startswith(_CADENCE_EVERY_PREFIX):
        body = cadence[len(_CADENCE_EVERY_PREFIX):].rstrip("d")
        try:
            n = float(body)
        except ValueError:
            n = 7.0
        if (now - last) <= timedelta(days=n) + grace:
            return False, None, "ok"
        return True, None, "overdue"

    # Unknown cadence -> assume weekly to fail safe.
    logger.warning("Unknown maintenance cadence %r; treating as weekly", cadence)
    if (now - last) <= timedelta(days=7) + grace:
        return False, None, "ok"
    return True, None, "overdue"


# ===================================================================== #
#  Data path
# ===================================================================== #

@dataclass
class RigTaskStatus:
    rig: str
    task: str
    tab_name: str
    cadence: str
    last_ts: datetime | None
    last_by: str
    last_notes: str
    overdue: bool
    status: str          # "never" / "ok" / "overdue"
    days_since: float | None


def _last_event(df: pd.DataFrame) -> tuple[datetime | None, str, str]:
    """Return (timestamp, initials, notes) of the latest row in *df*."""
    if df is None or df.empty:
        return None, "", ""
    ts_col = _find_column(df, ["Timestamp", "Time", "Date"])
    if ts_col is None:
        return None, "", ""
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    if ts.dropna().empty:
        return None, "", ""
    idx = ts.idxmax()
    row = df.loc[idx]
    last_dt = ts.loc[idx]
    initials_col = _find_column(df, ["Initials", "Operator"])
    notes_col = _find_column(df, ["Notes/Comments", "Notes", "Comments"])
    initials = _normalize_text(row.get(initials_col)) if initials_col else ""
    notes = _normalize_text(row.get(notes_col)) if notes_col else ""
    return last_dt.to_pydatetime(), initials, notes


def _maintenance_cfg(config: dict) -> dict:
    return (config or {}).get("maintenance", {}) or {}


def rig_status(config: dict, now: datetime | None = None,
               ttl_sec: float | None = None) -> list[RigTaskStatus]:
    """Compute current status for every configured (rig, task) pair."""
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    now = now or datetime.now()
    sheet_id = cfg.get("sheet_id", "")
    sa_file = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    if not sheet_id or not sa_file:
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0
    grace_default = float(cfg.get("default_grace_hours", 24))

    out: list[RigTaskStatus] = []
    for rig in cfg.get("rigs", []) or []:
        rig_name = str(rig.get("name", "?"))
        for task_key, task_cfg in (rig.get("tasks") or {}).items():
            tab_name = task_cfg.get("tab_name") or ""
            cadence = task_cfg.get("cadence", _CADENCE_WEEKLY)
            grace_h = float(task_cfg.get("grace_hours", grace_default))
            if not tab_name:
                continue
            df = _load_sheet_via_api(sheet_id, tab_name, sa_file, ttl_sec)
            last_ts, by, notes = _last_event(df)
            overdue, _due, status = is_overdue(last_ts, cadence, now, grace_h)
            days_since = (
                (now - last_ts).total_seconds() / 86400.0
                if last_ts is not None else None
            )
            out.append(RigTaskStatus(
                rig=rig_name, task=task_key, tab_name=tab_name,
                cadence=cadence, last_ts=last_ts, last_by=by,
                last_notes=notes, overdue=overdue, status=status,
                days_since=days_since,
            ))
    return out


def recent_events(config: dict, limit: int = 5,
                  ttl_sec: float | None = None) -> list[dict]:
    """Most-recent events merged across every configured (rig, task)."""
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    sheet_id = cfg.get("sheet_id", "")
    sa_file = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    if not sheet_id or not sa_file:
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0

    all_events: list[dict] = []
    for rig in cfg.get("rigs", []) or []:
        rig_name = str(rig.get("name", "?"))
        for task_key, task_cfg in (rig.get("tasks") or {}).items():
            tab_name = task_cfg.get("tab_name") or ""
            if not tab_name:
                continue
            df = _load_sheet_via_api(sheet_id, tab_name, sa_file, ttl_sec)
            if df is None or df.empty:
                continue
            ts_col = _find_column(df, ["Timestamp", "Time", "Date"])
            initials_col = _find_column(df, ["Initials", "Operator"])
            notes_col = _find_column(df, ["Notes/Comments", "Notes",
                                          "Comments"])
            if ts_col is None:
                continue
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            for idx in ts.dropna().sort_values(ascending=False).index[:limit]:
                row = df.loc[idx]
                all_events.append({
                    "when": ts.loc[idx].to_pydatetime(),
                    "rig": rig_name,
                    "task": task_key,
                    "by": (_normalize_text(row.get(initials_col))
                           if initials_col else ""),
                    "notes": (_normalize_text(row.get(notes_col))
                              if notes_col else ""),
                })
    all_events.sort(key=lambda x: x["when"], reverse=True)
    return all_events[:limit]


# ===================================================================== #
#  Layout
# ===================================================================== #

_STATUS_COLOR = {
    "ok": "#30d158",
    "overdue": "#ff453a",
    "never": "#a0a0b0",
}

_TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": "#1a1a2e", "color": "#e0e0ea",
        "fontWeight": "600",
        "border": "1px solid rgba(255,255,255,0.07)",
    },
    "style_cell": {
        "backgroundColor": "#13131f", "color": "#f0f0f5",
        "padding": "8px", "fontSize": "13px",
        "border": "1px solid rgba(255,255,255,0.05)",
        "textAlign": "left",
    },
}


def _task_label(task_key: str) -> str:
    return task_key.replace("_", " ").capitalize()


def _pill(status: str) -> html.Span:
    color = _STATUS_COLOR.get(status, "#a0a0b0")
    label = status.upper()
    return html.Span(
        label,
        style={
            "backgroundColor": color, "color": "white",
            "padding": "2px 10px", "borderRadius": "12px",
            "fontSize": "11px", "fontWeight": "600",
            "letterSpacing": "0.5px",
        },
    )


def _format_last(last_ts: datetime | None, days_since: float | None) -> str:
    if last_ts is None:
        return "never"
    if days_since is not None and days_since < 1.0:
        return f"{last_ts:%Y-%m-%d %H:%M} (today)"
    if days_since is not None:
        return f"{last_ts:%Y-%m-%d} ({days_since:.1f}d ago)"
    return f"{last_ts:%Y-%m-%d %H:%M}"


def _status_grid(rows: list[RigTaskStatus]) -> html.Div:
    if not rows:
        return html.Div("No rigs configured.",
                        style={"color": "#a0a0b0", "padding": "12px"})
    children = []
    # One styled row per (rig, task)
    for r in rows:
        children.append(html.Div([
            html.Div(f"Rig {r.rig}",
                     style={"flex": "0 0 80px", "fontWeight": "600",
                            "color": "#f0f0f5"}),
            html.Div(_task_label(r.task),
                     style={"flex": "0 0 200px", "color": "#d0d0da"}),
            html.Div(_format_last(r.last_ts, r.days_since),
                     style={"flex": "1", "color": "#a0a0b0",
                            "fontSize": "12px"}),
            html.Div(r.last_by or "-",
                     style={"flex": "0 0 80px", "color": "#a0a0b0",
                            "fontSize": "12px"}),
            _pill(r.status),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "12px",
            "padding": "10px 12px",
            "borderBottom": "1px solid rgba(255,255,255,0.05)",
            "backgroundColor": "#13131f",
        }))
    return html.Div(children, style={
        "borderRadius": "8px", "overflow": "hidden",
        "border": "1px solid rgba(255,255,255,0.08)",
    })


def _recent_table(events: list[dict]) -> html.Div:
    if not events:
        return html.Div(
            "No recent events.",
            style={"color": "#6c6c80", "fontSize": "13px",
                   "padding": "12px 0", "fontStyle": "italic"},
        )
    rows = [{
        "when": e["when"].strftime("%Y-%m-%d %H:%M"),
        "rig": f"Rig {e['rig']}",
        "task": _task_label(e["task"]),
        "by": e["by"] or "-",
        "notes": (e["notes"][:60] + "...") if len(e["notes"]) > 60
                 else e["notes"],
    } for e in events]
    columns = [
        {"name": "When", "id": "when"},
        {"name": "Rig", "id": "rig"},
        {"name": "Task", "id": "task"},
        {"name": "By", "id": "by"},
        {"name": "Notes", "id": "notes"},
    ]
    return html.Div([dash_table.DataTable(
        data=rows, columns=columns, page_size=10, **_TABLE_STYLE,
    )])


def _footer(rows: list[RigTaskStatus], sheet_id: str) -> html.Div:
    """Last-fetched timestamps per tab."""
    if not rows:
        return html.Div()
    lines = []
    for r in rows:
        key = f"api:{sheet_id}:{r.tab_name}:h1"
        ts = _last_fetched(key)
        when = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                if ts else "never")
        lines.append(html.Div(
            f"Rig {r.rig} {_task_label(r.task)} -- fetched {when}",
            style={"color": "#6c6c80", "fontSize": "11px",
                   "marginBottom": "2px"},
        ))
    return html.Div(lines, style={"marginTop": "16px"})


def _render(rows: list[RigTaskStatus], events: list[dict],
            sheet_id: str, today: date) -> html.Div:
    return html.Div([
        html.H3(f"Maintenance -- {today.isoformat()}",
                style={"color": "#f0f0f5", "marginBottom": "12px"}),
        html.H4("Status", style={
            "color": "#a0a0b0", "marginTop": "8px", "marginBottom": "8px",
            "fontSize": "13px", "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        }),
        _status_grid(rows),
        html.H4("Recent activity", style={
            "color": "#a0a0b0", "marginTop": "24px", "marginBottom": "8px",
            "fontSize": "13px", "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        }),
        _recent_table(events),
        _footer(rows, sheet_id),
    ])


# ===================================================================== #
#  Public layout + callbacks
# ===================================================================== #

def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = _maintenance_cfg(config or {})
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Maintenance", style={"color": "#f0f0f5"}),
            html.P(
                "Maintenance tracking is disabled. Enable it in "
                "config.yaml -> maintenance.enabled and configure the "
                "Sheet ID + rigs.",
                style={"color": "#a0a0b0"},
            ),
        ], style={"padding": "24px"})

    refresh_min = float(cfg.get("refresh_minutes", 10))
    return html.Div([
        html.Div([
            html.Button(
                "Refresh", id="maintenance-refresh-btn", n_clicks=0,
                style={
                    "backgroundColor": "#262638", "color": "#f0f0f5",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "padding": "8px 14px", "borderRadius": "6px",
                    "cursor": "pointer", "fontSize": "13px",
                },
            ),
        ], style={"display": "flex", "justifyContent": "flex-end",
                  "marginBottom": "12px"}),
        dcc.Interval(id="maintenance-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="maintenance-pane"),
    ], style={"padding": "24px"})


def register_callbacks(app, store: Store, config: dict | None = None) -> None:
    cfg = _maintenance_cfg(config or {})
    if not cfg.get("enabled", False):
        return

    default_ttl = float(cfg.get("refresh_minutes", 10)) * 60.0

    @app.callback(
        Output("maintenance-pane", "children"),
        Input("maintenance-refresh-btn", "n_clicks"),
        Input("maintenance-tick", "n_intervals"),
    )
    def _render_cb(_n_clicks, _n_intervals):
        triggered = (callback_context.triggered_id
                     if callback_context.triggered else None)
        ttl = 0.0 if triggered == "maintenance-refresh-btn" else default_ttl
        now = datetime.now()
        rows = rig_status(config, now=now, ttl_sec=ttl)
        events = recent_events(config, limit=10, ttl_sec=ttl)
        sheet_id = cfg.get("sheet_id", "")
        return _render(rows, events, sheet_id, now.date())
