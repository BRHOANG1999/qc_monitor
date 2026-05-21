"""Maintenance tab -- mirrors the lab's Apps Script semantics.

The Apps Script in the Maintenance Tracker spreadsheet is the source of
truth: it owns the per-day schedule, the per-shelf "active" filter,
and the operational reminder + escalation pipeline. This tab is a
read-only view: it answers "today, what's scheduled and who's done
with it?" using the same Schedule + Active Shelves tabs the GAS reads,
plus the per-rig event log tabs.

What you'll see for a (rig, task) cell:
  * NOT ACTIVE       -- rig has no active shelves in `Active Shelves`.
  * NOT SCHEDULED    -- today's weekday isn't in `Schedule` for this task.
  * DONE             -- scheduled today, every active shelf swapped today
                        (battery) or any entry recorded today (cage).
  * PENDING          -- scheduled today, not yet done, before the
                        escalation hour (3:30 PM battery / 5 PM cage).
  * OVERDUE          -- scheduled, not done, past the escalation hour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import pandas as pd
from dash import Input, Output, callback_context, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.components import (
    TABLE_STYLE, empty_state, pill, refresh_bar,
)
from src.dashboard.design import (
    COLOR_DIVIDER, COLOR_SURFACE_1, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY, FONT_SIZE_BODY,
    FONT_SIZE_CAPTION, RADIUS_MD, SPACE_2, SPACE_3, SPACE_4, SPACE_5,
)
from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path, _last_fetched,
    _find_column, _normalize_text,
)

logger = logging.getLogger("qc_monitor.dashboard.maintenance")

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]  # datetime.weekday() order

# Default escalation hours match the Apps Script CONFIG.
_DEFAULT_ESC = {"battery": 15.5, "cage": 17.0}
_RIG_RE = re.compile(r"Rig\s*([A-Za-z0-9]+)", re.IGNORECASE)


# ===================================================================== #
#  Status data class
# ===================================================================== #

@dataclass
class RigTaskStatus:
    rig: str
    task: str                 # "battery" or "cage"
    tab_name: str
    scheduled_today: bool
    is_active: bool           # rig has at least one active shelf
    done_today: bool
    status: str               # "not_active" | "not_scheduled" | "done"
                              # | "pending" | "overdue"
    pending_shelves: list[int] = field(default_factory=list)  # battery only
    done_shelves: list[int] = field(default_factory=list)      # battery only
    last_ts: datetime | None = None
    last_by: str = ""
    last_notes: str = ""
    assignees: list[tuple[str, str]] = field(default_factory=list)


def _task_pretty(task: str) -> str:
    return {"battery": "Battery", "cage": "Cage cleaning"}.get(task, task)


# ===================================================================== #
#  Config helpers
# ===================================================================== #

def _maintenance_cfg(config: dict) -> dict:
    return (config or {}).get("maintenance", {}) or {}


def _rig_letter(label: str) -> str | None:
    """Pull the rig letter out of "🐜 Rig A" / "Rig A" / "Rig-A"."""
    if not label:
        return None
    m = _RIG_RE.search(str(label))
    return m.group(1).upper() if m else None


def _is_truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "yes", "y", "1", "x")


# ===================================================================== #
#  Schedule + Active Shelves (data-driven)
# ===================================================================== #

def _load_schedule(config: dict, ttl_sec: float
                   ) -> dict[str, list[tuple[int, str, str]]]:
    """Return {task_label: [(weekday_int, name, email), ...]}.

    Reads the '📅 Schedule' tab (or whichever name is configured).
    Task labels and weekday names match the Apps Script convention
    ('Battery Change', 'Cage Cleaning', 'Monday'..'Sunday').
    """
    cfg = _maintenance_cfg(config)
    sa = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    sheet_id = cfg.get("sheet_id", "")
    tab = cfg.get("schedule_tab_name", "📅 Schedule")
    if not (sheet_id and sa and tab):
        return {}
    df = _load_sheet_via_api(sheet_id, tab, sa, ttl_sec)
    if df is None or df.empty:
        return {}

    col_task = _find_column(df, ["Task"])
    col_day = _find_column(df, ["Day"])
    col_name = _find_column(df, ["Assigned Name", "Name"])
    col_email = _find_column(df, ["Assigned Email", "Email"])
    if col_task is None or col_day is None:
        return {}

    name_to_wd = {n: i for i, n in enumerate(_DAY_NAMES)}
    out: dict[str, list[tuple[int, str, str]]] = {}
    for _i, row in df.iterrows():
        task = _normalize_text(row.get(col_task))
        day = _normalize_text(row.get(col_day))
        wd = name_to_wd.get(day)
        if not task or wd is None:
            continue
        name = _normalize_text(row.get(col_name)) if col_name else ""
        email = _normalize_text(row.get(col_email)) if col_email else ""
        out.setdefault(task, []).append((wd, name, email))
    return out


def _load_active_shelves(config: dict, ttl_sec: float
                         ) -> dict[str, dict[int, bool]]:
    """Return {'A': {1: True, 2: False, ...}, ...}.

    Rigs missing from the 'Active Shelves' tab default to all-False;
    that's how the GAS treats them too.
    """
    cfg = _maintenance_cfg(config)
    sa = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    sheet_id = cfg.get("sheet_id", "")
    tab = cfg.get("active_shelves_tab_name", "Active Shelves")
    if not (sheet_id and sa and tab):
        return {}
    df = _load_sheet_via_api(sheet_id, tab, sa, ttl_sec)
    if df is None or df.empty:
        return {}

    col_rig = _find_column(df, ["Rig"])
    if col_rig is None:
        return {}

    shelf_cols = {}
    for n in range(1, 9):
        c = _find_column(df, [f"Shelf {n}", f"shelf{n}", f"S{n}"])
        if c is not None:
            shelf_cols[n] = c

    out: dict[str, dict[int, bool]] = {}
    for _i, row in df.iterrows():
        letter = _rig_letter(row.get(col_rig))
        if not letter:
            continue
        out[letter] = {n: _is_truthy(row.get(c)) for n, c in shelf_cols.items()}
    return out


# ===================================================================== #
#  Event tabs (battery / cage)
# ===================================================================== #

def _today_rows(df: pd.DataFrame, today: date) -> pd.DataFrame | None:
    """Return rows of *df* whose Timestamp matches *today*."""
    if df is None or df.empty:
        return None
    ts_col = _find_column(df, ["Timestamp", "Time", "Date"])
    if ts_col is None:
        return None
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    mask = ts.dt.date == today
    if not mask.any():
        return df.iloc[0:0]
    return df.loc[mask]


def _battery_done_shelves(df: pd.DataFrame, today: date,
                          active_shelves: dict[int, bool]
                          ) -> tuple[list[int], list[int]]:
    """Return (done_shelves, pending_shelves) for the active set."""
    todays = _today_rows(df, today)
    done: set[int] = set()
    if todays is not None and not todays.empty:
        for n in range(1, 9):
            out_col = _find_column(df, [f"Shelf {n} - OUT",
                                         f"Shelf {n} OUT",
                                         f"Shelf {n}OUT"])
            in_col = _find_column(df, [f"Shelf {n} - IN",
                                        f"Shelf {n} IN",
                                        f"Shelf {n}IN"])
            for _i, row in todays.iterrows():
                if (out_col and _normalize_text(row.get(out_col))) or \
                   (in_col and _normalize_text(row.get(in_col))):
                    done.add(n)
                    break
    active = [n for n, on in active_shelves.items() if on]
    return (sorted(s for s in active if s in done),
            sorted(s for s in active if s not in done))


def _cage_done_today(df: pd.DataFrame, today: date) -> bool:
    todays = _today_rows(df, today)
    return todays is not None and not todays.empty


def _last_event(df: pd.DataFrame) -> tuple[datetime | None, str, str]:
    """Return (latest_ts, initials, notes) -- used for the recent feed."""
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
    initials_col = _find_column(df, ["Initials", "Operator"])
    notes_col = _find_column(df, ["Notes/Comments", "Notes", "Comments"])
    return (ts.loc[idx].to_pydatetime(),
            _normalize_text(row.get(initials_col)) if initials_col else "",
            _normalize_text(row.get(notes_col)) if notes_col else "")


# ===================================================================== #
#  Status computation
# ===================================================================== #

def _resolve_status(scheduled: bool, active: bool, done: bool,
                    now: datetime, esc_hour: float) -> str:
    if not active:
        return "not_active"
    if not scheduled:
        return "not_scheduled"
    if done:
        return "done"
    current = now.hour + now.minute / 60.0
    return "overdue" if current >= esc_hour else "pending"


def rig_status(config: dict, now: datetime | None = None,
               ttl_sec: float | None = None) -> list[RigTaskStatus]:
    """One RigTaskStatus per (rig, task) pair configured under
    `maintenance.rigs`."""
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    now = now or datetime.now()
    today = now.date()
    today_wd = now.weekday()

    sa = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    sheet_id = cfg.get("sheet_id", "")
    if not (sa and sheet_id):
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0

    battery_label = cfg.get("task_label_battery", "Battery Change")
    cage_label = cfg.get("task_label_cage", "Cage Cleaning")
    esc = cfg.get("escalation_hours", {}) or {}
    esc_battery = float(esc.get("battery", _DEFAULT_ESC["battery"]))
    esc_cage = float(esc.get("cage", _DEFAULT_ESC["cage"]))

    schedule = _load_schedule(config, ttl_sec)
    active_map = _load_active_shelves(config, ttl_sec)

    battery_today_assignees = [
        (n, e) for wd, n, e in schedule.get(battery_label, []) if wd == today_wd
    ]
    cage_today_assignees = [
        (n, e) for wd, n, e in schedule.get(cage_label, []) if wd == today_wd
    ]
    battery_scheduled = bool(battery_today_assignees) or any(
        wd == today_wd for wd, _, _ in schedule.get(battery_label, []))
    cage_scheduled = bool(cage_today_assignees) or any(
        wd == today_wd for wd, _, _ in schedule.get(cage_label, []))

    out: list[RigTaskStatus] = []
    for rig in cfg.get("rigs", []) or []:
        letter = str(rig.get("name", "")).upper()
        shelves = active_map.get(letter, {})
        any_active = any(shelves.values()) if shelves else False

        # ---- battery ----
        battery_tab = rig.get("battery_tab", "")
        if battery_tab:
            df = _load_sheet_via_api(sheet_id, battery_tab, sa, ttl_sec)
            done_shelves, pending = _battery_done_shelves(df, today, shelves)
            done = any_active and not pending
            status = _resolve_status(battery_scheduled, any_active, done,
                                     now, esc_battery)
            last_ts, by, notes = _last_event(df)
            out.append(RigTaskStatus(
                rig=letter, task="battery", tab_name=battery_tab,
                scheduled_today=battery_scheduled,
                is_active=any_active, done_today=done,
                status=status, done_shelves=done_shelves,
                pending_shelves=pending,
                last_ts=last_ts, last_by=by, last_notes=notes,
                assignees=list(battery_today_assignees),
            ))

        # ---- cage ----
        cage_tab = rig.get("cage_tab", "")
        if cage_tab:
            df = _load_sheet_via_api(sheet_id, cage_tab, sa, ttl_sec)
            done = any_active and _cage_done_today(df, today)
            status = _resolve_status(cage_scheduled, any_active, done,
                                     now, esc_cage)
            last_ts, by, notes = _last_event(df)
            out.append(RigTaskStatus(
                rig=letter, task="cage", tab_name=cage_tab,
                scheduled_today=cage_scheduled,
                is_active=any_active, done_today=done,
                status=status,
                last_ts=last_ts, last_by=by, last_notes=notes,
                assignees=list(cage_today_assignees),
            ))
    return out


def todays_schedule(config: dict, now: datetime | None = None,
                    ttl_sec: float | None = None) -> list[dict]:
    """Today's scheduled (task, assignee) pairs.

    Returns [] when nothing is scheduled. Used by the home page block.
    """
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    now = now or datetime.now()
    wd = now.weekday()
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0
    schedule = _load_schedule(config, ttl_sec)
    out: list[dict] = []
    for task_label, entries in schedule.items():
        for w, name, email in entries:
            if w == wd:
                out.append({"task": task_label, "name": name, "email": email})
    return out


def recent_events(config: dict, limit: int = 10,
                  ttl_sec: float | None = None) -> list[dict]:
    """Recent maintenance events merged across every rig x task tab."""
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    sa = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    sheet_id = cfg.get("sheet_id", "")
    if not (sa and sheet_id):
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0

    all_events: list[dict] = []
    for rig in cfg.get("rigs", []) or []:
        letter = str(rig.get("name", "")).upper()
        for task_key in ("battery", "cage"):
            tab = rig.get(f"{task_key}_tab", "")
            if not tab:
                continue
            df = _load_sheet_via_api(sheet_id, tab, sa, ttl_sec)
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
                    "rig": letter,
                    "task": task_key,
                    "by": (_normalize_text(row.get(initials_col))
                           if initials_col else ""),
                    "notes": (_normalize_text(row.get(notes_col))
                              if notes_col else ""),
                })
    all_events.sort(key=lambda x: x["when"], reverse=True)
    return all_events[:limit]


def recent_incidents(config: dict, limit: int = 10,
                     ttl_sec: float | None = None) -> list[dict]:
    """Recent rows from the Incident Report tab, newest first."""
    cfg = _maintenance_cfg(config)
    if not cfg.get("enabled", False):
        return []
    sa = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    sheet_id = cfg.get("sheet_id", "")
    tab = cfg.get("incident_tab_name", "⚠️ Incident Report form")
    if not (sa and sheet_id and tab):
        return []
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0

    df = _load_sheet_via_api(sheet_id, tab, sa, ttl_sec)
    if df is None or df.empty:
        return []
    ts_col = _find_column(df, ["Timestamp", "Time", "Date"])
    report_col = _find_column(df, ["Report", "Report (be as descriptive "
                                    "as needed)", "Description"])
    initials_col = _find_column(df, ["Initials", "Operator", "By"])
    if ts_col is None:
        return []
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    out: list[dict] = []
    for idx in ts.dropna().sort_values(ascending=False).index[:limit]:
        row = df.loc[idx]
        out.append({
            "when": ts.loc[idx].to_pydatetime(),
            "report": (_normalize_text(row.get(report_col))
                       if report_col else ""),
            "by": (_normalize_text(row.get(initials_col))
                   if initials_col else ""),
        })
    return out


# ===================================================================== #
#  Rendering
# ===================================================================== #

def _detail_text(r: RigTaskStatus) -> str:
    if r.task == "battery":
        if not r.is_active:
            return "no active shelves"
        if r.status == "not_scheduled":
            return "scheduled Mon/Wed/Fri" if r.assignees == [] else ""
        if r.done_shelves and not r.pending_shelves:
            return f"shelves done: {', '.join(map(str, r.done_shelves))}"
        if r.pending_shelves:
            pending = ", ".join(map(str, r.pending_shelves))
            done = (f"; done: {', '.join(map(str, r.done_shelves))}"
                    if r.done_shelves else "")
            return f"pending shelves: {pending}{done}"
        return "no shelves active"
    # cage
    if not r.is_active:
        return "no active shelves"
    if r.status == "not_scheduled":
        return ""
    return "entry logged today" if r.done_today else "no entry yet today"


def _status_grid(rows: list[RigTaskStatus]) -> html.Div:
    if not rows:
        return empty_state("No rigs configured.")
    children = []
    for r in rows:
        who = (", ".join(n for n, _e in r.assignees)
               if r.assignees else "-")
        children.append(html.Div([
            html.Div(f"Rig {r.rig}", style={
                "flex": "0 0 70px", "fontWeight": "600",
                "color": COLOR_TEXT_PRIMARY,
            }),
            html.Div(_task_pretty(r.task),
                     style={"flex": "0 0 130px",
                            "color": COLOR_TEXT_PRIMARY}),
            html.Div(_detail_text(r),
                     style={"flex": "1", "color": COLOR_TEXT_SECONDARY,
                            "fontSize": FONT_SIZE_CAPTION}),
            html.Div(who, style={"flex": "0 0 110px",
                                  "color": COLOR_TEXT_SECONDARY,
                                  "fontSize": FONT_SIZE_CAPTION}),
            pill(r.status),
        ], style={
            "display": "flex", "alignItems": "center", "gap": SPACE_3,
            "padding": f"{SPACE_3} {SPACE_3}",
            "borderBottom": f"1px solid {COLOR_DIVIDER}",
            "backgroundColor": COLOR_SURFACE_1,
        }))
    return html.Div(children, style={
        "borderRadius": RADIUS_MD, "overflow": "hidden",
        "border": f"1px solid {COLOR_DIVIDER}",
    })


def _recent_table(events: list[dict]) -> html.Div:
    if not events:
        return empty_state("No recent events.")
    rows = [{
        "when": e["when"].strftime("%Y-%m-%d %H:%M"),
        "rig": f"Rig {e['rig']}",
        "task": _task_pretty(e["task"]),
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
        data=rows, columns=columns, page_size=10, **TABLE_STYLE,
    )])


def _incidents_table(events: list[dict]) -> html.Div:
    if not events:
        return empty_state("No incident reports.")
    rows = [{
        "when": e["when"].strftime("%Y-%m-%d %H:%M"),
        "by": e["by"] or "-",
        "report": e["report"],
    } for e in events]
    columns = [
        {"name": "When", "id": "when"},
        {"name": "By", "id": "by"},
        {"name": "Report", "id": "report"},
    ]
    return html.Div([dash_table.DataTable(
        data=rows, columns=columns, page_size=10, **TABLE_STYLE,
    )])


def _footer(rows: list[RigTaskStatus], sheet_id: str) -> html.Div:
    if not rows:
        return html.Div()
    lines = []
    for r in rows:
        key = f"api:{sheet_id}:{r.tab_name}:h1"
        ts = _last_fetched(key)
        when = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                if ts else "never")
        lines.append(html.Div(
            f"Rig {r.rig} {_task_pretty(r.task)} -- fetched {when}",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_CAPTION,
                   "marginBottom": "2px"},
        ))
    return html.Div(lines, style={"marginTop": SPACE_4})


def _render(rows: list[RigTaskStatus], events: list[dict],
            incidents: list[dict], sheet_id: str, today: date) -> html.Div:
    section_h = {
        "color": COLOR_TEXT_SECONDARY, "marginTop": SPACE_5,
        "marginBottom": SPACE_2, "fontSize": FONT_SIZE_CAPTION,
        "textTransform": "uppercase", "letterSpacing": "0.5px",
    }
    return html.Div([
        html.H3(f"Maintenance -- {today.isoformat()}",
                style={"color": COLOR_TEXT_PRIMARY,
                       "marginBottom": SPACE_3}),
        html.P(
            "Mirrors the lab's Apps Script: today's scheduled tasks + "
            "per-shelf completion, read live from the same sheet. The "
            "QC Monitor doesn't send maintenance reminders -- the Apps "
            "Script does that.",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_CAPTION,
                   "marginBottom": SPACE_4},
        ),
        html.H4("Today's status", style=section_h),
        _status_grid(rows),
        html.H4("Recent activity", style=section_h),
        _recent_table(events),
        html.H4("Incident reports", style=section_h),
        _incidents_table(incidents),
        _footer(rows, sheet_id),
    ])


# ===================================================================== #
#  Public layout + callbacks
# ===================================================================== #

def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = _maintenance_cfg(config or {})
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Maintenance",
                    style={"color": COLOR_TEXT_PRIMARY}),
            html.P(
                "Maintenance tracking is disabled. Enable it in "
                "config.yaml -> maintenance.enabled and configure the "
                "Sheet ID + rigs.",
                style={"color": COLOR_TEXT_SECONDARY},
            ),
        ], style={"padding": SPACE_5})
    refresh_min = float(cfg.get("refresh_minutes", 10))
    return html.Div([
        refresh_bar("maintenance-refresh-btn"),
        dcc.Interval(id="maintenance-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="maintenance-pane"),
    ], style={"padding": SPACE_5})


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
        incidents = recent_incidents(config, limit=10, ttl_sec=ttl)
        sheet_id = cfg.get("sheet_id", "")
        return _render(rows, events, incidents, sheet_id, now.date())
