"""Surgeries tab -- read-only "what's on my plate today" view.

Reads one or more Google Sheets (published as CSV via File -> Share ->
Publish to web) and computes today's required actions for every active
surgery:

* day -1 (implant surgeries only): Pre-op Motrin
* day  0                          : Surgery day
* days +1 / +2 / +3               : Post-op day N

The Sheets stay canonical -- this tab never writes. A small module-level
TTL cache around the CSV fetches keeps the page snappy without hammering
Google's endpoint.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from threading import RLock

import pandas as pd
import requests
from dash import Input, Output, dash_table, dcc, html

from src.db.store import Store

logger = logging.getLogger("qc_monitor.dashboard.surgeries")

# Default candidate-name lists. The user's sheets predate this code so we
# can't assume column names; we look for any case-insensitive match.
_DEFAULT_ALIASES = {
    "animal_id":    ["animal_id", "animal", "animal #", "id", "subject"],
    "surgery_date": ["surgery_date", "date", "surgery date", "op date",
                      "date of surgery"],
    "surgery_type": ["surgery_type", "type", "surgery", "procedure"],
    "notes":        ["notes", "comment", "comments", "note"],
}
_DEFAULT_IMPLANT_KEYWORDS = ["implant", "headstage", "headcap", "electrode"]

# (epoch_sec_fetched, dataframe_or_None) keyed by URL
_cache: "dict[str, tuple[float, pd.DataFrame | None]]" = {}
_cache_lock = RLock()
_FETCH_TIMEOUT_SEC = 10


# ===================================================================== #
#  Data path
# ===================================================================== #

def _load_sheet(url: str, ttl_sec: float) -> pd.DataFrame | None:
    """Return a DataFrame for *url*, hitting the cache when fresh.

    Network errors fall back to the previous cached frame (if any) so a
    flaky Google response doesn't blank the tab.
    """
    assert isinstance(url, str) and url, "url required"
    assert ttl_sec >= 0, "ttl_sec must be non-negative"

    now = time.time()
    with _cache_lock:
        hit = _cache.get(url)
        if hit is not None and (now - hit[0]) < ttl_sec:
            return hit[1]

    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT_SEC)
        resp.raise_for_status()
        df = pd.read_csv(BytesIO(resp.content))
    except Exception as e:
        logger.warning("Surgery sheet fetch failed: %s (url=%s)", e, url)
        with _cache_lock:
            stale = _cache.get(url)
        return stale[1] if stale is not None else None

    with _cache_lock:
        _cache[url] = (now, df)
    return df


def _last_fetched(url: str) -> float | None:
    with _cache_lock:
        hit = _cache.get(url)
    return hit[0] if hit is not None else None


# ===================================================================== #
#  Column discovery and parsing
# ===================================================================== #

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column (case-insensitive), else None."""
    if df is None or df.empty:
        return None
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        hit = lower_map.get(cand.lower().strip())
        if hit is not None:
            return hit
    return None


def _safe_date(value) -> date | None:
    """Parse a heterogeneous date value to a date, else None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if isinstance(ts, datetime):
        return ts.date()
    return ts


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


# ===================================================================== #
#  Compute today's tasks
# ===================================================================== #

def _step_for_delta(delta_days: int, is_implant: bool) -> str | None:
    """Map (today - surgery_date) to the step due today, or None."""
    if delta_days == -1 and is_implant:
        return "Pre-op (Motrin)"
    if delta_days == 0:
        return "Surgery day"
    if 1 <= delta_days <= 3:
        return f"Post-op day {delta_days}"
    return None


def _tasks_from_frame(df: pd.DataFrame, sheet_label: str, today: date,
                      aliases: dict, implant_kw: list[str]
                      ) -> tuple[list[dict], int]:
    """Walk a DataFrame and emit task dicts for rows due today/in window.

    Returns (tasks, dropped) where *dropped* counts rows skipped because a
    required column was missing or unparseable.
    """
    assert isinstance(today, date), "today must be a date"

    if df is None or df.empty:
        return [], 0

    col_animal = _find_column(df, aliases["animal_id"])
    col_date = _find_column(df, aliases["surgery_date"])
    col_type = _find_column(df, aliases["surgery_type"])
    col_notes = _find_column(df, aliases["notes"])

    if col_date is None or col_animal is None:
        return [], len(df)

    tasks: list[dict] = []
    dropped = 0
    for r in df.itertuples(index=False):
        rd = r._asdict()
        animal = _normalize_text(rd.get(col_animal))
        d = _safe_date(rd.get(col_date))
        if not animal or d is None:
            dropped += 1
            continue
        type_str = _normalize_text(rd.get(col_type)) if col_type else ""
        notes = _normalize_text(rd.get(col_notes)) if col_notes else ""
        is_implant = any(k in type_str.lower() for k in implant_kw)
        delta = (today - d).days
        step = _step_for_delta(delta, is_implant)
        if step is None:
            continue
        tasks.append({
            "step": step,
            "animal": animal,
            "type": type_str or "—",
            "surgery_date": d.isoformat(),
            "sheet": sheet_label,
            "notes": notes,
        })
    return tasks, dropped


def _upcoming(df: pd.DataFrame, sheet_label: str, today: date,
              aliases: dict, days_ahead: int = 7) -> list[dict]:
    """Surgeries scheduled in (today, today+days_ahead]."""
    if df is None or df.empty:
        return []
    col_animal = _find_column(df, aliases["animal_id"])
    col_date = _find_column(df, aliases["surgery_date"])
    col_type = _find_column(df, aliases["surgery_type"])
    if col_date is None or col_animal is None:
        return []
    horizon = today + timedelta(days=days_ahead)
    out: list[dict] = []
    for r in df.itertuples(index=False):
        rd = r._asdict()
        d = _safe_date(rd.get(col_date))
        animal = _normalize_text(rd.get(col_animal))
        if d is None or not animal:
            continue
        if today < d <= horizon:
            out.append({
                "animal": animal,
                "type": _normalize_text(rd.get(col_type)) if col_type else "",
                "surgery_date": d.isoformat(),
                "in_days": (d - today).days,
                "sheet": sheet_label,
            })
    out.sort(key=lambda x: x["in_days"])
    return out


# ===================================================================== #
#  Rendering
# ===================================================================== #

_GROUP_ORDER = [
    "Pre-op (Motrin)",
    "Surgery day",
    "Post-op day 1",
    "Post-op day 2",
    "Post-op day 3",
]

_TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": "#1a1a2e", "color": "#e0e0ea",
        "fontWeight": "600", "border": "1px solid rgba(255,255,255,0.07)",
    },
    "style_cell": {
        "backgroundColor": "#13131f", "color": "#f0f0f5",
        "padding": "8px", "fontSize": "13px",
        "border": "1px solid rgba(255,255,255,0.05)",
        "textAlign": "left",
    },
    "style_data_conditional": [{
        "if": {"row_index": "odd"},
        "backgroundColor": "#16162a",
    }],
}


def _empty_group(label: str) -> html.Div:
    return html.Div(
        f"No animals due for {label.lower()} today.",
        style={"color": "#6c6c80", "fontSize": "13px",
               "padding": "12px 0", "fontStyle": "italic"},
    )


def _task_table(tasks: list[dict]) -> html.Div:
    columns = [
        {"name": "Animal", "id": "animal"},
        {"name": "Type", "id": "type"},
        {"name": "Surgery date", "id": "surgery_date"},
        {"name": "Sheet", "id": "sheet"},
        {"name": "Notes", "id": "notes"},
    ]
    return html.Div([dash_table.DataTable(
        data=tasks, columns=columns, page_size=20, **_TABLE_STYLE,
    )])


def _upcoming_table(rows: list[dict]) -> html.Div:
    if not rows:
        return html.Div(
            "No surgeries scheduled in the next 7 days.",
            style={"color": "#6c6c80", "fontSize": "13px",
                   "padding": "12px 0", "fontStyle": "italic"},
        )
    columns = [
        {"name": "Animal", "id": "animal"},
        {"name": "Type", "id": "type"},
        {"name": "Surgery date", "id": "surgery_date"},
        {"name": "In days", "id": "in_days"},
        {"name": "Sheet", "id": "sheet"},
    ]
    return html.Div([dash_table.DataTable(
        data=rows, columns=columns, page_size=20, **_TABLE_STYLE,
    )])


def _footer(sheets_cfg: list[dict], dropped_by_sheet: dict[str, int]
            ) -> html.Div:
    rows = []
    for s in sheets_cfg:
        url = s.get("url", "")
        ts = _last_fetched(url)
        when = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                if ts else "never")
        dropped = dropped_by_sheet.get(s.get("label", ""), 0)
        msg = f"{s.get('label', url)} -- fetched {when}"
        if dropped:
            msg += f" -- {dropped} row(s) dropped (missing date/animal)"
        rows.append(html.Div(msg, style={"color": "#6c6c80",
                                         "fontSize": "12px",
                                         "marginBottom": "4px"}))
    return html.Div(rows, style={"marginTop": "24px"})


def _render_panels(today: date, grouped: dict[str, list[dict]],
                   upcoming: list[dict],
                   sheets_cfg: list[dict],
                   dropped_by_sheet: dict[str, int]) -> html.Div:
    children = [
        html.H3(f"Today's checklist -- {today.isoformat()}",
                style={"color": "#f0f0f5", "marginBottom": "12px"}),
    ]
    for group in _GROUP_ORDER:
        children.append(html.H4(group, style={
            "color": "#a0a0b0", "marginTop": "16px",
            "marginBottom": "8px", "fontSize": "14px",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        }))
        bucket = grouped.get(group, [])
        if bucket:
            children.append(_task_table(bucket))
        else:
            children.append(_empty_group(group))

    children.append(html.H4("Next 7 days", style={
        "color": "#a0a0b0", "marginTop": "24px",
        "marginBottom": "8px", "fontSize": "14px",
        "textTransform": "uppercase", "letterSpacing": "0.5px",
    }))
    children.append(_upcoming_table(upcoming))

    children.append(_footer(sheets_cfg, dropped_by_sheet))
    return html.Div(children)


# ===================================================================== #
#  Public layout + callbacks
# ===================================================================== #

def _surgeries_cfg(config: dict) -> dict:
    return (config or {}).get("surgeries", {}) or {}


def _aliases(config: dict) -> dict:
    overrides = _surgeries_cfg(config).get("column_aliases", {}) or {}
    merged = {k: list(v) for k, v in _DEFAULT_ALIASES.items()}
    for key, vals in overrides.items():
        if key in merged and isinstance(vals, list):
            merged[key] = list(vals) + merged[key]
    return merged


def _implant_kw(config: dict) -> list[str]:
    custom = _surgeries_cfg(config).get("implant_keywords")
    if isinstance(custom, list) and custom:
        return [str(k).lower() for k in custom]
    return list(_DEFAULT_IMPLANT_KEYWORDS)


def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = _surgeries_cfg(config or {})
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Surgeries", style={"color": "#f0f0f5"}),
            html.P(
                "Surgery tracker is disabled. Enable it in "
                "config.yaml -> surgeries.enabled and add sheet URLs.",
                style={"color": "#a0a0b0"},
            ),
        ], style={"padding": "24px"})

    refresh_min = float(cfg.get("refresh_minutes", 10))
    return html.Div([
        html.Div([
            html.Button(
                "Refresh", id="surgeries-refresh-btn", n_clicks=0,
                style={
                    "backgroundColor": "#262638", "color": "#f0f0f5",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "padding": "8px 14px", "borderRadius": "6px",
                    "cursor": "pointer", "fontSize": "13px",
                },
            ),
        ], style={"display": "flex", "justifyContent": "flex-end",
                  "marginBottom": "12px"}),
        dcc.Interval(id="surgeries-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="surgeries-pane"),
    ], style={"padding": "24px"})


def register_callbacks(app, store: Store, config: dict | None = None) -> None:
    cfg = _surgeries_cfg(config or {})
    if not cfg.get("enabled", False):
        return

    sheets_cfg = cfg.get("sheets", []) or []
    ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0
    aliases = _aliases(config or {})
    implant_kw = _implant_kw(config or {})

    @app.callback(
        Output("surgeries-pane", "children"),
        Input("surgeries-refresh-btn", "n_clicks"),
        Input("surgeries-tick", "n_intervals"),
    )
    def _render(_n_clicks, _n_intervals):
        today = date.today()
        grouped: dict[str, list[dict]] = {g: [] for g in _GROUP_ORDER}
        upcoming_all: list[dict] = []
        dropped: dict[str, int] = {}

        for s in sheets_cfg:
            label = s.get("label") or s.get("url", "")
            url = s.get("url", "")
            if not url:
                continue
            df = _load_sheet(url, ttl_sec)
            tasks, n_drop = _tasks_from_frame(
                df, label, today, aliases, implant_kw,
            )
            dropped[label] = n_drop
            for t in tasks:
                grouped.setdefault(t["step"], []).append(t)
            upcoming_all.extend(_upcoming(df, label, today, aliases))

        upcoming_all.sort(key=lambda x: x["in_days"])
        return _render_panels(today, grouped, upcoming_all,
                              sheets_cfg, dropped)
