"""Data Log Diff -- KMrecorder Sheet vs monitor.db cross-reference.

Sheet 1 (`KMrecorder Data Log`) is an operator-maintained log of every
recording session across multiple PCs. monitor.db tracks every file the
QC Monitor processed. They should agree -- this tab flags the
discrepancies:

* logged-not-QC'd: the operator recorded the session in the Sheet but
  the QC Monitor never saw the file (different PC, failed copy, ...).
* QC'd-not-logged: the QC Monitor processed a file but it never made it
  into the Sheet.

Both sides match on the trailing ``___YYYY_MM_DD__HH_MM_SS`` substring
that ``src/watcher.py:MAT_PATTERN`` already extracts from .mat filenames.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime

import pandas as pd
from dash import Input, Output, callback_context, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.tabs.surgeries import (
    _load_sheet_via_api, _resolve_sa_path, _last_fetched,
    _find_column, _normalize_text,
)
from src.watcher import MAT_PATTERN

logger = logging.getLogger("qc_monitor.dashboard.data_log_xref")

# Match the timestamp substring in either a Sheet path (no .mat) or a
# DB path (with .mat). The Sheet log uses the same naming convention so
# the regex below is more lenient than MAT_PATTERN (no trailing .mat).
_TS_RE = re.compile(r"___(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})")


# ===================================================================== #
#  Key extraction
# ===================================================================== #

def _extract_key(path_or_filename: str) -> str | None:
    """Return the YYYY_MM_DD__HH_MM_SS substring, or None."""
    if not path_or_filename:
        return None
    m = _TS_RE.search(str(path_or_filename))
    return m.group(1) if m else None


def _xref_cfg(config: dict) -> dict:
    return (config or {}).get("data_log_xref", {}) or {}


def _sheet_records(config: dict, ttl_sec: float
                   ) -> tuple[dict[str, dict], int, int]:
    """Pull the KMrecorder log and key it by extracted timestamp.

    Returns (records, total_rows, dropped_rows). *records* maps the
    timestamp key -> a small dict with display fields.
    """
    cfg = _xref_cfg(config)
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "")
    sa_file = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    if not (sheet_id and tab_name and sa_file):
        return {}, 0, 0

    df = _load_sheet_via_api(sheet_id, tab_name, sa_file, ttl_sec)
    if df is None or df.empty:
        return {}, 0, 0

    filename_col = _find_column(df, ["Filename", "File", "Path"])
    date_col = _find_column(df, ["Date"])
    animal_col = _find_column(df, ["Animal_ID", "Animal"])
    channels_col = _find_column(df, ["Channels"])
    operator_col = _find_column(df, ["Operator"])
    pc_col = _find_column(df, ["PC Name", "PC"])
    if filename_col is None:
        return {}, len(df), len(df)

    records: dict[str, dict] = {}
    dropped = 0
    for _idx, row in df.iterrows():
        key = _extract_key(row.get(filename_col))
        if key is None:
            dropped += 1
            continue
        records[key] = {
            "key": key,
            "date": (_normalize_text(row.get(date_col))
                     if date_col else ""),
            "animal": (_normalize_text(row.get(animal_col))
                       if animal_col else ""),
            "channels": (_normalize_text(row.get(channels_col))
                         if channels_col else ""),
            "operator": (_normalize_text(row.get(operator_col))
                         if operator_col else ""),
            "pc": (_normalize_text(row.get(pc_col)) if pc_col else ""),
            "filename": _normalize_text(row.get(filename_col)),
        }
    return records, len(df), dropped


def _db_records(store: Store) -> dict[str, dict]:
    """All processed_files rows, keyed by extracted timestamp."""
    records: dict[str, dict] = {}
    conn = store._connect()
    try:
        rows = conn.execute(
            "SELECT id, file_path, session_name, chunk_datetime, status "
            "FROM processed_files",
        ).fetchall()
    finally:
        conn.close()
    for r in rows:
        key = _extract_key(r["file_path"])
        if key is None:
            continue
        records[key] = dict(r)
    return records


# ===================================================================== #
#  Public helpers (also used by Overview home block)
# ===================================================================== #

def diff_counts(store: Store, config: dict,
                ttl_sec: float | None = None) -> dict:
    """Return {'logged_not_qcd', 'qcd_not_logged', 'sheet_total',
    'db_total'} counts. Used by the home page block too."""
    cfg = _xref_cfg(config)
    if not cfg.get("enabled", False):
        return {"logged_not_qcd": 0, "qcd_not_logged": 0,
                "sheet_total": 0, "db_total": 0, "enabled": False}
    if ttl_sec is None:
        ttl_sec = float(cfg.get("refresh_minutes", 15)) * 60.0

    sheet_records, sheet_total, _drop = _sheet_records(config, ttl_sec)
    db_records = _db_records(store)
    sheet_keys = set(sheet_records.keys())
    db_keys = set(db_records.keys())
    return {
        "logged_not_qcd": len(sheet_keys - db_keys),
        "qcd_not_logged": len(db_keys - sheet_keys),
        "sheet_total": sheet_total,
        "db_total": len(db_records),
        "enabled": True,
    }


# ===================================================================== #
#  Rendering
# ===================================================================== #

_TABLE_STYLE = {
    "style_table": {"overflowX": "auto"},
    "style_header": {
        "backgroundColor": "#1a1a2e", "color": "#e0e0ea",
        "fontWeight": "600",
        "border": "1px solid rgba(255,255,255,0.07)",
    },
    "style_cell": {
        "backgroundColor": "#13131f", "color": "#f0f0f5",
        "padding": "8px", "fontSize": "12px",
        "border": "1px solid rgba(255,255,255,0.05)",
        "textAlign": "left",
        "whiteSpace": "normal",
        "height": "auto",
    },
}


def _logged_not_qcd_table(rows: list[dict]) -> html.Div:
    if not rows:
        return html.Div(
            "None -- every Sheet row has a matching QC'd file.",
            style={"color": "#6c6c80", "fontSize": "13px",
                   "padding": "12px 0", "fontStyle": "italic"},
        )
    data = [{
        "ts": r["key"],
        "date": r["date"],
        "animal": r["animal"],
        "channels": r["channels"],
        "operator": r["operator"],
        "pc": r["pc"],
        "filename": (r["filename"][-60:] if len(r["filename"]) > 60
                     else r["filename"]),
    } for r in rows[:50]]
    columns = [
        {"name": "Timestamp", "id": "ts"},
        {"name": "Date", "id": "date"},
        {"name": "Animal", "id": "animal"},
        {"name": "Ch", "id": "channels"},
        {"name": "Operator", "id": "operator"},
        {"name": "PC", "id": "pc"},
        {"name": "Filename (tail)", "id": "filename"},
    ]
    return html.Div([dash_table.DataTable(
        data=data, columns=columns, page_size=10, **_TABLE_STYLE,
    )])


def _qcd_not_logged_table(rows: list[dict]) -> html.Div:
    if not rows:
        return html.Div(
            "None -- every QC'd file has a matching Sheet row.",
            style={"color": "#6c6c80", "fontSize": "13px",
                   "padding": "12px 0", "fontStyle": "italic"},
        )
    data = [{
        "ts": r.get("chunk_datetime") or "",
        "session": r.get("session_name") or "",
        "status": r.get("status") or "",
        "file_path": ((r.get("file_path") or "")[-60:]
                      if r.get("file_path") and len(r["file_path"]) > 60
                      else r.get("file_path") or ""),
    } for r in rows[:50]]
    columns = [
        {"name": "Timestamp", "id": "ts"},
        {"name": "Session", "id": "session"},
        {"name": "Status", "id": "status"},
        {"name": "File (tail)", "id": "file_path"},
    ]
    return html.Div([dash_table.DataTable(
        data=data, columns=columns, page_size=10, **_TABLE_STYLE,
    )])


def _counter_card(label: str, value: int, color: str) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontSize": "11px", "color": "#a0a0b0",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "marginBottom": "4px",
        }),
        html.Div(str(value), style={
            "fontSize": "26px", "fontWeight": "700", "color": color,
        }),
    ], style={
        "backgroundColor": "#13131f", "padding": "16px 22px",
        "borderRadius": "8px",
        "border": "1px solid rgba(255,255,255,0.07)",
        "flex": "0 0 200px",
    })


def _render(today: date, sheet_records: dict[str, dict],
            db_records: dict[str, dict], sheet_total: int,
            sheet_dropped: int, sheet_id: str, tab_name: str) -> html.Div:
    sheet_keys = set(sheet_records.keys())
    db_keys = set(db_records.keys())
    logged_not_qcd_keys = sorted(sheet_keys - db_keys, reverse=True)
    qcd_not_logged_keys = sorted(db_keys - sheet_keys, reverse=True)

    logged_not_qcd_rows = [sheet_records[k] for k in logged_not_qcd_keys]
    qcd_not_logged_rows = [db_records[k] for k in qcd_not_logged_keys]

    fetch_key = f"api:{sheet_id}:{tab_name}:h1"
    ts = _last_fetched(fetch_key)
    fetched_str = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                   if ts else "never")

    return html.Div([
        html.H3(f"Data log diff -- {today.isoformat()}",
                style={"color": "#f0f0f5", "marginBottom": "12px"}),
        html.Div([
            _counter_card("Logged, not QC'd", len(logged_not_qcd_rows),
                          "#ff9f0a"),
            _counter_card("QC'd, not logged", len(qcd_not_logged_rows),
                          "#5e7ce2"),
            _counter_card("Sheet rows", sheet_total, "#a0a0b0"),
            _counter_card("DB rows", len(db_records), "#a0a0b0"),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "20px"}),

        html.H4("Logged but not QC'd", style={
            "color": "#a0a0b0", "marginTop": "16px", "marginBottom": "8px",
            "fontSize": "13px", "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        }),
        _logged_not_qcd_table(logged_not_qcd_rows),

        html.H4("QC'd but not logged", style={
            "color": "#a0a0b0", "marginTop": "24px", "marginBottom": "8px",
            "fontSize": "13px", "textTransform": "uppercase",
            "letterSpacing": "0.5px",
        }),
        _qcd_not_logged_table(qcd_not_logged_rows),

        html.Div(
            f"Sheet fetched {fetched_str} -- {sheet_dropped} sheet rows "
            f"dropped (no parseable timestamp).",
            style={"color": "#6c6c80", "fontSize": "11px",
                   "marginTop": "16px"},
        ),
    ])


# ===================================================================== #
#  Public layout + callbacks
# ===================================================================== #

def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = _xref_cfg(config or {})
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Data log diff", style={"color": "#f0f0f5"}),
            html.P(
                "Cross-reference disabled. Enable it in config.yaml -> "
                "data_log_xref and point sheet_id + tab_name at the "
                "KMrecorder Data Log Sessions tab.",
                style={"color": "#a0a0b0"},
            ),
        ], style={"padding": "24px"})

    refresh_min = float(cfg.get("refresh_minutes", 15))
    return html.Div([
        html.Div([
            html.Button(
                "Refresh", id="dlx-refresh-btn", n_clicks=0,
                style={
                    "backgroundColor": "#262638", "color": "#f0f0f5",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "padding": "8px 14px", "borderRadius": "6px",
                    "cursor": "pointer", "fontSize": "13px",
                },
            ),
        ], style={"display": "flex", "justifyContent": "flex-end",
                  "marginBottom": "12px"}),
        dcc.Interval(id="dlx-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="dlx-pane"),
    ], style={"padding": "24px"})


def register_callbacks(app, store: Store, config: dict | None = None) -> None:
    cfg = _xref_cfg(config or {})
    if not cfg.get("enabled", False):
        return

    default_ttl = float(cfg.get("refresh_minutes", 15)) * 60.0

    @app.callback(
        Output("dlx-pane", "children"),
        Input("dlx-refresh-btn", "n_clicks"),
        Input("dlx-tick", "n_intervals"),
    )
    def _render_cb(_n_clicks, _n_intervals):
        triggered = (callback_context.triggered_id
                     if callback_context.triggered else None)
        ttl = 0.0 if triggered == "dlx-refresh-btn" else default_ttl
        sheet_records, sheet_total, sheet_dropped = _sheet_records(
            config, ttl,
        )
        db_records = _db_records(store)
        return _render(
            date.today(), sheet_records, db_records,
            sheet_total, sheet_dropped,
            cfg.get("sheet_id", ""), cfg.get("tab_name", ""),
        )
