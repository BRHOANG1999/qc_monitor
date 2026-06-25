"""Operator Log panel -- the KMrecorder operator notes, synced from the
Google Sheet into ``annotations`` (category ``operator_log``) and shown
here as a searchable/sortable table.

The background warmer (``src.utils.operator_log.start_warmer``) keeps the
DB fresh; this panel reads the DB so it never blocks on the Sheets API.
The Refresh button forces an immediate re-sync. Rows whose
``Session_Filename`` matched a QC'd recording are flagged "linked" -- the
same note also appears on that file in the Video Review tab.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from dash import Input, Output, callback_context, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.components import TABLE_STYLE, empty_state, kpi, refresh_bar
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, SPACE_3, SPACE_5,
)
from src.utils import operator_log as oplog

logger = logging.getLogger("qc_monitor.dashboard.operator_log")


def _read_rows(store: Store) -> list[dict]:
    """All operator-log notes from the DB, newest first, flattened for
    the table (context JSON unpacked into display columns)."""
    try:
        with store.connection() as conn:
            rows = conn.execute(
                """SELECT timestamp, note, context, file_id, session_dir
                   FROM annotations WHERE category = 'operator_log'
                   ORDER BY timestamp DESC LIMIT 2000""").fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[dict] = []
    for r in rows:
        try:
            ctx = json.loads(r["context"]) if r["context"] else {}
        except (TypeError, ValueError):
            ctx = {}
        ts = (r["timestamp"] or "").replace("T", " ")[:19]
        out.append({
            "when": ts,
            "operator": ctx.get("operator", ""),
            "linked": "✓" if r["file_id"] else "",
            "session": ctx.get("session_filename", "") or (r["session_dir"] or ""),
            "event": ctx.get("event", ""),
            "note": r["note"] or "",
        })
    return out


def _table(rows: list[dict]) -> html.Div:
    if not rows:
        return empty_state(
            "No operator-log notes yet. They sync from the KMrecorder "
            "Data Log sheet -- check config.yaml -> operator_log.tab_name.")
    columns = [
        {"name": "When", "id": "when"},
        {"name": "Operator", "id": "operator"},
        {"name": "Linked", "id": "linked"},
        {"name": "Session", "id": "session"},
        {"name": "Event", "id": "event"},
        {"name": "Note", "id": "note"},
    ]
    n_linked = sum(1 for r in rows if r["linked"])
    return html.Div([
        html.Div([
            kpi("Notes", len(rows), color=COLOR_TEXT_SECONDARY),
            kpi("Linked to a recording", n_linked, color=COLOR_ACCENT),
        ], style={"display": "flex", "gap": SPACE_3, "flexWrap": "wrap",
                   "marginBottom": SPACE_3}),
        dash_table.DataTable(
            data=rows, columns=columns,
            page_size=25,
            sort_action="native", filter_action="native",
            style_cell={"whiteSpace": "normal", "height": "auto",
                         "textAlign": "left"},
            css=[{"selector": ".dash-filter input",
                   "rule": "text-align: left;"}],
            **TABLE_STYLE,
        ),
    ])


def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = (config or {}).get("operator_log", {}) or {}
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Operator Log", style={"color": COLOR_TEXT_PRIMARY}),
            html.P("Disabled. Enable it in config.yaml -> operator_log "
                    "and set tab_name to the KMrecorder notes tab.",
                    style={"color": "#a0a0b0"}),
        ], style={"padding": "24px"})
    refresh_min = float(cfg.get("refresh_minutes", 15))
    return html.Div([
        html.H3("Operator Log", style={"color": COLOR_TEXT_PRIMARY,
                                         "marginBottom": SPACE_3}),
        refresh_bar("oplog-refresh-btn"),
        dcc.Interval(id="oplog-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="oplog-pane"),
    ], style={"padding": SPACE_5})


def register_callbacks(app, store: Store, config: dict | None = None) -> None:
    cfg = (config or {}).get("operator_log", {}) or {}
    if not cfg.get("enabled", False):
        return

    @app.callback(
        Output("oplog-pane", "children"),
        Input("oplog-refresh-btn", "n_clicks"),
        Input("oplog-tick", "n_intervals"),
    )
    def _render_cb(_n_clicks, _n_intervals):
        triggered = (callback_context.triggered_id
                     if callback_context.triggered else None)
        # The Refresh button forces an immediate re-sync from the sheet;
        # the interval just re-reads the DB (the warmer keeps it fresh).
        if triggered == "oplog-refresh-btn":
            try:
                oplog.sync_once(store, config or {}, ttl_sec=0.0)
            except Exception:
                logger.exception("operator_log manual sync failed")
        return _table(_read_rows(store))
