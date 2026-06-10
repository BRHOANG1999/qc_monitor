"""Activity Log tab -- filterable view of the processing_log table.

Layout: a time-range dropdown + a level dropdown over a DataTable.
The initial render is pre-populated server-side; the (hours, level)
callback below re-queries on every filter change.

The row-shaping logic (DB row -> table row dict) used to be inlined
*twice* in app.py -- once for the initial layout data and once in the
callback. It's a single ``_rows_for`` helper here so the two paths
can't drift.

Lifted out of ``src/dashboard/app.py`` following the per-tab pattern.
"""

from __future__ import annotations

import os

from dash import Input, Output, dash_table, dcc, html

from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, LABEL_STYLE, ZEBRA_STRIPE,
)
from src.dashboard.data_helpers import TIME_RANGE_OPTIONS
from src.db.store import Store

_LEVEL_OPTIONS = [
    {"label": "ALL", "value": "ALL"},
    {"label": "INFO", "value": "INFO"},
    {"label": "WARNING", "value": "WARNING"},
    {"label": "ERROR", "value": "ERROR"},
]

_COLUMNS = [
    {"name": "Timestamp", "id": "timestamp"},
    {"name": "Level", "id": "level"},
    {"name": "Action", "id": "action"},
    {"name": "Message", "id": "message"},
    {"name": "File", "id": "file_path"},
    {"name": "Duration (s)", "id": "duration"},
]


def _rows_for(logs: list[dict]) -> list[dict]:
    """Shape activity-log DB rows into DataTable rows. Single source
    of truth for both the initial layout data and the filter
    callback so the two can't diverge."""
    return [
        {
            "timestamp": entry.get("timestamp", "")[:19],
            "level": entry.get("level", ""),
            "action": entry.get("action", ""),
            "message": (entry.get("message") or "")[:200],
            "file_path": os.path.basename(entry.get("file_path") or ""),
            "duration": (f"{entry['duration_sec']:.2f}"
                          if entry.get("duration_sec") else ""),
        }
        for entry in logs
    ]


def layout(store: Store):
    """Build the Activity Log tab. Initial data pre-loaded; the
    callback below re-queries on filter changes."""
    try:
        initial_logs = store.get_activity_log(hours=24, limit=500)
    except Exception:
        initial_logs = []

    return html.Div([
        html.H3("Activity Log",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=24,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
            html.Div([
                html.Label("Level", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-level-dropdown",
                    options=_LEVEL_OPTIONS,
                    value="ALL",
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 150px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),

        dash_table.DataTable(
            id="activity-log-table",
            data=_rows_for(initial_logs),
            columns=_COLUMNS,
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{level} = ERROR"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{level} = WARNING"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{level} = INFO"},
                 "color": "#6bb5ff"},
            ],
            page_size=30,
            filter_action="native",
            sort_action="native",
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the (hours, level) -> table-data callback. Same ids as
    the layout above."""

    @app.callback(
        Output("activity-log-table", "data"),
        [Input("activity-log-hours-dropdown", "value"),
         Input("activity-log-level-dropdown", "value")],
    )
    def update_activity_log(hours, level):
        hours_val = int(hours) if hours else 24
        level_val = level if level and level != "ALL" else None
        try:
            logs = store.get_activity_log(
                hours=hours_val, level=level_val, limit=500)
        except Exception:
            logs = []
        return _rows_for(logs)
