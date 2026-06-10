"""Annotations tab -- free-text lab notes tagged by category +
optional session, attributed to the logged-in user.

Layout: a DataTable of existing annotations on top, an "Add Note"
form below. The submit callback inserts the row and returns the
refreshed table data.

The DB-row -> table-row shaping lives in ``_table_data`` so the
initial render and the post-submit refresh share one transform
(this was ``_annotations_table_data`` in app.py, used by both
paths).

Lifted out of ``src/dashboard/app.py`` following the per-tab pattern.
"""

from __future__ import annotations

import os
from datetime import datetime

from dash import Input, Output, State, dash_table, dcc, html, no_update

from src.dashboard.auth import current_user_email
from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, INPUT_STYLE, LABEL_STYLE,
    SECTION_STYLE, ZEBRA_STRIPE,
)
from src.dashboard.data_helpers import session_dropdown_options
from src.db.store import Store

_CATEGORY_OPTIONS = [
    {"label": "Electrode", "value": "electrode"},
    {"label": "Injection", "value": "injection"},
    {"label": "Experiment", "value": "experiment"},
    {"label": "Observation", "value": "observation"},
]

_COLUMNS = [
    {"name": "ID", "id": "id"},
    {"name": "Timestamp", "id": "timestamp"},
    {"name": "Category", "id": "category"},
    {"name": "Note", "id": "note"},
    {"name": "Session", "id": "session_dir"},
    {"name": "User", "id": "user"},
    {"name": "Created At", "id": "created_at"},
]


def _table_data(annotations: list[dict]) -> list[dict]:
    """Shape annotation DB rows into DataTable rows. Single source of
    truth for both the initial layout and the post-submit refresh."""
    return [
        {
            "id": a.get("id", ""),
            "timestamp": (a.get("timestamp") or "")[:19],
            "category": a.get("category", ""),
            "note": a.get("note", ""),
            "session_dir": os.path.basename(a.get("session_dir") or ""),
            "user": a.get("user_email", "") or "",
            "created_at": (a.get("created_at") or "")[:19],
        }
        for a in annotations
    ]


def layout(store: Store):
    """Build the Annotations tab: existing-notes table + add-note
    form."""
    session_options = session_dropdown_options(store)
    try:
        all_annotations = store.get_annotations()
    except Exception:
        all_annotations = []

    return html.Div([
        html.H3("Annotations",
                style={"color": "white", "marginBottom": "12px"}),

        dash_table.DataTable(
            id="annotation-table",
            data=_table_data(all_annotations),
            columns=_COLUMNS,
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{category} = electrode"},
                 "color": "#FFA15A"},
                {"if": {"filter_query": "{category} = injection"},
                 "color": "#AB63FA"},
                {"if": {"filter_query": "{category} = experiment"},
                 "color": "#636EFA"},
                {"if": {"filter_query": "{category} = observation"},
                 "color": "#00CC96"},
            ],
            page_size=20,
            filter_action="native",
            sort_action="native",
        ),

        html.Div([
            html.H4("Add Note",
                    style={"color": "white", "marginTop": "24px",
                           "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Timestamp", style=LABEL_STYLE),
                    dcc.Input(
                        id="annotation-timestamp",
                        value=datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S"),
                        type="text",
                        style=INPUT_STYLE,
                        placeholder="YYYY-MM-DDTHH:MM:SS",
                    ),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Label("Category", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-category",
                        options=_CATEGORY_OPTIONS,
                        value="observation",
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "0 0 180px"}),
                html.Div([
                    html.Label("Session (optional)", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-session-dropdown",
                        options=session_options,
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "1", "minWidth": "250px"}),
            ], style={"display": "flex", "gap": "16px",
                      "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Label("Note", style=LABEL_STYLE),
                dcc.Textarea(
                    id="annotation-note",
                    style={**INPUT_STYLE, "height": "80px",
                           "resize": "vertical"},
                    placeholder="Enter your annotation...",
                ),
            ], style={"marginBottom": "12px"}),

            html.Button(
                "Submit Annotation", id="annotation-submit-btn",
                n_clicks=0,
                style={"backgroundColor": "#00CC96", "color": "white",
                       "border": "none", "padding": "10px 24px",
                       "borderRadius": "6px", "cursor": "pointer",
                       "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(id="annotation-status",
                     style={"marginTop": "8px"}),
        ], style=SECTION_STYLE),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the submit-annotation callback. Returns (status message,
    refreshed table data) so the new row appears without a full tab
    re-render."""

    @app.callback(
        [Output("annotation-status", "children"),
         Output("annotation-table", "data")],
        Input("annotation-submit-btn", "n_clicks"),
        [State("annotation-timestamp", "value"),
         State("annotation-note", "value"),
         State("annotation-category", "value"),
         State("annotation-session-dropdown", "value")],
        prevent_initial_call=True,
    )
    def submit_annotation(n_clicks, timestamp_val, note, category,
                          session_dir):
        if not n_clicks:
            return no_update, no_update
        if not note or not note.strip():
            return (html.Div("Note cannot be empty",
                             style={"color": "#EF553B"}),
                    no_update)
        ts = timestamp_val or datetime.now().isoformat()
        try:
            ann_id = store.add_annotation(
                timestamp=ts,
                note=note.strip(),
                category=category or "observation",
                session_dir=session_dir or None,
                user_email=current_user_email(),
            )
            all_ann = store.get_annotations()
            return (
                html.Div(f"Annotation #{ann_id} saved",
                         style={"color": "#00CC96"}),
                _table_data(all_ann),
            )
        except Exception as e:
            return (html.Div(f"Error: {e}",
                             style={"color": "#EF553B"}),
                    no_update)
