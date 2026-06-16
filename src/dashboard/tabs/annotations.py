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
import re
from datetime import datetime

from dash import Input, Output, State, dash_table, dcc, html, no_update

from src.dashboard.auth import current_user_email
from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, INPUT_STYLE, LABEL_STYLE,
    SECTION_STYLE, ZEBRA_STRIPE,
)
from src.dashboard.design import COLOR_ACCENT
from src.dashboard.data_helpers import session_dropdown_options
from src.db.store import Store
from src.utils.animal import is_animal_channel

# Video Review prefixes its notes with the playback time, e.g.
# "[t=12.34s] grooming". Pull that out so a note can deep-link back to
# the exact moment.
_TS_RE = re.compile(r"\[t=([0-9]+(?:\.[0-9]+)?)s\]")


def _start_sec_from_note(note: str) -> float:
    """Parse the ``[t=..s]`` playback-time prefix Video Review writes;
    0.0 when the note has none."""
    m = _TS_RE.search(note or "")
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return 0.0


def _first_animal_channel_index(store: Store, session_dir: str) -> int:
    """First animal-bearing channel for the LFP stitch -- notes don't
    record a channel, so default to the same heuristic Video Review
    uses. Falls back to channel 0."""
    try:
        names = store._channel_names_for_session(session_dir)
    except Exception:
        return 0
    for i, n in enumerate(names or []):
        if isinstance(n, str) and is_animal_channel(n):
            return i
    return 0

_CATEGORY_OPTIONS = [
    {"label": "Electrode", "value": "electrode"},
    {"label": "Injection", "value": "injection"},
    {"label": "Experiment", "value": "experiment"},
    {"label": "Observation", "value": "observation"},
]

_COLUMNS = [
    # Clickable only for notes tied to a recording (file_id present):
    # the active_cell callback deep-links into Video Review at the
    # note's moment. Blank for free-form notes.
    {"name": "", "id": "open"},
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
            # "▶ Open" only when the note is tied to a recording; the
            # deep-link callback reads file_id off the annotation id.
            "open": "▶ Open" if a.get("file_id") else "",
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
        html.H3("Notes",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("Notes you save while reviewing video / LFP show up "
                 "here. Click ▶ Open on one to jump straight back "
                 "into Video Review at that recording and moment.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "12px"}),

        dash_table.DataTable(
            id="annotation-table",
            data=_table_data(all_annotations),
            columns=_COLUMNS,
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                # The Open affordance reads as an interactive link.
                {"if": {"column_id": "open"},
                 "color": COLOR_ACCENT, "fontWeight": "600",
                 "cursor": "pointer"},
                {"if": {"filter_query": "{category} = electrode"},
                 "color": "#FFA15A"},
                {"if": {"filter_query": "{category} = injection"},
                 "color": "#AB63FA"},
                {"if": {"filter_query": "{category} = experiment"},
                 "color": "#636EFA"},
                {"if": {"filter_query": "{category} = observation"},
                 "color": "#00CC96"},
                {"if": {"filter_query": "{category} = video_review"},
                 "color": COLOR_ACCENT},
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

    # Click a note's "▶ Open" cell -> hand the recording to Video
    # Review and seek to the note's moment, reusing the same
    # lfp-to-video-bridge + pending-seek path the LFP Browser uses
    # (video.py consumes both, then the clientside seek loop jumps the
    # player to start_sec). Only notes with a file_id are openable.
    @app.callback(
        Output("lfp-to-video-bridge", "data", allow_duplicate=True),
        Output("pending-seek", "data", allow_duplicate=True),
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Output("annotation-table", "active_cell"),
        Input("annotation-table", "active_cell"),
        prevent_initial_call=True,
    )
    def open_note_in_video(active_cell):
        nop = (no_update, no_update, no_update, no_update, no_update)
        # Clear active_cell on a handled open so re-entering the tab
        # doesn't re-navigate; ignore clicks on any other column.
        if not active_cell or active_cell.get("column_id") != "open":
            return nop
        ann_id = active_cell.get("row_id")
        cleared = (no_update, no_update, no_update, no_update, None)
        if ann_id is None:
            return cleared
        try:
            ann = next((a for a in store.get_annotations()
                        if a.get("id") == ann_id), None)
        except Exception:
            return cleared
        if not ann or not ann.get("file_id"):
            return cleared
        file_id = int(ann["file_id"])
        session_dir = ann.get("session_dir")
        duration = 0.0
        try:
            with store.connection() as conn:
                row = conn.execute(
                    "SELECT session_dir, duration_sec "
                    "FROM processed_files WHERE id = ?",
                    (file_id,),
                ).fetchone()
            if row:
                session_dir = session_dir or row["session_dir"]
                duration = float(row["duration_sec"] or 0.0)
        except Exception:
            pass
        if not session_dir:
            return cleared
        start_sec = _start_sec_from_note(ann.get("note") or "")
        channel = _first_animal_channel_index(store, session_dir)
        seq = int(datetime.now().timestamp() * 1000)
        bridge = {
            "session_dir": session_dir, "file_id": file_id,
            "channel": int(channel),
            "hp": 0, "lp": 0, "notch": 0, "smooth": 0,
            "start_sec": float(start_sec),
            "lfp_dur": duration, "seq": seq,
        }
        seek = {"start_sec": float(start_sec), "lfp_dur": duration}
        return bridge, seek, "analysis", "video", None
