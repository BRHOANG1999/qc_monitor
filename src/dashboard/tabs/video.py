"""Video Review tab.

Lets a logged-in reviewer pick an LFP file that has a companion video,
watch the ``_v1.mp4`` in the browser, and leave a structured note that
gets written to the ``annotations`` table tagged with their email and
``category='video_review'``.

The video is streamed from the local filesystem by
``src.dashboard.media_routes`` at ``/media/video/<file_id>``; that route
respects the same Cloudflare Access auth gate as the rest of the app.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime

from dash import Input, Output, State, dcc, html, no_update
from dash.dependencies import ALL  # noqa: F401  (reserved for future per-row buttons)

from src.db.store import Store
from src.dashboard.auth import current_user_email

logger = logging.getLogger("qc_monitor.dashboard.video")

LABEL_STYLE = {"color": "#888", "fontSize": "11px", "marginBottom": "3px",
               "display": "block", "letterSpacing": "0.3px"}
DROPDOWN_STYLE = {"backgroundColor": "#1e1e2f", "color": "white"}
SECTION_STYLE = {"background": "linear-gradient(180deg, #1e1e2f 0%, #181828 100%)",
                 "padding": "16px 20px", "borderRadius": "10px",
                 "border": "1px solid #2a2a3e", "marginBottom": "16px",
                 "boxShadow": "0 1px 6px rgba(0,0,0,0.3)"}


def _sessions_with_video(store: Store) -> list[dict]:
    """Return distinct sessions that have at least one file with has_video=1."""
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT session_dir, session_name, COUNT(*) AS n_videos
               FROM processed_files
               WHERE has_video = 1
               GROUP BY session_dir
               ORDER BY MAX(chunk_datetime) DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _files_with_video(store: Store, session_dir: str) -> list[dict]:
    """Return processed_files rows in *session_dir* that have a video."""
    if not session_dir:
        return []
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT id, file_path, chunk_datetime
               FROM processed_files
               WHERE session_dir = ? AND has_video = 1
               ORDER BY chunk_datetime""",
            (session_dir,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def layout(store: Store):
    sessions = _sessions_with_video(store)
    session_options = [
        {"label": f"{s['session_name']} ({s['n_videos']} videos)",
         "value": s["session_dir"]}
        for s in sessions
    ]
    default_session = sessions[0]["session_dir"] if sessions else None

    if not sessions:
        return html.Div([
            html.Div("No videos to review yet",
                     style={"fontSize": "15px", "fontWeight": "600",
                            "color": "#f0f0f5", "marginBottom": "8px",
                            "textAlign": "center"}),
            html.Div(
                "Videos appear here once the dispatcher detects a "
                "_v1.mp4 file alongside a processed .mat file.",
                style={"color": "#a0a0b0", "fontSize": "13px",
                       "textAlign": "center", "maxWidth": "520px",
                       "margin": "0 auto"},
            ),
        ], style={"textAlign": "center", "padding": "32px 24px",
                  "marginTop": "32px"})

    return html.Div([
        html.H3("Video Review", style={"color": "white", "marginBottom": "12px"}),

        # --- Pickers ----------------------------------------------------- #
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="video-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "260px"}),
            html.Div([
                html.Label("File (chunk)", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="video-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "2", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                  "flexWrap": "wrap"}),

        # --- Player + side panel ---------------------------------------- #
        html.Div([
            html.Div([
                html.Div(
                    id="video-player-container",
                    children=html.P(
                        "Pick a session and file to load the video.",
                        style={"color": "#888"},
                    ),
                    style={"backgroundColor": "#000", "borderRadius": "6px",
                           "minHeight": "360px", "display": "flex",
                           "alignItems": "center", "justifyContent": "center"},
                ),
            ], style={"flex": "2", "minWidth": "480px"}),

            html.Div([
                html.Label("Reviewer note", style=LABEL_STYLE),
                dcc.Textarea(
                    id="video-note-input",
                    placeholder="Notes about this video / LFP chunk...",
                    style={"backgroundColor": "#0f0f1a", "color": "white",
                           "border": "1px solid #333", "borderRadius": "6px",
                           "padding": "8px 10px", "width": "100%",
                           "height": "180px", "fontFamily": "inherit",
                           "fontSize": "13px"},
                ),
                html.Button(
                    "Save note", id="video-note-save-btn", n_clicks=0,
                    style={"backgroundColor": "#636EFA", "color": "white",
                           "border": "none", "padding": "8px 20px",
                           "borderRadius": "6px", "cursor": "pointer",
                           "fontSize": "13px", "fontWeight": "bold",
                           "marginTop": "8px"},
                ),
                html.Div(id="video-note-status",
                         style={"marginTop": "8px", "fontSize": "12px"}),
            ], style={"flex": "1", "minWidth": "260px", **SECTION_STYLE}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
                  "flexWrap": "wrap"}),

        # --- Existing video-review notes for this file ------------------ #
        html.Div([
            html.H4("Past notes for this file",
                    style={"color": "#ccc", "fontSize": "13px",
                           "marginTop": "20px", "marginBottom": "8px"}),
            html.Div(id="video-note-history",
                     style={"color": "#888", "fontSize": "12px"}),
        ]),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    @app.callback(
        Output("video-file-dropdown", "options"),
        Output("video-file-dropdown", "value"),
        Input("video-session-dropdown", "value"),
    )
    def _update_files(session_dir):
        files = _files_with_video(store, session_dir)
        options = [
            {"label": f"{(f['chunk_datetime'] or '')[:16]} - {os.path.basename(f['file_path'])}",
             "value": f["id"]}
            for f in files
        ]
        default = options[0]["value"] if options else None
        return options, default

    @app.callback(
        Output("video-player-container", "children"),
        Output("video-note-history", "children"),
        Input("video-file-dropdown", "value"),
    )
    def _update_player(file_id):
        if not file_id:
            return (
                html.P("Pick a file to load the video.",
                       style={"color": "#888"}),
                "",
            )
        # The /media/video/<id> route lives on the same Flask server
        # (registered by media_routes.register_media_routes). The
        # browser inherits the Cloudflare Access cookie, so it will be
        # accepted by the auth hook.
        player = html.Video(
            src=f"/media/video/{file_id}",
            controls=True,
            preload="metadata",
            style={"width": "100%", "maxHeight": "520px",
                   "borderRadius": "6px", "backgroundColor": "#000"},
        )
        history = _render_history(store, file_id)
        return player, history

    @app.callback(
        Output("video-note-status", "children"),
        Output("video-note-history", "children", allow_duplicate=True),
        Output("video-note-input", "value"),
        Input("video-note-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-note-input", "value"),
        prevent_initial_call=True,
    )
    def _save_note(n_clicks, file_id, note):
        if not n_clicks:
            return no_update, no_update, no_update
        if not file_id:
            return (html.Span("Select a file first.", style={"color": "#EF553B"}),
                    no_update, no_update)
        if not note or not note.strip():
            return (html.Span("Note cannot be empty.", style={"color": "#EF553B"}),
                    no_update, no_update)

        email = current_user_email() or "unknown"
        # session_dir for the note: reuse what's stored on processed_files
        session_dir = _session_dir_for_file(store, file_id)
        try:
            ann_id = store.add_annotation(
                timestamp=datetime.now().isoformat(),
                note=note.strip(),
                category="video_review",
                session_dir=session_dir,
                file_id=file_id,
                user_email=email,
            )
        except Exception as e:
            logger.error("video_review annotation save failed: %s", e, exc_info=True)
            return (html.Span(f"Error: {e}", style={"color": "#EF553B"}),
                    no_update, no_update)

        return (
            html.Span(f"Saved note #{ann_id} as {email}.",
                      style={"color": "#00CC96"}),
            _render_history(store, file_id),
            "",
        )


def _session_dir_for_file(store: Store, file_id: int) -> str | None:
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT session_dir FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["session_dir"] if row else None
    finally:
        conn.close()


def _render_history(store: Store, file_id: int):
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT timestamp, user_email, note
               FROM annotations
               WHERE file_id = ? AND category = 'video_review'
               ORDER BY timestamp DESC LIMIT 50""",
            (file_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    if not rows:
        return html.Span("No video-review notes yet for this file.")
    return html.Ul([
        html.Li([
            html.Span(f"{(r['timestamp'] or '')[:19]} ",
                      style={"color": "#888"}),
            html.Span(f"{r['user_email'] or 'unknown'}: ",
                      style={"color": "#636EFA"}),
            html.Span(r["note"], style={"color": "#ccc"}),
        ], style={"marginBottom": "4px"})
        for r in rows
    ], style={"paddingLeft": "16px"})
