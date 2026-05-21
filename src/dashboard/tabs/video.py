"""Video Review tab — synchronized video + LFP trace.

Lets a logged-in reviewer pick an LFP file that has a companion video,
watch the ``_v1.mp4`` in the browser, and see the corresponding LFP
channel time-locked to the video below. Scrubbing or playing the
video moves a cursor along the LFP trace; clicking the LFP trace
seeks the video to that moment. Reviewers can leave a structured
note that gets written to ``annotations`` tagged with their email
and ``category='video_review'``.

The video stream itself is served by ``src.dashboard.media_routes`` at
``/media/video/<file_id>`` and inherits the Cloudflare Access auth
gate.

Time-locking is done entirely on the client to keep latency low. The
server only renders the static LFP figure once per file/channel
selection; clientside callbacks handle the cursor + click-to-seek.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from collections import OrderedDict
from threading import RLock
from dash import Input, Output, State, dcc, html, no_update, Patch
from dash.dependencies import ClientsideFunction

from src.db.store import Store
from src.dashboard.auth import current_user_email
from src.utils.chunk_cache import get_chunk
from src.utils.decimate import (
    envelope, window_slice, choose_target_bins, parse_relayout,
)

logger = logging.getLogger("qc_monitor.dashboard.video")

# Initial-render decimation target. The zoom callback re-decimates the
# visible window dynamically, so this only governs the first paint and
# the auto-reset view.
INITIAL_TARGET_BINS = 60_000

# Stim-blanked per-channel series cache. Re-blanking on every zoom is
# wasteful when only the visible window changed; keyed by
# (file_path, channel, blank_pre_ms, blank_post_ms, stim_fingerprint).
_BLANKED_MAX = 8
_blanked_cache: "OrderedDict[tuple, tuple[np.ndarray, float]]" = OrderedDict()
_blanked_lock = RLock()

LABEL_STYLE = {
    "color": "#6c6c80", "fontSize": "11px", "marginBottom": "8px",
    "display": "block", "letterSpacing": "0.4px",
    "textTransform": "uppercase", "fontWeight": "600",
}
DROPDOWN_STYLE = {"backgroundColor": "#262638", "color": "#f0f0f5"}
SECTION_STYLE = {
    "background": "#13131f", "padding": "24px",
    "borderRadius": "10px", "border": "1px solid rgba(255,255,255,0.07)",
    "marginBottom": "16px",
}

# DOM id assigned to the <video> element so clientside JS can find it.
VIDEO_DOM_ID = "lfp-video"


# ===================================================================== #
#  Data helpers
# ===================================================================== #

def _sessions_with_video(store: Store) -> list[dict]:
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


def _file_path_for_id(store: Store, file_id: int) -> str | None:
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT file_path FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["file_path"] if row else None
    finally:
        conn.close()


def _stim_times_for_file(store: Store, file_id: int) -> np.ndarray:
    """Return stim onset times (sec) detected by the MATLAB pipeline.

    Reads ``evoked_features.epoch_time_sec`` — each epoch corresponds
    to one detected stim event, so this is exactly the list of onsets.
    Returns an empty array for files that haven't been MATLAB-processed.
    """
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT DISTINCT epoch_time_sec
               FROM evoked_features
               WHERE file_id = ? AND epoch_time_sec IS NOT NULL
               ORDER BY epoch_time_sec""",
            (file_id,),
        ).fetchall()
    finally:
        conn.close()
    return np.asarray([r["epoch_time_sec"] for r in rows], dtype=np.float64)


def _stim_copy_channels(store: Store, session_dir: str | None) -> set[int]:
    """Return the set of channel indices flagged as stim-copy.

    Stim-copy channels record the stimulator output itself; blanking
    them out would erase the only signal worth seeing, so we leave
    them as-is.
    """
    if not session_dir:
        return set()
    cfg = store.get_session_config(session_dir) or {}
    raw = cfg.get("stim_copy_channels")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, list):
        return {int(c) for c in raw}
    return set()


def _session_dir_for_file(store: Store, file_id: int) -> str | None:
    conn = store._connect()
    try:
        row = conn.execute(
            "SELECT session_dir FROM processed_files WHERE id = ?", (file_id,),
        ).fetchone()
        return row["session_dir"] if row else None
    finally:
        conn.close()


def _channel_options(store: Store, session_dir: str | None,
                     n_channels: int) -> list[dict]:
    """Build channel-name options for the dropdown.

    Prefers session_config.channel_names if present; otherwise falls
    back to "Ch0", "Ch1", ... so the dropdown always has labels.
    """
    options = [{"label": f"Ch{i}", "value": i} for i in range(n_channels)]
    if not session_dir:
        return options
    cfg = store.get_session_config(session_dir) or {}
    raw_names = cfg.get("channel_names")
    if isinstance(raw_names, str):
        try:
            raw_names = json.loads(raw_names)
        except json.JSONDecodeError:
            raw_names = None
    if isinstance(raw_names, list):
        for i in range(min(n_channels, len(raw_names))):
            options[i] = {"label": f"{raw_names[i]} (Ch{i})", "value": i}
    return options


def _stim_fingerprint(stim_times: np.ndarray | None) -> tuple:
    """Cheap stable fingerprint for use as a cache key."""
    if stim_times is None or len(stim_times) == 0:
        return (0, 0.0, 0.0)
    arr = np.asarray(stim_times, dtype=np.float64)
    return (int(arr.size), float(arr[0]), float(arr[-1]))


def _get_blanked_series(file_path: str, channel: int,
                        stim_times: np.ndarray | None,
                        blank_pre_ms: float,
                        blank_post_ms: float,
                        ) -> tuple[np.ndarray, float, int]:
    """Return the full single-channel series with stim windows NaN-blanked.

    Cached because zoom callbacks would otherwise redo the blanking on
    every relayout event. Returns (series, fs, n_blanked_pulses).
    """
    assert isinstance(file_path, str) and file_path, "file_path required"
    assert channel >= 0, "channel must be non-negative"

    key = (file_path, int(channel), float(blank_pre_ms), float(blank_post_ms),
           _stim_fingerprint(stim_times))

    with _blanked_lock:
        hit = _blanked_cache.pop(key, None)
        if hit is not None:
            _blanked_cache[key] = hit
            series, fs = hit
            # n_blanked is only consumed by the status line; on cache hit
            # report the requested event count (close enough for display).
            n_blanked = 0 if stim_times is None else int(len(stim_times))
            return series, fs, n_blanked

    chunk = get_chunk(file_path)
    fs = float(chunk.fs)
    sig = chunk.signal
    if sig.ndim != 2 or sig.shape[1] <= channel:
        raise ValueError(
            f"Channel {channel} not in file with shape {sig.shape}"
        )
    series = sig[:, channel].astype(np.float32, copy=True)

    n_blanked = 0
    if stim_times is not None and len(stim_times) > 0:
        pre = int(round(blank_pre_ms * 1e-3 * fs))
        post = int(round(blank_post_ms * 1e-3 * fs))
        n = len(series)
        for t_sec in stim_times:
            center = int(round(t_sec * fs))
            lo = max(0, center + pre)
            hi = min(n, center + post)
            if hi > lo:
                series[lo:hi] = np.nan
                n_blanked += 1

    with _blanked_lock:
        _blanked_cache[key] = (series, fs)
        while len(_blanked_cache) > _BLANKED_MAX:
            _blanked_cache.popitem(last=False)

    return series, fs, n_blanked


def _decimated_lfp(file_path: str, channel: int,
                   stim_times: np.ndarray | None = None,
                   blank_pre_ms: float = -5.0,
                   blank_post_ms: float = 15.0,
                   ) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Load one channel of LFP and decimate for the initial render.

    Delegates blanking + caching to ``_get_blanked_series`` and the
    decimation to the shared envelope helper, so the zoom callback can
    use the same path on a sub-window.

    Returns (time_sec, signal_uV, duration_sec, n_blanked_pulses).
    """
    series, fs, n_blanked = _get_blanked_series(
        file_path, channel, stim_times, blank_pre_ms, blank_post_ms,
    )
    target_bins = choose_target_bins(len(series)) or INITIAL_TARGET_BINS
    t, display, _decim = envelope(series, fs, target_bins, t_start=0.0)
    duration = float(len(series) / fs)
    return t, display, duration, n_blanked


def _build_lfp_figure(t: np.ndarray, signal: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=t, y=signal,
        mode="lines",
        line=dict(color="#5e7ce2", width=0.8),
        hovertemplate="t=%{x:.2f}s<br>%{y:.1f} μV<extra></extra>",
        name=label,
    ))
    # Cursor placeholder — clientside callback updates the x position.
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        height=220,
        margin=dict(l=60, r=20, t=10, b=40),
        xaxis=dict(title="Time (s)", showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(title="μV", showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        shapes=[dict(
            type="line", xref="x", yref="paper",
            x0=0, x1=0, y0=0, y1=1,
            line=dict(color="#ff9f0a", width=2),
        )],
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def _empty_lfp_fig(text: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=220,
        plot_bgcolor="#13131f", paper_bgcolor="#13131f",
        annotations=[dict(text=text, showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper",
                          font=dict(size=12, color="#a0a0b0"))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ===================================================================== #
#  Layout
# ===================================================================== #

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
            ], style={"flex": "2", "minWidth": "260px"}),
            html.Div([
                html.Label("File (chunk)", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="video-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "3", "minWidth": "320px"}),
            html.Div([
                html.Label("LFP channel", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="video-channel-dropdown",
                    options=[],
                    value=0,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                  "flexWrap": "wrap"}),

        # --- Player + side panel ---------------------------------------- #
        html.Div([
            html.Div([
                html.Div(
                    id="video-player-container",
                    children=html.P(
                        "Pick a session and file to load the video.",
                        style={"color": "#a0a0b0"},
                    ),
                    style={"backgroundColor": "#000",
                           "borderRadius": "10px",
                           "minHeight": "360px", "display": "flex",
                           "alignItems": "center", "justifyContent": "center",
                           "overflow": "hidden"},
                ),
            ], style={"flex": "2", "minWidth": "480px"}),

            html.Div([
                html.Label("Reviewer note", style=LABEL_STYLE),
                dcc.Textarea(
                    id="video-note-input",
                    placeholder="Notes about this video / LFP chunk...",
                    style={"backgroundColor": "#262638", "color": "#f0f0f5",
                           "border": "1px solid rgba(255,255,255,0.07)",
                           "borderRadius": "6px",
                           "padding": "10px 12px", "width": "100%",
                           "height": "180px", "fontFamily": "inherit",
                           "fontSize": "13px"},
                ),
                html.Button(
                    "Save note", id="video-note-save-btn", n_clicks=0,
                    style={"backgroundColor": "#5e7ce2", "color": "white",
                           "border": "none", "padding": "8px 20px",
                           "borderRadius": "6px", "cursor": "pointer",
                           "fontSize": "13px", "fontWeight": "600",
                           "marginTop": "8px"},
                ),
                html.Div(id="video-note-status",
                         style={"marginTop": "8px", "fontSize": "12px"}),
            ], style={"flex": "1", "minWidth": "260px", **SECTION_STYLE}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start",
                  "flexWrap": "wrap"}),

        # --- Time-locked LFP trace -------------------------------------- #
        html.Div([
            html.Div([
                html.Span("LFP — time-locked to video",
                          style={"color": "#a0a0b0", "fontSize": "11px",
                                 "letterSpacing": "0.4px",
                                 "textTransform": "uppercase",
                                 "fontWeight": "600"}),
                html.Span(id="video-lfp-status",
                          style={"color": "#6c6c80", "fontSize": "11px",
                                 "marginLeft": "12px"}),
                html.Span("Click the trace to seek the video.",
                          style={"color": "#6c6c80", "fontSize": "11px",
                                 "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "baseline",
                      "marginBottom": "8px"}),
            dcc.Graph(
                id="video-lfp-trace",
                figure=_empty_lfp_fig("Pick a file to load the LFP trace."),
                config={"displayModeBar": False},
            ),
        ], style={"marginTop": "20px"}),

        # --- Existing video-review notes for this file ------------------ #
        html.Div([
            html.H4("Past notes for this file",
                    style={"color": "#a0a0b0", "fontSize": "13px",
                           "marginTop": "20px", "marginBottom": "8px",
                           "fontWeight": "600"}),
            html.Div(id="video-note-history",
                     style={"color": "#a0a0b0", "fontSize": "12px"}),
        ]),

        # --- Hidden state ----------------------------------------------- #
        # Polls every 100ms while the tab is open. The clientside callback
        # samples the <video> element's currentTime; if no video is loaded
        # the callback no-ops. Cheap.
        dcc.Interval(id="video-time-tick", interval=100, n_intervals=0),
        dcc.Store(id="video-current-time", data=0.0),
        # Dummy outputs for clientside callbacks that have side effects on
        # the DOM (setting video.currentTime) rather than returning data.
        html.Div(id="video-seek-sink", style={"display": "none"}),
    ])


# ===================================================================== #
#  Callbacks
# ===================================================================== #

def register_callbacks(app, store: Store, config: dict) -> None:
    fa = (config or {}).get("feature_analysis", {}) or {}
    blank_pre_ms = float(fa.get("stim_artifact_start_ms", -5.0))
    blank_post_ms = float(fa.get("stim_artifact_end_ms", 15.0))
    # ---- file/channel options ---- #
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

    # ---- video player + history + channel options ---- #
    @app.callback(
        Output("video-player-container", "children"),
        Output("video-note-history", "children"),
        Output("video-channel-dropdown", "options"),
        Output("video-channel-dropdown", "value"),
        Input("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
    )
    def _update_player(file_id, current_channel):
        if not file_id:
            return (
                html.P("Pick a file to load the video.",
                       style={"color": "#a0a0b0"}),
                "",
                [], 0,
            )

        # Build channel options from the .mat itself (truth) +
        # session_config (for friendly names). We need the actual shape.
        file_path = _file_path_for_id(store, file_id)
        ch_options: list[dict] = []
        ch_value = 0
        if file_path:
            try:
                # Route the channel-count probe through the chunk cache
                # so the subsequent LFP callback hits a warm entry.
                chunk = get_chunk(file_path)
                n_ch = int(chunk.signal.shape[1])
                session_dir = _session_dir_for_file(store, file_id)
                ch_options = _channel_options(store, session_dir, n_ch)
                # Keep current channel if still in range, else 0
                ch_value = current_channel if (
                    current_channel is not None and current_channel < n_ch
                ) else 0
            except Exception as e:
                logger.warning("Channel probe failed for file %d: %s", file_id, e)

        player = html.Video(
            id=VIDEO_DOM_ID,
            src=f"/media/video/{file_id}",
            controls=True,
            preload="metadata",
            style={"width": "100%", "maxHeight": "520px",
                   "borderRadius": "10px", "backgroundColor": "#000"},
        )
        history = _render_history(store, file_id)
        return player, history, ch_options, ch_value

    # ---- LFP trace ---- #
    @app.callback(
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Output("video-lfp-status", "children"),
        Input("video-file-dropdown", "value"),
        Input("video-channel-dropdown", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_lfp(file_id, channel):
        if not file_id:
            return _empty_lfp_fig("Pick a file to load the LFP trace."), ""
        if channel is None:
            return _empty_lfp_fig("Pick an LFP channel."), ""
        file_path = _file_path_for_id(store, file_id)
        if not file_path:
            return _empty_lfp_fig("File not found in DB."), ""

        # Decide whether to apply stim blanking. Only blank channels
        # that aren't themselves the stim source — and only if there
        # are actually detected stim events on this file.
        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        is_stim_copy = channel in stim_copy
        stim_times = (np.asarray([], dtype=np.float64)
                      if is_stim_copy
                      else _stim_times_for_file(store, file_id))

        try:
            t, signal, duration, n_blanked = _decimated_lfp(
                file_path, channel,
                stim_times=stim_times if len(stim_times) else None,
                blank_pre_ms=blank_pre_ms,
                blank_post_ms=blank_post_ms,
            )
        except Exception as e:
            logger.warning("LFP load failed file=%s ch=%s: %s",
                           file_id, channel, e)
            return _empty_lfp_fig(f"LFP load error: {e}"), ""

        fig = _build_lfp_figure(t, signal, label=f"Ch{channel}")
        status_bits = [f"{duration:.1f}s", f"{len(t):,} display points"]
        if is_stim_copy:
            status_bits.append("stim-copy channel · not blanked")
        elif n_blanked:
            status_bits.append(
                f"stim-blanked: {n_blanked} pulses "
                f"({blank_pre_ms:+.0f}–{blank_post_ms:+.0f} ms)"
            )
        elif len(_stim_times_for_file(store, file_id)) == 0:
            status_bits.append("no stim events on file")
        return fig, " · ".join(status_bits)

    # ---- Zoom-driven dynamic decimation ----
    # When the reviewer zooms in, re-decimate just the visible window so
    # narrow features (e.g. 150 us stim pulses) resolve to real samples.
    # The orange cursor lives in figure.layout.shapes[0] and the clientside
    # cursor callback writes layout only, so patching figure.data here is
    # safe -- the cursor and zoom callbacks edit disjoint paths.
    @app.callback(
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-lfp-trace", "relayoutData"),
        State("video-file-dropdown", "value"),
        State("video-channel-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _video_lfp_zoom(relayout, file_id, channel):
        if not file_id or channel is None or not relayout:
            return no_update
        x0, x1, is_reset = parse_relayout(relayout)
        if x0 is None and x1 is None and not is_reset:
            return no_update

        file_path = _file_path_for_id(store, file_id)
        if not file_path:
            return no_update

        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        is_stim_copy = channel in stim_copy
        stim_times = (np.asarray([], dtype=np.float64)
                      if is_stim_copy
                      else _stim_times_for_file(store, file_id))

        try:
            series, fs, _ = _get_blanked_series(
                file_path, channel,
                stim_times if len(stim_times) else None,
                blank_pre_ms, blank_post_ms,
            )
        except Exception as e:
            logger.warning("zoom blanked-series load failed file=%s ch=%s: %s",
                           file_id, channel, e)
            return no_update

        n = len(series)
        if is_reset:
            lo, hi = 0, n
        else:
            lo, hi = window_slice(n, fs, x0, x1)
        if hi <= lo:
            return no_update

        target_bins = choose_target_bins(hi - lo) or INITIAL_TARGET_BINS
        x_p, y_p, _decim = envelope(
            series[lo:hi], fs, target_bins, t_start=lo / fs,
        )

        p = Patch()
        p["data"][0]["x"] = x_p
        p["data"][0]["y"] = y_p
        return p

    # ---- Note save (unchanged) ---- #
    @app.callback(
        Output("video-note-status", "children"),
        Output("video-note-history", "children", allow_duplicate=True),
        Output("video-note-input", "value"),
        Input("video-note-save-btn", "n_clicks"),
        State("video-file-dropdown", "value"),
        State("video-note-input", "value"),
        State("video-current-time", "data"),
        prevent_initial_call=True,
    )
    def _save_note(n_clicks, file_id, note, current_time):
        if not n_clicks:
            return no_update, no_update, no_update
        if not file_id:
            return (html.Span("Select a file first.", style={"color": "#ff453a"}),
                    no_update, no_update)
        if not note or not note.strip():
            return (html.Span("Note cannot be empty.", style={"color": "#ff453a"}),
                    no_update, no_update)

        # Prefix the note with the playback timestamp so reviewers can
        # navigate back to the moment they were commenting on.
        ts_prefix = ""
        if isinstance(current_time, (int, float)) and current_time > 0:
            ts_prefix = f"[t={current_time:.2f}s] "
        body = ts_prefix + note.strip()

        email = current_user_email() or "unknown"
        session_dir = _session_dir_for_file(store, file_id)
        try:
            ann_id = store.add_annotation(
                timestamp=datetime.now().isoformat(),
                note=body,
                category="video_review",
                session_dir=session_dir,
                file_id=file_id,
                user_email=email,
            )
        except Exception as e:
            logger.error("video_review annotation save failed: %s", e, exc_info=True)
            return (html.Span(f"Error: {e}", style={"color": "#ff453a"}),
                    no_update, no_update)

        return (
            html.Span(f"Saved note #{ann_id} as {email}.",
                      style={"color": "#30d158"}),
            _render_history(store, file_id),
            "",
        )

    # ================================================================== #
    #  Clientside time-locking
    # ================================================================== #

    # 1. Poll the <video> element 10 Hz; push currentTime into a Store.
    app.clientside_callback(
        """
        function(_n) {
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (!v || isNaN(v.currentTime)) {
                return window.dash_clientside.no_update;
            }
            return v.currentTime;
        }
        """,
        Output("video-current-time", "data"),
        Input("video-time-tick", "n_intervals"),
    )

    # 2. Move the LFP-trace cursor whenever the stored time changes.
    #    We mutate only the layout.shapes array — Plotly does an
    #    incremental re-render, so this stays smooth even on long files.
    app.clientside_callback(
        """
        function(currentTime, fig) {
            if (fig === undefined || fig === null) {
                return window.dash_clientside.no_update;
            }
            if (currentTime === null || currentTime === undefined) {
                return window.dash_clientside.no_update;
            }
            const newFig = {
                data: fig.data,
                layout: Object.assign({}, fig.layout, {
                    shapes: [{
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: currentTime, x1: currentTime,
                        y0: 0, y1: 1,
                        line: {color: '#ff9f0a', width: 2}
                    }]
                })
            };
            return newFig;
        }
        """,
        Output("video-lfp-trace", "figure", allow_duplicate=True),
        Input("video-current-time", "data"),
        State("video-lfp-trace", "figure"),
        prevent_initial_call=True,
    )

    # 3. Click on the LFP trace -> seek the video to that x value.
    app.clientside_callback(
        """
        function(clickData) {
            if (!clickData || !clickData.points || !clickData.points.length) {
                return '';
            }
            const t = clickData.points[0].x;
            const v = document.getElementById('""" + VIDEO_DOM_ID + """');
            if (v && isFinite(t)) {
                v.currentTime = t;
            }
            return '';
        }
        """,
        Output("video-seek-sink", "children"),
        Input("video-lfp-trace", "clickData"),
    )


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
                      style={"color": "#6c6c80"}),
            html.Span(f"{r['user_email'] or 'unknown'}: ",
                      style={"color": "#5e7ce2"}),
            html.Span(r["note"], style={"color": "#a0a0b0"}),
        ], style={"marginBottom": "4px"})
        for r in rows
    ], style={"paddingLeft": "16px"})
