"""LFP Browser tab -- full-file raw/filtered LFP viewer with live
band-filter presets, min/max envelope decimation, optional Welch PSD,
and per-channel "View video" jumps into Video Review.

Four callbacks:

* ``_lfp_apply_preset`` -- preset dropdown populates the HP/LP number
  inputs (user still clicks Apply).
* ``load_lfp`` -- the heavyweight one. Loads + filters the chunk
  (both LRU-cached), envelope-decimates each channel for the initial
  paint, optionally builds the PSD figure, and emits one pattern-
  matched "View video" button per channel.
* ``_on_view_video`` -- the per-channel jump. Writes the
  ``lfp-to-video-bridge`` + ``pending-seek`` Stores (consumed by the
  Video Review tab's ``_consume_bridge``) and flips the nav tabs to
  Video Review. This writes app-shell outputs, but it lives here
  rather than app.py because every one of its Inputs/States is
  LFP-tab-owned -- including the pattern-matched buttons that
  ``load_lfp`` itself creates. Splitting producer and consumer
  across modules would be worse than bending the "shell callbacks
  stay in app.py" rule.
* ``_lfp_zoom`` -- relayout-driven re-decimation of the visible
  window via a Patch, using the filter settings persisted in
  ``lfp-filter-state`` so zooming never re-filters.

Plus ``update_lfp_file_options`` to refill the file picker when the
session changes.

Lifted out of ``src/dashboard/app.py``.
"""

from __future__ import annotations

import logging
import os
import time

import plotly.graph_objects as go
from dash import (ALL, Input, Output, Patch, State, callback_context,
                   dcc, html, no_update)
from plotly.subplots import make_subplots

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import (
    channel_map, color_for_role, default_session, empty_fig,
    processed_files_for_session, session_dropdown_options,
)
from src.db.store import Store
from src.utils.chunk_cache import get_chunk
from src.utils.decimate import (
    choose_target_bins, envelope_channel, parse_relayout, window_slice,
)
from src.utils.filters import compute_psd, get_filtered

logger = logging.getLogger("qc_monitor.dashboard.lfp_browser")

# Preset -> (HP, LP) lookup. Selecting a preset populates the number
# inputs; the user still clicks Apply to run the filter.
_LFP_PRESETS = {
    "raw":   (0, 0),
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 100),
    "spike": (300, 3000),
}

_PRESET_OPTIONS = [
    {"label": "Raw (no filter)", "value": "raw"},
    {"label": "Delta (1-4 Hz)", "value": "delta"},
    {"label": "Theta (4-8 Hz)", "value": "theta"},
    {"label": "Alpha (8-13 Hz)", "value": "alpha"},
    {"label": "Beta (13-30 Hz)", "value": "beta"},
    {"label": "Gamma (30-100 Hz)", "value": "gamma"},
    {"label": "Spike band (300-3000 Hz)", "value": "spike"},
    {"label": "Custom", "value": "custom"},
]


def layout(store: Store, default: str | None = None):
    """Build the tab layout. The plot starts as an instructional
    empty state -- nothing heavy loads until the user clicks Load."""
    session_options = session_dropdown_options(store)
    initial_session = default_session(store, hint=default)

    return html.Div([
        html.H3("LFP Browser",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-session-dropdown",
                    options=session_options,
                    value=initial_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("File", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "2", "minWidth": "350px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("Load LFP", id="lfp-load-btn", n_clicks=0,
                            style={"backgroundColor": "#636EFA",
                                   "color": "white", "border": "none",
                                   "padding": "8px 20px",
                                   "borderRadius": "6px",
                                   "cursor": "pointer",
                                   "fontSize": "14px",
                                   "fontWeight": "bold"}),
            ], style={"flex": "0 0 120px", "display": "flex",
                      "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),

        html.P("Loads the entire raw LFP from the selected .mat file. "
               "Long files are min/max envelope-decimated so 150 us "
               "stim pulses remain visible. Channel names come from "
               "the file's fnstr.",
               style={"color": "#888", "fontSize": "12px",
                      "marginBottom": "8px"}),

        # --- Live filter strip ------------------------------------- #
        html.Div([
            html.Div([
                html.Label("Preset", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-filter-preset",
                    options=_PRESET_OPTIONS,
                    value="raw",
                    clearable=False,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1 1 200px", "minWidth": "180px"}),
            html.Div([
                html.Label("HP (Hz)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-hp", type="number", min=0,
                          step=0.5, value=0, style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label("LP (Hz)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-lp", type="number", min=0,
                          step=1, value=0, style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label("Notch", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-filter-notch",
                    options=[
                        {"label": "Off", "value": 0},
                        {"label": "50 Hz", "value": 50},
                        {"label": "60 Hz", "value": 60},
                    ],
                    value=0,
                    clearable=False,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-smooth", type="number",
                          min=0, step=1, value=0,
                          style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label("PSD", style=LABEL_STYLE),
                dcc.Checklist(
                    id="lfp-show-psd",
                    options=[{"label": " Show", "value": "on"}],
                    value=[],
                    style={"paddingTop": "6px"},
                    labelStyle={"color": "white"},
                ),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("Apply", id="lfp-apply-filter-btn",
                            n_clicks=0,
                            style={"backgroundColor": "#262638",
                                   "color": "white",
                                   "border": "1px solid #444",
                                   "padding": "8px 16px",
                                   "borderRadius": "6px",
                                   "cursor": "pointer",
                                   "fontSize": "13px"}),
            ], style={"flex": "0 0 100px", "display": "flex",
                      "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "12px",
                  "marginBottom": "12px", "flexWrap": "wrap",
                  "padding": "10px 12px",
                  "backgroundColor": "#13131f",
                  "borderRadius": "8px",
                  "border": "1px solid rgba(255,255,255,0.06)"}),

        # Persists the last-applied filter tuple so the zoom callback
        # uses the same settings without re-reading the inputs.
        dcc.Store(id="lfp-filter-state",
                  data={"hp": 0, "lp": 0, "notch": 0, "smooth": 0,
                        "show_psd": False}),

        dcc.Graph(id="lfp-plot",
                  figure=empty_fig(
                      "Select a session and file, then click Load",
                      height=600)),

        # Per-channel "View video" buttons. Populated dynamically by
        # load_lfp -- one button per LFP channel. Clicking jumps to
        # Video Review with the same file, channel, filter settings,
        # AND seeks the video to the current LFP zoom's left edge so
        # the operator picks up where they were looking.
        html.Div(
            id="lfp-view-video-row",
            style={"display": "flex", "flexWrap": "wrap",
                    "gap": "6px", "marginTop": "8px",
                    "marginBottom": "8px"},
        ),

        html.Div(
            dcc.Graph(id="lfp-psd-plot",
                      figure=empty_fig(
                          "Toggle 'Show PSD' and click Apply",
                          height=280)),
            id="lfp-psd-row",
            style={"display": "none", "marginTop": "12px"},
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the preset, load, zoom, view-video, and file-options
    callbacks."""

    @app.callback(
        Output("lfp-filter-hp", "value"),
        Output("lfp-filter-lp", "value"),
        Input("lfp-filter-preset", "value"),
        State("lfp-filter-hp", "value"),
        State("lfp-filter-lp", "value"),
        prevent_initial_call=True,
    )
    def _lfp_apply_preset(preset, current_hp, current_lp):
        if preset == "custom" or preset not in _LFP_PRESETS:
            return no_update, no_update
        hp, lp = _LFP_PRESETS[preset]
        return hp, lp

    @app.callback(
        Output("lfp-plot", "figure"),
        Output("lfp-psd-plot", "figure"),
        Output("lfp-psd-row", "style"),
        Output("lfp-filter-state", "data"),
        Output("lfp-view-video-row", "children"),
        Input("lfp-load-btn", "n_clicks"),
        Input("lfp-apply-filter-btn", "n_clicks"),
        State("lfp-session-dropdown", "value"),
        State("lfp-file-dropdown", "value"),
        State("lfp-filter-hp", "value"),
        State("lfp-filter-lp", "value"),
        State("lfp-filter-notch", "value"),
        State("lfp-filter-smooth", "value"),
        State("lfp-show-psd", "value"),
        prevent_initial_call=True,
    )
    def load_lfp(n_load, n_apply, session_dir, file_path,
                  hp, lp, notch, smooth_ms, show_psd_val):
        psd_hidden_style = {"display": "none", "marginTop": "12px"}
        if not file_path:
            return (empty_fig("Select a file and click Load",
                               height=600),
                    no_update, psd_hidden_style, no_update, [])

        try:
            chunk = get_chunk(file_path)
        except Exception as e:
            return (empty_fig(f"Error loading file: {e}", height=600),
                    no_update, psd_hidden_style, no_update, [])

        fs = chunk.fs
        # Apply the cached filter to the full signal. get_filtered is
        # cheap on a cache hit (typical for zoom callbacks) and ~1s on
        # a miss for an hour-long 20kHz dual-channel chunk.
        signal = get_filtered(file_path, chunk.signal, fs,
                               highpass=hp, lowpass=lp,
                               notch=notch, smoothing_ms=smooth_ms)
        n_samples, n_ch = signal.shape
        duration_sec = n_samples / fs

        info_for = _channel_info_fn(store, session_dir, chunk)

        target_bins = choose_target_bins(n_samples) or 60_000
        fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                            vertical_spacing=0.005)

        decim_used = 1
        for ch_idx in range(n_ch):
            info = info_for(ch_idx)
            color = color_for_role(info["role"])
            x_plot, y_plot, decim_used = envelope_channel(
                signal, ch_idx, fs, 0, n_samples, target_bins,
            )
            fig.add_trace(go.Scattergl(
                x=x_plot, y=y_plot,
                mode="lines", name=info["name"],
                line=dict(color=color, width=0.8),
            ), row=ch_idx + 1, col=1)
            fig.update_yaxes(title_text=info["name"],
                             row=ch_idx + 1, col=1,
                             title_font=dict(size=9, color="#aaa"),
                             tickfont=dict(size=8))

        fig.update_xaxes(title_text="Time (sec)", row=n_ch, col=1)
        filt_label = _filter_label(hp, lp, notch, smooth_ms)

        if decim_used > 1:
            bin_ms = decim_used / fs * 1000.0
            title = (f"LFP ({duration_sec:.1f} s) -- {n_ch} ch @ "
                     f"{fs:.0f} Hz -- envelope {bin_ms:.2f} ms/bin "
                     f"-- {filt_label}")
        else:
            title = (f"LFP ({duration_sec:.1f} s) -- {n_ch} ch @ "
                     f"{fs:.0f} Hz -- raw samples -- {filt_label}")
        fig.update_layout(
            title=title,
            height=max(600, n_ch * 80),
            showlegend=False,
        )

        # Persisted filter state -- used by the zoom callback so it
        # decimates from the same filtered cache entry.
        state = {"hp": hp, "lp": lp, "notch": notch,
                 "smooth": smooth_ms,
                 "show_psd": bool(show_psd_val)}

        # Per-channel "View video" buttons. Pattern-matched ID so a
        # single click callback handles any channel via
        # ctx.triggered_id.channel.
        view_video_buttons = [
            html.Button(
                f"View video · {info_for(ch_idx)['name']}",
                id={"type": "lfp-view-video", "channel": ch_idx},
                n_clicks=0,
                title=("Open Video Review with this channel + "
                        "current filter, seek to current LFP zoom"),
                style={"backgroundColor": "#262638",
                        "color": "white",
                        "border": "1px solid #444",
                        "padding": "4px 10px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "fontSize": "11px"},
            )
            for ch_idx in range(n_ch)
        ]

        if not show_psd_val:
            return (fig, no_update, psd_hidden_style, state,
                    view_video_buttons)

        psd_fig = _build_psd_figure(signal, fs, n_ch, info_for,
                                     filt_label)
        return (fig, psd_fig,
                {"display": "block", "marginTop": "12px"},
                state, view_video_buttons)

    # ---- "View video" buttons: jump to Video Review pre-filled ----
    # This writes app-shell outputs (group-tabs / tabs / the bridge
    # Stores) but lives here because every Input/State it reads is
    # LFP-tab-owned, including the pattern-matched buttons that
    # load_lfp above creates.
    @app.callback(
        Output("lfp-to-video-bridge", "data"),
        Output("pending-seek", "data"),
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input({"type": "lfp-view-video", "channel": ALL}, "n_clicks"),
        State("lfp-plot", "relayoutData"),
        State("lfp-session-dropdown", "value"),
        State("lfp-file-dropdown", "value"),
        State("lfp-filter-hp", "value"),
        State("lfp-filter-lp", "value"),
        State("lfp-filter-notch", "value"),
        State("lfp-filter-smooth", "value"),
        prevent_initial_call=True,
    )
    def _on_view_video(n_clicks_list, relayout, session_dir,
                        file_path, hp, lp, notch, smooth_ms):
        # Pattern-matched callbacks fire with n_clicks_list = list of
        # all matching components' n_clicks. The build step also
        # triggers this with zero clicks; only act when something was
        # actually clicked.
        if not n_clicks_list or not any(n_clicks_list):
            return no_update, no_update, no_update, no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update, no_update, no_update, no_update
        channel = int(trig.get("channel", 0))
        if not file_path:
            return no_update, no_update, no_update, no_update
        # Pull the LFP duration so the BHZ-style scaling works on the
        # Video Review side. get_chunk is cached so this is cheap on
        # a warm path.
        try:
            chunk = get_chunk(file_path)
            lfp_dur = float(chunk.signal.shape[0] / chunk.fs)
        except Exception as e:
            logger.warning("View video: chunk load failed: %s", e)
            lfp_dur = 0.0
        x_start = _visible_window_start(relayout)
        # file_path -> file_id (Video Review's dropdown uses file_id)
        file_id = None
        with store.connection() as conn:
            row = conn.execute(
                "SELECT id FROM processed_files WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            if row:
                file_id = int(row["id"])
        bridge = {
            "session_dir": session_dir,
            "file_id": file_id,
            "channel": channel,
            "hp": hp or 0, "lp": lp or 0,
            "notch": notch or 0,
            "smooth": smooth_ms or 0,
            "start_sec": max(0.0, x_start),
            "lfp_dur": lfp_dur,
            "seq": int(time.time() * 1000),
        }
        pending_seek = {
            "start_sec": bridge["start_sec"],
            "lfp_dur": lfp_dur,
            "seq": bridge["seq"],
        }
        return bridge, pending_seek, "analysis", "video"

    @app.callback(
        Output("lfp-plot", "figure", allow_duplicate=True),
        Input("lfp-plot", "relayoutData"),
        State("lfp-file-dropdown", "value"),
        State("lfp-filter-state", "data"),
        prevent_initial_call=True,
    )
    def _lfp_zoom(relayout, file_path, filter_state):
        if not file_path or not relayout:
            return no_update
        x0, x1, is_reset = parse_relayout(relayout)
        if x0 is None and x1 is None and not is_reset:
            return no_update
        try:
            chunk = get_chunk(file_path)
        except Exception:
            return no_update

        fs = chunk.fs
        # Use the persisted filter settings so zoom re-decimates from
        # the same filtered cache the time-domain plot built from.
        st = filter_state or {}
        signal = get_filtered(
            file_path, chunk.signal, fs,
            highpass=st.get("hp"), lowpass=st.get("lp"),
            notch=st.get("notch"), smoothing_ms=st.get("smooth"),
        )
        n_samples, n_ch = signal.shape
        if is_reset:
            lo, hi = 0, n_samples
        else:
            lo, hi = window_slice(n_samples, fs, x0, x1)
        if hi <= lo:
            return no_update

        target_bins = choose_target_bins(hi - lo) or 60_000
        patch = Patch()
        for ch in range(n_ch):
            x_p, y_p, _ = envelope_channel(
                signal, ch, fs, lo, hi, target_bins,
            )
            patch["data"][ch]["x"] = x_p
            patch["data"][ch]["y"] = y_p
        return patch

    @app.callback(
        Output("lfp-file-dropdown", "options"),
        Input("lfp-session-dropdown", "value"),
    )
    def update_lfp_file_options(session_dir):
        if not session_dir:
            return []
        files = processed_files_for_session(store, session_dir)
        return [
            {"label": f"{f['chunk_datetime'][:16]} - "
                       f"{os.path.basename(f['file_path'])}",
             "value": f["file_path"]}
            for f in files
        ]


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #

def _channel_info_fn(store: Store, session_dir: str | None, chunk):
    """Return an ``info_for(ch_idx) -> {"name", "role"}`` lookup that
    prefers the file's own fnstr names (which reflect what was
    actually recorded) over the session_config map, and falls back to
    numeric names when neither knows the channel."""
    sess_map = channel_map(store, session_dir) if session_dir else {}
    file_names = chunk.channel_names or []

    def info_for(ch_idx: int) -> dict:
        if ch_idx < len(file_names) and file_names[ch_idx]:
            name = file_names[ch_idx]
            role = "stim_copy" if "stim" in name.lower() else "eeg"
            return {"name": name, "role": role}
        return sess_map.get(ch_idx,
                             {"name": f"Ch{ch_idx}", "role": "eeg"})

    return info_for


def _filter_label(hp, lp, notch, smooth_ms) -> str:
    """Human-readable summary of the active filter cascade for plot
    titles. 'Raw' when nothing is enabled."""
    bits = []
    if hp and hp > 0:
        bits.append(f"HP={hp:g}")
    if lp and lp > 0:
        bits.append(f"LP={lp:g}")
    if notch and notch > 0:
        bits.append(f"Notch={notch}")
    if smooth_ms and smooth_ms > 0:
        bits.append(f"Smooth={smooth_ms:g}ms")
    return " | ".join(bits) if bits else "Raw"


def _visible_window_start(relayout) -> float:
    """Left edge of the user's current zoom from relayoutData, or 0.0
    (full file) when there's no zoom. relayoutData uses dotted keys
    ('xaxis.range[0]') after a drag-zoom and a list ('xaxis.range')
    after some programmatic updates -- handle both."""
    if not isinstance(relayout, dict):
        return 0.0
    if "xaxis.range[0]" in relayout:
        try:
            return float(relayout["xaxis.range[0]"])
        except (TypeError, ValueError):
            return 0.0
    rng = relayout.get("xaxis.range")
    if isinstance(rng, list) and rng:
        try:
            return float(rng[0])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _build_psd_figure(signal, fs, n_ch, info_for, filt_label):
    """One Welch PSD subplot per channel, log-y, x clipped to
    0..1000 Hz (the operationally interesting band; plotly zoom
    reveals higher). Dotted guides at the line-noise harmonics."""
    psd_fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06)
    x_cap = min(1000.0, fs / 2.0)
    for ch_idx in range(n_ch):
        info = info_for(ch_idx)
        color = color_for_role(info["role"])
        freqs, psd = compute_psd(signal[:, ch_idx], fs)
        if len(freqs) == 0:
            continue
        mask = freqs <= x_cap
        psd_fig.add_trace(go.Scattergl(
            x=freqs[mask], y=psd[mask],
            mode="lines", name=info["name"],
            line=dict(color=color, width=1.2),
        ), row=ch_idx + 1, col=1)
        psd_fig.update_yaxes(type="log", title_text=info["name"],
                              row=ch_idx + 1, col=1,
                              title_font=dict(size=9, color="#aaa"),
                              tickfont=dict(size=8))
        for line_hz in (50, 60, 120, 180):
            if line_hz < x_cap:
                psd_fig.add_vline(
                    x=line_hz, line_width=1,
                    line_dash="dot", line_color="#666",
                    row=ch_idx + 1, col=1,
                )
    psd_fig.update_xaxes(title_text="Hz", row=n_ch, col=1)
    psd_fig.update_layout(
        title=f"Power spectral density (Welch) -- {filt_label}",
        height=max(220, n_ch * 140),
        showlegend=False,
    )
    return psd_fig
