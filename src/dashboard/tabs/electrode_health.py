"""Electrode Health tab -- separates the channel QC time-series by
channel role.

Three stacked subplots:

* Stim-copy channels' RMS amplitude over time (does the stim signal
  itself look right?)
* Reference channels' RMS amplitude over time (is the reference
  drifting?)
* Line-noise ratio for every channel (one trace per channel,
  colour-coded by role)

This is read on render and recomputed on every (session, time-range)
change via the callback below. ``store.get_qc_timeseries`` returns
one row per (file, channel); the per-role split is done here so the
SQL stays unaware of channel roles.

Lifted out of ``src/dashboard/app.py``.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import (
    TIME_RANGE_OPTIONS, channel_map, color_for_role,
    default_session, empty_fig, session_dropdown_options,
)
from src.db.store import Store


def layout(store: Store, default: str | None = None):
    """Build the layout. Initial plot is the "select a session" empty
    state -- the real figure is built by the callback below."""
    session_options = session_dropdown_options(store)
    initial_session = default_session(store, hint=default)
    return html.Div([
        html.H3("Electrode Health",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-session-dropdown",
                    options=session_options,
                    value=initial_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),
        dcc.Graph(id="electrode-health-plot",
                  figure=empty_fig("Select a session", height=700)),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the (session, time-range) -> three-pane figure callback."""

    @app.callback(
        Output("electrode-health-plot", "figure"),
        [Input("electrode-health-session-dropdown", "value"),
         Input("electrode-health-hours-dropdown", "value")],
    )
    def update_electrode_health(session_dir, hours):
        if not session_dir:
            return empty_fig("Select a session", height=700)
        hours_val = int(hours) if hours else 0
        try:
            data = store.get_qc_timeseries(
                session_dir=session_dir, hours=hours_val)
        except Exception as e:
            return empty_fig(f"Error: {e}", height=700)
        if not data:
            return empty_fig("No QC data available", height=700)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Stim Copy RMS", "Reference RMS",
                             "Line Noise Ratio (60 Hz indicator)"],
            vertical_spacing=0.08,
        )
        _add_role_rms(fig, data, role="stim_copy",
                       color="#888888", row=1)
        _add_role_rms(fig, data, role="reference",
                       color="#00CC96", row=2)
        _add_line_noise_traces(fig, data, store, session_dir)

        fig.update_layout(
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="right", x=1),
        )
        fig.update_annotations(font=dict(color="white"))
        return fig


def _add_role_rms(fig: go.Figure, data: list[dict], role: str,
                    color: str, row: int) -> None:
    """Add one Scatter trace per channel with the given role to the
    indicated subplot row."""
    role_data = [d for d in data if d.get("channel_role") == role]
    channels = sorted(set(d["channel"] for d in role_data))
    for ch in channels:
        ch_data = [d for d in role_data if d["channel"] == ch]
        ch_name = ch_data[0].get("channel_name") or f"Ch{ch}"
        fig.add_trace(go.Scatter(
            x=[d["chunk_datetime"] for d in ch_data],
            y=[d["rms_amplitude"] for d in ch_data],
            mode="lines+markers",
            name=f"{ch_name} ({role})",
            line=dict(color=color), marker=dict(size=3),
        ), row=row, col=1)


def _add_line_noise_traces(fig: go.Figure, data: list[dict],
                              store: Store, session_dir: str) -> None:
    """Line-noise ratio panel (row 3) -- one trace per channel,
    coloured by role from the session config."""
    ch_map = channel_map(store, session_dir)
    channels = sorted(set(d["channel"] for d in data))
    for ch in channels:
        ch_data = [d for d in data if d["channel"] == ch]
        info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
        fig.add_trace(go.Scatter(
            x=[d["chunk_datetime"] for d in ch_data],
            y=[d["line_noise_ratio"] for d in ch_data],
            mode="lines", name=info["name"],
            line=dict(color=color_for_role(info["role"]), width=1),
            showlegend=False,
        ), row=3, col=1)
