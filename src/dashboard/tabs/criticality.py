"""Criticality tab -- dB time series for each EEG channel of one
session, over a user-selectable time window.

Read on render, recomputed on every change to the session-dropdown or
the time-range dropdown via the ``update_criticality`` callback below.
Non-EEG channels are filtered out at the plotting layer; the
underlying ``store.query_criticality_timeseries`` call doesn't filter
by role so the cache stays warm for any tab that later wants the
reference / stim_copy series too.

Lifted out of ``src/dashboard/app.py``. Imports only from the shared
``components`` (styles), ``data_helpers`` (figure builder + channel
map + role color), and ``src.db.store`` -- no upward dependency on
the app shell.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.dashboard.components import (
    DROPDOWN_STYLE, LABEL_STYLE,
)
from src.dashboard.data_helpers import (
    TIME_RANGE_OPTIONS, channel_map, color_for_role, empty_fig,
)
from src.db.store import Store


def layout(store: Store):
    """Build the tab layout. Data is loaded server-side via the
    callback below; the layout only places the two dropdowns + the
    placeholder Graph."""
    sessions = store.get_sessions()
    session_options = [
        {"label": s["session_name"], "value": s["session_dir"]}
        for s in sessions
    ]
    default_session = (sessions[0]["session_dir"]
                       if sessions else None)

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),
        dcc.Graph(id="criticality-plot",
                  style={"height": "550px"}),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the (session, time-range) -> figure callback. Mirrors
    the layout's dropdown ids."""

    @app.callback(
        Output("criticality-plot", "figure"),
        [Input("criticality-session-dropdown", "value"),
         Input("criticality-hours-dropdown", "value")],
    )
    def update_criticality(session_dir, hours):
        if not session_dir:
            return empty_fig("Select a session", height=550)

        ch_map = channel_map(store, session_dir)

        try:
            data = store.query_criticality_timeseries(
                session_dir=session_dir,
                hours=int(hours) if hours else None,
            )
        except Exception as e:
            return empty_fig(f"Error: {e}", height=550)

        if not data:
            return empty_fig("No criticality data", height=550)

        channels = sorted(set(d["channel"] for d in data))

        fig = go.Figure()
        for ch in channels:
            ch_data = [d for d in data if d["channel"] == ch]
            times = [d["chunk_datetime"] for d in ch_data]
            vals = [d["db_value"] for d in ch_data]

            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            # If session_config gave us a channel map at all, restrict
            # the plot to EEG channels. Without a map (legacy chunks
            # pre-auto-discovery) every channel is plotted under its
            # numeric name.
            if ch_map and info["role"] != "eeg":
                continue

            fig.add_trace(go.Scatter(
                x=times, y=vals, mode="lines+markers",
                name=info["name"],
                line=dict(color=color_for_role(info["role"])),
                marker=dict(size=3),
            ))

        fig.update_layout(
            title="Criticality (dB) Over Time -- EEG Channels",
            xaxis_title="Time",
            yaxis_title="dB Value",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                         xanchor="right", x=1),
        )
        return fig
