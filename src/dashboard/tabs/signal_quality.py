"""Signal Quality tab -- 48 h time series of RMS, artifact %, and
line-noise ratio for every channel of the most recent session.

Read-only, no callbacks. The whole figure is built server-side once
per tab open from a single ``store.get_qc_timeseries`` call. Each
channel is one Scatter trace per subplot; the legend is grouped so
toggling a channel hides all three rows at once.

Lifted out of ``src/dashboard/app.py`` following the per-tab pattern
already in place for ``tabs/alerts``, ``tabs/video``, ``tabs/surgeries``,
``tabs/maintenance``, ``tabs/data_log_xref``, ``tabs/event_verification``.
The tab knows about the shared design tokens + the
``data_helpers.channel_map`` / ``color_for_role`` plumbing; it has no
upward dependency on the app shell.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from src.dashboard.data_helpers import channel_map, color_for_role
from src.db.store import Store


def layout(store: Store):
    """Build the Signal Quality tab. Returns a Div containing one
    ``dcc.Graph`` with three stacked subplots."""
    sessions = store.get_sessions()
    if not sessions:
        return html.Div("No sessions found.", style={"color": "#888"})

    session_dir = sessions[0].get("session_dir", "")
    ch_map = channel_map(store, session_dir)

    data = store.get_qc_timeseries(session_dir=session_dir, hours=48)
    if not data:
        return html.Div("No QC data yet.", style={"color": "#888"})

    channels = sorted(set(d["channel"] for d in data))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["RMS Amplitude", "Artifact %",
                         "Line Noise Ratio"],
        vertical_spacing=0.08,
    )

    for ch in channels:
        ch_data = [d for d in data if d["channel"] == ch]
        ch_times = [d["chunk_datetime"] for d in ch_data]

        info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
        ch_name = info["name"]
        color = color_for_role(info["role"])

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["rms_amplitude"] for d in ch_data],
            name=ch_name, legendgroup=ch_name,
            line=dict(color=color), marker=dict(size=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["artifact_pct"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["line_noise_ratio"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=3, col=1)

    fig.update_layout(
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    fig.update_annotations(font=dict(color="white"))

    return html.Div([dcc.Graph(figure=fig)])


def register_callbacks(app, store: Store, config: dict) -> None:
    """No callbacks on this tab. Kept for symmetry with the other
    tab modules."""
    return
