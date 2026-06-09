"""Session Compare tab -- two-session side-by-side comparison of the
latest mean evoked waveform and the top-5 feature distributions.

Layout: two session-picker dropdowns + a smoothing-input + two
``dcc.Graph`` panes (waveform overlay, feature bar comparison).
Single callback rebuilds both figures on any input change.

Both panes' second-axis distributions are read once per change; the
inner ``store.get_evoked_feature_timeseries`` call is cheap because
``evoked_features`` is indexed on ``(file_id, epoch_index)`` so the
join is the slow part, not the value scan.

Lifted out of ``src/dashboard/app.py``. Imports the shared form-control
styles + evoked-feature constants + ``empty_fig`` + ``apply_filter``
(for optional Gaussian smoothing of the displayed waveform).
"""

from __future__ import annotations

import logging
import statistics

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import (
    EVOKED_FEATURE_LABELS, empty_fig, session_dropdown_options,
)
from src.db.store import Store
from src.utils.filters import apply_filter

logger = logging.getLogger("qc_monitor.dashboard.session_compare")

# Two sessions, two distinct colors. Matches the original app.py
# overlay coloring (Plotly indigo + Plotly orange-red).
_SESSION_A_COLOR = "#636EFA"
_SESSION_B_COLOR = "#EF553B"

# Bar chart compares the top-5 evoked features. Frozen list rather
# than user-configurable to keep the layout / callback shape simple;
# tabs/evoked covers the full picker.
_TOP_FEATURES = [
    "peak_amplitude", "trough_amplitude", "rms_amplitude",
    "line_length", "peak_to_trough",
]


def layout(store: Store):
    """Layout only -- the two figures are built server-side by the
    callback registered below."""
    session_options = session_dropdown_options(store)
    return html.Div([
        html.H3("Session Compare",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session A", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-a-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Session B", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-b-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(
                    id="compare-smooth", type="number", min=0,
                    step=0.5, value=0,
                    style={"backgroundColor": "#262638",
                           "color": "#f0f0f5", "width": "80px"},
                ),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),
        dcc.Graph(id="session-compare-waveform-plot",
                  style={"height": "450px"}),
        dcc.Graph(id="session-compare-features-plot",
                  style={"height": "450px", "marginTop": "16px"}),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the (session_a, session_b, smooth_ms) -> (waveform,
    features) callback. Same ids as the layout above."""

    @app.callback(
        [Output("session-compare-waveform-plot", "figure"),
         Output("session-compare-features-plot", "figure")],
        [Input("compare-session-a-dropdown", "value"),
         Input("compare-session-b-dropdown", "value"),
         Input("compare-smooth", "value")],
    )
    def update_session_compare(session_a, session_b, smooth_ms):
        if not session_a or not session_b:
            return (empty_fig("Select two sessions", height=450),
                    empty_fig("Select two sessions", height=450))

        smooth_ms = float(smooth_ms or 0)
        wf_fig = _build_waveform_overlay(store, session_a, session_b,
                                          smooth_ms)
        feat_fig = _build_feature_comparison(store, session_a,
                                              session_b)
        return wf_fig, feat_fig


def _build_waveform_overlay(store: Store, session_a: str,
                             session_b: str, smooth_ms: float
                             ) -> go.Figure:
    fig = go.Figure()
    for sess_dir, color, label in [
            (session_a, _SESSION_A_COLOR, "Session A"),
            (session_b, _SESSION_B_COLOR, "Session B")]:
        _add_waveform_traces(fig, store, sess_dir, color, label,
                              smooth_ms)
    fig.add_vline(x=0, line=dict(color="white", width=1,
                                  dash="dash"))
    fig.update_layout(
        title="Mean Evoked Waveform Overlay",
        xaxis_title="Time (ms)", yaxis_title="Amplitude",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    return fig


def _add_waveform_traces(fig: go.Figure, store: Store,
                          sess_dir: str, color: str, label: str,
                          smooth_ms: float) -> None:
    try:
        waveforms = store.get_evoked_waveforms_for_session(sess_dir)
    except Exception:
        waveforms = []
    if not waveforms:
        return

    wf = waveforms[-1]
    time_ms = wf["time_axis_ms"]
    mean_tr = wf["mean_trace"]
    sem_tr = wf["sem_trace"]

    # Optional Gaussian smoothing in display-ms. apply_filter wants
    # a sample rate, so we recover one from the first two time-axis
    # samples -- the source mat is always uniformly sampled.
    if smooth_ms > 0 and mean_tr and len(time_ms) > 1:
        try:
            dt_ms = float(time_ms[1] - time_ms[0])
            if dt_ms > 0:
                fs_proxy = 1000.0 / dt_ms
                mean_tr = apply_filter(
                    np.asarray(mean_tr, dtype=np.float32),
                    fs_proxy, smoothing_ms=smooth_ms,
                ).tolist()
                if sem_tr and len(sem_tr) == len(mean_tr):
                    sem_tr = apply_filter(
                        np.asarray(sem_tr, dtype=np.float32),
                        fs_proxy, smoothing_ms=smooth_ms,
                    ).tolist()
        except Exception as e:
            logger.debug("Compare smoothing skipped: %s", e)

    sname = _resolve_session_name(store, sess_dir)

    if sem_tr and len(sem_tr) == len(mean_tr):
        upper = [m + s for m, s in zip(mean_tr, sem_tr)]
        lower = [m - s for m, s in zip(mean_tr, sem_tr)]
        # Hardcoded translucent fills matching _SESSION_A_COLOR /
        # _SESSION_B_COLOR. Translating arbitrary hex -> rgba is
        # avoided because we only ever overlay these two sessions.
        fill_color = ("rgba(99,110,250,0.15)"
                       if color == _SESSION_A_COLOR
                       else "rgba(239,85,59,0.15)")
        fig.add_trace(go.Scatter(
            x=list(time_ms) + list(reversed(time_ms)),
            y=upper + list(reversed(lower)),
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=time_ms, y=mean_tr, mode="lines",
        name=f"{label}: {sname}",
        line=dict(color=color, width=2),
    ))


def _build_feature_comparison(store: Store, session_a: str,
                                session_b: str) -> go.Figure:
    fig = go.Figure()
    for sess_dir, color, label in [
            (session_a, _SESSION_A_COLOR, "Session A"),
            (session_b, _SESSION_B_COLOR, "Session B")]:
        means, stds, f_names = _feature_stats_for_session(store,
                                                            sess_dir)
        sname = _resolve_session_name(store, sess_dir)
        fig.add_trace(go.Bar(
            x=f_names, y=means,
            name=f"{label}: {sname}",
            marker_color=color,
            error_y=dict(type="data", array=stds, visible=True),
        ))
    fig.update_layout(
        title="Feature Distribution Comparison (top 5)",
        xaxis_title="Feature", yaxis_title="Value",
        barmode="group", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    return fig


def _feature_stats_for_session(store: Store, sess_dir: str
                                 ) -> tuple[list[float],
                                              list[float],
                                              list[str]]:
    means: list[float] = []
    stds: list[float] = []
    f_names: list[str] = []
    for feat in _TOP_FEATURES:
        try:
            fdata = store.get_evoked_feature_timeseries(
                feature_name=feat, session_dir=sess_dir)
            # Drop artifact epochs (is_artifact=1) so the comparison
            # is over real evoked responses, not bad detections.
            vals = [d["value"] for d in fdata
                    if d["value"] is not None
                    and not d.get("is_artifact", 0)]
        except Exception:
            vals = []
        f_names.append(EVOKED_FEATURE_LABELS.get(feat, feat))
        if vals:
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1
                          else 0)
        else:
            means.append(0)
            stds.append(0)
    return means, stds, f_names


def _resolve_session_name(store: Store, sess_dir: str) -> str:
    """Look up the human-readable name for a session_dir. Falls
    back to the dir itself when no row matches (which only happens
    if the picker holds a stale value)."""
    for s in store.get_sessions():
        if s["session_dir"] == sess_dir:
            return s["session_name"]
    return sess_dir
