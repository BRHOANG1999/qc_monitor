"""Evoked Features tab -- separate row per selected feature.

Layout: session dropdown + time-range dropdown + feature checklist
+ a stack of one plot per selected feature, plus a per-feature stats
summary block below.

The single callback rebuilds every plot + the stats block whenever
(features, session, hours) changes. Each plot is a 3-series scatter:
clean epochs (indigo), artifact-flagged epochs (red), ictal-flagged
epochs (orange).

Lifted out of ``src/dashboard/app.py``.
"""

from __future__ import annotations

import statistics
from datetime import datetime as _dt, timedelta as _td

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.dashboard.components import (
    DROPDOWN_STYLE, LABEL_STYLE, SECTION_STYLE,
)
from src.dashboard.data_helpers import (
    EVOKED_FEATURE_COLS, EVOKED_FEATURE_LABELS,
    TIME_RANGE_OPTIONS, session_dropdown_options,
)
from src.db.store import Store

# Open the tab with three commonly-used features pre-selected so the
# default render isn't empty. The user can pick anything from
# EVOKED_FEATURE_COLS; these are just the initial choice.
_DEFAULT_FEATURES = ["peak_amplitude", "line_length", "recovery_tau"]

# Features that aren't actually plottable as a numeric scatter (they
# tag epochs rather than describe them). Filtered out of the checklist.
_NON_PLOTTABLE = {"is_artifact", "is_ictal", "epoch_time_sec"}


def layout(store: Store):
    """Build the layout. Plots are populated by the callback below
    on initial render thanks to Dash firing it once for the default
    feature selection."""
    sessions = store.get_sessions()
    session_options = session_dropdown_options(store)
    plottable = [f for f in EVOKED_FEATURE_COLS
                  if f not in _NON_PLOTTABLE]
    feature_options = [
        {"label": EVOKED_FEATURE_LABELS.get(f, f), "value": f}
        for f in plottable
    ]
    default_session = (sessions[0]["session_dir"]
                        if sessions else None)

    return html.Div([
        html.H3("Evoked Features",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="evoked-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="evoked-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=0,    # all time by default
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "12px", "flexWrap": "wrap"}),

        html.Div([
            html.Label("Select features to display "
                       "(each gets its own plot row):",
                       style=LABEL_STYLE),
            dcc.Checklist(
                id="evoked-feature-checklist",
                options=feature_options,
                value=_DEFAULT_FEATURES,
                inline=True,
                style={"fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"color": "#ddd",
                             "marginRight": "16px",
                             "marginBottom": "4px"},
            ),
        ], style={**SECTION_STYLE, "marginBottom": "16px"}),

        # Hidden single-select for back-compat with any external
        # callback that may still reference the old id (kept from the
        # pre-checklist version of this tab; remove once verified
        # unreferenced).
        dcc.Dropdown(id="evoked-feature-dropdown",
                     value="peak_amplitude",
                     style={"display": "none"}),

        html.Div(id="evoked-multi-plots"),
        html.Div(id="evoked-stats"),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire (features, session, hours) -> (plots div, stats div)."""

    @app.callback(
        [Output("evoked-multi-plots", "children"),
         Output("evoked-stats", "children")],
        [Input("evoked-feature-checklist", "value"),
         Input("evoked-session-dropdown", "value"),
         Input("evoked-hours-dropdown", "value")],
    )
    def update_evoked_multi(selected_features, session_dir, hours):
        if not selected_features or not session_dir:
            return (html.P("Select features and a session",
                            style={"color": "#888"}),
                    "")

        hrs = int(hours) if hours else 0
        plots: list = []
        stats_rows: list[str] = []

        for feature_name in selected_features:
            plot, stat = _render_one_feature(
                store, feature_name, session_dir, hrs)
            plots.append(plot)
            if stat is not None:
                stats_rows.append(stat)

        summary = (
            html.Div([
                html.P(s, style={"color": "#aaa", "margin": "2px 0",
                                  "fontSize": "12px"})
                for s in stats_rows
            ]) if stats_rows else "")
        return html.Div(plots), summary


def _render_one_feature(store: Store, feature_name: str,
                          session_dir: str, hours: int):
    """Pull a single feature's time series, split into clean /
    artifact / ictal series, and return ``(plot_or_message,
    stats_line_or_None)``."""
    try:
        data = store.get_evoked_feature_timeseries(
            feature_name=feature_name,
            session_dir=session_dir,
            hours=hours,
        )
    except Exception as e:
        return (html.P(f"Error loading {feature_name}: {e}",
                        style={"color": "#ff6b6b"}),
                None)
    if not data:
        return (html.P(f"No data for {feature_name}",
                        style={"color": "#888"}),
                None)

    clean_t, clean_v = [], []
    artifact_t, artifact_v = [], []
    ictal_t, ictal_v = [], []
    all_values = []

    for d in data:
        v = d.get("value")
        if v is None:
            continue
        # Absolute epoch timestamp = chunk start + epoch_time_sec.
        # When epoch_time_sec is missing we fall back to file-level.
        t = _epoch_timestamp(d)
        all_values.append(v)
        if d.get("is_artifact", 0):
            artifact_t.append(t); artifact_v.append(v)
        elif d.get("is_ictal", 0):
            ictal_t.append(t); ictal_v.append(v)
        else:
            clean_t.append(t); clean_v.append(v)

    label = EVOKED_FEATURE_LABELS.get(feature_name, feature_name)
    fig = go.Figure()
    if clean_t:
        fig.add_trace(go.Scatter(
            x=clean_t, y=clean_v, mode="markers", name="Clean",
            marker=dict(color="#636EFA", size=4, opacity=0.6),
        ))
    if artifact_t:
        fig.add_trace(go.Scatter(
            x=artifact_t, y=artifact_v, mode="markers",
            name="Artifact",
            marker=dict(color="#EF553B", size=5, opacity=0.7),
        ))
    if ictal_t:
        fig.add_trace(go.Scatter(
            x=ictal_t, y=ictal_v, mode="markers", name="Ictal",
            marker=dict(color="#FFA15A", size=5, opacity=0.7),
        ))

    fig.update_layout(
        title=label, xaxis_title="Time", yaxis_title=label,
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )

    n_total = len(all_values)
    n_art = len(artifact_v)
    n_clean = len(clean_v)
    mean_val = statistics.mean(all_values) if all_values else 0
    std_val = (statistics.stdev(all_values)
               if len(all_values) > 1 else 0)
    stat_line = (
        f"{label}: mean={mean_val:.4g}, std={std_val:.4g}, "
        f"N={n_total} (clean={n_clean}, artifact={n_art})")
    return (dcc.Graph(figure=fig,
                       style={"marginBottom": "8px"}),
            stat_line)


def _epoch_timestamp(row: dict) -> str:
    """Recover an absolute timestamp for one epoch from its row.

    chunk_datetime is the file-level timestamp in KMrecorder's
    ``YYYY_MM_DD__HH_MM_SS`` format; epoch_time_sec is the offset
    inside the file. When parsing fails we fall back to the file
    timestamp so the point still plots (just not with epoch-level
    resolution).
    """
    epoch_sec = row.get("epoch_time_sec")
    chunk_dt = row.get("chunk_datetime", "")
    if epoch_sec is None or not chunk_dt:
        return chunk_dt
    try:
        base = _dt.fromisoformat(
            chunk_dt.replace("_", "-").replace("--", " ").replace(
                "__", "T"))
        return (base + _td(seconds=float(epoch_sec))).isoformat()
    except Exception:
        return chunk_dt
