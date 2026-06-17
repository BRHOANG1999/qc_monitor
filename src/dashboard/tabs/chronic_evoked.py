"""Chronic Evoked Analyzer tab -- one animal's evoked metric tracked
across ALL its recordings over weeks/months (the "chronic" timescale).

Fills the gap between the existing evoked tabs:
  * ``tabs/evoked.py``         -- per-session, per-epoch features.
  * ``tabs/session_compare.py`` -- overlays two sessions.

Reads the already-stored per-file ``evoked_summary`` metrics (no MATLAB,
no recompute) via ``store.query_evoked_summary_for_animal``. Animal picker
is the primary control. Ports the longitudinal view of the external
STiM-NET "Chronic" tab (see docs/chronic_evoked_reference.md).
"""

from __future__ import annotations

import statistics
from collections import OrderedDict
from datetime import datetime

import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import TIME_RANGE_OPTIONS, empty_fig
from src.db.store import Store

# Plottable per-file evoked_summary columns -> friendly labels. Distinct
# from EVOKED_FEATURE_LABELS (which describes the per-epoch table).
_SUMMARY_METRICS: list[tuple[str, str]] = [
    ("mean_peak_amplitude", "Mean peak amplitude"),
    ("std_peak_amplitude", "Peak amplitude SD"),
    ("mean_trough_amplitude", "Mean trough amplitude"),
    ("mean_peak_latency_ms", "Mean peak latency (ms)"),
    ("mean_line_length", "Mean line length"),
    ("mean_recovery_tau", "Mean recovery tau"),
    ("mean_template_correlation", "Mean template correlation"),
    ("num_stimuli_detected", "Stimuli detected"),
    ("num_traces_extracted", "Traces extracted"),
]
_METRIC_LABELS = dict(_SUMMARY_METRICS)
_DEFAULT_METRIC = "mean_peak_amplitude"

# Qualitative palette cycled per session.
_PALETTE = ["#5e7ce2", "#30d158", "#ff9f0a", "#bf5af2", "#ff453a",
            "#64d2ff", "#ffd60a", "#ff375f", "#a0a0b0", "#5ac8fa"]


def layout(store: Store):
    """Pickers (animal, metric, time range) + two plots + a drift note.
    Non-empty defaults so Dash renders on mount (like tabs/evoked.py)."""
    animals = store.list_all_animals()
    animal_options = [{"label": a, "value": a} for a in animals]
    default_animal = animals[0] if animals else None

    return html.Div([
        html.H3("Chronic Evoked Analyzer",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("Track one animal's evoked response across all of its "
                 "recordings over the chronic implant period.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Animal", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-animal-dropdown",
                    options=animal_options, value=default_animal,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Div([
                html.Label("Metric", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-metric-dropdown",
                    options=[{"label": lbl, "value": col}
                             for col, lbl in _SUMMARY_METRICS],
                    value=_DEFAULT_METRIC, clearable=False,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "220px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-hours-dropdown",
                    options=TIME_RANGE_OPTIONS, value=0,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "12px", "flexWrap": "wrap"}),

        html.Div(id="chronic-drift-note",
                 style={"color": "#cfd0d6", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "8px"}),
        dcc.Graph(id="chronic-timeseries-plot"),
        dcc.Graph(id="chronic-session-agg-plot"),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire (animal, metric, hours) -> (timeline, per-session bars, note)."""

    @app.callback(
        Output("chronic-timeseries-plot", "figure"),
        Output("chronic-session-agg-plot", "figure"),
        Output("chronic-drift-note", "children"),
        Input("chronic-animal-dropdown", "value"),
        Input("chronic-metric-dropdown", "value"),
        Input("chronic-hours-dropdown", "value"),
    )
    def _update(animal, metric, hours):
        if not animal:
            return (empty_fig("Select an animal"),
                    empty_fig("Select an animal"), "")
        metric = metric or _DEFAULT_METRIC
        try:
            rows = store.query_evoked_summary_for_animal(
                animal, hours=(hours or None))
        except Exception as e:
            return (empty_fig("Couldn't load evoked summaries",
                              hint=str(e)),
                    empty_fig("Couldn't load evoked summaries"), "")
        if not rows:
            msg = f"No evoked data for {animal}"
            return empty_fig(msg), empty_fig(msg), ""
        return (_build_timeseries(rows, metric, animal),
                _build_session_agg(rows, metric),
                _drift_note(rows, metric))


# --------------------------------------------------------------------- #
#  Builders (pure)
# --------------------------------------------------------------------- #

def _parse_dt(s) -> datetime | None:
    """Parse a file timestamp: ISO first, then KMrecorder
    ``YYYY_MM_DD__HH_MM_SS``. None when neither parses."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        pass
    try:
        if "__" in s:
            d, t = s.split("__", 1)
            return datetime.fromisoformat(
                d.replace("_", "-") + " " + t.replace("_", ":"))
    except (ValueError, TypeError):
        pass
    return None


def _by_session(rows):
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        grouped.setdefault(r.get("session_name") or "?", []).append(r)
    return grouped


def _build_timeseries(rows, metric, animal) -> go.Figure:
    label = _METRIC_LABELS.get(metric, metric)
    fig = go.Figure()
    for i, (sess, srows) in enumerate(_by_session(rows).items()):
        xs, ys, hov = [], [], []
        for r in srows:
            v = r.get(metric)
            dt = _parse_dt(r.get("chunk_datetime"))
            if v is None or dt is None:
                continue
            xs.append(dt)
            ys.append(float(v))
            hov.append(f"{sess}<br>{dt:%Y-%m-%d %H:%M}<br>"
                       f"{label}: {float(v):.4g}")
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=sess,
            line=dict(color=_PALETTE[i % len(_PALETTE)]),
            marker=dict(size=6), hovertext=hov, hoverinfo="text"))
    fig.update_layout(
        title=f"{label} over time — {animal}",
        xaxis_title="Recording time", yaxis_title=label,
        height=420, hovermode="closest",
        legend=dict(font=dict(size=10)))
    return fig


def _build_session_agg(rows, metric) -> go.Figure:
    label = _METRIC_LABELS.get(metric, metric)
    names, means, stds = [], [], []
    for sess, srows in _by_session(rows).items():
        vals = [float(r[metric]) for r in srows
                if r.get(metric) is not None]
        if not vals:
            continue
        names.append(sess)
        means.append(statistics.mean(vals))
        stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    if not names:
        return empty_fig("No values to aggregate")
    fig = go.Figure(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker=dict(color="#5e7ce2")))
    fig.update_layout(
        title=f"{label} — per-session mean ± SD",
        xaxis_title="Session", yaxis_title=label, height=360)
    return fig


def _drift_note(rows, metric) -> str:
    """Median of the last 7 days vs the prior 7 days, as a % shift.
    Empty when fewer than 3 points on either side."""
    label = _METRIC_LABELS.get(metric, metric)
    pts = []
    for r in rows:
        v = r.get(metric)
        dt = _parse_dt(r.get("chunk_datetime"))
        if v is not None and dt is not None:
            pts.append((dt, float(v)))
    if len(pts) < 6:
        return ""
    last = pts[-1][0]
    recent = [v for dt, v in pts if (last - dt).days <= 7]
    prior = [v for dt, v in pts if 7 < (last - dt).days <= 14]
    if len(recent) < 3 or len(prior) < 3:
        return ""
    r_med, p_med = statistics.median(recent), statistics.median(prior)
    if p_med == 0:
        return ""
    pct = (r_med - p_med) / abs(p_med) * 100.0
    arrow = "▲" if pct > 0 else "▼"
    return (f"{label}: last 7 days median {r_med:.3g} vs prior 7 days "
            f"{p_med:.3g}  ({arrow} {abs(pct):.0f}%).")
