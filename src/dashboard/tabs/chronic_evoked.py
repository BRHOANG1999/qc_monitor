"""Chronic Evoked Analyzer tab -- one animal's evoked response tracked
across ALL of its recordings, at **per-evoked-response resolution**.

For an implanted animal the full chronic record (back to the implant
baseline) lives in the MATLAB toolkit's ``evokedOutput`` folder, not in
the rolling monitor DB. This tab reads those per-recording
``*_evoked.mat`` files -- one point per stimulus -- via the cached
loader in ``src/utils/evoked_output.py`` (first read of an animal warms
a compact sqlite cache; later reads are instant).

Animal picker is the primary control. Plots: every evoked response over
the implant period (WebGL scatter, render-capped) plus a per-recording
mean trend line, with a 7d-vs-prior-7d drift note.
"""

from __future__ import annotations

import statistics
from collections import OrderedDict
from datetime import datetime

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import TIME_RANGE_OPTIONS, empty_fig
from src.utils.evoked_output import (
    ChronicEvokedCache, DEFAULT_EVOKED_DIR, DEFAULT_CACHE_DB, list_animals,
)

# Per-epoch metrics available directly from the evoked output (the raw
# ``*_evoked.mat`` only carries these scalars per stimulus).
_METRICS: list[tuple[str, str]] = [
    ("peak", "Peak amplitude"),
    ("trough", "Trough amplitude"),
    ("peak_to_trough", "Peak-to-trough"),
]
_METRIC_LABELS = dict(_METRICS)
_DEFAULT_METRIC = "peak"

# WebGL stays smooth to ~10^5 markers; above this we stride-sample the
# scatter (the per-recording mean line still shows the true trend).
_MAX_POINTS = 60000

_PALETTE = ["#5e7ce2", "#30d158", "#ff9f0a", "#bf5af2", "#ff453a",
            "#64d2ff", "#ffd60a", "#ff375f", "#a0a0b0", "#5ac8fa"]

# Configured in register_callbacks(); layout() reads them on tab open.
_EVOKED_DIR = DEFAULT_EVOKED_DIR
_CACHE_DB = DEFAULT_CACHE_DB
_cache_singleton: ChronicEvokedCache | None = None


def _cache() -> ChronicEvokedCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = ChronicEvokedCache(_EVOKED_DIR, _CACHE_DB)
    return _cache_singleton


def layout(store):
    """Pickers (animal, metric, range) + a refresh button + two plots,
    wrapped in dcc.Loading so a cache build never looks like a freeze."""
    animals = list_animals(_EVOKED_DIR)
    animal_options = [{"label": a, "value": a} for a in animals]
    default_animal = animals[0] if animals else None

    return html.Div([
        html.H3("Chronic Evoked Analyzer",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("Every evoked response for one animal across its whole "
                 "implant period, read from the toolkit's evokedOutput.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Animal", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-animal-dropdown",
                    options=animal_options, value=default_animal,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "180px"}),
            html.Div([
                html.Label("Metric", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-metric-dropdown",
                    options=[{"label": lbl, "value": col}
                             for col, lbl in _METRICS],
                    value=_DEFAULT_METRIC, clearable=False,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "200px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-hours-dropdown",
                    options=TIME_RANGE_OPTIONS, value=0,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "0 0 170px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("↻ Refresh from evokedOutput",
                            id="chronic-refresh-btn", n_clicks=0,
                            style={"width": "100%", "padding": "8px",
                                   "background": "#2a2d3a",
                                   "color": "#cfd0d6",
                                   "border": "1px solid #3a3d4a",
                                   "borderRadius": "6px",
                                   "cursor": "pointer"}),
            ], style={"flex": "0 0 200px"}),
        ], style={"display": "flex", "gap": "14px",
                  "marginBottom": "10px", "flexWrap": "wrap"}),

        html.Div(id="chronic-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px", "marginBottom": "4px"}),
        html.Div(id="chronic-drift-note",
                 style={"color": "#cfd0d6", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "8px"}),
        dcc.Loading(type="default", color="#5e7ce2", children=[
            dcc.Graph(id="chronic-timeseries-plot"),
            dcc.Graph(id="chronic-session-agg-plot"),
        ]),
    ])


def register_callbacks(app, store, config: dict) -> None:
    """Wire pickers + refresh -> (per-response scatter, per-recording
    mean line, drift note, status). Reads the configured evokedOutput
    dir from ``config.chronic_evoked``."""
    global _EVOKED_DIR, _CACHE_DB
    ce_cfg = (config or {}).get("chronic_evoked", {}) or {}
    _EVOKED_DIR = ce_cfg.get("evoked_output_dir") or DEFAULT_EVOKED_DIR
    _CACHE_DB = ce_cfg.get("cache_db") or DEFAULT_CACHE_DB

    @app.callback(
        Output("chronic-timeseries-plot", "figure"),
        Output("chronic-session-agg-plot", "figure"),
        Output("chronic-drift-note", "children"),
        Output("chronic-status", "children"),
        Input("chronic-animal-dropdown", "value"),
        Input("chronic-metric-dropdown", "value"),
        Input("chronic-hours-dropdown", "value"),
        Input("chronic-refresh-btn", "n_clicks"),
    )
    def _update(animal, metric, hours, _clicks):
        if not animal:
            return (empty_fig("Select an animal"),
                    empty_fig("Select an animal"), "", "")
        metric = metric if metric in _METRIC_LABELS else _DEFAULT_METRIC
        try:
            cache = _cache()
            stats = cache.ensure_animal(animal)
            rows = cache.query(animal, hours=(hours or None))
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            return (empty_fig("Couldn't read evokedOutput", hint=str(e)),
                    empty_fig("Couldn't read evokedOutput"), "",
                    f"Error: {e}")
        status = (f"{stats['files']} recordings cached "
                  f"({stats['built']} (re)built this load) · "
                  f"{len(rows)} evoked responses in range.")
        if not rows:
            msg = f"No evoked responses for {animal} in range"
            return empty_fig(msg), empty_fig(msg), "", status
        return (_build_timeseries(rows, metric, animal),
                _build_recording_agg(rows, metric),
                _drift_note(rows, metric), status)


# --------------------------------------------------------------------- #
#  Builders (pure)
# --------------------------------------------------------------------- #

def _by_channel(rows):
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        grouped.setdefault(r.get("channel") or "?", []).append(r)
    return grouped


def _stride(seq, cap):
    """Uniform stride-sample *seq* down to at most *cap* items."""
    n = len(seq)
    if n <= cap or cap <= 0:
        return seq, 1
    step = (n // cap) + 1
    return seq[::step], step


def _build_timeseries(rows, metric, animal) -> go.Figure:
    label = _METRIC_LABELS.get(metric, metric)
    fig = go.Figure()
    sampled = False
    for i, (chan, crows) in enumerate(_by_channel(rows).items()):
        pts = [(r.get("abs_dt"), r.get(metric)) for r in crows
               if r.get(metric) is not None and r.get("abs_dt")]
        kept, step = _stride(pts, _MAX_POINTS // max(1, len(_by_channel(rows))))
        if step > 1:
            sampled = True
        if not kept:
            continue
        fig.add_trace(go.Scattergl(
            x=[p[0] for p in kept], y=[float(p[1]) for p in kept],
            mode="markers", name=chan,
            marker=dict(size=4, color=_PALETTE[i % len(_PALETTE)],
                        opacity=0.55)))
    note = " (stride-sampled for display)" if sampled else ""
    fig.update_layout(
        title=f"{label} per evoked response — {animal}{note}",
        xaxis_title="Recording time", yaxis_title=label,
        height=430, hovermode="closest", legend=dict(font=dict(size=10)))
    return fig


def _build_recording_agg(rows, metric) -> go.Figure:
    """Per-recording mean of *metric* over time -- the chronic trend."""
    label = _METRIC_LABELS.get(metric, metric)
    fig = go.Figure()
    for i, (chan, crows) in enumerate(_by_channel(rows).items()):
        per_rec: "OrderedDict[str, list]" = OrderedDict()
        for r in crows:
            v = r.get(metric)
            if v is not None and r.get("rec_dt"):
                per_rec.setdefault(r["rec_dt"], []).append(float(v))
        xs = sorted(per_rec.keys())
        ys = [statistics.mean(per_rec[x]) for x in xs]
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=chan,
            line=dict(color=_PALETTE[i % len(_PALETTE)]),
            marker=dict(size=5)))
    fig.update_layout(
        title=f"{label} — per-recording mean over time",
        xaxis_title="Recording", yaxis_title=label, height=360,
        legend=dict(font=dict(size=10)))
    return fig


def _drift_note(rows, metric) -> str:
    """Median of the last 7 days vs the prior 7 days, as a % shift.
    Empty when fewer than 3 recordings on either side."""
    label = _METRIC_LABELS.get(metric, metric)
    pts = []
    for r in rows:
        v = r.get(metric)
        dt = _parse_iso(r.get("abs_dt"))
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


def _parse_iso(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
