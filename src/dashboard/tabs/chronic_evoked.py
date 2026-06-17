"""Chronic Evoked Analyzer tab -- a faithful port of the MATLAB toolkit's
"Chronic Evoked Features" tab.

For one implanted animal it plots the full evoked-feature set (computed
from the raw ``evokedData`` traces, cached in ``evoked_output``) per
stimulus across the whole implant period: a rainbow time-colored scatter
with trend fits, a per-recording feature trend, and a trend-stat strip.
(Waveform overlay, stim-evoked correlation, windowed scroll, circadian, and
a per-recording table arrive in later phases.)

Animal picker is the primary control. The cheap features are always
available; the heavy ones (recovery tau, template correlation, ...) appear
only when ``chronic_evoked.compute_expensive`` is set and the cache warmed
with ``--expensive``.
"""

from __future__ import annotations

import statistics
from collections import OrderedDict
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE
from src.dashboard.data_helpers import (
    EVOKED_FEATURE_LABELS, TIME_RANGE_OPTIONS, empty_fig)
from src.utils import evoked_features as ef
from src.utils.evoked_output import (
    ChronicEvokedCache, DEFAULT_EVOKED_DIR, DEFAULT_CACHE_DB, list_animals)

# Feature dropdown = every cached evoked feature, friendly labels from the
# app-wide map. Expensive ones are flagged so the UI can disable them.
_FEATURE_COLS = list(ef.ALL_COLUMNS)
_EXPENSIVE = set(ef.EXPENSIVE_COLUMNS)
_DEFAULT_FEATURE = "line_length"

# WebGL stays smooth to ~10^5 markers; above this we stride-sample.
_MAX_POINTS = 60000
_MA_WINDOW = 50            # per-animal moving-average window (epochs).

# Configured in register_callbacks(); layout() reads them on tab open.
_EVOKED_DIR = DEFAULT_EVOKED_DIR
_CACHE_DB = DEFAULT_CACHE_DB
_EXPENSIVE_ENABLED = False
_cache_singleton: ChronicEvokedCache | None = None


def _label(col: str) -> str:
    return EVOKED_FEATURE_LABELS.get(col, col.replace("_", " ").title())


def _cache() -> ChronicEvokedCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = ChronicEvokedCache(
            _EVOKED_DIR, _CACHE_DB, compute_expensive=_EXPENSIVE_ENABLED)
    return _cache_singleton


def _feature_options() -> list[dict]:
    opts = []
    for col in _FEATURE_COLS:
        disabled = (col in _EXPENSIVE) and not _EXPENSIVE_ENABLED
        suffix = "  (warm --expensive)" if disabled else ""
        opts.append({"label": _label(col) + suffix, "value": col,
                     "disabled": disabled})
    return opts


def layout(store):
    """Pickers + trend toggles + two plots, wrapped in dcc.Loading."""
    animals = list_animals(_EVOKED_DIR)
    default_animal = animals[0] if animals else None
    return html.Div([
        html.H3("Chronic Evoked Analyzer",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("One animal's evoked features across the whole implant "
                 "period, computed from the raw traces in evokedOutput.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Animal", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-animal-dropdown",
                    options=[{"label": a, "value": a} for a in animals],
                    value=default_animal, style=DROPDOWN_STYLE,
                    className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "170px"}),
            html.Div([
                html.Label("Feature", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-feature-dropdown", options=_feature_options(),
                    value=_DEFAULT_FEATURE, clearable=False,
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "220px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-hours-dropdown", options=TIME_RANGE_OPTIONS,
                    value=0, style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "0 0 160px"}),
            html.Div([
                html.Label("Trend", style=LABEL_STYLE),
                dcc.Checklist(
                    id="chronic-trend-toggles",
                    options=[{"label": " Linear", "value": "lin"},
                             {"label": " Quad", "value": "quad"},
                             {"label": " Moving avg", "value": "ma"}],
                    value=["lin"], inline=True,
                    style={"color": "#cfd0d6", "fontSize": "12px"},
                    inputStyle={"marginRight": "3px", "marginLeft": "8px"}),
            ], style={"flex": "0 0 240px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("↻ Refresh", id="chronic-refresh-btn",
                            n_clicks=0, style=_BTN_STYLE),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "10px",
                  "flexWrap": "wrap"}),

        html.Div(id="chronic-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px"}),
        html.Div(id="chronic-trend-stats",
                 style={"color": "#cfd0d6", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "6px"}),
        dcc.Loading(type="default", color="#5e7ce2", children=[
            dcc.Graph(id="chronic-feature-plot"),
            dcc.Graph(id="chronic-recording-plot"),
        ]),
    ])


_BTN_STYLE = {"width": "100%", "padding": "8px", "background": "#2a2d3a",
              "color": "#cfd0d6", "border": "1px solid #3a3d4a",
              "borderRadius": "6px", "cursor": "pointer"}


def register_callbacks(app, store, config: dict) -> None:
    """Wire pickers + trend toggles -> (rainbow scatter, per-recording trend,
    stats, status). Reads ``config.chronic_evoked``."""
    global _EVOKED_DIR, _CACHE_DB, _EXPENSIVE_ENABLED
    ce = (config or {}).get("chronic_evoked", {}) or {}
    _EVOKED_DIR = ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR
    _CACHE_DB = ce.get("cache_db") or DEFAULT_CACHE_DB
    _EXPENSIVE_ENABLED = bool(ce.get("compute_expensive", False))

    @app.callback(
        Output("chronic-feature-plot", "figure"),
        Output("chronic-recording-plot", "figure"),
        Output("chronic-trend-stats", "children"),
        Output("chronic-status", "children"),
        Input("chronic-animal-dropdown", "value"),
        Input("chronic-feature-dropdown", "value"),
        Input("chronic-hours-dropdown", "value"),
        Input("chronic-trend-toggles", "value"),
        Input("chronic-refresh-btn", "n_clicks"),
    )
    def _update(animal, feature, hours, overlays, _clicks):
        if not animal:
            return empty_fig("Select an animal"), empty_fig(""), "", ""
        feature = feature if feature in _FEATURE_COLS else _DEFAULT_FEATURE
        try:
            cache = _cache()
            stats = cache.ensure_animal(animal)
            rows = cache.query(animal, hours=(hours or None))
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            return (empty_fig("Couldn't read evokedOutput", hint=str(e)),
                    empty_fig(""), "", f"Error: {e}")
        pts = _feature_points(rows, feature)
        status = (f"{stats['files']} recordings cached "
                  f"({stats['built']} (re)built) · {len(rows)} responses "
                  f"· {len(pts[0])} with {_label(feature)}.")
        if not pts[0]:
            msg = f"No {_label(feature)} for {animal} in range"
            return empty_fig(msg), empty_fig(""), "", status
        return (_build_feature_scatter(pts, feature, animal, overlays or []),
                _build_recording_trend(rows, feature),
                _trend_stats(pts, feature), status)


# --------------------------------------------------------------------- #
#  Data shaping + builders (pure)
# --------------------------------------------------------------------- #

def _feature_points(rows, feature):
    """(iso_times, seconds, values) for rows with a finite feature value."""
    iso, secs, vals = [], [], []
    for r in rows:
        v = r.get(feature)
        dt = _parse_iso(r.get("abs_dt"))
        if v is None or dt is None:
            continue
        iso.append(r["abs_dt"])
        secs.append(dt.timestamp())
        vals.append(float(v))
    return iso, np.asarray(secs, float), np.asarray(vals, float)


def _stride(n, cap):
    if n <= cap or cap <= 0:
        return 1
    return (n // cap) + 1


def _build_feature_scatter(pts, feature, animal, overlays) -> go.Figure:
    iso, secs, vals = pts
    label = _label(feature)
    step = _stride(len(secs), _MAX_POINTS)
    color = _norm(secs)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=iso[::step], y=vals[::step], mode="markers",
        marker=dict(size=4, color=color[::step], colorscale="Turbo",
                    opacity=0.6, colorbar=dict(title="time", thickness=10)),
        name="responses", hoverinfo="x+y"))
    _add_trend_overlays(fig, iso, secs, vals, overlays)
    note = " (stride-sampled)" if step > 1 else ""
    fig.update_layout(
        title=f"{label} per evoked response — {animal}{note}",
        xaxis_title="Recording time", yaxis_title=label,
        height=430, hovermode="closest", showlegend=True,
        legend=dict(font=dict(size=10)))
    return fig


def _add_trend_overlays(fig, iso, secs, vals, overlays) -> None:
    if secs.size < 3:
        return
    order = np.argsort(secs)
    xs = np.asarray(iso)[order]
    t = secs[order] - secs[order][0]
    y = vals[order]
    if "lin" in overlays:
        _add_polyline(fig, xs, t, y, 1, "#ffffff", "linear")
    if "quad" in overlays and t.size >= 3:
        _add_polyline(fig, xs, t, y, 2, "#ffd60a", "quadratic")
    if "ma" in overlays:
        ma = ef.rolling_centered(y, _MA_WINDOW, "mean")
        fig.add_trace(go.Scattergl(x=xs, y=ma, mode="lines",
                                   line=dict(color="#30d158", width=2),
                                   name=f"moving avg ({_MA_WINDOW})"))


def _add_polyline(fig, xs, t, y, deg, color, name) -> None:
    try:
        coef = np.polyfit(t, y, deg)
    except (np.linalg.LinAlgError, ValueError):
        return
    fig.add_trace(go.Scattergl(
        x=xs, y=np.polyval(coef, t), mode="lines",
        line=dict(color=color, width=2, dash="solid" if deg == 1 else "dot"),
        name=name))


def _build_recording_trend(rows, feature) -> go.Figure:
    """Per-recording mean of the feature over time -- the chronic trend."""
    label = _label(feature)
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        v = r.get(feature)
        if v is not None and r.get("rec_dt"):
            grouped.setdefault(r["rec_dt"], []).append(float(v))
    xs = sorted(grouped)
    ys = [statistics.mean(grouped[x]) for x in xs]
    if not xs:
        return empty_fig("No per-recording values")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers",
                               line=dict(color="#5e7ce2"),
                               marker=dict(size=5)))
    fig.update_layout(title=f"{label} — per-recording mean over time",
                      xaxis_title="Recording", yaxis_title=label, height=320)
    return fig


def _trend_stats(pts, feature) -> str:
    """Linear slope / r / p + mean/std/CV/range over the feature series."""
    _, secs, vals = pts
    if vals.size < 3:
        return ""
    from scipy.stats import linregress
    days = (secs - secs.min()) / 86400.0
    lr = linregress(days, vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    cv = (std / mean) if mean != 0 else float("nan")
    rng = float(np.max(vals) - np.min(vals))
    return (f"slope {lr.slope:.3g}/day · r {lr.rvalue:.2f} · p {lr.pvalue:.1e}"
            f"  |  mean {mean:.3g} · SD {std:.3g} · CV {cv:.2f} · "
            f"range {rng:.3g} · n {vals.size}")


def _norm(x):
    if x.size == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _parse_iso(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
