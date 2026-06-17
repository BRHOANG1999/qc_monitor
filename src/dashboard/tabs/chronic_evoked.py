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
import threading
from collections import OrderedDict
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback_context, dash_table, dcc, html

from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, LABEL_STYLE, ZEBRA_STRIPE)
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


# Background warm: building an animal's cache (reading + feature-extracting
# hundreds of files) must NEVER run inside the render callback -- it would
# hang the UI. The render path is query-only; the Refresh button kicks an
# off-thread warm so the page stays responsive and the cache fills in.
_warm_threads: dict = {}
_warm_lock = threading.Lock()


def _kick_warm(animal: str) -> bool:
    """Start a background warm for *animal* unless one is already running."""
    if not animal:
        return False
    with _warm_lock:
        t = _warm_threads.get(animal)
        if t is not None and t.is_alive():
            return False
        th = threading.Thread(target=_warm_worker, args=(animal,),
                              daemon=True, name=f"chronic-warm-{animal}")
        _warm_threads[animal] = th
        th.start()
        return True


def _warm_worker(animal: str) -> None:
    try:
        _cache().ensure_animal(animal)
    except Exception:  # noqa: BLE001 -- lock contention is non-fatal here
        pass


def _is_warming(animal: str) -> bool:
    with _warm_lock:
        t = _warm_threads.get(animal)
        return t is not None and t.is_alive()


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
                html.Label("Trend fit", style=LABEL_STYLE),
                dcc.Checklist(
                    id="chronic-trend-toggles",
                    options=[{"label": "Linear", "value": "lin"},
                             {"label": "Quad", "value": "quad"},
                             {"label": "Moving avg", "value": "ma"}],
                    value=["lin"], inline=True,
                    style={"fontSize": "12px"},
                    labelStyle={"color": "#cfd0d6", "marginRight": "14px",
                                "display": "inline-flex",
                                "alignItems": "center"},
                    inputStyle={"marginRight": "5px"}),
            ], style={"flex": "0 0 260px"}),
            html.Div([
                html.Label("Cache", style=LABEL_STYLE),
                html.Button("↻ Refresh / warm", id="chronic-refresh-btn",
                            n_clicks=0, style=_BTN_STYLE,
                            title="Re-query the cache and warm this animal "
                                  "in the background if it isn't cached yet."),
            ], style={"flex": "0 0 140px"}),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "10px",
                  "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.Label("Window (h, 0=all)", style=LABEL_STYLE),
                dcc.Input(id="chronic-window-hours", type="number",
                          value=0, min=0, step=12,
                          style={"width": "100%", "padding": "6px",
                                 "background": "#1f2230", "color": "#cfd0d6",
                                 "border": "1px solid #3a3d4a",
                                 "borderRadius": "6px"}),
            ], style={"flex": "0 0 150px"}),
            html.Div([
                html.Label("Scroll", style=LABEL_STYLE),
                dcc.Slider(id="chronic-window-scroll", min=0, max=1,
                           step=0.01, value=0, marks=None,
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "1", "minWidth": "200px"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "center",
                  "marginBottom": "8px"}),
        html.Div(id="chronic-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px"}),
        html.Div(id="chronic-trend-stats",
                 style={"color": "#cfd0d6", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "6px"}),
        dcc.Loading(type="default", color="#5e7ce2", children=[
            dcc.Graph(id="chronic-feature-plot"),
            dcc.Graph(id="chronic-recording-plot"),
            dcc.Graph(id="chronic-circadian-plot"),
            dcc.Graph(id="chronic-waveform-plot"),
            html.Div([
                html.Label("Correlation", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-corr-mode",
                    options=[{"label": lbl, "value": v}
                             for v, lbl, _x, _y in _CORR_MODES],
                    value=_CORR_MODES[0][0], clearable=False,
                    style={**DROPDOWN_STYLE, "maxWidth": "320px"},
                    className="dark-dropdown"),
            ], style={"marginTop": "6px"}),
            dcc.Graph(id="chronic-corr-plot"),
            html.Div("Per-recording summary", style={**LABEL_STYLE,
                     "marginTop": "10px"}),
            dash_table.DataTable(
                id="chronic-stats-table", page_size=15,
                sort_action="native",
                columns=[{"name": c, "id": c} for c in
                         ["Recording", "N", "Mean", "SD"]],
                style_data_conditional=[ZEBRA_STRIPE],
                **DARK_TABLE_STYLE),
        ]),
    ])


# (value, label, stim-amplitude source, evoked-feature column).
# stim source: 'peak'/'trough' raw columns, 'p2p' = peak-trough.
_CORR_MODES = [
    ("peak_peak", "Stim Peak vs Evoked Peak", "peak", "peak_amplitude"),
    ("trough_trough", "Stim Trough vs Evoked Trough", "trough",
     "trough_amplitude"),
    ("p2p_p2p", "Stim P2P vs Evoked P2P", "p2p", "peak_to_trough"),
    ("peak_p2p", "Stim Peak vs Evoked P2P", "peak", "peak_to_trough"),
]
_CORR_MAP = {m[0]: m for m in _CORR_MODES}
# How many per-recording mean waveforms to overlay (evenly sampled).
_MAX_WAVEFORMS = 40


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
        Output("chronic-circadian-plot", "figure"),
        Output("chronic-waveform-plot", "figure"),
        Output("chronic-corr-plot", "figure"),
        Output("chronic-stats-table", "data"),
        Output("chronic-trend-stats", "children"),
        Output("chronic-status", "children"),
        Input("chronic-animal-dropdown", "value"),
        Input("chronic-feature-dropdown", "value"),
        Input("chronic-hours-dropdown", "value"),
        Input("chronic-trend-toggles", "value"),
        Input("chronic-corr-mode", "value"),
        Input("chronic-window-hours", "value"),
        Input("chronic-window-scroll", "value"),
        Input("chronic-refresh-btn", "n_clicks"),
    )
    def _update(animal, feature, hours, overlays, corr_mode,
                win_hours, scroll, _clicks):
        blank = empty_fig("")
        if not animal:
            return (empty_fig("Select an animal"), blank, blank, blank,
                    blank, [], "", "")
        feature = feature if feature in _FEATURE_COLS else _DEFAULT_FEATURE
        # Query-only render. The heavy cache build runs off-thread; the
        # Refresh button is the only thing that kicks/refreshes a warm.
        if callback_context.triggered_id == "chronic-refresh-btn":
            _kick_warm(animal)
        try:
            cache = _cache()
            rows = cache.query(animal, hours=(hours or None))
            means = cache.query_recording_means(animal, hours=(hours or None))
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            err = empty_fig("Couldn't read the cache", hint=str(e))
            return err, blank, blank, blank, blank, [], "", f"Error: {e}"
        rows, means, win_lbl = _apply_window(rows, means, win_hours, scroll)
        pts = _feature_points(rows, feature)
        warming = " · ⏳ warming in background…" if _is_warming(animal) else ""
        status = (f"{len(rows)} cached responses · {len(pts[0])} with "
                  f"{_label(feature)}{win_lbl}{warming}.")
        table = _stats_table_rows(rows, feature)
        if not pts[0]:
            hint = ("" if rows else
                    f" — not cached yet; click ↻ Refresh to warm {animal} "
                    "in the background, then Refresh again.")
            msg = f"No {_label(feature)} for {animal}{hint}"
            return (empty_fig(msg), blank, blank,
                    _build_waveform_overlay(means, animal),
                    _build_stim_corr(rows, corr_mode), table, "", status + hint)
        return (_build_feature_scatter(pts, feature, animal, overlays or []),
                _build_recording_trend(rows, feature),
                _build_circadian(rows, feature),
                _build_waveform_overlay(means, animal),
                _build_stim_corr(rows, corr_mode), table,
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


def _build_waveform_overlay(means, animal) -> go.Figure:
    """Per-recording mean evoked waveform, rainbow-colored early->late."""
    if not means:
        return empty_fig("No mean waveforms (warm the cache)")
    step = _stride(len(means), _MAX_WAVEFORMS)
    shown = means[::step]
    colors = _time_colors(len(shown))
    fig = go.Figure()
    for i, m in enumerate(shown):
        t = m.get("time_axis") or []
        y = m.get("mean_trace") or []
        if not t or not y:
            continue
        fig.add_trace(go.Scattergl(
            x=t, y=y, mode="lines", line=dict(color=colors[i], width=1),
            name=str(m.get("rec_dt", ""))[:16], showlegend=False,
            hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    note = f" ({len(shown)} of {len(means)})" if step > 1 else ""
    fig.update_layout(
        title=f"Mean evoked waveform per recording — {animal}{note}",
        xaxis_title="Time (ms)", yaxis_title="Amplitude", height=380)
    return fig


def _build_stim_corr(rows, mode) -> go.Figure:
    """Stim amplitude vs evoked amplitude scatter + regression."""
    m = _CORR_MAP.get(mode) or _CORR_MODES[0]
    _, label, x_src, y_col = m
    xs, ys, secs = [], [], []
    for r in rows:
        x = _stim_amp(r, x_src)
        y = r.get(y_col)
        dt = _parse_iso(r.get("abs_dt"))
        if x is None or y is None or dt is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        secs.append(dt.timestamp())
    if len(xs) < 3:
        return empty_fig("Not enough paired points for correlation")
    xa, ya = np.asarray(xs), np.asarray(ys)
    step = _stride(len(xa), _MAX_POINTS)
    fig = go.Figure(go.Scattergl(
        x=xa[::step], y=ya[::step], mode="markers",
        marker=dict(size=4, color=_norm(np.asarray(secs))[::step],
                    colorscale="Turbo", opacity=0.6),
        hoverinfo="x+y", name="epochs"))
    _add_regression(fig, xa, ya)
    fig.update_layout(title=f"{label}", xaxis_title="Stimulus amplitude",
                      yaxis_title="Evoked amplitude", height=380,
                      showlegend=True)
    return fig


def _stim_amp(r, src):
    if src == "p2p":
        pk, tr = r.get("peak"), r.get("trough")
        return (pk - tr) if (pk is not None and tr is not None) else None
    return r.get(src)


def _add_regression(fig, xa, ya) -> None:
    from scipy.stats import linregress
    lr = linregress(xa, ya)
    xline = np.array([xa.min(), xa.max()])
    fig.add_trace(go.Scattergl(
        x=xline, y=lr.slope * xline + lr.intercept, mode="lines",
        line=dict(color="#ff453a", width=2),
        name=f"r={lr.rvalue:.2f} r²={lr.rvalue**2:.2f} "
             f"p={lr.pvalue:.1e} n={xa.size}"))


def _apply_window(rows, means, win_hours, scroll):
    """Filter to a scrollable [start, start+window) slice of wall time."""
    if not win_hours or win_hours <= 0 or not rows:
        return rows, means, ""
    times = [_parse_iso(r.get("abs_dt")) for r in rows]
    valid = [t for t in times if t is not None]
    if not valid:
        return rows, means, ""
    t0, t1 = min(valid), max(valid)
    span = (t1 - t0).total_seconds() / 3600.0
    if span <= win_hours:
        return rows, means, ""
    start_h = float(scroll or 0) * (span - win_hours)
    start = t0.timestamp() + start_h * 3600.0
    end = start + win_hours * 3600.0
    rw = [r for r, t in zip(rows, times)
          if t is not None and start <= t.timestamp() < end]
    mw = [m for m in means
          if _in_window(m.get("rec_dt"), start, end)]
    lbl = f" · window {win_hours:g}h @ +{start_h:.0f}h"
    return rw, mw, lbl


def _in_window(rec_dt, start, end) -> bool:
    t = _parse_iso(rec_dt)
    return t is not None and start <= t.timestamp() < end


def _build_circadian(rows, feature) -> go.Figure:
    """Polar: theta = time-of-day, rho = feature; 24-bin mean + Rayleigh."""
    label = _label(feature)
    theta, rho = [], []
    for r in rows:
        v = r.get(feature)
        dt = _parse_iso(r.get("abs_dt"))
        if v is None or dt is None:
            continue
        hod = dt.hour + dt.minute / 60.0
        theta.append(hod / 24.0 * 360.0)
        rho.append(float(v))
    if len(theta) < 3:
        return empty_fig("Not enough points for a circadian view")
    th = np.asarray(theta)
    rh = np.asarray(rho)
    shift = -min(0.0, float(rh.min()))         # keep rho >= 0 for polar
    fig = go.Figure(go.Scatterpolargl(
        theta=th, r=rh + shift, mode="markers",
        marker=dict(size=4, color=th, colorscale="Turbo", opacity=0.5),
        name="responses"))
    _add_hourly_mean(fig, th, rh + shift)
    R, p = _rayleigh(np.deg2rad(th))
    fig.update_layout(
        title=f"{label} by time of day — Rayleigh R={R:.2f} p={p:.1e}",
        height=420, polar=dict(angularaxis=dict(
            rotation=90, direction="clockwise",
            tickmode="array", tickvals=[0, 90, 180, 270],
            ticktext=["0h", "6h", "12h", "18h"])))
    return fig


def _add_hourly_mean(fig, th, rho) -> None:
    bins = np.floor(th / 15.0).astype(int) % 24     # 24 bins of 15 deg
    cx, cy = [], []
    for b in range(24):
        m = bins == b
        if np.any(m):
            cx.append((b + 0.5) * 15.0)
            cy.append(float(np.mean(rho[m])))
    if cx:
        cx.append(cx[0])
        cy.append(cy[0])
        fig.add_trace(go.Scatterpolar(theta=cx, r=cy, mode="lines",
                      line=dict(color="#ff453a", width=2),
                      name="hourly mean"))


def _rayleigh(angles):
    """Rayleigh mean resultant length R and p-value for circular data."""
    n = angles.size
    if n == 0:
        return 0.0, 1.0
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    R = float(np.hypot(c, s))
    z = n * R * R
    p = float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)))
    return R, min(1.0, max(0.0, p))


def _stats_table_rows(rows, feature) -> list:
    """Per-recording N / mean / SD of the selected feature."""
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        v = r.get(feature)
        if v is not None and r.get("rec_dt"):
            grouped.setdefault(r["rec_dt"], []).append(float(v))
    out = []
    for rec, vals in grouped.items():
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append({"Recording": str(rec)[:19], "N": len(vals),
                    "Mean": f"{statistics.mean(vals):.4g}",
                    "SD": f"{sd:.4g}"})
    return out


def _time_colors(n) -> list:
    from plotly.colors import sample_colorscale
    if n <= 1:
        return ["#5e7ce2"] * max(1, n)
    return sample_colorscale("Turbo", [i / (n - 1) for i in range(n)])


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
