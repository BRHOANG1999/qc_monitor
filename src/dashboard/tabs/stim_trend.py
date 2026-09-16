"""Stim Artifact Trend tab.

Browse to ANY local folder of stim recordings (e.g. an ``electrodeTest-100nC``
electrode stress test, 71 hourly ``.mat`` that were never ingested into the DB)
and see, across the recordings over time:

* an early→late coloured OVERLAY of each recording's mean stim artifact, and
* metric-over-time TRENDS: artifact peak-to-peak, positive/negative peak, access
  resistance Rₐ (the clipping-robust ohmic step), slow-phase steady-state Z_ss,
  and saturation %.

The heavy read (71 × ~90 MB, off-proc via ``get_chunk``) runs on a background
daemon thread with a ``dcc.Interval`` progress poll (mirrors ``periictal_explorer``
/ ``sidecar_warm``); results are cached per folder so re-opening is instant and
switching the metric doesn't recompute. All compute lives in the pure
``src.utils.stim_artifact_trend``; this module is only the Dash shell.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
from collections import OrderedDict

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from plotly.express.colors import sample_colorscale

from src.dashboard import file_browser as _fb
from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE, loading_icon
from src.dashboard.data_helpers import empty_fig
from src.db.store import Store
from src.utils import stim_artifact_trend as _sat

logger = logging.getLogger("qc_monitor.dashboard.stim_trend")

_METRIC_LABEL = {
    "ptp": "Artifact peak-to-peak (LFP amplitude)",
    "pos": "Positive peak (rel. baseline)",
    "neg": "Negative peak (rel. baseline)",
    "ra": "Access resistance Rₐ (kΩ)",
    "zss": "Slow-phase steady-state Z_ss (kΩ)",
    "sat": "Saturation (% of samples at rail)",
}
_METRIC_OPTIONS = [{"label": v, "value": k} for k, v in _METRIC_LABEL.items()]

_BROWSE_BTN_STYLE = {"padding": "8px 14px", "background": "#2a2d3a",
                     "color": "#cfd0d6", "border": "1px solid #3a3d4a",
                     "borderRadius": "6px", "cursor": "pointer"}

# ------------------------------------------------ background job registry ---- #
_JOBS: dict = {}                        # folder -> {"status","progress"}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 4
_LOCK = threading.Lock()


def _cache_key(folder: str) -> str:
    """Folder + a cheap signature (count + newest mtime of its .mat) so adding or
    replacing recordings invalidates the cache."""
    mats = glob.glob(os.path.join(folder, "*.mat"))
    newest = max((os.path.getmtime(m) for m in mats), default=0.0)
    return f"{folder}|{len(mats)}:{newest:.0f}"


def _set(folder: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(folder, {}).update(kw)


def status(folder: str) -> dict:
    with _LOCK:
        return dict(_JOBS.get(folder, {}))


def _cached(folder: str):
    with _LOCK:
        return _CACHE.get(_cache_key(folder))


def _worker(folder: str) -> None:
    try:
        records = _sat.analyze_folder(
            folder, progress_cb=lambda i, n, p: _set(
                folder, status="running",
                progress=(f"Reading {i}/{n}: {os.path.basename(p)}" if p
                          else f"Reading {i}/{n}…")))
        ok = [r for r in records if r.get("ok")]
        with _LOCK:
            _CACHE[_cache_key(folder)] = {"records": records}
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
            _JOBS[folder] = {"status": "done",
                             "progress": f"{len(ok)}/{len(records)} recordings"}
    except Exception as e:              # noqa: BLE001 -- never leave the job hung
        logger.warning("stim-trend worker failed for %s: %s", folder, e)
        _set(folder, status="error", progress=str(e))


def _kick(folder: str) -> None:
    """Start the compute for *folder* unless it's already running or cached."""
    with _LOCK:
        if _CACHE.get(_cache_key(folder)) is not None:
            return
        st = _JOBS.get(folder)
        if st and st.get("status") == "running":
            return
        _JOBS[folder] = {"status": "running", "progress": "starting…"}
    threading.Thread(target=_worker, args=(folder,), daemon=True,
                     name="stim-trend").start()


# ------------------------------------------------------------- figures ---- #

def _time_colors(n: int) -> list:
    if n <= 1:
        return ["#5e7ce2"]
    return sample_colorscale("Turbo", [i / (n - 1) for i in range(n)])


def _build_overlay(records: list) -> go.Figure:
    recs = [r for r in records if r.get("ok") and r.get("mean_trace")]
    if not recs:
        return empty_fig("No readable recordings in this folder", height=420)
    colors = _time_colors(len(recs))
    fig = go.Figure()
    for r, c in zip(recs, colors):
        fig.add_trace(go.Scattergl(
            x=r["time_ms"], y=r["mean_trace"], mode="lines",
            line={"color": c, "width": 1.0}, opacity=0.8,
            name=(r["dt"].strftime("%m-%d %H:%M") if r.get("dt") else ""),
            showlegend=False,
            hovertext=(r["dt"].strftime("%Y-%m-%d %H:%M") if r.get("dt") else ""),
        ))
    fig.add_vline(x=0.0, line={"color": "#888", "width": 0.6})
    fig.update_layout(
        template="plotly_dark", height=420,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        title="Mean stim artifact per recording — early (blue) → late (red)",
        xaxis_title="time (ms)", yaxis_title="LFP amplitude",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _build_metric_trend(records: list, metric: str) -> go.Figure:
    pts = [(r["dt"], r["metrics"].get(metric)) for r in records
           if r.get("ok") and r.get("dt") is not None
           and r["metrics"].get(metric) is not None]
    label = _METRIC_LABEL.get(metric, metric)
    if not pts:
        note = (" — needs a STIM_REPORT.txt for gain"
                if metric in ("ra", "zss") else "")
        return empty_fig(f"No {label} values{note}", height=360)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers",
        line={"color": "#5e7ce2", "width": 1.5},
        marker={"size": 5, "color": "#5e7ce2"}))
    fig.update_layout(
        template="plotly_dark", height=360,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        title=f"{label} over the recording sequence",
        xaxis_title="recording time", yaxis_title=label,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _build_summary(records: list) -> str:
    ok = [r for r in records if r.get("ok")]
    if not ok:
        return "No readable recordings."
    dts = [r["dt"] for r in ok if r.get("dt")]
    span = ""
    if len(dts) >= 2:
        hrs = (max(dts) - min(dts)).total_seconds() / 3600.0
        span = f" spanning {hrs / 24:.1f} days ({min(dts):%m-%d %H:%M} → {max(dts):%m-%d %H:%M})"
    src = ok[0].get("stim_params", {}).get("source")
    have_ra = any(r["metrics"].get("ra") is not None for r in ok)
    imp = ("" if have_ra else
           " — no STIM_REPORT.txt gain, so Rₐ/Z_ss are unavailable "
           "(peak & saturation metrics still shown)")
    return (f"{len(ok)} readable recording(s){span}. "
            f"Stim params: {src}.{imp}")


# --------------------------------------------------------------- layout ---- #

def layout(store: Store = None, config: dict = None):
    return html.Div([
        html.H3("Stim Artifact Trend", style={"color": "white",
                                              "marginBottom": "8px"}),
        html.P("Browse to a folder of stim recordings (e.g. an electrode-test "
               "run). Each recording's stim artifact is averaged and trended "
               "over time — the mean-artifact overlay shows shape drift, the "
               "metric graph shows a chosen measure per recording.",
               style={"color": "#8a8d99", "fontSize": "12px",
                      "marginBottom": "10px"}),
        html.Div([
            html.Button("📂 Browse folder…", id="st-fb-open", n_clicks=0,
                        style=_BROWSE_BTN_STYLE,
                        title="Pick a folder of .mat stim recordings on the "
                              "server filesystem (not the monitor DB)."),
            html.Div([
                html.Label("Metric", style=LABEL_STYLE),
                dcc.Dropdown(id="stim-trend-metric", options=_METRIC_OPTIONS,
                             value="ptp", clearable=False,
                             style={**DROPDOWN_STYLE, "width": "320px"},
                             className="dark-dropdown"),
            ]),
        ], style={"display": "flex", "gap": "18px", "alignItems": "flex-end",
                  "flexWrap": "wrap", "marginBottom": "6px"}),
        _fb.modal("st"),
        dcc.Store(id="stim-trend-folder"),
        dcc.Interval(id="stim-trend-poll", interval=1500, disabled=True),
        html.Div(id="stim-trend-folder-label",
                 style={"color": "#cfd0d6", "fontSize": "11px",
                        "fontFamily": "ui-monospace, monospace",
                        "wordBreak": "break-all", "marginBottom": "2px"}),
        html.Div(id="stim-trend-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px"}),
        html.Div(id="stim-trend-summary",
                 style={"color": "#a9b0c0", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "6px"}),
        dcc.Loading(
            custom_spinner=loading_icon("Reading recordings…"),
            delay_show=150,
            overlay_style={"visibility": "visible", "opacity": 0.45},
            children=dcc.Graph(
                id="stim-trend-overlay",
                figure=empty_fig("Browse to a folder of stim recordings",
                                 height=420))),
        dcc.Graph(id="stim-trend-metric-graph",
                  figure=empty_fig("Pick a folder to see the metric trend",
                                   height=360)),
    ])


# ------------------------------------------------------------ callbacks ---- #

def register_callbacks(app, store: Store, config: dict) -> None:
    initial = ((config or {}).get("stim_trend", {}) or {}).get(
        "initial_dir") or r"D:\code\KMrecorder"
    _fb.register(app, "st", open_btn_id="st-fb-open", exts=(".mat",),
                 initial_path=initial)

    @app.callback(
        Output("stim-trend-folder", "data"),
        Output("stim-trend-poll", "disabled"),
        Output("stim-trend-status", "children"),
        Output("stim-trend-folder-label", "children"),
        Input("st-fb-result", "data"),
        prevent_initial_call=True,
    )
    def _pick_folder(result):
        if not result:
            return no_update, no_update, no_update, no_update
        folder = result.get("path")
        if not folder:
            return (no_update, no_update,
                    "Pick a FOLDER (Use this folder), not individual files.",
                    no_update)
        mats = _fb.list_dir(folder, exts=(".mat",))["files"]
        if not mats:
            return no_update, no_update, f"No .mat files in {folder}.", folder
        _kick(folder)
        cached = _cached(folder) is not None
        msg = ("Loaded from cache." if cached
               else f"Reading {len(mats)} recordings… (this takes a bit)")
        return folder, False, msg, folder

    @app.callback(
        Output("stim-trend-overlay", "figure"),
        Output("stim-trend-metric-graph", "figure"),
        Output("stim-trend-summary", "children"),
        Output("stim-trend-status", "children", allow_duplicate=True),
        Output("stim-trend-poll", "disabled", allow_duplicate=True),
        Input("stim-trend-poll", "n_intervals"),
        State("stim-trend-folder", "data"),
        State("stim-trend-metric", "value"),
        prevent_initial_call=True,
    )
    def _poll(_n, folder, metric):
        if not folder:
            return no_update, no_update, no_update, no_update, True
        st = status(folder)
        cached = _cached(folder)
        if cached is not None:
            records = cached["records"]
            return (_build_overlay(records),
                    _build_metric_trend(records, metric or "ptp"),
                    _build_summary(records), st.get("progress", "done"), True)
        if st.get("status") == "error":
            return (no_update, no_update, no_update,
                    f"Error: {st.get('progress')}", True)
        # still running
        return no_update, no_update, no_update, st.get("progress", "reading…"), False

    @app.callback(
        Output("stim-trend-metric-graph", "figure", allow_duplicate=True),
        Input("stim-trend-metric", "value"),
        State("stim-trend-folder", "data"),
        prevent_initial_call=True,
    )
    def _metric_change(metric, folder):
        if not folder:
            return no_update
        cached = _cached(folder)
        if cached is None:
            return no_update
        return _build_metric_trend(cached["records"], metric or "ptp")
