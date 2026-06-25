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

import logging
import os
import statistics
import threading
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger("qc_monitor.dashboard.chronic_evoked")

import numpy as np
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback_context, dash_table, dcc, html)

from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, LABEL_STYLE, ZEBRA_STRIPE, loading_icon)
from src.dashboard.data_helpers import (
    EVOKED_FEATURE_LABELS, TIME_RANGE_OPTIONS, empty_fig)
from src.utils import evoked_features as ef
from src.utils.animal import is_animal_channel, split_animal_electrode
from src.utils.decimate import parse_relayout
from src.utils.evoked_output import (
    DEFAULT_EVOKED_DIR, compute_feature_rows, list_animals,
    list_evoked_files, parse_recording_dt, parse_session, read_file_evoked,
    read_feature_sidecar, write_feature_sidecar,
    animals_in_filename, sessions_for_animal, sessions_with_dt_for_animal)

# Feature dropdown = every cached evoked feature, friendly labels from the
# app-wide map. Expensive ones are flagged so the UI can disable them.
_FEATURE_COLS = list(ef.ALL_COLUMNS)
_EXPENSIVE = set(ef.EXPENSIVE_COLUMNS)
_DEFAULT_FEATURE = "line_length"

# WebGL stays smooth to ~10^5 markers; above this we stride-sample.
_MAX_POINTS = 60000
_MA_WINDOW = 50            # per-animal moving-average window (epochs).

# Density heatmap render grid. Constant pixel resolution; the *temporal*
# resolution follows the zoom (relayout rebins to the visible window).
_DENSITY_NX = 300
_DENSITY_NY = 150
# Onset-aligned view: default lookback (hours before an onset) to include.
_ONSET_DEFAULT_LOOKBACK_H = 24

# Last-plotted density arrays, keyed by selection signature, so the zoom
# relayout callback rebins in RAM instead of re-reading sidecars. Bounded.
_PLOT_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PLOT_CACHE_MAX = 4

# Configured in register_callbacks(); layout() reads them on tab open.
_EVOKED_DIR = DEFAULT_EVOKED_DIR
_EXPENSIVE_ENABLED = False

# Selections (animal, sessions) auto-loaded at least once -- so an empty
# result AFTER a completed load shows a message instead of re-kicking the
# warm forever (the "loading the same files 3 times" loop).
_autoloaded: set = set()


def _sel_key(animal, sess):
    return (animal or "", tuple(sorted(sess or [])))


def _label(col: str) -> str:
    return EVOKED_FEATURE_LABELS.get(col, col.replace("_", " ").title())


def _window_summary(cfg_dict) -> str:
    """One-line description of the time window + processing the features are
    computed over, for the banner above the plot. Reflects the live Configure
    state: default = full extracted trace + toolkit defaults."""
    cfg = ef.FeatureConfig.from_dict(cfg_dict)
    ws, we = cfg.window_start_ms, cfg.window_end_ms
    if ws is None and we is None:
        win = "full extracted trace"
    else:
        lo = f"{ws:g}" if ws is not None else "trace start"
        hi = f"{we:g}" if we is not None else "trace end"
        win = f"{lo}–{hi} ms"
    proc = []
    if cfg.bandpass:
        proc.append(f"bandpass {cfg.bp_low_hz:g}–{cfg.bp_high_hz:g} Hz")
    if cfg.notch:
        proc.append(f"notch {cfg.notch_hz:g} Hz")
    if cfg.smoothing:
        proc.append(f"smoothing {cfg.smooth_ms:g} ms")
    if cfg.baseline:
        proc.append("baseline")
    tail = " · ".join(proc) if proc else "toolkit defaults"
    return f"Computing features over: {win} · {tail}"


# No sqlite cache: features live in per-(recording, animal) JSON sidecars next
# to each *_evoked.mat in evokedOutput. The render path reads those sidecars
# (computing+writing any missing in a background "load"); the heavy trace reads
# only happen once per recording, then the sidecar makes every re-plot instant.
#
# Background load: building an animal's sidecars (reading + feature-extracting
# hundreds of files) must NEVER run inside the render callback -- it would
# hang the UI. The render path is read-only; ▶ Plot kicks an off-thread load
# so the page stays responsive and the sidecars fill in.
_warm_threads: dict = {}
_warm_progress: dict = {}        # animal -> {"done": int, "total": int}
_warm_lock = threading.Lock()


def _kick_warm(animal: str, sessions: list | None = None,
               cfg_dict: dict | None = None, force: bool = False) -> bool:
    """Start a background sidecar-build for *animal* (optionally just
    *sessions*) unless one is already running."""
    if not animal:
        return False
    with _warm_lock:
        t = _warm_threads.get(animal)
        if t is not None and t.is_alive():
            return False
        _warm_progress[animal] = {"done": 0, "total": 0}
        th = threading.Thread(target=_warm_worker,
                              args=(animal, sessions, cfg_dict, force),
                              daemon=True, name=f"chronic-warm-{animal}")
        _warm_threads[animal] = th
        th.start()
        return True


def _set_warm_progress(animal: str, done: int, total: int,
                       path: str = "", phase: str = "") -> None:
    with _warm_lock:
        _warm_progress[animal] = {"done": int(done), "total": int(total),
                                  "path": path or "", "phase": phase or ""}


def _warm_worker(animal: str, sessions: list | None = None,
                 cfg_dict: dict | None = None, force: bool = False) -> None:
    """Build *animal*'s feature sidecars for the selection (default config) or
    recompute its rows into the in-memory store (custom Configure). Reads each
    *_evoked.mat at most once; an existing fresh sidecar is skipped."""
    try:
        cfg = ef.FeatureConfig.from_dict(cfg_dict)
        passthrough = cfg.is_passthrough()
        files = _selection_files(animal, sessions)
        n = len(files)
        recompute: list = []
        for i, fp in enumerate(files):
            _set_warm_progress(animal, i, n, os.path.basename(fp), "reading")
            if passthrough and not force and read_feature_sidecar(
                    fp, animal) is not None:
                continue
            rows = _preview_rows(fp, animal, cfg)
            if passthrough:
                if rows:
                    try:
                        write_feature_sidecar(fp, animal, rows)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("sidecar write failed %s: %s", fp, e)
            else:
                recompute.extend(rows)
        if not passthrough:
            recompute.sort(key=lambda r: r.get("abs_dt") or "")
            _RECOMPUTE_CACHE[_recompute_key(animal, sessions, cfg_dict)] = \
                recompute
            while len(_RECOMPUTE_CACHE) > _RECOMPUTE_MAX:
                _RECOMPUTE_CACHE.pop(next(iter(_RECOMPUTE_CACHE)))
        _set_warm_progress(animal, n, n, "", "done")
        logger.info("chronic load %s (sessions=%s, passthrough=%s): %d files",
                    animal, sessions, passthrough, n)
    except Exception as e:  # noqa: BLE001 -- never crash the daemon thread
        logger.warning("chronic load failed for %s: %s", animal, e)


def _is_warming(animal: str) -> bool:
    with _warm_lock:
        t = _warm_threads.get(animal)
        return t is not None and t.is_alive()


def list_warming() -> list[str]:
    """Animals whose background warm thread is currently alive (for the
    Jobs monitor). Runs in parallel -- one thread per animal."""
    with _warm_lock:
        return [a for a, t in list(_warm_threads.items())
                if t is not None and t.is_alive()]


def warm_progress(animal: str) -> dict | None:
    """``{"done","total"}`` for an in-flight warm, or None."""
    with _warm_lock:
        p = _warm_progress.get(animal)
        return dict(p) if p else None


_INPUT_STYLE = {"width": "100%", "padding": "6px", "background": "#1f2230",
                "color": "#cfd0d6", "border": "1px solid #3a3d4a",
                "borderRadius": "6px"}


def _feature_options() -> list[dict]:
    opts = []
    for col in _FEATURE_COLS:
        disabled = (col in _EXPENSIVE) and not _EXPENSIVE_ENABLED
        suffix = "  (warm --expensive)" if disabled else ""
        opts.append({"label": _label(col) + suffix, "value": col,
                     "disabled": disabled})
    return opts


def _session_label(rec: dict) -> str:
    """'session  ·  <date(s)>' from a {session, first, last} record."""
    s = rec["session"]
    first, last = rec.get("first") or "", rec.get("last") or ""
    if not first:
        return s
    fd = first[:10]
    if last and last[:10] != fd:
        return f"{s}  ·  {fd} → {last[:10]}"
    return f"{s}  ·  {first[:16].replace('T', ' ')}"


def _session_options(animal) -> list[dict]:
    """Session options for *animal* (label shows the recording timestamp),
    or [] (best-effort; never raises).

    Read straight from the evoked filenames so the dropdown populates the
    moment an animal is selected. The value stays the bare session label so
    filtering is unchanged.
    """
    if not animal:
        return []
    try:
        dated = sessions_with_dt_for_animal(_EVOKED_DIR, animal)
        return [{"label": _session_label(r), "value": r["session"]}
                for r in dated]
    except Exception:  # noqa: BLE001
        return []


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
            ], style={"flex": "1", "minWidth": "150px"}),
            html.Div([
                html.Label("Session(s) — blank = all", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-session-dropdown",
                    options=_session_options(default_animal), value=[],
                    multi=True, placeholder="All sessions",
                    style=DROPDOWN_STYLE, className="dark-dropdown"),
            ], style={"flex": "1.4", "minWidth": "220px"}),
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
                html.Label("Overlays", style=LABEL_STYLE),
                dcc.Checklist(
                    id="chronic-event-toggles",
                    options=[{"label": "Seizure onsets", "value": "onsets"}],
                    value=[], inline=True, style={"fontSize": "12px"},
                    labelStyle={"color": "#cfd0d6",
                                "display": "inline-flex",
                                "alignItems": "center"},
                    inputStyle={"marginRight": "5px"}),
            ], style={"flex": "0 0 150px"}),
            html.Div([
                html.Label("View", style=LABEL_STYLE),
                dcc.RadioItems(
                    id="chronic-view-mode",
                    options=[{"label": "Density", "value": "density"},
                             {"label": "Scatter", "value": "scatter"},
                             {"label": "Onset-aligned", "value": "onset"}],
                    value="density", inline=True, style={"fontSize": "12px"},
                    labelStyle={"color": "#cfd0d6", "marginRight": "10px",
                                "display": "inline-flex",
                                "alignItems": "center"},
                    inputStyle={"marginRight": "5px"}),
            ], style={"flex": "0 0 150px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("👁 Preview", id="chronic-preview-btn", n_clicks=0,
                            style=_BTN_STYLE,
                            title="Quick look at the most recent recording for "
                                  "this animal / session(s), read straight from "
                                  "evokedOutput — no full chronic load. Use it "
                                  "to confirm there's signal before ▶ Plot."),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("▶ Plot", id="chronic-load-btn", n_clicks=0,
                            style=_PLOT_BTN_STYLE,
                            title="Build the plots for the current animal / "
                                  "session(s) / feature. Nothing renders "
                                  "until you click this."),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("↻ Reload from disk", id="chronic-refresh-btn",
                            n_clicks=0, style=_BTN_STYLE,
                            title="Re-read this selection from evokedOutput "
                                  "(picks up new/changed files). ▶ Plot "
                                  "already loads automatically."),
            ], style={"flex": "0 0 150px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("⬇ Export CSV", id="chronic-export-btn",
                            n_clicks=0, style=_BTN_STYLE,
                            title="Download the per-epoch features for the "
                                  "current animal / session(s) as CSV. Uses "
                                  "the loaded data (Plot warms the full "
                                  "history; otherwise the latest recording)."),
            ], style={"flex": "0 0 130px"}),
            dcc.Download(id="chronic-export-dl"),
        ], style={"display": "flex", "gap": "14px", "marginBottom": "10px",
                  "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.Label("Rolling window (responses)", style=LABEL_STYLE),
                dcc.Input(id="chronic-roll-window", type="number",
                          value=301, min=11, step=20, style=_INPUT_STYLE),
            ], style={"flex": "0 0 190px"}),
            html.Div([
                html.Label("Window (h, 0=all)", style=LABEL_STYLE),
                dcc.Input(id="chronic-window-hours", type="number",
                          value=0, min=0, step=12, style=_INPUT_STYLE),
            ], style={"flex": "0 0 150px"}),
            html.Div([
                html.Label("Scroll", style=LABEL_STYLE),
                dcc.Slider(id="chronic-window-scroll", min=0, max=1,
                           step=0.01, value=0, marks=None,
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "1", "minWidth": "200px"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "center",
                  "marginBottom": "8px", "flexWrap": "wrap"}),
        # Controls for the Onset-aligned view (ignored by the other views).
        html.Div([
            html.Div([
                html.Label("Onset lookback (h)", style=LABEL_STYLE),
                dcc.Input(id="chronic-onset-lookback", type="number",
                          value=_ONSET_DEFAULT_LOOKBACK_H, min=1, step=6,
                          style=_INPUT_STYLE),
            ], style={"flex": "0 0 170px"}),
            html.Div([
                html.Label("Onset time axis", style=LABEL_STYLE),
                dcc.Checklist(
                    id="chronic-onset-logtime",
                    options=[{"label": " Log scale", "value": "log"}],
                    value=["log"], style={"fontSize": "12px"},
                    labelStyle={"color": "#cfd0d6", "display": "inline-flex",
                                "alignItems": "center"},
                    inputStyle={"marginRight": "5px"}),
            ], style={"flex": "0 0 170px"}),
            html.Div("Used by the Onset-aligned view — bins each epoch by time "
                     "before its nearest upcoming seizure onset.",
                     style={"color": "#8a8d99", "fontSize": "11px",
                            "alignSelf": "flex-end"}),
        ], style={"display": "flex", "gap": "14px", "alignItems": "center",
                  "marginBottom": "8px", "flexWrap": "wrap"}),
        _configure_card(),
        dcc.Store(id="chronic-selection"),
        dcc.Store(id="chronic-config"),   # Configure panel settings (Phase 2)
        dcc.Interval(id="chronic-warm-poll", interval=1500, disabled=True),
        html.Div(id="chronic-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px"}),
        html.Div(id="chronic-trend-stats",
                 style={"color": "#cfd0d6", "fontSize": "12px",
                        "minHeight": "16px", "marginBottom": "6px"}),
        # Live indicator of the window/processing features are computed over.
        html.Div(id="chronic-window-info", children=_window_summary(None),
                 style={"color": "#7f8290", "fontSize": "11px",
                        "fontStyle": "italic", "marginBottom": "6px"}),
        # Primary plot -- always rendered on ▶ Plot.
        dcc.Loading(
            custom_spinner=loading_icon("Plotting…"),
            overlay_style={"visibility": "visible", "opacity": 0.45},
            children=dcc.Graph(
                id="chronic-feature-plot",
                figure=empty_fig("Pick animal / session(s) / feature, "
                                 "then click ▶ Plot"))),
        # Other analyses -- expand each on demand (computed lazily).
        html.Div("Other analyses (expand to compute)",
                 style={**LABEL_STYLE, "marginTop": "12px",
                        "color": "#8a8d99"}),
        _expandable("rec", "Per-recording trend",
                    dcc.Graph(id="chronic-recording-plot")),
        _expandable("wave", "Mean waveform overlay",
                    dcc.Graph(id="chronic-waveform-plot")),
        _expandable("corr", "Stim↔evoked correlation", html.Div([
            html.Div([
                html.Label("Correlation mode", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="chronic-corr-mode",
                    options=[{"label": lbl, "value": v}
                             for v, lbl, _x, _y in _CORR_MODES],
                    value=_CORR_MODES[0][0], clearable=False,
                    style={**DROPDOWN_STYLE, "maxWidth": "320px"},
                    className="dark-dropdown"),
            ], style={"marginBottom": "6px"}),
            dcc.Graph(id="chronic-corr-plot"),
        ])),
        _expandable("circ", "Circadian (time of day)",
                    dcc.Graph(id="chronic-circadian-plot")),
        _expandable("rhythm", "Rhythm / cycles (periodogram + phase)",
                    html.Div([
                        html.Div([
                            html.Label("Period override (days, blank = "
                                       "auto-detect dominant)", style=LABEL_STYLE),
                            dcc.Input(id="chronic-rhythm-period", type="number",
                                      value=None, min=0.25, step="any",
                                      style={**_INPUT_STYLE, "maxWidth": "280px"}),
                        ], style={"marginBottom": "6px"}),
                        dcc.Graph(id="chronic-rhythm-plot"),
                    ])),
        _expandable("table", "Per-recording table", dash_table.DataTable(
            id="chronic-stats-table", page_size=15, sort_action="native",
            columns=[{"name": c, "id": c} for c in
                     ["Recording", "N", "Mean", "SD"]],
            style_data_conditional=[ZEBRA_STRIPE], **DARK_TABLE_STYLE)),
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
_PLOT_BTN_STYLE = {"width": "100%", "padding": "8px", "background": "#5e7ce2",
                   "color": "white", "border": "none", "fontWeight": "700",
                   "borderRadius": "6px", "cursor": "pointer"}
_SECTION_BTN_STYLE = {"width": "100%", "padding": "10px 14px",
                      "textAlign": "left", "background": "#1f2230",
                      "color": "#cfd0d6", "border": "1px solid #3a3d4a",
                      "borderRadius": "6px", "cursor": "pointer",
                      "fontSize": "13px", "fontWeight": "600"}

# Expandable analysis sections: (key, title, child-builder-tag). Each is a
# toggle button + a collapsed container that computes its figure lazily on
# first/each expand. (id, target-prop) pairs drive the per-section callbacks.
_SECTIONS = [
    ("rec", "Per-recording trend", "chronic-recording-plot", "figure"),
    ("wave", "Mean waveform overlay", "chronic-waveform-plot", "figure"),
    ("corr", "Stim↔evoked correlation", "chronic-corr-plot", "figure"),
    ("circ", "Circadian (time of day)", "chronic-circadian-plot", "figure"),
    ("table", "Per-recording table", "chronic-stats-table", "data"),
]


def _cfg_check(label: str, cid: str):
    return dcc.Checklist(
        id=cid, options=[{"label": f" {label}", "value": "on"}], value=[],
        inline=True, style={"display": "inline-block"},
        labelStyle={"color": "#cfd0d6", "fontSize": "12px"})


def _cfg_num(cid: str, val, **kw):
    return dcc.Input(id=cid, type="number", value=val, style=_INPUT_STYLE,
                     **kw)


def _configure_card() -> html.Div:
    """Collapsible Configure panel: crop the feature window + optional
    bandpass / notch / smoothing / baseline applied (on raw epoch traces)
    before the feature math. Defaults are a pass-through (= the toolkit's
    already-extracted traces). Settings are gathered into chronic-config and
    applied on ▶ Plot."""
    def cell(*children, w="0 0 150px", end=False):
        st = {"flex": w}
        if end:
            st["alignSelf"] = "flex-end"
        return html.Div(list(children), style=st)
    return html.Div([
        html.Button("▸ Configure (window / filter / smoothing / baseline)",
                    id="chronic-configure-btn", n_clicks=0,
                    style=_SECTION_BTN_STYLE),
        html.Div([
            html.Div([
                cell(html.Label("Window start (ms, blank=full)",
                                style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-win-start", None, step="any"), w="0 0 190px"),
                cell(html.Label("Window end (ms)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-win-end", None, step="any")),
                cell(_cfg_check("Baseline (subtract pre-stim mean)",
                                "chronic-cfg-baseline"), w="0 0 240px", end=True),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                      "marginBottom": "8px"}),
            html.Div([
                cell(_cfg_check("Bandpass", "chronic-cfg-bandpass"),
                     w="0 0 100px", end=True),
                cell(html.Label("Low (Hz)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-bp-lo", 1.0, step=0.5), w="0 0 100px"),
                cell(html.Label("High (Hz)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-bp-hi", 100.0, step=5), w="0 0 100px"),
                cell(_cfg_check("Notch", "chronic-cfg-notch"),
                     w="0 0 80px", end=True),
                cell(html.Label("Notch (Hz)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-notch-hz", 60.0, step=10), w="0 0 100px"),
                cell(_cfg_check("Smoothing", "chronic-cfg-smooth"),
                     w="0 0 110px", end=True),
                cell(html.Label("Smooth (ms)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-smooth-ms", 5.0, step=1), w="0 0 100px"),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                      "marginBottom": "6px"}),
            html.Div("Applied on ▶ Plot — features are recomputed from the raw "
                     "epoch traces (warm the animal first so its traces are "
                     "cached). Defaults = the toolkit's extracted traces.",
                     style={"color": "#8a8d99", "fontSize": "11px"}),
        ], id="chronic-configure-wrap",
           style={"display": "none", "padding": "8px",
                  "border": "1px solid #2a2d3a", "borderRadius": "6px"}),
    ], style={"marginBottom": "8px"})


def _expandable(key: str, title: str, child) -> html.Div:
    """A toggle button + collapsed (display:none) container wrapping *child*
    in its own spinner. Click expands + computes; click again collapses."""
    return html.Div([
        html.Button(f"▸ {title}", id=f"chronic-{key}-btn", n_clicks=0,
                    style=_SECTION_BTN_STYLE),
        html.Div(
            dcc.Loading(
                custom_spinner=loading_icon(f"Computing {title}…", small=True),
                overlay_style={"visibility": "visible", "opacity": 0.45},
                children=child),
            id=f"chronic-{key}-wrap", style={"display": "none"}),
    ], style={"marginTop": "8px"})


def register_callbacks(app, store, config: dict) -> None:
    """Wire pickers + trend toggles -> (rainbow scatter, per-recording trend,
    stats, status). Reads ``config.chronic_evoked``."""
    global _EVOKED_DIR, _EXPENSIVE_ENABLED
    ce = (config or {}).get("chronic_evoked", {}) or {}
    _EVOKED_DIR = ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR
    _EXPENSIVE_ENABLED = bool(ce.get("compute_expensive", False))

    @app.callback(
        Output("chronic-session-dropdown", "options"),
        Output("chronic-session-dropdown", "value"),
        Input("chronic-animal-dropdown", "value"),
        Input("chronic-refresh-btn", "n_clicks"),
    )
    def _sessions(animal, _clicks):
        # Refresh updates the options (new warmed sessions) but keeps the
        # current selection; an animal change clears it.
        from dash import no_update
        clear = callback_context.triggered_id == "chronic-animal-dropdown"
        return _session_options(animal), ([] if clear else no_update)

    # ---- Configure panel: collapse toggle + gather settings ---- #
    @app.callback(
        Output("chronic-configure-wrap", "style"),
        Input("chronic-configure-btn", "n_clicks"),
        State("chronic-configure-wrap", "style"),
        prevent_initial_call=True,
    )
    def _toggle_configure(_n, style):
        style = dict(style or {})
        style["display"] = "none" if style.get("display") != "none" else "block"
        return style

    @app.callback(
        Output("chronic-config", "data"),
        Input("chronic-cfg-win-start", "value"),
        Input("chronic-cfg-win-end", "value"),
        Input("chronic-cfg-baseline", "value"),
        Input("chronic-cfg-bandpass", "value"),
        Input("chronic-cfg-bp-lo", "value"),
        Input("chronic-cfg-bp-hi", "value"),
        Input("chronic-cfg-notch", "value"),
        Input("chronic-cfg-notch-hz", "value"),
        Input("chronic-cfg-smooth", "value"),
        Input("chronic-cfg-smooth-ms", "value"),
    )
    def _gather_config(ws, we, bl, bp, lo, hi, nt, nhz, sm, sms):
        def on(v):
            return bool(v) and "on" in v
        return {
            "window_start_ms": ws, "window_end_ms": we, "baseline": on(bl),
            "bandpass": on(bp), "bp_low_hz": lo or 1.0, "bp_high_hz": hi or 100.0,
            "notch": on(nt), "notch_hz": nhz or 60.0,
            "smoothing": on(sm), "smooth_ms": sms or 5.0,
        }

    # Live banner: which window + processing features are computed over. Driven
    # by chronic-config (populated on load), so it's correct from first render
    # and updates the instant the user edits Configure -- before ▶ Plot.
    @app.callback(
        Output("chronic-window-info", "children"),
        Input("chronic-config", "data"),
    )
    def _window_banner(cfg):
        return _window_summary(cfg)

    # Primary render: only the feature-vs-time plot + trend stats. Captures
    # the current selection into chronic-selection so the expandable sections
    # can re-query lazily. Fires only on ▶ Plot / ↻ Refresh.
    @app.callback(
        Output("chronic-feature-plot", "figure"),
        Output("chronic-trend-stats", "children"),
        Output("chronic-status", "children"),
        Output("chronic-selection", "data"),
        Output("chronic-warm-poll", "disabled"),
        Input("chronic-load-btn", "n_clicks"),
        Input("chronic-refresh-btn", "n_clicks"),
        Input("chronic-preview-btn", "n_clicks"),
        State("chronic-animal-dropdown", "value"),
        State("chronic-session-dropdown", "value"),
        State("chronic-feature-dropdown", "value"),
        State("chronic-hours-dropdown", "value"),
        State("chronic-trend-toggles", "value"),
        State("chronic-event-toggles", "value"),
        State("chronic-view-mode", "value"),
        State("chronic-window-hours", "value"),
        State("chronic-window-scroll", "value"),
        State("chronic-roll-window", "value"),
        State("chronic-config", "data"),
        State("chronic-onset-lookback", "value"),
        State("chronic-onset-logtime", "value"),
        prevent_initial_call=True,
    )
    def _update(_load, _refresh, _preview, animal, sessions, feature, hours,
                overlays, event_overlays, view_mode, win_hours, scroll,
                roll_window, cfg_dict, onset_lookback, onset_logtime):
        if not animal:
            return (empty_fig("Select an animal, then ▶ Plot"),
                    "", "", None, True)
        from src.dashboard import activity as _activity
        _activity.track(store, "chronic_evoked",
                        callback_context.triggered_id or "plot", animal,
                        {"feature": feature})
        feature = feature if feature in _FEATURE_COLS else _DEFAULT_FEATURE
        sess = sessions or None
        sel = {"animal": animal, "sessions": sess, "hours": hours,
               "win_hours": win_hours, "scroll": scroll, "feature": feature,
               "roll_window": roll_window, "config": cfg_dict or None}
        scope = (f"the {len(sessions)} selected session(s)" if sessions
                 else "all sessions")
        key = _sel_key(animal, sess)
        # Preview: read just the most recent recording for this selection
        # straight from evokedOutput (no cache, no warm) so the user can
        # confirm there's signal before committing to the full chronic load.
        if callback_context.triggered_id == "chronic-preview-btn":
            try:
                fig, status, stats = _build_preview(
                    animal, sess, feature, cfg_dict, overlays or [],
                    roll_window)
            except Exception as e:  # noqa: BLE001 -- surface, never crash UI
                return (empty_fig("Preview failed", hint=str(e)),
                        "", f"Preview error: {e}", sel, True)
            return (fig, stats, status, sel, True)
        # "Reload from disk" forces a fresh re-read -> clear the once-flag so
        # it retries, show the loading state, enable the progress poll.
        if callback_context.triggered_id == "chronic-refresh-btn":
            _autoloaded.discard(key)
            _autoloaded.add(key)
            _kick_warm(animal, sess, cfg_dict, force=True)
            return (empty_fig(f"Reloading {scope} for {animal} from "
                              f"evokedOutput…"),
                    "", f"Reloading {animal} from evokedOutput…", sel, False)
        try:
            rows, _means, win_lbl = _query_for_selection(sel, need_means=False)
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            return (empty_fig("Couldn't read the data", hint=str(e)),
                    "", f"Error: {e}", sel, True)
        if not rows:
            if _is_warming(animal):                       # still loading
                return (empty_fig(f"Loading {scope} for {animal} from "
                                  f"evokedOutput…"),
                        "", _warm_status_text(animal), sel, False)
            if key in _autoloaded:
                # Already auto-loaded once and STILL empty -> stop looping;
                # surface it instead of re-kicking the warm forever.
                return (empty_fig(
                    f"No evoked data for {animal} in this selection. The files "
                    f"may have no detected stimuli, or the load didn't persist "
                    f"(try ↻ Reload from disk; check the logs)."),
                    "", "No evoked data for this selection.", sel, True)
            # First time for this selection -> auto-load it.
            _autoloaded.add(key)
            _kick_warm(animal, sess, cfg_dict)
            return (empty_fig(f"Loading {scope} for {animal} from "
                              f"evokedOutput…"),
                    "", f"Loading {animal} from evokedOutput…", sel, False)
        # We have data: clear the once-flag so a later (e.g. cache-cleared)
        # empty result can auto-load again.
        _autoloaded.discard(key)
        pts = _feature_points(rows, feature)
        sess_lbl = (f" · {len(sessions)} session(s)" if sessions else "")
        loading = " · ⏳ loading…" if _is_warming(animal) else ""
        keep_poll = _is_warming(animal)
        status = (f"Loaded {len(rows)} evoked responses · {len(pts[0])} with "
                  f"{_label(feature)}{sess_lbl}{win_lbl}{loading}.")
        if not pts[0]:
            return (empty_fig(f"No {_label(feature)} values for {animal} "
                              f"in this selection."), "", status, sel,
                    not keep_poll)
        view = view_mode or "density"
        want_onsets = bool(event_overlays and "onsets" in event_overlays)
        events = (_onset_events_for_animal(config, animal)
                  if want_onsets or view == "onset" else None)
        stats = _trend_stats(pts, feature)
        if view == "onset":
            if not events:
                return (empty_fig(
                    f"No scored seizure onsets for {animal} — the onset-aligned "
                    f"view needs flagged BHZ events."),
                    "", status, sel, not keep_poll)
            fig = _build_onset_aligned(pts, feature, animal, events,
                                       onset_lookback,
                                       "log" in (onset_logtime or []))
            return (fig, stats, status, sel, not keep_poll)
        if view == "scatter":
            fig = _build_feature_scatter(pts, feature, animal, overlays or [],
                                         roll_window, events)
            return (fig, stats, status, sel, not keep_poll)
        fig, payload = _build_timeline_density(pts, feature, animal,
                                               overlays or [], roll_window,
                                               events)
        _PLOT_CACHE[_plot_key(sel)] = payload
        while len(_PLOT_CACHE) > _PLOT_CACHE_MAX:
            _PLOT_CACHE.popitem(last=False)
        return (fig, stats, status, sel, not keep_poll)

    # Zoom-rebin: on pan/zoom of the density timeline, re-histogram the cached
    # epochs to the visible x-range (constant pixel grid, finer time bins as you
    # zoom). Other views and non-range relayout events are ignored.
    @app.callback(
        Output("chronic-feature-plot", "figure", allow_duplicate=True),
        Input("chronic-feature-plot", "relayoutData"),
        State("chronic-selection", "data"),
        State("chronic-view-mode", "value"),
        prevent_initial_call=True,
    )
    def _zoom_rebin(relayout, sel, view_mode):
        from dash import no_update
        if not relayout or not sel or (view_mode or "density") != "density":
            return no_update
        payload = _PLOT_CACHE.get(_plot_key(sel))
        if not payload:
            return no_update
        x0, x1, is_reset = parse_relayout(relayout)
        if not is_reset and (x0 is None or x1 is None):
            return no_update
        ts0 = ts1 = None
        if not is_reset:
            ts0, ts1 = _relayout_ts(x0), _relayout_ts(x1)
            if ts0 is None or ts1 is None or ts1 <= ts0:
                return no_update
        return _timeline_density_fig(
            payload["secs"], payload["vals"], payload["iso"],
            payload["feature"], payload["animal"], payload["overlays"],
            payload["roll_window"], payload["events"], ts0, ts1)

    # Live load feedback: while an off-thread load runs, update progress; when
    # it finishes, stop polling + bump ▶ Plot so the data plots itself.
    # (_update is the sole enabler of the poll -- see its disabled output.)
    @app.callback(
        Output("chronic-status", "children", allow_duplicate=True),
        Output("chronic-warm-poll", "disabled", allow_duplicate=True),
        Output("chronic-load-btn", "n_clicks", allow_duplicate=True),
        Input("chronic-warm-poll", "n_intervals"),
        State("chronic-animal-dropdown", "value"),
        State("chronic-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _warm_poll(_n, animal, load_clicks):
        from dash import no_update
        if not animal:
            return no_update, True, no_update
        if _is_warming(animal):
            return _warm_status_text(animal), False, no_update
        # Load finished: stop polling + bump ▶ Plot so the data plots itself.
        return ("Loaded — plotting.", True, int(load_clicks or 0) + 1)

    # Export: download the current selection's per-epoch features as CSV.
    @app.callback(
        Output("chronic-export-dl", "data"),
        Output("chronic-status", "children", allow_duplicate=True),
        Input("chronic-export-btn", "n_clicks"),
        State("chronic-animal-dropdown", "value"),
        State("chronic-session-dropdown", "value"),
        State("chronic-config", "data"),
        prevent_initial_call=True,
    )
    def _export(_n, animal, sessions, cfg_dict):
        from dash import no_update
        if not animal:
            return no_update, "Pick an animal, then ⬇ Export CSV."
        try:
            rows, note = _export_rows(animal, sessions or None, cfg_dict)
        except Exception as e:  # noqa: BLE001 -- surface, never crash UI
            return no_update, f"Export failed: {e}"
        if not rows:
            return no_update, (f"No features to export for {animal}. "
                               f"Preview or ▶ Plot first.")
        fname = f"{animal}_evoked_features.csv"
        return (dict(content=_rows_to_csv(rows), filename=fname),
                f"Exported {len(rows)} epoch rows to {fname}.{note}")

    # One lazy compute callback per expandable section (factory-registered).
    for key, _title, target, prop in _SECTIONS:
        _register_section(app, key, target, prop)

    # Rhythm section gets its own callback (not the generic factory) so the
    # plot recomputes live when the period override changes, not only on
    # expand/collapse.
    @app.callback(
        Output("chronic-rhythm-plot", "figure"),
        Output("chronic-rhythm-wrap", "style"),
        Input("chronic-rhythm-btn", "n_clicks"),
        Input("chronic-rhythm-period", "value"),
        State("chronic-selection", "data"),
        State("chronic-rhythm-wrap", "style"),
        prevent_initial_call=True,
    )
    def _rhythm(_n, period, sel, style):
        from dash import no_update
        trig = callback_context.triggered_id
        style = dict(style or {})
        if trig == "chronic-rhythm-btn":
            if style.get("display", "none") != "none":
                style["display"] = "none"          # collapse; keep cached fig
                return no_update, style
            style["display"] = "block"
        if style.get("display", "none") == "none":  # period changed while hidden
            return no_update, no_update
        if not sel or not sel.get("animal"):
            return empty_fig("Click ▶ Plot first"), style
        try:
            rows, _means, _ = _query_for_selection(sel, need_means=False)
            fig = _build_rhythm(rows, sel.get("feature") or _DEFAULT_FEATURE,
                                period)
        except Exception as e:  # noqa: BLE001 -- never crash the UI
            fig = empty_fig("Couldn't compute rhythm", hint=str(e))
        return fig, style


# --------------------------------------------------------------------- #
#  Lazy section plumbing
# --------------------------------------------------------------------- #

# In-memory recompute cache: (animal, hours, sessions, config-hash) -> rows.
# Bounded; a recompute over thousands of epochs is a few seconds, so re-plots
# / lazy sections of the same selection are instant.
_RECOMPUTE_CACHE: "dict[tuple, list]" = {}
_RECOMPUTE_MAX = 6


def _config_key(cfg_dict) -> tuple:
    f = ef.FeatureConfig.from_dict(cfg_dict)
    return tuple(getattr(f, k) for k in f.__dataclass_fields__)


def _recompute_key(animal, sessions, cfg_dict) -> tuple:
    """Key for the in-memory custom-config row store (sidecars cover the
    default config; custom Configure settings are transient, kept in RAM)."""
    return (animal, tuple(sorted(sessions or ())), _config_key(cfg_dict))


def _filter_by_hours(rows: list, hours) -> list:
    """Keep rows whose recording is within the last *hours* (0/None = all).
    Sidecars hold a recording's full history, so the Time Range is applied
    here at read time rather than in a SQL WHERE."""
    if not hours:
        return rows
    from datetime import timedelta
    cutoff = (datetime.now() - timedelta(hours=float(hours))).isoformat()
    return [r for r in rows if (r.get("rec_dt") or "") > cutoff]


def _abs_dt_iso(rec_iso, seconds) -> str:
    """recording ISO + stim offset seconds -> ISO (mirrors evoked_output)."""
    if not rec_iso:
        return ""
    if seconds is None:
        return rec_iso
    try:
        from datetime import timedelta
        return (_parse_iso(rec_iso) + timedelta(seconds=float(seconds))
                ).isoformat()
    except (ValueError, TypeError):
        return rec_iso


def _recording_means_for_selection(animal: str, files: list) -> list:
    """Per-recording mean/std evoked waveform for the overlay, computed on
    demand from at most ``_MAX_WAVEFORMS`` evenly-sampled recordings (reading
    their traces straight from disk -- the overlay is a lazy section)."""
    if not files:
        return []
    step = max(1, len(files) // _MAX_WAVEFORMS)
    out: list = []
    for fp in files[::step]:
        rec_iso = (parse_recording_dt(fp) or datetime.min).isoformat()
        chans = read_file_evoked(fp, only_animals=[animal])
        for ch, rec in chans.items():
            a, _e = split_animal_electrode(ch)
            if a != animal or not is_animal_channel(ch):
                continue
            tr, tms = rec.get("traces"), rec.get("time_ms")
            if tr is None or tms is None or len(tr) < 1 or tms.size < 2:
                continue
            t, m, s = _decimate3(np.asarray(tms), tr.mean(axis=0),
                                 tr.std(axis=0))
            out.append({"channel": ch, "rec_dt": rec_iso,
                        "n_epochs": int(tr.shape[0]),
                        "time_axis": t, "mean_trace": m, "std_trace": s})
    return out


def _decimate3(t, m, s, target: int = 1000):
    """Stride-decimate three aligned arrays to <= *target* points (lists)."""
    n = len(t)
    if n <= target:
        return t.tolist(), m.tolist(), s.tolist()
    step = (n // target) + 1
    return t[::step].tolist(), m[::step].tolist(), s[::step].tolist()


def _query_for_selection(sel: dict, need_means: bool = True):
    """Rows + per-recording means + window-label for a saved selection dict,
    read from the evokedOutput feature sidecars (default config) or the
    in-memory recompute store (custom Configure). No sqlite cache: a selection
    with no sidecars yet reads empty and the caller kicks a background load."""
    assert isinstance(sel, dict), "selection must be a dict"
    animal = sel.get("animal")
    assert animal, "selection animal required"
    hours = sel.get("hours") or None
    sessions = sel.get("sessions") or None
    cfg = ef.FeatureConfig.from_dict(sel.get("config"))
    files = _selection_files(animal, sessions)
    if cfg.is_passthrough():
        rows: list = []
        for fp in files:
            sc = read_feature_sidecar(fp, animal)
            if sc:
                rows.extend(sc)
        rows.sort(key=lambda r: r.get("abs_dt") or "")
    else:
        rows = _RECOMPUTE_CACHE.get(
            _recompute_key(animal, sessions, sel.get("config"))) or []
    rows = _filter_by_hours(rows, hours)
    means = (_recording_means_for_selection(animal, files)
             if need_means else [])
    return _apply_window(rows, means, sel.get("win_hours"), sel.get("scroll"))


def _build_section(key: str, rows, means, sel: dict, corr_mode):
    """Dispatch one expandable section to its pure builder."""
    feature = sel.get("feature") or _DEFAULT_FEATURE
    animal = sel.get("animal") or "?"
    if key == "rec":
        return _build_recording_trend(rows, feature)
    if key == "wave":
        return _build_waveform_overlay(means, animal)
    if key == "corr":
        return _build_stim_corr(rows, corr_mode or _CORR_MODES[0][0])
    if key == "circ":
        return _build_circadian(rows, feature)
    if key == "table":
        return _stats_table_rows(rows, feature)
    return empty_fig("?")


def _register_section(app, key: str, target: str, prop: str) -> None:
    """Register a lazy expandable section: toggle button expands the
    container and computes its figure/table on demand; re-click collapses
    (figure stays cached, no recompute)."""

    @app.callback(
        Output(target, prop),
        Output(f"chronic-{key}-wrap", "style"),
        Input(f"chronic-{key}-btn", "n_clicks"),
        State("chronic-selection", "data"),
        State(f"chronic-{key}-wrap", "style"),
        State("chronic-corr-mode", "value"),
        prevent_initial_call=True,
    )
    def _toggle(_n, sel, style, corr_mode, _key=key, _prop=prop):
        from dash import no_update
        style = dict(style or {})
        if style.get("display", "none") != "none":
            style["display"] = "none"            # collapse; keep cached output
            return no_update, style
        style["display"] = "block"
        empty = ([] if _prop == "data" else empty_fig("Click ▶ Plot first"))
        if not sel or not sel.get("animal"):
            return empty, style
        try:
            rows, means, _ = _query_for_selection(
                sel, need_means=(_key == "wave"))
            out = _build_section(_key, rows, means, sel, corr_mode)
        except Exception as e:  # noqa: BLE001 -- never crash the UI
            out = [] if _prop == "data" else empty_fig("Couldn't compute",
                                                       hint=str(e))
        return out, style


_PHASE_LABEL = {"opening": "opening", "reading": "reading file",
                "computing features": "computing features",
                "saving": "saving", "done": "finishing"}


def _warm_status_text(animal: str) -> str:
    p = warm_progress(animal) or {}
    total = int(p.get("total") or 0)
    if not total:
        return f"⏳ Loading {animal} from evokedOutput…"
    done = min(int(p.get("done") or 0), total)
    raw_phase = p.get("phase") or ""
    phase = _PHASE_LABEL.get(raw_phase, raw_phase)   # unknown -> pass through
    # The session/protocol label = filename prefix before the channel list.
    raw = (p.get("path") or "").replace("\\", "/").rsplit("/", 1)[-1]
    sess = raw.split("__", 1)[0] if raw else ""
    bits = [f"⏳ Loading {animal} — file {max(done, 1)} of {total}"]
    if phase:
        bits.append(phase)
    if sess:
        bits.append(sess)
    return "  ·  ".join(bits) + "…"


# --------------------------------------------------------------------- #
#  Data shaping + builders (pure)
# --------------------------------------------------------------------- #

def _latest_preview_file(animal: str, sessions) -> str | None:
    """Most recent ``*_evoked.mat`` in evokedOutput for *animal* (optionally
    restricted to *sessions*). Used by the no-cache preview path."""
    assert animal, "animal required"
    sset = set(sessions) if sessions else None
    cands: list = []
    for i, f in enumerate(list_evoked_files(_EVOKED_DIR)):
        assert i < 1000000, "evoked file scan runaway"
        if animal not in animals_in_filename(f):
            continue
        if sset is not None and parse_session(f) not in sset:
            continue
        cands.append((parse_recording_dt(f) or datetime.min, f))
    if not cands:
        return None
    return max(cands, key=lambda t: t[0])[1]


def _preview_rows(path: str, animal: str, cfg) -> list:
    """Per-epoch feature rows for ONE evoked file (delegates to the shared
    pure ``compute_feature_rows``; honours the tab's expensive-features flag).
    Same row shape as the sidecar payload so plot/section code is unchanged."""
    assert path and animal, "path and animal required"
    return compute_feature_rows(path, animal, cfg, _EXPENSIVE_ENABLED)


def _build_preview(animal, sessions, feature, cfg_dict, overlays,
                   roll_window):
    """(figure, status, trend_stats) for a one-recording preview read
    directly from evokedOutput -- no sqlite cache, no background warm."""
    assert animal, "animal required"
    cfg = ef.FeatureConfig.from_dict(cfg_dict)
    path = _latest_preview_file(animal, sessions)
    if not path:
        return (empty_fig(f"No evokedOutput file found for {animal} in this "
                          f"selection."), "Nothing to preview.", "")
    fname = os.path.basename(path)
    rows = _preview_rows(path, animal, cfg)
    if not rows:
        return (empty_fig(f"{fname} has no readable epochs for {animal}."),
                f"Preview · {fname} · no epochs.", "")
    # Persist this recording's features beside the source .mat (only for the
    # default feature config -- a custom Configure window is a transient view,
    # not the canonical export). Best-effort; never break the preview.
    if cfg.is_passthrough():
        try:
            write_feature_sidecar(path, animal, rows)
        except Exception as e:  # noqa: BLE001
            logger.debug("sidecar write failed for %s: %s", fname, e)
    pts = _feature_points(rows, feature)
    status = (f"Preview · {fname} · {len(rows)} epochs from 1 recording · "
              f"{len(pts[0])} with {_label(feature)}. ▶ Plot loads the full "
              f"chronic history.")
    if not pts[0]:
        return (empty_fig(f"No {_label(feature)} values in this preview."),
                status, "")
    # One recording is sparse -> show individual epochs (scatter), not density.
    fig = _build_feature_scatter(pts, feature, animal, overlays, roll_window,
                                 None)
    fig.update_layout(title=f"PREVIEW (1 recording, {len(rows)} epochs) — "
                            f"{_label(feature)} · {fname}")
    return (fig, status, _trend_stats(pts, feature))


# Cold export (no warm cache, no sidecars) computes features file-by-file in
# the request thread; cap it so an unwarmed full-animal export can't hang the
# UI. Sidecars (written by Preview) + a warmed cache make larger exports
# instant, so this bound only bites a truly cold, broad selection.
_EXPORT_MAX_COMPUTE = 12


def _selection_files(animal: str, sessions) -> list:
    """evokedOutput files for *animal* (optionally restricted to *sessions*),
    oldest recording first."""
    assert animal, "animal required"
    sset = set(sessions) if sessions else None
    out: list = []
    for i, f in enumerate(list_evoked_files(_EVOKED_DIR)):
        assert i < 1000000, "evoked file scan runaway"
        if animal not in animals_in_filename(f):
            continue
        if sset is not None and parse_session(f) not in sset:
            continue
        out.append(f)
    return sorted(out, key=lambda f: parse_recording_dt(f) or datetime.min)


def _export_rows(animal: str, sessions, cfg_dict) -> tuple:
    """Per-epoch feature rows for the selection -> (rows, note).

    Cache first (instant when ▶ Plot warmed it). Otherwise sidecar-first per
    file (Preview writes those), computing+writing at most
    ``_EXPORT_MAX_COMPUTE`` missing files so a cold export never hangs.
    """
    assert animal, "animal required"
    cfg = ef.FeatureConfig.from_dict(cfg_dict)
    cached = _query_for_selection({"animal": animal, "sessions": sessions,
                                   "config": cfg_dict}, need_means=False)[0]
    if cached:
        return cached, ""
    files = _selection_files(animal, sessions)
    rows: list = []
    computed = 0
    skipped = 0
    for fp in files:
        sc = read_feature_sidecar(fp, animal) if cfg.is_passthrough() else None
        if sc is None:
            if computed >= _EXPORT_MAX_COMPUTE:
                skipped += 1
                continue
            sc = _preview_rows(fp, animal, cfg)
            computed += 1
            if cfg.is_passthrough() and sc:
                try:
                    write_feature_sidecar(fp, animal, sc)
                except Exception:  # noqa: BLE001 -- export is best-effort
                    pass
        rows.extend(sc)
    rows.sort(key=lambda r: r.get("abs_dt") or "")
    note = ("" if not skipped else
            f" ({skipped} recording(s) skipped — ▶ Plot to warm the full "
            f"history, then re-export).")
    return rows, note


def _rows_to_csv(rows: list) -> str:
    """Per-epoch feature rows -> CSV text (metadata columns + every feature)."""
    import csv
    import io
    assert isinstance(rows, list), "rows must be a list"
    meta = ["channel", "electrode", "rec_dt", "session", "stim_time_sec",
            "abs_dt", "peak", "trough"]
    cols = meta + list(ef.ALL_COLUMNS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c) for c in cols])
    return buf.getvalue()


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


# Scored BHZ events change rarely; cache the full parse per CSV dir so the
# onset overlay doesn't re-read ~200 CSVs on every Plot click.
_SCORED_EVENTS_CACHE: dict = {"dir": None, "events": None}


def _datenum_to_dt(dn):
    """MATLAB datenum (days, epoch year 0) -> python datetime, or None."""
    if dn is None:
        return None
    try:
        from datetime import timedelta
        return (datetime.fromordinal(int(dn)) + timedelta(days=dn % 1)
                - timedelta(days=366))
    except (ValueError, OverflowError, TypeError):
        return None


def _onset_events_for_animal(config, animal: str) -> list:
    """Deduped flagged seizure onsets for *animal* from the BHZ scored CSVs,
    as ``[{dt, type, racine}]`` sorted by time. Best-effort: [] if the CSV
    dir is unreachable / past_events disabled (the overlay just shows nothing).
    """
    if not animal:
        return []
    try:
        from src.utils.past_events import (
            bhz_csv_dir, load_scored_events, dedup_events)
        d = bhz_csv_dir(config or {})
        if not d or not os.path.isdir(d):
            return []
        cache = _SCORED_EVENTS_CACHE
        if cache["dir"] != d:
            cache["dir"], cache["events"] = d, load_scored_events(d)
        evs = dedup_events([e for e in cache["events"]
                            if e.get("animal") == animal])
    except Exception as e:  # noqa: BLE001 -- overlay is optional
        logger.debug("onset events load failed: %s", e)
        return []
    out = []
    for e in evs:
        dt = _datenum_to_dt(e.get("peak_stamp"))
        if dt is not None:
            out.append({"dt": dt, "type": e.get("type"),
                        "racine": e.get("racine")})
    out.sort(key=lambda r: r["dt"])
    return out


def _build_feature_scatter(pts, feature, animal, overlays,
                            roll_window, events=None) -> go.Figure:
    """Per-epoch faded marker cloud + a rolling 10-90/25-75 percentile ribbon
    with a median line. Optional linear/quad/MA trend overlays + seizure-onset
    vlines. (The default density timeline is built by _build_timeline_density;
    this is the explicit Scatter view, where seeing individual epochs helps.)
    """
    iso, secs, vals = pts
    label = _label(feature)
    order = np.argsort(secs)            # time-ascending for the ribbon
    iso_s = np.asarray(iso)[order]
    secs_s = secs[order]
    vals_s = vals[order]
    fig = go.Figure()
    step = _stride(len(secs_s), _MAX_POINTS)
    fig.add_trace(go.Scattergl(
        x=iso_s[::step], y=vals_s[::step], mode="markers",
        marker=dict(size=3, color="#8a8d99", opacity=0.25),
        name="responses", hoverinfo="x+y"))
    mode_note = " (markers stride-sampled)" if step > 1 else ""
    _add_ribbon(fig, secs_s, vals_s, roll_window)
    _add_trend_overlays(fig, iso_s, secs_s, vals_s, overlays)
    n_onsets = _add_onset_lines(fig, events, secs_s, vals_s)
    onset_note = f" · {n_onsets} seizure onset(s)" if n_onsets else ""
    fig.update_layout(
        title=f"{label} per evoked response — {animal}{mode_note}{onset_note}",
        xaxis_title="Recording time", yaxis_title=label,
        height=460, hovermode="closest", showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=1.02,
                    yanchor="bottom"))
    return fig


# --------------------------------------------------------------------- #
#  Density timeline (zoom-rebinned) + onset-aligned heatmap
# --------------------------------------------------------------------- #

def _plot_key(sel: dict) -> tuple:
    """Stable signature of a selection -> key into _PLOT_CACHE, so the zoom
    relayout callback can fetch exactly the arrays the plot was built from."""
    assert isinstance(sel, dict), "selection must be a dict"
    return (sel.get("animal") or "", tuple(sorted(sel.get("sessions") or [])),
            sel.get("feature") or "", sel.get("hours") or 0,
            sel.get("win_hours") or 0, round(float(sel.get("scroll") or 0), 4),
            sel.get("roll_window") or 0, _config_key(sel.get("config")))


def _relayout_ts(v) -> float | None:
    """A relayout x-bound (epoch float, or a Plotly date string) -> POSIX
    timestamp, or None when it can't be parsed."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    dt = _parse_iso(str(v).replace(" ", "T"))
    return dt.timestamp() if dt is not None else None


def _pct_clip(vals, lo=1, hi=99) -> tuple[float, float]:
    """(low, high) feature bounds at the *lo*/*hi* percentiles, never degenerate
    -- so empty axis isn't wasted on a few outliers."""
    assert vals.size > 0, "need values to clip"
    ylo, yhi = (float(v) for v in np.percentile(vals, [lo, hi]))
    if yhi <= ylo:
        ylo, yhi = float(np.min(vals)), float(np.max(vals))
        if yhi <= ylo:
            yhi = ylo + 1.0
    return ylo, yhi


def _density_heatmap_trace(secs, vals, x0=None, x1=None):
    """A go.Heatmap of epoch density over [x0,x1] (default full range) on a
    fixed _DENSITY_NX x _DENSITY_NY grid. y clipped to the 1-99 pct of the
    visible points; empty bins transparent. Returns (trace|None, ylo, yhi)."""
    assert secs.size == vals.size, "secs/vals length mismatch"
    if secs.size == 0:
        return None, 0.0, 1.0
    x_lo = float(secs.min()) if x0 is None else float(x0)
    x_hi = float(secs.max()) if x1 is None else float(x1)
    if x_hi <= x_lo:
        return None, 0.0, 1.0
    m = (secs >= x_lo) & (secs <= x_hi)
    vis = vals[m]
    if vis.size < 2:
        return None, 0.0, 1.0
    ylo, yhi = _pct_clip(vis)
    hist, xe, ye = np.histogram2d(
        secs[m], vis, bins=[_DENSITY_NX, _DENSITY_NY],
        range=[[x_lo, x_hi], [ylo, yhi]])
    z = hist.T
    z[z == 0] = np.nan                  # transparent empty bins
    xc = [datetime.fromtimestamp(s).isoformat()
          for s in (xe[:-1] + xe[1:]) / 2.0]
    yc = (ye[:-1] + ye[1:]) / 2.0
    trace = go.Heatmap(
        x=xc, y=yc, z=z, colorscale="Viridis",
        colorbar=dict(title=dict(text="epochs", font=dict(size=9)),
                      thickness=10, len=0.7, x=1.005),
        hovertemplate="%{x}<br>%{y}<br>%{z} epochs<extra></extra>",
        name="density")
    return trace, ylo, yhi


def _add_median_line(fig, secs_sorted, vals_sorted, roll_window) -> None:
    """Just the rolling median (no ribbon) -- the heatmap already shows spread."""
    try:
        win = max(11, int(roll_window))
    except (TypeError, ValueError):
        win = 301
    res = _rolling_percentiles(secs_sorted, vals_sorted, win)
    if res is None:
        return
    t_iso, _p10, _p25, p50, _p75, _p90 = res
    fig.add_trace(go.Scatter(
        x=t_iso, y=p50, mode="lines", line=dict(color="#5e7ce2", width=2.5),
        name=f"rolling median (win {win})"))


def _timeline_density_fig(secs_s, vals_s, iso_s, feature, animal, overlays,
                          roll_window, events, x0=None, x1=None) -> go.Figure:
    """Density heatmap (rebinned to [x0,x1] when given) + median + trend +
    onset lines. Shared by the initial render and the zoom relayout callback."""
    assert secs_s.size == vals_s.size, "secs/vals length mismatch"
    label = _label(feature)
    fig = go.Figure()
    trace, _ylo, yhi = _density_heatmap_trace(secs_s, vals_s, x0, x1)
    note = ""
    if trace is not None:
        fig.add_trace(trace)
    else:
        note = " · no epochs in view"
    _add_median_line(fig, secs_s, vals_s, roll_window)
    _add_trend_overlays(fig, iso_s, secs_s, vals_s, overlays)
    y_ref = np.array([yhi if trace is not None else
                      (float(vals_s.max()) if vals_s.size else 0.0)])
    n_onsets = _add_onset_lines(fig, events, secs_s, y_ref)
    onset_note = f" · {n_onsets} seizure onset(s)" if n_onsets else ""
    fig.update_layout(
        title=f"{label} per evoked response — {animal} (density)"
              f"{onset_note}{note}",
        xaxis_title="Recording time", yaxis_title=label, height=460,
        hovermode="closest", showlegend=True,
        uirevision=f"{animal}:{feature}",
        legend=dict(font=dict(size=10), orientation="h", y=1.02,
                    yanchor="bottom"))
    if x0 is not None and x1 is not None:
        fig.update_xaxes(range=[datetime.fromtimestamp(x0).isoformat(),
                                datetime.fromtimestamp(x1).isoformat()])
    else:
        fig.update_xaxes(autorange=True)
    if trace is not None:
        fig.update_yaxes(range=[_ylo, yhi])
    return fig


def _build_timeline_density(pts, feature, animal, overlays, roll_window,
                            events):
    """Initial full-range density figure + a cache payload the zoom callback
    refilters in RAM. Returns (figure, payload)."""
    iso, secs, vals = pts
    order = np.argsort(secs)
    iso_s = np.asarray(iso)[order]
    secs_s = secs[order]
    vals_s = vals[order]
    fig = _timeline_density_fig(secs_s, vals_s, iso_s, feature, animal,
                                overlays, roll_window, events)
    payload = {"secs": secs_s, "vals": vals_s, "iso": iso_s,
               "feature": feature, "animal": animal, "overlays": overlays,
               "roll_window": roll_window, "events": events}
    return fig, payload


def _binned_median(x, y, edges):
    """Median of *y* in each [edges[i], edges[i+1]) bin (NaN for empties),
    returned at the bin centres -- a readable trend line over the heatmap."""
    n = edges.size - 1
    cx = (edges[:-1] + edges[1:]) / 2.0
    cy = np.full(n, np.nan)
    for i in range(n):
        assert i < 100000, "bin loop runaway"
        seg = y[(x >= edges[i]) & (x < edges[i + 1])]
        if seg.size:
            cy[i] = float(np.median(seg))
    return cx, cy


def _build_onset_aligned(pts, feature, animal, events, lookback_h,
                         log_time) -> go.Figure:
    """Heatmap of feature density vs time-BEFORE the nearest upcoming seizure
    onset, stacked across all of *animal*'s onsets. Each epoch is claimed by
    the nearest onset that follows it (the one it leads up to), within
    *lookback_h*; epochs after the last onset or beyond the window are dropped.
    Nearest-following assignment means clustered seizures never double-count an
    epoch. Optional log time axis fits seconds-to-hours in one frame."""
    assert events, "onset-aligned view needs events"
    _iso, secs, vals = pts
    assert secs.size == vals.size, "secs/vals length mismatch"
    onset_ts = np.sort(np.array([e["dt"].timestamp() for e in events], float))
    lb = float(lookback_h or _ONSET_DEFAULT_LOOKBACK_H)
    idx = np.searchsorted(onset_ts, secs, side="left")   # nearest following
    keep = idx < onset_ts.size
    hours_before = np.full(secs.shape, np.nan)
    hb = (onset_ts[np.clip(idx, 0, onset_ts.size - 1)] - secs) / 3600.0
    hours_before[keep] = hb[keep]
    sel = keep & (hours_before > 0) & (hours_before <= lb)
    x, y = hours_before[sel], vals[sel]
    if x.size < 5:
        return empty_fig(
            f"Only {int(x.size)} epoch(s) within {lb:g} h before an onset for "
            f"{animal} — not enough for the onset-aligned view (try a larger "
            f"lookback).")
    ylo, yhi = _pct_clip(y)
    if log_time:
        x = np.clip(x, 1.0 / 3600.0, None)               # floor at 1 s
        xe = np.logspace(np.log10(x.min()), np.log10(x.max()),
                         _DENSITY_NX + 1)
    else:
        xe = np.linspace(x.min(), x.max(), _DENSITY_NX + 1)
    ye = np.linspace(ylo, yhi, _DENSITY_NY + 1)
    hist, _xe, _ye = np.histogram2d(x, y, bins=[xe, ye])
    z = hist.T
    z[z == 0] = np.nan
    xc = (xe[:-1] + xe[1:]) / 2.0
    yc = (ye[:-1] + ye[1:]) / 2.0
    fig = go.Figure(go.Heatmap(
        x=xc, y=yc, z=z, colorscale="Viridis",
        colorbar=dict(title=dict(text="epochs", font=dict(size=9)),
                      thickness=10, len=0.7, x=1.005),
        hovertemplate="%{x:.3g} h before onset<br>%{y}<br>%{z} epochs"
                      "<extra></extra>", name="density"))
    mx, my = _binned_median(x, y, xe)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="lines",
                             line=dict(color="#5e7ce2", width=2.5),
                             name="binned median", connectgaps=False))
    label = _label(feature)
    fig.update_layout(
        title=f"{label} aligned to seizure onset — {animal} "
              f"({onset_ts.size} onsets · {int(x.size)} epochs ≤{lb:g} h "
              f"before)",
        xaxis_title="Hours before onset" + (" (log)" if log_time else ""),
        yaxis_title=label, height=460, hovermode="closest", showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=1.02,
                    yanchor="bottom"))
    fig.update_xaxes(autorange="reversed", type="log" if log_time else "linear")
    return fig


# Cap on how many onset vlines to draw -- above this the figure gets a shape
# per line and reads as a solid band; an animal with hundreds of events in the
# window is better served by the hover markers alone.
_MAX_ONSET_LINES = 200


def _add_onset_lines(fig, events, secs_sorted, vals_sorted) -> int:
    """Draw a dashed vertical line at each flagged seizure onset that falls
    inside the plotted time range, plus one hover/legend marker trace. Returns
    how many onsets were in range (0 when *events* is None/empty)."""
    if not events or secs_sorted.size == 0:
        return 0
    lo, hi = float(secs_sorted[0]), float(secs_sorted[-1])
    in_range = [e for e in events if lo <= e["dt"].timestamp() <= hi]
    if not in_range:
        return 0
    y_top = float(np.nanmax(vals_sorted))
    if len(in_range) <= _MAX_ONSET_LINES:
        for e in in_range:
            fig.add_vline(x=e["dt"].isoformat(),
                          line=dict(color="rgba(239,85,59,0.45)", width=1,
                                    dash="dot"))
    # Hover/legend markers at the top of the plot (always drawn, even when
    # there are too many for individual lines).
    fig.add_trace(go.Scatter(
        x=[e["dt"].isoformat() for e in in_range],
        y=[y_top] * len(in_range), mode="markers",
        marker=dict(color="#EF553B", size=7, symbol="triangle-down"),
        name="seizure onset",
        text=[f"{e['dt']:%Y-%m-%d %H:%M}"
              f"  {(e.get('type') or '?')}  Racine {e.get('racine')}"
              for e in in_range],
        hoverinfo="text"))
    return len(in_range)


def _add_ribbon(fig, secs_sorted, vals_sorted, roll_window) -> None:
    """Rolling 10/25/50/75/90 percentiles over a sliding count window."""
    try:
        win = int(roll_window)
    except (TypeError, ValueError):
        win = 301
    win = max(11, win)
    res = _rolling_percentiles(secs_sorted, vals_sorted, win)
    if res is None:
        return
    t_iso, p10, p25, p50, p75, p90 = res
    # 10-90 band (light), then 25-75 (darker), then the median line.
    _band(fig, t_iso, p10, p90, "rgba(94,124,226,0.13)", "10–90%")
    _band(fig, t_iso, p25, p75, "rgba(94,124,226,0.30)", "25–75%")
    fig.add_trace(go.Scatter(
        x=t_iso, y=p50, mode="lines", line=dict(color="#5e7ce2", width=2.5),
        name=f"rolling median (win {win})"))


def _band(fig, t_iso, lo, hi, color, name) -> None:
    fig.add_trace(go.Scatter(
        x=list(t_iso) + list(t_iso[::-1]),
        y=list(hi) + list(lo[::-1]), fill="toself", fillcolor=color,
        line=dict(width=0), name=name, hoverinfo="skip"))


def _rolling_percentiles(secs_sorted, vals_sorted, win, n_eval=400):
    """Evaluate 10/25/50/75/90 percentiles in a centered count window at
    up to *n_eval* evenly-spaced positions. Returns (iso_times, p10, p25,
    p50, p75, p90) or None when there aren't enough points."""
    n = vals_sorted.size
    if n < max(11, win // 2):
        return None
    half = win // 2
    lo_i, hi_i = half, n - half - 1
    if hi_i <= lo_i:
        lo_i, hi_i = 0, n - 1
    idxs = np.unique(np.linspace(lo_i, hi_i,
                                 min(n_eval, hi_i - lo_i + 1)).astype(int))
    qs = np.array([10, 25, 50, 75, 90])
    out = np.empty((idxs.size, 5))
    for k, i in enumerate(idxs):
        seg = vals_sorted[max(0, i - half):min(n, i + half + 1)]
        out[k] = np.percentile(seg, qs)
    t_iso = [datetime.fromtimestamp(s).isoformat() for s in secs_sorted[idxs]]
    return (t_iso, out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4])


def _add_trend_overlays(fig, iso, secs, vals, overlays) -> None:
    if secs.size < 3:
        return
    t = secs - secs[0]
    if "lin" in overlays:
        _add_polyline(fig, iso, t, vals, 1, "#ffffff", "linear")
    if "quad" in overlays and t.size >= 3:
        _add_polyline(fig, iso, t, vals, 2, "#ffd60a", "quadratic")
    if "ma" in overlays:
        ma = ef.rolling_centered(vals, _MA_WINDOW, "mean")
        fig.add_trace(go.Scattergl(x=iso, y=ma, mode="lines",
                                   line=dict(color="#30d158", width=1.5),
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
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    cv = (std / mean) if mean != 0 else float("nan")
    rng = float(np.max(vals) - np.min(vals))
    base = (f"mean {mean:.3g} · SD {std:.3g} · CV {cv:.2f} · "
            f"range {rng:.3g} · n {vals.size}")
    days = (secs - secs.min()) / 86400.0
    if np.ptp(days) == 0:          # all one instant -> no slope/r/p
        return base
    from scipy.stats import linregress
    lr = linregress(days, vals)
    return (f"slope {lr.slope:.3g}/day · r {lr.rvalue:.2f} · "
            f"p {lr.pvalue:.1e}  |  " + base)


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
    # linregress fails when every x is identical (degenerate stim amplitudes);
    # skip the fit line in that case rather than crash the panel.
    if np.ptp(xa) == 0:
        return
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


# --------------------------------------------------------------------- #
#  Rhythm / cycles: Lomb-Scargle periodogram + phase plot (Baud-style)
# --------------------------------------------------------------------- #

def _per_recording_series(rows, feature):
    """Per-recording mean of *feature* -> (timestamps, values), time-ascending.
    One robust value per recording is the right granularity for a days-scale
    cycle estimate (per-epoch values are noisier and unevenly spaced)."""
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        v = r.get(feature)
        if v is not None and r.get("rec_dt"):
            grouped.setdefault(r["rec_dt"], []).append(float(v))
    pairs = []
    for rd, vals in grouped.items():
        dt = _parse_iso(rd)
        if dt is not None:
            pairs.append((dt.timestamp(), float(np.mean(vals))))
    pairs.sort()
    ts = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    return ts, y


def _lomb_scargle(ts, y, pmin_d=0.25, pmax_cap=40.0, n=2000):
    """Normalized Lomb-Scargle power over a period grid (handles the irregular
    recording times). Returns (periods_d, power, dominant_period_d, span_d) or
    None when the span is too short / the series is flat."""
    assert ts.size == y.size, "ts/y length mismatch"
    span_d = (ts[-1] - ts[0]) / 86400.0 if ts.size > 1 else 0.0
    pmax = min(pmax_cap, span_d / 2.0)
    if pmax <= pmin_d:
        return None
    yz = y - y.mean()
    if yz.std() == 0:
        return None
    from scipy.signal import lombscargle
    t_h = (ts - ts[0]) / 3600.0
    periods = np.linspace(pmin_d, pmax, n)
    ang = 2 * np.pi / (periods * 24.0)         # rad per hour
    power = lombscargle(t_h, yz / yz.std(), ang, normalize=True)
    return periods, power, float(periods[int(np.argmax(power))]), span_d


def _phase_at_period(ts, y, period_d):
    """Phase angle (rad) of each sample within a *period_d*-day cycle, plus a
    modulation index in [0,1] = how strongly the feature locks to that phase."""
    ang = 2 * np.pi * (((ts - ts[0]) / 86400.0) % period_d) / period_d
    w = y - y.mean()
    c, s = float(np.sum(w * np.cos(ang))), float(np.sum(w * np.sin(ang)))
    denom = float(np.sum(np.abs(w))) or 1.0
    return ang, float(np.hypot(c, s) / denom)


def _phase_means(ang, y, nb=24):
    """Mean *y* in each of *nb* phase bins -> (theta_deg, value) at bin centres."""
    deg = np.degrees(ang)
    bidx = np.floor(deg / (360.0 / nb)).astype(int) % nb
    bx, by = [], []
    for b in range(nb):
        m = bidx == b
        if np.any(m):
            bx.append((b + 0.5) * (360.0 / nb))
            by.append(float(np.mean(y[m])))
    return bx, by


def _build_rhythm(rows, feature, period_override) -> go.Figure:
    """Lomb-Scargle periodogram (left) + polar phase plot at the dominant (or
    overridden) period (right). The 'is there rhythm' view: the spectrum finds
    the cycle, the phase plot shows the feature's modulation within it."""
    from plotly.subplots import make_subplots
    label = _label(feature)
    ts, y = _per_recording_series(rows, feature)
    if ts.size < 12:
        return empty_fig(f"Only {int(ts.size)} recording(s) with {label} — a "
                         f"rhythm estimate needs ≥12 over several days. Warm a "
                         f"longer span first.")
    ls = _lomb_scargle(ts, y)
    if ls is None:
        return empty_fig(f"{label}: span too short or series flat for a "
                         f"periodogram (need a few days of recordings).")
    periods, power, dom_p, span_d = ls
    try:
        p_over = float(period_override) if period_override else 0.0
    except (TypeError, ValueError):
        p_over = 0.0
    period = p_over if p_over > 0 else dom_p
    ang, mod = _phase_at_period(ts, y, period)
    bx, by = _phase_means(ang, y)
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.55, 0.45],
        specs=[[{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=(f"Lomb–Scargle periodogram · span {span_d:.0f} d",
                        f"Phase at {period:.2f} d · modulation {mod:.2f}"))
    fig.add_trace(go.Scatter(x=periods, y=power, mode="lines",
                             line=dict(color="#5e7ce2"), name="power"),
                  row=1, col=1)
    fig.add_vline(x=period, line=dict(color="#ff453a", dash="dash"),
                  row=1, col=1)
    if periods[0] <= 1.0 <= periods[-1]:
        fig.add_vline(x=1.0, line=dict(color="#30d158", dash="dot"),
                      row=1, col=1)          # circadian reference
    shift = -min(0.0, float(y.min()))
    fig.add_trace(go.Scatterpolar(
        theta=np.degrees(ang), r=y + shift, mode="markers",
        marker=dict(size=4, color=np.degrees(ang), colorscale="Turbo",
                    opacity=0.4), name="recordings"), row=1, col=2)
    if bx:
        fig.add_trace(go.Scatterpolar(
            theta=bx + [bx[0]], r=[v + shift for v in by] + [by[0] + shift],
            mode="lines", line=dict(color="#ff453a", width=2),
            name="phase mean"), row=1, col=2)
    over = f" · viewing {period:.2f} d" if abs(period - dom_p) > 1e-6 else ""
    fig.update_layout(
        height=420, showlegend=False,
        title=f"{label} rhythm — dominant period {dom_p:.2f} d{over}")
    fig.update_xaxes(title_text="Period (days)", row=1, col=1)
    fig.update_yaxes(title_text="LS power", row=1, col=1)
    return fig


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
