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
                     _cfg_num("chronic-cfg-win-start", None, step=10), w="0 0 190px"),
                cell(html.Label("Window end (ms)", style=LABEL_STYLE),
                     _cfg_num("chronic-cfg-win-end", None, step=10)),
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
        State("chronic-window-hours", "value"),
        State("chronic-window-scroll", "value"),
        State("chronic-roll-window", "value"),
        State("chronic-config", "data"),
        prevent_initial_call=True,
    )
    def _update(_load, _refresh, _preview, animal, sessions, feature, hours,
                overlays, win_hours, scroll, roll_window, cfg_dict):
        if not animal:
            return (empty_fig("Select an animal, then ▶ Plot"),
                    "", "", None, True)
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
        return (_build_feature_scatter(pts, feature, animal, overlays or [],
                                       roll_window),
                _trend_stats(pts, feature), status, sel, not keep_poll)

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
    fig = _build_feature_scatter(pts, feature, animal, overlays, roll_window)
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


def _build_feature_scatter(pts, feature, animal, overlays,
                            roll_window) -> go.Figure:
    """Every evoked response (faded markers) + a sliding-window rolling
    median line with two percentile ribbons: 10-90 (light) and 25-75
    (darker). Optional linear/quad trend overlays."""
    iso, secs, vals = pts
    label = _label(feature)
    order = np.argsort(secs)            # time-ascending for the ribbon
    iso_s = np.asarray(iso)[order]
    secs_s = secs[order]
    vals_s = vals[order]
    step = _stride(len(secs_s), _MAX_POINTS)
    fig = go.Figure()
    # Faded raw responses so the ribbon reads on top.
    fig.add_trace(go.Scattergl(
        x=iso_s[::step], y=vals_s[::step], mode="markers",
        marker=dict(size=3, color="#8a8d99", opacity=0.25),
        name="responses", hoverinfo="x+y"))
    _add_ribbon(fig, secs_s, vals_s, roll_window)
    _add_trend_overlays(fig, iso_s, secs_s, vals_s, overlays)
    note = " (markers stride-sampled)" if step > 1 else ""
    fig.update_layout(
        title=f"{label} per evoked response — {animal}{note}",
        xaxis_title="Recording time", yaxis_title=label,
        height=460, hovermode="closest", showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=1.02,
                    yanchor="bottom"))
    return fig


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
