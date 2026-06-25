"""Evoked Waveforms tab -- two-row per channel view of the stim-artifact
window and the evoked LFP response, read straight from the MATLAB toolkit's
``evokedOutput`` folder (the same source as the Chronic Evoked Analyzer), NOT
the rolling monitor DB.

Pickers: animal -> session -> recording. Nothing is read until the user
clicks ``▶ Load`` (one recording's ``*_evoked.mat`` decodes the chosen
animal's ``evokedData``, which is heavy -- ~20 k samples x N epochs), so the
page stays responsive. Smoothing re-renders from the cached read.

Grid: one row per animal channel in the recording, two columns -- both
sliced from the SAME mean evoked trace (stim onset at t=0 ms), so no
separate stim-copy series is needed:

* Left column: stim-artifact window, x-zoomed to -1..1 ms.
* Right column: evoked response, x-zoomed to the configured analysis window
  (config.feature_analysis.analysis_start_ms / analysis_end_ms, default
  1..50 ms), with a translucent SEM band.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from plotly.subplots import make_subplots

from src.dashboard import file_browser as _fb
from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE, loading_icon
from src.dashboard.data_helpers import empty_fig, load_config
from src.db.store import Store
from src.utils.animal import is_animal_channel
from src.utils.evoked_output import (
    DEFAULT_EVOKED_DIR, animals_in_filename, list_animals, list_evoked_files,
    parse_recording_dt, parse_session, read_file_evoked,
    sessions_with_dt_for_animal,
)
from src.utils.filters import apply_filter

logger = logging.getLogger("qc_monitor.dashboard.waveforms")

# Stim-artifact / evoked overlay windows (ms, stim onset at t=0). The artifact
# window is hard-coded -- the alignment story doesn't vary by deployment; the
# evoked window comes from config so a Settings edit takes effect live.
_ARTIFACT_X_RANGE = (-1.0, 1.0)
_DEFAULT_EVOKED_WINDOW = (1.0, 50.0)

# Configured in register_callbacks(); layout() reads it on tab open.
_EVOKED_DIR = DEFAULT_EVOKED_DIR

# Last decoded recording, so changing only the smoothing re-renders without a
# re-read. Single entry: the tab shows one recording at a time.
_LAST_LOAD: dict = {"key": None, "channels": None}


_SMOOTH_STYLE = {"backgroundColor": "#262638", "color": "#f0f0f5",
                 "width": "80px"}
_LOAD_BTN_STYLE = {"width": "100%", "padding": "8px", "background": "#5e7ce2",
                   "color": "white", "border": "none", "fontWeight": "700",
                   "borderRadius": "6px", "cursor": "pointer"}
_BROWSE_BTN_STYLE = {"width": "100%", "padding": "8px", "background": "#2a2d3a",
                     "color": "#cfd0d6", "border": "1px solid #3a3d4a",
                     "borderRadius": "6px", "cursor": "pointer"}


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


def _session_options(animal: str | None) -> list[dict]:
    """Session options for *animal*, read from the evokedOutput filenames."""
    if not animal:
        return []
    try:
        dated = sessions_with_dt_for_animal(_EVOKED_DIR, animal)
        return [{"label": _session_label(r), "value": r["session"]}
                for r in dated]
    except Exception:  # noqa: BLE001 -- best-effort dropdown
        return []


def _recording_options(animal: str | None,
                       session: str | None) -> list[dict]:
    """``{label: recording timestamp, value: file path}`` for *animal* in
    *session*, oldest recording first (so the latest is the last option)."""
    if not animal or not session:
        return []
    out: list[tuple] = []
    for i, f in enumerate(list_evoked_files(_EVOKED_DIR)):
        assert i < 1000000, "evoked file scan runaway"
        if animal not in animals_in_filename(f):
            continue
        if parse_session(f) != session:
            continue
        dt = parse_recording_dt(f)
        out.append((dt, f))
    out.sort(key=lambda t: (t[0] is None, t[0]))
    return [{"label": (dt.strftime("%Y-%m-%d  %H:%M:%S") if dt else
                       os.path.basename(f)), "value": f}
            for dt, f in out]


def _latest_value(options: list[dict]):
    return options[-1]["value"] if options else None


def layout(store: Store, default: str | None = None):
    """Animal / session / recording pickers + a manual Load button + Graph.
    Pre-populates the latest recording of the first animal so the controls
    aren't empty, but renders nothing until ``▶ Load`` is clicked."""
    animals = list_animals(_EVOKED_DIR)
    animal0 = animals[0] if animals else None
    sess_opts = _session_options(animal0)
    sess0 = _latest_value(sess_opts)
    rec_opts = _recording_options(animal0, sess0)
    rec0 = _latest_value(rec_opts)

    return html.Div([
        html.H3("Evoked Waveforms",
                style={"color": "white", "marginBottom": "4px"}),
        html.Div("One row per animal channel for a recording (left = stim "
                 "artifact, right = evoked response), read from evokedOutput. "
                 "Pick animal / session / recording, then click ▶ Load.",
                 style={"color": "#a0a0b0", "fontSize": "12px",
                        "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Animal", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-animal-dropdown",
                    options=[{"label": a, "value": a} for a in animals],
                    value=animal0, style=DROPDOWN_STYLE,
                    className="dark-dropdown"),
            ], style={"flex": "1", "minWidth": "150px"}),
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-session-dropdown",
                    options=sess_opts, value=sess0, style=DROPDOWN_STYLE,
                    className="dark-dropdown"),
            ], style={"flex": "1.4", "minWidth": "240px"}),
            html.Div([
                html.Label("Recording", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-file-dropdown",
                    options=rec_opts, value=rec0, style=DROPDOWN_STYLE,
                    className="dark-dropdown"),
            ], style={"flex": "1.2", "minWidth": "220px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="waveform-smooth", type="number",
                          min=0, step=0.5, value=0, style=_SMOOTH_STYLE),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("📂 Browse…", id="wf-fb-open", n_clicks=0,
                            style=_BROWSE_BTN_STYLE,
                            title="Browse the server filesystem and pick a "
                                  "folder (its *_evoked.mat become the "
                                  "Recording list) or specific .mat files — "
                                  "instead of the configured evokedOutput "
                                  "folder."),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("▶ Load", id="waveform-load-btn", n_clicks=0,
                            style=_LOAD_BTN_STYLE,
                            title="Read the selected recording from "
                                  "evokedOutput and plot its per-channel "
                                  "waveforms. Nothing renders until you click "
                                  "this."),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "8px",
                  "flexWrap": "wrap", "alignItems": "flex-end"}),
        _fb.modal("wf"),
        html.Div(id="waveform-status",
                 style={"color": "#8a8d99", "fontSize": "11px",
                        "minHeight": "14px", "marginBottom": "6px"}),
        dcc.Loading(
            custom_spinner=loading_icon("Building waveforms…"),
            delay_show=150,
            overlay_style={"visibility": "visible", "opacity": 0.45},
            children=dcc.Graph(
                id="waveform-plot",
                figure=empty_fig("Pick animal / session / recording, then "
                                 "click ▶ Load", height=400)),
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the animal→session→recording picker cascade and the manual
    Load/smooth figure builder. Reads ``config.chronic_evoked`` for the
    evokedOutput dir and ``config.feature_analysis`` (live) for the window."""
    global _EVOKED_DIR
    ce_cfg = (config or {}).get("chronic_evoked", {}) or {}
    _EVOKED_DIR = ce_cfg.get("evoked_output_dir") or DEFAULT_EVOKED_DIR

    # File browser: pick a folder (its *_evoked.mat fill the Recording list)
    # or specific .mat files, instead of the configured evokedOutput folder.
    _fb.register(app, "wf", open_btn_id="wf-fb-open", exts=(".mat",),
                 initial_path=_EVOKED_DIR)

    @app.callback(
        Output("waveform-file-dropdown", "options", allow_duplicate=True),
        Output("waveform-file-dropdown", "value", allow_duplicate=True),
        Output("waveform-status", "children", allow_duplicate=True),
        Input("wf-fb-result", "data"),
        prevent_initial_call=True,
    )
    def _wf_browse_pick(result):
        from dash import no_update
        if not result:
            return no_update, no_update, no_update
        from src.dashboard import activity as _activity
        _activity.track(store, "file_browser",
                        "pick_folder" if result.get("path") else "pick_files",
                        result.get("path") or
                        f"{len(result.get('files') or [])} files")
        if result.get("path"):
            mats = _fb.list_dir(result["path"], exts=(".mat",))["files"]
            opts = [{"label": os.path.basename(f), "value": f} for f in mats]
            if not opts:
                return [], None, (f"No .mat files in {result['path']}.")
            return (opts, opts[0]["value"],
                    f"{len(opts)} .mat in {os.path.basename(result['path'])} "
                    f"— pick one + ▶ Load.")
        files = result.get("files") or []
        opts = [{"label": os.path.basename(f), "value": f} for f in files]
        if not opts:
            return no_update, no_update, no_update
        return (opts, opts[0]["value"],
                f"{len(opts)} file(s) selected — ▶ Load.")

    @app.callback(
        Output("waveform-session-dropdown", "options"),
        Output("waveform-session-dropdown", "value"),
        Input("waveform-animal-dropdown", "value"),
    )
    def _sessions(animal):
        opts = _session_options(animal)
        return opts, _latest_value(opts)

    @app.callback(
        Output("waveform-file-dropdown", "options"),
        Output("waveform-file-dropdown", "value"),
        Input("waveform-animal-dropdown", "value"),
        Input("waveform-session-dropdown", "value"),
    )
    def _recordings(animal, session):
        opts = _recording_options(animal, session)
        return opts, _latest_value(opts)

    @app.callback(
        Output("waveform-plot", "figure"),
        Output("waveform-status", "children"),
        Input("waveform-load-btn", "n_clicks"),
        Input("waveform-smooth", "value"),
        State("waveform-animal-dropdown", "value"),
        State("waveform-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _load_plot(_n, smooth_ms, _animal, file_path):
        # Smoothing edits should only re-render once something is loaded -- not
        # trigger a cold read before the user has clicked Load.
        triggered = callback_context.triggered_id
        if triggered == "waveform-smooth" and _LAST_LOAD["key"] is None:
            from dash import no_update
            return no_update, no_update
        if not file_path:
            return (empty_fig("Pick animal / session / recording, then ▶ Load",
                              height=400),
                    "Nothing selected.")
        if triggered == "waveform-load-btn":
            from src.dashboard import activity as _activity
            _activity.track(store, "waveforms", "load",
                            os.path.basename(file_path))
        try:
            channels = _read_recording(file_path)
        except Exception as e:  # noqa: BLE001 -- surface, never crash the UI
            logger.error("Evoked read error: %s", e, exc_info=True)
            return (empty_fig(f"Couldn't read recording: {e}", height=400),
                    f"Read error: {e}")
        ana_start, ana_end = _evoked_window()
        try:
            fig, n_ch, n_ep = _build_evoked_figure(
                channels, smooth_ms or 0, ana_start, ana_end)
        except Exception as e:  # noqa: BLE001
            logger.error("Waveform plot error: %s", e, exc_info=True)
            return (empty_fig(f"Plot error: {e}", height=400),
                    f"Plot error: {e}")
        fname = os.path.basename(file_path)
        status = (f"Loaded {fname} · {n_ch} channel(s) · up to {n_ep} "
                  f"epoch(s)/channel." if n_ch
                  else f"{fname}: no evoked traces in this recording.")
        return fig, status


# --------------------------------------------------------------------- #
#  evokedOutput read (cached) + figure builder
# --------------------------------------------------------------------- #

def _read_recording(file_path: str) -> dict:
    """Every animal channel for one recording (one row each in the figure),
    cached so a smoothing change doesn't re-decode the (heavy) evokedData."""
    assert file_path, "file_path required"
    if _LAST_LOAD["key"] != file_path:
        _LAST_LOAD["channels"] = read_file_evoked(file_path)
        _LAST_LOAD["key"] = file_path
    return _LAST_LOAD["channels"] or {}


def _evoked_window() -> tuple[float, float]:
    """Evoked analysis window (ms) from live config, default 1..50."""
    fa = (load_config() or {}).get("feature_analysis", {}) or {}
    lo = fa.get("analysis_start_ms", fa.get("window_start_ms",
                                            _DEFAULT_EVOKED_WINDOW[0]))
    hi = fa.get("analysis_end_ms", fa.get("window_end_ms",
                                          _DEFAULT_EVOKED_WINDOW[1]))
    return float(lo), float(hi)


def _build_evoked_figure(channels: dict, smooth_ms: float,
                         ana_start: float, ana_end: float):
    """One row per animal channel; left column = stim-artifact window
    (-1..1 ms), right column = evoked response (analysis window + SEM band),
    both sliced from the channel's mean evoked trace. Returns (figure,
    n_channels, max_epochs)."""
    chans = [(ch, rec) for ch, rec in sorted(channels.items())
             if rec.get("traces") is not None and is_animal_channel(ch)]
    if not chans:
        return empty_fig("No evoked traces in this recording",
                         height=400), 0, 0
    n_rows = len(chans)
    titles: list[str] = []
    for ch, _rec in chans:
        titles.append(f"Stim artifact — {ch} (−1 to 1 ms)")
        titles.append(f"Evoked — {ch} ({ana_start:g}–{ana_end:g} ms)")
    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes=False,
                        subplot_titles=titles, vertical_spacing=0.08,
                        horizontal_spacing=0.08)
    max_ep = 0
    for i, (ch, rec) in enumerate(chans):
        row = i + 1
        t = np.asarray(rec["time_ms"], dtype=np.float64).ravel()
        traces = rec["traces"]
        n_ep = int(traces.shape[0])
        max_ep = max(max_ep, n_ep)
        mean = traces.mean(axis=0)
        sem = (traces.std(axis=0, ddof=1) / np.sqrt(n_ep)
               if n_ep > 1 else np.zeros_like(mean))
        mean, sem = _smooth_trace(t, mean, sem, smooth_ms)
        _add_window_row(fig, row, 1, t, mean, None, ch, *_ARTIFACT_X_RANGE)
        _add_window_row(fig, row, 2, t, mean, sem, ch, ana_start, ana_end,
                        n_epochs=n_ep)
    fig.update_layout(
        height=max(300, 260 * n_rows),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    return fig, len(chans), max_ep


def _smooth_trace(t, mean, sem, smooth_ms):
    """Gaussian-smooth mean (+ SEM) in display-ms; sample rate recovered from
    the (uniform) time axis. No-op when smoothing is off / unavailable."""
    if not (smooth_ms and smooth_ms > 0 and len(t) > 1):
        return mean, sem
    try:
        dt_ms = float(t[1] - t[0])
        if dt_ms <= 0:
            return mean, sem
        fs = 1000.0 / dt_ms
        mean = apply_filter(np.asarray(mean, dtype=np.float32), fs,
                            smoothing_ms=smooth_ms)
        if sem is not None:
            sem = apply_filter(np.asarray(sem, dtype=np.float32), fs,
                               smoothing_ms=smooth_ms)
    except Exception as e:  # noqa: BLE001 -- smoothing is cosmetic
        logger.debug("Smoothing skipped: %s", e)
    return mean, sem


def _add_window_row(fig, row, col, t, mean, sem, name, x0, x1, *,
                    n_epochs=None) -> None:
    """Plot the mean evoked trace (+ optional SEM band) sliced to [x0, x1] ms
    into subplot (*row*, *col*), with a stim-onset marker and a tight y-range.
    Slicing keeps each SVG trace small -- raw traces are ~20 k samples over
    ±500 ms."""
    t = np.asarray(t, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    mask = (t >= x0) & (t <= x1)
    if not np.any(mask):
        return
    tw, mw = t[mask], mean[mask]
    sw = None
    if sem is not None:
        sem = np.asarray(sem, dtype=np.float64)
        if sem.shape == mean.shape:
            sw = sem[mask]
    if sw is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([tw, tw[::-1]]),
            y=np.concatenate([mw + sw, (mw - sw)[::-1]]),
            fill="toself", fillcolor="rgba(99,110,250,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=row, col=col)
    label = name if n_epochs is None else f"{name} (n={n_epochs})"
    fig.add_trace(go.Scatter(
        x=tw, y=mw, mode="lines", name=label,
        line=dict(color="#636EFA", width=2), showlegend=False),
        row=row, col=col)
    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"),
                  row=row, col=col)
    fig.update_xaxes(range=[x0, x1], title_text="ms", row=row, col=col)
    yv = mw if sw is None else np.concatenate([mw + sw, mw - sw])
    mn, mx = float(np.min(yv)), float(np.max(yv))
    pad = (mx - mn) * 0.1 if mx > mn else 0.001
    fig.update_yaxes(range=[mn - pad, mx + pad], row=row, col=col)
