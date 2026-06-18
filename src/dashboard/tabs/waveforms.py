"""Evoked Waveforms tab -- two-row per channel view of stim artifact
+ evoked LFP for the latest evoked waveform of each LFP channel in a
chosen file.

Layout: session dropdown + file dropdown + smoothing input + Graph.
The session-dropdown change refills the file-dropdown (callback A).
The file-dropdown / smoothing change rebuilds the figure (callback B).

The per-channel layout is two stacked subplots:

* Row A: stim copy (orange) overlaid with LFP (blue), x-zoomed to
  -1..1 ms around stim onset. Shows whether the stim artifact bleeds
  into the LFP and lets the reviewer eyeball the blanking decision.
* Row B: LFP evoked response zoomed to the configured analysis
  window (from config.feature_analysis.analysis_start_ms /
  analysis_end_ms). SEM band rendered as a translucent fill.

Lifted out of ``src/dashboard/app.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from src.dashboard.components import DROPDOWN_STYLE, LABEL_STYLE, loading_icon
from src.dashboard.data_helpers import (
    channel_map, default_session, empty_fig, load_config,
    processed_files_for_session, session_dropdown_options,
)
from src.db.store import Store
from src.utils.filters import apply_filter

logger = logging.getLogger("qc_monitor.dashboard.waveforms")

# Stim-artifact / LFP overlay window. Hard-coded because the
# alignment story (stim onset at t=0 ms) doesn't vary by deployment.
_ARTIFACT_X_RANGE = (-1.0, 1.0)


def layout(store: Store, default: str | None = None):
    """Build the tab layout. The file dropdown is pre-populated for
    the default session so the initial render doesn't show empty."""
    session_options = session_dropdown_options(store)
    initial_session = default_session(store, hint=default)

    file_options: list[dict] = []
    default_file = None
    if initial_session:
        files = processed_files_for_session(store, initial_session)
        file_options = [{"label": f["chunk_datetime"],
                          "value": f["id"]} for f in files]
        if file_options:
            # Latest file (rows are oldest-first).
            default_file = file_options[-1]["value"]

    return html.Div([
        html.H3("Evoked Waveforms",
                style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-session-dropdown",
                    options=session_options,
                    value=initial_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("File", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-file-dropdown",
                    options=file_options,
                    value=default_file,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="waveform-smooth", type="number",
                          min=0, step=0.5, value=0,
                          style={"backgroundColor": "#262638",
                                 "color": "#f0f0f5",
                                 "width": "80px"}),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "16px",
                  "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Loading(
            custom_spinner=loading_icon("Building waveforms…"),
            delay_show=150,
            overlay_style={"visibility": "visible", "opacity": 0.45},
            children=dcc.Graph(id="waveform-plot"),
        ),
    ])


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire the file-list refresh callback and the figure-building
    callback. The figure builder re-reads ``config.feature_analysis``
    from disk on every plot so a Settings-tab edit to the analysis
    window takes effect without a restart."""

    @app.callback(
        [Output("waveform-file-dropdown", "options"),
         Output("waveform-file-dropdown", "value")],
        [Input("waveform-session-dropdown", "value")],
    )
    def update_waveform_file_list(session_dir):
        if not session_dir:
            return [], None
        files = processed_files_for_session(store, session_dir)
        options = [{"label": f["chunk_datetime"], "value": f["id"]}
                   for f in files]
        latest = options[-1]["value"] if options else None
        return options, latest

    @app.callback(
        Output("waveform-plot", "figure"),
        [Input("waveform-file-dropdown", "value"),
         Input("waveform-smooth", "value")],
        [State("waveform-session-dropdown", "value")],
    )
    def update_waveform_plot(file_id, smooth_ms, session_dir):
        if not file_id:
            return empty_fig("Select a file", height=400)
        try:
            waveforms = store.get_evoked_waveform_by_file(file_id)
        except Exception as e:
            logger.error("Waveform query error: %s", e, exc_info=True)
            return empty_fig(f"Error: {e}", height=400)
        if not waveforms:
            return empty_fig("No evoked waveforms for this file",
                              height=400)
        try:
            return _build_waveform_figure(
                waveforms, store, session_dir,
                smooth_ms=smooth_ms or 0,
            )
        except Exception as e:
            logger.error("Waveform plot error: %s", e, exc_info=True)
            return empty_fig(f"Plot error: {e}", height=400)


# --------------------------------------------------------------------- #
#  Figure builder
# --------------------------------------------------------------------- #

def _build_waveform_figure(waveforms: list[dict], store: Store,
                            session_dir: str | None,
                            smooth_ms: float = 0.0):
    """Build the per-channel, two-row figure from the latest evoked
    waveform of each LFP channel. *waveforms* is one row per
    (file_id, channel)."""
    cfg = load_config()
    fa = cfg.get("feature_analysis", {})
    ana_start = fa.get("analysis_start_ms",
                        fa.get("window_start_ms", 2))
    ana_end = fa.get("analysis_end_ms",
                      fa.get("window_end_ms", 50))

    # Group: keep the last waveform per channel (rows arrive in
    # insertion order, so the last is the most recent rev for that
    # channel + version).
    latest_per_ch: dict[int, dict] = {}
    for wf in waveforms:
        latest_per_ch[wf.get("channel", 0)] = wf

    ch_map = channel_map(store, session_dir) if session_dir else {}
    lfp_chs = sorted(latest_per_ch.keys())
    n_rows = len(lfp_chs) * 2
    if n_rows == 0:
        return empty_fig("No waveform data", height=400)

    titles: list[str] = []
    for ch in lfp_chs:
        name = (latest_per_ch[ch].get("channel_name")
                or ch_map.get(ch, {}).get("name", f"Ch{ch}"))
        titles.append(f"Stim Artifact vs {name} (-1 to 1 ms)")
        titles.append(f"Evoked: {name} ({ana_start}-{ana_end} ms)")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=False,
        subplot_titles=titles, vertical_spacing=0.06,
    )

    row = 1
    for ch in lfp_chs:
        wf = latest_per_ch[ch]
        name = wf.get("channel_name", f"Ch{ch}")
        t = wf["time_axis_ms"]
        m = wf["mean_trace"]
        s = wf.get("sem_trace")
        stim_tr = wf.get("stim_mean_trace")
        n_ep = wf.get("n_epochs", 0)

        # Optional Gaussian smoothing in display-ms. Applied to mean,
        # SEM, and stim mean so the band stays consistent.
        m, s, stim_tr = _maybe_smooth(t, m, s, stim_tr, smooth_ms)

        _add_artifact_row(fig, row, t, m, stim_tr, name,
                           is_first=(row == 1))
        _add_evoked_row(fig, row + 1, t, m, s, name, n_ep,
                         ana_start, ana_end)
        row += 2

    fig.update_layout(
        height=250 * n_rows,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1),
    )
    return fig


def _maybe_smooth(t, mean_tr, sem_tr, stim_tr, smooth_ms):
    """Apply Gaussian smoothing to mean / SEM / stim traces if the
    user asked for it. Recovers a sample rate from the first two
    time-axis samples; .mat sources are always uniformly sampled."""
    if not (smooth_ms and smooth_ms > 0 and mean_tr and len(t) > 1):
        return mean_tr, sem_tr, stim_tr
    try:
        dt_ms = float(t[1] - t[0])
        if dt_ms <= 0:
            return mean_tr, sem_tr, stim_tr
        fs_proxy = 1000.0 / dt_ms
        mean_tr = apply_filter(
            np.asarray(mean_tr, dtype=np.float32),
            fs_proxy, smoothing_ms=smooth_ms).tolist()
        if sem_tr and len(sem_tr) == len(mean_tr):
            sem_tr = apply_filter(
                np.asarray(sem_tr, dtype=np.float32),
                fs_proxy, smoothing_ms=smooth_ms).tolist()
        if stim_tr and len(stim_tr) > 1:
            stim_tr = apply_filter(
                np.asarray(stim_tr, dtype=np.float32),
                fs_proxy, smoothing_ms=smooth_ms).tolist()
    except Exception as e:
        logger.debug("Smoothing skipped: %s", e)
    return mean_tr, sem_tr, stim_tr


def _add_artifact_row(fig, row, t, mean_tr, stim_tr, name,
                       *, is_first: bool) -> None:
    """Row A: stim copy (orange) overlaid on LFP (blue), x-zoomed to
    the stim window. Y-range autoscaled to the data inside that
    window so a flat post-onset LFP doesn't get crushed by a huge
    stim spike."""
    x0, x1 = _ARTIFACT_X_RANGE
    yvals: list[float] = []
    if stim_tr and len(stim_tr) > 0:
        stim_t = t[:len(stim_tr)] if len(stim_tr) <= len(t) else t
        fig.add_trace(go.Scatter(
            x=stim_t, y=stim_tr, mode="lines",
            name="StimCopy",
            line=dict(color="#FFA15A", width=2),
            showlegend=is_first,
        ), row=row, col=1)
        yvals += [v for tv, v in zip(stim_t, stim_tr)
                  if x0 <= tv <= x1]
    fig.add_trace(go.Scatter(
        x=t, y=mean_tr, mode="lines",
        name=name, line=dict(color="#636EFA", width=2),
        showlegend=is_first,
    ), row=row, col=1)
    yvals += [v for tv, v in zip(t, mean_tr) if x0 <= tv <= x1]
    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"),
                   row=row, col=1)
    fig.update_xaxes(range=[x0, x1], title_text="ms", row=row, col=1)
    if yvals:
        mn, mx = min(yvals), max(yvals)
        pad = (mx - mn) * 0.1 if mx > mn else 0.001
        fig.update_yaxes(range=[mn - pad, mx + pad], row=row, col=1)


def _add_evoked_row(fig, row, t, mean_tr, sem_tr, name,
                     n_epochs, ana_start, ana_end) -> None:
    """Row B: LFP evoked response zoomed to the analysis window.
    SEM rendered as a translucent fill band so the noisy chunks are
    obviously noisy without dominating the trace."""
    if sem_tr and len(sem_tr) == len(mean_tr):
        upper = [mv + sv for mv, sv in zip(mean_tr, sem_tr)]
        lower = [mv - sv for mv, sv in zip(mean_tr, sem_tr)]
        fig.add_trace(go.Scatter(
            x=list(t) + list(reversed(t)),
            y=upper + list(reversed(lower)),
            fill="toself", fillcolor="rgba(99,110,250,0.15)",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=mean_tr, mode="lines",
        name=f"{name} (n={n_epochs})",
        line=dict(color="#636EFA", width=2),
        showlegend=False,
    ), row=row, col=1)
    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"),
                   row=row, col=1)
    fig.update_xaxes(range=[ana_start, ana_end], title_text="ms",
                      row=row, col=1)
    yr = _yrange(t, mean_tr, ana_start, ana_end)
    if yr is None:
        return
    # Widen the y-range to include the SEM band edges so the fill
    # doesn't clip at the top/bottom.
    if sem_tr and len(sem_tr) == len(mean_tr):
        sem_vals = []
        for tv, mv, sv in zip(t, mean_tr, sem_tr):
            if ana_start <= tv <= ana_end:
                sem_vals.append(mv + sv)
                sem_vals.append(mv - sv)
        if sem_vals:
            yr = [min(yr[0], min(sem_vals)),
                  max(yr[1], max(sem_vals))]
            pad = (yr[1] - yr[0]) * 0.1
            yr = [yr[0] - pad, yr[1] + pad]
    fig.update_yaxes(range=yr, row=row, col=1)


def _yrange(times, values, x0, x1):
    """Tight y-range with 10% padding for values whose x is inside
    [x0, x1]. Returns None when no samples fall in range so the
    caller can skip the update."""
    vals = [v for tv, v in zip(times, values) if x0 <= tv <= x1]
    if not vals:
        return None
    mn, mx = min(vals), max(vals)
    pad = (mx - mn) * 0.1 if mx > mn else 0.001
    return [mn - pad, mx + pad]
