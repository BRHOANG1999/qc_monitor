"""Dash web dashboard for the QC Monitor — v3 with evoked waveforms, LFP browser,
electrode health, session compare, activity log, and annotations."""

import json
import logging
import os
import statistics
from datetime import datetime

import numpy as np
import yaml
from dash import Dash, html, dcc, dash_table, callback_context, no_update, ALL
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.db.store import Store
from src.utils.mat_loader import load_mat

logger = logging.getLogger("qc_monitor.dashboard")

# Full list of plottable evoked feature columns
EVOKED_FEATURE_COLS = [
    "line_length", "log_auc", "peak_amplitude", "trough_amplitude",
    "peak_to_trough", "rms_amplitude", "peak_latency_ms", "trough_latency_ms",
    "max_slope", "max_slope_time_ms", "early_area", "late_area",
    "early_late_ratio", "recovery_tau", "recovery_slope",
    "template_correlation", "pca_recon_error", "variance",
    "autocorrelation", "ac_width", "exp_fit_a", "sum_power_low",
    "freq_moment_low", "sum_power_high", "freq_moment_high",
    "is_artifact", "is_ictal", "epoch_time_sec",
]

# Human-readable labels
EVOKED_FEATURE_LABELS = {
    "line_length": "Line Length", "log_auc": "Log(AUC)",
    "peak_amplitude": "Peak Amplitude", "trough_amplitude": "Trough Amplitude",
    "peak_to_trough": "Peak-to-Trough", "rms_amplitude": "RMS Amplitude",
    "peak_latency_ms": "Peak Latency (ms)", "trough_latency_ms": "Trough Latency (ms)",
    "max_slope": "Max Slope", "max_slope_time_ms": "Max Slope Time (ms)",
    "early_area": "Early Area", "late_area": "Late Area",
    "early_late_ratio": "Early/Late Ratio", "recovery_tau": "Recovery Tau",
    "recovery_slope": "Recovery Slope", "template_correlation": "Template Correlation",
    "pca_recon_error": "PCA Recon Error", "variance": "Variance",
    "autocorrelation": "Autocorrelation", "ac_width": "AC Width",
    "exp_fit_a": "Exp Fit A", "sum_power_low": "Sum Power Low",
    "freq_moment_low": "Freq Moment Low", "sum_power_high": "Sum Power High",
    "freq_moment_high": "Freq Moment High", "is_artifact": "Is Artifact",
    "is_ictal": "Is Ictal", "epoch_time_sec": "Epoch Time (sec)",
}

# Channel role colors
ROLE_COLORS = {"eeg": "#636EFA", "stim_copy": "#888888", "reference": "#00CC96"}

# Config file path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
CONFIG_PATH = os.path.normpath(CONFIG_PATH)

# Standard time-range options used across tabs
TIME_RANGE_OPTIONS = [
    {"label": "Last 24h", "value": 24},
    {"label": "Last 48h", "value": 48},
    {"label": "Last 1 week", "value": 168},
    {"label": "Last 1 month", "value": 720},
    {"label": "All time", "value": 0},
]

# Dark card style
CARD_STYLE = {
    "borderRadius": "10px",
    "padding": "15px 20px",
    "minWidth": "130px",
    "textAlign": "center",
    "border": "1px solid rgba(255,255,255,0.06)",
    "background": "linear-gradient(180deg, #1e1e2f 0%, #181828 100%)",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.3)",
    "transition": "transform 0.15s, box-shadow 0.15s",
}

TAB_STYLE = {"backgroundColor": "#0d0d1a", "color": "#777", "padding": "10px 20px",
             "border": "1px solid #222", "borderBottom": "none",
             "borderRadius": "6px 6px 0 0", "fontSize": "13px"}
TAB_SELECTED_STYLE = {"backgroundColor": "#1a1a2e", "color": "white", "padding": "10px 20px",
                      "border": "1px solid #444", "borderBottom": "2px solid #636EFA",
                      "borderRadius": "6px 6px 0 0", "fontWeight": "bold", "fontSize": "13px",
                      "boxShadow": "0 2px 8px rgba(99,110,250,0.15)"}

DARK_TABLE_STYLE = {
    "style_header": {"backgroundColor": "#16162a", "color": "white", "fontWeight": "bold",
                     "border": "1px solid #333", "fontSize": "12px"},
    "style_data": {"backgroundColor": "#0f0f1a", "color": "#ccc", "border": "1px solid #222",
                   "fontSize": "12px"},
    "style_cell": {"textAlign": "left", "padding": "8px 12px", "fontSize": "12px"},
    "style_filter": {"backgroundColor": "#16162a", "color": "white"},
}

# Zebra stripe base — merge with table-specific conditional styles
ZEBRA_STRIPE = {"if": {"row_index": "odd"}, "backgroundColor": "#141425"}

SECTION_STYLE = {"background": "linear-gradient(180deg, #1e1e2f 0%, #181828 100%)",
                 "padding": "16px 20px", "borderRadius": "10px",
                 "border": "1px solid #2a2a3e", "marginBottom": "16px",
                 "boxShadow": "0 1px 6px rgba(0,0,0,0.3)"}
LABEL_STYLE = {"color": "#888", "fontSize": "11px", "marginBottom": "3px", "display": "block",
               "letterSpacing": "0.3px"}
INPUT_STYLE = {"backgroundColor": "#0f0f1a", "color": "white", "border": "1px solid #333",
               "borderRadius": "6px", "padding": "6px 10px", "width": "100%",
               "transition": "border-color 0.2s, box-shadow 0.2s"}
FIELD_STYLE = {"flex": "1", "minWidth": "180px"}
DROPDOWN_STYLE = {"backgroundColor": "#1e1e2f", "color": "white"}

# ====================================================================== #
#  Helpers
# ====================================================================== #


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def _parse_json_field(val):
    """Parse a JSON string from the DB or return the value as-is."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _get_channel_map(store: Store, session_dir: str) -> dict:
    """Build channel_index -> {name, role} map from session_config."""
    cfg = store.get_session_config(session_dir)
    if not cfg:
        return {}
    names = _parse_json_field(cfg.get("channel_names")) or []
    eeg_chs = _parse_json_field(cfg.get("eeg_channels")) or []
    stim_chs = _parse_json_field(cfg.get("stim_copy_channels")) or []
    ref_chs = _parse_json_field(cfg.get("reference_channels")) or []

    ch_map = {}
    for i, name in enumerate(names):
        role = "eeg"
        if i in stim_chs:
            role = "stim_copy"
        elif i in ref_chs:
            role = "reference"
        ch_map[i] = {"name": name, "role": role}
    return ch_map


def _color_for_role(role: str) -> str:
    return ROLE_COLORS.get(role, "#636EFA")


def _status_card(title: str, value: str, color: str = "#636EFA"):
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase",
                                "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ], style=CARD_STYLE)


def _empty_fig(text: str = "No data", height: int = 400) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", height=height,
        annotations=[dict(text=text, showarrow=False, font=dict(size=16, color="#888"))],
    )
    return fig


def _session_dropdown_options(store: Store) -> list[dict]:
    sessions = store.get_sessions()
    return [{"label": s["session_name"], "value": s["session_dir"]} for s in sessions]


def _default_session(store: Store) -> str | None:
    sessions = store.get_sessions()
    return sessions[0]["session_dir"] if sessions else None


def _get_processed_files_for_session(store: Store, session_dir: str) -> list[dict]:
    """Return processed file rows for a session, ordered by chunk_datetime."""
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT id, file_path, chunk_datetime, session_name
               FROM processed_files
               WHERE session_dir = ? AND status = 'done'
               ORDER BY chunk_datetime""",
            (session_dir,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ====================================================================== #
#  App factory
# ====================================================================== #

def create_app(config: dict, store: Store) -> Dash:
    refresh_sec = config.get("dashboard", {}).get("refresh_interval_sec", 10)

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    app = Dash(__name__, title="QC Monitor", suppress_callback_exceptions=True,
               assets_folder=assets_dir)

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("QC Monitor", style={"margin": "0", "fontSize": "18px", "fontWeight": "600",
                                          "letterSpacing": "1px"}),
        ], id="app-header",
           style={"padding": "10px 24px",
                  "background": "linear-gradient(135deg, #0d0d2b 0%, #1a1035 50%, #0d1a2e 100%)",
                  "color": "white",
                  "borderBottom": "1px solid rgba(99,110,250,0.3)",
                  "boxShadow": "0 2px 12px rgba(0,0,0,0.5)"}),

        # Tabs — ordered as specified
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Evoked Waveforms", value="waveforms",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Evoked Features", value="evoked",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Signal Quality", value="signal",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Criticality", value="criticality",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="LFP Browser", value="lfp",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Electrode Health", value="electrode_health",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Session Compare", value="session_compare",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Stim QC", value="stim",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Settings", value="settings",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Activity Log", value="activity_log",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Annotations", value="annotations",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Alerts", value="alerts",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Sessions", value="sessions",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"borderBottom": "none"}),

        html.Div(id="tab-content", style={"padding": "24px", "backgroundColor": "#0f0f1a",
                                           "minHeight": "80vh"}),

        dcc.Interval(id="refresh", interval=refresh_sec * 1000, n_intervals=0,
                     disabled=True),  # auto-refresh OFF by default
        html.Div([
            html.Button("Refresh", id="manual-refresh-btn",
                        style={"backgroundColor": "#333", "color": "white", "border": "1px solid #555",
                               "borderRadius": "4px", "padding": "4px 12px", "cursor": "pointer",
                               "marginRight": "8px"}),
            html.Span(id="last-refresh-label",
                      style={"color": "#555", "fontSize": "11px", "marginRight": "10px"}),
            dcc.Checklist(id="auto-refresh-toggle",
                          options=[{"label": " Auto-refresh", "value": True}],
                          value=[], inline=True,
                          style={"color": "#888", "display": "inline-block", "fontSize": "12px"}),
        ], id="refresh-bar",
           style={"position": "fixed", "top": "6px", "right": "20px", "zIndex": "9999",
                  "display": "flex", "alignItems": "center",
                  "background": "rgba(15,15,26,0.85)", "backdropFilter": "blur(8px)",
                  "padding": "4px 12px", "borderRadius": "8px",
                  "border": "1px solid rgba(255,255,255,0.06)"}),
        dcc.Store(id="refresh-trigger", data=0),
        dcc.Store(id="last-refresh-ts", data=None),
        dcc.Interval(id="elapsed-ticker", interval=5000, n_intervals=0),
        # Hidden stores
        dcc.Store(id="selected-session-dir"),
    ], style={"backgroundColor": "#0f0f1a", "fontFamily": "'Segoe UI', -apple-system, sans-serif",
              "color": "#ddd", "minHeight": "100vh"})

    # ------------------------------------------------------------------ #
    #  Refresh controls
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("refresh", "disabled"),
        [Input("auto-refresh-toggle", "value")]
    )
    def toggle_auto_refresh(val):
        return not bool(val)

    @app.callback(
        [Output("refresh-trigger", "data"), Output("last-refresh-ts", "data")],
        [Input("manual-refresh-btn", "n_clicks"), Input("refresh", "n_intervals")],
        [State("refresh-trigger", "data")]
    )
    def on_refresh(clicks, intervals, current):
        import time as _time
        return (current or 0) + 1, _time.time()

    @app.callback(
        Output("last-refresh-label", "children"),
        [Input("elapsed-ticker", "n_intervals")],
        [State("last-refresh-ts", "data")],
    )
    def update_elapsed(_, ts):
        if ts is None:
            return ""
        import time as _time
        elapsed = int(_time.time() - ts)
        if elapsed < 60:
            return f"Last refresh {elapsed}s ago"
        return f"Last refresh {elapsed // 60}m {elapsed % 60}s ago"

    # ------------------------------------------------------------------ #
    #  Main tab router
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "value"), Input("refresh-trigger", "data")]
    )
    def render_tab(tab, _n):
        try:
            if tab == "overview":
                return _overview_tab(store)
            elif tab == "waveforms":
                return _waveforms_tab_layout(store)
            elif tab == "signal":
                return _signal_quality_tab(store)
            elif tab == "evoked":
                return _evoked_tab_layout(store)
            elif tab == "criticality":
                return _criticality_tab_layout(store)
            elif tab == "lfp":
                return _lfp_browser_tab_layout(store)
            elif tab == "electrode_health":
                return _electrode_health_tab_layout(store)
            elif tab == "session_compare":
                return _session_compare_tab_layout(store)
            elif tab == "stim":
                return _stim_tab(store)
            elif tab == "settings":
                return _settings_tab_layout(store)
            elif tab == "activity_log":
                return _activity_log_tab_layout(store)
            elif tab == "annotations":
                return _annotations_tab_layout(store)
            elif tab == "alerts":
                return _alerts_tab(store)
            elif tab == "sessions":
                return _sessions_tab(store)
        except Exception as e:
            logger.error("Dashboard render error: %s", e, exc_info=True)
            return html.Div(f"Error rendering tab: {e}",
                            style={"color": "#ff6b6b", "padding": "20px"})

    # ------------------------------------------------------------------ #
    #  Evoked Features callback — separate row per selected feature
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("evoked-multi-plots", "children"),
         Output("evoked-stats", "children")],
        [Input("evoked-feature-checklist", "value"),
         Input("evoked-session-dropdown", "value"),
         Input("evoked-hours-dropdown", "value")],
    )
    def update_evoked_multi(selected_features, session_dir, hours):
        if not selected_features or not session_dir:
            return html.P("Select features and a session", style={"color": "#888"}), ""

        plots = []
        stats_rows = []
        hrs = int(hours) if hours else 0

        for feature_name in selected_features:
            try:
                data = store.get_evoked_feature_timeseries(
                    feature_name=feature_name,
                    session_dir=session_dir,
                    hours=hrs,
                )
            except Exception as e:
                plots.append(html.P(f"Error loading {feature_name}: {e}", style={"color": "#ff6b6b"}))
                continue

            if not data:
                plots.append(html.P(f"No data for {feature_name}", style={"color": "#888"}))
                continue

            clean_t, clean_v = [], []
            artifact_t, artifact_v = [], []
            ictal_t, ictal_v = [], []
            all_values = []

            for d in data:
                # Use epoch_time_sec for continuous x-axis (seconds within file)
                # Add to chunk_datetime for absolute timestamp
                epoch_sec = d.get("epoch_time_sec")
                chunk_dt = d.get("chunk_datetime", "")
                v = d.get("value")
                if v is None:
                    continue
                # Build absolute epoch time: chunk start + epoch offset
                if epoch_sec is not None and chunk_dt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td
                        base = _dt.fromisoformat(chunk_dt.replace("_", "-").replace("--", " ").replace("__", "T"))
                        t = (base + _td(seconds=float(epoch_sec))).isoformat()
                    except Exception:
                        t = chunk_dt  # fallback to file-level timestamp
                else:
                    t = chunk_dt
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
                fig.add_trace(go.Scatter(x=clean_t, y=clean_v, mode="markers", name="Clean",
                                         marker=dict(color="#636EFA", size=4, opacity=0.6)))
            if artifact_t:
                fig.add_trace(go.Scatter(x=artifact_t, y=artifact_v, mode="markers", name="Artifact",
                                         marker=dict(color="#EF553B", size=5, opacity=0.7)))
            if ictal_t:
                fig.add_trace(go.Scatter(x=ictal_t, y=ictal_v, mode="markers", name="Ictal",
                                         marker=dict(color="#FFA15A", size=5, opacity=0.7)))

            fig.update_layout(
                template="plotly_dark",
                title=label,
                xaxis_title="Time",
                yaxis_title=label,
                height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            plots.append(dcc.Graph(figure=fig, style={"marginBottom": "8px"}))

            # Per-feature stats
            n_total = len(all_values)
            n_art = len(artifact_v)
            n_clean = len(clean_v)
            mean_val = statistics.mean(all_values) if all_values else 0
            std_val = statistics.stdev(all_values) if len(all_values) > 1 else 0
            stats_rows.append(f"{label}: mean={mean_val:.4g}, std={std_val:.4g}, "
                              f"N={n_total} (clean={n_clean}, artifact={n_art})")

        summary = html.Div([
            html.P(s, style={"color": "#aaa", "margin": "2px 0", "fontSize": "12px"})
            for s in stats_rows
        ]) if stats_rows else ""

        return html.Div(plots), summary

    # ------------------------------------------------------------------ #
    #  Evoked Waveforms callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("waveform-plot", "figure"),
        [Input("waveform-session-dropdown", "value")],
    )
    def update_waveform_plot(session_dir):
        if not session_dir:
            return _empty_fig("Select a session", 550)

        try:
            waveforms = store.get_evoked_waveforms_for_session(session_dir)
        except Exception as e:
            return _empty_fig(f"Error: {e}", 550)

        if not waveforms:
            return _empty_fig("No evoked waveforms for this session", 550)

        # Read analysis window from config
        cfg = _load_config()
        fa = cfg.get("feature_analysis", {})
        ana_start = fa.get("analysis_start_ms", fa.get("window_start_ms", 2))
        ana_end = fa.get("analysis_end_ms", fa.get("window_end_ms", 50))

        # Group latest waveform per channel
        latest_per_ch = {}
        for wf in waveforms:
            ch = wf.get("channel", 0)
            latest_per_ch[ch] = wf

        ch_map = _get_channel_map(store, session_dir)

        # Build StimCopy→LFP pairs based on channel order
        # e.g., Ch0(stimCopy)→Ch1(LFP), Ch2(stimCopy)→Ch3(LFP)
        stim_chs = sorted([c for c in latest_per_ch if ch_map.get(c, {}).get("role") == "stim_copy"])
        lfp_chs = sorted([c for c in latest_per_ch if ch_map.get(c, {}).get("role") != "stim_copy"])

        pairs = []
        used_lfp = set()
        for sc in stim_chs:
            # Find the next LFP channel after this stimCopy
            partner = None
            for lc in lfp_chs:
                if lc > sc and lc not in used_lfp:
                    partner = lc
                    used_lfp.add(lc)
                    break
            if partner is not None:
                pairs.append((sc, partner))

        # Any unpaired LFP channels get their own row
        unpaired_lfp = [lc for lc in lfp_chs if lc not in used_lfp]

        # Layout: 2 rows per pair (artifact comparison + analysis window) + 1 per unpaired
        n_rows = len(pairs) * 2 + len(unpaired_lfp)
        if n_rows == 0:
            return _empty_fig("No channel pairs found", 400)

        titles = []
        for sc, lc in pairs:
            sc_name = ch_map.get(sc, {}).get("name", f"Ch{sc}")
            lc_name = ch_map.get(lc, {}).get("name", f"Ch{lc}")
            titles.append(f"Artifact: {sc_name} vs {lc_name} (-1 to 1 ms)")
            titles.append(f"Evoked: {lc_name} ({ana_start}-{ana_end} ms)")
        for lc in unpaired_lfp:
            lc_name = ch_map.get(lc, {}).get("name", f"Ch{lc}")
            titles.append(f"Evoked: {lc_name} ({ana_start}-{ana_end} ms)")

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=False,
            subplot_titles=titles,
            vertical_spacing=0.04,
        )

        row = 1
        for sc, lc in pairs:
            sc_wf = latest_per_ch[sc]
            lc_wf = latest_per_ch[lc]
            sc_name = ch_map.get(sc, {}).get("name", f"Ch{sc}")
            lc_name = ch_map.get(lc, {}).get("name", f"Ch{lc}")

            # Row A: Artifact comparison — both channels superimposed, -1 to 1 ms
            for wf, name, color in [(sc_wf, sc_name, "#FFA15A"), (lc_wf, lc_name, "#636EFA")]:
                fig.add_trace(go.Scatter(
                    x=wf["time_axis_ms"], y=wf["mean_trace"], mode="lines",
                    name=name, line=dict(color=color, width=2),
                    legendgroup=f"pair{sc}", showlegend=(row == 1),
                ), row=row, col=1)
            fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"), row=row, col=1)
            fig.update_xaxes(range=[-1, 1], title_text="ms", row=row, col=1)
            row += 1

            # Row B: LFP evoked response in analysis window
            t = lc_wf["time_axis_ms"]
            m = lc_wf["mean_trace"]
            s = lc_wf.get("sem_trace")
            n_ep = lc_wf.get("n_epochs", 0)
            if s and len(s) == len(m):
                upper = [mv + sv for mv, sv in zip(m, s)]
                lower = [mv - sv for mv, sv in zip(m, s)]
                fig.add_trace(go.Scatter(
                    x=list(t) + list(reversed(t)), y=upper + list(reversed(lower)),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=m, mode="lines",
                name=f"{lc_name} (n={n_ep})",
                line=dict(color="#636EFA", width=2),
                showlegend=False,
            ), row=row, col=1)
            fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"), row=row, col=1)
            fig.add_vrect(x0=ana_start, x1=ana_end,
                          fillcolor="rgba(0,204,150,0.08)", line_width=0, row=row, col=1)
            pad = (ana_end - ana_start) * 0.2
            fig.update_xaxes(range=[ana_start - pad, ana_end + pad],
                             title_text="ms", row=row, col=1)
            row += 1

        # Unpaired LFP channels
        for lc in unpaired_lfp:
            lc_wf = latest_per_ch[lc]
            lc_name = ch_map.get(lc, {}).get("name", f"Ch{lc}")
            t = lc_wf["time_axis_ms"]
            m = lc_wf["mean_trace"]
            s = lc_wf.get("sem_trace")
            n_ep = lc_wf.get("n_epochs", 0)
            if s and len(s) == len(m):
                upper = [mv + sv for mv, sv in zip(m, s)]
                lower = [mv - sv for mv, sv in zip(m, s)]
                fig.add_trace(go.Scatter(
                    x=list(t) + list(reversed(t)), y=upper + list(reversed(lower)),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=m, mode="lines", name=f"{lc_name} (n={n_ep})",
                line=dict(color="#636EFA", width=2), showlegend=False,
            ), row=row, col=1)
            fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"), row=row, col=1)
            fig.add_vrect(x0=ana_start, x1=ana_end,
                          fillcolor="rgba(0,204,150,0.08)", line_width=0, row=row, col=1)
            pad = (ana_end - ana_start) * 0.2
            fig.update_xaxes(range=[ana_start - pad, ana_end + pad],
                             title_text="ms", row=row, col=1)
            row += 1

        fig.update_layout(
            template="plotly_dark",
            height=220 * n_rows,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Criticality callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("criticality-plot", "figure"),
        [Input("criticality-session-dropdown", "value"),
         Input("criticality-hours-dropdown", "value")],
    )
    def update_criticality(session_dir, hours):
        if not session_dir:
            return _empty_fig("Select a session", 550)

        ch_map = _get_channel_map(store, session_dir)

        try:
            data = store.query_criticality_timeseries(
                session_dir=session_dir,
                hours=int(hours) if hours else None,
            )
        except Exception as e:
            return _empty_fig(f"Error: {e}", 550)

        if not data:
            return _empty_fig("No criticality data", 550)

        channels = sorted(set(d["channel"] for d in data))

        fig = go.Figure()
        for ch in channels:
            ch_data = [d for d in data if d["channel"] == ch]
            times = [d["chunk_datetime"] for d in ch_data]
            vals = [d["db_value"] for d in ch_data]

            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            if ch_map and info["role"] != "eeg":
                continue

            fig.add_trace(go.Scatter(
                x=times, y=vals, mode="lines+markers",
                name=info["name"],
                line=dict(color=_color_for_role(info["role"])),
                marker=dict(size=3),
            ))

        fig.update_layout(
            template="plotly_dark",
            title="Criticality (dB) Over Time -- EEG Channels",
            xaxis_title="Time",
            yaxis_title="dB Value",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  LFP Browser callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("lfp-plot", "figure"),
        [Input("lfp-load-btn", "n_clicks")],
        [State("lfp-session-dropdown", "value"),
         State("lfp-file-dropdown", "value")],
        prevent_initial_call=True,
    )
    def load_lfp(n_clicks, session_dir, file_path):
        if not n_clicks or not file_path:
            return _empty_fig("Select a file and click Load", 600)

        try:
            chunk = load_mat(file_path)
        except Exception as e:
            return _empty_fig(f"Error loading file: {e}", 600)

        fs = chunk.fs
        n_samples = int(5 * fs)
        signal = chunk.signal[:n_samples, :]
        n_ch = signal.shape[1]
        time_sec = np.arange(signal.shape[0]) / fs

        ch_map = _get_channel_map(store, session_dir) if session_dir else {}

        fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                            vertical_spacing=0.005)

        for ch_idx in range(n_ch):
            info = ch_map.get(ch_idx, {"name": f"Ch{ch_idx}", "role": "eeg"})
            color = _color_for_role(info["role"])
            fig.add_trace(go.Scatter(
                x=time_sec, y=signal[:, ch_idx],
                mode="lines", name=info["name"],
                line=dict(color=color, width=0.8),
            ), row=ch_idx + 1, col=1)
            fig.update_yaxes(title_text=info["name"], row=ch_idx + 1, col=1,
                             title_font=dict(size=9, color="#aaa"),
                             tickfont=dict(size=8))

        fig.update_xaxes(title_text="Time (sec)", row=n_ch, col=1)
        fig.update_layout(
            template="plotly_dark",
            title=f"Raw LFP (first 5 sec) -- {n_ch} channels @ {fs:.0f} Hz",
            height=max(600, n_ch * 80),
            showlegend=False,
        )
        return fig

    @app.callback(
        Output("lfp-file-dropdown", "options"),
        Input("lfp-session-dropdown", "value"),
    )
    def update_lfp_file_options(session_dir):
        if not session_dir:
            return []
        files = _get_processed_files_for_session(store, session_dir)
        return [{"label": f"{f['chunk_datetime'][:16]} - {os.path.basename(f['file_path'])}",
                 "value": f["file_path"]} for f in files]

    # ------------------------------------------------------------------ #
    #  Electrode Health callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("electrode-health-plot", "figure"),
        [Input("electrode-health-session-dropdown", "value"),
         Input("electrode-health-hours-dropdown", "value")],
    )
    def update_electrode_health(session_dir, hours):
        if not session_dir:
            return _empty_fig("Select a session", 700)

        hours_val = int(hours) if hours else 0
        try:
            data = store.get_qc_timeseries(session_dir=session_dir, hours=hours_val)
        except Exception as e:
            return _empty_fig(f"Error: {e}", 700)

        if not data:
            return _empty_fig("No QC data available", 700)

        # Separate by channel_role
        stim_copy_data = [d for d in data if d.get("channel_role") == "stim_copy"]
        reference_data = [d for d in data if d.get("channel_role") == "reference"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Stim Copy RMS", "Reference RMS",
                                            "Line Noise Ratio (60 Hz indicator)"],
                            vertical_spacing=0.08)

        # Stim copy RMS
        stim_channels = sorted(set(d["channel"] for d in stim_copy_data))
        for ch in stim_channels:
            ch_data = [d for d in stim_copy_data if d["channel"] == ch]
            ch_name = ch_data[0].get("channel_name") or f"Ch{ch}"
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["rms_amplitude"] for d in ch_data],
                mode="lines+markers", name=f"{ch_name} (stim_copy)",
                line=dict(color="#888888"), marker=dict(size=3),
            ), row=1, col=1)

        # Reference RMS
        ref_channels = sorted(set(d["channel"] for d in reference_data))
        for ch in ref_channels:
            ch_data = [d for d in reference_data if d["channel"] == ch]
            ch_name = ch_data[0].get("channel_name") or f"Ch{ch}"
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["rms_amplitude"] for d in ch_data],
                mode="lines+markers", name=f"{ch_name} (reference)",
                line=dict(color="#00CC96"), marker=dict(size=3),
            ), row=2, col=1)

        # Line noise ratio for ALL channels (aggregate trend)
        all_channels = sorted(set(d["channel"] for d in data))
        ch_map = _get_channel_map(store, session_dir)
        for ch in all_channels:
            ch_data = [d for d in data if d["channel"] == ch]
            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["line_noise_ratio"] for d in ch_data],
                mode="lines", name=info["name"],
                line=dict(color=_color_for_role(info["role"]), width=1),
                showlegend=False,
            ), row=3, col=1)

        fig.update_layout(
            height=700, template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_annotations(font=dict(color="white"))
        return fig

    # ------------------------------------------------------------------ #
    #  Session Compare callback
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("session-compare-waveform-plot", "figure"),
         Output("session-compare-features-plot", "figure")],
        [Input("compare-session-a-dropdown", "value"),
         Input("compare-session-b-dropdown", "value")],
    )
    def update_session_compare(session_a, session_b):
        if not session_a or not session_b:
            return (_empty_fig("Select two sessions", 450),
                    _empty_fig("Select two sessions", 450))

        # --- Waveform overlay ---
        wf_fig = go.Figure()
        for sess_dir, color, label in [(session_a, "#636EFA", "Session A"),
                                       (session_b, "#EF553B", "Session B")]:
            try:
                waveforms = store.get_evoked_waveforms_for_session(sess_dir)
            except Exception:
                waveforms = []

            if waveforms:
                # Use the latest waveform
                wf = waveforms[-1]
                time_ms = wf["time_axis_ms"]
                mean_tr = wf["mean_trace"]
                sem_tr = wf["sem_trace"]

                sessions = store.get_sessions()
                sname = sess_dir
                for s in sessions:
                    if s["session_dir"] == sess_dir:
                        sname = s["session_name"]
                        break

                if sem_tr and len(sem_tr) == len(mean_tr):
                    upper = [m + s for m, s in zip(mean_tr, sem_tr)]
                    lower = [m - s for m, s in zip(mean_tr, sem_tr)]
                    fill_color = color.replace(")", ",0.15)").replace("rgb", "rgba") if color.startswith("rgb") else color
                    if color == "#636EFA":
                        fill_color = "rgba(99,110,250,0.15)"
                    else:
                        fill_color = "rgba(239,85,59,0.15)"
                    fig_band_x = list(time_ms) + list(reversed(time_ms))
                    fig_band_y = upper + list(reversed(lower))
                    wf_fig.add_trace(go.Scatter(
                        x=fig_band_x, y=fig_band_y,
                        fill="toself", fillcolor=fill_color,
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))

                wf_fig.add_trace(go.Scatter(
                    x=time_ms, y=mean_tr, mode="lines",
                    name=f"{label}: {sname}",
                    line=dict(color=color, width=2),
                ))

        wf_fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
        wf_fig.update_layout(
            template="plotly_dark",
            title="Mean Evoked Waveform Overlay",
            xaxis_title="Time (ms)", yaxis_title="Amplitude",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # --- Feature distribution comparison (top 5 features) ---
        top_features = ["peak_amplitude", "trough_amplitude", "rms_amplitude",
                        "line_length", "peak_to_trough"]
        feat_fig = go.Figure()

        for sess_dir, color, label in [(session_a, "#636EFA", "Session A"),
                                       (session_b, "#EF553B", "Session B")]:
            means = []
            stds = []
            f_names = []
            for feat in top_features:
                try:
                    fdata = store.get_evoked_feature_timeseries(
                        feature_name=feat, session_dir=sess_dir)
                    vals = [d["value"] for d in fdata if d["value"] is not None
                            and not d.get("is_artifact", 0)]
                except Exception:
                    vals = []

                f_names.append(EVOKED_FEATURE_LABELS.get(feat, feat))
                if vals:
                    means.append(statistics.mean(vals))
                    stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            sessions = store.get_sessions()
            sname = sess_dir
            for s in sessions:
                if s["session_dir"] == sess_dir:
                    sname = s["session_name"]
                    break

            feat_fig.add_trace(go.Bar(
                x=f_names, y=means,
                name=f"{label}: {sname}",
                marker_color=color,
                error_y=dict(type="data", array=stds, visible=True),
            ))

        feat_fig.update_layout(
            template="plotly_dark",
            title="Feature Distribution Comparison (top 5)",
            xaxis_title="Feature", yaxis_title="Value",
            barmode="group", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return wf_fig, feat_fig

    # ------------------------------------------------------------------ #
    #  Activity Log callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("activity-log-table", "data"),
        [Input("activity-log-hours-dropdown", "value"),
         Input("activity-log-level-dropdown", "value")],
    )
    def update_activity_log(hours, level):
        hours_val = int(hours) if hours else 24
        level_val = level if level and level != "ALL" else None
        try:
            logs = store.get_activity_log(hours=hours_val, level=level_val, limit=500)
        except Exception:
            logs = []
        return [
            {
                "timestamp": entry.get("timestamp", "")[:19],
                "level": entry.get("level", ""),
                "action": entry.get("action", ""),
                "message": (entry.get("message") or "")[:200],
                "file_path": os.path.basename(entry.get("file_path") or ""),
                "duration": f"{entry['duration_sec']:.2f}" if entry.get("duration_sec") else "",
            }
            for entry in logs
        ]

    # ------------------------------------------------------------------ #
    #  Annotations callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("annotation-status", "children"),
         Output("annotation-table", "data")],
        Input("annotation-submit-btn", "n_clicks"),
        [State("annotation-timestamp", "value"),
         State("annotation-note", "value"),
         State("annotation-category", "value"),
         State("annotation-session-dropdown", "value")],
        prevent_initial_call=True,
    )
    def submit_annotation(n_clicks, timestamp_val, note, category, session_dir):
        if not n_clicks:
            return no_update, no_update
        if not note or not note.strip():
            return html.Div("Note cannot be empty", style={"color": "#EF553B"}), no_update
        ts = timestamp_val or datetime.now().isoformat()
        try:
            ann_id = store.add_annotation(
                timestamp=ts,
                note=note.strip(),
                category=category or "observation",
                session_dir=session_dir or None,
            )
            all_ann = store.get_annotations()
            table_data = _annotations_table_data(all_ann)
            return (
                html.Div(f"Annotation #{ann_id} saved", style={"color": "#00CC96"}),
                table_data,
            )
        except Exception as e:
            return html.Div(f"Error: {e}", style={"color": "#EF553B"}), no_update

    # ------------------------------------------------------------------ #
    #  Settings callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("settings-save-status", "children"),
        Input("btn-save-settings", "n_clicks"),
        [State("evoked-pre-stim", "value"),
         State("evoked-post-stim", "value"),
         State("evoked-baseline-correction", "value"),
         State("evoked-baseline-start", "value"),
         State("evoked-baseline-end", "value"),
         State("evoked-stim-thresh", "value"),
         State("evoked-min-stim-dist", "value"),
         State("evoked-notch60", "value"),
         State("evoked-notch50", "value"),
         State("evoked-hp-enabled", "value"),
         State("evoked-hp-cutoff", "value"),
         State("evoked-lp-enabled", "value"),
         State("evoked-lp-cutoff", "value"),
         State("crit-ar-order", "value"),
         State("crit-window-sec", "value"),
         State("crit-overlap", "value"),
         State("crit-fit-method", "value"),
         State("crit-target-sr", "value"),
         State("crit-type-b", "value"),
         State("crit-error-bars", "value"),
         State("seiz-spike-thresh", "value"),
         State("seiz-spike-min-w", "value"),
         State("seiz-spike-max-w", "value"),
         State("seiz-min-dur", "value"),
         State("seiz-glue-sec", "value"),
         State("seiz-min-spikes", "value"),
         State("seiz-outlier", "value"),
         State("art-method", "value"),
         State("art-fixed-thresh", "value"),
         State("art-mad-k", "value"),
         State("art-merge-gap", "value"),
         State("qc-artifact-warn", "value"),
         State("qc-flatline-std", "value"),
         State("qc-clipping-v", "value"),
         State("qc-linenoise-warn", "value"),
         State("features-checklist", "value"),
         # Step 2: Feature Analysis states
         State("feat-win-start", "value"),
         State("feat-win-end", "value"),
         State("feat-stim-art-start", "value"),
         State("feat-stim-art-end", "value"),
         State("feat-bp-enabled", "value"),
         State("feat-bp-hp", "value"),
         State("feat-bp-lp", "value"),
         State("feat-notch-enabled", "value"),
         State("feat-notch-freq", "value"),
         State("feat-smooth-enabled", "value"),
         State("feat-smooth-win", "value"),
         State("feat-baseline", "value"),
         State("feat-art-enabled", "value"),
         State("feat-art-method", "value"),
         State("feat-art-thresh", "value"),
         State("feat-art-merge", "value"),
         State("feat-tmpl-source", "value"),
         State("feat-tmpl-upper", "value"),
         State("feat-tmpl-lower", "value"),
         State("feat-rawamp-k", "value"),
         State("feat-ictal-rescue", "value"),
         State("feat-ictal-win", "value"),
         State("feat-analysis-start", "value"),
         State("feat-analysis-end", "value"),
         State("feat-early-start", "value"),
         State("feat-early-end", "value"),
         State("feat-late-start", "value"),
         State("feat-late-end", "value"),
         ],
        prevent_initial_call=True,
    )
    def save_settings(n_clicks, *values):
        if not n_clicks:
            return no_update
        try:
            cfg = _load_config()
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr, type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec, min_spikes, outlier,
             art_method, art_fixed, art_mad, art_merge,
             qc_art_warn, qc_flat, qc_clip, qc_ln,
             features_enabled,
             # Step 2 values
             f_win_start, f_win_end, f_stim_start, f_stim_end,
             f_bp_en, f_bp_hp, f_bp_lp, f_notch_en, f_notch_freq,
             f_smooth_en, f_smooth_win, f_baseline,
             f_art_en, f_art_method, f_art_thresh, f_art_merge,
             f_tmpl_src, f_tmpl_upper, f_tmpl_lower, f_rawamp_k,
             f_ictal_en, f_ictal_win,
             f_analysis_start, f_analysis_end,
             f_early_start, f_early_end, f_late_start, f_late_end,
             ) = values

            # Step 1: Epoch Extraction
            ee = cfg.setdefault("epoch_extraction", {})
            if pre_stim is not None: ee["pre_stimulus_ms"] = float(pre_stim)
            if post_stim is not None: ee["post_stimulus_ms"] = float(post_stim)
            ee["baseline_correction"] = bool(baseline_corr)
            ee["baseline_start_ms"] = float(bl_start or -60)
            ee["baseline_end_ms"] = float(bl_end or -10)
            if stim_thresh is not None: ee["stimulus_threshold_std"] = float(stim_thresh)
            if min_stim_dist is not None: ee["min_stimulus_distance_sec"] = float(min_stim_dist)
            ee["notch_60hz"] = bool(notch60)
            ee["notch_50hz"] = bool(notch50)
            ee["highpass_enabled"] = bool(hp_en)
            if hp_cut is not None: ee["highpass_cutoff_hz"] = float(hp_cut)
            ee["lowpass_enabled"] = bool(lp_en)
            if lp_cut is not None: ee["lowpass_cutoff_hz"] = float(lp_cut)
            cfg.pop("evoked", None)

            # Step 2: Feature Analysis
            fa = cfg.setdefault("feature_analysis", {})
            if f_win_start is not None: fa["window_start_ms"] = float(f_win_start)
            if f_win_end is not None: fa["window_end_ms"] = float(f_win_end)
            if f_stim_start is not None: fa["stim_artifact_start_ms"] = float(f_stim_start)
            if f_stim_end is not None: fa["stim_artifact_end_ms"] = float(f_stim_end)
            fa["bandpass_enabled"] = bool(f_bp_en)
            if f_bp_hp is not None: fa["bandpass_highpass_hz"] = float(f_bp_hp)
            if f_bp_lp is not None: fa["bandpass_lowpass_hz"] = float(f_bp_lp)
            fa["notch_enabled"] = bool(f_notch_en)
            if f_notch_freq is not None: fa["notch_frequency_hz"] = float(f_notch_freq)
            fa["smoothing_enabled"] = bool(f_smooth_en)
            if f_smooth_win is not None: fa["smoothing_window_ms"] = float(f_smooth_win)
            fa["baseline_correction"] = bool(f_baseline)
            fa["artifact_exclusion_enabled"] = bool(f_art_en)
            if f_art_method: fa["artifact_method"] = f_art_method
            if f_art_thresh is not None: fa["artifact_threshold"] = float(f_art_thresh)
            if f_art_merge is not None: fa["artifact_merge_gap_sec"] = float(f_art_merge)
            if f_tmpl_src: fa["template_source"] = f_tmpl_src
            if f_tmpl_upper is not None: fa["template_upper_r"] = float(f_tmpl_upper)
            if f_tmpl_lower is not None: fa["template_lower_r"] = float(f_tmpl_lower)
            if f_rawamp_k is not None: fa["rawamp_multiplier"] = float(f_rawamp_k)
            fa["ictal_rescue_enabled"] = bool(f_ictal_en)
            if f_ictal_win is not None: fa["ictal_rescue_window_ms"] = float(f_ictal_win)
            if f_analysis_start is not None: fa["analysis_start_ms"] = float(f_analysis_start)
            if f_analysis_end is not None: fa["analysis_end_ms"] = float(f_analysis_end)
            if f_early_start is not None: fa["early_area_start_ms"] = float(f_early_start)
            if f_early_end is not None: fa["early_area_end_ms"] = float(f_early_end)
            if f_late_start is not None: fa["late_area_start_ms"] = float(f_late_start)
            if f_late_end is not None: fa["late_area_end_ms"] = float(f_late_end)

            cfg["criticality"]["ar_order"] = int(ar_order) if ar_order is not None else cfg["criticality"]["ar_order"]
            cfg["criticality"]["window_sec"] = float(win_sec) if win_sec is not None else cfg["criticality"]["window_sec"]
            cfg["criticality"]["overlap_pct"] = int(overlap) if overlap is not None else cfg["criticality"]["overlap_pct"]
            cfg["criticality"]["fit_method"] = fit_method or cfg["criticality"]["fit_method"]
            cfg["criticality"]["target_sampling_rate"] = int(target_sr) if target_sr is not None else cfg["criticality"]["target_sampling_rate"]
            cfg["criticality"]["criticality_type_b"] = int(type_b) if type_b is not None else cfg["criticality"]["criticality_type_b"]
            cfg["criticality"]["calculate_error_bars"] = bool(err_bars)

            cfg["seizure"]["spike_threshold_uv"] = float(sp_thresh) if sp_thresh is not None else cfg["seizure"]["spike_threshold_uv"]
            cfg["seizure"]["spike_min_width"] = int(sp_min_w) if sp_min_w is not None else cfg["seizure"]["spike_min_width"]
            cfg["seizure"]["spike_max_width"] = int(sp_max_w) if sp_max_w is not None else cfg["seizure"]["spike_max_width"]
            cfg["seizure"]["min_seizure_duration_sec"] = float(min_dur) if min_dur is not None else cfg["seizure"]["min_seizure_duration_sec"]
            cfg["seizure"]["event_glue_sec"] = float(glue_sec) if glue_sec is not None else cfg["seizure"]["event_glue_sec"]
            cfg["seizure"]["min_spikes_per_sec"] = float(min_spikes) if min_spikes is not None else cfg["seizure"]["min_spikes_per_sec"]
            cfg["seizure"]["outlier_factor"] = float(outlier) if outlier is not None else cfg["seizure"]["outlier_factor"]

            cfg["artifact"]["method"] = art_method or cfg["artifact"]["method"]
            cfg["artifact"]["fixed_threshold"] = float(art_fixed) if art_fixed is not None else cfg["artifact"]["fixed_threshold"]
            cfg["artifact"]["mad_k"] = float(art_mad) if art_mad is not None else cfg["artifact"]["mad_k"]
            cfg["artifact"]["merge_gap_sec"] = float(art_merge) if art_merge is not None else cfg["artifact"]["merge_gap_sec"]

            cfg["qc_thresholds"]["artifact_pct_warning"] = float(qc_art_warn) if qc_art_warn is not None else cfg["qc_thresholds"]["artifact_pct_warning"]
            cfg["qc_thresholds"]["flatline_std"] = float(qc_flat) if qc_flat is not None else cfg["qc_thresholds"]["flatline_std"]
            cfg["qc_thresholds"]["clipping_voltage"] = float(qc_clip) if qc_clip is not None else cfg["qc_thresholds"]["clipping_voltage"]
            cfg["qc_thresholds"]["line_noise_ratio_warning"] = float(qc_ln) if qc_ln is not None else cfg["qc_thresholds"]["line_noise_ratio_warning"]

            cfg["features"]["enabled"] = features_enabled or []

            _save_config(cfg)
            return html.Div("Settings saved to config.yaml",
                            style={"color": "#00CC96", "marginTop": "10px"})
        except Exception as e:
            logger.error("Failed to save settings: %s", e, exc_info=True)
            return html.Div(f"Error saving: {e}",
                            style={"color": "#EF553B", "marginTop": "10px"})

    @app.callback(
        [Output("settings-version-status", "children"),
         Output("version-history-table", "data")],
        Input("btn-save-version", "n_clicks"),
        [State("version-label-input", "value"),
         State("evoked-pre-stim", "value"),
         State("evoked-post-stim", "value"),
         State("evoked-baseline-correction", "value"),
         State("evoked-baseline-start", "value"),
         State("evoked-baseline-end", "value"),
         State("evoked-stim-thresh", "value"),
         State("evoked-min-stim-dist", "value"),
         State("evoked-notch60", "value"),
         State("evoked-notch50", "value"),
         State("evoked-hp-enabled", "value"),
         State("evoked-hp-cutoff", "value"),
         State("evoked-lp-enabled", "value"),
         State("evoked-lp-cutoff", "value"),
         State("crit-ar-order", "value"),
         State("crit-window-sec", "value"),
         State("crit-overlap", "value"),
         State("crit-fit-method", "value"),
         State("crit-target-sr", "value"),
         State("crit-type-b", "value"),
         State("crit-error-bars", "value"),
         State("seiz-spike-thresh", "value"),
         State("seiz-spike-min-w", "value"),
         State("seiz-spike-max-w", "value"),
         State("seiz-min-dur", "value"),
         State("seiz-glue-sec", "value"),
         State("seiz-min-spikes", "value"),
         State("seiz-outlier", "value"),
         State("art-method", "value"),
         State("art-fixed-thresh", "value"),
         State("art-mad-k", "value"),
         State("art-merge-gap", "value"),
         State("qc-artifact-warn", "value"),
         State("qc-flatline-std", "value"),
         State("qc-clipping-v", "value"),
         State("qc-linenoise-warn", "value"),
         State("features-checklist", "value")],
        prevent_initial_call=True,
    )
    def save_version(n_clicks, label, *values):
        if not n_clicks:
            return no_update, no_update
        try:
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr, type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec, min_spikes, outlier,
             art_method, art_fixed, art_mad, art_merge,
             qc_art_warn, qc_flat, qc_clip, qc_ln,
             features_enabled) = values

            config_dict = {
                "epoch_extraction": {
                    "pre_stimulus_ms": float(pre_stim or -100),
                    "post_stimulus_ms": float(post_stim or 500),
                    "baseline_correction": bool(baseline_corr),
                    "baseline_start_ms": float(bl_start or -60),
                    "baseline_end_ms": float(bl_end or -10),
                    "stimulus_threshold_std": float(stim_thresh or 3.0),
                    "min_stimulus_distance_sec": float(min_stim_dist or 0.1),
                    "notch_60hz": bool(notch60),
                    "notch_50hz": bool(notch50),
                    "highpass_enabled": bool(hp_en),
                    "highpass_cutoff_hz": float(hp_cut or 1.0),
                    "lowpass_enabled": bool(lp_en),
                    "lowpass_cutoff_hz": float(lp_cut or 1000.0),
                },
                "feature_analysis": {
                    "analysis_start_ms": 5, "analysis_end_ms": 50,
                    "early_area_start_ms": 0, "early_area_end_ms": 50,
                    "late_area_start_ms": 50, "late_area_end_ms": 200,
                },
                "criticality": {
                    "ar_order": int(ar_order or 5),
                    "window_sec": float(win_sec or 2.0),
                    "overlap_pct": int(overlap or 50),
                    "fit_method": fit_method or "YuleWalker",
                    "target_sampling_rate": int(target_sr or 1000),
                    "criticality_type_b": int(type_b or 2),
                    "calculate_error_bars": bool(err_bars),
                },
                "seizure": {
                    "spike_threshold_uv": float(sp_thresh or 10),
                    "spike_min_width": int(sp_min_w or 5),
                    "spike_max_width": int(sp_max_w or 50),
                    "min_seizure_duration_sec": float(min_dur or 5),
                    "event_glue_sec": float(glue_sec or 2),
                    "min_spikes_per_sec": float(min_spikes or 2),
                    "outlier_factor": float(outlier or 3),
                },
                "artifact": {
                    "method": art_method or "fixed",
                    "fixed_threshold": float(art_fixed or 500),
                    "mad_k": float(art_mad or 4.0),
                    "merge_gap_sec": float(art_merge or 2.0),
                },
                "qc_thresholds": {
                    "artifact_pct_warning": float(qc_art_warn or 50),
                    "flatline_std": float(qc_flat or 1e-6),
                    "clipping_voltage": float(qc_clip or 10.0),
                    "line_noise_ratio_warning": float(qc_ln or 0.2),
                },
                "features": {"enabled": features_enabled or []},
            }

            vid = store.create_settings_version(config_dict, label=label or "")
            versions = store.get_all_settings_versions()
            table_data = _version_table_data(versions)
            return (
                html.Div(f"Version #{vid} saved with label '{label or ''}'",
                         style={"color": "#00CC96", "marginTop": "10px"}),
                table_data,
            )
        except Exception as e:
            logger.error("Failed to save version: %s", e, exc_info=True)
            return (
                html.Div(f"Error: {e}", style={"color": "#EF553B", "marginTop": "10px"}),
                no_update,
            )

    @app.callback(
        Output("reprocess-status", "children"),
        Input("btn-reprocess", "n_clicks"),
        prevent_initial_call=True,
    )
    def reprocess_session(n_clicks):
        if not n_clicks:
            return no_update
        try:
            conn = store._connect()
            try:
                conn.execute("UPDATE processed_files SET status = 'pending' WHERE status = 'done'")
                cnt = conn.execute("SELECT changes()").fetchone()[0]
                conn.commit()
            finally:
                conn.close()
            return html.Div(f"Queued {cnt} files for reprocessing",
                            style={"color": "#FFA15A", "marginTop": "10px"})
        except Exception as e:
            return html.Div(f"Error: {e}", style={"color": "#EF553B", "marginTop": "10px"})

    return app


# ====================================================================== #
#  Tab builders
# ====================================================================== #

def _overview_tab(store: Store):
    health = store.get_health_history(hours=1)
    latest = health[-1] if health else {}

    sessions = store.get_sessions()
    active_session = sessions[0] if sessions else {}
    session_dir = active_session.get("session_dir", "")

    ch_map = _get_channel_map(store, session_dir) if session_dir else {}

    recent_alerts = store.get_recent_alerts(hours=24)

    net_ok = latest.get("network_share_accessible")
    cpu = latest.get("cpu_pct", 0)
    mem = latest.get("memory_pct", 0)
    disk = latest.get("disk_free_gb", 0)
    fph = latest.get("files_processed_last_hour", 0)

    cards = html.Div([
        _status_card("Network", "OK" if net_ok else "DOWN",
                     "#00CC96" if net_ok else "#EF553B"),
        _status_card("CPU", f"{cpu:.0f}%",
                     "#00CC96" if cpu < 80 else "#FFA15A"),
        _status_card("Memory", f"{mem:.0f}%",
                     "#00CC96" if mem < 85 else "#FFA15A"),
        _status_card("Disk Free", f"{disk:.1f} GB",
                     "#00CC96" if disk > 50 else "#EF553B"),
        _status_card("Files/Hour", str(fph), "#636EFA"),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})

    # File queue progress across all sessions
    total_files = sum(s.get("num_files", 0) for s in sessions)
    total_done = sum(s.get("processed", 0) for s in sessions)
    total_errors = sum(s.get("errors", 0) for s in sessions)
    total_pending = total_files - total_done - total_errors
    pct_done = (100 * total_done / total_files) if total_files > 0 else 0

    queue_section = html.Div([
        html.H4("Processing Queue", style={"color": "#aaa", "marginTop": "16px", "marginBottom": "8px",
                                            "fontSize": "14px", "letterSpacing": "0.5px"}),
        html.Div([
            html.Div(
                f"{pct_done:.0f}%" if pct_done > 5 else "",
                style={
                    "width": f"{max(pct_done, 1):.1f}%",
                    "background": "linear-gradient(90deg, #00CC96 0%, #00AA80 100%)",
                    "height": "28px", "borderRadius": "6px",
                    "transition": "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "11px", "fontWeight": "bold", "color": "white",
                    "textShadow": "0 1px 2px rgba(0,0,0,0.5)",
                }),
        ], style={"backgroundColor": "#1a1a2e", "borderRadius": "6px", "overflow": "hidden",
                  "marginBottom": "8px", "boxShadow": "inset 0 1px 4px rgba(0,0,0,0.4)"}),
        html.Div([
            html.Span(f"{total_done} done", style={"color": "#00CC96", "marginRight": "16px"}),
            html.Span(f"{total_pending} queued", style={"color": "#FFA15A", "marginRight": "16px"}),
            html.Span(f"{total_errors} errors", style={"color": "#EF553B", "marginRight": "16px"}),
            html.Span(f"{total_files} total detected", style={"color": "#666"}),
        ], style={"fontSize": "13px"}),
    ], style=SECTION_STYLE)

    # Evoked waveform thumbnail — show latest file, ALL channels
    waveform_thumbnail = html.Div()
    if session_dir:
        try:
            waveforms = store.get_evoked_waveforms_for_session(session_dir)
            if waveforms:
                # Group by channel, take latest per channel
                latest_by_ch = {}
                latest_datetime = ""
                for wf in waveforms:
                    ch = wf.get("channel", 0)
                    latest_by_ch[ch] = wf
                    dt = wf.get("chunk_datetime", "")
                    if dt > latest_datetime:
                        latest_datetime = dt

                sorted_chs = sorted(latest_by_ch.keys())
                n_ch = len(sorted_chs)
                if n_ch > 0:
                    colors_list = ["#636EFA", "#00CC96", "#FFA15A", "#EF553B", "#AB63FA"]
                    thumb_fig = make_subplots(
                        rows=n_ch, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                    )
                    for ri, ch in enumerate(sorted_chs, 1):
                        wf = latest_by_ch[ch]
                        ch_name = wf.get("channel_name", f"Ch{ch}")
                        n_ep = wf.get("n_epochs", 0)
                        time_ms = wf["time_axis_ms"]
                        mean_tr = wf["mean_trace"]
                        color = colors_list[ri % len(colors_list)]

                        thumb_fig.add_trace(go.Scatter(
                            x=time_ms, y=mean_tr, mode="lines",
                            name=f"{ch_name} (n={n_ep})",
                            line=dict(color=color, width=1.5),
                        ), row=ri, col=1)
                        thumb_fig.add_vline(x=0, line=dict(color="white", width=0.5, dash="dash"),
                                            row=ri, col=1)
                        thumb_fig.update_yaxes(title_text=ch_name, title_font_size=10, row=ri, col=1)

                    thumb_fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=1)
                    thumb_fig.update_layout(
                        template="plotly_dark",
                        title=f"Latest Evoked — {latest_datetime[:16]}",
                        height=180 * n_ch, margin=dict(l=60, r=20, t=40, b=30),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="right", x=1, font_size=10),
                    )
                    waveform_thumbnail = html.Div([
                        dcc.Graph(figure=thumb_fig, style={"marginTop": "16px"}),
                    ])
        except Exception as e:
            logger.debug("Could not load waveform thumbnail: %s", e)

    # Active session info
    if active_session:
        session_info = html.Div([
            html.H3("Active Session", style={"color": "white", "marginTop": "24px"}),
            html.Div([
                html.Div([
                    html.Span("Session: ", style={"color": "#888"}),
                    html.Span(active_session.get("session_name", "N/A"),
                              style={"color": "white", "fontWeight": "bold"}),
                ]),
                html.Div([
                    html.Span("Files: ", style={"color": "#888"}),
                    html.Span(f"{active_session.get('processed', 0)} / {active_session.get('num_files', 0)} processed",
                              style={"color": "white"}),
                ]),
                html.Div([
                    html.Span("Errors: ", style={"color": "#888"}),
                    html.Span(str(active_session.get("errors", 0)),
                              style={"color": "#EF553B" if active_session.get("errors", 0) > 0 else "#00CC96"}),
                ]),
            ], style={"backgroundColor": "#1e1e2f", "padding": "12px 16px",
                      "borderRadius": "8px", "border": "1px solid #333", "marginTop": "8px"}),
        ])

        if ch_map:
            ch_rows = []
            for idx in sorted(ch_map.keys()):
                info = ch_map[idx]
                ch_rows.append({
                    "index": idx,
                    "name": info["name"],
                    "role": info["role"],
                })
            channel_table = html.Div([
                html.H4("Channel Map (Auto-Discovered)", style={"color": "#aaa", "marginTop": "16px"}),
                dash_table.DataTable(
                    data=ch_rows,
                    columns=[{"name": "Index", "id": "index"},
                             {"name": "Name", "id": "name"},
                             {"name": "Role", "id": "role"}],
                    **DARK_TABLE_STYLE,
                    style_data_conditional=[ZEBRA_STRIPE,
                        {"if": {"filter_query": "{role} = eeg"},
                         "color": ROLE_COLORS["eeg"]},
                        {"if": {"filter_query": "{role} = stim_copy"},
                         "color": ROLE_COLORS["stim_copy"]},
                        {"if": {"filter_query": "{role} = reference"},
                         "color": ROLE_COLORS["reference"]},
                    ],
                    page_size=32,
                ),
            ])
        else:
            channel_table = html.Div()
    else:
        session_info = html.P("No sessions yet", style={"color": "#888"})
        channel_table = html.Div()

    # Recent alerts
    if recent_alerts:
        alerts_section = html.Div([
            html.H3(f"Recent Alerts ({len(recent_alerts)})",
                     style={"color": "white", "marginTop": "24px"}),
            dash_table.DataTable(
                data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                       "type": a["alert_type"], "message": a["message"][:120]}
                      for a in recent_alerts[:15]],
                columns=[{"name": c, "id": c} for c in ["time", "severity", "type", "message"]],
                **DARK_TABLE_STYLE,
                style_data_conditional=[ZEBRA_STRIPE,
                    {"if": {"filter_query": "{severity} = critical"},
                     "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                    {"if": {"filter_query": "{severity} = warning"},
                     "backgroundColor": "#3d3011", "color": "#ffd93d"},
                    {"if": {"filter_query": "{severity} = info"},
                     "backgroundColor": "#112233", "color": "#6bb5ff"},
                ],
                page_size=10,
            ),
        ])
    else:
        alerts_section = html.Div([
            html.H3("Recent Alerts", style={"color": "white", "marginTop": "24px"}),
            html.P("No alerts in the last 24 hours", style={"color": "#888"}),
        ])

    return html.Div([cards, queue_section, waveform_thumbnail, session_info, channel_table, alerts_section])


# ------------------------------------------------------------------ #
#  Evoked Waveforms tab (NEW)
# ------------------------------------------------------------------ #

def _waveforms_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)
    default = _default_session(store)

    return html.Div([
        html.H3("Evoked Waveforms", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="waveform-plot", style={"height": "550px"}),
    ])


# ------------------------------------------------------------------ #
#  Signal Quality tab (existing)
# ------------------------------------------------------------------ #

def _signal_quality_tab(store: Store):
    sessions = store.get_sessions()
    if not sessions:
        return html.Div("No sessions found.", style={"color": "#888"})

    session_dir = sessions[0].get("session_dir", "")
    ch_map = _get_channel_map(store, session_dir)

    data = store.get_qc_timeseries(session_dir=session_dir, hours=48)
    if not data:
        return html.Div("No QC data yet.", style={"color": "#888"})

    channels = sorted(set(d["channel"] for d in data))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["RMS Amplitude", "Artifact %", "Line Noise Ratio"],
                        vertical_spacing=0.08)

    for ch in channels:
        ch_data = [d for d in data if d["channel"] == ch]
        ch_times = [d["chunk_datetime"] for d in ch_data]

        info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
        ch_name = info["name"]
        color = _color_for_role(info["role"])

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["rms_amplitude"] for d in ch_data],
            name=ch_name, legendgroup=ch_name,
            line=dict(color=color), marker=dict(size=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["artifact_pct"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["line_noise_ratio"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=3, col=1)

    fig.update_layout(
        height=750, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_annotations(font=dict(color="white"))

    return html.Div([dcc.Graph(figure=fig)])


# ------------------------------------------------------------------ #
#  Evoked Features tab (renamed from Evoked Response)
# ------------------------------------------------------------------ #

def _evoked_tab_layout(store: Store):
    """Build the Evoked Features tab — separate row per selected feature."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    plottable = [f for f in EVOKED_FEATURE_COLS
                 if f not in ("is_artifact", "is_ictal", "epoch_time_sec")]
    feature_options = [{"label": EVOKED_FEATURE_LABELS.get(f, f), "value": f}
                       for f in plottable]

    default_session = sessions[0]["session_dir"] if sessions else None
    # Default: show 3 key features
    default_features = ["peak_amplitude", "line_length", "recovery_tau"]

    return html.Div([
        html.H3("Evoked Features", style={"color": "white", "marginBottom": "12px"}),
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
                    value=0,  # all time by default
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "12px", "flexWrap": "wrap"}),

        html.Div([
            html.Label("Select features to display (each gets its own plot row):", style=LABEL_STYLE),
            dcc.Checklist(
                id="evoked-feature-checklist",
                options=feature_options,
                value=default_features,
                inline=True,
                style={"color": "#ddd", "fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px", "marginBottom": "4px"},
            ),
        ], style={**SECTION_STYLE, "marginBottom": "16px"}),

        # Hidden single-select for backward compat with callback
        dcc.Dropdown(id="evoked-feature-dropdown", value="peak_amplitude",
                     style={"display": "none"}),

        html.Div(id="evoked-multi-plots"),
        html.Div(id="evoked-stats"),
    ])


# ------------------------------------------------------------------ #
#  Criticality tab (existing)
# ------------------------------------------------------------------ #

def _criticality_tab_layout(store: Store):
    """Build the Criticality tab layout -- data loaded via callback."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    default_session = sessions[0]["session_dir"] if sessions else None

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="criticality-plot", style={"height": "550px"}),
    ])


# ------------------------------------------------------------------ #
#  LFP Browser tab (NEW)
# ------------------------------------------------------------------ #

def _lfp_browser_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)
    default = _default_session(store)

    return html.Div([
        html.H3("LFP Browser", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("File", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "2", "minWidth": "350px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("Load LFP", id="lfp-load-btn", n_clicks=0,
                            style={"backgroundColor": "#636EFA", "color": "white",
                                   "border": "none", "padding": "8px 20px",
                                   "borderRadius": "6px", "cursor": "pointer",
                                   "fontSize": "14px", "fontWeight": "bold"}),
            ], style={"flex": "0 0 120px", "display": "flex", "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        html.P("Loads the first 5 seconds of raw LFP from the selected .mat file.",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"}),

        dcc.Graph(id="lfp-plot", figure=_empty_fig("Select a session and file, then click Load", 600)),
    ])


# ------------------------------------------------------------------ #
#  Electrode Health tab (NEW)
# ------------------------------------------------------------------ #

def _electrode_health_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)
    default = _default_session(store)

    return html.Div([
        html.H3("Electrode Health", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="electrode-health-plot", figure=_empty_fig("Select a session", 700)),
    ])


# ------------------------------------------------------------------ #
#  Session Compare tab (NEW)
# ------------------------------------------------------------------ #

def _session_compare_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)

    return html.Div([
        html.H3("Session Compare", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session A", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-a-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Session B", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-b-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="session-compare-waveform-plot", style={"height": "450px"}),
        dcc.Graph(id="session-compare-features-plot", style={"height": "450px",
                                                              "marginTop": "16px"}),
    ])


# ------------------------------------------------------------------ #
#  Stim QC tab (existing)
# ------------------------------------------------------------------ #

def _stim_tab(store: Store):
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT pf.chunk_datetime, pf.session_name, sq.*
               FROM stim_qc sq JOIN processed_files pf ON sq.file_id = pf.id
               ORDER BY pf.chunk_datetime DESC LIMIT 500"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return html.Div("No stimulation data yet.", style={"color": "#888"})

    data = [dict(r) for r in rows]
    return html.Div([
        html.H3("Stimulation Delivery QC", style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{
                "time": d["chunk_datetime"][:16],
                "session": d.get("session_name", ""),
                "channel": d["stim_channel"],
                "charge_nC": f"{d['charge_nC']:.1f}" if d["charge_nC"] else "",
                "freq_Hz": f"{d['frequency_hz']:.1f}" if d["frequency_hz"] else "",
                "pulses": d["total_pulses"],
                "expected": d["expected_pulses"],
                "delivery_%": f"{d['delivery_pct']:.1f}" if d["delivery_pct"] else "N/A",
            } for d in data],
            columns=[{"name": c, "id": c} for c in
                     ["time", "session", "channel", "charge_nC", "freq_Hz",
                      "pulses", "expected", "delivery_%"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{delivery_%} contains '0.0'"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ),
    ])


# ------------------------------------------------------------------ #
#  Settings tab (existing)
# ------------------------------------------------------------------ #

def _settings_tab_layout(store: Store):
    """Build the Settings tab with all subsections."""
    try:
        cfg = _load_config()
    except Exception:
        cfg = {}

    ee = cfg.get("epoch_extraction", {})   # Step 1: extraction window
    fa = cfg.get("feature_analysis", {})   # Step 2: analysis sub-window
    cr = cfg.get("criticality", {})
    sz = cfg.get("seizure", {})
    ar = cfg.get("artifact", {})
    qc = cfg.get("qc_thresholds", {})
    ft = cfg.get("features", {})

    all_features = [
        "Line Length", "Log(AUC)", "Peak Amplitude", "Trough Amplitude",
        "Peak-to-Trough", "RMS Amplitude", "Peak Latency", "Trough Latency",
        "Max Slope", "Max Slope Time", "Early Area", "Late Area",
        "Early/Late Ratio", "Recovery Tau", "Recovery Slope",
        "Template Correlation", "PCA Recon Error", "Variance",
        "Autocorrelation", "AC Width", "Rolling Variance", "Rolling AR(1)",
        "Rolling CV", "Exp Fit A", "Sum Power Low", "Freq Moment Low",
        "Sum Power High", "Freq Moment High",
    ]

    versions = store.get_all_settings_versions()
    version_data = _version_table_data(versions)

    def _input(id_, val, type_="number", **kw):
        return dcc.Input(id=id_, value=val, type=type_, style=INPUT_STYLE, **kw)

    def _check(id_, val):
        return dcc.Checklist(
            id=id_, options=[{"label": " Enabled", "value": True}],
            value=[True] if val else [],
            style={"color": "#ddd"},
        )

    return html.Div([
        # --- Step 1: Epoch Extraction ---
        html.Div([
            html.H4("Step 1: Epoch Extraction", style={"color": "#636EFA", "marginTop": "0"}),
            html.P("Cuts a window around each detected stimulus to produce evoked.mat files. "
                   "These intermediary files can also be analyzed manually.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Pre-stimulus (ms, negative = before)", style=LABEL_STYLE),
                          _input("evoked-pre-stim", ee.get("pre_stimulus_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Post-stimulus (ms)", style=LABEL_STYLE),
                          _input("evoked-post-stim", ee.get("post_stimulus_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stimulus threshold (std)", style=LABEL_STYLE),
                          _input("evoked-stim-thresh", ee.get("stimulus_threshold_std", 3.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min stim distance (sec)", style=LABEL_STYLE),
                          _input("evoked-min-stim-dist", ee.get("min_stimulus_distance_sec", 0.1), step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Baseline correction", style=LABEL_STYLE),
                          _check("evoked-baseline-correction", ee.get("baseline_correction", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline start (ms)", style=LABEL_STYLE),
                          _input("evoked-baseline-start", ee.get("baseline_start_ms", -60))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline end (ms)", style=LABEL_STYLE),
                          _input("evoked-baseline-end", ee.get("baseline_end_ms", -10))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Notch 60 Hz", style=LABEL_STYLE),
                          _check("evoked-notch60", ee.get("notch_60hz", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch 50 Hz", style=LABEL_STYLE),
                          _check("evoked-notch50", ee.get("notch_50hz", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass enabled", style=LABEL_STYLE),
                          _check("evoked-hp-enabled", ee.get("highpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass cutoff (Hz)", style=LABEL_STYLE),
                          _input("evoked-hp-cutoff", ee.get("highpass_cutoff_hz", 1.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass enabled", style=LABEL_STYLE),
                          _check("evoked-lp-enabled", ee.get("lowpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass cutoff (Hz)", style=LABEL_STYLE),
                          _input("evoked-lp-cutoff", ee.get("lowpass_cutoff_hz", 1000.0))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE, "borderLeft": "3px solid #636EFA"}),

        # --- Step 2: Feature Analysis (Configure & Analyze) ---
        html.Div([
            html.H4("Step 2: Feature Analysis (Configure & Analyze)",
                     style={"color": "#00CC96", "marginTop": "0"}),
            html.P("Mirrors the Chronic Evoked Features dialog. Configures filtering, "
                   "artifact exclusion, and feature windows applied to extracted epochs.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),

            # Row 1: Evoked response window + stimulus artifact window
            html.H5("Windows", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Window start (ms)", style=LABEL_STYLE),
                          _input("feat-win-start", fa.get("window_start_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window end (ms)", style=LABEL_STYLE),
                          _input("feat-win-end", fa.get("window_end_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact start (ms)", style=LABEL_STYLE),
                          _input("feat-stim-art-start", fa.get("stim_artifact_start_ms", -5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact end (ms)", style=LABEL_STYLE),
                          _input("feat-stim-art-end", fa.get("stim_artifact_end_ms", 15))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 2: Filtering
            html.H5("Filtering", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Bandpass filter", style=LABEL_STYLE),
                          _check("feat-bp-enabled", fa.get("bandpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass (Hz)", style=LABEL_STYLE),
                          _input("feat-bp-hp", fa.get("bandpass_highpass_hz", 1.0), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass (Hz)", style=LABEL_STYLE),
                          _input("feat-bp-lp", fa.get("bandpass_lowpass_hz", 100.0), step=10)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch filter", style=LABEL_STYLE),
                          _check("feat-notch-enabled", fa.get("notch_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch freq (Hz)", style=LABEL_STYLE),
                          _input("feat-notch-freq", fa.get("notch_frequency_hz", 60))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 3: Smoothing + Baseline
            html.Div([
                html.Div([html.Label("Moving average smoothing", style=LABEL_STYLE),
                          _check("feat-smooth-enabled", fa.get("smoothing_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Smoothing window (ms)", style=LABEL_STYLE),
                          _input("feat-smooth-win", fa.get("smoothing_window_ms", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline correction", style=LABEL_STYLE),
                          _check("feat-baseline", fa.get("baseline_correction", True))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # Row 4: Artifact Exclusion
            html.H5("Artifact Exclusion", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Enable artifact exclusion", style=LABEL_STYLE),
                          _check("feat-art-enabled", fa.get("artifact_exclusion_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(id="feat-art-method",
                                       options=[{"label": m, "value": m} for m in
                                                ["fixed", "mad", "template", "rawamp", "noise"]],
                                       value=fa.get("artifact_method", "fixed"),
                                       style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Threshold / k-value", style=LABEL_STYLE),
                          _input("feat-art-thresh", fa.get("artifact_threshold", 500.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)", style=LABEL_STYLE),
                          _input("feat-art-merge", fa.get("artifact_merge_gap_sec", 2.0), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 5: Template-specific params
            html.Div([
                html.Div([html.Label("Template source", style=LABEL_STYLE),
                          dcc.Dropdown(id="feat-tmpl-source",
                                       options=[{"label": "Early 50ms", "value": "early50ms"},
                                                {"label": "Grand Mean", "value": "grandMean"}],
                                       value=fa.get("template_source", "early50ms"),
                                       style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Upper band (r)", style=LABEL_STYLE),
                          _input("feat-tmpl-upper", fa.get("template_upper_r", 0.7), step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lower band (r)", style=LABEL_STYLE),
                          _input("feat-tmpl-lower", fa.get("template_lower_r", 0.3), step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Raw amp multiplier (k)", style=LABEL_STYLE),
                          _input("feat-rawamp-k", fa.get("rawamp_multiplier", 4.0), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 6: Ictal rescue
            html.Div([
                html.Div([html.Label("Ictal spike rescue", style=LABEL_STYLE),
                          _check("feat-ictal-rescue", fa.get("ictal_rescue_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Rescue window (ms)", style=LABEL_STYLE),
                          _input("feat-ictal-win", fa.get("ictal_rescue_window_ms", 500))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # Row 7: Feature sub-windows
            html.H5("Feature Sub-Windows", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Analysis start (ms)", style=LABEL_STYLE),
                          _input("feat-analysis-start", fa.get("analysis_start_ms", 5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Analysis end (ms)", style=LABEL_STYLE),
                          _input("feat-analysis-end", fa.get("analysis_end_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area start (ms)", style=LABEL_STYLE),
                          _input("feat-early-start", fa.get("early_area_start_ms", 0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area end (ms)", style=LABEL_STYLE),
                          _input("feat-early-end", fa.get("early_area_end_ms", 50))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Late area start (ms)", style=LABEL_STYLE),
                          _input("feat-late-start", fa.get("late_area_start_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Late area end (ms)", style=LABEL_STYLE),
                          _input("feat-late-end", fa.get("late_area_end_ms", 200))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE, "borderLeft": "3px solid #00CC96"}),

        # --- Criticality ---
        html.Div([
            html.H4("Criticality", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("AR order", style=LABEL_STYLE),
                          _input("crit-ar-order", cr.get("ar_order", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window (sec)", style=LABEL_STYLE),
                          _input("crit-window-sec", cr.get("window_sec", 2.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Overlap %", style=LABEL_STYLE),
                          dcc.Slider(id="crit-overlap", min=0, max=90, step=10,
                                     value=cr.get("overlap_pct", 50),
                                     marks={i: str(i) for i in range(0, 91, 10)},
                                     tooltip={"placement": "bottom"})],
                         style={**FIELD_STYLE, "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Fit method", style=LABEL_STYLE),
                          dcc.Dropdown(id="crit-fit-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["YuleWalker", "Burg", "Covariance"]],
                                       value=cr.get("fit_method", "YuleWalker"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Target SR", style=LABEL_STYLE),
                          _input("crit-target-sr", cr.get("target_sampling_rate", 1000))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Type B", style=LABEL_STYLE),
                          _input("crit-type-b", cr.get("criticality_type_b", 2), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Error bars", style=LABEL_STYLE),
                          _check("crit-error-bars", cr.get("calculate_error_bars", False))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Seizure ---
        html.Div([
            html.H4("Seizure Detection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Spike threshold (uV)", style=LABEL_STYLE),
                          _input("seiz-spike-thresh", sz.get("spike_threshold_uv", 10))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike min width", style=LABEL_STYLE),
                          _input("seiz-spike-min-w", sz.get("spike_min_width", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike max width", style=LABEL_STYLE),
                          _input("seiz-spike-max-w", sz.get("spike_max_width", 50), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min seizure dur (sec)", style=LABEL_STYLE),
                          _input("seiz-min-dur", sz.get("min_seizure_duration_sec", 5), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Event glue (sec)", style=LABEL_STYLE),
                          _input("seiz-glue-sec", sz.get("event_glue_sec", 2), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min spikes/sec", style=LABEL_STYLE),
                          _input("seiz-min-spikes", sz.get("min_spikes_per_sec", 2), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Outlier factor", style=LABEL_STYLE),
                          _input("seiz-outlier", sz.get("outlier_factor", 3), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Artifact ---
        html.Div([
            html.H4("Artifact Rejection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(id="art-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["fixed", "mad"]],
                                       value=ar.get("method", "fixed"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Fixed threshold", style=LABEL_STYLE),
                          _input("art-fixed-thresh", ar.get("fixed_threshold", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("MAD k", style=LABEL_STYLE),
                          _input("art-mad-k", ar.get("mad_k", 4.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)", style=LABEL_STYLE),
                          _input("art-merge-gap", ar.get("merge_gap_sec", 2.0), step=0.1)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- QC Thresholds ---
        html.Div([
            html.H4("QC Thresholds", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Artifact % warning", style=LABEL_STYLE),
                          _input("qc-artifact-warn", qc.get("artifact_pct_warning", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Flatline std", style=LABEL_STYLE),
                          _input("qc-flatline-std", qc.get("flatline_std", 1e-6), step=1e-7)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Clipping voltage", style=LABEL_STYLE),
                          _input("qc-clipping-v", qc.get("clipping_voltage", 10.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Line noise ratio warning", style=LABEL_STYLE),
                          _input("qc-linenoise-warn", qc.get("line_noise_ratio_warning", 0.2), step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Features ---
        html.Div([
            html.H4("Features", style={"color": "white", "marginTop": "0"}),
            dcc.Checklist(
                id="features-checklist",
                options=[{"label": f"  {f}", "value": f} for f in all_features],
                value=ft.get("enabled", all_features),
                style={"color": "#ddd", "columns": "3", "columnGap": "20px"},
                inputStyle={"marginRight": "6px"},
            ),
        ], style=SECTION_STYLE),

        # --- Buttons ---
        html.Div([
            html.Button("Save Settings", id="btn-save-settings", n_clicks=0,
                         style={"backgroundColor": "#636EFA", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            dcc.Input(id="version-label-input", placeholder="Version label (optional)",
                      style={**INPUT_STYLE, "width": "250px"}),
            html.Button("Save as New Version", id="btn-save-version", n_clicks=0,
                         style={"backgroundColor": "#00CC96", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            html.Button("Reprocess Session", id="btn-reprocess", n_clicks=0,
                         style={"backgroundColor": "#FFA15A", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center",
                  "flexWrap": "wrap", "marginBottom": "12px"}),

        html.Div(id="settings-save-status"),
        html.Div(id="settings-version-status"),
        html.Div(id="reprocess-status"),

        # --- Version history ---
        html.Div([
            html.H4("Version History", style={"color": "white", "marginTop": "24px"}),
            dash_table.DataTable(
                id="version-history-table",
                data=version_data,
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Label", "id": "label"},
                    {"name": "Created", "id": "created_at"},
                    {"name": "Active", "id": "is_active"},
                    {"name": "Hash (short)", "id": "hash_short"},
                ],
                **DARK_TABLE_STYLE,
                style_data_conditional=[ZEBRA_STRIPE,
                    {"if": {"filter_query": "{is_active} = Yes"},
                     "backgroundColor": "#112211", "color": "#00CC96"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ], style=SECTION_STYLE),
    ])


# ------------------------------------------------------------------ #
#  Activity Log tab (NEW)
# ------------------------------------------------------------------ #

def _activity_log_tab_layout(store: Store):
    # Pre-load initial data
    try:
        initial_logs = store.get_activity_log(hours=24, limit=500)
    except Exception:
        initial_logs = []

    initial_data = [
        {
            "timestamp": entry.get("timestamp", "")[:19],
            "level": entry.get("level", ""),
            "action": entry.get("action", ""),
            "message": (entry.get("message") or "")[:200],
            "file_path": os.path.basename(entry.get("file_path") or ""),
            "duration": f"{entry['duration_sec']:.2f}" if entry.get("duration_sec") else "",
        }
        for entry in initial_logs
    ]

    return html.Div([
        html.H3("Activity Log", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=24,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
            html.Div([
                html.Label("Level", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-level-dropdown",
                    options=[
                        {"label": "ALL", "value": "ALL"},
                        {"label": "INFO", "value": "INFO"},
                        {"label": "WARNING", "value": "WARNING"},
                        {"label": "ERROR", "value": "ERROR"},
                    ],
                    value="ALL",
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 150px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dash_table.DataTable(
            id="activity-log-table",
            data=initial_data,
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Level", "id": "level"},
                {"name": "Action", "id": "action"},
                {"name": "Message", "id": "message"},
                {"name": "File", "id": "file_path"},
                {"name": "Duration (s)", "id": "duration"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{level} = ERROR"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{level} = WARNING"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{level} = INFO"},
                 "color": "#6bb5ff"},
            ],
            page_size=30,
            filter_action="native",
            sort_action="native",
        ),
    ])


# ------------------------------------------------------------------ #
#  Annotations tab (NEW)
# ------------------------------------------------------------------ #

def _annotations_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)

    try:
        all_annotations = store.get_annotations()
    except Exception:
        all_annotations = []

    table_data = _annotations_table_data(all_annotations)

    return html.Div([
        html.H3("Annotations", style={"color": "white", "marginBottom": "12px"}),

        # Existing annotations table
        dash_table.DataTable(
            id="annotation-table",
            data=table_data,
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Category", "id": "category"},
                {"name": "Note", "id": "note"},
                {"name": "Session", "id": "session_dir"},
                {"name": "Created At", "id": "created_at"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{category} = electrode"},
                 "color": "#FFA15A"},
                {"if": {"filter_query": "{category} = injection"},
                 "color": "#AB63FA"},
                {"if": {"filter_query": "{category} = experiment"},
                 "color": "#636EFA"},
                {"if": {"filter_query": "{category} = observation"},
                 "color": "#00CC96"},
            ],
            page_size=20,
            filter_action="native",
            sort_action="native",
        ),

        # Add Note form
        html.Div([
            html.H4("Add Note", style={"color": "white", "marginTop": "24px", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Timestamp", style=LABEL_STYLE),
                    dcc.Input(
                        id="annotation-timestamp",
                        value=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        type="text",
                        style=INPUT_STYLE,
                        placeholder="YYYY-MM-DDTHH:MM:SS",
                    ),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Label("Category", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-category",
                        options=[
                            {"label": "Electrode", "value": "electrode"},
                            {"label": "Injection", "value": "injection"},
                            {"label": "Experiment", "value": "experiment"},
                            {"label": "Observation", "value": "observation"},
                        ],
                        value="observation",
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "0 0 180px"}),
                html.Div([
                    html.Label("Session (optional)", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-session-dropdown",
                        options=session_options,
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "1", "minWidth": "250px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Label("Note", style=LABEL_STYLE),
                dcc.Textarea(
                    id="annotation-note",
                    style={**INPUT_STYLE, "height": "80px", "resize": "vertical"},
                    placeholder="Enter your annotation...",
                ),
            ], style={"marginBottom": "12px"}),

            html.Button("Submit Annotation", id="annotation-submit-btn", n_clicks=0,
                        style={"backgroundColor": "#00CC96", "color": "white",
                               "border": "none", "padding": "10px 24px",
                               "borderRadius": "6px", "cursor": "pointer",
                               "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(id="annotation-status", style={"marginTop": "8px"}),
        ], style=SECTION_STYLE),
    ])


def _annotations_table_data(annotations: list[dict]) -> list[dict]:
    return [
        {
            "id": a.get("id", ""),
            "timestamp": (a.get("timestamp") or "")[:19],
            "category": a.get("category", ""),
            "note": a.get("note", ""),
            "session_dir": os.path.basename(a.get("session_dir") or ""),
            "created_at": (a.get("created_at") or "")[:19],
        }
        for a in annotations
    ]


# ------------------------------------------------------------------ #
#  Alerts tab (existing)
# ------------------------------------------------------------------ #

def _alerts_tab(store: Store):
    alerts = store.get_recent_alerts(hours=168)  # 7 days
    return html.Div([
        html.H3(f"Alert History (last 7 days) -- {len(alerts)} alerts",
                 style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                   "type": a["alert_type"], "message": a["message"],
                   "session": a.get("session_dir", "")}
                  for a in alerts],
            columns=[{"name": c, "id": c}
                     for c in ["time", "severity", "type", "message", "session"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{severity} = critical"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{severity} = warning"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{severity} = info"},
                 "backgroundColor": "#112233", "color": "#6bb5ff"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ) if alerts else html.P("No alerts in the last 7 days", style={"color": "#888"}),
    ])


# ------------------------------------------------------------------ #
#  Sessions tab (existing)
# ------------------------------------------------------------------ #

def _sessions_tab(store: Store):
    sessions = store.get_sessions()
    all_configs = store.get_all_session_configs()
    config_by_dir = {c["session_dir"]: c for c in all_configs}

    rows = []
    for s in sessions:
        cfg = config_by_dir.get(s["session_dir"], {})
        ch_names = _parse_json_field(cfg.get("channel_names"))
        num_ch = cfg.get("num_channels") or (len(ch_names) if ch_names else "")
        sr = cfg.get("sampling_rate", "")
        stim_freq = cfg.get("stim_frequency_hz", "")
        stim_charge = cfg.get("stim_charge_nC", "")

        rows.append({
            "name": s["session_name"],
            "dir": s["session_dir"],
            "files": s["num_files"],
            "processed": s["processed"],
            "errors": s["errors"],
            "first": s["first_chunk"][:16] if s["first_chunk"] else "",
            "last": s["last_chunk"][:16] if s["last_chunk"] else "",
            "channels": num_ch,
            "sr": sr,
            "stim_freq": f"{stim_freq}" if stim_freq else "",
            "stim_charge": f"{stim_charge}" if stim_charge else "",
        })

    return html.Div([
        html.H3(f"Sessions ({len(sessions)})",
                 style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=rows,
            columns=[
                {"name": "Session", "id": "name"},
                {"name": "Directory", "id": "dir"},
                {"name": "Files", "id": "files"},
                {"name": "Processed", "id": "processed"},
                {"name": "Errors", "id": "errors"},
                {"name": "First Chunk", "id": "first"},
                {"name": "Last Chunk", "id": "last"},
                {"name": "Channels", "id": "channels"},
                {"name": "SR (Hz)", "id": "sr"},
                {"name": "Stim Freq", "id": "stim_freq"},
                {"name": "Stim Charge (nC)", "id": "stim_charge"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{errors} > 0"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=20,
            sort_action="native",
            filter_action="native",
        ),
    ])


# ------------------------------------------------------------------ #
#  Version table helper
# ------------------------------------------------------------------ #

def _version_table_data(versions: list[dict]) -> list[dict]:
    return [
        {
            "id": v["id"],
            "label": v.get("label", ""),
            "created_at": v["created_at"][:19],
            "is_active": "Yes" if v.get("is_active") else "No",
            "hash_short": v.get("version_hash", "")[:12],
        }
        for v in versions
    ]
