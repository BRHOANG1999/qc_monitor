"""Dash web dashboard for the QC Monitor — v2 with evoked, criticality, settings."""

import json
import logging
import os
from datetime import datetime

import yaml
from dash import Dash, html, dcc, dash_table, callback_context, no_update, ALL
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.db.store import Store

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

# Dark card style
CARD_STYLE = {
    "backgroundColor": "#1e1e2f",
    "borderRadius": "8px",
    "padding": "15px 20px",
    "minWidth": "130px",
    "textAlign": "center",
    "border": "1px solid #333",
}

TAB_STYLE = {"backgroundColor": "#0d0d1a", "color": "#aaa", "padding": "10px 20px",
             "border": "1px solid #333", "borderBottom": "none"}
TAB_SELECTED_STYLE = {"backgroundColor": "#1a1a2e", "color": "white", "padding": "10px 20px",
                      "border": "1px solid #555", "borderBottom": "none", "fontWeight": "bold"}

DARK_TABLE_STYLE = {
    "style_header": {"backgroundColor": "#1a1a2e", "color": "white", "fontWeight": "bold",
                     "border": "1px solid #444"},
    "style_data": {"backgroundColor": "#111", "color": "#ddd", "border": "1px solid #333"},
    "style_cell": {"textAlign": "left", "padding": "8px", "fontSize": "13px"},
    "style_filter": {"backgroundColor": "#1a1a2e", "color": "white"},
}


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


def create_app(config: dict, store: Store) -> Dash:
    refresh_sec = config.get("dashboard", {}).get("refresh_interval_sec", 10)

    app = Dash(__name__, title="QC Monitor", suppress_callback_exceptions=True)

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("QC Monitor", style={"margin": "0", "fontSize": "20px"}),
        ], style={"padding": "12px 24px", "backgroundColor": "#0d0d1a", "color": "white",
                  "borderBottom": "1px solid #333"}),

        # Tabs
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Overview", value="overview",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Signal Quality", value="signal",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Evoked Response", value="evoked",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Criticality", value="criticality",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Stim QC", value="stim",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Settings", value="settings",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Alerts", value="alerts",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            dcc.Tab(label="Sessions", value="sessions",
                    style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        ], style={"borderBottom": "none"}),

        html.Div(id="tab-content", style={"padding": "20px", "backgroundColor": "#111",
                                           "minHeight": "80vh"}),

        dcc.Interval(id="refresh", interval=refresh_sec * 1000, n_intervals=0),
        # Hidden stores for inter-callback data
        dcc.Store(id="selected-session-dir"),
    ], style={"backgroundColor": "#111", "fontFamily": "Segoe UI, sans-serif", "color": "#ddd"})

    # ------------------------------------------------------------------ #
    #  Main tab router
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "value"), Input("refresh", "n_intervals")]
    )
    def render_tab(tab, _n):
        try:
            if tab == "overview":
                return _overview_tab(store)
            elif tab == "signal":
                return _signal_quality_tab(store)
            elif tab == "evoked":
                return _evoked_tab_layout(store)
            elif tab == "criticality":
                return _criticality_tab_layout(store)
            elif tab == "stim":
                return _stim_tab(store)
            elif tab == "settings":
                return _settings_tab_layout(store)
            elif tab == "alerts":
                return _alerts_tab(store)
            elif tab == "sessions":
                return _sessions_tab(store)
        except Exception as e:
            logger.error("Dashboard render error: %s", e, exc_info=True)
            return html.Div(f"Error rendering tab: {e}",
                            style={"color": "#ff6b6b", "padding": "20px"})

    # ------------------------------------------------------------------ #
    #  Evoked Response callback
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("evoked-scatter", "figure"),
         Output("evoked-stats", "children")],
        [Input("evoked-feature-dropdown", "value"),
         Input("evoked-session-dropdown", "value"),
         Input("evoked-hours-dropdown", "value")],
    )
    def update_evoked(feature_name, session_dir, hours):
        if not feature_name or not session_dir:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark",
                                annotations=[dict(text="Select a session and feature",
                                                  showarrow=False, font=dict(size=16))])
            return empty, html.P("No data")

        try:
            data = store.get_evoked_feature_timeseries(
                feature_name=feature_name,
                session_dir=session_dir,
                hours=int(hours) if hours else None,
            )
        except Exception as e:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark",
                                annotations=[dict(text=f"Error: {e}",
                                                  showarrow=False, font=dict(size=14, color="red"))])
            return empty, html.P(f"Error: {e}")

        if not data:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark",
                                annotations=[dict(text="No evoked data for this selection",
                                                  showarrow=False, font=dict(size=16))])
            return empty, html.P("No data available")

        # Separate by type
        clean_t, clean_v = [], []
        artifact_t, artifact_v = [], []
        ictal_t, ictal_v = [], []
        all_values = []

        for d in data:
            t = d["chunk_datetime"]
            v = d["value"]
            if v is None:
                continue
            all_values.append(v)
            is_art = d.get("is_artifact", 0)
            is_ict = d.get("is_ictal", 0)
            if is_art:
                artifact_t.append(t)
                artifact_v.append(v)
            elif is_ict:
                ictal_t.append(t)
                ictal_v.append(v)
            else:
                clean_t.append(t)
                clean_v.append(v)

        fig = go.Figure()
        if clean_t:
            fig.add_trace(go.Scatter(
                x=clean_t, y=clean_v, mode="markers", name="Clean",
                marker=dict(color="#636EFA", size=5, opacity=0.7)))
        if artifact_t:
            fig.add_trace(go.Scatter(
                x=artifact_t, y=artifact_v, mode="markers", name="Artifact",
                marker=dict(color="#EF553B", size=6, opacity=0.8)))
        if ictal_t:
            fig.add_trace(go.Scatter(
                x=ictal_t, y=ictal_v, mode="markers", name="Ictal",
                marker=dict(color="#FFA15A", size=6, opacity=0.8)))

        label = EVOKED_FEATURE_LABELS.get(feature_name, feature_name)
        fig.update_layout(
            template="plotly_dark",
            title=f"{label} per Epoch Over Time",
            xaxis_title="Time",
            yaxis_title=label,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Summary stats
        n_total = len(all_values)
        n_art = len(artifact_v)
        n_ict = len(ictal_v)
        n_clean = len(clean_v)
        if all_values:
            import statistics
            mean_val = statistics.mean(all_values)
            std_val = statistics.stdev(all_values) if len(all_values) > 1 else 0.0
        else:
            mean_val, std_val = 0, 0

        stats = html.Div([
            html.Div([
                _status_card("Mean", f"{mean_val:.4g}", "#636EFA"),
                _status_card("Std", f"{std_val:.4g}", "#AB63FA"),
                _status_card("N Epochs", str(n_total), "#00CC96"),
                _status_card("N Artifact", str(n_art), "#EF553B"),
                _status_card("N Ictal", str(n_ict), "#FFA15A"),
                _status_card("N Clean", str(n_clean), "#636EFA"),
            ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "10px"}),
        ])

        return fig, stats

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
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              annotations=[dict(text="Select a session",
                                                showarrow=False, font=dict(size=16))])
            return fig

        ch_map = _get_channel_map(store, session_dir)

        try:
            data = store.query_criticality_timeseries(
                session_dir=session_dir,
                hours=int(hours) if hours else None,
            )
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              annotations=[dict(text=f"Error: {e}",
                                                showarrow=False, font=dict(size=14, color="red"))])
            return fig

        if not data:
            fig = go.Figure()
            fig.update_layout(template="plotly_dark",
                              annotations=[dict(text="No criticality data",
                                                showarrow=False, font=dict(size=16))])
            return fig

        # Group by channel
        channels = sorted(set(d["channel"] for d in data))

        # Only plot EEG channels if we have a channel map
        fig = go.Figure()
        for ch in channels:
            ch_data = [d for d in data if d["channel"] == ch]
            times = [d["chunk_datetime"] for d in ch_data]
            vals = [d["db_value"] for d in ch_data]

            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            # Only plot EEG channels
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
            title="Criticality (dB) Over Time — EEG Channels",
            xaxis_title="Time",
            yaxis_title="dB Value",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

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
         State("features-checklist", "value")],
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
             features_enabled) = values

            cfg["evoked"]["pre_stimulus_ms"] = float(pre_stim) if pre_stim is not None else cfg["evoked"]["pre_stimulus_ms"]
            cfg["evoked"]["post_stimulus_ms"] = float(post_stim) if post_stim is not None else cfg["evoked"]["post_stimulus_ms"]
            cfg["evoked"]["baseline_correction"] = bool(baseline_corr)
            cfg["evoked"]["baseline_window_ms"] = [float(bl_start or -60), float(bl_end or -10)]
            cfg["evoked"]["stimulus_threshold_std"] = float(stim_thresh) if stim_thresh is not None else cfg["evoked"]["stimulus_threshold_std"]
            cfg["evoked"]["min_stimulus_distance_sec"] = float(min_stim_dist) if min_stim_dist is not None else cfg["evoked"]["min_stimulus_distance_sec"]
            cfg["evoked"]["notch_60hz"] = bool(notch60)
            cfg["evoked"]["notch_50hz"] = bool(notch50)
            cfg["evoked"]["highpass_enabled"] = bool(hp_en)
            cfg["evoked"]["highpass_cutoff_hz"] = float(hp_cut) if hp_cut is not None else cfg["evoked"]["highpass_cutoff_hz"]
            cfg["evoked"]["lowpass_enabled"] = bool(lp_en)
            cfg["evoked"]["lowpass_cutoff_hz"] = float(lp_cut) if lp_cut is not None else cfg["evoked"]["lowpass_cutoff_hz"]

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
                "evoked": {
                    "pre_stimulus_ms": float(pre_stim or -100),
                    "post_stimulus_ms": float(post_stim or 500),
                    "baseline_correction": bool(baseline_corr),
                    "baseline_window_ms": [float(bl_start or -60), float(bl_end or -10)],
                    "stimulus_threshold_std": float(stim_thresh or 3.0),
                    "min_stimulus_distance_sec": float(min_stim_dist or 0.1),
                    "notch_60hz": bool(notch60),
                    "notch_50hz": bool(notch50),
                    "highpass_enabled": bool(hp_en),
                    "highpass_cutoff_hz": float(hp_cut or 1.0),
                    "lowpass_enabled": bool(lp_en),
                    "lowpass_cutoff_hz": float(lp_cut or 1000.0),
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

    # Channel map for active session
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

        # Channel map display
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
                    style_data_conditional=[
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
                style_data_conditional=[
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

    return html.Div([cards, session_info, channel_table, alerts_section])


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


def _evoked_tab_layout(store: Store):
    """Build the Evoked Response tab layout — data loaded via callback."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    feature_options = [{"label": EVOKED_FEATURE_LABELS.get(f, f), "value": f}
                       for f in EVOKED_FEATURE_COLS
                       if f not in ("is_artifact", "is_ictal")]

    default_session = sessions[0]["session_dir"] if sessions else None

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Session", style={"color": "#888", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="evoked-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style={"backgroundColor": "#1e1e2f", "color": "white"},
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Feature", style={"color": "#888", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="evoked-feature-dropdown",
                    options=feature_options,
                    value="peak_amplitude",
                    style={"backgroundColor": "#1e1e2f", "color": "white"},
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Hours", style={"color": "#888", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="evoked-hours-dropdown",
                    options=[
                        {"label": "Last 6h", "value": 6},
                        {"label": "Last 24h", "value": 24},
                        {"label": "Last 48h", "value": 48},
                        {"label": "Last 7d", "value": 168},
                        {"label": "All", "value": ""},
                    ],
                    value=48,
                    style={"backgroundColor": "#1e1e2f", "color": "white"},
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 150px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="evoked-scatter", style={"height": "500px"}),
        html.Div(id="evoked-stats"),
    ])


def _criticality_tab_layout(store: Store):
    """Build the Criticality tab layout — data loaded via callback."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    default_session = sessions[0]["session_dir"] if sessions else None

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Session", style={"color": "#888", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="criticality-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style={"backgroundColor": "#1e1e2f", "color": "white"},
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Hours", style={"color": "#888", "fontSize": "12px"}),
                dcc.Dropdown(
                    id="criticality-hours-dropdown",
                    options=[
                        {"label": "Last 6h", "value": 6},
                        {"label": "Last 24h", "value": 24},
                        {"label": "Last 48h", "value": 48},
                        {"label": "Last 7d", "value": 168},
                        {"label": "All", "value": ""},
                    ],
                    value=48,
                    style={"backgroundColor": "#1e1e2f", "color": "white"},
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 150px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="criticality-plot", style={"height": "550px"}),
    ])


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
            style_data_conditional=[
                {"if": {"filter_query": "{delivery_%} contains '0.0'"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ),
    ])


def _settings_tab_layout(store: Store):
    """Build the Settings tab with all subsections."""
    try:
        cfg = _load_config()
    except Exception:
        cfg = {}

    ev = cfg.get("evoked", {})
    cr = cfg.get("criticality", {})
    sz = cfg.get("seizure", {})
    ar = cfg.get("artifact", {})
    qc = cfg.get("qc_thresholds", {})
    ft = cfg.get("features", {})
    bl_window = ev.get("baseline_window_ms", [-60, -10])

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

    # Version history
    versions = store.get_all_settings_versions()
    version_data = _version_table_data(versions)

    section_style = {"backgroundColor": "#1e1e2f", "padding": "16px", "borderRadius": "8px",
                     "border": "1px solid #333", "marginBottom": "16px"}
    label_style = {"color": "#888", "fontSize": "12px", "marginBottom": "2px",
                   "display": "block"}
    input_style = {"backgroundColor": "#111", "color": "white", "border": "1px solid #444",
                   "borderRadius": "4px", "padding": "4px 8px", "width": "100%"}
    field_style = {"flex": "1", "minWidth": "180px"}

    def _input(id_, val, type_="number", **kw):
        return dcc.Input(id=id_, value=val, type=type_, style=input_style, **kw)

    def _check(id_, val):
        return dcc.Checklist(
            id=id_, options=[{"label": " Enabled", "value": True}],
            value=[True] if val else [],
            style={"color": "#ddd"},
        )

    return html.Div([
        # --- Evoked ---
        html.Div([
            html.H4("Evoked Analysis", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Pre-stimulus (ms)", style=label_style),
                          _input("evoked-pre-stim", ev.get("pre_stimulus_ms", -100))],
                         style=field_style),
                html.Div([html.Label("Post-stimulus (ms)", style=label_style),
                          _input("evoked-post-stim", ev.get("post_stimulus_ms", 500))],
                         style=field_style),
                html.Div([html.Label("Stimulus threshold (std)", style=label_style),
                          _input("evoked-stim-thresh", ev.get("stimulus_threshold_std", 3.0), step=0.1)],
                         style=field_style),
                html.Div([html.Label("Min stim distance (sec)", style=label_style),
                          _input("evoked-min-stim-dist", ev.get("min_stimulus_distance_sec", 0.1), step=0.01)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Baseline correction", style=label_style),
                          _check("evoked-baseline-correction", ev.get("baseline_correction", True))],
                         style=field_style),
                html.Div([html.Label("Baseline start (ms)", style=label_style),
                          _input("evoked-baseline-start", bl_window[0] if len(bl_window) > 0 else -60)],
                         style=field_style),
                html.Div([html.Label("Baseline end (ms)", style=label_style),
                          _input("evoked-baseline-end", bl_window[1] if len(bl_window) > 1 else -10)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Notch 60 Hz", style=label_style),
                          _check("evoked-notch60", ev.get("notch_60hz", True))],
                         style=field_style),
                html.Div([html.Label("Notch 50 Hz", style=label_style),
                          _check("evoked-notch50", ev.get("notch_50hz", False))],
                         style=field_style),
                html.Div([html.Label("Highpass enabled", style=label_style),
                          _check("evoked-hp-enabled", ev.get("highpass_enabled", False))],
                         style=field_style),
                html.Div([html.Label("Highpass cutoff (Hz)", style=label_style),
                          _input("evoked-hp-cutoff", ev.get("highpass_cutoff_hz", 1.0), step=0.1)],
                         style=field_style),
                html.Div([html.Label("Lowpass enabled", style=label_style),
                          _check("evoked-lp-enabled", ev.get("lowpass_enabled", False))],
                         style=field_style),
                html.Div([html.Label("Lowpass cutoff (Hz)", style=label_style),
                          _input("evoked-lp-cutoff", ev.get("lowpass_cutoff_hz", 1000.0))],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=section_style),

        # --- Criticality ---
        html.Div([
            html.H4("Criticality", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("AR order", style=label_style),
                          _input("crit-ar-order", cr.get("ar_order", 5), step=1)],
                         style=field_style),
                html.Div([html.Label("Window (sec)", style=label_style),
                          _input("crit-window-sec", cr.get("window_sec", 2.0), step=0.1)],
                         style=field_style),
                html.Div([html.Label("Overlap %", style=label_style),
                          dcc.Slider(id="crit-overlap", min=0, max=90, step=10,
                                     value=cr.get("overlap_pct", 50),
                                     marks={i: str(i) for i in range(0, 91, 10)},
                                     tooltip={"placement": "bottom"})],
                         style={**field_style, "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Fit method", style=label_style),
                          dcc.Dropdown(id="crit-fit-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["YuleWalker", "Burg", "Covariance"]],
                                       value=cr.get("fit_method", "YuleWalker"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=field_style),
                html.Div([html.Label("Target SR", style=label_style),
                          _input("crit-target-sr", cr.get("target_sampling_rate", 1000))],
                         style=field_style),
                html.Div([html.Label("Type B", style=label_style),
                          _input("crit-type-b", cr.get("criticality_type_b", 2), step=1)],
                         style=field_style),
                html.Div([html.Label("Error bars", style=label_style),
                          _check("crit-error-bars", cr.get("calculate_error_bars", False))],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=section_style),

        # --- Seizure ---
        html.Div([
            html.H4("Seizure Detection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Spike threshold (uV)", style=label_style),
                          _input("seiz-spike-thresh", sz.get("spike_threshold_uv", 10))],
                         style=field_style),
                html.Div([html.Label("Spike min width", style=label_style),
                          _input("seiz-spike-min-w", sz.get("spike_min_width", 5), step=1)],
                         style=field_style),
                html.Div([html.Label("Spike max width", style=label_style),
                          _input("seiz-spike-max-w", sz.get("spike_max_width", 50), step=1)],
                         style=field_style),
                html.Div([html.Label("Min seizure dur (sec)", style=label_style),
                          _input("seiz-min-dur", sz.get("min_seizure_duration_sec", 5), step=0.5)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Event glue (sec)", style=label_style),
                          _input("seiz-glue-sec", sz.get("event_glue_sec", 2), step=0.5)],
                         style=field_style),
                html.Div([html.Label("Min spikes/sec", style=label_style),
                          _input("seiz-min-spikes", sz.get("min_spikes_per_sec", 2), step=0.5)],
                         style=field_style),
                html.Div([html.Label("Outlier factor", style=label_style),
                          _input("seiz-outlier", sz.get("outlier_factor", 3), step=0.5)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=section_style),

        # --- Artifact ---
        html.Div([
            html.H4("Artifact Rejection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Method", style=label_style),
                          dcc.Dropdown(id="art-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["fixed", "mad"]],
                                       value=ar.get("method", "fixed"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=field_style),
                html.Div([html.Label("Fixed threshold", style=label_style),
                          _input("art-fixed-thresh", ar.get("fixed_threshold", 500))],
                         style=field_style),
                html.Div([html.Label("MAD k", style=label_style),
                          _input("art-mad-k", ar.get("mad_k", 4.0), step=0.1)],
                         style=field_style),
                html.Div([html.Label("Merge gap (sec)", style=label_style),
                          _input("art-merge-gap", ar.get("merge_gap_sec", 2.0), step=0.1)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=section_style),

        # --- QC Thresholds ---
        html.Div([
            html.H4("QC Thresholds", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Artifact % warning", style=label_style),
                          _input("qc-artifact-warn", qc.get("artifact_pct_warning", 50))],
                         style=field_style),
                html.Div([html.Label("Flatline std", style=label_style),
                          _input("qc-flatline-std", qc.get("flatline_std", 1e-6), step=1e-7)],
                         style=field_style),
                html.Div([html.Label("Clipping voltage", style=label_style),
                          _input("qc-clipping-v", qc.get("clipping_voltage", 10.0))],
                         style=field_style),
                html.Div([html.Label("Line noise ratio warning", style=label_style),
                          _input("qc-linenoise-warn", qc.get("line_noise_ratio_warning", 0.2), step=0.01)],
                         style=field_style),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=section_style),

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
        ], style=section_style),

        # --- Buttons ---
        html.Div([
            html.Button("Save Settings", id="btn-save-settings", n_clicks=0,
                         style={"backgroundColor": "#636EFA", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            dcc.Input(id="version-label-input", placeholder="Version label (optional)",
                      style={**input_style, "width": "250px"}),
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
                style_data_conditional=[
                    {"if": {"filter_query": "{is_active} = Yes"},
                     "backgroundColor": "#112211", "color": "#00CC96"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ], style=section_style),
    ])


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
            style_data_conditional=[
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
            style_data_conditional=[
                {"if": {"filter_query": "{errors} > 0"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=20,
            sort_action="native",
            filter_action="native",
        ),
    ])
