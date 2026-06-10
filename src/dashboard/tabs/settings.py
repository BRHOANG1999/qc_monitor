"""Settings tab -- edit config.yaml + manage settings versions in the
DB + queue a reprocess.

Three responsibilities, three callbacks:

* ``save_settings`` -- collect every input's State, merge into the
  config dict (defensive ``if v is not None: cfg[key] = ...`` per
  field), and round-trip to ``config/config.yaml``. Settings are
  picked up next time anything reads the file -- no restart needed.
* ``save_version`` -- build a clean settings dict from the same
  inputs and hand it to ``store.create_settings_version``. The
  version_hash dedupe inside Store means clicking "Save as New
  Version" with no field changes is a no-op.
* ``reprocess_session`` -- flip every ``status='done'`` row in
  ``processed_files`` back to ``pending`` so the dispatcher picks
  them up on the next poll. Mainly used when an analysis-pipeline
  config edit means the existing results are stale.

The layout itself is the bulk of the module (~390 lines of form
plumbing). It's verbatim from the original app.py move with the
internal ``_input`` and ``_check`` closures lifted to module-level
helpers for legibility.

Lifted out of ``src/dashboard/app.py``.
"""

from __future__ import annotations

import logging

from dash import Input, Output, State, dash_table, dcc, html, no_update

from src.dashboard.components import (
    DARK_TABLE_STYLE, DROPDOWN_STYLE, FIELD_STYLE, INPUT_STYLE,
    LABEL_STYLE, SECTION_STYLE, ZEBRA_STRIPE,
)
from src.dashboard.data_helpers import load_config, save_config
from src.db.store import Store

logger = logging.getLogger("qc_monitor.dashboard.settings")

# The full feature list shown in the "Features" checklist. Keys are
# the human-readable names; the same strings are persisted to
# ``features.enabled`` in config.yaml and read by the MATLAB pipeline.
_ALL_FEATURES = [
    "Line Length", "Log(AUC)", "Peak Amplitude", "Trough Amplitude",
    "Peak-to-Trough", "RMS Amplitude", "Peak Latency", "Trough Latency",
    "Max Slope", "Max Slope Time", "Early Area", "Late Area",
    "Early/Late Ratio", "Recovery Tau", "Recovery Slope",
    "Template Correlation", "PCA Recon Error", "Variance",
    "Autocorrelation", "AC Width", "Rolling Variance", "Rolling AR(1)",
    "Rolling CV", "Exp Fit A", "Sum Power Low", "Freq Moment Low",
    "Sum Power High", "Freq Moment High",
]


def _input(id_, val, type_: str = "number", **kw):
    """Wrapper that applies the standard form INPUT_STYLE to every
    numeric / text input in this tab."""
    return dcc.Input(id=id_, value=val, type=type_,
                     style=INPUT_STYLE, **kw)


def _check(id_, val):
    """Standard single-item Enabled checklist used dozens of times.
    *val* is a bool; we render it as the [True] / [] dance Dash
    expects for a Checklist."""
    return dcc.Checklist(
        id=id_,
        options=[{"label": " Enabled", "value": True}],
        value=[True] if val else [],
        labelStyle={"color": "#ddd"},
    )


def _version_table_data(versions: list[dict]) -> list[dict]:
    """Shape settings_versions rows for the bottom-of-tab history
    table."""
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


def layout(store: Store):
    """Build the Settings tab."""
    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    ee = cfg.get("epoch_extraction", {})   # Step 1: extraction window
    fa = cfg.get("feature_analysis", {})   # Step 2: analysis sub-window
    cr = cfg.get("criticality", {})
    sz = cfg.get("seizure", {})
    ar = cfg.get("artifact", {})
    qc = cfg.get("qc_thresholds", {})
    ft = cfg.get("features", {})

    versions = store.get_all_settings_versions()
    version_data = _version_table_data(versions)

    return html.Div([
        # --- Step 1: Epoch Extraction ---
        html.Div([
            html.H4("Step 1: Epoch Extraction",
                    style={"color": "#636EFA", "marginTop": "0"}),
            html.P("Cuts a window around each detected stimulus to "
                   "produce evoked.mat files. These intermediary "
                   "files can also be analyzed manually.",
                   style={"color": "#888", "fontSize": "12px",
                          "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Pre-stimulus (ms, negative = before)",
                                     style=LABEL_STYLE),
                          _input("evoked-pre-stim",
                                  ee.get("pre_stimulus_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Post-stimulus (ms)",
                                     style=LABEL_STYLE),
                          _input("evoked-post-stim",
                                  ee.get("post_stimulus_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stimulus threshold (std)",
                                     style=LABEL_STYLE),
                          _input("evoked-stim-thresh",
                                  ee.get("stimulus_threshold_std", 3.0),
                                  step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min stim distance (sec)",
                                     style=LABEL_STYLE),
                          _input("evoked-min-stim-dist",
                                  ee.get("min_stimulus_distance_sec", 0.1),
                                  step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Baseline correction",
                                     style=LABEL_STYLE),
                          _check("evoked-baseline-correction",
                                  ee.get("baseline_correction", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline start (ms)",
                                     style=LABEL_STYLE),
                          _input("evoked-baseline-start",
                                  ee.get("baseline_start_ms", -60))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline end (ms)",
                                     style=LABEL_STYLE),
                          _input("evoked-baseline-end",
                                  ee.get("baseline_end_ms", -10))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Notch 60 Hz", style=LABEL_STYLE),
                          _check("evoked-notch60",
                                  ee.get("notch_60hz", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch 50 Hz", style=LABEL_STYLE),
                          _check("evoked-notch50",
                                  ee.get("notch_50hz", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass enabled", style=LABEL_STYLE),
                          _check("evoked-hp-enabled",
                                  ee.get("highpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass cutoff (Hz)",
                                     style=LABEL_STYLE),
                          _input("evoked-hp-cutoff",
                                  ee.get("highpass_cutoff_hz", 1.0),
                                  step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass enabled", style=LABEL_STYLE),
                          _check("evoked-lp-enabled",
                                  ee.get("lowpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass cutoff (Hz)",
                                     style=LABEL_STYLE),
                          _input("evoked-lp-cutoff",
                                  ee.get("lowpass_cutoff_hz", 1000.0))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE, "borderLeft": "3px solid #636EFA"}),

        # --- Step 2: Feature Analysis (Configure & Analyze) ---
        html.Div([
            html.H4("Step 2: Feature Analysis (Configure & Analyze)",
                    style={"color": "#00CC96", "marginTop": "0"}),
            html.P("Mirrors the Chronic Evoked Features dialog. "
                   "Configures filtering, artifact exclusion, and "
                   "feature windows applied to extracted epochs.",
                   style={"color": "#888", "fontSize": "12px",
                          "marginBottom": "12px"}),

            html.H5("Windows",
                    style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Window start (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-win-start",
                                  fa.get("window_start_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window end (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-win-end",
                                  fa.get("window_end_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact start (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-stim-art-start",
                                  fa.get("stim_artifact_start_ms", -5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact end (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-stim-art-end",
                                  fa.get("stim_artifact_end_ms", 15))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),

            html.H5("Filtering",
                    style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Bandpass filter",
                                     style=LABEL_STYLE),
                          _check("feat-bp-enabled",
                                  fa.get("bandpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass (Hz)",
                                     style=LABEL_STYLE),
                          _input("feat-bp-hp",
                                  fa.get("bandpass_highpass_hz", 1.0),
                                  step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass (Hz)", style=LABEL_STYLE),
                          _input("feat-bp-lp",
                                  fa.get("bandpass_lowpass_hz", 100.0),
                                  step=10)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch filter",
                                     style=LABEL_STYLE),
                          _check("feat-notch-enabled",
                                  fa.get("notch_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch freq (Hz)",
                                     style=LABEL_STYLE),
                          _input("feat-notch-freq",
                                  fa.get("notch_frequency_hz", 60))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),

            html.Div([
                html.Div([html.Label("Moving average smoothing",
                                     style=LABEL_STYLE),
                          _check("feat-smooth-enabled",
                                  fa.get("smoothing_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Smoothing window (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-smooth-win",
                                  fa.get("smoothing_window_ms", 5),
                                  step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline correction",
                                     style=LABEL_STYLE),
                          _check("feat-baseline",
                                  fa.get("baseline_correction", True))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "16px"}),

            html.H5("Artifact Exclusion",
                    style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Enable artifact exclusion",
                                     style=LABEL_STYLE),
                          _check("feat-art-enabled",
                                  fa.get("artifact_exclusion_enabled",
                                           False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(
                              id="feat-art-method",
                              options=[{"label": m, "value": m}
                                       for m in ["fixed", "mad",
                                                  "template", "rawamp",
                                                  "noise"]],
                              value=fa.get("artifact_method", "fixed"),
                              style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Threshold / k-value",
                                     style=LABEL_STYLE),
                          _input("feat-art-thresh",
                                  fa.get("artifact_threshold", 500.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)",
                                     style=LABEL_STYLE),
                          _input("feat-art-merge",
                                  fa.get("artifact_merge_gap_sec", 2.0),
                                  step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),

            html.Div([
                html.Div([html.Label("Template source",
                                     style=LABEL_STYLE),
                          dcc.Dropdown(
                              id="feat-tmpl-source",
                              options=[
                                  {"label": "Early 50ms",
                                   "value": "early50ms"},
                                  {"label": "Grand Mean",
                                   "value": "grandMean"}],
                              value=fa.get("template_source",
                                            "early50ms"),
                              style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Upper band (r)",
                                     style=LABEL_STYLE),
                          _input("feat-tmpl-upper",
                                  fa.get("template_upper_r", 0.7),
                                  step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lower band (r)",
                                     style=LABEL_STYLE),
                          _input("feat-tmpl-lower",
                                  fa.get("template_lower_r", 0.3),
                                  step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Raw amp multiplier (k)",
                                     style=LABEL_STYLE),
                          _input("feat-rawamp-k",
                                  fa.get("rawamp_multiplier", 4.0),
                                  step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),

            html.Div([
                html.Div([html.Label("Ictal spike rescue",
                                     style=LABEL_STYLE),
                          _check("feat-ictal-rescue",
                                  fa.get("ictal_rescue_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Rescue window (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-ictal-win",
                                  fa.get("ictal_rescue_window_ms", 500))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "16px"}),

            html.H5("Feature Sub-Windows",
                    style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Analysis start (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-analysis-start",
                                  fa.get("analysis_start_ms", 5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Analysis end (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-analysis-end",
                                  fa.get("analysis_end_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area start (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-early-start",
                                  fa.get("early_area_start_ms", 0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area end (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-early-end",
                                  fa.get("early_area_end_ms", 50))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Late area start (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-late-start",
                                  fa.get("late_area_start_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Late area end (ms)",
                                     style=LABEL_STYLE),
                          _input("feat-late-end",
                                  fa.get("late_area_end_ms", 200))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE,
                  "borderLeft": "3px solid #00CC96"}),

        # --- Criticality ---
        html.Div([
            html.H4("Criticality",
                    style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("AR order", style=LABEL_STYLE),
                          _input("crit-ar-order",
                                  cr.get("ar_order", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window (sec)",
                                     style=LABEL_STYLE),
                          _input("crit-window-sec",
                                  cr.get("window_sec", 2.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Overlap %", style=LABEL_STYLE),
                          dcc.Slider(
                              id="crit-overlap", min=0, max=90, step=10,
                              value=cr.get("overlap_pct", 50),
                              marks={i: str(i)
                                       for i in range(0, 91, 10)},
                              tooltip={"placement": "bottom"})],
                         style={**FIELD_STYLE, "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Fit method", style=LABEL_STYLE),
                          dcc.Dropdown(
                              id="crit-fit-method",
                              options=[{"label": m, "value": m}
                                       for m in ["YuleWalker", "Burg",
                                                  "Covariance"]],
                              value=cr.get("fit_method", "YuleWalker"),
                              style={"backgroundColor": "#111",
                                     "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Target SR", style=LABEL_STYLE),
                          _input("crit-target-sr",
                                  cr.get("target_sampling_rate", 1000))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Type B", style=LABEL_STYLE),
                          _input("crit-type-b",
                                  cr.get("criticality_type_b", 2),
                                  step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Error bars", style=LABEL_STYLE),
                          _check("crit-error-bars",
                                  cr.get("calculate_error_bars", False))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Seizure ---
        html.Div([
            html.H4("Seizure Detection",
                    style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Spike threshold (uV)",
                                     style=LABEL_STYLE),
                          _input("seiz-spike-thresh",
                                  sz.get("spike_threshold_uv", 10))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike min width",
                                     style=LABEL_STYLE),
                          _input("seiz-spike-min-w",
                                  sz.get("spike_min_width", 5),
                                  step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike max width",
                                     style=LABEL_STYLE),
                          _input("seiz-spike-max-w",
                                  sz.get("spike_max_width", 50),
                                  step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min seizure dur (sec)",
                                     style=LABEL_STYLE),
                          _input("seiz-min-dur",
                                  sz.get("min_seizure_duration_sec", 5),
                                  step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Event glue (sec)",
                                     style=LABEL_STYLE),
                          _input("seiz-glue-sec",
                                  sz.get("event_glue_sec", 2),
                                  step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min spikes/sec",
                                     style=LABEL_STYLE),
                          _input("seiz-min-spikes",
                                  sz.get("min_spikes_per_sec", 2),
                                  step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Outlier factor",
                                     style=LABEL_STYLE),
                          _input("seiz-outlier",
                                  sz.get("outlier_factor", 3),
                                  step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Artifact ---
        html.Div([
            html.H4("Artifact Rejection",
                    style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(
                              id="art-method",
                              options=[{"label": m, "value": m}
                                       for m in ["fixed", "mad"]],
                              value=ar.get("method", "fixed"),
                              style={"backgroundColor": "#111",
                                     "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Fixed threshold",
                                     style=LABEL_STYLE),
                          _input("art-fixed-thresh",
                                  ar.get("fixed_threshold", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("MAD k", style=LABEL_STYLE),
                          _input("art-mad-k", ar.get("mad_k", 4.0),
                                  step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)",
                                     style=LABEL_STYLE),
                          _input("art-merge-gap",
                                  ar.get("merge_gap_sec", 2.0),
                                  step=0.1)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- QC Thresholds ---
        html.Div([
            html.H4("QC Thresholds",
                    style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Artifact % warning",
                                     style=LABEL_STYLE),
                          _input("qc-artifact-warn",
                                  qc.get("artifact_pct_warning", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Flatline std", style=LABEL_STYLE),
                          _input("qc-flatline-std",
                                  qc.get("flatline_std", 1e-6),
                                  step=1e-7)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Clipping voltage",
                                     style=LABEL_STYLE),
                          _input("qc-clipping-v",
                                  qc.get("clipping_voltage", 10.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Line noise ratio warning",
                                     style=LABEL_STYLE),
                          _input("qc-linenoise-warn",
                                  qc.get("line_noise_ratio_warning",
                                          0.2),
                                  step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px",
                      "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Features ---
        html.Div([
            html.H4("Features",
                    style={"color": "white", "marginTop": "0"}),
            dcc.Checklist(
                id="features-checklist",
                options=[{"label": f"  {f}", "value": f}
                         for f in _ALL_FEATURES],
                value=ft.get("enabled", _ALL_FEATURES),
                style={"columns": "3", "columnGap": "20px"},
                inputStyle={"marginRight": "6px"},
                labelStyle={"color": "#ddd"},
            ),
        ], style=SECTION_STYLE),

        # --- Buttons ---
        html.Div([
            html.Button("Save Settings", id="btn-save-settings",
                        n_clicks=0,
                        style={"backgroundColor": "#636EFA",
                               "color": "white", "border": "none",
                               "padding": "10px 24px",
                               "borderRadius": "6px",
                               "cursor": "pointer",
                               "fontSize": "14px",
                               "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            dcc.Input(id="version-label-input",
                      placeholder="Version label (optional)",
                      style={**INPUT_STYLE, "width": "250px"}),
            html.Button("Save as New Version",
                        id="btn-save-version", n_clicks=0,
                        style={"backgroundColor": "#00CC96",
                               "color": "white", "border": "none",
                               "padding": "10px 24px",
                               "borderRadius": "6px",
                               "cursor": "pointer",
                               "fontSize": "14px",
                               "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            html.Button("Reprocess Session", id="btn-reprocess",
                        n_clicks=0,
                        style={"backgroundColor": "#FFA15A",
                               "color": "white", "border": "none",
                               "padding": "10px 24px",
                               "borderRadius": "6px",
                               "cursor": "pointer",
                               "fontSize": "14px",
                               "fontWeight": "bold"}),
        ], style={"display": "flex", "gap": "10px",
                  "alignItems": "center", "flexWrap": "wrap",
                  "marginBottom": "12px"}),

        html.Div(id="settings-save-status"),
        html.Div(id="settings-version-status"),
        html.Div(id="reprocess-status"),

        # --- Version history ---
        html.Div([
            html.H4("Version History",
                    style={"color": "white", "marginTop": "24px"}),
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
                     "backgroundColor": "#112211",
                     "color": "#00CC96"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ], style=SECTION_STYLE),
    ])


# --------------------------------------------------------------------- #
#  Callbacks
# --------------------------------------------------------------------- #

# Every input id read by save_settings AND save_version. Kept as a
# module-level list so the two callbacks stay in lockstep; if you add
# a new field, add it here and to the unpack tuple in each callback.
_SHARED_STATE_IDS = [
    "evoked-pre-stim", "evoked-post-stim", "evoked-baseline-correction",
    "evoked-baseline-start", "evoked-baseline-end",
    "evoked-stim-thresh", "evoked-min-stim-dist",
    "evoked-notch60", "evoked-notch50",
    "evoked-hp-enabled", "evoked-hp-cutoff",
    "evoked-lp-enabled", "evoked-lp-cutoff",
    "crit-ar-order", "crit-window-sec", "crit-overlap",
    "crit-fit-method", "crit-target-sr", "crit-type-b",
    "crit-error-bars",
    "seiz-spike-thresh", "seiz-spike-min-w", "seiz-spike-max-w",
    "seiz-min-dur", "seiz-glue-sec", "seiz-min-spikes",
    "seiz-outlier",
    "art-method", "art-fixed-thresh", "art-mad-k", "art-merge-gap",
    "qc-artifact-warn", "qc-flatline-std", "qc-clipping-v",
    "qc-linenoise-warn",
    "features-checklist",
]

# Step 2 (Feature Analysis) ids are only read by save_settings -- the
# legacy save_version path predates Step 2 and the schema versions it
# writes deliberately use the smaller field set.
_STEP2_STATE_IDS = [
    "feat-win-start", "feat-win-end",
    "feat-stim-art-start", "feat-stim-art-end",
    "feat-bp-enabled", "feat-bp-hp", "feat-bp-lp",
    "feat-notch-enabled", "feat-notch-freq",
    "feat-smooth-enabled", "feat-smooth-win", "feat-baseline",
    "feat-art-enabled", "feat-art-method", "feat-art-thresh",
    "feat-art-merge",
    "feat-tmpl-source", "feat-tmpl-upper", "feat-tmpl-lower",
    "feat-rawamp-k",
    "feat-ictal-rescue", "feat-ictal-win",
    "feat-analysis-start", "feat-analysis-end",
    "feat-early-start", "feat-early-end",
    "feat-late-start", "feat-late-end",
]


def register_callbacks(app, store: Store, config: dict) -> None:
    """Wire save_settings (writes config.yaml), save_version (writes
    settings_versions row), reprocess_session (re-queues done files)."""

    @app.callback(
        Output("settings-save-status", "children"),
        Input("btn-save-settings", "n_clicks"),
        [State(i, "value") for i in
         _SHARED_STATE_IDS + _STEP2_STATE_IDS],
        prevent_initial_call=True,
    )
    def save_settings(n_clicks, *values):
        if not n_clicks:
            return no_update
        try:
            cfg = load_config()
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr,
             type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec,
             min_spikes, outlier,
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
             f_early_start, f_early_end,
             f_late_start, f_late_end,
             ) = values

            # Step 1: Epoch Extraction. Pattern: cast + assign only
            # if the user supplied a value, so a deleted field doesn't
            # silently zero the persisted setting.
            ee = cfg.setdefault("epoch_extraction", {})
            if pre_stim is not None:
                ee["pre_stimulus_ms"] = float(pre_stim)
            if post_stim is not None:
                ee["post_stimulus_ms"] = float(post_stim)
            ee["baseline_correction"] = bool(baseline_corr)
            ee["baseline_start_ms"] = float(bl_start or -60)
            ee["baseline_end_ms"] = float(bl_end or -10)
            if stim_thresh is not None:
                ee["stimulus_threshold_std"] = float(stim_thresh)
            if min_stim_dist is not None:
                ee["min_stimulus_distance_sec"] = float(min_stim_dist)
            ee["notch_60hz"] = bool(notch60)
            ee["notch_50hz"] = bool(notch50)
            ee["highpass_enabled"] = bool(hp_en)
            if hp_cut is not None:
                ee["highpass_cutoff_hz"] = float(hp_cut)
            ee["lowpass_enabled"] = bool(lp_en)
            if lp_cut is not None:
                ee["lowpass_cutoff_hz"] = float(lp_cut)
            # Legacy "evoked" block predates "epoch_extraction" and
            # had the same shape; drop it once we've persisted to the
            # newer key to avoid two-key drift on re-read.
            cfg.pop("evoked", None)

            # Step 2: Feature Analysis
            fa = cfg.setdefault("feature_analysis", {})
            if f_win_start is not None:
                fa["window_start_ms"] = float(f_win_start)
            if f_win_end is not None:
                fa["window_end_ms"] = float(f_win_end)
            if f_stim_start is not None:
                fa["stim_artifact_start_ms"] = float(f_stim_start)
            if f_stim_end is not None:
                fa["stim_artifact_end_ms"] = float(f_stim_end)
            fa["bandpass_enabled"] = bool(f_bp_en)
            if f_bp_hp is not None:
                fa["bandpass_highpass_hz"] = float(f_bp_hp)
            if f_bp_lp is not None:
                fa["bandpass_lowpass_hz"] = float(f_bp_lp)
            fa["notch_enabled"] = bool(f_notch_en)
            if f_notch_freq is not None:
                fa["notch_frequency_hz"] = float(f_notch_freq)
            fa["smoothing_enabled"] = bool(f_smooth_en)
            if f_smooth_win is not None:
                fa["smoothing_window_ms"] = float(f_smooth_win)
            fa["baseline_correction"] = bool(f_baseline)
            fa["artifact_exclusion_enabled"] = bool(f_art_en)
            if f_art_method:
                fa["artifact_method"] = f_art_method
            if f_art_thresh is not None:
                fa["artifact_threshold"] = float(f_art_thresh)
            if f_art_merge is not None:
                fa["artifact_merge_gap_sec"] = float(f_art_merge)
            if f_tmpl_src:
                fa["template_source"] = f_tmpl_src
            if f_tmpl_upper is not None:
                fa["template_upper_r"] = float(f_tmpl_upper)
            if f_tmpl_lower is not None:
                fa["template_lower_r"] = float(f_tmpl_lower)
            if f_rawamp_k is not None:
                fa["rawamp_multiplier"] = float(f_rawamp_k)
            fa["ictal_rescue_enabled"] = bool(f_ictal_en)
            if f_ictal_win is not None:
                fa["ictal_rescue_window_ms"] = float(f_ictal_win)
            if f_analysis_start is not None:
                fa["analysis_start_ms"] = float(f_analysis_start)
            if f_analysis_end is not None:
                fa["analysis_end_ms"] = float(f_analysis_end)
            if f_early_start is not None:
                fa["early_area_start_ms"] = float(f_early_start)
            if f_early_end is not None:
                fa["early_area_end_ms"] = float(f_early_end)
            if f_late_start is not None:
                fa["late_area_start_ms"] = float(f_late_start)
            if f_late_end is not None:
                fa["late_area_end_ms"] = float(f_late_end)

            # The criticality / seizure / artifact / qc_thresholds
            # blocks are guaranteed present in config.yaml (no
            # setdefault); ``or`` keeps the existing value when the
            # form left a field empty.
            cfg["criticality"]["ar_order"] = (
                int(ar_order) if ar_order is not None
                else cfg["criticality"]["ar_order"])
            cfg["criticality"]["window_sec"] = (
                float(win_sec) if win_sec is not None
                else cfg["criticality"]["window_sec"])
            cfg["criticality"]["overlap_pct"] = (
                int(overlap) if overlap is not None
                else cfg["criticality"]["overlap_pct"])
            cfg["criticality"]["fit_method"] = (
                fit_method or cfg["criticality"]["fit_method"])
            cfg["criticality"]["target_sampling_rate"] = (
                int(target_sr) if target_sr is not None
                else cfg["criticality"]["target_sampling_rate"])
            cfg["criticality"]["criticality_type_b"] = (
                int(type_b) if type_b is not None
                else cfg["criticality"]["criticality_type_b"])
            cfg["criticality"]["calculate_error_bars"] = bool(err_bars)

            cfg["seizure"]["spike_threshold_uv"] = (
                float(sp_thresh) if sp_thresh is not None
                else cfg["seizure"]["spike_threshold_uv"])
            cfg["seizure"]["spike_min_width"] = (
                int(sp_min_w) if sp_min_w is not None
                else cfg["seizure"]["spike_min_width"])
            cfg["seizure"]["spike_max_width"] = (
                int(sp_max_w) if sp_max_w is not None
                else cfg["seizure"]["spike_max_width"])
            cfg["seizure"]["min_seizure_duration_sec"] = (
                float(min_dur) if min_dur is not None
                else cfg["seizure"]["min_seizure_duration_sec"])
            cfg["seizure"]["event_glue_sec"] = (
                float(glue_sec) if glue_sec is not None
                else cfg["seizure"]["event_glue_sec"])
            cfg["seizure"]["min_spikes_per_sec"] = (
                float(min_spikes) if min_spikes is not None
                else cfg["seizure"]["min_spikes_per_sec"])
            cfg["seizure"]["outlier_factor"] = (
                float(outlier) if outlier is not None
                else cfg["seizure"]["outlier_factor"])

            cfg["artifact"]["method"] = (
                art_method or cfg["artifact"]["method"])
            cfg["artifact"]["fixed_threshold"] = (
                float(art_fixed) if art_fixed is not None
                else cfg["artifact"]["fixed_threshold"])
            cfg["artifact"]["mad_k"] = (
                float(art_mad) if art_mad is not None
                else cfg["artifact"]["mad_k"])
            cfg["artifact"]["merge_gap_sec"] = (
                float(art_merge) if art_merge is not None
                else cfg["artifact"]["merge_gap_sec"])

            cfg["qc_thresholds"]["artifact_pct_warning"] = (
                float(qc_art_warn) if qc_art_warn is not None
                else cfg["qc_thresholds"]["artifact_pct_warning"])
            cfg["qc_thresholds"]["flatline_std"] = (
                float(qc_flat) if qc_flat is not None
                else cfg["qc_thresholds"]["flatline_std"])
            cfg["qc_thresholds"]["clipping_voltage"] = (
                float(qc_clip) if qc_clip is not None
                else cfg["qc_thresholds"]["clipping_voltage"])
            cfg["qc_thresholds"]["line_noise_ratio_warning"] = (
                float(qc_ln) if qc_ln is not None
                else cfg["qc_thresholds"]["line_noise_ratio_warning"])

            cfg["features"]["enabled"] = features_enabled or []

            save_config(cfg)
            return html.Div("Settings saved to config.yaml",
                            style={"color": "#00CC96",
                                   "marginTop": "10px"})
        except Exception as e:
            logger.error("Failed to save settings: %s",
                          e, exc_info=True)
            return html.Div(f"Error saving: {e}",
                            style={"color": "#EF553B",
                                   "marginTop": "10px"})

    @app.callback(
        [Output("settings-version-status", "children"),
         Output("version-history-table", "data")],
        Input("btn-save-version", "n_clicks"),
        [State("version-label-input", "value")] +
        [State(i, "value") for i in _SHARED_STATE_IDS],
        prevent_initial_call=True,
    )
    def save_version(n_clicks, label, *values):
        if not n_clicks:
            return no_update, no_update
        try:
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr,
             type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec,
             min_spikes, outlier,
             art_method, art_fixed, art_mad, art_merge,
             qc_art_warn, qc_flat, qc_clip, qc_ln,
             features_enabled) = values

            # Build a clean settings dict from form state. Defaults
            # here are intentionally permissive ("or X") so a blank
            # field still produces a valid version row.
            config_dict = {
                "epoch_extraction": {
                    "pre_stimulus_ms": float(pre_stim or -100),
                    "post_stimulus_ms": float(post_stim or 500),
                    "baseline_correction": bool(baseline_corr),
                    "baseline_start_ms": float(bl_start or -60),
                    "baseline_end_ms": float(bl_end or -10),
                    "stimulus_threshold_std": float(stim_thresh or 3.0),
                    "min_stimulus_distance_sec":
                        float(min_stim_dist or 0.1),
                    "notch_60hz": bool(notch60),
                    "notch_50hz": bool(notch50),
                    "highpass_enabled": bool(hp_en),
                    "highpass_cutoff_hz": float(hp_cut or 1.0),
                    "lowpass_enabled": bool(lp_en),
                    "lowpass_cutoff_hz": float(lp_cut or 1000.0),
                },
                # Step 2 isn't editable via this version-save path
                # (predates Step 2). Fixed defaults so the version
                # dedupe hash isn't perturbed by random UI state.
                "feature_analysis": {
                    "analysis_start_ms": 5,
                    "analysis_end_ms": 50,
                    "early_area_start_ms": 0,
                    "early_area_end_ms": 50,
                    "late_area_start_ms": 50,
                    "late_area_end_ms": 200,
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

            vid = store.create_settings_version(
                config_dict, label=label or "")
            versions = store.get_all_settings_versions()
            return (
                html.Div(
                    f"Version #{vid} saved with label '{label or ''}'",
                    style={"color": "#00CC96", "marginTop": "10px"}),
                _version_table_data(versions),
            )
        except Exception as e:
            logger.error("Failed to save version: %s", e, exc_info=True)
            return (html.Div(f"Error: {e}",
                              style={"color": "#EF553B",
                                     "marginTop": "10px"}),
                    no_update)

    @app.callback(
        Output("reprocess-status", "children"),
        Input("btn-reprocess", "n_clicks"),
        prevent_initial_call=True,
    )
    def reprocess_session(n_clicks):
        if not n_clicks:
            return no_update
        try:
            with store.connection() as conn:
                conn.execute(
                    "UPDATE processed_files SET status = 'pending' "
                    "WHERE status = 'done'")
                cnt = conn.execute(
                    "SELECT changes()").fetchone()[0]
                conn.commit()
            return html.Div(f"Queued {cnt} files for reprocessing",
                            style={"color": "#FFA15A",
                                   "marginTop": "10px"})
        except Exception as e:
            return html.Div(f"Error: {e}",
                            style={"color": "#EF553B",
                                   "marginTop": "10px"})
