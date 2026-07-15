"""Peri-ictal Evoked Explorer tab.

Embeds an animal's per-stimulus evoked (or passive pre-stim) feature vectors and
colours them by CONTINUOUS time-to-next-seizure -- and, crucially, by the
confounds (stim fingerprint, hour-of-day) so an apparent proximity structure
that is really circadian or a stim-setting artefact is visible on the same
screen. Lasso/box-select a cluster to see, on demand, its time-to-onset and
hour-of-day distributions and which seizures/recordings it came from -- i.e.
whether it is a real signal or a confound.

This is an EXPLORATION surface, not a test. PCA is the default (an honest linear
projection whose axes carry variance); UMAP is opt-in and labelled a figure, not
evidence; and the effective sample size is stated as the number of SEIZURES, not
the ~10k stimuli. Design follows WCAG AA contrast, perceptually-uniform /
cyclic / colourblind-safe colour maps (see ``periictal.palette``), and
Shneiderman's "overview first, details on demand".

Builds run OFF the render thread (a daemon + a dcc.Interval poll, mirroring
chronic_evoked) so the page never freezes; recolouring reads the cached
embedding and is instant.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update

from src.dashboard.components import (DROPDOWN_STYLE, LABEL_STYLE, button, card,
                                      loading_icon, section_header)
from src.dashboard.data_helpers import empty_fig
from src.dashboard.design import (COLOR_ACCENT, COLOR_DIVIDER, COLOR_SUCCESS,
                                  COLOR_SURFACE_2, COLOR_TEXT_PRIMARY,
                                  COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
                                  COLOR_WARNING, FONT_SIZE_BODY,
                                  FONT_SIZE_CAPTION, FONT_SIZE_TITLE, RADIUS_SM,
                                  SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5)
from src.periictal import config as _cfg
from src.periictal import palette as _pal
from src.periictal import passive as _passive
from src.periictal import stim_map as _sm
from src.periictal.embed import confound_readout, embed
from src.periictal.persist import build_matrix_cached
from src.periictal import trajectory as _traj
from src.periictal.erpimage import (decimate_rows, gather_leadup_trials,
                                    sliding_trial_average)
from src.periictal.selection import summarize_selection
from src.preictal.isi import scored_seizures
from src.utils.evoked_features import FeatureConfig
from src.utils.evoked_output import list_animals

# Where the signature-keyed matrix cache lives.
_CACHE_DIR = None       # set in register_callbacks from the repo derivatives root

# --- background job registry (job_id -> state); results cached separately --- #
_JOBS: dict = {}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 6
_LOCK = threading.Lock()

# Readable dark-theme radio labels: `style` on a RadioItems does NOT reach the
# option <label>s (documented in video.py), so option text needs labelStyle.
_RADIO_LABEL = {"color": COLOR_TEXT_PRIMARY, "marginRight": SPACE_4,
                "fontSize": FONT_SIZE_BODY}
_RADIO_INPUT = {"marginRight": "4px"}


def _job_id(animal, protocol, variant, window_h, method, cap, wintok="") -> str:
    return f"{animal}|{protocol}|{variant}|{window_h}|{method}|{cap}|{wintok}"


def _tok(session_dir: str) -> str:
    return os.path.basename(session_dir or "").split("__", 1)[0]


# --------------------------------------------------------------------- #
#  Background build
# --------------------------------------------------------------------- #

def _set(job_id: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {}).update(kw)


def _kick(store, evoked_dir, job_id, animal, protocol, variant,
          window_h, method, cap, sidecar_variant, feature_cfg) -> None:
    """Start the build thread for *job_id* unless one is already running or the
    result is already cached."""
    with _LOCK:
        if job_id in _CACHE:
            return
        st = _JOBS.get(job_id)
        if st and st.get("status") == "running":
            return
        _JOBS[job_id] = {"status": "running", "progress": "starting…"}
    th = threading.Thread(
        target=_worker, name=f"periictal-{job_id}", daemon=True,
        args=(store, evoked_dir, job_id, animal, protocol, variant,
              window_h, method, cap, sidecar_variant, feature_cfg))
    th.start()


def _worker(store, evoked_dir, job_id, animal, protocol, variant,
            window_h, method, cap, sidecar_variant, feature_cfg) -> None:
    """Build the matrix + embedding for one job and cache the result. Never
    raises (records the error for the poll to surface)."""
    try:
        # A windowed variant (passive / custom-evoked) recomputes features from
        # the raw traces first; the shared default 'evoked' sidecar is pre-warm.
        if sidecar_variant not in (None, "evoked"):
            _set(job_id, progress="reading traces for the feature window…")
            _passive.warm_variant(
                animal, evoked_dir, sidecar_variant, feature_cfg,
                protocol=protocol or None,
                progress=lambda d, n, fp: _set(
                    job_id, progress=f"windowing traces… ({d}/{n})"))
        _set(job_id, progress="joining stimuli to seizures…")
        df = build_matrix_cached(
            store, animal, evoked_dir, _CACHE_DIR or _default_cache_dir(),
            protocol=protocol or None, window_sec=window_h * 3600.0,
            variant=variant, feature_cfg=feature_cfg,
            sidecar_variant=sidecar_variant)
        if df.empty:
            _finish(job_id, {"empty": True})
            return
        _set(job_id, progress=f"embedding {len(df):,} stimuli ({method.upper()})…")
        res = embed(df, method=method, cap=cap)
        sub = df.loc[res["rows"]].reset_index(drop=True)
        ro = confound_readout(res["emb"], sub)
        _finish(job_id, {"empty": False, "emb": res["emb"], "sub": sub,
                         "readout": ro, "meta": res["meta"],
                         "method": res["method"],
                         "n_seizures": int(df["seizure_idx"].nunique())})
    except Exception as e:                                     # noqa: BLE001
        _set(job_id, status="error", progress=f"error: {e}")


def _finish(job_id: str, result: dict) -> None:
    with _LOCK:
        _CACHE[job_id] = result
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
        _JOBS[job_id] = {"status": "done", "progress": "done"}


# --------------------------------------------------------------------- #
#  Layout
# --------------------------------------------------------------------- #

def _protocol_options(store, animal):
    if not animal:
        return [{"label": "All protocols", "value": ""}]
    toks = sorted(_sm.session_dir_by_token(store, animal).keys())
    return [{"label": "All protocols", "value": ""}] + \
        [{"label": t, "value": t} for t in toks]


def _default_protocol(store, animal) -> str:
    """Default to the animal's chronicStim group when it exists (the intended
    zero-click view), else all protocols. Must return a value that is an actual
    option -- the folder token is parameterized (chronicStim-5nC-2nC), so a bare
    'chronicStim' would be an invalid selection that Dash silently blanks."""
    if not animal:
        return ""
    toks = sorted(_sm.session_dir_by_token(store, animal).keys())
    chronic = [t for t in toks if "chronicstim" in t.lower()]
    return chronic[0] if chronic else ""


def layout(store):
    animals = list_animals(_evoked_dir(store))
    a0 = "BCH111" if "BCH111" in animals else (animals[0] if animals else None)
    return html.Div([
        _title_block(),
        card(section_header("Selection"), _controls(store, animals, a0),
             _window_panel()),
        card(html.Div(id="pex-preview", style={"marginBottom": SPACE_3}),
             dcc.Loading(
                 custom_spinner=loading_icon("Building…"),
                 overlay_style={"visibility": "visible", "opacity": 0.4},
                 children=dcc.Graph(
                     id="pex-graph", clear_on_unhover=True,
                     config={"displaylogo": False},
                     figure=empty_fig("Press ▶ Build to embed the lead-up stimuli",
                                      hint="Pick an animal and protocol above."))),
             _explainer(),
             html.Div(id="pex-status", style={"color": COLOR_TEXT_SECONDARY,
                                              "fontSize": FONT_SIZE_CAPTION,
                                              "minHeight": "14px",
                                              "marginTop": SPACE_2}),
             html.Div(id="pex-reading", style={"marginTop": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Lead-time trajectory"),
             html.Div([
                 html.Div("Median value per dyadic log-lead-time bin — one point "
                          "per bin, so every time-scale has equal weight and the "
                          "count imbalance (near-onset is sparse, far-onset "
                          "abundant) can't distort it. Faint lines = individual "
                          "seizures (a trend is credible only if consistent "
                          "across them); hollow markers = low-n bins.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 html.Div(_ctl("Trajectory y", dcc.Dropdown(
                     id="pex-traj-y", clearable=False,
                     style={**DROPDOWN_STYLE, "minWidth": "240px"}),
                     "Which value to trace against lead-time: any feature "
                     "(median + IQR per bin), a PC embedding coordinate, or "
                     "hour-of-day as a confound check."),
                          style={"marginTop": SPACE_3}),
             ]),
             dcc.Graph(id="pex-traj", config={"displaylogo": False},
                       figure=empty_fig("Build to see the lead-time trajectory")),
             style={"marginTop": SPACE_4}),
        card(section_header("Selected points — details on demand"),
             html.Div(id="pex-details", children=_details_view(None)),
             style={"marginTop": SPACE_4}),
        _erp_card(store, a0),
        dcc.Interval(id="pex-poll", interval=1200, disabled=True),
        dcc.Interval(id="pex-erp-poll", interval=1500, disabled=True),
        dcc.Store(id="pex-job"),
        dcc.Store(id="pex-erp-job"),
    ], style={"padding": SPACE_4})


def _erp_card(store, a0) -> object:
    """The ERP-image card: raw evoked waveforms of the trials leading into one
    selected seizure, stacked as a heatmap (columns = trials, rows = post-stim
    ms, colour = signed amplitude)."""
    controls = html.Div([
        _ctl("Seizure", dcc.Dropdown(
            id="pex-erp-seizure", clearable=False,
            style={**DROPDOWN_STYLE, "minWidth": "220px"}),
            "Which seizure's lead-up to show (the trials lead into its onset)."),
        _ctl("Lookback (min)", dcc.Input(
            id="pex-erp-lookback", type="number", value=60, min=1, max=360,
            step="any", debounce=True, style=_WIN_INP),
            "How far before onset to read trials. ~1 h ≈ one recording."),
        _ctl("Window from (ms)", dcc.Input(
            id="pex-erp-from", type="number", value=1, step="any", debounce=True,
            style=_WIN_INP), "Post-stim window start (excludes the t=0 artifact)."),
        _ctl("to (ms)", dcc.Input(
            id="pex-erp-to", type="number", value=200, step="any", debounce=True,
            style=_WIN_INP), "Post-stim window end."),
        _ctl("Avg trials (N)", dcc.Input(
            id="pex-erp-n", type="number", value=20, min=1, max=500, step=1,
            debounce=True, style=_WIN_INP),
            "Sliding trial-average window: each column = mean of N trials."),
        _ctl("Overlap (%)", dcc.Input(
            id="pex-erp-overlap", type="number", value=50, min=0, max=95, step=1,
            debounce=True, style=_WIN_INP),
            "Overlap between successive N-trial windows."),
        _ctl("Contrast (pct)", dcc.Input(
            id="pex-erp-contrast", type="number", value=99, min=80, max=100,
            step="any", debounce=True, style=_WIN_INP),
            "Colour range = ± this percentile of |amplitude| (clips outliers)."),
        html.Div(button("▶ Build ERP", "pex-erp-build", icon_name="play",
                        **{"title": "Read the raw traces for this seizure's "
                                    "lead-up and stack them. Runs in the "
                                    "background (~15 s per recording)."}),
                 style={"alignSelf": "flex-end"}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
              "alignItems": "flex-start"})
    return card(
        section_header("Evoked ERP-image — trials into onset"),
        html.Div("Each column is an evoked-response trial (smoothed by the "
                 "N-trial sliding average), ordered so they lead into the "
                 "seizure — onset at the RIGHT. Rows are post-stim samples; "
                 "colour is signed amplitude (diverging at 0), so the biphasic "
                 "waveform is visible. Reads raw traces on demand.",
                 style={"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
                        "maxWidth": "95ch", "marginBottom": SPACE_3}),
        controls,
        html.Div(id="pex-erp-warn", style={"marginTop": SPACE_2}),
        dcc.Loading(
            custom_spinner=loading_icon("Reading traces…"),
            overlay_style={"visibility": "visible", "opacity": 0.4},
            children=dcc.Graph(id="pex-erp", config={"displaylogo": False},
                               figure=empty_fig("Pick a seizure, then ▶ Build ERP"))),
        html.Div(id="pex-erp-status", style={"color": COLOR_TEXT_SECONDARY,
                                             "fontSize": FONT_SIZE_CAPTION,
                                             "minHeight": "14px", "marginTop": SPACE_2}),
        style={"marginTop": SPACE_4})


_EXPLAIN_P = {"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
              "margin": f"{SPACE_2} 0", "maxWidth": "95ch"}


def _explainer() -> html.Div:
    """Persistent 'what am I looking at' panel: what a point is, what sets its
    position (the features -- NOT the colour), and how to read UMAP honestly."""
    feats = ", ".join(_cfg.CHEAP_METRICS)
    always = html.Div([
        html.Span("How to read this   ",
                  style={"fontWeight": "600", "color": COLOR_TEXT_PRIMARY}),
        html.Span("Each point is one evoked response, positioned by its "
                  "standardized feature vector. Colour is an overlay — it never "
                  "moves points; recolour to test whether the layout tracks "
                  "seizure proximity or a confound.",
                  style={"color": COLOR_TEXT_SECONDARY}),
    ], style={"fontSize": FONT_SIZE_CAPTION, "marginBottom": SPACE_1})
    detail = html.Details([
        html.Summary("What determines the layout, and the features used",
                     style={"cursor": "pointer", "color": COLOR_TEXT_SECONDARY,
                            "fontSize": FONT_SIZE_CAPTION}),
        html.P([html.B("Position = feature similarity. "),
                "Each response's ", html.B("22 evoked (19 passive) features"),
                " — amplitudes, latencies, slope, line length, AUC, RMS, "
                "variance, autocorrelation, and spectral + wavelet-gamma powers "
                "— are z-scored and reduced to 2-D. Responses with similar "
                "features sit near each other; the blobs are regions of similar "
                "response morphology, not clusters 'of' time or seizures."],
               style=_EXPLAIN_P),
        html.P([html.B("Colour is not an input. "),
                "Time-to-onset, hour-of-day, seizure and stim fingerprint are "
                "painted on afterward, so a colour pattern means the feature "
                "layout happens to line up with that variable — recolour by "
                "hour-of-day or stim fingerprint to check it isn't a confound."],
               style=_EXPLAIN_P),
        html.P([html.B("UMAP caveat. "), "Blob shapes and the distances between "
                "separated blobs are not meaningful — treat UMAP as a picture. "
                "PCA axes, by contrast, carry real variance (shown on the axes)."],
               style=_EXPLAIN_P),
        html.Div("Features fed in: " + feats + "  (passive drops early_area, "
                 "late_area, early_late_ratio).",
                 style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
                        "marginTop": SPACE_2}),
    ])
    return html.Div([always, detail],
                    style={"marginTop": SPACE_2, "background": COLOR_SURFACE_2,
                           "padding": f"{SPACE_2} {SPACE_3}",
                           "borderRadius": RADIUS_SM})


def _title_block() -> html.Div:
    return html.Div([
        html.H3("Peri-ictal Evoked Explorer",
                style={"color": COLOR_TEXT_PRIMARY, "marginBottom": "2px",
                       "fontSize": FONT_SIZE_TITLE}),
        html.Div("Embed an animal's evoked-response features and see whether they "
                 "organise by time-to-next-seizure — or by a confound. "
                 "Colour by proximity, by hour-of-day, or by stim setting; "
                 "lasso a cluster to see what it actually is.",
                 style={"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_BODY,
                        "marginBottom": SPACE_3, "maxWidth": "70ch"}),
        _callout(
            "Exploratory tool. The effective sample size is the number of "
            "SEIZURES, not the thousands of stimuli. PCA is an honest linear "
            "projection; UMAP can invent clusters from noise — a figure, not a "
            "significance test.", COLOR_WARNING, "⚠"),
    ], style={"marginBottom": SPACE_4})


def _controls(store, animals, a0) -> html.Div:
    data_grp = html.Div([
        _ctl("Animal", dcc.Dropdown(
            id="pex-animal", options=[{"label": a, "value": a} for a in animals],
            value=a0, clearable=False, style=DROPDOWN_STYLE),
            "Which implanted animal to explore."),
        _ctl("Protocol / stim group", dcc.Dropdown(
            id="pex-protocol", options=_protocol_options(store, a0),
            value=_default_protocol(store, a0), clearable=False,
            style=DROPDOWN_STYLE),
            "Scope to one stim configuration. Pooling protocols mixes stim "
            "settings — a confound the colouring can reveal."),
        _ctl("Window", dcc.RadioItems(
            id="pex-variant", value="evoked", inline=True,
            labelStyle=_RADIO_LABEL, inputStyle=_RADIO_INPUT,
            options=[{"label": "evoked (post-stim)", "value": "evoked"},
                     {"label": "passive (pre-stim)", "value": "passive"}]),
            "Evoked = the response AFTER each pulse. Passive = the pre-stim LFP "
            "BEFORE it (less circadian-confounded)."),
        _ctl("Lead-up (hours)", dcc.Input(
            id="pex-window-h", type="number", value=6, min=0.1, max=72,
            step="any", debounce=True, style={**DROPDOWN_STYLE, "width": "80px"}),
            "How far before each seizure onset a stimulus still counts as "
            "pre-ictal."),
    ], style=_GRP)
    embed_grp = html.Div([
        _ctl("Embedding", dcc.RadioItems(
            id="pex-method", value="pca", inline=True,
            labelStyle=_RADIO_LABEL, inputStyle=_RADIO_INPUT,
            options=[{"label": "PCA", "value": "pca"},
                     {"label": "UMAP", "value": "umap"}]),
            "PCA: honest linear projection (axes carry variance). UMAP: "
            "non-linear, distances not meaningful — a figure only."),
        _ctl("Colour by", dcc.Dropdown(
            id="pex-colorby", clearable=False,
            style={**DROPDOWN_STYLE, "minWidth": "220px"}),
            "Recolour instantly (no rebuild). Try hour-of-day and stim "
            "fingerprint as confound checks."),
        html.Div(button("▶ Build", "pex-build", icon_name="play",
                        **{"title": "Embed the lead-up stimuli for the current "
                                    "selection. Runs in the background."}),
                 style={"alignSelf": "flex-end"}),
    ], style=_GRP)
    return html.Div([data_grp, _divider(), embed_grp],
                    style={"display": "flex", "flexWrap": "wrap",
                           "gap": SPACE_5, "alignItems": "flex-start"})


_GRP = {"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
        "alignItems": "flex-start"}
_WIN_INP = {**DROPDOWN_STYLE, "width": "90px"}


def _window_panel() -> html.Details:
    """Progressive-disclosure 'Feature window' panel: which slice of each evoked
    trace the features are computed over. Collapsed by default so it never
    clutters the common path."""
    return html.Details([
        html.Summary("Feature window (advanced)",
                     style={"cursor": "pointer", "color": COLOR_TEXT_SECONDARY,
                            "fontSize": FONT_SIZE_CAPTION, "fontWeight": "600",
                            "marginBottom": SPACE_2}),
        html.Div([
            _ctl("Evoked mode", dcc.RadioItems(
                id="pex-winmode", value="full", inline=True,
                labelStyle=_RADIO_LABEL, inputStyle=_RADIO_INPUT,
                options=[{"label": "full trace (fast)", "value": "full"},
                         {"label": "custom window", "value": "custom"}]),
                "Full = the toolkit's extracted trace (fast, but INCLUDES the "
                "stim artifact at t=0). Custom = recompute features over the ms "
                "window below, excluding the artifact. (Passive is always a "
                "window.)"),
            _ctl("From (ms)", dcc.Input(id="pex-win-from", type="number", value=1,
                                        step="any", debounce=True, style=_WIN_INP),
                 "Window start relative to the stimulus (t=0)."),
            _ctl("To (ms)", dcc.Input(id="pex-win-to", type="number", value=200,
                                      step="any", debounce=True, style=_WIN_INP),
                 "Window end relative to the stimulus."),
            _ctl("Artifact guard (ms)", dcc.Input(
                id="pex-win-guard", type="number", value=1, min=0, step="any",
                debounce=True, style=_WIN_INP),
                "Exclude ±this many ms around t=0 (the stim artifact); the bound "
                "nearest 0 is pushed out to here."),
        ], style=_GRP),
        html.Div("t = 0 is the stimulus. Evoked = a POST-stim window (e.g. 1→200 "
                 "ms); passive = PRE-stim (−200→−1). A custom window recomputes "
                 "features from the raw traces — a one-time background warm, then "
                 "cached.",
                 style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
                        "marginTop": SPACE_2, "maxWidth": "90ch"}),
    ], style={"marginTop": SPACE_3})


def _resolve_window(variant, winmode, frm, to, guard):
    """(sidecar_variant, feature_cfg) for the current selection. Evoked 'full'
    reads the fast shared default sidecar (cfg None); a custom evoked window uses
    the 'evokedw' variant; passive is always windowed. Raises ValueError on a bad
    window so the caller can surface it."""
    if variant != "passive" and winmode == "full":
        return "evoked", None
    try:
        frm, to, guard = float(frm), float(to), float(guard)
    except (TypeError, ValueError):
        raise ValueError("window bounds must be numbers")
    sv = "passive" if variant == "passive" else "evokedw"
    try:
        cfg = _passive.window_config(frm, to, artifact_half_ms=guard)
    except AssertionError as e:
        raise ValueError(f"invalid window ({e})")
    return sv, cfg


def _win_token(sidecar_variant, cfg) -> str:
    return f"{sidecar_variant}:{_passive.config_sig(cfg) if cfg is not None else ''}"


def _divider() -> html.Div:
    return html.Div(style={"width": "1px", "alignSelf": "stretch",
                           "background": COLOR_DIVIDER})


def _ctl(label, control, help_text=None):
    lab = html.Div(label, style=LABEL_STYLE, title=help_text or "")
    return html.Div([lab, control],
                    style={"display": "flex", "flexDirection": "column"})


def _callout(text, colour, glyph="") -> html.Div:
    return html.Div([
        html.Span(glyph + " ", style={"color": colour}) if glyph else "",
        html.Span(text, style={"color": COLOR_TEXT_PRIMARY,
                               "fontSize": FONT_SIZE_CAPTION}),
    ], style={"borderLeft": f"3px solid {colour}", "padding": f"6px {SPACE_3}",
              "background": COLOR_SURFACE_2, "borderRadius": RADIUS_SM,
              "maxWidth": "80ch"})


def _evoked_dir(store):
    from src.dashboard.data_helpers import load_config
    return load_config().get("chronic_evoked", {}).get("evoked_output_dir", "")


def _colorby_options(variant: str):
    base = [{"label": "time to next onset", "value": "time_to_onset_sec"},
            {"label": "hour of day (circadian check)", "value": "hour_of_day"},
            {"label": "stim fingerprint (confound check)", "value": "stim_key"},
            {"label": "seizure", "value": "seizure_idx"},
            {"label": "channel", "value": "channel"}]
    return base + [{"label": f"metric: {m}", "value": m}
                   for m in _cfg.metrics_for_variant(variant)]


def _pretty(color_by: str) -> str:
    return {"time_to_onset_sec": "time to onset", "hour_of_day": "hour of day",
            "stim_key": "stim fingerprint", "seizure_idx": "seizure",
            "channel": "channel"}.get(color_by, color_by)


# --------------------------------------------------------------------- #
#  Lead-time trajectory (count-agnostic: median per log-lead-time bin)
# --------------------------------------------------------------------- #

def _traj_y_options(variant: str):
    opts = [{"label": f"metric: {m}", "value": m}
            for m in _cfg.metrics_for_variant(variant)]
    return opts + [{"label": "PC1 (embedding position)", "value": "__pc1__"},
                   {"label": "PC2 (embedding position)", "value": "__pc2__"},
                   {"label": "hour of day (confound check)", "value": "hour_of_day"}]


def _traj_values(cached, y_key):
    """(values, y_label) for the chosen trajectory y -- a feature/hour column of
    the cached subsample, or a PC coordinate of the cached embedding."""
    sub, emb = cached["sub"], cached["emb"]
    if y_key == "__pc1__":
        return emb[:, 0], "PC1 (embedding)"
    if y_key == "__pc2__" and emb.shape[1] > 1:
        return emb[:, 1], "PC2 (embedding)"
    if y_key in sub.columns:
        return sub[y_key].to_numpy(dtype=float), _pretty(y_key)
    return None, y_key


def _trajectory_fig(cached, y_key) -> go.Figure:
    if not cached or cached.get("empty"):
        return empty_fig("Build to see the lead-time trajectory")
    values, ylabel = _traj_values(cached, y_key)
    if values is None:
        return empty_fig("Pick a trajectory metric")
    sub = cached["sub"]
    tto = sub["time_to_onset_sec"].to_numpy(dtype=float)
    hi = float(np.nanmax(tto)) if tto.size else 21600.0
    edges = _traj.default_edges(hi)
    tr = _traj.lead_time_trajectory(tto, values,
                                    sub["seizure_idx"].to_numpy(), edges)
    return _render_trajectory(tr, ylabel, hi)


def _render_trajectory(tr, ylabel, hi) -> go.Figure:
    c, med = tr["centers"], tr["median"]
    ok = np.isfinite(med)
    if not ok.any():
        return empty_fig("Not enough points to bin a trajectory")
    fig = go.Figure()
    for m in tr["per_seizure"].values():                 # faint per-seizure lines
        fig.add_trace(go.Scatter(
            x=c, y=m, mode="lines", connectgaps=False, hoverinfo="skip",
            showlegend=False, line=dict(width=1, color="rgba(160,160,176,0.22)")))
    xb = np.concatenate([c[ok], c[ok][::-1]])            # IQR band
    yb = np.concatenate([tr["p75"][ok], tr["p25"][ok][::-1]])
    fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself", mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False,
                             fillcolor="rgba(94,124,226,0.16)"))
    low = tr["low_n"][ok]
    fig.add_trace(go.Scatter(                            # aggregate median
        x=c[ok], y=med[ok], mode="lines+markers", showlegend=False,
        line=dict(width=2, color=COLOR_ACCENT),
        marker=dict(size=[6 if lo else 9 for lo in low], color=COLOR_ACCENT,
                    symbol=["circle-open" if lo else "circle" for lo in low],
                    line=dict(width=1.5, color=COLOR_ACCENT)),
        customdata=tr["n"][ok],
        hovertemplate="lead %{x:.3s}s · median %{y:.3g} · n=%{customdata}"
                      "<extra></extra>"))
    tv, tt = _pal.time_axis_ticks(hi)
    fig.update_layout(
        height=300, margin=dict(l=54, r=20, t=22, b=40),
        xaxis=dict(title="lead time before onset (log)", type="log",
                   tickvals=tv, ticktext=tt, autorange="reversed"),
        yaxis=dict(title=ylabel), uirevision="pex-traj")
    return fig


# --------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------- #

def _figure(emb, sub, color_by, method, meta) -> go.Figure:
    if emb is None or emb.shape[0] == 0:
        return empty_fig("No lead-up stimuli for this selection")
    n_unique = int(sub[color_by].nunique()) if color_by in sub else 0
    fig = (_categorical_fig(emb, sub, color_by)
           if _pal.is_categorical(color_by, n_unique)
           else _continuous_fig(emb, sub, color_by))
    return _finish_fig(fig, method, meta)


def _continuous_fig(emb, sub, color_by) -> go.Figure:
    spec = _pal.continuous_spec(color_by, sub[color_by].to_numpy(dtype=float))
    cbar = dict(title=dict(text=spec["label"], font=dict(size=9)),
                thickness=10, len=0.7, x=1.005)
    if spec["ticks"]:
        tv, tt = spec["ticks"]
        cbar.update(tickvals=tv, ticktext=tt)
    marker = dict(size=4, opacity=0.6, color=spec["vals"], colorscale=spec["scale"],
                  reversescale=spec["reverse"], showscale=True, colorbar=cbar)
    if spec["cmin"] is not None:
        marker["cmin"], marker["cmax"] = spec["cmin"], spec["cmax"]
    kw = dict(x=emb[:, 0], y=emb[:, 1], mode="markers", marker=marker,
              customdata=np.arange(emb.shape[0]))
    if color_by == "time_to_onset_sec":
        # colour is log10(seconds); hover shows a readable duration instead.
        kw["text"] = [_fmt_dur(s) for s in sub["time_to_onset_sec"].to_numpy()]
        kw["hovertemplate"] = "time to onset: %{text}<extra></extra>"
    else:
        kw["hovertemplate"] = f"{spec['label']}: %{{marker.color:.2f}}<extra></extra>"
    return go.Figure(go.Scattergl(**kw))


def _fmt_dur(s) -> str:
    """Human-readable duration for hover (the colour axis is log-seconds)."""
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _categorical_fig(emb, sub, color_by) -> go.Figure:
    raw = sub[color_by].astype(str).to_numpy()
    order, disp = _pal.fold_categories(raw)
    shown = np.array([disp[x] for x in raw])
    row_idx = np.arange(emb.shape[0])
    fig = go.Figure()
    for k, name in enumerate(order):
        m = shown == name
        fig.add_trace(go.Scattergl(
            x=emb[m, 0], y=emb[m, 1], mode="markers", name=name,
            marker=dict(size=4, opacity=0.65, color=_pal.hue_for(name, k)),
            customdata=row_idx[m],
            hovertemplate=f"{_pretty(color_by)}: {name}<extra></extra>"))
    fig.update_layout(legend=dict(title=_pretty(color_by), orientation="h",
                                  y=1.02, yanchor="bottom", font=dict(size=10),
                                  itemsizing="constant"))
    return fig


def _finish_fig(fig, method, meta) -> go.Figure:
    xt, yt = _axis_titles(method, meta)
    fig.update_layout(
        margin=dict(l=46, r=20, t=34, b=42), height=520, hovermode="closest",
        xaxis=dict(title=xt, showticklabels=False, zeroline=False),
        yaxis=dict(title=yt, showticklabels=False, zeroline=False),
        dragmode="lasso", uirevision="pex")
    if method == "umap":
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
            yanchor="bottom", showarrow=False,
            text="UMAP — cluster shapes and distances are not metric",
            font=dict(size=10, color=COLOR_WARNING))
    return fig


def _axis_titles(method, meta):
    ev = (meta or {}).get("explained_var")
    if method == "pca" and ev and len(ev) >= 2:
        return f"PC1 ({ev[0]:.0%} variance)", f"PC2 ({ev[1]:.0%} variance)"
    if method == "umap":
        return "UMAP-1 (relative)", "UMAP-2 (relative)"
    return "dim 1", "dim 2"


# --------------------------------------------------------------------- #
#  Guided preview + reading strip
# --------------------------------------------------------------------- #

def _preview_panel(store, animal, protocol, variant, window_h,
                   window_label="") -> html.Div:
    if not animal:
        return _callout("Pick an animal to begin.", COLOR_TEXT_SECONDARY)
    szs = [s for s in scored_seizures(store, animal)
           if not protocol or protocol in _tok(s.session_dir)]
    n = len(szs)
    proto = protocol or "all protocols"
    if n < 2:
        return _callout(
            f"{animal} has {n} scored seizure(s) in {proto} — need at least 2 to "
            f"define lead-up windows. Try another protocol or 'All protocols'.",
            COLOR_WARNING, "⚠")
    win = f" [{window_label}]" if window_label else ""
    return html.Div([
        html.Span("Ready — ", style={"color": COLOR_SUCCESS, "fontWeight": "600",
                                     "fontSize": FONT_SIZE_BODY}),
        html.Span(f"{animal} · {proto} · {variant} features{win} · "
                  f"{window_h:g} h lead-up",
                  style={"color": COLOR_TEXT_PRIMARY, "fontSize": FONT_SIZE_BODY}),
        html.Span(f"    Effective n = {n} seizures. Press ▶ Build to embed their "
                  f"lead-up stimuli.",
                  style={"color": COLOR_TEXT_SECONDARY,
                         "fontSize": FONT_SIZE_CAPTION}),
    ])


def _window_label(sidecar_variant, cfg) -> str:
    """Human label for the active feature window."""
    if cfg is None:
        return "full trace, incl. artifact"
    return f"{cfg.window_start_ms:g} to {cfg.window_end_ms:g} ms"


def _reading_strip(res) -> html.Div:
    ro = res.get("readout") or {}
    n = res.get("n_seizures", 0)
    meta = res.get("meta") or {}
    pts, tot = meta.get("n_points", 0), meta.get("n_total", 0)
    chips = [_confound_chip("Circadian (hour-of-day)", ro.get("time_of_day")),
             _confound_chip("Stim fingerprint", ro.get("stim_fingerprint"))]
    vals = [v for v in (ro.get("time_of_day"), ro.get("stim_fingerprint"))
            if v is not None]
    worst = max(vals) if vals else None
    if worst is not None and worst >= 0.5:
        msg, colour, glyph = (
            "The embedding largely tracks a confound flagged above, so any "
            "apparent time-to-onset structure is suspect — recolour by that "
            "confound to see it.", COLOR_WARNING, "⚠")
    elif worst is not None:
        msg, colour, glyph = (
            "Confound correlation is low — the layout isn't obviously driven by "
            "time-of-day or stim setting.", COLOR_SUCCESS, "✓")
    else:
        msg, colour, glyph = "", COLOR_TEXT_SECONDARY, ""
    prov = (f"n = {n} seizures · showing {pts:,} of {tot:,} stimuli"
            if tot and pts < tot else f"n = {n} seizures · {pts:,} stimuli")
    return html.Div([
        html.Div(prov, style={"color": COLOR_TEXT_TERTIARY,
                              "fontSize": FONT_SIZE_CAPTION, "marginBottom": SPACE_2}),
        html.Div([html.Span("Reading this plot   ",
                            style={"fontWeight": "600", "color": COLOR_TEXT_PRIMARY,
                                   "fontSize": FONT_SIZE_CAPTION}), *chips],
                 style={"marginBottom": SPACE_2, "display": "flex",
                        "flexWrap": "wrap", "alignItems": "center", "gap": SPACE_2}),
        _callout(msg, colour, glyph) if msg else html.Div(),
    ])


def _confound_chip(label, v) -> html.Span:
    if v is None:
        txt, border, glyph = "n/a", COLOR_DIVIDER, ""
    elif v >= 0.5:
        txt, border, glyph = f"|ρ| {v:.2f}", COLOR_WARNING, " ⚠"
    else:
        txt, border, glyph = f"|ρ| {v:.2f}", COLOR_SUCCESS, ""
    return html.Span([f"{label}: ", html.B(txt + glyph)],
                     style={"padding": f"2px {SPACE_3}", "marginRight": SPACE_2,
                            "borderRadius": RADIUS_SM, "fontSize": FONT_SIZE_CAPTION,
                            "color": COLOR_TEXT_PRIMARY,
                            "border": f"1px solid {border}"})


# --------------------------------------------------------------------- #
#  Details on demand (lasso/box selection)
# --------------------------------------------------------------------- #

_TWO_COL = {"display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
            "gap": SPACE_4, "marginTop": SPACE_3}


def _details_view(summary) -> html.Div:
    if not summary or summary.get("n", 0) == 0:
        return html.Div(
            "Lasso- or box-select points on the plot to inspect them — their "
            "time-to-onset, hour-of-day, and which seizures and recordings they "
            "came from (so you can tell a real cluster from a confound).",
            style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION})
    n, tot = summary["n"], summary["n_total"]
    head = f"{n:,} of {tot:,} points selected"
    lead = summary.get("lead_frac")
    if lead is not None:
        head += f"   ·   {lead:.0%} in one lead-time bin"
    return html.Div([
        html.Div(head, style={"color": COLOR_TEXT_PRIMARY, "fontWeight": "600",
                              "fontSize": FONT_SIZE_BODY, "marginBottom": SPACE_2}),
        html.Div([
            _mini_hist(summary["tto_hours"], "hours to onset",
                       "Time-to-onset of selection"),
            _mini_hist(summary["hour_of_day"], "hour of day",
                       "Hour-of-day of selection", xrange=[0, 24]),
        ], style=_TWO_COL),
        html.Div([
            _count_table("By seizure (onset time)", summary["by_seizure"]),
            _count_table("By recording", summary["by_recording"]),
        ], style=_TWO_COL),
    ])


def _mini_hist(values, xtitle, title, xrange=None) -> dcc.Graph:
    fig = go.Figure(go.Histogram(x=values, nbinsx=24,
                                 marker=dict(color=COLOR_ACCENT)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11), x=0.02),
        margin=dict(l=40, r=12, t=28, b=34), height=200, bargap=0.05,
        xaxis=dict(title=dict(text=xtitle, font=dict(size=10)), range=xrange),
        yaxis=dict(title=dict(text="stimuli", font=dict(size=10))))
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _count_table(title, rows) -> html.Div:
    header = html.Div(title, style={**LABEL_STYLE, "marginBottom": SPACE_2})
    if not rows:
        return html.Div([header, html.Div("—", style={"color": COLOR_TEXT_TERTIARY})])
    total = sum(c for _, c in rows) or 1
    body = html.Table([html.Tbody([
        html.Tr([
            html.Td(str(lab), style={"color": COLOR_TEXT_SECONDARY,
                                     "padding": "2px 8px 2px 0",
                                     "fontSize": FONT_SIZE_CAPTION}),
            html.Td(f"{c:,}", style={"color": COLOR_TEXT_PRIMARY,
                                     "textAlign": "right", "padding": "2px 8px",
                                     "fontVariantNumeric": "tabular-nums",
                                     "fontSize": FONT_SIZE_CAPTION}),
            html.Td(_bar(c / total), style={"width": "70px"}),
        ]) for lab, c in rows])], style={"borderCollapse": "collapse",
                                         "width": "100%"})
    return html.Div([header, body])


def _bar(frac) -> html.Div:
    return html.Div(style={"height": "8px", "borderRadius": "2px", "opacity": 0.85,
                           "width": f"{max(3.0, frac * 100):.0f}%",
                           "background": COLOR_ACCENT})


def _selected_indices(selected) -> list:
    """Row indices (customdata) of the lasso/box-selected points; defensive
    against the various shapes Plotly can hand back."""
    if not selected or "points" not in selected:
        return []
    out = []
    for p in selected["points"]:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is None:
            continue
        try:
            out.append(int(cd))
        except (TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------- #
#  Callbacks
# --------------------------------------------------------------------- #

def _default_cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "data", "derivatives", "periictal", "cache")


def _render_cached(cached, color_by, jid, traj_y):
    """(figure, reading, status, poll_disabled, job, trajectory) for a ready
    build."""
    if cached.get("empty"):
        msg = ("No lead-up stimuli — fewer than 2 seizures, none in this protocol, "
               "or no fresh sidecars for this animal.")
        return empty_fig(msg), "", msg, True, jid, empty_fig(msg)
    fig = _figure(cached["emb"], cached["sub"], color_by,
                  cached.get("method", "pca"), cached.get("meta"))
    traj = _trajectory_fig(cached, traj_y or "peak_to_trough")
    return fig, _reading_strip(cached), "✓ built", True, jid, traj


# --------------------------------------------------------------------- #
#  ERP-image: raw-trace gather job + heatmap
# --------------------------------------------------------------------- #

_ERP_JOBS: dict = {}
_ERP_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_ERP_CACHE_MAX = 4


def _scoped_seizures(store, animal, protocol):
    if not animal:
        return []
    szs = scored_seizures(store, animal)
    if protocol:
        szs = [s for s in szs if protocol in _tok(s.session_dir)]
    return szs


def _erp_seizure_options(store, animal, protocol):
    out = []
    for i, s in enumerate(_scoped_seizures(store, animal, protocol)):
        dt = datetime.fromtimestamp(s.onset_epoch)
        rac = s.racine if s.racine is not None else "?"
        out.append({"label": f"{dt:%Y-%m-%d %H:%M} · R{rac}", "value": i})
    return out


def _erp_key(animal, protocol, sz, lookback, frm, to) -> str:
    return f"{animal}|{protocol}|{sz}|{lookback}|{frm}|{to}"


def _erp_set(key, **kw):
    with _LOCK:
        _ERP_JOBS.setdefault(key, {}).update(kw)


def _erp_finish(key, result):
    with _LOCK:
        _ERP_CACHE[key] = result
        while len(_ERP_CACHE) > _ERP_CACHE_MAX:
            _ERP_CACHE.popitem(last=False)
        _ERP_JOBS[key] = {"status": "done", "progress": "done"}


def _erp_kick(store, evoked_dir, key, animal, protocol, sz, lookback, frm, to):
    with _LOCK:
        if key in _ERP_CACHE:
            return
        st = _ERP_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _ERP_JOBS[key] = {"status": "running", "progress": "starting…"}
    threading.Thread(
        target=_erp_worker, name=f"erp-{key}", daemon=True,
        args=(store, evoked_dir, key, animal, protocol, sz, lookback, frm, to)
    ).start()


def _erp_worker(store, evoked_dir, key, animal, protocol, sz, lookback, frm, to):
    """Gather the seizure's lead-up traces (heavy) and cache them. Never raises."""
    try:
        szs = _scoped_seizures(store, animal, protocol)
        if sz is None or sz >= len(szs):
            _erp_finish(key, {"empty": True})
            return
        s = szs[sz]
        rec_only = False
        try:
            rec_only = _sm.resolve_fingerprint(
                store, s.session_dir, animal).status == "record_only"
        except Exception:                                     # noqa: BLE001
            pass
        cfg = FeatureConfig(window_start_ms=float(frm), window_end_ms=float(to))
        res = gather_leadup_trials(
            evoked_dir, animal, s.onset_epoch, float(lookback) * 60.0, cfg=cfg,
            progress=lambda i, n, fp: _erp_set(
                key, progress=f"reading traces… ({i + 1}/{n})"))
        res["empty"] = int(res["trials"].shape[0]) == 0
        res["record_only"] = rec_only
        _erp_finish(key, res)
    except Exception as e:                                    # noqa: BLE001
        _erp_set(key, status="error", progress=f"error: {e}")


def _erp_figure(gathered, n, overlap, contrast) -> go.Figure:
    if not gathered or gathered.get("empty"):
        return empty_fig("No trials in this lead-up window")
    trials, row_ms, tto = gathered["trials"], gathered["row_ms"], gathered["tto"]
    step = max(1, round(float(n) * (1.0 - float(overlap) / 100.0)))
    z, col_tto = sliding_trial_average(trials, tto, int(n), step)
    z, rm = decimate_rows(z, row_ms, 300)
    absz = np.abs(z[np.isfinite(z)])
    zmax = float(np.percentile(absz, float(contrast))) if absz.size else 1.0
    zmax = zmax or 1.0
    ncol = z.shape[1]
    x = np.arange(ncol)
    ti = np.linspace(0, ncol - 1, min(7, ncol)).astype(int) if ncol else []
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=rm, colorscale="RdBu", reversescale=True, zmid=0,
        zmin=-zmax, zmax=zmax,
        colorbar=dict(title=dict(text="amplitude", font=dict(size=9)),
                      thickness=10, len=0.85, x=1.005),
        hovertemplate="%{y:.0f} ms · %{z:.3g}<extra></extra>"))
    fig.update_layout(
        height=420, margin=dict(l=56, r=20, t=22, b=44),
        xaxis=dict(title="trial windows  (far ← → onset)",
                   tickvals=[x[k] for k in ti],
                   ticktext=[_fmt_dur(col_tto[k]) for k in ti]),
        yaxis=dict(title="post-stim time (ms)"), uirevision="pex-erp")
    return fig


def _erp_render(gathered, n, overlap, contrast, key):
    """(figure, status, poll_disabled, job, warn) for a ready ERP gather."""
    if gathered.get("empty"):
        msg = "No trials in this seizure's lead-up (unreachable traces or none in window)."
        return empty_fig(msg), msg, True, key, ""
    warn = ""
    if gathered.get("record_only"):
        warn = _callout("This animal was record-only in this session (not "
                        "stimulated) — these traces are the passive LFP around "
                        "another animal's pulses, not true evoked responses.",
                        COLOR_WARNING, "⚠")
    nt = int(gathered["trials"].shape[0])
    return (_erp_figure(gathered, n, overlap, contrast),
            f"✓ built from {nt:,} trials", True, key, warn)


def register_callbacks(app, store, config):
    global _CACHE_DIR
    _CACHE_DIR = _default_cache_dir()
    evoked_dir = config.get("chronic_evoked", {}).get("evoked_output_dir", "")

    @app.callback(
        Output("pex-protocol", "options"),
        Output("pex-protocol", "value"),
        Output("pex-colorby", "options"),
        Output("pex-colorby", "value"),
        Output("pex-winmode", "value"),
        Output("pex-win-from", "value"),
        Output("pex-win-to", "value"),
        Output("pex-traj-y", "options"),
        Output("pex-traj-y", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
        State("pex-colorby", "value"),
        State("pex-traj-y", "value"),
    )
    def _on_animal(animal, variant, cur_color, cur_traj):
        opts = _protocol_options(store, animal)
        pval = _default_protocol(store, animal)
        copts = _colorby_options(variant or "evoked")
        cvals = {o["value"] for o in copts}
        cval = cur_color if cur_color in cvals else "time_to_onset_sec"
        topts = _traj_y_options(variant or "evoked")
        tvals = {o["value"] for o in topts}
        tval = cur_traj if cur_traj in tvals else "peak_to_trough"
        trig = callback_context.triggered_id
        # Reset the feature window to the variant's default when the variant
        # flips (passive needs pre-stim bounds, evoked post-stim).
        if variant == "passive":
            wm, w0, w1 = "custom", *_cfg.DEFAULT_PASSIVE_WINDOW_MS
        else:
            wm, w0, w1 = "full", *_cfg.DEFAULT_EVOKED_WINDOW_MS
        win_reset = trig == "pex-variant"
        return (opts, (pval if trig == "pex-animal" else no_update), copts, cval,
                wm if win_reset else no_update,
                w0 if win_reset else no_update,
                w1 if win_reset else no_update,
                topts, tval)

    @app.callback(
        Output("pex-preview", "children"),
        Input("pex-animal", "value"),
        Input("pex-protocol", "value"),
        Input("pex-variant", "value"),
        Input("pex-window-h", "value"),
        Input("pex-winmode", "value"),
        Input("pex-win-from", "value"),
        Input("pex-win-to", "value"),
        Input("pex-win-guard", "value"),
    )
    def _preview(animal, protocol, variant, window_h, winmode, wf, wt, wg):
        variant = variant or "evoked"
        try:
            sv, cfg = _resolve_window(variant, winmode or "full", wf, wt, wg)
        except ValueError as e:
            return _callout(f"Feature window: {e}.", COLOR_WARNING, "⚠")
        try:
            return _preview_panel(store, animal, protocol or "", variant,
                                  float(window_h or 6.0), _window_label(sv, cfg))
        except Exception:                                     # noqa: BLE001
            return no_update

    @app.callback(
        Output("pex-graph", "figure"),
        Output("pex-reading", "children"),
        Output("pex-status", "children"),
        Output("pex-poll", "disabled"),
        Output("pex-job", "data"),
        Output("pex-traj", "figure"),
        Input("pex-build", "n_clicks"),
        Input("pex-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-variant", "value"),
        State("pex-window-h", "value"),
        State("pex-method", "value"),
        State("pex-colorby", "value"),
        State("pex-winmode", "value"),
        State("pex-win-from", "value"),
        State("pex-win-to", "value"),
        State("pex-win-guard", "value"),
        State("pex-traj-y", "value"),
        prevent_initial_call=True,
    )
    def _build_or_poll(_n, _iv, animal, protocol, variant, window_h,
                       method, color_by, winmode, wf, wt, wg, traj_y):
        if not animal:
            return (no_update, no_update, "Pick an animal.", True, no_update,
                    no_update)
        try:
            sv, cfg = _resolve_window(variant, winmode or "full", wf, wt, wg)
        except ValueError as e:
            return (empty_fig("Invalid feature window", hint=str(e)),
                    "", f"⚠ Feature window: {e}", True, no_update, no_update)
        window_h = float(window_h or 6.0)
        cap = _cfg.INTERACTIVE_POINT_CAP
        jid = _job_id(animal, protocol or "", variant, window_h, method, cap,
                      _win_token(sv, cfg))
        with _LOCK:
            cached = _CACHE.get(jid)
            state = dict(_JOBS.get(jid) or {})
        if cached is not None:
            return _render_cached(cached, color_by, jid, traj_y)
        if state.get("status") == "error":
            return (empty_fig("Build failed", hint=state.get("progress", "")),
                    "", state.get("progress", "error"), True, no_update, no_update)
        _kick(store, evoked_dir, jid, animal, protocol or "", variant,
              window_h, method, cap, sv, cfg)
        prog = (_JOBS.get(jid) or {}).get("progress", "starting…")
        return no_update, no_update, f"⏳ {prog}", False, no_update, no_update

    @app.callback(
        Output("pex-graph", "figure", allow_duplicate=True),
        Input("pex-colorby", "value"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _recolor(color_by, jid):
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty"):
            return no_update
        return _figure(cached["emb"], cached["sub"], color_by,
                       cached.get("method", "pca"), cached.get("meta"))

    @app.callback(
        Output("pex-traj", "figure", allow_duplicate=True),
        Input("pex-traj-y", "value"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _retraj(traj_y, jid):
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty"):
            return no_update
        return _trajectory_fig(cached, traj_y or "peak_to_trough")

    @app.callback(
        Output("pex-details", "children"),
        Input("pex-graph", "selectedData"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _select(selected, jid):
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty"):
            return _details_view(None)
        idx = _selected_indices(selected)
        return _details_view(summarize_selection(cached["sub"], idx))

    # --- ERP-image callbacks --- #

    @app.callback(
        Output("pex-erp-seizure", "options"),
        Output("pex-erp-seizure", "value"),
        Input("pex-animal", "value"),
        Input("pex-protocol", "value"),
        State("pex-erp-seizure", "value"),
    )
    def _erp_seizures(animal, protocol, cur):
        opts = _erp_seizure_options(store, animal, protocol or "")
        vals = {o["value"] for o in opts}
        val = cur if cur in vals else (opts[-1]["value"] if opts else None)
        return opts, val

    @app.callback(
        Output("pex-erp", "figure"),
        Output("pex-erp-status", "children"),
        Output("pex-erp-poll", "disabled"),
        Output("pex-erp-job", "data"),
        Output("pex-erp-warn", "children"),
        Input("pex-erp-build", "n_clicks"),
        Input("pex-erp-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-erp-seizure", "value"),
        State("pex-erp-lookback", "value"),
        State("pex-erp-from", "value"),
        State("pex-erp-to", "value"),
        State("pex-erp-n", "value"),
        State("pex-erp-overlap", "value"),
        State("pex-erp-contrast", "value"),
        prevent_initial_call=True,
    )
    def _erp_build_or_poll(_n, _iv, animal, protocol, sz, lookback, frm, to,
                           navg, overlap, contrast):
        if not animal or sz is None:
            return no_update, "Pick a seizure.", True, no_update, no_update
        lookback = float(lookback or 60.0)
        key = _erp_key(animal, protocol or "", sz, lookback, frm, to)
        with _LOCK:
            gathered = _ERP_CACHE.get(key)
            state = dict(_ERP_JOBS.get(key) or {})
        if gathered is not None:
            return _erp_render(gathered, navg, overlap, contrast, key)
        if state.get("status") == "error":
            return (empty_fig("ERP build failed", hint=state.get("progress", "")),
                    state.get("progress", "error"), True, no_update, no_update)
        _erp_kick(store, evoked_dir, key, animal, protocol or "", sz, lookback,
                  frm, to)
        prog = (_ERP_JOBS.get(key) or {}).get("progress", "starting…")
        return no_update, f"⏳ {prog}", False, no_update, no_update

    @app.callback(
        Output("pex-erp", "figure", allow_duplicate=True),
        Input("pex-erp-n", "value"),
        Input("pex-erp-overlap", "value"),
        Input("pex-erp-contrast", "value"),
        State("pex-erp-job", "data"),
        prevent_initial_call=True,
    )
    def _erp_rerender(navg, overlap, contrast, key):
        gathered = _ERP_CACHE.get(key) if key else None
        if not gathered or gathered.get("empty"):
            return no_update
        return _erp_figure(gathered, navg, overlap, contrast)
