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

import base64
import logging
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
from src.periictal import embed_io as _eio
from src.periictal import forecast as _fc
from src.periictal import palette as _pal
from src.periictal import passive as _passive
from src.periictal import stim_map as _sm
from src.periictal.embed import confound_readout, embed
from src.periictal.persist import build_matrix_cached
from src.periictal import erpimage as _erp
from src.periictal import trajectory as _traj
from src.periictal.erpimage import (decimate_rows, gather_leadup_trials,
                                    sliding_trial_average)
from src.periictal.selection import summarize_selection
from src.periictal import trendtest as _tt
from src.preictal.isi import scored_seizures
from src.utils import evoked_features as _ef
from src.utils.evoked_features import FeatureConfig
from src.utils.evoked_output import list_animals

logger = logging.getLogger(__name__)

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

# Job-status dicts are keyed by a selection signature, so their key space is
# unbounded (every distinct animal/protocol/variant/window/method combination
# leaves a permanent entry). Entries are small, but "small x forever" is still
# a leak -- prune oldest-first. Dicts preserve insertion order (3.7+).
_JOBS_MAX = 200


def _prune_jobs(jobs: dict, max_n: int = _JOBS_MAX) -> None:
    """Drop oldest job-status entries beyond *max_n*. Caller holds _LOCK."""
    while len(jobs) > max_n:
        jobs.pop(next(iter(jobs)), None)


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {}).update(kw)
        _prune_jobs(_JOBS)


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
        _prune_jobs(_JOBS)
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
        # NOTE: deliberately NOT warm_missing. Computing a missing sidecar
        # means reading a multi-GB recording, and an animal can be hundreds of
        # recordings behind -- doing that inline made this build appear to hang
        # forever on "joining stimuli to seizures". Older-schema sidecars are
        # now READ as-is (newer feature columns simply come back NaN), and the
        # background warmer (utils.sidecar_warm) fills in genuinely missing
        # ones, so the tab stays responsive and catches up on its own.
        if df.empty:
            _finish(job_id, {"empty": True,
                             "reason": _empty_reason(store, evoked_dir,
                                                     animal, protocol)})
            return
        # The embedding / scatter / trajectory are PRE-onset only (unchanged);
        # the full frame (pre + post) is kept for the per-seizure trend test.
        pre = df[df["phase"] == "pre"].reset_index(drop=True)
        if pre.empty:
            _finish(job_id, {"empty": True,
                             "reason": _empty_reason(store, evoked_dir,
                                                     animal, protocol)})
            return
        _set(job_id, progress=f"embedding {len(pre):,} stimuli ({method.upper()})…")
        res = embed(pre, method=method, cap=cap)
        sub = pre.loc[res["rows"]].reset_index(drop=True)
        ro = confound_readout(res["emb"], sub)
        _finish(job_id, {"empty": False, "emb": res["emb"], "sub": sub,
                         "full": df.reset_index(drop=True),
                         "metrics": list(_cfg.metrics_for_variant(variant)),
                         "cols": list(res["cols"]),
                         "readout": ro, "meta": res["meta"],
                         "method": res["method"],
                         "n_seizures": int(pre["seizure_idx"].nunique())})
    except Exception as e:                                     # noqa: BLE001
        # Log the TRACEBACK, not just str(e). The UI can only show one line, and
        # a bare message ("assignment destination is read-only") names neither
        # the file nor the frame -- which made a real build failure impossible
        # to diagnose without re-deriving the whole call path by hand.
        logger.exception("peri-ictal build failed (job=%s animal=%s protocol=%s "
                         "variant=%s sidecar=%s window_h=%s method=%s): %s",
                         job_id, animal, protocol, variant, sidecar_variant,
                         window_h, method, e)
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


def _animals_a0(store):
    animals = list_animals(_evoked_dir(store))
    a0 = "BCH111" if "BCH111" in animals else (animals[0] if animals else None)
    return animals, a0


def scope_bar(store):
    """The shared scope header present on EVERY peri-ictal sub-tab: selection +
    feature window + build + preview + status, plus the build job store/poll.

    The scope controls persist across sub-tab swaps via the app's automatic
    session-persistence (``_enable_persistence``); the job Store persists
    explicitly (Stores are skipped by that walk) so each lens can redraw from the
    module cache on remount."""
    animals, a0 = _animals_a0(store)
    return html.Div([
        _title_block(),
        card(section_header("Selection"), _controls(store, animals, a0),
             _window_panel(), _seizure_panel()),
        card(html.Div(id="pex-preview", style={"marginBottom": SPACE_2}),
             html.Div(id="pex-status", style={"color": COLOR_TEXT_SECONDARY,
                                              "fontSize": FONT_SIZE_CAPTION,
                                              "minHeight": "14px"}),
             _save_load_row(),
             style={"marginTop": SPACE_3}),
        dcc.Interval(id="pex-poll", interval=1200, disabled=True),
        # storage_type="session" is how a dcc.Store persists its data across
        # sub-tab swaps (Store has no `persistence` prop) — so a lens can redraw
        # from the module cache on remount using the retained job id.
        dcc.Store(id="pex-job", storage_type="session"),
        dcc.Download(id="pex-export-dl"),
    ])


def _save_load_row() -> html.Div:
    """Export the currently-built embedding to a file, and import one back to
    restore the exact context (scatter + trend + PDF/CDF + forecasting) without
    rebuilding. See periictal.embed_io."""
    # The import control is a button-STYLED dcc.Upload, not a <button> inside an
    # Upload: a real button child captures the click and the file dialog never
    # opens. Styling the Upload itself keeps the whole area as the file target.
    up_style = {"display": "inline-block", "padding": "6px 12px",
                "borderRadius": RADIUS_SM, "border": f"1px solid {COLOR_DIVIDER}",
                "color": COLOR_TEXT_PRIMARY, "background": COLOR_SURFACE_2,
                "cursor": "pointer", "fontSize": FONT_SIZE_CAPTION,
                "userSelect": "none"}
    return html.Div([
        button("⬇ Export embedding", "pex-export-btn", variant="secondary",
               **{"title": "Save the current built embedding to a file you can "
                           "re-import later."}),
        dcc.Upload(id="pex-import-up", multiple=False,
                   children="⬆ Import embedding", style=up_style),
        html.Span(id="pex-io-status",
                  style={"color": COLOR_TEXT_SECONDARY,
                         "fontSize": FONT_SIZE_CAPTION, "marginLeft": SPACE_3}),
    ], style={"display": "flex", "alignItems": "center", "gap": SPACE_3,
              "flexWrap": "wrap", "marginTop": SPACE_2})


def layout_embedding(store):
    """Embedding lens: the PCA/UMAP scatter (+ confound strip), the feature-space
    loadings, the feature reference, and lasso details."""
    return html.Div([
        scope_bar(store),
        card(_embed_colorby_ctl(),
             dcc.Loading(
                 custom_spinner=loading_icon("Building…"),
                 overlay_style={"visibility": "visible", "opacity": 0.4},
                 children=dcc.Graph(
                     id="pex-graph", clear_on_unhover=True,
                     config={"displaylogo": False},
                     figure=empty_fig("Press ▶ Build to embed the lead-up stimuli",
                                      hint="Pick an animal and protocol above."))),
             _explainer(),
             html.Div(id="pex-reading", style={"marginTop": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Feature space — what defines the axes"),
             html.Div("Each axis is a weighted mix of the standardized features. "
                      "The bars are the PCA loadings: how much each feature pushes "
                      "PC1 and PC2 — i.e. the directions of the feature vector "
                      "space the scatter is a projection of. (PCA only; UMAP is "
                      "non-linear and has no loadings.)",
                      style={"color": COLOR_TEXT_SECONDARY,
                             "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch",
                             "marginBottom": SPACE_2}),
             dcc.Graph(id="pex-loadings", config={"displaylogo": False},
                       figure=empty_fig("Build to see which features define the "
                                        "axes")),
             style={"marginTop": SPACE_4}),
        card(section_header("Features fed to the embedding"),
             _feature_reference(),
             style={"marginTop": SPACE_4}),
        card(section_header("Selected points — details on demand"),
             html.Div(id="pex-details", children=_details_view(None)),
             style={"marginTop": SPACE_4}),
    ], style={"padding": SPACE_4})


def layout_trend(store):
    """Trend & test lens: the lead-time trajectory (visual) + the per-seizure
    trend test (quantitative — attached in the trend-test step)."""
    return html.Div([
        scope_bar(store),
        card(section_header("Lead-time trajectory"),
             html.Div([
                 html.Div("Median value per EQUAL (linear) lead-time bin, from "
                          "onset up to the cap — every bin spans the same amount "
                          "of real time (so at a uniform stim rate the bins hold "
                          "~equal counts). Faint lines = individual seizures (a "
                          "trend is credible only if consistent across them); "
                          "hollow markers = low-n bins.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 html.Div([
                     _ctl("Trajectory y", dcc.Dropdown(
                         id="pex-traj-y", clearable=False,
                         style={**DROPDOWN_STYLE, "minWidth": "240px"}),
                         "Which value to trace against lead-time: any feature "
                         "(median + IQR per bin), a PC embedding coordinate, or "
                         "hour-of-day as a confound check."),
                     _ctl("Up to (cap)", dcc.Input(
                         id="pex-traj-cap", type="number", value=2, min=0.001,
                         step="any", debounce=True,
                         style={**DROPDOWN_STYLE, "width": "80px"}),
                         "Bin from onset up to this much lead time; stimuli "
                         "further out are dropped from the trajectory."),
                     _ctl("unit", dcc.Dropdown(
                         id="pex-traj-cap-unit", clearable=False,
                         options=[{"label": "hours", "value": "h"},
                                  {"label": "minutes", "value": "m"},
                                  {"label": "seconds", "value": "s"}],
                         value="h", style={**DROPDOWN_STYLE, "minWidth": "110px"}),
                         "Unit for the lead-time cap."),
                 ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
                           "alignItems": "flex-start", "marginTop": SPACE_3}),
             ]),
             dcc.Graph(id="pex-traj", config={"displaylogo": False},
                       figure=empty_fig("Build to see the lead-time trajectory")),
             style={"marginTop": SPACE_4}),
        _trend_test_card(),
    ], style={"padding": SPACE_4})


def layout_waveform(store):
    """Waveform lens: the evoked ERP-image + click-a-column ERP."""
    _animals, a0 = _animals_a0(store)
    return html.Div([
        scope_bar(store),
        _erp_card(store, a0),
        dcc.Interval(id="pex-erp-poll", interval=1500, disabled=True),
        dcc.Store(id="pex-erp-job"),
        dcc.Store(id="pex-erp-col"),
    ], style={"padding": SPACE_4})


# --------------------------------------------------------------------- #
#  Preictal-vs-interictal lens (Chang et al. 2026 PDF/CDF + forecaster)
# --------------------------------------------------------------------- #
_PRE_COLOR = "#ff5a5f"        # preictal = red (paper convention)
_INT_COLOR = "#5a9bd4"        # interictal = blue-grey


_LR_MODEL_KEY = "__lr_model__"


def _pc_feature_options(variant: str):
    # The multivariable logistic model's OWN output is the paper's headline
    # metric (Fig 3A) -- offered first, above the raw features.
    return ([{"label": "★ multivariable model score (paper Fig 3A)",
              "value": _LR_MODEL_KEY}]
            + [{"label": f"metric: {m}", "value": m}
               for m in _cfg.metrics_for_variant(variant)])


def layout_pdfcdf(store):
    """Preictal-vs-interictal lens: per-feature PDF + CDF split by preictal
    (0-30 min before a seizure) vs interictal (60-90 min), the discrimination
    AUC + significance, and the paper's prospective logistic-regression
    forecaster. Recreates Chang et al. 2026's metric + modeling strategy."""
    return html.Div([
        scope_bar(store),
        card(section_header("Preictal vs interictal — distribution of a feature"),
             html.Div([
                 html.Div("Every lead-up stimulus is labelled PREICTAL (0-30 min "
                          "before a seizure onset) or INTERICTAL (60-90 min "
                          "before); the 30-60 min band is a redacted buffer. The "
                          "PDF (kernel density) and CDF (empirical) below show how "
                          "the chosen feature is distributed in each state, with "
                          "the rank AUC and its 500-permutation p (per-stimulus, "
                          "optimistic) alongside the honest per-seizure paired "
                          "test. After Chang et al. 2026.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 _ctl("Feature", dcc.Dropdown(
                     id="pex-pc-feature", clearable=False,
                     style={**DROPDOWN_STYLE, "minWidth": "260px"}),
                     "Which evoked feature to compare between the preictal and "
                     "interictal states."),
             ], style={"display": "flex", "flexDirection": "column",
                       "gap": SPACE_3, "marginTop": SPACE_2}),
             html.Div(id="pex-pc-verdict", style={"margin": f"{SPACE_3} 0"}),
             html.Div([
                 dcc.Graph(id="pex-pc-pdf", config={"displaylogo": False},
                           figure=empty_fig("Build to see the PDF"),
                           style={"flex": "1 1 380px", "minWidth": "0",
                                  "height": "375px"}),
                 dcc.Graph(id="pex-pc-cdf", config={"displaylogo": False},
                           figure=empty_fig("Build to see the CDF"),
                           style={"flex": "1 1 380px", "minWidth": "0",
                                  "height": "375px"}),
             ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Per-feature discrimination (ranked)"),
             html.Div("Rank AUC (normalised to ≥.5), 500-permutation p and "
                      "Benjamini–Hochberg q for every feature — the paper's "
                      "single-feature screen (Fig 2). The seizure is the honest "
                      "unit of replication, so the paired p is shown too.",
                      style={"color": COLOR_TEXT_TERTIARY,
                             "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch",
                             "marginBottom": SPACE_2}),
             html.Div(id="pex-pc-scan"),
             style={"marginTop": SPACE_4}),
        card(section_header("Multivariable forecaster (train phase P → test P+1)"),
             html.Div([
                 html.Div("The paper's prospective preictal detector: split the "
                          "recording into equal-seizure-count epilepsy phases, "
                          "train a logistic regression (paper's best-5 features) "
                          "on each phase and test on the NEXT (novel data). The "
                          "ROC, per-phase AUC and normalised coefficients follow "
                          "Fig 3 / Fig 5C.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 _ctl("Phases", dcc.Input(
                     id="pex-pc-nphases", type="number", value=6, min=2, max=20,
                     step=1, debounce=True,
                     style={**DROPDOWN_STYLE, "width": "90px"}),
                     "Number of equal-seizure-count epilepsy phases (clamped to "
                     "the seizure count). The paper used 10; fewer is steadier "
                     "when an animal has few seizures."),
             ], style={"display": "flex", "flexDirection": "column",
                       "gap": SPACE_3, "marginTop": SPACE_2}),
             html.Div(id="pex-pc-fverdict", style={"margin": f"{SPACE_3} 0"}),
             html.Div([
                 dcc.Graph(id="pex-pc-roc", config={"displaylogo": False},
                           figure=empty_fig("Build to see the ROC curves"),
                           style={"flex": "1 1 320px", "minWidth": "0",
                                  "height": "355px"}),
                 dcc.Graph(id="pex-pc-phaseauc", config={"displaylogo": False},
                           figure=empty_fig("Build to see AUC vs phase"),
                           style={"flex": "1 1 320px", "minWidth": "0",
                                  "height": "355px"}),
                 dcc.Graph(id="pex-pc-coef", config={"displaylogo": False},
                           figure=empty_fig("Build to see coefficients"),
                           style={"flex": "1 1 320px", "minWidth": "0",
                                  "height": "355px"}),
             ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_3}),
             style={"marginTop": SPACE_4}),
    ], style={"padding": SPACE_4})


def _pc_layout(title: str, xtitle: str, height: int = 360) -> dict:
    # A FIXED height + autosize off is essential: these graphs live in flex
    # rows, and an autosizing Plotly graph in a flexbox ratchets its container
    # taller on every resize (the "graph grows forever" bug).
    return dict(template="plotly_dark", title=title, height=height,
                autosize=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color=COLOR_TEXT_SECONDARY),
                margin=dict(l=55, r=15, t=45, b=45), xaxis_title=xtitle,
                legend=dict(orientation="h", y=1.12, x=0))


def _pdf_fig(pc: dict, feature: str) -> go.Figure:
    fig = go.Figure()
    if pc["grid"].size and pc["pre_kde"] is not None:
        fig.add_trace(go.Scatter(x=pc["grid"], y=pc["pre_kde"], mode="lines",
                                 name=f"preictal (n={pc['n_pre']})", fill="tozeroy",
                                 line=dict(color=_PRE_COLOR, width=2),
                                 fillcolor="rgba(255,90,95,0.18)"))
    if pc["grid"].size and pc["inter_kde"] is not None:
        fig.add_trace(go.Scatter(x=pc["grid"], y=pc["inter_kde"], mode="lines",
                                 name=f"interictal (n={pc['n_inter']})", fill="tozeroy",
                                 line=dict(color=_INT_COLOR, width=2),
                                 fillcolor="rgba(90,155,212,0.15)"))
    if len(fig.data) == 0:      # KDE undefined -> fall back to histograms
        fig.add_trace(go.Bar(x=pc["centers"], y=pc["pre_hist"], name="preictal",
                             marker_color=_PRE_COLOR, opacity=0.5))
        fig.add_trace(go.Bar(x=pc["centers"], y=pc["inter_hist"], name="interictal",
                             marker_color=_INT_COLOR, opacity=0.5))
        fig.update_layout(barmode="overlay")
    fig.update_layout(**_pc_layout(f"PDF · {feature}", feature, height=370),
                      yaxis_title="density")
    return fig


def _cdf_fig(pc: dict, feature: str) -> go.Figure:
    fig = go.Figure()
    if pc["pre_cdf_x"].size:
        fig.add_trace(go.Scatter(x=pc["pre_cdf_x"], y=pc["pre_cdf_y"],
                                 mode="lines", name="preictal",
                                 line=dict(color=_PRE_COLOR, width=2, shape="hv")))
    if pc["inter_cdf_x"].size:
        fig.add_trace(go.Scatter(x=pc["inter_cdf_x"], y=pc["inter_cdf_y"],
                                 mode="lines", name="interictal",
                                 line=dict(color=_INT_COLOR, width=2, shape="hv")))
    fig.update_layout(**_pc_layout(f"CDF · {feature}", feature, height=370),
                      yaxis_title="cumulative fraction", yaxis_range=[0, 1])
    return fig


def _pc_verdict(pc: dict, pp: dict, ps: dict) -> object:
    """Plain-language AUC + significance callout for the selected feature."""
    if not np.isfinite(pc.get("auc_norm", np.nan)):
        return _callout("No preictal/interictal samples for this feature yet — "
                        "build, or pick another feature.", COLOR_WARNING, "⚠")
    strong = pc["auc_norm"] >= 0.7 and pp.get("p", 1) < 0.05 and ps.get("p", 1) < 0.05
    colour = COLOR_SUCCESS if strong else COLOR_WARNING if pc["auc_norm"] >= 0.6 \
        else COLOR_TEXT_SECONDARY
    txt = (f"AUC {pc['auc_norm']:.3f} (preictal {pc['direction']}) · "
           f"permutation p {_fmt_p(pp.get('p'))} (per-stimulus) · "
           f"paired-seizure p {_fmt_p(ps.get('p'))} over {ps.get('n_seizures', 0)} "
           f"seizures · n = {pc['n_pre']} preictal / {pc['n_inter']} interictal "
           "stimuli.")
    return _callout(txt, colour, "✓" if strong else "•")


def _pc_scan_table(rows: list) -> object:
    """Compact ranked table of per-feature preictal/interictal AUC + p + q
    (top 25) for the preictal-vs-interictal lens."""
    if not rows:
        return html.Div("No features scored.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_CAPTION})
    head = [html.Th(h, style={"textAlign": "left", "padding": "4px 10px",
                              "color": COLOR_TEXT_TERTIARY,
                              "fontSize": FONT_SIZE_CAPTION})
            for h in ("feature", "AUC", "dir", "perm p", "q", "n pre/int")]
    body = []
    for r in rows[:25]:
        cells = [r["feature"], f"{r['auc_norm']:.3f}", r["direction"],
                 _fmt_p(r["p"]), _fmt_p(r["q"]),
                 f"{r['n_pre']}/{r['n_inter']}"]
        body.append(html.Tr([html.Td(c, style={"padding": "3px 10px",
                                                "fontSize": FONT_SIZE_CAPTION,
                                                "color": COLOR_TEXT_PRIMARY})
                             for c in cells]))
    return html.Table([html.Thead(html.Tr(head)), html.Tbody(body)],
                      style={"borderCollapse": "collapse", "width": "100%"})


def _roc_fig(res: dict) -> go.Figure:
    fig = go.Figure()
    # The chance diagonal = a classifier with ROC-AUC 0.5 (labelled so it reads
    # as the reference, not another phase).
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="chance (AUC 0.5)",
                             line=dict(color=COLOR_TEXT_TERTIARY, dash="dash",
                                       width=1.5), hoverinfo="skip"))
    phases = [p for p in res.get("phases", []) if p.get("fpr") is not None]
    for i, p in enumerate(phases):
        shade = 0.35 + 0.6 * (i / max(1, len(phases) - 1))
        fig.add_trace(go.Scatter(
            x=p["fpr"], y=p["tpr"], mode="lines",
            name=f"P{p['train_phase']}→{p['test_phase']} ({p['auc']:.2f})",
            line=dict(color=f"rgba(255,90,95,{shade:.2f})", width=2)))
    fig.update_layout(**_pc_layout("ROC per phase (test on P+1)",
                                   "false-positive rate", height=350),
                      yaxis_title="true-positive rate",
                      xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


def _phaseauc_fig(res: dict) -> go.Figure:
    phases = res.get("phases", [])
    xs = [p["test_phase"] for p in phases]
    ys = [p["auc"] for p in phases]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[min(xs) if xs else 0, max(xs) if xs else 1],
                             y=[0.5, 0.5], mode="lines", showlegend=False,
                             line=dict(color=COLOR_DIVIDER, dash="dash", width=1),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="test AUC",
                             line=dict(color=_PRE_COLOR, width=2),
                             marker=dict(size=8)))
    fig.update_layout(**_pc_layout("AUC vs epilepsy phase", "test phase",
                                   height=350),
                      yaxis_title="test AUC", yaxis_range=[0.3, 1.0])
    return fig


def _coef_fig(res: dict) -> go.Figure:
    coef = res.get("coefficients", {}) or {}
    items = sorted(coef.items(), key=lambda kv: kv[1])
    fig = go.Figure()
    if items:
        fig.add_trace(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                             orientation="h", marker_color=COLOR_ACCENT))
    fig.update_layout(**_pc_layout("Normalised |coefficient|", "importance (Σ=1)",
                                   height=350),
                      yaxis_title="")
    return fig


def _forecast_verdict(res: dict, n_seizures: int) -> object:
    mean_auc = res.get("mean_auc", float("nan"))
    if not np.isfinite(mean_auc):
        return _callout("Not enough phases/seizures with both classes to fit the "
                        "forecaster — lower the phase count or pick an animal with "
                        "more seizures.", COLOR_WARNING, "⚠")
    slope = res.get("phase_auc_slope", float("nan"))
    colour = COLOR_SUCCESS if mean_auc >= 0.75 else COLOR_WARNING \
        if mean_auc >= 0.6 else COLOR_TEXT_SECONDARY
    trend = (f"; AUC {'rises' if slope > 0 else 'falls'} over phases "
             f"(slope {slope:+.3f})") if np.isfinite(slope) else ""
    txt = (f"Mean prospective test AUC {mean_auc:.3f} across {res.get('n_phases')} "
           f"phases ({n_seizures} seizures){trend}. Features: "
           f"{', '.join(res.get('features', []))}.")
    return _callout(txt, colour, "✓" if mean_auc >= 0.75 else "•")


def _embed_colorby_ctl() -> html.Div:
    """The Embedding-lens colour controls: what to colour by, plus the
    time-to-onset colour CAP (value + unit). Colour is linear and equal up to the
    cap and saturates beyond it — no log. Recolours instantly (no rebuild)."""
    return html.Div([
        _ctl("Colour by", dcc.Dropdown(
            id="pex-colorby", clearable=False,
            style={**DROPDOWN_STYLE, "minWidth": "220px"}),
            "Recolour instantly (no rebuild). Try hour-of-day and stim "
            "fingerprint as confound checks."),
        _ctl("Time-colour cap", dcc.Input(
            id="pex-color-cap", type="number", value=2, min=0.001, step="any",
            debounce=True, style={**DROPDOWN_STYLE, "width": "80px"}),
            "Colour runs linearly and equally from onset up to this much lead "
            "time; stimuli further out saturate at the far colour. Only affects "
            "the time-to-onset colouring."),
        _ctl("unit", dcc.Dropdown(
            id="pex-color-cap-unit", clearable=False,
            options=[{"label": "hours", "value": "h"},
                     {"label": "minutes", "value": "m"},
                     {"label": "seconds", "value": "s"}],
            value="h", style={**DROPDOWN_STYLE, "minWidth": "110px"}),
            "Unit for the colour cap."),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
              "alignItems": "flex-start", "marginBottom": SPACE_3})


def _trend_test_card() -> object:
    """Per-seizure trend-test card (Trend & test lens): a plain-language verdict,
    a forest of per-seizure Spearman rho, and an all-features BH-FDR scan. Reads
    the cached matrix synchronously (no background job)."""
    return card(
        section_header("Per-seizure trend test"),
        html.Div("Collapses each seizure to ONE number — the Spearman correlation "
                 "of the selected feature with lead-time over that seizure's "
                 "pre-onset stimuli — then tests whether those per-seizure trends "
                 "are consistent across seizures (the seizure is the unit of "
                 "replication, not the stimulus). The post-onset window is a "
                 "built-in positive control; hour-of-day is the circadian check.",
                 style={"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
                        "maxWidth": "95ch", "marginBottom": SPACE_3}),
        html.Div(id="pex-trend-verdict", style={"marginBottom": SPACE_3}),
        dcc.Graph(id="pex-trend-forest", config={"displaylogo": False},
                  figure=empty_fig("Build to run the per-seizure trend test")),
        html.Div("Scanning every feature? Control the false-discovery rate — one "
                 "p<0.05 across ~22 features is expected by chance. The table "
                 "below reports Benjamini–Hochberg q-values.",
                 style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
                        "margin": f"{SPACE_3} 0 {SPACE_2}", "maxWidth": "95ch"}),
        html.Div(id="pex-trend-table"),
        style={"marginTop": SPACE_4})


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
        _ctl("Display", dcc.RadioItems(
            id="pex-erp-mode", value="amplitude", inline=True,
            labelStyle=_RADIO_LABEL, inputStyle=_RADIO_INPUT,
            options=[{"label": "raw", "value": "amplitude"},
                     {"label": "Δ from mean", "value": "deviation"},
                     {"label": "row z-score", "value": "zscore"}]),
            "Raw = the actual amplitude (dominated by the fixed early peak). "
            "Δ from mean subtracts the grand-average waveform per latency so only "
            "CHANGES show. Row z-score normalizes each latency — most sensitive."),
        html.Div(button("▶ Build ERP", "pex-erp-build", icon_name="play",
                        **{"title": "Read the raw traces for this seizure's "
                                    "lead-up and stack them. Runs in the "
                                    "background (~15 s per recording)."}),
                 style={"alignSelf": "flex-end"}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
              "alignItems": "flex-start"})
    return card(
        section_header("Evoked ERP-image — trials around onset"),
        html.Div("Each column is an evoked-response trial (smoothed by the "
                 "N-trial sliding average), ordered chronologically: before → "
                 "ONSET (dashed line) → after (an equal window past onset). Rows "
                 "are post-stim samples; colour is signed amplitude (diverging at "
                 "0), so the biphasic waveform is visible. Click a column (or use "
                 "◀ ▶) to see its averaged ERP below. Reads raw traces on demand.",
                 style={"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
                        "maxWidth": "95ch", "marginBottom": SPACE_3}),
        controls,
        html.Div(id="pex-erp-warn", style={"marginTop": SPACE_2}),
        dcc.Loading(
            custom_spinner=loading_icon("Reading traces…"),
            overlay_style={"visibility": "visible", "opacity": 0.4},
            children=dcc.Graph(id="pex-erp", config={"displaylogo": False},
                               figure=empty_fig("Pick a seizure, then ▶ Build ERP"))),
        html.Div([
            button("◀ Prev", "pex-erp-prev", variant="secondary"),
            html.Span(id="pex-erp-navlabel",
                      style={"color": COLOR_TEXT_SECONDARY,
                             "fontSize": FONT_SIZE_CAPTION,
                             "margin": f"0 {SPACE_3}", "minWidth": "22ch",
                             "display": "inline-block", "textAlign": "center"}),
            button("Next ▶", "pex-erp-next", variant="secondary"),
        ], style={"display": "flex", "alignItems": "center", "gap": SPACE_2,
                  "marginTop": SPACE_3, "flexWrap": "wrap"}),
        dcc.Graph(id="pex-erp-wave", config={"displaylogo": False},
                  figure=empty_fig("Click a column (or use ◀ ▶) to see its ERP")),
        html.Div(id="pex-erp-status", style={"color": COLOR_TEXT_SECONDARY,
                                             "fontSize": FONT_SIZE_CAPTION,
                                             "minHeight": "14px", "marginTop": SPACE_2}),
        style={"marginTop": SPACE_4})


_EXPLAIN_P = {"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
              "margin": f"{SPACE_2} 0", "maxWidth": "95ch"}


_FREF_TH = {"textAlign": "left", "padding": f"{SPACE_1} {SPACE_3}",
            "color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
            "borderBottom": f"1px solid {COLOR_DIVIDER}", "position": "sticky",
            "top": "0", "background": COLOR_SURFACE_2}
_FREF_TD = {"padding": f"{SPACE_1} {SPACE_3}", "fontSize": FONT_SIZE_CAPTION,
            "verticalAlign": "top", "borderBottom": f"1px solid {COLOR_DIVIDER}"}


def _feature_reference() -> html.Div:
    """Code-accurate reference of every feature fed to the embedding — the exact
    computation of each, for validation. Sourced from
    ``evoked_features.COLUMN_DOCS`` so the formulas can't drift from the code;
    'evoked only' marks the three post-stim-window features the passive variant
    drops. Rendered VISIBLY (its own card), not buried in a disclosure."""
    cols = list(_cfg.CHEAP_METRICS)
    rows = []
    for col in cols:
        passive_drop = col in _cfg.PASSIVE_INVALID
        name = [html.Code(col, style={"color": COLOR_TEXT_PRIMARY,
                                      "background": "transparent"})]
        if passive_drop:
            name.append(html.Span(" · evoked only",
                                  style={"color": COLOR_WARNING,
                                         "fontSize": "11px"}))
        rows.append(html.Tr([
            html.Td(name, style={**_FREF_TD, "whiteSpace": "nowrap"}),
            html.Td(_ef.COLUMN_DOCS.get(col, "—"),
                    style={**_FREF_TD, "color": COLOR_TEXT_SECONDARY,
                           "maxWidth": "70ch"}),
        ]))
    table = html.Div(html.Table([
        html.Thead(html.Tr([html.Th("Feature", style=_FREF_TH),
                            html.Th("How it's computed", style=_FREF_TH)])),
        html.Tbody(rows),
    ], style={"borderCollapse": "collapse", "width": "100%"}),
        style={"maxHeight": "360px", "overflowY": "auto", "marginTop": SPACE_2,
               "border": f"1px solid {COLOR_DIVIDER}", "borderRadius": RADIUS_SM})
    return html.Div([
        html.Div(f"The {len(cols)} features below are each standardized (z-scored "
                 "per column) then reduced to 2-D by PCA/UMAP — these are the "
                 "ONLY inputs to the layout. 'evoked only' rows are dropped for "
                 "the passive (pre-stim) variant. Formulas mirror "
                 "src/utils/evoked_features.py exactly.",
                 style={"color": COLOR_TEXT_SECONDARY, "fontSize": FONT_SIZE_CAPTION,
                        "maxWidth": "95ch"}),
        table,
        html.Div("Critical-slowing measures (recovery_tau, ac_width, "
                 "recovery_slope, …) are computed in evoked_features but are NOT "
                 "in this default set — they live in EXPENSIVE_COLUMNS.",
                 style={"color": COLOR_TEXT_TERTIARY, "fontSize": "11px",
                        "marginTop": SPACE_2, "fontStyle": "italic"}),
    ])


def _explainer() -> html.Div:
    """Persistent 'what am I looking at' panel: what a point is, what sets its
    position (the features -- NOT the colour), and how to read UMAP honestly."""
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
                "PCA axes, by contrast, carry real variance (shown on the axes). "
                "The features fed in — and how the axes weight them — are shown in "
                "the two panels below."],
               style=_EXPLAIN_P),
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


def _seizure_panel() -> html.Details:
    """Collapsed 'Seizure events' panel: an include/exclude checkbox per scored
    seizure so bad-data-period seizures can be dropped from the embedding.
    Persistent per animal; the list is filled by a callback on animal/protocol."""
    return html.Details([
        html.Summary("Seizure events — exclude bad-data periods",
                     style={"cursor": "pointer", "color": COLOR_TEXT_SECONDARY,
                            "fontSize": FONT_SIZE_CAPTION, "userSelect": "none"}),
        html.Div(id="pex-seizure-list",
                 style={"maxHeight": "220px", "overflowY": "auto",
                        "marginTop": SPACE_2}),
        html.Div(id="pex-seizure-status",
                 style={"color": COLOR_TEXT_TERTIARY,
                        "fontSize": FONT_SIZE_CAPTION, "marginTop": SPACE_1}),
    ], style={"marginTop": SPACE_2})


def _seizures_in_scope(store, animal, protocol) -> list:
    """The scored seizures for an (animal, protocol) scope, chronological."""
    if not animal:
        return []
    return [s for s in scored_seizures(store, animal)
            if not protocol or protocol in _tok(s.session_dir)]


def _seizure_key(s) -> str:
    """Stable checklist value for a seizure: file + rounded onset offset."""
    return f"{int(s.file_id)}:{round(float(s.eo_sec), 2)}"


def _seizure_label(s) -> str:
    try:
        when = datetime.fromtimestamp(s.onset_epoch).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, OverflowError):
        when = "?"
    rac = f"Racine {s.racine}" if s.racine is not None else "Racine —"
    typ = f" · {s.seizure_type}" if s.seizure_type else ""
    return f" {when} · {rac}{typ}"


def _keep_status(kept: int, excluded: int, total: int) -> str:
    return (f"{kept} kept · {excluded} excluded of {total}"
            + (" — press ▶ Build to apply." if excluded else ""))


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
                options=[{"label": "[1–200 ms] (fast)", "value": "full"},
                         {"label": "custom window", "value": "custom"}]),
                "Default = the shared sidecar, now windowed to [1–200 ms] "
                "post-stim (fast, pre-computed — use this). Custom = recompute "
                "features over a DIFFERENT ms window from the raw traces (slow "
                "first build). (Passive is always a window.)"),
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
        html.Div("t = 0 is the stimulus. Evoked = a POST-stim window (default "
                 "[1→200] ms); passive = PRE-stim (−200→−1). The default reads "
                 "the pre-computed sidecar; a CUSTOM window recomputes features "
                 "from the raw traces (slow first build, then cached).",
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


def _trajectory_fig(cached, y_key, cap_sec=None) -> go.Figure:
    if not cached or cached.get("empty"):
        return empty_fig("Build to see the lead-time trajectory")
    values, ylabel = _traj_values(cached, y_key)
    if values is None:
        return empty_fig("Pick a trajectory metric")
    sub = cached["sub"]
    tto = sub["time_to_onset_sec"].to_numpy(dtype=float)
    cap = float(cap_sec) if (cap_sec and cap_sec > 0) else _pal.DEFAULT_TIME_CAP_SEC
    edges = _traj.linear_edges(cap)                      # equal linear bins 0..cap
    tr = _traj.lead_time_trajectory(tto, values, sub["seizure_idx"].to_numpy(),
                                    edges, log_centers=False)
    return _render_trajectory(tr, ylabel, cap)


def _render_trajectory(tr, ylabel, cap) -> go.Figure:
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
    tv, tt = _pal.linear_time_ticks(cap)                 # equal linear ticks
    fig.update_layout(
        height=300, margin=dict(l=54, r=20, t=22, b=40),
        xaxis=dict(title="lead time before onset", tickvals=tv, ticktext=tt,
                   range=[cap, 0]),                      # onset (0) at the right
        yaxis=dict(title=ylabel), uirevision="pex-traj")
    return fig


# --------------------------------------------------------------------- #
#  Per-seizure trend test (Trend & test lens) — synchronous, from the cache
# --------------------------------------------------------------------- #

def _seizure_labels(full) -> dict:
    """{seizure_idx: 'R{racine} · MM-DD HH:MM'} from the matrix, for the forest
    row labels."""
    out: dict = {}
    g = full.groupby("seizure_idx")[["seizure_onset_epoch",
                                     "seizure_racine"]].first()
    for sid, row in g.iterrows():
        try:
            when = datetime.fromtimestamp(
                float(row["seizure_onset_epoch"])).strftime("%m-%d %H:%M")
        except (ValueError, OverflowError, OSError):
            when = "?"
        rac = row["seizure_racine"]
        rac = int(rac) if np.isfinite(rac) else "?"
        out[int(sid)] = f"R{rac} · {when}"
    return out


def _forest_fig(per_sz, summary, labels) -> go.Figure:
    """One marker per seizure at its Spearman rho (diverging colour at 0, size ∝
    √n), a dotted null line at 0, and a dashed median line annotated with the
    across-seizure p."""
    if not per_sz:
        return empty_fig("Not enough per-seizure data to test this feature")
    per_sz = sorted(per_sz, key=lambda r: r["rho"])
    rhos = [r["rho"] for r in per_sz]
    ys = [labels.get(r["seizure_idx"], f"sz {r['seizure_idx']}") for r in per_sz]
    ns = [r["n"] for r in per_sz]
    hrs = [r["hour_rho"] for r in per_sz]
    # size ∝ √n, but scaled RELATIVE to the busiest seizure (n is thousands of
    # stimuli here) so markers stay ~10–26 px, not giant overlapping blobs.
    mx = np.sqrt(max(ns)) if ns else 1.0
    sizes = [10.0 + 16.0 * np.sqrt(max(n, 1)) / mx for n in ns]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(width=1, color="rgba(160,160,176,0.55)",
                                 dash="dot"))
    med = summary["median_rho"]
    if np.isfinite(med):
        fig.add_vline(x=med, line=dict(width=2, color=COLOR_ACCENT, dash="dash"),
                      annotation_text=f"median ρ={med:+.2f} · p={_fmt_p(summary['p'])}",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=COLOR_ACCENT))
    fig.add_trace(go.Scatter(
        x=rhos, y=ys, mode="markers",
        marker=dict(size=sizes,
                    color=rhos, colorscale="RdBu", reversescale=True,
                    cmin=-1, cmax=1, line=dict(width=1, color="rgba(20,20,28,0.85)"),
                    colorbar=dict(title=dict(text="ρ", font=dict(size=9)),
                                  thickness=10, len=0.7)),
        customdata=np.column_stack([ns, hrs]),
        hovertemplate="ρ %{x:.2f} · n=%{customdata[0]:d} · circadian ρ="
                      "%{customdata[1]:.2f}<extra>%{y}</extra>", showlegend=False))
    fig.update_layout(
        height=max(200, 34 * len(per_sz) + 96),
        margin=dict(l=150, r=20, t=36, b=44),
        xaxis=dict(title="Spearman ρ  (feature vs lead-time, per seizure)",
                   range=[-1.05, 1.05], zeroline=False),
        yaxis=dict(title="", automargin=True), uirevision="pex-forest")
    return fig


def _verdict_view(summary, control, circadian, feature):
    """A plain-language verdict: consistency + significance, the post-ictal
    positive control, and the circadian caveat."""
    n = summary["n_seizures"]
    if n == 0:
        return _callout("Not enough per-seizure data to test this feature "
                        "(each seizure needs several pre-onset stimuli).",
                        COLOR_WARNING, "⚠")
    p, med = summary["p"], summary["median_rho"]
    sig = p < 0.05
    dom = max(summary["n_neg"], summary["n_pos"])
    lead = "rises" if med < 0 else "falls" if med > 0 else "is flat"
    main = (f"{dom}/{n} seizures trend the same way; median ρ={med:+.2f} → the "
            f"feature {lead} toward onset; signed-rank p={_fmt_p(p)} — "
            f"{'SIGNIFICANT' if sig else 'not significant'}.")
    if not control.get("available") or control["n_seizures"] == 0:
        pc = "Post-ictal control: n/a (no post-onset rows for this selection)."
    else:
        pcp = control["p"]
        pc = ("Post-ictal positive control p=" + _fmt_p(pcp) + " — "
              + ("method is sensitive (it detects the known post-ictal effect)."
                 if pcp < 0.05 else
                 "not significant either (low power or weak post-ictal effect)."))
    cir = (f"Circadian: median |ρ| vs hour-of-day = "
           f"{circadian:.2f}" if np.isfinite(circadian) else
           "Circadian: n/a")
    if np.isfinite(circadian) and circadian >= 0.5:
        cir += " ⚠ — the trend may be time-of-day, not pre-ictal."
    else:
        cir += "."
    tone = COLOR_SUCCESS if sig else COLOR_TEXT_SECONDARY
    return html.Div([
        html.Div(main, style={"color": tone, "fontWeight": "600",
                              "fontSize": FONT_SIZE_BODY, "marginBottom": SPACE_1}),
        html.Div(pc, style={"color": COLOR_TEXT_SECONDARY,
                            "fontSize": FONT_SIZE_CAPTION}),
        html.Div(cir, style={"color": COLOR_TEXT_SECONDARY,
                             "fontSize": FONT_SIZE_CAPTION}),
    ], style={"background": COLOR_SURFACE_2, "padding": f"{SPACE_2} {SPACE_3}",
              "borderRadius": RADIUS_SM})


_TREND_TH = {"textAlign": "left", "padding": f"{SPACE_1} {SPACE_3}",
             "color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
             "borderBottom": f"1px solid {COLOR_DIVIDER}", "whiteSpace": "nowrap"}
_TREND_TD = {"padding": f"{SPACE_1} {SPACE_3}", "color": COLOR_TEXT_SECONDARY,
             "fontSize": FONT_SIZE_CAPTION, "whiteSpace": "nowrap"}


def _scan_table(rows):
    """All-features BH-FDR scan as a ranked table; q<0.05 rows highlighted."""
    if not rows:
        return html.Div("No features to scan.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_CAPTION})
    head = ["feature", "dir", "n sz", "p", "q (BH)", "circadian |ρ|",
            "post-ictal p"]
    header = html.Tr([html.Th(h, style=_TREND_TH) for h in head])
    body = []
    for r in rows:
        q = r["q"]
        hot = np.isfinite(q) and q < 0.05
        direction = ("↑ onset" if r["median_rho"] < 0 else
                     "↓ onset" if r["median_rho"] > 0 else "—")
        cells = [r["feature"], direction, r["n_seizures"], _fmt_p(r["p"]),
                 _fmt_p(q),
                 f"{r['circadian']:.2f}" if np.isfinite(r["circadian"]) else "—",
                 _fmt_p(r["post_p"]) if r["post_available"] else "n/a"]
        rstyle = {"background": "rgba(94,124,226,0.16)"} if hot else {}
        body.append(html.Tr([html.Td(str(c), style=_TREND_TD) for c in cells],
                            style=rstyle))
    return html.Table([html.Thead(header), html.Tbody(body)],
                      style={"borderCollapse": "collapse", "width": "100%",
                             "marginTop": SPACE_2})


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _trend_test_views(cached, feature):
    """(forest fig, verdict view, scan table) for the trend-test card — computed
    synchronously from the cached full (pre+post) matrix."""
    full = cached.get("full")
    if full is None or cached.get("empty"):
        return (empty_fig("Build to run the per-seizure trend test"), "", None)
    table = _scan_table(_tt.scan_all_features(full, cached.get("metrics") or []))
    if feature not in full.columns:      # PC coordinates aren't matrix columns
        note = _callout("The per-seizure forest needs a feature column — PC "
                        "coordinates aren't in the matrix. Pick a metric above.",
                        COLOR_TEXT_SECONDARY, "ℹ")
        return (empty_fig("Pick a metric to see its per-seizure forest"),
                note, table)
    pre = full[full["phase"] == "pre"]
    per = _tt.per_seizure_trend(pre, feature)
    summary = _tt.across_seizure_test([r["rho"] for r in per])
    control = _tt.positive_control(full, feature)
    circ = (float(np.nanmedian([abs(r["hour_rho"]) for r in per]))
            if per else float("nan"))
    forest = _forest_fig(per, summary, _seizure_labels(full))
    verdict = _verdict_view(summary, control, circ, feature)
    return forest, verdict, table


# --------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------- #

_UNIT_SEC = {"h": 3600.0, "m": 60.0, "s": 1.0}


def _cap_seconds(cap_val, cap_unit) -> float:
    """The time-to-onset colour cap in seconds from the (value, unit) controls;
    falls back to the 2 h default on a blank/invalid entry."""
    try:
        val = float(cap_val)
    except (TypeError, ValueError):
        return _pal.DEFAULT_TIME_CAP_SEC
    sec = val * _UNIT_SEC.get(cap_unit or "h", 3600.0)
    return sec if sec > 0 else _pal.DEFAULT_TIME_CAP_SEC


def _figure(emb, sub, color_by, method, meta, cap_sec=None) -> go.Figure:
    if emb is None or emb.shape[0] == 0:
        return empty_fig("No lead-up stimuli for this selection")
    n_unique = int(sub[color_by].nunique()) if color_by in sub else 0
    fig = (_categorical_fig(emb, sub, color_by)
           if _pal.is_categorical(color_by, n_unique)
           else _continuous_fig(emb, sub, color_by, cap_sec))
    return _finish_fig(fig, method, meta)


def _continuous_fig(emb, sub, color_by, cap_sec=None) -> go.Figure:
    spec = _pal.continuous_spec(color_by, sub[color_by].to_numpy(dtype=float),
                                cap_sec)
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
        # colour is linear seconds, clamped to the cap; hover shows the TRUE
        # duration (so points beyond the cap still read their real lead time).
        kw["text"] = [_fmt_dur(s) for s in sub["time_to_onset_sec"].to_numpy()]
        kw["hovertemplate"] = "time to onset: %{text}<extra></extra>"
    else:
        kw["hovertemplate"] = f"{spec['label']}: %{{marker.color:.2f}}<extra></extra>"
    return go.Figure(go.Scattergl(**kw))


def _fmt_dur(s) -> str:
    """Human-readable duration for hover (the colour axis is linear seconds
    clamped to the cap; hover shows the TRUE duration)."""
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
            x=emb[m, 0], y=emb[m, 1], mode="markers", name=name, showlegend=True,
            marker=dict(size=4, opacity=0.65, color=_pal.hue_for(name, k)),
            customdata=row_idx[m],
            hovertemplate=f"{_pretty(color_by)}: {name}<extra></extra>"))
    # Force the legend even for a single category (Plotly hides a 1-trace legend
    # by default) so a single-valued colour-by reads as "one value here", not as
    # a broken control.
    fig.update_layout(showlegend=True,
                      legend=dict(title=_pretty(color_by), orientation="h",
                                  y=1.02, yanchor="bottom", font=dict(size=10),
                                  itemsizing="constant"))
    if len(order) <= 1:
        only = order[0] if order else "—"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            text=(f"Only one {_pretty(color_by)} here: <b>{only}</b><br>"
                  f"nothing to distinguish — this colour-by can't separate points "
                  f"for this selection."),
            font=dict(size=12, color=COLOR_WARNING),
            bgcolor="rgba(20,20,28,0.72)", bordercolor=COLOR_WARNING,
            borderwidth=1, borderpad=8, align="center")
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
#  Vector-space view: PCA loadings (which features define each axis)
# --------------------------------------------------------------------- #

def _pc_name(k: int, ev) -> str:
    """'PC{k} (NN%)' when the explained-variance for that axis is known, else
    'PC{k}' (guards a short/absent explained_var list — e.g. single-component
    PCA)."""
    if ev is not None and len(ev) >= k and np.isfinite(ev[k - 1]):
        return f"PC{k} ({ev[k - 1]:.0%})"
    return f"PC{k}"


def _loadings_fig(cached) -> go.Figure:
    """The feature-space view: each feature's loading on PC1/PC2 — i.e. how the
    standardized feature vectors project onto the two axes you see. PCA only;
    UMAP has no linear loadings, so it shows an honest note instead."""
    if not cached or cached.get("empty"):
        return empty_fig("Build to see which features define the axes")
    meta = cached.get("meta") or {}
    comps = meta.get("components")
    cols = cached.get("cols")
    if cached.get("method") != "pca" or not comps or not cols:
        return empty_fig("PCA loadings show which features define each axis — "
                         "switch Embedding to PCA to see them (UMAP is non-linear, "
                         "so it has no feature loadings).")
    comps = np.asarray(comps, dtype=float)
    if comps.ndim != 2 or comps.shape[0] < 1 or comps.shape[1] != len(cols):
        return empty_fig("Loadings unavailable for this build")
    ev = list(meta.get("explained_var") or [])
    pc1 = comps[0]
    order = np.argsort(pc1)                       # most −PC1 → most +PC1
    feats = [cols[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=feats, x=pc1[order], orientation="h", name=_pc_name(1, ev),
        marker_color=COLOR_ACCENT,
        hovertemplate="%{y}<br>PC1 loading %{x:.2f}<extra></extra>"))
    if comps.shape[0] > 1:                        # PC2 only if it exists
        fig.add_trace(go.Bar(
            y=feats, x=comps[1][order], orientation="h", name=_pc_name(2, ev),
            marker_color="#c98500",
            hovertemplate="%{y}<br>PC2 loading %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(width=1, color="rgba(160,160,176,0.5)"))
    fig.update_layout(
        barmode="group", bargap=0.25, height=max(320, 20 * len(feats) + 90),
        margin=dict(l=150, r=20, t=44, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", font=dict(size=10)),
        xaxis=dict(title="loading (standardized-feature weight on the axis)",
                   zeroline=False),
        yaxis=dict(title="", automargin=True), uirevision="pex-loadings")
    return fig


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
        # The default sidecar is now the [1, 200] ms post-stim window
        # (evoked_output.DEFAULT_EVOKED_CFG), not the full trace.
        return "1 to 200 ms"
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
    # Say so when UMAP was asked for and PCA was drawn instead -- an unlabelled
    # substitution would be read as the UMAP figure that was requested.
    fb = (f"Showing PCA, not UMAP: {meta.get('fallback_reason')}."
          if meta.get("fallback_from") == "umap" else "")
    return html.Div([
        html.Div(prov, style={"color": COLOR_TEXT_TERTIARY,
                              "fontSize": FONT_SIZE_CAPTION, "marginBottom": SPACE_2}),
        _callout(fb, COLOR_WARNING, "⚠") if fb else html.Div(),
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


_EMPTY_MSG = ("No lead-up stimuli — fewer than 2 seizures, none in this protocol, "
              "or no fresh sidecars for this animal.")


def _empty_reason(store, evoked_dir, animal, protocol) -> str:
    """Say WHICH of the three causes made the build empty.

    The generic message above lumps them together, which sends people hunting
    for a bug when the real answer is usually "features for this protocol
    haven't been computed yet" -- a backlog the warmer clears on its own.
    Only runs on an empty result, so the directory scan is not a hot path."""
    try:
        szs = scored_seizures(store, animal)
    except Exception:                                   # noqa: BLE001
        return _EMPTY_MSG
    n_all = len(szs)
    if n_all < 2:
        return (f"{animal} has {n_all} scored seizure(s) — at least 2 are "
                "needed to build lead-up windows.")
    if protocol:
        n_p = sum(1 for s in szs if protocol in str(s.session_dir or ""))
        if n_p < 2:
            return (f"{animal} has {n_all} scored seizures but only {n_p} in "
                    f"'{protocol}'. Try another protocol / stim group.")
    have = tot = 0
    try:
        from src.utils.evoked_output import (animals_in_filename,
                                             list_evoked_files,
                                             sidecar_is_current)
        for fp in list_evoked_files(evoked_dir):
            if animal not in animals_in_filename(fp):
                continue
            if protocol and protocol not in os.path.basename(fp):
                continue
            tot += 1
            # sidecar_is_current, NOT os.path.exists: after a schema bump every
            # recording still HAS a sidecar (the old version), so counting
            # existence read "196 of 197" and never moved while the warmer
            # rebuilt them -- the build needs the CURRENT (v5) sidecar.
            if sidecar_is_current(fp, animal):
                have += 1
    except Exception:                                   # noqa: BLE001
        return _EMPTY_MSG
    if tot and have < tot:
        scope = f"{animal}" + (f" / {protocol}" if protocol else "")
        pct = int(100 * have / tot) if tot else 0
        return (f"Features for {scope} are being (re)computed: {have} of {tot} "
                f"recordings up to date ({pct}%). The background warmer is "
                "re-reading each recording (slow — minutes/file); rebuild once "
                "it climbs. A recording that failed MATLAB has no evoked data "
                "and never reaches 100%.")
    return _EMPTY_MSG


def _embed_fig_reading(cached, color_by, cap_sec=None):
    """(scatter figure, reading strip) for a ready embedding build (empty-safe).
    The single source of truth for the Embedding lens's render."""
    if not cached or cached.get("empty"):
        return empty_fig((cached or {}).get("reason") or _EMPTY_MSG), ""
    fig = _figure(cached["emb"], cached["sub"], color_by,
                  cached.get("method", "pca"), cached.get("meta"), cap_sec)
    return fig, _reading_strip(cached)


def _build_status(cached, jid):
    """(status, poll_disabled, job) once a build is ready in cache — the pulse
    that wakes the per-lens render callbacks (they key off the job store)."""
    if cached.get("empty"):
        return cached.get("reason") or _EMPTY_MSG, True, jid
    return "✓ built", True, jid


def _render_cached(cached, color_by, jid, traj_y):
    """(figure, reading, status, poll_disabled, job, trajectory) — composes the
    per-lens renderers; retained as a stable helper for tests."""
    if cached.get("empty"):
        msg = cached.get("reason") or _EMPTY_MSG
        return empty_fig(msg), "", msg, True, jid, empty_fig(msg)
    fig, reading = _embed_fig_reading(cached, color_by)
    traj = _trajectory_fig(cached, traj_y or "peak_to_trough")
    return fig, reading, "✓ built", True, jid, traj


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
        _prune_jobs(_ERP_JOBS)


def _erp_finish(key, result):
    with _LOCK:
        _ERP_CACHE[key] = result
        while len(_ERP_CACHE) > _ERP_CACHE_MAX:
            _ERP_CACHE.popitem(last=False)
        _ERP_JOBS[key] = {"status": "done", "progress": "done"}
        _prune_jobs(_ERP_JOBS)


def _erp_kick(store, evoked_dir, key, animal, protocol, sz, lookback, frm, to):
    with _LOCK:
        if key in _ERP_CACHE:
            return
        st = _ERP_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _ERP_JOBS[key] = {"status": "running", "progress": "starting…"}
        _prune_jobs(_ERP_JOBS)
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
        logger.exception("peri-ictal ERP build failed (key=%s): %s", key, e)
        _erp_set(key, status="error", progress=f"error: {e}")


def _erp_transform(z, mode):
    """Per-latency (row) display transform: raw, deviation-from-mean-waveform, or
    row z-score -- so a constant response goes flat and only CHANGES show."""
    if mode == "deviation":
        return z - np.nanmean(z, axis=1, keepdims=True), "Δ amplitude"
    if mode == "zscore":
        m = np.nanmean(z, axis=1, keepdims=True)
        sd = np.nanstd(z, axis=1, keepdims=True)
        return (z - m) / np.where(sd > 0, sd, 1.0), "z-score"
    return z, "amplitude"


def _erp_figure(gathered, n, overlap, contrast, mode="amplitude") -> go.Figure:
    if not gathered or gathered.get("empty"):
        return empty_fig("No trials in this lead-up window")
    trials, row_ms, tto = gathered["trials"], gathered["row_ms"], gathered["tto"]
    step = max(1, round(float(n) * (1.0 - float(overlap) / 100.0)))
    z, col_tto = sliding_trial_average(trials, tto, int(n), step)
    z, rm = decimate_rows(z, row_ms, 300)
    z, cbar_title = _erp_transform(z, mode)
    absz = np.abs(z[np.isfinite(z)])
    zmax = float(np.percentile(absz, float(contrast))) if absz.size else 1.0
    zmax = zmax or 1.0
    ncol = z.shape[1]
    x = np.arange(ncol)
    ti = np.linspace(0, ncol - 1, min(9, ncol)).astype(int) if ncol else []
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=rm, colorscale="RdBu", reversescale=True, zmid=0,
        zmin=-zmax, zmax=zmax,
        colorbar=dict(title=dict(text=cbar_title, font=dict(size=9)),
                      thickness=10, len=0.85, x=1.005),
        hovertemplate="%{y:.0f} ms · %{z:.3g}<extra></extra>"))
    if ncol and col_tto.min() <= 0.0 <= col_tto.max():      # mark onset (tto≈0)
        oc = int(np.argmin(np.abs(col_tto)))
        fig.add_vline(x=oc, line=dict(color="#f0f0f5", width=1.5, dash="dash"),
                      annotation_text="onset", annotation_position="top",
                      annotation_font=dict(size=9, color="#f0f0f5"))
    fig.update_layout(
        height=420, margin=dict(l=56, r=20, t=22, b=44),
        xaxis=dict(title="time relative to onset  (− before · + after)",
                   tickvals=[x[k] for k in ti],
                   ticktext=[_fmt_from_onset(col_tto[k]) for k in ti]),
        yaxis=dict(title="post-stim time (ms)"), uirevision="pex-erp")
    return fig


def _fmt_from_onset(tto) -> str:
    """Signed time relative to onset: '−30m' before, 'onset', '+20m' after."""
    tto = float(tto)
    if abs(tto) < 2.0:
        return "onset"
    return ("−" if tto > 0 else "+") + _fmt_dur(abs(tto))


def _erp_wave_figure(gathered, n, overlap, col):
    """(figure, nav-label) for one column's averaged ERP waveform + ±SD band."""
    if not gathered or gathered.get("empty"):
        return empty_fig("Build the ERP-image first"), ""
    trials, row_ms, tto = gathered["trials"], gathered["row_ms"], gathered["tto"]
    step = max(1, round(float(n) * (1.0 - float(overlap) / 100.0)))
    mean, sd, col_tto, ncol, nin = _erp.column_waveform(
        trials, tto, int(n), step, int(col or 0))
    if mean.size == 0:
        return empty_fig("No column to show"), ""
    grand = np.nanmean(trials, axis=0)                   # grand-average waveform
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([row_ms, row_ms[::-1]]),
        y=np.concatenate([mean + sd, (mean - sd)[::-1]]), fill="toself",
        mode="lines", line=dict(width=0), fillcolor="rgba(94,124,226,0.15)",
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=row_ms, y=grand, mode="lines", name="mean of all",
                             line=dict(color=COLOR_TEXT_TERTIARY, width=1,
                                       dash="dash"),
                             hovertemplate="mean %{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=row_ms, y=mean, mode="lines", name="this column",
                             line=dict(color=COLOR_ACCENT, width=2),
                             hovertemplate="%{x:.0f} ms · %{y:.3g}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=COLOR_DIVIDER, width=1))
    lbl = f"{_fmt_from_onset(col_tto)} · avg of {nin} trials · column {int(col or 0) + 1}/{ncol}"
    fig.update_layout(
        height=240, margin=dict(l=56, r=20, t=26, b=40),
        title=dict(text=f"ERP at {lbl}", font=dict(size=11), x=0.02),
        xaxis=dict(title="post-stim time (ms)"), yaxis=dict(title="amplitude"),
        legend=dict(orientation="h", y=1.04, yanchor="bottom", x=1, xanchor="right",
                    font=dict(size=9)), uirevision="pex-erp-wave")
    return fig, lbl


def _erp_ncol_and_onset(gathered, n, overlap):
    """(n_columns, onset_column_index) for the current sliding-average settings —
    cheap (just the per-window median lead-times, no trace math)."""
    tto = gathered.get("tto")
    nt = int(tto.shape[0]) if tto is not None else 0
    step = max(1, round(float(n) * (1.0 - float(overlap) / 100.0)))
    starts = _erp.window_starts(nt, int(n), step)
    if not starts:
        return 0, 0
    ct = np.array([np.nanmedian(tto[s:s + int(n)]) for s in starts])
    return len(starts), int(np.argmin(np.abs(ct)))


def _erp_render(gathered, n, overlap, contrast, key, mode="amplitude"):
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
    return (_erp_figure(gathered, n, overlap, contrast, mode),
            f"✓ built from {nt:,} trials", True, key, warn)


def register_callbacks(app, store, config):
    global _CACHE_DIR
    _CACHE_DIR = _default_cache_dir()
    evoked_dir = config.get("chronic_evoked", {}).get("evoked_output_dir", "")

    @app.callback(
        Output("pex-protocol", "options"),
        Output("pex-protocol", "value"),
        Output("pex-winmode", "value"),
        Output("pex-win-from", "value"),
        Output("pex-win-to", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
    )
    def _on_animal(animal, variant):
        """Scope-only: protocol options + the variant's default feature window.
        The colour-by and trajectory-y options live with their own lenses."""
        opts = _protocol_options(store, animal)
        pval = _default_protocol(store, animal)
        trig = callback_context.triggered_id
        # Reset the feature window to the variant's default when the variant
        # flips (passive needs pre-stim bounds, evoked post-stim).
        if variant == "passive":
            wm, w0, w1 = "custom", *_cfg.DEFAULT_PASSIVE_WINDOW_MS
        else:
            wm, w0, w1 = "full", *_cfg.DEFAULT_EVOKED_WINDOW_MS
        win_reset = trig == "pex-variant"
        return (opts, (pval if trig == "pex-animal" else no_update),
                wm if win_reset else no_update,
                w0 if win_reset else no_update,
                w1 if win_reset else no_update)

    @app.callback(
        Output("pex-colorby", "options"),
        Output("pex-colorby", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
        State("pex-colorby", "value"),
        prevent_initial_call=False,
    )
    def _embed_colorby_opts(_animal, variant, cur):
        """Populate the Embedding lens's colour-by dropdown (on mount + on
        animal/variant change)."""
        copts = _colorby_options(variant or "evoked")
        cvals = {o["value"] for o in copts}
        return copts, (cur if cur in cvals else "time_to_onset_sec")

    @app.callback(
        Output("pex-traj-y", "options"),
        Output("pex-traj-y", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
        State("pex-traj-y", "value"),
        prevent_initial_call=False,
    )
    def _trend_traj_opts(_animal, variant, cur):
        """Populate the Trend lens's trajectory-y dropdown (on mount + on
        animal/variant change)."""
        topts = _traj_y_options(variant or "evoked")
        tvals = {o["value"] for o in topts}
        return topts, (cur if cur in tvals else "peak_to_trough")

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
        Output("pex-status", "children"),
        Output("pex-poll", "disabled"),
        Output("pex-job", "data"),
        Input("pex-build", "n_clicks"),
        Input("pex-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-variant", "value"),
        State("pex-window-h", "value"),
        State("pex-method", "value"),
        State("pex-winmode", "value"),
        State("pex-win-from", "value"),
        State("pex-win-to", "value"),
        State("pex-win-guard", "value"),
        prevent_initial_call=True,
    )
    def _build_or_poll(_n, _iv, animal, protocol, variant, window_h,
                       method, winmode, wf, wt, wg):
        """Compute + cache the matrix/embedding for the current scope; publish the
        job id ONLY when the cache is ready (that pulse drives the per-lens render
        callbacks). Writes no figures itself."""
        if not animal:
            return "Pick an animal.", True, no_update
        try:
            sv, cfg = _resolve_window(variant, winmode or "full", wf, wt, wg)
        except ValueError as e:
            return f"⚠ Feature window: {e}", True, no_update
        window_h = float(window_h or 6.0)
        cap = _cfg.INTERACTIVE_POINT_CAP
        jid = _job_id(animal, protocol or "", variant, window_h, method, cap,
                      _win_token(sv, cfg))
        with _LOCK:
            cached = _CACHE.get(jid)
            state = dict(_JOBS.get(jid) or {})
        if cached is not None:
            return _build_status(cached, jid)
        if state.get("status") == "error":
            return state.get("progress", "error"), True, no_update
        _kick(store, evoked_dir, jid, animal, protocol or "", variant,
              window_h, method, cap, sv, cfg)
        prog = (_JOBS.get(jid) or {}).get("progress", "starting…")
        return f"⏳ {prog}", False, no_update

    # ---- Export / import the built embedding (skip the whole rebuild) ---- #
    def _selection_jid(sel: dict) -> str:
        """Recompute the cache job id from a saved selection, EXACTLY as
        _build_or_poll does, so an imported context lands under the same key a
        later Build would use."""
        variant = sel.get("variant", "evoked")
        try:
            sv, cfg = _resolve_window(variant, sel.get("winmode") or "full",
                                      sel.get("win_from"), sel.get("win_to"),
                                      sel.get("win_guard"))
        except ValueError:
            sv, cfg = "evoked", None
        return _job_id(sel.get("animal"), sel.get("protocol") or "", variant,
                       float(sel.get("window_h") or 6.0),
                       sel.get("method", "pca"), _cfg.INTERACTIVE_POINT_CAP,
                       _win_token(sv, cfg))

    @app.callback(
        Output("pex-export-dl", "data"),
        Output("pex-io-status", "children"),
        Input("pex-export-btn", "n_clicks"),
        State("pex-job", "data"),
        State("pex-animal", "value"), State("pex-protocol", "value"),
        State("pex-variant", "value"), State("pex-window-h", "value"),
        State("pex-method", "value"), State("pex-winmode", "value"),
        State("pex-win-from", "value"), State("pex-win-to", "value"),
        State("pex-win-guard", "value"),
        prevent_initial_call=True,
    )
    def _export_embedding(_n, jid, animal, protocol, variant, window_h, method,
                          winmode, wf, wt, wg):
        cached = _CACHE.get(jid) if jid else None
        if cached is None or cached.get("empty"):
            return no_update, "Build an embedding first, then export."
        selection = {"animal": animal, "protocol": protocol or "",
                     "variant": variant, "window_h": window_h, "method": method,
                     "winmode": winmode, "win_from": wf, "win_to": wt,
                     "win_guard": wg}
        try:
            blob = _eio.dumps(selection, cached)
        except Exception as e:                            # noqa: BLE001
            logger.warning("embedding export failed: %s", e)
            return no_update, f"Export failed: {e}"
        name = _eio.suggested_name(selection)
        return (dcc.send_bytes(lambda buf: buf.write(blob), name),
                f"Exported {name} ({len(blob) // 1024} KB).")

    @app.callback(
        Output("pex-job", "data", allow_duplicate=True),
        Output("pex-io-status", "children", allow_duplicate=True),
        Output("pex-animal", "value", allow_duplicate=True),
        Output("pex-protocol", "value", allow_duplicate=True),
        Output("pex-variant", "value", allow_duplicate=True),
        Output("pex-window-h", "value", allow_duplicate=True),
        Output("pex-method", "value", allow_duplicate=True),
        Input("pex-import-up", "contents"),
        prevent_initial_call=True,
    )
    def _import_embedding(contents):
        if not contents:
            return (no_update,) * 7
        try:
            _hdr, _, b64 = contents.partition(",")
            selection, result = _eio.loads(base64.b64decode(b64))
        except Exception as e:                            # noqa: BLE001
            logger.warning("embedding import failed: %s", e)
            return (no_update, f"Import failed: {e}",
                    no_update, no_update, no_update, no_update, no_update)
        jid = _selection_jid(selection)
        with _LOCK:
            _CACHE[jid] = result
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.popitem(last=False)
            _JOBS[jid] = {"status": "done", "progress": "imported"}
            _prune_jobs(_JOBS)
        n = result.get("n_seizures")
        method = selection.get("method", "pca")
        msg = (f"✓ Loaded {selection.get('animal') or ''} · "
               f"{selection.get('protocol') or 'all'} · {str(method).upper()}"
               + (f" · n = {n} seizures" if n else "") + " (imported).")
        # Setting pex-job pulses every lens to redraw from the restored cache;
        # the dropdowns are set best-effort for a matching UI (the figures do
        # NOT depend on them, so a cascade reset of protocol is only cosmetic).
        return (jid, msg, selection.get("animal"),
                selection.get("protocol") or "", selection.get("variant"),
                selection.get("window_h"), method)

    # ---- Seizure-event include/exclude (persistent per animal) ---- #
    _seiz_hint = {"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION}

    @app.callback(
        Output("pex-seizure-list", "children"),
        Output("pex-seizure-status", "children"),
        Input("pex-animal", "value"),
        Input("pex-protocol", "value"),
    )
    def _render_seizure_keeplist(animal, protocol):
        if not animal:
            return html.Span("Pick an animal.", style=_seiz_hint), ""
        szs = _seizures_in_scope(store, animal, protocol)
        if not szs:
            return html.Span("No scored seizures in this scope.",
                             style=_seiz_hint), ""
        excl = store.excluded_seizure_keys(animal)
        opts, keep = [], []
        for s in szs:
            opts.append({"label": _seizure_label(s), "value": _seizure_key(s)})
            if store.seizure_excl_key(s.file_id, s.eo_sec) not in excl:
                keep.append(_seizure_key(s))
        lst = dcc.Checklist(
            id="pex-seizure-keep", options=opts, value=keep,
            labelStyle={"display": "block", "color": COLOR_TEXT_PRIMARY,
                        "fontSize": FONT_SIZE_CAPTION, "cursor": "pointer"},
            inputStyle={"marginRight": "6px"})
        return lst, _keep_status(len(keep), len(szs) - len(keep), len(szs))

    @app.callback(
        Output("pex-seizure-status", "children", allow_duplicate=True),
        Input("pex-seizure-keep", "value"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        prevent_initial_call=True,
    )
    def _persist_seizure_keep(keep, animal, protocol):
        if not animal:
            return no_update
        keep = set(keep or [])
        szs = _seizures_in_scope(store, animal, protocol)
        scope = {store.seizure_excl_key(s.file_id, s.eo_sec): s for s in szs}
        want_excl = {k for k, s in scope.items() if _seizure_key(s) not in keep}
        cur = store.excluded_seizure_keys(animal)
        # Only touch the diff within THIS scope; other protocols' marks stand.
        for k in want_excl - cur:
            s = scope[k]
            store.set_seizure_excluded(animal, s.file_id, s.eo_sec, True)
        for k in (cur & set(scope)) - want_excl:
            s = scope[k]
            store.set_seizure_excluded(animal, s.file_id, s.eo_sec, False)
        return _keep_status(len(szs) - len(want_excl), len(want_excl), len(szs))

    @app.callback(
        Output("pex-graph", "figure"),
        Output("pex-reading", "children"),
        Output("pex-loadings", "figure"),
        Input("pex-job", "data"),
        State("pex-colorby", "value"),
        State("pex-color-cap", "value"),
        State("pex-color-cap-unit", "value"),
        prevent_initial_call=False,
    )
    def _embed_render(jid, color_by, cap_val, cap_unit):
        """Draw the Embedding lens from cache when the job pulse arrives (or on
        mount, from the persisted job): the scatter, the reading strip, and the
        feature-space loadings."""
        cached = _CACHE.get(jid) if jid else None
        if cached is None:
            return no_update, no_update, no_update
        fig, reading = _embed_fig_reading(
            cached, color_by or "time_to_onset_sec",
            _cap_seconds(cap_val, cap_unit))
        return fig, reading, _loadings_fig(cached)

    @app.callback(
        Output("pex-graph", "figure", allow_duplicate=True),
        Input("pex-colorby", "value"),
        Input("pex-color-cap", "value"),
        Input("pex-color-cap-unit", "value"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _recolor(color_by, cap_val, cap_unit, jid):
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty"):
            return no_update
        return _figure(cached["emb"], cached["sub"], color_by,
                       cached.get("method", "pca"), cached.get("meta"),
                       _cap_seconds(cap_val, cap_unit))

    @app.callback(
        Output("pex-traj", "figure"),
        Output("pex-trend-forest", "figure"),
        Output("pex-trend-verdict", "children"),
        Output("pex-trend-table", "children"),
        Input("pex-job", "data"),
        Input("pex-traj-y", "value"),
        Input("pex-traj-cap", "value"),
        Input("pex-traj-cap-unit", "value"),
        prevent_initial_call=False,
    )
    def _trend_render(jid, traj_y, cap_val, cap_unit):
        """Draw the Trend lens from cache on the job pulse or a control change:
        the lead-time trajectory (linear bins up to the cap) + the per-seizure
        trend test (forest + verdict + all-features FDR table), synchronously."""
        cached = _CACHE.get(jid) if jid else None
        if cached is None:
            return no_update, no_update, no_update, no_update
        feature = traj_y or "peak_to_trough"
        traj = _trajectory_fig(cached, feature, _cap_seconds(cap_val, cap_unit))
        forest, verdict, table = _trend_test_views(cached, feature)
        return traj, forest, verdict, table

    # --- Preictal-vs-interictal lens callbacks --- #
    @app.callback(
        Output("pex-pc-feature", "options"),
        Output("pex-pc-feature", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
        State("pex-pc-feature", "value"),
        prevent_initial_call=False,
    )
    def _pc_feature_opts(_animal, variant, cur):
        opts = _pc_feature_options(variant or "evoked")
        vals = {o["value"] for o in opts}
        # Default to the multivariable model score -- the paper's headline.
        return opts, (cur if cur in vals else _LR_MODEL_KEY)

    @app.callback(
        Output("pex-pc-pdf", "figure"),
        Output("pex-pc-cdf", "figure"),
        Output("pex-pc-verdict", "children"),
        Input("pex-job", "data"),
        Input("pex-pc-feature", "value"),
        Input("pex-pc-nphases", "value"),
        prevent_initial_call=False,
    )
    def _pc_render(jid, feature, nphases):
        cached = _CACHE.get(jid) if jid else None
        if cached is None or cached.get("empty"):
            return no_update, no_update, no_update
        feature = feature or _LR_MODEL_KEY
        lab = _fc.label_classes(cached["full"])
        if feature == _LR_MODEL_KEY:
            # The combined multivariable model output as a metric (Fig 3A):
            # prospective train-P/test-P+1 scores, so it is not overfit.
            try:
                k = max(2, min(20, int(nphases or _cfg.DEFAULT_N_PHASES)))
            except (TypeError, ValueError):
                k = _cfg.DEFAULT_N_PHASES
            lab = lab.assign(lr_model=_fc.prospective_scores(lab, n_phases=k))
            feature = "lr_model"
        pc = _fc.pdf_cdf(lab, feature)
        pp = _fc.permutation_p(lab, feature, n_perm=500)
        ps = _fc.paired_seizure_test(lab, feature)
        return _pdf_fig(pc, feature), _cdf_fig(pc, feature), _pc_verdict(pc, pp, ps)

    @app.callback(
        Output("pex-pc-scan", "children"),
        Input("pex-job", "data"),
        prevent_initial_call=False,
    )
    def _pc_scan(jid):
        cached = _CACHE.get(jid) if jid else None
        if cached is None or cached.get("empty"):
            return no_update
        lab = _fc.label_classes(cached["full"])
        metrics = [m for m in cached.get("metrics", []) if m in lab.columns]
        return _pc_scan_table(_fc.scan_features(lab, metrics, n_perm=500))

    @app.callback(
        Output("pex-pc-roc", "figure"),
        Output("pex-pc-phaseauc", "figure"),
        Output("pex-pc-coef", "figure"),
        Output("pex-pc-fverdict", "children"),
        Input("pex-job", "data"),
        Input("pex-pc-nphases", "value"),
        prevent_initial_call=False,
    )
    def _pc_forecast(jid, nphases):
        cached = _CACHE.get(jid) if jid else None
        if cached is None or cached.get("empty"):
            return no_update, no_update, no_update, no_update
        lab = _fc.label_classes(cached["full"])
        try:
            k = int(nphases or _cfg.DEFAULT_N_PHASES)
        except (TypeError, ValueError):
            k = _cfg.DEFAULT_N_PHASES
        res = _fc.logistic_forecast(lab, n_phases=max(2, min(20, k)))
        n_sz = int(cached.get("n_seizures", 0))
        return (_roc_fig(res), _phaseauc_fig(res), _coef_fig(res),
                _forecast_verdict(res, n_sz))

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
        State("pex-erp-mode", "value"),
        prevent_initial_call=True,
    )
    def _erp_build_or_poll(_n, _iv, animal, protocol, sz, lookback, frm, to,
                           navg, overlap, contrast, mode):
        if not animal or sz is None:
            return no_update, "Pick a seizure.", True, no_update, no_update
        lookback = float(lookback or 60.0)
        key = _erp_key(animal, protocol or "", sz, lookback, frm, to)
        with _LOCK:
            gathered = _ERP_CACHE.get(key)
            state = dict(_ERP_JOBS.get(key) or {})
        if gathered is not None:
            return _erp_render(gathered, navg, overlap, contrast, key, mode)
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
        Input("pex-erp-mode", "value"),
        State("pex-erp-job", "data"),
        prevent_initial_call=True,
    )
    def _erp_rerender(navg, overlap, contrast, mode, key):
        gathered = _ERP_CACHE.get(key) if key else None
        if not gathered or gathered.get("empty"):
            return no_update
        return _erp_figure(gathered, navg, overlap, contrast, mode)

    @app.callback(
        Output("pex-erp-col", "data"),
        Input("pex-erp", "clickData"),
        Input("pex-erp-prev", "n_clicks"),
        Input("pex-erp-next", "n_clicks"),
        Input("pex-erp-job", "data"),
        State("pex-erp-col", "data"),
        State("pex-erp-n", "value"),
        State("pex-erp-overlap", "value"),
        prevent_initial_call=True,
    )
    def _erp_pick_col(click, _p, _nx, job, cur, navg, overlap):
        gathered = _ERP_CACHE.get(job) if job else None
        if not gathered or gathered.get("empty"):
            return no_update
        ncol, onset_col = _erp_ncol_and_onset(gathered, navg or 20, overlap or 50)
        if ncol == 0:
            return no_update
        trig = callback_context.triggered_id
        if trig == "pex-erp-job":                       # new build -> onset column
            return onset_col
        base = int(cur) if cur is not None else onset_col
        if trig == "pex-erp" and click and click.get("points"):
            col = int(round(click["points"][0].get("x", base)))
        elif trig == "pex-erp-prev":
            col = base - 1
        elif trig == "pex-erp-next":
            col = base + 1
        else:
            col = base
        return int(min(max(col, 0), ncol - 1))

    @app.callback(
        Output("pex-erp-wave", "figure"),
        Output("pex-erp-navlabel", "children"),
        Input("pex-erp-col", "data"),
        Input("pex-erp-n", "value"),
        Input("pex-erp-overlap", "value"),
        State("pex-erp-job", "data"),
        prevent_initial_call=True,
    )
    def _erp_wave(col, navg, overlap, job):
        gathered = _ERP_CACHE.get(job) if job else None
        if not gathered or gathered.get("empty"):
            return no_update, no_update
        fig, lbl = _erp_wave_figure(gathered, navg or 20, overlap or 50,
                                    col if col is not None else 0)
        return fig, lbl
