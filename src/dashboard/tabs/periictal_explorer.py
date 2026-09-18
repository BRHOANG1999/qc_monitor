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
import hashlib
import logging
import os
import threading
import time
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
from src.periictal import matrix as _matrix
from src.periictal import forecast as _fc
from src.periictal import palette as _pal
from src.periictal import passive as _passive
from src.periictal import nonstationarity as _nscmod
from src.periictal import stim_map as _sm
from src.periictal.embed import confound_readout, embed
from src.periictal.persist import build_matrix_cached
from src.periictal import erpimage as _erp
from src.periictal import trajectory as _traj
from src.periictal.erpimage import decimate_rows, sliding_trial_average
from src.periictal.selection import summarize_selection
from src.periictal import trendtest as _tt
from src.periictal import slow_report as _slow
from src.periictal import sliding_auc as _swa
from src.periictal import dist_animation as _dan
from src.periictal.trial_series import build_trial_series
from src.preictal.isi import scored_seizures
from src.utils import evoked_features as _ef
from src.utils.evoked_features import FeatureConfig
from src.utils.evoked_output import list_animals
from src.utils import evoked_reader as _er
from src.utils import heavy_admit as _ha
from src.utils import sidecar_warm as _sw

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


def _excl_tok(store, animal) -> str:
    """Short signature of the EXCLUDED-seizure set for *animal*, folded into the
    in-memory cache key (via wintok). Without it, the in-process ``_CACHE``
    short-circuits ``_build_or_poll`` before the disk layer, so toggling a
    seizure's inclusion + Build silently re-serves the STALE matrix that still
    contains the excluded seizure (the disk cache already honours exclusion via
    persist._seizure_sig, but is never consulted on an in-memory hit). Empty set
    -> stable constant, so unchanged exclusions never force a spurious rebuild."""
    try:
        keys = store.excluded_seizure_keys(animal) if animal else set()
    except Exception:                       # noqa: BLE001 -- never break the build
        keys = set()
    if not keys:
        return "|x0"
    digest = hashlib.sha1("|".join(sorted(map(str, keys))).encode()).hexdigest()
    return "|x" + digest[:8]


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


def _run_admitted(target, set_fn, tok, args) -> None:
    """Thread body for every heavy peri-ictal build: hold a process-wide
    heavy-compute slot (``heavy_admit``) for the whole build so N distinct
    selections can't all run at once, surfacing a 'waiting…' status via *set_fn*
    while parked, then run ``target(*args)``. The build workers swallow their own
    exceptions, so the ``with`` always releases the slot."""
    with _ha.admit(tok, progress=lambda m: set_fn(tok, progress=m)):
        target(*args)


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
        target=_run_admitted, name=f"periictal-{job_id}", daemon=True,
        args=(_worker, _set, job_id,
              (store, evoked_dir, job_id, animal, protocol, variant,
               window_h, method, cap, sidecar_variant, feature_cfg)))
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
            # Same near-seizure prefilter as the matrix read: a custom-window
            # build recomputes features from raw traces (~15 s/file), so skipping
            # files that can't contribute a row is the dominant speedup here.
            near = _matrix.near_seizure_filter(store, animal, window_h * 3600.0)
            _passive.warm_variant(
                animal, evoked_dir, sidecar_variant, feature_cfg,
                protocol=protocol or None, file_filter=near,
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
    """Default to ALL protocols ('') so the explorer organizes by SEIZURE across
    every experimental context, not by one protocol folder. Two differently-named
    protocols with equivalent stim are merged downstream by stim FINGERPRINT (the
    seizure label + marker symbol), so scoping to a single folder would wrongly
    hide equivalent seizures. The protocol dropdown stays available as an OPTIONAL
    narrowing filter. (store/animal kept for signature compatibility with callers.)"""
    _ = (store, animal)
    return ""


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
             # Live feature-warmer activity so a re-warm reads as "working, next
             # batch in Ns", never "frozen". Ticks independently of a build.
             html.Div(id="pex-warm-status",
                      style={"color": COLOR_TEXT_TERTIARY,
                             "fontSize": FONT_SIZE_CAPTION, "marginTop": SPACE_2,
                             "fontFamily": "monospace"}),
             style={"marginTop": SPACE_3}),
        dcc.Interval(id="pex-poll", interval=1200, disabled=True),
        dcc.Interval(id="pex-warm-tick", interval=2500),
        # storage_type="session" is how a dcc.Store persists its data across
        # sub-tab swaps (Store has no `persistence` prop) — so a lens can redraw
        # from the module cache on remount using the retained job id.
        dcc.Store(id="pex-job", storage_type="session"),
        dcc.Download(id="pex-export-dl"),
        dcc.Download(id="pex-nb-dl"),
    ])


def _warm_status_line(st: dict) -> str:
    """One live line for the feature warmer -- the reviewer's 'alive vs frozen'
    signal. Warming shows the current file + seconds elapsed; sleeping shows a
    countdown to the next batch; both show the running built count."""
    built, failed = int(st.get("built", 0)), int(st.get("failed", 0))
    tail = (f" · {built} built" + (f", {failed} failed" if failed else "")
            + " this run")
    phase = st.get("phase")
    if phase == "warming":
        started = st.get("started_at")
        ago = int(time.time() - started) if started else 0
        pos = st.get("pos")
        posn = f" [{pos[0]}/{pos[1]}]" if pos else ""
        who = st.get("animal") or ""
        return (f"⟳ warmer: computing {who}{posn} · {st.get('file') or ''} "
                f"— {ago}s{tail}")
    if phase == "idle":
        return f"✓ warmer: corpus up to date{tail}"
    if phase in ("sleeping", "error"):
        nxt = st.get("next_at")
        rem = max(0, int(nxt - time.time())) if nxt else 0
        label = "retry" if phase == "error" else "next batch"
        return f"⏸ warmer: waiting — {label} in {rem}s{tail}"
    return "warmer: starting…"


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
        button("⬇ Export notebook", "pex-nb-btn", variant="secondary",
               **{"title": "Download a self-contained Jupyter notebook + frozen "
                           "data that reproduces THIS lens offline (no DB/share). "
                           "Build first."}),
        html.Span(id="pex-io-status",
                  style={"color": COLOR_TEXT_SECONDARY,
                         "fontSize": FONT_SIZE_CAPTION, "marginLeft": SPACE_3}),
    ], style={"display": "flex", "alignItems": "center", "gap": SPACE_3,
              "flexWrap": "wrap", "marginTop": SPACE_2})


def _embed_column(kind: str, title: str) -> object:
    """One embedding column (*kind* = 'passive' | 'evoked'): the scatter, its
    reading strip, the PC loadings, and a lasso-details pane -- every id suffixed
    by *kind* so the two columns never collide. Dark tokens throughout."""
    return card(
        section_header(title),
        dcc.Loading(
            custom_spinner=loading_icon(f"Building {kind}…"),
            overlay_style={"visibility": "visible", "opacity": 0.4},
            children=dcc.Graph(
                id=f"pex-graph-{kind}", clear_on_unhover=True,
                config={"displaylogo": False},
                figure=empty_fig("Press ▶ Build passive + evoked"))),
        html.Div(id=f"pex-reading-{kind}", style={"marginTop": SPACE_2}),
        section_header("Feature space — what defines the axes"),
        dcc.Graph(id=f"pex-loadings-{kind}", config={"displaylogo": False},
                  figure=empty_fig("Build to see the PC loadings (PCA only)")),
        section_header("Selected points — details on demand"),
        html.Div(id=f"pex-details-{kind}", children=_details_view(None)),
    )


def layout_embedding(store):
    """Embedding lens: PASSIVE (left) and EVOKED (right) embeddings side by side,
    one shared colour-by. A dedicated build runs BOTH windows concurrently (two
    _kick()s into the shared cache); the passive column re-windows features from
    the raw pre-stim trace, so it is the slower of the two to appear."""
    return html.Div([
        scope_bar(store),
        card(
            _embed_colorby_ctl(),
            html.Div([
                button("▶ Build passive + evoked", "pex-embed2-build"),
                html.Div(id="pex-embed2-status",
                         style={"color": COLOR_TEXT_SECONDARY,
                                "fontSize": FONT_SIZE_CAPTION,
                                "fontFamily": "monospace", "minHeight": "16px"}),
            ], style={"display": "flex", "gap": SPACE_3, "alignItems": "center",
                      "flexWrap": "wrap", "margin": f"{SPACE_2} 0"}),
            _explainer(),
            # Responsive grid: passive first (DOM order = left on wide screens,
            # top when it collapses to one column on narrow ones). No horizontal
            # body scroll -- each column min-widths at 380px then wraps.
            html.Div([
                _embed_column("passive", "Passive · pre-stim LFP (−200→−1 ms)"),
                _embed_column("evoked",
                              "Evoked · post-stim response ([1–200 ms] or custom)"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(auto-fit, minmax(380px, 1fr))",
                      "gap": SPACE_4, "marginTop": SPACE_3}),
            dcc.Interval(id="pex-embed2-poll", interval=1200, disabled=True),
            dcc.Store(id="pex-embed2-job", storage_type="session"),
            style={"marginTop": SPACE_4}),
        card(section_header("Features fed to the embedding"),
             _feature_reference(),
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
    """{seizure_idx: 'R{racine} · MM-DD HH:MM · <stim group>'} from the matrix.

    The stim group is the hardware stim FINGERPRINT (charge/pulse-width/freq/gain),
    NOT the protocol folder name, so two differently-named protocols with
    equivalent stim carry the SAME group label -- the point of organizing by
    seizure with a stim-group label rather than by protocol folder."""
    out: dict = {}
    cols = ["seizure_onset_epoch", "seizure_racine"]
    has_stim = "stim_key" in full.columns
    if has_stim:
        cols.append("stim_key")
    g = full.groupby("seizure_idx")[cols].first()
    for sid, row in g.iterrows():
        try:
            when = datetime.fromtimestamp(
                float(row["seizure_onset_epoch"])).strftime("%m-%d %H:%M")
        except (ValueError, OverflowError, OSError):
            when = "?"
        rac = row["seizure_racine"]
        rac = int(rac) if np.isfinite(rac) else "?"
        label = f"R{rac} · {when}"
        if has_stim:
            sk = row["stim_key"]
            if isinstance(sk, str) and sk:
                label += f" · {sk}"
        out[int(sid)] = label
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


# Distinct marker symbols for the stim-group (fingerprint) dimension. Chosen to
# stay visually separable at size 4 and to render on Scattergl.
_STIM_SYMBOLS = ("circle", "triangle-up", "square", "diamond", "cross",
                 "star", "triangle-down", "x", "pentagon", "hexagon")


def _stim_symbol_map(stim_vals) -> dict:
    """Map each stim group (fingerprint) to a distinct marker symbol, in
    first-seen order. {group_str: plotly_symbol}."""
    seen: list = []
    for v in stim_vals:
        s = str(v)
        if s not in seen:
            seen.append(s)
    return {g: _STIM_SYMBOLS[i % len(_STIM_SYMBOLS)]
            for i, g in enumerate(seen)}


def _symbol_legend_traces(sym_map) -> list:
    """Legend-only (off-canvas) traces so the marker SYMBOL -> stim-group mapping
    is readable next to the seizure colour. Neutral grey = 'shape means group'."""
    return [go.Scattergl(
        x=[None], y=[None], mode="markers", name=g, showlegend=True,
        legendgroup="stimgroup", hoverinfo="skip",
        marker=dict(size=8, symbol=sym, color="#c8c8d4", line=dict(width=0)))
        for g, sym in sym_map.items()]


def _figure(emb, sub, color_by, method, meta, cap_sec=None) -> go.Figure:
    if emb is None or emb.shape[0] == 0:
        return empty_fig("No lead-up stimuli for this selection")
    if color_by not in sub:
        # A shared evoked-superset colour-by (e.g. early_area) selected while
        # drawing the PASSIVE column -- that frame has no such feature. Degrade
        # gracefully instead of KeyError-ing inside _continuous_fig.
        return _finish_fig(
            empty_fig(f"‘{color_by}’ is evoked-only — not defined for the "
                      "passive pre-stim window"), method, meta)
    # Encode the stim GROUP (fingerprint) as marker SYMBOL whenever we're not
    # already colouring by it -- so seizure colour + stim-group shape read at
    # once. Only when there's more than one group (else a symbol adds no info).
    sym = None
    if "stim_key" in sub and color_by != "stim_key":
        svals = sub["stim_key"].astype(str).to_numpy()
        smap = _stim_symbol_map(svals)
        if len(smap) > 1:
            sym = {"vals": svals, "map": smap}
    n_unique = int(sub[color_by].nunique())
    fig = (_categorical_fig(emb, sub, color_by, sym)
           if _pal.is_categorical(color_by, n_unique)
           else _continuous_fig(emb, sub, color_by, cap_sec, sym))
    return _finish_fig(fig, method, meta)


def _continuous_fig(emb, sub, color_by, cap_sec=None, sym=None) -> go.Figure:
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
    if sym is not None:                      # stim group -> marker symbol
        marker["symbol"] = [sym["map"][s] for s in sym["vals"]]
    kw = dict(x=emb[:, 0], y=emb[:, 1], mode="markers", marker=marker,
              customdata=np.arange(emb.shape[0]))
    if color_by == "time_to_onset_sec":
        # colour is linear seconds, clamped to the cap; hover shows the TRUE
        # duration (so points beyond the cap still read their real lead time).
        kw["text"] = [_fmt_dur(s) for s in sub["time_to_onset_sec"].to_numpy()]
        kw["hovertemplate"] = "time to onset: %{text}<extra></extra>"
    else:
        kw["hovertemplate"] = f"{spec['label']}: %{{marker.color:.2f}}<extra></extra>"
    fig = go.Figure(go.Scattergl(**kw))
    if sym is not None:
        # Colour rides a COLORBAR here, so the legend is free for the symbol->group
        # key -- one clean legend for stim group, colorbar for seizure.
        for t in _symbol_legend_traces(sym["map"]):
            fig.add_trace(t)
        fig.update_layout(showlegend=True,
                          legend=dict(title="stim group (marker)", orientation="h",
                                      y=1.02, yanchor="bottom", font=dict(size=9),
                                      itemsizing="constant"))
    return fig


def _fmt_dur(s) -> str:
    """Human-readable duration for hover (the colour axis is linear seconds
    clamped to the cap; hover shows the TRUE duration)."""
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _categorical_fig(emb, sub, color_by, sym=None) -> go.Figure:
    raw = sub[color_by].astype(str).to_numpy()
    order, disp = _pal.fold_categories(raw)
    shown = np.array([disp[x] for x in raw])
    row_idx = np.arange(emb.shape[0])
    fig = go.Figure()
    for k, name in enumerate(order):
        m = shown == name
        marker = dict(size=4, opacity=0.65, color=_pal.hue_for(name, k))
        if sym is not None:                  # stim group -> marker symbol
            marker["symbol"] = [sym["map"][s] for s in sym["vals"][m]]
        fig.add_trace(go.Scattergl(
            x=emb[m, 0], y=emb[m, 1], mode="markers", name=name, showlegend=True,
            legendgroup="color", marker=marker,
            customdata=row_idx[m],
            hovertemplate=f"{_pretty(color_by)}: {name}<extra></extra>"))
    if sym is not None:                      # symbol->group key alongside the colours
        for t in _symbol_legend_traces(sym["map"]):
            fig.add_trace(t)
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
                   window_label="", tab="") -> html.Div:
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
    # "embed" is only accurate on the Embedding lens; every other lens just needs
    # the shared matrix built (the embedding radio is irrelevant there).
    action = ("embed their lead-up stimuli" if tab == "periictal_embedding"
              else "prepare the matrix, then Run this lens")
    return html.Div([
        html.Span("Ready — ", style={"color": COLOR_SUCCESS, "fontWeight": "600",
                                     "fontSize": FONT_SIZE_BODY}),
        html.Span(f"{animal} · {proto} · {variant} features{win} · "
                  f"{window_h:g} h lead-up",
                  style={"color": COLOR_TEXT_PRIMARY, "fontSize": FONT_SIZE_BODY}),
        html.Span(f"    Effective n = {n} seizures. Press ▶ Build to {action}.",
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
        target=_run_admitted, name=f"erp-{key}", daemon=True,
        args=(_erp_worker, _erp_set, key,
              (store, evoked_dir, key, animal, protocol, sz, lookback, frm, to))
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
        # The lead-up gather reads many ~600 MB evoked files (~15 s each) and is
        # the ONLY peri-ictal view that reads raw traces -- run the WHOLE gather
        # in a child process (off the dashboard GIL) so it can't freeze a second
        # user. Only the final windowed trials return; per-file progress can't
        # cross the process boundary, so surface one coarse status instead.
        _erp_set(key, progress="reading lead-up traces…")
        res = _er.gather_leadup_offproc(
            evoked_dir, animal, s.onset_epoch, float(lookback) * 60.0, cfg=cfg)
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
    # Overlay this column's constituent trials (the ones averaged into `mean`),
    # thin + transparent, under the band/mean so the spread is visible.
    starts = _erp.window_starts(np.asarray(trials).shape[0], int(n), step)
    if starts:
        c = min(max(int(col or 0), 0), len(starts) - 1)
        seg = np.asarray(trials, dtype=float)[starts[c]:starts[c] + int(n)]
        stp = max(1, seg.shape[0] // 40)                 # cap the overlaid count
        for tr in seg[::stp]:
            fig.add_trace(go.Scattergl(
                x=row_ms, y=tr, mode="lines", opacity=0.12,
                line=dict(color=COLOR_ACCENT, width=0.5),
                hoverinfo="skip", showlegend=False))
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


# --------------------------------------------------------------------- #
#  Slow-dynamics lens (Part B): across-trial eigenvalue + circadian control
# --------------------------------------------------------------------- #

_SD_JOBS: dict = {}
_SD_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SD_CACHE_MAX = 6


def _sd_key(animal, feature, win) -> str:
    return f"{animal}|{feature}|{int(win)}"


def _sd_set(key, **kw):
    with _LOCK:
        _SD_JOBS.setdefault(key, {}).update(kw)
        _prune_jobs(_SD_JOBS)


def _sd_finish(key, result):
    with _LOCK:
        _SD_CACHE[key] = result
        while len(_SD_CACHE) > _SD_CACHE_MAX:
            _SD_CACHE.popitem(last=False)
        _SD_JOBS[key] = {"status": "done", "progress": "done"}
        _prune_jobs(_SD_JOBS)


def _sd_kick(store, evoked_dir, key, animal, feature, win):
    with _LOCK:
        if key in _SD_CACHE:
            return
        st = _SD_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _SD_JOBS[key] = {"status": "running", "progress": "starting…"}
        _prune_jobs(_SD_JOBS)
    threading.Thread(
        target=_run_admitted, name=f"periictal-sd-{key}", daemon=True,
        args=(_sd_worker, _sd_set, key,
              (store, evoked_dir, key, animal, feature, win))).start()


def _sd_worker(store, evoked_dir, key, animal, feature, win):
    """Build the full-timeline feature series (heavy: reads every sidecar) and run
    the Part B slow-dynamics summary. Never raises."""
    try:
        _sd_set(key, progress="reading the full feature timeline…")
        series = build_trial_series(store, animal, feature, evoked_dir)
        if series.n == 0:
            _sd_finish(key, {"summary": {"insufficient": True, "n": 0}})
            return
        _sd_set(key, progress=f"fitting slow dynamics on {series.n:,} stimuli…")
        _sd_finish(key, {"summary": _slow.summarize(series, win=int(win))})
    except Exception as e:                                    # noqa: BLE001
        logger.exception("slow-dynamics build failed (key=%s): %s", key, e)
        _sd_set(key, status="error", progress=f"error: {e}")


def _sd_dark(fig):
    fig.update_layout(paper_bgcolor=COLOR_SURFACE_2, plot_bgcolor=COLOR_SURFACE_2,
                      font_color=COLOR_TEXT_SECONDARY, margin=dict(l=48, r=16, t=28, b=40),
                      height=280)
    fig.update_xaxes(gridcolor=COLOR_DIVIDER, zeroline=False)
    fig.update_yaxes(gridcolor=COLOR_DIVIDER, zeroline=False)
    return fig


def _sd_phi_fig(s):
    """AR(1) phi vs lead time (near-onset on the right). Rising toward onset ->
    the state is slowing (eigenvalue -> 0)."""
    if not s or s.get("insufficient"):
        return empty_fig("Not enough trials for a slow-dynamics fit")
    lead = s["phi_leadtime"]
    fig = go.Figure(go.Scatter(x=lead["centers_h"], y=lead["phi"],
                               mode="lines+markers",
                               line=dict(color=COLOR_ACCENT, width=2)))
    fig.update_xaxes(autorange="reversed", title="hours before onset")
    fig.update_yaxes(title="AR(1) φ (detrended)")
    return _sd_dark(fig)


def _sd_circ_fig(s):
    """Preictal AUC: standard same-day baseline vs the circadian-matched control.
    Collapse toward 0.5 under matching => the effect was circadian."""
    if not s or s.get("insufficient"):
        return empty_fig("Circadian control needs non-seizure-day stimuli")
    y = [s.get("sameday_auc", float("nan")), s.get("matched_auc", float("nan"))]
    fig = go.Figure(go.Bar(x=["same-day (60–90 min)", "circadian-matched"], y=y,
                           marker_color=[COLOR_WARNING, COLOR_SUCCESS]))
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLOR_TEXT_TERTIARY)
    fig.update_yaxes(title="preictal AUC", range=[0, 1])
    return _sd_dark(fig)


def _sd_line(label, value, note=""):
    return html.Div([
        html.Span(f"{label}: ", style={"color": COLOR_TEXT_TERTIARY}),
        html.Span(value, style={"color": COLOR_TEXT_PRIMARY, "fontWeight": 600}),
        html.Span(f"  {note}" if note else "",
                  style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION})],
        style={"marginBottom": SPACE_1})


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}" if np.isfinite(float(v)) else "—"
    except (TypeError, ValueError):
        return "—"


def _sd_readout_view(s):
    if not s or s.get("insufficient"):
        return html.Div("Build to compute the across-trial eigenvalue.",
                        style={"color": COLOR_TEXT_TERTIARY})
    rho = s.get("coupling_rho")
    csd = ("consistent with critical slowing (φ & variance rise together)"
           if (rho is not None and np.isfinite(rho) and rho > 0.4)
           else "variance/φ decoupled → injected noise, not slowing"
           if (rho is not None and np.isfinite(rho)) else "n/a")
    return html.Div([
        _sd_line("Eigenvalue φ near→far onset",
                 f"{_fmt(s['phi_near'])} → {_fmt(s['phi_far'])}",
                 "(→1 = slowing; reported as φ/λ, never τ)"),
        _sd_line("λ = ln(φ)/Δt (near, far)",
                 f"{_fmt(s['lambda_near'], 5)}, {_fmt(s['lambda_far'], 5)} /s"),
        _sd_line("variance–φ coupling ρ", _fmt(rho), csd),
        _sd_line("infraslow band fraction (0.001–0.01 Hz)", _fmt(s.get("band_frac"))),
        _sd_line("preictal AUC: same-day → matched",
                 f"{_fmt(s.get('sameday_auc'))} → {_fmt(s.get('matched_auc'))}",
                 "(collapse toward 0.5 = circadian artefact)"),
        html.Div(f"n = {s.get('n_seizures', 0)} seizures · "
                 f"{s.get('n', 0):,} stimuli · channel {s.get('channel') or '?'} · "
                 "power scales with SEIZURE count, not trials — treat as elimination, "
                 "not demonstration.",
                 style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
                        "marginTop": SPACE_2, "fontStyle": "italic"}),
    ])


def _sd_controls(a0):
    return html.Div([
        html.Div([
            html.Label("Feature", style=LABEL_STYLE),
            dcc.Dropdown(id="pex-sd-feature",
                         options=[{"label": m, "value": m} for m in _cfg.CHEAP_METRICS],
                         value=_cfg.PAPER_BEST5[0], clearable=False,
                         style=DROPDOWN_STYLE)],
            style={"flex": "2", "minWidth": "220px"}),
        html.Div([
            html.Label("φ window (trials)", style=LABEL_STYLE),
            dcc.Input(id="pex-sd-win", type="number", value=300, min=30, step=10,
                      style={"width": "100%"})],
            style={"flex": "1", "minWidth": "120px"}),
        html.Div(button("Analyse", "pex-sd-build"),
                 style={"alignSelf": "flex-end"}),
    ], style={"display": "flex", "gap": SPACE_3, "alignItems": "flex-start",
              "flexWrap": "wrap", "marginBottom": SPACE_3})


def layout_slow_dynamics(store):
    """Slow-dynamics lens (Part B): the across-trial eigenvalue φ/λ vs lead time,
    the variance/φ decoupling verdict, infraslow band power, and the
    circadian-matched preictal AUC — the confound-robust readouts the evoked
    features (within-waveform, ms-scale) cannot give."""
    _animals, a0 = _animals_a0(store)
    return html.Div([
        scope_bar(store),
        card(
            section_header("Slow dynamics — across-trial eigenvalue & circadian control"),
            html.Div("The evoked features are within-waveform (ms-scale) and cannot "
                     "see a minute-scale mode; the slow eigenvalue lives in the "
                     "trial-to-trial series. AR(1) φ = exp(λΔt).",
                     style={"color": COLOR_TEXT_TERTIARY, "fontSize": FONT_SIZE_CAPTION,
                            "marginBottom": SPACE_2}),
            _sd_controls(a0),
            html.Div(id="pex-sd-status", style={"color": COLOR_TEXT_SECONDARY,
                                                "fontSize": FONT_SIZE_CAPTION,
                                                "minHeight": "16px"}),
            html.Div(id="pex-sd-readout", style={"marginTop": SPACE_2}),
            dcc.Graph(id="pex-sd-phi", figure=empty_fig("φ vs lead time")),
            dcc.Graph(id="pex-sd-circ", figure=empty_fig("circadian-matched AUC")),
            style={"marginTop": SPACE_3}),
        dcc.Interval(id="pex-sd-poll", interval=1500, disabled=True),
        dcc.Store(id="pex-sd-job"),
    ], style={"padding": SPACE_4})


# --------------------------------------------------------------------- #
#  Sliding-window ROC-AUC lens (src.periictal.sliding_auc)
# --------------------------------------------------------------------- #
_SWAUC_JOBS: dict = {}
_SWAUC_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SWAUC_CACHE_MAX = 4

# Distribution-animation (GIF) jobs. Result = {"mode","status","progress", and
# on success "src" (data-URI for the inline preview) OR "zip" (bytes) + "name"}.
_GIF_JOBS: dict = {}
_GIF_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_GIF_CACHE_MAX = 6


def _swauc_key(jid, nwin, band_lo_h, band_hi_h) -> str:
    return f"{jid}|swauc|{nwin}|{band_lo_h}|{band_hi_h}"


def _swauc_set(key, **kw):
    with _LOCK:
        _SWAUC_JOBS.setdefault(key, {}).update(kw)
        _prune_jobs(_SWAUC_JOBS)


def _swauc_finish(key, result):
    with _LOCK:
        _SWAUC_CACHE[key] = result
        while len(_SWAUC_CACHE) > _SWAUC_CACHE_MAX:
            _SWAUC_CACHE.popitem(last=False)
        _SWAUC_JOBS[key] = {"status": "done", "progress": "done"}
        _prune_jobs(_SWAUC_JOBS)


def _swauc_kick(store, key, jid, nwin, band_lo_sec, band_hi_sec):
    with _LOCK:
        if key in _SWAUC_CACHE:
            return
        st = _SWAUC_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _SWAUC_JOBS[key] = {"status": "running", "progress": "starting…"}
        _prune_jobs(_SWAUC_JOBS)
    threading.Thread(
        target=_run_admitted, name=f"swauc-{key}", daemon=True,
        args=(_swauc_worker, _swauc_set, key,
              (key, jid, nwin, band_lo_sec, band_hi_sec))).start()


def _swauc_group_matrix(result) -> np.ndarray:
    """Mean-across-seizures AUC matrix (feature x window); NaN where no seizure
    scored a cell."""
    import warnings
    mats = [ps["auc"] for ps in result["per_seizure"].values()]
    n_feat, n_win = len(result["features"]), result["offsets"].size
    if not mats:
        return np.full((n_feat, n_win), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        return np.nanmean(np.stack(mats), axis=0)


def _swauc_worker(key, jid, nwin, band_lo_sec, band_hi_sec):
    """Score the sliding-window AUC off the request thread. Never raises."""
    try:
        with _LOCK:
            cached = _CACHE.get(jid)
        if cached is None or cached.get("empty"):
            _swauc_finish(key, {"empty": True,
                                "reason": "Build the matrix first (scope bar)."})
            return
        full = cached["full"]
        feats = list(cached.get("metrics") or [])
        _swauc_set(key, progress="scoring windows…")
        result = _swa.sliding_window_auc(
            full, feats, n_windows=int(nwin),
            band_lo=float(band_lo_sec), band_hi=float(band_hi_sec))
        result["group_auc"] = _swauc_group_matrix(result)
        _swauc_set(key, progress="building distributions…")
        cls = _swa.sliding_class_column(full, n_windows=int(nwin),
                                        band_lo=float(band_lo_sec),
                                        band_hi=float(band_hi_sec))
        lab = full.assign(**{"class": cls})
        result["pc_by_feature"] = {f: _fc.pdf_cdf(lab, f)
                                   for f in result["features"]}
        result["sz_labels"] = _seizure_labels(full)
        result["empty"] = result["n_seizures_used"] == 0
        _swauc_finish(key, result)
    except Exception as e:                                    # noqa: BLE001
        logger.exception("sliding-AUC build failed (key=%s): %s", key, e)
        _swauc_set(key, status="error", progress=f"error: {e}")


# --- Distribution-animation (GIF) jobs --------------------------------- #

def _gif_key(jid, sel, feature, lookback_h, mode) -> str:
    return f"{jid}|gif|{sel}|{feature}|{lookback_h}|{mode}"


def _gif_set(key, **kw):
    with _LOCK:
        _GIF_JOBS.setdefault(key, {}).update(kw)
        _prune_jobs(_GIF_JOBS)


def _gif_finish(key, result):
    with _LOCK:
        _GIF_CACHE[key] = result
        while len(_GIF_CACHE) > _GIF_CACHE_MAX:
            _GIF_CACHE.popitem(last=False)
        _GIF_JOBS[key] = {"status": "done", "progress": "done"}
        _prune_jobs(_GIF_JOBS)


def _gif_seizure_ids(full) -> list:
    """Seizure indices with >= 1 pre-onset stimulus, oldest first (bounded)."""
    pre = full[full["phase"].to_numpy() == "pre"]
    return sorted({int(s) for s in pre["seizure_idx"].to_numpy().tolist()})[:500]


def _w1trend_fig(full, sel, feature, lookback_sec):
    """Wasserstein-1 step size (frame -> frame) + displacement (vs the far
    baseline) vs time-before-onset, for the selected Seizure + Feature. The
    prototype answer to "does W1 trend meaningfully from one distribution to the
    next as onset approaches?" Synchronous -- W1 over ~23 frames is trivial."""
    if not feature:
        return empty_fig("Pick a Feature to see the Wasserstein step size.")
    if full is None or feature not in full.columns:
        return empty_fig(f"'{feature}' is not in the built matrix.")
    sid = "group" if (sel in (None, "group", "")) else int(sel)
    try:
        d = _dan.frame_step_distances(full, sid, feature, lookback=lookback_sec)
    except Exception as e:                                   # noqa: BLE001
        return empty_fig(f"Couldn't compute W₁: {e}")
    h = d["hours_before"]
    if h.size == 0 or not np.any(np.isfinite(d["step_w1"])):
        return empty_fig("Not enough stimuli per frame for a W₁ trend",
                         hint="try the group, a wider band, or a busier seizure")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h, y=d["step_w1"], mode="lines+markers", connectgaps=False,
        name="step W₁ (frame → frame)",
        line=dict(color=COLOR_ACCENT, width=2), marker=dict(size=6),
        customdata=[[lab, int(n)] for lab, n in zip(d["labels"], d["n"])],
        hovertemplate="%{customdata[0]}<br>step W₁=%{y:.3g} "
                      "(n=%{customdata[1]})<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=h, y=d["disp_w1"], mode="lines", connectgaps=False,
        name="displacement W₁ (vs far baseline)",
        line=dict(color="#e26e6e", width=1.5, dash="dot"),
        hovertemplate="displacement W₁=%{y:.3g}<extra></extra>"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=56, r=20, t=34, b=46),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)),
        hovermode="x unified")
    fig.update_xaxes(autorange="reversed", title_text="hours before onset",
                     gridcolor=COLOR_DIVIDER, zeroline=False)
    fig.update_yaxes(title_text=f"Wasserstein-1  ({feature})",
                     gridcolor=COLOR_DIVIDER, rangemode="tozero")
    return fig


def _gif_kick(jid, key, sel, feature, lookback_sec, mode):
    with _LOCK:
        if key in _GIF_CACHE:
            return
        st = _GIF_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _GIF_JOBS[key] = {"status": "running", "progress": "starting…"}
        _prune_jobs(_GIF_JOBS)
    # Heavy matplotlib render (many frames x maybe many seizures) -> hold a
    # process-wide compute slot, same gate as the matrix/ERP/AUC builds.
    threading.Thread(
        target=_run_admitted, name=f"gif-{key}", daemon=True,
        args=(_gif_worker, _gif_set, key,
              (jid, key, sel, feature, lookback_sec, mode))).start()


def _gif_worker(jid, key, sel, feature, lookback_sec, mode):
    """Render the distribution GIF(s) off the request thread. Never raises."""
    try:
        cached = _CACHE.get(jid)
        if not cached or cached.get("empty") or "full" not in cached:
            _gif_finish(key, {"empty": True,
                              "reason": "Build the matrix first (scope bar)."})
            return
        if not feature:
            _gif_finish(key, {"empty": True, "reason": "Pick a Feature first."})
            return
        full = cached["full"]
        labels = _seizure_labels(full)
        if mode == "zip":
            sids = _gif_seizure_ids(full)
            if not sids:
                _gif_finish(key, {"empty": True, "reason": "No seizures to animate."})
                return
            blob = _dan.render_all_seizures_zip(
                full, feature, sids, labels, lookback=lookback_sec,
                progress=lambda i, n, lab: _gif_set(
                    key, progress=f"rendering seizure {i + 1}/{n}: {lab}…"))
            _gif_finish(key, {"empty": False, "mode": "zip", "zip": blob,
                              "name": f"{_dan.safe_name(feature)}_dist_gifs.zip"})
            return
        sid = "group" if (sel in (None, "group", "")) else int(sel)
        _gif_set(key, progress="rendering frames…")
        gif = _dan.render_seizure_gif(full, sid, feature,
                                      label=labels.get(sid, ""),
                                      lookback=lookback_sec)
        b64 = base64.b64encode(gif).decode("ascii")
        _gif_finish(key, {"empty": False, "mode": "one",
                          "src": f"data:image/gif;base64,{b64}"})
    except ValueError as e:                                   # no usable stimuli
        _gif_finish(key, {"empty": True, "reason": str(e)})
    except Exception as e:                                    # noqa: BLE001
        logger.exception("distribution-GIF build failed (key=%s): %s", key, e)
        _gif_set(key, status="error", progress=f"error: {e}")


def _swauc_feat(result, feature):
    """Selected PDF/CDF feature, defaulting to the group's #1 feature."""
    if feature in result.get("features", []):
        return feature
    top = result.get("group_top5") or result.get("features") or [None]
    return top[0]


def _swauc_seizure_opts(result) -> list:
    opts = [{"label": "All seizures (group)", "value": "group"}]
    labels = result.get("sz_labels", {})
    for s in sorted(result.get("per_seizure", {})):
        opts.append({"label": labels.get(s, f"seizure {s}"), "value": str(s)})
    return opts


def _swauc_notes(result) -> object:
    n = result.get("n_seizures_used", 0)
    notes = result.get("notes", [])
    colour = COLOR_SUCCESS if n else COLOR_WARNING
    head = (f"{n} seizure(s) scored · group top-5: "
            f"{', '.join(result.get('group_top5', [])) or '—'}")
    body = [html.Div(head, style={"color": COLOR_TEXT_PRIMARY,
                                  "fontSize": FONT_SIZE_CAPTION})]
    for nt in notes[:8]:
        body.append(html.Div("· " + nt, style={"color": COLOR_TEXT_TERTIARY,
                                               "fontSize": FONT_SIZE_CAPTION}))
    return html.Div(body, style={"borderLeft": f"3px solid {colour}",
                                 "padding": f"6px {SPACE_3}",
                                 "background": COLOR_SURFACE_2,
                                 "borderRadius": RADIUS_SM})


def _swauc_bar(result, sel) -> go.Figure:
    """Ranked mean-AUC horizontal bar: group (mean across seizures) or one
    seizure; group top-5 highlighted."""
    if sel == "group":
        score, ranked, top = (result["group"], result["group_ranked"],
                              set(result["group_top5"]))
        title = "Mean AUC across seizures"
    else:
        ps = result["per_seizure"].get(int(sel))
        if not ps:
            return empty_fig("No scorable windows for this seizure")
        score, ranked, top = ps["mean_auc"], ps["ranked"], set(ps["top5"])
        title = f"Mean AUC · seizure {sel}"
    show = [f for f in ranked if np.isfinite(score[f])][:12][::-1]
    colours = [COLOR_ACCENT if f in top else COLOR_DIVIDER for f in show]
    fig = go.Figure(go.Bar(
        x=[score[f] for f in show], y=show, orientation="h", marker_color=colours,
        hovertemplate="%{y}: AUC %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=COLOR_TEXT_TERTIARY, dash="dash", width=1))
    fig.update_layout(**_pc_layout(title, "mean AUC (≥.5)", height=390),
                      xaxis_range=[0.5, 1.0], yaxis_title="")
    return fig


def _swauc_heatmap(result, sel) -> go.Figure:
    feats, ranked, labels = (result["features"], result["group_ranked"],
                             result["win_labels"])
    if sel == "group":
        M, title = result["group_auc"], "AUC by feature × window (group mean)"
    else:
        ps = result["per_seizure"].get(int(sel))
        if not ps:
            return empty_fig("No scorable windows for this seizure")
        M, title = ps["auc"], f"AUC by feature × window · seizure {sel}"
    order = [feats.index(f) for f in ranked][::-1]        # best feature at top
    z = np.asarray(M)[order]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=[ranked[::-1][i] for i in range(len(order))],
        colorscale="Viridis", zmin=0.5, zmax=1.0,
        colorbar=dict(title=dict(text="AUC", font=dict(size=9)), thickness=10,
                      len=0.85, x=1.005),
        hovertemplate="%{y} · %{x} before onset · AUC %{z:.3f}<extra></extra>"))
    fig.update_layout(**_pc_layout(title, "interictal window (before onset)",
                                   height=530), yaxis_title="")
    return fig


def _swauc_best_bar(result) -> go.Figure:
    best, labels = result["best_per_seizure"], result.get("sz_labels", {})
    xs = [labels.get(b["seizure_idx"], f"sz {b['seizure_idx']}") for b in best]
    ys = [b["best_auc"] for b in best]
    # The PEAK is a concrete (feature, window) pair -- show BOTH the feature and
    # which interictal window (lead time before onset) it was, so "best AUC" is
    # never an unattributed average over windows.
    txt = [f"{(b['best_feature'] or '—')}<br>@{b.get('best_window') or '?'}"
           for b in best]
    cd = [[b.get("best_window") or "?", b["best_feature"] or "—"] for b in best]
    fig = go.Figure(go.Bar(
        x=xs, y=ys, text=txt, textposition="outside", customdata=cd,
        textfont=dict(size=9), marker_color=COLOR_ACCENT,
        hovertemplate="%{x}<br>peak AUC %{y:.3f}<br>%{customdata[1]} @ "
                      "%{customdata[0]} before onset<br><i>click to see the "
                      "window distribution</i><extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=COLOR_TEXT_TERTIARY, dash="dash", width=1))
    fig.update_layout(
        **_pc_layout("Peak AUC per seizure — best group top-5 feature @ its window",
                     "seizure (click a bar for its window distribution)",
                     height=390),
        yaxis_title="peak AUC", yaxis_range=[0.4, 1.0])
    return fig


def _swauc_render(result, sel, feature):
    pc = result.get("pc_by_feature", {}).get(feature)
    pdf = _pdf_fig(pc, feature) if pc else empty_fig("No distribution")
    cdf = _cdf_fig(pc, feature) if pc else empty_fig("No distribution")
    return (_swauc_bar(result, sel), _swauc_heatmap(result, sel),
            _swauc_best_bar(result), pdf, cdf, _swauc_notes(result))


def _swauc_dist_fig(result, clicked_sid, feature) -> go.Figure:
    """Window-AUC trajectory: x = interictal-window lead time before onset,
    y = that window's rank AUC for *feature*. One colour-coded line per seizure
    (the clicked one foregrounded) + a bold GROUP median with an IQR band, so you
    can read WHERE in the pre-onset run each seizure -- and the cohort -- separates
    best. Windows dropped by the post-ictal guard leave gaps."""
    feats = result.get("features", [])
    if feature not in feats:
        feature = (result.get("group_top5") or feats or [None])[0]
    if feature is None:
        return empty_fig("No feature to plot")
    fi = feats.index(feature)
    offsets = np.asarray(result["offsets"], dtype=float)
    x_h = offsets / 3600.0
    sids = sorted(result.get("per_seizure", {}))
    if not sids:
        return empty_fig("No seizures scored yet")
    labels = result.get("sz_labels", {})
    fig = go.Figure()
    rows = []
    for k, s in enumerate(sids):
        row = np.asarray(result["per_seizure"][s]["auc"])[fi]
        rows.append(row)
        if not np.isfinite(row).any():
            continue
        col = _pal.hue_for(labels.get(s, str(s)), k)
        clicked = clicked_sid is not None and int(s) == int(clicked_sid)
        fig.add_trace(go.Scatter(
            x=x_h, y=row, mode="lines+markers", name=labels.get(s, f"sz {s}"),
            connectgaps=False, opacity=1.0 if clicked else 0.4,
            line=dict(color=col, width=2.8 if clicked else 1.0),
            marker=dict(color=col, size=8 if clicked else 4),
            hovertemplate="%{x:.1f} h before onset · AUC %{y:.3f}<extra>"
                          + labels.get(s, f"sz {s}") + "</extra>"))
    # Group median + IQR band across seizures at each window.
    M = np.vstack(rows) if rows else np.empty((0, offsets.size))
    if M.size:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cols
            med = np.nanmedian(M, axis=0)
            q1 = np.nanpercentile(M, 25, axis=0)
            q3 = np.nanpercentile(M, 75, axis=0)
        ok = np.isfinite(med)
        if ok.any():
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_h[ok], x_h[ok][::-1]]),
                y=np.concatenate([q3[ok], q1[ok][::-1]]),
                fill="toself", mode="lines", line=dict(width=0),
                fillcolor="rgba(216,216,224,0.13)", hoverinfo="skip",
                name="group IQR", showlegend=False))
            fig.add_trace(go.Scatter(
                x=x_h[ok], y=med[ok], mode="lines+markers", name="group median",
                line=dict(color="#e2e2ea", width=3), marker=dict(size=6),
                hovertemplate="%{x:.1f} h · median AUC %{y:.3f}<extra>group</extra>"))
    fig.add_hline(y=0.5, line=dict(color=COLOR_TEXT_TERTIARY, dash="dash", width=1))
    fig.update_layout(
        **_pc_layout(f"Window AUC vs lead time · {feature}",
                     "time before onset (h)", height=460),
        yaxis_title="AUC (auc_norm)", yaxis_range=[0.45, 1.0],
        xaxis_autorange="reversed")            # onset-approaching to the right
    return fig


_SWAUC_MODAL_SHOWN = {
    "display": "flex", "position": "fixed", "top": "0", "left": "0",
    "right": "0", "bottom": "0", "background": "rgba(10,10,16,0.62)",
    "alignItems": "center", "justifyContent": "center", "zIndex": "2000",
    "padding": SPACE_4,
}


def layout_slidingauc(store):
    """Sliding-window ROC-AUC lens: the fixed 30-min preictal window vs N sampled
    30-min interictal windows across a 1-6 h band, scored per feature per seizure
    then aggregated -- surfacing per-seizure and consensus top features."""
    inp = {**DROPDOWN_STYLE, "width": "90px"}
    return html.Div([
        scope_bar(store),
        card(section_header("Sliding-window ROC-AUC — preictal vs sampled interictal"),
             html.Div([
                 html.Div("The preictal 30-min window (positive) is held fixed; the "
                          "interictal reference is sampled as N evenly-spaced 30-min "
                          "windows across a 1-6 h lookback band before onset (windows "
                          "within 1 h of the previous seizure are dropped). Each "
                          "feature's preictal-vs-window rank AUC is scored per seizure, "
                          "then averaged across seizures (the seizure is the unit). "
                          "Build the matrix in the scope bar first, then Run.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 html.Div([
                     _ctl("Windows / seizure", dcc.Input(
                         id="pex-swauc-nwin", type="number",
                         value=_cfg.SLIDING_N_WINDOWS, min=2, max=40, step=1,
                         debounce=True, style=inp),
                         "How many interictal windows to sample per seizure."),
                     _ctl("Band from (h)", dcc.Input(
                         id="pex-swauc-bandlo", type="number", value=1, min=0.5,
                         max=12, step=0.5, debounce=True, style=inp),
                         "Closest the windows get to onset."),
                     _ctl("Band to (h)", dcc.Input(
                         id="pex-swauc-bandhi", type="number", value=6, min=1,
                         max=24, step=0.5, debounce=True, style=inp),
                         "Furthest before onset the windows reach."),
                     _ctl("Seizure", dcc.Dropdown(
                         id="pex-swauc-seizure",
                         options=[{"label": "All seizures (group)",
                                   "value": "group"}], value="group",
                         clearable=False,
                         style={**DROPDOWN_STYLE, "minWidth": "230px"}),
                         "Group aggregate, or drill into one seizure."),
                     _ctl("Feature (PDF/CDF)", dcc.Dropdown(
                         id="pex-swauc-feature", clearable=False,
                         style={**DROPDOWN_STYLE, "minWidth": "200px"}),
                         "Which feature's distribution to show at the bottom."),
                     html.Div(button("▶ Run sliding-window AUC", "pex-swauc-build",
                                     icon_name="play"),
                              style={"alignSelf": "flex-end"}),
                 ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
                           "alignItems": "flex-start", "marginTop": SPACE_3}),
                 html.Div(id="pex-swauc-status",
                          style={"marginTop": SPACE_2, "color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION}),
                 html.Div(id="pex-swauc-notes", style={"marginTop": SPACE_2}),
             ], style={"display": "flex", "flexDirection": "column",
                       "gap": SPACE_2, "marginTop": SPACE_2}),
             style={"marginTop": SPACE_4}),
        card(section_header("Top features & AUC across windows"),
             html.Div([
                 dcc.Graph(id="pex-swauc-groupbar", config={"displaylogo": False},
                           figure=empty_fig("Run to see ranked features"),
                           style={"flex": "1 1 380px", "minWidth": "0",
                                  "height": "400px"}),
                 dcc.Graph(id="pex-swauc-heatmap", config={"displaylogo": False},
                           figure=empty_fig("Run to see AUC by window"),
                           style={"flex": "1 1 460px", "minWidth": "0",
                                  "height": "540px"}),
             ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Peak AUC per seizure — click a bar for its window "
                            "distribution"),
             dcc.Graph(id="pex-swauc-bestbar", config={"displaylogo": False},
                       figure=empty_fig("Run to see per-seizure best AUC")),
             style={"marginTop": SPACE_4}),
        card(section_header("Distribution of the selected feature "
                            "(preictal vs pooled interictal windows)"),
             html.Div([
                 dcc.Graph(id="pex-swauc-pdf", config={"displaylogo": False},
                           figure=empty_fig("Run to see the PDF"),
                           style={"flex": "1 1 380px", "minWidth": "0",
                                  "height": "375px"}),
                 dcc.Graph(id="pex-swauc-cdf", config={"displaylogo": False},
                           figure=empty_fig("Run to see the CDF"),
                           style={"flex": "1 1 380px", "minWidth": "0",
                                  "height": "375px"}),
             ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("🎞 Animate the distribution over time to onset"),
             html.Div([
                 html.Div("For the Seizure + Feature selected above, animate that "
                          "metric's distribution across a 30-min window sliding from "
                          "the 'Band to (h)' lookback down to the window right up to "
                          "the seizure. Each frame is one window's distribution "
                          "(histogram + smoothed density + the individual stimuli as "
                          "ticks); the faint grey shape is the far-from-onset "
                          "baseline, so a drift toward onset is visible. 'All seizures "
                          "(group)' pools every seizure by lead time; the ZIP writes "
                          "one GIF per seizure.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 html.Div([
                     button("🎞 Generate GIF (selected seizure)",
                            "pex-swauc-gif-btn"),
                     button("⬇ Download all seizures (.zip)", "pex-swauc-gifzip-btn",
                            variant="secondary"),
                 ], style={"display": "flex", "gap": SPACE_3, "flexWrap": "wrap",
                           "marginTop": SPACE_3, "alignItems": "center"}),
                 html.Div(id="pex-swauc-gif-status",
                          style={"marginTop": SPACE_2, "color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION}),
                 html.Img(id="pex-swauc-gif-img",
                          style={"marginTop": SPACE_3, "maxWidth": "680px",
                                 "width": "100%", "display": "none",
                                 "border": f"1px solid {COLOR_DIVIDER}",
                                 "borderRadius": RADIUS_SM}),
             ], style={"display": "flex", "flexDirection": "column"}),
             style={"marginTop": SPACE_4}),
        card(section_header("📉 Wasserstein step size between consecutive "
                            "distributions"),
             html.Div([
                 html.Div("For the Seizure + Feature selected above: the "
                          "Wasserstein-1 distance between each 30-min frame's "
                          "distribution and the previous one (blue = frame-to-frame "
                          "step; small means the distribution is holding still), and "
                          "each frame's distance from the far-from-onset baseline "
                          "(red dotted = how far it has drifted). Both sides of every "
                          "comparison are downsampled to equal n, so the distance "
                          "reflects distribution shape, not frame size. Updates live "
                          "with the Seizure / Feature / Band controls above — a "
                          "prototype trend view, no null test.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 dcc.Graph(id="pex-swauc-w1trend", config={"displaylogo": False},
                           figure=empty_fig("Pick a Seizure + Feature above to see "
                                            "the W₁ trend."),
                           style={"marginTop": SPACE_3, "height": "380px"}),
             ], style={"display": "flex", "flexDirection": "column"}),
             style={"marginTop": SPACE_4}),
        # Click-to-expand modal: the AUC-across-windows distribution for a
        # clicked seizure (+ all seizures + group). Hidden until a bar is clicked.
        html.Div(id="pex-swauc-modal", style={"display": "none"}, children=[
            html.Div([
                html.Div([
                    html.Span(id="pex-swauc-modal-title",
                              style={"color": COLOR_TEXT_PRIMARY,
                                     "fontSize": FONT_SIZE_TITLE,
                                     "fontWeight": "600"}),
                    html.Button("✕", id="pex-swauc-modal-close", n_clicks=0,
                                style={"marginLeft": "auto", "cursor": "pointer",
                                       "background": "#3a3a4a", "color": "#f0f0f5",
                                       "border": f"1px solid {COLOR_DIVIDER}",
                                       "borderRadius": RADIUS_SM,
                                       "padding": "2px 10px",
                                       "fontSize": FONT_SIZE_BODY}),
                ], style={"display": "flex", "alignItems": "center",
                          "gap": SPACE_3, "marginBottom": SPACE_2}),
                html.Div("Window AUC vs lead time (for the selected Feature): "
                         "x = how long before onset the interictal window sits, "
                         "y = that window's rank AUC. Each coloured line is a "
                         "seizure (the clicked one bold, the others faint); the "
                         "bold grey line is the group median with an IQR band. "
                         "Change the Feature dropdown to re-plot.",
                         style={"color": COLOR_TEXT_TERTIARY,
                                "fontSize": FONT_SIZE_CAPTION,
                                "marginBottom": SPACE_2, "maxWidth": "90ch"}),
                dcc.Graph(id="pex-swauc-modal-fig", config={"displaylogo": False},
                          figure=empty_fig("Click a seizure bar"),
                          style={"height": "470px"}),
            ], style={"background": "#1e1e2f",
                      "border": f"1px solid {COLOR_DIVIDER}",
                      "borderRadius": RADIUS_SM, "padding": SPACE_4,
                      "width": "92%", "maxWidth": "920px",
                      "boxShadow": "0 8px 40px rgba(0,0,0,0.5)"}),
        ]),
        dcc.Interval(id="pex-swauc-poll", interval=1200, disabled=True),
        dcc.Store(id="pex-swauc-job"),
        dcc.Interval(id="pex-swauc-gif-poll", interval=1200, disabled=True),
        dcc.Store(id="pex-swauc-gif-job"),
        dcc.Download(id="pex-swauc-gif-dl"),
    ], style={"padding": SPACE_4})


# ===================================================================== #
#  Nonstationarity control (null-onset test): is a preictal AUC real, or drift?
# ===================================================================== #
_NSC_JOBS: dict = {}
_NSC_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_NSC_CACHE_MAX = 3
_NSC_COLORS = {("evoked", "sz"): "#5e7ce2", ("evoked", "null"): "#9aabe8",
               ("passive", "sz"): "#e2a15e", ("passive", "null"): "#e8c79f"}


def _nsc_key(jid, seed, k, buffer_sec, nwin, blo, bhi, wtok) -> str:
    return f"{jid}|nsc|{seed}|{k}|{int(buffer_sec)}|{nwin}|{blo}|{bhi}|{wtok}"


def _nsc_set(key, **kw):
    with _LOCK:
        _NSC_JOBS.setdefault(key, {}).update(kw)
        _prune_jobs(_NSC_JOBS)


def _nsc_finish(key, result):
    with _LOCK:
        _NSC_CACHE[key] = result
        while len(_NSC_CACHE) > _NSC_CACHE_MAX:
            _NSC_CACHE.popitem(last=False)
        _NSC_JOBS[key] = {"status": "done", "progress": "done"}
        _prune_jobs(_NSC_JOBS)


def _nsc_kick(store, key, jid, animal, evoked_dir, cache_dir, seed, k, buffer_sec,
              nwin, blo, bhi, ev_sv, ev_cfg, ev_from, ev_to, protocol):
    with _LOCK:
        if key in _NSC_CACHE:
            return
        st = _NSC_JOBS.get(key)
        if st and st.get("status") == "running":
            return
        _NSC_JOBS[key] = {"status": "running", "progress": "starting…"}
        _prune_jobs(_NSC_JOBS)
    threading.Thread(
        target=_run_admitted, name=f"nsc-{key}", daemon=True,
        args=(_nsc_worker, _nsc_set, key,
              (key, store, jid, animal, evoked_dir, cache_dir, seed, k, buffer_sec,
               nwin, blo, bhi, ev_sv, ev_cfg, ev_from, ev_to, protocol))).start()


def _nsc_worker(key, store, jid, animal, evoked_dir, cache_dir, seed, k, buffer_sec,
                nwin, blo, bhi, ev_sv, ev_cfg, ev_from, ev_to, protocol):
    """Build the evoked matrix (cached; warm a custom window) then run the full
    nonstationarity control. Never raises -- errors surface in the poll."""
    try:
        if not animal:
            _nsc_finish(key, {"empty": True,
                              "reason": "Pick an animal in the scope bar first."})
            return
        if ev_sv not in (None, "evoked"):
            _nsc_set(key, progress="warming the evoked window…")
            _passive.warm_variant(
                animal, evoked_dir, ev_sv, ev_cfg, protocol=protocol or None,
                file_filter=_matrix.near_seizure_filter(store, animal, bhi),
                progress=lambda d, n, _fp: _nsc_set(
                    key, progress=f"windowing evoked traces… ({d}/{n})"))
        _nsc_set(key, progress="building the evoked matrix…")
        evoked_full = build_matrix_cached(
            store, animal, evoked_dir, cache_dir, protocol=protocol or None,
            window_sec=bhi, variant="evoked", sidecar_variant=ev_sv,
            feature_cfg=ev_cfg)
        feats = list(_cfg.metrics_for_variant("evoked"))
        res = _nscmod.run_control(
            store, animal, evoked_dir, cache_dir, base_full_evoked=evoked_full,
            feats_evoked=feats, seed=int(seed), k_draws=int(k),
            buffer_sec=float(buffer_sec), nwin=int(nwin), band_lo_sec=float(blo),
            band_hi_sec=float(bhi), evoked_sidecar_variant=ev_sv,
            evoked_feature_cfg=ev_cfg, evoked_from_ms=float(ev_from),
            evoked_to_ms=float(ev_to), protocol=protocol or None,
            progress=lambda m: _nsc_set(key, progress=m))
        _nsc_finish(key, res)
    except Exception as e:                                    # noqa: BLE001
        logger.exception("nonstationarity control failed (key=%s): %s", key, e)
        _nsc_set(key, status="error", progress=f"error: {e}")


def _nsc_band_traces(x, stack, name, color, dash, show):
    """Median line + translucent IQR band from a (units x W) AUC stack."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cols
        med = np.nanmedian(stack, axis=0)
        lo = np.nanpercentile(stack, 25, axis=0)
        hi = np.nanpercentile(stack, 75, axis=0)
    fill = (f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},"
            f"{int(color[5:7], 16)},0.12)")
    return [
        go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip",
                   showlegend=False),
        go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                   fillcolor=fill, hoverinfo="skip", showlegend=False),
        go.Scatter(x=x, y=med, mode="lines+markers", name=name,
                   line=dict(color=color, width=2.5, dash=dash), showlegend=show,
                   hovertemplate=name + ": AUC %{y:.3f}<extra></extra>"),
    ]


def _nsc_overlay_fig(res, feature) -> go.Figure:
    """AUC-vs-lead-time for *feature*: seizure vs null medians (+IQR bands) for
    both the evoked and passive windows. Seizure band above the null band = signal
    above chance beyond nonstationarity."""
    if not res or res.get("empty"):
        return empty_fig("Run the control")
    offs = np.asarray(res["offsets"], dtype=float) / 3600.0     # hours before onset
    fig = go.Figure()
    for variant in ("evoked", "passive"):
        v = res["variants"][variant]
        if feature not in v["feats"]:
            continue
        fi = v["feats"].index(feature)
        if v["sz_stack"].shape[0]:
            fig.add_traces(_nsc_band_traces(
                offs, v["sz_stack"][:, fi, :], f"{variant} · seizure",
                _NSC_COLORS[(variant, "sz")], None, True))
        if v["null_stack"].shape[0]:
            fig.add_traces(_nsc_band_traces(
                offs, v["null_stack"][:, fi, :], f"{variant} · null",
                _NSC_COLORS[(variant, "null")], "dot", True))
    fig.add_hline(y=0.5, line=dict(color=COLOR_TEXT_TERTIARY, dash="dash", width=1))
    fig.update_layout(**_pc_layout(f"Window AUC vs lead time · {feature}",
                                   "hours before onset", height=440))
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(title="window AUC", range=[0.4, 1.0])
    return fig


def _nsc_delta_heatmap(res, variant) -> go.Figure:
    """Δ = AUC(seizure) − AUC(null), feature × window, diverging at 0."""
    v = res["variants"][variant]
    feats, ranked, labels = v["feats"], v["ranked"], res["win_labels"]
    order = [feats.index(f) for f in ranked][::-1]
    z = np.asarray(v["delta"])[order]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=ranked[::-1],
        colorscale="RdBu", reversescale=True, zmid=0.0, zmin=-0.25, zmax=0.25,
        colorbar=dict(title=dict(text="Δ AUC", font=dict(size=9)), thickness=10,
                      len=0.85, x=1.005),
        hovertemplate="%{y} · %{x} · Δ %{z:+.3f}<extra></extra>"))
    fig.update_layout(**_pc_layout(f"Δ AUC (seizure − null) · {variant}",
                                   "interictal window (before onset)", height=530),
                      yaxis_title="")
    return fig


def _nsc_delta_bar(res, variant) -> go.Figure:
    """Per-feature Δ AUC (seizure − null median); green = survives (Δ>0, p<.05)."""
    v = res["variants"][variant]
    show = [f for f in v["ranked"]
            if np.isfinite(v["delta_feat"].get(f, np.nan))][:14][::-1]
    xs = [v["delta_feat"][f] for f in show]
    ps = [v["p"].get(f, np.nan) for f in show]
    colours = [COLOR_SUCCESS if (d > 0 and np.isfinite(p) and p < 0.05)
               else COLOR_DIVIDER for d, p in zip(xs, ps)]
    txt = [(f"p={p:.3f}" if np.isfinite(p) else "") for p in ps]
    fig = go.Figure(go.Bar(
        x=xs, y=show, orientation="h", marker_color=colours, text=txt,
        textposition="outside", textfont=dict(size=9),
        hovertemplate="%{y}: Δ %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=COLOR_TEXT_TERTIARY, dash="dash", width=1))
    fig.update_layout(**_pc_layout(f"Δ AUC per feature · {variant}",
                                   "Δ AUC (seizure − null)", height=440),
                      yaxis_title="")
    return fig


def _nsc_verdict(res) -> object:
    if not res or res.get("empty"):
        return _callout(res.get("reason", "Build the evoked matrix, then Run.")
                        if res else "Run the control.", COLOR_WARNING)
    v = res["variants"]["evoked"]
    top = v["ranked"][0] if v["ranked"] else None
    if top is None:
        return _callout("No scorable features.", COLOR_WARNING)
    a = v["sz_group"].get(top, float("nan"))
    m = v["null_group_median"].get(top, float("nan"))
    d = v["delta_feat"].get(top, float("nan"))
    p = v["p"].get(top, float("nan"))
    ok = np.isfinite(d) and d > 0 and np.isfinite(p) and p < 0.05
    colour = COLOR_SUCCESS if ok else COLOR_WARNING
    verdict = ("survives the null (real pre-ictal signal)" if ok else
               "≈ nonstationarity — NOT a pre-ictal signal")
    txt = (f"Top evoked feature '{top}': seizure AUC {a:.3f} · null {m:.3f} · "
           f"Δ {d:+.3f} · p {p:.3f}  →  {verdict}.  "
           f"({res.get('n_seizures')} seizures vs {res.get('n_null_placed')}/"
           f"{res.get('n_null_requested')} null onsets × {res.get('k_draws')} "
           f"draws.)")
    note = res.get("null_reason")
    if note:
        txt += f"  ⚠ {note}"
    return _callout(txt, colour)


def _nsc_render(res, feature):
    if not res or res.get("empty") or "variants" not in res:
        blank = empty_fig("Run the control")
        return (_nsc_overlay_fig(res, feature), blank, blank, blank,
                _nsc_verdict(res))
    return (_nsc_overlay_fig(res, feature),
            _nsc_delta_heatmap(res, "evoked"),
            _nsc_delta_heatmap(res, "passive"),
            _nsc_delta_bar(res, "evoked"),
            _nsc_verdict(res))


def layout_nonstationarity(store):
    """The capstone control: place random deep-interictal 'null' onsets and run the
    identical sliding-AUC on them, for the evoked window and its passive mirror.
    Δ = AUC(seizure) − AUC(null) says whether a preictal AUC is real or just
    feature drift. The heaviest peri-ictal step -- last in the progression."""
    inp = {**DROPDOWN_STYLE, "width": "90px"}
    return html.Div([
        scope_bar(store),
        card(section_header("Nonstationarity control — is the preictal AUC real, "
                            "or just drift?"),
             html.Div([
                 html.Div("Places N random NULL onsets in deep-interictal time "
                          "(matched to the seizure count, on days that had seizures, "
                          "≥ lookback+1 h clear of every real seizure on both sides) "
                          "and runs the IDENTICAL sliding-window AUC on them — for the "
                          "evoked window AND its matched passive (pre-stim) mirror. K "
                          "draws give a null band + p. If the null AUC matches the "
                          "seizure AUC (Δ≈0), the 'preictal' separation is feature "
                          "drift, not a pre-ictal signal. Build the matrix in the scope "
                          "bar first (any variant), then Run.",
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION, "maxWidth": "95ch"}),
                 html.Div([
                     _ctl("Windows", dcc.Input(
                         id="pex-nsc-nwin", type="number", value=_cfg.SLIDING_N_WINDOWS,
                         min=2, max=40, step=1, debounce=True, style=inp),
                         "Interictal windows per onset."),
                     _ctl("Band from (h)", dcc.Input(
                         id="pex-nsc-bandlo", type="number", value=1, min=0.5, max=12,
                         step=0.5, debounce=True, style=inp),
                         "Closest the windows get to onset."),
                     _ctl("Band to (h)", dcc.Input(
                         id="pex-nsc-bandhi", type="number", value=6, min=1, max=24,
                         step=0.5, debounce=True, style=inp),
                         "Furthest before onset (the lookback)."),
                     _ctl("Null draws K", dcc.Input(
                         id="pex-nsc-draws", type="number", value=20, min=1, max=200,
                         step=1, debounce=True, style=inp),
                         "Independent matched null draws (band + p). K=1 = one draw."),
                     _ctl("Seed", dcc.Input(
                         id="pex-nsc-seed", type="number", value=0, min=0, step=1,
                         debounce=True, style=inp), "Reproducible RNG seed."),
                     _ctl("Buffer (h)", dcc.Input(
                         id="pex-nsc-buffer", type="number", value=7, min=1, max=48,
                         step=0.5, debounce=True, style=inp),
                         "Seizure-free clearance each side (≥ lookback+1 h)."),
                     _ctl("Feature", dcc.Dropdown(
                         id="pex-nsc-feature", clearable=False,
                         style={**DROPDOWN_STYLE, "minWidth": "200px"}),
                         "Which feature's AUC-vs-lead-time to overlay."),
                     html.Div(button("▶ Run nonstationarity control",
                                     "pex-nsc-build", icon_name="play"),
                              style={"alignSelf": "flex-end"}),
                 ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
                           "alignItems": "flex-start", "marginTop": SPACE_3}),
                 html.Div(id="pex-nsc-status",
                          style={"marginTop": SPACE_2, "color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION}),
                 html.Div(id="pex-nsc-verdict", style={"marginTop": SPACE_2}),
             ], style={"display": "flex", "flexDirection": "column",
                       "gap": SPACE_2, "marginTop": SPACE_2}),
             style={"marginTop": SPACE_4}),
        card(section_header("Window AUC vs lead time — seizure vs null "
                            "(evoked & passive)"),
             dcc.Graph(id="pex-nsc-overlay", config={"displaylogo": False},
                       figure=empty_fig("Run to compare seizure vs null"),
                       style={"height": "450px"}),
             style={"marginTop": SPACE_4}),
        card(section_header("Δ AUC (seizure − null) by feature × window — "
                            "blue = seizure exceeds null"),
             html.Div([
                 dcc.Graph(id="pex-nsc-delta-evoked", config={"displaylogo": False},
                           figure=empty_fig("evoked Δ"),
                           style={"flex": "1 1 460px", "minWidth": "0",
                                  "height": "540px"}),
                 dcc.Graph(id="pex-nsc-delta-passive", config={"displaylogo": False},
                           figure=empty_fig("passive Δ"),
                           style={"flex": "1 1 460px", "minWidth": "0",
                                  "height": "540px"}),
             ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Δ AUC per feature (evoked) — green survives the null "
                            "(Δ>0, p<0.05)"),
             dcc.Graph(id="pex-nsc-delta-bar", config={"displaylogo": False},
                       figure=empty_fig("Run"), style={"height": "450px"}),
             style={"marginTop": SPACE_4}),
        dcc.Interval(id="pex-nsc-poll", interval=1500, disabled=True),
        dcc.Store(id="pex-nsc-job"),
    ], style={"padding": SPACE_4})


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
        State("pex-colorby", "value"),
        prevent_initial_call=False,
    )
    def _embed_colorby_opts(_animal, cur):
        """Populate the Embedding lens's shared colour-by dropdown. It drives BOTH
        the passive and evoked columns, so it offers the EVOKED SUPERSET (every
        metric stays reachable for the evoked column); an evoked-only metric
        degrades gracefully on the passive column via the _figure guard. No longer
        keyed on pex-variant (that only scopes the other lenses)."""
        copts = _colorby_options("evoked")
        cvals = {o["value"] for o in copts}
        # Default to SEIZURE so the embedding organizes by seizure out of the box
        # (stim group rides along as the marker symbol); keep the user's choice
        # if they've already picked one.
        return copts, (cur if cur in cvals else "seizure_idx")

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
        Input("tabs", "value"),
    )
    def _preview(animal, protocol, variant, window_h, winmode, wf, wt, wg, tab):
        variant = variant or "evoked"
        try:
            sv, cfg = _resolve_window(variant, winmode or "full", wf, wt, wg)
        except ValueError as e:
            return _callout(f"Feature window: {e}.", COLOR_WARNING, "⚠")
        try:
            return _preview_panel(store, animal, protocol or "", variant,
                                  float(window_h or 6.0), _window_label(sv, cfg),
                                  tab=tab or "")
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
                      _win_token(sv, cfg) + _excl_tok(store, animal))
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
                       _win_token(sv, cfg) + _excl_tok(store, sel.get("animal")))

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
        Output("pex-nb-dl", "data"),
        Output("pex-io-status", "children", allow_duplicate=True),
        Input("pex-nb-btn", "n_clicks"),
        State("tabs", "value"),
        State("pex-job", "data"),
        State("pex-animal", "value"), State("pex-protocol", "value"),
        State("pex-variant", "value"), State("pex-window-h", "value"),
        State("pex-winmode", "value"), State("pex-win-from", "value"),
        State("pex-win-to", "value"), State("pex-win-guard", "value"),
        State("pex-swauc-nwin", "value"), State("pex-swauc-bandlo", "value"),
        State("pex-swauc-bandhi", "value"), State("pex-pc-nphases", "value"),
        State("pex-traj-y", "value"), State("pex-method", "value"),
        prevent_initial_call=True,
    )
    def _export_notebook(_n, tab, jid, animal, protocol, variant, window_h,
                         winmode, wf, wt, wg, nwin, blo, bhi, nphases, trend_feat,
                         method):
        """Export the CURRENT lens as a self-contained reproducible notebook +
        frozen matrix (tab-aware; only the matrix-based lenses)."""
        from src.periictal import notebook_export as _nx
        if not _nx.supported(tab or ""):
            return no_update, "Notebook export isn't available for this lens."
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty") or "full" not in cached:
            return no_update, "Build first (scope bar), then export the notebook."
        variant = variant or "evoked"
        try:
            sv, cfg = _resolve_window(variant, winmode or "full", wf, wt, wg)
            wlabel = _window_label(sv, cfg)
        except Exception:                                 # noqa: BLE001
            wlabel = ""
        meta = {"animal": animal or "", "protocol": protocol or "all protocols",
                "variant": variant, "feature_window": wlabel, "lead_up_h": window_h,
                "n_seizures": int(cached.get("n_seizures", 0)),
                "exported_at": datetime.now().isoformat(timespec="seconds")}
        if tab == "periictal_slidingauc":
            params = {"variant": variant, "n_windows": int(nwin or 12),
                      "band_lo": float(blo or 1) * 3600.0,
                      "band_hi": float(bhi or 6) * 3600.0}
        elif tab == "periictal_pdfcdf":
            params = {"variant": variant, "nphases": int(nphases or 6)}
        elif tab == "periictal_trend":
            params = {"variant": variant, "feature": trend_feat or "peak_to_trough"}
        else:                                             # periictal_embedding
            params = {"variant": variant, "method": method or "pca"}
        try:
            blob, fname = _nx.build_export(tab, cached["full"], meta, params)
        except Exception as e:                            # noqa: BLE001
            logger.warning("notebook export failed: %s", e)
            return no_update, f"Notebook export failed: {e}"
        return (dcc.send_bytes(lambda buf: buf.write(blob), fname),
                f"Exported {fname} ({len(blob) // 1024} KB) — unzip & Run All.")

    @app.callback(
        Output("pex-job", "data", allow_duplicate=True),
        Output("pex-io-status", "children", allow_duplicate=True),
        Output("pex-animal", "value", allow_duplicate=True),
        Output("pex-protocol", "value", allow_duplicate=True),
        Output("pex-variant", "value", allow_duplicate=True),
        Output("pex-window-h", "value", allow_duplicate=True),
        Output("pex-method", "value", allow_duplicate=True),
        Output("pex-embed2-job", "data", allow_duplicate=True),
        Input("pex-import-up", "contents"),
        prevent_initial_call=True,
    )
    def _import_embedding(contents):
        if not contents:
            return (no_update,) * 8
        try:
            _hdr, _, b64 = contents.partition(",")
            selection, result = _eio.loads(base64.b64decode(b64))
        except Exception as e:                            # noqa: BLE001
            logger.warning("embedding import failed: %s", e)
            return (no_update, f"Import failed: {e}",
                    no_update, no_update, no_update, no_update, no_update,
                    no_update)
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
        # pex-job pulses the Trend / Waveform / slow lenses; the Embedding lens
        # now renders from pex-embed2-job, so publish the imported jid under its
        # variant's column key ({variant: jid}) too, or the two-column scatter
        # stays on its placeholder. The other column is absent (an import carries
        # one variant), so it simply keeps its current figure.
        embed2 = {(selection.get("variant") or "evoked"): jid}
        return (jid, msg, selection.get("animal"),
                selection.get("protocol") or "", selection.get("variant"),
                selection.get("window_h"), method, embed2)

    # ---- Live feature-warmer status (alive vs frozen) ---- #
    @app.callback(
        Output("pex-warm-status", "children"),
        Input("pex-warm-tick", "n_intervals"),
    )
    def _render_warm_status(_n):
        try:
            return _warm_status_line(_sw.status())
        except Exception:                             # noqa: BLE001
            return ""

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

    # ---- Two-column embedding: build BOTH passive + evoked, render side by side #
    @app.callback(
        Output("pex-embed2-status", "children"),
        Output("pex-embed2-poll", "disabled"),
        Output("pex-embed2-job", "data"),
        Input("pex-embed2-build", "n_clicks"),
        Input("pex-embed2-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-window-h", "value"),
        State("pex-method", "value"),
        State("pex-winmode", "value"),
        State("pex-win-from", "value"),
        State("pex-win-to", "value"),
        State("pex-win-guard", "value"),
        State("pex-embed2-job", "data"),
        prevent_initial_call=True,
    )
    def _embed2_build_or_poll(_n, _iv, animal, protocol, window_h, method,
                              winmode, wf, wt, wg, cur):
        """Build the passive AND evoked embeddings for the current scope; publish
        {passive: jid, evoked: jid} ONLY when both are cached. Two _kick()s into
        the shared _CACHE (distinct _job_id keys) run concurrently and reuse any
        single-variant build already done. Ignores pex-variant by design -- it
        builds both. Local button+poll => cannot mis-fire on the other lenses."""
        if not animal:
            return "Pick an animal.", True, no_update
        window_h = float(window_h or 6.0)
        cap = _cfg.INTERACTIVE_POINT_CAP
        try:
            ev_sv, ev_cfg = _resolve_window("evoked", winmode or "full", wf, wt, wg)
        except ValueError as e:
            return f"⚠ Evoked feature window: {e}", True, no_update
        # Passive ALWAYS uses the canonical -200..-1 ms pre-stim window -- never
        # the shared win-from/to, which may currently hold an evoked window. Wrap
        # like the evoked resolve: a cleared guard reaches validation HERE (the
        # evoked 'full' path skips the guard), so an unwrapped call would 500.
        try:
            pa_sv, pa_cfg = _resolve_window("passive", "custom",
                                            *_cfg.DEFAULT_PASSIVE_WINDOW_MS, wg)
        except ValueError as e:
            return f"⚠ Passive feature window: {e}", True, no_update
        jids = {}
        for kind, sv, fcfg in (("passive", pa_sv, pa_cfg),
                               ("evoked", ev_sv, ev_cfg)):
            jid = _job_id(animal, protocol or "", kind, window_h, method, cap,
                          _win_token(sv, fcfg) + _excl_tok(store, animal))
            jids[kind] = jid
            with _LOCK:
                have = jid in _CACHE
                st = dict(_JOBS.get(jid) or {})
            if not have and st.get("status") != "error":
                _kick(store, evoked_dir, jid, animal, protocol or "", kind,
                      window_h, method, cap, sv, fcfg)

        def _status_of(kind):
            """(human status, published-jid-or-None) for one column."""
            jid = jids[kind]
            with _LOCK:
                cached = jid in _CACHE
                st = dict(_JOBS.get(jid) or {})
            if cached:
                return "✓", jid
            if st.get("status") == "error":
                return f"error: {st.get('progress', '')}", None
            return st.get("progress", "starting…"), None

        s_pass, pub_pass = _status_of("passive")
        s_evk, pub_evk = _status_of("evoked")
        # Publish per column: that column's jid once IT caches, else None. The
        # dict changes as each column completes, so _embed2_render re-fires and
        # draws the fast (evoked) column before the slow passive re-windowing
        # lands -- progressive, not all-or-nothing.
        pub = {"passive": pub_pass, "evoked": pub_evk}

        def _settled(s):
            return s == "✓" or s.startswith("error")
        done = _settled(s_pass) and _settled(s_evk)
        status = f"passive: {s_pass} · evoked: {s_evk}"
        # Only write the job store when a column's cached-state actually CHANGES,
        # never every poll -- otherwise Dash re-fires _embed2_render each tick and
        # rebuilds the ~10k-point evoked scatter (2-7 s) over and over, saturating
        # the render threads and starving the slow passive build. Status still
        # updates every tick (cheap text) so progress stays live.
        job_out = pub if pub != (cur or {}) else no_update
        return ("" if done else "⏳ ") + status, done, job_out

    @app.callback(
        Output("pex-graph-passive", "figure"),
        Output("pex-reading-passive", "children"),
        Output("pex-loadings-passive", "figure"),
        Output("pex-graph-evoked", "figure"),
        Output("pex-reading-evoked", "children"),
        Output("pex-loadings-evoked", "figure"),
        Input("pex-embed2-job", "data"),
        State("pex-colorby", "value"),
        State("pex-color-cap", "value"),
        State("pex-color-cap-unit", "value"),
        prevent_initial_call=False,
    )
    def _embed2_render(jobs, color_by, cap_val, cap_unit):
        """Draw both columns from cache on the pair-job pulse (or on remount from
        the persisted store). no_update per column whose job isn't cached yet, so
        the faster (evoked) column can appear before the slower passive one."""
        jobs = jobs or {}
        cap = _cap_seconds(cap_val, cap_unit)
        cb = color_by or "time_to_onset_sec"
        out = []
        for kind in ("passive", "evoked"):
            cached = _CACHE.get(jobs.get(kind)) if jobs.get(kind) else None
            if cached is None:
                out += [no_update, no_update, no_update]
                continue
            fig, reading = _embed_fig_reading(cached, cb, cap)
            out += [fig, reading, _loadings_fig(cached)]
        return tuple(out)

    @app.callback(
        Output("pex-graph-passive", "figure", allow_duplicate=True),
        Output("pex-graph-evoked", "figure", allow_duplicate=True),
        Input("pex-colorby", "value"),
        Input("pex-color-cap", "value"),
        Input("pex-color-cap-unit", "value"),
        State("pex-embed2-job", "data"),
        prevent_initial_call=True,
    )
    def _embed2_recolor(color_by, cap_val, cap_unit, jobs):
        """Instant recolour of BOTH columns from cache (the second writer of the
        two graph figures; allow_duplicate lives ONLY here)."""
        jobs = jobs or {}
        cap = _cap_seconds(cap_val, cap_unit)
        out = []
        for kind in ("passive", "evoked"):
            cached = _CACHE.get(jobs.get(kind)) if jobs.get(kind) else None
            if not cached or cached.get("empty"):
                out.append(no_update)
                continue
            out.append(_figure(cached["emb"], cached["sub"], color_by,
                               cached.get("method", "pca"), cached.get("meta"), cap))
        return tuple(out)

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

    # --- Sliding-window ROC-AUC callbacks --- #

    @app.callback(
        Output("pex-swauc-feature", "options"),
        Output("pex-swauc-feature", "value"),
        Input("pex-variant", "value"),
        State("pex-swauc-feature", "value"),
        prevent_initial_call=False,
    )
    def _swauc_feature_opts(variant, cur):
        opts = [{"label": m, "value": m}
                for m in _cfg.metrics_for_variant(variant or "evoked")]
        vals = {o["value"] for o in opts}
        return opts, (cur if cur in vals else (opts[0]["value"] if opts else None))

    @app.callback(
        Output("pex-swauc-groupbar", "figure"),
        Output("pex-swauc-heatmap", "figure"),
        Output("pex-swauc-bestbar", "figure"),
        Output("pex-swauc-pdf", "figure"),
        Output("pex-swauc-cdf", "figure"),
        Output("pex-swauc-status", "children"),
        Output("pex-swauc-poll", "disabled"),
        Output("pex-swauc-job", "data"),
        Output("pex-swauc-notes", "children"),
        Output("pex-swauc-seizure", "options"),
        Input("pex-swauc-build", "n_clicks"),
        Input("pex-swauc-poll", "n_intervals"),
        State("pex-job", "data"),
        State("pex-swauc-nwin", "value"),
        State("pex-swauc-bandlo", "value"),
        State("pex-swauc-bandhi", "value"),
        State("pex-swauc-seizure", "value"),
        State("pex-swauc-feature", "value"),
        prevent_initial_call=True,
    )
    def _swauc_build_or_poll(_n, _iv, jid, nwin, blo, bhi, sel, feature):
        grp_opt = [{"label": "All seizures (group)", "value": "group"}]
        if not jid or _CACHE.get(jid) is None:
            msg = "Build the matrix in the scope bar above first, then Run."
            return (no_update,) * 5 + (msg, True, no_update, "", no_update)
        nwin = int(nwin or _cfg.SLIDING_N_WINDOWS)
        blo, bhi = float(blo or 1.0), float(bhi or 6.0)
        key = _swauc_key(jid, nwin, blo, bhi)
        with _LOCK:
            result = _SWAUC_CACHE.get(key)
            state = dict(_SWAUC_JOBS.get(key) or {})
        if result is not None:
            if result.get("empty"):
                ef = empty_fig("No seizure had both a clean preictal window and a "
                               "sampled interictal window")
                return (ef, ef, ef, ef, ef, result.get("reason", "no data"),
                        True, key, _swauc_notes(result), grp_opt)
            feat = _swauc_feat(result, feature)
            g, h, b, pdf, cdf, notes = _swauc_render(result, sel or "group", feat)
            return (g, h, b, pdf, cdf,
                    f"✓ {result['n_seizures_used']} seizure(s) scored", True, key,
                    notes, _swauc_seizure_opts(result))
        if state.get("status") == "error":
            ef = empty_fig("Run failed", hint=state.get("progress", ""))
            return (ef, ef, ef, ef, ef, state.get("progress", "error"),
                    True, no_update, "", no_update)
        _swauc_kick(store, key, jid, nwin, blo * 3600.0, bhi * 3600.0)
        prog = (_SWAUC_JOBS.get(key) or {}).get("progress", "starting…")
        return (no_update,) * 5 + (f"⏳ {prog}", False, no_update, no_update,
                                   no_update)

    @app.callback(
        Output("pex-swauc-groupbar", "figure", allow_duplicate=True),
        Output("pex-swauc-heatmap", "figure", allow_duplicate=True),
        Output("pex-swauc-bestbar", "figure", allow_duplicate=True),
        Output("pex-swauc-pdf", "figure", allow_duplicate=True),
        Output("pex-swauc-cdf", "figure", allow_duplicate=True),
        Input("pex-swauc-seizure", "value"),
        Input("pex-swauc-feature", "value"),
        State("pex-swauc-job", "data"),
        prevent_initial_call=True,
    )
    def _swauc_rerender(sel, feature, key):
        result = _SWAUC_CACHE.get(key) if key else None
        if not result or result.get("empty"):
            return (no_update,) * 5
        feat = _swauc_feat(result, feature)
        g, h, b, pdf, cdf, _notes = _swauc_render(result, sel or "group", feat)
        return g, h, b, pdf, cdf

    @app.callback(
        Output("pex-swauc-w1trend", "figure"),
        Input("pex-swauc-seizure", "value"),
        Input("pex-swauc-feature", "value"),
        Input("pex-swauc-bandhi", "value"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _swauc_w1trend(sel, feature, bhi, jid):
        cached = _CACHE.get(jid) if jid else None
        if not cached or cached.get("empty") or "full" not in cached:
            return empty_fig("Build the matrix in the scope bar above first.")
        lookback = float(bhi or 6.0) * 3600.0
        return _w1trend_fig(cached["full"], sel, feature, lookback)

    # --- Nonstationarity control (null test) callbacks --- #
    @app.callback(
        Output("pex-nsc-feature", "options"),
        Output("pex-nsc-feature", "value"),
        Input("pex-variant", "value"),
        State("pex-nsc-feature", "value"),
        prevent_initial_call=False,
    )
    def _nsc_feature_opts(_variant, cur):
        # The overlay is per-feature and spans both windows; evoked names are the
        # superset that always populate, so drive the picker from them.
        opts = [{"label": m, "value": m}
                for m in _cfg.metrics_for_variant("evoked")]
        vals = {o["value"] for o in opts}
        return opts, (cur if cur in vals else (opts[0]["value"] if opts else None))

    @app.callback(
        Output("pex-nsc-overlay", "figure"),
        Output("pex-nsc-delta-evoked", "figure"),
        Output("pex-nsc-delta-passive", "figure"),
        Output("pex-nsc-delta-bar", "figure"),
        Output("pex-nsc-verdict", "children"),
        Output("pex-nsc-status", "children"),
        Output("pex-nsc-poll", "disabled"),
        Output("pex-nsc-job", "data"),
        Input("pex-nsc-build", "n_clicks"),
        Input("pex-nsc-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-winmode", "value"),
        State("pex-win-from", "value"),
        State("pex-win-to", "value"),
        State("pex-win-guard", "value"),
        State("pex-nsc-nwin", "value"),
        State("pex-nsc-bandlo", "value"),
        State("pex-nsc-bandhi", "value"),
        State("pex-nsc-draws", "value"),
        State("pex-nsc-seed", "value"),
        State("pex-nsc-buffer", "value"),
        State("pex-nsc-feature", "value"),
        prevent_initial_call=True,
    )
    def _nsc_build_or_poll(_n, _iv, animal, protocol, winmode, wf, wt, wg,
                           nwin, blo, bhi, draws, seed, buffer_h, feature):
        if not animal:
            return (no_update,) * 4 + (
                _callout("Pick an animal in the scope bar first.", COLOR_WARNING),
                "Pick an animal.", True, no_update)
        nwin = int(nwin or _cfg.SLIDING_N_WINDOWS)
        blo, bhi = float(blo or 1.0), float(bhi or 6.0)
        k, seed = int(draws or 20), int(seed or 0)
        buffer_h = float(buffer_h or (bhi + 1.0))
        # The evoked window: a custom scope-bar window becomes the evoked window
        # (its passive mirror is derived inside run_control); otherwise the fast
        # default [1, 200] ms.
        if (winmode or "full") == "custom" and wf is not None and wt is not None:
            try:
                sv, cfg = _resolve_window("evoked", "custom", wf, wt, wg)
            except ValueError as e:
                return (no_update,) * 4 + (
                    _callout(f"Evoked window: {e}.", COLOR_WARNING),
                    f"⚠ {e}", True, no_update)
            ev_from, ev_to = float(wf), float(wt)
        else:
            sv, cfg, ev_from, ev_to = "evoked", None, 1.0, 200.0
        wtok = _win_token(sv, cfg) + _excl_tok(store, animal)
        base = f"{animal}|{protocol or ''}|{bhi}"
        key = _nsc_key(base, seed, k, buffer_h * 3600.0, nwin, blo, bhi, wtok)
        with _LOCK:
            result = _NSC_CACHE.get(key)
            state = dict(_NSC_JOBS.get(key) or {})
        if result is not None:
            over, de, dp, bar, verdict = _nsc_render(result, feature)
            if result.get("empty"):
                return (over, de, dp, bar, verdict,
                        result.get("reason", "no data"), True, key)
            return (over, de, dp, bar, verdict,
                    f"✓ {result['n_seizures']} seizures vs "
                    f"{result['n_null_placed']}×{result['k_draws']} null onsets "
                    f"(evoked & passive)", True, key)
        if state.get("status") == "error":
            return (no_update,) * 4 + (
                _callout(state.get("progress", "error"), COLOR_WARNING),
                state.get("progress", "error"), True, no_update)
        _nsc_kick(store, key, base, animal, evoked_dir,
                  _CACHE_DIR or _default_cache_dir(), seed, k, buffer_h * 3600.0,
                  nwin, blo * 3600.0, bhi * 3600.0, sv, cfg, ev_from, ev_to,
                  protocol or "")
        prog = (_NSC_JOBS.get(key) or {}).get("progress", "starting…")
        return (no_update,) * 5 + (f"⏳ {prog}", False, no_update)

    @app.callback(
        Output("pex-nsc-overlay", "figure", allow_duplicate=True),
        Input("pex-nsc-feature", "value"),
        State("pex-nsc-job", "data"),
        prevent_initial_call=True,
    )
    def _nsc_rerender(feature, key):
        result = _NSC_CACHE.get(key) if key else None
        if not result or result.get("empty"):
            return no_update
        return _nsc_overlay_fig(result, feature)

    @app.callback(
        Output("pex-swauc-modal", "style"),
        Output("pex-swauc-modal-fig", "figure"),
        Output("pex-swauc-modal-title", "children"),
        # Reset clickData on close so re-clicking the SAME bar is a value change
        # and re-fires (Dash only fires on a changed Input value; a bar's clickData
        # is deterministic, so without this the modal can't be reopened for a
        # seizure just closed).
        Output("pex-swauc-bestbar", "clickData", allow_duplicate=True),
        Input("pex-swauc-bestbar", "clickData"),
        Input("pex-swauc-modal-close", "n_clicks"),
        State("pex-swauc-job", "data"),
        State("pex-swauc-feature", "value"),
        prevent_initial_call=True,
    )
    def _swauc_open_dist(click, _close, key, feature):
        # Close button (or no click payload) -> hide the modal + clear the click.
        if callback_context.triggered_id == "pex-swauc-modal-close" or not click:
            return {"display": "none"}, no_update, no_update, None
        result = _SWAUC_CACHE.get(key) if key else None
        if not result or result.get("empty"):
            return {"display": "none"}, no_update, no_update, None
        best = result.get("best_per_seizure", [])
        try:
            idx = int(click["points"][0].get("pointNumber", 0))
        except (KeyError, IndexError, TypeError, ValueError):
            idx = -1
        if idx < 0 or idx >= len(best):
            return no_update, no_update, no_update, no_update
        sid = best[idx]["seizure_idx"]
        label = result.get("sz_labels", {}).get(sid, f"seizure {sid}")
        feat = _swauc_feat(result, feature)
        return (_SWAUC_MODAL_SHOWN, _swauc_dist_fig(result, sid, feat),
                f"Window AUC vs lead time · {label}", no_update)

    _GIF_IMG_BASE = {"marginTop": SPACE_3, "maxWidth": "680px", "width": "100%",
                     "border": f"1px solid {COLOR_DIVIDER}", "borderRadius": RADIUS_SM}
    _GIF_IMG_HIDDEN = {**_GIF_IMG_BASE, "display": "none"}
    _GIF_IMG_SHOWN = {**_GIF_IMG_BASE, "display": "block"}

    @app.callback(
        Output("pex-swauc-gif-img", "src"),
        Output("pex-swauc-gif-img", "style"),
        Output("pex-swauc-gif-status", "children"),
        Output("pex-swauc-gif-dl", "data"),
        Output("pex-swauc-gif-poll", "disabled"),
        Output("pex-swauc-gif-job", "data"),
        Input("pex-swauc-gif-btn", "n_clicks"),
        Input("pex-swauc-gifzip-btn", "n_clicks"),
        Input("pex-swauc-gif-poll", "n_intervals"),
        State("pex-job", "data"),
        State("pex-swauc-seizure", "value"),
        State("pex-swauc-feature", "value"),
        State("pex-swauc-bandhi", "value"),
        State("pex-swauc-gif-job", "data"),
        prevent_initial_call=True,
    )
    def _swauc_gif_build_or_poll(_n1, _n2, _iv, jid, sel, feature, bandhi, cur_key):
        trig = callback_context.triggered_id
        lookback = float(bandhi or 6.0) * 3600.0
        # A button press: validate the matrix + kick a fresh render job.
        if trig in ("pex-swauc-gif-btn", "pex-swauc-gifzip-btn"):
            cached = _CACHE.get(jid) if jid else None
            if not cached or cached.get("empty"):
                return (no_update, _GIF_IMG_HIDDEN,
                        "Build the matrix in the scope bar first.", no_update,
                        True, no_update)
            if not feature:
                return (no_update, _GIF_IMG_HIDDEN, "Pick a Feature first.",
                        no_update, True, no_update)
            mode = "one" if trig == "pex-swauc-gif-btn" else "zip"
            key = _gif_key(jid, sel or "group", feature, round(lookback), mode)
            _gif_kick(jid, key, sel, feature, lookback, mode)
            busy = ("⏳ rendering the selected seizure…" if mode == "one"
                    else "⏳ rendering one GIF per seizure…")
            return no_update, _GIF_IMG_HIDDEN, busy, no_update, False, key
        # Poll tick: surface progress, or deliver the finished result (once).
        if not cur_key:
            return no_update, no_update, no_update, no_update, True, no_update
        with _LOCK:
            result = _GIF_CACHE.get(cur_key)
            state = dict(_GIF_JOBS.get(cur_key) or {})
        if result is not None:
            if result.get("empty"):
                return (no_update, _GIF_IMG_HIDDEN,
                        f"⚠ {result.get('reason', 'nothing to animate')}",
                        no_update, True, no_update)
            if result.get("mode") == "zip":
                blob = result["zip"]
                dl = dcc.send_bytes(lambda b: b.write(blob), result["name"])
                return (no_update, _GIF_IMG_HIDDEN,
                        f"✓ {result['name']} ({len(blob) // 1024} KB) downloaded.",
                        dl, True, no_update)
            return (result["src"], _GIF_IMG_SHOWN,
                    "✓ done — the looping GIF is below.", no_update, True, no_update)
        if state.get("status") == "error":
            return (no_update, _GIF_IMG_HIDDEN, state.get("progress", "error"),
                    no_update, True, no_update)
        return (no_update, no_update, f"⏳ {state.get('progress', 'working…')}",
                no_update, False, no_update)

    def _make_select_cb(kind):
        @app.callback(
            Output(f"pex-details-{kind}", "children"),
            Input(f"pex-graph-{kind}", "selectedData"),
            State("pex-embed2-job", "data"),
            prevent_initial_call=True,
        )
        def _select(selected, jobs, _kind=kind):
            cached = _CACHE.get((jobs or {}).get(_kind))
            if not cached or cached.get("empty"):
                return _details_view(None)
            return _details_view(
                summarize_selection(cached["sub"], _selected_indices(selected)))
        return _select

    _select_passive = _make_select_cb("passive")   # noqa: F841 (registers cb)
    _select_evoked = _make_select_cb("evoked")      # noqa: F841

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

    # ---- Slow-dynamics lens: across-trial eigenvalue + circadian control ---- #
    @app.callback(
        Output("pex-sd-phi", "figure"),
        Output("pex-sd-circ", "figure"),
        Output("pex-sd-readout", "children"),
        Output("pex-sd-status", "children"),
        Output("pex-sd-poll", "disabled"),
        Output("pex-sd-job", "data"),
        Input("pex-sd-build", "n_clicks"),
        Input("pex-sd-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-sd-feature", "value"),
        State("pex-sd-win", "value"),
        prevent_initial_call=True,
    )
    def _sd_build_or_poll(_n, _iv, animal, feature, win):
        if not animal or not feature:
            return (no_update, no_update, no_update, "Pick an animal and feature.",
                    True, no_update)
        win = int(win or 300)
        key = _sd_key(animal, feature, win)
        with _LOCK:
            cached = _SD_CACHE.get(key)
            state = dict(_SD_JOBS.get(key) or {})
        if cached is not None:
            s = cached.get("summary")
            return (_sd_phi_fig(s), _sd_circ_fig(s), _sd_readout_view(s),
                    "done", True, key)
        if state.get("status") == "error":
            return (no_update, no_update, no_update,
                    state.get("progress", "error"), True, no_update)
        _sd_kick(store, evoked_dir, key, animal, feature, win)
        prog = (_SD_JOBS.get(key) or {}).get("progress", "starting…")
        return no_update, no_update, no_update, f"⏳ {prog}", False, no_update
