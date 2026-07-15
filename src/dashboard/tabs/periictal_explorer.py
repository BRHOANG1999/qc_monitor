"""Peri-ictal Evoked Explorer tab.

Embeds an animal's per-stimulus evoked (or passive pre-stim) feature vectors and
colours them by CONTINUOUS time-to-next-seizure -- and, crucially, by the
confounds (stim fingerprint, hour-of-day) so an apparent proximity structure
that is really circadian or a stim-setting artefact is visible on the same
screen. This is an EXPLORATION surface, not a test: PCA is the default (honest
linear projection), UMAP is opt-in and labelled "figure, not evidence", and a
banner states the effective n = number of seizures, not the ~10k stimuli.

Builds run OFF the render thread (a daemon + a dcc.Interval poll, mirroring
chronic_evoked) so the page never freezes; recolouring reads the cached
embedding and is instant.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update

from src.dashboard.components import (DROPDOWN_STYLE, LABEL_STYLE, button, card,
                                      loading_icon, section_header)
from src.dashboard.data_helpers import empty_fig
from src.dashboard.design import (COLOR_TEXT_SECONDARY, COLOR_WARNING, SPACE_3,
                                  SPACE_4)
from src.periictal import config as _cfg
from src.periictal import passive as _passive
from src.periictal import stim_map as _sm
from src.periictal.embed import confound_readout, embed
from src.periictal.persist import build_matrix_cached
from src.utils.evoked_output import list_animals

# Where the signature-keyed matrix cache lives.
_CACHE_DIR = None       # set in register_callbacks from the repo derivatives root

# Colour-by choices that are categorical (one trace per level) vs continuous.
_CATEGORICAL = ("stim_key", "seizure_idx", "channel", "stim_status")
_CONT_LABEL = {"time_to_onset_sec": "hours to next onset",
               "hour_of_day": "hour of day"}

# --- background job registry (job_id -> state); results cached separately --- #
_JOBS: dict = {}
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 6
_LOCK = threading.Lock()


def _job_id(animal, protocol, variant, window_h, method, cap) -> str:
    return f"{animal}|{protocol}|{variant}|{window_h}|{method}|{cap}"


# --------------------------------------------------------------------- #
#  Background build
# --------------------------------------------------------------------- #

def _set(job_id: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {}).update(kw)


def _kick(store, evoked_dir, job_id, animal, protocol, variant,
          window_h, method, cap) -> None:
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
              window_h, method, cap))
    th.start()


def _worker(store, evoked_dir, job_id, animal, protocol, variant,
            window_h, method, cap) -> None:
    """Build the matrix + embedding for one job and cache the result. Never
    raises (records the error for the poll to surface)."""
    try:
        passive_cfg = None
        if variant == "passive":
            passive_cfg = _passive.passive_config()
            _set(job_id, progress=f"warming passive {protocol or 'all'}…")
            _passive.warm_passive(
                animal, evoked_dir, passive_cfg, protocol=protocol or None,
                progress=lambda d, n, fp: _set(
                    job_id, progress=f"passive sidecar {d}/{n}…"))
        _set(job_id, progress="joining seizures…")
        df = build_matrix_cached(
            store, animal, evoked_dir, _CACHE_DIR or _default_cache_dir(),
            protocol=protocol or None, window_sec=window_h * 3600.0,
            variant=variant, passive_cfg=passive_cfg)
        if df.empty:
            _finish(job_id, {"empty": True})
            return
        _set(job_id, progress=f"embedding {len(df)} stimuli ({method})…")
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
    controls = html.Div([
        _ctl("Animal", dcc.Dropdown(
            id="pex-animal", options=[{"label": a, "value": a} for a in animals],
            value=a0, clearable=False, style=DROPDOWN_STYLE)),
        _ctl("Protocol / stim group", dcc.Dropdown(
            id="pex-protocol", options=_protocol_options(store, a0),
            value=_default_protocol(store, a0),
            clearable=False, style=DROPDOWN_STYLE)),
        _ctl("Window", dcc.RadioItems(
            id="pex-variant", options=[{"label": " evoked (post-stim)", "value": "evoked"},
                                       {"label": " passive (pre-stim)", "value": "passive"}],
            value="evoked", inline=True)),
        _ctl("Lead-up (hours)", dcc.Input(
            id="pex-window-h", type="number", value=6, min=0.1, max=72, step=0.5,
            style={**DROPDOWN_STYLE, "width": "80px"})),
        _ctl("Embedding", dcc.RadioItems(
            id="pex-method", options=[{"label": " PCA", "value": "pca"},
                                      {"label": " UMAP", "value": "umap"}],
            value="pca", inline=True)),
        _ctl("Colour by", dcc.Dropdown(
            id="pex-colorby", clearable=False, style={**DROPDOWN_STYLE, "minWidth": "200px"})),
        html.Div(button("▶ Build", "pex-build", icon_name="play"),
                 style={"alignSelf": "flex-end"}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": SPACE_4,
              "alignItems": "flex-start"})

    return html.Div([
        card(section_header("Peri-ictal Evoked Explorer"),
             html.Div(id="pex-scope", style={"color": COLOR_TEXT_SECONDARY,
                                             "fontSize": "12px", "marginBottom": SPACE_3}),
             controls),
        card(dcc.Loading(custom_spinner=loading_icon("Building…"),
                         overlay_style={"visibility": "visible", "opacity": 0.4},
                         children=dcc.Graph(id="pex-graph",
                                            figure=empty_fig("Press ▶ Build"))),
             html.Div(id="pex-readout", style={"marginTop": SPACE_3}),
             html.Div(id="pex-status", style={"color": COLOR_TEXT_SECONDARY,
                                             "fontSize": "12px", "marginTop": SPACE_3}),
             style={"marginTop": SPACE_4}),
        dcc.Interval(id="pex-poll", interval=1200, disabled=True),
        dcc.Store(id="pex-job"),
    ], style={"padding": SPACE_4})


def _ctl(label, control):
    return html.Div([html.Div(label, style=LABEL_STYLE), control],
                    style={"display": "flex", "flexDirection": "column"})


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


# --------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------- #

def _figure(emb: np.ndarray, sub: pd.DataFrame, color_by: str) -> go.Figure:
    if emb is None or emb.shape[0] == 0:
        return empty_fig("No lead-up stimuli for this selection")
    if color_by in _CATEGORICAL:
        return _categorical_fig(emb, sub, color_by)
    return _continuous_fig(emb, sub, color_by)


def _continuous_fig(emb, sub, color_by) -> go.Figure:
    vals = sub[color_by].to_numpy(dtype=float)
    reverse = False
    scale, title = "Viridis", color_by
    if color_by == "time_to_onset_sec":
        vals, scale, reverse, title = vals / 3600.0, "Plasma", True, "hrs to onset"
    elif color_by == "hour_of_day":
        scale, title = "HSV", "hour of day"
    fig = go.Figure(go.Scattergl(
        x=emb[:, 0], y=emb[:, 1], mode="markers",
        marker=dict(size=4, opacity=0.55, color=vals, colorscale=scale,
                    reversescale=reverse, showscale=True,
                    colorbar=dict(title=title)),
        hovertext=[f"{title}={v:.2f}" for v in vals], hoverinfo="text"))
    return _finish_fig(fig)


def _categorical_fig(emb, sub, color_by) -> go.Figure:
    cats = pd.Categorical(sub[color_by].astype(str))
    fig = go.Figure()
    for k, name in enumerate(cats.categories):
        m = cats.codes == k
        fig.add_trace(go.Scattergl(
            x=emb[m, 0], y=emb[m, 1], mode="markers", name=str(name),
            marker=dict(size=4, opacity=0.55)))
    fig.update_layout(legend=dict(title=color_by, itemsizing="constant"))
    return _finish_fig(fig)


def _finish_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(l=30, r=20, t=20, b=30), height=520,
        xaxis=dict(title="dim 1", showticklabels=False),
        yaxis=dict(title="dim 2", showticklabels=False), dragmode="lasso")
    return fig


def _readout_view(res: dict) -> html.Div:
    ro = res.get("readout") or {}
    meta = res.get("meta") or {}
    chips = []
    for name, label in (("time_of_day", "circadian (hour-of-day)"),
                        ("stim_fingerprint", "stim fingerprint")):
        v = ro.get(name)
        txt = "n/a" if v is None else f"{v:.2f}"
        warn = v is not None and v >= 0.5
        chips.append(html.Span(
            f"{label}: |ρ| {txt}",
            style={"padding": f"2px {SPACE_3}", "marginRight": SPACE_3,
                   "borderRadius": "4px", "fontSize": "12px",
                   "background": (COLOR_WARNING if warn else "transparent"),
                   "color": ("#000" if warn else COLOR_TEXT_SECONDARY),
                   "border": f"1px solid {COLOR_WARNING if warn else COLOR_TEXT_SECONDARY}"}))
    note = ("  — high |ρ| means the embedding is that confound, not seizure "
            "proximity" if chips else "")
    ev = meta.get("explained_var")
    ev_txt = (f"  ·  PCA var: {', '.join(f'{x:.0%}' for x in ev)}"
              if ev else "")
    return html.Div([html.Span("Confound readout: ", style={"fontWeight": "600"}),
                     *chips, html.Span(note + ev_txt,
                                       style={"color": COLOR_TEXT_SECONDARY,
                                              "fontSize": "12px"})])


def _scope_text(res: dict, variant: str) -> str:
    n = res.get("n_seizures", 0)
    meta = res.get("meta") or {}
    pts = meta.get("n_points", 0)
    tot = meta.get("n_total", 0)
    sub = f"showing {pts:,} of {tot:,} stimuli" if tot and pts < tot else \
        f"{pts:,} stimuli"
    return (f"⚠ EXPLORATORY — effective n = {n} seizures (not the {tot:,} "
            f"stimuli). {sub}. {variant} features; UMAP/PCA is a figure, not a "
            f"significance test.")


# --------------------------------------------------------------------- #
#  Callbacks
# --------------------------------------------------------------------- #

def _default_cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "data", "derivatives", "periictal", "cache")


def register_callbacks(app, store, config):
    global _CACHE_DIR
    _CACHE_DIR = _default_cache_dir()
    evoked_dir = config.get("chronic_evoked", {}).get("evoked_output_dir", "")

    @app.callback(
        Output("pex-protocol", "options"),
        Output("pex-protocol", "value"),
        Output("pex-colorby", "options"),
        Output("pex-colorby", "value"),
        Input("pex-animal", "value"),
        Input("pex-variant", "value"),
        State("pex-colorby", "value"),
    )
    def _on_animal(animal, variant, cur_color):
        opts = _protocol_options(store, animal)
        pval = _default_protocol(store, animal)
        copts = _colorby_options(variant or "evoked")
        cvals = {o["value"] for o in copts}
        cval = cur_color if cur_color in cvals else "time_to_onset_sec"
        # protocol value only resets when the animal changed, not the variant.
        trig = callback_context.triggered_id
        return opts, (pval if trig == "pex-animal" else no_update), copts, cval

    @app.callback(
        Output("pex-graph", "figure"),
        Output("pex-readout", "children"),
        Output("pex-status", "children"),
        Output("pex-scope", "children"),
        Output("pex-poll", "disabled"),
        Output("pex-job", "data"),
        Input("pex-build", "n_clicks"),
        Input("pex-poll", "n_intervals"),
        State("pex-animal", "value"),
        State("pex-protocol", "value"),
        State("pex-variant", "value"),
        State("pex-window-h", "value"),
        State("pex-method", "value"),
        State("pex-colorby", "value"),
        prevent_initial_call=True,
    )
    def _build_or_poll(_n, _iv, animal, protocol, variant, window_h,
                       method, color_by):
        if not animal:
            return (no_update, no_update, "Pick an animal.", no_update, True,
                    no_update)
        window_h = float(window_h or 6.0)
        cap = _cfg.INTERACTIVE_POINT_CAP
        jid = _job_id(animal, protocol or "", variant, window_h, method, cap)
        with _LOCK:
            cached = _CACHE.get(jid)
            state = dict(_JOBS.get(jid) or {})
        if cached is not None:
            return _render_cached(cached, color_by, variant, jid)
        if state.get("status") == "error":
            return (empty_fig("Build failed"), "", state.get("progress", "error"),
                    no_update, True, no_update)
        # Not ready: ensure the build is running and keep polling.
        _kick(store, evoked_dir, jid, animal, protocol or "", variant,
              window_h, method, cap)
        prog = (_JOBS.get(jid) or {}).get("progress", "starting…")
        return (no_update, no_update, f"⏳ {prog}", no_update, False, no_update)

    @app.callback(
        Output("pex-graph", "figure", allow_duplicate=True),
        Input("pex-colorby", "value"),
        State("pex-job", "data"),
        prevent_initial_call=True,
    )
    def _recolor(color_by, jid):
        if not jid:
            return no_update
        with _LOCK:
            cached = _CACHE.get(jid)
        if not cached or cached.get("empty"):
            return no_update
        return _figure(cached["emb"], cached["sub"], color_by)


def _render_cached(cached, color_by, variant, jid):
    """(figure, readout, status, scope, poll_disabled, job) for a ready build."""
    if cached.get("empty"):
        msg = "No lead-up stimuli — <2 seizures, none in this protocol, or no fresh sidecars."
        return (empty_fig(msg), "", msg, "", True, jid)
    fig = _figure(cached["emb"], cached["sub"], color_by)
    return (fig, _readout_view(cached), "✓ built", _scope_text(cached, variant),
            True, jid)
