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
                                  SPACE_2, SPACE_3, SPACE_4, SPACE_5)
from src.periictal import config as _cfg
from src.periictal import palette as _pal
from src.periictal import passive as _passive
from src.periictal import stim_map as _sm
from src.periictal.embed import confound_readout, embed
from src.periictal.persist import build_matrix_cached
from src.periictal.selection import summarize_selection
from src.preictal.isi import scored_seizures
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


def _job_id(animal, protocol, variant, window_h, method, cap) -> str:
    return f"{animal}|{protocol}|{variant}|{window_h}|{method}|{cap}"


def _tok(session_dir: str) -> str:
    return os.path.basename(session_dir or "").split("__", 1)[0]


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
                    job_id, progress=f"reading pre-stim traces… ({d}/{n})"))
        _set(job_id, progress="joining stimuli to seizures…")
        df = build_matrix_cached(
            store, animal, evoked_dir, _CACHE_DIR or _default_cache_dir(),
            protocol=protocol or None, window_sec=window_h * 3600.0,
            variant=variant, passive_cfg=passive_cfg)
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
        card(section_header("Selection"), _controls(store, animals, a0)),
        card(html.Div(id="pex-preview", style={"marginBottom": SPACE_3}),
             dcc.Loading(
                 custom_spinner=loading_icon("Building…"),
                 overlay_style={"visibility": "visible", "opacity": 0.4},
                 children=dcc.Graph(
                     id="pex-graph", clear_on_unhover=True,
                     config={"displaylogo": False},
                     figure=empty_fig("Press ▶ Build to embed the lead-up stimuli",
                                      hint="Pick an animal and protocol above."))),
             html.Div(id="pex-status", style={"color": COLOR_TEXT_SECONDARY,
                                              "fontSize": FONT_SIZE_CAPTION,
                                              "minHeight": "14px",
                                              "marginTop": SPACE_2}),
             html.Div(id="pex-reading", style={"marginTop": SPACE_3}),
             style={"marginTop": SPACE_4}),
        card(section_header("Selected points — details on demand"),
             html.Div(id="pex-details", children=_details_view(None)),
             style={"marginTop": SPACE_4}),
        dcc.Interval(id="pex-poll", interval=1200, disabled=True),
        dcc.Store(id="pex-job"),
    ], style={"padding": SPACE_4})


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
    return go.Figure(go.Scattergl(
        x=emb[:, 0], y=emb[:, 1], mode="markers", marker=marker,
        customdata=np.arange(emb.shape[0]),
        hovertemplate=f"{spec['label']}: %{{marker.color:.2f}}<extra></extra>"))


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

def _preview_panel(store, animal, protocol, variant, window_h) -> html.Div:
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
    return html.Div([
        html.Span("Ready — ", style={"color": COLOR_SUCCESS, "fontWeight": "600",
                                     "fontSize": FONT_SIZE_BODY}),
        html.Span(f"{animal} · {proto} · {variant} features · {window_h:g} h lead-up",
                  style={"color": COLOR_TEXT_PRIMARY, "fontSize": FONT_SIZE_BODY}),
        html.Span(f"    Effective n = {n} seizures. Press ▶ Build to embed their "
                  f"lead-up stimuli.",
                  style={"color": COLOR_TEXT_SECONDARY,
                         "fontSize": FONT_SIZE_CAPTION}),
    ])


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


def _render_cached(cached, color_by, jid):
    """(figure, reading, status, poll_disabled, job) for a ready build."""
    if cached.get("empty"):
        msg = ("No lead-up stimuli — fewer than 2 seizures, none in this protocol, "
               "or no fresh sidecars for this animal.")
        return empty_fig(msg), "", msg, True, jid
    fig = _figure(cached["emb"], cached["sub"], color_by,
                  cached.get("method", "pca"), cached.get("meta"))
    return fig, _reading_strip(cached), "✓ built", True, jid


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
        trig = callback_context.triggered_id
        return opts, (pval if trig == "pex-animal" else no_update), copts, cval

    @app.callback(
        Output("pex-preview", "children"),
        Input("pex-animal", "value"),
        Input("pex-protocol", "value"),
        Input("pex-variant", "value"),
        Input("pex-window-h", "value"),
    )
    def _preview(animal, protocol, variant, window_h):
        try:
            return _preview_panel(store, animal, protocol or "",
                                  variant or "evoked", float(window_h or 6.0))
        except Exception:                                     # noqa: BLE001
            return no_update

    @app.callback(
        Output("pex-graph", "figure"),
        Output("pex-reading", "children"),
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
        State("pex-colorby", "value"),
        prevent_initial_call=True,
    )
    def _build_or_poll(_n, _iv, animal, protocol, variant, window_h,
                       method, color_by):
        if not animal:
            return no_update, no_update, "Pick an animal.", True, no_update
        window_h = float(window_h or 6.0)
        cap = _cfg.INTERACTIVE_POINT_CAP
        jid = _job_id(animal, protocol or "", variant, window_h, method, cap)
        with _LOCK:
            cached = _CACHE.get(jid)
            state = dict(_JOBS.get(jid) or {})
        if cached is not None:
            return _render_cached(cached, color_by, jid)
        if state.get("status") == "error":
            return (empty_fig("Build failed", hint=state.get("progress", "")),
                    "", state.get("progress", "error"), True, no_update)
        _kick(store, evoked_dir, jid, animal, protocol or "", variant,
              window_h, method, cap)
        prog = (_JOBS.get(jid) or {}).get("progress", "starting…")
        return no_update, no_update, f"⏳ {prog}", False, no_update

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
