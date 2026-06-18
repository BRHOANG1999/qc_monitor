"""Training tab -- students blind-re-score PI-validated recordings and
get an agreement score vs the validated answer.

Three maturation stages (see src/utils/training.STAGES):
  1 Detection      -- is there a BHZ? (yes/no)
  2 Type + Racine  -- per event: onset type (LVF/HYP) + Racine, by EO
  3 Full           -- type + Racine + all five landmarks

Advancement is auto-flagged (rolling mean >= pass_pct over `window`) but
PI-confirmed. Examples are served unseen-first then recycle oldest-seen
(store.next_training_file). The validated answer is revealed only AFTER
the student submits.

All component ids are ``training-`` / ``tr-`` namespaced so nothing
collides with the Video Review tab, whose figure builders + media route
this tab reuses.
"""

from __future__ import annotations

import json
import logging
import os
import random

from dash import (Input, Output, State, dcc, html, no_update, ALL,
                   callback_context)

from src.dashboard.auth import current_user_email
from src.dashboard.components import (
    button, empty_state, LABEL_STYLE, card)
from src.dashboard.design import (
    COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER,
    COLOR_SURFACE_1, COLOR_SURFACE_2, COLOR_DIVIDER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY, RADIUS_MD, RADIUS_SM,
    SPACE_2, SPACE_3, SPACE_4, FONT_SIZE_BODY, FONT_SIZE_CAPTION)
from src.db.store import Store
from src.utils.animal import is_animal_channel
from src.utils import training as _grade

# Reused from the Video Review tab -- the LFP/Hilbert figure builders and
# the file-path lookup. video.py never imports this module, so no cycle.
from src.dashboard.tabs.video import (
    _file_path_for_id, _decimated_lfp, _build_lfp_figure,
    _render_hilbert_trace, _render_auc_trace, _empty_lfp_fig,
    _session_dir_for_file, _stim_times_for_file, _stim_copy_channels)

logger = logging.getLogger("qc_monitor.dashboard.training")

_LANDMARKS = _grade.LANDMARKS  # ("EO","LAS","BO","PID","BB")
_AUC_WINDOW_SEC = 5.0          # sliding-window AUC width for the LFP toggle


# ===================================================================== #
#  Helpers
# ===================================================================== #

def _is_pi(config: dict, email: str | None) -> bool:
    if not email:
        return False
    pis = (((config or {}).get("review_queue", {}) or {})
           .get("pi_emails", []) or [])
    return email.lower() in {e.lower() for e in pis}


def _cfg(config: dict) -> dict:
    t = (config or {}).get("training", {}) or {}
    return {
        "pass_pct": float(t.get("pass_pct", 85)),
        "window": int(t.get("window", 10)),
        "onset_tol_s": float(t.get("onset_tol_s", 10)),
        "racine_tol": int(t.get("racine_tol", 1)),
        "examples_per_round": int(t.get("examples_per_round", 10)),
    }


def _first_animal_channel(store: Store, session_dir: str | None) -> int:
    try:
        names = store._channel_names_for_session(session_dir)
    except Exception:
        return 0
    for i, n in enumerate(names or []):
        if isinstance(n, str) and is_animal_channel(n):
            return i
    return 0


def _filtered_trace(store: Store, file_id: int, channel: int,
                     mode: str):
    """The student's chosen 2nd-panel trace: raw Hilbert envelope (default)
    or its sliding-window AUC. Falls back to a placeholder on error."""
    try:
        if mode == "auc":
            fig, _s = _render_auc_trace(store, file_id, channel,
                                        _AUC_WINDOW_SEC)
        else:
            fig, _s = _render_hilbert_trace(store, file_id, channel)
        return fig
    except Exception:
        return _empty_lfp_fig("Trace failed to load.")


def _build_figures(store: Store, file_id: int, channel: int,
                    mode: str = "hilbert"):
    """(lfp_fig, filtered_fig) for a recording -- stim-blanked like the
    Video Review default. *mode* picks the 2nd panel (hilbert | auc)."""
    file_path = _file_path_for_id(store, file_id)
    if not file_path:
        ph = _empty_lfp_fig("Recording file not found.")
        return ph, ph
    try:
        session_dir = _session_dir_for_file(store, file_id)
        stim_copy = _stim_copy_channels(store, session_dir)
        do_blank = channel not in stim_copy
        stim_times = (_stim_times_for_file(store, file_id)
                      if do_blank else None)
        t, sig, dur, _n = _decimated_lfp(
            file_path, channel,
            stim_times=stim_times if (stim_times is not None
                                       and len(stim_times)) else None)
        lfp = _build_lfp_figure(t, sig, label=f"Ch{channel}",
                                 uirevision=f"train:{file_id}:{channel}")
    except Exception:
        lfp = _empty_lfp_fig("LFP failed to load.")
    return lfp, _filtered_trace(store, file_id, channel, mode)


def _ensure_round(store: Store, email: str, stage: int, n: int):
    """The open round for (email, stage), creating a balanced one when none
    is open. None when there are no candidate files."""
    rnd = store.current_training_round(email, stage)
    if rnd is not None:
        return rnd
    pool = store.training_candidate_pool(email)
    if not pool:
        return None
    seen = {c["file_id"]: (1 if c["last_seen"] else 0) for c in pool}
    files = _grade.select_round(pool, int(stage), n, seen, random.Random())
    if not files:
        return None
    return store.create_training_round(email, int(stage), files)


def _round_idx(store: Store, email: str, stage: int, rnd: dict) -> int:
    """Examples of this round the student has already submitted."""
    return len(store.round_scores(email, int(stage),
                                  rnd["started_after_id"]))


def _history_text(store: Store, email: str, stage: int, rnd=None) -> str:
    life = store.training_lifetime_stats(email, int(stage))
    bits = []
    if rnd is not None:
        idx = min(_round_idx(store, email, stage, rnd), rnd["examples"])
        bits.append(f"round {idx}/{rnd['examples']}")
    if life["n"]:
        bits.append(f"lifetime {life['mean'] * 100:.0f}% / {life['n']}")
    return "  ·  ".join(bits)


def _load_next(store: Store, email: str, stage: int, mode: str, n: int):
    """('loaded', payload) for the next round example, ('complete', rnd) when
    the round is finished, or None when no candidate files exist."""
    rnd = _ensure_round(store, email, stage, n)
    if rnd is None:
        return None
    idx = _round_idx(store, email, stage, rnd)
    if idx >= rnd["examples"]:
        return "complete", rnd
    file_id = int(rnd["files"][idx])
    session_dir = _session_dir_for_file(store, file_id)
    channel = _first_animal_channel(store, session_dir)
    lfp, hil = _build_figures(store, file_id, channel, mode)
    cur = {"file_id": file_id, "channel": channel,
           "validated": store.validated_events_for_file(file_id),
           "session_dir": session_dir}
    now = (f"Now scoring: example {idx + 1} of {rnd['examples']} "
           f"(file #{file_id})")
    return "loaded", (cur, _vid_player(file_id), lfp, hil, now,
                      _history_text(store, email, stage, rnd))


def _build_prompt(store: Store, email: str, stage: int,
                   rnd: dict) -> html.Div:
    scores = store.round_scores(email, int(stage), rnd["started_after_id"])
    rmean = (sum(scores) / len(scores) * 100) if scores else 0.0
    life = store.training_lifetime_stats(email, int(stage))
    lifetxt = (f"{life['mean'] * 100:.0f}% over {life['n']} attempts"
               if life["n"] else "—")
    return card(
        html.Div(f"Round complete — {rmean:.0f}% on these "
                 f"{rnd['examples']} examples (lifetime {lifetxt}).",
                 style={"fontWeight": "700", "color": COLOR_TEXT_PRIMARY,
                        "marginBottom": SPACE_3}),
        html.Div("How confident are you in your scoring now? (1 = unsure, "
                 "5 = very confident)",
                 style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginBottom": SPACE_2}),
        dcc.RadioItems(
            id="training-confidence",
            options=[{"label": f" {i}", "value": i} for i in range(1, 6)],
            value=3, inline=True,
            labelStyle={"marginRight": SPACE_3, "color": COLOR_TEXT_PRIMARY,
                         "fontSize": FONT_SIZE_BODY}),
        html.Div([
            button("Review more examples", "training-review-more-btn",
                   variant="secondary", style={"marginRight": SPACE_3}),
            button("I'm confident — move on", "training-moveon-btn",
                   variant="primary", tone="success"),
        ], style={"marginTop": SPACE_3}),
        html.Div(id="training-prompt-status",
                 style={"color": COLOR_TEXT_TERTIARY,
                        "fontSize": FONT_SIZE_CAPTION, "marginTop": SPACE_2}),
        style={"borderLeft": f"3px solid {COLOR_ACCENT}",
               "marginTop": SPACE_4},
    )


def _notify_pi(store: Store, config: dict, email: str, stage: int,
                rnd: dict) -> bool:
    """Email the PI(s) that a student finished a stage. Non-fatal."""
    pis = (((config or {}).get("review_queue", {}) or {})
           .get("pi_emails", []) or [])
    if not pis:
        return False
    scores = store.round_scores(email, int(stage), rnd["started_after_id"])
    rmean = (sum(scores) / len(scores) * 100) if scores else 0.0
    life = store.training_lifetime_stats(email, int(stage))
    lifetxt = (f"{life['mean'] * 100:.0f}% over {life['n']} attempts"
               if life["n"] else "—")
    label = STAGE_LABEL.get(int(stage), stage)
    subject = f"Training: {email} finished Stage {stage} ({label})"
    body = (f"{email} just finished a round of Stage {stage} ({label}) and "
            f"says they're confident.\n\n"
            f"Round score: {rmean:.0f}%\nLifetime: {lifetxt}\n\n"
            f"Open the Training tab -> Student progress (PI) to review their "
            f"performance and advance them.")
    try:
        from src.alerting.email_alert import EmailAlerter
        return EmailAlerter(config).send(subject, body, recipients=pis,
                                          subject_prefix=False)
    except Exception as e:  # noqa: BLE001 -- email must never break the UI
        logger.warning("PI training-stage email failed: %s", e)
        return False


def _events_table(events: list, title: str, color: str) -> html.Div:
    """Compact read-only listing of an event list for the feedback
    side-by-side (student vs validated)."""
    rows = []
    for i, e in enumerate(_grade._real_events(events)):
        bits = [f"#{i + 1}", e.get("type") or "?",
                f"R{e.get('racine')}" if e.get("racine") is not None
                else "R?"]
        eo = e.get("EO_sec")
        bits.append(f"EO {float(eo):.1f}s" if eo is not None else "EO ?")
        rows.append(html.Div(" · ".join(bits),
                             style={"fontSize": FONT_SIZE_CAPTION,
                                     "color": COLOR_TEXT_SECONDARY,
                                     "padding": "2px 0"}))
    if not rows:
        rows = [html.Div("No events.",
                         style={"fontSize": FONT_SIZE_CAPTION,
                                 "color": COLOR_TEXT_TERTIARY})]
    return html.Div([
        html.Div(title, style={"fontSize": FONT_SIZE_CAPTION,
                                "fontWeight": "700", "color": color,
                                "marginBottom": SPACE_2}),
        *rows,
    ], style={"flex": "1 1 220px", "padding": SPACE_3,
              "background": COLOR_SURFACE_1,
              "border": f"1px solid {COLOR_DIVIDER}",
              "borderRadius": RADIUS_SM})


# ===================================================================== #
#  Layout
# ===================================================================== #

def layout(store: Store, config: dict | None = None):
    email = current_user_email()
    is_pi = _is_pi(config or {}, email)

    return html.Div([
        html.H3("Training",
                style={"color": COLOR_TEXT_PRIMARY, "marginBottom": "2px"}),
        html.Div("Score recordings that already have a validated answer, "
                 "then see how closely you agreed. Work through the "
                 "stages; your PI signs off when you're consistent enough.",
                 style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_BODY, "marginBottom": SPACE_4}),

        # Stage selector + readiness badge.
        card(
            html.Div([
                html.Span("Stage:", style={**LABEL_STYLE,
                                            "display": "inline",
                                            "marginRight": SPACE_3}),
                dcc.RadioItems(
                    id="training-stage",
                    options=[{"label": f" {s}. {STAGE_LABEL[s]}",
                              "value": s} for s in (1, 2, 3)],
                    value=1, inline=True,
                    labelStyle={"marginRight": SPACE_4,
                                 "color": COLOR_TEXT_PRIMARY,
                                 "fontSize": FONT_SIZE_BODY}),
                html.Span(id="training-history",
                          style={"marginLeft": "auto",
                                  "color": COLOR_TEXT_TERTIARY,
                                  "fontSize": FONT_SIZE_CAPTION}),
                html.Span(id="training-ready-badge",
                          style={"marginLeft": SPACE_3}),
            ], style={"display": "flex", "alignItems": "center",
                       "flexWrap": "wrap", "gap": SPACE_2}),
            html.Div(id="training-stage-hint",
                      style={"color": COLOR_TEXT_TERTIARY,
                              "fontSize": FONT_SIZE_CAPTION,
                              "marginTop": SPACE_2}),
        ),

        # Viewer: video | LFP + Hilbert.
        html.Div([
            dcc.Loading(
                html.Div(id="training-video",
                          style={"background": "#000",
                                  "borderRadius": RADIUS_MD,
                                  "minHeight": "300px",
                                  "display": "flex",
                                  "alignItems": "center",
                                  "justifyContent": "center"}),
                type="default"),
            dcc.Loading(html.Div([
                dcc.Graph(id="training-lfp",
                          figure=_empty_lfp_fig("Load an example below."),
                          config={"displayModeBar": True,
                                   "displaylogo": False,
                                   "scrollZoom": True}),
                html.Div([
                    html.Span("2nd panel:", style={
                        "color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginRight": SPACE_2}),
                    dcc.RadioItems(
                        id="training-lfp-mode",
                        options=[{"label": " Hilbert envelope",
                                  "value": "hilbert"},
                                 {"label": " AUC", "value": "auc"}],
                        value="hilbert", inline=True,
                        labelStyle={"marginRight": SPACE_3,
                                     "color": COLOR_TEXT_PRIMARY,
                                     "fontSize": FONT_SIZE_CAPTION}),
                ], style={"display": "flex", "alignItems": "center",
                           "margin": f"{SPACE_2} 0"}),
                dcc.Graph(id="training-hilbert",
                          figure=_empty_lfp_fig(""),
                          config={"displayModeBar": True,
                                   "displaylogo": False,
                                   "scrollZoom": True}),
            ]), type="default"),
        ], style={"display": "grid",
                   "gridTemplateColumns": "minmax(320px,1fr) "
                                           "minmax(320px,1fr)",
                   "gap": SPACE_4, "marginTop": SPACE_4,
                   "marginBottom": SPACE_4}),

        html.Div([
            button("Load an example", "training-next-btn",
                   variant="primary", icon_name="video"),
            html.Span(id="training-now",
                      style={"marginLeft": SPACE_4,
                              "color": COLOR_TEXT_SECONDARY,
                              "fontSize": FONT_SIZE_CAPTION}),
        ], style={"display": "flex", "alignItems": "center",
                   "marginBottom": SPACE_4}),

        # Your impression (stage-dependent form).
        card(
            html.Div("Your impression",
                      style={**LABEL_STYLE, "marginBottom": SPACE_3}),
            html.Div(id="training-form"),
            html.Div([
                button("Submit", "training-submit-btn", variant="primary",
                       tone="success", icon_name="check-circle"),
            ], style={"marginTop": SPACE_3}),
            style={"marginBottom": SPACE_4},
        ),

        # Feedback (student vs validated), revealed after submit.
        html.Div(id="training-feedback"),

        # Post-round confidence prompt (hidden until a round completes).
        html.Div(id="training-prompt", style={"display": "none"}),

        # PI roster.
        html.Div(
            card(
                html.Div("Student progress (PI)",
                          style={**LABEL_STYLE, "marginBottom": SPACE_3}),
                html.Div(id="training-roster"),
            ) if is_pi else "",
            id="training-roster-wrap",
            style={"marginTop": SPACE_4} if is_pi else {"display": "none"},
        ),

        # State.
        dcc.Store(id="training-current"),          # {file_id, channel,
                                                   #  validated events, ...}
        dcc.Store(id="training-events", data=[]),  # student events
        dcc.Store(id="training-refresh", data=0),  # bump -> roster/badge
        dcc.Store(id="training-is-pi", data=bool(is_pi)),
    ])


STAGE_LABEL = {s: _grade.STAGES[s]["label"] for s in (1, 2, 3)}


# ===================================================================== #
#  Callbacks
# ===================================================================== #

def register_callbacks(app, store: Store, config: dict) -> None:
    cfg = _cfg(config)

    # ---- Stage access: disable locked stages for this student ---- #
    @app.callback(
        Output("training-stage", "options"),
        Output("training-stage", "value"),
        Output("training-stage-hint", "children"),
        Output("training-ready-badge", "children"),
        Input("training-refresh", "data"),
        State("training-stage", "value"),
    )
    def _stage_access(_refresh, cur_stage):
        email = current_user_email()
        prog = (store.get_training_progress(email) if email
                else {"unlocked_stage": 1, "certified_at": None})
        unlocked = int(prog.get("unlocked_stage", 1))
        opts = [{"label": f" {s}. {STAGE_LABEL[s]}", "value": s,
                 "disabled": s > unlocked} for s in (1, 2, 3)]
        value = cur_stage if (cur_stage and cur_stage <= unlocked) \
            else unlocked
        hint = _grade.STAGES[value]["scored"]
        badge = ""
        if email:
            # Readiness is round-scoped (resets when the student reviews more).
            rnd = store.current_training_round(email, value)
            scores = (store.round_scores(email, value, rnd["started_after_id"])
                      if rnd else
                      store.recent_training_scores(email, value,
                                                   limit=cfg["window"]))
            if prog.get("certified_at"):
                badge = _pill("Certified ✓", COLOR_SUCCESS)
            elif _grade.rolling_ready(scores, cfg["pass_pct"], cfg["window"]):
                badge = _pill(
                    "Ready to certify — ask your PI" if value == 3
                    else "Ready to advance — ask your PI", COLOR_WARNING)
        return opts, value, hint, badge

    # ---- Load the next example ---- #
    @app.callback(
        Output("training-current", "data"),
        Output("training-video", "children"),
        Output("training-lfp", "figure"),
        Output("training-hilbert", "figure"),
        Output("training-events", "data", allow_duplicate=True),
        Output("training-feedback", "children", allow_duplicate=True),
        Output("training-now", "children"),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Input("training-next-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-lfp-mode", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _load_example(_n, stage, mode):
        hide = {"display": "none"}
        email = current_user_email()
        if not email:
            ph = _empty_lfp_fig("Sign in to start training.")
            return (None, _vid_placeholder("Sign in to start training."),
                    ph, ph, [], "", "", "", hide, "")
        stage = int(stage or 1)
        res = _load_next(store, email, stage, mode or "hilbert",
                         cfg["examples_per_round"])
        if res is None:
            ph = _empty_lfp_fig("No validated recordings yet.")
            return (None, _vid_placeholder(
                "No validated recordings to train on yet — check back once "
                "the PI has approved some."), ph, ph, [], "", "", "", hide, "")
        kind, payload = res
        if kind == "complete":
            prompt = _build_prompt(store, email, stage, payload)
            return (no_update, no_update, no_update, no_update, no_update, "",
                    "Round complete — choose below.", prompt,
                    {"display": "block"},
                    _history_text(store, email, stage, payload))
        cur, vid, lfp, hil, now, hist = payload
        return cur, vid, lfp, hil, [], "", now, "", hide, hist

    # ---- 2nd-panel filter: Hilbert envelope vs AUC ---- #
    @app.callback(
        Output("training-hilbert", "figure", allow_duplicate=True),
        Input("training-lfp-mode", "value"),
        State("training-current", "data"),
        prevent_initial_call=True,
    )
    def _toggle_lfp_mode(mode, current):
        if not current or not current.get("file_id"):
            return no_update
        return _filtered_trace(store, int(current["file_id"]),
                               int(current.get("channel") or 0),
                               mode or "hilbert")

    # ---- Stage-dependent scoring form ---- #
    @app.callback(
        Output("training-form", "children"),
        Input("training-stage", "value"),
        Input("training-events", "data"),
    )
    def _render_form(stage, events):
        stage = int(stage or 1)
        if stage == 1:
            return dcc.RadioItems(
                id="training-detect",
                options=[{"label": " No BHZ in this recording",
                          "value": "no"},
                         {"label": " Yes — there is at least one BHZ",
                          "value": "yes"}],
                value=None,
                labelStyle={"display": "block", "color": COLOR_TEXT_PRIMARY,
                             "fontSize": FONT_SIZE_BODY,
                             "marginBottom": SPACE_2})
        # Stages 2 / 3: event rows + add button. training-detect still
        # present (hidden) so the submit State always resolves.
        rows = [_event_row(e, i, stage)
                for i, e in enumerate(events or [])]
        return html.Div([
            dcc.RadioItems(id="training-detect", options=[], value=None,
                            style={"display": "none"}),
            html.Div(rows or [html.Div(
                "No events added. If you think this recording has no "
                "BHZ, just submit. Otherwise add one.",
                style={"color": COLOR_TEXT_TERTIARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginBottom": SPACE_2})]),
            button("+ Add event", "training-add-btn", variant="secondary",
                   style={"marginTop": SPACE_2}),
        ])

    # ---- Events editor: add / remove / set fields ---- #
    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input("training-add-btn", "n_clicks"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _add_event(_n, events):
        if not _n:
            return no_update
        events = list(events or [])
        events.append({"type": "", "racine": None, "EO_sec": None,
                       "LAS_sec": None, "BO_sec": None, "PID_sec": None,
                       "BB_sec": None})
        return events

    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input({"type": "tr-remove", "idx": ALL}, "n_clicks"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _remove_event(_clicks, events):
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        events.pop(idx)
        return events

    @app.callback(
        Output("training-events", "data", allow_duplicate=True),
        Input({"type": "tr-type", "idx": ALL}, "value"),
        Input({"type": "tr-racine", "idx": ALL}, "value"),
        Input({"type": "tr-lm", "idx": ALL, "field": ALL}, "value"),
        State("training-events", "data"),
        prevent_initial_call=True,
    )
    def _set_fields(_types, _racines, _lms, events):
        events = list(events or [])
        if not events:
            return no_update
        changed = False
        for group in (callback_context.inputs_list or []):
            for item in group:
                cid = item.get("id") or {}
                idx = cid.get("idx")
                if idx is None or idx < 0 or idx >= len(events):
                    continue
                val = item.get("value")
                kind = cid.get("type")
                ev = dict(events[idx])
                if kind == "tr-type":
                    new = val or ""
                    if new != (ev.get("type") or ""):
                        ev["type"] = new
                        changed = True
                elif kind == "tr-racine":
                    new = int(val) if val not in (None, "") else None
                    if new != ev.get("racine"):
                        ev["racine"] = new
                        changed = True
                elif kind == "tr-lm":
                    field = f"{cid.get('field')}_sec"
                    new = float(val) if val not in (None, "") else None
                    if new != ev.get(field):
                        ev[field] = new
                        changed = True
                if changed:
                    events[idx] = ev
        return events if changed else no_update

    # ---- Submit: grade, persist, reveal feedback ---- #
    @app.callback(
        Output("training-feedback", "children"),
        Output("training-refresh", "data"),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Input("training-submit-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-detect", "value"),
        State("training-events", "data"),
        State("training-current", "data"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _submit(_n, stage, detect, events, current, refresh):
        nope = (no_update,) * 5
        if not _n:
            return nope
        email = current_user_email()
        if not email:
            return (_note("Sign in to submit.", COLOR_DANGER),) + nope[1:]
        if not current:
            return (_note("Load an example first.", COLOR_WARNING),) + nope[1:]
        stage = int(stage or 1)
        validated = current.get("validated") or []
        if stage == 1:
            if detect not in ("yes", "no"):
                return (_note("Pick yes or no first.", COLOR_WARNING),
                        ) + nope[1:]
            answer = {"events_present": detect == "yes", "events": []}
        else:
            answer = {"events_present": bool(events), "events": events or []}
        result = _grade.grade_attempt(
            stage, validated, answer,
            onset_tol_s=cfg["onset_tol_s"], racine_tol=cfg["racine_tol"])
        score = result["score"]
        try:
            store.add_training_attempt(
                email, stage, int(current["file_id"]),
                json.dumps(answer), float(score),
                json.dumps(result.get("breakdown")))
        except Exception as e:
            return (_note(f"Save failed: {e}", COLOR_DANGER),) + nope[1:]
        fb = _feedback(stage, score, result.get("breakdown", {}),
                       answer, validated)
        hide = {"display": "none"}
        rnd = store.current_training_round(email, stage)
        hist = _history_text(store, email, stage, rnd)
        nxt = int(refresh or 0) + 1
        if rnd is not None and _round_idx(store, email, stage,
                                          rnd) >= rnd["examples"]:
            return (fb, nxt, _build_prompt(store, email, stage, rnd),
                    {"display": "block"}, hist)
        return fb, nxt, no_update, hide, hist

    # ---- Post-round prompt: review more (reset grade) ---- #
    @app.callback(
        Output("training-current", "data", allow_duplicate=True),
        Output("training-video", "children", allow_duplicate=True),
        Output("training-lfp", "figure", allow_duplicate=True),
        Output("training-hilbert", "figure", allow_duplicate=True),
        Output("training-events", "data", allow_duplicate=True),
        Output("training-feedback", "children", allow_duplicate=True),
        Output("training-now", "children", allow_duplicate=True),
        Output("training-prompt", "children", allow_duplicate=True),
        Output("training-prompt", "style", allow_duplicate=True),
        Output("training-history", "children", allow_duplicate=True),
        Output("training-refresh", "data", allow_duplicate=True),
        Input("training-review-more-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-lfp-mode", "value"),
        State("training-confidence", "value"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _review_more(_n, stage, mode, confidence, refresh):
        if not _n:
            return (no_update,) * 11
        email = current_user_email()
        stage = int(stage or 1)
        hide = {"display": "none"}
        bump = int(refresh or 0) + 1
        rnd = store.current_training_round(email, stage)
        if rnd is not None:
            # Close the round -> a fresh round starts a new reset boundary,
            # so the current grade restarts (history is kept).
            store.finish_training_round(rnd["round_id"], confidence,
                                        "review_more")
        res = _load_next(store, email, stage, mode or "hilbert",
                         cfg["examples_per_round"])
        if res is None or res[0] != "loaded":
            return ((no_update,) * 7 + ("", hide,
                    _history_text(store, email, stage, None), bump))
        cur, vid, lfp, hil, now, hist = res[1]
        return cur, vid, lfp, hil, [], "", now, "", hide, hist, bump

    # ---- Post-round prompt: move on (notify PI) ---- #
    @app.callback(
        Output("training-prompt-status", "children"),
        Output("training-refresh", "data", allow_duplicate=True),
        Input("training-moveon-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-confidence", "value"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _move_on(_n, stage, confidence, refresh):
        if not _n:
            return no_update, no_update
        email = current_user_email()
        stage = int(stage or 1)
        rnd = store.current_training_round(email, stage)
        if rnd is None:
            return "No active round.", no_update
        store.finish_training_round(rnd["round_id"], confidence, "move_on")
        sent = (_notify_pi(store, config, email, stage, rnd)
                if store.mark_round_notified(rnd["round_id"]) else False)
        msg = ("Sent to your PI — they'll review your performance and "
               "advance you." if sent else
               "Recorded. Your PI reviews and gates the next stage.")
        return msg, int(refresh or 0) + 1

    # ---- PI roster ---- #
    @app.callback(
        Output("training-roster", "children"),
        Input("training-refresh", "data"),
        State("training-is-pi", "data"),
    )
    def _roster(_refresh, is_pi):
        if not is_pi:
            return no_update
        roster = store.all_training_progress()
        if not roster:
            return empty_state("No students yet",
                                hint="Attempts appear here as students "
                                     "start training.", icon_name="inbox")
        body = []
        for r in roster:
            em = r["student_email"]
            unlocked = int(r.get("unlocked_stage", 1))
            certified = bool(r.get("certified_at"))
            # Current grade is round-scoped (resets when they review more);
            # lifetime is the full history.
            rnd = store.current_training_round(em, unlocked)
            scores = (store.round_scores(em, unlocked,
                                         rnd["started_after_id"]) if rnd else
                      store.recent_training_scores(em, unlocked,
                                                   limit=cfg["window"]))
            mean = (sum(scores) / len(scores)) if scores else 0.0
            ready = _grade.rolling_ready(scores, cfg["pass_pct"],
                                         cfg["window"])
            life = store.training_lifetime_stats(em, unlocked)
            lifetxt = (f"{life['mean'] * 100:.0f}% / {life['n']}"
                       if life["n"] else "—")
            status = ("Certified" if certified
                      else ("Ready" if ready else "In progress"))
            label = ("Certify" if unlocked >= 3 else
                     f"Advance to {unlocked + 1}")
            body.append(html.Tr([
                html.Td(em, style=_TD),
                html.Td(f"{unlocked}. {STAGE_LABEL[unlocked]}", style=_TD),
                html.Td(f"{mean * 100:.0f}% (n={len(scores)})", style=_TD),
                html.Td(lifetxt, style=_TD),
                html.Td(status, style={**_TD, "color": (
                    COLOR_SUCCESS if certified or ready
                    else COLOR_TEXT_SECONDARY)}),
                html.Td(button(
                    label, {"type": "training-advance", "email": em},
                    variant="secondary",
                    style={"padding": "2px 10px",
                            "fontSize": FONT_SIZE_CAPTION},
                    **({"disabled": True} if certified else {})),
                    style=_TD),
            ]))
        return html.Table([
            html.Thead(html.Tr([
                html.Th(h, style=_TH) for h in
                ("Student", "Stage", "Round grade", "Lifetime", "Status",
                 "")])),
            html.Tbody(body),
        ], style={"width": "100%", "borderCollapse": "collapse"})

    # ---- PI advance ---- #
    @app.callback(
        Output("training-refresh", "data", allow_duplicate=True),
        Input({"type": "training-advance", "email": ALL}, "n_clicks"),
        State("training-refresh", "data"),
        State("training-is-pi", "data"),
        prevent_initial_call=True,
    )
    def _advance(_clicks, refresh, is_pi):
        if not is_pi:
            return no_update
        if not any((t.get("value") or 0)
                   for t in (callback_context.triggered or [])):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        email = trig.get("email")
        if not email:
            return no_update
        prog = store.get_training_progress(email)
        store.advance_student(email, int(prog.get("unlocked_stage", 1)) + 1)
        return int(refresh or 0) + 1


# ===================================================================== #
#  Small render helpers (module-level; no app/store needed)
# ===================================================================== #

_TD = {"padding": f"{SPACE_2} {SPACE_3}",
       "borderBottom": f"1px solid {COLOR_DIVIDER}",
       "fontSize": FONT_SIZE_BODY, "color": COLOR_TEXT_PRIMARY}
_TH = {"padding": f"{SPACE_2} {SPACE_3}",
       "borderBottom": f"1px solid {COLOR_DIVIDER}",
       "fontSize": FONT_SIZE_CAPTION, "color": COLOR_TEXT_SECONDARY,
       "textAlign": "left", "textTransform": "uppercase",
       "letterSpacing": "0.4px"}


def _pill(text: str, color: str) -> html.Span:
    return html.Span(text, style={
        "padding": "3px 10px", "borderRadius": "10px",
        "background": color, "color": "#10120a" if color == COLOR_WARNING
        else "white", "fontSize": FONT_SIZE_CAPTION, "fontWeight": "700"})


def _note(text: str, color: str) -> html.Div:
    return html.Div(text, style={"color": color, "fontSize": FONT_SIZE_BODY,
                                  "padding": SPACE_2})


def _vid_placeholder(text: str) -> html.Div:
    return empty_state("No recording loaded", hint=text, icon_name="video")


def _vid_player(file_id: int) -> html.Video:
    return html.Video(
        src=f"/media/video/{file_id}",
        controls=True, preload="metadata",
        style={"width": "100%", "maxHeight": "360px",
               "borderRadius": RADIUS_MD, "background": "#000"})


def _event_row(event: dict, idx: int, stage: int) -> html.Div:
    type_opts = ([{"label": " LVF", "value": "LVF"},
                  {"label": " HYP", "value": "HYP"}]
                 + ([{"label": " Undefined", "value": "Undefined"}]
                    if stage == 3 else []))
    controls = [
        html.Span(f"Event {idx + 1}",
                  style={"fontWeight": "700", "color": COLOR_TEXT_PRIMARY,
                          "fontSize": FONT_SIZE_BODY,
                          "marginRight": SPACE_3}),
        dcc.RadioItems(id={"type": "tr-type", "idx": idx},
                        options=type_opts, value=event.get("type") or None,
                        inline=True,
                        labelStyle={"marginRight": SPACE_3,
                                     "color": COLOR_TEXT_PRIMARY,
                                     "fontSize": FONT_SIZE_CAPTION}),
        html.Span("Racine", style={"color": COLOR_TEXT_SECONDARY,
                                     "fontSize": FONT_SIZE_CAPTION,
                                     "marginRight": SPACE_2}),
        dcc.Dropdown(id={"type": "tr-racine", "idx": idx},
                      options=[{"label": str(n), "value": n}
                               for n in range(1, 9)],
                      value=event.get("racine"),
                      style={"width": "70px"}, className="dark-dropdown"),
    ]
    lms = _LANDMARKS if stage == 3 else ("EO",)
    for lm in lms:
        controls.append(html.Span(
            lm, style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginLeft": SPACE_2, "marginRight": "2px"}))
        controls.append(dcc.Input(
            id={"type": "tr-lm", "idx": idx, "field": lm},
            type="number", value=event.get(f"{lm}_sec"),
            placeholder="s", step="any",
            style={"width": "70px", "backgroundColor": COLOR_SURFACE_2,
                   "color": COLOR_TEXT_PRIMARY,
                   "border": f"1px solid {COLOR_DIVIDER}",
                   "borderRadius": RADIUS_SM, "padding": "3px 6px"}))
    controls.append(button("✕", {"type": "tr-remove", "idx": idx},
                            variant="ghost",
                            style={"marginLeft": "auto",
                                    "padding": "2px 8px"}))
    return html.Div(controls, style={
        "display": "flex", "alignItems": "center", "flexWrap": "wrap",
        "gap": f"{SPACE_2} {SPACE_2}", "padding": f"{SPACE_2} {SPACE_3}",
        "marginBottom": SPACE_2, "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}", "borderRadius": RADIUS_SM})


def _feedback(stage, score, breakdown, answer, validated) -> html.Div:
    pct = round(score * 100)
    color = (COLOR_SUCCESS if pct >= 85 else
             COLOR_WARNING if pct >= 60 else COLOR_DANGER)
    if stage == 1:
        you = ("events present" if answer.get("events_present")
               else "no events")
        val = ("events present" if _grade._real_events(validated)
               else "no events")
        detail = html.Div(f"You said: {you}  ·  Validated: {val}",
                          style={"color": COLOR_TEXT_SECONDARY,
                                  "fontSize": FONT_SIZE_CAPTION,
                                  "marginTop": SPACE_2})
    else:
        b = breakdown or {}
        detail = html.Div([
            html.Div(
                f"matched {b.get('matched', 0)} · missed "
                f"{b.get('missed', 0)} · extra {b.get('false_pos', 0)} "
                f"(of {b.get('validated', 0)} validated)",
                style={"color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginTop": SPACE_2}),
            html.Div([
                _events_table(answer.get("events") or [], "Your answer",
                              COLOR_ACCENT),
                _events_table(validated, "Validated answer", COLOR_SUCCESS),
            ], style={"display": "flex", "gap": SPACE_3,
                       "flexWrap": "wrap", "marginTop": SPACE_3}),
        ])
    return card(
        html.Div([
            html.Span(f"{pct}%", style={"fontSize": "28px",
                                         "fontWeight": "800",
                                         "color": color}),
            html.Span(" agreement with the validated answer",
                      style={"color": COLOR_TEXT_SECONDARY,
                              "fontSize": FONT_SIZE_BODY,
                              "marginLeft": SPACE_2}),
        ]),
        detail,
        html.Div("Use “Load an example” above for the next recording.",
                 style={"color": COLOR_TEXT_TERTIARY,
                         "fontSize": FONT_SIZE_CAPTION,
                         "marginTop": SPACE_3}),
        style={"borderLeft": f"3px solid {color}", "marginTop": SPACE_4},
    )
