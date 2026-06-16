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
import os

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
    _render_hilbert_trace, _empty_lfp_fig, _session_dir_for_file,
    _stim_times_for_file, _stim_copy_channels)

_LANDMARKS = _grade.LANDMARKS  # ("EO","LAS","BO","PID","BB")


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


def _build_figures(store: Store, file_id: int, channel: int):
    """(lfp_fig, hilbert_fig) for a recording -- stim-blanked like the
    Video Review default. Falls back to placeholders on error."""
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
    try:
        hil, _status = _render_hilbert_trace(store, file_id, channel)
    except Exception:
        hil = _empty_lfp_fig("Hilbert failed to load.")
    return lfp, hil


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
                html.Span(id="training-ready-badge",
                          style={"marginLeft": "auto"}),
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
        # Ready badge for the CURRENT stage.
        badge = ""
        if email:
            scores = store.recent_training_scores(email, value,
                                                   limit=cfg["window"])
            if prog.get("certified_at"):
                badge = _pill("Certified ✓", COLOR_SUCCESS)
            elif value < 3 and value >= unlocked and _grade.rolling_ready(
                    scores, cfg["pass_pct"], cfg["window"]):
                badge = _pill("Ready to advance — ask your PI",
                              COLOR_WARNING)
            elif value == 3 and _grade.rolling_ready(
                    scores, cfg["pass_pct"], cfg["window"]):
                badge = _pill("Ready to certify — ask your PI",
                              COLOR_WARNING)
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
        Input("training-next-btn", "n_clicks"),
        prevent_initial_call="initial_duplicate",
    )
    def _load_example(_n):
        email = current_user_email()
        if not email:
            ph = _empty_lfp_fig("Sign in to start training.")
            return (None, _vid_placeholder("Sign in to start training."),
                    ph, ph, [], "", "")
        nxt = store.next_training_file(email)
        if not nxt:
            ph = _empty_lfp_fig("No validated recordings yet.")
            return (None,
                    _vid_placeholder("No validated recordings to train "
                                     "on yet — check back once the PI "
                                     "has approved some."),
                    ph, ph, [], "", "")
        file_id = int(nxt["file_id"])
        channel = _first_animal_channel(store, nxt.get("session_dir"))
        lfp, hil = _build_figures(store, file_id, channel)
        cur = {"file_id": file_id, "channel": channel,
               "validated": nxt.get("events") or [],
               "session_dir": nxt.get("session_dir"),
               "chunk_datetime": nxt.get("chunk_datetime")}
        vid = _vid_player(file_id)
        when = (nxt.get("chunk_datetime") or "")[:16]
        now = f"Now scoring: file #{file_id}" + (f" · {when}" if when
                                                  else "")
        return cur, vid, lfp, hil, [], "", now

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
        Input("training-submit-btn", "n_clicks"),
        State("training-stage", "value"),
        State("training-detect", "value"),
        State("training-events", "data"),
        State("training-current", "data"),
        State("training-refresh", "data"),
        prevent_initial_call=True,
    )
    def _submit(_n, stage, detect, events, current, refresh):
        if not _n:
            return no_update, no_update
        email = current_user_email()
        if not email:
            return _note("Sign in to submit.", COLOR_DANGER), no_update
        if not current:
            return _note("Load an example first.", COLOR_WARNING), no_update
        stage = int(stage or 1)
        validated = current.get("validated") or []
        if stage == 1:
            if detect not in ("yes", "no"):
                return _note("Pick yes or no first.",
                             COLOR_WARNING), no_update
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
            return _note(f"Save failed: {e}", COLOR_DANGER), no_update
        fb = _feedback(stage, score, result.get("breakdown", {}),
                       answer, validated)
        return fb, int(refresh or 0) + 1

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
            # Rolling mean in the current (highest unlocked) stage.
            scores = store.recent_training_scores(em, unlocked,
                                                   limit=cfg["window"])
            mean = (sum(scores) / len(scores)) if scores else 0.0
            ready = _grade.rolling_ready(scores, cfg["pass_pct"],
                                         cfg["window"])
            status = ("Certified" if certified
                      else ("Ready" if ready else "In progress"))
            label = ("Certify" if unlocked >= 3 else
                     f"Advance to {unlocked + 1}")
            body.append(html.Tr([
                html.Td(em, style=_TD),
                html.Td(f"{unlocked}. {STAGE_LABEL[unlocked]}", style=_TD),
                html.Td(f"{mean * 100:.0f}% (n={len(scores)})", style=_TD),
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
                ("Student", "Stage", "Recent agreement", "Status", "")])),
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
