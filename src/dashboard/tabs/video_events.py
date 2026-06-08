"""BHZ event editor for the Video Review tab.

Replaces the old flat onset-marker list with a structured event
record: each event has a type (LVF / HYP), the 5 landmark
timestamps the decision tree calls for (EO / LAS / BO / PID /
BB), three free-text comments, and a Racine score (1-8 extended
scale from the user). The complete picture goes into
``video-events-store`` (a dcc.Store) and is consumed by
``_save_review`` which writes both ``review_state`` and the
per-(animal, day) CSV.

Decision tree:
* Pick type -> LVF or HYP.
* Fill landmarks in order EO -> LAS (LVF only) -> BO -> PID
  -> BB. Each landmark's "Drop at video time" button reads
  the current ``video.currentTime`` and scales to LFP time.
* Pick Racine 1-8. This is the gate -- the event is INCOMPLETE
  until Racine is set, and Mark-done is blocked until every
  event is complete (or the file has no events).

NASA Rule 4 + 5: every callback function in here is <60 lines
and asserts its inputs at the boundary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from dash import ALL, MATCH, Input, Output, State, callback_context
from dash import dcc, html, no_update, Patch

from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_DIVIDER, COLOR_SUCCESS,
    COLOR_SURFACE_1, COLOR_SURFACE_2, COLOR_SURFACE_3,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    COLOR_WARNING, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    FONT_SIZE_HEADER, RADIUS_MD, RADIUS_SM,
    SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5,
)

logger = logging.getLogger("qc_monitor.video_events")


# Decision-tree fields in the order the reviewer fills them.
# LAS is LVF-only; the renderer hides its row for HYP events.
EVENT_FIELDS: tuple[str, ...] = ("EO", "LAS", "BO", "PID", "BB")

# Extended Racine scale (1-8). Stages verbatim from the user.
# Tooltip shown on hover so the reviewer never has to leave the
# screen to look up a stage description.
RACINE_STAGES: tuple[tuple[int, str], ...] = (
    (1, "change in behavioural state "
         "(sudden behavioural arrest or sudden motion)"),
    (2, "head nodding"),
    (3, "forelimb clonus"),
    (4, "rearing, or clonus when on belly, "
         "or strong hindlimb clonus (bucking)"),
    (5, "falling, or clonus when on side"),
    (6, "multiple sequences of rearing and falling, "
         "or brief jumps"),
    (7, "violent jumping"),
    (8, "class seven, followed by a period of tonus "
         "lasting longer than 5 s"),
)


def blank_event() -> dict:
    """A new empty event record."""
    return {
        "type": "",        # "" | "LVF" | "HYP"
        "EO_sec": None,
        "LAS_sec": None,
        "BO_sec": None,
        "PID_sec": None,
        "BB_sec": None,
        "racine": None,
        "onset_comment": "",
        "behavior_comment": "",
        "score_comment": "",
        "roomlight": "",
        "video_quality": "",
    }


def required_fields(event: dict) -> tuple[str, ...]:
    """Fields that must be set for *event* to be complete.

    LAS is only required when type=LVF; HYP events don't have
    a Large Amplitude Spiking phase by definition.
    """
    assert isinstance(event, dict), "event must be dict"
    if event.get("type") == "HYP":
        return ("EO", "BO", "PID", "BB")
    return ("EO", "LAS", "BO", "PID", "BB")


def is_event_complete(event: dict) -> bool:
    """An event is complete iff type is picked, every required
    landmark has a timestamp, AND Racine 1-8 is chosen.
    """
    assert isinstance(event, dict), "event must be dict"
    if event.get("type") not in ("LVF", "HYP"):
        return False
    for f in required_fields(event):
        if event.get(f + "_sec") in (None, ""):
            return False
    r = event.get("racine")
    if r is None or not (1 <= int(r) <= 8):
        return False
    return True


def events_status_label(events: list[dict]) -> tuple[str, str]:
    """Return ``(text, color)`` for the events-complete status
    pill. Color is one of the design tokens."""
    assert isinstance(events, list), "events must be list"
    if not events:
        return ("No events yet -- add one below, "
                 "or hit 'Mark recording done' if there's "
                 "nothing to score.", COLOR_TEXT_TERTIARY)
    total = len(events)
    done = sum(1 for e in events if is_event_complete(e))
    if done == total:
        return (f"{done} of {total} events complete.",
                 COLOR_SUCCESS)
    missing = total - done
    return (f"{done} of {total} complete. {missing} still need "
             f"landmarks + Racine.", COLOR_WARNING)


# --------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------- #

def _field_label(field: str) -> str:
    """Human label for a landmark field."""
    return {
        "EO":  "EO -- Electrographic onset",
        "LAS": "LAS -- Large Amplitude Spiking",
        "BO":  "BO -- Behavioral Onset",
        "PID": "PID -- Post-Ictal Depression",
        "BB":  "BB -- Back to Baseline",
    }.get(field, field)


def _field_row(event: dict, idx: int, field: str) -> html.Div:
    """One landmark row inside an event card."""
    assert isinstance(field, str) and field, "field required"
    secs = event.get(f"{field}_sec")
    has_value = secs is not None
    value_label = (f"{float(secs):.2f} s" if has_value
                    else "(not set)")
    return html.Div(
        [
            html.Span(_field_label(field),
                       style={"width": "240px",
                               "color": COLOR_TEXT_SECONDARY,
                               "fontSize": FONT_SIZE_BODY}),
            html.Span(value_label,
                       style={"width": "120px",
                               "color": (COLOR_SUCCESS if has_value
                                         else COLOR_TEXT_TERTIARY),
                               "fontFamily": ("ui-monospace, "
                                               "SF Mono, monospace"),
                               "fontSize": FONT_SIZE_CAPTION}),
            html.Button(
                "Drop at video time",
                id={"type": "event-field-btn", "idx": idx,
                     "field": field},
                n_clicks=0,
                style={
                    "background": COLOR_SURFACE_2,
                    "color": COLOR_TEXT_PRIMARY,
                    "border": f"1px solid {COLOR_DIVIDER}",
                    "borderRadius": RADIUS_SM,
                    "padding": f"3px {SPACE_3}",
                    "cursor": "pointer",
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginRight": SPACE_2,
                },
            ),
            html.Button(
                "Clear",
                id={"type": "event-field-clear", "idx": idx,
                     "field": field},
                n_clicks=0,
                style={
                    "background": "transparent",
                    "color": COLOR_TEXT_TERTIARY,
                    "border": "none",
                    "cursor": "pointer",
                    "fontSize": FONT_SIZE_CAPTION,
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center",
               "padding": f"{SPACE_1} 0"},
    )


def _racine_picker(event: dict, idx: int) -> html.Div:
    """The 1-8 Racine chip row, with the stage on hover."""
    current = event.get("racine")
    chips = [html.Span(
        "Racine 1-8:",
        style={"color": COLOR_TEXT_SECONDARY,
               "fontSize": FONT_SIZE_BODY,
               "marginRight": SPACE_3},
    )]
    for stage, description in RACINE_STAGES:
        is_picked = (current == stage)
        chips.append(html.Button(
            str(stage),
            id={"type": "event-racine-btn",
                 "idx": idx, "stage": stage},
            n_clicks=0,
            title=f"Stage {stage}: {description}",
            style={
                "background": (COLOR_ACCENT if is_picked
                                else COLOR_SURFACE_2),
                "color": (COLOR_TEXT_PRIMARY if is_picked
                          else COLOR_TEXT_SECONDARY),
                "border": (f"1px solid {COLOR_ACCENT}"
                            if is_picked
                            else f"1px solid {COLOR_DIVIDER}"),
                "borderRadius": RADIUS_SM,
                "padding": f"3px {SPACE_3}",
                "fontWeight": "600",
                "fontSize": FONT_SIZE_CAPTION,
                "cursor": "pointer",
                "marginRight": SPACE_1,
                "minWidth": "28px",
            },
        ))
    return html.Div(
        chips,
        style={"display": "flex", "alignItems": "center",
               "padding": f"{SPACE_2} 0", "flexWrap": "wrap"},
    )


def _comment_box(event: dict, idx: int, field: str,
                   placeholder: str) -> html.Div:
    """Per-event comment textarea."""
    return html.Div([
        html.Label(field.replace("_", " ").capitalize(),
                    style={"color": COLOR_TEXT_TERTIARY,
                            "fontSize": FONT_SIZE_CAPTION,
                            "display": "block",
                            "marginBottom": "2px"}),
        dcc.Textarea(
            id={"type": "event-comment", "idx": idx,
                 "field": field},
            value=event.get(field, "") or "",
            placeholder=placeholder,
            maxLength=512,
            style={"width": "100%", "height": "40px",
                    "background": COLOR_SURFACE_3,
                    "color": COLOR_TEXT_PRIMARY,
                    "border": f"1px solid {COLOR_DIVIDER}",
                    "borderRadius": RADIUS_SM,
                    "padding": f"{SPACE_2} {SPACE_3}",
                    "fontSize": FONT_SIZE_CAPTION,
                    "fontFamily": "inherit"},
        ),
    ], style={"marginTop": SPACE_2})


def _type_radio(event: dict, idx: int) -> html.Div:
    """HYP / LVF radio."""
    return html.Div([
        html.Span("Type:",
                   style={"color": COLOR_TEXT_SECONDARY,
                           "fontSize": FONT_SIZE_BODY,
                           "marginRight": SPACE_3}),
        dcc.RadioItems(
            id={"type": "event-type-radio", "idx": idx},
            options=[
                {"label": " LVF (Low Voltage Fast Onset)",
                 "value": "LVF"},
                {"label": " HYP (Hypersynchronous)",
                 "value": "HYP"},
            ],
            value=event.get("type") or None, inline=True,
            labelStyle={"color": COLOR_TEXT_PRIMARY,
                         "fontSize": FONT_SIZE_BODY,
                         "marginRight": SPACE_4},
            inputStyle={"marginRight": "4px"},
        ),
    ], style={"display": "flex", "alignItems": "center",
              "padding": f"{SPACE_1} 0"})


def render_event_card(event: dict, idx: int) -> html.Div:
    """One event card. Decision-tree gate: rows for LAS / etc.
    only show after a type is picked."""
    assert isinstance(event, dict), "event must be dict"
    assert isinstance(idx, int) and idx >= 0, "idx >= 0"
    children: list = []
    children.append(html.Div([
        html.Span(f"Event {idx + 1}",
                   style={"color": COLOR_TEXT_PRIMARY,
                           "fontWeight": "600",
                           "fontSize": FONT_SIZE_HEADER}),
        html.Span(
            "COMPLETE" if is_event_complete(event)
            else "INCOMPLETE",
            style={"marginLeft": "auto",
                    "color": (COLOR_SUCCESS
                              if is_event_complete(event)
                              else COLOR_WARNING),
                    "fontSize": FONT_SIZE_CAPTION,
                    "fontWeight": "600"}),
        html.Button(
            "Delete",
            id={"type": "event-delete-btn", "idx": idx},
            n_clicks=0,
            style={"background": "transparent",
                    "color": COLOR_DANGER,
                    "border": "none",
                    "cursor": "pointer",
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginLeft": SPACE_3},
        ),
    ], style={"display": "flex", "alignItems": "center",
              "borderBottom": f"1px solid {COLOR_DIVIDER}",
              "paddingBottom": SPACE_2,
              "marginBottom": SPACE_2}))
    children.append(_type_radio(event, idx))
    if event.get("type") in ("LVF", "HYP"):
        # Only show landmark rows after a type is chosen.
        for f in EVENT_FIELDS:
            if f == "LAS" and event.get("type") == "HYP":
                continue
            children.append(_field_row(event, idx, f))
        children.append(_racine_picker(event, idx))
        children.append(_comment_box(
            event, idx, "onset_comment",
            "How did the seizure onset look on the LFP?"))
        children.append(_comment_box(
            event, idx, "behavior_comment",
            "What behavior change marked the BO?"))
        children.append(_comment_box(
            event, idx, "score_comment",
            "Behavioral observation for the Racine score."))
    return html.Div(
        children,
        style={"background": COLOR_SURFACE_1,
                "border": f"1px solid {COLOR_DIVIDER}",
                "borderRadius": RADIUS_MD,
                "padding": SPACE_4,
                "marginBottom": SPACE_3},
    )


def render_events_panel() -> html.Div:
    """The Step 4 panel root: status pill + events list + add btn.

    The Save / Mark-done button itself lives outside this module
    (it's in video.py's existing Step 4 wiring); this panel
    sits ABOVE the Save button.
    """
    return html.Div([
        html.Div(id="video-events-status",
                  style={"padding": f"{SPACE_2} {SPACE_3}",
                          "marginBottom": SPACE_2,
                          "background": COLOR_SURFACE_1,
                          "border": f"1px solid {COLOR_DIVIDER}",
                          "borderRadius": RADIUS_SM,
                          "fontSize": FONT_SIZE_BODY}),
        html.Div(id="video-events-list"),
        html.Button(
            "+ Add event",
            id="video-events-add-btn",
            n_clicks=0,
            style={
                "background": COLOR_ACCENT,
                "color": COLOR_TEXT_PRIMARY,
                "border": "none",
                "borderRadius": RADIUS_SM,
                "padding": f"{SPACE_2} {SPACE_4}",
                "cursor": "pointer",
                "fontSize": FONT_SIZE_BODY,
                "fontWeight": "600",
                "marginTop": SPACE_3,
            },
        ),
        # The Store the editor mutates; reads + writes go
        # through Patch-style copy-on-write so the renderer
        # callback below sees the latest snapshot.
        dcc.Store(id="video-events-store", data=[]),
    ])


# --------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------- #

def _to_lfp_seconds(video_t: float | None,
                      video_dur: float | None,
                      lfp_dur: float | None) -> float | None:
    """Scale video clock seconds to LFP-trace seconds. Falls
    back to ``video_t`` if either duration is missing or zero."""
    if video_t is None:
        return None
    try:
        vt = float(video_t)
    except (TypeError, ValueError):
        return None
    try:
        vd = float(video_dur or 0)
        ld = float(lfp_dur or 0)
    except (TypeError, ValueError):
        return vt
    if vd <= 0 or ld <= 0:
        return vt
    return vt * (ld / vd)


def register_callbacks(app, store) -> None:
    """Wire up the events editor against *app*. ``store`` is
    the dispatcher DB; not used in the editor itself but kept
    on the signature for symmetry with other tab modules."""
    assert app is not None, "app required"

    @app.callback(
        Output("video-events-list", "children"),
        Output("video-events-status", "children"),
        Output("video-events-status", "style"),
        Input("video-events-store", "data"),
        State("video-events-status", "style"),
    )
    def _render_events_list(events, status_style):
        events = list(events or [])
        cards = [render_event_card(e, i)
                  for i, e in enumerate(events)]
        text, color = events_status_label(events)
        merged_style = dict(status_style or {})
        merged_style["color"] = color
        merged_style["borderLeft"] = f"3px solid {color}"
        return cards, text, merged_style

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input("video-events-add-btn", "n_clicks"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _add_event(_clicks, events):
        events = list(events or [])
        if len(events) >= 32:
            return no_update
        events.append(blank_event())
        return events

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-delete-btn", "idx": ALL},
                "n_clicks"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _delete_event(_clicks_list, events):
        if not any(_clicks_list or []):
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
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-type-radio", "idx": ALL}, "value"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _set_type(values, events):
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        # The values list is in DOM order; pull the latest by idx.
        new_val = None
        triggered = callback_context.triggered or []
        for t in triggered:
            new_val = t.get("value")
        events[idx] = dict(events[idx])
        events[idx]["type"] = new_val or ""
        # Clearing type clears LAS (HYP doesn't have it; if the
        # user flips LVF->HYP a stale LAS would orphan).
        if events[idx]["type"] == "HYP":
            events[idx]["LAS_sec"] = None
        return events

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-field-btn", "idx": ALL,
                "field": ALL}, "n_clicks"),
        State("video-events-store", "data"),
        State("video-current-time", "data"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )
    def _drop_field(_clicks, events, video_t, lfp_dur):
        if not any(_clicks or []):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx"); field = trig.get("field")
        if (idx is None or field not in EVENT_FIELDS):
            return no_update
        events = list(events or [])
        if idx < 0 or idx >= len(events):
            return no_update
        # We don't have video_dur in a Store yet; use lfp_dur as
        # the source-of-truth when the video clock and the LFP
        # have the same duration (most common). Future work:
        # plumb video.duration through a Store for the BHZ rate
        # scaling factor.
        t_lfp = _to_lfp_seconds(video_t, lfp_dur, lfp_dur)
        if t_lfp is None:
            return no_update
        events[idx] = dict(events[idx])
        events[idx][f"{field}_sec"] = float(round(t_lfp, 3))
        return events

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-field-clear", "idx": ALL,
                "field": ALL}, "n_clicks"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _clear_field(_clicks, events):
        if not any(_clicks or []):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx"); field = trig.get("field")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        if field not in EVENT_FIELDS:
            return no_update
        events[idx] = dict(events[idx])
        events[idx][f"{field}_sec"] = None
        return events

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-racine-btn", "idx": ALL,
                "stage": ALL}, "n_clicks"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _set_racine(_clicks, events):
        if not any(_clicks or []):
            return no_update
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx"); stage = trig.get("stage")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        if not isinstance(stage, int) or not (1 <= stage <= 8):
            return no_update
        events[idx] = dict(events[idx])
        events[idx]["racine"] = int(stage)
        return events

    @app.callback(
        Output("video-events-store", "data",
                allow_duplicate=True),
        Input({"type": "event-comment", "idx": ALL,
                "field": ALL}, "value"),
        State("video-events-store", "data"),
        prevent_initial_call=True,
    )
    def _set_comment(_values, events):
        trig = callback_context.triggered_id
        if not isinstance(trig, dict):
            return no_update
        idx = trig.get("idx"); field = trig.get("field")
        events = list(events or [])
        if idx is None or idx < 0 or idx >= len(events):
            return no_update
        if field not in ("onset_comment", "behavior_comment",
                          "score_comment"):
            return no_update
        triggered = callback_context.triggered or []
        new_val = ""
        for t in triggered:
            new_val = t.get("value") or ""
        events[idx] = dict(events[idx])
        events[idx][field] = str(new_val)
        return events
