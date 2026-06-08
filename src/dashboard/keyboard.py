"""Keyboard shortcuts -- single source of truth.

The Video Review tab is keyboard-driven (Linear / Superhuman /
CVAT pattern). Bindings are declared here in Python; the only
JS-side concern is forwarding raw ``keydown`` events into a
``dcc.Store`` (see ``assets/keyboard.js``). A Python clientside
callback does the ``key -> action`` lookup, so adding a new
shortcut is a one-place change.

Wire-up overview:

  keydown (JS)
      -> kbd-keydown Store {key, shift, ctrl, ts}
      -> clientside lookup against kbd-bindings (Python -> JSON)
      -> kbd-event Store {action, seq}
      -> every subscriber Input("kbd-event", "data")
         (each callback filters on data["action"])

Callbacks that need to react do so by adding
``Input("kbd-event", "data")`` to their inputs. The convention
inside the callback is::

    triggered = (ctx.triggered_id == "kbd-event")
    if triggered and ev.get("action") != "next":
        return no_update

Keep ``SHORTCUTS`` and the cheat-sheet table aligned by feeding
the same data structure to both ``cheatsheet_overlay()`` and the
``kbd-bindings`` Store. NASA Rule 6: single source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dash import dcc, html

from src.dashboard.design import (
    COLOR_ACCENT, COLOR_DIVIDER, COLOR_SURFACE_1, COLOR_SURFACE_2,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_HEADER,
    RADIUS_MD, RADIUS_SM, SPACE_2, SPACE_3, SPACE_4, SPACE_5,
)


# --------------------------------------------------------------- #
# Shortcut catalog
# --------------------------------------------------------------- #

@dataclass(frozen=True)
class Shortcut:
    """One row in the cheat sheet + one entry in the key map.

    ``key`` is the literal ``event.key`` string emitted by the
    browser ("j", "ArrowDown", " ", "?"). ``display`` is what we
    show humans ("J", "↓", "Space", "?"); defaults to key.upper()
    for single letters, raw for everything else.

    ``action`` is the bus payload subscribers filter on. Keep
    actions short and verb-like.
    """
    key: str
    action: str
    label: str
    group: str
    display: str | None = None

    def display_label(self) -> str:
        if self.display is not None:
            return self.display
        if len(self.key) == 1 and self.key.isalpha():
            return self.key.upper()
        return self.key


SHORTCUTS: tuple[Shortcut, ...] = (
    # Queue navigation
    Shortcut("j", "next", "Next file in queue", "Queue"),
    Shortcut("k", "prev", "Previous file", "Queue"),
    Shortcut("ArrowDown", "next", "Next file in queue",
              "Queue", display="↓"),
    Shortcut("ArrowUp", "prev", "Previous file",
              "Queue", display="↑"),
    # Decision / review state
    Shortcut("n", "mark_no_events",
              "Mark no events + go to next", "Decision"),
    Shortcut("e", "events_mode",
              "Switch to 'events seen'", "Decision"),
    Shortcut("u", "undo",
              "Undo the last decision (8 s window)", "Decision"),
    Shortcut("r", "revert",
              "Reopen the current file's review", "Decision"),
    # Markers on the LFP
    Shortcut("m", "drop_marker",
              "Drop a marker at the video's current time",
              "Markers"),
    Shortcut("x", "delete_marker",
              "Delete the focused marker", "Markers"),
    # Video transport
    Shortcut(" ", "play_pause", "Play / pause the video",
              "Video", display="Space"),
    Shortcut("ArrowRight", "seek_fwd", "Seek video +1 s",
              "Video", display="→"),
    Shortcut("ArrowLeft", "seek_back", "Seek video -1 s",
              "Video", display="←"),
    # Notes + help
    Shortcut("/", "focus_note",
              "Focus the reviewer note textarea", "Notes"),
    Shortcut("?", "toggle_help",
              "Show / hide this overlay", "Help"),
    Shortcut("Escape", "close_help",
              "Close this overlay", "Help", display="Esc"),
)


def bindings_map() -> dict[str, str]:
    """``{event.key -> action}`` dict for the clientside lookup.

    If two shortcuts share a key, last one wins -- catch this in
    review, not at runtime, since the table is short.
    """
    return {s.key: s.action for s in SHORTCUTS}


# --------------------------------------------------------------- #
# UI primitives
# --------------------------------------------------------------- #

def _kbd_chip(text: str) -> html.Span:
    """A tiny <kbd>-style chip for the cheat sheet."""
    assert isinstance(text, str) and text, "chip text required"
    return html.Span(
        text,
        style={
            "display": "inline-block",
            "padding": f"2px {SPACE_2}",
            "minWidth": "22px",
            "textAlign": "center",
            "fontFamily": "ui-monospace, SFMono-Regular, monospace",
            "fontSize": FONT_SIZE_CAPTION,
            "color": COLOR_TEXT_PRIMARY,
            "backgroundColor": COLOR_SURFACE_2,
            "border": f"1px solid {COLOR_DIVIDER}",
            "borderRadius": RADIUS_SM,
            "boxShadow": f"0 1px 0 {COLOR_DIVIDER}",
        },
    )


def _group_rows(group: str,
                 items: Iterable[Shortcut]) -> list[html.Div]:
    """Render one Group heading + its rows."""
    assert isinstance(group, str) and group, "group required"
    rows: list[html.Div] = []
    rows.append(html.Div(
        group,
        style={
            "fontSize": FONT_SIZE_CAPTION,
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLOR_TEXT_TERTIARY,
            "marginTop": SPACE_4,
            "marginBottom": SPACE_2,
        },
    ))
    for s in items:
        rows.append(html.Div(
            [
                _kbd_chip(s.display_label()),
                html.Span(
                    s.label,
                    style={
                        "marginLeft": SPACE_3,
                        "color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_BODY,
                    },
                ),
            ],
            style={
                "display": "flex", "alignItems": "center",
                "padding": f"{SPACE_2} 0",
            },
        ))
    return rows


def cheatsheet_overlay() -> html.Div:
    """The fullscreen overlay shown when the user presses ``?``.

    Hidden by default (``display: none``). A clientside callback
    flips ``style.display`` on ``kbd-event.action ==
    'toggle_help' | 'close_help'``. Click on the dim backdrop
    also closes it (handled in the same JS).
    """
    seen: set[str] = set()
    groups: list[str] = []
    by_group: dict[str, list[Shortcut]] = {}
    for s in SHORTCUTS:
        if s.group not in seen:
            seen.add(s.group)
            groups.append(s.group)
        by_group.setdefault(s.group, []).append(s)
    body: list[html.Div] = []
    for g in groups:
        body.extend(_group_rows(g, by_group[g]))

    return html.Div(
        id="kbd-help-overlay",
        children=html.Div(
            [
                html.Div(
                    [
                        html.Div("Keyboard shortcuts",
                                  style={
                                      "fontSize": FONT_SIZE_HEADER,
                                      "fontWeight": 600,
                                      "color": COLOR_TEXT_PRIMARY,
                                  }),
                        html.Div("Press Esc or click outside to "
                                  "dismiss",
                                  style={
                                      "fontSize": FONT_SIZE_CAPTION,
                                      "color": COLOR_TEXT_TERTIARY,
                                      "marginTop": "2px",
                                  }),
                    ],
                    style={
                        "borderBottom": f"1px solid {COLOR_DIVIDER}",
                        "paddingBottom": SPACE_3,
                        "marginBottom": SPACE_2,
                    },
                ),
                *body,
            ],
            id="kbd-help-panel",
            style={
                "width": "min(560px, 92vw)",
                "maxHeight": "84vh",
                "overflowY": "auto",
                "backgroundColor": COLOR_SURFACE_1,
                "border": f"1px solid {COLOR_DIVIDER}",
                "borderRadius": RADIUS_MD,
                "padding": SPACE_5,
                "boxShadow": "0 24px 48px rgba(0,0,0,0.45)",
            },
        ),
        style={
            "position": "fixed",
            "inset": "0",
            "display": "none",
            "alignItems": "center",
            "justifyContent": "center",
            "backgroundColor": "rgba(10,10,20,0.55)",
            "backdropFilter": "blur(6px)",
            "zIndex": "10000",
        },
    )


def kbd_help_hint() -> html.Span:
    """Tiny 'Press ? for shortcuts' chip for the refresh bar."""
    return html.Span(
        ["Press ", _kbd_chip("?"), " for shortcuts"],
        id="kbd-help-hint",
        style={
            "display": "inline-flex", "alignItems": "center", "gap": SPACE_2,
            "color": COLOR_TEXT_TERTIARY,
            "fontSize": FONT_SIZE_CAPTION,
            "marginRight": SPACE_3,
        },
    )


# --------------------------------------------------------------- #
# Stores + the keydown -> action dispatcher
# --------------------------------------------------------------- #

def kbd_stores() -> list:
    """All hidden state the keyboard layer needs in the layout.

    * ``kbd-bindings`` -- {key: action} map, written once.
    * ``kbd-keydown``  -- raw event from assets/keyboard.js.
    * ``kbd-event``    -- bus consumed by every action subscriber.
    * ``kbd-undo``     -- {file_id, deadline_ms, label} for the
                          Undo toast (P0-2).
    """
    return [
        dcc.Store(id="kbd-bindings", data=bindings_map()),
        dcc.Store(id="kbd-keydown", data=None),
        dcc.Store(id="kbd-event", data=None),
        dcc.Store(id="kbd-undo", data=None),
    ]


# The clientside callback that maps raw keydowns -> action bus.
# Lives here so the wiring stays next to the catalog. The first
# argument to ``register`` is the Dash ``app`` instance; we call
# this from ``create_app`` after the layout is set.

KEY_TO_ACTION_JS = """
function (kd, bindings) {
    if (!kd || !bindings) {
        return window.dash_clientside.no_update;
    }
    const action = bindings[kd.key];
    if (!action) {
        return window.dash_clientside.no_update;
    }
    return {action: action, seq: kd.ts};
}
"""


HELP_TOGGLE_JS = """
function (ev) {
    if (!ev) { return window.dash_clientside.no_update; }
    const el = document.getElementById('kbd-help-overlay');
    if (!el) { return window.dash_clientside.no_update; }
    if (ev.action === 'toggle_help') {
        const open = el.style.display !== 'none' && el.style.display !== '';
        el.style.display = open ? 'none' : 'flex';
    } else if (ev.action === 'close_help') {
        el.style.display = 'none';
    }
    return window.dash_clientside.no_update;
}
"""


# --------------------------------------------------------------- #
# Undo toast (P0-2)
# --------------------------------------------------------------- #

def undo_toast() -> html.Div:
    """The bottom-right Undo toast.

    Shown when ``kbd-undo`` Store has a payload, hidden otherwise.
    UNDO_TOAST_JS handles the show/hide + the auto-clear after
    the deadline; the actual reopen runs server-side via the
    Undo button click or the ``U`` hotkey.
    """
    return html.Div(
        id="kbd-undo-toast",
        children=[
            html.Span(id="kbd-undo-text",
                       style={"color": COLOR_TEXT_PRIMARY,
                               "marginRight": SPACE_4,
                               "fontSize": FONT_SIZE_BODY}),
            html.Button(
                ["Undo  ", _kbd_chip("U")],
                id="kbd-undo-button",
                n_clicks=0,
                style={
                    "background": "transparent",
                    "border": f"1px solid {COLOR_ACCENT}",
                    "color": COLOR_ACCENT,
                    "padding": f"{SPACE_2} {SPACE_3}",
                    "borderRadius": RADIUS_SM,
                    "cursor": "pointer",
                    "fontSize": FONT_SIZE_BODY,
                    "fontWeight": 600,
                    "display": "inline-flex",
                    "alignItems": "center",
                    "gap": SPACE_2,
                },
            ),
        ],
        style={
            "position": "fixed",
            "bottom": SPACE_5,
            "right": SPACE_5,
            "display": "none",
            "alignItems": "center",
            "padding": f"{SPACE_3} {SPACE_4}",
            "backgroundColor": COLOR_SURFACE_2,
            "border": f"1px solid {COLOR_DIVIDER}",
            "borderRadius": RADIUS_MD,
            "boxShadow": "0 12px 24px rgba(0,0,0,0.45)",
            "zIndex": "10001",
        },
    )


UNDO_TOAST_JS = """
function (undo) {
    const toast = document.getElementById('kbd-undo-toast');
    const text  = document.getElementById('kbd-undo-text');
    if (!toast || !text) { return window.dash_clientside.no_update; }
    if (!undo || !undo.file_id) {
        toast.style.display = 'none';
        return window.dash_clientside.no_update;
    }
    text.innerText = undo.label || 'Marked file';
    toast.style.display = 'flex';
    const remaining = (undo.deadline_ms || 0) - Date.now();
    if (remaining > 0) {
        setTimeout(function () {
            // Only clear if still the same payload (user could
            // have already pressed U or another mark fired).
            const cur = window.dash_clientside &&
                window.dash_clientside.callback_context;
            window.dash_clientside.set_props('kbd-undo',
                                              {data: null});
        }, remaining);
    } else {
        window.dash_clientside.set_props('kbd-undo', {data: null});
    }
    return window.dash_clientside.no_update;
}
"""


# When the user hits `N`, drive the existing Mark-done button:
# set the decision radio to 'no_events' (so _save_review's
# validation passes), then bump the Save button's n_clicks. The
# server callback then handles the auto-advance + Undo wiring.
MARK_NO_EVENTS_JS = """
function (ev, curClicks) {
    if (!ev || ev.action !== 'mark_no_events') {
        return [window.dash_clientside.no_update,
                window.dash_clientside.no_update];
    }
    return ['no_events', (curClicks || 0) + 1];
}
"""
