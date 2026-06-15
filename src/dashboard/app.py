"""Dash web dashboard for the QC Monitor — v3 with evoked waveforms, LFP browser,
electrode health, session compare, activity log, and annotations."""

import logging
import os
from datetime import datetime, date

from dash import Dash, html, dcc, no_update, ALL
from dash.dependencies import Input, Output, State

from src.db.store import Store
from src.dashboard.auth import register_auth, current_user_email
from src.dashboard import keyboard as _kbd
from src.utils import assignments as _assignments
from src.utils import event_clip as _event_clip
from src.utils import mass_analyze as _mass_analyze
from src.utils.version import qc_monitor_version
from src.dashboard.media_routes import register_media_routes
from src.dashboard.tabs import video as tabs_video
from src.dashboard.tabs import surgeries as tabs_surgeries
from src.dashboard.tabs import maintenance as tabs_maintenance
from src.dashboard.tabs import data_log_xref as tabs_data_log_xref
from src.dashboard.tabs import activity_log as tabs_activity_log
from src.dashboard.tabs import alerts as tabs_alerts
from src.dashboard.tabs import annotations as tabs_annotations
from src.dashboard.tabs import criticality as tabs_criticality
from src.dashboard.tabs import electrode_health as tabs_electrode_health
from src.dashboard.tabs import evoked as tabs_evoked
from src.dashboard.tabs import lfp_browser as tabs_lfp_browser
from src.dashboard.tabs import overview as tabs_overview
from src.dashboard.tabs import session_compare as tabs_session_compare
from src.dashboard.tabs import sessions as tabs_sessions
from src.dashboard.tabs import settings as tabs_settings
from src.dashboard.tabs import signal_quality as tabs_signal_quality
from src.dashboard.tabs import stim as tabs_stim
from src.dashboard.tabs import waveforms as tabs_waveforms
from src.dashboard.components import (
    card as _card, pill as _pill, section_header as _section_header,
)
from src.dashboard.data_helpers import (
    channel_map as _get_channel_map,
    color_for_role as _color_for_role,
    parse_json_field as _parse_json_field,
)
# Side-effect import: registers the qc_dark Plotly template as default
from src.dashboard import plotly_template  # noqa: F401

logger = logging.getLogger("qc_monitor.dashboard")

# EVOKED_FEATURE_COLS / EVOKED_FEATURE_LABELS moved to
# src/dashboard/data_helpers.py so tab modules can share one source
# of truth. Imported below so the 6 existing in-module references
# keep resolving.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    EVOKED_FEATURE_COLS, EVOKED_FEATURE_LABELS,
)

# --------------------------------------------------------------------- #
#  Design tokens (Apple HIG-inspired)
#
#  Semantic colors first, raw hex values second. Every component below
#  references these tokens so the whole app re-skins from one place.
# --------------------------------------------------------------------- #

# Design tokens live in src/dashboard/design.py (single source of truth
# for the Python side; CSS reads the same values via :root variables
# injected through index_string below).
from src.dashboard.design import (
    COLOR_SURFACE_0, COLOR_SURFACE_1, COLOR_SURFACE_2, COLOR_SURFACE_3,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    COLOR_DIVIDER, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING,
    COLOR_DANGER, ROLE_COLORS, FONT_STACK,
    FONT_SIZE_TITLE, FONT_SIZE_HEADER, FONT_SIZE_BODY, FONT_SIZE_CAPTION,
    SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5, SPACE_6,
    RADIUS_SM, RADIUS_MD,
    as_css_root_block,
)

# CONFIG_PATH + _load_config + _save_config moved to data_helpers.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    CONFIG_PATH,
    load_config as _load_config,
    processed_files_for_session as _get_processed_files_for_session,
    save_config as _save_config,
)

# Standard time-range options used across tabs
# TIME_RANGE_OPTIONS moved to src/dashboard/data_helpers.py.
from src.dashboard.data_helpers import TIME_RANGE_OPTIONS  # noqa: E402,F401

# --------------------------------------------------------------------- #
#  Navigation taxonomy — top-level groups with sub-tabs.
#
#  The sub-tab ``id`` is what the existing ``render_tab`` callback
#  switches on, so adding a new tab means: (1) add an entry below,
#  (2) add a branch to the existing render_tab, (3) build a layout.
# --------------------------------------------------------------------- #
NAV_GROUPS = [
    {"id": "overview", "label": "Overview", "subs": [
        {"id": "overview", "label": "Overview"},
    ]},
    {"id": "sessions", "label": "Sessions", "subs": [
        {"id": "sessions", "label": "All sessions"},
        {"id": "session_compare", "label": "Compare"},
    ]},
    {"id": "analysis", "label": "Analysis", "subs": [
        {"id": "waveforms", "label": "Evoked waveforms"},
        {"id": "evoked", "label": "Evoked features"},
        {"id": "criticality", "label": "Criticality"},
        {"id": "lfp", "label": "LFP browser"},
        {"id": "video", "label": "Video review"},
    ]},
    {"id": "quality", "label": "Quality", "subs": [
        {"id": "signal", "label": "Signal quality"},
        {"id": "electrode_health", "label": "Electrode health"},
        {"id": "stim", "label": "Stim QC"},
    ]},
    {"id": "lab", "label": "Lab", "subs": [
        {"id": "surgeries", "label": "Surgeries"},
        {"id": "maintenance", "label": "Maintenance"},
        {"id": "data_log_xref", "label": "Data log diff"},
    ]},
    {"id": "system", "label": "System", "subs": [
        {"id": "annotations", "label": "Notes"},
        {"id": "review_status", "label": "Review status"},
        {"id": "event_verification",
          "label": "Event Verification"},
        {"id": "activity_log", "label": "Activity log"},
        {"id": "alerts", "label": "Alerts"},
        {"id": "settings", "label": "Settings"},
    ]},
]
SUBTAB_TO_GROUP: dict[str, str] = {
    sub["id"]: g["id"] for g in NAV_GROUPS for sub in g["subs"]
}
GROUP_BY_ID: dict[str, dict] = {g["id"]: g for g in NAV_GROUPS}

# Sub-tab styling — slightly smaller / quieter than the top group tabs.
SUBTAB_STYLE = {
    "backgroundColor": "transparent",
    "color": COLOR_TEXT_TERTIARY,
    "padding": f"{SPACE_2} {SPACE_3}",
    "border": "none",
    "borderBottom": "2px solid transparent",
    "borderRadius": "0",
    "fontSize": FONT_SIZE_CAPTION,
    "fontWeight": "500",
    "letterSpacing": "0.3px",
    "textTransform": "uppercase",
}
SUBTAB_SELECTED_STYLE = {
    **SUBTAB_STYLE,
    "color": COLOR_TEXT_PRIMARY,
    "borderBottom": f"2px solid {COLOR_TEXT_PRIMARY}",
}

# --------------------------------------------------------------------- #
#  Component styles built from tokens
# --------------------------------------------------------------------- #

TAB_STYLE = {
    "backgroundColor": "transparent",
    "color": COLOR_TEXT_TERTIARY,
    "padding": f"{SPACE_3} {SPACE_4}",
    "border": "none",
    "borderBottom": "2px solid transparent",
    "borderRadius": "0",
    "fontSize": FONT_SIZE_BODY,
    "fontWeight": "500",
    "letterSpacing": "0.1px",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "color": COLOR_TEXT_PRIMARY,
    "borderBottom": f"2px solid {COLOR_ACCENT}",
    "fontWeight": "600",
}

# DARK_TABLE_STYLE / ZEBRA_STRIPE live in src/dashboard/components.py
# so the four tab modules that still inline DataTable usage in this
# file and the carved-out tabs share one source of truth.
from src.dashboard.components import DARK_TABLE_STYLE, ZEBRA_STRIPE  # noqa: E402,F401

# SECTION_STYLE / LABEL_STYLE / INPUT_STYLE / FIELD_STYLE /
# DROPDOWN_STYLE moved to src/dashboard/components.py so the tab
# modules carved out of this file can share one source of truth.
from src.dashboard.components import (  # noqa: E402,F401
    DROPDOWN_STYLE, FIELD_STYLE, INPUT_STYLE, LABEL_STYLE, SECTION_STYLE,
)

# ====================================================================== #
#  Helpers
# ====================================================================== #


# _collapsible / _status_card / _fmt_disk_space / _status_pill moved to
# src/dashboard/tabs/overview.py (their only consumer).


_QC_MONITOR_VERSION = qc_monitor_version()


# _parse_json_field, _get_channel_map, and _color_for_role moved to
# src/dashboard/data_helpers.py so the tab modules carved out of this
# file can reuse them without importing the app shell. The aliased
# imports above preserve every existing in-module callsite.


# _empty_fig moved to src/dashboard/data_helpers.empty_fig. The
# alias below keeps the 19 in-module callsites working unchanged.
from src.dashboard.data_helpers import empty_fig as _empty_fig  # noqa: E402,F401


# _empty_state moved to src/dashboard/components.tab_empty_state.
# Aliased here so the in-module callsites keep working.
from src.dashboard.components import (  # noqa: E402,F401
    tab_empty_state as _empty_state,
)


# _session_dropdown_options moved to src/dashboard/data_helpers.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    session_dropdown_options as _session_dropdown_options,
)


# _default_session moved to src/dashboard/data_helpers.default_session.
from src.dashboard.data_helpers import (  # noqa: E402,F401
    default_session as _default_session,
)


# _get_processed_files_for_session moved to data_helpers (re-imported above).


# ====================================================================== #
#  App factory
# ====================================================================== #

def create_app(config: dict, store: Store) -> Dash:
    refresh_sec = config.get("dashboard", {}).get("refresh_interval_sec", 10)

    # Warm the Reviewer Assignments cache in a daemon thread so the
    # Video Review picker never blocks on the Sheets API in any
    # render path. Idempotent across reloads.
    _assignments.start_warmer(config)
    # Spawn the PI verification event-clip extractor worker.
    # Daemon thread that drains event_clip_job 'pending' rows
    # via ffmpeg. Idempotent across reloads.
    _event_clip.start_worker(store, config)
    # Mass Analyze (PI bulk pre-screen) worker. Drains
    # mass_analyze_job rows; same idempotent pattern.
    _mass_analyze.start_worker(store, config)

    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    app = Dash(__name__, title="QC Monitor", suppress_callback_exceptions=True,
               assets_folder=assets_dir)

    # Inject design tokens as CSS custom properties on :root so theme.css
    # and ad-hoc inline `var(--*)` references stay in sync with Python.
    app.index_string = (
        """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        """ + as_css_root_block() + """
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""
    )

    # Identity gate (Cloudflare Access JWT) + media-streaming routes both
    # attach to the underlying Flask server.
    register_auth(app.server, store, config)
    register_media_routes(app.server, store, config)

    app.layout = html.Div([
        # Header — quiet chrome, content-first.
        html.Div([
            html.H1("QC Monitor",
                    style={"margin": "0", "fontSize": FONT_SIZE_TITLE,
                           "fontWeight": "600", "letterSpacing": "-0.2px",
                           "display": "inline-block",
                           "color": COLOR_TEXT_PRIMARY}),
            html.Span(_QC_MONITOR_VERSION,
                       id="header-version",
                       title="git commit + date "
                              "(dirty = uncommitted local changes)",
                       style={
                           "display": "inline-block",
                           "marginLeft": SPACE_3,
                           "padding": "2px 8px",
                           "borderRadius": "10px",
                           "background": COLOR_SURFACE_2,
                           "color": COLOR_TEXT_TERTIARY,
                           "fontSize": FONT_SIZE_CAPTION,
                           "fontFamily": "ui-monospace, monospace",
                           "letterSpacing": "0.2px",
                           "verticalAlign": "middle",
                       }),
            html.Span(
                id="header-status-dot",
                className="status-dot",
                title="Health: unknown",
                style={
                    "display": "inline-block",
                    "width": "8px", "height": "8px",
                    "borderRadius": "50%",
                    "backgroundColor": COLOR_TEXT_TERTIARY,
                    "marginLeft": SPACE_3,
                    "verticalAlign": "middle",
                },
            ),
            html.Span(id="logged-in-as",
                      style={"marginLeft": SPACE_4,
                             "color": COLOR_TEXT_TERTIARY,
                             "fontSize": FONT_SIZE_CAPTION,
                             "letterSpacing": "0.2px"}),
        ], id="app-header",
           style={"padding": f"{SPACE_4} {SPACE_6}",
                  "background": COLOR_SURFACE_0,
                  "borderBottom": f"1px solid {COLOR_DIVIDER}"}),

        # Top-level nav: 5 groups. Apple HIG "reduce" — each group fits
        # comfortably without horizontal scrolling.
        dcc.Tabs(
            id="group-tabs", value="overview",
            children=[
                dcc.Tab(label=g["label"], value=g["id"],
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
                for g in NAV_GROUPS
            ],
            style={"borderBottom": f"1px solid {COLOR_DIVIDER}",
                   "padding": f"0 {SPACE_5}",
                   "backgroundColor": COLOR_SURFACE_0},
        ),

        # Second-level nav: sub-tabs within the active group. Children
        # are populated by a callback when the group changes; the
        # ``tabs`` id is preserved so all existing sub-tab callbacks
        # (Input("tabs", "value")) continue to work unchanged.
        dcc.Tabs(
            id="tabs", value="overview", children=[],
            style={"borderBottom": f"1px solid {COLOR_DIVIDER}",
                   "padding": f"{SPACE_2} {SPACE_5} 0",
                   "backgroundColor": COLOR_SURFACE_0,
                   "minHeight": "36px"},
        ),

        html.Div(id="tab-content",
                 style={"padding": f"{SPACE_6} {SPACE_5}",
                        "backgroundColor": COLOR_SURFACE_0,
                        "minHeight": "80vh"}),

        dcc.Interval(id="refresh", interval=refresh_sec * 1000, n_intervals=0,
                     disabled=True),  # auto-refresh OFF by default
        html.Div([
            _kbd.kbd_help_hint(),
            html.Button("Focus", id="focus-toggle-btn",
                        title="Hide header + tabs nav and show only "
                               "the tab content (click again to restore)",
                        style={"backgroundColor": COLOR_SURFACE_2,
                               "color": COLOR_TEXT_PRIMARY,
                               "border": f"1px solid {COLOR_DIVIDER}",
                               "borderRadius": RADIUS_SM,
                               "padding": f"{SPACE_1} {SPACE_3}",
                               "cursor": "pointer",
                               "marginRight": SPACE_2,
                               "fontSize": FONT_SIZE_CAPTION,
                               "fontFamily": FONT_STACK}),
            # Manual PiP toggle (replaces the IntersectionObserver
            # auto-pin). Click flips the video-pip-state Store
            # which a clientside callback in video.py mirrors to
            # the .qc-pip-grid class. Active-state styling
            # (filled accent vs outline) is managed by a
            # clientside callback subscribed to the same Store.
            html.Button("PiP", id="pip-toggle-btn",
                        title="Pop the video + LFP into a floating "
                               "PiP at bottom-right (P hotkey). "
                               "Click again to dock.",
                        n_clicks=0,
                        style={"backgroundColor": COLOR_SURFACE_2,
                               "color": COLOR_TEXT_PRIMARY,
                               "border": f"1px solid {COLOR_DIVIDER}",
                               "borderRadius": RADIUS_SM,
                               "padding": f"{SPACE_1} {SPACE_3}",
                               "cursor": "pointer",
                               "marginRight": SPACE_2,
                               "fontSize": FONT_SIZE_CAPTION,
                               "fontFamily": FONT_STACK}),
            html.Button("Refresh", id="manual-refresh-btn",
                        style={"backgroundColor": COLOR_SURFACE_2,
                               "color": COLOR_TEXT_PRIMARY,
                               "border": f"1px solid {COLOR_DIVIDER}",
                               "borderRadius": RADIUS_SM,
                               "padding": f"{SPACE_1} {SPACE_3}",
                               "cursor": "pointer",
                               "marginRight": SPACE_2,
                               "fontSize": FONT_SIZE_CAPTION,
                               "fontFamily": FONT_STACK}),
            html.Span(id="last-refresh-label",
                      style={"color": COLOR_TEXT_TERTIARY,
                             "fontSize": FONT_SIZE_CAPTION,
                             "marginRight": SPACE_3}),
            dcc.Checklist(id="auto-refresh-toggle",
                          options=[{"label": " Auto", "value": True}],
                          value=[], inline=True,
                          style={"display": "inline-block",
                                 "marginRight": SPACE_2},
                          labelStyle={"color": COLOR_TEXT_SECONDARY,
                                      "fontSize": FONT_SIZE_CAPTION}),
            # Interval picker. Updates the dcc.Interval below via
            # callback; default matches config.dashboard.
            # refresh_interval_sec so the dropdown starts in sync.
            dcc.Dropdown(
                id="auto-refresh-interval",
                options=[
                    {"label": "5s",   "value": 5},
                    {"label": "10s",  "value": 10},
                    {"label": "30s",  "value": 30},
                    {"label": "1min", "value": 60},
                    {"label": "5min", "value": 300},
                ],
                value=int(refresh_sec) if refresh_sec else 10,
                clearable=False, searchable=False,
                style={"width": "78px", "fontSize": FONT_SIZE_CAPTION,
                        "color": COLOR_TEXT_PRIMARY},
            ),
        ], id="refresh-bar",
           style={"position": "fixed", "top": SPACE_2, "right": SPACE_5, "zIndex": "9999",
                  "display": "flex", "alignItems": "center",
                  "background": "rgba(19,19,31,0.85)", "backdropFilter": "blur(12px)",
                  "padding": f"{SPACE_1} {SPACE_3}", "borderRadius": RADIUS_SM,
                  "border": f"1px solid {COLOR_DIVIDER}"}),
        dcc.Store(id="refresh-trigger", data=0),
        dcc.Store(id="last-refresh-ts", data=None),
        # Video PiP toggle. False = grid sits in normal Step 2
        # layout; True = grid floats at bottom-right via the
        # .qc-pip-grid CSS class. Flipped by (a) the PiP button
        # in this refresh bar, (b) the in-PiP Dock button, or
        # (c) the P hotkey via kbd-event.
        dcc.Store(id="video-pip-state", data=False),
        # Cache-version Store for the Reviewer Assignments sheet.
        # Bumped by a background warmer thread; subscribed to by
        # the Video Review animal picker so it auto-updates when
        # new data lands without paying the Sheets API latency on
        # the render path. The 2 s polling Interval here is
        # server-side cheap (it just reads an int).
        dcc.Store(id="assignments-version", data=0),
        dcc.Interval(id="assignments-poll", interval=2000,
                       n_intervals=0),
        # Focus-mode state: True when the user wants chrome (header,
        # group tabs, sub-tabs) hidden so only the active tab content
        # remains visible. Toggled by the Focus button in the refresh
        # bar; the bar itself stays visible (it's the only way out).
        dcc.Store(id="focus-mode", data=False),
        # Snapshot expanded state: False = small thumbnail in the
        # third column, True = full-width image. Toggled by clicking
        # the snapshot itself.
        dcc.Store(id="snapshot-expanded", data=False),
        # LFP Browser -> Video Review bridge. Populated by a "View
        # video" button click in the LFP Browser tab; consumed by
        # Video Review to prefill session/file/channel/filters.
        # Includes a seq counter so re-clicking the same channel
        # re-fires the consumer even if the payload didn't change.
        dcc.Store(id="lfp-to-video-bridge", data=None),
        # Pending video seek (in LFP seconds + LFP duration so the
        # BHZ scale factor can be applied). A clientside callback
        # on video-time-tick consumes this once the master <video>
        # has loaded metadata; it's set to None after seeking.
        dcc.Store(id="pending-seek", data=None),
        dcc.Interval(id="elapsed-ticker", interval=5000, n_intervals=0),
        # Hidden stores
        dcc.Store(id="selected-session-dir"),
        # Keyboard-shortcut layer (see src/dashboard/keyboard.py).
        # kbd-keydown is written by assets/keyboard.js on each
        # raw keypress; a clientside callback maps it through
        # kbd-bindings into kbd-event which downstream subscribers
        # filter by action.
        *_kbd.kbd_stores(),
        _kbd.cheatsheet_overlay(),
        _kbd.undo_toast(),
    ], style={"backgroundColor": COLOR_SURFACE_0,
              "fontFamily": FONT_STACK,
              "color": COLOR_TEXT_PRIMARY,
              "minHeight": "100vh",
              "fontSize": FONT_SIZE_BODY,
              "letterSpacing": "0.1px"})

    # ------------------------------------------------------------------ #
    #  Refresh + focus-mode controls
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("refresh", "disabled"),
        [Input("auto-refresh-toggle", "value")]
    )
    def toggle_auto_refresh(val):
        return not bool(val)

    @app.callback(
        Output("refresh", "interval"),
        Input("auto-refresh-interval", "value"),
        prevent_initial_call=True,
    )
    def update_auto_refresh_interval(seconds):
        # dcc.Interval.interval is in ms; clamp to a sane window.
        s = int(seconds or 10)
        s = max(1, min(s, 3600))
        return s * 1000

    @app.callback(
        Output("focus-mode", "data"),
        Output("focus-toggle-btn", "children"),
        Input("focus-toggle-btn", "n_clicks"),
        State("focus-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_focus_mode(_clicks, current):
        new_state = not bool(current)
        return new_state, ("Exit focus" if new_state else "Focus")

    @app.callback(
        Output("app-header", "style"),
        Output("group-tabs", "style"),
        Output("tabs", "style"),
        Output("tab-content", "style"),
        Input("focus-mode", "data"),
    )
    def apply_focus_mode(focus):
        # Default (chrome visible) styles. Mirror what app.layout sets
        # so toggling back restores the original geometry.
        header_default = {"padding": f"{SPACE_4} {SPACE_6}",
                          "background": COLOR_SURFACE_0,
                          "borderBottom": f"1px solid {COLOR_DIVIDER}"}
        group_tabs_default = {"borderBottom": f"1px solid {COLOR_DIVIDER}",
                              "padding": f"0 {SPACE_5}",
                              "backgroundColor": COLOR_SURFACE_0}
        tabs_default = {"borderBottom": f"1px solid {COLOR_DIVIDER}",
                        "padding": f"{SPACE_2} {SPACE_5} 0",
                        "backgroundColor": COLOR_SURFACE_0,
                        "minHeight": "36px"}
        content_default = {"padding": f"{SPACE_6} {SPACE_5}",
                           "backgroundColor": COLOR_SURFACE_0,
                           "minHeight": "80vh"}
        if not focus:
            return (header_default, group_tabs_default,
                    tabs_default, content_default)
        # Focus mode: hide all chrome and pull the content padding
        # in so the visible viewport is just the tab body.
        hidden = {"display": "none"}
        content_focus = {"padding": f"{SPACE_4} {SPACE_5}",
                         "backgroundColor": COLOR_SURFACE_0,
                         "minHeight": "100vh"}
        return hidden, hidden, hidden, content_focus

    @app.callback(
        [Output("refresh-trigger", "data"), Output("last-refresh-ts", "data")],
        [Input("manual-refresh-btn", "n_clicks"), Input("refresh", "n_intervals")],
        [State("refresh-trigger", "data")]
    )
    def on_refresh(clicks, intervals, current):
        import time as _time
        return (current or 0) + 1, _time.time()

    # Publish the assignments cache version into a Store so picker
    # callbacks can react to "data ready" without polling the
    # Sheets API themselves. Returns no_update unless the warmer
    # actually bumped the counter, so downstream callbacks don't
    # refire every 2 s for no reason.
    @app.callback(
        Output("assignments-version", "data"),
        Input("assignments-poll", "n_intervals"),
        State("assignments-version", "data"),
    )
    def _publish_assignments_version(_n, prev):
        cur = _assignments.cache_version()
        if cur == (prev or 0):
            return no_update
        return cur

    # Keyboard shortcut bus. assets/keyboard.js pushes raw keydowns
    # into kbd-keydown; this clientside map looks the key up in
    # kbd-bindings (Python-owned) and bumps kbd-event with the
    # action name. Downstream callbacks subscribe to kbd-event and
    # filter on ``data["action"]``.
    app.clientside_callback(
        _kbd.KEY_TO_ACTION_JS,
        Output("kbd-event", "data"),
        Input("kbd-keydown", "data"),
        State("kbd-bindings", "data"),
        prevent_initial_call=True,
    )

    # Help overlay open/close. Done in JS because it only flips a
    # style.display flag -- no need for a server round trip.
    app.clientside_callback(
        _kbd.HELP_TOGGLE_JS,
        Output("kbd-help-overlay", "id"),  # write-only sink
        Input("kbd-event", "data"),
        prevent_initial_call=True,
    )

    # Click on the "Press ? for shortcuts" chip emits the same
    # toggle_help action the ? key does. Goes through kbd-event
    # so the existing HELP_TOGGLE_JS subscriber handles the open/
    # close; no duplicated DOM logic.
    app.clientside_callback(
        _kbd.HELP_HINT_CLICK_JS,
        Output("kbd-event", "data", allow_duplicate=True),
        Input("kbd-help-hint", "n_clicks"),
        prevent_initial_call=True,
    )

    # Pulsate "+ Add event" when the reviewer picked "Events
    # seen" but hasn't added any events yet -- a one-time visual
    # nudge so the first event isn't missed.
    app.clientside_callback(
        """
        function (decision, events) {
            const empty = !events || events.length === 0;
            const should_pulse = (decision === 'has_events'
                                    && empty);
            return should_pulse ? 'qc-pulse' : '';
        }
        """,
        Output("video-events-add-btn", "className"),
        Input("video-review-decision", "value"),
        Input("video-events-store", "data"),
    )

    # ---- Track C: manual PiP toggle ---- #
    # The PiP button in the refresh bar, the Dock button inside
    # the floating PiP, and the P hotkey all flip the same
    # video-pip-state Store. A separate clientside callback
    # mirrors the Store into the DOM via assets/pip_video.js.
    app.clientside_callback(
        """
        function (btn_n, dock_n, kbd_ev, state) {
            const ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || !ctx.triggered.length) {
                return window.dash_clientside.no_update;
            }
            const trig = ctx.triggered[0].prop_id || '';
            if (trig.startsWith('kbd-event')) {
                if (!kbd_ev || kbd_ev.action !== 'toggle_pip') {
                    return window.dash_clientside.no_update;
                }
            } else if (trig.startsWith('pip-toggle-btn')
                        && !btn_n) {
                return window.dash_clientside.no_update;
            } else if (trig.startsWith('pip-dock-btn')
                        && !dock_n) {
                return window.dash_clientside.no_update;
            }
            return !state;
        }
        """,
        Output("video-pip-state", "data"),
        Input("pip-toggle-btn", "n_clicks"),
        Input("pip-dock-btn", "n_clicks"),
        Input("kbd-event", "data"),
        State("video-pip-state", "data"),
        prevent_initial_call=True,
    )

    # Mirror the Store into the DOM. Calls into
    # window.qcPipVideo.applyPipClass which lives in
    # assets/pip_video.js.
    app.clientside_callback(
        """
        function (on) {
            if (window.qcPipVideo
                    && window.qcPipVideo.applyPipClass) {
                window.qcPipVideo.applyPipClass(!!on);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("video-pip-state", "id"),  # write-only sink
        Input("video-pip-state", "data"),
    )

    # Active-state styling on the refresh-bar PiP button
    # (filled accent vs outlined) so the reviewer can read the
    # PiP state at a glance from the top of the screen.
    app.clientside_callback(
        """
        function (on) {
            const btn = document.getElementById(
                'pip-toggle-btn');
            if (!btn) {
                return window.dash_clientside.no_update;
            }
            if (on) {
                btn.style.backgroundColor = '#5e7ce2';
                btn.style.color = '#ffffff';
                btn.style.borderColor = '#5e7ce2';
            } else {
                btn.style.backgroundColor = '';
                btn.style.color = '';
                btn.style.borderColor = '';
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("pip-toggle-btn", "title"),  # write-only sink
        Input("video-pip-state", "data"),
    )

    # PI-only frame sampler: on click, draw the <video>'s current
    # frame to an offscreen canvas, compute grayscale (BT.601)
    # pixel variance, and report it back. Wide enough to capture
    # behavioral detail; capped at 640 px so a one-hour 1080p
    # video doesn't OOM the browser.
    app.clientside_callback(
        """
        function (n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            const v = document.getElementById('lfp-video');
            if (!v || v.readyState < 2 || !v.videoWidth) {
                return {error: 'Video not ready -- load a '
                                + 'recording first.', variance: null};
            }
            const TARGET_W = 640;
            const w = Math.min(TARGET_W, v.videoWidth);
            const h = Math.round(w * (v.videoHeight / v.videoWidth));
            const c = document.createElement('canvas');
            c.width = w; c.height = h;
            const ctx = c.getContext('2d');
            try {
                ctx.drawImage(v, 0, 0, w, h);
            } catch (e) {
                return {error: 'Cross-origin video; can\\'t sample.',
                         variance: null};
            }
            const px = ctx.getImageData(0, 0, w, h).data;
            const n = w * h;
            let sum = 0, sumsq = 0;
            for (let i = 0; i < px.length; i += 4) {
                const g = 0.299 * px[i]
                        + 0.587 * px[i+1]
                        + 0.114 * px[i+2];
                sum += g;
                sumsq += g * g;
            }
            const mean = sum / n;
            const variance = (sumsq / n) - mean * mean;
            return {
                variance: variance,
                time: v.currentTime,
                paused: v.paused,
                w: w, h: h,
            };
        }
        """,
        Output("video-pi-variance-store", "data"),
        Input("video-pi-sample-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    # Undo toast show/hide. Reads the kbd-undo Store, mirrors its
    # presence into the toast's visibility, and auto-clears the
    # Store after the deadline passes.
    app.clientside_callback(
        _kbd.UNDO_TOAST_JS,
        Output("kbd-undo-toast", "title"),  # write-only sink
        Input("kbd-undo", "data"),
        prevent_initial_call=True,
    )

    # N hotkey -> set decision radio + fire the Save button via
    # the existing _save_review server callback.
    app.clientside_callback(
        _kbd.MARK_NO_EVENTS_JS,
        Output("video-review-decision", "value",
                allow_duplicate=True),
        Output("video-review-save-btn", "n_clicks",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        State("video-review-save-btn", "n_clicks"),
        prevent_initial_call=True,
    )

    # E hotkey -> arm "Events seen" mode (reveals the marker
    # editor + lets click-on-LFP append onset markers).
    app.clientside_callback(
        _kbd.EVENTS_MODE_JS,
        Output("video-review-decision", "value",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        prevent_initial_call=True,
    )

    # Space / arrows / slash -- pure DOM side effects on the
    # <video> element + the note textarea. Writes nothing into
    # Dash state, hence the sink Output.
    app.clientside_callback(
        _kbd.SHORTCUT_DOM_JS,
        Output("kbd-dom-sink", "data", allow_duplicate=True),
        Input("kbd-event", "data"),
        prevent_initial_call=True,
    )

    # M hotkey -> append onset marker at the video's current
    # time (BHZ-rescaled into LFP seconds).
    app.clientside_callback(
        _kbd.DROP_MARKER_JS,
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        State("video-review-marker-store", "data"),
        State("video-lfp-duration", "data"),
        prevent_initial_call=True,
    )

    # X hotkey -> drop the most-recent marker.
    app.clientside_callback(
        _kbd.DELETE_MARKER_JS,
        Output("video-review-marker-store", "data",
                allow_duplicate=True),
        Input("kbd-event", "data"),
        State("video-review-marker-store", "data"),
        prevent_initial_call=True,
    )

    # Overview refresh / thumbnail / snapshot callbacks moved to
    # src/dashboard/tabs/overview.py.


    @app.callback(
        Output("header-status-dot", "style"),
        Output("header-status-dot", "title"),
        Output("header-status-dot", "className"),
        Input("refresh-trigger", "data"),
    )
    def update_header_status_dot(_n):
        """One-glance health indicator beside the title. Green +
        pulsing when the SMB share is up and the queue isn't stuck;
        red + pulsing otherwise."""
        try:
            history = store.get_health_history(hours=1)
        except Exception:
            history = []
        latest = history[-1] if history else {}
        net_ok = bool(latest.get("network_share_accessible"))
        # Crude queue-stalled heuristic: 0 files processed in the last
        # hour while there's a non-empty pending queue. Cheap to compute.
        files_hour = int(latest.get("files_processed_last_hour", 0) or 0)
        try:
            pending = bool(store.get_pending_files(limit=1))
        except Exception:
            pending = False
        queue_stalled = pending and files_hour == 0
        ok = net_ok and not queue_stalled
        color = COLOR_SUCCESS if ok else COLOR_DANGER
        title = ("Healthy: network up, queue moving"
                 if ok
                 else ("Network share unreachable" if not net_ok
                       else "Queue stalled (no files processed last hour)"))
        cls = "status-dot " + ("pulse-ok" if ok else "pulse-bad")
        style = {
            "display": "inline-block",
            "width": "8px", "height": "8px",
            "borderRadius": "50%",
            "backgroundColor": color,
            "marginLeft": SPACE_3,
            "verticalAlign": "middle",
        }
        return style, title, cls

    @app.callback(
        Output("last-refresh-label", "children"),
        [Input("elapsed-ticker", "n_intervals")],
        [State("last-refresh-ts", "data")],
    )
    def update_elapsed(_, ts):
        if ts is None:
            return ""
        import time as _time
        elapsed = int(_time.time() - ts)
        if elapsed < 60:
            return f"Last refresh {elapsed}s ago"
        return f"Last refresh {elapsed // 60}m {elapsed % 60}s ago"

    # ------------------------------------------------------------------ #
    #  Direct-manipulation: click a Sessions row -> jump to Analysis
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("selected-session-dir", "data"),
        Output("group-tabs", "value"),
        Output("tabs", "value", allow_duplicate=True),
        Input("sessions-table", "active_cell"),
        State("sessions-table", "data"),
        prevent_initial_call=True,
    )
    def jump_to_session(active_cell, table_data):
        if not active_cell or not table_data:
            return no_update, no_update, no_update
        row_idx = active_cell.get("row")
        if row_idx is None or row_idx >= len(table_data):
            return no_update, no_update, no_update
        session_dir = table_data[row_idx].get("dir")
        if not session_dir:
            return no_update, no_update, no_update
        # Take the user to Evoked Waveforms with the session pre-filled.
        return session_dir, "analysis", "waveforms"

    # Home-grid 'Open' button callback moved to src/dashboard/tabs/overview.py.


    # ------------------------------------------------------------------ #
    #  Two-tier nav: top groups -> sub-tabs
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("tabs", "children"),
        Output("tabs", "value"),
        Input("group-tabs", "value"),
        State("tabs", "value"),
    )
    def populate_subtabs(group_id, current_sub):
        """Render the sub-tab strip for the active group.

        Preserves the current sub-tab if it belongs to the new group;
        otherwise falls back to the first sub-tab of the group.
        """
        group = GROUP_BY_ID.get(group_id) or NAV_GROUPS[0]
        subs = group["subs"]
        valid = {s["id"] for s in subs}
        new_sub = current_sub if current_sub in valid else subs[0]["id"]
        children = [
            dcc.Tab(label=s["label"], value=s["id"],
                    style=SUBTAB_STYLE, selected_style=SUBTAB_SELECTED_STYLE)
            for s in subs
        ]
        return children, new_sub

    # ------------------------------------------------------------------ #
    #  Main tab router — re-renders on tab change AND manual refresh
    # ------------------------------------------------------------------ #
    # Walk a Dash component tree and turn on session-scoped
    # persistence for every form control that has an id. Solves the
    # "I switched tabs and lost my session/file/filter selection"
    # problem in one place instead of touching every Dropdown/Input
    # call site. Skip components in _PERSIST_SKIP -- those are
    # ephemeral (load buttons share the same Input class, free-text
    # search fields shouldn't pin a stale query, etc).
    _PERSIST_KINDS = (dcc.Dropdown, dcc.Input, dcc.Checklist,
                       dcc.RadioItems, dcc.Slider, dcc.RangeSlider,
                       dcc.Tabs)
    _PERSIST_SKIP = {
        # Load / Apply / Refresh buttons aren't inputs but they
        # share a class hierarchy with the wrapped ones below;
        # listing here for documentation.
        "manual-refresh-btn", "focus-toggle-btn",
    }

    def _enable_persistence(node):
        # depth-first walk; bound at 5000 nodes (NASA rule 2) so
        # a malformed tree can't spin forever.
        stack = [node]
        guard = 0
        while stack and guard < 5000:
            guard += 1
            cur = stack.pop()
            if isinstance(cur, _PERSIST_KINDS):
                cid = getattr(cur, "id", None)
                if (isinstance(cid, str)
                        and cid and cid not in _PERSIST_SKIP):
                    # Only set if author didn't already opt in/out.
                    if getattr(cur, "persistence", None) is None:
                        cur.persistence = True
                        cur.persistence_type = "session"
            kids = getattr(cur, "children", None)
            if kids is None:
                continue
            if not isinstance(kids, list):
                kids = [kids]
            for k in kids:
                if k is not None:
                    stack.append(k)
        return node

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        State("selected-session-dir", "data"),
        State("lfp-to-video-bridge", "data"),
    )
    def render_tab(tab, session_hint, video_bridge):
        try:
            if tab == "overview":
                return _enable_persistence(tabs_overview.layout(store, config))
            elif tab == "waveforms":
                return _enable_persistence(
                    tabs_waveforms.layout(store, default=session_hint))
            elif tab == "signal":
                return _enable_persistence(tabs_signal_quality.layout(store))
            elif tab == "evoked":
                return _enable_persistence(tabs_evoked.layout(store))
            elif tab == "criticality":
                return _enable_persistence(tabs_criticality.layout(store))
            elif tab == "lfp":
                return _enable_persistence(
                    tabs_lfp_browser.layout(store, default=session_hint))
            elif tab == "video":
                # Pass the LFP-Browser / Event-Verification hand-off
                # bridge so the session is pre-selected AT BUILD TIME.
                # Consuming it via a post-mount callback raced the lazy
                # tab render (the callback fired before the dropdown
                # existed), so deep-links silently fell back to the
                # newest file.
                return _enable_persistence(
                    tabs_video.layout(store, bridge=video_bridge))
            elif tab == "electrode_health":
                return _enable_persistence(
                    tabs_electrode_health.layout(store, default=session_hint))
            elif tab == "session_compare":
                return _enable_persistence(
                    tabs_session_compare.layout(store))
            elif tab == "stim":
                return _enable_persistence(tabs_stim.layout(store))
            elif tab == "settings":
                return _enable_persistence(tabs_settings.layout(store))
            elif tab == "activity_log":
                return _enable_persistence(tabs_activity_log.layout(store))
            elif tab == "annotations":
                return _enable_persistence(tabs_annotations.layout(store))
            elif tab == "alerts":
                return _enable_persistence(tabs_alerts.layout(store))
            elif tab == "sessions":
                return _enable_persistence(tabs_sessions.layout(store))
            elif tab == "surgeries":
                return _enable_persistence(tabs_surgeries.layout(store, config))
            elif tab == "maintenance":
                return _enable_persistence(tabs_maintenance.layout(store, config))
            elif tab == "data_log_xref":
                return _enable_persistence(tabs_data_log_xref.layout(store, config))
            elif tab == "review_status":
                from src.dashboard.tabs import review_status as tabs_review_status
                return _enable_persistence(
                    tabs_review_status.layout(store, config))
            elif tab == "event_verification":
                from src.dashboard.tabs import event_verification as tabs_evtv
                return _enable_persistence(
                    tabs_evtv.layout(store, config))
        except Exception as e:
            logger.error("Dashboard render error: %s", e, exc_info=True)
            return html.Div(f"Error rendering tab: {e}",
                            style={"color": COLOR_DANGER, "padding": SPACE_5})

    # Evoked Features callbacks moved to src/dashboard/tabs/evoked.py.

    # Evoked Waveforms callbacks moved to src/dashboard/tabs/waveforms.py.

    # Criticality callback moved to src/dashboard/tabs/criticality.py.

    # LFP Browser callbacks moved to src/dashboard/tabs/lfp_browser.py.

    # ------------------------------------------------------------------ #
    # Electrode Health callback moved to
    # src/dashboard/tabs/electrode_health.py.

    # Session Compare callback moved to
    # src/dashboard/tabs/session_compare.py.

    # Activity Log callback moved to
    # src/dashboard/tabs/activity_log.py.

    # Annotations callback moved to
    # src/dashboard/tabs/annotations.py.

    # Settings callbacks moved to src/dashboard/tabs/settings.py.

    # ------------------------------------------------------------------ #
    #  Logged-in indicator in the header
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("logged-in-as", "children"),
        Input("tabs", "value"),
    )
    def show_user(_tab):
        email = current_user_email()
        if not email:
            return ""
        return f"signed in as {email}"

    # ------------------------------------------------------------------ #
    #  Video Review tab callbacks (modular)
    # ------------------------------------------------------------------ #
    tabs_video.register_callbacks(app, store, config)
    tabs_surgeries.register_callbacks(app, store, config)
    tabs_maintenance.register_callbacks(app, store, config)
    tabs_data_log_xref.register_callbacks(app, store, config)
    tabs_alerts.register_callbacks(app, store, config)
    tabs_signal_quality.register_callbacks(app, store, config)
    tabs_stim.register_callbacks(app, store, config)
    tabs_criticality.register_callbacks(app, store, config)
    tabs_session_compare.register_callbacks(app, store, config)
    tabs_activity_log.register_callbacks(app, store, config)
    tabs_annotations.register_callbacks(app, store, config)
    tabs_sessions.register_callbacks(app, store, config)
    tabs_electrode_health.register_callbacks(app, store, config)
    tabs_waveforms.register_callbacks(app, store, config)
    tabs_evoked.register_callbacks(app, store, config)
    tabs_settings.register_callbacks(app, store, config)
    tabs_lfp_browser.register_callbacks(app, store, config)
    tabs_overview.register_callbacks(app, store, config)
    # PI verification tab (gated on pi_emails). Import inline
    # so the legacy bootstrap path stays minimal.
    from src.dashboard.tabs import event_verification as _tabs_evtv
    _tabs_evtv.register_callbacks(app, store, config)

    # Opt-in dev diagnostic: with suppress_callback_exceptions=True a
    # callback wired to an id that no layout produces fails *silently*
    # at click time, which is exactly the regression the tab-extraction
    # work risks. Set QC_MONITOR_CHECK_CALLBACKS=1 to log any such
    # dangling ids at boot. Never raises; off by default so production
    # boot stays fast.
    if os.environ.get("QC_MONITOR_CHECK_CALLBACKS"):
        _diagnose_callback_wiring(app, store, config)

    return app


# ====================================================================== #
#  Callback-wiring diagnostic (opt-in, dev only)
# ====================================================================== #

def _collect_component_ids(component, into: set) -> None:
    """Recursively collect every plain-string component id in a Dash
    component tree. Dict (pattern-matching ALL/MATCH) ids are skipped
    -- they can't be resolved statically."""
    cid = getattr(component, "id", None)
    if isinstance(cid, str):
        into.add(cid)
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for ch in children:
            if hasattr(ch, "children") or getattr(ch, "id", None):
                _collect_component_ids(ch, into)
    elif hasattr(children, "children") or getattr(children, "id", None):
        _collect_component_ids(children, into)


def _known_component_ids(store: Store, config: dict) -> set:
    """Union of plain-string ids from the base layout plus every tab
    layout we can materialize. Each tab layout is built best-effort
    inside try/except so one failing tab doesn't blind the check for
    the rest."""
    ids: set = set()
    # Each modular tab. (store) and (store, config) signatures both
    # appear, so try the bare call and fall back to passing config.
    # The base shell layout is seeded separately by the caller.
    tab_layouts = [
        tabs_alerts.layout, tabs_signal_quality.layout,
        tabs_stim.layout, tabs_criticality.layout,
        tabs_session_compare.layout, tabs_activity_log.layout,
        tabs_annotations.layout, tabs_sessions.layout,
        tabs_electrode_health.layout, tabs_waveforms.layout,
        tabs_evoked.layout, tabs_settings.layout,
        tabs_video.layout, tabs_surgeries.layout,
        tabs_maintenance.layout, tabs_data_log_xref.layout,
        tabs_lfp_browser.layout, tabs_overview.layout,
    ]
    # review_status + event_verification are imported inline in
    # render_tab; pull them in here too so their ids count as known.
    try:
        from src.dashboard.tabs import review_status as _rs
        tab_layouts.append(_rs.layout)
    except Exception:
        pass
    try:
        from src.dashboard.tabs import event_verification as _ev
        tab_layouts.append(_ev.layout)
    except Exception:
        pass
    for fn in tab_layouts:
        try:
            comp = fn(store)
        except TypeError:
            try:
                comp = fn(store, config)
            except Exception:
                continue
        except Exception:
            continue
        _collect_component_ids(comp, ids)
    return ids


def _callback_referenced_ids(app) -> set:
    """Collect plain-string ids referenced as Output/Input/State by
    any registered callback. Dict ids and clientside-only specs are
    skipped."""
    ids: set = set()
    for cb in getattr(app, "_callback_list", []):
        for key in ("output", "inputs", "state"):
            spec = cb.get(key)
            if spec is None:
                continue
            items = spec if isinstance(spec, (list, tuple)) else [spec]
            for item in items:
                comp_id = None
                if isinstance(item, dict):
                    comp_id = item.get("id")
                else:
                    comp_id = getattr(item, "component_id", None)
                if isinstance(comp_id, str):
                    ids.add(comp_id)
    return ids


def _diagnose_callback_wiring(app, store: Store, config: dict) -> None:
    """Log any callback-referenced component id that no known layout
    produces. Best-effort + never raises: a noisy warning is far
    cheaper than a silently-dead tab.

    Caveats it can't see through (so a listed id may be a false
    positive): components created *inside* a callback's returned
    children, and pattern-matching dict ids. Treat the output as a
    lead, not a verdict."""
    try:
        known = _known_component_ids(store, config)
        # Seed the base shell ids too.
        if getattr(app, "layout", None) is not None:
            _collect_component_ids(app.layout, known)
        referenced = _callback_referenced_ids(app)
        dangling = sorted(referenced - known)
        logger.info(
            "Callback wiring check: %d callbacks, %d referenced ids, "
            "%d not found in any static layout.",
            len(getattr(app, "_callback_list", [])),
            len(referenced), len(dangling),
        )
        if dangling:
            logger.warning(
                "Callback ids not found in any static layout "
                "(may be dynamically created, or genuinely dangling): "
                "%s", ", ".join(dangling),
            )
    except Exception as e:
        logger.debug("Callback wiring check skipped: %s", e)


# ====================================================================== #
#  Tab builders
# ====================================================================== #

# Overview tab (home page) moved to src/dashboard/tabs/overview.py.


# Evoked Waveforms tab moved to src/dashboard/tabs/waveforms.py.


# Signal Quality tab moved to src/dashboard/tabs/signal_quality.py.


# Evoked Features tab moved to src/dashboard/tabs/evoked.py.


# Criticality tab moved to src/dashboard/tabs/criticality.py.


# LFP Browser tab moved to src/dashboard/tabs/lfp_browser.py.


# Electrode Health tab moved to src/dashboard/tabs/electrode_health.py.


# Session Compare tab moved to src/dashboard/tabs/session_compare.py.


# Stim QC tab moved to src/dashboard/tabs/stim.py.
# Settings tab moved to src/dashboard/tabs/settings.py.


