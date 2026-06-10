"""Dash web dashboard for the QC Monitor — v3 with evoked waveforms, LFP browser,
electrode health, session compare, activity log, and annotations."""

import json
import logging
import os
import statistics
from datetime import datetime, date, timedelta

import numpy as np
import yaml
from dash import Dash, html, dcc, dash_table, callback_context, no_update, ALL, Patch
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.db.store import Store
from src.utils.mat_loader import load_mat
from src.utils.chunk_cache import get_chunk
from src.utils.decimate import (
    envelope_channel, window_slice, choose_target_bins, parse_relayout,
)
from src.utils.filters import (
    SUPPORTED_NOTCH, apply_filter, compute_psd, get_filtered,
)
from src.dashboard.auth import register_auth, current_user_email
from src.dashboard import keyboard as _kbd
from src.utils import assignments as _assignments
from src.utils import event_clip as _event_clip
from src.utils import mass_analyze as _mass_analyze
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

CARD_STYLE = {
    "borderRadius": RADIUS_MD,
    "padding": f"{SPACE_4} {SPACE_5}",
    "minWidth": "130px",
    "textAlign": "center",
    "background": COLOR_SURFACE_1,
    "border": f"1px solid {COLOR_DIVIDER}",
    "transition": "transform 0.15s ease, background 0.15s ease",
}

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


def _collapsible(title: str, content, *, open_default: bool = True,
                  badge: str | None = None,
                  badge_color: str | None = None,
                  full_width: bool = False):
    """Section wrapped in a native <details> element so the user can
    collapse anything they don't want to see.

    Apple HIG: one container per section, not nested chrome. The
    collapsible IS the card. Inner content should NOT carry its own
    border/background/padding -- it'd duplicate the visual frame.
    The summary row is the title strip; the body just gets a thin
    horizontal padding so the inner figure / table edges align.
    """
    summary_children = [
        html.Span(title, style={
            "color": "#ddd", "fontSize": "12px", "fontWeight": "600",
            "letterSpacing": "0.2px",
        }),
    ]
    if badge:
        summary_children.append(html.Span(
            badge,
            style={
                "marginLeft": "6px",
                "padding": "0 6px",
                "borderRadius": "8px",
                "background": (badge_color or "rgba(255,255,255,0.08)"),
                "color": "#fff", "fontSize": "10px",
                "fontWeight": "600", "lineHeight": "16px",
            }))
    base_style = {
        "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}",
        "borderRadius": "6px",
        "marginBottom": "6px",
    }
    if full_width:
        base_style["gridColumn"] = "1 / -1"
    return html.Details([
        html.Summary(summary_children, style={
            "cursor": "pointer", "userSelect": "none",
            "padding": "5px 10px",
            "borderBottom":
                "1px solid rgba(255,255,255,0.04)",
            "listStyle": "none",
        }),
        html.Div(content, style={"padding": "6px 10px"}),
    ], open=open_default, style=base_style)


def _qc_monitor_version() -> str:
    """Short, human-readable version string for the dashboard header.

    There is no setup.py / pyproject.toml here, so we derive from git:
    ``<short-sha>[-dirty]  <YYYY-MM-DD>``. If git isn't reachable
    (deployed without .git, no PATH entry), returns ``unknown``.
    Computed once at import so the subprocess cost doesn't repeat on
    every page render.
    """
    import subprocess
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo_root, "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        dirty = False
    try:
        ts = subprocess.check_output(
            ["git", "-C", repo_root, "log", "-1", "--format=%cs"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        ts = ""
    label = sha + ("-dirty" if dirty else "")
    return f"{label}  {ts}".strip() if ts else label


_QC_MONITOR_VERSION = _qc_monitor_version()


# _parse_json_field, _get_channel_map, and _color_for_role moved to
# src/dashboard/data_helpers.py so the tab modules carved out of this
# file can reuse them without importing the app shell. The aliased
# imports above preserve every existing in-module callsite.


def _status_card(title: str, value: str, color: str = "#636EFA"):
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase",
                                "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ], style=CARD_STYLE)


def _fmt_disk_space(gb: float) -> str:
    """Format a disk-free value with auto-scaling unit.

    Anything 1024 GB and up reads as TB with one decimal (1.2 TB);
    smaller values stay as GB without a decimal (543 GB). Matches
    the "round up to the natural unit" pattern macOS Finder uses.
    """
    if gb is None:
        return "—"
    try:
        gb_f = float(gb)
    except (TypeError, ValueError):
        return "—"
    if gb_f >= 1024:
        return f"{gb_f / 1024:.1f} TB"
    return f"{gb_f:.0f} GB"


def _status_pill(title: str, value: str, color: str = "#636EFA"):
    """Compact single-line pill. ~28px tall instead of _status_card's
    ~72px. Reference info doesn't need to read as a headline; the
    operator only consults these when something looks wrong, so they
    earn their visual weight from color (not size)."""
    return html.Div([
        html.Span(title, style={
            "color": "#888", "fontSize": "10px",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
            "marginRight": "6px",
        }),
        html.Span(value, style={
            "color": color, "fontSize": "13px",
            "fontWeight": "600",
        }),
    ], style={
        "padding": "4px 10px",
        "background": COLOR_SURFACE_1,
        "border": f"1px solid {COLOR_DIVIDER}",
        "borderRadius": "999px",
        "fontFamily": "ui-monospace, SF Mono, monospace",
        "whiteSpace": "nowrap",
    })


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

    # Fine-grained refresh: replace just the volatile Overview cards +
    # queue children instead of re-rendering the whole tab. Eliminates
    # the white flash that used to happen on every refresh tick. The
    # callback is a no-op when the user's on any other tab (the target
    # divs don't exist in the rendered tree).
    @app.callback(
        Output("overview-cards", "children"),
        Output("overview-queue", "children"),
        Output("overview-home-grid", "children"),
        Output("overview-km-log", "children"),
        Input("refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def refresh_overview_dynamic(_n):
        from datetime import date as _date
        return (
            _build_overview_cards(store),
            _build_overview_queue(store),
            _build_home_grid_children(store, config, _date.today()),
            _build_km_log_section(config),
        )

    # Thumbnail has its own callback because the radio adds an
    # additional Input. Re-renders on either trigger; the figure
    # is cheap enough (5 channels × 138 files in overlay) that we
    # don't bother caching.
    @app.callback(
        Output("overview-thumbnail", "children"),
        Input("overview-trace-mode", "value"),
        Input("refresh-trigger", "data"),
    )
    def refresh_overview_thumbnail(trace_mode, _n):
        sessions = store.get_sessions()
        session_dir = sessions[0]["session_dir"] if sessions else ""
        return _build_overview_thumbnail(
            store, config, session_dir, trace_mode or "mean")

    @app.callback(
        Output("snapshot-expanded", "data"),
        Input("overview-snapshot-img", "n_clicks"),
        State("snapshot-expanded", "data"),
        prevent_initial_call=True,
    )
    def toggle_snapshot_size(n_clicks, current):
        # Each click flips between thumb and expanded states.
        return not bool(current)

    @app.callback(
        Output("overview-snapshot-img", "style"),
        Input("snapshot-expanded", "data"),
    )
    def apply_snapshot_size(expanded):
        if expanded:
            return {
                "display": "block",
                "width": "100%",
                "height": "auto",
                "maxWidth": "100%",
                "maxHeight": "none",
                "objectFit": "contain",
                "borderRadius": "4px",
                "background": "#0a0a14",
                "cursor": "zoom-out",
            }
        # Small thumbnail: maxWidth caps total width and the
        # browser keeps the aspect ratio via objectFit: contain.
        return {
            "display": "block",
            "maxWidth": "240px",
            "maxHeight": "180px",
            "width": "100%",
            "height": "auto",
            "objectFit": "contain",
            "borderRadius": "4px",
            "background": "#0a0a14",
            "cursor": "zoom-in",
        }

    @app.callback(
        Output("overview-snapshot-img", "src"),
        Output("overview-snapshot-caption", "children"),
        Input("refresh-trigger", "data"),
    )
    def refresh_overview_snapshot(_n):
        # ?t= cache-buster forces the browser to re-fetch each tick;
        # the Flask side caches the JPEG for ~8s so we don't actually
        # pummel the SMB share.
        import time as _t
        src = f"/media/latest-snapshot.jpg?t={int(_t.time())}"
        # Caption: recording window of the underlying chunk + camera
        # count so the operator can tell single- vs multi-camera
        # sessions apart at a glance.
        with store.connection() as conn:
            row = conn.execute(
                "SELECT file_path, chunk_datetime, duration_sec "
                "FROM processed_files "
                "WHERE has_video = 1 "
                "ORDER BY chunk_datetime DESC LIMIT 1"
            ).fetchone()
        if not row or not row["chunk_datetime"]:
            return src, "No videos available"
        from src.utils.video import companion_video_paths
        try:
            n_cams = len(companion_video_paths(row["file_path"] or ""))
        except Exception:
            n_cams = 0
        cam_tag = (f" · {n_cams} cameras" if n_cams > 1
                    else "")
        start_dt = _parse_chunk_dt(row["chunk_datetime"])
        if start_dt is None:
            return src, f"Latest recording{cam_tag}"
        dur = float(row["duration_sec"] or 3600.0)
        end_dt = start_dt + timedelta(seconds=dur)
        return src, (f"Chunk {start_dt.strftime('%H:%M')} – "
                      f"{end_dt.strftime('%H:%M')} (last frame)"
                      f"{cam_tag}")

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

    # ------------------------------------------------------------------ #
    #  Home page "Open >" buttons -> jump to the matching tab
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("group-tabs", "value", allow_duplicate=True),
        Output("tabs", "value", allow_duplicate=True),
        Input("home-open-surgeries", "n_clicks"),
        Input("home-open-schedule", "n_clicks"),
        Input("home-open-maintenance", "n_clicks"),
        Input("home-open-incidents", "n_clicks"),
        Input("home-open-datalog", "n_clicks"),
        prevent_initial_call=True,
    )
    def _home_open(_s, _sch, _m, _inc, _d):
        # Gate on a real button press. The home grid is re-instantiated
        # by refresh_overview_dynamic on every refresh-trigger tick;
        # without this check Dash interprets the fresh n_clicks=0 buttons
        # as a triggering event and would tab-switch the user out of
        # Overview into Lab on every poll. Also covers the initial-page
        # load case (Dash sometimes ignores prevent_initial_call when
        # multiple callbacks share Output via allow_duplicate).
        if not callback_context.triggered:
            return no_update, no_update
        trig_payload = callback_context.triggered[0]
        if not trig_payload.get("value"):  # n_clicks 0 / None
            return no_update, no_update
        trig = callback_context.triggered_id
        mapping = {
            "home-open-surgeries": ("lab", "surgeries"),
            "home-open-schedule": ("lab", "maintenance"),
            "home-open-maintenance": ("lab", "maintenance"),
            "home-open-incidents": ("lab", "maintenance"),
            "home-open-datalog": ("lab", "data_log_xref"),
        }
        if trig in mapping:
            return mapping[trig]
        return no_update, no_update

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
    )
    def render_tab(tab, session_hint):
        try:
            if tab == "overview":
                return _enable_persistence(_overview_tab(store, config))
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
                return _enable_persistence(tabs_video.layout(store))
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
        tabs_lfp_browser.layout,
        _overview_tab,
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

_HOME_ROW_STYLE = {
    "display": "flex", "alignItems": "center",
    "padding": f"{SPACE_1} 0",
}


def _home_surgery_block(store: Store, config: dict | None,
                         today: date) -> html.Div:
    """Today's surgery checklist in compact form."""
    try:
        from src.notifications.surgery_digest import _build_today
        grouped, _upcoming, _ = _build_today(today, config or {})
    except Exception as e:
        logger.warning("Home: surgery block failed: %s", e)
        grouped = {}

    rows = []
    bucket_labels = [
        ("Pre-op (Motrin)", "Pre-op (Motrin)"),
        ("Surgery day", "Surgery day"),
        ("Post-op day 1", "Post-op day 1"),
        ("Post-op day 2–3", ["Post-op day 2", "Post-op day 3"]),
    ]
    for label, key in bucket_labels:
        if isinstance(key, list):
            items = []
            for k in key:
                items.extend(grouped.get(k, []))
        else:
            items = grouped.get(key, [])
        n = len(items)
        animals = ", ".join(t["animal"] for t in items[:6])
        color = COLOR_TEXT_PRIMARY if n else COLOR_TEXT_TERTIARY
        rows.append(html.Div([
            html.Span(label, style={"flex": "0 0 200px", "color": color}),
            html.Span(str(n), style={"flex": "0 0 32px",
                                     "color": color, "fontWeight": "600"}),
            html.Span(animals, style={"flex": "1",
                                       "color": COLOR_TEXT_SECONDARY,
                                       "fontSize": FONT_SIZE_CAPTION}),
        ], style=_HOME_ROW_STYLE))

    return _card(
        _section_header("Surgeries today",
                         on_open_id="home-open-surgeries"),
        html.Div(rows),
    )


def _home_maintenance_block(config: dict | None) -> html.Div:
    """Today's maintenance status, mirroring the Apps Script truth."""
    try:
        from src.dashboard.tabs.maintenance import rig_status
        rows = rig_status(config or {})
    except Exception as e:
        logger.warning("Home: maintenance block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div("Maintenance tracker disabled or no data.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_BODY})
    else:
        attention = [r for r in rows
                     if r.status in ("pending", "overdue")]
        nothing_active = all(r.status == "not_active" for r in rows)
        if nothing_active:
            body = html.Div(
                "No active rigs today.",
                style={"color": COLOR_TEXT_TERTIARY,
                       "fontSize": FONT_SIZE_BODY},
            )
        elif not attention:
            done_count = sum(1 for r in rows if r.status == "done")
            if done_count:
                msg = f"All {done_count} scheduled task(s) done today."
                color = COLOR_SUCCESS
            else:
                msg = "Nothing scheduled today."
                color = COLOR_ACCENT
            body = html.Div(msg, style={
                "color": color, "fontSize": FONT_SIZE_BODY,
                "fontWeight": "600",
            })
        else:
            by_rig: dict[str, list] = {}
            for r in attention:
                by_rig.setdefault(r.rig, []).append(r)
            body_rows = []
            for rig in sorted(by_rig.keys()):
                chunks = []
                for r in by_rig[rig]:
                    short = ("battery" if r.task == "battery" else "cage")
                    label = short
                    if r.task == "battery" and r.pending_shelves:
                        label += (" shelf "
                                  + ",".join(map(str, r.pending_shelves)))
                    chunks.append(html.Span(
                        _pill(r.status, label=label),
                        style={"marginRight": SPACE_2},
                    ))
                body_rows.append(html.Div([
                    html.Span(f"Rig {rig}", style={
                        "flex": "0 0 80px", "fontWeight": "600",
                        "color": COLOR_TEXT_PRIMARY,
                    }),
                    html.Span(chunks, style={"flex": "1",
                                              "fontSize": FONT_SIZE_CAPTION}),
                ], style=_HOME_ROW_STYLE))
            body = html.Div(body_rows)

    return _card(
        _section_header("Maintenance",
                         on_open_id="home-open-maintenance"),
        body,
    )


def _home_schedule_block(config: dict | None) -> html.Div:
    """Today's scheduled tasks + assignees, sourced from the
    Maintenance Tracker '📅 Schedule' tab."""
    try:
        from src.dashboard.tabs.maintenance import todays_schedule
        rows = todays_schedule(config or {})
    except Exception as e:
        logger.warning("Home: schedule block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div(
            "Nothing scheduled today.",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_BODY},
        )
    else:
        by_task: dict[str, list] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        items = []
        for task in sorted(by_task.keys()):
            names = ", ".join(r["name"] or r["email"]
                              for r in by_task[task]
                              if (r["name"] or r["email"]))
            items.append(html.Div([
                html.Span(task, style={"flex": "0 0 200px",
                                        "color": COLOR_TEXT_PRIMARY,
                                        "fontWeight": "600"}),
                html.Span(names or "-",
                          style={"flex": "1",
                                 "color": COLOR_TEXT_SECONDARY,
                                 "fontSize": FONT_SIZE_CAPTION}),
            ], style=_HOME_ROW_STYLE))
        body = html.Div(items)

    return _card(
        _section_header("Today's schedule",
                         on_open_id="home-open-schedule"),
        body,
    )


import re as _re_incident

# Software is checked FIRST so "dashboard crashed" wins over the
# substring "board" in the hardware bucket.
_INCIDENT_TAGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("software", "#AB63FA",
        ("software", "code", "crash", "freeze", "hang", "deploy",
         "bug", "version", "update", "script", "matlab", "python",
         "dashboard", "kmrecord")),
    ("signal", "#19D3F3",
        ("noise", "drift", "artifact", "spike", "baseline",
         "channel", "filter", "60hz", "stim", "amplifier",
         "amp")),
    ("hardware", "#FFA15A",
        ("battery", "batteries", "electrode", "headstage", "cable",
         "connector", "board", "circuit", "broken", "charge",
         "charging", "wire", "led", "solder", "pin", "shaft")),
)
# Word-boundary regex per word so "dashboard" stops matching "board",
# "line" stops matching "online", etc. Compiled once.
_INCIDENT_PATTERNS: list[tuple[str, str, _re_incident.Pattern]] = [
    (label, color,
     _re_incident.compile(r"\b(" + "|".join(_re_incident.escape(w)
                                              for w in words) + r")",
                           _re_incident.IGNORECASE))
    for label, color, words in _INCIDENT_TAGS
]


def _classify_incident(text: str) -> tuple[str, str] | None:
    """First matching (label, color) for a free-form incident report,
    or None when no keyword family fires. Word-boundary regex --
    'Dashboard crashed' must not classify as hardware via 'board'."""
    if not text:
        return None
    for label, color, pattern in _INCIDENT_PATTERNS:
        if pattern.search(text):
            return (label, color)
    return None


def _incident_age_style(days_old: float | None) -> dict:
    """Visual aging for incidents based on how long ago they happened.

    Fresh (0-2 d): full opacity, normal weight.
    Recent (3-14 d): slight fade.
    Stale (15+ d): heavy fade so old open items don't read identical
    to fresh ones at a glance.
    """
    if days_old is None or days_old < 3:
        return {"opacity": "1.0"}
    if days_old < 15:
        return {"opacity": "0.75"}
    return {"opacity": "0.5"}


def _fmt_days_ago(days_old: float | None) -> tuple[str, str]:
    """(label, color) for a 'N days ago' badge."""
    if days_old is None:
        return "", "#888"
    if days_old < 1:
        return "today", "#00CC96"
    if days_old < 2:
        return "yesterday", "#00CC96"
    if days_old < 7:
        return f"{int(days_old)}d ago", "#FFA15A"
    if days_old < 30:
        return f"{int(days_old)}d ago", "#EF553B"
    return f"{int(days_old)}d ago", "#EF553B"


def _home_incidents_block(config: dict | None) -> html.Div:
    """Latest 3 incident reports from the Maintenance Tracker.

    Stale incidents fade visually so an open battery report from
    three weeks ago doesn't claim the same attention as a fresh
    one. Each item gets a keyword-derived severity badge (hardware
    / signal / software) so the operator can triage at a glance.
    """
    try:
        from src.dashboard.tabs.maintenance import recent_incidents
        rows = recent_incidents(config or {}, limit=3)
    except Exception as e:
        logger.warning("Home: incidents block failed: %s", e)
        rows = []

    if not rows:
        body = html.Div(
            "No incident reports.",
            style={"color": COLOR_TEXT_TERTIARY,
                   "fontSize": FONT_SIZE_BODY},
        )
    else:
        now = datetime.now()
        items = []
        for r in rows:
            when_dt = r["when"]
            when = when_dt.strftime("%Y-%m-%d %H:%M")
            who = r.get("by") or "?"
            report = r.get("report") or ""
            if len(report) > 140:
                report = report[:137] + "..."
            try:
                days_old = max(0.0,
                                (now - when_dt).total_seconds() / 86400.0)
            except (TypeError, ValueError):
                days_old = None
            age_label, age_color = _fmt_days_ago(days_old)
            tag = _classify_incident(report)
            age_style = _incident_age_style(days_old)
            header_children = [
                html.Span(when, style={
                    "color": COLOR_TEXT_SECONDARY,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginRight": SPACE_3,
                }),
                html.Span(who, style={
                    "color": COLOR_ACCENT,
                    "fontSize": FONT_SIZE_CAPTION,
                    "fontWeight": "600",
                    "marginRight": SPACE_3,
                }),
            ]
            if age_label:
                header_children.append(html.Span(
                    age_label,
                    style={
                        "color": age_color, "fontSize": "10px",
                        "fontWeight": "600",
                        "marginRight": SPACE_3,
                    }))
            if tag is not None:
                tag_label, tag_color = tag
                header_children.append(html.Span(
                    tag_label,
                    style={
                        "color": "#fff", "fontSize": "9px",
                        "fontWeight": "600",
                        "background": tag_color,
                        "padding": "1px 6px",
                        "borderRadius": "8px",
                        "letterSpacing": "0.3px",
                        "textTransform": "uppercase",
                    }))
            items.append(html.Div([
                html.Div(header_children, style={
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "rowGap": "2px",
                }),
                html.Div(report, style={
                    "color": COLOR_TEXT_PRIMARY,
                    "fontSize": FONT_SIZE_BODY,
                    "marginTop": "2px",
                }),
            ], style={
                "padding": f"{SPACE_2} 0",
                "borderBottom": f"1px solid {COLOR_DIVIDER}",
                **age_style,
            }))
        body = html.Div(items)

    return _card(
        _section_header("Recent incidents",
                         on_open_id="home-open-incidents"),
        body,
    )


def _home_data_log_block(store: Store, config: dict | None) -> html.Div:
    try:
        from src.dashboard.tabs.data_log_xref import diff_counts
        counts = diff_counts(store, config or {})
    except Exception as e:
        logger.warning("Home: data log block failed: %s", e)
        counts = {"enabled": False}

    if not counts.get("enabled"):
        body = html.Div("Data log cross-reference disabled.",
                        style={"color": COLOR_TEXT_TERTIARY,
                               "fontSize": FONT_SIZE_BODY})
    else:
        ln = counts.get("logged_not_qcd", 0)
        qn = counts.get("qcd_not_logged", 0)
        if ln == 0 and qn == 0:
            body = html.Div("Sheet and DB are in sync.",
                            style={"color": COLOR_SUCCESS,
                                   "fontSize": FONT_SIZE_BODY,
                                   "fontWeight": "600"})
        else:
            body = html.Div([
                html.Span(f"{ln} ", style={"color": COLOR_WARNING,
                                            "fontWeight": "700",
                                            "fontSize": "16px"}),
                html.Span("logged, not QC'd",
                         style={"color": COLOR_TEXT_SECONDARY,
                                "marginRight": SPACE_5,
                                "fontSize": FONT_SIZE_BODY}),
                html.Span(f"{qn} ", style={"color": COLOR_ACCENT,
                                            "fontWeight": "700",
                                            "fontSize": "16px"}),
                html.Span("QC'd, not logged",
                         style={"color": COLOR_TEXT_SECONDARY,
                                "fontSize": FONT_SIZE_BODY}),
            ])

    return _card(
        _section_header("Data log diff",
                         on_open_id="home-open-datalog"),
        body,
    )


def _home_quick_links_block(config: dict | None) -> html.Div:
    """One tile holding every external resource the Overview cards
    point at (Google Sheets, drives, dashboards) so the operator
    doesn't have to dig into config or a separate doc to find a URL.

    Apple HIG: grouped list of related entry points. Each row is a
    plain anchor opening in a new tab; the sheet_id comes from
    config (same source the digest / data-log tabs already use).
    """
    cfg = config or {}
    links: list[tuple[str, str, str]] = []

    def _sheet_url(sid: str) -> str:
        return f"https://docs.google.com/spreadsheets/d/{sid}/edit"

    # Maintenance + incidents (one sheet, multiple tabs)
    maint_id = (cfg.get("maintenance", {}) or {}).get("sheet_id")
    if maint_id:
        links.append(("Maintenance tracker", "sheet",
                      _sheet_url(maint_id)))

    # Surgery / implant sheets
    for s in (cfg.get("surgeries", {}) or {}).get("sheets", []) or []:
        sid = s.get("sheet_id")
        label = s.get("label") or "Surgery log"
        if sid:
            links.append((label, "sheet", _sheet_url(sid)))

    # JAX cage index
    cage_cfg = ((cfg.get("surgeries", {}) or {})
                 .get("cage_index", {}) or {})
    cage_id = cage_cfg.get("sheet_id")
    if cage_id and cage_cfg.get("enabled"):
        links.append(("JAX cage index", "sheet", _sheet_url(cage_id)))

    # KMrecorder Data Log
    xref_id = (cfg.get("data_log_xref", {}) or {}).get("sheet_id")
    if xref_id:
        links.append(("KMrecorder Data Log", "sheet",
                      _sheet_url(xref_id)))

    if not links:
        body = html.Div(
            "No external resources configured.",
            style={"color": COLOR_TEXT_TERTIARY,
                    "fontSize": FONT_SIZE_BODY},
        )
    else:
        items = []
        for label, kind, href in links:
            tag_color = ("#5e7ce2" if kind == "sheet"
                          else COLOR_TEXT_TERTIARY)
            items.append(html.A([
                html.Span(label, style={
                    "color": COLOR_TEXT_PRIMARY,
                    "fontSize": FONT_SIZE_BODY,
                    "flex": "1",
                }),
                html.Span(kind.upper(), style={
                    "color": tag_color,
                    "fontSize": "10px", "letterSpacing": "0.5px",
                    "fontWeight": "600",
                    "marginLeft": SPACE_3,
                }),
            ], href=href, target="_blank", rel="noopener",
               style={
                   "display": "flex", "alignItems": "center",
                   "padding": f"{SPACE_2} 0",
                   "borderBottom": f"1px solid {COLOR_DIVIDER}",
                   "textDecoration": "none",
               }))
        body = html.Div(items)

    return _card(
        _section_header("Quick links"),
        body,
    )


def _build_home_grid_children(store: Store, config: dict | None,
                                today: date) -> list:
    """Lab-side home tiles. Extracted so the refresh callback can
    rebuild them in place without re-rendering everything else."""
    return [
        _home_surgery_block(store, config, today),
        _home_schedule_block(config),
        _home_maintenance_block(config),
        _home_incidents_block(config),
        _home_data_log_block(store, config),
        _home_quick_links_block(config),
    ]


def _build_overview_cards(store: Store):
    """Inner content for the system-health card strip. Returned without
    a wrapping Div so a callback can swap it via Patch / children."""
    health = store.get_health_history(hours=1)
    latest = health[-1] if health else {}
    net_ok = latest.get("network_share_accessible")
    cpu = latest.get("cpu_pct", 0)
    mem = latest.get("memory_pct", 0)
    disk = latest.get("disk_free_gb", 0)
    fph = latest.get("files_processed_last_hour", 0)
    # Video QC: 24-hour status mix. Color reflects worst severity in
    # the window (critical > warning > ok). Empty window stays gray.
    try:
        videos = store.get_recent_video_qc(hours=24)
    except Exception:
        videos = []
    n_ok = sum(1 for v in videos if v.get("status") == "ok")
    n_warn = sum(1 for v in videos if v.get("status") == "warning")
    n_crit = sum(1 for v in videos if v.get("status") == "critical")
    if videos:
        vid_label = f"{n_ok}/{n_warn}/{n_crit}"
    else:
        vid_label = "—"
    vid_color = ("#EF553B" if n_crit
                  else "#FFA15A" if n_warn
                  else "#00CC96" if n_ok
                  else "#666")
    return [
        _status_pill("Network", "OK" if net_ok else "DOWN",
                     "#00CC96" if net_ok else "#EF553B"),
        _status_pill("CPU", f"{cpu:.0f}%",
                     "#00CC96" if cpu < 80 else "#FFA15A"),
        _status_pill("Memory", f"{mem:.0f}%",
                     "#00CC96" if mem < 85 else "#FFA15A"),
        _status_pill("Disk Free", _fmt_disk_space(disk),
                     "#00CC96" if disk > 50 else "#EF553B"),
        _status_pill("Files/Hour", str(fph), "#636EFA"),
        _status_pill("Videos (24h)", vid_label, vid_color),
    ]


def _yrange_pad(time_ms, trace, x0: float, x1: float,
                  fill_frac: float = 0.8) -> tuple[float, float] | None:
    """Y-axis range such that data within [x0, x1] spans *fill_frac*
    of the plot's vertical extent (default 80%, i.e. 10% padding each
    side). Returns None when there are fewer than 3 finite samples in
    the window so the caller can fall back to Plotly's autorange.
    """
    if not time_ms or not trace or len(time_ms) != len(trace):
        return None
    vals = [v for t, v in zip(time_ms, trace)
            if t is not None and v is not None
            and x0 <= t <= x1 and isinstance(v, (int, float)) and v == v]
    if len(vals) < 3:
        return None
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span <= 0:
        # Flat trace; give the axis a tiny window so the line draws.
        eps = abs(hi) * 0.05 if hi != 0 else 1.0
        return lo - eps, hi + eps
    # We want data span = fill_frac * total span ⇒ total span =
    # data_span / fill_frac. Pad = (total - data) / 2 on each side.
    pad = span * (1.0 - fill_frac) / (2.0 * fill_frac)
    return lo - pad, hi + pad


def _build_overview_thumbnail(store: Store, config: dict | None,
                                session_dir: str, trace_mode: str
                                ) -> list:
    """Return the children list for the Overview waveform-thumbnail
    Div. *trace_mode* ∈ {"mean", "sem", "overlay"}.

    Both left (stim ±1 ms) and right (analysis window) panels show the
    LFP mean_trace -- the left panel is just a zoomed view around t=0,
    not the stim copy channel. Y-axis on every panel scales so the
    data spans 80% of the plot height.
    """
    if not session_dir:
        return [html.Div()]
    try:
        waveforms = store.get_evoked_waveforms_for_session(session_dir)
    except Exception as e:
        logger.debug("Could not load waveform thumbnail: %s", e)
        return [html.Div()]
    if not waveforms:
        return [html.Div()]

    fa_cfg = (config or {}).get("feature_analysis", {}) or {}
    evoked_x0 = float(fa_cfg.get("analysis_start_ms", 2.0))
    evoked_x1 = float(fa_cfg.get("analysis_end_ms", 200.0))
    stim_x0, stim_x1 = -1.0, 1.0

    # Latest waveform per channel + the full per-channel history so
    # overlay mode has something to draw under the mean.
    latest_by_ch: dict[int, dict] = {}
    history_by_ch: dict[int, list[dict]] = {}
    latest_datetime = ""
    for wf in waveforms:
        ch = int(wf.get("channel", 0))
        latest_by_ch[ch] = wf
        history_by_ch.setdefault(ch, []).append(wf)
        dt = wf.get("chunk_datetime", "") or ""
        if dt > latest_datetime:
            latest_datetime = dt
    sorted_chs = sorted(latest_by_ch.keys())
    n_ch = len(sorted_chs)
    if n_ch == 0:
        return [html.Div()]

    colors_list = ["#636EFA", "#00CC96", "#FFA15A", "#EF553B", "#AB63FA"]
    chan_label = {
        ch: latest_by_ch[ch].get("channel_name", f"Ch{ch}")
        for ch in sorted_chs
    }
    # Per-cell titles dropped (channel name lives on y-axis, column
    # name lives on the figure-level annotations below). Spacing
    # bumped to 0.08 + non-bottom rows hide their x ticks, so tick
    # labels from row N can no longer collide with row N+1's plot
    # top or with the next row's title region.
    thumb_fig = make_subplots(
        rows=n_ch, cols=2, column_widths=[0.16, 0.84],
        shared_xaxes=False, subplot_titles=None,
        vertical_spacing=0.08, horizontal_spacing=0.035,
    )

    def _hex_to_rgba(hex_str: str, alpha: float) -> str:
        h = hex_str.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for ri, ch in enumerate(sorted_chs, 1):
        wf = latest_by_ch[ch]
        nm = chan_label[ch]
        n_ep = wf.get("n_epochs", 0)
        time_ms = wf["time_axis_ms"]
        mean_tr = wf["mean_trace"]
        sem_tr = wf.get("sem_trace")
        color = colors_list[(ri - 1) % len(colors_list)]
        band_color = _hex_to_rgba(color, 0.18)
        overlay_color = _hex_to_rgba(color, 0.12)
        # Group every trace for this channel under one legendgroup so
        # clicking the legend entry hides stim+evoked+SEM band+overlay
        # simultaneously. Lets the user isolate one channel cleanly
        # (Plotly: single-click hide, double-click isolate).
        lg = f"ch_{ch}"

        # Overlay mode: draw every file's mean_trace under this
        # channel's latest mean. Cheap stylized backdrop, no legend.
        if trace_mode == "overlay":
            for prev in history_by_ch[ch]:
                if prev is wf:
                    continue
                pt = prev.get("time_axis_ms")
                pm = prev.get("mean_trace")
                if not pt or not pm or len(pt) != len(pm):
                    continue
                # Left + right panels both get the overlay so the
                # stim-window context matches the evoked window.
                for col_idx in (1, 2):
                    thumb_fig.add_trace(go.Scatter(
                        x=pt, y=pm, mode="lines",
                        line=dict(color=overlay_color, width=0.7),
                        hoverinfo="skip", showlegend=False,
                        legendgroup=lg,
                    ), row=ri, col=col_idx)

        # SEM band: filled ribbon mean±sem, mean drawn on top.
        if trace_mode == "sem" and sem_tr and len(sem_tr) == len(mean_tr):
            upper = [m + s for m, s in zip(mean_tr, sem_tr)]
            lower = [m - s for m, s in zip(mean_tr, sem_tr)]
            for col_idx in (1, 2):
                thumb_fig.add_trace(go.Scatter(
                    x=time_ms, y=upper, mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=lg,
                ), row=ri, col=col_idx)
                thumb_fig.add_trace(go.Scatter(
                    x=time_ms, y=lower, mode="lines",
                    fill="tonexty", fillcolor=band_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=lg,
                ), row=ri, col=col_idx)

        # Left panel: LFP mean, zoomed to ±1 ms around stim.
        thumb_fig.add_trace(go.Scatter(
            x=time_ms, y=mean_tr, mode="lines",
            name=f"{nm} stim",
            line=dict(color=color, width=1.5),
            showlegend=False, legendgroup=lg,
        ), row=ri, col=1)
        thumb_fig.add_vline(
            x=0, line=dict(color="white", width=0.5, dash="dash"),
            row=ri, col=1)
        thumb_fig.update_xaxes(range=[stim_x0, stim_x1], row=ri, col=1)
        thumb_fig.update_yaxes(title_text=nm, title_font_size=10,
                                row=ri, col=1)
        y_left = _yrange_pad(time_ms, mean_tr, stim_x0, stim_x1)
        if y_left is not None:
            thumb_fig.update_yaxes(range=list(y_left), row=ri, col=1)

        # Right panel: LFP mean, evoked window. The visible legend
        # entry lives on this trace (showlegend=True default) and
        # carries the channel group so clicking it hides the whole
        # row across both columns.
        thumb_fig.add_trace(go.Scatter(
            x=time_ms, y=mean_tr, mode="lines",
            name=f"{nm} (n={n_ep})",
            line=dict(color=color, width=1.5),
            legendgroup=lg,
        ), row=ri, col=2)
        thumb_fig.update_xaxes(range=[evoked_x0, evoked_x1],
                                row=ri, col=2)
        y_right = _yrange_pad(time_ms, mean_tr, evoked_x0, evoked_x1)
        if y_right is not None:
            thumb_fig.update_yaxes(range=list(y_right), row=ri, col=2)

        # Only the bottom row carries x-axis ticks + title. Non-
        # bottom rows hide tick labels so the "-1 -0.5 0 0.5 1"
        # labels of row N can't collide with row N+1's plot top
        # at tight vertical spacing.
        if ri < n_ch:
            thumb_fig.update_xaxes(showticklabels=False, row=ri, col=1)
            thumb_fig.update_xaxes(showticklabels=False, row=ri, col=2)

    thumb_fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=1)
    thumb_fig.update_xaxes(title_text="Time (ms)", row=n_ch, col=2)
    mode_label = {"mean": "mean",
                   "sem": "mean ± SEM",
                   "overlay": "mean + per-file overlay"}.get(
        trace_mode, "mean")
    # Viewport-aware: autosize=True + a CSS calc() height on the
    # Graph container makes Plotly redraw to fill whatever vertical
    # space the viewport allows. minHeight keeps each channel row
    # readable on a 1080p screen; maxHeight prevents the figure from
    # going absurdly tall on a 4K monitor.
    # Legend back on, but as a vertical strip in the right margin --
    # avoids the title collision the top-right horizontal legend
    # caused, and Plotly's built-in legend behavior (click to hide,
    # double-click to isolate) gives the user a way to focus on one
    # channel when three overlapping traces get noisy.
    thumb_fig.update_layout(
        title=dict(
            text=f"Latest Evoked — {latest_datetime[:16]} ({mode_label})",
            font=dict(size=11), x=0.5, xanchor="center",
            y=0.985, yanchor="top"),
        autosize=True,
        margin=dict(l=42, r=130, t=44, b=32),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.005, y=1, xanchor="left",
            yanchor="top", font_size=10, bgcolor="rgba(0,0,0,0)",
            itemclick="toggle",        # single-click = hide/show
            itemdoubleclick="toggleothers",  # double = isolate
        ),
    )
    # Column headers as figure-level annotations, sitting just below
    # the title. x-positions track column_widths=[0.16, 0.84] +
    # horizontal_spacing=0.035; update both if the subplot grid
    # geometry changes.
    thumb_fig.add_annotation(
        xref="paper", yref="paper",
        x=0.04, y=1.01, xanchor="left", yanchor="bottom",
        text=f"Stim window  {stim_x0:g} to {stim_x1:g} ms",
        showarrow=False,
        font=dict(size=10, color="#888"),
    )
    thumb_fig.add_annotation(
        xref="paper", yref="paper",
        x=0.22, y=1.01, xanchor="left", yanchor="bottom",
        text=f"Evoked  {evoked_x0:g}–{evoked_x1:g} ms",
        showarrow=False,
        font=dict(size=10, color="#888"),
    )
    return [dcc.Graph(
        figure=thumb_fig, responsive=True,
        style={
            "marginTop": "4px",
            # 58vh (down from 62) keeps the title strip clearly
            # readable at common viewports without losing channel
            # legibility. min keeps each row readable on a small
            # laptop; max caps absurd 4K heights.
            "height": "58vh",
            "minHeight": f"{max(420, 108 * n_ch)}px",
            "maxHeight": "780px",
        },
    )]


def _parse_end_datetime(s: str | None) -> datetime | None:
    """Parse the sheet's End_DateTime column.

    KMrecorder writes the row when the chunk finishes, so the
    End_DateTime cell is the closest the sheet has to "row submitted
    at." Values look like '6/1/2026 16:06:27' (no leading zeros) but
    we accept the ISO form too in case the column gets normalised.
    """
    if not s:
        return None
    text = str(s).strip()
    if not text:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace(" ", "T"))
    except ValueError:
        return None


def _km_log_summary(config: dict | None) -> dict:
    """Most-recent KMrecorder data-log entries + freshness in minutes.

    Pulls from the same Google Sheet the data_log_xref tab uses, so
    we share the existing TTL cache (no extra API quota cost). The
    sheet has a Filename column (recording start, derived from the
    .mat filename) and an End_DateTime column (when the chunk
    finished -- KMrecorder submits the row at that point, so we
    use it as "submitted at").
    """
    out = {
        "enabled": False, "rows": [], "latest_iso": None,
        "age_min": None,
        "latest_submitted_iso": None,
        "submitted_age_min": None,
    }
    cfg = (config or {}).get("data_log_xref", {}) or {}
    if not cfg.get("enabled", False):
        return out
    try:
        from src.dashboard.tabs.surgeries import (
            _load_sheet_via_api, _resolve_sa_path, _find_column,
            _normalize_text,
        )
        from src.dashboard.tabs.data_log_xref import _extract_key
    except Exception:
        return out
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "")
    sa = _resolve_sa_path(config or {},
                           cfg.get("service_account_file", ""))
    if not (sheet_id and tab_name and sa):
        return out
    ttl_sec = float(cfg.get("refresh_minutes", 15)) * 60.0
    try:
        df = _load_sheet_via_api(sheet_id, tab_name, sa, ttl_sec)
    except Exception as e:
        logger.debug("KM log fetch failed: %s", e)
        return out
    if df is None or df.empty:
        out["enabled"] = True
        return out

    filename_col = _find_column(df, ["Filename", "File", "Path"])
    animal_col = _find_column(df, ["Animal_ID", "Animal"])
    pc_col = _find_column(df, ["PC Name", "PC"])
    end_dt_col = _find_column(df, ["End_DateTime", "EndDateTime",
                                      "End"])
    operator_col = _find_column(df, ["Operator"])
    date_col = _find_column(df, ["Date"])
    rec_end_col = _find_column(df, ["Recording_End", "RecordingEnd"])
    if filename_col is None:
        out["enabled"] = True
        return out

    def _resolve_submitted(row) -> datetime | None:
        """Submission timestamp: prefer End_DateTime column; fall back
        to Date + Recording_End when the recorder doesn't populate
        End_DateTime (recent KMrecorder versions leave it blank)."""
        if end_dt_col:
            dt = _parse_end_datetime(row.get(end_dt_col))
            if dt is not None:
                return dt
        if not (date_col and rec_end_col):
            return None
        date_s = _normalize_text(row.get(date_col))
        end_s = _normalize_text(row.get(rec_end_col))
        if not (date_s and end_s):
            return None
        # Date is "YYYY-MM-DD"; Recording_End is "HH:MM:SS".
        try:
            return datetime.strptime(f"{date_s} {end_s}",
                                       "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    # Build records keyed by recording-start (chunk filename key)
    # so they sort chronologically lexicographically.
    records: dict[str, dict] = {}
    for _idx, row in df.iterrows():
        key = _extract_key(row.get(filename_col))
        if key is None:
            continue
        records[key] = {
            "key": key,
            "animal": (_normalize_text(row.get(animal_col))
                        if animal_col else ""),
            "pc": (_normalize_text(row.get(pc_col))
                    if pc_col else ""),
            "operator": (_normalize_text(row.get(operator_col))
                          if operator_col else ""),
            "submitted_dt": _resolve_submitted(row),
        }
    if not records:
        out["enabled"] = True
        return out

    sorted_keys = sorted(records.keys(), reverse=True)
    rows = [records[k] for k in sorted_keys[:5]]
    latest_key = sorted_keys[0]
    try:
        latest_dt = datetime.strptime(latest_key, _CHUNK_DT_FMT)
        age_min = (datetime.now() - latest_dt).total_seconds() / 60.0
    except ValueError:
        latest_dt, age_min = None, None
    latest_sub_dt = records[latest_key].get("submitted_dt")
    sub_age_min = ((datetime.now()
                     - latest_sub_dt).total_seconds() / 60.0
                    if latest_sub_dt else None)
    out.update({
        "enabled": True, "rows": rows,
        "latest_iso": latest_dt.isoformat() if latest_dt else None,
        "age_min": age_min,
        "latest_submitted_iso": (latest_sub_dt.isoformat()
                                    if latest_sub_dt else None),
        "submitted_age_min": sub_age_min,
    })
    return out


def _fmt_age(minutes: float | None) -> tuple[str, str]:
    """(label, color) for an age in minutes. Green <90 min, orange
    <24 h, red older. Used for both 'recorded' and 'submitted'."""
    if minutes is None:
        return "unknown", "#888"
    if minutes < 90:
        return f"{minutes:.0f} min ago", "#00CC96"
    if minutes < 24 * 60:
        return f"{minutes / 60:.1f} h ago", "#FFA15A"
    return f"{minutes / (24 * 60):.1f} d ago", "#EF553B"


def _build_km_log_section(config: dict | None) -> html.Div:
    """KM Recorder freshness header + last-5-entries strip. Each row
    shows the recording-start timestamp (from the filename) AND the
    sheet's End_DateTime, which is when KMrecorder finished the chunk
    and submitted the row -- the closest proxy this lab has for
    'last contact' since the recorder is what writes to the sheet."""
    summary = _km_log_summary(config)
    if not summary["enabled"]:
        return html.Div()

    rec_text, rec_color = _fmt_age(summary["age_min"])
    sub_text, sub_color = _fmt_age(summary["submitted_age_min"])

    rows = summary["rows"]
    if rows:
        tail = []
        for r in rows:
            ts_str = r.get("key", "")[:16].replace("_", "/")
            sub_dt = r.get("submitted_dt")
            sub_str = (sub_dt.strftime("%m/%d %H:%M")
                        if sub_dt else "?")
            label = (
                f"rec {ts_str}  ->  submitted {sub_str}  ·  "
                f"{r.get('animal') or '?'}  ·  {r.get('pc') or '?'}"
            )
            tail.append(html.Div(label, style={
                "color": "#aaa", "fontSize": "11px",
                "padding": "2px 0",
                "overflow": "hidden", "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }))
    else:
        tail = [html.Div("No rows from KMrecorder log yet.",
                          style={"color": "#888", "fontSize": "11px"})]

    return html.Div([
        html.Div([
            html.Span(f"recording {rec_text}",
                       style={"color": rec_color, "fontSize": "11px",
                               "fontWeight": "600",
                               "marginRight": "10px"}),
            html.Span(f"submitted {sub_text}",
                       style={"color": sub_color, "fontSize": "11px",
                               "fontWeight": "600"}),
        ], style={"marginBottom": "4px"}),
        html.Div(tail),
    ])


_CHUNK_DT_FMT = "%Y_%m_%d__%H_%M_%S"


def _parse_chunk_dt(ts: str | None) -> datetime | None:
    """Parse a chunk_datetime string into a datetime.

    chunk_datetime is stored in filename format (YYYY_MM_DD__HH_MM_SS)
    not ISO -- the recorder writes the timestamp into the filename and
    the watcher persists it as-is. Falls back to fromisoformat() so a
    future schema change doesn't silently break this.
    """
    if not ts:
        return None
    try:
        return datetime.strptime(ts, _CHUNK_DT_FMT)
    except ValueError:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None


def _recording_arrivals(store: Store, hours: int
                          ) -> list[tuple[str, float]]:
    """Back-compat shim returning (chunk_datetime, duration_sec)."""
    return [(ts, dur) for ts, dur, _sd, _sn
            in _recording_arrivals_with_session(store, hours)]


def _recording_arrivals_with_session(
        store: Store, hours: int
        ) -> list[tuple[str, float, str, str]]:
    """(chunk_datetime, duration_sec, session_dir, session_name) for
    every chunk whose coverage might overlap the last *hours* window.

    The cutoff is widened by 2h so a chunk that started just before
    the window and runs INTO it still appears.
    """
    assert hours > 0, "hours must be positive"
    cutoff_dt = datetime.now() - timedelta(hours=hours + 2)
    cutoff_str = cutoff_dt.strftime(_CHUNK_DT_FMT)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT chunk_datetime, duration_sec, session_dir, "
            "       session_name "
            "FROM processed_files "
            "WHERE chunk_datetime >= ? ORDER BY chunk_datetime",
            (cutoff_str,),
        ).fetchall()
        out: list[tuple[str, float, str, str]] = []
        for r in rows:
            ts = r["chunk_datetime"]
            if not ts:
                continue
            dur = float(r["duration_sec"] or 0.0)
            if dur <= 0:
                dur = 3600.0
            sd = str(r["session_dir"] or "")
            sn = str(r["session_name"] or sd.rsplit("/", 1)[-1] or "?")
            out.append((ts, dur, sd, sn))
        return out


# Per-session color palette for the 24h band. Cycles when there are
# more sessions than colors. Greens/blues/oranges/etc. are deliberate
# -- the "no recording" gray (#2a2a40) stays clearly distinct.
_SESSION_PALETTE = [
    "#00CC96", "#636EFA", "#FFA15A", "#EF553B", "#AB63FA",
    "#19D3F3", "#FF6692", "#FECB52", "#B6E880", "#7be3c0",
]
_NO_REC_COLOR = "#2a2a40"


def _session_color_map(arrivals_with_session: list[tuple]
                        ) -> dict[str, str]:
    """Assign a stable color to each session_dir in arrival order.

    The first session to appear gets the first palette color, etc.
    Sessions beyond the palette wrap around -- the visual gives a hint
    that a new session started, even if two distant ones reuse a color.
    """
    order: list[str] = []
    seen: set[str] = set()
    for _ts, _dur, sd, _sn in arrivals_with_session:
        if sd and sd not in seen:
            seen.add(sd)
            order.append(sd)
    return {sd: _SESSION_PALETTE[i % len(_SESSION_PALETTE)]
            for i, sd in enumerate(order)}


def _build_recording_24h_fig(
        arrivals_with_session: list[tuple[str, float, str, str]],
        session_colors: dict[str, str] | None = None
        ) -> go.Figure:
    """24-hour recording-uptime band. 5-minute bins (288 across the
    day) so short interruptions and quick session swaps are visible.
    Color = the session whose chunk covers that bin; gray = no chunk
    covered the bin. If a bin is covered by multiple sessions (rare
    -- only at hand-off seams), the LATEST-starting chunk wins."""
    now = datetime.now()
    bin_min = 5
    n_bins = 24 * 60 // bin_min   # 288
    bin_starts = [now - timedelta(minutes=(n_bins - i) * bin_min)
                   for i in range(n_bins)]
    bin_ends = [b + timedelta(minutes=bin_min) for b in bin_starts]
    sess_per_bin: list[str | None] = [None] * n_bins
    # Latest chunk start wins on overlap, so iterate in original
    # (ascending-by-start) order and overwrite.
    cover_start: list[datetime | None] = [None] * n_bins
    for ts, dur, sd, _sn in arrivals_with_session:
        dt = _parse_chunk_dt(ts)
        if dt is None or not sd:
            continue
        ch_end = dt + timedelta(seconds=dur)
        if ch_end <= bin_starts[0] or dt >= bin_ends[-1]:
            continue
        for i in range(n_bins):
            if dt >= bin_ends[i]:
                continue
            if ch_end <= bin_starts[i]:
                break
            if (cover_start[i] is None or dt >= cover_start[i]):
                sess_per_bin[i] = sd
                cover_start[i] = dt
    colors = (session_colors
               if session_colors is not None
               else _session_color_map(arrivals_with_session))
    sess_short = {sd: sn for ts, dur, sd, sn in arrivals_with_session
                   if sd}
    x_labels = [b.strftime("%H:%M") for b in bin_starts]
    bar_colors = [colors.get(s, _SESSION_PALETTE[0])
                   if s else _NO_REC_COLOR
                   for s in sess_per_bin]
    hovertext = []
    for b, s in zip(bin_starts, sess_per_bin):
        if s:
            label = sess_short.get(s, "?")[:48]
            hovertext.append(f"{b.strftime('%a %H:%M')}: {label}")
        else:
            hovertext.append(f"{b.strftime('%a %H:%M')}: no recording")
    fig = go.Figure(go.Bar(
        x=x_labels, y=[1] * n_bins,
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertext=hovertext, hoverinfo="text",
    ))
    fig.update_layout(
        height=32, margin=dict(l=8, r=8, t=2, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0,
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=8, color="#888"),
                    tickmode="array",
                    tickvals=[x_labels[0],
                              x_labels[n_bins // 2],
                              x_labels[-1]],
                    ticktext=["24h ago", "12h ago", "now"]),
        yaxis=dict(showgrid=False, showticklabels=False,
                    zeroline=False, range=[0, 1]),
    )
    return fig


def _build_recording_24h_legend(
        arrivals_with_session: list[tuple[str, float, str, str]],
        session_colors: dict[str, str]) -> html.Div:
    """Horizontal swatch strip naming the colors used in the 24h band.

    Always includes "no recording" + one swatch per session that
    actually appears. Sessions are listed in arrival order; long
    session names are truncated to 28 chars so the strip fits in
    the queue block at narrow viewports.
    """
    seen: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for _ts, _dur, sd, sn in arrivals_with_session:
        if sd and sd not in seen_keys:
            seen_keys.add(sd)
            seen.append((sd, sn))
    swatches = [
        html.Span([
            html.Span(style={
                "display": "inline-block",
                "width": "10px", "height": "10px",
                "background": _NO_REC_COLOR, "borderRadius": "2px",
                "marginRight": "4px",
                "verticalAlign": "middle",
            }),
            html.Span("no recording",
                      style={"verticalAlign": "middle"}),
        ], style={"marginRight": "12px"}),
    ]
    for sd, sn in seen:
        color = session_colors.get(sd, _SESSION_PALETTE[0])
        label = sn if len(sn) <= 28 else sn[:25] + "..."
        swatches.append(html.Span([
            html.Span(style={
                "display": "inline-block",
                "width": "10px", "height": "10px",
                "background": color, "borderRadius": "2px",
                "marginRight": "4px",
                "verticalAlign": "middle",
            }),
            html.Span(label, style={"verticalAlign": "middle"}),
        ], style={"marginRight": "12px"}))
    return html.Div(swatches, style={
        "fontSize": "10px", "color": "#aaa", "marginTop": "4px",
        "display": "flex", "flexWrap": "wrap",
        "rowGap": "4px",
    })


def _build_recording_7d_fig(
        arrivals: list[tuple[str, float]]) -> go.Figure:
    """7-day recording uptime as a day × hour heatmap. Each cell = one
    hour on one day; value = minutes of chunk coverage within that
    hour (0..60). Color saturates at full-hour coverage so partial-
    hour chunks still show clearly. The current-hour cell on today's
    row gets a yellow outline so the user can spot 'now' instantly."""
    today = date.today()
    now = datetime.now()
    # Coverage in MINUTES per (day, hour) cell.
    z = [[0.0] * 24 for _ in range(7)]
    for ts, dur in arrivals:
        dt = _parse_chunk_dt(ts)
        if dt is None:
            continue
        ch_end = dt + timedelta(seconds=dur)
        # Walk hour-by-hour from chunk start to end.
        cur = dt
        guard = 0
        while cur < ch_end and guard < 200:   # NASA rule 2
            guard += 1
            hour_end = cur.replace(minute=0, second=0,
                                    microsecond=0) + timedelta(hours=1)
            seg_end = min(ch_end, hour_end)
            offset = (today - cur.date()).days
            if 0 <= offset < 7:
                mins = (seg_end - cur).total_seconds() / 60.0
                z[6 - offset][cur.hour] += min(60.0, mins)
                # Clamp at 60 so two-back-to-back chunks in the same
                # hour don't shift the colorscale.
                if z[6 - offset][cur.hour] > 60.0:
                    z[6 - offset][cur.hour] = 60.0
            cur = seg_end
    y_labels = [(today - timedelta(days=d)).strftime("%a %m/%d")
                for d in range(6, -1, -1)]
    x_labels = [f"{h:02d}" for h in range(24)]
    # customdata is a 7×24 grid of uptime percent (0..100) -- z gives
    # minutes (0..60) and we want the hover to surface BOTH so a
    # 45-min hour reads "45 min recorded (75%)".
    custom = [[round(z[r][c] / 60.0 * 100.0, 0) for c in range(24)]
               for r in range(7)]
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, zmin=0, zmax=60,
        customdata=custom,
        colorscale=[[0.0, "#2a2a40"], [0.001, "#1f5a44"],
                     [0.5, "#00CC96"], [1.0, "#7be3c0"]],
        showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y} %{x}:00 — %{z:.0f} min recorded "
                       "(%{customdata:.0f}%)<extra></extra>",
    ))
    # Current-hour highlight: a thin gold outline around the cell at
    # (today, current_hour). Using numeric indices because shape
    # x/y on a categorical axis maps to the underlying index, where
    # hour 17 -> x=17 and row "Mon 06/01" -> y=6 (last row index).
    days_ago = (today - now.date()).days
    if 0 <= days_ago < 7:
        x_idx = now.hour
        y_idx = 6 - days_ago
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x_idx - 0.5, x1=x_idx + 0.5,
            y0=y_idx - 0.5, y1=y_idx + 0.5,
            line=dict(color="#FFD700", width=2),
            fillcolor="rgba(255, 215, 0, 0.18)",
            layer="above",
        )
    fig.update_layout(
        height=120, margin=dict(l=52, r=8, t=2, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#888"),
                    tickmode="array",
                    tickvals=["00", "06", "12", "18", "23"],
                    ticktext=["00", "06", "12", "18", "23"],
                    title=dict(text="hour of day",
                                font=dict(size=9, color="#666"))),
        yaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#aaa"),
                    autorange="reversed"),
    )
    return fig


def _overview_today_stats(store: Store, session_dir: str) -> dict:
    """One-shot snapshot for the Overview Today block.

    Today is wall-clock local; pending count is total (not today-only).
    """
    today = date.today()
    start = datetime.combine(today, datetime.min.time()).isoformat()
    end = datetime.combine(today, datetime.max.time()).isoformat()
    last_24h = (datetime.now() - timedelta(hours=24)).isoformat()
    last_1h = (datetime.now() - timedelta(hours=1)).isoformat()
    with store.connection() as conn:
        row = conn.execute(
            """SELECT
                 SUM(CASE WHEN status='done'  THEN 1 ELSE 0 END) AS done,
                 SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors
               FROM processed_files
               WHERE processed_at BETWEEN ? AND ?""",
            (start, end),
        ).fetchone()
        done_today = int((row["done"] if row and row["done"] else 0) or 0)
        errs_today = int((row["errors"] if row and row["errors"] else 0) or 0)

        prow = conn.execute(
            "SELECT COUNT(*) AS n, MIN(chunk_datetime) AS oldest "
            "FROM processed_files WHERE status='pending'"
        ).fetchone()
        pending = int(prow["n"] or 0) if prow else 0
        oldest_pending_iso = (prow["oldest"] if prow else None)

        # 24h hourly throughput.
        hourly_rows = conn.execute(
            """SELECT strftime('%Y-%m-%d %H', processed_at) AS bucket,
                      COUNT(*) AS n
               FROM processed_files
               WHERE status='done' AND processed_at >= ?
               GROUP BY bucket
               ORDER BY bucket""",
            (last_24h,),
        ).fetchall()

        # Active-session "this hour" count.
        sess_hour = 0
        if session_dir:
            srow = conn.execute(
                "SELECT COUNT(*) AS n FROM processed_files "
                "WHERE session_dir = ? AND status='done' "
                "AND processed_at >= ?",
                (session_dir, last_1h),
            ).fetchone()
            sess_hour = int(srow["n"] or 0) if srow else 0

    # Oldest-pending age in minutes (or None when nothing pending).
    oldest_age_min: float | None = None
    if oldest_pending_iso:
        oldest_dt = _parse_chunk_dt(oldest_pending_iso)
        if oldest_dt is not None:
            oldest_age_min = (
                datetime.now() - oldest_dt).total_seconds() / 60.0

    return {
        "done_today": done_today,
        "errors_today": errs_today,
        "pending": pending,
        "oldest_pending_age_min": oldest_age_min,
        "hourly_done_24h": [dict(r) for r in hourly_rows],
        "session_hour_done": sess_hour,
    }


def _build_overview_queue(store: Store):
    """Stacked Today + Active Session block.

    The old "lifetime queue progress" bar always pinned at 100% once
    the backlog finished, so it was useless for spotting current
    problems. This replaces it with today-only counters + a 24h
    hourly throughput sparkline + an active-session progress strip.
    """
    sessions = store.get_sessions()
    active = sessions[0] if sessions else {}
    session_dir = active.get("session_dir", "")
    stats = _overview_today_stats(store, session_dir)

    # ----- Today section ----- #
    pending = stats["pending"]
    oldest_min = stats["oldest_pending_age_min"]
    if oldest_min is None:
        oldest_str = "—"
    elif oldest_min < 60:
        oldest_str = f"{oldest_min:.0f} min"
    else:
        oldest_str = f"{oldest_min / 60:.1f} h"

    today_counters = html.Div([
        html.Span(f"{stats['done_today']} done today",
                  style={"color": "#00CC96", "marginRight": "16px",
                          "fontWeight": "600"}),
        html.Span(f"{stats['errors_today']} errors",
                  style={"color": "#EF553B" if stats["errors_today"]
                          else "#888", "marginRight": "16px"}),
        html.Span(f"{pending} pending",
                  style={"color": "#FFA15A" if pending else "#888",
                          "marginRight": "16px"}),
        html.Span(f"oldest pending: {oldest_str}",
                  style={"color": "#888"}),
    ], style={"fontSize": "13px", "marginTop": "4px"})

    # 24-hour recording-uptime band: chunk_datetime not processed_at;
    # 5-min bins (288 across the day) so short interruptions and
    # session-to-session hand-offs are visible. Each session paints a
    # different color from the palette; gaps where no chunk's coverage
    # interval touches stay gray.
    arrivals_24h_full = _recording_arrivals_with_session(store, hours=24)
    session_colors_24h = _session_color_map(arrivals_24h_full)
    rec_24h = dcc.Graph(
        figure=_build_recording_24h_fig(arrivals_24h_full,
                                          session_colors_24h),
        config={"displayModeBar": False},
        style={"marginTop": "4px"},
    )
    rec_24h_legend = _build_recording_24h_legend(arrivals_24h_full,
                                                   session_colors_24h)
    arrivals_7d = _recording_arrivals(store, hours=24 * 7)
    rec_7d = dcc.Graph(
        figure=_build_recording_7d_fig(arrivals_7d),
        config={"displayModeBar": False},
        style={"marginTop": "4px"},
    )

    today_section = html.Div([
        today_counters,
        html.Div("24h recording",
                  style={"color": "#666", "fontSize": "10px",
                          "marginTop": "6px",
                          "letterSpacing": "0.4px",
                          "textTransform": "uppercase"}),
        rec_24h,
        rec_24h_legend,
    ])

    recording_7d_section = html.Div([
        html.Div("Last 7 days",
                style={"color": "#666", "marginTop": "10px",
                        "marginBottom": "2px", "fontSize": "10px",
                        "letterSpacing": "0.4px",
                        "textTransform": "uppercase"}),
        rec_7d,
        html.Div([
            html.Span([
                html.Span(style={
                    "display": "inline-block",
                    "width": "10px", "height": "10px",
                    "background": _NO_REC_COLOR, "borderRadius": "2px",
                    "marginRight": "4px",
                    "verticalAlign": "middle",
                }),
                html.Span("no recording",
                          style={"verticalAlign": "middle"}),
            ], style={"marginRight": "12px"}),
            html.Span([
                html.Span(style={
                    "display": "inline-block",
                    "width": "10px", "height": "10px",
                    "background":
                        "linear-gradient(90deg, #1f5a44, #7be3c0)",
                    "borderRadius": "2px", "marginRight": "4px",
                    "verticalAlign": "middle",
                }),
                html.Span("recording (darker = partial hour, "
                           "brighter = full hour)",
                          style={"verticalAlign": "middle"}),
            ]),
        ], style={"fontSize": "10px", "color": "#aaa",
                   "marginTop": "4px", "display": "flex",
                   "flexWrap": "wrap", "rowGap": "4px"}),
    ])

    # ----- Active Session section ----- #
    if active:
        num = int(active.get("num_files", 0) or 0)
        done = int(active.get("processed", 0) or 0)
        errs = int(active.get("errors", 0) or 0)
        pct = (100.0 * done / num) if num > 0 else 0.0
        bar = html.Div([
            html.Div(
                f"{pct:.0f}%" if pct > 5 else "",
                style={
                    "width": f"{max(pct, 1):.1f}%",
                    "background":
                        "linear-gradient(90deg, #00CC96 0%, #00AA80 100%)",
                    "height": "22px", "borderRadius": "5px",
                    "transition":
                        "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "11px", "fontWeight": "bold",
                    "color": "white",
                    "textShadow": "0 1px 2px rgba(0,0,0,0.5)",
                }),
        ], style={"backgroundColor": "#1a1a2e", "borderRadius": "5px",
                   "overflow": "hidden", "marginTop": "4px",
                   "marginBottom": "6px",
                   "boxShadow": "inset 0 1px 4px rgba(0,0,0,0.4)"})
        active_counters = html.Div([
            html.Span(active.get("session_name", "(none)"),
                      style={"color": "white", "fontWeight": "600",
                              "marginRight": "12px"}),
            html.Span(f"{done} / {num} files",
                      style={"color": "#aaa", "marginRight": "12px"}),
            html.Span(f"{errs} errors",
                      style={"color": "#EF553B" if errs else "#888",
                              "marginRight": "12px"}),
            html.Span(f"+{stats['session_hour_done']} in last hour",
                      style={"color": "#888"}),
        ], style={"fontSize": "12px"})
        active_section = html.Div([
            html.Div("Active session",
                    style={"color": "#666", "marginTop": "10px",
                            "marginBottom": "2px", "fontSize": "10px",
                            "letterSpacing": "0.4px",
                            "textTransform": "uppercase"}),
            bar, active_counters,
        ])
    else:
        active_section = html.Div([
            html.Div("Active session",
                    style={"color": "#666", "marginTop": "10px",
                            "fontSize": "10px",
                            "letterSpacing": "0.4px",
                            "textTransform": "uppercase"}),
            html.Div("No sessions yet",
                     style={"color": "#888", "fontSize": "11px"}),
        ])

    return [today_section, active_section, recording_7d_section]


def _overview_tab(store: Store, config: dict | None = None):
    sessions = store.get_sessions()
    active_session = sessions[0] if sessions else {}
    session_dir = active_session.get("session_dir", "")

    ch_map = _get_channel_map(store, session_dir) if session_dir else {}
    recent_alerts = store.get_recent_alerts(hours=24)

    # Pills as a full-width horizontal strip at the very top of
    # the Overview tab. flexWrap on so the strip wraps to a second
    # row on narrow viewports rather than overflowing.
    cards = html.Div(
        _build_overview_cards(store),
        id="overview-cards",
        style={"display": "flex", "flexWrap": "wrap",
                "gap": "6px", "marginBottom": "10px"},
    )

    # Per-animal behavioral seizure analysis status. Sits
    # between the pill strip and the 3-column grid so it's
    # always above the fold. Glance-able: top line answers
    # "are we keeping up?" with the rate delta; the table
    # below breaks it down per animal so the user can spot
    # which animal is the bottleneck.
    bsz_status = html.Div(
        _build_behavioral_seizure_status_card(store),
        id="overview-bsz-status",
        style={"marginBottom": "12px"},
    )

    # No SECTION_STYLE on these wrappers -- the _collapsible they're
    # placed inside is the visible card. Nested chrome was making
    # the dashboard read as "card-on-card-on-card."
    km_section = html.Div(
        _build_km_log_section(config),
        id="overview-km-log",
    )
    queue_section = html.Div(
        _build_overview_queue(store),
        id="overview-queue",
    )

    # Evoked thumbnail: radio-selected trace mode +
    # callback-rendered figure. The mode and figure persist across
    # refresh ticks because the radio lives outside the container Div
    # that gets re-children'd by refresh_overview_thumbnail.
    trace_mode_radio = html.Div([
        html.Span("Trace mode: ",
                  style={"color": "#888", "fontSize": "12px",
                          "marginRight": "10px"}),
        dcc.RadioItems(
            id="overview-trace-mode",
            options=[
                {"label": " Mean", "value": "mean"},
                {"label": " Mean ± SEM", "value": "sem"},
                {"label": " Overlay all files", "value": "overlay"},
            ],
            value="mean",
            inline=True,
            labelStyle={"color": "#ddd", "fontSize": "12px",
                         "marginRight": "16px"},
            inputStyle={"marginRight": "4px"},
        ),
    ], style={"marginTop": "16px", "marginBottom": "4px"})
    waveform_thumbnail = html.Div([
        trace_mode_radio,
        html.Div(
            _build_overview_thumbnail(store, config, session_dir, "mean"),
            id="overview-thumbnail",
        ),
    ])

    # Active session basics (session name / file counts / errors) now
    # live in the queue block's Active Session section, so just emit
    # an empty placeholder here to keep the channel-map branch below.
    session_info = html.Div()
    if active_session:
        if ch_map:
            ch_rows = []
            for idx in sorted(ch_map.keys()):
                info = ch_map[idx]
                ch_rows.append({
                    "index": idx,
                    "name": info["name"],
                    "role": info["role"],
                })
            channel_table = html.Div([
                dash_table.DataTable(
                    data=ch_rows,
                    columns=[{"name": "Index", "id": "index"},
                             {"name": "Name", "id": "name"},
                             {"name": "Role", "id": "role"}],
                    **DARK_TABLE_STYLE,
                    style_data_conditional=[ZEBRA_STRIPE,
                        {"if": {"filter_query": "{role} = eeg"},
                         "color": ROLE_COLORS["eeg"]},
                        {"if": {"filter_query": "{role} = stim_copy"},
                         "color": ROLE_COLORS["stim_copy"]},
                        {"if": {"filter_query": "{role} = reference"},
                         "color": ROLE_COLORS["reference"]},
                    ],
                    page_size=8,
                ),
            ], style={"maxHeight": "300px", "overflow": "hidden"})
        else:
            channel_table = html.Div()
    else:
        session_info = html.P("No sessions yet", style={"color": "#888"})
        channel_table = html.Div()

    # Recent alerts -- capped + scrollable so it doesn't push the
    # rest of the Overview off the viewport. Title lives in the
    # collapsible summary so we drop the inner header.
    if recent_alerts:
        alerts_section = html.Div([
            dash_table.DataTable(
                data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                       "type": a["alert_type"], "message": a["message"][:120]}
                      for a in recent_alerts[:15]],
                columns=[{"name": c, "id": c} for c in ["time", "severity", "type", "message"]],
                **DARK_TABLE_STYLE,
                style_data_conditional=[ZEBRA_STRIPE,
                    {"if": {"filter_query": "{severity} = critical"},
                     "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                    {"if": {"filter_query": "{severity} = warning"},
                     "backgroundColor": "#3d3011", "color": "#ffd93d"},
                    {"if": {"filter_query": "{severity} = info"},
                     "backgroundColor": "#112233", "color": "#6bb5ff"},
                ],
                page_size=5,
            ),
        ], style={"maxHeight": "220px", "overflow": "hidden"})
    else:
        alerts_section = html.Div(
            "No alerts in the last 24 hours",
            style={"color": "#666", "fontSize": "11px",
                    "padding": "4px 2px"})

    today = date.today()
    # Home grid (surgeries / schedule / maintenance / incidents /
    # data log) sits at the TOP of the Overview now, in a 3-col-when-
    # wide auto-fit grid. Each tile keeps its existing internal
    # header. minmax(380px, 1fr) means the layout shows 1 column under
    # ~760px, 2 under ~1140px, 3 above that.
    # Lab tiles stacked vertically -- they live in the first
    # column of the new 3-col top section (Lab | Pills | Evoked).
    # The container is a flex column so each tile gets the full
    # column width and stacks predictably regardless of how many
    # tiles _build_home_grid_children returns.
    home_grid = html.Div(
        _build_home_grid_children(store, config, today),
        id="overview-home-grid",
        style={
            "display": "flex", "flexDirection": "column",
            "gap": "8px",
        },
    )

    # Recent alerts becomes a single header row with the count so it
    # stays as a visible divider even when collapsed.
    n_alerts = len(recent_alerts) if recent_alerts else 0
    alerts_collapsible = _collapsible(
        f"Recent Alerts",
        alerts_section,
        open_default=bool(recent_alerts),
        badge=str(n_alerts) if n_alerts else "0",
        badge_color=("#EF553B" if any(
            a.get("severity") == "critical" for a in (recent_alerts or []))
            else "#FFA15A" if recent_alerts else "rgba(255,255,255,0.08)"),
    )

    km_wrapped = _collapsible(
        "KM Recorder log", km_section, open_default=True,
    )
    # Latest video snapshot: most recent decodable frame from the
    # newest companion video. Renders as a small thumbnail by
    # default so it doesn't push the column tall; click the image
    # to expand it inline. The src is cache-busted on every refresh-
    # trigger tick so we always see fresh frames without reload.
    snapshot_card = html.Div([
        html.Img(
            id="overview-snapshot-img",
            src="/media/latest-snapshot.jpg",
            title="Click to expand / collapse",
            n_clicks=0,
            style={
                # 320 px max width fits a 2-camera side-by-side
                # collage without crushing it; single-camera
                # sessions still cap at 180 px via maxHeight.
                "display": "block",
                "maxWidth": "320px",
                "maxHeight": "180px",
                "width": "100%",
                "height": "auto",
                "objectFit": "contain",
                "borderRadius": "4px",
                "background": "#0a0a14",
                "cursor": "zoom-in",
            },
        ),
        html.Div(
            id="overview-snapshot-caption",
            style={"color": "#888", "fontSize": "10px",
                    "marginTop": "4px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.4px"},
        ),
    ])
    snapshot_wrapped = _collapsible(
        "Latest snapshot", snapshot_card, open_default=True,
    )
    queue_wrapped = _collapsible(
        "Today · Recording uptime", queue_section, open_default=True,
    )
    # Latest Evoked lives in column 3 of the top section; not full
    # width. User explicitly asked for it to be narrower than the
    # whole page. The 1fr grid cell still leaves it the widest
    # visible element after the two narrow sidebar columns.
    thumb_wrapped = _collapsible(
        "Latest Evoked", waveform_thumbnail,
        open_default=True,
    )
    channel_wrapped = _collapsible(
        "Channel Map", channel_table,
        open_default=False,
    )
    # Status pills are a full-width row at the very top, above
    # everything else. The 3-column grid below holds:
    #   Col 1 (280px): Lab tiles
    #   Col 2 (1fr):   Latest Evoked
    #   Col 3 (1fr):   Today + KM stacked
    # Channel Map drops underneath Evoked in column 2; Recent
    # Alerts drops underneath the snapshot in column 3. Both stay
    # collapsed by default so they just add a thin header strip,
    # but they prevent the ragged uneven dead zone the user saw
    # below the three columns. No separate bottom row anymore --
    # the page now reads as a single 3-column slab.
    evoked_col = html.Div([
        thumb_wrapped, channel_wrapped,
    ], style={
        "display": "flex", "flexDirection": "column", "gap": "8px",
    })
    ops_col = html.Div([
        queue_wrapped, km_wrapped, snapshot_wrapped,
        alerts_collapsible,
    ], style={
        "display": "flex", "flexDirection": "column", "gap": "8px",
    })
    top_section = html.Div([
        home_grid,
        evoked_col,
        ops_col,
    ], style={
        "display": "grid",
        # Sidebar (lab tiles) at 320 px. minmax(0, 1fr) on cols 2
        # and 3 stops nowrap content (KM log filenames, active-
        # session names) from forcing those columns wider.
        "gridTemplateColumns": "320px minmax(0, 1fr) minmax(0, 1fr)",
        "gap": "12px",
        "alignItems": "start",
        "marginBottom": "10px",
    })

    return html.Div([
        cards,         # pills strip, full width
        bsz_status,    # per-animal seizure analysis status
        top_section,   # sidebar | (Evoked + Channel Map) | (Today + KM + Snapshot + Alerts)
    ])


def _build_behavioral_seizure_status_card(store):
    """Per-animal behavioral seizure analysis status.

    Designed so the user can answer two questions at a glance:
      1. Are we keeping up?  -- rate delta on top.
      2. Which animal is the bottleneck?  -- per-row breakdown.

    Sorted so animals with the steepest backlog growth float
    to the top.
    """
    rows = store.behavioral_seizure_status_per_animal(days=7)
    if not rows:
        return html.Div(
            "No animals ingested yet.",
            style={"padding": "12px",
                    "color": "#a0a0b0",
                    "fontSize": "12px",
                    "background": COLOR_SURFACE_1,
                    "border": f"1px solid {COLOR_DIVIDER}",
                    "borderRadius": RADIUS_MD})
    # Totals + rate verdict.
    total_created = sum(r["created_window"] for r in rows)
    total_approved = sum(r["approved_window"] for r in rows)
    total_queue = sum(r["n_queue"] for r in rows)
    total_pending = sum(r["n_pending_pi"] for r in rows)
    rate_delta = total_created - total_approved
    rate_per_day = rate_delta / 7.0
    if rate_per_day > 5:
        verdict_text = (
            f"Backlog growing by +{rate_per_day:.1f} files/day. "
            "Consider hiring more reviewers.")
        verdict_color = "#ff453a"
    elif rate_per_day > 0.5:
        verdict_text = (
            f"Backlog growing slowly (+{rate_per_day:.1f} "
            "files/day). Team marginal.")
        verdict_color = "#ff9f0a"
    elif rate_per_day > -0.5:
        verdict_text = (
            "Team keeping up. Throughput matches arrivals.")
        verdict_color = "#30d158"
    else:
        verdict_text = (
            f"Backlog shrinking ({rate_per_day:.1f} files/day). "
            "Team ahead of schedule.")
        verdict_color = "#30d158"
    header = html.Div([
        html.Div([
            html.Span("Behavioral seizure analysis status",
                       style={"color": "#f0f0f5",
                               "fontWeight": "600",
                               "fontSize": "13px"}),
            html.Span("  (last 7 days)",
                       style={"color": "#888",
                               "fontSize": "11px"}),
        ]),
        html.Div([
            html.Span(f"{total_created} created  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_approved} approved  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_queue} in queue  ·  ",
                       style={"color": "#cfd0d6"}),
            html.Span(f"{total_pending} pending PI",
                       style={"color": "#cfd0d6"}),
        ], style={"fontSize": "12px",
                   "marginTop": "2px"}),
        html.Div(verdict_text,
                  style={"color": verdict_color,
                          "fontWeight": "600",
                          "fontSize": "12px",
                          "marginTop": "6px"}),
    ], style={"padding": "12px 14px",
               "borderBottom": f"1px solid {COLOR_DIVIDER}"})
    # Per-animal table.
    cells = []
    for r in rows:
        delta = (r["created_window"]
                  - r["approved_window"])
        delta_color = ("#ff453a" if delta > 5
                        else "#ff9f0a" if delta > 0
                        else "#30d158")
        delta_label = (f"+{delta}" if delta > 0
                        else f"{delta}")
        last = r["last_activity_at"][:16] or "—"
        cells.append(html.Div([
            html.Div(r["animal_id"],
                      style={"color": "#f0f0f5",
                              "fontWeight": "600",
                              "fontSize": "13px",
                              "marginBottom": "4px"}),
            html.Div([
                html.Div([
                    html.Span("queue",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_queue']}",
                              style={"color": "#cfd0d6",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("pending PI",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_pending_pi']}",
                              style={"color": "#5e7ce2",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("approved",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(f"{r['n_approved']}",
                              style={"color": "#30d158",
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
                html.Div([
                    html.Span("7 d net",
                               style={"color": "#888",
                                       "fontSize": "10px",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.5px"}),
                    html.Div(delta_label,
                              style={"color": delta_color,
                                      "fontSize": "16px",
                                      "fontWeight": "600"}),
                ]),
            ], style={"display": "grid",
                       "gridTemplateColumns":
                           "repeat(4, 1fr)",
                       "gap": "8px",
                       "marginBottom": "4px"}),
            html.Div(
                f"created 7 d: {r['created_window']}  ·  "
                f"approved 7 d: {r['approved_window']}  ·  "
                f"last activity: {last}",
                style={"color": "#888",
                        "fontSize": "10px"}),
        ], style={"padding": "10px 12px",
                   "background": COLOR_SURFACE_1,
                   "border": f"1px solid {COLOR_DIVIDER}",
                   "borderRadius": RADIUS_SM,
                   "borderLeft": f"3px solid {delta_color}"}))
    grid = html.Div(
        cells,
        style={"display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(260px, 1fr))",
                "gap": "8px",
                "padding": "12px 14px"})
    return html.Div([header, grid],
                      style={"background": COLOR_SURFACE_1,
                              "border":
                                  f"1px solid {COLOR_DIVIDER}",
                              "borderRadius": RADIUS_MD})


# Evoked Waveforms tab moved to src/dashboard/tabs/waveforms.py.


# Signal Quality tab moved to src/dashboard/tabs/signal_quality.py.


# Evoked Features tab moved to src/dashboard/tabs/evoked.py.


# Criticality tab moved to src/dashboard/tabs/criticality.py.


# LFP Browser tab moved to src/dashboard/tabs/lfp_browser.py.


# Electrode Health tab moved to src/dashboard/tabs/electrode_health.py.


# Session Compare tab moved to src/dashboard/tabs/session_compare.py.


# Stim QC tab moved to src/dashboard/tabs/stim.py.
# Settings tab moved to src/dashboard/tabs/settings.py.


