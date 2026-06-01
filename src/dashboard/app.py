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
from src.dashboard.media_routes import register_media_routes
from src.dashboard.tabs import video as tabs_video
from src.dashboard.tabs import surgeries as tabs_surgeries
from src.dashboard.tabs import maintenance as tabs_maintenance
from src.dashboard.tabs import data_log_xref as tabs_data_log_xref
from src.dashboard.components import (
    card as _card, pill as _pill, section_header as _section_header,
)
# Side-effect import: registers the qc_dark Plotly template as default
from src.dashboard import plotly_template  # noqa: F401

logger = logging.getLogger("qc_monitor.dashboard")

# Full list of plottable evoked feature columns
EVOKED_FEATURE_COLS = [
    "line_length", "log_auc", "peak_amplitude", "trough_amplitude",
    "peak_to_trough", "rms_amplitude", "peak_latency_ms", "trough_latency_ms",
    "max_slope", "max_slope_time_ms", "early_area", "late_area",
    "early_late_ratio", "recovery_tau", "recovery_slope",
    "template_correlation", "pca_recon_error", "variance",
    "autocorrelation", "ac_width", "exp_fit_a", "sum_power_low",
    "freq_moment_low", "sum_power_high", "freq_moment_high",
    "is_artifact", "is_ictal", "epoch_time_sec",
]

# Human-readable labels
EVOKED_FEATURE_LABELS = {
    "line_length": "Line Length", "log_auc": "Log(AUC)",
    "peak_amplitude": "Peak Amplitude", "trough_amplitude": "Trough Amplitude",
    "peak_to_trough": "Peak-to-Trough", "rms_amplitude": "RMS Amplitude",
    "peak_latency_ms": "Peak Latency (ms)", "trough_latency_ms": "Trough Latency (ms)",
    "max_slope": "Max Slope", "max_slope_time_ms": "Max Slope Time (ms)",
    "early_area": "Early Area", "late_area": "Late Area",
    "early_late_ratio": "Early/Late Ratio", "recovery_tau": "Recovery Tau",
    "recovery_slope": "Recovery Slope", "template_correlation": "Template Correlation",
    "pca_recon_error": "PCA Recon Error", "variance": "Variance",
    "autocorrelation": "Autocorrelation", "ac_width": "AC Width",
    "exp_fit_a": "Exp Fit A", "sum_power_low": "Sum Power Low",
    "freq_moment_low": "Freq Moment Low", "sum_power_high": "Sum Power High",
    "freq_moment_high": "Freq Moment High", "is_artifact": "Is Artifact",
    "is_ictal": "Is Ictal", "epoch_time_sec": "Epoch Time (sec)",
}

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

# Config file path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
CONFIG_PATH = os.path.normpath(CONFIG_PATH)

# Standard time-range options used across tabs
TIME_RANGE_OPTIONS = [
    {"label": "Last 24h", "value": 24},
    {"label": "Last 48h", "value": 48},
    {"label": "Last 1 week", "value": 168},
    {"label": "Last 1 month", "value": 720},
    {"label": "All time", "value": 0},
]

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

DARK_TABLE_STYLE = {
    "style_header": {
        "backgroundColor": COLOR_SURFACE_2,
        "color": COLOR_TEXT_PRIMARY,
        "fontWeight": "600",
        "border": "none",
        "borderBottom": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_CAPTION,
        "textTransform": "uppercase",
        "letterSpacing": "0.5px",
    },
    "style_data": {
        "backgroundColor": COLOR_SURFACE_1,
        "color": COLOR_TEXT_SECONDARY,
        "border": "none",
        "borderBottom": f"1px solid {COLOR_DIVIDER}",
        "fontSize": FONT_SIZE_BODY,
    },
    "style_cell": {
        "textAlign": "left",
        "padding": f"{SPACE_3} {SPACE_4}",
        "fontSize": FONT_SIZE_BODY,
        "fontFamily": FONT_STACK,
    },
    "style_filter": {
        "backgroundColor": COLOR_SURFACE_2,
        "color": COLOR_TEXT_PRIMARY,
    },
}

# Zebra stripe — very subtle, just a lighter surface.
ZEBRA_STRIPE = {"if": {"row_index": "odd"}, "backgroundColor": COLOR_SURFACE_2}

SECTION_STYLE = {
    "background": COLOR_SURFACE_1,
    "padding": f"{SPACE_5} {SPACE_5}",
    "borderRadius": RADIUS_MD,
    "border": f"1px solid {COLOR_DIVIDER}",
    "marginBottom": SPACE_4,
}
LABEL_STYLE = {
    "color": COLOR_TEXT_TERTIARY,
    "fontSize": FONT_SIZE_CAPTION,
    "marginBottom": SPACE_2,
    "display": "block",
    "letterSpacing": "0.4px",
    "textTransform": "uppercase",
    "fontWeight": "600",
}
INPUT_STYLE = {
    "backgroundColor": COLOR_SURFACE_3,
    "color": COLOR_TEXT_PRIMARY,
    "border": f"1px solid {COLOR_DIVIDER}",
    "borderRadius": RADIUS_SM,
    "padding": f"{SPACE_2} {SPACE_3}",
    "width": "100%",
    "fontSize": FONT_SIZE_BODY,
    "fontFamily": FONT_STACK,
    "transition": "border-color 0.15s ease, box-shadow 0.15s ease",
}
FIELD_STYLE = {"flex": "1", "minWidth": "200px"}
DROPDOWN_STYLE = {"backgroundColor": COLOR_SURFACE_3, "color": COLOR_TEXT_PRIMARY}

# ====================================================================== #
#  Helpers
# ====================================================================== #


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


def _load_config() -> dict:
    # Force UTF-8 -- config contains emoji in the maintenance tab names
    # and Windows' default cp1252 codec chokes on them. (Same fix that
    # was applied to main.py's load_config.)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(cfg: dict):
    # UTF-8 + allow_unicode keep emoji intact through a round-trip
    # (matters for the maintenance tab names in the config block).
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)


def _parse_json_field(val):
    """Parse a JSON string from the DB or return the value as-is."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _get_channel_map(store: Store, session_dir: str) -> dict:
    """Build channel_index -> {name, role} map from session_config."""
    cfg = store.get_session_config(session_dir)
    if not cfg:
        return {}
    names = _parse_json_field(cfg.get("channel_names")) or []
    eeg_chs = _parse_json_field(cfg.get("eeg_channels")) or []
    stim_chs = _parse_json_field(cfg.get("stim_copy_channels")) or []
    ref_chs = _parse_json_field(cfg.get("reference_channels")) or []

    ch_map = {}
    for i, name in enumerate(names):
        role = "eeg"
        if i in stim_chs:
            role = "stim_copy"
        elif i in ref_chs:
            role = "reference"
        ch_map[i] = {"name": name, "role": role}
    return ch_map


def _color_for_role(role: str) -> str:
    return ROLE_COLORS.get(role, "#636EFA")


def _status_card(title: str, value: str, color: str = "#636EFA"):
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "color": "#888", "textTransform": "uppercase",
                                "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
    ], style=CARD_STYLE)


def _empty_fig(text: str = "Nothing to show yet",
               hint: str | None = None,
               height: int = 400) -> go.Figure:
    """Friendly empty figure used in place of a plot when there's no data.

    *text* is the headline. *hint* is an optional one-liner explaining
    what the user can do next, rendered below the headline in a quieter
    color. Apple HIG: empty states should be informative, not silent.
    """
    fig = go.Figure()
    annotations = [dict(
        text=f"<b>{text}</b>", showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.55, font=dict(size=14, color=COLOR_TEXT_SECONDARY),
    )]
    if hint:
        annotations.append(dict(
            text=hint, showarrow=False, xref="paper", yref="paper",
            x=0.5, y=0.42, font=dict(size=11, color=COLOR_TEXT_TERTIARY),
        ))
    fig.update_layout(
        height=height,
        plot_bgcolor=COLOR_SURFACE_1, paper_bgcolor=COLOR_SURFACE_1,
        annotations=annotations,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def _empty_state(headline: str, hint: str = "",
                 glyph: str = "·") -> html.Div:
    """Tab-level empty state: small glyph, headline, and a friendly hint.

    Used by tabs that have no data to render at all (e.g. no sessions
    discovered, no files with videos). Looks consistent across the app.
    """
    return html.Div([
        html.Div(glyph, style={"fontSize": "32px",
                               "color": COLOR_TEXT_TERTIARY,
                               "marginBottom": SPACE_3,
                               "letterSpacing": "0"}),
        html.Div(headline, style={"fontSize": FONT_SIZE_HEADER,
                                  "fontWeight": "600",
                                  "color": COLOR_TEXT_PRIMARY,
                                  "marginBottom": SPACE_2}),
        html.Div(hint, style={"fontSize": FONT_SIZE_BODY,
                              "color": COLOR_TEXT_SECONDARY,
                              "maxWidth": "520px",
                              "lineHeight": "1.5"}),
    ], style={"textAlign": "center",
              "padding": f"{SPACE_6} {SPACE_5}",
              "marginTop": SPACE_6})


def _session_dropdown_options(store: Store) -> list[dict]:
    sessions = store.get_sessions()
    return [{"label": s["session_name"], "value": s["session_dir"]} for s in sessions]


def _default_session(store: Store, hint: str | None = None) -> str | None:
    """Return a default session_dir for tab dropdowns.

    *hint* is an explicit choice from the click-through Store (set when
    a user clicks a row in the Sessions table). If the hint matches a
    real session it wins, so the destination tab opens with that
    session pre-selected. Otherwise we fall back to the session with
    the most processed files.
    """
    sessions = store.get_sessions()
    if not sessions:
        return None
    if hint:
        valid = {s["session_dir"] for s in sessions}
        if hint in valid:
            return hint
    best = max(sessions, key=lambda s: s.get("processed", 0))
    return best["session_dir"] if best.get("processed", 0) > 0 else sessions[0]["session_dir"]


def _get_processed_files_for_session(store: Store, session_dir: str) -> list[dict]:
    """Return processed file rows for a session, ordered by chunk_datetime."""
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT id, file_path, chunk_datetime, session_name
               FROM processed_files
               WHERE session_dir = ? AND status = 'done'
               ORDER BY chunk_datetime""",
            (session_dir,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ====================================================================== #
#  App factory
# ====================================================================== #

def create_app(config: dict, store: Store) -> Dash:
    refresh_sec = config.get("dashboard", {}).get("refresh_interval_sec", 10)

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
                          options=[{"label": " Auto-refresh", "value": True}],
                          value=[], inline=True,
                          style={"color": COLOR_TEXT_SECONDARY,
                                 "display": "inline-block",
                                 "fontSize": FONT_SIZE_CAPTION}),
        ], id="refresh-bar",
           style={"position": "fixed", "top": SPACE_2, "right": SPACE_5, "zIndex": "9999",
                  "display": "flex", "alignItems": "center",
                  "background": "rgba(19,19,31,0.85)", "backdropFilter": "blur(12px)",
                  "padding": f"{SPACE_1} {SPACE_3}", "borderRadius": RADIUS_SM,
                  "border": f"1px solid {COLOR_DIVIDER}"}),
        dcc.Store(id="refresh-trigger", data=0),
        dcc.Store(id="last-refresh-ts", data=None),
        dcc.Interval(id="elapsed-ticker", interval=5000, n_intervals=0),
        # Hidden stores
        dcc.Store(id="selected-session-dir"),
    ], style={"backgroundColor": COLOR_SURFACE_0,
              "fontFamily": FONT_STACK,
              "color": COLOR_TEXT_PRIMARY,
              "minHeight": "100vh",
              "fontSize": FONT_SIZE_BODY,
              "letterSpacing": "0.1px"})

    # ------------------------------------------------------------------ #
    #  Refresh controls
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("refresh", "disabled"),
        [Input("auto-refresh-toggle", "value")]
    )
    def toggle_auto_refresh(val):
        return not bool(val)

    @app.callback(
        [Output("refresh-trigger", "data"), Output("last-refresh-ts", "data")],
        [Input("manual-refresh-btn", "n_clicks"), Input("refresh", "n_intervals")],
        [State("refresh-trigger", "data")]
    )
    def on_refresh(clicks, intervals, current):
        import time as _time
        return (current or 0) + 1, _time.time()

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
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        State("selected-session-dir", "data"),
    )
    def render_tab(tab, session_hint):
        try:
            if tab == "overview":
                return _overview_tab(store, config)
            elif tab == "waveforms":
                return _waveforms_tab_layout(store, default_session=session_hint)
            elif tab == "signal":
                return _signal_quality_tab(store)
            elif tab == "evoked":
                return _evoked_tab_layout(store)
            elif tab == "criticality":
                return _criticality_tab_layout(store)
            elif tab == "lfp":
                return _lfp_browser_tab_layout(store, default_session=session_hint)
            elif tab == "video":
                return tabs_video.layout(store)
            elif tab == "electrode_health":
                return _electrode_health_tab_layout(store, default_session=session_hint)
            elif tab == "session_compare":
                return _session_compare_tab_layout(store)
            elif tab == "stim":
                return _stim_tab(store)
            elif tab == "settings":
                return _settings_tab_layout(store)
            elif tab == "activity_log":
                return _activity_log_tab_layout(store)
            elif tab == "annotations":
                return _annotations_tab_layout(store)
            elif tab == "alerts":
                return _alerts_tab(store)
            elif tab == "sessions":
                return _sessions_tab(store)
            elif tab == "surgeries":
                return tabs_surgeries.layout(store, config)
            elif tab == "maintenance":
                return tabs_maintenance.layout(store, config)
            elif tab == "data_log_xref":
                return tabs_data_log_xref.layout(store, config)
        except Exception as e:
            logger.error("Dashboard render error: %s", e, exc_info=True)
            return html.Div(f"Error rendering tab: {e}",
                            style={"color": COLOR_DANGER, "padding": SPACE_5})

    # ------------------------------------------------------------------ #
    #  Evoked Features callback — separate row per selected feature
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("evoked-multi-plots", "children"),
         Output("evoked-stats", "children")],
        [Input("evoked-feature-checklist", "value"),
         Input("evoked-session-dropdown", "value"),
         Input("evoked-hours-dropdown", "value")],
    )
    def update_evoked_multi(selected_features, session_dir, hours):
        if not selected_features or not session_dir:
            return html.P("Select features and a session", style={"color": "#888"}), ""

        plots = []
        stats_rows = []
        hrs = int(hours) if hours else 0

        for feature_name in selected_features:
            try:
                data = store.get_evoked_feature_timeseries(
                    feature_name=feature_name,
                    session_dir=session_dir,
                    hours=hrs,
                )
            except Exception as e:
                plots.append(html.P(f"Error loading {feature_name}: {e}", style={"color": "#ff6b6b"}))
                continue

            if not data:
                plots.append(html.P(f"No data for {feature_name}", style={"color": "#888"}))
                continue

            clean_t, clean_v = [], []
            artifact_t, artifact_v = [], []
            ictal_t, ictal_v = [], []
            all_values = []

            for d in data:
                # Use epoch_time_sec for continuous x-axis (seconds within file)
                # Add to chunk_datetime for absolute timestamp
                epoch_sec = d.get("epoch_time_sec")
                chunk_dt = d.get("chunk_datetime", "")
                v = d.get("value")
                if v is None:
                    continue
                # Build absolute epoch time: chunk start + epoch offset
                if epoch_sec is not None and chunk_dt:
                    try:
                        from datetime import datetime as _dt, timedelta as _td
                        base = _dt.fromisoformat(chunk_dt.replace("_", "-").replace("--", " ").replace("__", "T"))
                        t = (base + _td(seconds=float(epoch_sec))).isoformat()
                    except Exception:
                        t = chunk_dt  # fallback to file-level timestamp
                else:
                    t = chunk_dt
                all_values.append(v)
                if d.get("is_artifact", 0):
                    artifact_t.append(t); artifact_v.append(v)
                elif d.get("is_ictal", 0):
                    ictal_t.append(t); ictal_v.append(v)
                else:
                    clean_t.append(t); clean_v.append(v)

            label = EVOKED_FEATURE_LABELS.get(feature_name, feature_name)
            fig = go.Figure()
            if clean_t:
                fig.add_trace(go.Scatter(x=clean_t, y=clean_v, mode="markers", name="Clean",
                                         marker=dict(color="#636EFA", size=4, opacity=0.6)))
            if artifact_t:
                fig.add_trace(go.Scatter(x=artifact_t, y=artifact_v, mode="markers", name="Artifact",
                                         marker=dict(color="#EF553B", size=5, opacity=0.7)))
            if ictal_t:
                fig.add_trace(go.Scatter(x=ictal_t, y=ictal_v, mode="markers", name="Ictal",
                                         marker=dict(color="#FFA15A", size=5, opacity=0.7)))

            fig.update_layout(
                title=label,
                xaxis_title="Time",
                yaxis_title=label,
                height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            plots.append(dcc.Graph(figure=fig, style={"marginBottom": "8px"}))

            # Per-feature stats
            n_total = len(all_values)
            n_art = len(artifact_v)
            n_clean = len(clean_v)
            mean_val = statistics.mean(all_values) if all_values else 0
            std_val = statistics.stdev(all_values) if len(all_values) > 1 else 0
            stats_rows.append(f"{label}: mean={mean_val:.4g}, std={std_val:.4g}, "
                              f"N={n_total} (clean={n_clean}, artifact={n_art})")

        summary = html.Div([
            html.P(s, style={"color": "#aaa", "margin": "2px 0", "fontSize": "12px"})
            for s in stats_rows
        ]) if stats_rows else ""

        return html.Div(plots), summary

    # ------------------------------------------------------------------ #
    #  Evoked Waveforms callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("waveform-file-dropdown", "options"),
         Output("waveform-file-dropdown", "value")],
        [Input("waveform-session-dropdown", "value")],
    )
    def update_waveform_file_list(session_dir):
        if not session_dir:
            return [], None
        files = _get_processed_files_for_session(store, session_dir)
        options = [{"label": f["chunk_datetime"], "value": f["id"]} for f in files]
        latest = options[-1]["value"] if options else None
        return options, latest

    @app.callback(
        Output("waveform-plot", "figure"),
        [Input("waveform-file-dropdown", "value"),
         Input("waveform-smooth", "value")],
        [State("waveform-session-dropdown", "value")],
    )
    def update_waveform_plot(file_id, smooth_ms, session_dir):
        if not file_id:
            return _empty_fig("Select a file", 400)

        try:
            waveforms = store.get_evoked_waveform_by_file(file_id)
        except Exception as e:
            logger.error("Waveform query error: %s", e, exc_info=True)
            return _empty_fig(f"Error: {e}", 400)

        if not waveforms:
            return _empty_fig("No evoked waveforms for this file", 400)

        try:
            return _build_waveform_figure(waveforms, store, session_dir,
                                            smooth_ms=smooth_ms or 0)
        except Exception as e:
            logger.error("Waveform plot error: %s", e, exc_info=True)
            return _empty_fig(f"Plot error: {e}", 400)

    def _build_waveform_figure(waveforms, store, session_dir,
                                smooth_ms: float = 0.0):

        cfg = _load_config()
        fa = cfg.get("feature_analysis", {})
        ana_start = fa.get("analysis_start_ms", fa.get("window_start_ms", 2))
        ana_end = fa.get("analysis_end_ms", fa.get("window_end_ms", 50))

        # Group latest waveform per LFP channel
        latest_per_ch = {}
        for wf in waveforms:
            ch = wf.get("channel", 0)
            latest_per_ch[ch] = wf

        ch_map = _get_channel_map(store, session_dir)
        lfp_chs = sorted(latest_per_ch.keys())

        # For each LFP channel: 2 rows
        #   Row A: Stim artifact vs LFP superimposed (-1 to 1ms)
        #   Row B: LFP evoked response zoomed to analysis window
        # The stim_mean_trace is stored ON each LFP waveform (not as separate channel)

        n_rows = len(lfp_chs) * 2
        if n_rows == 0:
            return _empty_fig("No waveform data", 400)

        titles = []
        for ch in lfp_chs:
            name = latest_per_ch[ch].get("channel_name", ch_map.get(ch, {}).get("name", f"Ch{ch}"))
            titles.append(f"Stim Artifact vs {name} (-1 to 1 ms)")
            titles.append(f"Evoked: {name} ({ana_start}-{ana_end} ms)")

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=False,
            subplot_titles=titles,
            vertical_spacing=0.06,
        )

        row = 1
        for ch in lfp_chs:
            wf = latest_per_ch[ch]
            name = wf.get("channel_name", f"Ch{ch}")
            t = wf["time_axis_ms"]
            m = wf["mean_trace"]
            s = wf.get("sem_trace")
            stim_tr = wf.get("stim_mean_trace")
            n_ep = wf.get("n_epochs", 0)

            # Optional smoothing (Gaussian, in display-ms). Apply to
            # mean + SEM + stim mean so the band stays consistent.
            if smooth_ms and smooth_ms > 0 and m and len(t) > 1:
                try:
                    dt_ms = float(t[1] - t[0])
                    if dt_ms > 0:
                        fs_proxy = 1000.0 / dt_ms
                        m = apply_filter(
                            np.asarray(m, dtype=np.float32),
                            fs_proxy, smoothing_ms=smooth_ms).tolist()
                        if s and len(s) == len(m):
                            s = apply_filter(
                                np.asarray(s, dtype=np.float32),
                                fs_proxy, smoothing_ms=smooth_ms).tolist()
                        if stim_tr and len(stim_tr) > 1:
                            stim_tr = apply_filter(
                                np.asarray(stim_tr, dtype=np.float32),
                                fs_proxy, smoothing_ms=smooth_ms).tolist()
                except Exception as e:
                    logger.debug("Smoothing skipped: %s", e)

            # Helper: compute y-range from data within x-range
            def _yrange(times, values, x0, x1):
                vals = [v for tv, v in zip(times, values) if x0 <= tv <= x1]
                if not vals:
                    return None
                mn, mx = min(vals), max(vals)
                pad = (mx - mn) * 0.1 if mx > mn else 0.001
                return [mn - pad, mx + pad]

            # Row A: Artifact comparison — stim copy (orange) vs LFP (blue), zoomed -1 to 1ms
            art_yvals = []
            if stim_tr and len(stim_tr) > 0:
                stim_t = t[:len(stim_tr)] if len(stim_tr) <= len(t) else t
                fig.add_trace(go.Scatter(
                    x=stim_t, y=stim_tr, mode="lines",
                    name="StimCopy", line=dict(color="#FFA15A", width=2),
                    showlegend=(row == 1),
                ), row=row, col=1)
                art_yvals += [v for tv, v in zip(stim_t, stim_tr) if -1 <= tv <= 1]
            fig.add_trace(go.Scatter(
                x=t, y=m, mode="lines",
                name=name, line=dict(color="#636EFA", width=2),
                showlegend=(row == 1),
            ), row=row, col=1)
            art_yvals += [v for tv, v in zip(t, m) if -1 <= tv <= 1]
            fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"), row=row, col=1)
            fig.update_xaxes(range=[-1, 1], title_text="ms", row=row, col=1)
            if art_yvals:
                mn, mx = min(art_yvals), max(art_yvals)
                pad = (mx - mn) * 0.1 if mx > mn else 0.001
                fig.update_yaxes(range=[mn - pad, mx + pad], row=row, col=1)
            row += 1

            # Row B: LFP evoked response zoomed to analysis window
            if s and len(s) == len(m):
                upper = [mv + sv for mv, sv in zip(m, s)]
                lower = [mv - sv for mv, sv in zip(m, s)]
                fig.add_trace(go.Scatter(
                    x=list(t) + list(reversed(t)), y=upper + list(reversed(lower)),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=t, y=m, mode="lines",
                name=f"{name} (n={n_ep})",
                line=dict(color="#636EFA", width=2),
                showlegend=False,
            ), row=row, col=1)
            fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"), row=row, col=1)
            fig.update_xaxes(range=[ana_start, ana_end], title_text="ms", row=row, col=1)
            yr = _yrange(t, m, ana_start, ana_end)
            if yr:
                # Include SEM bounds if available
                if s and len(s) == len(m):
                    sem_vals = [mv + sv for tv, mv, sv in zip(t, m, s) if ana_start <= tv <= ana_end]
                    sem_vals += [mv - sv for tv, mv, sv in zip(t, m, s) if ana_start <= tv <= ana_end]
                    if sem_vals:
                        yr = [min(yr[0], min(sem_vals)), max(yr[1], max(sem_vals))]
                        pad = (yr[1] - yr[0]) * 0.1
                        yr = [yr[0] - pad, yr[1] + pad]
                fig.update_yaxes(range=yr, row=row, col=1)
            row += 1

        fig.update_layout(
            height=250 * n_rows,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Criticality callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("criticality-plot", "figure"),
        [Input("criticality-session-dropdown", "value"),
         Input("criticality-hours-dropdown", "value")],
    )
    def update_criticality(session_dir, hours):
        if not session_dir:
            return _empty_fig("Select a session", 550)

        ch_map = _get_channel_map(store, session_dir)

        try:
            data = store.query_criticality_timeseries(
                session_dir=session_dir,
                hours=int(hours) if hours else None,
            )
        except Exception as e:
            return _empty_fig(f"Error: {e}", 550)

        if not data:
            return _empty_fig("No criticality data", 550)

        channels = sorted(set(d["channel"] for d in data))

        fig = go.Figure()
        for ch in channels:
            ch_data = [d for d in data if d["channel"] == ch]
            times = [d["chunk_datetime"] for d in ch_data]
            vals = [d["db_value"] for d in ch_data]

            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            if ch_map and info["role"] != "eeg":
                continue

            fig.add_trace(go.Scatter(
                x=times, y=vals, mode="lines+markers",
                name=info["name"],
                line=dict(color=_color_for_role(info["role"])),
                marker=dict(size=3),
            ))

        fig.update_layout(
            title="Criticality (dB) Over Time -- EEG Channels",
            xaxis_title="Time",
            yaxis_title="dB Value",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  LFP Browser callbacks
    # ------------------------------------------------------------------ #

    # Preset -> (HP, LP) lookup. Selecting a preset populates the
    # number inputs (via the clientside dispatch below) before the
    # user clicks Apply.
    _LFP_PRESETS = {
        "raw":   (0, 0),
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 100),
        "spike": (300, 3000),
    }

    @app.callback(
        Output("lfp-filter-hp", "value"),
        Output("lfp-filter-lp", "value"),
        Input("lfp-filter-preset", "value"),
        State("lfp-filter-hp", "value"),
        State("lfp-filter-lp", "value"),
        prevent_initial_call=True,
    )
    def _lfp_apply_preset(preset, current_hp, current_lp):
        if preset == "custom" or preset not in _LFP_PRESETS:
            return no_update, no_update
        hp, lp = _LFP_PRESETS[preset]
        return hp, lp

    @app.callback(
        Output("lfp-plot", "figure"),
        Output("lfp-psd-plot", "figure"),
        Output("lfp-psd-row", "style"),
        Output("lfp-filter-state", "data"),
        Input("lfp-load-btn", "n_clicks"),
        Input("lfp-apply-filter-btn", "n_clicks"),
        State("lfp-session-dropdown", "value"),
        State("lfp-file-dropdown", "value"),
        State("lfp-filter-hp", "value"),
        State("lfp-filter-lp", "value"),
        State("lfp-filter-notch", "value"),
        State("lfp-filter-smooth", "value"),
        State("lfp-show-psd", "value"),
        prevent_initial_call=True,
    )
    def load_lfp(n_load, n_apply, session_dir, file_path,
                  hp, lp, notch, smooth_ms, show_psd_val):
        psd_hidden_style = {"display": "none", "marginTop": "12px"}
        if not file_path:
            return (_empty_fig("Select a file and click Load", 600),
                    no_update, psd_hidden_style, no_update)

        try:
            chunk = get_chunk(file_path)
        except Exception as e:
            return (_empty_fig(f"Error loading file: {e}", 600),
                    no_update, psd_hidden_style, no_update)

        fs = chunk.fs
        # Apply the cached filter to the full signal. get_filtered
        # is cheap on a cache hit (typical for zoom callbacks) and
        # ~1s on a miss for an hour-long 20kHz dual-channel chunk.
        signal = get_filtered(file_path, chunk.signal, fs,
                               highpass=hp, lowpass=lp,
                               notch=notch, smoothing_ms=smooth_ms)
        n_samples, n_ch = signal.shape
        duration_sec = n_samples / fs

        sess_map = _get_channel_map(store, session_dir) if session_dir else {}
        file_names = chunk.channel_names or []

        def _info_for(ch_idx: int) -> dict:
            if ch_idx < len(file_names) and file_names[ch_idx]:
                name = file_names[ch_idx]
                role = "stim_copy" if "stim" in name.lower() else "eeg"
                return {"name": name, "role": role}
            return sess_map.get(ch_idx, {"name": f"Ch{ch_idx}", "role": "eeg"})

        target_bins = choose_target_bins(n_samples) or 60_000
        fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                            vertical_spacing=0.005)

        decim_used = 1
        for ch_idx in range(n_ch):
            info = _info_for(ch_idx)
            color = _color_for_role(info["role"])
            x_plot, y_plot, decim_used = envelope_channel(
                signal, ch_idx, fs, 0, n_samples, target_bins,
            )
            fig.add_trace(go.Scattergl(
                x=x_plot, y=y_plot,
                mode="lines", name=info["name"],
                line=dict(color=color, width=0.8),
            ), row=ch_idx + 1, col=1)
            fig.update_yaxes(title_text=info["name"], row=ch_idx + 1, col=1,
                             title_font=dict(size=9, color="#aaa"),
                             tickfont=dict(size=8))

        fig.update_xaxes(title_text="Time (sec)", row=n_ch, col=1)
        filt_bits = []
        if hp and hp > 0: filt_bits.append(f"HP={hp:g}")
        if lp and lp > 0: filt_bits.append(f"LP={lp:g}")
        if notch and notch > 0: filt_bits.append(f"Notch={notch}")
        if smooth_ms and smooth_ms > 0: filt_bits.append(f"Smooth={smooth_ms:g}ms")
        filt_label = " | ".join(filt_bits) if filt_bits else "Raw"

        if decim_used > 1:
            bin_ms = decim_used / fs * 1000.0
            title = (f"LFP ({duration_sec:.1f} s) -- {n_ch} ch @ {fs:.0f} Hz "
                     f"-- envelope {bin_ms:.2f} ms/bin -- {filt_label}")
        else:
            title = (f"LFP ({duration_sec:.1f} s) -- {n_ch} ch @ {fs:.0f} Hz "
                     f"-- raw samples -- {filt_label}")
        fig.update_layout(
            title=title,
            height=max(600, n_ch * 80),
            showlegend=False,
        )

        # Persisted filter state -- used by the zoom callback so it
        # decimates from the same filtered cache entry.
        state = {"hp": hp, "lp": lp, "notch": notch,
                 "smooth": smooth_ms,
                 "show_psd": bool(show_psd_val)}

        show_psd = bool(show_psd_val)
        if not show_psd:
            return fig, no_update, psd_hidden_style, state

        # --- PSD figure ------------------------------------------- #
        psd_fig = make_subplots(rows=n_ch, cols=1, shared_xaxes=True,
                                vertical_spacing=0.06)
        for ch_idx in range(n_ch):
            info = _info_for(ch_idx)
            color = _color_for_role(info["role"])
            sig_ch = signal[:, ch_idx]
            freqs, psd = compute_psd(sig_ch, fs)
            if len(freqs) == 0:
                continue
            # Clip x to 0..1000 Hz so the operationally interesting
            # band is centered; user can zoom plotly to see higher.
            x_cap = min(1000.0, fs / 2.0)
            mask = freqs <= x_cap
            psd_fig.add_trace(go.Scattergl(
                x=freqs[mask], y=psd[mask],
                mode="lines", name=info["name"],
                line=dict(color=color, width=1.2),
            ), row=ch_idx + 1, col=1)
            psd_fig.update_yaxes(type="log", title_text=info["name"],
                                  row=ch_idx + 1, col=1,
                                  title_font=dict(size=9, color="#aaa"),
                                  tickfont=dict(size=8))
            # Line-noise guides
            for line_hz in (50, 60, 120, 180):
                if line_hz < x_cap:
                    psd_fig.add_vline(
                        x=line_hz, line_width=1,
                        line_dash="dot", line_color="#666",
                        row=ch_idx + 1, col=1,
                    )
        psd_fig.update_xaxes(title_text="Hz", row=n_ch, col=1)
        psd_fig.update_layout(
            title=f"Power spectral density (Welch) -- {filt_label}",
            height=max(220, n_ch * 140),
            showlegend=False,
        )
        return fig, psd_fig, {"display": "block",
                              "marginTop": "12px"}, state

    @app.callback(
        Output("lfp-plot", "figure", allow_duplicate=True),
        Input("lfp-plot", "relayoutData"),
        State("lfp-file-dropdown", "value"),
        State("lfp-filter-state", "data"),
        prevent_initial_call=True,
    )
    def _lfp_zoom(relayout, file_path, filter_state):
        if not file_path or not relayout:
            return no_update
        x0, x1, is_reset = parse_relayout(relayout)
        if x0 is None and x1 is None and not is_reset:
            return no_update
        try:
            chunk = get_chunk(file_path)
        except Exception:
            return no_update

        fs = chunk.fs
        # Use the persisted filter settings so zoom re-decimates from
        # the same filtered cache the time-domain plot built from.
        st = filter_state or {}
        signal = get_filtered(
            file_path, chunk.signal, fs,
            highpass=st.get("hp"), lowpass=st.get("lp"),
            notch=st.get("notch"), smoothing_ms=st.get("smooth"),
        )
        n_samples, n_ch = signal.shape
        if is_reset:
            lo, hi = 0, n_samples
        else:
            lo, hi = window_slice(n_samples, fs, x0, x1)
        if hi <= lo:
            return no_update

        target_bins = choose_target_bins(hi - lo) or 60_000
        patch = Patch()
        for ch in range(n_ch):
            x_p, y_p, _ = envelope_channel(
                signal, ch, fs, lo, hi, target_bins,
            )
            patch["data"][ch]["x"] = x_p
            patch["data"][ch]["y"] = y_p
        return patch

    @app.callback(
        Output("lfp-file-dropdown", "options"),
        Input("lfp-session-dropdown", "value"),
    )
    def update_lfp_file_options(session_dir):
        if not session_dir:
            return []
        files = _get_processed_files_for_session(store, session_dir)
        return [{"label": f"{f['chunk_datetime'][:16]} - {os.path.basename(f['file_path'])}",
                 "value": f["file_path"]} for f in files]

    # ------------------------------------------------------------------ #
    #  Electrode Health callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("electrode-health-plot", "figure"),
        [Input("electrode-health-session-dropdown", "value"),
         Input("electrode-health-hours-dropdown", "value")],
    )
    def update_electrode_health(session_dir, hours):
        if not session_dir:
            return _empty_fig("Select a session", 700)

        hours_val = int(hours) if hours else 0
        try:
            data = store.get_qc_timeseries(session_dir=session_dir, hours=hours_val)
        except Exception as e:
            return _empty_fig(f"Error: {e}", 700)

        if not data:
            return _empty_fig("No QC data available", 700)

        # Separate by channel_role
        stim_copy_data = [d for d in data if d.get("channel_role") == "stim_copy"]
        reference_data = [d for d in data if d.get("channel_role") == "reference"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=["Stim Copy RMS", "Reference RMS",
                                            "Line Noise Ratio (60 Hz indicator)"],
                            vertical_spacing=0.08)

        # Stim copy RMS
        stim_channels = sorted(set(d["channel"] for d in stim_copy_data))
        for ch in stim_channels:
            ch_data = [d for d in stim_copy_data if d["channel"] == ch]
            ch_name = ch_data[0].get("channel_name") or f"Ch{ch}"
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["rms_amplitude"] for d in ch_data],
                mode="lines+markers", name=f"{ch_name} (stim_copy)",
                line=dict(color="#888888"), marker=dict(size=3),
            ), row=1, col=1)

        # Reference RMS
        ref_channels = sorted(set(d["channel"] for d in reference_data))
        for ch in ref_channels:
            ch_data = [d for d in reference_data if d["channel"] == ch]
            ch_name = ch_data[0].get("channel_name") or f"Ch{ch}"
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["rms_amplitude"] for d in ch_data],
                mode="lines+markers", name=f"{ch_name} (reference)",
                line=dict(color="#00CC96"), marker=dict(size=3),
            ), row=2, col=1)

        # Line noise ratio for ALL channels (aggregate trend)
        all_channels = sorted(set(d["channel"] for d in data))
        ch_map = _get_channel_map(store, session_dir)
        for ch in all_channels:
            ch_data = [d for d in data if d["channel"] == ch]
            info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
            fig.add_trace(go.Scatter(
                x=[d["chunk_datetime"] for d in ch_data],
                y=[d["line_noise_ratio"] for d in ch_data],
                mode="lines", name=info["name"],
                line=dict(color=_color_for_role(info["role"]), width=1),
                showlegend=False,
            ), row=3, col=1)

        fig.update_layout(
            height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_annotations(font=dict(color="white"))
        return fig

    # ------------------------------------------------------------------ #
    #  Session Compare callback
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("session-compare-waveform-plot", "figure"),
         Output("session-compare-features-plot", "figure")],
        [Input("compare-session-a-dropdown", "value"),
         Input("compare-session-b-dropdown", "value"),
         Input("compare-smooth", "value")],
    )
    def update_session_compare(session_a, session_b, smooth_ms):
        if not session_a or not session_b:
            return (_empty_fig("Select two sessions", 450),
                    _empty_fig("Select two sessions", 450))

        smooth_ms = float(smooth_ms or 0)

        # --- Waveform overlay ---
        wf_fig = go.Figure()
        for sess_dir, color, label in [(session_a, "#636EFA", "Session A"),
                                       (session_b, "#EF553B", "Session B")]:
            try:
                waveforms = store.get_evoked_waveforms_for_session(sess_dir)
            except Exception:
                waveforms = []

            if waveforms:
                # Use the latest waveform
                wf = waveforms[-1]
                time_ms = wf["time_axis_ms"]
                mean_tr = wf["mean_trace"]
                sem_tr = wf["sem_trace"]

                # Optional smoothing -- Gaussian filter in display-ms.
                if smooth_ms > 0 and mean_tr and len(time_ms) > 1:
                    try:
                        dt_ms = float(time_ms[1] - time_ms[0])
                        if dt_ms > 0:
                            fs_proxy = 1000.0 / dt_ms
                            mean_tr = apply_filter(
                                np.asarray(mean_tr, dtype=np.float32),
                                fs_proxy, smoothing_ms=smooth_ms,
                            ).tolist()
                            if sem_tr and len(sem_tr) == len(mean_tr):
                                sem_tr = apply_filter(
                                    np.asarray(sem_tr, dtype=np.float32),
                                    fs_proxy, smoothing_ms=smooth_ms,
                                ).tolist()
                    except Exception as e:
                        logger.debug("Compare smoothing skipped: %s", e)

                sessions = store.get_sessions()
                sname = sess_dir
                for s in sessions:
                    if s["session_dir"] == sess_dir:
                        sname = s["session_name"]
                        break

                if sem_tr and len(sem_tr) == len(mean_tr):
                    upper = [m + s for m, s in zip(mean_tr, sem_tr)]
                    lower = [m - s for m, s in zip(mean_tr, sem_tr)]
                    fill_color = color.replace(")", ",0.15)").replace("rgb", "rgba") if color.startswith("rgb") else color
                    if color == "#636EFA":
                        fill_color = "rgba(99,110,250,0.15)"
                    else:
                        fill_color = "rgba(239,85,59,0.15)"
                    fig_band_x = list(time_ms) + list(reversed(time_ms))
                    fig_band_y = upper + list(reversed(lower))
                    wf_fig.add_trace(go.Scatter(
                        x=fig_band_x, y=fig_band_y,
                        fill="toself", fillcolor=fill_color,
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))

                wf_fig.add_trace(go.Scatter(
                    x=time_ms, y=mean_tr, mode="lines",
                    name=f"{label}: {sname}",
                    line=dict(color=color, width=2),
                ))

        wf_fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
        wf_fig.update_layout(
            title="Mean Evoked Waveform Overlay",
            xaxis_title="Time (ms)", yaxis_title="Amplitude",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # --- Feature distribution comparison (top 5 features) ---
        top_features = ["peak_amplitude", "trough_amplitude", "rms_amplitude",
                        "line_length", "peak_to_trough"]
        feat_fig = go.Figure()

        for sess_dir, color, label in [(session_a, "#636EFA", "Session A"),
                                       (session_b, "#EF553B", "Session B")]:
            means = []
            stds = []
            f_names = []
            for feat in top_features:
                try:
                    fdata = store.get_evoked_feature_timeseries(
                        feature_name=feat, session_dir=sess_dir)
                    vals = [d["value"] for d in fdata if d["value"] is not None
                            and not d.get("is_artifact", 0)]
                except Exception:
                    vals = []

                f_names.append(EVOKED_FEATURE_LABELS.get(feat, feat))
                if vals:
                    means.append(statistics.mean(vals))
                    stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)

            sessions = store.get_sessions()
            sname = sess_dir
            for s in sessions:
                if s["session_dir"] == sess_dir:
                    sname = s["session_name"]
                    break

            feat_fig.add_trace(go.Bar(
                x=f_names, y=means,
                name=f"{label}: {sname}",
                marker_color=color,
                error_y=dict(type="data", array=stds, visible=True),
            ))

        feat_fig.update_layout(
            title="Feature Distribution Comparison (top 5)",
            xaxis_title="Feature", yaxis_title="Value",
            barmode="group", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return wf_fig, feat_fig

    # ------------------------------------------------------------------ #
    #  Activity Log callback
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("activity-log-table", "data"),
        [Input("activity-log-hours-dropdown", "value"),
         Input("activity-log-level-dropdown", "value")],
    )
    def update_activity_log(hours, level):
        hours_val = int(hours) if hours else 24
        level_val = level if level and level != "ALL" else None
        try:
            logs = store.get_activity_log(hours=hours_val, level=level_val, limit=500)
        except Exception:
            logs = []
        return [
            {
                "timestamp": entry.get("timestamp", "")[:19],
                "level": entry.get("level", ""),
                "action": entry.get("action", ""),
                "message": (entry.get("message") or "")[:200],
                "file_path": os.path.basename(entry.get("file_path") or ""),
                "duration": f"{entry['duration_sec']:.2f}" if entry.get("duration_sec") else "",
            }
            for entry in logs
        ]

    # ------------------------------------------------------------------ #
    #  Annotations callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        [Output("annotation-status", "children"),
         Output("annotation-table", "data")],
        Input("annotation-submit-btn", "n_clicks"),
        [State("annotation-timestamp", "value"),
         State("annotation-note", "value"),
         State("annotation-category", "value"),
         State("annotation-session-dropdown", "value")],
        prevent_initial_call=True,
    )
    def submit_annotation(n_clicks, timestamp_val, note, category, session_dir):
        if not n_clicks:
            return no_update, no_update
        if not note or not note.strip():
            return html.Div("Note cannot be empty", style={"color": "#EF553B"}), no_update
        ts = timestamp_val or datetime.now().isoformat()
        try:
            ann_id = store.add_annotation(
                timestamp=ts,
                note=note.strip(),
                category=category or "observation",
                session_dir=session_dir or None,
                user_email=current_user_email(),
            )
            all_ann = store.get_annotations()
            table_data = _annotations_table_data(all_ann)
            return (
                html.Div(f"Annotation #{ann_id} saved", style={"color": "#00CC96"}),
                table_data,
            )
        except Exception as e:
            return html.Div(f"Error: {e}", style={"color": "#EF553B"}), no_update

    # ------------------------------------------------------------------ #
    #  Settings callbacks
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("settings-save-status", "children"),
        Input("btn-save-settings", "n_clicks"),
        [State("evoked-pre-stim", "value"),
         State("evoked-post-stim", "value"),
         State("evoked-baseline-correction", "value"),
         State("evoked-baseline-start", "value"),
         State("evoked-baseline-end", "value"),
         State("evoked-stim-thresh", "value"),
         State("evoked-min-stim-dist", "value"),
         State("evoked-notch60", "value"),
         State("evoked-notch50", "value"),
         State("evoked-hp-enabled", "value"),
         State("evoked-hp-cutoff", "value"),
         State("evoked-lp-enabled", "value"),
         State("evoked-lp-cutoff", "value"),
         State("crit-ar-order", "value"),
         State("crit-window-sec", "value"),
         State("crit-overlap", "value"),
         State("crit-fit-method", "value"),
         State("crit-target-sr", "value"),
         State("crit-type-b", "value"),
         State("crit-error-bars", "value"),
         State("seiz-spike-thresh", "value"),
         State("seiz-spike-min-w", "value"),
         State("seiz-spike-max-w", "value"),
         State("seiz-min-dur", "value"),
         State("seiz-glue-sec", "value"),
         State("seiz-min-spikes", "value"),
         State("seiz-outlier", "value"),
         State("art-method", "value"),
         State("art-fixed-thresh", "value"),
         State("art-mad-k", "value"),
         State("art-merge-gap", "value"),
         State("qc-artifact-warn", "value"),
         State("qc-flatline-std", "value"),
         State("qc-clipping-v", "value"),
         State("qc-linenoise-warn", "value"),
         State("features-checklist", "value"),
         # Step 2: Feature Analysis states
         State("feat-win-start", "value"),
         State("feat-win-end", "value"),
         State("feat-stim-art-start", "value"),
         State("feat-stim-art-end", "value"),
         State("feat-bp-enabled", "value"),
         State("feat-bp-hp", "value"),
         State("feat-bp-lp", "value"),
         State("feat-notch-enabled", "value"),
         State("feat-notch-freq", "value"),
         State("feat-smooth-enabled", "value"),
         State("feat-smooth-win", "value"),
         State("feat-baseline", "value"),
         State("feat-art-enabled", "value"),
         State("feat-art-method", "value"),
         State("feat-art-thresh", "value"),
         State("feat-art-merge", "value"),
         State("feat-tmpl-source", "value"),
         State("feat-tmpl-upper", "value"),
         State("feat-tmpl-lower", "value"),
         State("feat-rawamp-k", "value"),
         State("feat-ictal-rescue", "value"),
         State("feat-ictal-win", "value"),
         State("feat-analysis-start", "value"),
         State("feat-analysis-end", "value"),
         State("feat-early-start", "value"),
         State("feat-early-end", "value"),
         State("feat-late-start", "value"),
         State("feat-late-end", "value"),
         ],
        prevent_initial_call=True,
    )
    def save_settings(n_clicks, *values):
        if not n_clicks:
            return no_update
        try:
            cfg = _load_config()
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr, type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec, min_spikes, outlier,
             art_method, art_fixed, art_mad, art_merge,
             qc_art_warn, qc_flat, qc_clip, qc_ln,
             features_enabled,
             # Step 2 values
             f_win_start, f_win_end, f_stim_start, f_stim_end,
             f_bp_en, f_bp_hp, f_bp_lp, f_notch_en, f_notch_freq,
             f_smooth_en, f_smooth_win, f_baseline,
             f_art_en, f_art_method, f_art_thresh, f_art_merge,
             f_tmpl_src, f_tmpl_upper, f_tmpl_lower, f_rawamp_k,
             f_ictal_en, f_ictal_win,
             f_analysis_start, f_analysis_end,
             f_early_start, f_early_end, f_late_start, f_late_end,
             ) = values

            # Step 1: Epoch Extraction
            ee = cfg.setdefault("epoch_extraction", {})
            if pre_stim is not None: ee["pre_stimulus_ms"] = float(pre_stim)
            if post_stim is not None: ee["post_stimulus_ms"] = float(post_stim)
            ee["baseline_correction"] = bool(baseline_corr)
            ee["baseline_start_ms"] = float(bl_start or -60)
            ee["baseline_end_ms"] = float(bl_end or -10)
            if stim_thresh is not None: ee["stimulus_threshold_std"] = float(stim_thresh)
            if min_stim_dist is not None: ee["min_stimulus_distance_sec"] = float(min_stim_dist)
            ee["notch_60hz"] = bool(notch60)
            ee["notch_50hz"] = bool(notch50)
            ee["highpass_enabled"] = bool(hp_en)
            if hp_cut is not None: ee["highpass_cutoff_hz"] = float(hp_cut)
            ee["lowpass_enabled"] = bool(lp_en)
            if lp_cut is not None: ee["lowpass_cutoff_hz"] = float(lp_cut)
            cfg.pop("evoked", None)

            # Step 2: Feature Analysis
            fa = cfg.setdefault("feature_analysis", {})
            if f_win_start is not None: fa["window_start_ms"] = float(f_win_start)
            if f_win_end is not None: fa["window_end_ms"] = float(f_win_end)
            if f_stim_start is not None: fa["stim_artifact_start_ms"] = float(f_stim_start)
            if f_stim_end is not None: fa["stim_artifact_end_ms"] = float(f_stim_end)
            fa["bandpass_enabled"] = bool(f_bp_en)
            if f_bp_hp is not None: fa["bandpass_highpass_hz"] = float(f_bp_hp)
            if f_bp_lp is not None: fa["bandpass_lowpass_hz"] = float(f_bp_lp)
            fa["notch_enabled"] = bool(f_notch_en)
            if f_notch_freq is not None: fa["notch_frequency_hz"] = float(f_notch_freq)
            fa["smoothing_enabled"] = bool(f_smooth_en)
            if f_smooth_win is not None: fa["smoothing_window_ms"] = float(f_smooth_win)
            fa["baseline_correction"] = bool(f_baseline)
            fa["artifact_exclusion_enabled"] = bool(f_art_en)
            if f_art_method: fa["artifact_method"] = f_art_method
            if f_art_thresh is not None: fa["artifact_threshold"] = float(f_art_thresh)
            if f_art_merge is not None: fa["artifact_merge_gap_sec"] = float(f_art_merge)
            if f_tmpl_src: fa["template_source"] = f_tmpl_src
            if f_tmpl_upper is not None: fa["template_upper_r"] = float(f_tmpl_upper)
            if f_tmpl_lower is not None: fa["template_lower_r"] = float(f_tmpl_lower)
            if f_rawamp_k is not None: fa["rawamp_multiplier"] = float(f_rawamp_k)
            fa["ictal_rescue_enabled"] = bool(f_ictal_en)
            if f_ictal_win is not None: fa["ictal_rescue_window_ms"] = float(f_ictal_win)
            if f_analysis_start is not None: fa["analysis_start_ms"] = float(f_analysis_start)
            if f_analysis_end is not None: fa["analysis_end_ms"] = float(f_analysis_end)
            if f_early_start is not None: fa["early_area_start_ms"] = float(f_early_start)
            if f_early_end is not None: fa["early_area_end_ms"] = float(f_early_end)
            if f_late_start is not None: fa["late_area_start_ms"] = float(f_late_start)
            if f_late_end is not None: fa["late_area_end_ms"] = float(f_late_end)

            cfg["criticality"]["ar_order"] = int(ar_order) if ar_order is not None else cfg["criticality"]["ar_order"]
            cfg["criticality"]["window_sec"] = float(win_sec) if win_sec is not None else cfg["criticality"]["window_sec"]
            cfg["criticality"]["overlap_pct"] = int(overlap) if overlap is not None else cfg["criticality"]["overlap_pct"]
            cfg["criticality"]["fit_method"] = fit_method or cfg["criticality"]["fit_method"]
            cfg["criticality"]["target_sampling_rate"] = int(target_sr) if target_sr is not None else cfg["criticality"]["target_sampling_rate"]
            cfg["criticality"]["criticality_type_b"] = int(type_b) if type_b is not None else cfg["criticality"]["criticality_type_b"]
            cfg["criticality"]["calculate_error_bars"] = bool(err_bars)

            cfg["seizure"]["spike_threshold_uv"] = float(sp_thresh) if sp_thresh is not None else cfg["seizure"]["spike_threshold_uv"]
            cfg["seizure"]["spike_min_width"] = int(sp_min_w) if sp_min_w is not None else cfg["seizure"]["spike_min_width"]
            cfg["seizure"]["spike_max_width"] = int(sp_max_w) if sp_max_w is not None else cfg["seizure"]["spike_max_width"]
            cfg["seizure"]["min_seizure_duration_sec"] = float(min_dur) if min_dur is not None else cfg["seizure"]["min_seizure_duration_sec"]
            cfg["seizure"]["event_glue_sec"] = float(glue_sec) if glue_sec is not None else cfg["seizure"]["event_glue_sec"]
            cfg["seizure"]["min_spikes_per_sec"] = float(min_spikes) if min_spikes is not None else cfg["seizure"]["min_spikes_per_sec"]
            cfg["seizure"]["outlier_factor"] = float(outlier) if outlier is not None else cfg["seizure"]["outlier_factor"]

            cfg["artifact"]["method"] = art_method or cfg["artifact"]["method"]
            cfg["artifact"]["fixed_threshold"] = float(art_fixed) if art_fixed is not None else cfg["artifact"]["fixed_threshold"]
            cfg["artifact"]["mad_k"] = float(art_mad) if art_mad is not None else cfg["artifact"]["mad_k"]
            cfg["artifact"]["merge_gap_sec"] = float(art_merge) if art_merge is not None else cfg["artifact"]["merge_gap_sec"]

            cfg["qc_thresholds"]["artifact_pct_warning"] = float(qc_art_warn) if qc_art_warn is not None else cfg["qc_thresholds"]["artifact_pct_warning"]
            cfg["qc_thresholds"]["flatline_std"] = float(qc_flat) if qc_flat is not None else cfg["qc_thresholds"]["flatline_std"]
            cfg["qc_thresholds"]["clipping_voltage"] = float(qc_clip) if qc_clip is not None else cfg["qc_thresholds"]["clipping_voltage"]
            cfg["qc_thresholds"]["line_noise_ratio_warning"] = float(qc_ln) if qc_ln is not None else cfg["qc_thresholds"]["line_noise_ratio_warning"]

            cfg["features"]["enabled"] = features_enabled or []

            _save_config(cfg)
            return html.Div("Settings saved to config.yaml",
                            style={"color": "#00CC96", "marginTop": "10px"})
        except Exception as e:
            logger.error("Failed to save settings: %s", e, exc_info=True)
            return html.Div(f"Error saving: {e}",
                            style={"color": "#EF553B", "marginTop": "10px"})

    @app.callback(
        [Output("settings-version-status", "children"),
         Output("version-history-table", "data")],
        Input("btn-save-version", "n_clicks"),
        [State("version-label-input", "value"),
         State("evoked-pre-stim", "value"),
         State("evoked-post-stim", "value"),
         State("evoked-baseline-correction", "value"),
         State("evoked-baseline-start", "value"),
         State("evoked-baseline-end", "value"),
         State("evoked-stim-thresh", "value"),
         State("evoked-min-stim-dist", "value"),
         State("evoked-notch60", "value"),
         State("evoked-notch50", "value"),
         State("evoked-hp-enabled", "value"),
         State("evoked-hp-cutoff", "value"),
         State("evoked-lp-enabled", "value"),
         State("evoked-lp-cutoff", "value"),
         State("crit-ar-order", "value"),
         State("crit-window-sec", "value"),
         State("crit-overlap", "value"),
         State("crit-fit-method", "value"),
         State("crit-target-sr", "value"),
         State("crit-type-b", "value"),
         State("crit-error-bars", "value"),
         State("seiz-spike-thresh", "value"),
         State("seiz-spike-min-w", "value"),
         State("seiz-spike-max-w", "value"),
         State("seiz-min-dur", "value"),
         State("seiz-glue-sec", "value"),
         State("seiz-min-spikes", "value"),
         State("seiz-outlier", "value"),
         State("art-method", "value"),
         State("art-fixed-thresh", "value"),
         State("art-mad-k", "value"),
         State("art-merge-gap", "value"),
         State("qc-artifact-warn", "value"),
         State("qc-flatline-std", "value"),
         State("qc-clipping-v", "value"),
         State("qc-linenoise-warn", "value"),
         State("features-checklist", "value")],
        prevent_initial_call=True,
    )
    def save_version(n_clicks, label, *values):
        if not n_clicks:
            return no_update, no_update
        try:
            (pre_stim, post_stim, baseline_corr, bl_start, bl_end,
             stim_thresh, min_stim_dist, notch60, notch50,
             hp_en, hp_cut, lp_en, lp_cut,
             ar_order, win_sec, overlap, fit_method, target_sr, type_b, err_bars,
             sp_thresh, sp_min_w, sp_max_w, min_dur, glue_sec, min_spikes, outlier,
             art_method, art_fixed, art_mad, art_merge,
             qc_art_warn, qc_flat, qc_clip, qc_ln,
             features_enabled) = values

            config_dict = {
                "epoch_extraction": {
                    "pre_stimulus_ms": float(pre_stim or -100),
                    "post_stimulus_ms": float(post_stim or 500),
                    "baseline_correction": bool(baseline_corr),
                    "baseline_start_ms": float(bl_start or -60),
                    "baseline_end_ms": float(bl_end or -10),
                    "stimulus_threshold_std": float(stim_thresh or 3.0),
                    "min_stimulus_distance_sec": float(min_stim_dist or 0.1),
                    "notch_60hz": bool(notch60),
                    "notch_50hz": bool(notch50),
                    "highpass_enabled": bool(hp_en),
                    "highpass_cutoff_hz": float(hp_cut or 1.0),
                    "lowpass_enabled": bool(lp_en),
                    "lowpass_cutoff_hz": float(lp_cut or 1000.0),
                },
                "feature_analysis": {
                    "analysis_start_ms": 5, "analysis_end_ms": 50,
                    "early_area_start_ms": 0, "early_area_end_ms": 50,
                    "late_area_start_ms": 50, "late_area_end_ms": 200,
                },
                "criticality": {
                    "ar_order": int(ar_order or 5),
                    "window_sec": float(win_sec or 2.0),
                    "overlap_pct": int(overlap or 50),
                    "fit_method": fit_method or "YuleWalker",
                    "target_sampling_rate": int(target_sr or 1000),
                    "criticality_type_b": int(type_b or 2),
                    "calculate_error_bars": bool(err_bars),
                },
                "seizure": {
                    "spike_threshold_uv": float(sp_thresh or 10),
                    "spike_min_width": int(sp_min_w or 5),
                    "spike_max_width": int(sp_max_w or 50),
                    "min_seizure_duration_sec": float(min_dur or 5),
                    "event_glue_sec": float(glue_sec or 2),
                    "min_spikes_per_sec": float(min_spikes or 2),
                    "outlier_factor": float(outlier or 3),
                },
                "artifact": {
                    "method": art_method or "fixed",
                    "fixed_threshold": float(art_fixed or 500),
                    "mad_k": float(art_mad or 4.0),
                    "merge_gap_sec": float(art_merge or 2.0),
                },
                "qc_thresholds": {
                    "artifact_pct_warning": float(qc_art_warn or 50),
                    "flatline_std": float(qc_flat or 1e-6),
                    "clipping_voltage": float(qc_clip or 10.0),
                    "line_noise_ratio_warning": float(qc_ln or 0.2),
                },
                "features": {"enabled": features_enabled or []},
            }

            vid = store.create_settings_version(config_dict, label=label or "")
            versions = store.get_all_settings_versions()
            table_data = _version_table_data(versions)
            return (
                html.Div(f"Version #{vid} saved with label '{label or ''}'",
                         style={"color": "#00CC96", "marginTop": "10px"}),
                table_data,
            )
        except Exception as e:
            logger.error("Failed to save version: %s", e, exc_info=True)
            return (
                html.Div(f"Error: {e}", style={"color": "#EF553B", "marginTop": "10px"}),
                no_update,
            )

    @app.callback(
        Output("reprocess-status", "children"),
        Input("btn-reprocess", "n_clicks"),
        prevent_initial_call=True,
    )
    def reprocess_session(n_clicks):
        if not n_clicks:
            return no_update
        try:
            conn = store._connect()
            try:
                conn.execute("UPDATE processed_files SET status = 'pending' WHERE status = 'done'")
                cnt = conn.execute("SELECT changes()").fetchone()[0]
                conn.commit()
            finally:
                conn.close()
            return html.Div(f"Queued {cnt} files for reprocessing",
                            style={"color": "#FFA15A", "marginTop": "10px"})
        except Exception as e:
            return html.Div(f"Error: {e}", style={"color": "#EF553B", "marginTop": "10px"})

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

    return app


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


def _home_incidents_block(config: dict | None) -> html.Div:
    """Latest 3 incident reports from the Maintenance Tracker."""
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
        items = []
        for r in rows:
            when = r["when"].strftime("%Y-%m-%d %H:%M")
            who = r.get("by") or "?"
            report = r.get("report") or ""
            if len(report) > 140:
                report = report[:137] + "..."
            items.append(html.Div([
                html.Div([
                    html.Span(when, style={
                        "color": COLOR_TEXT_SECONDARY,
                        "fontSize": FONT_SIZE_CAPTION,
                        "marginRight": SPACE_3,
                    }),
                    html.Span(who, style={"color": COLOR_ACCENT,
                                           "fontSize": FONT_SIZE_CAPTION,
                                           "fontWeight": "600"}),
                ]),
                html.Div(report, style={"color": COLOR_TEXT_PRIMARY,
                                         "fontSize": FONT_SIZE_BODY,
                                         "marginTop": "2px"}),
            ], style={"padding": f"{SPACE_2} 0",
                      "borderBottom": f"1px solid {COLOR_DIVIDER}"}))
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


def _build_home_grid_children(store: Store, config: dict | None,
                                today: date) -> list:
    """The five lab-side home blocks. Extracted so the refresh
    callback can rebuild them in place without re-rendering the
    cards / queue / waveform thumbnail above and below."""
    return [
        _home_surgery_block(store, config, today),
        _home_schedule_block(config),
        _home_maintenance_block(config),
        _home_incidents_block(config),
        _home_data_log_block(store, config),
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
        _status_card("Network", "OK" if net_ok else "DOWN",
                     "#00CC96" if net_ok else "#EF553B"),
        _status_card("CPU", f"{cpu:.0f}%",
                     "#00CC96" if cpu < 80 else "#FFA15A"),
        _status_card("Memory", f"{mem:.0f}%",
                     "#00CC96" if mem < 85 else "#FFA15A"),
        _status_card("Disk Free", f"{disk:.1f} GB",
                     "#00CC96" if disk > 50 else "#EF553B"),
        _status_card("Files/Hour", str(fph), "#636EFA"),
        _status_card("Videos (24h)", vid_label, vid_color),
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
    subplot_titles = []
    for ch in sorted_chs:
        nm = chan_label[ch]
        subplot_titles.append(
            f"Stim window {nm} ({stim_x0:g} to {stim_x1:g} ms)")
        subplot_titles.append(
            f"Evoked {nm} ({evoked_x0:g}–{evoked_x1:g} ms)")
    thumb_fig = make_subplots(
        rows=n_ch, cols=2, column_widths=[0.2, 0.8],
        shared_xaxes=False, subplot_titles=subplot_titles,
        vertical_spacing=0.08, horizontal_spacing=0.06,
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
                ), row=ri, col=col_idx)
                thumb_fig.add_trace(go.Scatter(
                    x=time_ms, y=lower, mode="lines",
                    fill="tonexty", fillcolor=band_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip", showlegend=False,
                ), row=ri, col=col_idx)

        # Left panel: LFP mean, zoomed to ±1 ms around stim.
        thumb_fig.add_trace(go.Scatter(
            x=time_ms, y=mean_tr, mode="lines",
            name=f"{nm} stim",
            line=dict(color=color, width=1.5),
            showlegend=False,
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

        # Right panel: LFP mean, evoked window.
        thumb_fig.add_trace(go.Scatter(
            x=time_ms, y=mean_tr, mode="lines",
            name=f"{nm} (n={n_ep})",
            line=dict(color=color, width=1.5),
        ), row=ri, col=2)
        thumb_fig.update_xaxes(range=[evoked_x0, evoked_x1],
                                row=ri, col=2)
        y_right = _yrange_pad(time_ms, mean_tr, evoked_x0, evoked_x1)
        if y_right is not None:
            thumb_fig.update_yaxes(range=list(y_right), row=ri, col=2)

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
    thumb_fig.update_layout(
        title=dict(
            text=f"Latest Evoked — {latest_datetime[:16]} ({mode_label})",
            font=dict(size=12)),
        autosize=True,
        margin=dict(l=50, r=10, t=36, b=22),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                     xanchor="right", x=1, font_size=9),
    )
    thumb_fig.update_annotations(font_size=10)
    return [dcc.Graph(
        figure=thumb_fig, responsive=True,
        style={
            "marginTop": "4px",
            # vh-based so a 1440p / 4K monitor gets a taller chart
            # without code changes. minHeight keeps each channel
            # row readable on a small laptop; maxHeight stops
            # absurdly tall plots on a giant monitor.
            "height": "62vh",
            "minHeight": f"{max(440, 110 * n_ch)}px",
            "maxHeight": "820px",
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
            html.Span("KM Recorder log",
                       style={"color": "#aaa", "fontSize": "12px",
                               "letterSpacing": "0.5px",
                               "marginRight": "12px"}),
            html.Span(f"last recording {rec_text}",
                       style={"color": rec_color, "fontSize": "12px",
                               "fontWeight": "600",
                               "marginRight": "12px"}),
            html.Span(f"last submitted {sub_text}",
                       style={"color": sub_color, "fontSize": "12px",
                               "fontWeight": "600"}),
        ], style={"marginBottom": "6px"}),
        html.Div(tail),
    ], style={**SECTION_STYLE, "padding": "10px 14px"})


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
    conn = store._connect()
    try:
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
    finally:
        conn.close()


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
        height=44, margin=dict(l=10, r=10, t=4, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0,
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#888"),
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
    hour chunks still show clearly."""
    today = date.today()
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
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, zmin=0, zmax=60,
        colorscale=[[0.0, "#2a2a40"], [0.001, "#1f5a44"],
                     [0.5, "#00CC96"], [1.0, "#7be3c0"]],
        showscale=False, xgap=1, ygap=1,
        hovertemplate="%{y} %{x}:00 — %{z:.0f} min recorded"
                       "<extra></extra>",
    ))
    fig.update_layout(
        height=148, margin=dict(l=60, r=10, t=4, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,
                    tickfont=dict(size=9, color="#888"),
                    tickmode="array",
                    tickvals=["00", "06", "12", "18", "23"],
                    ticktext=["00", "06", "12", "18", "23"],
                    title=dict(text="hour of day",
                                font=dict(size=10, color="#666"))),
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
    conn = store._connect()
    try:
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
    finally:
        conn.close()

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
        html.H4("Today",
                style={"color": "#aaa", "marginTop": "0px",
                        "marginBottom": "4px", "fontSize": "14px",
                        "letterSpacing": "0.5px"}),
        today_counters,
        html.Div("Recording uptime (last 24h)",
                  style={"color": "#888", "fontSize": "11px",
                          "marginTop": "8px",
                          "letterSpacing": "0.5px"}),
        rec_24h,
        rec_24h_legend,
    ])

    recording_7d_section = html.Div([
        html.H4("Recording uptime (last 7 days)",
                style={"color": "#aaa", "marginTop": "16px",
                        "marginBottom": "4px", "fontSize": "14px",
                        "letterSpacing": "0.5px"}),
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
            html.H4("Active Session",
                    style={"color": "#aaa", "marginTop": "16px",
                            "marginBottom": "4px", "fontSize": "14px",
                            "letterSpacing": "0.5px"}),
            bar, active_counters,
        ])
    else:
        active_section = html.Div([
            html.H4("Active Session",
                    style={"color": "#aaa", "marginTop": "16px",
                            "fontSize": "14px",
                            "letterSpacing": "0.5px"}),
            html.Div("No sessions yet",
                     style={"color": "#888", "fontSize": "12px"}),
        ])

    return [today_section, active_section, recording_7d_section]


def _overview_tab(store: Store, config: dict | None = None):
    sessions = store.get_sessions()
    active_session = sessions[0] if sessions else {}
    session_dir = active_session.get("session_dir", "")

    ch_map = _get_channel_map(store, session_dir) if session_dir else {}
    recent_alerts = store.get_recent_alerts(hours=24)

    cards = html.Div(
        _build_overview_cards(store),
        id="overview-cards",
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
    )

    km_section = html.Div(
        _build_km_log_section(config),
        id="overview-km-log",
    )

    queue_section = html.Div(
        _build_overview_queue(store),
        id="overview-queue", style=SECTION_STYLE,
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
                html.H4("Channel Map (Auto-Discovered)",
                         style={"color": "#aaa",
                                 "marginTop": "16px",
                                 "fontSize": "14px"}),
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
            ], style={"maxHeight": "320px", "overflow": "hidden"})
        else:
            channel_table = html.Div()
    else:
        session_info = html.P("No sessions yet", style={"color": "#888"})
        channel_table = html.Div()

    # Recent alerts -- capped + scrollable so it doesn't push the
    # rest of the Overview off the viewport.
    if recent_alerts:
        alerts_section = html.Div([
            html.H4(f"Recent Alerts ({len(recent_alerts)})",
                     style={"color": "#aaa", "marginTop": "16px",
                             "fontSize": "14px",
                             "letterSpacing": "0.5px"}),
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
        ], style={"maxHeight": "240px", "overflow": "hidden"})
    else:
        alerts_section = html.Div([
            html.H4("Recent Alerts",
                     style={"color": "#aaa", "marginTop": "16px",
                             "fontSize": "14px",
                             "letterSpacing": "0.5px"}),
            html.P("No alerts in the last 24 hours",
                    style={"color": "#888", "fontSize": "11px"}),
        ])

    today = date.today()
    # Two-column responsive grid -- collapses to one column under ~880px.
    # The whole grid is wrapped in a fixed-height scroll container so
    # the lab tiles don't push the rest of the Overview off-screen --
    # the user can wheel inside the strip if they want to see more.
    home_grid = html.Div(
        html.Div(
            _build_home_grid_children(store, config, today),
            id="overview-home-grid",
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(440px, 1fr))",
                "gap": "12px",
            },
        ),
        style={
            "marginTop": "12px",
            "maxHeight": "28vh",
            "overflowY": "auto",
            "paddingRight": "4px",
        },
    )

    # Two-column responsive layout for the dense upper portion.
    # Left column carries the operational tiles (cards, KM log,
    # queue + recording timelines, alerts); right column carries the
    # latest-evoked thumbnail + channel map. Collapses to a single
    # column below ~1280 px so narrower viewports still read cleanly.
    upper_grid = html.Div([
        html.Div([
            cards, km_section, queue_section,
            session_info, alerts_section,
        ]),
        html.Div([
            waveform_thumbnail, channel_table,
        ]),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fit, minmax(560px, 1fr))",
        "gap": "16px",
    })

    return html.Div([
        upper_grid, home_grid,
    ])


# ------------------------------------------------------------------ #
#  Evoked Waveforms tab (NEW)
# ------------------------------------------------------------------ #

def _waveforms_tab_layout(store: Store, default_session: str | None = None):
    session_options = _session_dropdown_options(store)
    default = _default_session(store, hint=default_session)

    # Build file options for the default session
    file_options = []
    default_file = None
    if default:
        files = _get_processed_files_for_session(store, default)
        file_options = [{"label": f["chunk_datetime"], "value": f["id"]} for f in files]
        if file_options:
            default_file = file_options[-1]["value"]  # latest file

    return html.Div([
        html.H3("Evoked Waveforms", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("File", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="waveform-file-dropdown",
                    options=file_options,
                    value=default_file,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="waveform-smooth", type="number", min=0,
                          step=0.5, value=0,
                          style={"backgroundColor": "#262638",
                                 "color": "#f0f0f5", "width": "80px"}),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="waveform-plot"),
    ])


# ------------------------------------------------------------------ #
#  Signal Quality tab (existing)
# ------------------------------------------------------------------ #

def _signal_quality_tab(store: Store):
    sessions = store.get_sessions()
    if not sessions:
        return html.Div("No sessions found.", style={"color": "#888"})

    session_dir = sessions[0].get("session_dir", "")
    ch_map = _get_channel_map(store, session_dir)

    data = store.get_qc_timeseries(session_dir=session_dir, hours=48)
    if not data:
        return html.Div("No QC data yet.", style={"color": "#888"})

    channels = sorted(set(d["channel"] for d in data))

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["RMS Amplitude", "Artifact %", "Line Noise Ratio"],
                        vertical_spacing=0.08)

    for ch in channels:
        ch_data = [d for d in data if d["channel"] == ch]
        ch_times = [d["chunk_datetime"] for d in ch_data]

        info = ch_map.get(ch, {"name": f"Ch{ch}", "role": "eeg"})
        ch_name = info["name"]
        color = _color_for_role(info["role"])

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["rms_amplitude"] for d in ch_data],
            name=ch_name, legendgroup=ch_name,
            line=dict(color=color), marker=dict(size=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["artifact_pct"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=ch_times, y=[d["line_noise_ratio"] for d in ch_data],
            name=ch_name, legendgroup=ch_name, showlegend=False,
            line=dict(color=color), marker=dict(size=2),
        ), row=3, col=1)

    fig.update_layout(
        height=750, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_annotations(font=dict(color="white"))

    return html.Div([dcc.Graph(figure=fig)])


# ------------------------------------------------------------------ #
#  Evoked Features tab (renamed from Evoked Response)
# ------------------------------------------------------------------ #

def _evoked_tab_layout(store: Store):
    """Build the Evoked Features tab — separate row per selected feature."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    plottable = [f for f in EVOKED_FEATURE_COLS
                 if f not in ("is_artifact", "is_ictal", "epoch_time_sec")]
    feature_options = [{"label": EVOKED_FEATURE_LABELS.get(f, f), "value": f}
                       for f in plottable]

    default_session = sessions[0]["session_dir"] if sessions else None
    # Default: show 3 key features
    default_features = ["peak_amplitude", "line_length", "recovery_tau"]

    return html.Div([
        html.H3("Evoked Features", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="evoked-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="evoked-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=0,  # all time by default
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "12px", "flexWrap": "wrap"}),

        html.Div([
            html.Label("Select features to display (each gets its own plot row):", style=LABEL_STYLE),
            dcc.Checklist(
                id="evoked-feature-checklist",
                options=feature_options,
                value=default_features,
                inline=True,
                style={"color": "#ddd", "fontSize": "12px"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px", "marginBottom": "4px"},
            ),
        ], style={**SECTION_STYLE, "marginBottom": "16px"}),

        # Hidden single-select for backward compat with callback
        dcc.Dropdown(id="evoked-feature-dropdown", value="peak_amplitude",
                     style={"display": "none"}),

        html.Div(id="evoked-multi-plots"),
        html.Div(id="evoked-stats"),
    ])


# ------------------------------------------------------------------ #
#  Criticality tab (existing)
# ------------------------------------------------------------------ #

def _criticality_tab_layout(store: Store):
    """Build the Criticality tab layout -- data loaded via callback."""
    sessions = store.get_sessions()
    session_options = [{"label": s["session_name"], "value": s["session_dir"]}
                       for s in sessions]
    default_session = sessions[0]["session_dir"] if sessions else None

    return html.Div([
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-session-dropdown",
                    options=session_options,
                    value=default_session,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="criticality-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="criticality-plot", style={"height": "550px"}),
    ])


# ------------------------------------------------------------------ #
#  LFP Browser tab (NEW)
# ------------------------------------------------------------------ #

def _lfp_browser_tab_layout(store: Store, default_session: str | None = None):
    session_options = _session_dropdown_options(store)
    default = _default_session(store, hint=default_session)

    return html.Div([
        html.H3("LFP Browser", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("File", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-file-dropdown",
                    options=[],
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "2", "minWidth": "350px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("Load LFP", id="lfp-load-btn", n_clicks=0,
                            style={"backgroundColor": "#636EFA", "color": "white",
                                   "border": "none", "padding": "8px 20px",
                                   "borderRadius": "6px", "cursor": "pointer",
                                   "fontSize": "14px", "fontWeight": "bold"}),
            ], style={"flex": "0 0 120px", "display": "flex", "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        html.P("Loads the entire raw LFP from the selected .mat file. Long files are min/max envelope-decimated so 150 us stim pulses remain visible. Channel names come from the file's fnstr.",
               style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"}),

        # --- Live filter strip ------------------------------------- #
        html.Div([
            html.Div([
                html.Label("Preset", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-filter-preset",
                    options=[
                        {"label": "Raw (no filter)", "value": "raw"},
                        {"label": "Delta (1-4 Hz)", "value": "delta"},
                        {"label": "Theta (4-8 Hz)", "value": "theta"},
                        {"label": "Alpha (8-13 Hz)", "value": "alpha"},
                        {"label": "Beta (13-30 Hz)", "value": "beta"},
                        {"label": "Gamma (30-100 Hz)", "value": "gamma"},
                        {"label": "Spike band (300-3000 Hz)",
                         "value": "spike"},
                        {"label": "Custom", "value": "custom"},
                    ],
                    value="raw",
                    clearable=False,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1 1 200px", "minWidth": "180px"}),
            html.Div([
                html.Label("HP (Hz)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-hp", type="number", min=0,
                          step=0.5, value=0, style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label("LP (Hz)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-lp", type="number", min=0,
                          step=1, value=0, style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label("Notch", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="lfp-filter-notch",
                    options=[
                        {"label": "Off", "value": 0},
                        {"label": "50 Hz", "value": 50},
                        {"label": "60 Hz", "value": 60},
                    ],
                    value=0,
                    clearable=False,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="lfp-filter-smooth", type="number", min=0,
                          step=1, value=0, style=DROPDOWN_STYLE),
            ], style={"flex": "0 0 110px"}),
            html.Div([
                html.Label("PSD", style=LABEL_STYLE),
                dcc.Checklist(
                    id="lfp-show-psd",
                    options=[{"label": " Show", "value": "on"}],
                    value=[],
                    style={"color": "white", "paddingTop": "6px"},
                ),
            ], style={"flex": "0 0 90px"}),
            html.Div([
                html.Label(" ", style=LABEL_STYLE),
                html.Button("Apply", id="lfp-apply-filter-btn",
                            n_clicks=0,
                            style={"backgroundColor": "#262638",
                                   "color": "white", "border": "1px solid #444",
                                   "padding": "8px 16px",
                                   "borderRadius": "6px", "cursor": "pointer",
                                   "fontSize": "13px"}),
            ], style={"flex": "0 0 100px", "display": "flex",
                      "alignItems": "flex-end"}),
        ], style={"display": "flex", "gap": "12px",
                  "marginBottom": "12px", "flexWrap": "wrap",
                  "padding": "10px 12px", "backgroundColor": "#13131f",
                  "borderRadius": "8px",
                  "border": "1px solid rgba(255,255,255,0.06)"}),

        # Persists the last-applied filter tuple so the zoom callback
        # uses the same settings without re-reading the inputs.
        dcc.Store(id="lfp-filter-state",
                  data={"hp": 0, "lp": 0, "notch": 0, "smooth": 0,
                        "show_psd": False}),

        dcc.Graph(id="lfp-plot", figure=_empty_fig("Select a session and file, then click Load", 600)),

        html.Div(
            dcc.Graph(id="lfp-psd-plot",
                      figure=_empty_fig("Toggle 'Show PSD' and click Apply", 280)),
            id="lfp-psd-row",
            style={"display": "none", "marginTop": "12px"},
        ),
    ])


# ------------------------------------------------------------------ #
#  Electrode Health tab (NEW)
# ------------------------------------------------------------------ #

def _electrode_health_tab_layout(store: Store, default_session: str | None = None):
    session_options = _session_dropdown_options(store)
    default = _default_session(store, hint=default_session)

    return html.Div([
        html.H3("Electrode Health", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-session-dropdown",
                    options=session_options,
                    value=default,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "250px"}),
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="electrode-health-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=48,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="electrode-health-plot", figure=_empty_fig("Select a session", 700)),
    ])


# ------------------------------------------------------------------ #
#  Session Compare tab (NEW)
# ------------------------------------------------------------------ #

def _session_compare_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)

    return html.Div([
        html.H3("Session Compare", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Session A", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-a-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Session B", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="compare-session-b-dropdown",
                    options=session_options,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "1", "minWidth": "300px"}),
            html.Div([
                html.Label("Smooth (ms)", style=LABEL_STYLE),
                dcc.Input(id="compare-smooth", type="number", min=0,
                          step=0.5, value=0,
                          style={"backgroundColor": "#262638",
                                 "color": "#f0f0f5", "width": "80px"}),
            ], style={"flex": "0 0 110px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dcc.Graph(id="session-compare-waveform-plot", style={"height": "450px"}),
        dcc.Graph(id="session-compare-features-plot", style={"height": "450px",
                                                              "marginTop": "16px"}),
    ])


# ------------------------------------------------------------------ #
#  Stim QC tab (existing)
# ------------------------------------------------------------------ #

def _stim_tab(store: Store):
    conn = store._connect()
    try:
        rows = conn.execute(
            """SELECT pf.chunk_datetime, pf.session_name, sq.*
               FROM stim_qc sq JOIN processed_files pf ON sq.file_id = pf.id
               ORDER BY pf.chunk_datetime DESC LIMIT 500"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return html.Div("No stimulation data yet.", style={"color": "#888"})

    data = [dict(r) for r in rows]
    return html.Div([
        html.H3("Stimulation Delivery QC", style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{
                "time": d["chunk_datetime"][:16],
                "session": d.get("session_name", ""),
                "channel": d["stim_channel"],
                "charge_nC": f"{d['charge_nC']:.1f}" if d["charge_nC"] else "",
                "freq_Hz": f"{d['frequency_hz']:.1f}" if d["frequency_hz"] else "",
                "pulses": d["total_pulses"],
                "expected": d["expected_pulses"],
                "delivery_%": f"{d['delivery_pct']:.1f}" if d["delivery_pct"] else "N/A",
            } for d in data],
            columns=[{"name": c, "id": c} for c in
                     ["time", "session", "channel", "charge_nC", "freq_Hz",
                      "pulses", "expected", "delivery_%"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{delivery_%} contains '0.0'"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ),
    ])


# ------------------------------------------------------------------ #
#  Settings tab (existing)
# ------------------------------------------------------------------ #

def _settings_tab_layout(store: Store):
    """Build the Settings tab with all subsections."""
    try:
        cfg = _load_config()
    except Exception:
        cfg = {}

    ee = cfg.get("epoch_extraction", {})   # Step 1: extraction window
    fa = cfg.get("feature_analysis", {})   # Step 2: analysis sub-window
    cr = cfg.get("criticality", {})
    sz = cfg.get("seizure", {})
    ar = cfg.get("artifact", {})
    qc = cfg.get("qc_thresholds", {})
    ft = cfg.get("features", {})

    all_features = [
        "Line Length", "Log(AUC)", "Peak Amplitude", "Trough Amplitude",
        "Peak-to-Trough", "RMS Amplitude", "Peak Latency", "Trough Latency",
        "Max Slope", "Max Slope Time", "Early Area", "Late Area",
        "Early/Late Ratio", "Recovery Tau", "Recovery Slope",
        "Template Correlation", "PCA Recon Error", "Variance",
        "Autocorrelation", "AC Width", "Rolling Variance", "Rolling AR(1)",
        "Rolling CV", "Exp Fit A", "Sum Power Low", "Freq Moment Low",
        "Sum Power High", "Freq Moment High",
    ]

    versions = store.get_all_settings_versions()
    version_data = _version_table_data(versions)

    def _input(id_, val, type_="number", **kw):
        return dcc.Input(id=id_, value=val, type=type_, style=INPUT_STYLE, **kw)

    def _check(id_, val):
        return dcc.Checklist(
            id=id_, options=[{"label": " Enabled", "value": True}],
            value=[True] if val else [],
            style={"color": "#ddd"},
        )

    return html.Div([
        # --- Step 1: Epoch Extraction ---
        html.Div([
            html.H4("Step 1: Epoch Extraction", style={"color": "#636EFA", "marginTop": "0"}),
            html.P("Cuts a window around each detected stimulus to produce evoked.mat files. "
                   "These intermediary files can also be analyzed manually.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Pre-stimulus (ms, negative = before)", style=LABEL_STYLE),
                          _input("evoked-pre-stim", ee.get("pre_stimulus_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Post-stimulus (ms)", style=LABEL_STYLE),
                          _input("evoked-post-stim", ee.get("post_stimulus_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stimulus threshold (std)", style=LABEL_STYLE),
                          _input("evoked-stim-thresh", ee.get("stimulus_threshold_std", 3.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min stim distance (sec)", style=LABEL_STYLE),
                          _input("evoked-min-stim-dist", ee.get("min_stimulus_distance_sec", 0.1), step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Baseline correction", style=LABEL_STYLE),
                          _check("evoked-baseline-correction", ee.get("baseline_correction", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline start (ms)", style=LABEL_STYLE),
                          _input("evoked-baseline-start", ee.get("baseline_start_ms", -60))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline end (ms)", style=LABEL_STYLE),
                          _input("evoked-baseline-end", ee.get("baseline_end_ms", -10))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Notch 60 Hz", style=LABEL_STYLE),
                          _check("evoked-notch60", ee.get("notch_60hz", True))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch 50 Hz", style=LABEL_STYLE),
                          _check("evoked-notch50", ee.get("notch_50hz", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass enabled", style=LABEL_STYLE),
                          _check("evoked-hp-enabled", ee.get("highpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass cutoff (Hz)", style=LABEL_STYLE),
                          _input("evoked-hp-cutoff", ee.get("highpass_cutoff_hz", 1.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass enabled", style=LABEL_STYLE),
                          _check("evoked-lp-enabled", ee.get("lowpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass cutoff (Hz)", style=LABEL_STYLE),
                          _input("evoked-lp-cutoff", ee.get("lowpass_cutoff_hz", 1000.0))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE, "borderLeft": "3px solid #636EFA"}),

        # --- Step 2: Feature Analysis (Configure & Analyze) ---
        html.Div([
            html.H4("Step 2: Feature Analysis (Configure & Analyze)",
                     style={"color": "#00CC96", "marginTop": "0"}),
            html.P("Mirrors the Chronic Evoked Features dialog. Configures filtering, "
                   "artifact exclusion, and feature windows applied to extracted epochs.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "12px"}),

            # Row 1: Evoked response window + stimulus artifact window
            html.H5("Windows", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Window start (ms)", style=LABEL_STYLE),
                          _input("feat-win-start", fa.get("window_start_ms", -100))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window end (ms)", style=LABEL_STYLE),
                          _input("feat-win-end", fa.get("window_end_ms", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact start (ms)", style=LABEL_STYLE),
                          _input("feat-stim-art-start", fa.get("stim_artifact_start_ms", -5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Stim artifact end (ms)", style=LABEL_STYLE),
                          _input("feat-stim-art-end", fa.get("stim_artifact_end_ms", 15))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 2: Filtering
            html.H5("Filtering", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Bandpass filter", style=LABEL_STYLE),
                          _check("feat-bp-enabled", fa.get("bandpass_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Highpass (Hz)", style=LABEL_STYLE),
                          _input("feat-bp-hp", fa.get("bandpass_highpass_hz", 1.0), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lowpass (Hz)", style=LABEL_STYLE),
                          _input("feat-bp-lp", fa.get("bandpass_lowpass_hz", 100.0), step=10)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch filter", style=LABEL_STYLE),
                          _check("feat-notch-enabled", fa.get("notch_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Notch freq (Hz)", style=LABEL_STYLE),
                          _input("feat-notch-freq", fa.get("notch_frequency_hz", 60))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 3: Smoothing + Baseline
            html.Div([
                html.Div([html.Label("Moving average smoothing", style=LABEL_STYLE),
                          _check("feat-smooth-enabled", fa.get("smoothing_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Smoothing window (ms)", style=LABEL_STYLE),
                          _input("feat-smooth-win", fa.get("smoothing_window_ms", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Baseline correction", style=LABEL_STYLE),
                          _check("feat-baseline", fa.get("baseline_correction", True))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # Row 4: Artifact Exclusion
            html.H5("Artifact Exclusion", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Enable artifact exclusion", style=LABEL_STYLE),
                          _check("feat-art-enabled", fa.get("artifact_exclusion_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(id="feat-art-method",
                                       options=[{"label": m, "value": m} for m in
                                                ["fixed", "mad", "template", "rawamp", "noise"]],
                                       value=fa.get("artifact_method", "fixed"),
                                       style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Threshold / k-value", style=LABEL_STYLE),
                          _input("feat-art-thresh", fa.get("artifact_threshold", 500.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)", style=LABEL_STYLE),
                          _input("feat-art-merge", fa.get("artifact_merge_gap_sec", 2.0), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 5: Template-specific params
            html.Div([
                html.Div([html.Label("Template source", style=LABEL_STYLE),
                          dcc.Dropdown(id="feat-tmpl-source",
                                       options=[{"label": "Early 50ms", "value": "early50ms"},
                                                {"label": "Grand Mean", "value": "grandMean"}],
                                       value=fa.get("template_source", "early50ms"),
                                       style=DROPDOWN_STYLE)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Upper band (r)", style=LABEL_STYLE),
                          _input("feat-tmpl-upper", fa.get("template_upper_r", 0.7), step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Lower band (r)", style=LABEL_STYLE),
                          _input("feat-tmpl-lower", fa.get("template_lower_r", 0.3), step=0.05)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Raw amp multiplier (k)", style=LABEL_STYLE),
                          _input("feat-rawamp-k", fa.get("rawamp_multiplier", 4.0), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),

            # Row 6: Ictal rescue
            html.Div([
                html.Div([html.Label("Ictal spike rescue", style=LABEL_STYLE),
                          _check("feat-ictal-rescue", fa.get("ictal_rescue_enabled", False))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Rescue window (ms)", style=LABEL_STYLE),
                          _input("feat-ictal-win", fa.get("ictal_rescue_window_ms", 500))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # Row 7: Feature sub-windows
            html.H5("Feature Sub-Windows", style={"color": "#aaa", "marginBottom": "4px"}),
            html.Div([
                html.Div([html.Label("Analysis start (ms)", style=LABEL_STYLE),
                          _input("feat-analysis-start", fa.get("analysis_start_ms", 5))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Analysis end (ms)", style=LABEL_STYLE),
                          _input("feat-analysis-end", fa.get("analysis_end_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area start (ms)", style=LABEL_STYLE),
                          _input("feat-early-start", fa.get("early_area_start_ms", 0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Early area end (ms)", style=LABEL_STYLE),
                          _input("feat-early-end", fa.get("early_area_end_ms", 50))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Late area start (ms)", style=LABEL_STYLE),
                          _input("feat-late-start", fa.get("late_area_start_ms", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Late area end (ms)", style=LABEL_STYLE),
                          _input("feat-late-end", fa.get("late_area_end_ms", 200))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style={**SECTION_STYLE, "borderLeft": "3px solid #00CC96"}),

        # --- Criticality ---
        html.Div([
            html.H4("Criticality", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("AR order", style=LABEL_STYLE),
                          _input("crit-ar-order", cr.get("ar_order", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Window (sec)", style=LABEL_STYLE),
                          _input("crit-window-sec", cr.get("window_sec", 2.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Overlap %", style=LABEL_STYLE),
                          dcc.Slider(id="crit-overlap", min=0, max=90, step=10,
                                     value=cr.get("overlap_pct", 50),
                                     marks={i: str(i) for i in range(0, 91, 10)},
                                     tooltip={"placement": "bottom"})],
                         style={**FIELD_STYLE, "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Fit method", style=LABEL_STYLE),
                          dcc.Dropdown(id="crit-fit-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["YuleWalker", "Burg", "Covariance"]],
                                       value=cr.get("fit_method", "YuleWalker"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Target SR", style=LABEL_STYLE),
                          _input("crit-target-sr", cr.get("target_sampling_rate", 1000))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Type B", style=LABEL_STYLE),
                          _input("crit-type-b", cr.get("criticality_type_b", 2), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Error bars", style=LABEL_STYLE),
                          _check("crit-error-bars", cr.get("calculate_error_bars", False))],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Seizure ---
        html.Div([
            html.H4("Seizure Detection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Spike threshold (uV)", style=LABEL_STYLE),
                          _input("seiz-spike-thresh", sz.get("spike_threshold_uv", 10))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike min width", style=LABEL_STYLE),
                          _input("seiz-spike-min-w", sz.get("spike_min_width", 5), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Spike max width", style=LABEL_STYLE),
                          _input("seiz-spike-max-w", sz.get("spike_max_width", 50), step=1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min seizure dur (sec)", style=LABEL_STYLE),
                          _input("seiz-min-dur", sz.get("min_seizure_duration_sec", 5), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Label("Event glue (sec)", style=LABEL_STYLE),
                          _input("seiz-glue-sec", sz.get("event_glue_sec", 2), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Min spikes/sec", style=LABEL_STYLE),
                          _input("seiz-min-spikes", sz.get("min_spikes_per_sec", 2), step=0.5)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Outlier factor", style=LABEL_STYLE),
                          _input("seiz-outlier", sz.get("outlier_factor", 3), step=0.5)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Artifact ---
        html.Div([
            html.H4("Artifact Rejection", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Method", style=LABEL_STYLE),
                          dcc.Dropdown(id="art-method",
                                       options=[{"label": m, "value": m}
                                                for m in ["fixed", "mad"]],
                                       value=ar.get("method", "fixed"),
                                       style={"backgroundColor": "#111", "color": "white"})],
                         style=FIELD_STYLE),
                html.Div([html.Label("Fixed threshold", style=LABEL_STYLE),
                          _input("art-fixed-thresh", ar.get("fixed_threshold", 500))],
                         style=FIELD_STYLE),
                html.Div([html.Label("MAD k", style=LABEL_STYLE),
                          _input("art-mad-k", ar.get("mad_k", 4.0), step=0.1)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Merge gap (sec)", style=LABEL_STYLE),
                          _input("art-merge-gap", ar.get("merge_gap_sec", 2.0), step=0.1)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- QC Thresholds ---
        html.Div([
            html.H4("QC Thresholds", style={"color": "white", "marginTop": "0"}),
            html.Div([
                html.Div([html.Label("Artifact % warning", style=LABEL_STYLE),
                          _input("qc-artifact-warn", qc.get("artifact_pct_warning", 50))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Flatline std", style=LABEL_STYLE),
                          _input("qc-flatline-std", qc.get("flatline_std", 1e-6), step=1e-7)],
                         style=FIELD_STYLE),
                html.Div([html.Label("Clipping voltage", style=LABEL_STYLE),
                          _input("qc-clipping-v", qc.get("clipping_voltage", 10.0))],
                         style=FIELD_STYLE),
                html.Div([html.Label("Line noise ratio warning", style=LABEL_STYLE),
                          _input("qc-linenoise-warn", qc.get("line_noise_ratio_warning", 0.2), step=0.01)],
                         style=FIELD_STYLE),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=SECTION_STYLE),

        # --- Features ---
        html.Div([
            html.H4("Features", style={"color": "white", "marginTop": "0"}),
            dcc.Checklist(
                id="features-checklist",
                options=[{"label": f"  {f}", "value": f} for f in all_features],
                value=ft.get("enabled", all_features),
                style={"color": "#ddd", "columns": "3", "columnGap": "20px"},
                inputStyle={"marginRight": "6px"},
            ),
        ], style=SECTION_STYLE),

        # --- Buttons ---
        html.Div([
            html.Button("Save Settings", id="btn-save-settings", n_clicks=0,
                         style={"backgroundColor": "#636EFA", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            dcc.Input(id="version-label-input", placeholder="Version label (optional)",
                      style={**INPUT_STYLE, "width": "250px"}),
            html.Button("Save as New Version", id="btn-save-version", n_clicks=0,
                         style={"backgroundColor": "#00CC96", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(style={"flex": "0 0 20px"}),
            html.Button("Reprocess Session", id="btn-reprocess", n_clicks=0,
                         style={"backgroundColor": "#FFA15A", "color": "white", "border": "none",
                                "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                                "fontSize": "14px", "fontWeight": "bold"}),
        ], style={"display": "flex", "gap": "10px", "alignItems": "center",
                  "flexWrap": "wrap", "marginBottom": "12px"}),

        html.Div(id="settings-save-status"),
        html.Div(id="settings-version-status"),
        html.Div(id="reprocess-status"),

        # --- Version history ---
        html.Div([
            html.H4("Version History", style={"color": "white", "marginTop": "24px"}),
            dash_table.DataTable(
                id="version-history-table",
                data=version_data,
                columns=[
                    {"name": "ID", "id": "id"},
                    {"name": "Label", "id": "label"},
                    {"name": "Created", "id": "created_at"},
                    {"name": "Active", "id": "is_active"},
                    {"name": "Hash (short)", "id": "hash_short"},
                ],
                **DARK_TABLE_STYLE,
                style_data_conditional=[ZEBRA_STRIPE,
                    {"if": {"filter_query": "{is_active} = Yes"},
                     "backgroundColor": "#112211", "color": "#00CC96"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ], style=SECTION_STYLE),
    ])


# ------------------------------------------------------------------ #
#  Activity Log tab (NEW)
# ------------------------------------------------------------------ #

def _activity_log_tab_layout(store: Store):
    # Pre-load initial data
    try:
        initial_logs = store.get_activity_log(hours=24, limit=500)
    except Exception:
        initial_logs = []

    initial_data = [
        {
            "timestamp": entry.get("timestamp", "")[:19],
            "level": entry.get("level", ""),
            "action": entry.get("action", ""),
            "message": (entry.get("message") or "")[:200],
            "file_path": os.path.basename(entry.get("file_path") or ""),
            "duration": f"{entry['duration_sec']:.2f}" if entry.get("duration_sec") else "",
        }
        for entry in initial_logs
    ]

    return html.Div([
        html.H3("Activity Log", style={"color": "white", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Label("Time Range", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-hours-dropdown",
                    options=TIME_RANGE_OPTIONS,
                    value=24,
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 180px"}),
            html.Div([
                html.Label("Level", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="activity-log-level-dropdown",
                    options=[
                        {"label": "ALL", "value": "ALL"},
                        {"label": "INFO", "value": "INFO"},
                        {"label": "WARNING", "value": "WARNING"},
                        {"label": "ERROR", "value": "ERROR"},
                    ],
                    value="ALL",
                    style=DROPDOWN_STYLE,
                    className="dark-dropdown",
                ),
            ], style={"flex": "0 0 150px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "flexWrap": "wrap"}),

        dash_table.DataTable(
            id="activity-log-table",
            data=initial_data,
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Level", "id": "level"},
                {"name": "Action", "id": "action"},
                {"name": "Message", "id": "message"},
                {"name": "File", "id": "file_path"},
                {"name": "Duration (s)", "id": "duration"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{level} = ERROR"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{level} = WARNING"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{level} = INFO"},
                 "color": "#6bb5ff"},
            ],
            page_size=30,
            filter_action="native",
            sort_action="native",
        ),
    ])


# ------------------------------------------------------------------ #
#  Annotations tab (NEW)
# ------------------------------------------------------------------ #

def _annotations_tab_layout(store: Store):
    session_options = _session_dropdown_options(store)

    try:
        all_annotations = store.get_annotations()
    except Exception:
        all_annotations = []

    table_data = _annotations_table_data(all_annotations)

    return html.Div([
        html.H3("Annotations", style={"color": "white", "marginBottom": "12px"}),

        # Existing annotations table
        dash_table.DataTable(
            id="annotation-table",
            data=table_data,
            columns=[
                {"name": "ID", "id": "id"},
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Category", "id": "category"},
                {"name": "Note", "id": "note"},
                {"name": "Session", "id": "session_dir"},
                {"name": "User", "id": "user"},
                {"name": "Created At", "id": "created_at"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{category} = electrode"},
                 "color": "#FFA15A"},
                {"if": {"filter_query": "{category} = injection"},
                 "color": "#AB63FA"},
                {"if": {"filter_query": "{category} = experiment"},
                 "color": "#636EFA"},
                {"if": {"filter_query": "{category} = observation"},
                 "color": "#00CC96"},
            ],
            page_size=20,
            filter_action="native",
            sort_action="native",
        ),

        # Add Note form
        html.Div([
            html.H4("Add Note", style={"color": "white", "marginTop": "24px", "marginBottom": "12px"}),
            html.Div([
                html.Div([
                    html.Label("Timestamp", style=LABEL_STYLE),
                    dcc.Input(
                        id="annotation-timestamp",
                        value=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                        type="text",
                        style=INPUT_STYLE,
                        placeholder="YYYY-MM-DDTHH:MM:SS",
                    ),
                ], style={"flex": "1", "minWidth": "220px"}),
                html.Div([
                    html.Label("Category", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-category",
                        options=[
                            {"label": "Electrode", "value": "electrode"},
                            {"label": "Injection", "value": "injection"},
                            {"label": "Experiment", "value": "experiment"},
                            {"label": "Observation", "value": "observation"},
                        ],
                        value="observation",
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "0 0 180px"}),
                html.Div([
                    html.Label("Session (optional)", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="annotation-session-dropdown",
                        options=session_options,
                        style=DROPDOWN_STYLE,
                        className="dark-dropdown",
                    ),
                ], style={"flex": "1", "minWidth": "250px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "12px", "flexWrap": "wrap"}),

            html.Div([
                html.Label("Note", style=LABEL_STYLE),
                dcc.Textarea(
                    id="annotation-note",
                    style={**INPUT_STYLE, "height": "80px", "resize": "vertical"},
                    placeholder="Enter your annotation...",
                ),
            ], style={"marginBottom": "12px"}),

            html.Button("Submit Annotation", id="annotation-submit-btn", n_clicks=0,
                        style={"backgroundColor": "#00CC96", "color": "white",
                               "border": "none", "padding": "10px 24px",
                               "borderRadius": "6px", "cursor": "pointer",
                               "fontSize": "14px", "fontWeight": "bold"}),
            html.Div(id="annotation-status", style={"marginTop": "8px"}),
        ], style=SECTION_STYLE),
    ])


def _annotations_table_data(annotations: list[dict]) -> list[dict]:
    return [
        {
            "id": a.get("id", ""),
            "timestamp": (a.get("timestamp") or "")[:19],
            "category": a.get("category", ""),
            "note": a.get("note", ""),
            "session_dir": os.path.basename(a.get("session_dir") or ""),
            "user": a.get("user_email", "") or "",
            "created_at": (a.get("created_at") or "")[:19],
        }
        for a in annotations
    ]


# ------------------------------------------------------------------ #
#  Alerts tab (existing)
# ------------------------------------------------------------------ #

def _alerts_tab(store: Store):
    alerts = store.get_recent_alerts(hours=168)  # 7 days
    return html.Div([
        html.H3(f"Alert History (last 7 days) -- {len(alerts)} alerts",
                 style={"color": "white", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=[{"time": a["sent_at"][:19], "severity": a["severity"],
                   "type": a["alert_type"], "message": a["message"],
                   "session": a.get("session_dir", "")}
                  for a in alerts],
            columns=[{"name": c, "id": c}
                     for c in ["time", "severity", "type", "message", "session"]],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{severity} = critical"},
                 "backgroundColor": "#3d1111", "color": "#ff6b6b"},
                {"if": {"filter_query": "{severity} = warning"},
                 "backgroundColor": "#3d3011", "color": "#ffd93d"},
                {"if": {"filter_query": "{severity} = info"},
                 "backgroundColor": "#112233", "color": "#6bb5ff"},
            ],
            page_size=25,
            filter_action="native",
            sort_action="native",
        ) if alerts else html.P("No alerts in the last 7 days", style={"color": "#888"}),
    ])


# ------------------------------------------------------------------ #
#  Sessions tab (existing)
# ------------------------------------------------------------------ #

def _sessions_tab(store: Store):
    sessions = store.get_sessions()
    if not sessions:
        return _empty_state(
            "No sessions yet",
            "Sessions appear here once the watcher discovers .mat files in "
            "the network share configured under watch.paths.",
        )

    all_configs = store.get_all_session_configs()
    config_by_dir = {c["session_dir"]: c for c in all_configs}

    rows = []
    for s in sessions:
        cfg = config_by_dir.get(s["session_dir"], {})
        ch_names = _parse_json_field(cfg.get("channel_names"))
        num_ch = cfg.get("num_channels") or (len(ch_names) if ch_names else "")
        sr = cfg.get("sampling_rate", "")
        stim_freq = cfg.get("stim_frequency_hz", "")
        stim_charge = cfg.get("stim_charge_nC", "")

        rows.append({
            "name": s["session_name"],
            "dir": s["session_dir"],
            "files": s["num_files"],
            "processed": s["processed"],
            "errors": s["errors"],
            "first": s["first_chunk"][:16] if s["first_chunk"] else "",
            "last": s["last_chunk"][:16] if s["last_chunk"] else "",
            "channels": num_ch,
            "sr": sr,
            "stim_freq": f"{stim_freq}" if stim_freq else "",
            "stim_charge": f"{stim_charge}" if stim_charge else "",
        })

    return html.Div([
        html.Div([
            html.H3(f"Sessions ({len(sessions)})",
                    style={"color": COLOR_TEXT_PRIMARY,
                           "fontSize": FONT_SIZE_HEADER,
                           "fontWeight": "600",
                           "margin": "0"}),
            html.Span(
                "Click any row to open that session in Evoked Waveforms.",
                style={"color": COLOR_TEXT_TERTIARY,
                       "fontSize": FONT_SIZE_CAPTION,
                       "marginLeft": SPACE_3},
            ),
        ], style={"display": "flex", "alignItems": "baseline",
                  "marginBottom": SPACE_4}),
        dash_table.DataTable(
            id="sessions-table",
            data=rows,
            columns=[
                {"name": "Session", "id": "name"},
                {"name": "Directory", "id": "dir"},
                {"name": "Files", "id": "files"},
                {"name": "Processed", "id": "processed"},
                {"name": "Errors", "id": "errors"},
                {"name": "First Chunk", "id": "first"},
                {"name": "Last Chunk", "id": "last"},
                {"name": "Channels", "id": "channels"},
                {"name": "SR (Hz)", "id": "sr"},
                {"name": "Stim Freq", "id": "stim_freq"},
                {"name": "Stim Charge (nC)", "id": "stim_charge"},
            ],
            **DARK_TABLE_STYLE,
            style_data_conditional=[ZEBRA_STRIPE,
                {"if": {"filter_query": "{errors} > 0"},
                 "backgroundColor": "rgba(255,69,58,0.10)",
                 "color": COLOR_DANGER},
                {"if": {"state": "active"},
                 "backgroundColor": "rgba(94,124,226,0.18)",
                 "border": f"1px solid {COLOR_ACCENT}"},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "name"},
                 "cursor": "pointer", "fontWeight": "600",
                 "color": COLOR_ACCENT},
            ],
            page_size=20,
            sort_action="native",
            filter_action="native",
            cell_selectable=True,
            active_cell=None,
        ),
    ])


# ------------------------------------------------------------------ #
#  Version table helper
# ------------------------------------------------------------------ #

def _version_table_data(versions: list[dict]) -> list[dict]:
    return [
        {
            "id": v["id"],
            "label": v.get("label", ""),
            "created_at": v["created_at"][:19],
            "is_active": "Yes" if v.get("is_active") else "No",
            "hash_short": v.get("version_hash", "")[:12],
        }
        for v in versions
    ]
