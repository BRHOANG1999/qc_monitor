"""Shared helpers for dashboard tabs.

Helpers used by more than one tab (store-aware accessors, JSON
unwrapping, the canonical "empty figure" plotly builder, the
time-range dropdown options) live here instead of being duplicated
across each tab module. The functions themselves came out of
``src/dashboard/app.py`` where they were used by 8+ render paths.

Anything UI-shaped (Dash components, dash_table + form-control
styles) belongs in ``src/dashboard/components.py``; anything purely
about colors / typography belongs in ``src/dashboard/design.py``.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go

from src.dashboard.design import (
    COLOR_SURFACE_1, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    ROLE_COLORS,
)
from src.db.store import Store


# Standard time-range dropdown options used across tabs (Criticality,
# Evoked, Activity log, ...). Value is hours; 0 means "all time".
TIME_RANGE_OPTIONS: list[dict] = [
    {"label": "Last 24h", "value": 24},
    {"label": "Last 48h", "value": 48},
    {"label": "Last 1 week", "value": 168},
    {"label": "Last 1 month", "value": 720},
    {"label": "All time", "value": 0},
]


def empty_fig(text: str = "Nothing to show yet",
              hint: str | None = None,
              height: int = 400) -> go.Figure:
    """Friendly empty figure used in place of a plot when there's no
    data. *text* is the headline; *hint* is an optional one-liner
    rendered below the headline in a quieter color. Apple HIG: empty
    states should be informative, not silent."""
    fig = go.Figure()
    annotations = [dict(
        text=f"<b>{text}</b>", showarrow=False,
        xref="paper", yref="paper",
        x=0.5, y=0.55,
        font=dict(size=14, color=COLOR_TEXT_SECONDARY),
    )]
    if hint:
        annotations.append(dict(
            text=hint, showarrow=False, xref="paper", yref="paper",
            x=0.5, y=0.42,
            font=dict(size=11, color=COLOR_TEXT_TERTIARY),
        ))
    fig.update_layout(
        height=height,
        plot_bgcolor=COLOR_SURFACE_1, paper_bgcolor=COLOR_SURFACE_1,
        annotations=annotations,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def parse_json_field(val):
    """Parse a JSON string from the DB or return the value as-is.

    session_config columns like ``channel_names`` / ``eeg_channels`` are
    serialised as JSON text by ``Store.upsert_session_config`` so the
    DataFrame-style fetch returns strings the tabs need to unwrap
    before iterating. Returns the original value when parsing fails
    so a malformed cell never crashes the render path.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def channel_map(store: Store, session_dir: str) -> dict:
    """Return ``{channel_index: {"name": str, "role": str}}`` for one
    session. Role is one of ``eeg`` / ``stim_copy`` / ``reference``;
    channels not flagged in the session config default to ``eeg``.

    Empty dict when the session has no recorded config yet -- callers
    fall back to ``f"Ch{i}"`` naming in that case.
    """
    cfg = store.get_session_config(session_dir)
    if not cfg:
        return {}
    names = parse_json_field(cfg.get("channel_names")) or []
    eeg_chs = parse_json_field(cfg.get("eeg_channels")) or []
    stim_chs = parse_json_field(cfg.get("stim_copy_channels")) or []
    ref_chs = parse_json_field(cfg.get("reference_channels")) or []
    # eeg_chs is read for parity with the existing helper but unused --
    # the default role *is* "eeg", so an explicit membership check is
    # only needed for the non-eeg roles. Left in the unwrap so the
    # caller can adopt the same pattern when a fourth role appears.
    _ = eeg_chs
    out: dict[int, dict] = {}
    for i, name in enumerate(names):
        role = "eeg"
        if i in stim_chs:
            role = "stim_copy"
        elif i in ref_chs:
            role = "reference"
        out[i] = {"name": name, "role": role}
    return out


def color_for_role(role: str) -> str:
    """Look up the design-token color for a channel role; defaults to
    the Plotly indigo so an unknown role still renders."""
    return ROLE_COLORS.get(role, "#636EFA")
