"""Shared store-aware helpers for dashboard tabs.

Helpers that wrap ``Store`` calls or do JSON-field unwrapping for
multiple tabs live here instead of being duplicated across each tab
module. The functions themselves came out of ``src/dashboard/app.py``
where they were used by 8+ render paths.

Anything UI-shaped (Dash components, dash_table styles) belongs in
``src/dashboard/components.py``; anything purely about colors/typography
belongs in ``src/dashboard/design.py``. This module is for plumbing
that *reads from the store* on behalf of a tab.
"""

from __future__ import annotations

import json

from src.dashboard.design import ROLE_COLORS
from src.db.store import Store


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
