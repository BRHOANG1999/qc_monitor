"""Resolve the evoked_figures config: paths, roll window, primary-channel
overrides, and the DECLARED (expected) metric set.

The declared set is fixed across animals — the cheap 22 columns by default,
plus the 6 expensive fit-columns when ``include_expensive`` is on. Rendering the
declared set (not whatever incidental columns a given sidecar happens to hold)
keeps the figure set identical per animal, so a missing metric is a real
coverage signal rather than silent variation, and stray columns don't churn.
"""

from __future__ import annotations

import os

from src.utils import evoked_features as ef
from src.utils.evoked_output import DEFAULT_EVOKED_DIR

_DEFAULT_OUT = os.path.join("data", "derivatives", "evoked_figures")


def _cfg(config: dict) -> dict:
    return (config or {}).get("evoked_figures", {}) or {}


def evoked_dir(config: dict) -> str:
    ce = (config or {}).get("chronic_evoked", {}) or {}
    return ce.get("evoked_output_dir") or DEFAULT_EVOKED_DIR


def out_root(config: dict) -> str:
    return _cfg(config).get("out_root") or _DEFAULT_OUT


def roll_window(config: dict) -> int:
    try:
        return max(11, int(_cfg(config).get("roll_window", 301)))
    except (TypeError, ValueError):
        return 301


def gap_break_sec(config: dict) -> float:
    """Break the percentile bands/median across gaps longer than this, so a
    stretch with no recordings shows as a gap instead of an interpolated
    trend. 0 disables. Default 24 h."""
    try:
        return max(0.0, float(_cfg(config).get("gap_break_hours", 24.0))) * 3600.0
    except (TypeError, ValueError):
        return 24 * 3600.0


def primary_overrides(config: dict) -> dict:
    return dict(_cfg(config).get("primary_channel", {}) or {})


def enabled(config: dict) -> bool:
    return bool(_cfg(config).get("enabled", False))


def declared_metrics(config: dict) -> list[str]:
    """The stable expected metric set: cheap 22 (+6 expensive if enabled),
    optionally narrowed to an explicit ``metrics`` list. Order from ALL_COLUMNS."""
    c = _cfg(config)
    base = list(ef.CHEAP_COLUMNS)
    if c.get("include_expensive"):
        base += list(ef.EXPENSIVE_COLUMNS)
    want = c.get("metrics", ["*"])
    if want and want != ["*"] and "*" not in want:
        wset = set(want)
        base = [m for m in base if m in wset]
    return base
