"""Picking a type for one event must not clear another's.

Regression for the ALL-pattern type-radio cross-assignment bug:
selecting a type for event 2 cleared event 1's type because the
callback applied "the last triggered value" to "the first triggered
idx". apply_event_types keys strictly by idx.

Run with: pytest tests/test_event_types.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs.video_events import apply_event_types  # noqa: E402


def _ev(type_=""):
    return {"type": type_, "EO_sec": None, "LAS_sec": None}


def test_setting_event2_type_keeps_event1():
    # Event 1 = LVF; event 2 just got HYP. The ALL callback fires with
    # BOTH radios' current values -- event 1 must stay LVF.
    events = [_ev("LVF"), _ev("")]
    pairs = [(0, "LVF"), (1, "HYP")]
    out, changed = apply_event_types(events, pairs)
    assert changed is True
    assert out[0]["type"] == "LVF"   # untouched sibling
    assert out[1]["type"] == "HYP"


def test_recreation_is_a_noop():
    # Re-render recreates every radio reporting the stored values ->
    # nothing differs -> no change (so the store isn't rewritten and
    # no sibling gets clobbered).
    events = [_ev("LVF"), _ev("HYP")]
    out, changed = apply_event_types(events, [(0, "LVF"), (1, "HYP")])
    assert changed is False
    assert out[0]["type"] == "LVF" and out[1]["type"] == "HYP"


def test_switch_to_hyp_drops_orphan_las():
    events = [{"type": "LVF", "LAS_sec": 4.2}]
    out, changed = apply_event_types(events, [(0, "HYP")])
    assert changed is True
    assert out[0]["type"] == "HYP"
    assert out[0]["LAS_sec"] is None


def test_out_of_range_idx_ignored():
    events = [_ev("LVF")]
    out, changed = apply_event_types(events, [(5, "HYP"), (None, "LVF")])
    assert changed is False
    assert out[0]["type"] == "LVF"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
