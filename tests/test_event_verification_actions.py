"""Guard the approve/flag handler against pattern-button recreation.

Dash recreates the per-row approve/flag buttons every time the pending
list re-renders, which re-fires the ALL-pattern callback with
n_clicks=None. Before the guard, _on_action treated that as a click and
silently approved files (it auto-approved 181 files in one incident).
The recreation path is a Dash-runtime behaviour, so we test the guard
predicate (_has_real_click) directly.

Run with: pytest tests/test_event_verification_actions.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs.event_verification import (  # noqa: E402
    _has_real_click,
)


def test_recreation_with_null_nclicks_is_not_a_click():
    # Buttons recreated by a list re-render report n_clicks=None.
    triggered = [
        {"prop_id": '{"file_id":1,"type":"evtv-approve-one-btn"}.n_clicks',
         "value": None},
        {"prop_id": '{"file_id":2,"type":"evtv-approve-one-btn"}.n_clicks',
         "value": None},
    ]
    assert _has_real_click(triggered) is False


def test_zero_nclicks_is_not_a_click():
    triggered = [{"prop_id": "x.n_clicks", "value": 0}]
    assert _has_real_click(triggered) is False


def test_empty_triggered_is_not_a_click():
    assert _has_real_click([]) is False
    assert _has_real_click(None) is False


def test_real_click_has_truthy_nclicks():
    triggered = [
        {"prop_id": '{"file_id":7,"type":"evtv-approve-one-btn"}.n_clicks',
         "value": 1},
    ]
    assert _has_real_click(triggered) is True


def test_bulk_button_click_counts():
    # The bulk approve button's n_clicks increments and persists.
    triggered = [{"prop_id": "evtv-approve-sel-btn.n_clicks", "value": 5}]
    assert _has_real_click(triggered) is True


def test_mixed_recreation_plus_one_real_click_counts():
    # If a real click and recreated buttons land together, the real
    # click still wins (truthy value present).
    triggered = [
        {"prop_id": "a.n_clicks", "value": None},
        {"prop_id": "b.n_clicks", "value": 1},
        {"prop_id": "c.n_clicks", "value": None},
    ]
    assert _has_real_click(triggered) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
