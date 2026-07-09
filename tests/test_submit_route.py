"""Phase 1c: single-Submit state routing (video_events.submit_route).

The confirmed rule: EO + Racine is the minimum to submit an event; fully
complete events route to PI review, scored-but-incomplete to 'Needs more
onsets', and an event missing EO or Racine blocks the submit.

Run with: pytest tests/test_submit_route.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs.video_events import (  # noqa: E402
    blank_event, submit_route, event_has_minimum_to_submit,
    event_is_meaningful, is_event_complete)


def _complete_lvf(**over):
    """A fully-complete LVF event (type + all 5 landmarks + Racine)."""
    e = blank_event()
    e.update(type="LVF", EO_sec=1.0, LAS_sec=2.0, BO_sec=3.0,
             PID_sec=4.0, BB_sec=5.0, racine=4)
    e.update(over)
    return e


def test_no_meaningful_events_routes_no_events():
    assert submit_route([]) == ("no_events", [])
    assert submit_route([blank_event()]) == ("no_events", [])
    assert submit_route([blank_event(), blank_event()]) == ("no_events", [])


def test_complete_event_routes_to_pi():
    assert is_event_complete(_complete_lvf())
    assert submit_route([_complete_lvf()]) == ("pending_pi_review", [])
    assert submit_route([_complete_lvf(), _complete_lvf()]) \
        == ("pending_pi_review", [])


def test_scored_but_missing_landmark_routes_needs_scoring():
    # EO + Racine present, but a secondary landmark missing -> Needs more onsets
    assert submit_route([_complete_lvf(BB_sec=None)]) == ("needs_scoring", [])
    assert submit_route([_complete_lvf(LAS_sec=None, PID_sec=None)]) \
        == ("needs_scoring", [])


def test_missing_eo_or_racine_blocks():
    assert submit_route([_complete_lvf(EO_sec=None)]) == (None, [1])
    assert submit_route([_complete_lvf(racine=None)]) == (None, [1])
    # blocking index is the 1-based position in the full list
    assert submit_route(
        [_complete_lvf(), _complete_lvf(EO_sec=None)]) == (None, [2])


def test_block_takes_priority_over_partial():
    # One event fine-but-partial, one missing EO -> the whole submit blocks.
    r = submit_route([_complete_lvf(BB_sec=None), _complete_lvf(racine=None)])
    assert r == (None, [2])


def test_blank_events_are_ignored_for_routing():
    # A pristine blank alongside a complete event doesn't block or downgrade.
    assert submit_route([blank_event(), _complete_lvf()]) \
        == ("pending_pi_review", [])
    # Blank alongside a partial -> still needs_scoring, blank ignored.
    assert submit_route([blank_event(), _complete_lvf(BB_sec=None)]) \
        == ("needs_scoring", [])


def test_event_has_minimum_predicate():
    assert event_has_minimum_to_submit(_complete_lvf()) is True
    assert event_has_minimum_to_submit(_complete_lvf(EO_sec=None)) is False
    assert event_has_minimum_to_submit(_complete_lvf(racine=None)) is False
    assert event_has_minimum_to_submit(
        {**blank_event(), "EO_sec": 1.0, "racine": 9}) is False  # racine>8
    assert event_is_meaningful(blank_event()) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
