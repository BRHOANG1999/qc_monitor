"""SingleFlight watchdog + _CachedBuilder freshness/stall behaviour.

The point of these guards is that a wedged build is OBSERVABLE (held_for /
is_stalled), AUDIBLE (warn_if_stalled logs once per window), and never silently
overlaps itself. Run with: pytest tests/test_single_flight.py -q
"""

from __future__ import annotations

import logging
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dash import no_update  # noqa: E402

from src.dashboard.single_flight import SingleFlight  # noqa: E402


# --------------------------------------------------------------------------- #
#  SingleFlight
# --------------------------------------------------------------------------- #

def test_single_flight_one_builder_at_a_time():
    sf = SingleFlight("t")
    assert sf.try_begin() is True          # first caller wins
    assert sf.try_begin() is False         # second caller is turned away
    assert sf.held_for() > 0.0
    sf.end()
    assert sf.held_for() == 0.0
    assert sf.try_begin() is True          # free again after end()
    sf.end()


def test_is_stalled_uses_hold_time():
    sf = SingleFlight("t")
    assert sf.is_stalled(0.05) is False    # idle -> never stalled
    sf.try_begin()
    time.sleep(0.08)
    assert sf.is_stalled(0.05) is True
    sf.end()


def test_warn_if_stalled_logs_once_per_window(caplog):
    sf = SingleFlight("wedged-card")
    sf.try_begin()
    time.sleep(0.08)
    with caplog.at_level(logging.ERROR):
        first = sf.warn_if_stalled(threshold=0.05, every=60.0)
        second = sf.warn_if_stalled(threshold=0.05, every=60.0)
    sf.end()
    assert first is True and second is True
    # Loud, but throttled: exactly one ERROR despite two stalled calls.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR
              and "wedged-card" in r.getMessage()]
    assert len(errors) == 1


def test_warn_not_emitted_when_healthy(caplog):
    sf = SingleFlight("healthy")
    sf.try_begin()
    with caplog.at_level(logging.ERROR):
        assert sf.warn_if_stalled(threshold=60.0) is False   # young build
    sf.end()
    assert not [r for r in caplog.records if "healthy" in r.getMessage()]


# --------------------------------------------------------------------------- #
#  _CachedBuilder
# --------------------------------------------------------------------------- #

def _cache(ttl):
    from src.dashboard.tabs.overview import _CachedBuilder
    return _CachedBuilder(ttl, "unit")


def test_cached_builder_builds_once_then_serves_cache():
    cache = _cache(60.0)
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return f"card-{calls['n']}"

    assert cache.built_at() is None and cache.age() is None
    assert cache.get(build) == "card-1"
    assert cache.get(build) == "card-1"        # within TTL -> no rebuild
    assert calls["n"] == 1
    assert cache.built_at() is not None
    assert cache.age() is not None and cache.age() < 60.0
    assert cache.inflight_for() == 0.0         # nothing building now


def test_cached_builder_serves_stale_while_a_rebuild_is_in_flight():
    cache = _cache(0.0)                         # everything is immediately stale
    assert cache.get(lambda: "good") == "good"
    # Simulate another thread mid-rebuild by holding the guard here (the Lock is
    # non-reentrant, so get()'s own try_begin will fail just like a real caller).
    assert cache._sf.try_begin() is True
    try:
        # Stale value is served rather than recomputed or blanked.
        assert cache.get(lambda: "SHOULD-NOT-RUN") == "good"
    finally:
        cache._sf.end()


def test_cached_builder_cold_and_in_flight_returns_no_update():
    cache = _cache(0.0)
    assert cache._sf.try_begin() is True       # in flight, nothing cached yet
    try:
        assert cache.get(lambda: "SHOULD-NOT-RUN") is no_update
    finally:
        cache._sf.end()
