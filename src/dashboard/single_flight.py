"""Single-flight guard with a stall watchdog.

A non-blocking lock that ALSO records when the current holder started and logs
loudly when a later caller finds it held past a threshold -- so a build that
hangs (e.g. a wedged network read) can't freeze a card silently forever.

Replaces the bare ``threading.Lock()`` + ``acquire(blocking=False)`` idiom the
Overview fill callbacks used, which recorded nothing and surfaced nothing: a
wedged build returned ``no_update`` on every subsequent tick with zero errors
anywhere. This primitive keeps the single-flight behaviour (one builder at a
time, others don't pile on) but makes a wedge observable (``held_for`` /
``is_stalled``) and audible (``warn_if_stalled`` emits one loud error per
window).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("qc_monitor.dashboard.single_flight")

# A build in flight longer than this is treated as "stalled" (wedged). Sized well
# past the DB busy-timeout (30 s) and the Sheets fetch timeout (10 s) so it only
# fires on a genuine hang, not ordinary slowness.
STALL_SEC = 90.0
# Throttle the loud "wedged" log to at most once per this window per guard.
_WARN_EVERY = 60.0


class SingleFlight:
    """One-builder-at-a-time guard that tracks hold time and warns on stalls.

    Usage::

        if not guard.try_begin():
            guard.warn_if_stalled()          # loud log if a prior build wedged
            return no_update                 # someone else is building
        try:
            ... build ...
        finally:
            guard.end()
    """

    __slots__ = ("label", "_lock", "_started_at", "_last_warned")

    def __init__(self, label: str):
        assert label, "label required"
        self.label = label
        self._lock = threading.Lock()
        self._started_at = 0.0      # time.time() while a build is in flight, else 0
        self._last_warned = 0.0

    def try_begin(self) -> bool:
        """Become the single builder if free. True -> acquired (caller MUST call
        ``end()`` in a finally); False -> a build is already in flight."""
        got = self._lock.acquire(blocking=False)
        if got:
            self._started_at = time.time()
        return got

    def end(self) -> None:
        """Release the guard (call in a finally after a successful try_begin)."""
        self._started_at = 0.0
        try:
            self._lock.release()
        except RuntimeError:            # not held -- defensive; never raise here
            pass

    def held_for(self) -> float:
        """Seconds the current build has been in flight (0.0 if idle)."""
        t = self._started_at
        return (time.time() - t) if t else 0.0

    def is_stalled(self, threshold: float = STALL_SEC) -> bool:
        return self.held_for() > threshold

    def warn_if_stalled(self, threshold: float = STALL_SEC,
                        every: float = _WARN_EVERY) -> bool:
        """If the in-flight build has exceeded *threshold*, emit ONE loud error
        per *every* seconds. Returns True when currently stalled."""
        held = self.held_for()
        if held <= threshold:
            return False
        now = time.time()
        if now - self._last_warned >= every:
            self._last_warned = now
            logger.error(
                "card build '%s' has been in flight for %.0fs -- likely wedged "
                "(a background read may be hung); refresh ticks are being "
                "skipped until it returns", self.label, held)
        return True
