"""A hung evokedOutput .mat read must NOT wedge the thumbnail build.

Regression for the "Overview cards load forever" incident: the thumbnail's
network-share fallback read had no timeout and ran under _THUMB_LOCK, so a stuck
share froze the whole page. The read is now deadline-bounded and returns a
visible "source slow" node instead of hanging.

Run with: pytest tests/test_overview_thumbnail_timeout.py -q
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs import overview  # noqa: E402


def test_hung_fallback_read_times_out_to_unavailable_node(monkeypatch):
    # Fallback source hangs far longer than the (shortened) deadline.
    monkeypatch.setattr(overview, "_THUMB_READ_TIMEOUT", 0.3, raising=True)

    def _hang(_config):
        time.sleep(30)
        return [], ""

    monkeypatch.setattr(overview, "_latest_evoked_from_output", _hang,
                        raising=True)

    t0 = time.perf_counter()
    # session_dir="" -> skip the DB read, go straight to the share fallback.
    children = overview._build_overview_thumbnail(None, {}, "", "mean")
    elapsed = time.perf_counter() - t0

    assert elapsed < 3.0, f"thumbnail blocked {elapsed:.1f}s on a hung read"
    assert isinstance(children, list) and children
    text = getattr(children[0], "children", "")
    assert "slow or unreachable" in text


def test_thumb_lock_is_free_after_a_hung_read(monkeypatch):
    # After the deadline fires, the single-flight guard must be releasable/usable
    # again -- a wedge that permanently held the lock was the original bug.
    monkeypatch.setattr(overview, "_THUMB_READ_TIMEOUT", 0.3, raising=True)
    monkeypatch.setattr(overview, "_latest_evoked_from_output",
                        lambda _c: (time.sleep(30), ([], ""))[1], raising=True)

    overview._build_overview_thumbnail(None, {}, "", "mean")
    assert overview._THUMB_LOCK.try_begin() is True
    overview._THUMB_LOCK.end()
