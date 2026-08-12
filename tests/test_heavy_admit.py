"""heavy_admit: the process-wide gate must bound how many heavy builds run at
once, and must publish a 'waiting…' status for the ones that park.

Run with: pytest tests/test_heavy_admit.py -q
"""

from __future__ import annotations

import os
import sys
import threading
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import heavy_admit as ha  # noqa: E402


def test_configure_and_limit():
    ha.configure(2)
    assert ha.limit() == 2
    ha.configure(1)
    assert ha.limit() == 1
    # Floors at 1.
    ha.configure(0)
    assert ha.limit() == 1


def test_admit_bounds_concurrency_and_reports_wait():
    ha.configure(2)
    lock = threading.Lock()
    state = {"live": 0, "peak": 0}
    waited: list = []

    def work(i):
        saw_wait = {"v": False}
        with ha.admit(f"job{i}", progress=lambda _m: saw_wait.__setitem__("v", True)):
            with lock:
                state["live"] += 1
                state["peak"] = max(state["peak"], state["live"])
            time.sleep(0.25)
            with lock:
                state["live"] -= 1
        waited.append(saw_wait["v"])

    ts = [threading.Thread(target=work, args=(i,)) for i in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # Never more than the configured 2 in the body at once.
    assert state["peak"] <= 2, state["peak"]
    # 5 jobs, 2 slots -> exactly 3 had to park and see the waiting status.
    assert sum(waited) == 3, waited


def test_admit_releases_on_exception():
    """A slot is returned even when the body raises, so a failing build never
    permanently consumes a slot."""
    ha.configure(1)
    try:
        with ha.admit("boom"):
            raise ValueError("boom")
    except ValueError:
        pass
    # Slot is free again: a non-blocking acquire inside admit must succeed.
    got = {"v": False}
    with ha.admit("after", progress=lambda _m: got.__setitem__("v", True)):
        pass
    assert got["v"] is False   # did not have to wait -> slot was released


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
