"""Now-viewing banner labels a multi-animal session by the REVIEWED
animal, not the session's first animal.

Run with: pytest tests/test_now_viewing_animal.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.dashboard.tabs.video import (  # noqa: E402
    _animal_for_session, _next_in_pool)


def test_animal_for_session_prefers_reviewed_animal(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    sd = "sess_multi"
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels, discovered_at)
               VALUES (?, ?, ?, '2026-01-01')""",
            (sd,
             json.dumps(["stimCopy", "BCH061SLM", "BCH062SR",
                          "BCH060SR"]),
             json.dumps([1, 2, 3])))
        conn.commit()
    # First animal in the session is BCH061 -- the historical bug.
    assert _animal_for_session(store, sd) == "BCH061"
    # But the reviewer picked BCH062: the banner must show BCH062.
    assert _animal_for_session(store, sd, prefer="BCH062") == "BCH062"
    assert _animal_for_session(store, sd, prefer="BCH060") == "BCH060"
    # Prefer an animal NOT in the session -> fall back to first.
    assert _animal_for_session(store, sd, prefer="BCH999") == "BCH061"
    # No session -> '?'.
    assert _animal_for_session(store, "", prefer="BCH062") == "?"


def test_next_in_pool_advances_within_the_pool(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    with store.connection() as conn:
        for fid in (10, 11, 12):
            conn.execute(
                "INSERT INTO processed_files (id, file_path, "
                "session_dir) VALUES (?, ?, ?)",
                (fid, f"/f/{fid}.mat", f"sess{fid}"))
        conn.commit()
    view = {"pool1": [], "pool2": [10, 11, 12], "pool3": []}

    # Browsing pool2 at file 10 (idx 0) -> Save advances to 11 (idx 1).
    cursor = {"active": "2", "idx": 0}
    sd, fid, new_cursor = _next_in_pool(store, view, cursor, 10)
    assert fid == 11 and new_cursor == {"active": "2", "idx": 1}
    assert sd == "sess11"

    # At the last file -> no next (None), so Save falls back to queue.
    assert _next_in_pool(store, view, {"active": "2", "idx": 2}, 12) \
        is None

    # Loaded file isn't the pool's current entry -> not pool-browsing.
    assert _next_in_pool(store, view, {"active": "2", "idx": 0}, 99) \
        is None

    # No active pool -> None.
    assert _next_in_pool(store, view, {"active": None, "idx": 0}, 10) \
        is None
    assert _next_in_pool(store, view, None, 10) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
