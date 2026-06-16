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
from src.dashboard.tabs.video import _animal_for_session  # noqa: E402


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
