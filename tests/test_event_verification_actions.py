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
    _pending_table_rows,
    _format_chunk_dt,
)
from src.db.store import Store  # noqa: E402


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


import json  # noqa: E402


def test_format_chunk_dt():
    assert _format_chunk_dt("2026_03_10__04_59_00") == "2026-03-10 04:59"
    assert _format_chunk_dt("garbage") == "garbage"
    assert _format_chunk_dt(None) == ""


def _seed_pending(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels, discovered_at)
               VALUES ('S1', ?, ?, '2026-01-01T00:00:00')""",
            (json.dumps(["stimCopy", "BCH061SR", "stimCopy", "BCH061SLM"]),
             json.dumps([1, 3])),
        )
        conn.execute(
            """INSERT INTO processed_files
               (id, file_path, session_dir, chunk_datetime, has_video)
               VALUES (1, '/f/a.mat', 'S1', '2026_03_10__04_59_00', 1)""")
        conn.commit()
    # One pending submission with 2 events, one no-event submission.
    store.mark_review(1, "u@lab", "pending_pi_review",
                       markers=[{"type": "sz"}, {"type": "sz"}])
    return store


def test_pending_table_rows_shape(tmp_path):
    store = _seed_pending(tmp_path)
    rows = store.pi_pending_files(limit=50)
    data = _pending_table_rows(store, rows)
    assert len(data) == 1
    r = data[0]
    assert r["id"] == 1
    assert r["animal"] == "BCH061"
    assert r["date"] == "2026-03-10 04:59"
    assert r["n_events"] == 2
    assert r["submitter"] == "u@lab"
    assert r["view"] == "Open >"


def test_approve_all_pending_covers_every_file(tmp_path):
    """Approve-all must approve every pending file, not just a table
    page (the bug: finalize CSV only covered the first 25 rows)."""
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    n_files = 40   # > one 25-row page
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels, discovered_at)
               VALUES ('S1', ?, ?, '2026-01-01T00:00:00')""",
            (json.dumps(["stimCopy", "BCH061SR"]), json.dumps([1])))
        for i in range(1, n_files + 1):
            conn.execute(
                "INSERT INTO processed_files (id, file_path, session_dir, "
                "chunk_datetime) VALUES (?, ?, 'S1', '2026_03_10__04_59_00')",
                (i, f"/f/{i}.mat"))
        conn.commit()
    for i in range(1, n_files + 1):
        store.mark_review(i, "u@lab", "pending_pi_review",
                          markers=[{"type": "sz"}])

    assert len(store.pi_pending_files(limit=500)) == n_files
    approved = store.pi_approve_all_pending("pi@lab")
    assert approved == n_files
    # Nothing left pending; every file is now pi_approved.
    assert store.pi_pending_files(limit=500) == []
    with store.connection() as conn:
        n_appr = conn.execute(
            "SELECT COUNT(*) FROM review_state WHERE status='pi_approved'"
        ).fetchone()[0]
    assert n_appr == n_files
    # A second call is a no-op (nothing pending).
    assert store.pi_approve_all_pending("pi@lab") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
