"""Video Review navigation modes: Queue / Flag / Needs Scoring.

Pins the two behaviors that back the mode selector:
* store.get_review_queue(..., flagged_only=True) = the still-reviewable queue
  intersected with the auto-filter has-events flag (review_event_log
  action='auto_filter_flag'), same shape/filters as the full queue.
* the video tab's _fetch_queue_by_mode dispatch returns uniform id-keyed dicts
  for all three modes.

Run with: pytest tests/test_review_queue_modes.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


_A = "BCH900"
_CHANNELS = f'["{_A}SR", "{_A}SLM"]'


def _store(tmp_path):
    """Store seeded with one session_config + a few has-video processed_files
    for animal BCH900."""
    s = Store(str(tmp_path / "data" / "m.db"))
    with s.connection() as conn:
        conn.execute(
            """INSERT INTO session_config (session_dir, channel_names,
               eeg_channels, discovered_at)
               VALUES (?, ?, ?, ?)""",
            ("/sess", _CHANNELS, "[0, 1]", "2026-07-01T00:00:00"))
        for fid, dt in [(1, "2026_07_01__00_00_00"),
                        (2, "2026_07_02__00_00_00"),
                        (3, "2026_07_03__00_00_00"),
                        (4, "2026_07_04__00_00_00")]:
            conn.execute(
                """INSERT INTO processed_files (id, file_path, session_dir,
                   chunk_datetime, duration_sec, has_video)
                   VALUES (?, ?, '/sess', ?, 3600, 1)""",
                (fid, f"/sess/{fid}.mat", dt))
        conn.commit()
    return s


# ===================================================================== #
#  flagged_only
# ===================================================================== #

def test_flagged_only_is_queue_intersect_flag(tmp_path):
    s = _store(tmp_path)
    email = "r@lab"
    # Baseline: all 4 files are in the normal queue.
    full = s.get_review_queue([_A], email, limit=100)
    assert {r["id"] for r in full} == {1, 2, 3, 4}

    # Auto-filter flags files 2 and 3 (has-events).
    for fid in (2, 3):
        s.insert_review_event(fid, "auto-filter@system", "auto_filter_flag",
                              {"n_peaks": 3}, animal_id=_A)

    flagged = s.get_review_queue([_A], email, limit=100, flagged_only=True)
    assert {r["id"] for r in flagged} == {2, 3}
    # Same shape as the full queue (id-keyed, carries session_dir + dt).
    assert all("id" in r and "session_dir" in r and "chunk_datetime" in r
               for r in flagged)


def test_flagged_only_excludes_finished(tmp_path):
    s = _store(tmp_path)
    email = "r@lab"
    for fid in (2, 3):
        s.insert_review_event(fid, "auto-filter@system", "auto_filter_flag",
                              {}, animal_id=_A)
    # File 2 gets finished -> leaves the queue -> leaves the flag subset too.
    s.mark_review(2, email, "has_events", markers=[], animal_id=_A)
    flagged = s.get_review_queue([_A], email, limit=100, flagged_only=True)
    assert {r["id"] for r in flagged} == {3}


def test_flagged_only_empty_when_nothing_flagged(tmp_path):
    s = _store(tmp_path)
    flagged = s.get_review_queue([_A], "r@lab", limit=100, flagged_only=True)
    assert flagged == []


# ===================================================================== #
#  _fetch_queue_by_mode dispatch (uniform id-keyed shape)
# ===================================================================== #

def test_fetch_queue_by_mode_uniform_shape(tmp_path):
    from src.dashboard.tabs.video import _fetch_queue_by_mode
    s = _store(tmp_path)
    email = "r@lab"
    s.insert_review_event(3, "auto-filter@system", "auto_filter_flag",
                          {}, animal_id=_A)
    # File 4 gets quick-flagged for scoring.
    s.mark_review(4, email, "needs_scoring", markers=[{"EO_sec": 5}],
                  animal_id=_A)

    q = _fetch_queue_by_mode(s, "queue", [_A], email, None, 100)
    f = _fetch_queue_by_mode(s, "flagged", [_A], email, None, 100)
    n = _fetch_queue_by_mode(s, "needs_scoring", [_A], email, None, 100)

    # needs_scoring pulls 4 out of the queue; flag = {3}; queue = the rest.
    assert {r["id"] for r in q} == {1, 2, 3}
    assert {r["id"] for r in f} == {3}
    assert {r["id"] for r in n} == {4}
    # All three modes yield rows the carousel can read (id present).
    for rows in (q, f, n):
        assert all(r.get("id") is not None for r in rows)


def test_fetch_queue_by_mode_no_animal(tmp_path):
    from src.dashboard.tabs.video import _fetch_queue_by_mode
    s = _store(tmp_path)
    assert _fetch_queue_by_mode(s, "queue", [], "r@lab", None, 100) == []
