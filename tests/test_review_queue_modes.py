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


# ===================================================================== #
#  neighbor_queue_file: advance by chunk_datetime, not the queue head
# ===================================================================== #

def test_neighbor_advances_by_date_when_current_left_queue(tmp_path):
    """A manually-picked MID-queue file, once finalised (out of the queue),
    advances to the next file BY DATE -- not the oldest queue head (the Sept-2
    snap-back bug)."""
    s = _store(tmp_path)              # files 1..4 dated Jul 1..4
    email = "r@lab"
    # Finalise file 2 -> it leaves the queue; queue is now {1, 3, 4}.
    s.mark_review(2, email, "has_events", markers=[], animal_id=_A)
    nxt = s.neighbor_queue_file(current_file_id=2, animal_ids=[_A],
                                user_email=email, direction=1)
    assert nxt == 3, "should go to the next-by-date file (Jul 3), not head (1)"
    prev = s.neighbor_queue_file(current_file_id=2, animal_ids=[_A],
                                 user_email=email, direction=-1)
    assert prev == 1, "backwards should land on the last-earlier file (Jul 1)"


def test_neighbor_none_when_nothing_later(tmp_path):
    """Finalising the newest file has nothing later -> None (no wrap to head)."""
    s = _store(tmp_path)
    email = "r@lab"
    s.mark_review(4, email, "has_events", markers=[], animal_id=_A)
    assert s.neighbor_queue_file(current_file_id=4, animal_ids=[_A],
                                 user_email=email, direction=1) is None


def test_neighbor_head_fallback_when_datetime_unknown(tmp_path):
    """A current file with no known chunk_datetime (e.g. absent) still falls
    back to head/tail so the reviewer is never stuck."""
    s = _store(tmp_path)
    email = "r@lab"
    assert s.neighbor_queue_file(current_file_id=999, animal_ids=[_A],
                                 user_email=email, direction=1) == 1
    assert s.neighbor_queue_file(current_file_id=999, animal_ids=[_A],
                                 user_email=email, direction=-1) == 4


def test_neighbor_in_queue_is_index_step(tmp_path):
    """When the current file is still in the queue, it's a plain +/-1 step."""
    s = _store(tmp_path)
    email = "r@lab"
    assert s.neighbor_queue_file(current_file_id=1, animal_ids=[_A],
                                 user_email=email, direction=1) == 2
    assert s.neighbor_queue_file(current_file_id=3, animal_ids=[_A],
                                 user_email=email, direction=-1) == 2


def test_neighbor_flagged_only_stays_in_pool(tmp_path):
    """Pool-aware advance: in the flagged pool, advancing skips unflagged files."""
    s = _store(tmp_path)
    email = "r@lab"
    for fid in (2, 3):
        s.insert_review_event(fid, "auto-filter@system", "auto_filter_flag",
                              {}, animal_id=_A)
    # Finalise flagged file 2 -> flagged pool now {3}; next flagged after Jul 2.
    s.mark_review(2, email, "has_events", markers=[], animal_id=_A)
    nxt = s.neighbor_queue_file(current_file_id=2, animal_ids=[_A],
                                user_email=email, direction=1, flagged_only=True)
    assert nxt == 3, "flagged advance must stay in the flagged subset"


# ===================================================================== #
#  Poor-lighting cleanup pool
# ===================================================================== #

def test_poor_lighting_pool_lists_light_1_files(tmp_path):
    s = _store(tmp_path)
    email = "r@lab"
    # File 2 has a poor-light event, file 3 a good-light one, file 4 no light.
    s.mark_review(2, email, "has_events",
                  markers=[{"EO_sec": 5, "light": 1}], animal_id=_A)
    s.mark_review(3, email, "has_events",
                  markers=[{"EO_sec": 5, "light": 2}], animal_id=_A)
    s.mark_review(4, email, "has_events", markers=[{"EO_sec": 5}], animal_id=_A)
    poor = s.files_with_poor_lighting(_A)
    assert {r["file_id"] for r in poor} == {2}
    assert all("chunk_datetime" in r and "session_dir" in r for r in poor)


def test_flip_poor_lighting_removes_from_pool(tmp_path):
    s = _store(tmp_path)
    email = "r@lab"
    for fid in (2, 3):
        s.mark_review(fid, email, "needs_scoring",
                      markers=[{"EO_sec": 5, "light": 1},
                               {"EO_sec": 9, "light": 1}], animal_id=_A)
    assert {r["file_id"] for r in s.files_with_poor_lighting(_A)} == {2, 3}
    flipped = s.flip_poor_lighting_to_good(2, _A)
    assert flipped == 2                     # both events flipped
    # File 2 leaves the pool; status is preserved (still needs_scoring).
    assert {r["file_id"] for r in s.files_with_poor_lighting(_A)} == {3}
    latest = s.get_review_state(2, animal_id=_A)
    assert latest["status"] == "needs_scoring"
    import json as _j
    assert all(e["light"] == 2 for e in _j.loads(latest["markers_json"]))


def test_flip_poor_lighting_noop_when_none(tmp_path):
    s = _store(tmp_path)
    s.mark_review(2, "r@lab", "has_events",
                  markers=[{"EO_sec": 5, "light": 2}], animal_id=_A)
    assert s.flip_poor_lighting_to_good(2, _A) == 0


def test_fetch_queue_by_mode_poor_lighting(tmp_path):
    from src.dashboard.tabs.video import _fetch_queue_by_mode
    s = _store(tmp_path)
    s.mark_review(3, "r@lab", "pending_pi_review",
                  markers=[{"EO_sec": 5, "light": 1}], animal_id=_A)
    rows = _fetch_queue_by_mode(s, "poor_lighting", [_A], "r@lab", None, 100)
    assert {r["id"] for r in rows} == {3}


def test_next_poor_lighting_advances_by_date(tmp_path):
    from src.dashboard.tabs.video import _next_poor_lighting
    s = _store(tmp_path)
    for fid in (2, 3, 4):
        s.mark_review(fid, "r@lab", "has_events",
                      markers=[{"EO_sec": 5, "light": 1}], animal_id=_A)
    # After (conceptually) flipping file 2, the next poor file by date is 3.
    _sd, nxt = _next_poor_lighting(s, [_A], 2)
    assert nxt == 3
