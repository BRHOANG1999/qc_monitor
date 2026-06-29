"""Phase 2: quick-flag 'needs_scoring' pool.

Run with: pytest tests/test_quick_flag.py -q
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
import src.utils.mass_analyze as ma  # noqa: E402


def _seed_animal_file(store, fid, session_dir="sessA",
                        animal="BCH001"):
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels,
                discovered_at)
               VALUES (?, ?, ?, '2026-01-01')
               ON CONFLICT(session_dir) DO NOTHING""",
            (session_dir,
             json.dumps([f"{animal}SR", f"{animal}SLM"]),
             json.dumps([0, 1])))
        conn.execute(
            """INSERT INTO processed_files
               (id, file_path, session_dir, has_video, chunk_datetime)
               VALUES (?, ?, ?, 1, '2026_01_01__00_00_00')""",
            (fid, f"/f/{fid}.mat", session_dir))
        conn.commit()


def test_needs_scoring_in_fresh_check(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    Store(db)
    sql = sqlite3.connect(db).execute(
        "SELECT sql FROM sqlite_master WHERE name='review_state'"
    ).fetchone()[0]
    assert "needs_scoring" in sql


def test_migration_adds_needs_scoring_to_legacy_db(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    Store(db)  # build schema
    # Rewind: rebuild review_state WITHOUT needs_scoring (pre-Phase-2).
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        DROP TABLE review_state;
        CREATE TABLE review_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES processed_files(id),
            user_email TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('claimed','no_events','has_events',
                      'abandoned','pending_pi_review','pi_approved',
                      'pi_flagged')),
            markers_json TEXT, note TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO processed_files (id, file_path)
            VALUES (1, '/f/a.mat');
        INSERT INTO review_state
            (file_id, user_email, status, markers_json, note,
             created_at, updated_at)
            VALUES (1, 'u', 'has_events', '[]', NULL,
                    '2026-01-01', '2026-01-01');
        """)
    conn.commit()
    conn.close()
    assert "needs_scoring" not in sqlite3.connect(db).execute(
        "SELECT sql FROM sqlite_master WHERE name='review_state'"
    ).fetchone()[0]

    # Boot -> migration widens the CHECK, preserves the row.
    Store(db)
    new = sqlite3.connect(db)
    assert "needs_scoring" in new.execute(
        "SELECT sql FROM sqlite_master WHERE name='review_state'"
    ).fetchone()[0]
    assert new.execute(
        "SELECT COUNT(*) FROM review_state").fetchone()[0] == 1


def test_quick_flag_excluded_from_queue_and_pending(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)
    # Before flagging: eligible for the queue.
    q0 = store.get_review_queue(["BCH001"], "u@lab")
    assert any(r["id"] == 1 for r in q0)

    store.mark_review(1, "u@lab", "needs_scoring",
                       markers=[{"EO_sec": 12.0, "draft": True}],
                       animal_id="BCH001")

    # After: gone from the FIFO queue + from MA pending, present in
    # the needs-scoring pool.
    q1 = store.get_review_queue(["BCH001"], "u@lab")
    assert not any(r["id"] == 1 for r in q1)
    pend = ma.pending_files_for_animal(store, "BCH001")
    assert not any(int(f["file_id"]) == 1 for f in pend)
    ns = store.files_needing_scoring_for_animal("BCH001")
    assert [r["file_id"] for r in ns] == [1]
    drafts = json.loads(ns[0]["markers_json"])
    assert drafts[0]["EO_sec"] == 12.0 and drafts[0]["draft"] is True


def _seed_multi_animal_file(store, fid, session_dir="sessM"):
    """A recording with two animals (BCH062 ch0, BCH061 ch1)."""
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels, discovered_at)
               VALUES (?, ?, ?, '2026-01-01')
               ON CONFLICT(session_dir) DO NOTHING""",
            (session_dir, json.dumps(["BCH062SLM", "BCH061SR"]),
             json.dumps([0, 1])))
        conn.execute(
            """INSERT INTO processed_files
               (id, file_path, session_dir, has_video, chunk_datetime)
               VALUES (?, ?, ?, 1, '2026_01_01__00_00_00')""",
            (fid, f"/f/{fid}.mat", session_dir))
        conn.commit()


def test_flags_are_strictly_per_animal(tmp_path):
    """A flag for one animal must never surface for another (the leak)."""
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_multi_animal_file(store, 7)
    # Flag onsets for BCH062 only (the channel-0 electrode).
    store.mark_review(7, "u@lab", "needs_scoring",
                       markers=[{"EO_sec": 10.5, "draft": True}],
                       animal_id="BCH062")

    # Needs-scoring pool: present for BCH062, absent for BCH061.
    assert [r["file_id"] for r in
            store.files_needing_scoring_for_animal("BCH062")] == [7]
    assert store.files_needing_scoring_for_animal("BCH061") == []

    # get_review_state: B's row never returned when querying for A.
    assert (store.get_review_state(7, animal_id="BCH062") or {}
            ).get("status") == "needs_scoring"
    assert store.get_review_state(7, animal_id="BCH061") is None

    # A legacy whole-file (NULL) row leaks to NEITHER specific animal.
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO review_state
               (file_id, user_email, status, markers_json, animal_id,
                created_at, updated_at)
               VALUES (7, 'u@lab', 'needs_scoring', '[]', NULL,
                       '2026-02-01', '2026-02-01')""")
        conn.commit()
    assert store.get_review_state(7, animal_id="BCH061") is None
    # BCH062's own row is still its latest (the NULL row is invisible).
    assert (store.get_review_state(7, animal_id="BCH062") or {}
            ).get("status") == "needs_scoring"


def test_animal_for_file_channel(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_multi_animal_file(store, 8)
    assert store.animal_for_file_channel(8, 0) == "BCH062"
    assert store.animal_for_file_channel(8, 1) == "BCH061"
    assert store.animal_for_file_channel(8, 9) is None     # out of range
    assert store.animal_for_file_channel(8, None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
