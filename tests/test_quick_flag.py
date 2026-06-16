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
                       markers=[{"EO_sec": 12.0, "draft": True}])

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
