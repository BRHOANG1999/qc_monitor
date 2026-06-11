"""Soft-claim tests: file_claim table + queue/Mass-Analyze exclusions.

A reviewer who opens a recording claims it; the claim hides the file
from OTHER reviewers' queues (and from the Mass Analyze pre-screen) for
Store.CLAIM_TTL_MINUTES, then expires on its own. These tests pin that
behaviour: own claims stay visible, expired claims reappear, release is
immediate, and the audit log isn't spammed by re-opens.

Run with: pytest tests/test_file_claim.py -q
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.utils import mass_analyze as ma  # noqa: E402

ANIMAL = "BCH060"
REVIEWER_A = "a@lab"
REVIEWER_B = "b@lab"


def _seed(tmp_path) -> Store:
    """Store with one session for BCH060 and three companion-video
    recordings, oldest first."""
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    now = datetime.now().isoformat()
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels,
                discovered_at)
               VALUES (?, ?, ?, ?)""",
            ("S1",
             json.dumps(["stimCopy", f"{ANIMAL}SR",
                          "stimCopy", f"{ANIMAL}SLM"]),
             json.dumps([1, 3]), now),
        )
        for i in range(1, 4):
            conn.execute(
                """INSERT INTO processed_files
                   (id, file_path, session_dir, session_name,
                    chunk_datetime, has_video)
                   VALUES (?, ?, 'S1', 'S1', ?, 1)""",
                (i, f"/fake/f{i}.mat",
                 f"2026-01-0{i}T00:00:00"),
            )
        conn.commit()
    return store


def _queue_ids(store: Store, email: str) -> list[int]:
    return [r["id"] for r in
            store.get_review_queue([ANIMAL], email, limit=100)]


def _backdate_claim(store: Store, file_id: int) -> None:
    """Push a claim past the TTL so it reads as expired."""
    old = (datetime.now()
           - timedelta(minutes=Store.CLAIM_TTL_MINUTES + 5)).isoformat()
    with store.connection() as conn:
        conn.execute(
            "UPDATE file_claim SET claimed_at = ? WHERE file_id = ?",
            (old, file_id))
        conn.commit()


# ===================================================================== #
#  Queue exclusion
# ===================================================================== #

def test_seed_has_all_three_in_both_queues(tmp_path):
    store = _seed(tmp_path)
    assert _queue_ids(store, REVIEWER_A) == [1, 2, 3]
    assert _queue_ids(store, REVIEWER_B) == [1, 2, 3]


def test_claim_hides_file_from_other_reviewer(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    # B no longer sees file 1; A still does (own claim).
    assert _queue_ids(store, REVIEWER_B) == [2, 3]
    assert _queue_ids(store, REVIEWER_A) == [1, 2, 3]


def test_expired_claim_reappears(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    _backdate_claim(store, 1)
    assert _queue_ids(store, REVIEWER_B) == [1, 2, 3]


def test_release_claim_reappears_immediately(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    assert _queue_ids(store, REVIEWER_B) == [2, 3]
    store.release_claim(1)
    assert _queue_ids(store, REVIEWER_B) == [1, 2, 3]


# ===================================================================== #
#  Mass Analyze exclusion
# ===================================================================== #

def test_fresh_claim_excluded_from_mass_analyze(tmp_path):
    store = _seed(tmp_path)
    assert ma.count_pending_for_animal(store, ANIMAL) == 3
    store.claim_file(1, REVIEWER_A)
    assert ma.count_pending_for_animal(store, ANIMAL) == 2
    ids = {f["file_id"] for f in ma.pending_files_for_animal(store, ANIMAL)}
    assert ids == {2, 3}


def test_stale_claim_not_excluded_from_mass_analyze(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    _backdate_claim(store, 1)
    assert ma.count_pending_for_animal(store, ANIMAL) == 3


# ===================================================================== #
#  Audit log
# ===================================================================== #

def test_claim_logs_once_per_new_claim(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    store.claim_file(1, REVIEWER_A)  # immediate re-open: no new log
    with store.connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM review_event_log "
            "WHERE file_id = 1 AND action = 'claim'"
        ).fetchone()[0]
    assert n == 1


def test_takeover_by_other_user_logs_again(tmp_path):
    store = _seed(tmp_path)
    store.claim_file(1, REVIEWER_A)
    store.claim_file(1, REVIEWER_B)  # different claimer: new audit row
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT user_email FROM review_event_log "
            "WHERE file_id = 1 AND action = 'claim' "
            "ORDER BY at"
        ).fetchall()
    assert [r["user_email"] for r in rows] == [REVIEWER_A, REVIEWER_B]
    # UPSERT keeps a single claim row, now owned by B.
    with store.connection() as conn:
        owner = conn.execute(
            "SELECT user_email FROM file_claim WHERE file_id = 1"
        ).fetchone()["user_email"]
    assert owner == REVIEWER_B


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
