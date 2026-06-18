"""Training store round-trip: attempts, progress, advancement, and the
unseen-first / recycle-oldest example picker.

Run with: pytest tests/test_training_store.py -q
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


def _store(tmp_path):
    return Store(str(tmp_path / "data" / "monitor.db"))


def _pf(store, file_id, session="sessA"):
    """Insert a bare processed_files row (training_attempt.file_id FKs it)."""
    with store.connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_files (id, file_path, "
            "session_dir, duration_sec) VALUES (?, ?, ?, ?)",
            (file_id, f"/f/{file_id}.mat", session, 3600.0))
        conn.commit()


def _approve(store, file_id, events, session="sessA"):
    """Insert a processed_files row + a pi_approved review_state row."""
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "duration_sec) VALUES (?, ?, ?, ?)",
            (file_id, f"/f/{file_id}.mat", session, 3600.0))
        conn.execute(
            """INSERT INTO review_state
               (file_id, user_email, status, markers_json,
                created_at, updated_at)
               VALUES (?, 'pi@x', 'pi_approved', ?, ?, ?)""",
            (file_id, json.dumps(events),
             f"2026-01-0{file_id}", f"2026-01-0{file_id}"))
        conn.commit()


def test_attempt_and_rolling_scores(tmp_path):
    store = _store(tmp_path)
    for fid in (10, 11, 12, 99):
        _pf(store, fid)
    for i, s in enumerate([0.5, 0.8, 1.0]):
        store.add_training_attempt("Stu@X", 2, 10 + i,
                                    json.dumps({}), s)
    # Different stage shouldn't bleed in.
    store.add_training_attempt("stu@x", 1, 99, json.dumps({}), 0.0)
    scores = store.recent_training_scores("stu@x", 2, limit=10)
    assert scores == [0.5, 0.8, 1.0]   # chronological, email lowered


def test_progress_default_and_advance(tmp_path):
    store = _store(tmp_path)
    assert store.get_training_progress("new@x") == {
        "unlocked_stage": 1, "certified_at": None}
    store.advance_student("new@x", 2)
    assert store.get_training_progress("new@x")["unlocked_stage"] == 2
    # Never regress.
    store.advance_student("new@x", 1)
    assert store.get_training_progress("new@x")["unlocked_stage"] == 2
    # Past stage 3 -> certified.
    store.advance_student("new@x", 4)
    prog = store.get_training_progress("new@x")
    assert prog["unlocked_stage"] == 3 and prog["certified_at"]


def test_next_file_unseen_first_then_recycle_oldest(tmp_path):
    store = _store(tmp_path)
    _approve(store, 1, [{"type": "LVF", "EO_sec": 5.0}])
    _approve(store, 2, [])
    _approve(store, 3, [{"type": "HYP", "EO_sec": 9.0}])

    # Nothing seen yet -> returns an approved file with its events.
    nxt = store.next_training_file("stu@x")
    assert nxt is not None and nxt["file_id"] in (1, 2, 3)
    assert isinstance(nxt["events"], list)

    # Attempt files 1 and 3; file 2 stays unseen -> must come next.
    store.add_training_attempt("stu@x", 1, 1, json.dumps({}), 1.0)
    store.add_training_attempt("stu@x", 1, 3, json.dumps({}), 1.0)
    assert store.next_training_file("stu@x")["file_id"] == 2

    # Now all seen -> recycle the oldest-seen (file 1, lowest attempt id).
    store.add_training_attempt("stu@x", 1, 2, json.dumps({}), 1.0)
    assert store.next_training_file("stu@x")["file_id"] == 1


def test_next_file_none_when_no_approved(tmp_path):
    store = _store(tmp_path)
    assert store.next_training_file("stu@x") is None


def test_roster_lists_students(tmp_path):
    store = _store(tmp_path)
    _pf(store, 1)
    store.add_training_attempt("a@x", 1, 1, json.dumps({}), 1.0)
    store.advance_student("b@x", 2)
    roster = {r["student_email"]: r for r in store.all_training_progress()}
    assert "a@x" in roster and "b@x" in roster
    assert roster["b@x"]["unlocked_stage"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_candidate_pool_labels(tmp_path):
    store = _store(tmp_path)
    # session with two animal channels so animals resolve.
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO session_config (session_dir, channel_names, "
            "eeg_channels, discovered_at) VALUES ('sessA', ?, ?, '2026-01-01')",
            (json.dumps(["stimCopy", "BCH062SR", "BCH061SLM"]),
             json.dumps([1])))
        conn.commit()
    _approve(store, 1, [{"type": "LVF", "EO_sec": 1.0, "racine": 4},
                        {"type": "HYP", "EO_sec": 3.0, "racine": 7}])
    _approve(store, 2, [])
    pool = {c["file_id"]: c for c in store.training_candidate_pool("stu@x")}
    assert pool[1]["has_seizure"] and not pool[2]["has_seizure"]
    assert pool[1]["rep_racine"] == 7          # highest-Racine event
    assert pool[1]["rep_type"] == "HYP"
    assert set(pool[1]["animals"]) == {"BCH062", "BCH061"}


def test_round_scoped_grade_reset_keeps_history(tmp_path):
    store = _store(tmp_path)
    for fid in (1, 2, 3):
        _pf(store, fid)
    store.add_training_attempt("stu@x", 1, 1, "{}", 0.2)   # pre-round
    r1 = store.create_training_round("stu@x", 1, [1, 2, 3])
    store.add_training_attempt("stu@x", 1, 1, "{}", 1.0)
    store.add_training_attempt("stu@x", 1, 2, "{}", 1.0)
    assert store.round_scores("stu@x", 1, r1["started_after_id"]) == [1.0, 1.0]
    # New round = new boundary; round 1's attempts excluded from round 2.
    store.finish_training_round(r1["round_id"], 4, "review_more")
    r2 = store.create_training_round("stu@x", 1, [3])
    assert store.round_scores("stu@x", 1, r2["started_after_id"]) == []
    # Lifetime keeps ALL attempts (history persists across resets).
    life = store.training_lifetime_stats("stu@x", 1)
    assert life["n"] == 3


def test_current_round_and_notify_dedup(tmp_path):
    store = _store(tmp_path)
    _pf(store, 1)
    r = store.create_training_round("stu@x", 2, [1])
    assert store.current_training_round("stu@x", 2)["round_id"] == r["round_id"]
    assert store.mark_round_notified(r["round_id"]) is True
    assert store.mark_round_notified(r["round_id"]) is False   # deduped
    store.finish_training_round(r["round_id"], 5, "move_on")
    assert store.current_training_round("stu@x", 2) is None     # ended
