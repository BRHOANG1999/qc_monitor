"""Phase 1b: durable autosave of in-progress scores.

Covers Store.upsert_scoring_draft -- the in-place draft persistence that
stops silent score-loss without growing review_state per edit or spamming
review_event_log -- plus the video-tab meaningfulness guard helpers.

Run with: pytest tests/test_autosave_draft.py -q
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


def _seed_animal_file(store, fid, session_dir="sessA", animal="BCH001"):
    with store.connection() as conn:
        conn.execute(
            """INSERT INTO session_config
               (session_dir, channel_names, eeg_channels, discovered_at)
               VALUES (?, ?, ?, '2026-01-01')
               ON CONFLICT(session_dir) DO NOTHING""",
            (session_dir, json.dumps([f"{animal}SR", f"{animal}SLM"]),
             json.dumps([0, 1])))
        conn.execute(
            """INSERT INTO processed_files
               (id, file_path, session_dir, has_video, chunk_datetime)
               VALUES (?, ?, ?, 1, '2026_01_01__00_00_00')""",
            (fid, f"/f/{fid}.mat", session_dir))
        conn.commit()


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


def _counts(store, fid):
    with store.connection() as conn:
        rs = conn.execute(
            "SELECT COUNT(*) FROM review_state WHERE file_id=?", (fid,)
        ).fetchone()[0]
        el = conn.execute(
            "SELECT COUNT(*) FROM review_event_log WHERE file_id=?", (fid,)
        ).fetchone()[0]
    return rs, el


_MARKERS_1 = [{"type": "LVF", "EO_sec": 12.0, "racine": 4, "light": 2}]
_MARKERS_2 = [{"type": "LVF", "EO_sec": 12.0, "racine": 5, "light": 1},
              {"type": "LVF", "EO_sec": 88.0, "racine": 3, "light": 2}]


def test_seed_creates_single_needs_scoring_row(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)

    res = store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_1)
    assert res == "seeded"
    assert (store.get_review_state(1, animal_id="BCH001") or {}
            ).get("status") == "needs_scoring"
    assert [r["file_id"] for r in
            store.files_needing_scoring_for_animal("BCH001")] == [1]
    # Exactly one review_state row + one audit event (quick_flag).
    assert _counts(store, 1) == (1, 1)
    with store.connection() as conn:
        act = conn.execute(
            "SELECT action FROM review_event_log WHERE file_id=1"
        ).fetchone()[0]
    assert act == "quick_flag"


def test_update_in_place_no_row_growth(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)

    assert store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_1) \
        == "seeded"
    row0 = store.get_review_state(1, animal_id="BCH001")
    rid0, created0 = row0["id"], row0["created_at"]

    # Two more autosaves with new content -> in-place UPDATE each time.
    assert store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_2) \
        == "updated"
    assert store.upsert_scoring_draft(
        1, "u@lab", "BCH001",
        _MARKERS_2 + [{"type": "LVF", "EO_sec": 99.0, "racine": 6}]) \
        == "updated"

    row1 = store.get_review_state(1, animal_id="BCH001")
    assert row1["id"] == rid0                       # same row, not a new one
    assert row1["updated_at"] >= created0
    assert len(json.loads(row1["markers_json"])) == 3   # latest content wins
    # No row growth, no audit spam: still 1 review_state + 1 event_log row.
    assert _counts(store, 1) == (1, 1)


def test_scoped_strictly_per_animal(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_multi_animal_file(store, 7)

    store.upsert_scoring_draft(7, "u@lab", "BCH062", _MARKERS_1)
    # BCH061 must never see BCH062's draft.
    assert store.get_review_state(7, animal_id="BCH061") is None
    assert store.files_needing_scoring_for_animal("BCH061") == []
    assert [r["file_id"] for r in
            store.files_needing_scoring_for_animal("BCH062")] == [7]

    # Updating BCH062 never mutates a BCH061 row (there is none).
    store.upsert_scoring_draft(7, "u@lab", "BCH062", _MARKERS_2)
    assert store.get_review_state(7, animal_id="BCH061") is None


def test_skipped_when_finalised(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)
    # Reviewer submitted -> pending_pi_review is the latest row.
    store.mark_review(1, "u@lab", "pending_pi_review",
                       markers=_MARKERS_1, animal_id="BCH001")
    before = _counts(store, 1)

    res = store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_2)
    assert res == "skipped"
    # Nothing written; the submitted state is untouched.
    assert _counts(store, 1) == before
    assert (store.get_review_state(1, animal_id="BCH001") or {}
            ).get("status") == "pending_pi_review"


def test_mark_done_supersedes_draft(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)
    store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_1)
    store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_2)

    # Terminal submit inserts a higher-id row that supersedes the draft.
    store.mark_review(1, "u@lab", "pending_pi_review",
                       markers=_MARKERS_2, animal_id="BCH001")
    assert (store.get_review_state(1, animal_id="BCH001") or {}
            ).get("status") == "pending_pi_review"
    # File leaves the needs-scoring pool...
    assert store.files_needing_scoring_for_animal("BCH001") == []
    # ...but the draft row is preserved as history (draft + submit = 2 rows).
    rs, _ = _counts(store, 1)
    assert rs == 2


def test_draft_survives_reload_roundtrip(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)
    store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_2)

    # Exercise the exact read the video tab's _rescope_events uses on reload.
    latest = store.get_review_state(1, animal_id="BCH001")
    assert latest["status"] == "needs_scoring"
    reloaded = json.loads(latest["markers_json"])
    assert [e["racine"] for e in reloaded] == [5, 3]
    assert [e["EO_sec"] for e in reloaded] == [12.0, 88.0]
    assert [e["light"] for e in reloaded] == [1, 2]


def test_queue_excludes_seeded_draft(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_animal_file(store, 1)
    assert any(r["id"] == 1 for r in
               store.get_review_queue(["BCH001"], "u@lab"))

    store.upsert_scoring_draft(1, "u@lab", "BCH001", _MARKERS_1)
    assert not any(r["id"] == 1 for r in
                   store.get_review_queue(["BCH001"], "u@lab"))


def test_meaningful_edits_guard():
    """The save-guard predicate: a pristine blank event is NOT meaningful,
    but any scored field flips it true."""
    from src.dashboard.tabs.video import (
        _event_is_meaningful, _has_meaningful_edits)
    from src.dashboard.tabs.video_events import blank_event
    blank = blank_event()
    assert _event_is_meaningful(blank) is False
    assert _has_meaningful_edits([blank]) is False
    assert _has_meaningful_edits([]) is False
    assert _event_is_meaningful({**blank, "racine": 3}) is True
    assert _event_is_meaningful({**blank, "EO_sec": 10.0}) is True
    assert _event_is_meaningful({**blank, "type": "LVF"}) is True
    assert _has_meaningful_edits([blank, {**blank, "racine": 3}]) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
