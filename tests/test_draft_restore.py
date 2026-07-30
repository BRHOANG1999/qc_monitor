"""Draft re-hydration from review_state.

Regression cover for the incident where BCH062 onsets looked ERASED after
``src/maintenance/revert_stranded_drafts.py`` moved stranded drafts back to the
Flag pool. That script sets status to 'abandoned' and PRESERVES markers_json by
contract -- but the Video Review editor only re-hydrated markers when the status
was exactly 'needs_scoring', so 55 recordings opened blank with their onsets
still sitting in the DB.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.tabs.video import (  # noqa: E402
    restorable_drafts,
    _events_scope_matches,
)
from src.db.store import Store  # noqa: E402


def _row(status, events):
    return {"status": status, "markers_json": json.dumps(events)}


_ONSET = [{"type": "LVF", "EO_sec": 1627.901, "racine": None, "draft": True}]


# ------------------------------------------------- statuses that restore --- #

def test_needs_scoring_draft_restores():
    assert restorable_drafts(_row("needs_scoring", _ONSET)) == _ONSET


def test_abandoned_draft_restores():
    """The actual incident: reverted stranded draft must not read as empty."""
    assert restorable_drafts(_row("abandoned", _ONSET)) == _ONSET


def test_abandoned_row_preserves_the_exact_onset():
    got = restorable_drafts(_row("abandoned", _ONSET))
    assert got[0]["EO_sec"] == 1627.901
    assert got[0]["type"] == "LVF"


# --------------------------------------------- statuses that must NOT ----- #

def test_finalised_work_is_never_resurrected():
    """Submitted/approved rows are done -- re-hydrating them into the editor
    would present finalised scoring as unsaved draft work."""
    for st in ("pending_pi_review", "pi_approved", "has_events", "no_events"):
        assert restorable_drafts(_row(st, _ONSET)) == [], st


def test_reattributed_row_has_nothing_to_restore():
    """Store.reattribute_review writes the abandoned row with markers=[], so
    onsets moved to another animal never bleed back into the old one."""
    assert restorable_drafts(_row("abandoned", [])) == []


# ------------------------------------------------------------ robustness --- #

def test_missing_and_malformed_rows_are_safe():
    assert restorable_drafts(None) == []
    assert restorable_drafts({}) == []
    assert restorable_drafts({"status": "abandoned"}) == []
    assert restorable_drafts({"status": "abandoned",
                              "markers_json": "{not json"}) == []
    assert restorable_drafts({"status": "abandoned",
                              "markers_json": '{"a": 1}'}) == []


# ------------------------------- get_review_state tie-break determinism --- #

def _insert_review_row(conn, rid, fid, animal, status, events, updated_at):
    conn.execute(
        """INSERT INTO review_state (id, file_id, user_email, status,
           markers_json, animal_id, created_at, updated_at)
           VALUES (?, ?, 'r@lab', ?, ?, ?, ?, ?)""",
        (rid, fid, status, json.dumps(events), animal, updated_at, updated_at))


def test_get_review_state_breaks_updated_at_ties_by_newest_id(tmp_path):
    """The BCH062 Flag-pool incident: revert_stranded_drafts stamps an IDENTICAL
    updated_at across sibling 'abandoned' rows -- one empty (old approved row
    flipped) and one carrying the preserved onsets (higher id). get_review_state
    must return the markers-bearing row, not an arbitrary sibling."""
    s = Store(str(tmp_path / "data" / "m.db"))
    fid, animal, ts = 4245, "BCH062", "2026-07-16T13:58:59.359764"
    with s.connection() as conn:
        conn.execute(
            """INSERT INTO processed_files (id, file_path, session_dir,
               chunk_datetime, duration_sec, has_video)
               VALUES (?, '/s/1.mat', '/s', '2026_07_01__00_00_00', 3600, 1)""",
            (fid,))
        # Empty siblings first (lower ids), markers row inserted last (higher id)
        _insert_review_row(conn, 1923, fid, animal, "abandoned", [], ts)
        _insert_review_row(conn, 1924, fid, animal, "abandoned", [], ts)
        _insert_review_row(conn, 4468, fid, animal, "abandoned", _ONSET, ts)
        conn.commit()

    latest = s.get_review_state(fid, animal_id=animal)
    assert latest is not None and latest["id"] == 4468
    assert restorable_drafts(latest) == _ONSET


# ------------------------- cross-file replication guard (scope matching) --- #
#
# The store->DB write paths (Submit / autosave) read video-events-store, which
# lags file navigation by one callback round. Writing in that window filed the
# PREVIOUS recording's onsets under the new file -- the identical EO=1493.289 on
# four BCH062 files / EO=729.993 on two BCH111 files (2026-07). Writes now gate
# on video-events-current-key == "{file}:{animal}".

def test_scope_matches_when_key_is_the_current_file():
    assert _events_scope_matches("4245:BCH062", 4245, "BCH062") is True


def test_scope_rejects_stale_key_after_navigation():
    """Navigated to 4249 but the store still holds 4245's key -> must NOT write
    (this is exactly the replication the guard blocks)."""
    assert _events_scope_matches("4245:BCH062", 4249, "BCH062") is False


def test_scope_rejects_wrong_animal():
    assert _events_scope_matches("4245:BCH061", 4245, "BCH062") is False


def test_scope_rejects_unset_or_incomplete():
    assert _events_scope_matches(None, 4245, "BCH062") is False
    assert _events_scope_matches("4245:BCH062", None, "BCH062") is False
    assert _events_scope_matches("4245:BCH062", 4245, None) is False
