"""Multi-animal onset re-attribution: store.reattribute_review + session_animals.

An onset scored in a multi-animal recording is filed under the scored channel's
animal. When it lands on the wrong one (picked BCH061's channel but the seizure
is BCH062's), the PI moves it. These tests pin that move: the events follow to
the target animal, the mis-filed row is retired, and an invalid target (an
animal not in the recording) is rejected.

Run with: pytest tests/test_reattribution.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


# A multi-animal session: BCH061 + BCH062 on the same recording.
_CHANNELS = '["stimCopy", "BCH061SLM", "BCH062SR", "BCH062SLM"]'


def _store(tmp_path):
    s = Store(str(tmp_path / "data" / "m.db"))
    with s.connection() as conn:
        conn.execute(
            """INSERT INTO session_config (session_dir, channel_names,
               eeg_channels, discovered_at) VALUES (?, ?, ?, ?)""",
            ("/sess", _CHANNELS, "[1, 2, 3]", "2026-05-27T00:00:00"))
        conn.execute(
            """INSERT INTO processed_files (id, file_path, session_dir,
               chunk_datetime, has_video) VALUES
               (4213, '/sess/a.mat', '/sess', '2026_05_27__08_20_12', 1)""")
        conn.commit()
    return s


def _latest_status(store, file_id, animal):
    with store.connection() as conn:
        row = conn.execute(
            """SELECT rs.status, rs.markers_json FROM review_state rs
               WHERE rs.file_id = ? AND rs.animal_id = ?
                 AND rs.id = (SELECT MAX(rs2.id) FROM review_state rs2
                              WHERE rs2.file_id = rs.file_id
                                AND rs2.animal_id = rs.animal_id)""",
            (file_id, animal)).fetchone()
    if not row:
        return None, []
    try:
        evs = json.loads(row["markers_json"] or "[]")
    except json.JSONDecodeError:
        evs = []
    return row["status"], evs


# ===================================================================== #
#  session_animals
# ===================================================================== #

def test_session_animals(tmp_path):
    s = _store(tmp_path)
    assert s.session_animals("/sess") == ["BCH061", "BCH062"]  # stimCopy excluded


# ===================================================================== #
#  reattribute_review
# ===================================================================== #

def test_reattribute_moves_events_and_retires_old(tmp_path):
    s = _store(tmp_path)
    email = "pi@lab"
    onsets = [{"type": "LVF", "EO_sec": 1046.57, "racine": 3},
              {"type": "LVF", "EO_sec": 3389.18, "racine": 4}]
    s.mark_review(4213, "rev@lab", "needs_scoring", markers=onsets,
                  animal_id="BCH061")

    ok = s.reattribute_review(4213, "BCH061", "BCH062", email)
    assert ok is True

    # Events now under BCH062, same status.
    st, evs = _latest_status(s, 4213, "BCH062")
    assert st == "needs_scoring"
    assert [e["EO_sec"] for e in evs] == [1046.57, 3389.18]
    # BCH061 row retired.
    st_old, _ = _latest_status(s, 4213, "BCH061")
    assert st_old == "abandoned"
    # Audit event written.
    with s.connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM review_event_log "
            "WHERE file_id=4213 AND action='reattribute'").fetchone()[0]
    assert n == 1


def test_reattribute_rejects_animal_not_in_recording(tmp_path):
    s = _store(tmp_path)
    s.mark_review(4213, "rev@lab", "needs_scoring",
                  markers=[{"EO_sec": 100.0, "racine": 2}], animal_id="BCH061")
    # BCH999 is not one of this recording's animals.
    assert s.reattribute_review(4213, "BCH061", "BCH999", "pi@lab") is False
    # Untouched.
    st, _ = _latest_status(s, 4213, "BCH061")
    assert st == "needs_scoring"


def test_reattribute_noop_same_animal_and_missing_source(tmp_path):
    s = _store(tmp_path)
    assert s.reattribute_review(4213, "BCH061", "BCH061", "pi@lab") is False
    # No source row for BCH062 yet.
    assert s.reattribute_review(4213, "BCH062", "BCH061", "pi@lab") is False
