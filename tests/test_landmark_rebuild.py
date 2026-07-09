"""Regression: red onset lines must reflect the CURRENT file on a switch.

_update_lfp / _update_analysis draw landmark verticals on every heavy trace
rebuild. On a file/channel switch the live video-events-store still holds the
PREVIOUS file's events, so the builders now read the newly-selected file's
saved draft from the DB via _saved_landmarks_for. This test locks that in,
including strict per-animal scoping on a multi-animal recording.

Run with: pytest tests/test_landmark_rebuild.py -q
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
from src.dashboard.tabs.video import _saved_landmarks_for  # noqa: E402


def _seed_multi_animal_file(store, fid, session_dir="sessM"):
    """BCH062 on channel 0, BCH061 on channel 1 (one recording)."""
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


def test_saved_landmarks_reads_current_file_animal_draft(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed_multi_animal_file(store, 7)
    store.upsert_scoring_draft(
        7, "u@lab", "BCH062",
        [{"type": "LVF", "EO_sec": 3220.71, "racine": 4}])

    # Channel 0 -> BCH062: the saved onset comes back.
    lm = _saved_landmarks_for(store, 7, 0)
    assert len(lm) == 1 and lm[0]["EO_sec"] == 3220.71

    # Channel 1 -> BCH061 has no draft: no onsets bleed across animals.
    assert _saved_landmarks_for(store, 7, 1) == []
    # Out-of-range / non-animal channel: safe empty.
    assert _saved_landmarks_for(store, 7, 99) == []
    assert _saved_landmarks_for(store, 7, None) == []


def test_saved_landmarks_empty_when_no_draft(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed_multi_animal_file(store, 7)
    # No draft saved yet -> nothing to draw (not the previous file's onsets).
    assert _saved_landmarks_for(store, 7, 0) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
