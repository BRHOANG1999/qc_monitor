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


def test_csv_preview_line_reflects_structured_event():
    """The Submit preview shows the structured event as it would hit the CSV
    (not the legacy '0 markers'), including em-dashes for unset fields."""
    from src.dashboard.tabs.video import _csv_preview_line, _fmt_sec
    from src.dashboard.tabs.video_events import blank_event
    assert _fmt_sec(3220.71) == "3220.7s"
    assert _fmt_sec(None) == "—" and _fmt_sec("") == "—"
    scored = {**blank_event(), "type": "LVF", "EO_sec": 3220.71,
              "racine": 4, "light": 2}
    line = _csv_preview_line(1, scored)
    assert "Onset=LVF" in line and "EO=3220.7s" in line
    assert "Score=4" in line and "Light=2" in line
    # An EO-only draft (the screenshot case): onset shows, rest are dashes.
    eo_only = {**blank_event(), "type": "LVF", "EO_sec": 3220.71}
    l2 = _csv_preview_line(1, eo_only)
    assert "EO=3220.7s" in l2 and "Score=—" in l2 and "Light=—" in l2


def test_chunk_start_dt_parses_both_formats(tmp_path):
    from datetime import datetime
    from src.dashboard.tabs.video import _chunk_start_dt
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed_multi_animal_file(store, 7)   # chunk_datetime '2026_01_01__00_00_00'
    assert _chunk_start_dt(store, 7) == datetime(2026, 1, 1, 0, 0, 0)
    with store.connection() as conn:
        conn.execute("INSERT INTO processed_files (id, file_path, "
                     "chunk_datetime) VALUES (8, '/f/8.mat', "
                     "'2026-06-04T03:41:00')")          # ISO variant
        conn.execute("INSERT INTO processed_files (id, file_path, "
                     "chunk_datetime) VALUES (9, '/f/9.mat', NULL)")
        conn.commit()
    assert _chunk_start_dt(store, 8) == datetime(2026, 6, 4, 3, 41, 0)
    assert _chunk_start_dt(store, 9) is None            # unparseable -> None


def test_all_files_for_animal_is_pool_independent(tmp_path):
    # 'Show all timestamps' lists every recording for the animal regardless
    # of review pool.
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed_multi_animal_file(store, 7)                       # BCH062 + BCH061
    _seed_multi_animal_file(store, 8, session_dir="sessM2")
    assert sorted(r["id"] for r in store.all_files_for_animal("BCH062")) \
        == [7, 8]
    # A review state on one file doesn't change the all-files listing.
    store.upsert_scoring_draft(7, "u@lab", "BCH062",
                               [{"type": "LVF", "EO_sec": 1.0, "racine": 3}])
    assert sorted(r["id"] for r in store.all_files_for_animal("BCH062")) \
        == [7, 8]
    assert store.all_files_for_animal("BCH061")             # both animals map
    assert store.all_files_for_animal("BCH999") == []       # unknown -> none


def test_wall_clock_axis_labels():
    from datetime import datetime
    import plotly.graph_objects as go
    from src.dashboard.tabs.video import _apply_mmss_xaxis
    fig = go.Figure()
    _apply_mmss_xaxis(fig, 3600.0, start_dt=datetime(2026, 6, 4, 3, 41, 0))
    assert list(fig.layout.xaxis.ticktext)[0] == "03:41:00"
    # Without a start time we keep mm:ss elapsed (other views unchanged).
    fig2 = go.Figure()
    _apply_mmss_xaxis(fig2, 3600.0)
    assert list(fig2.layout.xaxis.ticktext)[0] == "00:00"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
