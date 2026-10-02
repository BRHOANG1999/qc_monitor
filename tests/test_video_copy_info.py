"""Video Review 'copy file info (CSV row)' button: the store summary + CSV helper.

Run: pytest tests/test_video_copy_info.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                                   # noqa: E402
from src.dashboard.tabs.video import (_video_info_csv,           # noqa: E402
                                      _VIDEO_INFO_COLUMNS)


def _seed(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "session_name, chunk_datetime, duration_sec) VALUES (?,?,?,?,?,?)",
            (1, "G:/data/sessA/rec_001.mat", "G:/data/2026_09_30__sessA",
             "sessA", "2026_09_30__12_00_00", 3725.0))
        conn.execute(
            "INSERT INTO session_config (session_dir, channel_names, "
            "discovered_at) VALUES (?,?,?)",
            ("G:/data/2026_09_30__sessA",
             json.dumps(["BCH062SR", "BCH062SLM", "stimCopy"]),
             "2026-09-30T12:00:00"))
        conn.commit()
    return store


def test_summary_fields(tmp_path):
    store = _seed(tmp_path)
    s = store.video_file_summary(1, "BCH062", channel=1)
    assert s["session_dir"] == "G:/data/2026_09_30__sessA"
    assert s["session_name"] == "sessA"
    assert s["chunk_datetime"] == "2026_09_30__12_00_00"
    assert s["duration_sec"] == 3725.0
    assert s["channel"] == 1 and s["channel_name"] == "BCH062SLM"
    assert s["animal_id"] == "BCH062"


def test_summary_missing_file(tmp_path):
    store = _seed(tmp_path)
    s = store.video_file_summary(999, "BCH062")
    assert s["file_id"] == 999
    assert s["session_dir"] == "" and s["file_path"] == ""


def test_summary_channel_out_of_range(tmp_path):
    store = _seed(tmp_path)
    s = store.video_file_summary(1, "BCH062", channel=99)
    assert s["channel_name"] == ""          # no crash, just blank


def test_csv_header_toggle_and_quoting():
    row = {c: "" for c in _VIDEO_INFO_COLUMNS}
    row.update({"animal": "BCH062", "session": "2026_09_30__sess,A",
                "file_id": 1, "channel": 1, "channel_name": "BCH062SLM",
                "chunk_datetime": "2026_09_30__12_00_00", "duration": "1:02:05",
                "file_path": "G:/data/sessA/rec.mat"})
    import csv as _csv
    with_h = _video_info_csv(row, True)
    lines = with_h.split("\n")
    assert len(lines) == 2
    assert lines[0].split(",")[0] == "animal"       # header present
    # a session with a comma must be quoted, so the data line is not over-split
    parsed = next(_csv.reader([lines[1]]))
    assert parsed == [str(row[c]) for c in _VIDEO_INFO_COLUMNS]
    assert "2026_09_30__sess,A" in parsed

    no_h = _video_info_csv(row, False)
    assert "\n" not in no_h                          # single data row, no header
    assert next(_csv.reader([no_h]))[0] == "BCH062"


def test_csv_handles_none_values():
    row = {c: None for c in _VIDEO_INFO_COLUMNS}
    row["animal"] = "BCH062"
    out = _video_info_csv(row, False)
    cells = next(__import__("csv").reader([out]))
    assert cells[0] == "BCH062"
    assert all(c == "" for c in cells[1:])           # None -> empty cell


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
