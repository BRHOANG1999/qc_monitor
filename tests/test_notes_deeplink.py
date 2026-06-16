"""Notes tab: video/LFP notes are openable and carry their moment.

Run with: pytest tests/test_notes_deeplink.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.tabs.annotations import (  # noqa: E402
    _start_sec_from_note, _table_data)


def test_start_sec_parses_video_review_prefix():
    # Video Review writes "[t=12.34s] ..." -- pull the seconds out.
    assert _start_sec_from_note("[t=12.34s] grooming") == 12.34
    assert _start_sec_from_note("[t=0s] start") == 0.0
    assert _start_sec_from_note("[t=600.5s] late event") == 600.5
    # Embedded later in the body still parses.
    assert _start_sec_from_note("note [t=5s] tail") == 5.0


def test_start_sec_defaults_to_zero_without_prefix():
    assert _start_sec_from_note("free-form note, no timestamp") == 0.0
    assert _start_sec_from_note("") == 0.0
    assert _start_sec_from_note(None) == 0.0  # type: ignore[arg-type]


def test_open_cell_only_for_file_linked_notes():
    rows = _table_data([
        {"id": 1, "note": "[t=3s] seizure", "file_id": 42,
         "session_dir": "/x/sessA", "category": "video_review"},
        {"id": 2, "note": "lab observation", "file_id": None,
         "session_dir": None, "category": "observation"},
    ])
    by_id = {r["id"]: r for r in rows}
    # File-linked note is openable; free-form note is not.
    assert by_id[1]["open"] == "▶ Open"
    assert by_id[2]["open"] == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
