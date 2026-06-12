"""Tests for round-2 scoring additions: Light column, Undefined onset,
and event-completeness for the new type.

Run with: pytest tests/test_event_scoring_round2.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import bhz_csv  # noqa: E402
from src.dashboard.tabs.video_events import (  # noqa: E402
    blank_event,
    is_event_complete,
    required_fields,
)

FS = 20000.0
META = {"Target": "LFP"}


def _full_event(etype: str) -> dict:
    e = blank_event()
    e["type"] = etype
    for f in ("EO", "LAS", "BO", "PID", "BB"):
        e[f"{f}_sec"] = 1.0
    e["racine"] = 3
    return e


def test_light_maps_to_csv_column():
    e = _full_event("LVF")
    e["light"] = 2
    row = bhz_csv._event_to_row(e, META, FS)
    assert row["Light"] == 2


def test_light_none_when_unset():
    row = bhz_csv._event_to_row(_full_event("LVF"), META, FS)
    assert row["Light"] is None


def test_undefined_onset_string():
    row = bhz_csv._event_to_row(_full_event("Undefined"), META, FS)
    assert row["Onset"] == bhz_csv.ONSET_UNDEFINED == "Undefined"


def test_lvf_and_hyp_onset_strings():
    assert bhz_csv._event_to_row(
        _full_event("LVF"), META, FS)["Onset"] == bhz_csv.ONSET_LVF
    hyp = _full_event("HYP")
    assert bhz_csv._event_to_row(hyp, META, FS)["Onset"] == bhz_csv.ONSET_HYP


def test_video_quality_maps_through():
    e = _full_event("LVF")
    e["video_quality"] = "dim, bedding occlusion"
    assert bhz_csv._event_to_row(
        e, META, FS)["VideoQuality"] == "dim, bedding occlusion"


def test_undefined_uses_full_landmark_set():
    assert required_fields({"type": "Undefined"}) == (
        "EO", "LAS", "BO", "PID", "BB")
    assert required_fields({"type": "HYP"}) == ("EO", "BO", "PID", "BB")


def test_undefined_event_can_be_complete():
    assert is_event_complete(_full_event("Undefined")) is True
    # Missing LAS -> incomplete for Undefined (LVF set).
    e = _full_event("Undefined")
    e["LAS_sec"] = None
    assert is_event_complete(e) is False


def test_reject_peak_roundtrip(tmp_path):
    from src.db.store import Store
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path) "
            "VALUES (1, '/f/a.mat')")
        conn.commit()
    store.reject_peak(1, 0, 12.345, "pi@lab")
    store.reject_peak(1, 0, 99.0, "pi@lab")
    store.reject_peak(1, 0, 12.345, "pi@lab")  # idempotent
    assert store.get_rejected_peaks(1, 0) == [12.345, 99.0]
    # Channel scoping.
    assert store.get_rejected_peaks(1, 1) == []
    # Undo removes the most recent.
    assert store.unreject_last_peak(1, 0) is True
    assert store.get_rejected_peaks(1, 0) == [12.345]
    assert store.unreject_last_peak(1, 0) is True
    assert store.get_rejected_peaks(1, 0) == []
    assert store.unreject_last_peak(1, 0) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
