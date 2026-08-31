"""Unit tests for src/chronic_stability/maintenance.py (parsing + classify)."""

from datetime import datetime

from src.chronic_stability import maintenance as mnt


def test_classify():
    assert mnt._classify("Battery change") == "battery"
    assert mnt._classify("Cage cleaning") == "cage"
    assert mnt._classify("Battery Replacement and Cage Cleaning") == "both"
    assert mnt._classify("Swapped electrode configurations") == "other"
    assert mnt._classify("") == "other"


def test_event():
    assert mnt._event("Recording_Stop") == "stop"
    assert mnt._event("Recording_Start") == "start"
    assert mnt._event("Manual") == "manual"


def test_notes_epoch_formats():
    a = mnt._notes_epoch("2026-07-20", "13:56:32")
    assert a == datetime(2026, 7, 20, 13, 56, 32).timestamp()
    b = mnt._notes_epoch("2026-07-29", "9:35")       # single-digit hour, no sec
    assert b == datetime(2026, 7, 29, 9, 35).timestamp()
    assert mnt._notes_epoch("", "13:00") is None
    assert mnt._notes_epoch("not-a-date", "x") is None


def test_sheet_epoch_formats():
    a = mnt._sheet_epoch("7/20/2026 12:20:09")
    assert a == datetime(2026, 7, 20, 12, 20, 9).timestamp()
    b = mnt._sheet_epoch("8/5/2026 11:15")
    assert b == datetime(2026, 8, 5, 11, 15).timestamp()
    assert mnt._sheet_epoch("") is None


def test_figure_lines_from_stops():
    ev = {"available": True, "software": [
        {"epoch": 1.0, "event": "stop", "kind": "battery"},
        {"epoch": 1.2, "event": "start", "kind": "battery"},   # start ignored
        {"epoch": 2.0, "event": "stop", "kind": "cage"},
        {"epoch": 3.0, "event": "stop", "kind": "both"},       # counts in both
    ]}
    lines = mnt.figure_lines(ev)
    assert lines["battery"] == [1.0, 3.0]
    assert lines["cage"] == [2.0, 3.0]


def test_figure_lines_unavailable():
    assert mnt.figure_lines({"available": False}) == {"battery": [], "cage": []}


def test_load_events_disabled():
    ev = mnt.load_events({"chronic_stability": {"maintenance": {"enabled": False}}},
                         0.0, 1.0)
    assert ev["available"] is False and ev["software"] == []
