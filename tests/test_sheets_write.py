"""Phase 4: Google-Sheet daily upsert.

Mocks the Sheets service so no live API call happens.

Run with: pytest tests/test_sheets_write.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import sheets_write as sw  # noqa: E402


def test_norm():
    assert sw._norm("# Seizures!") == "seizures"
    assert sw._norm("Max Racine") == "maxracine"


def test_map_row_to_header_aliases():
    header = ["Date", "Animal", "# Seizures", "Max Racine", "Notes"]
    stats = {"date": "2026-06-11", "animal": "BCH062",
              "n_events": 3, "max_racine": 5}
    row = sw.map_row_to_header(header, stats)
    # "# Seizures" -> n_events alias; "Notes" unknown -> blank.
    assert row == ["2026-06-11", "BCH062", 3, 5, ""]


def test_map_row_to_header_column_map_override():
    header = ["Day", "Subject", "Count"]
    stats = {"date": "2026-06-11", "animal": "BCH062", "n_events": 3}
    # "Count" isn't a known alias; pin it explicitly.
    row = sw.map_row_to_header(
        header, stats, column_map={"Count": "n_events"})
    assert row == ["2026-06-11", "BCH062", 3]


# --------------------------------------------------------------- #
# Upsert against a fake Sheets service
# --------------------------------------------------------------- #

class _FakeValues:
    def __init__(self, grid):
        self.grid = grid
        self.updates: list = []
        self.appends: list = []

    def get(self, spreadsheetId, range):
        outer = self

        class _Exec:
            def execute(self_inner):
                return {"values": outer.grid}
        return _Exec()

    def update(self, spreadsheetId, range, valueInputOption, body):
        self.updates.append((range, body["values"]))

        class _Exec:
            def execute(self_inner):
                return {}
        return _Exec()

    def append(self, spreadsheetId, range, valueInputOption,
                insertDataOption, body):
        self.appends.append(body["values"])

        class _Exec:
            def execute(self_inner):
                return {}
        return _Exec()


class _FakeSheets:
    def __init__(self, grid):
        self._values = _FakeValues(grid)

    def spreadsheets(self):
        outer = self

        class _SS:
            def values(self_inner):
                return outer._values
        return _SS()


def test_upsert_updates_existing_and_appends_new(monkeypatch):
    grid = [
        ["Date", "Animal", "# Seizures"],
        ["2026-06-11", "BCH062", "1"],   # existing -> update
    ]
    fake = _FakeSheets(grid)
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda sa: fake)

    rows = [
        {"date": "2026-06-11", "animal": "BCH062", "n_events": 3},
        {"date": "2026-06-12", "animal": "BCH062", "n_events": 2},
    ]
    res = sw.upsert_day_rows(
        "sa.json", "sheet123", "Tab", ["Date", "Animal"], rows)
    assert res == {"updated": 1, "appended": 1}
    # The existing (2026-06-11, BCH062) row was updated in place at A2.
    assert fake._values.updates[0][0] == "Tab!A2"
    assert fake._values.updates[0][1] == [["2026-06-11", "BCH062", 3]]
    # The new day was appended.
    assert fake._values.appends[0] == [["2026-06-12", "BCH062", 2]]


def test_upsert_raises_on_missing_key_column(monkeypatch):
    fake = _FakeSheets([["Foo", "Bar"]])
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda sa: fake)
    with pytest.raises(ValueError):
        sw.upsert_day_rows("sa.json", "s", "T", ["Date", "Animal"],
                            [{"date": "x", "animal": "y"}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
