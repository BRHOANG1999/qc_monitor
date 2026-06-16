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
    def __init__(self, grid, titles=("BCH062",)):
        self._values = _FakeValues(grid)
        self._titles = list(titles)

    def spreadsheets(self):
        outer = self

        class _SS:
            def values(self_inner):
                return outer._values

            def get(self_inner, spreadsheetId):
                class _Exec:
                    def execute(self_x):
                        return {"sheets": [
                            {"properties": {"title": t}}
                            for t in outer._titles]}
                return _Exec()
        return _SS()


def test_upsert_merges_existing_preserves_human_cols(monkeypatch):
    # Header has a human-maintained column ("Notes") we never touch.
    grid = [
        ["Date", "# Seizures", "Notes"],
        ["2026-06-11", "1", "hand-typed"],   # existing -> merge-update
    ]
    fake = _FakeSheets(grid, titles=("BCH062",))
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda sa: fake)

    rows_by_tab = {
        "BCH062": [
            {"date": "2026-06-11", "n_events": 3},   # update
            {"date": "2026-06-12", "n_events": 2},   # append
        ]
    }
    res = sw.upsert_day_rows(
        "sa.json", "sheet123", rows_by_tab, ["Date"])
    assert res["updated"] == 1 and res["appended"] == 1
    # Update merged: Date+#Seizures written, Notes left as the human
    # value (not blanked).
    assert fake._values.updates[0][0] == "'BCH062'!A2"
    assert fake._values.updates[0][1] == [
        ["2026-06-11", 3, "hand-typed"]]
    # Append: unmatched cols blank for a brand-new row.
    assert fake._values.appends[0] == [["2026-06-12", 2, ""]]


def test_upsert_skips_missing_animal_tab(monkeypatch):
    fake = _FakeSheets([["Date", "# Seizures"]], titles=("BCH062",))
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda sa: fake)
    res = sw.upsert_day_rows(
        "sa.json", "s", {"BCH999": [{"date": "x", "n_events": 1}]},
        ["Date"])
    assert res["skipped_tabs"] == ["BCH999"]
    assert res["updated"] == 0 and res["appended"] == 0


def test_upsert_raises_on_missing_key_column(monkeypatch):
    fake = _FakeSheets([["Foo", "Bar"]], titles=("BCH062",))
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda sa: fake)
    with pytest.raises(ValueError):
        sw.upsert_day_rows("sa.json", "s",
                            {"BCH062": [{"date": "x"}]}, ["Date"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
