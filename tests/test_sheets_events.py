"""Auto-collect scored seizures into the Google Sheet.

Covers the per-event Sheets writer (ensure_tab create-then-idempotent,
upsert_event_rows append-then-update via an in-memory fake Sheets API) and
the per-event row builder (_event_sheet_row: wall-clock onset + fields).

Run with: pytest tests/test_sheets_events.py -q
"""

from __future__ import annotations

import os
import re
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import sheets_write as sw  # noqa: E402

_HEADER = ["Date", "Onset (clock)", "Racine", "Type", "Light", "File"]
_KEY = ["File", "Onset (clock)"]


# --- in-memory fake of the google-api-python-client Sheets service --------- #
class _Req:
    def __init__(self, result=None, effect=None):
        self._result, self._effect = result, effect

    def execute(self):
        if self._effect:
            self._effect()
        return self._result


class _Values:
    def __init__(self, ss):
        self.ss = ss

    def get(self, spreadsheetId=None, range=None, valueRenderOption=None):
        tab = self.ss._tab(range)
        return _Req({"values": [list(r) for r in self.ss.data.get(tab, [])]})

    def update(self, spreadsheetId=None, range=None, valueInputOption=None,
               body=None):
        tab, start = self.ss._tab(range), self.ss._row(range)
        vals = body["values"]

        def eff():
            grid = self.ss.data.setdefault(tab, [])
            for k, rowvals in enumerate(vals):
                r = start + k
                while len(grid) < r:
                    grid.append([])
                grid[r - 1] = list(rowvals)
        return _Req({}, eff)

    def clear(self, spreadsheetId=None, range=None, body=None):
        tab, start = self.ss._tab(range), self.ss._row(range)

        def eff():
            self.ss.data[tab] = self.ss.data.get(tab, [])[:start - 1]
        return _Req({}, eff)

    def append(self, spreadsheetId=None, range=None, valueInputOption=None,
               insertDataOption=None, body=None):
        tab = self.ss._tab(range)

        def eff():
            self.ss.data.setdefault(tab, []).extend(
                [list(r) for r in body["values"]])
        return _Req({}, eff)


class _Spreadsheets:
    def __init__(self, ss):
        self.ss = ss

    def get(self, spreadsheetId=None):
        return _Req({"sheets": [{"properties": {"title": t}}
                                 for t in self.ss.data]})

    def batchUpdate(self, spreadsheetId=None, body=None):
        def eff():
            for req in (body or {}).get("requests", []):
                title = (req.get("addSheet", {}).get("properties", {})
                          .get("title"))
                if title:
                    self.ss.data.setdefault(title, [])
        return _Req({}, eff)

    def values(self):
        return _Values(self.ss)


class FakeSheets:
    def __init__(self, initial=None):
        self.data = {t: [list(r) for r in rows]
                     for t, rows in (initial or {}).items()}

    def spreadsheets(self):
        return _Spreadsheets(self)

    @staticmethod
    def _tab(rng):
        s = rng.split("!")[0].strip()
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1].replace("''", "'")
        return s

    @staticmethod
    def _row(rng):
        return int(re.search(r"(\d+)", rng.split("!")[1]).group(1))


def test_ensure_tab_creates_then_idempotent():
    svc = FakeSheets(initial={"BCH062": [["Date"]]})     # summary tab exists
    assert sw.ensure_tab(svc, "SID", "BCH062 Events", _HEADER) is True
    assert svc.data["BCH062 Events"][0] == _HEADER       # header written
    # Second call: already exists -> no create, header untouched.
    assert sw.ensure_tab(svc, "SID", "BCH062 Events", _HEADER) is False
    assert len(svc.data["BCH062 Events"]) == 1


def test_upsert_event_rows_append_then_update(monkeypatch):
    svc = FakeSheets()
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    row = {"Date": "2026-06-04", "Onset (clock)": "2026-06-04 04:34:40.710",
           "Racine": 4, "Type": "LVF", "Light": 2, "File": "rec.mat"}

    res = sw.upsert_event_rows("sa.json", "SID",
                               {"BCH062 Events": [row]}, _KEY, _HEADER)
    assert res["appended"] == 1 and res["updated"] == 0
    assert "BCH062 Events" in res["created_tabs"]
    body = svc.data["BCH062 Events"]
    assert body[0] == _HEADER and len(body) == 2          # header + 1 row
    assert body[1][_HEADER.index("Racine")] == 4

    # Re-upsert the SAME event (same File + Onset) with a new Racine -> UPDATE
    # in place, not a duplicate row.
    res2 = sw.upsert_event_rows(
        "sa.json", "SID", {"BCH062 Events": [{**row, "Racine": 5}]},
        _KEY, _HEADER)
    assert res2["updated"] == 1 and res2["appended"] == 0
    assert res2["created_tabs"] == []                     # tab already there
    body = svc.data["BCH062 Events"]
    assert len(body) == 2                                  # still one data row
    assert body[1][_HEADER.index("Racine")] == 5


def test_event_sheet_row_wall_clock_and_fields():
    from src.dashboard.tabs.event_verification import _event_sheet_row
    file_meta = {"Peak_Date": "2026-06-04", "Peak_Time": "03:41:00",
                 "Peak_Index": None, "filename": "rec.mat"}
    event = {"type": "LVF", "EO_sec": 3220.71, "racine": 4, "light": 2}
    r = _event_sheet_row(event, file_meta, 20000.0)
    assert r["Date"] == "2026-06-04"
    # 03:41:00 + 3220.71 s = 04:34:40.710
    assert r["Onset (clock)"] == "2026-06-04 04:34:40.710"
    assert r["Racine"] == 4 and r["Type"] == "LVF" and r["Light"] == 2
    assert r["File"] == "rec.mat"
    # Unscored / no-onset fields degrade to "".
    blank = _event_sheet_row({"type": "", "EO_sec": None}, file_meta, 20000.0)
    assert blank["Onset (clock)"] == "" and blank["Racine"] == ""


def test_canon_key_serial_vs_iso():
    # Google date serial 46168 == 2026-05-26.
    assert sw._canon_key(46168) == "2026-05-26"
    assert sw._canon_key("2026-05-26") == "2026-05-26"
    # Datetime serial + string collapse to the same second.
    assert sw._canon_key("2026-03-02 17:20:31.710") == "2026-03-02 17:20:31"
    # Non-dates fall back to _norm (racine stays comparable).
    assert sw._canon_key(4) == sw._norm(4)
    assert sw._canon_key("rec.mat") == sw._norm("rec.mat")


def test_upsert_idempotent_across_date_formats(monkeypatch):
    # The tab already stores the date as a Google SERIAL (46168), the way
    # USER_ENTERED writes it -- an ISO-string upsert must UPDATE, not append.
    svc = FakeSheets(initial={
        "BCH062": [["Date", "Number of Behavioral Events"], [46168, 3]]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    res = sw.upsert_day_rows(
        "sa.json", "SID",
        {"BCH062": [{"date": "2026-05-26", "n_events": 7}]}, ["Date"])
    assert res["updated"] == 1 and res["appended"] == 0
    assert len(svc.data["BCH062"]) == 2          # header + 1 row, no dup
    assert svc.data["BCH062"][1][1] == 7         # n_events updated in place


def test_upsert_day_rows_creates_new_animal_tab(monkeypatch):
    # BCH062 exists (its header is the template); BCH999 is a new animal.
    tmpl = ["Date", "MouseID", "Number of Behavioral Events"]
    svc = FakeSheets(initial={"BCH062": [tmpl]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    res = sw.upsert_day_rows(
        "sa.json", "SID",
        {"BCH999": [{"date": "2026-07-01", "animal": "BCH999",
                     "n_events": 2}]},
        ["Date"], create_missing=True)
    assert "BCH999" in res["created_tabs"] and res["appended"] == 1
    assert svc.data["BCH999"][0] == tmpl               # header copied
    body = svc.data["BCH999"][1]
    assert body[0] == "2026-07-01"                      # Date
    assert body[2] == 2                                 # mapped n_events


def test_upsert_day_rows_skips_when_create_missing_false(monkeypatch):
    svc = FakeSheets(initial={"BCH062": [["Date"]]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    res = sw.upsert_day_rows(
        "sa.json", "SID", {"BCH999": [{"date": "2026-07-01"}]},
        ["Date"], create_missing=False)
    assert res["skipped_tabs"] == ["BCH999"]
    assert res["created_tabs"] == [] and "BCH999" not in svc.data


def test_dedupe_tab_collapses_same_date(monkeypatch):
    svc = FakeSheets(initial={"BCH062": [
        ["Date", "# Seizures", "Notes"],
        ["2026-05-27", 5, ""],          # first-seen (lab-ish, real count)
        ["2026-05-27", 0, "hand"],       # dup of same date; different cols
        ["2026-05-28", 1, ""],
    ]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    dry = sw.dedupe_tab(svc, "SID", "BCH062", ["Date"], dry_run=True)
    assert dry == {"tab": "BCH062", "rows_before": 3,
                   "rows_after": 2, "removed": 1}
    assert len(svc.data["BCH062"]) == 4               # unchanged (dry run)
    res = sw.dedupe_tab(svc, "SID", "BCH062", ["Date"], dry_run=False)
    assert res["removed"] == 1
    body = svc.data["BCH062"][1:]
    assert len(body) == 2
    # 2026-05-27 merged: keep first-seen 5, fill Notes from the dup.
    assert body[0] == ["2026-05-27", 5, "hand"]
    assert body[1] == ["2026-05-28", 1, ""]


def test_dedupe_tab_protects_keys(monkeypatch):
    svc = FakeSheets(initial={"BCH040": [
        ["Date", "# Seizures", "RecFile"],
        ["2026-03-19", 0, ""],            # my dup (day row)
        ["2026-03-19", 2, "rec_v1.mat"],   # genuine per-event row -> protect
        ["2026-05-01", 0, ""],            # my dup
        ["2026-05-01", 0, ""],            # my dup, same date, no per-event
    ]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    res = sw.dedupe_tab(svc, "SID", "BCH040", ["Date"], dry_run=False,
                        protect_keys={"2026-03-19"})
    body = svc.data["BCH040"][1:]
    assert res["removed"] == 1
    # Protected date keeps BOTH rows; the other date collapses to one.
    assert sum(1 for r in body if r[0] == "2026-03-19") == 2
    assert sum(1 for r in body if r[0] == "2026-05-01") == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
