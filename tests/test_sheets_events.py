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

    def batchUpdate(self, spreadsheetId=None, body=None):
        def eff():
            for d in body["data"]:
                tab, start = self.ss._tab(d["range"]), self.ss._row(d["range"])
                grid = self.ss.data.setdefault(tab, [])
                for k, rv in enumerate(d["values"]):
                    r = start + k
                    while len(grid) < r:
                        grid.append([])
                    grid[r - 1] = list(rv)
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
        # Tab ORDER is the dict order; sheetId is stable across renames.
        return _Req({"sheets": [
            {"properties": {"title": t, "sheetId": self.ss.sids[t],
                             "index": i}}
            for i, t in enumerate(self.ss.data)]})

    def batchUpdate(self, spreadsheetId=None, body=None):
        def eff():
            for req in (body or {}).get("requests", []):
                if "addSheet" in req:
                    title = req["addSheet"]["properties"]["title"]
                    self.ss.data.setdefault(title, [])
                    self.ss.sids.setdefault(title, self.ss._new_id())
                elif "updateSheetProperties" in req:
                    up = req["updateSheetProperties"]
                    props, fields = up["properties"], up.get("fields", "")
                    title = self.ss._title_of(props["sheetId"])
                    if "title" in fields:
                        self.ss._rename(title, props["title"])
                    if "index" in fields:
                        self.ss._move(title, props["index"])
        return _Req({}, eff)

    def values(self):
        return _Values(self.ss)


class FakeSheets:
    def __init__(self, initial=None):
        self.data = {t: [list(r) for r in rows]
                     for t, rows in (initial or {}).items()}
        self._next_id = 1
        self.sids = {t: self._new_id() for t in self.data}

    def spreadsheets(self):
        return _Spreadsheets(self)

    # -- helpers the fake's batchUpdate leans on ------------------------- #
    def _new_id(self):
        i, self._next_id = self._next_id, self._next_id + 1
        return i

    def _title_of(self, sheet_id):
        for t, i in self.sids.items():
            if i == sheet_id:
                return t
        raise KeyError(sheet_id)

    def _rename(self, old, new):
        """Rename in place -- a real rename keeps the tab's position + id."""
        self.data = {(new if t == old else t): r
                     for t, r in self.data.items()}
        self.sids = {(new if t == old else t): i
                     for t, i in self.sids.items()}

    def _move(self, title, index):
        order = [t for t in self.data if t != title]
        order.insert(index, title)
        self.data = {t: self.data[t] for t in order}

    def order(self):
        return list(self.data)

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


def test_event_sheet_row_status():
    from src.dashboard.tabs.event_verification import (
        _event_sheet_row, _STATUS_NEEDS)
    file_meta = {"Peak_Date": "2026-06-04", "Peak_Time": "03:41:00",
                 "Peak_Index": None, "filename": "rec.mat"}
    event = {"type": "LVF", "EO_sec": 10.0, "racine": 3}
    r = _event_sheet_row(event, file_meta, 20000.0, status=_STATUS_NEEDS)
    assert r["Status"] == _STATUS_NEEDS
    # Default (no status arg) -> blank Status cell (approved backfill etc.).
    assert _event_sheet_row(event, file_meta, 20000.0)["Status"] == ""


def test_ensure_header_columns_appends_missing():
    # A tab created before the "Status" column: extension adds it at the end,
    # existing columns/order untouched, and it's idempotent.
    old = ["Date", "Onset (clock)", "Racine", "Type", "Light", "File"]
    svc = FakeSheets(initial={"BCH062 Events": [
        list(old), ["2026-06-04", "x", 3, "LVF", 2, "rec.mat"]]})
    added = sw._ensure_header_columns(svc, "SID", "BCH062 Events",
                                       old + ["Status"])
    assert added == 1
    assert svc.data["BCH062 Events"][0] == old + ["Status"]
    assert sw._ensure_header_columns(svc, "SID", "BCH062 Events",
                                      old + ["Status"]) == 0


def test_upsert_event_rows_extends_old_header(monkeypatch):
    # Existing Events tab has the OLD 6-column header. Upserting the same
    # event (File+Onset) with a Status value must extend the header and write
    # Status in place, not drop it or duplicate the row.
    from src.dashboard.tabs.event_verification import (
        _EVENT_SHEET_HEADER, _EVENT_SHEET_KEY, _STATUS_NEEDS)
    old = ["Date", "Onset (clock)", "Racine", "Type", "Light", "File"]
    svc = FakeSheets(initial={"BCH062 Events": [
        list(old), ["2026-06-04", "2026-06-04 04:34:40.710", 4, "LVF", 2,
                    "rec.mat"]]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    row = {"Date": "2026-06-04", "Onset (clock)": "2026-06-04 04:34:40.710",
           "Racine": 4, "Type": "LVF", "Light": 2, "File": "rec.mat",
           "Status": _STATUS_NEEDS}
    res = sw.upsert_event_rows("sa.json", "SID", {"BCH062 Events": [row]},
                               _EVENT_SHEET_KEY, _EVENT_SHEET_HEADER)
    assert res["updated"] == 1 and res["appended"] == 0
    hdr = svc.data["BCH062 Events"][0]
    assert "Status" in hdr and len(svc.data["BCH062 Events"]) == 2
    assert svc.data["BCH062 Events"][1][hdr.index("Status")] == _STATUS_NEEDS


# --- tab housekeeping: rename + canonical order + the pending column ------ #

def test_target_tab_order_groups_and_sorts():
    # Insertion order is a mess; the flow is: other (relative order kept),
    # then summaries sorted, then events sorted.
    got = sw._target_tab_order(
        ["BCH062 Daily", "Fall 25", "BCH040 Events", "BCH040 Daily",
         "BCH062 Events", "BCH039 Daily"], " Daily", " Events")
    assert got == ["Fall 25",
                   "BCH039 Daily", "BCH040 Daily", "BCH062 Daily",
                   "BCH040 Events", "BCH062 Events"]


def test_rename_tab_is_idempotent():
    svc = FakeSheets(initial={"BCH111": [["Date"]], "Fall 25": [["Folder"]]})
    assert sw.rename_tab(svc, "SID", "BCH111", "BCH111 Daily") is True
    assert "BCH111 Daily" in svc.data and "BCH111" not in svc.data
    # Re-run: source gone -> no-op, not an error (migration is re-runnable).
    assert sw.rename_tab(svc, "SID", "BCH111", "BCH111 Daily") is False


def test_reorder_tabs_applies_canonical_order():
    svc = FakeSheets(initial={
        "BCH062 Daily": [["Date"]], "Fall 25": [["Folder"]],
        "BCH040 Events": [["Date"]], "BCH040 Daily": [["Date"]]})
    final = sw.reorder_tabs(svc, "SID", " Daily", " Events")
    assert final == ["Fall 25", "BCH040 Daily", "BCH062 Daily",
                     "BCH040 Events"]
    assert svc.order() == final     # the fake actually moved the sheets


def test_reorder_tabs_noop_on_empty_summary_suffix():
    # Every title endswith "" -> classification would collapse; must bail out.
    svc = FakeSheets(initial={"BCH040": [["Date"]], "Fall 25": [["F"]]})
    assert sw.reorder_tabs(svc, "SID", "", " Events") == []


def test_upsert_day_rows_ensure_columns_adds_pending(monkeypatch):
    # A lab summary tab with no pending column: the column is APPENDED and the
    # hand-filled cell ("hand") survives the merge-update.
    svc = FakeSheets(initial={"BCH111 Daily": [
        ["Date", "Number of Behavioral Events", "Notes"],
        ["2026-07-02", 0, "hand"]]})
    monkeypatch.setattr(sw, "_sheets_api_rw", lambda _p: svc)
    res = sw.upsert_day_rows(
        "sa.json", "SID",
        {"BCH111 Daily": [{"date": "2026-07-02", "n_events": 0,
                            "n_pending_events": 4}]},
        ["Date"], ensure_columns=["Events Needing Onsets"])
    assert res["updated"] == 1 and res["appended"] == 0
    hdr = svc.data["BCH111 Daily"][0]
    assert hdr == ["Date", "Number of Behavioral Events", "Notes",
                   "Events Needing Onsets"]
    row = svc.data["BCH111 Daily"][1]
    assert row[hdr.index("Events Needing Onsets")] == 4
    assert row[hdr.index("Number of Behavioral Events")] == 0   # approved tally
    assert row[hdr.index("Notes")] == "hand"                    # not blanked


def test_resolve_summary_header_skips_events_tabs():
    # An Events tab also has a "Date" column -- it must NOT be picked as the
    # header template for a brand-new animal's summary tab.
    svc = FakeSheets(initial={
        "AAA Events": [["Date", "Onset (clock)", "Racine"]],
        "BCH040 Daily": [["Date", "MouseID", "Number of Behavioral Events"]]})
    hdr = sw._resolve_summary_header(
        svc, "SID", set(svc.data), None, exclude_suffix=" Events")
    assert hdr == ["Date", "MouseID", "Number of Behavioral Events"]
    # Without the exclusion the Events tab wins the alphabetical scan (the bug).
    assert sw._resolve_summary_header(
        svc, "SID", set(svc.data), None)[1] == "Onset (clock)"


# --- lab-column correctness: text coercion, 1-based channels, protocol ---- #

def test_force_text_guards_date_shaped_values():
    # These are the real corruptions: Sheets read "2, 3" as Feb 3 (serial
    # 46056) and "3/4" as Mar 4 (45720) under USER_ENTERED.
    assert sw.force_text("2, 3") == "'2, 3"
    assert sw.force_text("3/4") == "'3/4"
    # Plain integers can't be misread as a date -> left numeric-looking, so
    # they match the lab's own hand-typed cells.
    assert sw.force_text("2") == "2"
    assert sw.force_text("") == "" and sw.force_text(None) == ""


def test_protocol_from_session_matches_lab_labels():
    from src.dashboard.tabs.event_verification import _protocol_from_session
    f = _protocol_from_session
    assert f("//host/db/MARCH_2026\\20260318_stimBaseline__stimCopy_BCH040SR_"
             "stimCopy_saline_") == "stimBaseline"
    assert f("baseline__BCH061SLM_BCH061SR_") == "baseline"
    assert f("chronicEvoked_15-4nC_150us__stimCopy-BCH040SR_") == "chronicEvoked"
    assert f("20240206_baseline__BCH039stimSRRecSLM_NULL_") == "baseline"
    assert f("baseline-salineTest__x_") == "baseline-salineTest"
    assert f("") == ""
    # Typo'd 9-digit date prefix -- a \d{8} strip would report the DATE as the
    # protocol.
    assert f("G:\\BHZ\\JANUARY_2026\\292601012_stimBaseline-salineTest__x_") \
        == "stimBaseline-salineTest"
    # Deleted recordings linger as $RECYCLE.BIN paths -- never a protocol.
    assert f("//host/database/$RECYCLE.BIN\\S-1-5-21-1005\\$RD7H40J") == ""


def test_session_meta_joins_multi_protocol_days_the_lab_way():
    from src.dashboard.tabs.event_verification import _session_meta
    st = _FakeStore([{"channel_index": 1, "location": "SR"}])
    meta = _session_meta(st, "BCH040",
                          ["20260101_baseline__x_", "20260101_stimBaseline__x_"])
    # The lab writes "baseline + stim", not "baseline, stim".
    assert meta["type_of_recording"] == "baseline + stimBaseline"


class _FakeStore:
    """Minimal store for _session_meta: one animal on two 0-based channels."""
    def __init__(self, electrodes, cfg=None):
        self._e, self._cfg = electrodes, cfg or {}

    def electrodes_for_animal_in_session(self, sd, animal):
        return self._e

    def get_session_config(self, sd):
        return self._cfg


def test_session_meta_channels_are_1_based_and_text():
    from src.dashboard.tabs.event_verification import _session_meta
    # channel_index is the 0-based position in channel_names; the lab counts
    # from 1. 0-based (1, 2) must surface as "2, 3" -- and as TEXT, since
    # USER_ENTERED would otherwise store "2, 3" as the date Feb 3.
    st = _FakeStore([{"channel_index": 1, "location": "SR"},
                     {"channel_index": 2, "location": "SLM"}])
    meta = _session_meta(st, "BCH040", ["20260318_stimBaseline__x_"])
    assert meta["channels"] == "'2, 3"          # 1-based + force-text
    assert meta["recording_location"] == "SR, SLM"
    assert meta["type_of_recording"] == "stimBaseline"
    # Single channel stays a bare number (can't be misread as a date).
    st1 = _FakeStore([{"channel_index": 1, "location": "SR"}])
    assert _session_meta(st1, "BCH040", ["baseline__x_"])["channels"] == "2"
    # A 0-based "channel 0" -- impossible in the lab's notation -- becomes 1.
    st0 = _FakeStore([{"channel_index": 0, "location": "SR"}])
    assert _session_meta(st0, "BCH040", ["baseline__x_"])["channels"] == "1"


def test_day_stats_counts_pending_seizures():
    # A needs_scoring event is a CONFIRMED seizure (scored from video, missing
    # only the remaining EEG landmarks) -- it must count toward "Number of
    # Behavioral Events" and Max Racine, not sit in a separate bucket. Before
    # this, every BCH062 day read 0 events while Events Needing Onsets read 5-7.
    import datetime
    from src.dashboard.tabs.event_verification import _day_stats
    st = _FakeStore([{"channel_index": 1, "location": "SR"}])
    pending = [{"file_id": 1, "session_dir": "20260527_stimTest-40nC__x_",
                "user_email": "a@b.c",
                "events": [{"EO_sec": 10.0, "racine": 4},
                            {"EO_sec": 20.0, "racine": 3},
                            {"EO_sec": None, "racine": 2}]}]   # no onset
    row = _day_stats(st, "BCH062", datetime.date(2026, 5, 27), [],
                      "20260527_BCH062.csv", pending_items=pending)
    assert row["n_events"] == 2           # the two real onsets are counted
    assert row["n_pending_events"] == 2   # ...and both still need landmarks
    assert row["max_racine"] == 4         # Racine picked up from pending too
    assert row["type_of_recording"] == "stimTest-40nC"


def test_repair_tool_sheet_row_mapping():
    # REGRESSION: this off-by-one wrote/deleted the NEIGHBOURING row and
    # damaged PM's hand-entered rows in BCH040/052/053. grid[0] is the header
    # (sheet row 1), so grid[1] -- the first DATA row -- is sheet row 2.
    from tools.sheets_repair_channels import _sheet_row, _recover
    assert _sheet_row(1) == 2      # NOT 3
    assert _sheet_row(2) == 3
    with pytest.raises(AssertionError):
        _sheet_row(0)              # the header is not a data row
    # _recover strips a year the cell's date format tacked on ("3, 4, 2005").
    assert _recover(38415, "3, 4, 2005") == "3, 4"
    assert _recover(46056, "2, 3") == "2, 3"
    assert _recover(45720, "3/4") == "3/4"
    assert _recover(46056, "not a channel list") == ""   # skipped, not written


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
