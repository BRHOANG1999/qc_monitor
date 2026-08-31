"""sheets_write.set_header_notes + bhz_column_notes.annotate_all, via a fake
Sheets client (no network).

Run: pytest tests/test_bhz_column_notes.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import bhz_column_notes as bn  # noqa: E402
from src.utils import sheets_write as sw  # noqa: E402


class _Req:
    def __init__(self, r):
        self._r = r

    def execute(self):
        return self._r


class _Values:
    def __init__(self, headers_by_tab):
        self._h = headers_by_tab

    def get(self, spreadsheetId=None, range=None, **kw):
        tab = str(range).split("!")[0].strip("'")
        return _Req({"values": [self._h.get(tab, [])]})


class _Spreadsheets:
    def __init__(self, store, headers_by_tab, ids):
        self.store = store
        self._h = headers_by_tab
        self._ids = ids

    def get(self, spreadsheetId=None, **kw):
        return _Req({"sheets": [
            {"properties": {"title": t, "sheetId": i, "index": i}}
            for i, t in enumerate(self._ids)]})

    def values(self):
        return _Values(self._h)

    def batchUpdate(self, spreadsheetId=None, body=None):
        self.store.setdefault("batches", []).append(body)
        return _Req({})


class FakeSvc:
    def __init__(self, headers_by_tab):
        self.store = {}
        self._h = headers_by_tab
        self._ids = list(headers_by_tab)

    def spreadsheets(self):
        return _Spreadsheets(self.store, self._h, self._ids)


def test_set_header_notes_matches_by_norm_and_skips_unknown():
    svc = FakeSvc({"Summary": ["Overall Bh SZ rate", "MouseID", "Comments"]})
    n = sw.set_header_notes(svc, "SID", "Summary", bn.SUMMARY_NOTES)
    assert n == 1                                   # only the rate col matched
    req = svc.store["batches"][0]["requests"][0]["updateCells"]
    assert req["start"]["columnIndex"] == 0 and req["fields"] == "note"
    assert "recording-day" in req["rows"][0]["values"][0]["note"]


def test_annotate_all_routes_each_tab_to_its_notes():
    svc = FakeSvc({
        "Summary": ["Overall Bh SZ rate"],
        "BCH039 Daily": ["Number of Behavioral Events", "Comments"],
        "BCH039 Events": ["Onset (clock)", "Status"],
        "JAX Cages": ["whatever"],                  # not a log tab -> skipped
    })
    res = bn.annotate_all(svc, "SID")
    assert res == {"Summary": 1, "BCH039 Daily": 1, "BCH039 Events": 2}
    assert "JAX Cages" not in res


def test_notes_are_plain_nonempty_strings():
    for d in (bn.DAILY_NOTES, bn.EVENTS_NOTES, bn.SUMMARY_NOTES):
        assert d and all(isinstance(v, str) and v.strip() for v in d.values())


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
