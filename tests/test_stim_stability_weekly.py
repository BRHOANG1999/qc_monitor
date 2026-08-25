"""stim_stability_weekly: figure/render + sheet-row shaping + the full send path
with fakes (store, emailer, Drive, Sheets). No network, no real SA.

Run: pytest tests/test_stim_stability_weekly.py -q
"""

from __future__ import annotations

import math
import os
import sys
from datetime import date

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import stim_stability_weekly as m  # noqa: E402
from tests.test_drive_upload import FakeDrive  # noqa: E402  (reuse the fake)

_ANIMAL, _CH = "BCH062", "BCH062SR"
_T = [i * 0.1 for i in range(60)]          # 0 .. 5.9 ms


def _trace(fid, day, phase):
    stim = [math.sin(i * 0.3 + phase) for i in range(60)]
    evoked = [0.5 * math.sin(i * 0.2 + phase) + 0.01 * fid for i in range(60)]
    return {"file_id": fid, "chunk_datetime": f"{day}__{fid:02d}_00_00",
            "time_ms": _T, "stim_trace": stim, "evoked_trace": evoked,
            "charge_nc": 5.0}


class FakeStore:
    def __init__(self):
        self._traces = [_trace(1, "2026_08_24", 0.0), _trace(2, "2026_08_24", 0.4),
                        _trace(3, "2026_08_25", 0.8), _trace(4, "2026_08_25", 1.2)]
        self._imp = [{"chunk_datetime": f"2026_08_{d:02d}__10_00_00",
                      "access_r_kohm": 100.0 + (0 if d < 24 else 55.0)}
                     for d in range(10, 26)]

    def active_impedance_channel_keys(self):
        return {(_ANIMAL, _CH)}

    def impedance_series_by_channel(self, exclude=None):
        return {(_ANIMAL, _CH): list(self._imp)}

    def channel_stim_evoked_traces_in_range(self, animal, channel, s, e):
        return list(self._traces) if (animal, channel) == (_ANIMAL, _CH) else []


class FakeEmailer:
    def __init__(self):
        self.calls = []

    def send(self, **kw):
        self.calls.append(kw)
        return True


def _config(**over):
    cfg = {"enabled": True, "recipients": ["lab@x.z"], "sheet_id": "SID",
           "shared_drive_folder_id": "FID", "perfile_span": "week"}
    cfg.update(over)
    return {
        "notifications": {"stim_stability_weekly": cfg},
        "google_sheets": {"spreadsheet_id": "SID2",
                          "service_account_file": "secrets/x.json"},
        "google_drive": {"shared_drive_folder_id": "FID",
                         "service_account_file": "secrets/x.json"},
        "alerting": {"smtp": {"recipients": []}},
    }


def _patch_backends(monkeypatch):
    """Fake Drive + capture the Sheets upsert."""
    monkeypatch.setattr(m._drive, "drive_from_config",
                        lambda mode, **kw: FakeDrive())
    captured = {}
    monkeypatch.setattr(m._sw, "_sheets_api_rw", lambda sa: object())
    monkeypatch.setattr(m._sw, "ensure_tab",
                        lambda svc, sid, tab, header: captured.update(
                            {"tab": tab, "header": header}) or False)

    def fake_upsert(svc, sid, tab, key_cols, rows, column_map=None, **kw):
        captured.update({"key_cols": key_cols, "rows": rows,
                         "column_map": column_map, "sheet_id": sid})
        return {"updated": 0, "appended": len(rows)}
    monkeypatch.setattr(m._sw, "upsert_rows_into_tab", fake_upsert)
    return captured


# ------------------------------------------------------------- helpers ---- #

def test_week_span():
    s, e, wk = m._week_span(date(2026, 8, 30))
    assert e == "2026-08-30" and s == "2026-08-24"
    assert wk.startswith("2026-W")


def test_magnitude_pairs():
    xs, ys, ts = m._magnitude_pairs([_trace(1, "2026_08_24", 0.0),
                                     _trace(2, "2026_08_24", 0.9)], None)
    assert len(xs) == 2 and len(ys) == 2 and len(ts) == 2
    assert all(v >= 0 for v in xs)


def test_crop_window():
    t = [i * 0.1 for i in range(-20, 21)]      # -2.0 .. 2.0 ms
    y = [float(i) for i in range(len(t))]
    ct, cy = m._crop(t, y, (-1.0, 1.0))
    assert ct and all(-1.0 <= x <= 1.0 for x in ct)
    # empty/degenerate window falls back to the full trace
    assert m._crop(t, y, (100.0, 200.0)) == (t, y)


# --------------------------------------------------------- full send --- #

def test_full_send(monkeypatch, tmp_path):
    cap = _patch_backends(monkeypatch)
    store, emailer = FakeStore(), FakeEmailer()
    res = m.send_stim_stability_weekly(date(2026, 8, 30), _config(), store,
                                       emailer, out_dir=str(tmp_path))
    assert res["sent"] is True and res["channels"] == 1
    # sheet upsert keyed correctly, one row, formula cells present
    assert cap["key_cols"] == ["Week", "Animal", "Channel"]
    assert set(cap["column_map"].keys()) == set(m.HEADER)
    row = cap["rows"][0]
    assert row["Animal"] == _ANIMAL and row["Channel"] == _CH
    assert row["Weekly stim"].startswith('=IMAGE(') and \
        row["Figures"].startswith('=HYPERLINK(')
    assert row["n files"] == 4
    # email carried inline PNG attachments
    assert emailer.calls and emailer.calls[0]["subject_prefix"] is False
    assert len(emailer.calls[0]["attachments"]) >= 3


def test_disabled_no_send(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    res = m.send_stim_stability_weekly(date(2026, 8, 30), _config(enabled=False),
                                       FakeStore(), FakeEmailer())
    assert res == {"sent": False, "reason": "disabled"}


def test_no_recipients_no_send(monkeypatch, tmp_path):
    _patch_backends(monkeypatch)
    res = m.send_stim_stability_weekly(
        date(2026, 8, 30), _config(recipients=[]), FakeStore(), FakeEmailer())
    assert res == {"sent": False, "reason": "no recipients"}


def test_dry_run_renders_no_side_effects(monkeypatch, tmp_path):
    cap = _patch_backends(monkeypatch)
    emailer = FakeEmailer()
    res = m.send_stim_stability_weekly(date(2026, 8, 30), _config(), FakeStore(),
                                       emailer, dry_run=True, out_dir=str(tmp_path))
    assert res["dry_run"] is True and res["channels"] == 1
    assert res["rows"] and res["rows"][0]["Animal"] == _ANIMAL
    assert emailer.calls == []                 # no email
    assert "rows" not in cap                    # no sheet upsert
    # PNGs really rendered
    pngs = [p for _, _, files in os.walk(tmp_path) for p in files
            if p.endswith(".png")]
    assert any("weekly" in p for p in pngs)
    assert any("impedance" in p for p in pngs)
    assert any("correlation" in p for p in pngs)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
