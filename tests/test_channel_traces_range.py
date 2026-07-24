"""Date-range + bounds for the Overview stim-artifact overlay.

Store.channel_traces_in_range / channel_trace_date_bounds back the selectable
overlay: pick an animal + stim location + a recording date window instead of the
active channels' most-recent-N. These mirror recent_channel_traces' guarantees
(dominant charge only, newest-first) but filter by chunk_datetime instead of a
count.

Run: pytest tests/test_channel_traces_range.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402

_ANIMAL, _CH = "BCH062", "BCH062SR"
_T = [0.0, 0.5, 1.0]


def _rec(store, fid, chunk_dt, *, charge=5.0, ra=100.0, channel=1,
         animal=_ANIMAL, ch_name=_CH, mean=None):
    """One recording: processed_files + channel_impedance + evoked_waveforms."""
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "chunk_datetime, has_video) VALUES (?,?,?,?,1)",
            (fid, f"/f{fid}.mat", "/s", chunk_dt))
        conn.execute(
            "INSERT INTO channel_impedance (file_id, channel, channel_name, "
            "animal_id, charge_nc, access_r_kohm, computed_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (fid, channel, ch_name, animal, charge, ra, chunk_dt))
        conn.execute(
            "INSERT INTO evoked_waveforms (file_id, channel, channel_name, "
            "time_axis_ms, mean_trace) VALUES (?,?,?,?,?)",
            (fid, channel, ch_name, json.dumps(_T),
             json.dumps(mean or [float(fid)] * len(_T))))
        conn.commit()


def _store(tmp_path):
    return Store(str(tmp_path / "data" / "m.db"))


def _dates(traces):
    return [t["chunk_datetime"] for t in traces]


# ------------------------------------------------------------ in range --- #

def test_returns_only_in_range_newest_first(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_01__10_00_00")
    _rec(s, 2, "2026_06_10__10_00_00")
    _rec(s, 3, "2026_06_20__10_00_00")
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-05", "2026-06-15")
    assert _dates(got) == ["2026_06_10__10_00_00"]     # only file 2


def test_end_date_is_inclusive_through_end_of_day(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_10__23_59_58")                 # late on the end day
    _rec(s, 2, "2026_06_11__00_00_01")                 # next day -> excluded
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-10")
    assert _dates(got) == ["2026_06_10__23_59_58"]


def test_newest_first_ordering(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_02__10_00_00")
    _rec(s, 2, "2026_06_04__10_00_00")
    _rec(s, 3, "2026_06_03__10_00_00")
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-30")
    assert _dates(got) == ["2026_06_04__10_00_00", "2026_06_03__10_00_00",
                           "2026_06_02__10_00_00"]


def test_dominant_charge_only(tmp_path):
    """Two 5nC recordings, one 20nC -> the odd charge is excluded (mixing
    charges makes the overlay meaningless)."""
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_01__10_00_00", charge=5.0)
    _rec(s, 2, "2026_06_02__10_00_00", charge=5.0)
    _rec(s, 3, "2026_06_03__10_00_00", charge=20.0)
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-30")
    assert len(got) == 2                                # the two 5nC ones
    assert all(d.startswith("2026_06_0") for d in _dates(got))


def test_max_traces_cap_keeps_newest(tmp_path):
    s = _store(tmp_path)
    for fid in range(1, 6):
        _rec(s, fid, f"2026_06_0{fid}__10_00_00")
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-30",
                                    max_traces=2)
    assert _dates(got) == ["2026_06_05__10_00_00", "2026_06_04__10_00_00"]


def test_other_channel_and_animal_excluded(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_10__10_00_00")
    _rec(s, 2, "2026_06_11__10_00_00", ch_name="BCH062SLM", channel=2)
    _rec(s, 3, "2026_06_12__10_00_00", animal="BCH061", ch_name="BCH061SR")
    got = s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-30")
    assert _dates(got) == ["2026_06_10__10_00_00"]


def test_empty_when_nothing_in_range(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_05_01__10_00_00")
    assert s.channel_traces_in_range(_ANIMAL, _CH, "2026-06-01", "2026-06-30") == []


# ------------------------------------------------------------- bounds --- #

def test_date_bounds(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_03__10_00_00")
    _rec(s, 2, "2026_06_01__10_00_00")
    _rec(s, 3, "2026_06_07__10_00_00")
    lo, hi = s.channel_trace_date_bounds(_ANIMAL, _CH)
    assert lo == "2026_06_01__10_00_00"
    assert hi == "2026_06_07__10_00_00"


def test_date_bounds_none_when_no_history(tmp_path):
    s = _store(tmp_path)
    assert s.channel_trace_date_bounds(_ANIMAL, _CH) is None


def test_bounds_respect_dominant_charge(tmp_path):
    """Bounds come from the dominant-charge series only, matching the traces
    the overlay actually draws."""
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_01__10_00_00", charge=5.0)
    _rec(s, 2, "2026_06_09__10_00_00", charge=5.0)
    _rec(s, 3, "2026_06_20__10_00_00", charge=20.0)    # odd charge, out
    lo, hi = s.channel_trace_date_bounds(_ANIMAL, _CH)
    assert lo == "2026_06_01__10_00_00"
    assert hi == "2026_06_09__10_00_00"                 # not the 20nC one
