"""Store.channel_stim_evoked_traces_in_range + trace_average.average_traces.

The weekly stim-stability job reads per-file STIM (stim_mean_trace) and EVOKED
(mean_trace) traces for a stimulated channel over a date range, then averages
them per day / per week. This covers both the reader (dominant-charge, in-range,
NULL-stim drop, both traces decoded) and the pure averaging helper.

Run: pytest tests/test_stim_trace_reader.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.utils.trace_average import average_traces  # noqa: E402

_ANIMAL, _CH = "BCH062", "BCH062SR"
_T = [0.0, 0.5, 1.0]


def _rec(store, fid, chunk_dt, *, charge=5.0, ra=100.0, channel=1,
         animal=_ANIMAL, ch_name=_CH, stim=None, evoked=None):
    """One recording with BOTH stim_mean_trace and mean_trace (stim=None keeps
    the column NULL, exercising the drop path)."""
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
            "time_axis_ms, mean_trace, stim_mean_trace) VALUES (?,?,?,?,?,?)",
            (fid, channel, ch_name, json.dumps(_T),
             json.dumps(evoked or [float(fid)] * len(_T)),
             None if stim is None else json.dumps(stim)))
        conn.commit()


def _store(tmp_path):
    return Store(str(tmp_path / "data" / "m.db"))


# --------------------------------------------------------------- reader --- #

def test_reader_returns_both_traces_oldest_first(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_03__10_00_00", stim=[1, 2, 3], evoked=[4, 5, 6])
    _rec(s, 2, "2026_06_01__10_00_00", stim=[7, 8, 9], evoked=[1, 1, 1])
    got = s.channel_stim_evoked_traces_in_range(_ANIMAL, _CH,
                                                "2026-06-01", "2026-06-30")
    assert [r["chunk_datetime"] for r in got] == [
        "2026_06_01__10_00_00", "2026_06_03__10_00_00"]        # oldest-first
    assert got[0]["stim_trace"] == [7, 8, 9]
    assert got[0]["evoked_trace"] == [1, 1, 1]
    assert got[0]["time_ms"] == _T and got[0]["charge_nc"] == 5.0
    assert "file_id" in got[0]


def test_reader_drops_null_stim(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_01__10_00_00", stim=[1, 2, 3])
    _rec(s, 2, "2026_06_02__10_00_00", stim=None)              # NULL stim -> out
    got = s.channel_stim_evoked_traces_in_range(_ANIMAL, _CH,
                                                "2026-06-01", "2026-06-30")
    assert [r["file_id"] for r in got] == [1]


def test_reader_dominant_charge_only(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_01__10_00_00", charge=5.0, stim=[1, 1, 1])
    _rec(s, 2, "2026_06_02__10_00_00", charge=5.0, stim=[2, 2, 2])
    _rec(s, 3, "2026_06_03__10_00_00", charge=20.0, stim=[9, 9, 9])
    got = s.channel_stim_evoked_traces_in_range(_ANIMAL, _CH,
                                                "2026-06-01", "2026-06-30")
    assert len(got) == 2 and all(r["charge_nc"] == 5.0 for r in got)


def test_reader_other_channel_animal_excluded(tmp_path):
    s = _store(tmp_path)
    _rec(s, 1, "2026_06_10__10_00_00", stim=[1, 1, 1])
    _rec(s, 2, "2026_06_11__10_00_00", ch_name="BCH062SLM", channel=2,
         stim=[2, 2, 2])
    _rec(s, 3, "2026_06_12__10_00_00", animal="BCH061", ch_name="BCH061SR",
         stim=[3, 3, 3])
    got = s.channel_stim_evoked_traces_in_range(_ANIMAL, _CH,
                                                "2026-06-01", "2026-06-30")
    assert [r["file_id"] for r in got] == [1]


# ------------------------------------------------------------ averaging --- #

def test_average_equal_length():
    tr = [{"time_ms": _T, "stim_trace": [1.0, 2.0, 3.0]},
          {"time_ms": _T, "stim_trace": [3.0, 4.0, 5.0]}]
    t, y = average_traces(tr)
    assert t == _T and y == [2.0, 3.0, 4.0]


def test_average_unequal_length_interp():
    tr = [{"time_ms": [0.0, 1.0], "stim_trace": [0.0, 10.0]},
          {"time_ms": [0.0, 0.5, 1.0], "stim_trace": [0.0, 5.0, 10.0]}]
    t, y = average_traces(tr)
    assert t == [0.0, 0.5, 1.0]          # ref grid = the longer axis
    # interp of the 2-pt trace at 0.5 -> 5.0, averaged with 5.0 -> 5.0
    assert y == [0.0, 5.0, 10.0]


def test_average_ignores_nan_per_sample():
    tr = [{"time_ms": _T, "stim_trace": [1.0, float("nan"), 3.0]},
          {"time_ms": _T, "stim_trace": [3.0, 4.0, 5.0]}]
    _t, y = average_traces(tr)
    assert y[0] == 2.0 and y[1] == 4.0 and y[2] == 4.0    # nan dropped at idx1


def test_average_empty_returns_none():
    assert average_traces([]) == (None, None)
    assert average_traces([{"time_ms": None, "stim_trace": None}]) == (None, None)


def test_average_evoked_key():
    tr = [{"time_ms": _T, "evoked_trace": [2.0, 2.0, 2.0]},
          {"time_ms": _T, "evoked_trace": [4.0, 4.0, 4.0]}]
    _t, y = average_traces(tr, value_key="evoked_trace")
    assert y == [3.0, 3.0, 3.0]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
