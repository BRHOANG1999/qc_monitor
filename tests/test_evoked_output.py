"""ChronicEvokedCache: parse evokedOutput filenames, lazily extract
per-epoch scalars from a (synthetic) v7.3-style HDF5 ``*_evoked.mat``,
cache them, and query per-animal at per-evoked-response resolution.

Run with: pytest tests/test_evoked_output.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

h5py = pytest.importorskip("h5py")  # noqa: E402

from src.utils.evoked_output import (  # noqa: E402
    ChronicEvokedCache, animals_in_filename, parse_recording_dt,
)


def _write_evoked(path, channels, traces=None, time_ms=None):
    """Write allAnimalResults/<ch>/{stimulusTimes,Peak,Trough[,evokedData,
    timeAxis]}. *traces* maps channel -> ndarray[epochs x samples]."""
    import numpy as np
    with h5py.File(path, "w") as g:
        root = g.create_group("allAnimalResults")
        for ch, (times, peak, trough) in channels.items():
            grp = root.create_group(ch)
            grp.create_dataset("stimulusTimes", data=[times])      # (1, n)
            grp.create_dataset("stimulusPeakAmplitudes",
                               data=[[v] for v in peak])           # (n, 1)
            grp.create_dataset("stimulusTroughAmplitudes",
                               data=[[v] for v in trough])
            if traces is not None and ch in traces:
                grp.create_dataset("evokedData",
                                   data=np.asarray(traces[ch]))    # (E, T)
                grp.create_dataset("timeAxis", data=np.asarray(time_ms))


def test_filename_parsers():
    fn = "stimBaseline__stimCopy_BCH062SR___2026_05_10__06_00_00_evoked.mat"
    assert parse_recording_dt(fn) == datetime(2026, 5, 10, 6, 0, 0)
    assert animals_in_filename(fn) == {"BCH062"}
    assert parse_recording_dt("nope.mat") is None


def test_cache_build_query_and_guards(tmp_path):
    ed = tmp_path / "evokedOutput"
    ed.mkdir()
    # One BCH062 baseline; a stimCopy channel must be filtered out.
    _write_evoked(
        str(ed / "base__BCH062SR___2026_05_10__06_00_00_evoked.mat"),
        {"BCH062SR": ([10.0, 20.0], [1.0, 2.0], [-0.5, -1.0]),
         "stimCopy": ([10.0], [9.0], [-9.0])})
    # A multi-animal recording: BCH062SR + BCH061SLM in one file.
    _write_evoked(
        str(ed / "mix__BCH062SR_BCH061SLM___2026_05_12__06_00_00_evoked.mat"),
        {"BCH062SR": ([5.0], [3.0], [-1.5]),
         "BCH061SLM": ([5.0], [7.0], [-2.0])})

    cache = ChronicEvokedCache(str(ed), str(tmp_path / "cache.db"))
    stats = cache.ensure_animal("BCH062")
    assert stats == {"files": 2, "built": 2}

    rows = cache.query("BCH062")
    # 2 epochs from the baseline + 1 from the mix; stimCopy excluded.
    assert len(rows) == 3
    assert all(r["channel"] == "BCH062SR" for r in rows)
    # Oldest first; abs_dt = recording time + stim_time_sec.
    assert rows[0]["abs_dt"] == datetime(2026, 5, 10, 6, 0, 10).isoformat()
    assert rows[-1]["abs_dt"] == datetime(2026, 5, 12, 6, 0, 5).isoformat()
    # Raw stim scalars are kept on peak/trough (the correlation X axis).
    assert rows[0]["peak"] == pytest.approx(1.0)
    assert rows[0]["trough"] == pytest.approx(-0.5)
    # No evokedData in these files -> computed evoked features stay NULL.
    assert rows[0]["peak_to_trough"] is None
    assert rows[0]["line_length"] is None

    # Warming is per-animal now: BCH062's warm built only BCH062's channel,
    # so the other animal isn't cached until it's warmed too.
    assert cache.query("BCH061") == []
    cache.ensure_animal("BCH061")
    other = cache.query("BCH061")
    assert len(other) == 1 and other[0]["channel"] == "BCH061SLM"

    # Rebuild is incremental: nothing changed -> built 0.
    assert cache.ensure_animal("BCH062")["built"] == 0
    # Unknown / empty animal -> [].
    assert cache.query("BCH999") == []
    assert cache.query("") == []


def test_ensure_animal_session_scope(tmp_path):
    ed = tmp_path / "evokedOutput"
    ed.mkdir()
    # Two sessions for BCH062 (session = filename prefix before '__').
    _write_evoked(
        str(ed / "sessA__BCH062SR___2026_05_10__06_00_00_evoked.mat"),
        {"BCH062SR": ([1.0], [1.0], [-1.0])})
    _write_evoked(
        str(ed / "sessB__BCH062SR___2026_05_12__06_00_00_evoked.mat"),
        {"BCH062SR": ([2.0], [2.0], [-2.0])})
    cache = ChronicEvokedCache(str(ed), str(tmp_path / "cache.db"))
    # files_for_animal restricts to the requested session(s).
    assert len(cache.files_for_animal("BCH062")) == 2
    assert len(cache.files_for_animal("BCH062", ["sessA"])) == 1
    # Scoped warm builds only sessA -> only its epoch is queryable by session.
    stats = cache.ensure_animal("BCH062", sessions=["sessA"])
    assert stats["files"] == 1 and stats["built"] == 1
    assert len(cache.query("BCH062", sessions=["sessA"])) == 1
    assert cache.query("BCH062", sessions=["sessB"]) == []


def test_epoch_trace_cache_roundtrip(tmp_path):
    import numpy as np
    ed = tmp_path / "evokedOutput"
    ed.mkdir()
    rng = np.random.default_rng(0)
    tr = rng.standard_normal((6, 600)) * 80.0          # 6 epochs, 600 samples
    tax = np.linspace(-100, 500, 600)
    _write_evoked(
        str(ed / "base__BCH062SR___2026_05_10__06_00_00_evoked.mat"),
        {"BCH062SR": ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      [1.0] * 6, [-1.0] * 6)},
        traces={"BCH062SR": tr}, time_ms=tax)

    cache = ChronicEvokedCache(str(ed), str(tmp_path / "cache.db"),
                               store_traces=True)   # opt-in trace storage
    cache.ensure_animal("BCH062")

    blocks = cache.query_epoch_traces("BCH062")
    assert len(blocks) == 1
    b = blocks[0]
    assert b["channel"] == "BCH062SR"
    assert b["traces"].shape == (6, 600)
    # int16 round-trip is accurate to well under 1 uV.
    assert np.max(np.abs(b["traces"] - tr)) < 0.1
    assert b["time_ms"][0] == pytest.approx(-100.0)
    assert b["time_ms"][-1] == pytest.approx(500.0)
    assert len(b["stim_times"]) == 6 and b["peaks"][0] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
