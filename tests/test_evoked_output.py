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


def _write_evoked(path, channels):
    """Write allAnimalResults/<ch>/{stimulusTimes,Peak,Trough}."""
    with h5py.File(path, "w") as g:
        root = g.create_group("allAnimalResults")
        for ch, (times, peak, trough) in channels.items():
            grp = root.create_group(ch)
            grp.create_dataset("stimulusTimes", data=[times])      # (1, n)
            grp.create_dataset("stimulusPeakAmplitudes",
                               data=[[v] for v in peak])           # (n, 1)
            grp.create_dataset("stimulusTroughAmplitudes",
                               data=[[v] for v in trough])


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
    # Derived peak_to_trough.
    assert rows[0]["peak_to_trough"] == pytest.approx(1.5)

    # The other animal is isolated to its own channel.
    other = cache.query("BCH061")
    assert len(other) == 1 and other[0]["channel"] == "BCH061SLM"

    # Rebuild is incremental: nothing changed -> built 0.
    assert cache.ensure_animal("BCH062")["built"] == 0
    # Unknown / empty animal -> [].
    assert cache.query("BCH999") == []
    assert cache.query("") == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
