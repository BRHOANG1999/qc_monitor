"""evoked_output: parse evokedOutput filenames, extract per-epoch features
from a (synthetic) v7.3-style HDF5 ``*_evoked.mat`` (``compute_feature_rows``),
and round-trip the per-(recording, animal) JSON feature sidecars.

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
    animals_in_filename, compute_feature_rows, parse_recording_dt,
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


def test_compute_feature_rows(tmp_path):
    """compute_feature_rows reads the animal's traces, computes features, and
    drops non-animal channels -- the canonical sidecar payload."""
    import numpy as np
    ed = tmp_path / "evokedOutput"
    ed.mkdir()
    rng = np.random.default_rng(0)
    tr = rng.standard_normal((3, 600)) * 80.0          # 3 epochs, 600 samples
    tax = np.linspace(-100, 500, 600)
    # BCH062SR has traces; a stimCopy channel must be dropped.
    _write_evoked(
        str(ed / "base__BCH062SR_stimCopy___2026_05_10__06_00_00_evoked.mat"),
        {"BCH062SR": ([10.0, 20.0, 30.0], [1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]),
         "stimCopy": ([10.0], [9.0], [-9.0])},
        traces={"BCH062SR": tr}, time_ms=tax)

    rows = compute_feature_rows(
        str(ed / "base__BCH062SR_stimCopy___2026_05_10__06_00_00_evoked.mat"),
        "BCH062")
    # 3 epochs, all on the animal channel (stimCopy excluded).
    assert len(rows) == 3
    assert all(r["channel"] == "BCH062SR" for r in rows)
    # Oldest first; abs_dt = recording time + stim_time_sec.
    assert rows[0]["abs_dt"] == datetime(2026, 5, 10, 6, 0, 10).isoformat()
    # Raw stim scalars carried through; features computed (not None).
    assert rows[0]["peak"] == pytest.approx(1.0)
    assert rows[0]["trough"] == pytest.approx(-0.5)
    assert rows[0]["line_length"] is not None
    # A file with no evokedData yields no rows (nothing to feature-extract).
    _write_evoked(
        str(ed / "scal__BCH061SR___2026_05_12__06_00_00_evoked.mat"),
        {"BCH061SR": ([5.0], [3.0], [-1.5])})
    assert compute_feature_rows(
        str(ed / "scal__BCH061SR___2026_05_12__06_00_00_evoked.mat"),
        "BCH061") == []


def test_feature_sidecar_roundtrip_and_guards(tmp_path):
    from src.utils.evoked_output import (
        feature_sidecar_path, read_feature_sidecar, write_feature_sidecar)
    mat = tmp_path / "s__BCH062SLM___2026_05_10__06_00_00_evoked.mat"
    mat.write_bytes(b"raw")
    rows = [{"channel": "BCH062SLM", "abs_dt": "2026-05-10T06:00:01",
             "line_length": 238.0}]
    sp = write_feature_sidecar(str(mat), "BCH062", rows)
    assert sp == feature_sidecar_path(str(mat), "BCH062")
    assert sp.endswith(".features.BCH062.json")
    # Round-trips; a different animal has no sidecar.
    assert read_feature_sidecar(str(mat), "BCH062") == rows
    assert read_feature_sidecar(str(mat), "BCH999") is None
    # Source change -> the sidecar reads as stale (None), not wrong data.
    import os
    import time
    future = time.time() + 10
    os.utime(str(mat), (future, future))
    assert read_feature_sidecar(str(mat), "BCH062") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
