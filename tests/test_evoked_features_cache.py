"""ChronicEvokedCache v2: extract evokedData traces, compute the full
feature set per epoch, store per-recording mean waveforms, and serve them
back. Uses synthetic v7.3-style HDF5 files.

Run with: pytest tests/test_evoked_features_cache.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

h5py = pytest.importorskip("h5py")  # noqa: E402

import src.utils.evoked_features as ef  # noqa: E402
from src.utils.evoked_output import ChronicEvokedCache, SCHEMA_VERSION  # noqa: E402

FS = 1000.0
TIME = np.linspace(-100.0, 199.0, 300)


def _trace(seed, n):
    rng = np.random.default_rng(seed)
    base = np.sin(2 * np.pi * 8 * (TIME / 1000.0))
    return base[None, :] + 0.05 * rng.standard_normal((n, 300))


def _write(path, channels):
    """channels: {ch: (n_epochs, stim_peaks, stim_troughs)} with evokedData."""
    with h5py.File(path, "w") as g:
        root = g.create_group("allAnimalResults")
        for ch, (n, sp, st) in channels.items():
            grp = root.create_group(ch)
            traces = _trace(hash(ch) % 1000, n)
            grp.create_dataset("stimulusTimes", data=[np.arange(n) * 2.0])
            grp.create_dataset("stimulusPeakAmplitudes", data=sp.reshape(n, 1))
            grp.create_dataset("stimulusTroughAmplitudes",
                               data=st.reshape(n, 1))
            grp.create_dataset("evokedData", data=traces)        # (n, samples)
            grp.create_dataset("timeAxis", data=TIME.reshape(-1, 1))


def _cache(tmp_path, expensive=False):
    ed = tmp_path / "eo"
    ed.mkdir(exist_ok=True)
    n = 12
    _write(str(ed / f"base__BCH062SR___2026_05_10__06_00_00_evoked.mat"),
           {"BCH062SR": (n, np.linspace(1, 2, n), np.linspace(-1, -0.5, n))})
    return ChronicEvokedCache(str(ed), str(tmp_path / "c.db"),
                              compute_expensive=expensive), ed


def test_features_and_recording_mean_populate(tmp_path):
    cache, _ = _cache(tmp_path)
    assert cache.ensure_animal("BCH062")["built"] == 1

    rows = cache.query("BCH062")
    assert len(rows) == 12
    r0 = rows[0]
    # All cheap feature columns present and finite; raw stim kept on peak.
    for col in ef.CHEAP_COLUMNS:
        assert r0[col] is not None, col
        assert np.isfinite(r0[col]), col
    assert r0["peak"] == pytest.approx(1.0)        # raw stim, not evoked peak
    # Expensive columns NULL when the flag is off.
    for col in ef.EXPENSIVE_COLUMNS:
        assert r0[col] is None, col

    means = cache.query_recording_means("BCH062")
    assert len(means) == 1
    m = means[0]
    assert len(m["time_axis"]) == 300
    assert len(m["mean_trace"]) == 300
    assert m["n_epochs"] == 12


def test_expensive_columns_filled_when_enabled(tmp_path):
    cache, _ = _cache(tmp_path, expensive=True)
    cache.ensure_animal("BCH062")
    rows = cache.query("BCH062")
    # At least one expensive feature is finite for a real waveform.
    assert any(rows[-1][c] is not None for c in ef.EXPENSIVE_COLUMNS)


def test_session_filtering(tmp_path):
    ed = tmp_path / "eo"
    ed.mkdir(exist_ok=True)
    n = 6
    sp = np.linspace(1, 2, n)
    st = np.linspace(-1, -0.5, n)
    # Two sessions (distinct filename prefixes) for the same animal.
    _write(str(ed / "20251210_stimBaseline__BCH062SR___"
               "2025_12_10__06_00_00_evoked.mat"), {"BCH062SR": (n, sp, st)})
    _write(str(ed / "stimTest-40nC__BCH062SR___"
               "2026_05_10__06_00_00_evoked.mat"), {"BCH062SR": (n, sp, st)})
    cache = ChronicEvokedCache(str(ed), str(tmp_path / "c.db"))
    cache.ensure_animal("BCH062")
    assert cache._has_session
    sessions = cache.list_sessions("BCH062")
    assert sessions == ["20251210_stimBaseline", "stimTest-40nC"]
    # All sessions vs one session (combine = pass a list).
    assert len(cache.query("BCH062")) == 12
    one = cache.query("BCH062", sessions=["stimTest-40nC"])
    assert len(one) == 6
    assert all(r["session"] == "stimTest-40nC" for r in one)
    both = cache.query("BCH062", sessions=sessions)
    assert len(both) == 12
    assert cache.query_recording_means(
        "BCH062", sessions=["20251210_stimBaseline"]) != []


def test_incremental_and_schema_rebuild(tmp_path):
    cache, ed = _cache(tmp_path)
    cache.ensure_animal("BCH062")
    assert cache.ensure_animal("BCH062")["built"] == 0     # unchanged
    # force=True recomputes even unchanged files.
    assert cache.ensure_animal("BCH062", force=True)["built"] == 1

    # A schema-version bump wipes + rebuilds: simulate by stamping an old
    # version, then re-opening.
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "c.db"))
    conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    cache2 = ChronicEvokedCache(str(ed), str(tmp_path / "c.db"))
    assert cache2.query("BCH062") == []                    # wiped
    conn = sqlite3.connect(str(tmp_path / "c.db"))
    v = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    conn.close()
    assert v == SCHEMA_VERSION


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
