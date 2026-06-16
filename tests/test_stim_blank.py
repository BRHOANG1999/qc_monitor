"""Tests for stim-blanking shared between the trace and the scan.

Run with: pytest tests/test_stim_blank.py -q
"""

from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.utils.mass_analyze as ma  # noqa: E402
from src.utils.stim_blank import blank_series_with_stim_times  # noqa: E402
from src.utils.hilbert_envelope import (  # noqa: E402
    hilbert_envelope_20_200, windowed_auc)
from src.db.store import Store  # noqa: E402


def _artifact_series(fs: float, dur_s: float, t_stim: float):
    n = int(fs * dur_s)
    rng = np.random.default_rng(0)
    s = (rng.standard_normal(n) * 0.001).astype(np.float32)
    c = int(t_stim * fs)
    s[c:c + 20] = 5.0  # big stim artifact
    return s


# --------------------------------------------------------------- #
# Pure blanking core
# --------------------------------------------------------------- #

def test_blank_nans_the_window_and_suppresses_envelope():
    fs = 2000.0
    s = _artifact_series(fs, 5.0, 2.5)
    blk = blank_series_with_stim_times(s, fs, np.array([2.5]), -5.0, 15.0)
    assert np.isnan(blk[int(2.5 * fs)])
    # The artifact dominates the raw envelope; blanking removes it.
    assert hilbert_envelope_20_200(blk, fs).max() < \
        hilbert_envelope_20_200(s, fs).max() / 10
    assert windowed_auc(hilbert_envelope_20_200(blk, fs), fs, 5.0).max() < \
        windowed_auc(hilbert_envelope_20_200(s, fs), fs, 5.0).max() / 10


def test_detect_stim_onsets_finds_pulses():
    from src.utils.stim_blank import detect_stim_onsets
    fs, n = 1000.0, 10000
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.5  # baseline noise
    for t in np.arange(0.5, 10.0, 0.5):  # 19 pulses
        c = int(t * fs)
        x[c:c + 3] = 120.0
    onsets = detect_stim_onsets(x, fs)
    assert len(onsets) == 19
    # Onsets land near the injected times.
    assert abs(onsets[0] - 0.5) < 0.01


def test_detect_stim_onsets_baseline_is_empty():
    from src.utils.stim_blank import detect_stim_onsets
    rng = np.random.default_rng(1)
    assert detect_stim_onsets(rng.standard_normal(10000) * 0.5,
                               1000.0).size == 0
    assert detect_stim_onsets(np.zeros(0), 1000.0).size == 0


def test_blank_no_stim_times_is_passthrough():
    fs = 2000.0
    s = _artifact_series(fs, 2.0, 1.0)
    out = blank_series_with_stim_times(s, fs, np.array([]), -5.0, 15.0)
    assert not np.isnan(out).any()
    assert np.allclose(out, s)


# --------------------------------------------------------------- #
# Scan compute functions now blank
# --------------------------------------------------------------- #

class _FakeChunk:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs


def _seed_file(store):
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir) "
            "VALUES (1, '/f/a.mat', 's')")
        conn.commit()


def test_scan_compute_blanks_stim(tmp_path, monkeypatch):
    fs = 2000.0
    series = _artifact_series(fs, 5.0, 2.5)
    chunk = _FakeChunk(series.reshape(-1, 1), fs)

    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    _seed_file(store)

    monkeypatch.setattr(ma, "get_chunk", lambda path: chunk)
    monkeypatch.setattr(ma.stim_blank, "stim_copy_channels",
                         lambda store, sd: set())

    # With the stim time supplied, the artifact is blanked -> 0 events.
    monkeypatch.setattr(ma.stim_blank, "stim_times_for_file",
                         lambda store, fid: np.array([2.5]))
    blanked = ma._compute_auc_count_uncached(
        store, 1, 0, 0.005, 5.0, ma.DEFAULT_MIN_PEAK_DIST_SEC)
    blanked_peak = ma._compute_peak_count_uncached(
        store, 1, 0, 0.005, ma.DEFAULT_MIN_PEAK_DIST_SEC)

    # With no stim times, the raw artifact is detected.
    monkeypatch.setattr(ma.stim_blank, "stim_times_for_file",
                         lambda store, fid: np.array([]))
    raw = ma._compute_auc_count_uncached(
        store, 1, 0, 0.005, 5.0, ma.DEFAULT_MIN_PEAK_DIST_SEC)
    raw_peak = ma._compute_peak_count_uncached(
        store, 1, 0, 0.005, ma.DEFAULT_MIN_PEAK_DIST_SEC)

    assert blanked.n_events < raw.n_events
    assert blanked.n_events == 0 and raw.n_events >= 1
    assert blanked_peak.n_peaks < raw_peak.n_peaks


# --------------------------------------------------------------- #
# One-time cache eviction
# --------------------------------------------------------------- #

def _insert_cache_row(db):
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO processed_files (id, file_path) "
            "VALUES (1, '/f/a.mat')")
        conn.execute(
            """INSERT INTO envelope_peak_cache
               (file_id, channel, cutoff, min_peak_dist_sec,
                n_peaks, peak_times_json, computed_at)
               VALUES (1, 0, 0.05, 150.0, 3, '[]', 'now')""")
        conn.commit()
    finally:
        conn.close()


def _peak_cache_count(db) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM envelope_peak_cache").fetchone()[0]
    finally:
        conn.close()


def _user_version(db) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def test_one_time_cache_eviction(tmp_path):
    db = str(tmp_path / "data" / "monitor.db")
    Store(db)  # first boot stamps the current cache-schema version
    assert _user_version(db) == 2

    # Simulate a legacy DB: stale row + version reset to 0.
    _insert_cache_row(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()
    assert _peak_cache_count(db) == 1

    # Re-boot: the guard fires once, clears the stale cache.
    Store(db)
    assert _peak_cache_count(db) == 0
    assert _user_version(db) == 2

    # A fresh row must survive a subsequent boot (guard already spent).
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO envelope_peak_cache
           (file_id, channel, cutoff, min_peak_dist_sec,
            n_peaks, peak_times_json, computed_at)
           VALUES (1, 1, 0.05, 150.0, 2, '[]', 'now')""")
    conn.commit()
    conn.close()
    Store(db)
    assert _peak_cache_count(db) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
