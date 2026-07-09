"""Pre-ictal ISI accounting: dual-format datetime parsing, absolute-onset
seizure enumeration, ISI-derived lookback ceilings, and dyadic lead-time bins.

Run with: pytest tests/test_preictal_isi.py -q
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.preictal import isi  # noqa: E402


# ------------------------------------------------------------------ #
#  parse_chunk_datetime -- BOTH stored formats
# ------------------------------------------------------------------ #

def test_parse_chunk_datetime_both_formats():
    iso = isi.parse_chunk_datetime("2025-02-09T13:59:16")
    us = isi.parse_chunk_datetime("2026_03_02__17_18_51")
    assert iso == datetime(2025, 2, 9, 13, 59, 16)
    assert us == datetime(2026, 3, 2, 17, 18, 51)
    assert isi.parse_chunk_datetime("") is None
    assert isi.parse_chunk_datetime("garbage") is None


# ------------------------------------------------------------------ #
#  lead-time bins (pure)
# ------------------------------------------------------------------ #

def test_leadtime_bins_dyadic_bounded_by_ceiling():
    b = isi.leadtime_bins(ceiling_sec=850.0, min_leadtime_sec=1.0, factor=2.0)
    # 1,2,4,...,512 then the ceiling 850 as the final edge.
    assert b.edges_sec[0] == 1.0
    assert b.edges_sec[-1] == 850.0
    assert all(e <= 850.0 for e in b.edges_sec)         # never past the ceiling
    assert len(b.centers_sec) == len(b.edges_sec) - 1
    # centers are geometric means, strictly increasing.
    assert list(b.centers_sec) == sorted(b.centers_sec)


def test_leadtime_bins_empty_when_isi_too_short():
    # Ceiling below the minimum lead-time -> no defined pre-ictal bin.
    b = isi.leadtime_bins(ceiling_sec=0.5, min_leadtime_sec=1.0)
    assert b.edges_sec == () and b.centers_sec == ()


# ------------------------------------------------------------------ #
#  ceilings (pure) -- ISI - buffer, drop first + buffer-eaten
# ------------------------------------------------------------------ #

def _sz(onset_epoch, bb=None):
    return isi.Seizure(animal_id="BCH040", file_id=1, file_path="/f.mat",
                       session_dir="/s", chunk_datetime="x", eo_sec=0.0,
                       bb_sec=bb, racine=3, seizure_type="LVF",
                       onset_epoch=onset_epoch)


def test_ceilings_isi_minus_buffer():
    szs = [_sz(0.0), _sz(1150.0), _sz(3500.0)]
    assert isi.inter_seizure_intervals(szs) == [None, 1150.0, 2350.0]
    ceil = isi.lookback_ceilings(szs, post_ictal_buffer_sec=300.0)
    # first has no predecessor; others = ISI - 300.
    assert ceil == [None, 850.0, 2050.0]


def test_ceiling_none_when_buffer_eats_interval():
    szs = [_sz(0.0), _sz(200.0)]              # ISI 200 < buffer 300
    assert isi.lookback_ceilings(szs, post_ictal_buffer_sec=300.0) == [None,
                                                                       None]


# ------------------------------------------------------------------ #
#  behavioral_seizures -- absolute onset across chunk boundaries + mixed fmt
# ------------------------------------------------------------------ #

def _seed(store, rows):
    """rows: list of (file_id, chunk_datetime, eo_sec, bb_sec, racine)."""
    with store.connection() as conn:
        for fid, cd, eo, bb, rac in rows:
            conn.execute(
                "INSERT INTO processed_files (id, file_path, session_dir, "
                "chunk_datetime) VALUES (?,?,?,?)",
                (fid, f"/sess/{fid}.mat", "/sess", cd))
            markers = [{"EO_sec": eo, "BB_sec": bb, "racine": rac,
                        "type": "LVF"}]
            conn.execute(
                "INSERT INTO review_state (file_id, user_email, status, "
                "animal_id, markers_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (fid, "pi@x", "pi_approved", "BCH040", json.dumps(markers),
                 "2026-01-01", "2026-01-01"))
        conn.commit()


def test_behavioral_seizures_absolute_onset_and_order(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    # Two chunk-datetime formats on purpose; onsets must still order correctly.
    #   A: 00:00:00 + 100s -> t+100
    #   B: 00:20:00 (=1200) + 50s -> t+1250   (ISO format)
    #   C: 01:00:00 (=3600) + 0s  -> t+3600   (underscore format)
    _seed(store, [
        (1, "2026_03_02__00_00_00", 100.0, 150.0, 3),
        (2, "2026-03-02T00:20:00",   50.0,  90.0, 4),
        (3, "2026_03_02__01_00_00",   0.0,  30.0, 2),
    ])
    szs = isi.behavioral_seizures(store, "BCH040")
    assert [s.file_id for s in szs] == [1, 2, 3]          # chronological
    base = datetime(2026, 3, 2, 0, 0, 0).timestamp()
    assert abs(szs[0].onset_epoch - (base + 100)) < 1e-6
    assert abs(szs[1].onset_epoch - (base + 1250)) < 1e-6
    assert abs(szs[2].onset_epoch - (base + 3600)) < 1e-6
    assert szs[0].bb_sec == 150.0 and szs[0].racine == 3
    # ISI + ceilings across the boundary.
    assert isi.inter_seizure_intervals(szs) == [None, 1150.0, 2350.0]
    assert isi.lookback_ceilings(szs, 300.0) == [None, 850.0, 2050.0]


def test_behavioral_seizures_skips_unscored(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    with store.connection() as conn:
        conn.execute("INSERT INTO processed_files (id, file_path, "
                     "chunk_datetime) VALUES (1,'/f.mat','2026_03_02__00_00_00')")
        # No racine -> not a confirmed behavioral seizure.
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, animal_id, "
            "markers_json, created_at, updated_at) VALUES "
            "(1,'p','pi_approved','BCH040',?, 't','t')",
            (json.dumps([{"EO_sec": 10.0, "racine": None}]),))
        conn.commit()
    assert isi.behavioral_seizures(store, "BCH040") == []
