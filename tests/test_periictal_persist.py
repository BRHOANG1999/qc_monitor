"""Peri-ictal matrix disk cache: a signature hit reuses the pickle, and scoring
a NEW seizure (which retroactively rewrites lead-times) changes the signature so
the cache is correctly rebuilt.

Run with: pytest tests/test_periictal_persist.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                             # noqa: E402
from src.periictal import persist                          # noqa: E402
from src.utils.evoked_output import write_feature_sidecar  # noqa: E402

_ANIMAL = "BCH040"


def _seed_seizure(store, fid, chunk_dt, eo, rac=3):
    with store.connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_files (id, file_path, session_dir, "
            "chunk_datetime) VALUES (?,?,?,?)",
            (fid, f"/s/{fid}.mat", "/s", chunk_dt))
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, animal_id, "
            "markers_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (fid, "p", "pi_approved", _ANIMAL,
             json.dumps([{"EO_sec": eo, "racine": rac}]), "t", "t"))
        conn.commit()


def _seed_sidecar(evoked_dir):
    os.makedirs(evoked_dir, exist_ok=True)
    mat = os.path.join(
        evoked_dir, f"sess__stimCopy_{_ANIMAL}SR___2026_03_02__00_00_00_evoked.mat")
    with open(mat, "wb") as f:
        f.write(b"\x00")
    rows = [{"channel": f"{_ANIMAL}SR", "abs_dt": "2026-03-02T00:59:00",
             "session": "sess", "line_length": 1.0}]
    write_feature_sidecar(mat, _ANIMAL, rows)


def test_cache_hit_then_invalidated_by_new_seizure(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    _seed_seizure(store, 1, "2026_03_01__00_00_00", 3600.0)   # A
    _seed_seizure(store, 2, "2026_03_02__00_00_00", 3600.0)   # B (onset 01:00)
    evoked_dir = str(tmp_path / "evoked")
    _seed_sidecar(evoked_dir)
    cache = str(tmp_path / "cache")

    df1 = persist.build_matrix_cached(store, _ANIMAL, evoked_dir, cache,
                                      window_sec=6 * 3600.0,
                                      attach_fingerprint=False)
    files1 = sorted(os.listdir(cache))
    assert len(files1) == 1 and len(df1) == 1
    assert round(df1["time_to_onset_sec"].iloc[0]) == 60.0

    # Second call, unchanged -> a signature HIT (no new cache file).
    df2 = persist.build_matrix_cached(store, _ANIMAL, evoked_dir, cache,
                                      window_sec=6 * 3600.0,
                                      attach_fingerprint=False)
    assert sorted(os.listdir(cache)) == files1
    assert df2.equals(df1)

    # Score a NEW seizure 30 s after the stimulus -> its lead-time changes,
    # so the signature changes and the cache is rebuilt (old file pruned).
    _seed_seizure(store, 3, "2026_03_02__00_59_30", 0.0)
    df3 = persist.build_matrix_cached(store, _ANIMAL, evoked_dir, cache,
                                      window_sec=6 * 3600.0,
                                      attach_fingerprint=False)
    assert round(df3["time_to_onset_sec"].iloc[0]) == 30.0     # relabelled
    files3 = sorted(os.listdir(cache))
    assert files3 != files1 and len(files3) == 1               # stale pruned


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
