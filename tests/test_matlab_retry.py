"""MATLAB Tier-2 failure handling: attempt-aware failure listing, the
'exhausted / gave up' split, bounded auto-retry eligibility, and the
blocked-on-missing-gain legacy-animal allowlist.

Run: pytest tests/test_matlab_retry.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                              # noqa: E402


def _iso(days=0, hours=0):
    # Real-now relative: matlab_failed_files / matlab_files_to_retry filter on
    # datetime.now(), so fixtures must be relative to the real clock.
    return (datetime.now() - timedelta(days=days, hours=hours)).isoformat()


def _pf(conn, fid, status, processed_days=1):
    conn.execute(
        "INSERT INTO processed_files (id, file_path, status, processed_at) "
        "VALUES (?,?,?,?)",
        (fid, f"/f{fid}.mat", status, _iso(days=processed_days)))


def _attempt(conn, fid, computed_days=None, computed_hours=0):
    ca = (_iso(days=computed_days, hours=computed_hours)
          if computed_days is not None else _iso(hours=computed_hours))
    conn.execute(
        "INSERT INTO matlab_results (file_id, exit_status, error_message, "
        "computed_at) VALUES (?,?,?,?)",
        (fid, "error", "MATLAB timed out after 1800s", ca))


def _seed(store):
    with store.connection() as conn:
        # 1: active + eligible (1 attempt, 2 days ago)
        _pf(conn, 1, "matlab_error", processed_days=2)
        _attempt(conn, 1, computed_days=2)
        # 2: exhausted (4 attempts)
        _pf(conn, 2, "matlab_error", processed_days=2)
        for _ in range(4):
            _attempt(conn, 2, computed_days=2)
        # 3: active but WITHIN backoff (1 attempt, 1 hour ago) -> not eligible yet
        _pf(conn, 3, "matlab_error", processed_days=0)
        _attempt(conn, 3, computed_hours=1)
        # 4: active + eligible, 0 attempts (failed before MATLAB ran)
        _pf(conn, 4, "matlab_error", processed_days=2)
        # 5: done -> never appears
        _pf(conn, 5, "done", processed_days=1)
        conn.commit()


def test_failed_files_excludes_exhausted_and_counts_attempts(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    _seed(store)
    active = store.matlab_failed_files(hours=168, max_attempts=4)
    ids = {r["file_id"] for r in active}
    assert ids == {1, 3, 4}                       # 2 exhausted, 5 done -> out
    by_id = {r["file_id"]: r["attempts"] for r in active}
    assert by_id[1] == 1 and by_id[4] == 0        # attempt counts surfaced
    # Without the cap, the exhausted file is still listed.
    assert 2 in {r["file_id"] for r in store.matlab_failed_files(hours=168)}


def test_exhausted_files(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    _seed(store)
    ex = store.matlab_exhausted_files(max_attempts=4)
    assert [r["file_id"] for r in ex] == [2] and ex[0]["attempts"] == 4


def test_files_to_retry_respects_cap_and_backoff(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    _seed(store)
    retry = set(store.matlab_files_to_retry(max_attempts=4, backoff_hours=12))
    # 1 (2d old) + 4 (no attempt yet) eligible; 3 too recent (<12h), 2 exhausted.
    assert retry == {1, 4}


def test_blocked_on_missing_gain_allowlist(tmp_path):
    store = Store(str(tmp_path / "d" / "m.db"))
    with store.connection() as conn:
        for fid, animal in ((1, "BCH040"), (2, "BCH040"), (3, "BCH062")):
            _pf(conn, fid, "done")               # channel_impedance.file_id FK
            conn.execute(
                "INSERT INTO channel_impedance (file_id, channel, channel_name, "
                "animal_id, gain, access_r_kohm, computed_at) "
                "VALUES (?,?,?,?,NULL,NULL,?)",
                (fid, 0, f"{animal}SR", animal, _iso()))
        conn.commit()
    all_blocked = {b["animal_id"]: b["files"]
                   for b in store.files_blocked_on_missing_gain()}
    assert all_blocked == {"BCH040": 2, "BCH062": 1}
    # legacy BCH040 excluded (case-insensitive) -> only BCH062 remains
    kept = store.files_blocked_on_missing_gain(exclude=["bch040"])
    assert [b["animal_id"] for b in kept] == ["BCH062"]
