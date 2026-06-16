"""Phase 1: pool filtering by session + stim.

Run with: pytest tests/test_pool_filter.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.utils.mass_analyze as ma  # noqa: E402
from src.db.store import Store  # noqa: E402


def _seed(store, fid, session_dir, has_report=0):
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "has_stim_report) VALUES (?, ?, ?, ?)",
            (fid, f"/f/{fid}.mat", session_dir, has_report))
        conn.commit()


def test_has_stim_via_report_flag(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed(store, 1, "sessA", has_report=1)
    _seed(store, 2, "sessA", has_report=0)
    # No evoked_features, no stimCopy -> file 2 is baseline.
    monkeypatch.setattr(ma.stim_blank, "stim_times_for_file",
                         lambda s, f: np.zeros(0))
    assert ma.has_stim_for_file(store, 1) is True
    assert ma.has_stim_for_file(store, 2) is False


def test_has_stim_via_evoked_features(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed(store, 3, "sessA", has_report=0)
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO evoked_features (file_id, epoch_index, "
            "epoch_time_sec) VALUES (3, 0, 12.5)")
        conn.commit()
    monkeypatch.setattr(ma.stim_blank, "stim_times_for_file",
                         lambda s, f: np.zeros(0))
    assert ma.has_stim_for_file(store, 3) is True


def test_has_stim_via_stimcopy_fallback(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    _seed(store, 4, "sessA", has_report=0)
    monkeypatch.setattr(ma.stim_blank, "stim_times_for_file",
                         lambda s, f: np.array([1.0, 2.0, 3.0]))
    assert ma.has_stim_for_file(store, 4) is True


def test_session_filter_logic():
    # Mirrors the session-only _keep predicate in video.py's filter
    # callback (Stim filter was removed; stim is implicit per session).
    meta = {
        "1": {"session_dir": "A"},
        "2": {"session_dir": "A"},
        "3": {"session_dir": "B"},
    }
    valid = {m["session_dir"] for m in meta.values()}  # {A, B}

    def keep(fid, sess_val):
        # Defensive: a session not in this animal's options -> "all".
        if sess_val not in (None, "__all__") and sess_val not in valid:
            sess_val = "__all__"
        if sess_val in (None, "__all__"):
            return True
        m = meta.get(str(fid))
        if m is None:
            return True
        return m["session_dir"] == sess_val

    pool = [1, 2, 3]
    assert [f for f in pool if keep(f, "__all__")] == [1, 2, 3]
    assert [f for f in pool if keep(f, "A")] == [1, 2]
    assert [f for f in pool if keep(f, "B")] == [3]
    # Stale session from another animal -> treated as "all" (Pool-2 bug).
    assert [f for f in pool if keep(f, "ZZZ_other_animal")] == [1, 2, 3]


def test_pool_meta_is_db_only(tmp_path, monkeypatch):
    # pool_meta must NOT carry has_stim anymore (that caused the stall).
    store = Store(str(tmp_path / "data" / "monitor.db"))
    monkeypatch.setattr(
        ma, "pending_files_for_animal",
        lambda s, a: [{"file_id": 1, "session_dir": "sess/A"}])
    # If pool_meta still called has_stim_for_file, this would blow up.
    monkeypatch.setattr(
        ma, "has_stim_for_file",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("pool_meta must not check stim")))
    meta = ma.pool_meta(store, "BCH001")
    assert meta["1"] == {"session_dir": "sess/A", "session_name": "A"}
    assert "has_stim" not in meta["1"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
