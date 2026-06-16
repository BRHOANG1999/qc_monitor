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


def test_filter_intersection_logic():
    # Mirrors the _keep predicate in video.py's filter callback.
    meta = {
        "1": {"session_dir": "A", "has_stim": True},
        "2": {"session_dir": "A", "has_stim": False},
        "3": {"session_dir": "B", "has_stim": True},
    }

    def keep(fid, sess_val, stim_val):
        m = meta.get(str(fid))
        if m is None:
            return True
        if sess_val and sess_val != "__all__" \
                and m["session_dir"] != sess_val:
            return False
        if stim_val == "stim" and not m["has_stim"]:
            return False
        if stim_val == "baseline" and m["has_stim"]:
            return False
        return True

    pool = [1, 2, 3]
    assert [f for f in pool if keep(f, "__all__", "all")] == [1, 2, 3]
    assert [f for f in pool if keep(f, "A", "all")] == [1, 2]
    assert [f for f in pool if keep(f, "__all__", "stim")] == [1, 3]
    assert [f for f in pool if keep(f, "__all__", "baseline")] == [2]
    assert [f for f in pool if keep(f, "A", "stim")] == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
