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


def test_pending_files_include_reviewed_keeps_flagged(tmp_path):
    # A flagged (needs_scoring) file is dropped from the pending set but
    # MUST stay when include_reviewed=True, so the pool browser/tracker
    # doesn't lose it on a tab-reopen rebuild.
    import json as _json
    store = Store(str(tmp_path / "data" / "monitor.db"))
    sd = "//srv/db\\stimBaseline_BCH062SR_BCH061SLM_"
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO session_config (session_dir, channel_names, "
            "eeg_channels, discovered_at) VALUES (?, ?, ?, '2026-01-01')",
            (sd, _json.dumps(["stimCopy", "BCH062SR", "BCH061SLM"]),
             _json.dumps([1, 2])))
        for fid in (10, 11):
            conn.execute(
                "INSERT INTO processed_files (id, file_path, session_dir, "
                "chunk_datetime) VALUES (?, ?, ?, ?)",
                (fid, f"/f/{fid}.mat", sd, f"2026-01-0{fid}"))
        # File 10 flagged for scoring; file 11 untouched.
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, "
            "created_at, updated_at) VALUES "
            "(10,'u','needs_scoring','t','t')")
        conn.commit()

    pending = {int(f["file_id"])
               for f in ma.pending_files_for_animal(store, "BCH062")}
    allf = {int(f["file_id"]) for f in ma.pending_files_for_animal(
        store, "BCH062", include_reviewed=True)}
    assert 10 not in pending and 11 in pending      # flagged dropped
    assert 10 in allf and 11 in allf                # kept with flag


def test_review_statuses_for_files_batch(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    for fid in (1, 2, 3, 4):
        _seed(store, fid, "sessA")
    with store.connection() as conn:
        # File 1 flagged; file 2 done (pending PI); file 3 just claimed;
        # file 4 has two rows -> latest (needs_scoring) wins.
        conn.execute(
            "INSERT INTO review_state (file_id, user_email, status, "
            "created_at, updated_at) VALUES "
            "(1,'u','needs_scoring','t','t'),"
            "(2,'u','pending_pi_review','t','t'),"
            "(3,'u','claimed','t','t'),"
            "(4,'u','pending_pi_review','t','t'),"
            "(4,'u','needs_scoring','t','t')")
        conn.commit()
    m = store.review_statuses_for_files([1, 2, 3, 4, 999])
    assert m[1] == "needs_scoring"
    assert m[2] == "pending_pi_review"
    assert m[3] == "claimed"
    assert m[4] == "needs_scoring"   # latest row by id wins
    assert 999 not in m              # no row -> absent
    assert store.review_statuses_for_files([]) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
