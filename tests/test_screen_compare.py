"""Tests for the two-screen comparison benchmark.

Covers the ground-truth mapping, the AUC cache roundtrip, the
confusion-matrix accumulation in run_screen_benchmark, and the
pool_files two-screen split.

Run with: pytest tests/test_screen_compare.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.utils.mass_analyze as ma  # noqa: E402
from src.db.store import Store  # noqa: E402


# --------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------- #

def test_file_ground_truth_mapping():
    g = ma.file_ground_truth
    assert g("has_events", None) == "pos"
    assert g("no_events", "[]") == "neg"
    assert g("pi_approved", json.dumps([{"type": "LVF"}])) == "pos"
    assert g("pi_approved", "[]") == "neg"
    assert g("pi_approved", None) == "neg"
    assert g("pending_pi_review", "[]") is None
    assert g("claimed", None) is None
    assert g("abandoned", None) is None


# --------------------------------------------------------------- #
# AUC cache roundtrip
# --------------------------------------------------------------- #

def test_auc_cache_read_skips_compute(tmp_path, monkeypatch):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path) "
            "VALUES (1, '/f/a.mat')")
        conn.execute(
            """INSERT INTO envelope_auc_cache
               (file_id, channel, auc_threshold, window_sec,
                min_peak_dist_sec, n_events, computed_at)
               VALUES (1, 0, 0.5, 5.0, 150.0, 3, '2026-01-01')""")
        conn.commit()

    # If the cache is honoured, the uncached computer never runs.
    def _boom(*a, **k):
        raise AssertionError("should have hit the cache")
    monkeypatch.setattr(ma, "_compute_auc_count_uncached", _boom)

    res = ma.get_or_compute_auc_count(store, 1, 0, 0.5, 5.0)
    assert res.n_events == 3
    assert res.auc_threshold == 0.5 and res.window_sec == 5.0


# --------------------------------------------------------------- #
# Confusion-matrix accumulation
# --------------------------------------------------------------- #

def test_tally():
    c = {k: 0 for k in ("x_tp", "x_fp", "x_tn", "x_fn")}
    ma._tally(c, "x", pred_pos=True, truth_pos=True)    # TP
    ma._tally(c, "x", pred_pos=False, truth_pos=True)   # FN
    ma._tally(c, "x", pred_pos=True, truth_pos=False)   # FP
    ma._tally(c, "x", pred_pos=False, truth_pos=False)  # TN
    assert (c["x_tp"], c["x_fn"], c["x_fp"], c["x_tn"]) == (1, 1, 1, 1)


def test_run_screen_benchmark_accumulates(tmp_path, monkeypatch):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    job_id = ma.create_screen_eval_job(
        store, "pi@lab", "BCH001", 0.05, 0.5, 5.0)

    # Synthetic labeled set: (file_id, truth, env_pos, auc_pos).
    # Designed so the AUC screen has fewer false positives.
    fixtures = [
        (1, "pos", True,  True),   # both right
        (2, "pos", True,  False),  # env right, auc misses -> env-only-true
        (3, "neg", True,  False),  # env false alarm -> env-only-false
        (4, "neg", False, False),  # both right (TN)
        (5, "pos", False, True),   # auc right, env misses -> auc-only-true
    ]
    truth_by_id = {fid: t for (fid, t, _e, _a) in fixtures}
    screen_by_id = {fid: (e, a) for (fid, _t, e, a) in fixtures}

    monkeypatch.setattr(
        ma, "labeled_files_for_animal",
        lambda store, animal: [
            {"file_id": fid, "session_dir": "s", "truth": t}
            for (fid, t, _e, _a) in fixtures])
    monkeypatch.setattr(
        ma, "first_animal_channel_index", lambda store, sd: 0)
    monkeypatch.setattr(
        ma, "screen_file",
        lambda store, fid, ch, pc, at, win, **k: screen_by_id[fid])

    out = ma.run_screen_benchmark(store, job_id)
    assert out["status"] == "done"

    job = ma.get_screen_eval_job(store, job_id)
    # Envelope screen: TP from files 1,2; FP from 3; TN from 4,5.
    assert job["p1_tp"] == 2
    assert job["p1_fp"] == 1
    assert job["p1_tn"] == 1   # file 4
    assert job["p1_fn"] == 1   # file 5 (pos, env said no)
    # AUC screen: TP from 1,5; FP 0; TN from 3,4; FN from 2.
    assert job["p2_tp"] == 2
    assert job["p2_fp"] == 0
    assert job["p2_tn"] == 2
    assert job["p2_fn"] == 1
    # Disagreement breakdown.
    assert job["env_only_true"] == 1   # file 2
    assert job["env_only_false"] == 1  # file 3
    assert job["auc_only_true"] == 1   # file 5
    assert job["auc_only_false"] == 0
    assert job["scanned_files"] == 5


# --------------------------------------------------------------- #
# pool_files two-screen split
# --------------------------------------------------------------- #

def test_pool_files_two_screen_split(tmp_path, monkeypatch):
    db = str(tmp_path / "data" / "monitor.db")
    store = Store(db)
    # file -> (n_peaks, n_auc_events)
    fixtures = {10: (2, 2), 11: (1, 0), 12: (0, 3), 13: (0, 0)}
    with store.connection() as conn:
        for fid, (npk, nauc) in fixtures.items():
            conn.execute(
                "INSERT INTO processed_files (id, file_path) "
                "VALUES (?, ?)", (fid, f"/f/{fid}.mat"))
            conn.execute(
                """INSERT INTO envelope_peak_cache
                   (file_id, channel, cutoff, min_peak_dist_sec,
                    n_peaks, peak_times_json, computed_at)
                   VALUES (?, 0, 0.05, 150.0, ?, '[]', 'now')""",
                (fid, npk))
            conn.execute(
                """INSERT INTO envelope_auc_cache
                   (file_id, channel, auc_threshold, window_sec,
                    min_peak_dist_sec, n_events, computed_at)
                   VALUES (?, 0, 0.5, 5.0, 150.0, ?, 'now')""",
                (fid, nauc))
        conn.commit()

    monkeypatch.setattr(
        ma, "pending_files_for_animal",
        lambda store, animal: [
            {"file_id": fid, "session_dir": "s"}
            for fid in fixtures])
    monkeypatch.setattr(
        ma, "first_animal_channel_index", lambda store, sd: 0)

    pools = ma.pool_files(store, "BCH001", 0.05, 0.5, 5.0)
    assert pools["pool1"] == [10, 11]        # envelope positives
    assert pools["pool2"] == [10, 12]        # AUC positives
    # Disagreement: 11 envelope-only, 12 auc-only.
    p3 = {e["file_id"]: e["only"] for e in pools["pool3"]}
    assert p3 == {11: "envelope", 12: "auc"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
