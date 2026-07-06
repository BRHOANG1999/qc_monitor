"""Auto-filter: flag KEPT (has-events) files + count/reason bookkeeping.

The sweep triages every pending file into CLEAR (-> pending_pi_review) or FLAG
(crosses a screen -> has candidate events -> a one-time auto_filter_flag audit
event, no state change so it stays in the reviewer queue) or ERROR. These tests
pin the flag-once idempotency, the count/reason bookkeeping, and that
screen_file_verdict carries the peak/event counts used by the flag + logging.

Run with: pytest tests/test_auto_filter_flag.py -q
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.utils import mass_analyze as ma  # noqa: E402


def _cfg():
    return {"animal_id": "BCH111", "enabled": 1, "screen_mode": "both",
            "peak_cutoff": 0.05, "auc_threshold": 0.2, "auc_window_sec": 5.0}


def _store(tmp_path, file_ids):
    """Fresh Store with processed_files rows for the given ids (the
    review_event_log FK needs them to exist)."""
    s = Store(str(tmp_path / "data" / "m.db"))
    with s.connection() as conn:
        for fid in file_ids:
            conn.execute("INSERT INTO processed_files (id, file_path) "
                         "VALUES (?, ?)", (fid, f"/fake/{fid}.mat"))
        conn.commit()
    return s


def _keep_env(fid=None):
    return ("keep", {"channel": 5, "env_pos": True, "auc_pos": False,
                     "n_peaks": 3, "n_events": 0,
                     "screens": ["auc", "envelope"]})


# ===================================================================== #
#  Flag once, keep counting
# ===================================================================== #

def test_flag_written_once_then_idempotent(tmp_path, monkeypatch):
    fids = [101, 102, 103]
    store = _store(tmp_path, fids)
    files = [{"file_id": i, "session_dir": "/s"} for i in fids]
    monkeypatch.setattr(ma, "pending_files_for_animal",
                        lambda s, a, **k: files)
    monkeypatch.setattr(ma, "screen_file_verdict",
                        lambda *a, **k: _keep_env())

    r1 = ma.auto_screen_for_animal(store, "BCH111", _cfg())
    assert r1["n_cleared"] == 0
    assert r1["n_flagged"] == 3
    assert r1["n_newly_flagged"] == 3          # all three brand-new
    assert r1["reasons"] == {"env_only": 3, "auc_only": 0, "both": 0}
    assert r1["pool"] == 3
    assert store.auto_filter_flagged_file_ids("BCH111") == set(fids)

    # A second sweep still SEES them as flagged (has events -> stay in queue)
    # but writes NO new audit events -> idempotent.
    r2 = ma.auto_screen_for_animal(store, "BCH111", _cfg())
    assert r2["n_flagged"] == 3
    assert r2["n_newly_flagged"] == 0
    # Exactly one auto_filter_flag event per file, not two.
    with store.connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM review_event_log "
                         "WHERE action='auto_filter_flag'").fetchone()[0]
    assert n == 3


def test_mixed_verdicts_count_and_classify(tmp_path, monkeypatch):
    fids = [1, 2, 3, 4, 5]
    store = _store(tmp_path, fids)
    files = [{"file_id": i, "session_dir": "/s"} for i in fids]
    monkeypatch.setattr(ma, "pending_files_for_animal",
                        lambda s, a, **k: files)
    # 1 clears, 2 env-only, 3 auc-only, 4 both, 5 errors.
    verdicts = {
        1: ("clear", {"channel": 5, "env_pos": False, "auc_pos": False,
                      "n_peaks": 0, "n_events": 0, "screens": ["envelope"]}),
        2: ("keep", {"channel": 5, "env_pos": True, "auc_pos": False,
                     "n_peaks": 2, "n_events": 0, "screens": ["envelope"]}),
        3: ("keep", {"channel": 5, "env_pos": False, "auc_pos": True,
                     "n_peaks": 0, "n_events": 1, "screens": ["auc"]}),
        4: ("keep", {"channel": 5, "env_pos": True, "auc_pos": True,
                     "n_peaks": 4, "n_events": 2, "screens": ["auc",
                                                              "envelope"]}),
        5: ("keep", {"reason": "error"}),
    }
    monkeypatch.setattr(ma, "pending_files_for_animal",
                        lambda s, a, **k: files)
    monkeypatch.setattr(ma, "screen_file_verdict",
                        lambda s, fid, *a, **k: verdicts[fid])
    # Don't touch review_state (FK / status churn) -- just confirm the clear
    # path is counted.
    monkeypatch.setattr(ma, "_apply_auto_clear", lambda *a, **k: True)

    r = ma.auto_screen_for_animal(store, "BCH111", _cfg())
    assert r["n_cleared"] == 1
    assert r["n_flagged"] == 3
    assert r["n_error"] == 1
    assert r["pool"] == 5
    assert r["reasons"] == {"env_only": 1, "auc_only": 1, "both": 1}


def test_disabled_and_no_threshold_are_noops(tmp_path):
    store = _store(tmp_path, [])
    off = ma.auto_screen_for_animal(store, "BCH111",
                                    {"animal_id": "BCH111", "enabled": 0})
    assert off["reason"] == "disabled"
    assert off["n_flagged"] == 0 and off["n_cleared"] == 0
    nothr = ma.auto_screen_for_animal(
        store, "BCH111",
        {"animal_id": "BCH111", "enabled": 1, "screen_mode": "both"})
    assert nothr["reason"] == "no_usable_threshold"


# ===================================================================== #
#  Verdict carries the peak/event counts (used by flag payload + logging)
# ===================================================================== #

def test_verdict_carries_counts(monkeypatch):
    monkeypatch.setattr(ma, "animal_channel_index", lambda *a, **k: 5)
    monkeypatch.setattr(ma, "get_or_compute_peak_count",
                        lambda *a, **k: SimpleNamespace(n_peaks=4))
    monkeypatch.setattr(ma, "get_or_compute_auc_count",
                        lambda *a, **k: SimpleNamespace(n_events=0))
    v, det = ma.screen_file_verdict(None, 1, "/s", "BCH111", _cfg())
    assert v == "keep"                 # env positive -> crosses -> keep
    assert det["n_peaks"] == 4
    assert det["n_events"] == 0
    assert det["env_pos"] is True
    assert det["auc_pos"] is False


def test_verdict_clear_when_all_zero(monkeypatch):
    monkeypatch.setattr(ma, "animal_channel_index", lambda *a, **k: 5)
    monkeypatch.setattr(ma, "get_or_compute_peak_count",
                        lambda *a, **k: SimpleNamespace(n_peaks=0))
    monkeypatch.setattr(ma, "get_or_compute_auc_count",
                        lambda *a, **k: SimpleNamespace(n_events=0))
    v, det = ma.screen_file_verdict(None, 1, "/s", "BCH111", _cfg())
    assert v == "clear"
    assert det["n_peaks"] == 0 and det["n_events"] == 0
