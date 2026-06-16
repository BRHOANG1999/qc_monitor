"""Phase 3: per-session statistics.

Run with: pytest tests/test_session_stats.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
import src.utils.mass_analyze as ma  # noqa: E402


def test_session_review_stats(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    sd = "sessA"
    with store.connection() as conn:
        for fid in (1, 2, 3, 4):
            conn.execute(
                "INSERT INTO processed_files (id, file_path, "
                "session_dir) VALUES (?, ?, ?)",
                (fid, f"/f/{fid}.mat", sd))
        conn.commit()
    # file1: approved with 2 events (max racine 5); file2: approved
    # no-events; file3: needs_scoring 1 draft; file4: untouched.
    store.mark_review(1, "u", "pi_approved", markers=[
        {"type": "LVF", "racine": 3}, {"type": "HYP", "racine": 5}])
    store.mark_review(2, "u", "pi_approved", markers=[])
    store.mark_review(3, "u", "needs_scoring",
                       markers=[{"EO_sec": 1.0, "draft": True}])

    # Only file1 has stim.
    monkeypatch.setattr(
        ma, "has_stim_for_file",
        lambda s, fid, sdir=None: fid == 1)

    st = store.session_review_stats(sd)
    assert st["n_files"] == 4
    assert st["n_reviewed"] == 2          # file1 + file2
    assert st["n_approved"] == 2
    assert st["n_needs_scoring"] == 1
    assert st["n_files_with_events"] == 1  # file1
    assert st["n_events"] == 2
    assert st["max_racine"] == 5
    assert st["n_no_event_files"] == 1     # file2
    assert st["n_stim_files"] == 1         # file1
    assert st["n_during_stim_events"] == 2  # file1's 2 events


def test_session_review_stats_empty():
    # No session -> zeroed dict, no raise.
    import tempfile
    store = Store(os.path.join(tempfile.mkdtemp(), "data", "m.db"))
    st = store.session_review_stats("")
    assert st["n_files"] == 0 and st["n_events"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
