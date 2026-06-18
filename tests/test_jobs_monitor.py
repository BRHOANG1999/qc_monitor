"""list_active_jobs: union active + recent background jobs with FIFO queue
positions that mirror the single worker's drain order.

Run with: pytest tests/test_jobs_monitor.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.utils.mass_analyze import list_active_jobs  # noqa: E402


def _ma_job(conn, jid, animal, status, created, scanned=0, total=10):
    conn.execute(
        "INSERT INTO mass_analyze_job (id, pi_email, animal_id, cutoff, "
        "status, total_files, scanned_files, created_at) "
        "VALUES (?, 'pi@lab', ?, 0.05, ?, ?, ?, ?)",
        (jid, animal, status, total, scanned, created))


def _sc_job(conn, jid, animal, status, created):
    conn.execute(
        "INSERT INTO screen_eval_job (id, pi_email, animal_id, peak_cutoff, "
        "auc_threshold, auc_window_sec, status, total_files, scanned_files, "
        "created_at) VALUES (?, 'pi@lab', ?, 0.05, 0.1, 1.0, ?, 5, 0, ?)",
        (jid, animal, status, created))


def test_list_active_jobs_queue_order(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    with store.connection() as conn:
        _ma_job(conn, 1, "BCH040", "running", "2026-06-18T10:00:00", 3, 10)
        _ma_job(conn, 2, "BCH062", "pending", "2026-06-18T10:01:00")
        _ma_job(conn, 3, "BCH060", "pending", "2026-06-18T10:02:00")
        _ma_job(conn, 4, "BCH110", "done", "2026-06-18T09:00:00", 10, 10)
        _sc_job(conn, 5, "BCH061", "pending", "2026-06-18T10:00:30")
        conn.commit()

    jobs = list_active_jobs(store)
    by_scope = {e["scope"]: e for e in jobs}

    # Running -> queue_pos 0; carries scanned/total.
    assert by_scope["BCH040"]["status"] == "running"
    assert by_scope["BCH040"]["queue_pos"] == 0
    assert (by_scope["BCH040"]["scanned"], by_scope["BCH040"]["total"]) == (3, 10)

    # Pending scans ranked by created_at: B(2)->1, C(3)->2.
    assert by_scope["BCH062"]["queue_pos"] == 1
    assert by_scope["BCH060"]["queue_pos"] == 2
    # The shared worker drains all pending scans BEFORE any benchmark, even
    # though the screen_eval job was created earlier than BCH060's scan.
    assert by_scope["BCH061"]["kind"] == "screen_eval"
    assert by_scope["BCH061"]["queue_pos"] == 3

    # Terminal job present, no queue position.
    assert by_scope["BCH110"]["status"] == "done"
    assert by_scope["BCH110"]["queue_pos"] is None

    # Kinds tagged.
    assert by_scope["BCH040"]["kind"] == "mass_analyze"


def test_empty_db_no_jobs(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    assert list_active_jobs(store) == []


def test_recent_limit_excludes_old(tmp_path):
    store = Store(str(tmp_path / "data" / "monitor.db"))
    with store.connection() as conn:
        for i in range(5):
            _ma_job(conn, i + 1, f"A{i}", "done",
                    f"2026-06-1{i}T08:00:00", 10, 10)
        conn.commit()
    assert len(list_active_jobs(store, recent_limit=2)) == 2
    assert list_active_jobs(store, recent_limit=0) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
