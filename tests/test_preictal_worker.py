"""Pre-ictal worker: drains a queued job through the engine to 'done', and
respects the enabled flag.

Run with: pytest tests/test_preictal_worker.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.preictal import worker  # noqa: E402


def _cfg(tmp_path):
    return {"preictal": {"enabled": True,
                         "derivatives_root": str(tmp_path / "deriv"),
                         "features": ["*"],
                         "worker": {"enabled": True, "poll_interval_sec": 1},
                         "events": {"post_ictal_buffer_sec": 300.0},
                         "trajectory": {"step_sec": 1.0},
                         "cwt": {"min_leadtime_sec": 1.0}}}


def test_worker_runs_queued_job(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    # No seizures for this animal -> a fast, clean run.
    jid = store.enqueue_preictal_job("adhoc", None, None, ["BCH999"])
    job = store.claim_next_preictal_job()
    assert job["id"] == jid
    worker._run_one(store, _cfg(tmp_path), job)
    assert store.preictal_job_status(jid) == "done"
    run = store.latest_preictal_run("adhoc")
    assert run and run["status"] == "done"
    # The job links to the run it produced.
    with store.connection() as conn:
        rid = conn.execute("SELECT run_id FROM preictal_job WHERE id=?",
                           (jid,)).fetchone()["run_id"]
    assert rid == run["id"]


def test_start_worker_noop_when_disabled(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    # enabled: False -> must not raise and must not start a thread.
    worker.start_worker(store, {"preictal": {"enabled": False}})
    assert worker._worker_started is False
