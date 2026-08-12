"""bhz_csv cross-process lock: the dashboard and daemon both write the same
{date}_{animal}.csv in the split topology, so the read-modify-write must be
serialised ACROSS processes, not just across threads. This verifies two real
processes cannot hold the per-file lock at the same time.

Run with: pytest tests/test_bhz_csv_crossproc_lock.py -q
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _hold(lock_target: str, events, hold_sec: float) -> None:
    """Child process: take the cross-process CSV lock on *lock_target*, record its
    hold window into the shared *events* list, hold it briefly, release."""
    sys.path.insert(0, _ROOT)
    from src.utils.bhz_csv import _cross_proc_csv_lock
    with _cross_proc_csv_lock(Path(lock_target)):
        events.append(("enter", os.getpid(), time.time()))
        time.sleep(hold_sec)
        events.append(("exit", os.getpid(), time.time()))


def test_cross_proc_lock_is_mutually_exclusive(tmp_path):
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    events = mgr.list()
    target = str(tmp_path / "20260101_BCH040.csv")   # need not exist to be locked

    procs = [ctx.Process(target=_hold, args=(target, events, 0.4))
             for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"child failed: exitcode={p.exitcode}"

    ev = sorted(events, key=lambda e: e[2])          # by timestamp
    assert len(ev) == 4, ev
    # Mutual exclusion => the sequence is enter,exit,enter,exit (never two
    # consecutive 'enter's, which would mean overlapping hold windows).
    phases = [e[0] for e in ev]
    assert phases == ["enter", "exit", "enter", "exit"], phases
    # And the two holders are different PIDs.
    assert ev[0][1] != ev[2][1], "expected two distinct processes"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
