"""Count-based peri-ictal trigger in the DigestScheduler: arms a baseline that
excludes the pre-existing backlog, fires an animal's peri-ictal email once it
accumulates >= threshold NEW needs_scoring entries, is per-animal, and dedups
across a restart.

Run: pytest tests/test_peri_stim_trigger.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402

ANIMAL_A, ANIMAL_B = "BCH111", "BCH222"


class _ImmediateThread:
    """Runs the target synchronously on start() so the test needn't join."""

    def __init__(self, target=None, name=None, daemon=None):
        self._target = target

    def start(self):
        if self._target:
            self._target()


def _seed(tmp_path) -> Store:
    store = Store(str(tmp_path / "data" / "m.db"))
    now = datetime.now().isoformat()
    with store.connection() as conn:
        for i in range(1, 80):
            conn.execute(
                """INSERT INTO processed_files
                   (id, file_path, session_dir, session_name,
                    chunk_datetime, has_video)
                   VALUES (?, ?, 'S1', 'S1', ?, 1)""",
                (i, f"/fake/f{i}.mat", now))
        conn.commit()
    return store


_next_file = {"i": 1}


def _quick_flag(store, animal, n=1):
    for _ in range(n):
        fid = _next_file["i"]
        _next_file["i"] += 1
        store.mark_review(fid, "rev@lab.org", "needs_scoring",
                          markers=[{"EO_sec": 1.0, "racine": 4}],
                          animal_id=animal)


def _cfg(tmp_path, threshold=5):
    return {"notifications": {
        "state_file": str(tmp_path / "notif_state.json"),
        "peri_stim": {"enabled": True, "threshold": threshold},
    }}


def _scheduler(cfg, store, monkeypatch, sink):
    from src.alerting.email_alert import EmailAlerter
    import src.notifications.scheduler as sched_mod
    import src.notifications.peri_stim_needs_scoring as sender
    monkeypatch.setattr(sched_mod, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        sender, "send_peri_stim_artifact",
        lambda animal, cfg, store, emailer, reason="", out_dir=None:
        sink.append(animal) or {"sent": True, "animal": animal})
    emailer = EmailAlerter.__new__(EmailAlerter)
    return sched_mod.DigestScheduler(cfg, emailer, store=store)


def test_baseline_excludes_backlog_then_fires_per_animal(tmp_path, monkeypatch):
    _next_file["i"] = 1
    sink = []
    store = _seed(tmp_path)
    _quick_flag(store, ANIMAL_A, 10)               # pre-existing backlog
    sched = _scheduler(_cfg(tmp_path), store, monkeypatch, sink)
    # First tick only ARMS the baseline -- the 10 backlog events must not fire.
    fired0 = sched.tick(now=datetime(2026, 9, 21, 10, 0))
    assert fired0.get("peri_stim_baseline") is True
    assert sink == []
    # 5 new for A (fires), 3 new for B (below threshold).
    _quick_flag(store, ANIMAL_A, 5)
    _quick_flag(store, ANIMAL_B, 3)
    fired1 = sched.tick(now=datetime(2026, 9, 21, 10, 1))
    assert sink == [ANIMAL_A]
    assert fired1.get(f"peri_stim:{ANIMAL_A}") is True
    assert f"peri_stim:{ANIMAL_B}" not in fired1
    # 2 more for B -> reaches 5 -> fires; A stays quiet (already consumed).
    _quick_flag(store, ANIMAL_B, 2)
    sched.tick(now=datetime(2026, 9, 21, 10, 2))
    assert sink == [ANIMAL_A, ANIMAL_B]
    # No new events -> no more fires.
    sched.tick(now=datetime(2026, 9, 21, 10, 3))
    assert sink == [ANIMAL_A, ANIMAL_B]


def test_dedups_across_restart(tmp_path, monkeypatch):
    _next_file["i"] = 1
    sink = []
    store = _seed(tmp_path)
    sched = _scheduler(_cfg(tmp_path), store, monkeypatch, sink)
    sched.tick(now=datetime(2026, 9, 21, 10, 0))     # arm baseline
    _quick_flag(store, ANIMAL_A, 5)
    sched.tick(now=datetime(2026, 9, 21, 10, 1))
    assert sink == [ANIMAL_A]
    # Fresh scheduler on the same state file == a daemon restart.
    sched2 = _scheduler(_cfg(tmp_path), store, monkeypatch, sink)
    sched2.tick(now=datetime(2026, 9, 21, 10, 2))
    assert sink == [ANIMAL_A]                        # no re-fire of the batch


def test_disabled_never_fires(tmp_path, monkeypatch):
    _next_file["i"] = 1
    sink = []
    store = _seed(tmp_path)
    cfg = _cfg(tmp_path)
    cfg["notifications"]["peri_stim"]["enabled"] = False
    sched = _scheduler(cfg, store, monkeypatch, sink)
    _quick_flag(store, ANIMAL_A, 8)
    sched.tick(now=datetime(2026, 9, 21, 10, 0))
    assert sink == []


def test_state_round_trips(tmp_path):
    import json
    from src.alerting.email_alert import EmailAlerter
    from src.notifications.scheduler import DigestScheduler
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"peri_stim_baseline_id": 42,
                             "peri_stim_marks": {"BCH111": 50}}))
    cfg = {"notifications": {"state_file": str(p)}}
    sched = DigestScheduler(cfg, EmailAlerter.__new__(EmailAlerter))
    assert sched._state.peri_stim_baseline_id == 42
    assert sched._state.peri_stim_marks == {"BCH111": 50}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
