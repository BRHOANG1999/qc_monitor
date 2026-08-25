"""DigestScheduler wiring for the weekly stim-stability digest: fires once on
Sunday >= hour, dedups across a simulated restart, and the state field
round-trips.

Run: pytest tests/test_stim_stability_scheduler.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


def _sunday_9am() -> datetime:
    d = datetime(2026, 8, 30, 9, 0, 0)
    while d.weekday() != 6:                 # 6 = Sunday
        d += timedelta(days=1)
    return d


def _cfg(tmp_path) -> dict:
    return {
        "notifications": {
            "state_file": str(tmp_path / "notif_state.json"),
            "stim_stability_weekly": {"enabled": True, "weekday": 6, "hour": 8},
        },
    }


def _scheduler(cfg, store):
    from src.alerting.email_alert import EmailAlerter
    import src.notifications.scheduler as sched_mod
    emailer = EmailAlerter.__new__(EmailAlerter)   # no SMTP config needed
    return sched_mod.DigestScheduler(cfg, emailer, store=store), sched_mod


def _patch_send(monkeypatch, sink):
    import src.notifications.stim_stability_weekly as mod
    monkeypatch.setattr(
        mod, "send_stim_stability_weekly",
        lambda day, cfg, store, emailer: sink.append(day) or {"sent": True})


def test_fires_once_on_sunday(tmp_path, monkeypatch):
    sink = []
    _patch_send(monkeypatch, sink)
    store = Store(str(tmp_path / "data" / "m.db"))
    sched, _ = _scheduler(_cfg(tmp_path), store)
    fired = sched.tick(now=_sunday_9am())
    assert fired.get("stim_stability_weekly") is True
    assert len(sink) == 1


def test_dedups_across_restart(tmp_path, monkeypatch):
    sink = []
    _patch_send(monkeypatch, sink)
    store = Store(str(tmp_path / "data" / "m.db"))
    now = _sunday_9am()
    s1, _ = _scheduler(_cfg(tmp_path), store)
    s1.tick(now=now)
    # Fresh scheduler on the same DB == a daemon restart, same day.
    s2, _ = _scheduler(_cfg(tmp_path), store)
    fired2 = s2.tick(now=now)
    assert "stim_stability_weekly" not in fired2
    assert len(sink) == 1                    # still exactly one send


def test_not_due_off_sunday_or_early(tmp_path, monkeypatch):
    sink = []
    _patch_send(monkeypatch, sink)
    store = Store(str(tmp_path / "data" / "m.db"))
    sched, _ = _scheduler(_cfg(tmp_path), store)
    saturday = _sunday_9am() - timedelta(days=1)
    assert "stim_stability_weekly" not in sched.tick(now=saturday)
    early = _sunday_9am().replace(hour=6)
    assert "stim_stability_weekly" not in sched.tick(now=early)
    assert sink == []


def test_disabled_never_fires(tmp_path, monkeypatch):
    sink = []
    _patch_send(monkeypatch, sink)
    store = Store(str(tmp_path / "data" / "m.db"))
    cfg = _cfg(tmp_path)
    cfg["notifications"]["stim_stability_weekly"]["enabled"] = False
    sched, _ = _scheduler(cfg, store)
    assert "stim_stability_weekly" not in sched.tick(now=_sunday_9am())
    assert sink == []


def test_state_field_round_trips(tmp_path):
    from src.notifications.scheduler import NotificationState, DigestScheduler
    st = NotificationState(stim_stability_weekly="2026-08-30")
    from dataclasses import asdict
    assert asdict(st)["stim_stability_weekly"] == "2026-08-30"
    # _load_state must restore it from the JSON file.
    import json
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"stim_stability_weekly": "2026-08-30"}))
    from src.alerting.email_alert import EmailAlerter
    cfg = {"notifications": {"state_file": str(p)}}
    sched = DigestScheduler(cfg, EmailAlerter.__new__(EmailAlerter))
    assert sched._state.stim_stability_weekly == "2026-08-30"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
