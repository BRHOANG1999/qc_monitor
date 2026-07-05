"""Restart-proof digest dedup + current-fidelity store helpers.

Two concerns, both in src/db/store.py:

* ``record_notification_sent`` / ``notification_already_sent`` -- the atomic
  check-and-set that makes "send a digest at most once per calendar day"
  survive daemon restarts (the JSON state file could revert and re-fire, which
  is exactly the duplicate-surgery-reminder bug this fixes).
* ``latest_current_fidelity_per_channel`` + the ``_current_fidelity_pct``
  helper -- the two-edge current-delivery fidelity metric.

Run with: pytest tests/test_notification_dedup.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store, _current_fidelity_pct  # noqa: E402


def _db_path(tmp_path) -> str:
    return str(tmp_path / "data" / "monitor.db")


# ===================================================================== #
#  Notification dedup (atomic check-and-set)
# ===================================================================== #

def test_record_notification_sent_is_atomic_check_and_set(tmp_path):
    s = Store(_db_path(tmp_path))
    # First claim for the day wins.
    assert s.record_notification_sent("surgery", "2026-07-05") is True
    # Second claim for the SAME day loses (already sent) -- the core of the
    # duplicate-reminder fix.
    assert s.record_notification_sent("surgery", "2026-07-05") is False
    # A different day is a fresh claim.
    assert s.record_notification_sent("surgery", "2026-07-06") is True
    # A different digest, same day, is independent.
    assert s.record_notification_sent("eod", "2026-07-05") is True


def test_notification_already_sent_reflects_claims(tmp_path):
    s = Store(_db_path(tmp_path))
    assert s.notification_already_sent("surgery", "2026-07-05") is False
    s.record_notification_sent("surgery", "2026-07-05")
    assert s.notification_already_sent("surgery", "2026-07-05") is True
    assert s.notification_already_sent("surgery", "2026-07-06") is False


def test_claim_survives_a_simulated_restart(tmp_path):
    """A new Store on the same DB (a daemon restart) still sees the claim, so
    the digest can't re-fire -- the JSON state file's failure mode."""
    db = _db_path(tmp_path)
    assert Store(db).record_notification_sent("surgery", "2026-07-05") is True
    # Fresh Store instance == process restart.
    assert Store(db).record_notification_sent("surgery", "2026-07-05") is False


# ===================================================================== #
#  DigestScheduler: at most one send per day across restarts
# ===================================================================== #

class _FakeEmailer:
    def __init__(self):
        self.sent = []

    def send(self, subject, body, severity="info"):
        self.sent.append((subject, body, severity))


def _surgery_config():
    return {"surgery_digest": {"enabled": True, "send_hour": 8}}


def test_scheduler_sends_surgery_once_across_restarts(tmp_path, monkeypatch):
    from datetime import datetime
    from src.alerting.email_alert import EmailAlerter
    import src.notifications.scheduler as sched_mod

    sends = []
    monkeypatch.setattr(
        sched_mod, "send_today_digest",
        lambda day, cfg, emailer: sends.append(day) or {"sent": True})

    db = _db_path(tmp_path)
    store = Store(db)
    cfg = _surgery_config()
    # Point the JSON state at the tmp dir so we don't touch the real one.
    cfg["notifications"] = {"state_file": str(tmp_path / "notif_state.json")}
    emailer = EmailAlerter.__new__(EmailAlerter)  # avoid SMTP config needs
    now = datetime(2026, 7, 5, 9, 0, 0)

    sched1 = sched_mod.DigestScheduler(cfg, emailer, store=store)
    fired1 = sched1.tick(now=now)
    assert fired1.get("surgery") is True
    assert len(sends) == 1

    # Simulate a daemon restart: brand-new scheduler on the same DB, same day.
    sched2 = sched_mod.DigestScheduler(cfg, emailer, store=store)
    fired2 = sched2.tick(now=now)
    assert "surgery" not in fired2          # blocked by the DB claim
    assert len(sends) == 1                  # still exactly one send


# ===================================================================== #
#  Current-delivery fidelity helper + latest-per-channel
# ===================================================================== #

def test_current_fidelity_pct_zero_when_edges_agree():
    # BCH062SR: reversal 0.397 / offset 0.395 → ≈ +0.5%.
    f = _current_fidelity_pct(0.397, 0.395)
    assert abs(f - 0.505) < 0.05


def test_current_fidelity_pct_negative_when_reversal_sags():
    # Reversal edge lower than offset → current not fully delivered.
    f = _current_fidelity_pct(0.30, 0.40)
    assert f < 0


def test_current_fidelity_pct_none_when_edge_missing():
    assert _current_fidelity_pct(None, 0.4) is None
    assert _current_fidelity_pct(0.4, None) is None
    assert _current_fidelity_pct(0.0, 0.0) is None
