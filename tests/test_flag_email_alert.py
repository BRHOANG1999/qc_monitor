"""Per-animal flagged-event email opt-in (Overview seizure-status card).

Covers the store subscription/dedup layer and the AlertRuleEngine send path:
  - the checkbox persists a global per-animal opt-in;
  - a NEW auto-filter flagged file emails the configured recipients exactly
    once (deduped per file), so a standing flag pool never re-emails;
  - enabling seeds the current backlog so turning it on is not a mail-bomb;
  - a transient send failure releases the claim so the next tick retries.

Run: pytest tests/test_flag_email_alert.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.alerting.rules import AlertRuleEngine   # noqa: E402
from src.db.store import Store                    # noqa: E402


class _FakeEmailer:
    """Records sends; ``ok`` controls the success return the engine checks."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.sent = []

    def send(self, subject, body, severity):
        self.sent.append((subject, body, severity))
        return self.ok


def _flag(conn, fid, animal, has_video=1):
    conn.execute(
        "INSERT INTO processed_files (id, file_path, session_dir, has_video) "
        "VALUES (?,?,?,?)", (fid, f"/f{fid}.mat", "/s", has_video))
    conn.execute(
        "INSERT INTO review_event_log (file_id, user_email, action, "
        "payload_json, animal_id, at) VALUES (?,?,?,?,?,?)",
        (fid, "auto", "auto_filter_flag", "{}", animal, "2026-01-01"))


def _store(tmp_path):
    return Store(str(tmp_path / "data" / "m.db"))


def _engine(store, ok=True):
    eng = AlertRuleEngine(store, _FakeEmailer(ok=ok), {"alerting": {"rules": {}}})
    return eng, eng.emailer


# ------------------------------------------------------- subscription --- #

def test_subscription_roundtrip(tmp_path):
    s = _store(tmp_path)
    assert s.flag_email_subscribed_animals() == set()
    assert s.is_flag_email_subscribed("BCH062") is False
    s.set_flag_email_subscription("BCH062", True)
    assert s.is_flag_email_subscribed("BCH062") is True
    assert s.flag_email_subscribed_animals() == {"BCH062"}
    s.set_flag_email_subscription("BCH062", False)
    assert s.is_flag_email_subscribed("BCH062") is False
    assert s.flag_email_subscribed_animals() == set()


def test_flagged_file_ids_exclude_no_video(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH062")
        _flag(conn, 2, "BCH062")
        _flag(conn, 3, "BCH061")
        _flag(conn, 4, "BCH062", has_video=0)     # no video -> excluded
        conn.commit()
    ids = s.flagged_pool_file_ids_per_animal()
    assert ids["BCH062"] == {1, 2}
    assert ids["BCH061"] == {3}


# ------------------------------------------------------------ sending --- #

def test_no_subscription_no_email(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH062")
        conn.commit()
    eng, mail = _engine(s)
    eng.check_flagged_events()
    assert mail.sent == []


def test_new_flag_emails_once(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH062")
        conn.commit()
    s.set_flag_email_subscription("BCH062", True)
    eng, mail = _engine(s)
    eng.check_flagged_events()
    assert len(mail.sent) == 1
    assert "BCH062" in mail.sent[0][0]
    eng.check_flagged_events()                     # dedup: no re-send
    assert len(mail.sent) == 1


def test_other_animal_flag_does_not_email(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH061")                   # not subscribed
        conn.commit()
    s.set_flag_email_subscription("BCH062", True)
    eng, mail = _engine(s)
    eng.check_flagged_events()
    assert mail.sent == []


def test_enabling_seeds_baseline_so_backlog_is_not_emailed(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH062")
        _flag(conn, 2, "BCH062")
        conn.commit()
    s.set_flag_email_subscription("BCH062", True)
    seeded = s.seed_flag_email_baseline("BCH062")  # mirrors the UI toggle
    assert seeded == 2
    eng, mail = _engine(s)
    eng.check_flagged_events()
    assert mail.sent == []                          # standing backlog suppressed
    # A NEW flag after opt-in DOES email.
    with s.connection() as conn:
        _flag(conn, 3, "BCH062")
        conn.commit()
    eng.check_flagged_events()
    assert len(mail.sent) == 1


def test_send_failure_releases_claim_and_retries(tmp_path):
    s = _store(tmp_path)
    with s.connection() as conn:
        _flag(conn, 1, "BCH062")
        conn.commit()
    s.set_flag_email_subscription("BCH062", True)
    eng, mail = _engine(s, ok=False)               # SMTP down
    eng.check_flagged_events()
    assert len(mail.sent) == 1                       # attempted
    assert s.notification_already_sent(
        s.flag_email_digest("BCH062", 1), s.FLAG_EMAIL_SENT_KEY) is False
    eng.emailer.ok = True                            # SMTP recovers
    eng.check_flagged_events()
    assert len(mail.sent) == 2                        # retried and delivered


def test_check_never_raises_on_bad_state(tmp_path, monkeypatch):
    s = _store(tmp_path)
    s.set_flag_email_subscription("BCH062", True)
    monkeypatch.setattr(s, "flagged_pool_file_ids_per_animal",
                        lambda: (_ for _ in ()).throw(RuntimeError("db")))
    eng, mail = _engine(s)
    eng.check_flagged_events()                       # must not raise
    assert mail.sent == []
