"""Health watchdog: detection, bounded remediation ledger, and the cycle decisions.

No real services or DB are touched (subprocess/service calls are monkeypatched or
avoided). Run: pytest tests/test_health_watchdog.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.health import checks as ck            # noqa: E402
from src.health import remediate as rem        # noqa: E402
from src.health import watchdog as wd          # noqa: E402


# --------------------------- checks --------------------------- #
def _write_log(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_recent_error_labels_time_filtered(tmp_path):
    now = datetime(2026, 9, 29, 10, 0, 0)
    log = tmp_path / "qc.log"
    _write_log(log, [
        "[2026-09-29 03:00:00] ERROR scheduler - Stim-stability daily send raised: x",
        "[2026-09-29 09:50:00] ERROR scheduler - Stim-stability daily send raised: x",
        "[2026-09-29 09:55:00] INFO scheduler - Stim-stability daily fire",
    ])
    hits = ck._recent_error_labels(str(log), now - timedelta(hours=1))
    assert len(hits) == 1 and "09:50:00" in hits[0]


def test_job_overdue_daily_and_weekly():
    now = datetime(2026, 9, 29, 12, 0, 0)
    daily = ck.Job("j", ("j",), "daily", "J")
    weekly = ck.Job("w", ("w",), "weekly", "W")
    # past 08:00 + 3h grace, last sent yesterday -> overdue
    assert ck._job_overdue(daily, {"hour": 8}, "2026-09-28", now, 3.0) is True
    # sent today -> not overdue
    assert ck._job_overdue(daily, {"hour": 8}, "2026-09-29", now, 3.0) is False
    # weekly sent 9 days ago -> overdue
    assert ck._job_overdue(weekly, {}, "2026-09-20", now, 3.0) is True
    assert ck._job_overdue(weekly, {}, "2026-09-26", now, 3.0) is False


def test_check_email_jobs_flags_error_loop(tmp_path):
    now = datetime(2026, 9, 29, 12, 0, 0)
    log = tmp_path / "qc.log"
    state = tmp_path / "notif.json"
    _write_log(log, [
        "[2026-09-29 11:50:00] ERROR s - Stim-stability daily send raised: boom",
        "[2026-09-29 11:52:00] ERROR s - Stim-stability daily send raised: boom",
    ])
    state.write_text('{"stim_stability_daily": "2026-09-28"}', encoding="utf-8")
    config = {
        "health": {"log_file": str(log)},
        "notifications": {"state_file": str(state),
                          "stim_stability_daily": {"enabled": True, "hour": 8}},
    }
    res = {r.name: r for r in ck.check_email_jobs(config, now=now)}
    r = res["job:stim_stability_daily"]
    assert r.status == "fail" and r.remedy == "fixjob:stim_stability_daily"


def test_eval_cpu_runaway_flags_pegging_and_bloated():
    now = 1000.0
    current = {
        111: {"name": "MATLABWindow", "cpu_s": 2000.0, "create": 5.0},  # pegging
        222: {"name": "MATLAB", "cpu_s": 200000.0, "create": 6.0},      # bloated
        333: {"name": "MATLAB", "cpu_s": 100.0, "create": 7.0},         # fine
    }
    prev = {
        "111": {"cpu_s": 800.0, "create": 5.0, "ts": 400.0},   # +1200 CPU-s / 600 s = 2 cores
        "222": {"cpu_s": 199900.0, "create": 6.0, "ts": 400.0},
        "333": {"cpu_s": 90.0, "create": 7.0, "ts": 400.0},
    }
    flagged = {f["pid"]: f for f in ck._eval_cpu_runaway(
        current, prev, now, cores_thresh=1.5, total_hours_thresh=24.0)}
    assert 111 in flagged and flagged[111]["reason"] == "pegging cores"
    assert flagged[111]["cores"] == 2.0
    assert 222 in flagged and flagged[222]["reason"] == "huge cumulative CPU"
    assert 333 not in flagged


def test_eval_cpu_runaway_ignores_reused_pid():
    # pid 111 reused: create time differs -> no core-rate computed, cpu_s small -> fine
    now = 1000.0
    current = {111: {"name": "MATLAB", "cpu_s": 50.0, "create": 950.0}}
    prev = {"111": {"cpu_s": 800.0, "create": 5.0, "ts": 400.0}}
    flagged = ck._eval_cpu_runaway(current, prev, now, cores_thresh=1.5,
                                   total_hours_thresh=24.0)
    assert flagged == []


def test_check_services_flags_stopped(monkeypatch):
    monkeypatch.setattr(ck, "_service_status",
                        lambda name: "Stopped" if name == "QCMonitorDaemon" else "Running")
    config = {"health": {"services": ["QCMonitorDaemon", "QCMonitorDashboard"]}}
    out = {r.name: r for r in ck.check_services(config)}
    assert out["service:QCMonitorDaemon"].status == "fail"
    assert out["service:QCMonitorDaemon"].remedy == "restart:QCMonitorDaemon"
    assert out["service:QCMonitorDaemon"].severity == "critical"
    assert out["service:QCMonitorDashboard"].status == "ok"


# --------------------------- remediation ledger --------------------------- #
def _ledger(tmp_path):
    cfg = {"health": {"state_file": str(tmp_path / "health.json"),
                      "cooldown_sec": 100.0, "max_attempts": 3,
                      "escalate_window_sec": 10_000.0}}
    return rem.RemediationLedger(cfg)


def test_ledger_cooldown_and_backoff(tmp_path):
    lg = _ledger(tmp_path)
    key = "restart:QCMonitorDaemon"
    t = 1000.0
    ok, _ = lg.allowed(key, now=t)
    assert ok
    lg.record(key, success=False, now=t)
    # within cooldown -> blocked
    assert lg.allowed(key, now=t + 50)[0] is False
    # after 1st backoff (100 * 2^1 = 200) -> allowed
    assert lg.allowed(key, now=t + 250)[0] is True


def test_ledger_escalates_after_max_attempts(tmp_path):
    lg = _ledger(tmp_path)
    key = "restart:QCMonitorDaemon"
    # cooldown=100, backoff x2^attempts -> clear each backoff but stay in the 10_000s
    # window so the 3 attempts count together and escalate.
    for t in (0.0, 200.0, 600.0):
        assert lg.allowed(key, now=t)[0] is True
        lg.record(key, success=False, now=t)
    # after 3 failed attempts within the window -> escalated, not allowed
    allowed, why = lg.allowed(key, now=700.0)
    assert allowed is False
    assert key in lg.escalated_keys() or "escalat" in why.lower()


def test_ledger_success_resets(tmp_path):
    lg = _ledger(tmp_path)
    key = "restart:svc"
    lg.record(key, success=False, now=0.0)
    lg.record(key, success=True, now=10.0)
    assert lg.allowed(key, now=10_000.0)[0] is True
    assert key not in lg.escalated_keys()


def test_apply_remedies_dry_run(tmp_path):
    lg = _ledger(tmp_path)
    results = [ck.HealthResult("service:X", "fail", "down", remedy="restart:X")]
    actions = rem.apply_remedies(results, lg, dry_run=True)
    assert actions[0]["attempted"] is False
    assert "dry-run" in actions[0]["note"]


def test_fixjob_releases_claim_and_restarts(tmp_path, monkeypatch):
    lg = _ledger(tmp_path)
    released = {}

    class FakeStore:
        def release_notification_sent(self, digest, date):
            released["digest"] = digest

    monkeypatch.setattr(rem, "restart_service", lambda name, **k: (True, "restarted"))
    results = [ck.HealthResult("job:stim_stability_daily", "fail", "loop",
                               remedy="fixjob:stim_stability_daily")]
    actions = rem.apply_remedies(results, lg, dry_run=False, store=FakeStore())
    assert actions[0]["attempted"] is True and actions[0]["ok"] is True
    assert released.get("digest") == "stim_stability_daily"
    assert "released" in actions[0]["note"]


# --------------------------- watchdog cycle --------------------------- #
def test_overall_and_should_email():
    ok = [ck.HealthResult("a", "ok", "")]
    warn = [ck.HealthResult("a", "warn", "")]
    fail = [ck.HealthResult("a", "fail", "", remedy="restart:X")]
    assert wd._overall(ok) == "ok"
    assert wd._overall(warn) == "warn"
    assert wd._overall(fail) == "fail"
    # email on fail, on attempted action, and on recovery; not on steady healthy
    assert wd._should_email("fail", "ok", [], False) is True
    assert wd._should_email("ok", "fail", [], False) is True          # recovery
    assert wd._should_email("ok", "ok", [], False) is False
    assert wd._should_email("warn", "warn",
                            [{"attempted": True, "ok": True}], False) is True
    # a NEWLY appearing warn (e.g. a cpu runaway) alerts once; steady warn stays quiet
    assert wd._should_email("warn", "warn", [], False, new_problem=True) is True
    assert wd._should_email("warn", "warn", [], False, new_problem=False) is False


def test_run_cycle_dry_run_no_email(tmp_path, monkeypatch):
    fail = [ck.HealthResult("job:x", "fail", "looping",
                            remedy="restart:QCMonitorDaemon")]
    monkeypatch.setattr(wd._checks, "run_all", lambda *a, **k: fail)
    config = {"health": {"state_file": str(tmp_path / "h.json")}}
    summary = wd.run_cycle(config, dry_run=True, store=None)
    assert summary["overall"] == "fail"
    assert summary["emailed"] is False
    assert summary["actions"][0]["attempted"] is False
    assert "FAIL" in summary["report"]


def test_run_cycle_healthy_quiet(tmp_path, monkeypatch):
    good = [ck.HealthResult("all", "ok", "fine")]
    monkeypatch.setattr(wd._checks, "run_all", lambda *a, **k: good)
    sent = {}
    monkeypatch.setattr(wd, "_send", lambda *a, **k: sent.setdefault("x", True))
    config = {"health": {"state_file": str(tmp_path / "h.json")}}
    summary = wd.run_cycle(config, dry_run=False, store=None)
    assert summary["overall"] == "ok"
    assert summary["emailed"] is False
    assert "x" not in sent


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
