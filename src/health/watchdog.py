"""Watchdog cycle: run every probe, apply bounded auto-fixes, email a report.

Designed to run from a Windows Scheduled Task every ~10 min, independent of the
daemon. Emails only when something is wrong, when a fix was attempted, or on recovery,
so a healthy system stays quiet.

  python -m src.health.watchdog            # one cycle: check, fix, email
  python -m src.health.watchdog --dry-run  # check + report only (no restart, no email)
  python -m src.health.watchdog --once     # explicit single cycle (default)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from src.health import checks as _checks
from src.health import remediate as _rem

_ICON = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}


def _overall(results) -> str:
    if any(r.status == "fail" for r in results):
        return "fail"
    if any(r.status == "warn" for r in results):
        return "warn"
    return "ok"


def _open_store(config: dict):
    try:
        from src.db.store import Store
        return Store((config.get("database", {}) or {}).get("path"))
    except Exception:                                    # noqa: BLE001
        return None


def build_report(results, actions, overall: str, escalated: list[str]) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [f"QC Monitor health watchdog - {overall.upper()} - {now} UTC", ""]
    for r in results:
        lines.append(f"  [{_ICON.get(r.status, '?')}] {r.name}: {r.detail}")
    if actions:
        lines += ["", "Auto-fix actions:"]
        for a in actions:
            tag = "attempted" if a["attempted"] else "skipped"
            ok = "ok" if a["ok"] else "no"
            lines.append(f"  - {a['remedy']}: {tag} (success={ok}) - {a['note']}")
    if escalated:
        lines += ["", "ESCALATED (auto-fix gave up, needs a human):"]
        lines += [f"  - {k}" for k in escalated]
    return "\n".join(lines)


def _severity(results, escalated: list[str]) -> str:
    if escalated or any(r.status == "fail" and r.severity == "critical"
                        for r in results):
        return "critical"
    return "warning" if any(r.status == "fail" for r in results) else "info"


def _should_email(overall: str, prev: str, actions: list, force: bool) -> bool:
    if force:
        return True
    if overall == "fail":
        return True
    if any(a["attempted"] for a in actions):
        return True
    return prev == "fail" and overall != "fail"          # recovery notice


def run_cycle(config: dict, *, dry_run: bool = False, force_email: bool = False,
              emailer=None, store=None, log=None) -> dict:
    """One full check -> remediate -> report -> email cycle. Returns a summary dict.
    ``dry_run`` skips restarts and email."""
    store = store if store is not None else _open_store(config)
    results = _checks.run_all(config, store)
    ledger = _rem.RemediationLedger(config)
    actions = _rem.apply_remedies(results, ledger, dry_run=dry_run)
    escalated = ledger.escalated_keys()
    overall = _overall(results)
    report = build_report(results, actions, overall, escalated)
    if log is not None:
        log(report)
    prev = ledger.last_status()
    sent = False
    if not dry_run:
        if _should_email(overall, prev, actions, force_email):
            sent = _send(config, report, _severity(results, escalated), emailer)
        ledger.set_status(overall)
        ledger.save()
    return {"overall": overall, "n_fail": sum(1 for r in results if r.status == "fail"),
            "actions": actions, "escalated": escalated, "emailed": sent,
            "report": report}


def _send(config: dict, report: str, severity: str, emailer) -> bool:
    hc = (config.get("health", {}) or {})
    recipients = hc.get("recipients")            # None -> emailer default list
    try:
        if emailer is None:
            from src.alerting.email_alert import EmailAlerter
            emailer = EmailAlerter(config)
        return bool(emailer.send(
            "health watchdog", report, severity=severity, recipients=recipients))
    except Exception:                                    # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="check + print report; no restart, no email")
    ap.add_argument("--once", action="store_true", help="single cycle (default)")
    ap.add_argument("--force-email", action="store_true",
                    help="email the report even if healthy")
    a = ap.parse_args(argv)
    from src.dashboard.data_helpers import load_config
    config = load_config()
    summary = run_cycle(config, dry_run=a.dry_run, force_email=a.force_email,
                        log=lambda m: print(m, flush=True))
    print("---", flush=True)
    print(f"overall={summary['overall']} fails={summary['n_fail']} "
          f"emailed={summary['emailed']} escalated={summary['escalated']}", flush=True)
    return 2 if summary["overall"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
