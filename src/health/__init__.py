"""24/7 upkeep watchdog: check the health of everything that must stay up
(Windows services, the scheduled email jobs, the processing pipeline, disk and log
health), attempt BOUNDED auto-fixes for the fixable ones, and email a report.

Runs as a STANDALONE process (``python -m src.health.watchdog``) on a Windows
Scheduled Task, independent of the daemon, so it can detect and restart the daemon
itself when that is what is down. See tools/install_health_watchdog.ps1.

Modules:
  checks.py      pure read-only health probes -> list[HealthResult]
  remediate.py   cooldown/backoff ledger + service restart (the only auto-fix)
  watchdog.py    orchestrate: check -> remediate (bounded) -> report -> email
"""

from __future__ import annotations

__all__ = ["checks", "remediate", "watchdog"]
