"""Bounded auto-remediation: restart a service at most once per cooldown, back off on
repeats, and ESCALATE (stop trying, flag for the email) after a few failures inside a
window. This is what stops a real on-disk code bug from turning into a restart loop.

State persists in data/health_state.json so the cooldown survives across the Scheduled
Task's separate invocations.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime

_DEFAULT_STATE = os.path.join("data", "health_state.json")
_NSSM = os.path.join("tools", "nssm", "nssm.exe")


class RemediationLedger:
    """Cooldown + backoff + escalation, keyed by remedy string (e.g.
    ``restart:QCMonitorDaemon``). Persisted atomically."""

    def __init__(self, config: dict, *, path: str | None = None) -> None:
        hc = (config or {}).get("health", {}) or {}
        self.cooldown_sec = float(hc.get("cooldown_sec", 1800.0))
        self.max_attempts = int(hc.get("max_attempts", 3))
        self.window_sec = float(hc.get("escalate_window_sec", 6 * 3600.0))
        self.path = path or hc.get("state_file") or _DEFAULT_STATE
        self._data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            d = {}
        d.setdefault("remedies", {})
        return d

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def _entry(self, key: str, now: float) -> dict:
        e = self._data["remedies"].setdefault(
            key, {"last_attempt": 0.0, "attempts": 0, "last_success": 0.0,
                  "escalated": False, "window_start": now})
        if now - e.get("window_start", 0.0) > self.window_sec:      # window rolled over
            e.update(attempts=0, escalated=False, window_start=now)
        return e

    def allowed(self, key: str, *, now: float | None = None) -> tuple[bool, str]:
        """Whether *key* may be attempted now, and why not. Applies escalation cap,
        attempt cap, and exponential-backoff cooldown."""
        now = time.time() if now is None else now
        e = self._entry(key, now)
        if e.get("escalated"):
            return False, "escalated (repeated failures); alert only"
        if e["attempts"] >= self.max_attempts:
            e["escalated"] = True
            return False, f"reached {self.max_attempts} attempts in window; escalating"
        factor = min(2 ** e["attempts"], 8)
        wait = self.cooldown_sec * factor
        since = now - e.get("last_attempt", 0.0)
        if e.get("last_attempt") and since < wait:
            return False, f"in cooldown ({int(wait - since)}s of {int(wait)}s left)"
        return True, "allowed"

    def record(self, key: str, success: bool, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        e = self._entry(key, now)
        e["last_attempt"] = now
        e["attempts"] = int(e["attempts"]) + 1
        if success:
            e["last_success"] = now
            e.update(attempts=0, escalated=False, window_start=now)

    def escalated_keys(self) -> list[str]:
        return [k for k, e in self._data["remedies"].items() if e.get("escalated")]

    # ------- recovery-email bookkeeping -------
    def last_status(self) -> str:
        return str(self._data.get("last_status", "ok"))

    def set_status(self, status: str) -> None:
        self._data["last_status"] = status


def restart_service(name: str, *, nssm_path: str | None = None,
                    verify_wait_sec: float = 8.0) -> tuple[bool, str]:
    """Restart a Windows service and verify it is Running. Tries nssm first, then
    Restart-Service. Returns (ok, message). Access-denied (no admin) is reported, not
    raised, so the watchdog can escalate by email."""
    assert name, "service name required"
    nssm = nssm_path or _NSSM
    msgs = []
    for cmd in ([nssm, "restart", name] if os.path.exists(nssm) else None,
                ["powershell", "-NoProfile", "-Command",
                 f"Restart-Service -Name '{name}' -Force"]):
        if cmd is None:
            continue
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except (OSError, subprocess.SubprocessError) as e:
            msgs.append(f"{cmd[0]}: {e}")
            continue
        out = (r.stdout + r.stderr).strip().replace("\x00", "")
        if _is_running(name, verify_wait_sec):
            return True, f"{os.path.basename(str(cmd[0]))} restart OK; service Running"
        msgs.append(f"{os.path.basename(str(cmd[0]))}: rc={r.returncode} {out[:160]}")
    return False, "; ".join(msgs) or "restart failed"


def _is_running(name: str, wait_sec: float) -> bool:
    deadline = time.time() + max(0.0, wait_sec)
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-Service -Name '{name}' -ErrorAction SilentlyContinue).Status"],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        if (r.stdout or "").strip() == "Running":
            return True
        time.sleep(1.0)
    return False


_DAEMON = "QCMonitorDaemon"


def _release_claim(digest: str, store) -> str:
    """Clear today's dedup claim for *digest* so a resend can proceed (the fix for a
    stale claim left by a crash mid-send). Best-effort; returns a note."""
    if store is None:
        return "no store handle (cannot release claim)"
    from datetime import date
    try:
        store.release_notification_sent(digest, date.today().isoformat())
        return f"released today's claim for {digest}"
    except Exception as e:                               # noqa: BLE001
        return f"claim release failed: {e}"


def _do_remedy(key: str, store) -> tuple[bool, str]:
    """Execute one remedy. ``restart:<svc>`` restarts a service. ``fixjob:<digest>``
    releases the job's stale claim THEN restarts the daemon (reload code + un-suppress
    the resend), in that order so the fixed code re-fires against a clear claim."""
    if key.startswith("restart:"):
        return restart_service(key.split(":", 1)[1])
    if key.startswith("fixjob:"):
        digest = key.split(":", 1)[1]
        ok, note = restart_service(_DAEMON)
        rel = _release_claim(digest, store)
        return ok, f"{note}; {rel}"
    return False, "no auto-fix for this remedy"


def apply_remedies(results, ledger: RemediationLedger, *, dry_run: bool,
                   store=None, now: float | None = None) -> list[dict]:
    """Apply the bounded auto-fix for each distinct remedy named by a failing result.
    Returns a list of ``{remedy, attempted, ok, note}``. Auto-fixable remedies are
    ``restart:<service>`` and ``fixjob:<digest>`` (release stale claim + restart
    daemon); anything else is alert-only."""
    remedies = sorted({r.remedy for r in results
                       if r.status == "fail" and r.remedy})
    actions: list[dict] = []
    for key in remedies:
        allowed, why = ledger.allowed(key, now=now)
        if not allowed:
            actions.append({"remedy": key, "attempted": False, "ok": False,
                            "note": why})
            continue
        if not (key.startswith("restart:") or key.startswith("fixjob:")):
            actions.append({"remedy": key, "attempted": False, "ok": False,
                            "note": "no auto-fix for this remedy"})
            continue
        if dry_run:
            actions.append({"remedy": key, "attempted": False, "ok": False,
                            "note": "dry-run (would auto-fix)"})
            continue
        ok, note = _do_remedy(key, store)
        ledger.record(key, ok, now=now)
        actions.append({"remedy": key, "attempted": True, "ok": ok, "note": note})
    return actions
