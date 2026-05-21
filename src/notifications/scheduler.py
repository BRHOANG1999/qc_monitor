"""Once-per-day digest scheduler.

The watcher's main loop calls ``DigestScheduler.tick()`` on every poll;
this class compares wall-clock time to the configured send hour and
fires the digest the first tick after that hour, persisting the date
of the most-recent send to a small JSON state file so a restart doesn't
double-send (or skip a send if we boot at 08:01).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date
from threading import Lock

from src.alerting.email_alert import EmailAlerter
from src.notifications.surgery_digest import send_today_digest

logger = logging.getLogger("qc_monitor.notifications.scheduler")


@dataclass
class DigestState:
    last_sent: str = ""        # ISO date of last successful send


class DigestScheduler:
    """Fire the surgery digest once per day at or after *send_hour*."""

    def __init__(self, config: dict, emailer: EmailAlerter):
        assert isinstance(config, dict), "config must be a dict"
        assert isinstance(emailer, EmailAlerter), "emailer required"
        self._config = config
        self._emailer = emailer
        cfg = config.get("surgery_digest", {}) or {}
        self._enabled = bool(cfg.get("enabled", False))
        hour = int(cfg.get("send_hour", 8))
        assert 0 <= hour <= 23, "send_hour must be 0..23"
        self._send_hour = hour
        self._state_path = self._resolve_state_path(cfg)
        self._lock = Lock()
        self._state = self._load_state()

    # ----- state I/O -------------------------------------------------- #

    @staticmethod
    def _resolve_state_path(cfg: dict) -> str:
        path = cfg.get("state_file") or "data/surgery_digest_state.json"
        if os.path.isabs(path):
            return path
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", ".."))
        return os.path.normpath(os.path.join(project_root, path))

    def _load_state(self) -> DigestState:
        try:
            with open(self._state_path) as f:
                raw = json.load(f)
            return DigestState(last_sent=str(raw.get("last_sent", "")))
        except (FileNotFoundError, json.JSONDecodeError):
            return DigestState()

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"last_sent": self._state.last_sent}, f)
        os.replace(tmp, self._state_path)

    # ----- main loop entry ------------------------------------------- #

    def tick(self, now: datetime | None = None) -> bool:
        """Called once per main-loop iteration. Sends if today is past
        the configured hour AND we haven't already sent today.

        Returns True iff a send fired this tick.
        """
        if not self._enabled:
            return False
        now = now or datetime.now()
        today_iso = now.date().isoformat()
        with self._lock:
            already_sent_today = (self._state.last_sent == today_iso)
            if already_sent_today:
                return False
            if now.hour < self._send_hour:
                return False

            logger.info("Surgery digest fire: %s %02d:%02d (window: hour>=%d)",
                        today_iso, now.hour, now.minute, self._send_hour)
            try:
                result = send_today_digest(
                    now.date(), self._config, self._emailer,
                )
            except Exception as e:
                logger.error("Digest send raised: %s", e, exc_info=True)
                return False

            # Mark today as sent regardless of partial-failure -- we
            # don't want to retry every 30 s if SMTP is briefly down.
            # The next attempt is tomorrow.
            self._state.last_sent = today_iso
            self._save_state()
            logger.info("Surgery digest result: %s", result)
            return bool(result.get("sent"))

    # ----- ops helpers ----------------------------------------------- #

    def force_now(self, now: datetime | None = None) -> bool:
        """Send the digest right now regardless of hour or last_sent
        state, then mark today as sent. Used by an admin "test" hook."""
        now = now or datetime.now()
        with self._lock:
            try:
                result = send_today_digest(
                    now.date(), self._config, self._emailer,
                )
            except Exception as e:
                logger.error("Forced digest raised: %s", e, exc_info=True)
                return False
            self._state.last_sent = now.date().isoformat()
            self._save_state()
            logger.info("Forced surgery digest result: %s", result)
            return bool(result.get("sent"))
