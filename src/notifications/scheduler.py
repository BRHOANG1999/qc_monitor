"""Notification scheduler.

Owns the cadence + persistence for every recurring email the
QC Monitor sends. One ``tick()`` per main-loop iteration fires
whatever is due, persists state, and is otherwise a cheap no-op.

Schedules supported:

* ``surgery_digest`` -- daily at/after ``send_hour``
  (existing config key: ``surgery_digest``)
* ``notifications.eod_digest`` -- daily at/after ``hour``
* ``notifications.video_weekly`` -- weekly on ``weekday``
  (0 = Monday) at/after ``hour``
* ``notifications.evoked_weekly`` -- weekly, same shape
* ``notifications.queue_watch`` -- every tick, edge-triggered;
  one "stuck" email per stuck episode, one "clear" on recovery.

State lives in ``data/notification_state.json`` (overridable via
``notifications.state_file``). A legacy
``data/surgery_digest_state.json`` is migrated on first load so
upgrades don't double-send the morning surgery digest.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from threading import Lock

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store
from src.notifications.surgery_digest import send_today_digest
from src.notifications.eod_digest import send_eod_digest
from src.notifications.video_weekly import send_video_weekly
from src.notifications.evoked_weekly import send_evoked_weekly
from src.notifications import queue_watch

logger = logging.getLogger("qc_monitor.notifications.scheduler")


@dataclass
class NotificationState:
    """Per-digest 'last successful send' dates + the queue-watch flag.

    Dates are ISO ``YYYY-MM-DD``; absent / empty means 'never sent'.
    ``queue_stuck`` is True only while we're inside a stuck episode
    so we send exactly one alert per stall and one all-clear.
    """
    surgery: str = ""
    eod: str = ""
    video_weekly: str = ""
    evoked_weekly: str = ""
    queue_stuck: bool = False


def _project_path(relative: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.normpath(os.path.join(project_root, relative))


class DigestScheduler:
    """Schedule + dispatch every recurring email the monitor sends.

    Backwards-compatible: ``DigestScheduler(config, emailer)`` still
    works (surgery digest only). Passing *store* enables the new
    digests + queue watch.
    """

    def __init__(self, config: dict, emailer: EmailAlerter,
                  store: Store | None = None):
        assert isinstance(config, dict), "config must be a dict"
        assert isinstance(emailer, EmailAlerter), "emailer required"
        self._config = config
        self._emailer = emailer
        self._store = store
        self._lock = Lock()

        # Surgery digest (existing, top-level key).
        surg_cfg = config.get("surgery_digest", {}) or {}
        self._surg_enabled = bool(surg_cfg.get("enabled", False))
        hour = int(surg_cfg.get("send_hour", 8))
        assert 0 <= hour <= 23, "send_hour must be 0..23"
        self._surg_hour = hour

        notif_cfg = (config.get("notifications", {}) or {})
        self._state_path = self._resolve_state_path(notif_cfg, surg_cfg)
        self._state = self._load_state()

    # ----- state I/O -------------------------------------------------- #

    @staticmethod
    def _resolve_state_path(notif_cfg: dict, surg_cfg: dict) -> str:
        # Prefer the unified path; fall back to the legacy surgery
        # path so first-boot doesn't blow away a known-good record.
        unified = notif_cfg.get("state_file") or \
            "data/notification_state.json"
        path = unified if os.path.isabs(unified) else _project_path(unified)
        return path

    def _legacy_surgery_state_path(self) -> str:
        cfg = self._config.get("surgery_digest", {}) or {}
        p = cfg.get("state_file") or "data/surgery_digest_state.json"
        return p if os.path.isabs(p) else _project_path(p)

    def _load_state(self) -> NotificationState:
        # Try unified first.
        try:
            with open(self._state_path) as f:
                raw = json.load(f)
            return NotificationState(
                surgery=str(raw.get("surgery") or raw.get("last_sent") or ""),
                eod=str(raw.get("eod") or ""),
                video_weekly=str(raw.get("video_weekly") or ""),
                evoked_weekly=str(raw.get("evoked_weekly") or ""),
                queue_stuck=bool(raw.get("queue_stuck", False)),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        # Migrate legacy surgery state if present.
        try:
            with open(self._legacy_surgery_state_path()) as f:
                raw = json.load(f)
            return NotificationState(
                surgery=str(raw.get("last_sent", "")),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return NotificationState()

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self._state), f)
        os.replace(tmp, self._state_path)

    # ----- per-digest predicates ------------------------------------- #

    @staticmethod
    def _due_daily(now: datetime, hour: int, last_sent_iso: str) -> bool:
        if now.hour < hour:
            return False
        return last_sent_iso != now.date().isoformat()

    @staticmethod
    def _due_weekly(now: datetime, weekday: int, hour: int,
                     last_sent_iso: str) -> bool:
        if now.weekday() != weekday:
            return False
        if now.hour < hour:
            return False
        return last_sent_iso != now.date().isoformat()

    # ----- main loop entry ------------------------------------------- #

    def tick(self, now: datetime | None = None) -> dict:
        """Send any digests that are due this tick. Returns a dict of
        ``{digest_name: True}`` for every fire."""
        now = now or datetime.now()
        fired: dict[str, bool] = {}
        with self._lock:
            self._tick_surgery(now, fired)
            self._tick_eod(now, fired)
            self._tick_video_weekly(now, fired)
            self._tick_evoked_weekly(now, fired)
            self._tick_queue_watch(now, fired)
            if fired:
                self._save_state()
        return fired

    # ----- per-digest tick handlers ---------------------------------- #

    def _tick_surgery(self, now: datetime, fired: dict) -> None:
        if not self._surg_enabled:
            return
        if not self._due_daily(now, self._surg_hour, self._state.surgery):
            return
        logger.info("Surgery digest fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_today_digest(now.date(), self._config,
                                        self._emailer)
        except Exception as e:
            logger.error("Surgery digest send raised: %s", e,
                          exc_info=True)
            return
        # Mark as sent regardless of partial-failure so SMTP downtime
        # doesn't retry every poll. Next attempt is tomorrow.
        self._state.surgery = now.date().isoformat()
        fired["surgery"] = bool(result.get("sent"))
        logger.info("Surgery digest result: %s", result)

    def _tick_eod(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("eod_digest", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        hour = int(cfg.get("hour", 18))
        if not self._due_daily(now, hour, self._state.eod):
            return
        logger.info("EOD digest fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_eod_digest(now.date(), self._config,
                                       self._store, self._emailer)
        except Exception as e:
            logger.error("EOD digest send raised: %s", e, exc_info=True)
            return
        self._state.eod = now.date().isoformat()
        fired["eod"] = bool(result.get("sent"))
        logger.info("EOD digest result: %s", result)

    def _tick_video_weekly(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("video_weekly", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        weekday = int(cfg.get("weekday", 0))   # 0=Mon
        hour = int(cfg.get("hour", 8))
        if not self._due_weekly(now, weekday, hour,
                                  self._state.video_weekly):
            return
        logger.info("Video weekly fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_video_weekly(now.date(), self._config,
                                         self._store, self._emailer)
        except Exception as e:
            logger.error("Video weekly send raised: %s", e, exc_info=True)
            return
        self._state.video_weekly = now.date().isoformat()
        fired["video_weekly"] = bool(result.get("sent"))
        logger.info("Video weekly result: %s", result)

    def _tick_evoked_weekly(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("evoked_weekly", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        weekday = int(cfg.get("weekday", 6))   # 6=Sunday
        hour = int(cfg.get("hour", 8))
        if not self._due_weekly(now, weekday, hour,
                                  self._state.evoked_weekly):
            return
        logger.info("Evoked weekly fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_evoked_weekly(now.date(), self._config,
                                          self._store, self._emailer)
        except Exception as e:
            logger.error("Evoked weekly send raised: %s", e, exc_info=True)
            return
        self._state.evoked_weekly = now.date().isoformat()
        fired["evoked_weekly"] = bool(result.get("sent"))
        logger.info("Evoked weekly result: %s", result)

    def _tick_queue_watch(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("queue_watch", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        try:
            result = queue_watch.evaluate(self._store, self._config,
                                            now=now)
        except Exception as e:
            logger.error("Queue watch evaluate raised: %s", e,
                          exc_info=True)
            return
        is_stuck = bool(result.get("stuck"))
        was_stuck = bool(self._state.queue_stuck)
        if is_stuck and not was_stuck:
            logger.warning("Queue stuck: %s", result)
            try:
                queue_watch.send_stuck_alert(result, self._config,
                                              self._emailer)
                fired["queue_stuck"] = True
            except Exception as e:
                logger.error("Queue stuck alert raised: %s", e,
                              exc_info=True)
            self._state.queue_stuck = True
        elif was_stuck and not is_stuck:
            logger.info("Queue recovered: %s", result)
            try:
                queue_watch.send_clear(result, self._config,
                                        self._emailer)
                fired["queue_clear"] = True
            except Exception as e:
                logger.error("Queue clear alert raised: %s", e,
                              exc_info=True)
            self._state.queue_stuck = False

    # ----- ops helpers ----------------------------------------------- #

    def force_now(self, now: datetime | None = None) -> bool:
        """Force-send the surgery digest right now. Kept for backwards
        compatibility with the admin "test" hook."""
        now = now or datetime.now()
        with self._lock:
            try:
                result = send_today_digest(now.date(), self._config,
                                             self._emailer)
            except Exception as e:
                logger.error("Forced surgery digest raised: %s", e,
                              exc_info=True)
                return False
            self._state.surgery = now.date().isoformat()
            self._save_state()
            logger.info("Forced surgery digest result: %s", result)
            return bool(result.get("sent"))
