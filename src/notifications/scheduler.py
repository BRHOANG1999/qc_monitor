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
from datetime import datetime, date, timedelta
from threading import Lock, Thread

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store
from src.notifications.surgery_digest import send_today_digest
from src.notifications.eod_digest import send_eod_digest
from src.notifications.video_weekly import send_video_weekly
from src.notifications.evoked_weekly import send_evoked_weekly
from src.notifications.review_weekly import send_review_weekly
from src.notifications.coverage_monitor import send_coverage_digest
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
    review_weekly: str = ""
    coverage: str = ""
    stim_stability_weekly: str = ""
    stim_stability_daily: str = ""
    evoked_daily: str = ""
    evoked_resp_weekly: str = ""
    queue_stuck: bool = False
    # Count-based peri-ictal trigger (non-calendar): fire an animal's peri-ictal
    # email once each time it accumulates >= threshold NEW needs_scoring entries.
    # ``peri_stim_baseline_id`` is the review_event_log.id high-water mark set on
    # first watch so the pre-existing backlog never fires; ``peri_stim_marks`` is
    # ``{animal_id: last_consumed_event_id}`` advanced only when that animal fires.
    peri_stim_baseline_id: int = -1
    peri_stim_marks: dict = field(default_factory=dict)


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
        # Animals whose peri-ictal render+email thread is in flight, so a later
        # tick doesn't spawn a second render for the same animal.
        self._peri_stim_inflight: set = set()

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
                # review_weekly was dropped here on load, so its persisted
                # send-once date never round-tripped -> the weekly review
                # digest could re-send after a restart. Round-trip it.
                review_weekly=str(raw.get("review_weekly") or ""),
                coverage=str(raw.get("coverage") or ""),
                stim_stability_weekly=str(
                    raw.get("stim_stability_weekly") or ""),
                stim_stability_daily=str(
                    raw.get("stim_stability_daily") or ""),
                evoked_daily=str(raw.get("evoked_daily") or ""),
                evoked_resp_weekly=str(raw.get("evoked_resp_weekly") or ""),
                queue_stuck=bool(raw.get("queue_stuck", False)),
                peri_stim_baseline_id=int(
                    raw.get("peri_stim_baseline_id", -1)),
                peri_stim_marks=dict(raw.get("peri_stim_marks") or {}),
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

    def _claim(self, digest: str, now: datetime) -> bool:
        """Atomically claim a daily/weekly send for today. Returns True only if
        WE won the claim (nobody has sent this digest today). Restart-proof: the
        DB row survives restarts and backup reverts, so a relaunched daemon
        can't re-fire a digest the JSON state file happened to lose. Falls back
        to True when there's no store (legacy surgery-only construction) — the
        JSON ``_due_*`` gate is then the only guard, as before."""
        if self._store is None:
            return True
        try:
            return self._store.record_notification_sent(
                digest, now.date().isoformat())
        except Exception as e:  # noqa: BLE001 -- never crash the tick
            # Fail CLOSED: skip this tick rather than send. A transient WAL
            # lock clears and the next tick retries (the JSON _due_* gate is
            # still True since we didn't send), so nothing is permanently
            # lost -- whereas failing OPEN would defeat the atomic dedup and
            # re-introduce the multi-restart double-send this guard exists to
            # prevent.
            logger.error("notification claim failed for %s: %s "
                         "(skipping this tick; will retry)", digest, e)
            return False

    def _release(self, digest: str, now: datetime) -> None:
        """Undo a claim after the send RAISED, so a transient send exception
        doesn't leave the digest permanently suppressed by a dead claim."""
        if self._store is None:
            return
        try:
            self._store.release_notification_sent(
                digest, now.date().isoformat())
        except Exception as e:  # noqa: BLE001
            logger.error("notification release failed for %s: %s", digest, e)

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
            self._tick_stim_stability_weekly(now, fired)
            self._tick_stim_stability_daily(now, fired)
            self._tick_evoked_daily(now, fired)
            self._tick_evoked_resp_weekly(now, fired)
            self._tick_review_weekly(now, fired)
            self._tick_coverage(now, fired)
            self._tick_queue_watch(now, fired)
            self._tick_peri_stim(now, fired)
            if fired:
                self._save_state()
        return fired

    # ----- per-digest tick handlers ---------------------------------- #

    def _tick_surgery(self, now: datetime, fired: dict) -> None:
        if not self._surg_enabled:
            return
        if not self._due_daily(now, self._surg_hour, self._state.surgery):
            return
        if not self._claim("surgery", now):
            self._state.surgery = now.date().isoformat()   # heal JSON
            return
        logger.info("Surgery digest fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_today_digest(now.date(), self._config,
                                        self._emailer)
        except Exception as e:
            logger.error("Surgery digest send raised: %s", e,
                          exc_info=True)
            self._release("surgery", now)   # dead claim -> retry next tick
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
        if not self._claim("eod", now):
            self._state.eod = now.date().isoformat()
            return
        logger.info("EOD digest fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_eod_digest(now.date(), self._config,
                                       self._store, self._emailer)
        except Exception as e:
            logger.error("EOD digest send raised: %s", e, exc_info=True)
            self._release("eod", now)
            return
        self._state.eod = now.date().isoformat()
        fired["eod"] = bool(result.get("sent"))
        logger.info("EOD digest result: %s", result)

    def _tick_coverage(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("coverage_digest", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        hour = int(cfg.get("hour", 8))
        if not self._due_daily(now, hour, self._state.coverage):
            return
        if not self._claim("coverage", now):
            self._state.coverage = now.date().isoformat()
            return
        logger.info("Coverage digest fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_coverage_digest(now.date(), self._config,
                                          self._store, self._emailer)
        except Exception as e:
            logger.error("Coverage digest raised: %s", e, exc_info=True)
            self._release("coverage", now)
            return
        # Mark sent for the day whether or not it emailed -- a HEALTHY day
        # legitimately returns sent=False, and we don't want to re-run the
        # (heavy) scan every tick for the rest of the day.
        self._state.coverage = now.date().isoformat()
        fired["coverage"] = bool(result.get("sent"))
        logger.info("Coverage digest result: %s", result)

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
        if not self._claim("video_weekly", now):
            self._state.video_weekly = now.date().isoformat()
            return
        logger.info("Video weekly fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_video_weekly(now.date(), self._config,
                                         self._store, self._emailer)
        except Exception as e:
            logger.error("Video weekly send raised: %s", e, exc_info=True)
            self._release("video_weekly", now)
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
        if not self._claim("evoked_weekly", now):
            self._state.evoked_weekly = now.date().isoformat()
            return
        logger.info("Evoked weekly fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_evoked_weekly(now.date(), self._config,
                                          self._store, self._emailer)
        except Exception as e:
            logger.error("Evoked weekly send raised: %s", e, exc_info=True)
            self._release("evoked_weekly", now)
            return
        self._state.evoked_weekly = now.date().isoformat()
        fired["evoked_weekly"] = bool(result.get("sent"))
        logger.info("Evoked weekly result: %s", result)

    def _tick_stim_stability_weekly(self, now: datetime, fired: dict) -> None:
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("stim_stability_weekly", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        weekday = int(cfg.get("weekday", 6))   # 6=Sunday
        hour = int(cfg.get("hour", 8))
        if not self._due_weekly(now, weekday, hour,
                                 self._state.stim_stability_weekly):
            return
        if not self._claim("stim_stability_weekly", now):
            self._state.stim_stability_weekly = now.date().isoformat()
            return
        logger.info("Stim-stability weekly fire: %s %02d:%02d",
                    now.date().isoformat(), now.hour, now.minute)
        # Lazy import: keeps matplotlib/plotly/scipy out of the daemon startup
        # path when this (heavy, optional) digest is disabled.
        from src.notifications.stim_stability_weekly import (
            send_stim_stability_weekly)
        try:
            result = send_stim_stability_weekly(now.date(), self._config,
                                                self._store, self._emailer)
        except Exception as e:
            logger.error("Stim-stability weekly send raised: %s", e,
                         exc_info=True)
            self._release("stim_stability_weekly", now)
            return
        self._state.stim_stability_weekly = now.date().isoformat()
        fired["stim_stability_weekly"] = bool(result.get("sent"))
        logger.info("Stim-stability weekly result: %s", result)

    def _tick_stim_stability_daily(self, now: datetime, fired: dict) -> None:
        """Daily stim-stability digest: the SAME figures as the weekly one but
        sent every morning, covering the PREVIOUS full day's per-file figures +
        a running 7-day week ending yesterday (email-only)."""
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("stim_stability_daily", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        hour = int(cfg.get("hour", 8))
        if not self._due_daily(now, hour, self._state.stim_stability_daily):
            return
        if not self._claim("stim_stability_daily", now):
            self._state.stim_stability_daily = now.date().isoformat()
            return
        logger.info("Stim-stability daily fire: %s %02d:%02d",
                    now.date().isoformat(), now.hour, now.minute)
        from src.notifications.stim_stability_weekly import (
            send_stim_stability_weekly)
        report_date = now.date() - timedelta(days=1)      # cover YESTERDAY
        try:
            result = send_stim_stability_weekly(
                report_date, self._config, self._store, self._emailer,
                period="daily")
        except Exception as e:
            logger.error("Stim-stability daily send raised: %s", e,
                         exc_info=True)
            self._release("stim_stability_daily", now)
            return
        self._state.stim_stability_daily = now.date().isoformat()
        fired["stim_stability_daily"] = bool(result.get("sent"))
        logger.info("Stim-stability daily result: %s", result)

    def _tick_evoked_daily(self, now: datetime, fired: dict) -> None:
        """Daily evoked-response digest: per-feature percentile-band "kinds" of
        evoked response for the PREVIOUS day, per animal (email-only)."""
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("evoked_daily", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        hour = int(cfg.get("hour", 8))
        if not self._due_daily(now, hour, self._state.evoked_daily):
            return
        if not self._claim("evoked_daily", now):
            self._state.evoked_daily = now.date().isoformat()      # heal JSON
            return
        logger.info("Evoked digest daily fire: %s %02d:%02d",
                    now.date().isoformat(), now.hour, now.minute)
        from src.notifications.evoked_daily import send_evoked_daily
        try:
            result = send_evoked_daily(now, self._config, self._store,
                                       self._emailer)               # reports yesterday
        except Exception as e:
            logger.error("Evoked digest daily send raised: %s", e, exc_info=True)
            self._release("evoked_daily", now)
            return
        self._state.evoked_daily = now.date().isoformat()
        fired["evoked_daily"] = bool(result.get("sent"))
        logger.info("Evoked digest daily result: %s", result)

    def _tick_evoked_resp_weekly(self, now: datetime, fired: dict) -> None:
        """Weekly evoked-response digest: the SAME percentile-band "kinds" figures
        as the daily one but over a 7-day window (config: the nested
        ``notifications.evoked_daily.weekly`` block; render params inherit the
        daily block)."""
        daily = (self._config.get("notifications", {}) or {}) \
            .get("evoked_daily", {}) or {}
        cfg = daily.get("weekly", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        weekday = int(cfg.get("weekday", 6))              # 6 = Sunday
        hour = int(cfg.get("hour", 9))
        if not self._due_weekly(now, weekday, hour, self._state.evoked_resp_weekly):
            return
        if not self._claim("evoked_resp_weekly", now):
            self._state.evoked_resp_weekly = now.date().isoformat()
            return
        logger.info("Evoked digest weekly fire: %s %02d:%02d",
                    now.date().isoformat(), now.hour, now.minute)
        from src.notifications.evoked_daily import send_evoked_daily
        try:
            result = send_evoked_daily(now, self._config, self._store,
                                       self._emailer, window_days=7)
        except Exception as e:
            logger.error("Evoked digest weekly send raised: %s", e, exc_info=True)
            self._release("evoked_resp_weekly", now)
            return
        self._state.evoked_resp_weekly = now.date().isoformat()
        fired["evoked_resp_weekly"] = bool(result.get("sent"))
        logger.info("Evoked digest weekly result: %s", result)

    def _tick_review_weekly(self, now: datetime, fired: dict) -> None:
        cfg = ((self._config.get("review_queue", {}) or {})
                .get("weekly_digest", {}) or {})
        if not cfg.get("enabled", False) or self._store is None:
            return
        weekday = int(cfg.get("weekday", 4))   # 4=Friday
        hour = int(cfg.get("hour", 17))
        if not self._due_weekly(now, weekday, hour,
                                  self._state.review_weekly):
            return
        if not self._claim("review_weekly", now):
            self._state.review_weekly = now.date().isoformat()
            return
        logger.info("Review weekly fire: %s %02d:%02d",
                     now.date().isoformat(), now.hour, now.minute)
        try:
            result = send_review_weekly(now.date(), self._config,
                                          self._store, self._emailer)
        except Exception as e:
            logger.error("Review weekly send raised: %s", e,
                          exc_info=True)
            self._release("review_weekly", now)
            return
        self._state.review_weekly = now.date().isoformat()
        fired["review_weekly"] = bool(result.get("sent"))
        logger.info("Review weekly result: %s", result)

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

    def _tick_peri_stim(self, now: datetime, fired: dict) -> None:
        """Count-based, non-calendar trigger: when an animal accumulates >=
        ``threshold`` (default 5) NEW ``needs_scoring`` entries since its last
        peri-ictal email, render + email that animal's peri-ictal figures. Edge-
        triggered off ``review_event_log`` ids against a persisted per-animal
        high-water mark; the pre-existing backlog is baselined out on first watch
        so enabling it never blasts. Rendering is heavy, so the send runs on a
        background thread and the mark advances optimistically (a failed email
        won't re-fire the whole backlog)."""
        cfg = (self._config.get("notifications", {}) or {}) \
            .get("peri_stim", {}) or {}
        if not cfg.get("enabled", False) or self._store is None:
            return
        threshold = max(1, int(cfg.get("threshold", 5)))
        st = self._state
        if st.peri_stim_baseline_id < 0:                 # first watch: baseline
            try:
                st.peri_stim_baseline_id = self._store.max_review_event_id()
            except Exception as e:                        # noqa: BLE001
                logger.error("peri-stim baseline query failed: %s", e)
                return
            fired["peri_stim_baseline"] = True            # persist the baseline
            logger.info("peri-stim trigger armed at event id %d",
                        st.peri_stim_baseline_id)
            return
        floor = min([st.peri_stim_baseline_id, *st.peri_stim_marks.values()])
        try:
            rows = self._store.needs_scoring_quick_flags(since_id=floor)
        except Exception as e:                            # noqa: BLE001
            logger.error("peri-stim quick_flag query failed: %s", e)
            return
        per_animal: dict[str, list] = {}
        for r in rows:
            animal = r.get("animal_id")
            mark = st.peri_stim_marks.get(animal, st.peri_stim_baseline_id)
            if animal and int(r["id"]) > mark:
                per_animal.setdefault(animal, []).append(int(r["id"]))
        for animal, ids in per_animal.items():
            if len(ids) < threshold or animal in self._peri_stim_inflight:
                continue
            st.peri_stim_marks[animal] = max(ids)         # consume optimistically
            fired[f"peri_stim:{animal}"] = True
            self._spawn_peri_stim(animal, len(ids))

    def _spawn_peri_stim(self, animal: str, n_new: int) -> None:
        """Render + email one animal's peri-ictal figures on a daemon thread so
        the (multi-minute, raw-.mat) render never blocks the main tick loop."""
        self._peri_stim_inflight.add(animal)
        reason = (f"{n_new} recording(s) moved into needs-onset-scoring for "
                  f"{animal} — the peri-ictal stim-artifact summary follows.")

        def _work():
            try:
                from src.notifications.peri_stim_needs_scoring import (
                    send_peri_stim_artifact)
                res = send_peri_stim_artifact(animal, self._config,
                                              self._store, self._emailer,
                                              reason=reason)
                logger.info("peri-stim send for %s: %s", animal, res)
            except Exception as e:                        # noqa: BLE001
                logger.error("peri-stim send for %s raised: %s", animal, e,
                             exc_info=True)
            finally:
                self._peri_stim_inflight.discard(animal)

        Thread(target=_work, name=f"peri_stim_{animal}", daemon=True).start()

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
