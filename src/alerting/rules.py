"""Alert rule engine with rate limiting."""

import logging
import os
from datetime import datetime
from html import escape

from src.db.store import Store
from src.alerting.email_alert import EmailAlerter

logger = logging.getLogger("qc_monitor.alerting.rules")


def _median(vals: list) -> float | None:
    s = sorted(v for v in vals if v is not None)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _drift_stat(values: list, window: int, min_history: int) -> dict | None:
    """Rolling-median drift of the newest value vs the prior *window*.
    Returns ``{latest, baseline, drift_pct}`` or None if too little history
    / a degenerate baseline. Shared conceptually with the Overview card."""
    vals = [v for v in values if v is not None]
    if len(vals) < max(2, min_history):
        return None
    latest = vals[-1]
    baseline = _median(vals[max(0, len(vals) - 1 - window):-1])
    if baseline is None or baseline == 0:
        return None
    return {"latest": latest, "baseline": baseline,
            "drift_pct": (latest - baseline) / baseline * 100.0}


class AlertRuleEngine:
    def __init__(self, store: Store, emailer: EmailAlerter, config: dict):
        self.store = store
        self.emailer = emailer
        rules_cfg = config.get("alerting", {}).get("rules", {})
        self.rate_limit_minutes = rules_cfg.get("rate_limit_per_type_minutes", 60)
        self.network_down_minutes = rules_cfg.get("network_down_minutes", 5)
        self.no_data_hours = rules_cfg.get("no_data_hours", 2)
        # Exclude files that exhausted their auto-retries from the MATLAB-failure
        # alert so a permanently-too-heavy recording stops re-alerting.
        self.matlab_max_attempts = int(config.get("matlab_max_attempts", 4))
        # Transfer-impedance drift thresholds (per channel, per phase).
        self.impedance_shift_pct = float(
            rules_cfg.get("impedance_shift_pct", 40))
        self.impedance_shift_pct_critical = float(
            rules_cfg.get("impedance_shift_pct_critical", 75))
        self.impedance_baseline_window = int(
            rules_cfg.get("impedance_baseline_window", 10))
        self.impedance_min_history = int(
            rules_cfg.get("impedance_min_history", 4))
        # Slow-phase steady-state impedance (Z_ss) drift: the same rolling-median
        # drift test as access resistance, but on the settled slow-phase plateau
        # (a stim-consistency signal). Defaults to the access-resistance
        # thresholds above when unset; shares the baseline window + min history.
        self.impedance_ss_shift_pct = float(
            rules_cfg.get("impedance_ss_shift_pct", self.impedance_shift_pct))
        self.impedance_ss_shift_pct_critical = float(
            rules_cfg.get("impedance_ss_shift_pct_critical",
                          self.impedance_shift_pct_critical))
        # Suppress an impedance-drift alert when the newest measurement in the
        # (recent-protocol) series is older than this many days -- a channel with
        # no recent measurement has no basis to judge drift and would re-fire a
        # stale value forever.
        self.impedance_staleness_days = float(
            rules_cfg.get("impedance_staleness_days", 3))
        # Z_ss saturation guard: at high stim charge the amplifier saturates and
        # the slow-phase plateau collapses (Z_ss << Rₐ). Treat a Z_ss row as
        # invalid (drop it) when slow_ss_kohm < this fraction of access_r_kohm, so
        # a saturating therapeutic protocol can't false-alarm the Z_ss drift.
        self.impedance_ss_saturation_ratio = float(
            rules_cfg.get("impedance_ss_saturation_ratio", 0.3))
        # If more than this fraction of a channel's Z_ss measurements are
        # saturated, the whole channel is on a saturating (high-charge) protocol
        # -> Z_ss is unreliable noise around ~0 and its rolling-median drift is
        # meaningless, so skip the channel entirely (don't just drop rows).
        self.impedance_ss_max_saturated_frac = float(
            rules_cfg.get("impedance_ss_max_saturated_frac", 0.3))
        # Attach the Rₐ / Z_ss trend figure to the drift-alert emails so a
        # recipient can see the trend instead of guessing from a bare percentage.
        self.impedance_alert_figures = bool(
            rules_cfg.get("impedance_alert_figures", True))
        # Current-delivery fidelity: fire when a channel's rolling-median
        # two-edge fidelity drops below this (negative) percent.
        self.current_sag_pct = float(rules_cfg.get("current_sag_pct", -15))
        self.impedance_exclude = (
            (config.get("overview", {}) or {}).get("exclude_animals")
            or ["Randles"])

        self._network_down_since: datetime | None = None

    def check_network(self, accessible: bool):
        now = datetime.now()
        if not accessible:
            if self._network_down_since is None:
                self._network_down_since = now
            elif (now - self._network_down_since).total_seconds() > self.network_down_minutes * 60:
                self._fire_alert(
                    "network_down", "critical",
                    f"Network share unreachable for >{self.network_down_minutes} minutes"
                )
        else:
            self._network_down_since = None

    def check_processing_stalled(self, scan_stats: dict):
        """Fire when a recording NEWER (by its embedded recording time) than
        anything we've processed has sat on the share, unprocessed, for longer
        than ``no_data_hours``. Keyed on the recording's filename datetime, NOT
        file mtime -- the daemon watches a daily backup that re-copies old
        files with fresh mtimes, so mtime would false-alarm every backup.
        Catches a daemon hang, a queue stall, AND the silent-skip case (a
        recorder rename where nothing ever reaches the queue, so pending
        stays 0)."""
        share_dt = (scan_stats or {}).get("newest_chunk_dt") or ""
        if not share_dt:
            return                       # no recordings on the share to judge
        proc_dt = self.store.newest_processed_chunk_datetime() or ""
        if share_dt <= proc_dt:
            return                       # nothing on the share newer than
            #                              what we've processed (a backup
            #                              re-copy of old files doesn't count)
        try:
            share_when = datetime.strptime(share_dt, "%Y_%m_%d__%H_%M_%S")
        except (ValueError, TypeError):
            return
        if (datetime.now() - share_when).total_seconds() \
                <= self.no_data_hours * 3600:
            return                       # fresh -- give it time to process
        newest_str = share_when.strftime("%Y-%m-%d %H:%M")
        try:
            last_str = datetime.strptime(
                proc_dt, "%Y_%m_%d__%H_%M_%S").strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            last_str = proc_dt or "never"
        n_skip = (scan_stats or {}).get("n_skipped_recordinglike") or 0
        skip_note = (f" {n_skip} file(s) skipped the filename pattern."
                     if n_skip else "")
        self._fire_alert(
            "processing_stalled", "critical",
            f"Processing may be stalled: a recording from {newest_str} on the "
            f"share is newer than the last processed recording ({last_str}) "
            f"and has been unprocessed for >{self.no_data_hours}h.{skip_note}")

    def check_unrecognized_recordings(self, scan_stats: dict):
        """Fire when the scan saw .mat files that look like recordings (have a
        recorder timestamp) but didn't match the filename pattern -- the direct
        alarm for a format change like the ___/__ one that silently skipped
        ingest for two days."""
        n = (scan_stats or {}).get("n_skipped_recordinglike") or 0
        if n <= 0:
            return
        samples = ", ".join((scan_stats or {}).get("skipped_samples") or [])
        self._fire_alert(
            "unrecognized_recordings", "warning",
            f"{n} .mat file(s) look like recordings but don't match the "
            f"filename pattern and are NOT being processed (possible recorder "
            f"format change). Examples: {samples}")

    def check_matlab_failures(self, lookback_hours: int = 1):
        """Fire (rate-limited) when files have failed the MATLAB step
        recently, so a per-file crash isn't silent -- it also shows on the
        Overview 'failed processing' card and the Alerts tab."""
        try:
            failed = self.store.matlab_failed_files(
                hours=lookback_hours, max_attempts=self.matlab_max_attempts)
        except Exception:  # noqa: BLE001 -- alerting must never crash the loop
            return
        if not failed:
            return
        newest = os.path.basename(failed[0].get("file_path") or "")
        self._fire_alert(
            "matlab_error", "warning",
            f"{len(failed)} recording(s) failed MATLAB processing in the last "
            f"{lookback_hours}h (most recent: {newest}). They produce no "
            f"evoked data -- see the Overview 'failed processing' card.")

    def _series_fresh(self, rows) -> bool:
        """True if the newest recording in this per-channel series is within
        ``impedance_staleness_days`` of now. A frozen series (the protocol
        stopped / changed) has no recent basis to judge drift and would otherwise
        re-fire a stale value forever. Rows are ordered oldest->newest, so the
        last one is the newest measurement."""
        if not rows:
            return False
        cd = (rows[-1] or {}).get("chunk_datetime") or ""
        try:
            when = datetime.strptime(cd, "%Y_%m_%d__%H_%M_%S")
        except (ValueError, TypeError):
            return True                       # unparseable -> fail open
        return (datetime.now() - when).total_seconds() \
            <= self.impedance_staleness_days * 86400

    def _valid_zss_values(self, rows) -> list:
        """``slow_ss_kohm`` per row with SATURATED rows dropped (see
        ``_zss_valid_and_saturation``). Order preserved so the rolling-median
        drift is over the VALID measurements only."""
        return self._zss_valid_and_saturation(rows)[0]

    def _zss_valid_and_saturation(self, rows):
        """``(valid_values_in_order, saturated_fraction)``. At high stim charge
        the amp saturates and the slow-phase plateau collapses to ``Z_ss << Rₐ``;
        a row is saturated when ``slow_ss_kohm`` falls below
        ``impedance_ss_saturation_ratio`` of ``access_r_kohm``. The fraction is
        over rows that HAVE a Z_ss, so a channel dominated by saturated rows can
        be skipped wholesale rather than drifting on the few noisy survivors."""
        considered = saturated = 0
        valid = []
        for r in rows:
            z = r.get("slow_ss_kohm")
            ra = r.get("access_r_kohm")
            if z is None:
                continue
            considered += 1
            if ra and ra > 0 and z < self.impedance_ss_saturation_ratio * ra:
                saturated += 1
            else:
                valid.append(z)
        frac = (saturated / considered) if considered else 0.0
        return valid, frac

    @staticmethod
    def _drift_signature(flags, severity: str) -> str:
        """Stable per-condition key for daily dedup: the sorted set of drifting
        (animal, channel) + severity. The SAME condition maps to the same key
        (one email/day); a new channel or an escalation to critical is a new key
        and fires again."""
        chans = "|".join(sorted(f"{f[0]}:{f[1]}" for f in flags))
        return f"{chans}#{severity}"

    def check_impedance_shift(self):
        """Fire (rate-limited) when a channel's ACCESS RESISTANCE (Rₐ, from
        the ohmic step of the stim pulse) drifts past ``impedance_shift_pct``
        from its rolling-median baseline. One aggregated alert lists every
        drifting (animal, channel); severity escalates to critical past
        ``impedance_shift_pct_critical``. Silent until a channel has
        ``impedance_min_history`` measurements."""
        try:
            series = self.store.impedance_series_by_channel(
                exclude=self.impedance_exclude)
            active = self.store.active_impedance_channel_keys()
        except Exception:  # noqa: BLE001 -- alerting must not crash the loop
            return
        # Only alert on the animals currently on the rig (most-recent session).
        if active:
            series = {k: v for k, v in series.items() if k in active}
        flags = self._impedance_flags(series)
        if not flags:
            return
        crit = any(abs(st["drift_pct"]) >= self.impedance_shift_pct_critical
                   for *_r, st in flags)
        severity = "critical" if crit else "warning"
        lines = [f"{a} {c} {st['latest']:.3f}kΩ vs "
                 f"{st['baseline']:.3f} ({st['drift_pct']:+.0f}%)"
                 for a, c, st in flags[:8]]
        more = f" (+{len(flags) - 8} more)" if len(flags) > 8 else ""
        self._fire_alert(
            "impedance_shift", severity,
            f"{len(flags)} channel(s) drifted "
            f">{self.impedance_shift_pct:.0f}% from baseline access "
            f"resistance: " + "; ".join(lines) + more,
            resend_key=self._drift_signature(flags, severity),
            figure_builder=(
                (lambda: self._drift_figures(
                    series, flags, value_key="access_r_kohm", ylabel="Rₐ (kΩ)",
                    series_name="Rₐ", metric_label="access resistance (Rₐ)",
                    pct=self.impedance_shift_pct))
                if self.impedance_alert_figures else None))

    def _impedance_flags(self, series: dict) -> list:
        """Every (animal, channel, drift_stat) whose access resistance is past
        the warn pct. Stale (frozen) series are skipped."""
        flags = []
        for (animal, ch), rows in (series or {}).items():
            if not self._series_fresh(rows):
                continue
            st = _drift_stat([r.get("access_r_kohm") for r in rows],
                             self.impedance_baseline_window,
                             self.impedance_min_history)
            if st and abs(st["drift_pct"]) >= self.impedance_shift_pct:
                flags.append((animal, ch, st))
        flags.sort(key=lambda f: -abs(f[2]["drift_pct"]))
        return flags

    def check_impedance_ss_shift(self):
        """Fire (rate-limited) when a channel's SLOW-PHASE STEADY-STATE
        IMPEDANCE (Z_ss -- the settled plateau of the long slow recharge phase,
        measured after the transients have died out) drifts past
        ``impedance_ss_shift_pct`` from its rolling-median baseline. A
        stim-consistency signal complementary to access resistance (Rₐ): both
        ride the same per-channel series, but Z_ss tracks the steady-state
        plateau while Rₐ tracks the instantaneous ohmic step, so a channel can
        drift in one without the other. One aggregated alert lists every
        drifting (animal, channel); severity escalates to critical past
        ``impedance_ss_shift_pct_critical``. Scoped to the channels currently on
        the rig; silent until a channel has ``impedance_min_history``
        measurements."""
        try:
            series = self.store.impedance_series_by_channel(
                exclude=self.impedance_exclude)
            active = self.store.active_impedance_channel_keys()
        except Exception:  # noqa: BLE001 -- alerting must not crash the loop
            return
        # Only alert on the animals currently on the rig (most-recent session).
        if active:
            series = {k: v for k, v in series.items() if k in active}
        flags = self._impedance_ss_flags(series)
        if not flags:
            return
        crit = any(abs(st["drift_pct"]) >= self.impedance_ss_shift_pct_critical
                   for *_r, st in flags)
        severity = "critical" if crit else "warning"
        lines = [f"{a} {c} {st['latest']:.3f}kΩ vs "
                 f"{st['baseline']:.3f} ({st['drift_pct']:+.0f}%)"
                 for a, c, st in flags[:8]]
        more = f" (+{len(flags) - 8} more)" if len(flags) > 8 else ""
        self._fire_alert(
            "impedance_ss_shift", severity,
            f"{len(flags)} channel(s) drifted "
            f">{self.impedance_ss_shift_pct:.0f}% from baseline slow-phase "
            f"steady-state impedance (Z_ss): " + "; ".join(lines) + more,
            resend_key=self._drift_signature(flags, severity),
            figure_builder=(
                (lambda: self._drift_figures(
                    series, flags, value_key="slow_ss_kohm", ylabel="Z_ss (kΩ)",
                    series_name="Z_ss",
                    metric_label="slow-phase steady-state impedance (Z_ss)",
                    pct=self.impedance_ss_shift_pct))
                if self.impedance_alert_figures else None))

    def _impedance_ss_flags(self, series: dict) -> list:
        """Every (animal, channel, drift_stat) whose slow-phase steady-state
        impedance is past the warn pct. Stale series are skipped; saturated Z_ss
        rows (plateau collapsed at high charge) are dropped; and a channel whose
        Z_ss is MOSTLY saturated (a high-charge protocol) is skipped entirely --
        its Z_ss is unreliable noise around ~0, so a rolling-median drift on the
        few rows that bounce above the saturation floor is meaningless (the
        BCH111SR-at-10nC false alarm)."""
        flags = []
        for (animal, ch), rows in (series or {}).items():
            if not self._series_fresh(rows):
                continue
            valid, sat_frac = self._zss_valid_and_saturation(rows)
            if sat_frac > self.impedance_ss_max_saturated_frac:
                continue                 # saturating protocol -> Z_ss unreliable
            st = _drift_stat(valid, self.impedance_baseline_window,
                             self.impedance_min_history)
            if st and abs(st["drift_pct"]) >= self.impedance_ss_shift_pct:
                flags.append((animal, ch, st))
        flags.sort(key=lambda f: -abs(f[2]["drift_pct"]))
        return flags

    def check_current_sag(self):
        """Fire (rate-limited) when a channel's CURRENT-DELIVERY FIDELITY (the
        two-edge ratio) sags below ``current_sag_pct`` (negative). The two
        ohmic edges of the biphasic pulse each back out the same series R using
        their exact commanded ΔI, so this is impedance-invariant: a persistent
        negative means the reversal (high-ΔI) edge is sagging => the commanded
        current is not being fully delivered (compliance) — the earliest
        warning, before the impedance itself drifts. Rolling-median so a single
        noisy edge doesn't trip it. Scoped to the channels currently on the
        rig; silent until a channel has ``impedance_min_history`` points."""
        try:
            series = self.store.impedance_series_by_channel(
                exclude=self.impedance_exclude)
            active = self.store.active_impedance_channel_keys()
        except Exception:  # noqa: BLE001 -- alerting must not crash the loop
            return
        if active:
            series = {k: v for k, v in series.items() if k in active}
        flags = self._current_sag_flags(series)
        if not flags:
            return
        lines = [f"{a} {c} {med:+.1f}% (latest {latest:+.1f}%)"
                 for a, c, med, latest in flags[:8]]
        more = f" (+{len(flags) - 8} more)" if len(flags) > 8 else ""
        self._fire_alert(
            "current_sag", "warning",
            f"{len(flags)} channel(s) show current-delivery sag "
            f"(<{self.current_sag_pct:.0f}% two-edge fidelity → commanded "
            f"current not fully delivered): " + "; ".join(lines) + more,
            resend_key=self._drift_signature(flags, "warning"))

    def _current_sag_flags(self, series: dict) -> list:
        """Every (animal, channel, median_fidelity, latest) whose rolling-median
        current fidelity is below the (negative) sag threshold. Most-negative
        first. Stale (frozen) series are skipped."""
        flags = []
        for (animal, ch), rows in (series or {}).items():
            if not self._series_fresh(rows):
                continue
            vals = [r.get("current_fidelity_pct") for r in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) < max(2, self.impedance_min_history):
                continue
            med = _median(vals[max(0, len(vals) - self.impedance_baseline_window):])
            if med is not None and med < self.current_sag_pct:
                flags.append((animal, ch, med, vals[-1]))
        flags.sort(key=lambda f: f[2])
        return flags

    def _fire_alert(self, alert_type: str, severity: str, message: str,
                    file_id: int = None, session_dir: str = None, *,
                    resend_key: str = None, figure_builder=None):
        """Record + email an alert.

        *resend_key* enables per-CONDITION daily dedup: with it, a STANDING
        condition emails at most once per day (not once per rate-limit window),
        so a drift that never clears stops spamming hourly; a changed signature
        (new channel / escalation) is a new digest and fires again. Without it,
        the legacy per-TYPE rate limit applies. *figure_builder*, when given, is
        called only after the alert clears both gates and returns temp PNG paths
        to embed inline (the caller-agnostic figure is deleted after send)."""
        # Per-type burst rate limit (unchanged) -- caps any one type's cadence.
        last = self.store.get_last_alert_time(alert_type)
        if last and (datetime.now() - last).total_seconds() < self.rate_limit_minutes * 60:
            logger.debug("Alert rate-limited: %s", alert_type)
            return
        # Per-condition daily dedup: atomically claim (type, signature) for today.
        today = datetime.now().strftime("%Y-%m-%d")
        digest = f"{alert_type}:{resend_key}" if resend_key is not None else None
        if digest is not None and not self.store.record_notification_sent(
                digest, today):
            logger.debug("Alert deduped for today: %s", digest)
            return

        # Record it for the UI alerts log regardless. CHECK the email result:
        # send() returns False when disabled/misconfigured or on SMTP failure,
        # and silently ignoring that meant a failed CRITICAL email went nowhere.
        self.store.insert_alert(alert_type, severity, message, file_id, session_dir)
        figs = []
        if figure_builder is not None:
            try:
                figs = figure_builder() or []
            except Exception:  # noqa: BLE001 -- a figure must never block the alert
                logger.exception("alert figure render failed [%s]", alert_type)
                figs = []
        body_html = (self._alert_body_html(severity, message, figs)
                     if figs else None)
        try:
            ok = self.emailer.send(
                f"{alert_type}: {message[:80]}", message, severity,
                body_html=body_html, attachments=(figs or None))
        except Exception as e:  # noqa: BLE001 -- alerting must not crash the loop
            ok = False
            logger.error("Alert email raised [%s/%s]: %s (%s)",
                         severity, alert_type, message, e)
        finally:
            for p in figs:
                try:
                    os.remove(p)
                except OSError:
                    pass
        if not ok and digest is not None:
            # Send failed -> release the daily claim so a later tick can retry
            # rather than permanently suppressing this condition for the day.
            try:
                self.store.release_notification_sent(digest, today)
            except Exception:  # noqa: BLE001
                logger.debug("release_notification_sent failed", exc_info=True)
        if ok:
            logger.warning("Alert fired [%s/%s]: %s",
                           severity, alert_type, message)
        else:
            logger.error("Alert recorded but email NOT sent [%s/%s]: %s",
                         severity, alert_type, message)

    @staticmethod
    def _alert_body_html(severity: str, message: str, figure_paths: list) -> str:
        """HTML alert body that renders the message + inline trend figures
        (referenced by Content-ID = basename, matching EmailAlerter._load_images)."""
        color = {"critical": "#c0392b", "warning": "#e8833a"}.get(
            severity, "#5e7ce2")
        imgs = "".join(
            f'<img src="cid:{os.path.basename(p)}" '
            'style="max-width:760px;display:block;margin:12px 0">'
            for p in figure_paths)
        return (
            '<html><body style="font-family:-apple-system,sans-serif">'
            f'<h3 style="color:{color};margin:0 0 8px">{escape(message[:120])}</h3>'
            '<pre style="white-space:pre-wrap;color:#333;font-size:13px">'
            f'{escape(message)}</pre>{imgs}<hr>'
            '<small style="color:#888">QC Monitor - '
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>'
            '</body></html>')

    def _drift_figures(self, series: dict, flags: list, *, value_key: str,
                       ylabel: str, series_name: str, metric_label: str,
                       pct: float, max_figs: int = 3) -> list:
        """Render the per-channel drift-trend PNG for the top drifting channels
        to temp files (the caller deletes them after the email is sent). Reuses
        the weekly report's ``impedance_png`` so the alert figure matches the
        dashboard/report look."""
        import tempfile
        from src.notifications.stim_figures import impedance_png
        paths = []
        for f in flags[:max_figs]:
            animal, ch = f[0], f[1]
            rows = (series or {}).get((animal, ch)) or []
            fd, p = tempfile.mkstemp(prefix="qc_alert_", suffix=".png")
            os.close(fd)
            out = impedance_png(
                rows, p, title=f"{animal} {ch} — {metric_label}", pct=pct,
                window=self.impedance_baseline_window,
                min_history=self.impedance_min_history,
                value_key=value_key, ylabel=ylabel, series_name=series_name)
            if out:
                paths.append(out)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return paths

    def check_flagged_events(self):
        """Email the configured recipients when a NEW seizure event is
        auto-filter flagged for an animal opted in via the Overview card
        checkbox.

        Edge-triggered and deduped per flagged file (``notification_log``), so a
        standing flag pool never re-emails -- only a file flagged AFTER opt-in
        does (existing flags are seeded at opt-in by
        ``Store.seed_flag_email_baseline``). Does NOT go through ``_fire_alert``:
        that rate-limits per alert TYPE, which would swallow a second animal's
        flag within the window; here every distinct file must get through once.
        Never raises -- alerting must not crash the poll loop.
        """
        try:
            subs = self.store.flag_email_subscribed_animals()
            if not subs:
                return
            per_animal = self.store.flagged_pool_file_ids_per_animal()
        except Exception as e:  # noqa: BLE001 -- alerting must not crash the loop
            logger.error("flag-email check failed to read state: %s", e)
            return
        for animal in sorted(subs):
            for fid in sorted(per_animal.get(animal, set())):
                self._fire_flagged_event(animal, fid)

    def _fire_flagged_event(self, animal: str, file_id: int):
        """Send (once) the flagged-event email for one (animal, file)."""
        digest = self.store.flag_email_digest(animal, file_id)
        sent_key = self.store.FLAG_EMAIL_SENT_KEY
        # Atomic claim: the first caller to see this (animal, file) wins, so the
        # email goes out exactly once across daemon restarts / multiple daemons.
        if not self.store.record_notification_sent(digest, sent_key):
            return
        subject = f"{animal}: new flagged seizure event"
        body = (f"A seizure event was auto-filter flagged for {animal} "
                f"(recording file id {file_id}) and is awaiting review in the "
                f"Video Review 'Flag' queue.\n\n"
                f"You are receiving this because {animal}'s flagged-event email "
                f"alert is enabled on the Overview behavioral-seizure card. "
                f"Turn it off there to stop these.")
        try:
            ok = self.emailer.send(subject, body, "warning")
        except Exception as e:  # noqa: BLE001 -- alerting must not crash the loop
            ok = False
            logger.error("flag-email raised [%s/file %s]: %s", animal, file_id, e)
        if ok:
            self.store.insert_alert("flagged_event", "warning", body)
            logger.warning("flag-email sent: %s file %s", animal, file_id)
        else:
            # Release the claim so a later tick retries rather than permanently
            # suppressing this file on a transient SMTP failure.
            self.store.release_notification_sent(digest, sent_key)
            logger.error("flag-email NOT sent, claim released: %s file %s",
                         animal, file_id)
