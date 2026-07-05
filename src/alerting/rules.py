"""Alert rule engine with rate limiting."""

import logging
import os
from datetime import datetime

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
        # Transfer-impedance drift thresholds (per channel, per phase).
        self.impedance_shift_pct = float(
            rules_cfg.get("impedance_shift_pct", 40))
        self.impedance_shift_pct_critical = float(
            rules_cfg.get("impedance_shift_pct_critical", 75))
        self.impedance_baseline_window = int(
            rules_cfg.get("impedance_baseline_window", 10))
        self.impedance_min_history = int(
            rules_cfg.get("impedance_min_history", 4))
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
            failed = self.store.matlab_failed_files(hours=lookback_hours)
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
        lines = [f"{a} {c} {st['latest']:.3f}kΩ vs "
                 f"{st['baseline']:.3f} ({st['drift_pct']:+.0f}%)"
                 for a, c, st in flags[:8]]
        more = f" (+{len(flags) - 8} more)" if len(flags) > 8 else ""
        self._fire_alert(
            "impedance_shift", "critical" if crit else "warning",
            f"{len(flags)} channel(s) drifted "
            f">{self.impedance_shift_pct:.0f}% from baseline access "
            f"resistance: " + "; ".join(lines) + more)

    def _impedance_flags(self, series: dict) -> list:
        """Every (animal, channel, drift_stat) whose access resistance is past
        the warn pct."""
        flags = []
        for (animal, ch), rows in (series or {}).items():
            st = _drift_stat([r.get("access_r_kohm") for r in rows],
                             self.impedance_baseline_window,
                             self.impedance_min_history)
            if st and abs(st["drift_pct"]) >= self.impedance_shift_pct:
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
            f"current not fully delivered): " + "; ".join(lines) + more)

    def _current_sag_flags(self, series: dict) -> list:
        """Every (animal, channel, median_fidelity, latest) whose rolling-median
        current fidelity is below the (negative) sag threshold. Most-negative
        first."""
        flags = []
        for (animal, ch), rows in (series or {}).items():
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
                    file_id: int = None, session_dir: str = None):
        # Rate limiting
        last = self.store.get_last_alert_time(alert_type)
        if last and (datetime.now() - last).total_seconds() < self.rate_limit_minutes * 60:
            logger.debug("Alert rate-limited: %s", alert_type)
            return

        self.store.insert_alert(alert_type, severity, message, file_id, session_dir)
        self.emailer.send(f"{alert_type}: {message[:80]}", message, severity)
        logger.warning("Alert fired [%s/%s]: %s", severity, alert_type, message)
