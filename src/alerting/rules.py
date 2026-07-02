"""Alert rule engine with rate limiting."""

import logging
import os
from datetime import datetime

from src.db.store import Store
from src.alerting.email_alert import EmailAlerter

logger = logging.getLogger("qc_monitor.alerting.rules")


class AlertRuleEngine:
    def __init__(self, store: Store, emailer: EmailAlerter, config: dict):
        self.store = store
        self.emailer = emailer
        rules_cfg = config.get("alerting", {}).get("rules", {})
        self.rate_limit_minutes = rules_cfg.get("rate_limit_per_type_minutes", 60)
        self.network_down_minutes = rules_cfg.get("network_down_minutes", 5)
        self.no_data_hours = rules_cfg.get("no_data_hours", 2)

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
        """Fire when a recording has sat on the share, unprocessed, for longer
        than ``no_data_hours`` -- i.e. the newest .mat on the share is that
        much newer than the last file we finished. Catches a daemon hang, a
        queue stall, AND the silent-skip case (a recorder rename where nothing
        ever reaches the queue, so pending stays 0)."""
        newest_share = (scan_stats or {}).get("newest_mat_mtime") or 0.0
        if newest_share <= 0:
            return                       # no files on the share to judge
        last_proc = self.store.newest_processed_at()
        last_proc_ts = last_proc.timestamp() if last_proc else 0.0
        if (newest_share - last_proc_ts) <= self.no_data_hours * 3600:
            return                       # keeping up (or nothing newer)
        newest_str = datetime.fromtimestamp(newest_share).strftime(
            "%Y-%m-%d %H:%M")
        last_str = last_proc.strftime("%Y-%m-%d %H:%M") if last_proc else "never"
        n_skip = (scan_stats or {}).get("n_skipped_recordinglike") or 0
        skip_note = (f" {n_skip} file(s) skipped the filename pattern."
                     if n_skip else "")
        self._fire_alert(
            "processing_stalled", "critical",
            f"Processing may be stalled: newest recording on the share "
            f"({newest_str}) is >{self.no_data_hours}h newer than the last "
            f"processed file (last processed {last_str}).{skip_note}")

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
