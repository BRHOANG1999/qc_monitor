"""Alert rule engine with rate limiting."""

import logging
from datetime import datetime, timedelta

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
        self.artifact_pct_warning = rules_cfg.get("artifact_pct_warning", 50)

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

    def check_chunk_qc(self, file_id: int, channel: int, metrics: dict,
                       session_dir: str = None):
        # Artifact warning
        artifact_pct = metrics.get("artifact_pct", 0)
        if artifact_pct > self.artifact_pct_warning:
            self._fire_alert(
                "high_artifact", "warning",
                f"Artifact {artifact_pct:.1f}% on channel {channel} (threshold: {self.artifact_pct_warning}%)",
                file_id=file_id, session_dir=session_dir,
            )

        # Flat-line warning
        if metrics.get("flat_line_pct", 0) > 50:
            self._fire_alert(
                "flatline", "warning",
                f"Flat-line detected on channel {channel} ({metrics['flat_line_pct']:.1f}% of chunk)",
                file_id=file_id, session_dir=session_dir,
            )

    def check_stim_delivery(self, file_id: int, stim_metrics: dict,
                            session_dir: str = None):
        if stim_metrics.get("total_pulses", 0) == 0 and stim_metrics.get("expected_pulses", 0) > 0:
            self._fire_alert(
                "stim_delivery_failure", "warning",
                f"Stimulation channel {stim_metrics.get('stim_channel', '?')}: "
                f"0 pulses delivered (expected {stim_metrics.get('expected_pulses', '?')})",
                file_id=file_id, session_dir=session_dir,
            )

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
