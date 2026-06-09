"""Alert rule engine with rate limiting."""

import logging
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
