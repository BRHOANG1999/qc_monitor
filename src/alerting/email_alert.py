"""Gmail SMTP email alerting with App Password."""

import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger("qc_monitor.alerting")


class EmailAlerter:
    def __init__(self, config: dict):
        smtp_cfg = config.get("alerting", {}).get("smtp", {})
        self.server = smtp_cfg.get("server", "smtp.gmail.com")
        self.port = smtp_cfg.get("port", 587)
        self.from_email = smtp_cfg.get("from_email", "")
        self.recipients = smtp_cfg.get("recipients", [])
        self.enabled = config.get("alerting", {}).get("enabled", True)

        # Password from environment variable
        pw_env = smtp_cfg.get("password_env_var", "QC_MONITOR_EMAIL_PASSWORD")
        self.password = os.environ.get(pw_env, "")

    def send(self, subject: str, body: str, severity: str = "info",
             recipients: list[str] | None = None,
             body_html: str | None = None,
             subject_prefix: bool = True) -> bool:
        """Send an email.

        *recipients* overrides the default alerting recipient list (used
        by the daily surgery digest to target a different mailbox / an
        email-to-SMS gateway). *body_html*, when provided, replaces the
        auto-generated severity-styled HTML; when None, the plaintext
        *body* is wrapped in a default template. *subject_prefix*
        attaches the "[INFO] QC Monitor:" prefix; set False to send a
        bare subject (the surgery digest does).
        """
        recipients = recipients if recipients is not None else self.recipients
        if not self.enabled or not self.password or not recipients:
            logger.debug("Email not sent (disabled or missing config): %s", subject)
            return False

        if subject_prefix:
            severity_prefix = {"critical": "[CRITICAL]", "warning": "[WARNING]",
                                "info": "[INFO]"}
            full_subject = f"{severity_prefix.get(severity, '')} QC Monitor: {subject}"
        else:
            full_subject = subject

        msg = MIMEMultipart("alternative")
        msg["Subject"] = full_subject
        msg["From"] = self.from_email
        msg["To"] = ", ".join(recipients)

        if body_html is None:
            body_html = f"""
            <html><body>
            <h3 style="color: {'red' if severity == 'critical' else 'orange' if severity == 'warning' else 'blue'}">
                {full_subject}
            </h3>
            <pre>{body}</pre>
            <hr>
            <small>QC Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </body></html>
            """
        msg.attach(MIMEText(body, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP(self.server, self.port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(self.from_email, self.password)
                smtp.sendmail(self.from_email, recipients, msg.as_string())
            logger.info("Alert email sent to %d recipient(s): %s",
                        len(recipients), subject)
            return True
        except Exception as e:
            logger.error("Failed to send email: %s", e)
            return False
