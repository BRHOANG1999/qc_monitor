"""Gmail SMTP email alerting with App Password."""

import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
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
        # Password lookup order, re-evaluated on every send so the user
        # can rotate the App Password without restarting the watcher:
        #   1. env var QC_MONITOR_EMAIL_PASSWORD (or `password_env_var`)
        #   2. file at `password_file` (default secrets/smtp_password.txt)
        # The file path falls back to the project-root relative form
        # when not absolute. Windows services don't inherit interactive-
        # shell env vars cleanly, so the file is the reliable channel.
        self._pw_env_var = smtp_cfg.get(
            "password_env_var", "QC_MONITOR_EMAIL_PASSWORD")
        self._pw_file = smtp_cfg.get(
            "password_file", "secrets/smtp_password.txt")

    @property
    def password(self) -> str:
        # File first: the explicit, user-written secret beats whatever
        # env var the NSSM service may have captured at install time.
        # Lets the user rotate the App Password by overwriting the
        # file -- no env-var dance, no service restart needed.
        if self._pw_file:
            path = self._pw_file
            if not os.path.isabs(path):
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", ".."))
                path = os.path.normpath(os.path.join(project_root, path))
            try:
                if os.path.isfile(path):
                    # utf-8-sig auto-strips the BOM that PowerShell's
                    # `Set-Content -Encoding utf8` prepends -- otherwise
                    # the BOM lands in the password string and SMTP's
                    # ASCII auth fails with
                    # "'ascii' codec can't encode character '\\ufeff'".
                    with open(path, "r", encoding="utf-8-sig") as f:
                        raw = f.read()
                    # Strip ALL whitespace (incl. internal spaces).
                    # Gmail App Passwords are typically shown as four
                    # space-separated groups; both forms authenticate.
                    pw = "".join(raw.split())
                    if pw:
                        return pw
            except OSError as e:
                logger.warning("SMTP password_file %s unreadable: %s",
                                path, e)
        # Env var fallback -- still useful for interactive testing
        # without touching files.
        return os.environ.get(self._pw_env_var, "").strip()

    @staticmethod
    def _load_images(paths: list[str]):
        """Load image files as (cid, MIMEImage) with inline Content-ID =
        basename, so HTML can reference them as ``cid:<basename>``. Unreadable
        paths are skipped."""
        out = []
        for p in paths or []:
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            cid = os.path.basename(p)
            img = MIMEImage(data)
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=cid)
            out.append((cid, img))
        return out

    @staticmethod
    def _load_files(paths: list[str]):
        """Load arbitrary files as downloadable attachments (Content-Disposition
        attachment). Unreadable paths are skipped."""
        from email.mime.application import MIMEApplication
        out = []
        for p in paths or []:
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            name = os.path.basename(p)
            sub = "html" if name.lower().endswith((".html", ".htm")) else \
                  "octet-stream"
            part = MIMEApplication(data, _subtype=sub)
            part.add_header("Content-Disposition", "attachment", filename=name)
            out.append(part)
        return out

    def send(self, subject: str, body: str, severity: str = "info",
             recipients: list[str] | None = None,
             body_html: str | None = None,
             subject_prefix: bool = True,
             attachments: list[str] | None = None,
             file_attachments: list[str] | None = None) -> bool:
        """Send an email.

        *recipients* overrides the default alerting recipient list (used
        by the daily surgery digest to target a different mailbox / an
        email-to-SMS gateway). *body_html*, when provided, replaces the
        auto-generated severity-styled HTML; when None, the plaintext
        *body* is wrapped in a default template. *subject_prefix*
        attaches the "[INFO] QC Monitor:" prefix; set False to send a
        bare subject (the surgery digest does). *attachments* are image
        file paths embedded INLINE via Content-ID (referenced in the HTML
        as ``cid:<basename>``) so figures render in the email body; a
        client that can't render them still gets them as attachments.
        *file_attachments* are arbitrary files (e.g. the standalone HTML
        report) attached as downloadable, not inlined.
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
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body, "plain"))
        alt.attach(MIMEText(body_html, "html"))

        images = self._load_images(attachments or [])
        if images:
            # multipart/related so the HTML's cid: refs resolve inline.
            content = MIMEMultipart("related")
            content.attach(alt)
            for cid, img in images:
                content.attach(img)
        else:
            content = alt

        files = self._load_files(file_attachments or [])
        if files:
            # multipart/mixed wraps the body (+inline images) and the
            # downloadable attachments (the standalone HTML report).
            msg = MIMEMultipart("mixed")
            msg.attach(content)
            for part in files:
                msg.attach(part)
        else:
            msg = content
        msg["Subject"] = full_subject
        msg["From"] = self.from_email
        msg["To"] = ", ".join(recipients)

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
