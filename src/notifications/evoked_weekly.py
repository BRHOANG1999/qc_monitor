"""Weekly evoked-response drift digest.

For each active session (last_chunk in past 7 days), compute the
median peak amplitude per LFP channel in the analysis window for
this week vs last week, and flag channels whose magnitude shifted
by more than ``drift_threshold_pct`` (default 30%). Sent Sunday
mornings so the user gets a single weekly summary instead of one
email per session.

This is intentionally simple math: peak-to-trough range of the mean
trace within the analysis window. Anything more sophisticated
belongs in the dedicated Evoked tab; the digest just flags
candidates worth eyeballing.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, date, timedelta
from html import escape

from src.alerting.email_alert import EmailAlerter
from src.db.store import Store

logger = logging.getLogger("qc_monitor.notifications.evoked_weekly")


def _peak_amplitude(time_ms: list[float], mean_trace: list[float],
                     x0: float, x1: float) -> float | None:
    """Peak-to-trough range of *mean_trace* within [x0, x1] ms.

    Returns None if the window has fewer than 3 samples or all-NaN
    values, so the caller can skip rather than divide by zero.
    """
    if not time_ms or not mean_trace or len(time_ms) != len(mean_trace):
        return None
    vals = [m for t, m in zip(time_ms, mean_trace)
             if t is not None and m is not None
             and x0 <= t <= x1 and isinstance(m, (int, float))]
    # Drop NaNs (json-decoded NaN becomes float('nan') in Python).
    vals = [v for v in vals if v == v]
    if len(vals) < 3:
        return None
    return float(max(vals) - min(vals))


def _median_amp_per_channel(waveforms: list[dict],
                             since: datetime, until: datetime,
                             x0: float, x1: float) -> dict[int, float]:
    """{channel: median peak-amplitude} over chunks in [since, until)."""
    bucket: dict[int, list[float]] = {}
    for wf in waveforms:
        dt_str = wf.get("chunk_datetime") or ""
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            continue
        if dt < since or dt >= until:
            continue
        amp = _peak_amplitude(
            wf.get("time_axis_ms") or [], wf.get("mean_trace") or [],
            x0, x1,
        )
        if amp is None:
            continue
        bucket.setdefault(int(wf.get("channel", 0)), []).append(amp)
    return {ch: statistics.median(v) for ch, v in bucket.items() if v}


def _collect_drift(store: Store, config: dict, today: date,
                    drift_pct: float) -> list[dict]:
    """Return per-session, per-channel drift rows whose magnitude
    moved >= drift_pct% week-over-week."""
    fa = (config.get("feature_analysis", {}) or {})
    x0 = float(fa.get("analysis_start_ms", 2.0))
    x1 = float(fa.get("analysis_end_ms", 200.0))

    now = datetime.combine(today, datetime.min.time())
    this_week_start = now - timedelta(days=7)
    last_week_start = now - timedelta(days=14)

    sessions = store.get_sessions()
    active = [s for s in sessions
              if (s.get("last_chunk") or "") >= this_week_start.isoformat()]

    drifts = []
    for s in active:
        try:
            waveforms = store.get_evoked_waveforms_for_session(
                s["session_dir"]
            )
        except Exception as e:
            logger.warning("Evoked fetch failed for %s: %s",
                            s.get("session_name"), e)
            continue
        if not waveforms:
            continue
        this_week = _median_amp_per_channel(
            waveforms, this_week_start, now, x0, x1,
        )
        last_week = _median_amp_per_channel(
            waveforms, last_week_start, this_week_start, x0, x1,
        )
        chan_names = {
            int(wf.get("channel", 0)):
                wf.get("channel_name", f"Ch{wf.get('channel', 0)}")
            for wf in waveforms
        }
        for ch, this_amp in this_week.items():
            last_amp = last_week.get(ch)
            if not last_amp:
                continue
            pct = 100.0 * (this_amp - last_amp) / last_amp
            if abs(pct) >= drift_pct:
                drifts.append({
                    "session_name": s.get("session_name", "?"),
                    "channel": ch,
                    "channel_name": chan_names.get(ch, f"Ch{ch}"),
                    "this_amp": this_amp,
                    "last_amp": last_amp,
                    "pct_change": pct,
                })
    drifts.sort(key=lambda d: abs(d["pct_change"]), reverse=True)
    return drifts


def _format_text(today: date, drifts: list[dict],
                  drift_pct: float) -> str:
    lines = [
        f"Weekly evoked drift -- {today.isoformat()}",
        f"  (channels with >= {drift_pct:.0f}% week-over-week change)",
        "",
    ]
    if not drifts:
        lines.append("  No channels above the drift threshold.")
        return "\n".join(lines)
    for d in drifts:
        sign = "+" if d["pct_change"] >= 0 else ""
        lines.append(
            f"  {sign}{d['pct_change']:.0f}%  "
            f"{d['channel_name']:>10}  "
            f"{d['session_name']}  "
            f"(this={d['this_amp']:.2f}, last={d['last_amp']:.2f})"
        )
    return "\n".join(lines).rstrip()


def _format_html(today: date, drifts: list[dict],
                  drift_pct: float) -> str:
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;'
        'color:#1a1a2e">',
        f'<h2>Weekly evoked drift -- {today.isoformat()}</h2>',
        f'<p style="color:#6c6c80">Channels with at least '
        f'{drift_pct:.0f}% week-over-week change in peak-to-trough '
        f'amplitude within the analysis window.</p>',
    ]
    if not drifts:
        parts.append('<p><em>No channels above the drift threshold '
                     'this week.</em></p>')
    else:
        parts.append(
            '<table cellpadding="6" '
            'style="border-collapse:collapse;border:1px solid #ddd">'
            '<tr style="background:#f4f6fb">'
            '<th>Δ%</th><th>Channel</th><th>Session</th>'
            '<th>This week</th><th>Last week</th></tr>'
        )
        for d in drifts:
            color = "#c0392b" if abs(d["pct_change"]) >= 50 else "#b8860b"
            sign = "+" if d["pct_change"] >= 0 else ""
            parts.append(
                f'<tr><td style="color:{color};font-weight:bold">'
                f'{sign}{d["pct_change"]:.0f}%</td>'
                f'<td>{escape(str(d["channel_name"]))}</td>'
                f'<td>{escape(str(d["session_name"]))}</td>'
                f'<td>{d["this_amp"]:.2f}</td>'
                f'<td>{d["last_amp"]:.2f}</td></tr>'
            )
        parts.append('</table>')
    parts.append(
        '<hr><small style="color:#6c6c80">QC Monitor evoked weekly '
        'digest. Edit threshold or recipients in config.yaml -> '
        'notifications.evoked_weekly.</small></body></html>'
    )
    return "".join(parts)


def send_evoked_weekly(today: date, config: dict, store: Store,
                        emailer: EmailAlerter) -> dict:
    """Build and dispatch the weekly evoked-drift summary."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    cfg = (config.get("notifications", {}) or {}).get(
        "evoked_weekly", {}) or {}
    if not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}

    recipients = list(
        cfg.get("recipients")
        or ((config.get("alerting", {}) or {}).get("smtp", {})
            .get("recipients", []) or [])
    )
    if not recipients:
        return {"sent": False, "reason": "no recipients"}

    drift_pct = float(cfg.get("drift_threshold_pct", 30.0))
    drifts = _collect_drift(store, config, today, drift_pct)
    subject = (f"QC weekly evoked drift -- {today.isoformat()} "
                f"({len(drifts)} channel"
                f"{'s' if len(drifts) != 1 else ''})")
    text_body = _format_text(today, drifts, drift_pct)
    html_body = _format_html(today, drifts, drift_pct)
    ok = emailer.send(
        subject=subject, body=text_body, body_html=html_body,
        recipients=recipients, subject_prefix=False,
    )
    return {
        "sent": bool(ok),
        "recipients": len(recipients),
        "drift_channels": len(drifts),
    }
