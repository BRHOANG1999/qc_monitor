"""Daily EVOKED-RESPONSE digest email -- the "kinds" of response by feature.

Distinct from the stim-quality / stim-stability digest: for each animal's primary
recording channel over YESTERDAY, split each evoked feature into even percentile
bands and show, per band, the mean response waveform (the top-amplitude kind, the
bottom-line-length kind, ...). Email-only (Drive/Sheet archiving deferred).

Cadence is driven by ``notifications.scheduler.DigestScheduler._tick_evoked_daily``.
On-demand / prototyping: ``python -m src.notifications.evoked_daily --dry-run``.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from datetime import datetime, timedelta

from src.notifications import evoked_digest as _ed
from src.utils.evoked_output import (list_evoked_files, animals_in_filename,
                                     parse_recording_dt)

logger = logging.getLogger("qc_monitor.notifications.evoked_daily")

_ATTACH_CAP = 20 * 1024 * 1024      # 20 MB total, mirrors the stim digest
_CAPTIONS = {
    "peak_to_trough": "Amplitude (peak−trough): top band = the consistently "
                      "strong responses, bottom band = flat / weak.",
    "line_length": "Line length (waveform wiggliness): bottom band = the "
                   "consistently weak-line-length responses.",
    "rms_amplitude": "RMS amplitude (overall response size) by percentile band.",
    "peak_latency_ms": "Peak latency by band — fast vs slow responders.",
    "recovery_tau": "Recovery tau (decay time constant) by band.",
}


def _animals_with_data(evoked_dir: str, end_day: datetime,
                       window_days: int = 1) -> list[str]:
    """Sorted animal ids with an evoked recording in the *window_days* ending on
    *end_day* (inclusive)."""
    d1 = datetime(end_day.year, end_day.month, end_day.day) + timedelta(days=1)
    d0 = d1 - timedelta(days=max(1, int(window_days)))
    out: set = set()
    for fp in list_evoked_files(evoked_dir):
        dt = parse_recording_dt(fp)
        if dt is not None and d0 <= dt < d1:
            out |= animals_in_filename(fp)
    return sorted(out)


def _split_attachments(results: list) -> tuple:
    """Inline EVERY feature figure (so all show in-body), overflowing to
    downloadable attachments only once the cumulative inline budget is hit.
    Returns (inline_paths, file_paths)."""
    inline, files, budget = [], [], _ATTACH_CAP
    for res in results:
        for _feat, png in res.get("pngs", {}).items():
            try:
                sz = os.path.getsize(png)
            except OSError:
                continue
            if sz <= budget:
                inline.append(png)
                budget -= sz
            else:
                files.append(png)
    return inline, files


def _email_html(results: list, period_label: str, features: list) -> str:
    parts = [f"<h2 style='font-family:sans-serif'>Evoked-response digest "
             f"— {period_label}</h2>",
             "<p style='font-family:sans-serif;color:#444'>Each figure splits a "
             "feature into 10 equal-population percentile bands; the right panel "
             "is the mean evoked response of each band (the “kinds”).</p>"]
    for res in results:
        parts.append(f"<h3 style='font-family:sans-serif'>{res['animal']} "
                     f"· {res['channel']} · n={res['n_responses']}</h3>")
        for feat in features:
            png = res.get("pngs", {}).get(feat)
            if not png:
                continue
            cid = os.path.basename(png)
            cap = _CAPTIONS.get(feat, feat.replace("_", " "))
            parts.append(
                f"<div style='margin:6px 0 16px'>"
                f"<img src='cid:{cid}' style='max-width:900px;width:100%'><br>"
                f"<span style='font-family:sans-serif;font-size:12px;color:#666'>"
                f"{cap}</span></div>")
        skipped = res.get("skipped") or []
        if skipped:
            parts.append("<p style='font-family:sans-serif;font-size:11px;"
                         "color:#999'>Skipped (no data this period): "
                         f"{', '.join(skipped)}.</p>")
    return "\n".join(parts)


def _email_text(results: list) -> str:
    lines = ["Evoked-response digest (percentile-band kinds).", ""]
    for res in results:
        lines.append(f"{res['animal']} · {res['channel']} · "
                     f"n={res['n_responses']} · features: "
                     f"{', '.join(res.get('pngs', {}))}")
    return "\n".join(lines)


def send_evoked_daily(today: datetime, config: dict, store, emailer, *,
                      window_days: int | None = None,
                      dry_run: bool = False, out_dir: str | None = None) -> dict:
    """Render + send the evoked-response digest ending the day BEFORE *today*.
    ``window_days`` (1 = daily, 7 = weekly) overrides the config. Returns
    ``{"sent", "animals", "recipients", "work_dir", ...}``. Never raises for
    per-animal data gaps (they're skipped)."""
    cfg = (config.get("notifications", {}) or {}).get("evoked_daily", {}) or {}
    evoked_dir = (config.get("chronic_evoked", {}) or {}).get("evoked_output_dir", "")
    features = cfg.get("features") or _ed.DEFAULT_FEATURES
    n_bands = int(cfg.get("n_bands", _ed.DEFAULT_N_BANDS))
    win = int(window_days if window_days is not None else cfg.get("window_days", 1))
    max_trace_files = int(cfg.get("max_trace_files", 24))
    overrides = cfg.get("channel_overrides", {}) or {}
    report_date = (today - timedelta(days=1))
    period_label = _ed._date_label(report_date, win)
    cadence = "weekly" if win > 1 else "daily"
    animals = cfg.get("animals") or _animals_with_data(evoked_dir, report_date, win)
    work = out_dir or tempfile.mkdtemp(prefix="evoked_digest_")
    os.makedirs(work, exist_ok=True)

    results = []
    for animal in animals:
        try:
            res = _ed.build_animal(animal, evoked_dir, report_date,
                                   features=features, n_bands=n_bands,
                                   window_days=win, max_trace_files=max_trace_files,
                                   work_dir=work, channel_override=overrides.get(animal))
        except Exception as e:                                   # noqa: BLE001
            logger.error("evoked digest: %s failed: %s", animal, e)
            continue
        if not res.get("empty"):
            results.append(res)

    if not results:
        return {"sent": False, "reason": "no evoked data", "animals": [],
                "work_dir": work}
    if dry_run:
        return {"sent": False, "dry_run": True, "work_dir": work,
                "period": period_label,
                "animals": [r["animal"] for r in results], "results": results}

    recipients = (cfg.get("recipients")
                  or (config.get("alerting", {}).get("smtp", {}) or {}).get("recipients"))
    inline, files = _split_attachments(results)
    sent = bool(emailer.send(
        subject=f"QC evoked-response {cadence} digest — {period_label} "
                f"({len(results)} animal{'s' if len(results) != 1 else ''})",
        body=_email_text(results),
        body_html=_email_html(results, period_label, features),
        recipients=recipients, subject_prefix=False,
        attachments=inline, file_attachments=files))
    return {"sent": sent, "animals": [r["animal"] for r in results],
            "recipients": recipients, "work_dir": work, "period": period_label}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--date", default=None, help="report day YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--animal", default=None, help="limit to one animal")
    ap.add_argument("--window-days", type=int, default=1,
                    help="1 = daily, 7 = weekly (window ending on --date)")
    ap.add_argument("--dry-run", action="store_true", help="render PNGs, do not send")
    ap.add_argument("--out", default=None, help="output dir for the PNGs")
    a = ap.parse_args(argv)
    if not a.dry_run:
        print("This CLI only renders (--dry-run). Real sends go through the "
              "daemon scheduler (notifications.evoked_daily.enabled).")
        return 2
    import yaml
    config = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.animal:
        (config.setdefault("notifications", {})
               .setdefault("evoked_daily", {}))["animals"] = [a.animal]
    # --date is the REPORT day; send_evoked_daily reports (today-1), so pass +1.
    today = (datetime.fromisoformat(a.date) + timedelta(days=1)
             if a.date else datetime.now())
    res = send_evoked_daily(today, config, None, None, window_days=a.window_days,
                            dry_run=True, out_dir=a.out)
    print("dry-run:", {k: v for k, v in res.items() if k != "results"})
    for r in res.get("results", []):
        print(f"  {r['animal']} {r['channel']} n={r['n_responses']}: "
              f"{list(r.get('pngs', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
