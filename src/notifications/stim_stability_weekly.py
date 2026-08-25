"""Weekly stimulus-stability report: figures -> Google Drive -> running Sheet ->
Sunday email.

For every stimulated channel currently on the rig (``active_impedance_channel_keys``
-- a stimCopy channel + the eeg channel after it; each such ``evoked_waveforms``
row carries ``stim_mean_trace`` = the stim and ``mean_trace`` = the evoked
response), this renders:

  * average-stim waveform PNGs: per file (the most recent day by default,
    ``perfile_span: week`` for all), per day, and a weekly grand average;
  * the access-resistance (Rₐ) drift figure, as on the Overview tab;
  * a stim-magnitude vs evoked-magnitude (1-50 ms) correlation scatter.

The PNGs are uploaded to a per-week Shared-Drive folder; a running "Stim
Stability" sheet gets one upserted row per (week, animal, channel) with =IMAGE
thumbnails + a =HYPERLINK to the folder; and a Sunday digest email carries the
headline figures inline. Wired into ``DigestScheduler`` like ``evoked_weekly``.

Reuses: ``store.channel_stim_evoked_traces_in_range`` +
``store.impedance_series_by_channel``; ``trace_average.average_traces``;
``stim_figures`` (renderers); ``evoked_weekly._peak_amplitude`` (magnitudes);
``rules._drift_stat``; ``drive_upload``; ``sheets_write``; ``EmailAlerter.send``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import date, timedelta
from html import escape
from types import SimpleNamespace

from src.alerting.email_alert import EmailAlerter
from src.alerting.rules import _drift_stat
from src.db.store import Store
from src.notifications import stim_figures as _fig
from src.notifications.evoked_weekly import _peak_amplitude
from src.utils import drive_upload as _drive
from src.utils import sheets_write as _sw
from src.utils.sheets import _resolve_sa_path
from src.utils.trace_average import average_traces

logger = logging.getLogger("qc_monitor.notifications.stim_stability_weekly")

HEADER = ["Week", "Animal", "Channel", "Ra kΩ", "Ra Δ%", "corr r", "corr p",
          "n files", "Weekly stim", "Impedance", "Correlation", "Figures"]
_TAB = "Stim Stability"


# ------------------------------------------------------------- helpers ---- #

def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s))


def _week_span(today: date) -> tuple[str, str, str]:
    """(start_date, end_date, 'YYYY-Www') for the 7 days ending *today*."""
    start = today - timedelta(days=6)
    iso = today.isocalendar()
    return start.isoformat(), today.isoformat(), f"{iso[0]}-W{iso[1]:02d}"


def _resolve_params(cfg: dict, config: dict) -> SimpleNamespace:
    gs = (config.get("google_sheets", {}) or {})
    gd = (config.get("google_drive", {}) or {})
    sa_raw = (gd.get("service_account_file") or gs.get("service_account_file")
              or (config.get("surgeries", {}) or {}).get("service_account_file"))
    tok_raw = gd.get("token_file") or "secrets/drive_oauth_token.json"
    return SimpleNamespace(
        sa_path=_resolve_sa_path(config, sa_raw) if sa_raw else "",
        drive_auth=(gd.get("auth_mode") or "service_account"),
        token_file=_resolve_sa_path(config, tok_raw),
        sheet_id=cfg.get("sheet_id") or gs.get("spreadsheet_id") or "",
        tab=cfg.get("tab_name", _TAB),
        folder_id=cfg.get("folder_id") or cfg.get("shared_drive_folder_id")
        or gd.get("folder_id") or gd.get("shared_drive_folder_id") or "",
        exclude=list(cfg.get("exclude", []) or []),
        pct=float(cfg.get("drift_threshold_pct", 40.0)),
        window=int(cfg.get("baseline_weeks", 10)),
        min_history=int(cfg.get("min_history", 4)),
        perfile_span=str(cfg.get("perfile_span", "last_day")),
        stim_window=cfg.get("stim_window_ms"),
        throttle=float(cfg.get("upload_throttle_sec", 0.1)),
    )


def _excluded(animal: str, exclude: list[str]) -> bool:
    a = (animal or "").lower()
    return any(e.lower() in a for e in exclude if e)


def _group_by_day(traces: list[dict]) -> dict:
    """{'YYYY_MM_DD': [traces...]} preserving order."""
    out: dict = {}
    for t in traces:
        day = str(t.get("chunk_datetime", ""))[:10]
        out.setdefault(day, []).append(t)
    return out


def _magnitude_pairs(traces: list[dict], stim_win) -> tuple[list, list]:
    """Per-file (stim peak-to-trough, evoked 1-50 ms peak-to-trough)."""
    xs, ys = [], []
    for t in traces:
        tm = t.get("time_ms") or []
        if not tm:
            continue
        sx0, sx1 = (stim_win if stim_win else (min(tm), max(tm)))
        sm = _peak_amplitude(tm, t.get("stim_trace") or [], sx0, sx1)
        ev = _peak_amplitude(tm, t.get("evoked_trace") or [], 1.0, 50.0)
        if sm is not None and ev is not None:
            xs.append(sm)
            ys.append(ev)
    return xs, ys


# --------------------------------------------------------- per channel ---- #

def _process_channel(animal, channel, traces, imp_rows, p, work) -> dict | None:
    """Render every PNG + compute the scalars for one channel. Returns a result
    dict, or None when the channel has no stim traces this week."""
    if not traces:
        return None
    base = os.path.join(work, _safe(animal), _safe(channel))
    os.makedirs(base, exist_ok=True)
    tag = f"{_safe(animal)}_{_safe(channel)}"
    days = _group_by_day(traces)
    perfile_src = (days[max(days)] if p.perfile_span == "last_day" and days
                   else traces)
    # per-file
    perfile = []
    for t in perfile_src:
        pth = os.path.join(base, f"file_{_safe(t['chunk_datetime'])}.png")
        if _fig.plot_stim_trace(t["time_ms"], t["stim_trace"],
                                f"{animal} {channel} · {t['chunk_datetime']}", pth):
            perfile.append(pth)
    # per-day averages
    daily = []
    for day, dts in sorted(days.items()):
        tm, y = average_traces(dts)
        pth = os.path.join(base, f"day_{_safe(day)}.png")
        if _fig.plot_stim_trace(tm, y, f"{animal} {channel} · {day} avg", pth):
            daily.append(pth)
    # weekly grand average
    tm, y = average_traces(traces)
    weekly = os.path.join(base, f"{tag}_weekly.png")
    if not _fig.plot_stim_trace(tm, y, f"{animal} {channel} · weekly avg stim",
                                weekly):
        weekly = None
    # impedance trend + correlation
    imp = os.path.join(base, f"{tag}_impedance.png")
    imp = _fig.impedance_png(imp_rows, imp, title=f"{animal} {channel} Rₐ",
                             pct=p.pct, window=p.window,
                             min_history=p.min_history)
    xs, ys = _magnitude_pairs(traces, p.stim_window)
    corr_png = os.path.join(base, f"{tag}_correlation.png")
    corr = _fig.plot_stim_vs_evoked(xs, ys, f"{animal} {channel}", corr_png)
    stat = _drift_stat([r.get("access_r_kohm") for r in imp_rows
                        if r.get("access_r_kohm") is not None],
                       p.window, p.min_history)
    return {
        "animal": animal, "channel": channel, "n_files": len(traces),
        "ra": (stat["latest"] if stat else None),
        "ra_drift": (stat["drift_pct"] if stat else None),
        "corr": corr,
        "png_weekly": weekly, "png_impedance": imp, "png_corr": corr["png"],
        "perfile": perfile, "daily": daily,
        "urls": {}, "folder_link": "",
    }


def _upload_channel(drive, week_folder_id, res, throttle) -> None:
    """Upload every PNG into a per-channel subfolder; record headline URLs +
    the folder link on *res*."""
    sub = _drive.ensure_folder(
        drive, f"{_safe(res['animal'])}_{_safe(res['channel'])}", week_folder_id)
    res["folder_link"] = sub.get("webViewLink", "")
    for key, path in (("weekly", res["png_weekly"]),
                      ("impedance", res["png_impedance"]),
                      ("corr", res["png_corr"])):
        if path:
            res["urls"][key] = _drive.upload_png(
                drive, path, sub["id"], throttle_sec=throttle)["image_url"]
    for path in res["daily"] + res["perfile"]:            # archived, not linked
        _drive.upload_png(drive, path, sub["id"], throttle_sec=throttle)


def _num(v, fmt: str) -> str:
    return format(v, fmt) if isinstance(v, (int, float)) and v == v else ""


def _sheet_row(week: str, res: dict) -> dict:
    def img(u):
        return f'=IMAGE("{u}")' if u else ""
    urls = res["urls"]
    c = res["corr"]
    folder = res["folder_link"]
    return {
        "Week": week, "Animal": res["animal"], "Channel": res["channel"],
        "Ra kΩ": _num(res["ra"], ".1f"), "Ra Δ%": _num(res["ra_drift"], "+.0f"),
        "corr r": _num(c.get("r"), ".2f"), "corr p": _num(c.get("p"), ".1e"),
        "n files": res["n_files"],
        "Weekly stim": img(urls.get("weekly")),
        "Impedance": img(urls.get("impedance")),
        "Correlation": img(urls.get("corr")),
        "Figures": (f'=HYPERLINK("{folder}","open folder")' if folder else ""),
    }


# --------------------------------------------------------------- email ---- #

_EMAIL_ATTACH_CAP = 20 * 1024 * 1024      # keep the digest email under ~20 MB


def _email_pngs(results, cap_bytes: int) -> tuple[list, list, int]:
    """Split every rendered PNG into (inline headline, downloadable rest,
    dropped). The 3 headline figures per channel go inline (cid); daily +
    per-file go as file attachments until the size cap, so figures survive even
    if Drive/token ever lapses."""
    inline, files, size, dropped = [], [], 0, 0
    for r in results:
        for path in (r["png_weekly"], r["png_impedance"], r["png_corr"]):
            if path and os.path.exists(path):
                inline.append(path)
                size += os.path.getsize(path)
    for r in results:
        for path in r["daily"] + r["perfile"]:
            if not (path and os.path.exists(path)):
                continue
            s = os.path.getsize(path)
            if size + s > cap_bytes:
                dropped += 1
                continue
            files.append(path)
            size += s
    return inline, files, dropped


def _email_html(today, week, results, folder_link, sheet_id, dropped=0) -> str:
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;color:#1a1a2e">',
        f'<h2>Weekly stimulus stability — {week} ({today.isoformat()})</h2>',
        f'<p style="color:#6c6c80">{len(results)} stimulated channel(s). '
        'Rₐ = access resistance; corr = stim-magnitude vs evoked (1–50 ms).</p>',
        '<table cellpadding="6" style="border-collapse:collapse;'
        'border:1px solid #ddd"><tr style="background:#f4f6fb">'
        '<th>Channel</th><th>Rₐ kΩ</th><th>Rₐ Δ%</th><th>corr r</th>'
        '<th>corr p</th><th>n</th></tr>',
    ]
    for r in results:
        drift = r["ra_drift"]
        col = ("#c0392b" if isinstance(drift, (int, float)) and abs(drift) >= 40
               else "#1a1a2e")
        parts.append(
            f'<tr><td>{escape(r["animal"])} {escape(r["channel"])}</td>'
            f'<td>{_num(r["ra"], ".1f")}</td>'
            f'<td style="color:{col}">{_num(drift, "+.0f")}</td>'
            f'<td>{_num(r["corr"].get("r"), ".2f")}</td>'
            f'<td>{_num(r["corr"].get("p"), ".1e")}</td>'
            f'<td>{r["n_files"]}</td></tr>')
    parts.append('</table>')
    for r in results:                                     # headline figures inline
        parts.append(f'<h3>{escape(r["animal"])} {escape(r["channel"])}</h3>')
        for path in (r["png_weekly"], r["png_impedance"], r["png_corr"]):
            if path:
                parts.append(f'<img src="cid:{os.path.basename(path)}" '
                             'style="max-width:640px;display:block;margin:6px 0">')
    if sheet_id:
        parts.append(f'<p><a href="https://docs.google.com/spreadsheets/d/'
                     f'{sheet_id}">Open the Stim Stability sheet</a></p>')
    if folder_link:
        parts.append(f'<p><a href="{folder_link}">Open this week\'s Drive '
                     'folder (all figures)</a></p>')
    if dropped:
        parts.append(f'<p style="color:#6c6c80"><small>{dropped} more per-file/'
                     'daily figure(s) omitted to keep this email small — see the '
                     'Drive folder for the full set.</small></p>')
    parts.append('<hr><small style="color:#6c6c80">QC Monitor stimulus-stability '
                 'weekly. Edit config.yaml → notifications.stim_stability_weekly.'
                 '</small></body></html>')
    return "".join(parts)


def _email_text(today, week, results) -> str:
    lines = [f"Weekly stimulus stability -- {week} ({today.isoformat()})",
             f"  {len(results)} stimulated channel(s)", ""]
    for r in results:
        lines.append(f"  {r['animal']} {r['channel']}: Ra={_num(r['ra'], '.1f')} kΩ "
                     f"Δ={_num(r['ra_drift'], '+.0f')}% "
                     f"corr r={_num(r['corr'].get('r'), '.2f')} "
                     f"p={_num(r['corr'].get('p'), '.1e')} n={r['n_files']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- entry --- #

def send_stim_stability_weekly(today: date, config: dict, store: Store,
                               emailer: EmailAlerter, *, dry_run: bool = False,
                               out_dir: str | None = None) -> dict:
    """Build and dispatch the weekly stimulus-stability report. *dry_run* renders
    PNGs into *out_dir* (or a temp dir) and returns the would-be rows/summary
    WITHOUT uploading to Drive, writing the sheet, or emailing."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    cfg = (config.get("notifications", {}) or {}).get(
        "stim_stability_weekly", {}) or {}
    if not dry_run and not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}
    recipients = list(cfg.get("recipients")
                      or ((config.get("alerting", {}) or {}).get("smtp", {})
                          .get("recipients", []) or []))
    if not dry_run and not recipients:
        return {"sent": False, "reason": "no recipients"}

    p = _resolve_params(cfg, config)
    start_date, end_date, week = _week_span(today)
    channels = [c for c in sorted(store.active_impedance_channel_keys())
                if not _excluded(c[0], p.exclude)]
    imp_series = store.impedance_series_by_channel(exclude=p.exclude)
    work = out_dir or tempfile.mkdtemp(prefix="stimstab_")

    drive = week_folder = None
    if not dry_run and p.folder_id:
        try:
            drive = _drive.drive_from_config(p.drive_auth, sa_path=p.sa_path,
                                             token_file=p.token_file)
            week_folder = _drive.ensure_folder(drive, week, p.folder_id)
        except Exception as e:                            # noqa: BLE001
            logger.warning("Drive unavailable (%s), emailing figures only: %s",
                           p.drive_auth, e)
            drive = week_folder = None

    rows, results = [], []
    for animal, channel in channels:
        traces = store.channel_stim_evoked_traces_in_range(
            animal, channel, start_date, end_date)
        res = _process_channel(animal, channel, traces,
                               imp_series.get((animal, channel), []), p, work)
        if res is None:
            continue
        if drive is not None and week_folder is not None:
            try:
                _upload_channel(drive, week_folder["id"], res, p.throttle)
            except Exception as e:                        # noqa: BLE001
                logger.warning("Drive upload failed for %s %s: %s",
                               animal, channel, e)
        rows.append(_sheet_row(week, res))
        results.append(res)

    sheet_res = {"updated": 0, "appended": 0}
    if rows and not dry_run and p.sheet_id and p.sa_path:
        try:
            svc = _sw._sheets_api_rw(p.sa_path)
            _sw.ensure_tab(svc, p.sheet_id, p.tab, HEADER)
            sheet_res = _sw.upsert_rows_into_tab(
                svc, p.sheet_id, p.tab, ["Week", "Animal", "Channel"], rows,
                column_map={h: h for h in HEADER})
        except Exception as e:                            # noqa: BLE001
            logger.warning("Stim-stability sheet write failed: %s", e)

    folder_link = week_folder.get("webViewLink", "") if week_folder else ""
    sent = False
    if not dry_run and recipients and results:
        inline, files, dropped = _email_pngs(results, _EMAIL_ATTACH_CAP)
        sent = bool(emailer.send(
            subject=f"QC weekly stim stability -- {week} "
                    f"({len(results)} channel{'s' if len(results) != 1 else ''})",
            body=_email_text(today, week, results),
            body_html=_email_html(today, week, results, folder_link, p.sheet_id,
                                  dropped=dropped),
            recipients=recipients, subject_prefix=False,
            attachments=inline, file_attachments=files))

    if not dry_run and out_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return {"sent": sent, "channels": len(results), "recipients": len(recipients),
            "sheet": sheet_res, "week": week, "work_dir": work,
            "dry_run": dry_run,
            "rows": rows if dry_run else None}


def _main(argv=None) -> int:
    """Dry-run CLI: render every PNG locally and print the would-be rows, with NO
    Drive upload / sheet write / email. Eyeball the figures before enabling."""
    import argparse
    import datetime as _dt

    import yaml

    ap = argparse.ArgumentParser(description="Weekly stim-stability dry run")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--db", default=None, help="override database.path")
    ap.add_argument("--out-dir", default="data/stim_stability_preview")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db = args.db or (config.get("database", {}) or {}).get("path")
    store = Store(db)
    emailer = EmailAlerter(config)
    today = (_dt.date.fromisoformat(args.today) if args.today
             else _dt.date.today())
    os.makedirs(args.out_dir, exist_ok=True)
    res = send_stim_stability_weekly(today, config, store, emailer,
                                     dry_run=True, out_dir=args.out_dir)
    print(f"week={res['week']} channels={res['channels']} "
          f"PNGs under {res['work_dir']}")
    for row in (res.get("rows") or []):
        print(f"  {row['Animal']} {row['Channel']}: Ra={row['Ra kΩ']} "
              f"Δ={row['Ra Δ%']} corr_r={row['corr r']} n={row['n files']}")
    if not res.get("rows"):
        print("  (no stimulated channels had traces in the last 7 days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
