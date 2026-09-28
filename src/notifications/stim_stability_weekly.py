"""Weekly stimulus-stability report: figures -> Google Drive -> running Sheet ->
Sunday email.

For every stimulated channel currently on the rig (``active_impedance_channel_keys``
-- a stimCopy channel + the eeg channel after it; each such ``evoked_waveforms``
row carries ``stim_mean_trace`` = the stim and ``mean_trace`` = the evoked
response), this renders:

  * average-stim waveform PNGs: per file (the most recent day by default,
    ``perfile_span: week`` for all), per day, and a weekly grand average;
  * the access-resistance (Rₐ) drift figure, as on the Overview tab;
  * the slow-phase steady-state impedance (Z_ss) drift figure -- the settled
    recharge-phase plateau, a stim-consistency signal complementary to Rₐ;
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
from datetime import date, datetime, timedelta
from html import escape
from types import SimpleNamespace

import numpy as np

from src.alerting.email_alert import EmailAlerter
from src.alerting.rules import _drift_stat
from src.db.store import Store
from src.notifications import stim_figures as _fig
from src.notifications.evoked_weekly import _peak_amplitude
from src.utils import drive_upload as _drive
from src.utils import sheets_write as _sw
from src.utils.animal import split_animal_electrode
from src.utils.evoked_output import read_file_evoked
from src.utils.sheets import _resolve_sa_path
from src.utils.trace_average import align_average_with_traces

logger = logging.getLogger("qc_monitor.notifications.stim_stability_weekly")

HEADER = ["Week", "Animal", "Channel", "Ra kΩ", "Ra Δ%", "Z_ss kΩ", "Z_ss Δ%",
          "corr r", "corr p", "n files", "LFP artifact", "LFP evoked",
          "Impedance", "Z_ss", "Correlation", "Figures"]
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
        write_sheet=bool(cfg.get("write_sheet", True)),
        tab=cfg.get("tab_name", _TAB),
        folder_id=cfg.get("folder_id") or cfg.get("shared_drive_folder_id")
        or gd.get("folder_id") or gd.get("shared_drive_folder_id") or "",
        exclude=list(cfg.get("exclude", []) or []),
        pct=float(cfg.get("drift_threshold_pct", 40.0)),
        window=int(cfg.get("baseline_weeks", 10)),
        min_history=int(cfg.get("min_history", 4)),
        perfile_span=str(cfg.get("perfile_span", "last_day")),
        artifact_window=list(cfg.get("artifact_window_ms", [-1.0, 1.0])),
        evoked_window=list(cfg.get("evoked_window_ms", [1.0, 50.0])),
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


def _group_by_hour(traces: list[dict]) -> dict:
    """{'YYYY_MM_DD__HH': [traces...]} preserving order. Recordings are ~hourly
    but not strictly one-per-clock-hour, so group by the hour of the start
    timestamp -- ``chunk_datetime[:14]`` ('YYYY_MM_DD' + '__' + 'HH')."""
    out: dict = {}
    for t in traces:
        hour = str(t.get("chunk_datetime", ""))[:14]
        out.setdefault(hour, []).append(t)
    return out


def _magnitude_pairs(traces: list[dict], artifact_win, evoked_win
                     ) -> tuple[list, list, list]:
    """Per-file, from the SAME LFP (mean_trace): (stim-artifact peak-to-trough
    over *artifact_win*, evoked peak-to-trough over *evoked_win*, recording epoch
    seconds). The last drives the correlation graph's time colouring."""
    xs, ys, days = [], [], []
    for t in traces:
        tm = t.get("time_ms") or []
        lfp = t.get("evoked_trace") or []
        if not tm:
            continue
        art = _peak_amplitude(tm, lfp, artifact_win[0], artifact_win[1])
        ev = _peak_amplitude(tm, lfp, evoked_win[0], evoked_win[1])
        if art is None or ev is None:
            continue
        xs.append(art)
        ys.append(ev)
        days.append(str(t.get("chunk_datetime", ""))[:10])
    return xs, ys, days


def _crop(time_ms, y, win) -> tuple:
    """Restrict a (time_ms, y) trace to the *win* = (x0, x1) ms window so the
    stim-pulse figures show the pulse, not the full +/-500 ms record. Falls back
    to the full trace when the window would keep < 2 samples."""
    if not win or time_ms is None or y is None:
        return time_ms, y
    x0, x1 = win
    ct = [t for t in time_ms if x0 <= t <= x1]
    cy = [v for t, v in zip(time_ms, y) if x0 <= t <= x1]
    return (ct, cy) if len(ct) >= 2 else (time_ms, y)


# --------------------------------------------------------- per channel ---- #

_WIN_LABEL = {"artifact": "stim artifact", "evoked": "evoked response"}


def _lfp_figs(tm, y, base, name, title, wins, sem=None, color=None,
              overlay=None) -> dict:
    """The LFP trace at each window -> {window_key: png path}. Both windows are
    of the SAME LFP (mean_trace): 'artifact' and 'evoked'. Averaged figures pass
    *sem* for a shaded mean +/- SEM band; *color* tints the line+band (the daily
    averages are drawn in their day colour). *overlay* is a list of the
    constituent traces (on the same grid as *y*) drawn thin+transparent under the
    mean, so the spread of the averaged recordings is visible."""
    out = {}
    for wkey, win in wins.items():
        ct, cy = _crop(tm, y, win)
        cs = _crop(tm, sem, win)[1] if sem is not None else None
        ov = ([_crop(tm, o, win)[1] for o in overlay] if overlay else None)
        pth = os.path.join(base, f"{name}_{wkey}.png")
        lbl = f"{title} · {_WIN_LABEL[wkey]} ({win[0]:g} to {win[1]:g} ms)"
        if _fig.plot_stim_trace(ct, cy, lbl, pth, ylabel="LFP amplitude", sem=cs,
                                color=color or _fig._ACCENT, overlay=ov):
            out[wkey] = pth
    return out


def _overlays(traces, day_mean, sdays, wins, day_color, base, tag,
              animal, channel) -> dict:
    """Per window, one figure overlaying all file traces (thin, day-coloured) +
    the daily-mean traces (bold) on top. Returns {window_key: png path}."""
    out = {}
    for wkey, win in wins.items():
        ftr = [(str(t["chunk_datetime"])[:10],
                *_crop(t["time_ms"], t["evoked_trace"], win)) for t in traces]
        dtr = [(d, *_crop(day_mean[d][0], day_mean[d][1], win)) for d in sdays]
        pth = os.path.join(base, f"{tag}_overlay_{wkey}.png")
        p = _fig.plot_day_overlay(
            ftr, dtr, f"{animal} {channel} — {_WIN_LABEL[wkey]} ({win[0]:g} to "
            f"{win[1]:g} ms) across the week: every recording (thin) + daily "
            f"means (bold)", pth, day_color)
        if p:
            out[wkey] = p
    return out


def _metric_figs(imp_rows, base, tag, animal, channel, week, start, end, p,
                 *, value_key, ylabel, series_name, metric_label, fname):
    """(full-history <metric> with THIS WEEK shaded, this-week zoom) for one
    per-channel impedance metric. *value_key* selects it: 'access_r_kohm' (the
    access-resistance Rₐ trend) or 'slow_ss_kohm' (the slow-phase steady-state
    impedance Z_ss trend). Both read the SAME ``imp_rows`` (each row carries
    both metrics), so Z_ss costs no extra DB read."""
    hl = (datetime.fromisoformat(start),
          datetime.fromisoformat(end) + timedelta(days=1))
    full = _fig.impedance_png(
        imp_rows, os.path.join(base, f"{tag}_{fname}.png"), highlight=hl,
        title=f"{animal} {channel} — {metric_label}, full implant history",
        pct=p.pct, window=p.window, min_history=p.min_history,
        value_key=value_key, ylabel=ylabel, series_name=series_name)
    lo, hi = start.replace("-", "_"), end.replace("-", "_")
    wk = [r for r in imp_rows if lo <= str(r.get("chunk_datetime", ""))[:10] <= hi]
    week_png = _fig.impedance_png(
        wk, os.path.join(base, f"{tag}_{fname}_week.png"),
        title=f"{animal} {channel} — {metric_label}, this week ({week})",
        pct=p.pct, window=p.window, min_history=p.min_history,
        value_key=value_key, ylabel=ylabel, series_name=series_name) if wk else None
    return full, week_png


# Cap on per-recording epochs overlaid on ONE figure: enough to show the spread
# without turning hundreds of thin lines into an opaque smear (or a slow render).
# Beyond this, epochs are evenly subsampled (logged, never silently dropped).
_MAX_EPOCH_OVERLAY = 150


def _recording_epochs(store, file_id, animal, channel) -> list[dict]:
    """The per-epoch evoked traces of ONE recording+channel as
    ``[{"evoked_trace": [...], "time_ms": [...]}, ...]`` -- read from that
    recording's ``*_evoked.mat`` (``read_file_evoked``) so a per-recording figure
    can overlay its constituent stimulus repetitions under the mean. ``[]`` when
    the .mat is missing/unreadable or carries no per-epoch ``evokedData`` (the
    caller then renders the DB mean alone).

    Reads only this animal's channel (``only_animals``) and keeps just the
    (sub-sampled) epochs -- the full ``evokedData`` (e.g. 1795 x 20001 ~= 287 MB)
    is released before the next recording, so the digest never accumulates one
    array per file (cf. the 2026-07 memory-leak incident)."""
    if not file_id:
        return []
    path = store.evoked_output_path_for_file(int(file_id))
    chans = (read_file_evoked(path, only_animals=[animal])
             if path and os.path.exists(path) else {})
    rec = chans.get(channel)
    if rec is None:                       # channel_name vs evokedOutput key drift
        want = split_animal_electrode(channel)
        for k, v in chans.items():
            if split_animal_electrode(k) == want:
                rec = v
                break
    if not rec:
        return []
    traces, tms = rec.get("traces"), rec.get("time_ms")
    if traces is None or tms is None or getattr(traces, "ndim", 0) != 2:
        return []
    tm_list = np.asarray(tms, dtype=float).tolist()
    n = int(traces.shape[0])
    if n > _MAX_EPOCH_OVERLAY:
        logger.info("overlay: subsampling %d epochs -> %d for %s %s (file %s)",
                    n, _MAX_EPOCH_OVERLAY, animal, channel, file_id)
        idx = np.linspace(0, n - 1, _MAX_EPOCH_OVERLAY).astype(int)
    else:
        idx = range(n)
    return [{"evoked_trace": np.asarray(traces[i], dtype=float).tolist(),
             "time_ms": tm_list} for i in idx]


def _perfile_figs(store, perfile_src, animal, channel, base, wins) -> list:
    """Per-recording LFP figures. Each figure overlays that ONE recording's own
    epochs (its stimulus repetitions, read from the ``*_evoked.mat``) thin +
    transparent under their rising-edge-aligned mean -- a per-recording mean is
    itself an average over epochs, so it gets its constituents like every other
    average. Falls back to the DB per-recording mean alone when the .mat is
    unavailable."""
    out = []
    for t in perfile_src:
        name = f"file_{_safe(t['chunk_datetime'])}"
        epochs = _recording_epochs(store, t.get("file_id"), animal, channel)
        etm = emean = esem = None
        eov = []
        if epochs:
            etm, emean, esem, eov = align_average_with_traces(
                epochs, value_key="evoked_trace", align="rising_edge")
        if etm is not None and eov:
            figs = _lfp_figs(
                etm, emean, base, name,
                f"{animal} {channel} — single recording {t['chunk_datetime']} "
                f"(mean + {len(eov)} epochs)", wins, sem=esem, overlay=eov)
        else:
            figs = _lfp_figs(
                t["time_ms"], t["evoked_trace"], base, name,
                f"{animal} {channel} — single recording {t['chunk_datetime']}",
                wins)
        out += list(figs.values())
    return out


def _hour_overlay(day_traces, day_mean_d, wins, base, tag, d,
                  animal, channel) -> list:
    """Per window: each HOUR's stimCopy-aligned average overlaid on the full-day
    mean, coloured early->late (a colorbar) so intra-day drift reads at a glance.
    Returns [png paths]; empty when the day has fewer than 2 hours to compare."""
    hours = _group_by_hour(day_traces)
    shours = sorted(hours)
    if len(shours) < 2:
        return []
    hmeans = {h: align_average_with_traces(
                  hours[h], value_key="evoked_trace",
                  align_key="stim_trace", align="rising_edge")
              for h in shours}
    n = len(shours)
    dtm, dy = day_mean_d[0], day_mean_d[1]
    out = []
    for wkey, win in wins.items():
        gtr = []
        for i, h in enumerate(shours):
            htm, hy = hmeans[h][0], hmeans[h][1]
            if htm is None:
                continue
            gtr.append((i / (n - 1), *_crop(htm, hy, win)))
        base_c = _crop(dtm, dy, win) if dtm else None
        pth = os.path.join(base, f"{tag}_hourly_{_safe(d)}_{wkey}.png")
        p = _fig.plot_gradient_overlay(
            gtr, base_c,
            f"{animal} {channel} — {_WIN_LABEL[wkey]} ({win[0]:g} to {win[1]:g} ms)"
            f" per hour on {d}: each hour vs the full-day mean", pth)
        if p:
            out.append(p)
    return out


_ZSS_SAT_RATIO = 0.3       # Z_ss < ratio * Rₐ = saturated (amp railed) -> drop
_ZSS_MIN_DAY_N = 3         # a day with fewer valid Z_ss rows gives a wild mean -> NaN


def _zss_valid_rows(imp_rows) -> list:
    """Rows with a NON-saturated Z_ss (Z_ss >= _ZSS_SAT_RATIO * Rₐ). At high
    charge the amp rails and the plateau collapses to Z_ss << Rₐ; those near-zero
    rows would blow up a %-change, so drop them."""
    out = []
    for r in imp_rows:
        z, ra = r.get("slow_ss_kohm"), r.get("access_r_kohm")
        if z is None:
            continue
        if ra and ra > 0 and z < _ZSS_SAT_RATIO * ra:
            continue
        out.append(r)
    return out


def _zss_pct_change(imp_rows) -> dict | None:
    """Per-hour mean Z_ss (valid rows) + % changes: hour-over-hour, vs the hour's
    DAY mean, and vs the WEEK mean. ``None`` when fewer than 2 hours of valid
    data. Keys: x (datetimes), hz, pct_hoh, pct_day, pct_week, week_mean."""
    rows = _zss_valid_rows(imp_rows)
    byh: dict = {}
    byd: dict = {}
    for r in rows:
        cd = str(r["chunk_datetime"])
        byh.setdefault(cd[:14], []).append(r["slow_ss_kohm"])
        byd.setdefault(cd[:10], []).append(r["slow_ss_kohm"])
    hours = sorted(byh)
    if len(hours) < 2:
        return None
    hz = [float(np.mean(byh[h])) for h in hours]
    day_mean = {d: float(np.mean(v)) for d, v in byd.items()}
    week_mean = float(np.mean([r["slow_ss_kohm"] for r in rows]))

    def _pct(a, b):
        return (a - b) / b * 100.0 if b else None
    pct_hoh = [None] + [_pct(hz[i], hz[i - 1]) for i in range(1, len(hz))]
    pct_day = [_pct(hz[i], day_mean[hours[i][:10]]) for i in range(len(hz))]
    pct_week = [_pct(z, week_mean) for z in hz]
    x = []
    for h in hours:
        try:
            x.append(datetime.strptime(h, "%Y_%m_%d__%H"))
        except ValueError:
            x.append(None)
    # per-DAY series (far less noisy than hourly): daily mean Z_ss + day-over-day %
    # change. A day with < _ZSS_MIN_DAY_N valid rows is None (a 1-2 sample day gives
    # a wild mean -- the source of the biggest %-swings in the hourly view).
    days = sorted(byd)
    dz = [float(np.mean(byd[d])) if len(byd[d]) >= _ZSS_MIN_DAY_N else None
          for d in days]
    pct_dod = [None]
    for i in range(1, len(dz)):
        prev = next((dz[j] for j in range(i - 1, -1, -1) if dz[j] is not None), None)
        pct_dod.append(_pct(dz[i], prev) if (dz[i] is not None and prev) else None)
    xday = []
    for d in days:
        try:
            xday.append(datetime.strptime(d, "%Y_%m_%d").replace(hour=12))
        except ValueError:
            xday.append(None)
    return {"x": x, "hz": hz, "pct_hoh": pct_hoh, "pct_day": pct_day,
            "pct_week": pct_week, "week_mean": week_mean,
            "xday": xday, "dz": dz, "pct_dod": pct_dod}


def _zss_pct_change_fig(imp_rows, base, tag, animal, channel, week):
    """The Z_ss %-change figure (per-hour Z_ss + the three %Δ series). Returns
    the PNG path or None when there isn't >=2 hours of valid Z_ss."""
    d = _zss_pct_change(imp_rows)
    if d is None:
        return None
    return _fig.plot_zss_pct_change(
        d["x"], d["hz"], d["pct_hoh"], d["pct_day"], d["pct_week"],
        d["week_mean"], f"{animal} {channel} — Z_ss % change ({week})",
        os.path.join(base, f"{tag}_zss_pct.png"),
        xday=d.get("xday"), dz=d.get("dz"), pct_dod=d.get("pct_dod"))


def _process_channel(store, animal, channel, traces, imp_rows, p, work,
                     week, start, end) -> dict | None:
    """Render every PNG + scalars for one channel: two LFP windows per level
    (stim-artifact + evoked), a per-window daily overlay under the weekly average,
    the impedance trend (full + this-week), and the day-coloured LFP correlation."""
    if not traces:
        return None
    base = os.path.join(work, _safe(animal), _safe(channel))
    os.makedirs(base, exist_ok=True)
    tag = f"{_safe(animal)}_{_safe(channel)}"
    days = _group_by_day(traces)
    sdays = sorted(days)
    day_color = _fig.day_color_map(sdays)
    # Averages are aligned by the stimCopy command's rising edge (via align_key)
    # -- a sharp, unambiguous pulse -- NOT the LFP artifact's own edge, which is
    # biphasic and occasionally mis-detects the recovery rise, shifting a trace
    # ~0.2 ms. They carry their constituent traces so figures can overlay them.
    day_mean = {d: align_average_with_traces(
                    days[d], value_key="evoked_trace",
                    align_key="stim_trace", align="rising_edge")
                for d in sdays}
    wins = {"artifact": p.artifact_window, "evoked": p.evoked_window}
    perfile_src = (days[max(days)] if p.perfile_span == "last_day" and days
                   else traces)
    # Per file: 2 LFP figs, each overlaying that recording's own epochs.
    perfile = _perfile_figs(store, perfile_src, animal, channel, base, wins)
    daily = []
    for d in sdays:                                        # per day: avg + traces
        tm, mean, sem, dtraces = day_mean[d]
        daily += list(_lfp_figs(
            tm, mean, base, f"day_{_safe(d)}",
            f"{animal} {channel} — daily average LFP {d} "
            f"(mean + {len(dtraces)} recordings)", wins,
            sem=sem, color=day_color[d], overlay=dtraces).values())
        daily += _hour_overlay(days[d], day_mean[d], wins, base, tag, d,
                               animal, channel)
    tmw, yw, semw, wtraces = align_average_with_traces(   # weekly
        traces, value_key="evoked_trace", align_key="stim_trace",
        align="rising_edge")
    weekly = _lfp_figs(
        tmw, yw, base, f"{tag}_weekly",
        f"{animal} {channel} — weekly average LFP, {week} "
        f"(mean + {len(wtraces)} recordings)", wins, sem=semw, overlay=wtraces)
    over = _overlays(traces, day_mean, sdays, wins, day_color, base, tag,
                     animal, channel)
    imp, imp_week = _metric_figs(
        imp_rows, base, tag, animal, channel, week, start, end, p,
        value_key="access_r_kohm", ylabel="Rₐ (kΩ)", series_name="Rₐ",
        metric_label="electrode access resistance (Rₐ)", fname="impedance")
    zss, zss_week = _metric_figs(
        imp_rows, base, tag, animal, channel, week, start, end, p,
        value_key="slow_ss_kohm", ylabel="Z_ss (kΩ)", series_name="Z_ss",
        metric_label="slow-phase steady-state impedance (Z_ss)", fname="zss")
    # Slow-phase plateau VOLTAGE (V_ss, mV) -- the raw measurement behind Z_ss
    # (Z_ss = V_ss / commanded slow current), so a voltage change reads directly
    # instead of being folded through the current normalisation.
    vss, vss_week = _metric_figs(
        imp_rows, base, tag, animal, channel, week, start, end, p,
        value_key="v_ss_mv", ylabel="V_ss (mV)", series_name="V_ss",
        metric_label="slow-phase plateau voltage (V_ss)", fname="vss")
    zss_pct = _zss_pct_change_fig(imp_rows, base, tag, animal, channel, week)
    xs, ys, dys = _magnitude_pairs(traces, p.artifact_window, p.evoked_window)
    corr = _fig.plot_stim_vs_evoked(
        xs, ys, f"{animal} {channel} — stim-artifact vs evoked magnitude",
        os.path.join(base, f"{tag}_correlation.png"), days=dys,
        day_color=day_color, stim_win=p.artifact_window,
        evoked_win=tuple(p.evoked_window))
    stat = _drift_stat([r.get("access_r_kohm") for r in imp_rows
                        if r.get("access_r_kohm") is not None],
                       p.window, p.min_history)
    zss_stat = _drift_stat([r.get("slow_ss_kohm") for r in imp_rows
                            if r.get("slow_ss_kohm") is not None],
                           p.window, p.min_history)
    return {
        "animal": animal, "channel": channel, "n_files": len(traces),
        "ra": (stat["latest"] if stat else None),
        "ra_drift": (stat["drift_pct"] if stat else None),
        "zss": (zss_stat["latest"] if zss_stat else None),
        "zss_drift": (zss_stat["drift_pct"] if zss_stat else None),
        "corr": corr,
        "png_artifact": weekly.get("artifact"), "png_evoked": weekly.get("evoked"),
        "png_artifact_overlay": over.get("artifact"),
        "png_evoked_overlay": over.get("evoked"),
        "png_impedance": imp, "png_impedance_week": imp_week,
        "png_zss": zss, "png_zss_week": zss_week,
        "png_vss": vss, "png_vss_week": vss_week,
        "png_zss_pct": zss_pct,
        "png_corr": corr["png"],
        "perfile": perfile, "daily": daily,
        "urls": {}, "folder_link": "",
    }


# Sheet thumbnails = the two daily-overlay figures + Rₐ + Z_ss + correlation.
_SHEET_HEADLINE = (("artifact", "png_artifact_overlay"),
                   ("evoked", "png_evoked_overlay"),
                   ("impedance", "png_impedance"), ("zss", "png_zss"),
                   ("corr", "png_corr"))
# Email inlines, per window, the weekly average then its daily overlay beneath;
# then full-history Rₐ, this-week Rₐ, full-history Z_ss, this-week Z_ss, and the
# correlation.
_EMAIL_HEADLINE = ("png_artifact", "png_artifact_overlay",
                   "png_evoked", "png_evoked_overlay",
                   "png_impedance", "png_impedance_week",
                   "png_zss", "png_zss_week",
                   "png_vss", "png_vss_week", "png_zss_pct", "png_corr")


def _upload_channel(drive, week_folder_id, res, throttle) -> None:
    """Upload every PNG into a per-channel subfolder; record the sheet-thumbnail
    URLs + the folder link on *res*."""
    sub = _drive.ensure_folder(
        drive, f"{_safe(res['animal'])}_{_safe(res['channel'])}", week_folder_id)
    res["folder_link"] = sub.get("webViewLink", "")
    for key, field in _SHEET_HEADLINE:
        if res.get(field):
            res["urls"][key] = _drive.upload_png(
                drive, res[field], sub["id"], throttle_sec=throttle)["image_url"]
    archive = [res.get("png_artifact"), res.get("png_evoked"),
               res.get("png_impedance_week"), res.get("png_zss_week"),
               res.get("png_vss"), res.get("png_vss_week"),
               res.get("png_zss_pct")] \
        + res["daily"] + res["perfile"]                   # archived, not linked
    for path in archive:
        if path:
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
        "Z_ss kΩ": _num(res.get("zss"), ".3f"),
        "Z_ss Δ%": _num(res.get("zss_drift"), "+.0f"),
        "corr r": _num(c.get("r"), ".2f"), "corr p": _num(c.get("p"), ".1e"),
        "n files": res["n_files"],
        "LFP artifact": img(urls.get("artifact")),
        "LFP evoked": img(urls.get("evoked")),
        "Impedance": img(urls.get("impedance")),
        "Z_ss": img(urls.get("zss")),
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
        for field in _EMAIL_HEADLINE:
            path = r.get(field)
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


_SUMMARY = (
    "This report tracks whether stimulation is being delivered consistently and "
    "whether the electrode and the brain's response stay stable over time. Per "
    "stimulated channel it shows the stimulus artifact and the evoked LFP "
    "response (weekly average + day-by-day), the electrode's access resistance "
    "and slow-phase steady-state impedance, and whether the artifact and the "
    "response track together.")

_BATTERY_TIP = (
    "Review tip: right after every battery change, check the stimulus-artifact "
    "figures — a shift there is the earliest sign that delivery has changed.")

_CAPTIONS = {
    "png_artifact": "Weekly-average stimulus artifact on the LFP electrode "
    "(-1 to 2 ms after each pulse), mean +/- SEM over the week. A consistent "
    "shape and amplitude means stimulus delivery is stable.",
    "png_artifact_overlay": "The -1 to 2 ms artifact for every recording this "
    "week (thin) with each day's mean bold, one colour per day. Overlapping days "
    "= stable; a colour-ordered drift = change across the week.",
    "png_evoked": "Weekly-average evoked LFP response (2 to 50 ms after each "
    "pulse), mean +/- SEM -- the brain's response to the stimulus.",
    "png_evoked_overlay": "Evoked response (2 to 50 ms) for every recording this "
    "week, coloured by day with daily means bold -- within- and across-day "
    "variability at a glance.",
    "png_impedance": "Electrode access resistance (Ra) over the whole implant "
    "history -- a proxy for electrode/tissue health. Dashed = rolling baseline, "
    "blue band = +/- threshold, orange = the week in this report. A large "
    "sustained rise can indicate encapsulation.",
    "png_impedance_week": "The same access resistance zoomed to this week (the "
    "orange region highlighted above).",
    "png_zss": "Slow-phase steady-state impedance (Z_ss) over the whole implant "
    "history -- the settled plateau of the long recharge phase, after the pulse "
    "transients have died out, divided by the commanded slow current. A "
    "stim-consistency signal complementary to access resistance: same dashed "
    "baseline + threshold band, orange = the week in this report.",
    "png_zss_week": "The same slow-phase steady-state impedance (Z_ss) zoomed to "
    "this week.",
    "png_vss": "Slow-phase plateau VOLTAGE (V_ss, mV) over the whole implant "
    "history -- the raw settled voltage behind Z_ss (Z_ss = V_ss / commanded slow "
    "current). Because the current is constant while the charge is unchanged, a "
    "leap here IS a real voltage change, not a current/impedance artifact; if the "
    "charge ever changes, V_ss and Z_ss diverge.",
    "png_vss_week": "The same slow-phase plateau voltage (V_ss) zoomed to this "
    "week.",
    "png_zss_pct": "Z_ss % change by hour: per-hour mean Z_ss (top, with the week "
    "mean dashed) and its % change hour-over-hour, vs that hour's full-day mean, "
    "and vs the full-week mean (bottom). Saturated (railed) rows are dropped so "
    "the % change tracks the real plateau, not the collapses.",
    "png_corr": "Does a bigger stimulus artifact drive a bigger evoked response? "
    "One point per recording (coloured by day); the red line is the linear fit "
    "(r, p). A strong positive r means the response scales with the delivered "
    "stimulus.",
}


def _email_html(today, span, results, folder_link, sheet_id, dropped=0,
                label="weekly", daily=False) -> str:
    intro = escape(_SUMMARY) + (
        " <b>This daily digest covers the previous full day's per-recording "
        "figures (attached + archived to Drive) plus a running 7-day week.</b>"
        if daily else "")
    parts = [
        '<html><body style="font-family:-apple-system,sans-serif;color:#1a1a2e">',
        f'<h2>{label.capitalize()} stimulus stability — {span} '
        f'({today.isoformat()})</h2>',
        f'<p style="color:#333;max-width:72ch">{intro}</p>',
        '<p style="border-left:3px solid #ff9f0a;padding:6px 10px;'
        f'background:#fff8ec;color:#7a5b00;max-width:72ch">{escape(_BATTERY_TIP)}'
        '</p>',
        f'<p style="color:#6c6c80">{len(results)} stimulated channel(s). '
        'Ra = electrode access resistance; Z_ss = slow-phase steady-state '
        'impedance (settled recharge-phase plateau); corr r = '
        'artifact-vs-response correlation.</p>',
        '<table cellpadding="6" style="border-collapse:collapse;'
        'border:1px solid #ddd"><tr style="background:#f4f6fb">'
        '<th>Channel</th><th>Rₐ kΩ</th><th>Rₐ Δ%</th>'
        '<th>Z_ss kΩ</th><th>Z_ss Δ%</th><th>corr r</th>'
        '<th>corr p</th><th>n</th></tr>',
    ]
    for r in results:
        drift = r["ra_drift"]
        col = ("#c0392b" if isinstance(drift, (int, float)) and abs(drift) >= 40
               else "#1a1a2e")
        zdrift = r.get("zss_drift")
        zcol = ("#c0392b" if isinstance(zdrift, (int, float)) and abs(zdrift) >= 40
                else "#1a1a2e")
        parts.append(
            f'<tr><td>{escape(r["animal"])} {escape(r["channel"])}</td>'
            f'<td>{_num(r["ra"], ".1f")}</td>'
            f'<td style="color:{col}">{_num(drift, "+.0f")}</td>'
            f'<td>{_num(r.get("zss"), ".3f")}</td>'
            f'<td style="color:{zcol}">{_num(zdrift, "+.0f")}</td>'
            f'<td>{_num(r["corr"].get("r"), ".2f")}</td>'
            f'<td>{_num(r["corr"].get("p"), ".1e")}</td>'
            f'<td>{r["n_files"]}</td></tr>')
    parts.append('</table>')
    for r in results:                                     # figures + captions
        parts.append(f'<h3>{escape(r["animal"])} {escape(r["channel"])}</h3>')
        for field in _EMAIL_HEADLINE:
            path = r.get(field)
            if not path:
                continue
            parts.append(f'<img src="cid:{os.path.basename(path)}" '
                         'style="max-width:640px;display:block;margin:10px 0 2px">')
            cap = _CAPTIONS.get(field)
            if cap:
                parts.append('<p style="color:#6c6c80;font-size:12px;'
                             f'margin:0 0 16px;max-width:66ch"><em>{escape(cap)}'
                             '</em></p>')
    if sheet_id:
        parts.append(f'<p><a href="https://docs.google.com/spreadsheets/d/'
                     f'{sheet_id}">Open the Stim Stability sheet</a></p>')
    if folder_link:
        parts.append(f'<p><a href="{folder_link}">Open the Drive '
                     'folder (all figures)</a></p>')
    if dropped:
        parts.append(f'<p style="color:#6c6c80"><small>{dropped} more per-file/'
                     'daily figure(s) omitted to keep this email small — see the '
                     'Drive folder for the full set.</small></p>')
    parts.append('<hr><small style="color:#6c6c80">QC Monitor stimulus-stability '
                 f'{label}. Edit config.yaml → notifications.stim_stability_{label}.'
                 '</small></body></html>')
    return "".join(parts)


def _email_text(today, span, results, label="weekly") -> str:
    lines = [f"{label.capitalize()} stimulus stability -- {span} "
             f"({today.isoformat()})", "",
             _SUMMARY, "", _BATTERY_TIP, "",
             f"{len(results)} stimulated channel(s):"]
    for r in results:
        lines.append(f"  {r['animal']} {r['channel']}: Ra={_num(r['ra'], '.1f')} kΩ "
                     f"Δ={_num(r['ra_drift'], '+.0f')}% "
                     f"Z_ss={_num(r.get('zss'), '.3f')} kΩ "
                     f"Δ={_num(r.get('zss_drift'), '+.0f')}% "
                     f"corr r={_num(r['corr'].get('r'), '.2f')} "
                     f"p={_num(r['corr'].get('p'), '.1e')} n={r['n_files']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- entry --- #

def send_stim_stability_weekly(today: date, config: dict, store: Store,
                               emailer: EmailAlerter, *, dry_run: bool = False,
                               out_dir: str | None = None,
                               period: str = "weekly") -> dict:
    """Build and dispatch the stimulus-stability report.

    *period* ``'weekly'`` (default) or ``'daily'``: the daily digest renders the
    SAME figures but reads the ``stim_stability_daily`` config, is EMAIL-ONLY (no
    Drive/Sheet, to keep the daily cadence light), and labels itself 'daily'. The
    caller passes YESTERDAY as *today* so it covers the previous full day's
    per-file (per-hour) figures + the running 7-day week ending then.

    *dry_run* renders PNGs into *out_dir* (or a temp dir) and returns the
    would-be rows/summary WITHOUT uploading to Drive, writing the sheet, or
    emailing."""
    assert isinstance(today, date), "today must be a date"
    assert isinstance(config, dict), "config must be a dict"
    assert period in ("weekly", "daily"), "period must be weekly|daily"
    daily = (period == "daily")
    notif = (config.get("notifications", {}) or {})
    weekly_cfg = notif.get("stim_stability_weekly", {}) or {}
    cfg = (notif.get("stim_stability_daily", {}) or {}) if daily else weekly_cfg
    if not dry_run and not cfg.get("enabled", False):
        return {"sent": False, "reason": "disabled"}
    recipients = list(cfg.get("recipients")
                      or (weekly_cfg.get("recipients") if daily else None)
                      or ((config.get("alerting", {}) or {}).get("smtp", {})
                          .get("recipients", []) or []))
    if not dry_run and not recipients:
        return {"sent": False, "reason": "no recipients"}

    # Daily inherits the weekly section's rendering params (same figures); its
    # own keys override. Daily figures ARE archived to Drive (into a Daily/<date>
    # subfolder), but daily does NOT write the per-week Sheet.
    p = _resolve_params({**weekly_cfg, **cfg} if daily else cfg, config)
    if daily:
        p.write_sheet = False
    label = "daily" if daily else "weekly"
    start_date, end_date, week = _week_span(today)
    span_label = end_date if daily else week
    channels = [c for c in sorted(store.active_impedance_channel_keys())
                if not _excluded(c[0], p.exclude)]
    imp_series = store.impedance_series_by_channel(exclude=p.exclude)
    work = out_dir or tempfile.mkdtemp(prefix="stimstab_")

    drive = week_folder = None
    if not dry_run and p.folder_id:
        try:
            drive = _drive.drive_from_config(p.drive_auth, sa_path=p.sa_path,
                                             token_file=p.token_file)
            if daily:
                # Nest daily figures under a "Daily" parent so per-date folders
                # don't clutter the top-level folder alongside the weekly ones.
                parent = _drive.ensure_folder(drive, "Daily", p.folder_id)
                week_folder = _drive.ensure_folder(
                    drive, span_label, parent["id"])
            else:
                week_folder = _drive.ensure_folder(drive, week, p.folder_id)
        except Exception as e:                            # noqa: BLE001
            logger.warning("Drive unavailable (%s), emailing figures only: %s",
                           p.drive_auth, e)
            drive = week_folder = None

    rows, results = [], []
    for animal, channel in channels:
        traces = store.channel_stim_evoked_traces_in_range(
            animal, channel, start_date, end_date)
        res = _process_channel(store, animal, channel, traces,
                               imp_series.get((animal, channel), []), p, work,
                               week, start_date, end_date)
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
    if rows and not dry_run and p.write_sheet and p.sheet_id and p.sa_path:
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
            subject=f"QC {label} stim stability -- {span_label} "
                    f"({len(results)} channel{'s' if len(results) != 1 else ''})",
            body=_email_text(today, span_label, results, label),
            body_html=_email_html(today, span_label, results, folder_link,
                                  p.sheet_id, dropped=dropped, label=label,
                                  daily=daily),
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
