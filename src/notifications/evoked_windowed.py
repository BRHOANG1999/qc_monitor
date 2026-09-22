"""Windowed, FULL-DENSITY evoked-response figure package.

Extends the evoked package along two axes the operator asked for:

1. MULTIPLE ANALYSIS WINDOWS (2-50, 2-100, 2-500, 2-1000 ms). The window is not a
   display crop -- every feature is RE-MEASURED over each window (line-length, RMS,
   peak, ... computed on that segment), so a response's decile changes per window.
   Features are recomputed in-memory from the raw traces (one trace read per file,
   reused across all windows); nothing is filtered or decimated.

2. FULL-DENSITY grids. The per-decile grid draws EVERY trace, not a 30-sample, via
   the streaming rasterizer in ``trace_density`` (fixed memory, individual lines,
   lone outliers still visible above the smear).

Memory/I-O shape: ONE values pass reads+recomputes every file once (the costly
step) and fans each response's windowed feature values to its day + the week; then
a per-period trace pass re-reads that period's traces (no recompute) to accumulate
Welford mean/sd + a per-band density buffer, renders, and frees the buffers -- so
peak memory is one period's buffers, independent of the trace count.

CLI: ``python -m src.notifications.evoked_windowed --animal BCH111 --date 2026-09-20``
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from src.notifications import evoked_digest as _ed       # noqa: E402
from src.notifications import evoked_package as _pkg      # noqa: E402
from src.notifications.trace_density import LineDensity, robust_ylim  # noqa: E402
from src.evoked_figures import data as _d                # noqa: E402
from src.periictal import passive as _passive            # noqa: E402
from src.utils import evoked_features as _ef              # noqa: E402
from src.utils.evoked_output import read_feature_sidecar, read_file_evoked  # noqa: E402

# (start, end) ms. All start at 2 ms (just past the 1 ms artifact guard), so the
# crop always keeps >=2 samples -- no silent full-trace fallback. Wide windows are
# simply capped by the file's data extent (~+/-200 or +/-500 ms).
WINDOWS_MS = [(2.0, 50.0), (2.0, 100.0), (2.0, 500.0)]

_FD_W, _FD_H = 620, 360             # density raster (pixels); memory ~ W*H per band
_A0 = 0.30                          # per-trace opacity the density reproduces


def _win_tag(w) -> str:
    return f"{int(round(w[0]))}-{int(round(w[1]))}ms"


def _win_cfg(w):
    """A crop-only FeatureConfig for window *w* (no filtering; artifact guard 1 ms
    keeps start>=1, so a 2 ms start is unchanged)."""
    return _passive.window_config(float(w[0]), float(w[1]))


# --------------------------------------------------------------------- #
#  values pass: windowed feature values per response, fanned to periods
# --------------------------------------------------------------------- #
def _fs_of(time_ms) -> float:
    d = np.diff(np.asarray(time_ms, dtype=float))
    return 1000.0 / float(np.mean(d)) if d.size else 20000.0


def windowed_values(files, animal, channel, features, windows, week_key,
                    progress=None) -> tuple[dict, dict]:
    """ONE trace pass. For each response, recompute *features* over each window and
    fan (timestamp, value) into its DAY period and the WEEK. Also track, per window,
    the robust amplitude extent (for a shared y-limit). Returns
    ``(series[wtag][period][feat] = {"secs","metrics"}, ylim[wtag] = (y0,y1))``."""
    cfgs = {_win_tag(w): _win_cfg(w) for w in windows}
    # The CWT bank dominates compute (~15 s/file); skip it unless a selected
    # feature actually needs the wavelet columns (the curated set does not).
    want_wavelet = any(f in _ef.WAVELET_COLUMNS for f in features)
    # raw accumulators: secs/vals per (wtag, period, feat)
    secs: dict = defaultdict(lambda: defaultdict(list))          # [wtag][period] -> [ts...]
    vals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    extent: dict = {wt: [] for wt in cfgs}                       # [wtag] -> [(p1,p99)...]
    n_files = len(files)
    for i, fp in enumerate(files):
        if progress and i % 5 == 0:
            progress(f"windowed values… ({i}/{n_files})")
        rows = [r for r in (read_feature_sidecar(fp, animal) or [])
                if r.get("channel") == channel]
        if not rows:
            continue
        ev = read_file_evoked(fp, only_animals=[animal]).get(channel)
        if not ev or ev.get("traces") is None:
            continue
        traces = np.asarray(ev["traces"], dtype=float)
        time_ms = np.asarray(ev["time_ms"], dtype=float)
        st_times = np.asarray(ev["times"], dtype=float)
        fs = _fs_of(time_ms)
        avg = _ef.trial_moving_average(traces)
        # timestamp per epoch, matched to a sidecar row by stim time.
        st2dt = {round(float(_d._nan(r.get("stim_time_sec"))), 4):
                 (_d.parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None)
                 for r in rows if np.isfinite(_d._nan(r.get("stim_time_sec")))}
        dt_by_epoch = [st2dt.get(round(float(s), 4)) for s in st_times]
        for wt, cfg in cfgs.items():
            feats = _ef.compute_all(avg, time_ms, fs, expensive=False, cfg=cfg,
                                    include_wavelet=want_wavelet)
            for j, dt in enumerate(dt_by_epoch):
                if dt is None:
                    continue
                ts = dt.timestamp()
                dk = dt.date().isoformat()
                for period in (dk, week_key):
                    secs[wt][period].append(ts)
                    for feat in features:
                        arr = feats.get(feat)
                        vals[wt][period][feat].append(
                            float(arr[j]) if arr is not None
                            and np.isfinite(arr[j]) else np.nan)
            extent[wt].append(_win_amp_extent(avg, time_ms, cfg))
    return _finalize_values(secs, vals, features), _finalize_ylim(extent)


def _win_amp_extent(traces, time_ms, cfg):
    """(p1, p99) baseline-corrected amplitude within *cfg*'s window across a file's
    epochs -- a robust vertical extent for the density y-limit."""
    t = np.asarray(time_ms, dtype=float)
    ws = cfg.window_start_ms if cfg.window_start_ms is not None else t[0]
    we = cfg.window_end_ms if cfg.window_end_ms is not None else t[-1]
    m = (t >= ws) & (t <= we)
    base = (t >= -50) & (t <= -5)
    if m.sum() < 2:
        return (np.nan, np.nan)
    seg = traces[:, m]
    if base.any():
        seg = seg - np.nanmean(traces[:, base], axis=1, keepdims=True)
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(seg, 1.0)), float(np.percentile(seg, 99.0)))


def _finalize_values(secs, vals, features) -> dict:
    out: dict = {}
    for wt in secs:
        out[wt] = {}
        for period in secs[wt]:
            s = np.asarray(secs[wt][period], dtype=float)
            order = np.argsort(s, kind="stable")
            out[wt][period] = {
                "secs": s[order],
                "metrics": {f: np.asarray(vals[wt][period][f], dtype=float)[order]
                            for f in features}}
    return out


def _finalize_ylim(extent) -> dict:
    out = {}
    for wt, pairs in extent.items():
        arr = np.asarray(pairs, dtype=float)
        out[wt] = robust_ylim(arr) if arr.size else (-1.0, 1.0)
    return out


# --------------------------------------------------------------------- #
#  per-period render pass: Welford mean/sd + full-density per band
# --------------------------------------------------------------------- #
class _Accum:
    """DAY strategy: per (window, feature) Welford mean/sd (all traces) + a
    ``LineDensity`` per band (all traces, streamed -> full-density overlay)."""

    def __init__(self, n_bands, xlim, ylim):
        self.mean = _ed._BandAccum(n_bands)
        self.dens = [LineDensity(xlim, ylim, _FD_W, _FD_H) for _ in range(n_bands)]

    def add(self, band, trace, time_ms, tdec_x, tdec_y):
        self.mean.add(band, trace, time_ms)
        self.dens[band].add(tdec_x, tdec_y)


class _RunMean:
    """Streaming mean of a fixed-length vector (one day's mean trace in a band)."""

    def __init__(self):
        self.sum = None
        self.n = 0

    def add(self, v):
        v = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(v)):
            return
        if self.sum is None:
            self.sum = np.zeros(v.size)
        if v.size != self.sum.size:
            return
        self.sum += v
        self.n += 1

    def mean(self):
        return self.sum / self.n if self.n else None


class _WeekAccum:
    """WEEK strategy (NOT an all-trace overlay): per band, the MEAN trace of each
    DAY -- so the week grid shows day-to-day drift within each decile -- plus the
    week Welford mean/sd for the bands_sd figure."""

    def __init__(self, n_bands):
        self.mean = _ed._BandAccum(n_bands)
        self.days = [defaultdict(_RunMean) for _ in range(n_bands)]

    def add(self, band, day, trace, time_ms):
        self.mean.add(band, trace, time_ms)
        if day is not None:
            self.days[band][day].add(trace)


def _blank_baseline(trace, time_ms):
    """Baseline-correct (mean over -50..-5 ms) and NaN the |t|<=1.5 ms artifact --
    the same convention as the sampled overlays, so density and mean align."""
    t = np.asarray(time_ms, dtype=float)
    y = np.asarray(trace, dtype=float).copy()
    base = (t >= -50) & (t <= -5)
    if base.any():
        y = y - np.nanmean(y[base])
    y[np.abs(t) <= 1.5] = np.nan
    return y


def _render_period(files, animal, channel, period, wtags_windows, info_pw, ylim,
                   labels, pkg, is_week=False, progress=None) -> list:
    """One trace pass over *period*'s files for ALL windows. DAY periods accumulate
    a full-density overlay per band; the WEEK accumulates a per-day MEAN per band
    (a different strategy). Render, free. Returns figure records."""
    accs: dict = {}
    for wt, w in wtags_windows:
        for feat, info in info_pw[wt].items():
            nb = info["n_bands"] + 1
            accs[(wt, feat)] = (_WeekAccum(nb) if is_week
                                else _Accum(nb, (w[0], w[1]), ylim[wt]))
    maps = {(wt, feat): info["band_of_ts"]
            for wt, _w in wtags_windows for feat, info in info_pw[wt].items()}
    for fp in files:
        rows = [r for r in (read_feature_sidecar(fp, animal) or [])
                if r.get("channel") == channel]
        if not rows:
            continue
        ev = read_file_evoked(fp, only_animals=[animal]).get(channel)
        if not ev or ev.get("traces") is None:
            continue
        times = np.asarray(ev["times"], dtype=float)
        traces, time_ms = ev["traces"], np.asarray(ev["time_ms"], dtype=float)
        order = np.argsort(times)
        ts_sorted = times[order]
        for r in rows:
            st = _d._nan(r.get("stim_time_sec"))
            if not np.isfinite(st):
                continue
            j = int(np.searchsorted(ts_sorted, st))
            cand = [k for k in (j - 1, j) if 0 <= k < ts_sorted.size]
            if not cand:
                continue
            k = min(cand, key=lambda k: abs(ts_sorted[k] - st))
            if abs(ts_sorted[k] - st) > _ed._MATCH_TOL_SEC:
                continue
            trace = traces[order[k]]
            key = _ed._ts_key(r)
            dt = _d.parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None
            day = dt.date().isoformat() if dt else None
            y = None if is_week else _blank_baseline(trace, time_ms)
            for (wt, feat), m in maps.items():
                b = m.get(key, -1)
                if b < 0:
                    continue
                if is_week:
                    accs[(wt, feat)].add(b, day, trace, time_ms)
                else:
                    accs[(wt, feat)].add(b, trace, time_ms, time_ms, y)
    # render
    figs = []
    for wt, w in wtags_windows:
        for feat, info in info_pw[wt].items():
            acc = accs[(wt, feat)]
            band_wf = acc.mean.result()
            if not band_wf:
                continue
            fdir = os.path.join(pkg, wt, _pkg._safe(period), feat)
            os.makedirs(fdir, exist_ok=True)
            tm = acc.mean.time_ms
            meta = {"animal": animal, "channel": channel,
                    "date": labels.get(period, period), "window": wt}
            if progress:
                progress(f"{wt} · {period} · {feat}")
            _pkg._fig_bands_sd(feat, meta, info, band_wf, tm,
                               os.path.join(fdir, "bands_sd.png"))
            if is_week:
                grid = "deciles_grid_daymeans.png"
                _fig_grid_week(feat, meta, info, acc, band_wf, tm, (w[0], w[1]),
                               ylim[wt], os.path.join(fdir, grid))
            else:
                grid = "deciles_grid_fulldensity.png"
                _fig_grid_fd(feat, meta, info, band_wf, acc.dens, tm,
                             os.path.join(fdir, grid))
            figs.append({"period": period, "window": wt, "feature": feat,
                         "dir": os.path.relpath(fdir, pkg).replace("\\", "/"),
                         "grid": grid, "n_zero": info.get("n_zero", 0)})
    return figs


# --------------------------------------------------------------------- #
#  full-density grid figure
# --------------------------------------------------------------------- #
def _fd_panel(ax, dens: LineDensity, band_wf, b, tm, color, title, ylim) -> None:
    ax.set_facecolor(_ed._PANEL)
    d = dens.dens[b] if hasattr(dens, "dens") else dens
    n = d.n
    ax.imshow(d.rgba(color, a0=_A0), extent=d.extent(), origin="upper",
              aspect="auto", interpolation="nearest", zorder=1)
    wf = band_wf.get(b)
    if wf:
        mean = _ed._baseline(wf["mean"], tm)
        mp = mean.copy(); mp[np.abs(tm) <= 1.5] = np.nan
        ax.plot(tm, mp, color="#f0f0f5", lw=1.6, zorder=3)
    ax.set_xlim(d.x0, d.x1)
    ax.set_ylim(ylim)
    ax.set_title(f"{title}  (n={n})", color=_ed._TEXT, fontsize=9, loc="left")


def _fig_grid_week(feat, meta, info, wacc, band_wf, tm, xlim, ylim, out) -> str:
    """WEEK grid (a different strategy than the all-trace overlay): per decile, the
    MEAN trace of each DAY (turbo ramp early->late) + the week mean in white -- so
    day-to-day drift within each band reads at a glance."""
    n_bands, edges = info["n_bands"], info["edges"]
    days = sorted({d for b in range(n_bands) for d in wacc.days[b]})
    cmap = plt.cm.turbo(np.linspace(0.10, 0.92, max(1, len(days))))
    dcol = {d: cmap[i] for i, d in enumerate(days)}
    art = np.abs(tm) <= 1.5
    ncol = 5
    nrow = int(np.ceil(n_bands / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    seen: dict = {}
    for b in range(n_bands):
        ax = axes[b // ncol][b % ncol]
        ax.set_facecolor(_ed._PANEL)
        for d in days:
            rm = wacc.days[b].get(d)
            if not rm or rm.n < 3:
                continue
            m = _ed._baseline(rm.mean(), tm).copy()
            m[art] = np.nan
            seen[d], = ax.plot(tm, m, color=dcol[d], lw=1.0, alpha=0.9)
        wf = band_wf.get(b)
        if wf:
            wm = _ed._baseline(wf["mean"], tm).copy()
            wm[art] = np.nan
            seen["week"], = ax.plot(tm, wm, color="#f0f0f5", lw=2.2, zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(_ed._band_label(b, edges, n_bands), color=_ed._TEXT,
                     fontsize=9, loc="left")
    for jj in range(n_bands, nrow * ncol):
        axes[jj // ncol][jj % ncol].axis("off")
    order = [d for d in days if d in seen] + (["week"] if "week" in seen else [])
    if order:
        leg = fig.legend([seen[k] for k in order],
                         [("week" if k == "week" else k[5:]) for k in order],
                         loc="upper right", ncol=min(8, len(order)), fontsize=8,
                         frameon=False)
        for txt in leg.get_texts():
            txt.set_color(_ed._TEXT)
    fig.suptitle(f"{meta['animal']} · {meta['date']} · {meta['window']} · "
                 f"{_ed._pretty(feat)} · per-decile DAILY MEANS across the week",
                 color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat)


def _fig_grid_fd(feat, meta, info, band_wf, dens_list, tm, out) -> str:
    n_bands, edges = info["n_bands"], info["edges"]
    ylim = (dens_list[0].y0, dens_list[0].y1)
    # bright decile ramp (turbo, clipped off the near-black ends) -> reads on dark.
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, n_bands))[:, :3]
    ncol = 5
    nrow = int(np.ceil(n_bands / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    for b in range(n_bands):
        ax = axes[b // ncol][b % ncol]
        _fd_panel(ax, dens_list[b], band_wf, b, tm, tuple(colors[b]),
                  _ed._band_label(b, edges, n_bands), ylim)
    if band_wf.get(n_bands):                    # the zero/pile band, if any
        pass
    for jj in range(n_bands, nrow * ncol):
        axes[jj // ncol][jj % ncol].axis("off")
    fig.suptitle(f"{meta['animal']} · {meta['date']} · {meta['window']} · "
                 f"{_ed._pretty(feat)} · per-decile FULL-DENSITY (every trace) "
                 f"+ mean", color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat)


# --------------------------------------------------------------------- #
#  package assembly
# --------------------------------------------------------------------- #
def _fig_metric_trends(feat, per_win, meta, out) -> str:
    """Track a feature OVER TIME: per window, the per-day median + IQR band (light
    raw points behind), days-primary x-axis with dates. Built from the windowed
    values -- no extra trace reads."""
    import matplotlib.dates as mdates
    nwin = max(1, len(per_win))
    fig, axes = plt.subplots(1, nwin, figsize=(5.2 * nwin, 4.0),
                             facecolor=_ed._BG, squeeze=False)
    for i, (wt, secs, vals) in enumerate(per_win):
        ax = axes[0][i]
        ax.set_facecolor(_ed._PANEL)
        s = np.asarray(secs, dtype=float)
        v = np.asarray(vals, dtype=float)
        fin = np.isfinite(s) & np.isfinite(v)
        s, v = s[fin], v[fin]
        if s.size == 0:
            ax.set_title(f"{wt} — no data", color=_ed._MUTED, fontsize=10)
            continue
        dts = np.array([datetime.fromtimestamp(x) for x in s])
        ax.scatter(dts, v, s=3, c="#5e7ce2", alpha=0.10, linewidths=0)
        days = np.array([d.date() for d in dts])
        xs, md, lo, hi = [], [], [], []
        for d in sorted(set(days)):
            mask = days == d
            xs.append(datetime(d.year, d.month, d.day, 12))
            md.append(np.median(v[mask]))
            lo.append(np.percentile(v[mask], 25))
            hi.append(np.percentile(v[mask], 75))
        xs = np.array(xs)
        ax.fill_between(xs, lo, hi, color="#5e7ce2", alpha=0.25, linewidth=0,
                        label="IQR (p25–p75)")
        ax.plot(xs, md, color="#f0f0f5", lw=1.8, marker="o", ms=4,
                label="daily median")
        ax.set_title(wt, color=_ed._TEXT, fontsize=10, loc="left")
        ax.tick_params(colors=_ed._MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_ha("right")
        if i == 0:
            leg = ax.legend(fontsize=8, frameon=False)
            for t in leg.get_texts():
                t.set_color(_ed._TEXT)
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · {_ed._pretty(feat)} · "
                 f"metric OVER TIME (per-day median ± IQR, per window)",
                 color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat)


def build_windowed_package(animal, evoked_dir, end_day, out_root, *,
                           features=None, windows=None, n_bands=10,
                           channel_override=None, include_week=True, n_days=7,
                           progress=None) -> dict:
    """Windowed + full-density package ending *end_day*. Renders each of the last
    *n_days* days as a full-density (all-trace) grid, and (when *include_week*) the
    week as a per-day-means grid. ``n_days=1, include_week=False`` -> a single-day
    package (the daily email); the defaults -> the weekly package. Returns a
    manifest dict (also written as manifest.json + index.html)."""
    features = [f for f in (features or _pkg.PACKAGE_FEATURES) if f in _ef.ALL_COLUMNS]
    windows = windows or WINDOWS_MS
    wtags_windows = [(_win_tag(w), w) for w in windows]
    win_files, win_series = _pkg._period_series(animal, evoked_dir, end_day, n_days,
                                                features)
    if not win_files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    channel = _d.primary_channel(animal, win_series, features, channel_override)
    if channel is None or channel not in win_series:
        return {"animal": animal, "empty": True, "reason": "no channel data"}

    week_key = "week"
    days = [(end_day - timedelta(days=i)) for i in range(n_days - 1, -1, -1)]
    if progress:
        progress(f"recomputing {len(features)} features over {len(windows)} "
                 f"windows for {len(win_files)} recordings…")
    series, ylim = windowed_values(win_files, animal, channel, features, windows,
                                   week_key, progress)

    # band per (window, period, feature)
    info: dict = {}                                  # info[period][wtag][feat]
    week_label = _ed._date_label(end_day, n_days)
    labels = {week_key: week_label}
    for wt, _w in wtags_windows:
        for period, ser in series.get(wt, {}).items():
            info.setdefault(period, {}).setdefault(wt, {})
            for feat in features:
                bi = _ed.compute_bands(ser["secs"], ser["metrics"][feat], n_bands)
                if bi is not None:
                    info[period][wt][feat] = bi
    for d in days:
        dk = datetime(d.year, d.month, d.day).date().isoformat()
        labels.setdefault(dk, dk)

    cadence = "week" if include_week else "day"
    pkg = os.path.join(out_root, f"evoked_windowed_{cadence}_{animal}_"
                       f"{week_label}".replace(" ", "").replace("–", "_"))
    os.makedirs(pkg, exist_ok=True)

    # render period by period (bounded memory). week first (if any), then each day.
    order = ([week_key] if include_week else []) + \
        [datetime(d.year, d.month, d.day).date().isoformat()
         for d in reversed(days)]
    figures: list = []
    for period in order:
        info_pw = {wt: info.get(period, {}).get(wt, {}) for wt, _w in wtags_windows}
        info_pw = {wt: v for wt, v in info_pw.items() if v}
        if not info_pw:
            continue
        p_windows = [(wt, w) for wt, w in wtags_windows if wt in info_pw]
        pfiles = (win_files if period == week_key
                  else _ed.list_day_files(animal, evoked_dir,
                                          datetime.fromisoformat(period)))
        if progress:
            progress(f"rendering {period} ({len(pfiles)} recordings)…")
        figures += _render_period(pfiles, animal, channel, period, p_windows,
                                  info_pw, ylim, labels, pkg,
                                  is_week=(period == week_key), progress=progress)

    # Metrics-over-time trends for the primary period (built from the windowed
    # values -- cheap, no trace reads). One figure per feature (panel per window).
    primary = week_key if include_week else \
        datetime(end_day.year, end_day.month, end_day.day).date().isoformat()
    trends: dict = {}
    if progress:
        progress("building metric-over-time trends…")
    trend_dir = os.path.join(pkg, "_trends")
    os.makedirs(trend_dir, exist_ok=True)
    for feat in features:
        per_win = []
        for wt, _w in wtags_windows:
            ser = series.get(wt, {}).get(primary)
            if ser is not None:
                per_win.append((wt, ser["secs"], ser["metrics"][feat]))
        if not per_win:
            continue
        tp = os.path.join(trend_dir, f"{feat}.png")
        _fig_metric_trends(feat, per_win, {"animal": animal, "channel": channel},
                           tp)
        trends[feat] = os.path.relpath(tp, pkg).replace("\\", "/")

    manifest = {"animal": animal, "channel": channel, "week": labels[week_key],
                "windows": [wt for wt, _w in wtags_windows],
                "periods": order, "features": features, "primary": primary,
                "docs": {f: _ed.feature_doc(f) for f in features},
                "trends": trends, "figures": figures}
    with open(os.path.join(pkg, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    _write_index(pkg, manifest, labels)
    manifest["pkg_dir"] = pkg
    manifest["index"] = os.path.join(pkg, "index.html")
    return manifest


def _write_index(pkg, manifest, labels) -> None:
    figs = {(f["window"], f["period"], f["feature"]): f for f in manifest["figures"]}
    wins = manifest["windows"]
    feats = manifest["features"]
    periods = manifest["periods"]
    parts = ["<!doctype html><meta charset='utf-8'>",
             "<style>body{background:#16161f;color:#f0f0f5;font-family:sans-serif;"
             "margin:18px}h2{margin:6px 0}table{border-collapse:collapse;margin:10px 0 26px}"
             "td,th{border:1px solid #2c2c40;padding:5px 8px;font-size:12px;vertical-align:top}"
             "a{color:#7aa2ff}.na{color:#555}img{display:block;width:230px;border:1px solid #2c2c40}"
             ".def{color:#9a9ab0;font-size:11px;max-width:34ch}</style>",
             f"<h1>{html.escape(manifest['animal'])} · {html.escape(manifest['channel'])}"
             f" · windowed full-density · {html.escape(manifest['week'])}</h1>",
             "<p class='def'>Each feature is RE-MEASURED over each window (magnitudes "
             "&amp; deciles differ per window). Grids draw EVERY trace (full density); "
             "mean overlaid in white.</p>"]
    trends = manifest.get("trends") or {}
    if trends:
        parts.append("<h2>① Metrics over time (per-day median ± IQR)</h2>")
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:12px'>")
        for feat in feats:
            rel = trends.get(feat)
            if not rel:
                continue
            parts.append(f"<div><div class='def'>{html.escape(feat)}</div>"
                         f"<a href='{rel}'><img src='{rel}' "
                         f"style='width:360px'></a></div>")
        parts.append("</div>")
        parts.append("<h2>② Response “kinds” by decile</h2>")
    for wt in wins:
        parts.append(f"<h2>window {html.escape(wt)}</h2>")
        parts.append("<table><tr><th>period</th>"
                     + "".join(f"<th>{html.escape(f)}<div class='def'>"
                               f"{html.escape(manifest['docs'].get(f,''))}</div></th>"
                               for f in feats) + "</tr>")
        for p in periods:
            row = [f"<th style='text-align:left'>{html.escape(labels.get(p,p))}</th>"]
            for feat in feats:
                f = figs.get((wt, p, feat))
                if not f:
                    row.append("<td class='na'>—</td>")
                    continue
                d = f["dir"]
                grid = f.get("grid", "deciles_grid_fulldensity.png")
                row.append(
                    f"<td><a href='{d}/{grid}'>"
                    f"<img src='{d}/{grid}'></a>"
                    f"<a href='{d}/bands_sd.png'>mean±sd</a></td>")
            parts.append("<tr>" + "".join(row) + "</tr>")
        parts.append("</table>")
    with open(os.path.join(pkg, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


_ATTACH_CAP = 20 * 1024 * 1024        # 20 MB inline budget (mirrors the stim digest)


def _browse_url(manifest) -> str:
    """A clickable file:// link to the browsable package index (the daemon runs on
    the operator's own machine, so the local path opens for them)."""
    idx = os.path.abspath(manifest.get("index", ""))
    return "file:///" + idx.replace("\\", "/")


def _stage_email(manifest, pkg, wins, staged):
    """Copy the emailed figures to *staged* under UNIQUE names (so each inline
    image gets its own Content-ID -- the earlier collision made every grid render
    as the same picture). Returns (inline_paths, overflow_files, by_win, trends).
    Inlines the over-time trends first, then the primary period's decile grids,
    greedy under the 20 MB budget."""
    import shutil
    period = manifest.get("primary") or "week"
    budget = _ATTACH_CAP
    inline, files, trends = [], [], []
    for feat, rel in (manifest.get("trends") or {}).items():
        src = os.path.join(pkg, rel)
        if not os.path.isfile(src):
            continue
        cid = f"trend_{feat}.png"
        dst = os.path.join(staged, cid)
        shutil.copyfile(src, dst)
        sz = os.path.getsize(dst)
        if sz <= budget:
            inline.append(dst)
            budget -= sz
            trends.append((feat, cid))
    by_win: dict = {w: [] for w in wins}
    for f in manifest["figures"]:
        if f["period"] != period or f["window"] not in by_win:
            continue
        src = os.path.join(pkg, f["dir"], f.get("grid", "deciles_grid.png"))
        if not os.path.isfile(src):
            continue
        cid = f"grid_{f['window']}_{f['feature']}.png"
        dst = os.path.join(staged, cid)
        shutil.copyfile(src, dst)
        sz = os.path.getsize(dst)
        if sz <= budget:
            inline.append(dst)
            budget -= sz
            by_win[f["window"]].append((f["feature"], cid))
        else:
            files.append(src)
    return inline, files, by_win, trends


def _email_html(manifest, by_win, trends, cadence) -> str:
    period = manifest.get("primary") or "week"
    browse = _browse_url(manifest)
    kind = ("per-day MEANS across the week" if period == "week"
            else "FULL-DENSITY (every trace)")
    S = "font-family:sans-serif"
    parts = [
        f"<h2 style='{S}'>Evoked windowed {cadence} digest — "
        f"{html.escape(manifest['animal'])} · {html.escape(manifest['channel'])} · "
        f"{html.escape(manifest['week'])}</h2>",
        f"<p style='{S};color:#444'>Each feature is RE-MEASURED over each window "
        f"({', '.join(manifest['windows'])}) — magnitudes &amp; deciles differ per "
        f"window.</p>",
        f"<p style='{S};font-size:14px'>📂 <a href='{html.escape(browse)}'>"
        f"<b>Open the full browsable package (all windows × all periods)</b></a>"
        f"<br><span style='color:#888;font-size:12px'>{html.escape(browse)}</span>"
        f"</p>",
        f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
        f"① Metrics over time — per-day median ± IQR (this {period})</h3>"]
    for feat, cid in trends:
        parts.append(
            f"<div style='margin:4px 0 12px'><div style='{S};font-size:13px;"
            f"color:#333'><b>{html.escape(_ed._pretty(feat))}</b></div>"
            f"<img src='cid:{cid}' style='max-width:1100px;width:100%'></div>")
    parts.append(
        f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
        f"② Response “kinds” by decile — {kind}</h3>")
    for win, items in by_win.items():
        if not items:
            continue
        parts.append(f"<h4 style='{S};color:#5e7ce2'>window {html.escape(win)}</h4>")
        for feat, cid in items:
            parts.append(
                f"<div style='margin:4px 0 12px'><div style='{S};font-size:13px;"
                f"color:#333'>{html.escape(_ed._pretty(feat))}</div>"
                f"<img src='cid:{cid}' style='max-width:1100px;width:100%'></div>")
    parts.append(f"<p style='{S};font-size:12px;color:#666'>📂 Full browsable "
                 f"package: <a href='{html.escape(browse)}'>{html.escape(browse)}"
                 f"</a></p>")
    return "\n".join(parts)


def send_windowed_email(animal, evoked_dir, end_day, config, emailer, *,
                        cadence="weekly", out_root=None, features=None,
                        windows=None, email_windows=None, dry_run=False,
                        progress=None) -> dict:
    """Build the windowed package for *cadence* ('daily'|'weekly') and email it,
    organized: a clickable browse link, the metric-over-time trends, then the
    decile grids. Never raises for a data gap."""
    import tempfile
    out_root = out_root or os.path.join("data", "evoked_windowed_packages")
    os.makedirs(out_root, exist_ok=True)
    weekly = cadence == "weekly"
    manifest = build_windowed_package(
        animal, evoked_dir, end_day, out_root, features=features, windows=windows,
        include_week=weekly, n_days=7 if weekly else 1, progress=progress)
    if manifest.get("empty"):
        return {"sent": False, "reason": manifest.get("reason", "no data"),
                "animal": animal}
    # weekly grids are small day-means (inline all windows); daily grids are big
    # full-density (default to the narrow + mid window to stay under the cap).
    ew = email_windows or (manifest["windows"] if weekly
                           else ["2-50ms", "2-500ms"])
    staged = tempfile.mkdtemp(prefix="evoked_win_mail_")
    inline, files, by_win, trends = _stage_email(manifest, manifest["pkg_dir"],
                                                 ew, staged)
    if dry_run:
        return {"sent": False, "dry_run": True, "manifest": manifest,
                "inline": len(inline), "attach": len(files), "trends": len(trends)}
    recipients = (config.get("notifications", {}).get("evoked_windowed", {})
                  .get("recipients")
                  or (config.get("alerting", {}).get("smtp", {}) or {})
                  .get("recipients"))
    sent = bool(emailer.send(
        subject=f"QC evoked windowed {cadence} — {animal} · {manifest['week']}",
        body=f"Evoked windowed {cadence} digest for {animal}. Metrics-over-time + "
             f"decile grids in the HTML body; full package: {_browse_url(manifest)}",
        body_html=_email_html(manifest, by_win, trends, cadence),
        recipients=recipients, subject_prefix=False,
        attachments=inline, file_attachments=files))
    return {"sent": sent, "animal": animal, "cadence": cadence,
            "recipients": recipients, "index": manifest.get("index"),
            "inline": len(inline), "attach": len(files), "trends": len(trends)}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--date", default=None, help="week-ending day YYYY-MM-DD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--features", default=None, help="comma list (default: curated)")
    ap.add_argument("--windows", default=None,
                    help="comma list like 2-50,2-100 (ms); default: the four")
    ap.add_argument("--cadence", choices=["daily", "weekly"], default="weekly",
                    help="daily = single-day full-density; weekly = week day-means")
    ap.add_argument("--send", action="store_true",
                    help="also email the primary-period grids (else build only)")
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    end = datetime.fromisoformat(a.date) if a.date else datetime.now()
    feats = a.features.split(",") if a.features else None
    windows = ([tuple(float(x) for x in w.split("-")) for w in a.windows.split(",")]
               if a.windows else None)
    out = a.out or os.path.join("data", "evoked_windowed_packages")
    os.makedirs(out, exist_ok=True)
    prog = lambda m: print("  ", m)                              # noqa: E731
    if a.send:
        from src.alerting.email_alert import EmailAlerter
        res = send_windowed_email(a.animal, evoked_dir, end, cfg,
                                  EmailAlerter(cfg), cadence=a.cadence, out_root=out,
                                  features=feats, windows=windows, progress=prog)
    else:
        weekly = a.cadence == "weekly"
        res = build_windowed_package(end_day=end, animal=a.animal,
                                     evoked_dir=evoked_dir, out_root=out,
                                     features=feats, windows=windows,
                                     include_week=weekly, n_days=7 if weekly else 1,
                                     progress=prog)
    print("RESULT:", {k: v for k, v in res.items()
                      if k not in ("figures", "docs", "manifest")})
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
