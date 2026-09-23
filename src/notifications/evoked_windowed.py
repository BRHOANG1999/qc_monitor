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

# High-frequency band-power features (Σ PSD of the cropped window per response).
# Added as FULL features (trends + decile grids) alongside the curated set.
HF_BANDS = [("hf_500_1000", (500.0, 1000.0)), ("hf_1000_2000", (1000.0, 2000.0)),
            ("hf_2000_4000", (2000.0, 4000.0)), ("hf_4000_8000", (4000.0, 8000.0))]
HF_NAMES = [n for n, _ in HF_BANDS]
_HF_DOCS = {n: f"HF band power {int(lo)}–{int(hi)} Hz — Σ periodogram power over "
               f"the cropped analysis window." for n, (lo, hi) in HF_BANDS}

# Circadian: four 6 h bins over a 07:00-ANCHORED 24 h cycle (so a bin never splits
# across midnight). "day" = 07:00–19:00, "night" = 19:00–07:00.
_CIRC_LABELS = ["day-early (07–13)", "day-late (13–19)",
                "night-early (19–01)", "night-late (01–07)"]
_CIRC_COLORS = ["#f2c14e", "#e08a3c", "#6a8cff", "#3550a0"]   # warm=day, cool=night


def _circadian(dt):
    """(07:00-anchored cycle-date iso, bin 0..3) for a datetime. bin 0=day-early
    07–13, 1=day-late 13–19, 2=night-early 19–01, 3=night-late 01–07."""
    shifted = dt - timedelta(hours=7)
    return shifted.date().isoformat(), int(min(3, shifted.hour // 6))


def _pretty(feat: str) -> str:
    """Label for a feature, including the HF bands (which _ed._pretty doesn't know)."""
    if feat in _HF_DOCS:
        lo, hi = dict(HF_BANDS)[feat]
        return f"HF power {int(lo)}–{int(hi)} Hz"
    return _ed._pretty(feat)


def _doc(feat: str) -> str:
    return _HF_DOCS.get(feat) or _ed.feature_doc(feat)


def _hf_powers(win_traces, fs) -> dict:
    """Σ-PSD power per epoch in each HF band. *win_traces* = [epochs, samples]
    already cropped to the analysis window. Top edge clamped to 0.49*fs."""
    from scipy.signal import periodogram
    from scipy.fft import next_fast_len
    win_traces = np.asarray(win_traces, dtype=float)
    n = win_traces.shape[0]
    if win_traces.ndim != 2 or win_traces.shape[1] < 4:
        return {nm: np.full(n, np.nan) for nm in HF_NAMES}
    f, pxx = periodogram(win_traces, fs=fs,
                         nfft=next_fast_len(win_traces.shape[1]), axis=1)
    nyq = 0.49 * fs
    return {nm: _ef._band_sum(f, pxx, (lo, min(hi, nyq)))
            for nm, (lo, hi) in HF_BANDS}


def _crop_window(traces, time_ms, cfg):
    """Columns of *traces* inside the cfg's [start,end] ms window (crop only)."""
    t = np.asarray(time_ms, dtype=float)
    ws = cfg.window_start_ms if cfg.window_start_ms is not None else t[0]
    we = cfg.window_end_ms if cfg.window_end_ms is not None else t[-1]
    return np.asarray(traces, dtype=float)[:, (t >= ws) & (t <= we)]


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
                    progress=None) -> tuple:
    """ONE trace pass. Per response: recompute *features* (incl. HF band powers)
    over each window and fan (timestamp, value) to its DAY + the WEEK; track the
    stimulus peak-to-peak amplitude (window-independent); and accumulate the mean
    RESPONSE WAVEFORM per 07:00-anchored circadian bin. Returns
    ``(series, ylim, stim, circ)`` where ``stim[period] = {"secs","p2p"}`` and
    ``circ = {"time_ms", "cycles": {cyc: [mean×4]}, "week": [mean×4]}``."""
    cfgs = {_win_tag(w): _win_cfg(w) for w in windows}
    want_wavelet = any(f in _ef.WAVELET_COLUMNS for f in features)
    hf_names = [f for f in features if f in HF_NAMES]
    secs: dict = defaultdict(lambda: defaultdict(list))
    vals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    extent: dict = {wt: [] for wt in cfgs}
    stim_secs: dict = defaultdict(list)
    stim_p2p: dict = defaultdict(list)
    circ_cycles: dict = defaultdict(lambda: [_RunMean() for _ in range(4)])
    circ_week = [_RunMean() for _ in range(4)]
    circ_time = [None]
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
        s_peak = np.asarray(ev.get("stim_peak") or [], dtype=float)
        s_trough = np.asarray(ev.get("stim_trough") or [], dtype=float)
        fs = _fs_of(time_ms)
        avg = _ef.trial_moving_average(traces)
        if circ_time[0] is None:
            circ_time[0] = time_ms
        st2dt = {round(float(_d._nan(r.get("stim_time_sec"))), 4):
                 (_d.parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None)
                 for r in rows if np.isfinite(_d._nan(r.get("stim_time_sec")))}
        dt_by_epoch = [st2dt.get(round(float(s), 4)) for s in st_times]
        # window-INDEPENDENT: stimulus P2P + circadian mean waveform (once/response)
        for j, dt in enumerate(dt_by_epoch):
            if dt is None:
                continue
            ts = dt.timestamp()
            p2p = (float(s_peak[j] - s_trough[j])
                   if j < s_peak.size and j < s_trough.size else np.nan)
            for period in (dt.date().isoformat(), week_key):
                stim_secs[period].append(ts)
                stim_p2p[period].append(p2p)
            cyc, b = _circadian(dt)
            circ_cycles[cyc][b].add(traces[j])
            circ_week[b].add(traces[j])
        # window-DEPENDENT: features (incl. HF band powers merged in)
        for wt, cfg in cfgs.items():
            feats = _ef.compute_all(avg, time_ms, fs, expensive=False, cfg=cfg,
                                    include_wavelet=want_wavelet)
            if hf_names:
                feats = dict(feats)
                feats.update(_hf_powers(_crop_window(avg, time_ms, cfg), fs))
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
    circ = {"time_ms": circ_time[0],
            "cycles": {c: [rm.mean() for rm in bins]
                       for c, bins in sorted(circ_cycles.items())},
            "week": [rm.mean() for rm in circ_week]}
    return (_finalize_values(secs, vals, features), _finalize_ylim(extent),
            _finalize_stim(stim_secs, stim_p2p), circ)


def _finalize_stim(stim_secs, stim_p2p) -> dict:
    out = {}
    for period in stim_secs:
        s = np.asarray(stim_secs[period], dtype=float)
        p = np.asarray(stim_p2p[period], dtype=float)
        order = np.argsort(s, kind="stable")
        out[period] = {"secs": s[order], "p2p": p[order]}
    return out


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
                 f"{_pretty(feat)} · per-decile DAILY MEANS across the week",
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
                 f"{_pretty(feat)} · per-decile FULL-DENSITY (every trace) "
                 f"+ mean", color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat)


# --------------------------------------------------------------------- #
#  package assembly
# --------------------------------------------------------------------- #
def _bin_circadian(secs, vals):
    """Aggregate (secs, vals) into 07:00-anchored 6 h circadian bins → (xs[datetime],
    median, p25, p75) sorted by time, ONE point per (cycle, bin) = 4 points/day."""
    s = np.asarray(secs, dtype=float)
    v = np.asarray(vals, dtype=float)
    fin = np.isfinite(s) & np.isfinite(v)
    s, v = s[fin], v[fin]
    groups: dict = {}
    for ts, val in zip(s, v):
        cyc, b = _circadian(datetime.fromtimestamp(ts))
        groups.setdefault((cyc, b), []).append(val)
    rows = []
    for (cyc, b), vs in groups.items():
        cy = datetime.fromisoformat(cyc)
        x = datetime(cy.year, cy.month, cy.day, 7) + timedelta(hours=6 * b + 3)
        a = np.asarray(vs)
        rows.append((x, float(np.median(a)), float(np.percentile(a, 25)),
                     float(np.percentile(a, 75))))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    xs, md, lo, hi = zip(*rows)
    return np.array(xs), np.array(md), np.array(lo), np.array(hi)


def _fig_metric_trends(feat, per_win, stim, meta, out) -> str:
    """Track a feature OVER TIME at circadian (6 h) resolution: per window, the
    per-bin median + IQR (4 points/day, light raw points behind), with the stimulus
    peak-to-peak amplitude overlaid on a 2nd y-axis to check whether the metric's
    swings track the stimulus. Built from the windowed values -- no trace reads."""
    import matplotlib.dates as mdates
    nwin = max(1, len(per_win))
    fig, axes = plt.subplots(1, nwin, figsize=(5.6 * nwin, 4.2),
                             facecolor=_ed._BG, squeeze=False)
    stim_binned = _bin_circadian(stim["secs"], stim["p2p"]) if stim else None
    for i, (wt, secs, vals) in enumerate(per_win):
        ax = axes[0][i]
        ax.set_facecolor(_ed._PANEL)
        s = np.asarray(secs, dtype=float)
        v = np.asarray(vals, dtype=float)
        fin = np.isfinite(s) & np.isfinite(v)
        if fin.sum() == 0:
            ax.set_title(f"{wt} — no data", color=_ed._MUTED, fontsize=10)
            continue
        dts = np.array([datetime.fromtimestamp(x) for x in s[fin]])
        ax.scatter(dts, v[fin], s=3, c="#6a8cff", alpha=0.08, linewidths=0)
        b = _bin_circadian(s, v)
        if b is not None:
            xs, md, lo, hi = b
            ax.fill_between(xs, lo, hi, color="#6a8cff", alpha=0.22, linewidth=0,
                            label="IQR (p25–p75)")
            ax.plot(xs, md, color="#f0f0f5", lw=1.6, marker="o", ms=4,
                    label="6 h median")
        ax.set_title(wt, color=_ed._TEXT, fontsize=10, loc="left")
        ax.tick_params(colors=_ed._MUTED, labelsize=8)
        ax.set_ylabel(_pretty(feat), color="#c7d0ff", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_ha("right")
        # stimulus P2P on a 2nd axis (window-independent, same on every panel)
        if stim_binned is not None:
            sx, smd, _slo, _shi = stim_binned
            ax2 = ax.twinx()
            ax2.plot(sx, smd, color="#e08a3c", lw=1.4, marker="s", ms=3,
                     alpha=0.9, label="stim P2P")
            ax2.set_ylabel("stim P2P (raw)", color="#e08a3c", fontsize=9)
            ax2.tick_params(axis="y", colors="#e08a3c", labelsize=8)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = (ax2.get_legend_handles_labels()
                      if stim_binned is not None else ([], []))
            leg = ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False,
                            loc="upper left")
            for t in leg.get_texts():
                t.set_color(_ed._TEXT)
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · {_pretty(feat)} · "
                 f"OVER TIME (6 h circadian median ± IQR) · stim P2P overlaid",
                 color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat if feat not in _HF_DOCS else None)


def _fig_circadian_means(circ, meta, out) -> str:
    """Per 07:00-anchored day-cycle, the 4 circadian-bin MEAN response waveforms
    (day-early→night-late), baseline-corrected + artifact-blanked. Panel 0 = the
    whole-period aggregate; then one panel per cycle. Window-independent."""
    tm = circ.get("time_ms")
    if tm is None:
        fig = plt.figure(facecolor=_ed._BG)
        return _pkg._finish(fig, out)
    tm = np.asarray(tm, dtype=float)
    cycles = circ.get("cycles", {})
    panels = [("whole period", circ.get("week"))] + \
        [(c, cycles[c]) for c in sorted(cycles)]
    art = np.abs(tm) <= 1.5
    xm = (tm >= -20) & (tm <= 200)
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    for idx, (label, bins) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_facecolor(_ed._PANEL)
        for b in range(4):
            m = bins[b] if bins else None
            if m is None:
                continue
            y = _ed._baseline(np.asarray(m, dtype=float), tm).copy()
            y[art] = np.nan
            ax.plot(tm[xm], y[xm], color=_CIRC_COLORS[b], lw=1.3,
                    label=_CIRC_LABELS[b] if idx == 0 else None)
        ax.set_title(label, color=_ed._TEXT, fontsize=9, loc="left")
        ax.tick_params(colors=_ed._MUTED, labelsize=7)
        ax.set_xlim(-20, 200)
    for jj in range(len(panels), nrow * ncol):
        axes[jj // ncol][jj % ncol].axis("off")
    # legend INSIDE the first panel (a figure-level legend collided with the title)
    leg = axes[0][0].legend(loc="lower right", fontsize=7, frameon=False,
                            labelspacing=0.3)
    if leg:
        for t in leg.get_texts():
            t.set_color(_ed._TEXT)
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · circadian MEAN response "
                 f"per 6 h bin (07:00-anchored day-cycles)",
                 color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out)


def build_windowed_package(animal, evoked_dir, end_day, out_root, *,
                           features=None, windows=None, n_bands=10,
                           channel_override=None, include_week=True, n_days=7,
                           progress=None) -> dict:
    """Windowed + full-density package ending *end_day*. Renders each of the last
    *n_days* days as a full-density (all-trace) grid, and (when *include_week*) the
    week as a per-day-means grid. ``n_days=1, include_week=False`` -> a single-day
    package (the daily email); the defaults -> the weekly package. Returns a
    manifest dict (also written as manifest.json + index.html)."""
    features = [f for f in (features or (_pkg.PACKAGE_FEATURES + HF_NAMES))
                if f in _ef.ALL_COLUMNS or f in HF_NAMES]
    windows = windows or WINDOWS_MS
    wtags_windows = [(_win_tag(w), w) for w in windows]
    # channel/file discovery reads the DEFAULT-window sidecars, which don't carry
    # the HF columns -- discover with the sidecar-backed features only.
    sidecar_feats = [f for f in features if f not in HF_NAMES] or features
    win_files, win_series = _pkg._period_series(animal, evoked_dir, end_day, n_days,
                                                sidecar_feats)
    if not win_files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    channel = _d.primary_channel(animal, win_series, sidecar_feats, channel_override)
    if channel is None or channel not in win_series:
        return {"animal": animal, "empty": True, "reason": "no channel data"}

    week_key = "week"
    days = [(end_day - timedelta(days=i)) for i in range(n_days - 1, -1, -1)]
    if progress:
        progress(f"recomputing {len(features)} features over {len(windows)} "
                 f"windows for {len(win_files)} recordings…")
    series, ylim, stim, circ = windowed_values(win_files, animal, channel, features,
                                               windows, week_key, progress)

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

    # Metrics-over-time trends ALWAYS span the full window (the week_key series
    # holds every response over n_days) -- cheap, no trace reads. One figure per
    # feature (panel per window). The decile GRIDS shown in the email are the
    # "primary_periods": the week (day-means) when include_week, else each day.
    trend_src = week_key
    primary_periods = ["week"] if include_week else \
        [p for p in order if p != week_key]
    trends: dict = {}
    if progress:
        progress("building metric-over-time trends…")
    trend_dir = os.path.join(pkg, "_trends")
    os.makedirs(trend_dir, exist_ok=True)
    for feat in features:
        per_win = []
        for wt, _w in wtags_windows:
            ser = series.get(wt, {}).get(trend_src)
            if ser is not None:
                per_win.append((wt, ser["secs"], ser["metrics"][feat]))
        if not per_win:
            continue
        tp = os.path.join(trend_dir, f"{feat}.png")
        _fig_metric_trends(feat, per_win, stim.get(trend_src),
                           {"animal": animal, "channel": channel}, tp)
        trends[feat] = os.path.relpath(tp, pkg).replace("\\", "/")

    # Circadian mean-response waveforms (window-independent; one figure/package).
    if progress:
        progress("building circadian mean-response figure…")
    circ_dir = os.path.join(pkg, "_circadian")
    os.makedirs(circ_dir, exist_ok=True)
    cp = os.path.join(circ_dir, "circadian_means.png")
    _fig_circadian_means(circ, {"animal": animal, "channel": channel}, cp)
    circadian_rel = os.path.relpath(cp, pkg).replace("\\", "/")

    manifest = {"animal": animal, "channel": channel, "week": labels[week_key],
                "windows": [wt for wt, _w in wtags_windows],
                "periods": order, "features": features,
                "primary_periods": primary_periods,
                "docs": {f: _doc(f) for f in features},
                "trends": trends, "circadian": circadian_rel, "figures": figures}
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
        parts.append("<h2>① Metrics over time (6 h circadian median ± IQR · "
                     "stim P2P overlaid)</h2>")
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:12px'>")
        for feat in feats:
            rel = trends.get(feat)
            if not rel:
                continue
            parts.append(f"<div><div class='def'>{html.escape(feat)}</div>"
                         f"<a href='{rel}'><img src='{rel}' "
                         f"style='width:360px'></a></div>")
        parts.append("</div>")
    circ = manifest.get("circadian")
    if circ:
        parts.append("<h2>② Circadian mean response (4 bins/day, 07:00-anchored)</h2>")
        parts.append(f"<a href='{circ}'><img src='{circ}' "
                     f"style='width:760px'></a>")
    parts.append("<h2>③ Response “kinds” by decile</h2>")
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
    """Copy emailed figures to *staged* under UNIQUE names (unique Content-IDs).
    Inlines the over-time trends, then each PRIMARY period's decile grids (weekly
    = the week; daily = each of the last 2 days), greedy under the budget. Returns
    (inline_paths, overflow_files, sections, trends) where sections =
    ``[(period, {window: [(feature, cid)]})]``."""
    import shutil
    periods = manifest.get("primary_periods") or ["week"]
    budget = _ATTACH_CAP
    inline, files, trends = [], [], []
    for feat, rel in (manifest.get("trends") or {}).items():
        src = os.path.join(pkg, rel)
        if not os.path.isfile(src):
            continue
        cid = f"trend_{feat}.png"
        dst = os.path.join(staged, cid)
        shutil.copyfile(src, dst)
        if os.path.getsize(dst) <= budget:
            inline.append(dst)
            budget -= os.path.getsize(dst)
            trends.append((feat, cid))
    circ_cid = None
    crel = manifest.get("circadian")
    if crel and os.path.isfile(os.path.join(pkg, crel)):
        cid = "circadian_means.png"
        dst = os.path.join(staged, cid)
        shutil.copyfile(os.path.join(pkg, crel), dst)
        if os.path.getsize(dst) <= budget:
            inline.append(dst)
            budget -= os.path.getsize(dst)
            circ_cid = cid
    fig_by: dict = {}
    for f in manifest["figures"]:
        if f["period"] in periods and f["window"] in wins:
            fig_by.setdefault(f["period"], {}).setdefault(f["window"], []).append(f)
    sections = []
    for period in periods:
        by_win: dict = {}
        for wt in wins:
            for f in fig_by.get(period, {}).get(wt, []):
                src = os.path.join(pkg, f["dir"], f.get("grid", "grid.png"))
                if not os.path.isfile(src):
                    continue
                sz = os.path.getsize(src)
                if sz > budget:
                    files.append(src)
                    continue
                cid = (f"grid_{period}_{f['window']}_{f['feature']}.png"
                       .replace(":", "-"))
                dst = os.path.join(staged, cid)
                shutil.copyfile(src, dst)
                inline.append(dst)
                budget -= sz
                by_win.setdefault(wt, []).append((f["feature"], cid))
        if by_win:
            sections.append((period, by_win))
    return inline, files, sections, trends, circ_cid


def _period_label(manifest, period) -> str:
    return manifest["week"] if period == "week" else period


def _email_html(manifest, sections, trends, circ_cid, cadence) -> str:
    browse = _browse_url(manifest)
    multiday = len(sections) > 1
    span = manifest["week"] if cadence == "weekly" else (
        f"{sections[0][0]} … {sections[-1][0]}" if sections else manifest["week"])
    S = "font-family:sans-serif"
    parts = [
        f"<h2 style='{S}'>Evoked windowed {cadence} digest — "
        f"{html.escape(manifest['animal'])} · {html.escape(manifest['channel'])} · "
        f"{html.escape(span)}</h2>",
        f"<p style='{S};color:#444'>Each feature is RE-MEASURED over each window "
        f"({', '.join(manifest['windows'])}) — magnitudes &amp; deciles differ per "
        f"window.</p>",
        f"<p style='{S};font-size:14px'>📂 <a href='{html.escape(browse)}'>"
        f"<b>Open the full browsable package (all windows × all periods)</b></a>"
        f"<br><span style='color:#888;font-size:12px'>{html.escape(browse)}</span>"
        f"</p>",
        f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
        f"① Metrics over time — 6 h circadian median ± IQR, stim P2P overlaid</h3>"]
    for feat, cid in trends:
        parts.append(
            f"<div style='margin:4px 0 12px'><div style='{S};font-size:13px;"
            f"color:#333'><b>{html.escape(_pretty(feat))}</b></div>"
            f"<img src='cid:{cid}' style='max-width:1100px;width:100%'></div>")
    if circ_cid:
        parts.append(
            f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
            f"② Circadian mean response — 4 bins/day (07:00-anchored)</h3>"
            f"<img src='cid:{circ_cid}' style='max-width:1100px;width:100%'>")
    kind = ("per-day MEANS across the week" if cadence == "weekly"
            else "FULL-DENSITY (every trace)")
    parts.append(
        f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
        f"③ Response “kinds” by decile — {kind}</h3>")
    for period, by_win in sections:
        if multiday:
            parts.append(f"<h3 style='{S};color:#e2a45e;margin-top:16px'>"
                         f"{html.escape(_period_label(manifest, period))}</h3>")
        for win, items in by_win.items():
            parts.append(f"<h4 style='{S};color:#5e7ce2'>window "
                         f"{html.escape(win)}</h4>")
            for feat, cid in items:
                parts.append(
                    f"<div style='margin:4px 0 12px'><div style='{S};font-size:13px;"
                    f"color:#333'>{html.escape(_pretty(feat))}</div>"
                    f"<img src='cid:{cid}' style='max-width:1100px;width:100%'></div>")
    parts.append(f"<p style='{S};font-size:12px;color:#666'>📂 Full browsable "
                 f"package: <a href='{html.escape(browse)}'>{html.escape(browse)}"
                 f"</a></p>")
    return "\n".join(parts)


def send_windowed_email(animal, evoked_dir, end_day, config, emailer, *,
                        cadence="weekly", out_root=None, features=None,
                        windows=None, email_windows=None, daily_days=2,
                        dry_run=False, progress=None) -> dict:
    """Build the windowed package for *cadence* ('daily'|'weekly') and email it,
    organized: a clickable browse link, the metric-over-time trends, then the
    decile grids. Daily covers the last ``daily_days`` days. Never raises for a
    data gap."""
    import tempfile
    out_root = out_root or os.path.join("data", "evoked_windowed_packages")
    os.makedirs(out_root, exist_ok=True)
    weekly = cadence == "weekly"
    # daily = the LAST TWO DAYS (fast); weekly = the 7-day week with day-means.
    n_days = 7 if weekly else int(daily_days or 2)
    manifest = build_windowed_package(
        animal, evoked_dir, end_day, out_root, features=features, windows=windows,
        include_week=weekly, n_days=n_days, progress=progress)
    if manifest.get("empty"):
        return {"sent": False, "reason": manifest.get("reason", "no data"),
                "animal": animal}
    # weekly day-means grids are small -> inline all windows. Daily full-density
    # grids are BIG (~600 KB): 2 days × 3 windows × 9 feats base64-encodes past
    # Gmail's 25 MB limit, so inline ONE balanced window per day (the trends still
    # span all 3 windows; every window is in the browsable folder).
    ew = email_windows or (manifest["windows"] if weekly else ["2-100ms"])
    staged = tempfile.mkdtemp(prefix="evoked_win_mail_")
    inline, files, sections, trends, circ_cid = _stage_email(
        manifest, manifest["pkg_dir"], ew, staged)
    if dry_run:
        return {"sent": False, "dry_run": True, "manifest": manifest,
                "inline": len(inline), "attach": len(files), "trends": len(trends)}
    recipients = (config.get("notifications", {}).get("evoked_windowed", {})
                  .get("recipients")
                  or (config.get("alerting", {}).get("smtp", {}) or {})
                  .get("recipients"))
    span = manifest["week"] if weekly else (
        f"{sections[0][0]}…{sections[-1][0]}" if sections else manifest["week"])
    sent = bool(emailer.send(
        subject=f"QC evoked windowed {cadence} — {animal} · {span}",
        body=f"Evoked windowed {cadence} digest for {animal}. Metrics-over-time + "
             f"decile grids in the HTML body; full package: {_browse_url(manifest)}",
        body_html=_email_html(manifest, sections, trends, circ_cid, cadence),
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
