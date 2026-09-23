"""Windowed evoked-response figure package (decile SHAPES + trends).

1. MULTIPLE ANALYSIS WINDOWS (2-50, 2-100, 2-500 ms). The window is not a display
   crop -- every feature is RE-MEASURED over each window (line-length, RMS, peak, HF
   band power, ...), so a response's decile changes per window. Recomputed in-memory
   from the raw traces (one read/file, reused across windows); nothing is filtered.

2. Per feature, per window: NORMALIZED metric-vs-stim over-time trends (robust z,
   6 h circadian median, per response); a CIRCADIAN mean±SD response figure (4 bins/
   day); and the decile "kinds" as clean SHAPE views -- a stacked ridgeline (mean±SD),
   a peak-normalized overlay, and a per-decile mean±SD grid. No full-density overlay
   (it was too busy and slow to render) -- every figure shows a mean WITH its spread.

Memory/I-O shape: ONE values pass reads+recomputes every file once and fans each
response's windowed values to its day + the week; then a per-period trace pass
re-reads that period's traces (no recompute) to accumulate Welford mean/SD per
decile, renders, and frees -- peak memory independent of the trace count.

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
from src.notifications.trace_density import robust_ylim  # noqa: E402
from src.evoked_figures import data as _d                # noqa: E402
from src.periictal import passive as _passive            # noqa: E402
from src.utils import evoked_features as _ef              # noqa: E402
from src.utils.evoked_output import read_feature_sidecar, read_file_evoked  # noqa: E402

# (start, end) ms. All start at 2 ms (just past the 1 ms artifact guard), so the
# crop always keeps >=2 samples -- no silent full-trace fallback. Wide windows are
# simply capped by the file's data extent (~+/-200 or +/-500 ms).
WINDOWS_MS = [(2.0, 50.0), (2.0, 100.0), (2.0, 500.0)]

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
    def _pack(bins):
        return [{"mean": rm.mean(), "sd": rm.sd()} for rm in bins]
    circ = {"time_ms": circ_time[0],
            "cycles": {c: _pack(bins) for c, bins in sorted(circ_cycles.items())},
            "week": _pack(circ_week)}
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
#  per-period render pass: Welford mean/SD per decile (no full-density)
# --------------------------------------------------------------------- #
class _RunMean:
    """Streaming mean + SD of a fixed-length vector (one bin's mean trace + spread).
    We never plot a mean without its spread, so sum-of-squares is tracked too."""

    def __init__(self):
        self.sum = None
        self.sumsq = None
        self.n = 0

    def add(self, v):
        v = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(v)):
            return
        if self.sum is None:
            self.sum = np.zeros(v.size)
            self.sumsq = np.zeros(v.size)
        if v.size != self.sum.size:
            return
        self.sum += v
        self.sumsq += v * v
        self.n += 1

    def mean(self):
        return self.sum / self.n if self.n else None

    def sd(self):
        if not self.n:
            return None
        m = self.sum / self.n
        return np.sqrt(np.maximum(self.sumsq / self.n - m * m, 0.0))


def _render_period(files, animal, channel, period, wtags_windows, info_pw, ylim,
                   labels, pkg, is_week=False, progress=None) -> list:
    """One trace pass over *period*'s files for ALL windows: per (window, feature)
    accumulate a Welford MEAN + SD per decile (no full-density -- dropped), then
    render the clean shape views (ridgeline, peak-normalized, per-decile mean±SD
    grid), each showing spread. Render, free. Returns figure records."""
    accs: dict = {(wt, feat): _ed._BandAccum(info["n_bands"] + 1)
                  for wt, _w in wtags_windows for feat, info in info_pw[wt].items()}
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
            for (wt, feat), m in maps.items():
                b = m.get(key, -1)
                if b >= 0:
                    accs[(wt, feat)].add(b, trace, time_ms)
    # render
    figs = []
    for wt, w in wtags_windows:
        for feat, info in info_pw[wt].items():
            band_wf = accs[(wt, feat)].result()
            if not band_wf:
                continue
            fdir = os.path.join(pkg, wt, _pkg._safe(period), feat)
            os.makedirs(fdir, exist_ok=True)
            tm = accs[(wt, feat)].time_ms
            meta = {"animal": animal, "channel": channel,
                    "date": labels.get(period, period), "window": wt}
            if progress:
                progress(f"{wt} · {period} · {feat}")
            _fig_ridgeline(feat, meta, info, band_wf, tm, (w[0], w[1]),
                           os.path.join(fdir, "deciles_ridgeline.png"))
            _fig_normalized(feat, meta, info, band_wf, tm, (w[0], w[1]),
                            os.path.join(fdir, "deciles_normalized.png"))
            _fig_decile_grid(feat, meta, info, band_wf, tm, (w[0], w[1]),
                             ylim[wt], os.path.join(fdir, "deciles_grid.png"))
            figs.append({"period": period, "window": wt, "feature": feat,
                         "dir": os.path.relpath(fdir, pkg).replace("\\", "/"),
                         "ridgeline": "deciles_ridgeline.png",
                         "normalized": "deciles_normalized.png",
                         "grid": "deciles_grid.png",
                         "n_zero": info.get("n_zero", 0)})
    return figs


# --------------------------------------------------------------------- #
#  clean "shapes" figures: stacked decile means (ridgeline) + normalized
# --------------------------------------------------------------------- #
def _decile_means(band_wf, info, tm, xlim):
    """List of (b, mean, sd) per populated decile — mean baseline-corrected+blanked,
    sd artifact-blanked (we never plot a mean without its spread). Cropped to xlim."""
    n_bands = info["n_bands"]
    art = np.abs(tm) <= 1.5
    xm = (tm >= xlim[0]) & (tm <= xlim[1])
    out = []
    for b in range(n_bands):
        wf = band_wf.get(b)
        if wf is None:
            continue
        y = _ed._baseline(np.asarray(wf["mean"], dtype=float), tm).copy()
        y[art] = np.nan
        sd = None
        if wf.get("sd") is not None:
            sd = np.asarray(wf["sd"], dtype=float).copy()
            sd[art] = np.nan
        out.append((b, y, sd))
    return out, xm


def _fig_ridgeline(feat, meta, info, band_wf, tm, xlim, out) -> str:
    """The 10 decile MEAN waveforms STACKED with a vertical offset (bottom decile
    at the bottom), each on its own baseline -> all shapes legible at once, no
    smear. Offset = 0.6× the pooled p1–p99 range."""
    n_bands, edges = info["n_bands"], info["edges"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, n_bands))
    means, xm = _decile_means(band_wf, info, tm, xlim)
    if not means:
        return _pkg._finish(plt.figure(facecolor=_ed._BG), out)
    allv = np.concatenate([m[xm][np.isfinite(m[xm])] for _b, m, _sd in means])
    span = (np.percentile(allv, 99) - np.percentile(allv, 1)) if allv.size else 1.0
    step = 0.7 * span if span > 0 else 1.0
    fig, ax = plt.subplots(figsize=(9.0, 7.5), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    yticks, ylabels = [], []

    def _row(b, m, sd, color, label):
        off = b * step
        if sd is not None:                        # ±1 SD ribbon (never mean-alone)
            ax.fill_between(tm[xm], m[xm] + off - sd[xm], m[xm] + off + sd[xm],
                            color=color, alpha=0.16, linewidth=0)
        ax.plot(tm[xm], m[xm] + off, color=color, lw=1.5)
        yticks.append(off)
        ylabels.append(label)

    for b, m, sd in means:
        _row(b, m, sd, colors[b], _ed._band_label(b, edges, n_bands))
    if band_wf.get(n_bands):                       # zero/pile band on top, red
        y = _ed._baseline(np.asarray(band_wf[n_bands]["mean"], float), tm).copy()
        y[np.abs(tm) <= 1.5] = np.nan
        zsd = None
        if band_wf[n_bands].get("sd") is not None:
            zsd = np.asarray(band_wf[n_bands]["sd"], float).copy()
            zsd[np.abs(tm) <= 1.5] = np.nan
        _row(n_bands, y, zsd, _ed._ZCOLOR, "zeros")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.tick_params(colors=_ed._MUTED, labelsize=8)
    ax.set_xlim(xlim)
    ax.set_xlabel("ms since stim (artifact blanked)", color=_ed._TEXT, fontsize=10)
    ax.set_title(f"{meta['animal']} · {meta['date']} · {meta['window']} · "
                 f"{_pretty(feat)} · decile MEAN ± 1 SD, stacked (low→high)",
                 color=_ed._TEXT, fontsize=11, loc="left")
    return _pkg._finish(fig, out, feat if feat not in _HF_DOCS else None)


def _fig_normalized(feat, meta, info, band_wf, tm, xlim, out) -> str:
    """Each decile MEAN peak-normalized to unit amplitude and overlaid — reveals
    whether the SHAPE changes across deciles or it is only amplitude scaling."""
    n_bands = info["n_bands"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, n_bands))
    means, xm = _decile_means(band_wf, info, tm, xlim)
    if not means:
        return _pkg._finish(plt.figure(facecolor=_ed._BG), out)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    for b, m, sd in means:
        pk = np.nanmax(np.abs(m[xm]))
        pk = pk if (pk and np.isfinite(pk)) else 1.0
        yn = m / pk
        if sd is not None:                        # faint ±1 SD (never mean-alone)
            sn = sd / pk
            ax.fill_between(tm[xm], (yn - sn)[xm], (yn + sn)[xm], color=colors[b],
                            alpha=0.06, linewidth=0)
        ax.plot(tm[xm], yn[xm], color=colors[b], lw=1.2, alpha=0.9,
                label=f"D{b + 1}")
    ax.axhline(0, color=_ed._MUTED, lw=0.5, alpha=0.4)
    ax.set_xlim(xlim)
    ax.tick_params(colors=_ed._MUTED, labelsize=8)
    ax.set_xlabel("ms since stim (artifact blanked)", color=_ed._TEXT, fontsize=10)
    ax.set_ylabel("peak-normalized (±SD faint)", color=_ed._TEXT, fontsize=10)
    leg = ax.legend(fontsize=7, ncol=2, frameon=False, labelspacing=0.3,
                    loc="lower right")
    for t in leg.get_texts():
        t.set_color(_ed._TEXT)
    ax.set_title(f"{meta['animal']} · {meta['date']} · {meta['window']} · "
                 f"{_pretty(feat)} · decile means PEAK-NORMALIZED (shape only)",
                 color=_ed._TEXT, fontsize=11, loc="left")
    return _pkg._finish(fig, out, feat if feat not in _HF_DOCS else None)


def _fig_decile_grid(feat, meta, info, band_wf, tm, xlim, ylim, out) -> str:
    """One panel per decile: that decile's MEAN ± 1 SD response — the spread of the
    responses inside the decile (n in the thousands, so SD not SEM). Colored
    low→high, n annotated. Replaces the busy full-density overlay."""
    n_bands, edges = info["n_bands"], info["edges"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, n_bands))
    art = np.abs(tm) <= 1.5
    xm = (tm >= xlim[0]) & (tm <= xlim[1])
    ncol = 5
    nrow = int(np.ceil(n_bands / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    for b in range(n_bands):
        ax = axes[b // ncol][b % ncol]
        ax.set_facecolor(_ed._PANEL)
        wf = band_wf.get(b)
        if wf:
            m = _ed._baseline(np.asarray(wf["mean"], dtype=float), tm).copy()
            m[art] = np.nan
            if wf.get("sd") is not None:
                sd = np.asarray(wf["sd"], dtype=float).copy()
                sd[art] = np.nan
                ax.fill_between(tm[xm], (m - sd)[xm], (m + sd)[xm],
                                color=colors[b], alpha=0.22, linewidth=0)
            ax.plot(tm[xm], m[xm], color=colors[b], lw=1.6)
            ax.set_title(f"{_ed._band_label(b, edges, n_bands)}  (n={wf.get('n', 0)})",
                         color=_ed._TEXT, fontsize=9, loc="left")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(colors=_ed._MUTED, labelsize=7)
    for jj in range(n_bands, nrow * ncol):
        axes[jj // ncol][jj % ncol].axis("off")
    fig.suptitle(f"{meta['animal']} · {meta['date']} · {meta['window']} · "
                 f"{_pretty(feat)} · per-decile MEAN ± 1 SD",
                 color=_ed._TEXT, fontsize=12)
    return _pkg._finish(fig, out, feat if feat not in _HF_DOCS else None)

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


def _robust_z(v):
    """Robust z (median-centered, MAD-scaled) so a tiny-but-consistent drift is
    visible on the SAME scale as the metric. Falls back to std, then to zeros for a
    truly constant series."""
    v = np.asarray(v, dtype=float)
    fin = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if fin.sum() < 2:
        return out
    med = np.median(v[fin])
    scale = 1.4826 * np.median(np.abs(v[fin] - med))
    if scale <= 0:
        scale = np.std(v[fin])
    if scale <= 0:
        out[fin] = 0.0
        return out
    out[fin] = (v[fin] - med) / scale
    return out


def _fig_metric_trends(feat, per_win, stim, meta, out) -> str:
    """Track a feature OVER TIME vs the STIMULUS, both NORMALIZED (robust z) onto one
    axis so their trends are comparable regardless of absolute scale — the raw stim
    P2P barely moves, but its z-trend reveals any real drift. Per window: each series
    shows its per-response points (light) + the 6 h circadian median line."""
    import matplotlib.dates as mdates
    nwin = max(1, len(per_win))
    fig, axes = plt.subplots(1, nwin, figsize=(5.8 * nwin, 4.3),
                             facecolor=_ed._BG, squeeze=False)
    have_stim = stim is not None and len(np.asarray(stim.get("secs", []))) > 1
    for i, (wt, secs, vals) in enumerate(per_win):
        ax = axes[0][i]
        ax.set_facecolor(_ed._PANEL)
        s = np.asarray(secs, dtype=float)
        vz = _robust_z(vals)
        fin = np.isfinite(s) & np.isfinite(vz)
        if fin.sum() == 0:
            ax.set_title(f"{wt} — no data", color=_ed._MUTED, fontsize=10)
            continue
        mdts = np.array([datetime.fromtimestamp(x) for x in s[fin]])
        ax.scatter(mdts, vz[fin], s=3, c="#6a8cff", alpha=0.06, linewidths=0)
        bm = _bin_circadian(s, vz)
        if bm is not None:
            ax.plot(bm[0], bm[1], color="#c7d0ff", lw=1.8, marker="o", ms=4,
                    label=f"{_pretty(feat)} (z)", zorder=4)
        if have_stim:                              # stim P2P, same z-scale, per-response
            ss = np.asarray(stim["secs"], dtype=float)
            sz = _robust_z(stim["p2p"])
            sfin = np.isfinite(ss) & np.isfinite(sz)
            sdts = np.array([datetime.fromtimestamp(x) for x in ss[sfin]])
            ax.scatter(sdts, sz[sfin], s=3, c="#e08a3c", alpha=0.05, linewidths=0)
            bs = _bin_circadian(ss, sz)
            if bs is not None:
                ax.plot(bs[0], bs[1], color="#e08a3c", lw=1.8, marker="s", ms=3,
                        label="stim P2P (z)", zorder=5)
        ax.axhline(0, color=_ed._MUTED, lw=0.5, alpha=0.3)
        # clip the y-view to the bulk so heavy-tailed outliers don't compress the
        # trend lines (the median lines are the point); outliers run off-screen.
        pooled = vz[fin]
        if have_stim:
            pooled = np.concatenate([pooled, sz[sfin]])
        if pooled.size:
            lo_y, hi_y = np.nanpercentile(pooled, [1, 99])
            pad = 0.4 * (hi_y - lo_y) + 0.5
            ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_title(wt, color=_ed._TEXT, fontsize=10, loc="left")
        ax.set_ylabel("robust z (per window)", color="#aaa", fontsize=9)
        ax.tick_params(colors=_ed._MUTED, labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(35)
            lab.set_ha("right")
        if i == 0:
            leg = ax.legend(fontsize=8, frameon=False, loc="upper left")
            for t in leg.get_texts():
                t.set_color(_ed._TEXT)
    fig.suptitle(f"{meta['animal']} · {meta['channel']} · {_pretty(feat)} vs stim "
                 f"P2P · OVER TIME, NORMALIZED (robust z; 6 h median lines)",
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
            rec = bins[b] if bins else None
            if not rec or rec.get("mean") is None:
                continue
            y = _ed._baseline(np.asarray(rec["mean"], dtype=float), tm).copy()
            y[art] = np.nan
            if rec.get("sd") is not None:            # ±1 SD (never mean-alone)
                sd = np.asarray(rec["sd"], dtype=float).copy()
                sd[art] = np.nan
                ax.fill_between(tm[xm], (y - sd)[xm], (y + sd)[xm],
                                color=_CIRC_COLORS[b], alpha=0.12, linewidth=0)
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
    """Windowed decile-shapes package ending *end_day*. Renders, per (period,
    window, feature): the ridgeline (mean±SD), peak-normalized overlay, and
    per-decile mean±SD grid, plus the metric-vs-stim trends and circadian means.
    ``n_days=2, include_week=False`` -> the daily package (last 2 days); the
    defaults -> the weekly. Returns a manifest (also written as manifest.json +
    index.html)."""
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
             f" · windowed decile shapes · {html.escape(manifest['week'])}</h1>",
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
                grid = f.get("grid", "deciles_grid.png")
                ridge = f.get("ridgeline", "deciles_ridgeline.png")
                row.append(
                    f"<td><a href='{d}/{ridge}'>"
                    f"<img src='{d}/{ridge}'></a>"
                    f"<a href='{d}/deciles_normalized.png'>normalized</a> · "
                    f"<a href='{d}/{grid}'>mean±SD grid</a></td>")
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
    # Section ③ shows the CLEAN shapes with spread: the stacked ridgeline (mean±SD),
    # the peak-normalized overlay, and the per-decile mean±SD grid.
    sections = []
    for period in periods:
        by_win: dict = {}
        for wt in wins:
            for f in fig_by.get(period, {}).get(wt, []):
                cids = {}
                for kind in ("ridgeline", "normalized", "grid"):
                    src = os.path.join(pkg, f["dir"], f.get(kind, ""))
                    if not f.get(kind) or not os.path.isfile(src):
                        continue
                    sz = os.path.getsize(src)
                    if sz > budget:
                        files.append(src)
                        continue
                    cid = (f"{kind}_{period}_{f['window']}_{f['feature']}.png"
                           .replace(":", "-"))
                    shutil.copyfile(src, os.path.join(staged, cid))
                    inline.append(os.path.join(staged, cid))
                    budget -= sz
                    cids[kind] = cid
                if cids:
                    by_win.setdefault(wt, []).append((f["feature"], cids))
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
    parts.append(
        f"<h3 style='{S};border-bottom:2px solid #5e7ce2;padding-bottom:3px'>"
        f"③ Decile SHAPES — ridgeline (mean±SD) + peak-normalized + per-decile mean±SD grid</h3>"
        f"<p style='{S};font-size:12px;color:#666'>The 10 decile mean waveforms, "
        f"stacked so each shape is legible, and peak-normalized so shape reads "
        f"apart from amplitude. (Full-density all-trace grids are in the folder.)</p>")
    for period, by_win in sections:
        if multiday:
            parts.append(f"<h3 style='{S};color:#e2a45e;margin-top:16px'>"
                         f"{html.escape(_period_label(manifest, period))}</h3>")
        for win, items in by_win.items():
            parts.append(f"<h4 style='{S};color:#5e7ce2'>window "
                         f"{html.escape(win)}</h4>")
            for feat, cids in items:
                imgs = "".join(
                    f"<img src='cid:{cids[k]}' style='max-width:560px;width:49%;"
                    f"display:inline-block;vertical-align:top'>"
                    for k in ("ridgeline", "normalized") if k in cids)
                if "grid" in cids:                 # per-decile mean±SD, full width
                    imgs += (f"<img src='cid:{cids['grid']}' "
                             f"style='max-width:1100px;width:100%'>")
                parts.append(
                    f"<div style='margin:4px 0 12px'><div style='{S};font-size:13px;"
                    f"color:#333'>{html.escape(_pretty(feat))}</div>{imgs}</div>")
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
                    help="daily = last 2 days; weekly = the 7-day week")
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
