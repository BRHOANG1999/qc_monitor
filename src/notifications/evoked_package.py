"""Evoked-response FIGURE PACKAGE: per-day + weekly, deciles, overlay/std, zeros.

The deep-dive counterpart to the compact evoked email digest. For one animal over
the last week it renders, per (period, feature): the mean-by-decile figure
(scatter + response + a zeros stim-artifact-window panel), a mean+/-sd view, an
overlay-of-sample-traces view, a 10-panel per-decile grid, and 10 per-decile
files -- into a browsable folder with an ``index.html`` table + ``manifest.json``.

ONE bounded trace pass over the week's evenly-sampled recordings fans every
response to its DAY period and the WEEK period; per (period, feature, band) it
keeps a Welford mean+std (full-res) and a seeded reservoir of decimated sample
traces (for overlays). Daemon/offline safe (in-process reads, matplotlib Agg).

CLI: ``python -m src.notifications.evoked_package --animal BCH111 --date 2026-09-20``
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from src.notifications import evoked_digest as _ed   # noqa: E402
from src.evoked_figures import data as _d            # noqa: E402
from src.utils import evoked_features as _ef          # noqa: E402
from src.utils.evoked_output import read_feature_sidecar, read_file_evoked  # noqa: E402

# Curated relevant features (all in CHEAP_COLUMNS): amplitude, area, timing,
# slope, shape. Any not present / all-NaN for an animal are simply skipped.
PACKAGE_FEATURES = ["peak_to_trough", "trough_amplitude", "line_length",
                    "rms_amplitude", "log_auc", "peak_latency_ms",
                    "trough_latency_ms", "max_slope", "early_late_ratio"]
_FIG_TYPES = ["bands_mean", "bands_sd", "bands_overlay", "deciles_grid"]
_DEFAULT_MAX_FILES = None          # None/0 = read EVERY recording (no downsampling)
_OVERLAY_SAMPLES = 30              # individual full-res traces drawn per band (a
                                   # rendering limit only; mean/sd use ALL responses)


class _BandStats:
    """Welford mean + M2 (-> sd) at full resolution + a seeded reservoir of
    decimated sample traces per band. One instance per (period, feature)."""

    def __init__(self, n_bands: int, k_samples: int, rng):
        self.n_bands = n_bands
        self.k = k_samples
        self.rng = rng
        self.mean = [None] * n_bands
        self.M2 = [None] * n_bands
        self.count = [0] * n_bands
        self.samples: list = [[] for _ in range(n_bands)]
        self.seen = [0] * n_bands
        self.T = None
        self.time_ms = None
        self.stride = 1

    def add(self, b: int, trace, time_ms) -> None:
        t = np.asarray(trace, dtype=float)
        if self.T is None:
            self.T = t.size
            self.time_ms = np.asarray(time_ms, dtype=float)
            self.stride = 1                              # never downsample (full res)
        if t.size != self.T or not np.all(np.isfinite(t)):
            return
        self.count[b] += 1
        n = self.count[b]
        if self.mean[b] is None:
            self.mean[b] = np.zeros(self.T)
            self.M2[b] = np.zeros(self.T)
        d = t - self.mean[b]
        self.mean[b] += d / n
        self.M2[b] += d * (t - self.mean[b])
        self.seen[b] += 1
        td = t[::self.stride]
        if len(self.samples[b]) < self.k:
            self.samples[b].append(td)
        else:                                            # reservoir replace
            j = int(self.rng.integers(0, self.seen[b]))
            if j < self.k:
                self.samples[b][j] = td

    def result(self) -> dict:
        out: dict = {}
        for b in range(self.n_bands):
            n = self.count[b]
            if n == 0:
                continue
            var = np.maximum(self.M2[b] / n, 0.0)
            out[b] = {"mean": self.mean[b], "sd": np.sqrt(var), "n": n,
                      "samples": list(self.samples[b])}
        return out

    def time_decim(self):
        return self.time_ms[::self.stride] if self.time_ms is not None else None


# --------------------------------------------------------------------- #
#  the single fan-out trace pass
# --------------------------------------------------------------------- #
def collect_stats(files, animal, channel, info_by_period: dict, week_key: str,
                  n_bands: int, k_samples: int, rng) -> dict:
    """ONE trace pass. For each response, bin it into its DAY period and the WEEK
    period (band looked up per-period by timestamp). Returns
    ``{period: {feature: _BandStats}}``."""
    stats = {p: {f: _BandStats(n_bands + 1, k_samples, rng) for f in feats}
             for p, feats in info_by_period.items()}
    for fp in files:
        rows = [r for r in (read_feature_sidecar(fp, animal) or [])
                if r.get("channel") == channel]
        if not rows:
            continue
        ev = read_file_evoked(fp, only_animals=[animal]).get(channel)
        if not ev or ev.get("traces") is None:
            continue
        times = np.asarray(ev["times"], dtype=float)
        traces, time_ms = ev["traces"], ev["time_ms"]
        order = np.argsort(times)
        ts_arr = times[order]
        for r in rows:
            st = _d._nan(r.get("stim_time_sec"))
            if not np.isfinite(st):
                continue
            j = int(np.searchsorted(ts_arr, st))
            cand = [k for k in (j - 1, j) if 0 <= k < ts_arr.size]
            if not cand:
                continue
            k = min(cand, key=lambda k: abs(ts_arr[k] - st))
            if abs(ts_arr[k] - st) > _ed._MATCH_TOL_SEC:
                continue
            trace = traces[order[k]]
            key = _ed._ts_key(r)
            dt = _d.parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None
            if dt is None:
                continue
            dk = dt.date().isoformat()
            for period in ({dk, week_key} & set(stats)):
                for feat, info in info_by_period[period].items():
                    b = info["band_of_ts"].get(key, -1)
                    if b >= 0:
                        stats[period][feat].add(b, trace, time_ms)
    return stats


# --------------------------------------------------------------------- #
#  figure builders (reuse evoked_digest draw_* helpers where possible)
# --------------------------------------------------------------------- #
def _finish(fig, out_png: str, feature: str | None = None) -> str:
    for ax in fig.axes:
        ax.tick_params(colors=_ed._MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a52")
    fig.tight_layout(rect=[0, 0.07, 1, 1] if feature else [0, 0, 1, 1])
    if feature:
        _ed.doc_footer(fig, feature)
    fig.savefig(out_png, dpi=125, facecolor=_ed._BG)
    plt.close(fig)
    return out_png


def _overlay_samples(ax, tdec, band_wf, b, color) -> None:
    d = band_wf.get(b)
    if not d:
        return
    base_m = (tdec >= -50) & (tdec <= -5)
    art_m = np.abs(tdec) <= 1.5
    for s in d["samples"]:
        y = s - (np.nanmean(s[base_m]) if base_m.any() else 0.0)
        y = y.copy()
        y[art_m] = np.nan
        ax.plot(tdec, y, color=color, lw=0.5, alpha=0.30)


def _fig_bands_sd(feat, meta, info, band_wf, tm, out) -> str:
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=_ed._BG)
    _ed.draw_response(ax, tm, band_wf, info, blank=True, show_sd=True)
    ax.set_title(f"{meta['date']} · {_ed._pretty(feat)} · mean±sd by decile",
                 color=_ed._TEXT, fontsize=11, loc="left")
    return _finish(fig, out, feat)


def _fig_bands_overlay(feat, meta, info, band_wf, tdec, out) -> str:
    n_bands = info["n_bands"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=_ed._BG)
    ax.set_facecolor(_ed._PANEL)
    for b in range(n_bands):
        _overlay_samples(ax, tdec, band_wf, b, colors[b])
    if band_wf.get(n_bands):
        _overlay_samples(ax, tdec, band_wf, n_bands, _ed._ZCOLOR)
    ax.set_xlim(-20, 200)
    ax.axvline(0, color=_ed._MUTED, lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("ms since stim (artifact blanked)", color=_ed._TEXT, fontsize=10)
    ax.set_ylabel("evoked (a.u.)", color=_ed._TEXT, fontsize=10)
    ax.set_title(f"{meta['date']} · {_ed._pretty(feat)} · sample traces by decile "
                 "(viridis low→high, zeros red)", color=_ed._TEXT, fontsize=10,
                 loc="left")
    return _finish(fig, out, feat)


def _decile_panel(ax, tdec, tm, band_wf, b, color, title) -> None:
    ax.set_facecolor(_ed._PANEL)
    _overlay_samples(ax, tdec, band_wf, b, color)
    d = band_wf.get(b)
    if d:
        mean = _ed._baseline(d["mean"], tm)
        art = np.abs(tm) <= 1.5
        mp = mean.copy(); mp[art] = np.nan
        sd = d["sd"].copy(); sd[art] = np.nan
        ax.plot(tm, mp, color=color, lw=2.0)
        ax.fill_between(tm, mp - sd, mp + sd, color=color, alpha=0.15, linewidth=0)
    ax.set_xlim(-20, 200)
    ax.axvline(0, color=_ed._MUTED, lw=0.5, ls=":", alpha=0.5)
    ax.set_title(title, color=_ed._TEXT, fontsize=9, loc="left")


def _fig_deciles_grid(feat, meta, info, band_wf, tm, tdec, out) -> str:
    n_bands, edges = info["n_bands"], info["edges"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    ncol = 5
    nrow = int(np.ceil(n_bands / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow),
                             facecolor=_ed._BG, squeeze=False)
    for b in range(n_bands):
        ax = axes[b // ncol][b % ncol]
        _decile_panel(ax, tdec, tm, band_wf, b, colors[b],
                      _ed._band_label(b, edges, n_bands))
    for j in range(n_bands, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{meta['animal']} · {meta['date']} · {_ed._pretty(feat)} · "
                 "per-decile overlay + mean±sd", color=_ed._TEXT, fontsize=12)
    return _finish(fig, out, feat)


def _fig_decile(feat, meta, info, band_wf, tm, tdec, b, out) -> str:
    n_bands, edges = info["n_bands"], info["edges"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    fig, ax = plt.subplots(figsize=(7.6, 4.8), facecolor=_ed._BG)
    _decile_panel(ax, tdec, tm, band_wf, b, colors[b],
                  f"{meta['date']} · {_ed._pretty(feat)} · "
                  f"{_ed._band_label(b, edges, n_bands)}")
    ax.set_xlabel("ms since stim (artifact blanked)", color=_ed._TEXT, fontsize=10)
    ax.set_ylabel("evoked (a.u.)", color=_ed._TEXT, fontsize=10)
    return _finish(fig, out, feat)


# --------------------------------------------------------------------- #
#  package assembly
# --------------------------------------------------------------------- #
def _period_series(animal, evoked_dir, day, window_days, features):
    files = _ed.list_window_files(animal, evoked_dir, day, window_days)
    series = _ed.day_series(files, animal, features) if files else {}
    return files, series


def build_package(animal: str, evoked_dir: str, end_day: datetime, out_root: str,
                  *, features=None, n_bands: int = 10,
                  k_samples: int = _OVERLAY_SAMPLES,
                  max_trace_files=_DEFAULT_MAX_FILES,
                  channel_override: str | None = None, seed: int = 0,
                  progress=None) -> dict:
    """Render the full figure package for *animal* over the week ending *end_day*.
    Returns a manifest dict (also written as manifest.json + index.html)."""
    features = [f for f in (features or PACKAGE_FEATURES) if f in _ef.ALL_COLUMNS]
    rng = np.random.default_rng(seed)
    week_files, week_series = _period_series(animal, evoked_dir, end_day, 7, features)
    if not week_files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    channel = _d.primary_channel(animal, week_series, features, channel_override)
    if channel is None or channel not in week_series:
        return {"animal": animal, "empty": True, "reason": "no channel data"}

    # periods: each of the last 7 days (keyed by date) + the week.
    days = [(end_day - timedelta(days=i)) for i in range(6, -1, -1)]
    week_key = "week"
    info_by_period: dict = {}
    series_by_period: dict = {}
    label_by_period: dict = {}
    for dday in days:
        dk = datetime(dday.year, dday.month, dday.day).date().isoformat()
        _f, s = _period_series(animal, evoked_dir, dday, 1, features)
        if channel not in s:
            continue
        info_by_period[dk] = _bands_for(s[channel], features, n_bands)
        series_by_period[dk] = s[channel]
        label_by_period[dk] = dk
    info_by_period[week_key] = _bands_for(week_series[channel], features, n_bands)
    series_by_period[week_key] = week_series[channel]
    label_by_period[week_key] = _ed._date_label(end_day, 7)
    info_by_period = {p: v for p, v in info_by_period.items() if v}   # drop empties

    trace_files = (_ed._sample_files(week_files, max_trace_files)
                   if max_trace_files else week_files)     # default: every recording
    if progress:
        progress(f"reading traces from {len(trace_files)} recordings (one pass)…")
    stats = collect_stats(trace_files, animal, channel, info_by_period, week_key,
                          n_bands, k_samples, rng)

    pkg = os.path.join(out_root,
                       f"evoked_package_{animal}_{label_by_period[week_key]}"
                       .replace(" ", "").replace("–", "_"))
    figures: list = []
    order = [week_key] + [datetime(d.year, d.month, d.day).date().isoformat()
                          for d in reversed(days)]
    for period in [p for p in order if p in info_by_period]:
        for feat in [f for f in features if f in info_by_period[period]]:
            band_wf = stats[period][feat].result()
            if not band_wf:
                continue
            info = info_by_period[period][feat]
            d = series_by_period[period]
            meta = {"animal": animal, "channel": channel,
                    "date": label_by_period[period]}
            fdir = os.path.join(pkg, _safe(period), feat)
            os.makedirs(fdir, exist_ok=True)
            tm = stats[period][feat].time_ms
            tdec = stats[period][feat].time_decim()
            if progress:
                progress(f"{period} · {feat}")
            _ed.feature_figure(feat, d["secs"], d["metrics"][feat], info, tm,
                               band_wf, meta, os.path.join(fdir, "bands_mean.png"))
            _fig_bands_sd(feat, meta, info, band_wf, tm,
                          os.path.join(fdir, "bands_sd.png"))
            _fig_bands_overlay(feat, meta, info, band_wf, tdec,
                               os.path.join(fdir, "bands_overlay.png"))
            _fig_deciles_grid(feat, meta, info, band_wf, tm, tdec,
                              os.path.join(fdir, "deciles_grid.png"))
            for b in range(info["n_bands"]):
                if b in band_wf:
                    _fig_decile(feat, meta, info, band_wf, tm, tdec, b,
                                os.path.join(fdir, f"decile_{b + 1:02d}.png"))
            figures.append({"period": period, "label": label_by_period[period],
                            "feature": feat,
                            "dir": os.path.relpath(fdir, pkg).replace("\\", "/"),
                            "n_zero": info.get("n_zero", 0)})

    manifest = {"animal": animal, "channel": channel,
                "week": label_by_period[week_key],
                "periods": order, "features": features,
                "docs": {f: _ed.feature_doc(f) for f in features},
                "trace_files_used": len(trace_files),
                "figures": figures}
    with open(os.path.join(pkg, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    _write_index(pkg, manifest, label_by_period)
    manifest["pkg_dir"] = pkg
    manifest["index"] = os.path.join(pkg, "index.html")
    return manifest


def _bands_for(chan_series: dict, features: list, n_bands: int) -> dict:
    out = {}
    for feat in features:
        vals = chan_series["metrics"].get(feat)
        if vals is None:
            continue
        info = _ed.compute_bands(chan_series["secs"], vals, n_bands)
        if info is not None:
            out[feat] = info
    return out


def _safe(s: str) -> str:
    return "day_" + s if s and s[0].isdigit() else s


# --------------------------------------------------------------------- #
#  the browsable index
# --------------------------------------------------------------------- #
def _write_index(pkg: str, manifest: dict, labels: dict) -> None:
    figs = {(f["period"], f["feature"]): f for f in manifest["figures"]}
    periods = [p for p in manifest["periods"] if any(p == k[0] for k in figs)]
    feats = manifest["features"]
    types = _FIG_TYPES + [f"decile_{i:02d}" for i in range(1, 11)]
    rows = []
    for p in periods:
        cells = [f"<th style='text-align:left'>{html.escape(labels.get(p, p))}</th>"]
        for feat in feats:
            f = figs.get((p, feat))
            if not f:
                cells.append("<td class='na'>—</td>")
                continue
            d = f["dir"]
            thumb = f"{d}/bands_mean.png"
            links = " · ".join(
                f"<a href='{d}/{t}.png'>{t.replace('bands_', '').replace('_', '')}</a>"
                for t in _FIG_TYPES)
            dl = "".join(f"<a href='{d}/decile_{i:02d}.png'>{i}</a> "
                         for i in range(1, 11))
            zt = (f" · <b style='color:#ff8080'>{f['n_zero']} zero</b>"
                  if f.get("n_zero") else "")
            cells.append(
                f"<td><a href='{thumb}'><img src='{thumb}' loading='lazy'></a><br>"
                f"<span class='lk'>{links}{zt}<br>deciles: {dl}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr><th></th>" + "".join(
        f"<th>{html.escape(_ed._pretty(x))}"
        f"<br><span class=def>{html.escape(_ed.feature_doc(x))}</span></th>"
        for x in feats) + "</tr>"
    doc = f"""<!doctype html><meta charset=utf-8>
<title>Evoked package · {html.escape(manifest['animal'])}</title>
<style>
 body{{background:#12121c;color:#e8e8f0;font:13px system-ui;margin:16px}}
 h1{{font-size:18px}} .sub{{color:#9a9ab0}}
 table{{border-collapse:collapse}} td,th{{border:1px solid #2a2a3a;padding:6px;vertical-align:top}}
 th{{background:#1e1e2f;position:sticky;top:0}} img{{width:300px;display:block;border:1px solid #2a2a3a}}
 .lk{{font-size:11px;color:#9a9ab0}} .lk a{{color:#8ab4ff;text-decoration:none}}
 .na{{color:#555;text-align:center}} a{{color:#8ab4ff}}
 .def{{display:block;max-width:280px;font-weight:400;font-size:10px;color:#8a8aa0;white-space:normal}}
</style>
<h1>Evoked-response package · {html.escape(manifest['animal'])}
 · {html.escape(manifest['channel'])}</h1>
<div class=sub>week {html.escape(manifest['week'])} ·
 {len(manifest['figures'])} feature-panels · traces from
 {manifest['trace_files_used']} sampled recordings ·
 rows = periods (week first, then each day) · columns = features ·
 each cell: bands_mean thumbnail + links to sd / overlay / decilesgrid + the 10 per-decile files</div>
<table>{head}{''.join(rows)}</table>
"""
    with open(os.path.join(pkg, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--date", default=None, help="week END day YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--out", default=None, help="output root dir")
    ap.add_argument("--features", default=None, help="comma-separated override")
    ap.add_argument("--max-trace-files", type=int, default=_DEFAULT_MAX_FILES)
    a = ap.parse_args(argv)
    import yaml
    config = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = (config.get("chronic_evoked", {}) or {}).get("evoked_output_dir", "")
    end = (datetime.fromisoformat(a.date) if a.date
           else datetime.now() - timedelta(days=1))
    out = a.out or os.path.join("data", "evoked_packages")
    os.makedirs(out, exist_ok=True)
    feats = a.features.split(",") if a.features else None
    res = build_package(a.animal, evoked_dir, end, out, features=feats,
                        max_trace_files=a.max_trace_files,
                        progress=lambda m: print("  ", m, flush=True))
    print("RESULT:", {k: v for k, v in res.items()
                      if k not in ("figures", "docs")})
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
