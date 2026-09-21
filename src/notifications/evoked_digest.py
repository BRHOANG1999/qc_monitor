"""Evoked-response daily digest: per-feature even-percentile "kinds".

For one animal over one day, split each evoked feature's per-stimulus values into
EVEN-POPULATION percentile bands (quintiles by default) and show, per band, the
mean +/- sd evoked waveform of the responses in it -- so the top-amplitude band
and the bottom-line-length band etc. are surfaced as actual response shapes (the
"kinds"), not just numbers.

Data: per-stimulus features from the JSON sidecars (via evoked_figures.data);
waveforms from the recording *_evoked.mat (evoked_output.read_file_evoked).
Daemon-safe (in-process reads; matplotlib Agg) and Dash-free / unit-testable.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from src.evoked_figures import data as _d  # noqa: E402
from src.utils import evoked_features as _ef  # noqa: E402
from src.utils.evoked_output import (list_evoked_files, animals_in_filename,  # noqa: E402
                                     parse_recording_dt, read_feature_sidecar,
                                     read_file_evoked)

# Amplitude (peak_to_trough), wiggliness (line_length), size (rms), timing
# (peak latency), decay (recovery tau). All are ALL_COLUMNS members.
DEFAULT_FEATURES = ["peak_to_trough", "line_length", "rms_amplitude",
                    "peak_latency_ms", "recovery_tau"]
HEADLINE_FEATURES = ["peak_to_trough", "line_length"]     # inline in the email
DEFAULT_N_BANDS = 5
_MIN_PER_FEATURE = 40          # too few finite responses -> skip that feature
_MATCH_TOL_SEC = 0.5           # stim-time match tolerance (row <-> trace)
_BG = "#1e1e2f"
_PANEL = "#26263a"
_TEXT = "#f0f0f5"
_MUTED = "#9a9ab0"


def _pretty(feature: str) -> str:
    return feature.replace("_", " ")


def day_bounds(day: datetime) -> tuple[float, float]:
    """[start, end) unix seconds for the calendar day of *day*."""
    d0 = datetime(day.year, day.month, day.day)
    return d0.timestamp(), (d0 + timedelta(days=1)).timestamp()


def list_day_files(animal: str, evoked_dir: str, day: datetime) -> list[str]:
    """The animal's *_evoked.mat files recorded on *day*, oldest first."""
    d0 = datetime(day.year, day.month, day.day)
    d1 = d0 + timedelta(days=1)
    out = []
    for fp in list_evoked_files(evoked_dir):
        if animal not in animals_in_filename(fp):
            continue
        dt = parse_recording_dt(fp)
        if dt is not None and d0 <= dt < d1:
            out.append(fp)
    return sorted(out)


def day_series(day_files: list[str], animal: str, metrics: list[str]) -> dict:
    """Per-channel per-stimulus feature arrays over *day_files* (sidecar read).
    Returns ``{channel: {"secs": float[n], "metrics": {m: float[n]}}}`` sorted by
    time -- the day-scoped analogue of evoked_figures.data.animal_series."""
    csecs: dict = defaultdict(list)
    cvals: dict = defaultdict(lambda: defaultdict(list))
    for fp in day_files:
        rows = read_feature_sidecar(fp, animal) or []
        by_ch: dict = defaultdict(list)
        for r in rows:
            if r.get("channel"):
                by_ch[r["channel"]].append(r)
        for ch, rws in by_ch.items():
            csecs[ch].append(np.fromiter(
                ((_d.parse_iso(r.get("abs_dt")) or datetime.min).timestamp()
                 if r.get("abs_dt") else np.nan for r in rws),
                dtype=float, count=len(rws)))
            for m in metrics:
                cvals[ch][m].append(np.fromiter(
                    (_d._nan(r.get(m)) for r in rws), dtype=float, count=len(rws)))
    series: dict = {}
    for ch, parts in csecs.items():
        secs = np.concatenate(parts)
        order = np.argsort(secs, kind="stable")
        series[ch] = {"secs": secs[order],
                      "metrics": {m: np.concatenate(cvals[ch][m])[order]
                                  for m in metrics}}
    return series


def even_band_edges(vals: np.ndarray, n_bands: int = DEFAULT_N_BANDS):
    """Percentile band edges from the finite values -- equal-population where the
    distribution allows, but a large point mass (e.g. many flat 0-amplitude
    responses on inactive stimuli) collapses to ONE wide band rather than being
    split arbitrarily, so value->band stays deterministic. Returns the unique
    edge array (>= 3 edges => >= 2 bands) or None when degenerate (single value /
    too few points). The effective band count is ``len(edges) - 1``."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < max(n_bands, _MIN_PER_FEATURE):
        return None
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bands + 1)))
    if edges.size < 3:                               # < 2 distinct bands => skip
        return None
    edges[-1] = np.nextafter(edges[-1], np.inf)      # include the max in the top band
    return edges


def n_bands_of(edges: np.ndarray) -> int:
    return len(edges) - 1


def assign_band(value: float, edges: np.ndarray) -> int:
    """Band index [0, n_bands) for *value*, or -1 if non-finite/out of range."""
    if not np.isfinite(value):
        return -1
    b = int(np.searchsorted(edges, value, side="right") - 1)
    return b if 0 <= b < len(edges) - 1 else -1


class _BandAccum:
    """Streaming per-band mean + sd of the evoked trace (never holds all traces)."""

    def __init__(self, n_bands: int):
        self.n_bands = n_bands
        self.sum: list = [None] * n_bands
        self.sumsq: list = [None] * n_bands
        self.count = [0] * n_bands
        self.T: int | None = None
        self.time_ms = None

    def add(self, band: int, trace: np.ndarray, time_ms) -> None:
        t = np.asarray(trace, dtype=float)
        if self.T is None:
            self.T, self.time_ms = t.size, np.asarray(time_ms, dtype=float)
        if t.size != self.T or not np.all(np.isfinite(t)):
            return
        if self.sum[band] is None:
            self.sum[band] = np.zeros(self.T)
            self.sumsq[band] = np.zeros(self.T)
        self.sum[band] += t
        self.sumsq[band] += t * t
        self.count[band] += 1

    def result(self) -> dict:
        out: dict = {}
        for b in range(self.n_bands):
            n = self.count[b]
            if n == 0:
                continue
            mean = self.sum[b] / n
            var = np.maximum(self.sumsq[b] / n - mean * mean, 0.0)
            out[b] = {"mean": mean, "sd": np.sqrt(var), "n": n}
        return out


def band_waveforms_multi(day_files: list[str], animal: str, channel: str,
                         edges_by_feat: dict) -> dict:
    """ONE streaming trace pass over the day's *channel* recordings: match each
    epoch to its sidecar row by stim time, and for EVERY feature accumulate its
    per-band mean +/- sd waveform. ``edges_by_feat = {feature: edges}``; returns
    ``{feature: (time_ms, {band: {mean, sd, n}})}``. Reading each recording's
    traces once (not once per feature) keeps the daemon's I/O bounded."""
    accs = {f: _BandAccum(n_bands_of(e)) for f, e in edges_by_feat.items()}
    for fp in day_files:
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
        ts = times[order]
        for r in rows:
            st = _d._nan(r.get("stim_time_sec"))
            if not np.isfinite(st):
                continue
            j = int(np.searchsorted(ts, st))
            cand = [k for k in (j - 1, j) if 0 <= k < ts.size]
            if not cand or abs(ts[min(cand, key=lambda k: abs(ts[k] - st))] - st) \
                    > _MATCH_TOL_SEC:
                continue
            trace = traces[order[min(cand, key=lambda k: abs(ts[k] - st))]]
            for f, edges in edges_by_feat.items():
                b = assign_band(_d._nan(r.get(f)), edges)
                if b >= 0:
                    accs[f].add(b, trace, time_ms)
    return {f: (acc.time_ms, acc.result()) for f, acc in accs.items()}


def band_waveforms(day_files: list[str], animal: str, channel: str,
                   feature: str, edges: np.ndarray) -> tuple:
    """Single-feature convenience wrapper over ``band_waveforms_multi``."""
    return band_waveforms_multi(day_files, animal, channel, {feature: edges})[feature]


def _band_label(b: int, edges: np.ndarray, n_bands: int, n: int,
                total: int) -> str:
    lo, hi = edges[b], edges[b + 1]
    share = f"{100 * n / total:.0f}%" if total else "?"
    tag = "  (top)" if b == n_bands - 1 else "  (bottom)" if b == 0 else ""
    return f"{lo:.3g}–{hi:.3g}  n={n} ({share}){tag}"


def feature_figure(feature: str, secs: np.ndarray, vals: np.ndarray,
                   edges: np.ndarray, time_ms, band_wf: dict, meta: dict,
                   out_png: str, n_bands: int | None = None) -> str:
    """2-panel PNG: (left) feature-vs-time scatter with the y-axis split at the
    percentile band edges, points colored by band; (right) each band's mean +/-
    sd evoked waveform. Saves to *out_png*, returns the path."""
    n_bands = n_bands_of(edges) if n_bands is None else n_bands
    total = int(sum(d["n"] for d in band_wf.values())) or 1
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, n_bands))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=_BG,
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    ok = np.isfinite(vals) & np.isfinite(secs)
    s, v = secs[ok], vals[ok]
    bands = np.array([assign_band(x, edges) for x in v])
    dts = mdates.date2num([datetime.fromtimestamp(x) for x in s])
    axL.set_facecolor(_PANEL)
    for b in range(n_bands):
        m = bands == b
        if m.any():
            axL.scatter(dts[m], v[m], s=7, color=colors[b], alpha=0.6,
                        edgecolors="none")
    for e in edges[1:-1]:
        axL.axhline(e, color=_MUTED, lw=0.8, ls="--", alpha=0.5)
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axL.set_xlabel("time of day", color=_TEXT, fontsize=10)
    axL.set_ylabel(_pretty(feature), color=_TEXT, fontsize=10)
    axL.set_title(f"{meta['animal']} · {meta['channel']} · "
                  f"{meta['date']}  (n={int(ok.sum())})", color=_TEXT,
                  fontsize=11, loc="left")

    axR.set_facecolor(_PANEL)
    tm = np.asarray(time_ms, dtype=float)
    base_m = (tm >= -50) & (tm <= -5)                 # pre-stim baseline window
    art_m = np.abs(tm) <= 1.5                          # blank the stim artifact
    vis_m = (tm >= -20) & (tm <= 200) & ~art_m        # response window shown
    ys = []
    for b in sorted(band_wf):
        d = band_wf[b]
        mean = d["mean"] - (np.nanmean(d["mean"][base_m]) if base_m.any() else 0.0)
        m_plot, sd_plot = mean.copy(), d["sd"].copy()
        m_plot[art_m] = np.nan                         # gap at the artifact
        sd_plot[art_m] = np.nan
        lw = 2.4 if b in (0, n_bands - 1) else 1.4
        axR.plot(tm, m_plot, color=colors[b], lw=lw,
                 label=_band_label(b, edges, n_bands, d["n"], total))
        axR.fill_between(tm, m_plot - sd_plot, m_plot + sd_plot,
                         color=colors[b], alpha=0.07, linewidth=0)
        seg = mean[vis_m]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            ys.append(np.percentile(seg, [0.5, 99.5]))
    axR.set_xlim(-20, 200)
    if ys:
        lo = min(a[0] for a in ys); hi = max(a[1] for a in ys)
        pad = 0.15 * (hi - lo) + 1e-9
        axR.set_ylim(lo - pad, hi + pad)
    axR.axvline(0, color=_MUTED, lw=0.6, ls=":", alpha=0.5)
    axR.set_xlabel("ms since stim (artifact blanked)", color=_TEXT, fontsize=10)
    axR.set_ylabel("evoked (a.u.)", color=_TEXT, fontsize=10)
    axR.set_title(f"mean±sd response by {_pretty(feature)} band",
                  color=_TEXT, fontsize=11, loc="left")
    leg = axR.legend(fontsize=7, facecolor=_PANEL, edgecolor="#3a3a52",
                     labelcolor=_TEXT, loc="best")
    if leg:
        leg.get_frame().set_alpha(0.85)
    for ax in (axL, axR):
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a52")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out_png


def build_animal(animal: str, evoked_dir: str, day: datetime, *,
                 features=None, n_bands: int = DEFAULT_N_BANDS,
                 work_dir: str, channel_override: str | None = None) -> dict:
    """Render one 2-panel PNG per feature for *animal* on *day*. Returns
    ``{"animal", "channel", "date", "n_responses", "pngs": {feature: path}}`` or
    ``{"empty": True, "reason": ...}`` when there's nothing to show."""
    features = [f for f in (features or DEFAULT_FEATURES) if f in _ef.ALL_COLUMNS]
    day_files = list_day_files(animal, evoked_dir, day)
    if not day_files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    series = day_series(day_files, animal, features)
    channel = _d.primary_channel(animal, series, features, channel_override)
    if channel is None or channel not in series:
        return {"animal": animal, "empty": True, "reason": "no channel data"}
    d = series[channel]
    date_str = datetime(day.year, day.month, day.day).date().isoformat()
    meta = {"animal": animal, "channel": channel, "date": date_str}
    edges_by_feat = {}
    for feat in features:
        vals = d["metrics"].get(feat)
        if vals is None:
            continue
        edges = even_band_edges(vals, n_bands)
        if edges is not None:
            edges_by_feat[feat] = edges
    if not edges_by_feat:
        return {"animal": animal, "empty": True, "reason": "no bandable feature"}
    wf = band_waveforms_multi(day_files, animal, channel, edges_by_feat)   # one trace pass
    pngs: dict = {}
    for feat, edges in edges_by_feat.items():
        time_ms, band_wf = wf[feat]
        if not band_wf or time_ms is None:
            continue
        png = os.path.join(work_dir, f"evoked_{animal}_{channel}_{feat}.png")
        feature_figure(feat, d["secs"], d["metrics"][feat], edges, time_ms,
                       band_wf, meta, png)
        pngs[feat] = png
    if not pngs:
        return {"animal": animal, "empty": True, "reason": "no bandable feature"}
    return {"animal": animal, "channel": channel, "date": date_str,
            "n_responses": int(d["secs"].size), "pngs": pngs}
