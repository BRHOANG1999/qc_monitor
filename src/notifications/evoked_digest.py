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
DEFAULT_N_BANDS = 10          # deciles: 10 even-population bands, 10 traces
_MIN_PER_FEATURE = 40          # too few finite responses -> skip that feature
_PILE_FRAC = 0.02              # a boundary value held by >= this fraction = a pile-up
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


def list_window_files(animal: str, evoked_dir: str, end_day: datetime,
                      window_days: int = 1) -> list[str]:
    """The animal's *_evoked.mat files in the *window_days* ending on *end_day*
    (inclusive), oldest first. ``window_days=1`` = just *end_day* (daily);
    ``7`` = a running week (weekly counterpart)."""
    d1 = datetime(end_day.year, end_day.month, end_day.day) + timedelta(days=1)
    d0 = d1 - timedelta(days=max(1, int(window_days)))
    out = []
    for fp in list_evoked_files(evoked_dir):
        if animal not in animals_in_filename(fp):
            continue
        dt = parse_recording_dt(fp)
        if dt is not None and d0 <= dt < d1:
            out.append(fp)
    return sorted(out)


def list_day_files(animal: str, evoked_dir: str, day: datetime) -> list[str]:
    """The animal's *_evoked.mat files recorded on *day* (daily window)."""
    return list_window_files(animal, evoked_dir, day, 1)


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


def compute_bands(secs: np.ndarray, vals: np.ndarray,
                  n_bands: int = DEFAULT_N_BANDS):
    """RANK-based equal-population bands (deciles by default): each band holds
    ~n/n_bands responses. Assigning by RANK (not value edge) means a large point
    mass -- e.g. the many flat 0-amplitude responses on inactive stimuli -- is
    split evenly across the bottom bands instead of collapsing them, so you
    always get *n_bands* equal-count traces. Ties are broken by time order.

    Keyed to each response by its absolute timestamp (rounded to ms) so the trace
    pass can look up a band without re-deriving it. Returns
    ``{"band_of_ts", "bands" (per-index, -1 = non-finite), "edges" (display value
    edges), "counts", "n_bands"}`` or None when too few finite responses."""
    v = np.asarray(vals, dtype=float)
    s = np.asarray(secs, dtype=float)
    fin = np.isfinite(v) & np.isfinite(s)
    if fin.sum() < max(4 * n_bands, _MIN_PER_FEATURE):
        return None
    # A degenerate "pile-up" at a BOUNDARY value is a distinct population, not a
    # band: exact-0 flat responses (amplitude/line-length) OR the window floor
    # (peak_latency piles at its 1 ms minimum = peak-at-artifact). Split the modal
    # value out only when it sits at the min/max (where degenerate/sentinel values
    # collect) and holds a meaningful fraction -- an interior bulk mode is a real
    # band, not a pile.
    uv, uc = np.unique(v[fin], return_counts=True)
    mi = int(np.argmax(uc))
    mode_val = float(uv[mi])
    at_boundary = mode_val in (float(uv[0]), float(uv[-1]))
    if at_boundary and uc[mi] >= max(5, _PILE_FRAC * fin.sum()):
        pile, pile_val = fin & (v == mode_val), mode_val
    else:
        pile, pile_val = np.zeros(v.shape, dtype=bool), None
    nz = np.where(fin & ~pile)[0]                          # the non-pile responses
    if nz.size < max(4 * n_bands, _MIN_PER_FEATURE):
        return None
    order = nz[np.argsort(v[nz], kind="stable")]           # non-pile, value-sorted
    n = order.size
    band_sorted = np.minimum((np.arange(n) * n_bands) // n, n_bands - 1)
    bands = np.full(v.size, -1, dtype=int)
    bands[order] = band_sorted
    zidx = np.where(pile)[0]
    bands[zidx] = n_bands                                  # the pile -> the extra band
    band_of_ts = {round(float(s[i]), 3): int(bands[i])
                  for i in np.concatenate([order, zidx])}
    edges = np.quantile(v[nz], np.linspace(0.0, 1.0, n_bands + 1))
    counts = np.bincount(band_sorted, minlength=n_bands)
    return {"band_of_ts": band_of_ts, "bands": bands, "edges": edges,
            "counts": counts, "n_zero": int(pile.sum()), "pile_val": pile_val,
            "n_bands": int(n_bands)}


def _ts_key(row) -> float:
    dt = _d.parse_iso(row.get("abs_dt")) if row.get("abs_dt") else None
    return round((dt or datetime.min).timestamp(), 3)


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
                         info_by_feat: dict) -> dict:
    """ONE streaming trace pass over the *channel* recordings: match each epoch to
    its sidecar row by stim time, look up its (rank) band per feature by the
    response's timestamp, and accumulate per-band mean +/- sd waveforms.
    ``info_by_feat = {feature: compute_bands(...)}``; returns
    ``{feature: (time_ms, {band: {mean, sd, n}})}``. Reading each recording's
    traces once (not once per feature) keeps the daemon's I/O bounded."""
    accs = {f: _BandAccum(info["n_bands"] + 1)               # +1 = the zero band
            for f, info in info_by_feat.items()}
    maps = {f: info["band_of_ts"] for f, info in info_by_feat.items()}
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
            key = _ts_key(r)
            for f, m in maps.items():
                b = m.get(key, -1)
                if b >= 0:
                    accs[f].add(b, trace, time_ms)
    return {f: (acc.time_ms, acc.result()) for f, acc in accs.items()}


def band_waveforms(day_files: list[str], animal: str, channel: str,
                   feature: str, info: dict) -> tuple:
    """Single-feature convenience wrapper over ``band_waveforms_multi``."""
    return band_waveforms_multi(day_files, animal, channel, {feature: info})[feature]


def _band_label(b: int, edges: np.ndarray, n_bands: int) -> str:
    lo, hi = edges[b], edges[b + 1]
    tag = "  ← top" if b == n_bands - 1 else "  ← bottom" if b == 0 else ""
    return f"D{b + 1} [{lo:.3g}–{hi:.3g}]{tag}"


def _pile_label(info: dict, n: int) -> str:
    pv = info.get("pile_val")
    return f"pile @{pv:g} (n={n})" if pv is not None else f"pile (n={n})"


def feature_doc(feature: str) -> str:
    """Code-accurate one-line definition of *feature* (source of truth)."""
    return _ef.COLUMN_DOCS.get(feature, "")


def doc_footer(fig, feature: str) -> None:
    """Stamp 'how computed' along the bottom of a figure. Reserve room first
    with ``fig.tight_layout(rect=[0, 0.07, 1, 1])``."""
    doc = feature_doc(feature)
    if doc:
        fig.text(0.5, 0.012, f"how computed — {_pretty(feature)}: {doc}",
                 ha="center", va="bottom", color=_MUTED, fontsize=8, wrap=True)


_ZCOLOR = "#ff5c5c"                                       # the zero-value group


def _baseline(mean: np.ndarray, tm: np.ndarray) -> np.ndarray:
    base_m = (tm >= -50) & (tm <= -5)
    return mean - (np.nanmean(mean[base_m]) if base_m.any() else 0.0)


def draw_scatter(ax, feature, secs, vals, info, meta) -> None:
    """Feature-vs-time scatter colored by band, zeros highlighted, robust y."""
    n_bands, edges, bands_idx = info["n_bands"], info["edges"], info["bands"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    ax.set_facecolor(_PANEL)
    ok = bands_idx >= 0
    s, v, bb = secs[ok], vals[ok], bands_idx[ok]
    dts = mdates.date2num([datetime.fromtimestamp(x) for x in s])
    for b in range(n_bands):
        m = bb == b
        if m.any():
            ax.scatter(dts[m], v[m], s=6, color=colors[b], alpha=0.6, edgecolors="none")
    mz = bb == n_bands
    if mz.any():
        ax.scatter(dts[mz], v[mz], s=8, color=_ZCOLOR, alpha=0.7, edgecolors="none")
    for e in np.unique(edges[1:-1]):
        ax.axhline(e, color=_MUTED, lw=0.7, ls="--", alpha=0.4)
    _loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(_loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(_loc))
    vv = v[np.isfinite(v)]
    if vv.size:
        ylo, yhi = np.percentile(vv, [0.2, 99.5])
        ax.set_ylim(ylo - 0.05 * (yhi - ylo) - 1e-9, yhi + 0.05 * (yhi - ylo) + 1e-9)
    ax.set_xlabel("time", color=_TEXT, fontsize=10)
    ax.set_ylabel(_pretty(feature), color=_TEXT, fontsize=10)
    pv = info.get("pile_val")
    zc = (f" · {int(mz.sum())} @{pv:g}" if mz.any() and pv is not None
          else f" · {int(mz.sum())} pile" if mz.any() else "")
    ax.set_title(f"{meta['animal']} · {meta['channel']} · {meta['date']}  "
                 f"(n={int(ok.sum())}{zc})", color=_TEXT, fontsize=11, loc="left")


def draw_response(ax, tm, band_wf, info, *, blank=True, show_sd=False) -> None:
    """Per-band mean response (baseline-corrected; artifact blanked if *blank*),
    plus the zeros trace. sd fills when *show_sd*."""
    n_bands, edges = info["n_bands"], info["edges"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    ax.set_facecolor(_PANEL)
    art_m = np.abs(tm) <= 1.5
    vis_m = (tm >= -20) & (tm <= 200) & (~art_m if blank else np.ones(tm.shape, bool))
    ys = []

    def _plot(b, color, lw, ls="-", lbl=None):
        d = band_wf.get(b)
        if not d or d["n"] == 0:
            return
        mean = _baseline(d["mean"], tm)
        mp = mean.copy()
        if blank:
            mp[art_m] = np.nan
        ax.plot(tm, mp, color=color, lw=lw, ls=ls, label=lbl)
        if show_sd:
            sd = d["sd"].copy()
            if blank:
                sd[art_m] = np.nan
            ax.fill_between(tm, mp - sd, mp + sd, color=color, alpha=0.08, linewidth=0)
        seg = mean[vis_m]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            ys.append(np.percentile(seg, [0.5, 99.5]))

    for b in range(n_bands):
        _plot(b, colors[b], 2.4 if b in (0, n_bands - 1) else 1.3,
              lbl=_band_label(b, edges, n_bands))
    if band_wf.get(n_bands) and band_wf[n_bands]["n"]:
        _plot(n_bands, _ZCOLOR, 2.0, ls="--",
              lbl=_pile_label(info, band_wf[n_bands]["n"]))
    ax.set_xlim(-20, 200)
    if ys:
        lo = min(a[0] for a in ys); hi = max(a[1] for a in ys)
        pad = 0.15 * (hi - lo) + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
    ax.axvline(0, color=_MUTED, lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel("ms since stim" + (" (artifact blanked)" if blank else ""),
                  color=_TEXT, fontsize=10)
    ax.set_ylabel("evoked (a.u.)", color=_TEXT, fontsize=10)
    ax.legend(fontsize=6, facecolor=_PANEL, edgecolor="#3a3a52", labelcolor=_TEXT,
              loc="upper right", framealpha=0.85)


def draw_artifact_panel(ax, tm, band_wf, info) -> None:
    """UNBLANKED stim-artifact window (~-5..25 ms) for the zeros vs the top band."""
    n_bands = info["n_bands"]
    colors = plt.cm.viridis(np.linspace(0.06, 0.96, n_bands))
    ax.set_facecolor(_PANEL)
    aw = (tm >= -5) & (tm <= 25)
    d = band_wf[n_bands]
    zm = _baseline(d["mean"], tm)
    ax.plot(tm[aw], zm[aw], color=_ZCOLOR, lw=2.2, label=_pile_label(info, d["n"]))
    ax.fill_between(tm[aw], (zm - d["sd"])[aw], (zm + d["sd"])[aw],
                    color=_ZCOLOR, alpha=0.12, linewidth=0)
    if band_wf.get(n_bands - 1) and band_wf[n_bands - 1]["n"]:
        tmn = _baseline(band_wf[n_bands - 1]["mean"], tm)
        ax.plot(tm[aw], tmn[aw], color=colors[-1], lw=1.5, label=f"D{n_bands} (top)")
    ax.axvline(0, color=_MUTED, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlim(-5, 25)
    ax.set_xlabel("ms since stim (UNBLANKED)", color=_TEXT, fontsize=10)
    ax.set_ylabel("evoked (a.u.)", color=_TEXT, fontsize=10)
    ax.set_title("stim-artifact window · pile-up (unblanked)", color=_TEXT,
                 fontsize=11, loc="left")
    ax.legend(fontsize=7, facecolor=_PANEL, edgecolor="#3a3a52", labelcolor=_TEXT,
              loc="best", framealpha=0.85)


def draw_computation_panel(ax, feature, tm, band_wf, info) -> None:
    """Plot HOW the feature is measured, geometrically, on a reference response
    (the top decile's mean): the peak/trough it takes, the latency it reads, the
    windows it integrates, etc. The figure's footer carries the exact formula."""
    n_bands = info["n_bands"]
    rb = (n_bands - 1 if band_wf.get(n_bands - 1) and band_wf[n_bands - 1]["n"]
          else next((b for b in sorted(band_wf) if b < n_bands), None))
    ax.set_facecolor(_PANEL)
    if rb is None:
        ax.axis("off")
        return
    ref = _baseline(band_wf[rb]["mean"], tm)
    win = (tm >= 1) & (tm <= 200)                         # response window only
    tw, yw = tm[win], ref[win]
    ax.plot(tw, yw, color="#cfd0e0", lw=1.6, zorder=3)    # artifact (t<1) excluded
    C = "#ffd24a"
    if yw.size >= 2:
        ia, ii = int(np.argmax(yw)), int(np.argmin(yw))
        if feature in ("peak_to_trough", "peak_amplitude"):
            ax.plot(tw[ia], yw[ia], "o", color=C, ms=7)
            ax.annotate("peak", (tw[ia], yw[ia]), color=C, fontsize=8,
                        xytext=(3, 4), textcoords="offset points")
        if feature in ("peak_to_trough", "trough_amplitude"):
            ax.plot(tw[ii], yw[ii], "o", color=C, ms=7)
            ax.annotate("trough", (tw[ii], yw[ii]), color=C, fontsize=8,
                        xytext=(3, -10), textcoords="offset points")
        if feature == "peak_to_trough":
            ax.annotate("", (tw[ia], yw[ii]), (tw[ia], yw[ia]),
                        arrowprops=dict(arrowstyle="<->", color=C, lw=1.5))
            ax.annotate("max−min", (tw[ia], (yw[ia] + yw[ii]) / 2), color=C,
                        fontsize=8, xytext=(5, 0), textcoords="offset points")
        elif feature == "rms_amplitude":
            r = float(np.sqrt(np.mean(yw ** 2)))
            for s in (r, -r):
                ax.axhline(s, color=C, ls="--", lw=1)
            ax.annotate(f"±rms={r:.2g}", (tw[-1], r), color=C, fontsize=8,
                        ha="right", va="bottom")
        elif feature in ("peak_latency_ms", "trough_latency_ms"):
            k = ia if feature == "peak_latency_ms" else ii
            ax.axvline(tw[k], color=C, ls="--", lw=1.2)
            ax.plot(tw[k], yw[k], "o", color=C, ms=7)
            y0 = ax.get_ylim()[0] * 0.6
            ax.annotate("", (tw[k], y0), (0, y0),
                        arrowprops=dict(arrowstyle="<->", color=C))
            ax.annotate(f"{tw[k]:.0f} ms", (tw[k] / 2, y0), color=C, fontsize=8,
                        va="bottom", ha="center")
        elif feature == "max_slope":
            d = np.diff(yw); k = int(np.argmax(np.abs(d)))
            dx, dy = tw[k + 1] - tw[k], yw[k + 1] - yw[k]
            ax.plot([tw[k] - 4 * dx, tw[k] + 5 * dx],
                    [yw[k] - 4 * dy, yw[k] + 5 * dy], color=C, lw=2)
            ax.plot(tw[k], yw[k], "o", color=C, ms=6)
            ax.annotate("steepest Δy/dt", (tw[k], yw[k]), color=C, fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        elif feature == "log_auc":
            ax.fill_between(tw, 0, np.abs(yw), color=C, alpha=0.18)
            ax.annotate("area Σ|y|·dt", (tw[len(tw) // 2], 0), color=C,
                        fontsize=8, ha="center")
        elif feature == "early_late_ratio":
            ax.axvspan(1, 50, color="#5edc7c", alpha=0.13)
            ax.axvspan(50, 200, color="#e2a05e", alpha=0.13)
            yt = ax.get_ylim()[1]
            ax.annotate("early 1–50", (25, yt), color="#5edc7c", fontsize=8,
                        ha="center", va="top")
            ax.annotate("late 50–200", (125, yt), color="#e2a05e", fontsize=8,
                        ha="center", va="top")
        elif feature == "line_length":
            cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(yw)))])
            ax2 = ax.twinx()
            ax2.plot(tw, cum, color=C, lw=1.6, ls="--")
            ax2.set_ylabel("cumulative Σ|Δy|", color=C, fontsize=8)
            ax2.tick_params(colors=C, labelsize=7)
            for sp in ax2.spines.values():
                sp.set_color("#3a3a52")
    ax.axvline(0, color=_MUTED, lw=0.6, ls=":", alpha=0.5)
    ax.set_xlim(-2, 200)
    ax.set_xlabel("ms since stim", color=_TEXT, fontsize=10)
    ax.set_ylabel("evoked (a.u.)", color=_TEXT, fontsize=10)
    ax.set_title(f"how measured · {_pretty(feature)}", color=_TEXT,
                 fontsize=11, loc="left")


def feature_figure(feature: str, secs: np.ndarray, vals: np.ndarray,
                   info: dict, time_ms, band_wf: dict, meta: dict,
                   out_png: str) -> str:
    """Scatter | mean response by decile (+zeros) | how-measured | (if zeros)
    unblanked stim-artifact window. Saves to *out_png*, returns the path."""
    n_bands = info["n_bands"]
    tm = np.asarray(time_ms, dtype=float)
    has_zero = bool(band_wf.get(n_bands) and band_wf[n_bands]["n"] > 0)
    ncols = 4 if has_zero else 3
    widths = [1.15, 1.0, 0.95, 0.7] if has_zero else [1.15, 1.0, 0.95]
    fig, axes = plt.subplots(1, ncols, figsize=(5.7 * ncols + 1.0, 4.8),
                             facecolor=_BG, gridspec_kw={"width_ratios": widths})
    draw_scatter(axes[0], feature, secs, vals, info, meta)
    draw_response(axes[1], tm, band_wf, info, blank=True, show_sd=(n_bands <= 5))
    axes[1].set_title(f"mean±sd response by {_pretty(feature)} band",
                      color=_TEXT, fontsize=11, loc="left")
    draw_computation_panel(axes[2], feature, tm, band_wf, info)
    if has_zero:
        draw_artifact_panel(axes[3], tm, band_wf, info)
    for ax in axes:
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a52")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    doc_footer(fig, feature)
    fig.savefig(out_png, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out_png


def _date_label(end_day: datetime, window_days: int) -> str:
    end = datetime(end_day.year, end_day.month, end_day.day).date()
    if window_days <= 1:
        return end.isoformat()
    start = (datetime(end_day.year, end_day.month, end_day.day)
             - timedelta(days=window_days - 1)).date()
    return f"{start.isoformat()} – {end.isoformat()}"


def _sample_files(files: list, cap) -> list:
    """Evenly sample at most *cap* files across the window; ``None``/0 = all."""
    if not cap or cap <= 0 or len(files) <= cap:
        return files
    idx = np.linspace(0, len(files) - 1, cap).round().astype(int)
    return [files[i] for i in sorted(set(idx.tolist()))]


def build_animal(animal: str, evoked_dir: str, day: datetime, *,
                 features=None, n_bands: int = DEFAULT_N_BANDS,
                 window_days: int = 1, max_trace_files=None,
                 work_dir: str, channel_override: str | None = None) -> dict:
    """Render one 2-panel PNG per feature for *animal* over the *window_days*
    ending on *day* (1 = daily, 7 = weekly). Returns ``{"animal","channel",
    "date","n_responses","skipped","pngs":{feature: path}}`` or
    ``{"empty": True, "reason": ...}`` when there's nothing to show."""
    features = [f for f in (features or DEFAULT_FEATURES) if f in _ef.ALL_COLUMNS]
    files = list_window_files(animal, evoked_dir, day, window_days)
    if not files:
        return {"animal": animal, "empty": True, "reason": "no recordings"}
    series = day_series(files, animal, features)
    channel = _d.primary_channel(animal, series, features, channel_override)
    if channel is None or channel not in series:
        return {"animal": animal, "empty": True, "reason": "no channel data"}
    d = series[channel]
    date_str = _date_label(day, window_days)
    meta = {"animal": animal, "channel": channel, "date": date_str}
    info_by_feat, skipped = {}, []
    for feat in features:
        vals = d["metrics"].get(feat)
        info = compute_bands(d["secs"], vals, n_bands) if vals is not None else None
        if info is not None:
            info_by_feat[feat] = info
        else:
            skipped.append(feat)                         # all-NaN / too few
    if not info_by_feat:
        return {"animal": animal, "empty": True, "reason": "no bandable feature"}
    # Banding/scatter use ALL responses (cheap sidecars); the waveform pass reads
    # traces from a bounded, evenly-sampled subset of recordings (I/O-bound).
    trace_files = _sample_files(files, max_trace_files)
    wf = band_waveforms_multi(trace_files, animal, channel, info_by_feat)
    pngs: dict = {}
    for feat, info in info_by_feat.items():
        time_ms, band_wf = wf[feat]
        if not band_wf or time_ms is None:
            skipped.append(feat)
            continue
        png = os.path.join(work_dir, f"evoked_{animal}_{channel}_{feat}.png")
        feature_figure(feat, d["secs"], d["metrics"][feat], info, time_ms,
                       band_wf, meta, png)
        pngs[feat] = png
    if not pngs:
        return {"animal": animal, "empty": True, "reason": "no bandable feature"}
    return {"animal": animal, "channel": channel, "date": date_str,
            "n_responses": int(d["secs"].size), "skipped": skipped, "pngs": pngs}
