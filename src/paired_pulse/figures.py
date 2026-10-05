"""Dark-theme paired-pulse figures: PPR over time, S1/S2/residual overlay, PPR vs
seizure proximity, and the PPR distribution."""

from __future__ import annotations

import datetime as _dt
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.dates as mdates          # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

from . import config as C                  # noqa: E402


def _savefig(fig, out, **kw):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for i in range(5):                                   # Windows errno22 retry
        try:
            fig.savefig(out, **kw); return out
        except OSError:
            if i == 4:
                raise
            time.sleep(0.3)


def _dark(ax, title=None):
    ax.set_facecolor(C.PANEL)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    ax.tick_params(colors=C.MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=C.TEXT, fontsize=10, loc="left")


def _base_note(extra=""):
    """Standard analysis-parameters footnote string (window / ISI / band)."""
    s = (f"BCH111SR · ISI {C.ISI_MS:.0f} ms · per-pulse window "
         f"{C.WIN[0]:.0f}–{C.WIN[1]:.0f} ms at each pulse's own onset · "
         f"{C.BP_LOW_HZ:.0f}–{C.BP_HIGH_HZ:.0f} Hz · PPR = feature(S2)/feature(S1)")
    return s + (f" · {extra}" if extra else "")


def _span_str(mat):
    import datetime as _d
    t = mat["t_epoch"].to_numpy(float)
    return (f"{_d.datetime.fromtimestamp(t.min()):%m-%d %H:%M}–"
            f"{_d.datetime.fromtimestamp(t.max()):%m-%d %H:%M} · {len(mat):,} pairs")


def _footnote(fig, text):
    fig.text(0.006, 0.004, text, color=C.MUTED, fontsize=6.5, ha="left", va="bottom")


def _binned(t, v, bin_sec):
    """(centers_dt, median, p25, p75) of v over fixed time bins."""
    lo, hi = np.nanmin(t), np.nanmax(t)
    edges = np.arange(lo, hi + bin_sec, bin_sec)
    which = np.clip(np.searchsorted(edges, t, "right") - 1, 0, edges.size - 2)
    cen, med, q1, q3 = [], [], [], []
    for b in range(edges.size - 1):
        vb = v[(which == b) & np.isfinite(v)]
        if vb.size:
            cen.append(_dt.datetime.fromtimestamp(edges[b] + bin_sec / 2))
            m = np.percentile(vb, [50, 25, 75]); med.append(m[0]); q1.append(m[1]); q3.append(m[2])
    return cen, np.array(med), np.array(q1), np.array(q3)


def ppr_over_time_fig(mat, out_png, *, onsets=None, bin_min=30.0) -> str:
    """One row per feature: PPR over the record (binned median + IQR), dashed line at
    1 (depression<->facilitation), seizure onsets marked (white)."""
    feats = [f for f in C.FEATURES if f"ppr_{f}" in mat.columns]
    t = mat["t_epoch"].to_numpy(float)
    fig, axes = plt.subplots(len(feats), 1, figsize=(14, 1.9 * len(feats) + 1),
                             sharex=True, facecolor=C.BG, squeeze=False)
    ons = np.sort(np.asarray(onsets, float)) if onsets is not None else np.empty(0)
    for ax, f in zip(axes[:, 0], feats):
        v = mat[f"ppr_{f}"].to_numpy(float)
        cen, med, q1, q3 = _binned(t, v, bin_min * 60)
        if cen:
            ax.fill_between(cen, q1, q3, color=C.ACCENT, alpha=0.18, lw=0)
            ax.plot(cen, med, color=C.ACCENT, lw=1.3)
        ax.axhline(1.0, color=C.MUTED, lw=0.9, ls="--")
        vf = v[np.isfinite(v)]
        for o in ons:
            if t.min() <= o <= t.max():
                ax.axvline(mdates.date2num(_dt.datetime.fromtimestamp(o)),
                           color=C.SEIZURE_COLOR, lw=1.0, zorder=3)
        _dark(ax, f"PPR {f}  (median {np.median(vf):.2f}, "
                  f"{100*np.mean(vf>1):.0f}% facilitation)")
        ax.set_ylabel("S2/S1", color=C.TEXT, fontsize=8)
    ax = axes[-1, 0]
    span_h = (t.max() - t.min()) / 3600.0
    if span_h <= 48:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3 if span_h <= 30 else 6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xlabel("date", color=C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · paired-pulse ratio over time "
                 f"(ISI {C.ISI_MS:.0f} ms, {bin_min:.0f}-min median ± IQR; "
                 f"white = lead seizure)", color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note(_span_str(mat)))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def s1_s2_overlay_fig(wf, out_png, *, n_traces=40, seed=0) -> str:
    """S1 (blue) and S2 (orange) windowed responses: mean + sample traces, and the
    overlay with the S2-S1 residual + p2p PPR in the title."""
    tw, s1, s2 = wf["twin"], wf["s1"], wf["s2"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(s1.shape[0], size=min(n_traces, s1.shape[0]), replace=False)
    m1, m2 = np.nanmean(s1, 0), np.nanmean(s2, 0)
    ppr = float(np.ptp(m2) / np.ptp(m1)) if np.ptp(m1) else np.nan
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=C.BG)
    for k in idx:
        a.plot(tw, s1[k], color=C.S1_COLOR, lw=0.4, alpha=0.18)
        b.plot(tw, s2[k], color=C.S2_COLOR, lw=0.4, alpha=0.18)
    a.plot(tw, m1, color=C.S1_COLOR, lw=2.2); _dark(a, "pulse 1 (S1)")
    b.plot(tw, m2, color=C.S2_COLOR, lw=2.2); _dark(b, "pulse 2 (S2)")
    c.plot(tw, m1, color=C.S1_COLOR, lw=2.0, label="S1")
    c.plot(tw, m2, color=C.S2_COLOR, lw=2.0, label="S2")
    c.plot(tw, m2 - m1, color=C.TEXT, lw=1.4, ls="--", label="S2 − S1 residual")
    c.axhline(0, color=C.MUTED, lw=0.6)
    _dark(c, f"overlay — p2p PPR = {ppr:.2f}")
    leg = c.legend(fontsize=8, framealpha=0.1)
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    for ax in (a, b, c):
        ax.set_xlabel("ms from pulse onset", color=C.TEXT)
    a.set_ylabel("LFP (1-500 Hz)", color=C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · paired-pulse evoked responses "
                 f"(n={s1.shape[0]} pairs; mean + {len(idx)} traces)",
                 color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note("S1 blue / S2 orange; each windowed 1–49 ms at its own "
                              "onset; residual = S2 − S1"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def ppr_vs_seizure_fig(mat, out_png, *, feature=None, cap_h=12.0) -> str:
    """PPR vs log time-to-next-lead-seizure (does paired-pulse inhibition change toward
    onset?), reusing periictal.trajectory. Placeholder if no in-window seizures."""
    from src.periictal import trajectory as _TR
    feature = feature or C.PRIMARY
    tto = mat["time_to_onset_sec"].to_numpy(float)
    v = mat[f"ppr_{feature}"].to_numpy(float)
    m = np.isfinite(tto) & (tto > C.WIN[1] / 1000.0) & (tto <= cap_h * 3600) & np.isfinite(v)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG)
    if m.sum() >= 20:
        edges = _TR.default_edges(cap_h * 3600, 2.0)
        tj = _TR.lead_time_trajectory(tto[m], v[m], np.zeros(int(m.sum())), edges,
                                      log_centers=True)
        cen = tj["centers"] / 60.0
        ax.fill_between(cen, tj["p25"], tj["p75"], color=C.ACCENT, alpha=0.18, lw=0)
        ax.plot(cen, tj["median"], "-o", color=C.ACCENT, lw=1.8, ms=4)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("min to next lead seizure (log)", color=C.TEXT)
    else:
        ax.text(0.5, 0.5, "no lead seizures within the paired-pulse window yet\n"
                "(analysis accumulates as data lands)", ha="center", va="center",
                color=C.MUTED, transform=ax.transAxes)
        ax.set_xticks([])
    ax.axhline(1.0, color=C.MUTED, lw=0.9, ls="--")
    _dark(ax, f"PPR {feature} vs time-to-seizure  (n={int(m.sum())} epochs < {cap_h:.0f} h)")
    ax.set_ylabel("PPR (S2/S1)", color=C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · paired-pulse ratio vs seizure proximity",
                 color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note(_span_str(mat)))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


_HLAB = {600: "10 min", 1800: "30 min", 3600: "1 h", 7200: "2 h"}
_HCOL = {600: "#5e7ce2", 1800: "#2ee6a6", 3600: "#e6800f", 7200: "#d62f2f"}


def metric_trend_fig(mat, cols, onset_col, out_png, *, labels=None, title="",
                     cap_h=12.0, ref1=False) -> str:
    """Pre-ictal trend: each metric vs LOG time-to-onset (median + IQR), onset at the
    right. ref1 draws the PPR=1 (equal) line."""
    from src.periictal import trajectory as _TR
    tto = mat[onset_col].to_numpy(float)
    edges = _TR.default_edges(cap_h * 3600, 2.0)
    labels = labels or {}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 4.2),
                             facecolor=C.BG, squeeze=False)
    for ax, col in zip(axes[0], cols):
        v = mat[col].to_numpy(float)
        m = (np.isfinite(tto) & (tto > C.WIN[1] / 1000.0)
             & (tto <= cap_h * 3600) & np.isfinite(v))
        if m.sum() >= 20:
            tj = _TR.lead_time_trajectory(tto[m], v[m], np.zeros(int(m.sum())),
                                          edges, log_centers=True)
            cen = tj["centers"] / 60.0
            ax.fill_between(cen, tj["p25"], tj["p75"], color=C.ACCENT, alpha=0.18, lw=0)
            ax.plot(cen, tj["median"], "-o", color=C.ACCENT, lw=1.6, ms=3)
            ax.set_xscale("log"); ax.invert_xaxis()
        else:
            ax.text(0.5, 0.5, f"n={int(m.sum())}", ha="center", va="center",
                    color=C.MUTED, transform=ax.transAxes)
        if ref1:
            ax.axhline(1.0, color=C.MUTED, lw=0.9, ls="--")
        _dark(ax, labels.get(col, col))
        ax.set_xlabel("min to onset (log)", color=C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note(f"pre-ictal trend, median ± IQR per dyadic log bin · "
                              f"{_span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def seizure_prob_vs_metric_fig(mat, cols, onset_col, out_png, *, horizons,
                               labels=None, title="", nbins=6) -> str:
    """Metric on x (quantile bins), P(event within H) on y, one curve per horizon
    (+ dotted base rate). Does a metric level predict an imminent event?"""
    tto = mat[onset_col].to_numpy(float)
    labels = labels or {}
    fig, axes = plt.subplots(1, len(cols), figsize=(3.7 * len(cols), 4.4),
                             facecolor=C.BG, squeeze=False)
    for ax, col in zip(axes[0], cols):
        x = mat[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(tto)
        xq, tq = x[ok], tto[ok]
        uniq = np.unique(xq)
        binary = uniq.size <= 2                           # e.g. pHFO 0/1
        if binary:
            cen = uniq
            groups = [xq == u for u in uniq]
        else:
            edges = np.unique(np.quantile(xq, np.linspace(0, 1, nbins + 1)))
            if edges.size < 3:
                ax.text(0.5, 0.5, "constant", ha="center", va="center",
                        color=C.MUTED, transform=ax.transAxes)
                _dark(ax, labels.get(col, col)); continue
            cen = 0.5 * (edges[:-1] + edges[1:])
            bidx = np.clip(np.searchsorted(edges, xq, "right") - 1, 0, edges.size - 2)
            groups = [bidx == b for b in range(edges.size - 1)]
        for H in horizons:
            hit = (tq > 0) & (tq <= H)
            P = [100 * np.mean(hit[g]) if g.any() else np.nan for g in groups]
            ax.plot(cen, P, "-o", color=_HCOL.get(H, C.ACCENT), lw=1.5, ms=4,
                    label=_HLAB.get(H, f"{H}s"))
            ax.axhline(100 * np.mean(hit), color=_HCOL.get(H, C.ACCENT), lw=0.7, ls=":")
        _dark(ax, labels.get(col, col))
        ax.set_xlabel(labels.get(col, col), color=C.TEXT)
        if binary and set(uniq) <= {0.0, 1.0}:
            ax.set_xticks([0, 1]); ax.set_xticklabels(["absent", "present"])
    axes[0][0].set_ylabel("P(event within H) %", color=C.TEXT)
    h, lab = axes[0][0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, lab, fontsize=8, framealpha=0.1, title="horizon",
                         loc="upper right")
        for t in leg.get_texts():
            t.set_color(C.TEXT)
        leg.get_title().set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note(f"P(next event within H | metric quantile bin); dotted "
                              f"= base rate · {_span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def periictal_trajectory_fig(mat, onsets, out_png, *, feature="ppr_peak_to_trough",
                             pre_h=2.0, post_h=1.0, bin_min=10.0, title="") -> str:
    """Peri-ictal PPR trajectory: one line per seizure + bold mean, over -pre_h..+post_h
    around onset. y=1 is equal (above = facilitation, below = depression)."""
    t = mat["t_epoch"].to_numpy(float)
    v = mat[feature].to_numpy(float)
    ons = np.sort(np.asarray(onsets, float))
    edges = np.arange(-pre_h * 60, post_h * 60 + bin_min, bin_min)
    cen = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=C.BG)
    lines = []
    for o in ons:
        rel = (t - o) / 60.0
        m = (rel >= -pre_h * 60) & (rel <= post_h * 60) & np.isfinite(v)
        if m.sum() < 5:
            continue
        bi = np.clip(np.searchsorted(edges, rel[m], "right") - 1, 0, cen.size - 1)
        line = np.array([np.median(v[m][bi == b]) if np.any(bi == b) else np.nan
                         for b in range(cen.size)])
        ax.plot(cen, line, color=C.MUTED, lw=0.9, alpha=0.55)
        lines.append(line)
    if lines:
        mean = np.nanmean(np.vstack(lines), axis=0)
        ax.plot(cen, mean, color=C.S2_COLOR, lw=2.8, label=f"mean ({len(lines)} seizures)")
    ax.axhline(1.0, color=C.SEIZURE_COLOR, lw=1.1, ls="--")
    ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.3)
    ax.text(0.015, 0.97, "facilitation ↑", transform=ax.transAxes, color=C.FACIL_COLOR,
            fontsize=9, va="top")
    ax.text(0.015, 0.03, "depression ↓", transform=ax.transAxes, color=C.DEPR_COLOR,
            fontsize=9, va="bottom")
    _dark(ax, title)
    ax.set_xlabel("minutes from seizure onset (− pre-ictal / + post-ictal)", color=C.TEXT)
    ax.set_ylabel(f"PPR ({feature.replace('ppr_', '')})", color=C.TEXT)
    leg = ax.legend(fontsize=8, framealpha=0.1, loc="upper right")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    _footnote(fig, _base_note(f"{bin_min:.0f}-min bins; per-seizure median then mean "
                              f"across seizures · {_span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def example_pairs_fig(ex, out_png, *, title="") -> str:
    """Grid of individual paired-pulse epochs: S1 (blue) + S2 (orange) overlaid, PPR +
    timestamp per panel. Example evoked responses across the window."""
    tw = ex["twin"]; pairs = ex["pairs"]
    n = len(pairs)
    ncol = 4
    nrow = int(np.ceil(n / ncol)) if n else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.3 * nrow + 0.6),
                             facecolor=C.BG, squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < n:
            pr = pairs[i]
            ax.plot(tw, pr["s1"], color=C.S1_COLOR, lw=1.3)
            ax.plot(tw, pr["s2"], color=C.S2_COLOR, lw=1.3)
            ax.axhline(0, color=C.MUTED, lw=0.5)
            _dark(ax, f"{pr['label']} · PPR {pr['ppr']:.2f}")
            if i // ncol == nrow - 1:
                ax.set_xlabel("ms from onset", color=C.TEXT, fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note("S1 blue / S2 orange; individual epochs sampled across "
                              "the window"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def waveform_by_bin_fig(wb, out_png, *, onset_set="lead", which="s2", title="",
                        min_n=30) -> str:
    """Averaged evoked waveform per LOG lead-time bin, overlaid and color-coded by
    time-to-onset. which='s2' = pulse-2 response; which='resid' = S2 − S1 residual."""
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    tw, cen = wb["twin"], wb["centers"]
    data = wb[onset_set][which]; cnt = wb[onset_set]["cnt"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=C.BG)
    if data is not None:
        valid = np.where((cnt >= min_n) & np.all(np.isfinite(data), axis=1))[0]
        if valid.size:
            norm = _mc.LogNorm(vmin=max(cen[valid].min(), 1e-2), vmax=cen[valid].max())
            cmap = _cm.turbo
            for b in valid:
                ax.plot(tw, data[b], color=cmap(norm(cen[b])), lw=1.5)
            sm = _cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            cb = fig.colorbar(sm, ax=ax)
            cb.set_label("min to onset (log; blue=near, red=far)", color=C.TEXT, fontsize=8)
            cb.ax.tick_params(colors=C.MUTED, labelsize=7); cb.outline.set_edgecolor(C.MUTED)
    ax.axhline(0, color=C.MUTED, lw=0.6)
    _dark(ax, title)
    ax.set_xlabel("ms from pulse onset", color=C.TEXT)
    ax.set_ylabel("S2 LFP (1–500 Hz)" if which == "s2" else "S2 − S1 residual",
                  color=C.TEXT)
    _footnote(fig, _base_note(f"{onset_set} seizures · averaged per dyadic log "
                              f"lead-time bin (n≥{min_n}) · pre-ictal only (on/post-onset "
                              f"excluded)"))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _ctraj_matrix(mat, onsets, feature, pre_min, post_min, bin_min):
    import datetime as _dt
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    ons = np.sort(np.asarray(onsets, float))
    edges = np.arange(-pre_min, post_min + bin_min, bin_min)
    M = np.full((ons.size, edges.size - 1), np.nan)
    for r, o in enumerate(ons):
        rel = (t - o) / 60.0
        for c in range(edges.size - 1):
            s = v[(rel >= edges[c]) & (rel < edges[c + 1])]
            s = s[np.isfinite(s)]
            if s.size:
                M[r, c] = np.median(s)
    labels = [_dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M") for o in ons]
    return M, edges, labels


def continuous_trajectory_heatmap(mat, onsets, out_png, *, feature, pre_min=120.0,
                                  post_min=20.0, bin_min=10.0, diverging=False,
                                  title="") -> str:
    """Per-seizure peri-ictal heatmap of a CONTINUOUS measure (one row per seizure,
    x = min from onset, colour = median feature per 10-min bin). Replaces the discrete
    k-means state coloring. diverging=True centers the colormap at 1 (for PPR)."""
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    M, edges, labels = _ctraj_matrix(mat, onsets, feature, pre_min, post_min, bin_min)
    Mm = np.ma.masked_invalid(M)
    fin = M[np.isfinite(M)]
    if diverging and fin.size:
        d = np.nanpercentile(np.abs(fin - 1.0), 98)
        norm = _mc.TwoSlopeNorm(vcenter=1.0, vmin=1 - d, vmax=1 + d); cmap = _cm.RdBu_r
    else:
        lo, hi = (np.nanpercentile(fin, [2, 98]) if fin.size else (0, 1))
        norm = _mc.Normalize(vmin=lo, vmax=hi); cmap = _cm.viridis
    cmap = cmap.copy(); cmap.set_bad(C.NODATA_COLOR)
    fig, ax = plt.subplots(figsize=(13, 0.4 * M.shape[0] + 2.4), facecolor=C.BG)
    im = ax.imshow(Mm, aspect="auto", cmap=cmap, norm=norm, origin="upper",
                   extent=[edges[0], edges[-1], M.shape[0] - 0.5, -0.5],
                   interpolation="nearest")
    ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.6)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(labels, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(feature.replace("_", " ") + (" (1 = equal)" if diverging else ""),
                 color=C.TEXT, fontsize=8)
    cb.ax.tick_params(colors=C.MUTED, labelsize=7); cb.outline.set_edgecolor(C.MUTED)
    _dark(ax, title)
    ax.set_xlabel("minutes from seizure onset", color=C.TEXT)
    ax.set_ylabel("seizure", color=C.TEXT)
    _footnote(fig, _base_note(f"{bin_min:.0f}-min bins; median per bin; white line = onset"))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def continuous_timeline_fig(mat, onsets, out_png, *, feature, bin_min=10.0,
                            diverging=False, title="") -> str:
    """Continuous measure over the whole paired-pulse record as a colour strip (median
    per 10-min bin), seizure onsets marked. The continuous-measure analogue of the
    discrete state timeline."""
    import datetime as _dt
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    import matplotlib.dates as _md
    t = mat["t_epoch"].to_numpy(float); v = mat[feature].to_numpy(float)
    lo, hi = t.min(), t.max()
    edges = np.arange(lo, hi + bin_min * 60, bin_min * 60)
    cen = edges[:-1] + bin_min * 30
    which = np.clip(np.searchsorted(edges, t, "right") - 1, 0, cen.size - 1)
    row = np.array([np.median(v[(which == c) & np.isfinite(v)])
                    if np.any((which == c) & np.isfinite(v)) else np.nan
                    for c in range(cen.size)])
    fin = row[np.isfinite(row)]
    if diverging and fin.size:
        d = np.nanpercentile(np.abs(fin - 1.0), 98)
        norm = _mc.TwoSlopeNorm(vcenter=1.0, vmin=1 - d, vmax=1 + d); cmap = _cm.RdBu_r
    else:
        q = np.nanpercentile(fin, [2, 98]) if fin.size else (0, 1)
        norm = _mc.Normalize(vmin=q[0], vmax=q[1]); cmap = _cm.viridis
    rgba = _cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(row)
    rgba[~np.isfinite(row)] = _mc.to_rgba(C.NODATA_COLOR)
    fig, ax = plt.subplots(figsize=(15, 3.0), facecolor=C.BG)
    ax.imshow(rgba[None, :, :], aspect="auto", origin="lower",
              extent=[_md.date2num(_dt.datetime.fromtimestamp(cen[0])),
                      _md.date2num(_dt.datetime.fromtimestamp(cen[-1])), 0, 1],
              interpolation="nearest")
    for o in np.sort(np.asarray(onsets, float)):
        if lo <= o <= hi:
            ax.axvline(_md.date2num(_dt.datetime.fromtimestamp(o)),
                       color=C.SEIZURE_COLOR, lw=1.1)
    ax.set_yticks([]); ax.xaxis_date()
    ax.xaxis.set_major_locator(_md.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(_md.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(colors=C.MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    sm = _cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label(feature.replace("_", " ") + (" (1 = equal)" if diverging else ""),
                 color=C.TEXT, fontsize=8)
    cb.ax.tick_params(colors=C.MUTED, labelsize=7); cb.outline.set_edgecolor(C.MUTED)
    ax.set_title(title, color=C.TEXT, fontsize=12, loc="left")
    ax.set_xlabel("time", color=C.TEXT)
    _footnote(fig, _base_note(f"{bin_min:.0f}-min bins; median per bin; "
                              f"white = seizure onset · {_span_str(mat)}"))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def trend_null_fig(results, out_png, *, labels=None, title="", ref1=False) -> str:
    """Pre-ictal trend vs circular-shift null, per metric (small multiples). Observed
    bin-mean (accent) + shift-null 95% band + null median; bins where observed exits
    the band (p<0.05) marked. x = log min to onset (onset at right)."""
    labels = labels or {}
    cols = list(results)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 4.3),
                             facecolor=C.BG, squeeze=False)
    for ax, col in zip(axes[0], cols):
        r = results[col]; cen = r["centers"]
        ax.fill_between(cen, r["lo"], r["hi"], color=C.MUTED, alpha=0.28,
                        label="shift-null 95%")
        ax.plot(cen, r["med"], color=C.MUTED, lw=0.9, ls="--")
        ax.plot(cen, r["obs"], "-o", color=C.ACCENT, lw=1.7, ms=3, label="observed")
        sig = r["p"] < 0.05
        if sig.any():
            ax.plot(cen[sig], r["obs"][sig], "o", color=C.FACIL_COLOR, ms=6,
                    label="p<0.05")
        if ref1:
            ax.axhline(1.0, color=C.SEIZURE_COLOR, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.invert_xaxis()
        _dark(ax, labels.get(col, col))
        ax.set_xlabel("min to onset (log)", color=C.TEXT)
    h, lab = axes[0][0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, lab, fontsize=8, framealpha=0.1, loc="upper right")
        for t in leg.get_texts():
            t.set_color(C.TEXT)
    fig.suptitle(title, color=C.TEXT, fontsize=12)
    ns = next(iter(results.values()))["n_surr"] if results else 0
    _footnote(fig, _base_note(f"circular-shift null ({ns} onset shifts, wrapped, ≥3 h "
                              f"floor); observed bin-mean vs null band"))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def trajectory_null_fig(res, out_png, *, title="") -> str:
    """Peri-ictal PPR trajectory vs circular-shift null: observed mean + shift-null 95%
    band + null median, y=1 = equal. Bins where observed exits the band (p<0.05) are
    marked (per-bin, uncorrected)."""
    cen = res["centers"]
    fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=C.BG)
    ax.fill_between(cen, res["lo"], res["hi"], color=C.MUTED, alpha=0.3,
                    label="shift-null 95%")
    ax.plot(cen, res["med"], color=C.MUTED, lw=1.0, ls="--", label="null median")
    ax.plot(cen, res["obs"], color=C.S2_COLOR, lw=2.6, label="observed mean")
    sig = res["p"] < 0.05
    if sig.any():
        ax.plot(cen[sig], res["obs"][sig], "o", color=C.FACIL_COLOR, ms=6,
                label="p<0.05 (per-bin)")
    ax.axhline(1.0, color=C.SEIZURE_COLOR, lw=1.0, ls="--")
    ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.2)
    _dark(ax, title)
    ax.set_xlabel("minutes from seizure onset (− pre-ictal / + post-ictal)", color=C.TEXT)
    ax.set_ylabel(f"PPR ({res['feature'].replace('ppr_', '')})", color=C.TEXT)
    leg = ax.legend(fontsize=8, framealpha=0.1, loc="upper right")
    for tt in leg.get_texts():
        tt.set_color(C.TEXT)
    _footnote(fig, _base_note(f"{res['bin_min']:.0f}-min bins · circular-shift null "
                              f"({res['n_surr']} shifts, wrapped, ≥3 h) · "
                              f"n={res['n_onsets']} seizures"))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def ppr_distribution_fig(mat, out_png) -> str:
    """Per-feature PPR histogram: facilitation (>1) vs depression (<1)."""
    feats = [f for f in C.FEATURES if f"ppr_{f}" in mat.columns]
    fig, axes = plt.subplots(1, len(feats), figsize=(3.4 * len(feats), 4),
                             facecolor=C.BG, squeeze=False)
    for ax, f in zip(axes[0], feats):
        v = mat[f"ppr_{f}"].to_numpy(float); v = v[np.isfinite(v)]
        v = v[(v > 0) & (v < np.nanpercentile(v, 99.5))]
        ax.hist(v, bins=50, color=C.ACCENT, alpha=0.85)
        ax.axvline(1.0, color=C.MUTED, lw=1.0, ls="--")
        ax.axvline(float(np.median(v)), color=C.FACIL_COLOR, lw=1.6)
        _dark(ax, f"{f}\nmed {np.median(v):.2f} · {100*np.mean(v>1):.0f}% facil")
        ax.set_xlabel("PPR (S2/S1)", color=C.TEXT)
    axes[0][0].set_ylabel("pairs", color=C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · paired-pulse ratio distribution "
                 f"(ISI {C.ISI_MS:.0f} ms, n={len(mat)} pairs)", color=C.TEXT, fontsize=12)
    _footnote(fig, _base_note(_span_str(mat)))
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png
