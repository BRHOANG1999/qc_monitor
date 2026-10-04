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
    axes[-1, 0].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1, 0].set_xlabel("date", color=C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · paired-pulse ratio over time "
                 f"(ISI {C.ISI_MS:.0f} ms, {bin_min:.0f}-min median ± IQR; "
                 f"white = lead seizure)", color=C.TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png
