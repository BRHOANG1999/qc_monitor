"""Dark-theme figures for the pre-ictal biomarker rebuild.

Phase-0 anchor figure (PCA states + state profiles + occupancy) to confirm the
re-derived states match the original 05b/05d. Later phases add the occupancy
null-comparison redesign and the null-method intermediaries.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.patches import Patch        # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from . import config as C                   # noqa: E402


def _fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=C.BG)
    return fig, ax


def _savefig(_fig, out_png, **kw):
    """savefig with retry — the Windows data drive intermittently throws
    OSError(22) mid-write; a short retry clears it."""
    import time as _t
    kw.setdefault("facecolor", C.BG)
    kw.setdefault("dpi", 130)
    err = None
    for _ in range(5):
        try:
            _fig.savefig(out_png, **kw)
            return out_png
        except OSError as e:                          # flaky I/O (Errno 22)
            err = e
            _t.sleep(0.6)
    raise err


def _dark(ax, title=None):
    ax.set_facecolor(C.PANEL)
    for sp in ax.spines.values():
        sp.set_color(C.MUTED)
    ax.tick_params(colors=C.MUTED, labelsize=8)
    ax.xaxis.label.set_color(C.TEXT)
    ax.yaxis.label.set_color(C.TEXT)
    if title:
        ax.set_title(title, color=C.TEXT, fontsize=10, loc="left")


def _state_color(s):
    return C.STATE_COLORS[int(s) % len(C.STATE_COLORS)]


def pca_scatter(ax, model, highlight_mask=None, pcx=0, pcy=1, *,
                plot_cap=40000, hl_cap=900, seed=0):
    """PC{pcx+1} vs PC{pcy+1}, colored by state; highlight_mask drawn as small
    white dots. Both layers are plot-subsampled so a dense cloud stays legible."""
    df = model.df
    xs = df[f"pc{pcx + 1}"].to_numpy()
    ys = df[f"pc{pcy + 1}"].to_numpy()
    st = df["state"].to_numpy()
    rng = np.random.default_rng(seed)
    for s in range(C.K_STATES):
        idx = np.where(st == s)[0]
        if idx.size > plot_cap:
            idx = rng.choice(idx, plot_cap, replace=False)
        ax.scatter(xs[idx], ys[idx], s=5, c=_state_color(s), alpha=0.35,
                   linewidths=0, label=f"state {s} · {model.labels.get(s, '')}")
    if highlight_mask is not None and np.any(highlight_mask):
        hi = np.where(highlight_mask)[0]
        if hi.size > hl_cap:
            hi = rng.choice(hi, hl_cap, replace=False)
        ax.scatter(xs[hi], ys[hi], s=9, c=C.SEIZURE_COLOR, alpha=0.55,
                   linewidths=0, zorder=6,
                   label=f"pre-ictal ≤30 min (n={int(highlight_mask.sum())})")
    _dark(ax, f"states (k={C.K_STATES}) on {len(model.explained_var)} PCs · "
              f"PC{pcx+1} vs PC{pcy+1}")
    ax.set_xlabel(f"PC{pcx + 1}")
    ax.set_ylabel(f"PC{pcy + 1}")
    leg = ax.legend(fontsize=6.5, framealpha=0.15, loc="upper right")
    for t in leg.get_texts():
        t.set_color(C.TEXT)


def state_heatmap(ax, model):
    """State x feature mean z-score profile."""
    P = model.profiles
    Z = P.to_numpy(float)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(len(P.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in P.columns], rotation=35,
                       ha="right", fontsize=7)
    ax.set_yticks(range(len(P.index)))
    ax.set_yticklabels([f"state {s}" for s in P.index], fontsize=8)
    _dark(ax, "state metric profiles (mean z)")
    ax.tick_params(colors=C.MUTED)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045)
    cb.ax.tick_params(colors=C.MUTED, labelsize=7)


def occupancy_bars(ax, occ, model):
    """Pre-ictal vs clean-baseline occupancy per state (grouped bars), each bar =
    % of THAT group's own epochs (gold bars sum to 100%, blue bars sum to 100%).
    The ``N×`` label over each state is the enrichment = pre-ictal ÷ baseline
    fraction (1.0× = no change; >1 = more likely pre-ictally)."""
    k = C.K_STATES
    x = np.arange(k)
    w = 0.38
    pre = np.asarray(occ["preictal"], float)
    base = np.asarray(occ["baseline"], float)
    ax.bar(x - w / 2, pre, w, color="#f2c744",
           label=f"pre-ictal {int(C.PREICTAL_SEC//60)} min (n={occ['n_pre']})")
    ax.bar(x + w / 2, base, w, color="#5b8def",
           label=f"clean baseline ≥{int(C.BASELINE_MIN_SEC//3600)} h "
                 f"(n={occ['n_base']})")
    ymax = max(pre.max(), base.max())
    for i in range(k):
        fc = pre[i] / base[i] if base[i] > 0 else np.nan
        y = max(pre[i], base[i])
        ax.text(i, y + 0.012 * ymax, f"{fc:.2f}×" if np.isfinite(fc) else "—",
                ha="center", va="bottom", fontsize=8,
                color="#f0a500" if (np.isfinite(fc) and abs(fc - 1) > 0.15)
                else C.MUTED)
        # epoch COUNT inside each bar (pct × group total), rotated
        npre_i = int(round(pre[i] / 100.0 * occ["n_pre"]))
        nbase_i = int(round(base[i] / 100.0 * occ["n_base"]))
        for xoff, bar, cnt, col in ((-w / 2, pre[i], npre_i, "#1a1a1a"),
                                    (w / 2, base[i], nbase_i, "#f0f0f5")):
            inside = bar > 0.10 * ymax
            ax.text(i + xoff, bar / 2 if inside else bar + 0.004 * ymax,
                    f"{cnt:,} = {bar:.0f}%", ha="center",
                    va="center" if inside else "bottom", rotation=90,
                    fontsize=6.5, color=col if inside else C.MUTED)
    _dark(ax, f"occupancy = % WITHIN each group (pre n={occ['n_pre']:,} vs "
              f"baseline n={occ['n_base']:,}, ~{occ['n_base']/max(1,occ['n_pre']):.0f}× "
              f"bigger) · N× = ratio of the %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"state {s}" for s in range(k)])
    ax.set_xlabel("state")
    ax.set_ylabel("% of each group's own epochs")
    ax.set_ylim(0, max(pre.max(), base.max()) * 1.18)
    leg = ax.legend(fontsize=7.5, framealpha=0.15, loc="upper right")
    for t in leg.get_texts():
        t.set_color(C.TEXT)


def anchor_figure(model, occ, highlight_mask, out_png: str) -> str:
    """Three-panel anchor: PCA scatter, state profiles, occupancy bars."""
    fig = plt.figure(figsize=(16, 4.6), facecolor=C.BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0], wspace=0.3)
    pca_scatter(fig.add_subplot(gs[0, 0]), model, highlight_mask)
    state_heatmap(fig.add_subplot(gs[0, 1]), model)
    occupancy_bars(fig.add_subplot(gs[0, 2]), occ, model)
    ev = model.explained_var
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · PCA/k-means state space "
                 f"({len(ev)} PCs, {100*ev.sum():.0f}% var) · anchor check",
                 color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
#  Phase B/C — occupancy null-vs-observed comparison + null-method figures
# --------------------------------------------------------------------------- #

def occupancy_null_compare(null_res, out_png: str, *, baseline=None) -> str:
    """Redesigned, easy-to-read null-vs-observed occupancy.

    Left: per state, the circular-shift null distribution as a violin (grey) with
    its 95% band, the baseline occupancy (blue line), and the OBSERVED pre-ictal
    occupancy as a gold diamond + p. Right: the standardised effect
    (observed - null median) / null SD per state, with the +/-1.96 sigma band --
    so "is the pre-ictal bar outside chance?" reads at a glance."""
    k = null_res["null"].shape[1]
    obs, med = null_res["observed"], null_res["median"]
    null = null_res["null"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=C.BG,
                                 gridspec_kw={"width_ratios": [1.6, 1.0]})
    # --- left: violins of the null + observed dots ---
    data = [null[np.isfinite(null[:, j]), j] for j in range(k)]
    parts = a1.violinplot(data, positions=range(k), widths=0.8,
                          showmeans=False, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(C.MUTED); b.set_alpha(0.28); b.set_edgecolor("none")
    for j in range(k):
        a1.plot([j - 0.4, j + 0.4], [null_res["lo"][j]] * 2, color=C.MUTED, lw=0.8)
        a1.plot([j - 0.4, j + 0.4], [null_res["hi"][j]] * 2, color=C.MUTED, lw=0.8)
        a1.plot(j, med[j], "_", color="#ff6b6b", ms=18, mew=2, zorder=4)
        if baseline is not None:
            a1.plot([j - 0.35, j + 0.35], [baseline[j]] * 2, color="#5b8def",
                    lw=2.2, zorder=3)
        a1.plot(j, obs[j], "D", color="#f2c744", ms=11, zorder=5,
                markeredgecolor="#000", mew=0.5)
        a1.text(j, obs[j], f"  p={null_res['p'][j]:.2f}", color=C.TEXT,
                fontsize=8, va="center")
    _dark(a1, "pre-ictal occupancy vs circular-shift null (per state)")
    a1.set_xticks(range(k)); a1.set_xticklabels([f"state {s}" for s in range(k)])
    a1.set_ylabel("% of pre-ictal epochs in state")
    handles = [plt.Line2D([0], [0], marker="D", color="#f2c744", ls="",
                          label="observed pre-ictal"),
               plt.Line2D([0], [0], marker="_", color="#ff6b6b", ls="",
                          mew=2, ms=14, label="null median"),
               Patch(facecolor=C.MUTED, alpha=0.3, label="null distribution")]
    if baseline is not None:
        handles.append(plt.Line2D([0], [0], color="#5b8def", lw=2.2,
                                  label="clean baseline"))
    leg = a1.legend(handles=handles, fontsize=7.5, framealpha=0.1, loc="best")
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    # --- right: standardised effect (null sigma) ---
    sd = np.nanstd(null, axis=0)
    z = (obs - med) / np.where(sd > 0, sd, np.nan)
    a2.axhspan(-1.96, 1.96, color=C.MUTED, alpha=0.18)
    a2.axhline(0, color=C.MUTED, lw=0.8)
    a2.bar(range(k), z, color=["#f2c744" if abs(v) <= 1.96 else "#d62f2f"
                               for v in z], width=0.6)
    _dark(a2, "effect in null SDs  (inside ±1.96 = chance)")
    a2.set_xticks(range(k)); a2.set_xticklabels([f"state {s}" for s in range(k)])
    a2.set_ylabel("(observed − null median) / null SD")
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal STATE OCCUPANCY vs "
                 f"circular-shift null (N={null_res['n_surr']} shuffles)",
                 color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def null_schematic(df, onsets, out_png: str, *, n_examples=3, seed=1) -> str:
    """Methods figure: how the circular-shift surrogate is built. Row 0 = true
    onsets + 30-min pre-windows; rows 1..n = example shifted onset sets (wrapped),
    showing the shuffle preserves spacing and only moves the alignment."""
    import datetime as _dt
    t = df["t_epoch"].astype(float)
    lo, hi = t.min(), t.max(); span = hi - lo
    ons = np.sort(np.asarray(onsets, float))
    rng = np.random.default_rng(seed)
    rows = [("observed", ons, "#ffffff")]
    for i in range(n_examples):
        sh = lo + ((ons - lo + rng.uniform(0, span)) % span)
        rows.append((f"surrogate {i+1}", np.sort(sh), C.STATE_COLORS[i % 4]))
    fig, ax = plt.subplots(figsize=(14, 3.6), facecolor=C.BG)
    _dark(ax, "circular-shift null: rotate ALL onsets by a random offset, wrap "
              "around the record, re-collect 30-min pre-windows, recompute "
              "occupancy (×N)")
    for r, (lab, arr, col) in enumerate(rows):
        y = len(rows) - 1 - r
        ax.hlines(y, mdates_num(lo), mdates_num(hi), color=C.MUTED, lw=0.6, alpha=0.5)
        for o in arr:
            ax.vlines(mdates_num(o), y - 0.32, y + 0.32, color=col, lw=1.6)
            ax.add_patch(plt.Rectangle(
                (mdates_num(o - C.PREICTAL_SEC), y - 0.18),
                mdates_num(o) - mdates_num(o - C.PREICTAL_SEC), 0.36,
                color=col, alpha=0.30, lw=0))
        ax.text(mdates_num(lo) - 0.03 * (mdates_num(hi) - mdates_num(lo)), y, lab,
                color=C.TEXT, fontsize=8, ha="right", va="center")
    ax.set_yticks([]); ax.set_ylim(-0.6, len(rows) - 0.4)
    import matplotlib.dates as _md
    ax.xaxis_date(); ax.xaxis.set_major_formatter(_md.DateFormatter("%m-%d"))
    ax.set_xlabel("record time", color=C.TEXT)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def null_worked_example(null_res, out_png: str) -> str:
    """2x2 histograms: per state, the null occupancy distribution with the
    observed value (gold line) and p -- the honest test, shown not just stated."""
    k = null_res["null"].shape[1]
    obs = null_res["observed"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=C.BG)
    for j, ax in enumerate(axes.ravel()[:k]):
        x = null_res["null"][np.isfinite(null_res["null"][:, j]), j]
        ax.hist(x, bins=40, color=C.MUTED, alpha=0.6)
        ax.axvline(null_res["median"][j], color="#ff6b6b", lw=1.5, label="null median")
        ax.axvline(null_res["lo"][j], color=C.MUTED, lw=0.8, ls="--")
        ax.axvline(null_res["hi"][j], color=C.MUTED, lw=0.8, ls="--", label="null 95%")
        ax.axvline(obs[j], color="#f2c744", lw=2.4, label="observed")
        _dark(ax, f"state {j} · p={null_res['p'][j]:.2f}")
        ax.set_xlabel("% of pre-ictal epochs in state")
    leg = axes[0, 0].legend(fontsize=7.5, framealpha=0.1)
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · occupancy null distributions "
                 f"(observed inside the null = chance)", color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def mdates_num(epoch):
    import datetime as _dt
    import matplotlib.dates as _md
    return _md.date2num(_dt.datetime.fromtimestamp(float(epoch)))


def incremental_test_fig(res, out_png: str) -> str:
    """Does evoked add beyond clock+tsl? Left: baseline vs baseline+evoked LOSO-AUC.
    Right: null distribution of the increment ΔAUC with the observed increment."""
    nd = res["null_delta"][np.isfinite(res["null_delta"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=C.BG)
    a1.bar([0, 1], [res["auc_base"], res["auc_full"]],
           color=[C.MUTED, C.ACCENT], width=0.6)
    a1.axhline(0.5, color=C.MUTED, lw=1.0, ls=":")
    _dark(a1, "LOSO-AUC: timing vs timing + evoked")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["clock + time-since-seizure", "+ evoked (31f)"], fontsize=9)
    a1.set_ylabel("ROC-AUC")
    for i, v in enumerate([res["auc_base"], res["auc_full"]]):
        a1.text(i, v + 0.01, f"{v:.3f}", ha="center", color=C.TEXT, fontsize=9)
    a2.hist(nd, bins=40, color=C.MUTED, alpha=0.6, label="shift-null ΔAUC")
    a2.axvline(0, color=C.MUTED, lw=1.0, ls=":")
    a2.axvline(res["null_delta_med"], color="#ff6b6b", lw=1.3, label="null median")
    a2.axvline(res["delta"], color="#f2c744", lw=2.6,
               label=f"observed Δ={res['delta']:+.3f}")
    _dark(a2, f"evoked increment vs null · p={res['p']:.3f}")
    a2.set_xlabel("ΔAUC (evoked beyond timing)"); a2.set_ylabel("surrogate count")
    leg = a2.legend(fontsize=8, framealpha=0.1)
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · does the evoked response add beyond "
                 "clock-hour + time-since-seizure?", color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def classifier_null_fig(res, out_png: str) -> str:
    """Pre-ictal-vs-baseline classifier: null AUC distribution (circular-shift)
    with the observed LOSO-AUC and chance marked. Observed left of the null's
    upper tail = not separable above chance."""
    null = res["null"][np.isfinite(res["null"])]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG)
    ax.hist(null, bins=40, color=C.MUTED, alpha=0.6, label="circular-shift null")
    ax.axvline(0.5, color=C.MUTED, lw=1.0, ls=":", label="chance (0.5)")
    ax.axvline(res["null_median"], color="#ff6b6b", lw=1.4, label="null median")
    ax.axvline(res["null_hi"], color="#ff6b6b", lw=1.0, ls="--", label="null 95th")
    ax.axvline(res["auc"], color="#f2c744", lw=2.6, label=f"observed AUC={res['auc']:.3f}")
    _dark(ax, f"{C.ANIMAL} · {C.CHANNEL} · pre-ictal vs baseline classifier "
              f"(LOSO, {res['n_features']} features)")
    ax.set_xlabel("ROC-AUC (leave-one-seizure-out)")
    ax.set_ylabel("surrogate count")
    ax.text(0.02, 0.96, f"p = {res['p']:.3f}\nn_pre={res['n_pre']}  "
            f"n_base={res['n_base']}\nsurrogates={res['n_surr']}",
            transform=ax.transAxes, va="top", color=C.TEXT, fontsize=9,
            bbox=dict(boxstyle="round", fc=C.PANEL, ec=C.MUTED, alpha=0.8))
    leg = ax.legend(fontsize=8, framealpha=0.1, loc="upper right")
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def cluster_traces_fig(waveforms, time_ms, states, out_png, *, k=None,
                       n_traces=25, seed=0, title_suffix="") -> str:
    """Per state: the MEAN evoked waveform (thick white) over *n_traces* random
    example traces (thin, state colour). Shows what each cluster's response looks
    like."""
    k = int(k or C.K_STATES)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, k, figsize=(4.0 * k, 4.2), facecolor=C.BG,
                             sharey=True)
    axes = np.atleast_1d(axes)
    for s in range(k):
        ax = axes[s]
        idx = np.where(states == s)[0]
        if idx.size:
            pick = rng.choice(idx, min(n_traces, idx.size), replace=False)
            for j in pick:
                ax.plot(time_ms, waveforms[j], color=C.STATE_COLORS[s], lw=0.5,
                        alpha=0.30)
            ax.plot(time_ms, np.nanmean(waveforms[idx], axis=0), color="#ffffff",
                    lw=2.4, label="mean")
        _dark(ax, f"state {s} · {C.STATE_COLORS[s] and ''}n={idx.size:,}")
        ax.set_xlabel("ms since stim")
    axes[0].set_ylabel("evoked response (gain-norm, a.u.)")
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · per-state evoked waveform "
                 f"(mean + {n_traces} random traces){title_suffix}",
                 color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _savefig(fig, out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _hlabel(sec):
    return f"{int(sec)}s" if sec < 60 else f"{int(sec//60)}min" if sec < 3600 \
        else f"{int(sec//3600)}h"


def seizure_prob_fig(res, out_png: str, *, title_suffix="") -> str:
    """Forward predictive map. Left: P(seizure within H | state) vs horizon, one
    line per state + dashed marginal base rate. Right: predictive LIFT
    (P|state ÷ base rate); >1 = state carries above-chance seizure risk."""
    P, base, hs = res["P"], res["base"], res["horizons_sec"]
    k = P.shape[0]
    xl = [_hlabel(s) for s in hs]
    x = np.arange(len(hs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=C.BG)
    for s in range(k):
        a1.plot(x, P[s], "-o", color=C.STATE_COLORS[s], lw=1.8, ms=4,
                label=f"state {s} (n={res['n_state'][s]})")
    a1.plot(x, base, "--", color=C.MUTED, lw=1.6, label="base rate (marginal)")
    _dark(a1, "P(seizure within H | state)")
    a1.set_xticks(x); a1.set_xticklabels(xl)
    a1.set_xlabel("horizon H"); a1.set_ylabel("P(seizure within H) %")
    leg = a1.legend(fontsize=7.5, framealpha=0.1, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    for s in range(k):
        lift = P[s] / np.where(base > 0, base, np.nan)
        a2.plot(x, lift, "-o", color=C.STATE_COLORS[s], lw=1.8, ms=4,
                label=f"state {s}")
    a2.axhline(1.0, color=C.MUTED, lw=1.2, ls="--")
    _dark(a2, "predictive lift  (P|state ÷ base rate; >1 = above chance)")
    a2.set_xticks(x); a2.set_xticklabels(xl)
    a2.set_xlabel("horizon H"); a2.set_ylabel("lift ×")
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · forward seizure-probability by "
                 f"state{title_suffix}", color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def seizure_prob_null_fig(res, out_png: str, *, labels=None,
                          title_suffix="") -> str:
    """The 09 forward map WITH the circular-shift-of-states null band on BOTH panels.
    Left: observed P(seizure within H | state) + per-state 2.5-97.5 shift-null band +
    dashed base rate. Right: observed lift + per-state band + dashed chance line. An
    observed curve inside its band = not distinguishable from chance."""
    hs = res["horizons_sec"]; k = res["obs"].shape[0]
    xl = [_hlabel(s) for s in hs]; x = np.arange(len(hs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=C.BG)
    for s in range(k):
        col = C.STATE_COLORS[s]
        lab = labels[s] if labels else f"state {s}"
        a1.fill_between(x, res["loP"][s], res["hiP"][s], color=col, alpha=0.13, lw=0)
        a1.plot(x, res["obs"][s], "-o", color=col, lw=1.8, ms=4,
                label=f"{lab} (n={res['n_state'][s]})")
    a1.plot(x, res["base"], "--", color=C.MUTED, lw=1.6, label="base rate (marginal)")
    _dark(a1, "P(seizure within H | state)  + shift-null 95% band")
    a1.set_xticks(x); a1.set_xticklabels(xl)
    a1.set_xlabel("horizon H"); a1.set_ylabel("P(seizure within H) %")
    leg = a1.legend(fontsize=7.5, framealpha=0.1, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    for s in range(k):
        col = C.STATE_COLORS[s]
        a2.fill_between(x, res["loL"][s], res["hiL"][s], color=col, alpha=0.13, lw=0)
        a2.plot(x, res["obs_lift"][s], "-o", color=col, lw=1.8, ms=4)
    a2.axhline(1.0, color=C.MUTED, lw=1.2, ls="--")
    _dark(a2, "predictive lift  + shift-null 95% band")
    a2.set_xticks(x); a2.set_xticklabels(xl)
    a2.set_xlabel("horizon H"); a2.set_ylabel("lift ×")
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · forward seizure-probability vs "
                 f"circular-shift-of-states null ({res['n_surr']} shifts)"
                 f"{title_suffix}", color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def state_risk_bin_null_fig(res, out_png: str, *, labels=None,
                            title_suffix="") -> str:
    """Per state: observed state x time-to-seizure-bin LIFT vs the circular-shift-of-
    STATES null. Shaded = 2.5-97.5 chance band; dashed = null median; dotted = lift 1;
    one-sided p per bin (* p<0.05). Observed inside the band => state->risk is not
    distinguishable from chance (the link disappears under the null)."""
    obs, lo, hi, med, p = res["obs"], res["lo"], res["hi"], res["med"], res["p"]
    names = res["names"]; k = obs.shape[0]
    nprox = len(names) - 1                                # drop the 'none' bin
    x = np.arange(nprox)
    rows = (k + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.1 * rows + 1), facecolor=C.BG,
                             squeeze=False)
    for s in range(k):
        ax = axes[s // 2][s % 2]
        ax.fill_between(x, lo[s, :nprox], hi[s, :nprox], color=C.MUTED, alpha=0.25,
                        lw=0, label="shift-null 95% band")
        ax.plot(x, med[s, :nprox], "--", color=C.MUTED, lw=1.0, label="null median")
        ax.plot(x, obs[s, :nprox], "-o", color=C.STATE_COLORS[s], lw=2.0, ms=5,
                label="observed")
        ax.axhline(1.0, color=C.TEXT, lw=0.8, ls=":")
        for j in range(nprox):
            if np.isfinite(p[s, j]) and np.isfinite(obs[s, j]):
                sig = "*" if p[s, j] < 0.05 else ""
                ax.annotate(f"p={p[s, j]:.2f}{sig}", (x[j], obs[s, j]),
                            textcoords="offset points", xytext=(0, 8), ha="center",
                            color=(C.ACCENT if sig else C.TEXT), fontsize=7)
        lab = labels[s] if labels else f"state {s}"
        _dark(ax, f"{lab}  (n={res['n_state'][s]})")
        ax.set_xticks(x); ax.set_xticklabels(names[:nprox])
        ax.set_xlabel("time to next seizure (min)"); ax.set_ylabel("lift ×")
        leg = ax.legend(fontsize=7, framealpha=0.1, loc="upper left")
        for t in leg.get_texts():
            t.set_color(C.TEXT)
    for s in range(k, rows * 2):
        axes[s // 2][s % 2].axis("off")
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · state→seizure-risk vs circular-shift"
                 f"-of-states null ({res['n_surr']} shifts, ≥"
                 f"{res['min_shift_sec']/3600:.0f} h){title_suffix}",
                 color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def metric_evolution_fig(model, onsets, out_png: str, *, follower_onsets=None,
                         metrics=None, bin_min: float = 30.0, state_filter=None,
                         title_suffix="") -> str:
    """Full-span evolution over the WHOLE record (all weeks), not a 2-h window: a
    dominant-state strip on top, then PC1/PC2 and key raw evoked metrics as binned
    median (+ IQR band) time series. Lead onsets = solid white lines; follower
    (non-lead) onsets = dashed, more transparent. Shows the long-timescale drift the
    per-seizure views can't capture."""
    import datetime as _dt
    import matplotlib.dates as mdates
    from . import trajectory as _T
    df = model.df
    series = ["pc1", "pc2"] + list(metrics or ["peak_to_trough", "line_length",
                                               "log_auc", "csd_variance"])
    series = [s for s in series if s in df.columns]
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    fin = np.isfinite(t); t = t[fin]
    st = df["state"].to_numpy()[fin]
    keep = np.ones(t.size, bool) if state_filter is None else (st == state_filter)
    lo, hi = t.min(), t.max()
    edges = np.arange(lo, hi + bin_min * 60, bin_min * 60)
    centers = edges[:-1] + bin_min * 30
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, centers.size - 1)
    xdt = [_dt.datetime.fromtimestamp(c) for c in centers]
    ons = np.sort(np.asarray(onsets, float))
    tl = _T.dominant_state_timeline(df, bin_sec=max(600.0, bin_min * 60))
    nrows = 1 + len(series)
    fig, axes = plt.subplots(nrows, 1, figsize=(15, 1.5 * nrows + 1), sharex=True,
                             facecolor=C.BG)
    from . import timeline as _TL
    ax0 = axes[0]; ax0.set_facecolor(C.PANEL)
    if state_filter is None:                              # all-state dominant strip
        rgb = _TL._rgb_row(tl["dominant"][(tl["centers"] >= lo) & (tl["centers"] <= hi)])
        ax0_label = "state"
    else:                                                 # state-filter occurrence strip
        ft = _T.state_fraction_timeline(df, target=state_filter,
                                        bin_sec=max(600.0, bin_min * 60))
        fsel = (ft["centers"] >= lo) & (ft["centers"] <= hi)
        pos = ft["frac"][np.isfinite(ft["frac"]) & (ft["frac"] > 0)]
        vmx = float(np.nanpercentile(pos, 95)) if pos.size else 1.0
        rgb = _TL._frac_rgb_row(ft["frac"][fsel], state_filter, vmx if vmx > 0 else 1.0)
        tl = ft; ax0_label = f"state {state_filter}"
    sc = tl["centers"][(tl["centers"] >= lo) & (tl["centers"] <= hi)]
    ax0.imshow(rgb, aspect="auto", origin="lower",
               extent=[mdates.date2num(_dt.datetime.fromtimestamp(sc[0])),
                       mdates.date2num(_dt.datetime.fromtimestamp(sc[-1])), 0, 1],
               interpolation="nearest", zorder=0)
    ax0.set_yticks([]); ax0.set_ylabel(ax0_label, color=C.TEXT, fontsize=9,
                                       rotation=0, ha="right", va="center")
    for ax, s in zip(axes[1:], series):
        ax.set_facecolor(C.PANEL)
        v = pd.to_numeric(df[s], errors="coerce").to_numpy(float)[fin]
        med = np.full(centers.size, np.nan)
        q1 = np.full(centers.size, np.nan); q3 = np.full(centers.size, np.nan)
        for c in range(centers.size):
            vv = v[(which == c) & keep]; vv = vv[np.isfinite(vv)]
            if vv.size:
                med[c] = np.median(vv)
                q1[c], q3[c] = np.percentile(vv, [25, 75])
        ax.fill_between(xdt, q1, q3, color=C.ACCENT, alpha=0.18, lw=0)
        mk = "." if state_filter is not None else None    # sparse cluster -> markers
        ax.plot(xdt, med, color=C.ACCENT, lw=1.0, marker=mk, ms=2.5)
        ax.set_ylabel(s, color=C.TEXT, fontsize=8, rotation=0, ha="right",
                      va="center")
        ax.tick_params(colors=C.MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(C.MUTED)
    fol = np.sort(np.asarray(follower_onsets, float)) if follower_onsets is not None \
        else np.empty(0)
    for ax in axes:                                       # onset lines on every panel
        for o in fol:                                     # followers: dashed, faint
            if lo <= o <= hi:
                ax.axvline(mdates.date2num(_dt.datetime.fromtimestamp(o)),
                           color=C.SEIZURE_COLOR, lw=0.8, ls="--", alpha=0.45,
                           zorder=2)
        for o in ons:                                     # leaders: solid, opaque
            if lo <= o <= hi:
                ax.axvline(mdates.date2num(_dt.datetime.fromtimestamp(o)),
                           color=C.SEIZURE_COLOR, lw=1.1, alpha=1.0, zorder=3)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("date", color=C.TEXT)
    ndays = (hi - lo) / 86400.0
    scope = f" · CLUSTER {state_filter} ONLY" if state_filter is not None else ""
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · full-span metric evolution{scope} "
                 f"({ndays:.0f} days, {bin_min:.0f}-min median ± IQR; solid = lead, "
                 f"dashed = follower onset){title_suffix}", color=C.TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def cluster_validity_fig(res, out_png: str, *, title_suffix="") -> str:
    """Four-panel 'are there discrete states?' diagnostic: PC1 histogram + dip p,
    GMM BIC vs k, bootstrap ARI vs a covariance-matched Gaussian null, silhouette vs
    k. Unimodal PCs + monotone BIC + low silhouette => a continuum, not states."""
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(12, 8), facecolor=C.BG)
    a.hist(res["Zpc1"], bins=60, color=C.ACCENT, alpha=0.85)
    dp = res["dips"][0]
    _dark(a, f"PC1 distribution  (dip p={dp['p']:.2f}, bimodality={dp['bimodality']:.2f})")
    a.set_xlabel("PC1"); a.set_ylabel("count")
    if res["bic"]:
        ks = sorted(res["bic"])
        b.plot(ks, [res["bic"][k] for k in ks], "-o", color=C.STATE_COLORS[0])
        if res["bic_min_k"] is not None:
            b.axvline(res["bic_min_k"], color=C.MUTED, ls="--", lw=1)
    _dark(b, "GMM BIC vs k  (interior min => a natural k; monotone => continuum)")
    b.set_xlabel("k (GMM components)"); b.set_ylabel("BIC (lower = better)")
    s = res["stability"]
    c.plot(s["k"], s["ari"], "-o", color=C.STATE_COLORS[1], label="observed")
    c.plot(s["k"], s["ari_null"], "--o", color=C.MUTED, label="Gaussian null")
    c.fill_between(s["k"], s["ari_null"], s["ari"], color=C.STATE_COLORS[1], alpha=0.15)
    _dark(c, "bootstrap cluster stability vs matched Gaussian null")
    c.set_xlabel("k (k-means)"); c.set_ylabel("ARI (reproducibility)")
    leg = c.legend(fontsize=8, framealpha=0.1)
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    sk = sorted(res["silhouette"])
    d.plot(sk, [res["silhouette"][k] for k in sk], "-o", color=C.STATE_COLORS[2])
    d.axhline(0.5, color=C.MUTED, ls="--", lw=1)
    _dark(d, "silhouette vs k  (~0.5+ = separated; ~0 = no structure)")
    d.set_xlabel("k (k-means)"); d.set_ylabel("silhouette")
    verdict = "DISCRETE states" if res["discrete"] else "CONTINUUM (no discrete states)"
    fig.suptitle(f"{C.ANIMAL} · {C.CHANNEL} · cluster validity → {verdict}  "
                 f"(N={res['n']:,}, {res['ncomp']} PCs @ {int(res['var_target']*100)}%)"
                 f"{title_suffix}", color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def state_fraction_trajectory_fig(traj, out_png: str, *, title_suffix="") -> str:
    """Per-seizure heatmap of the FRACTION of epochs in one target state per 10-min
    bin (one row per seizure; x = minutes from onset). Tracks a rare state's
    occurrence, which the dominant-state view hides. Bright = more; NaN = no data."""
    from matplotlib.colors import LinearSegmentedColormap
    tgt = int(traj["target"])
    M = np.ma.masked_invalid(traj["matrix"].astype(float))
    cmap = LinearSegmentedColormap.from_list("frac", [C.PANEL, C.STATE_COLORS[tgt]])
    cmap.set_bad(C.NODATA_COLOR)
    pos = traj["matrix"][np.isfinite(traj["matrix"]) & (traj["matrix"] > 0)]
    vmax = float(np.nanpercentile(pos, 95)) if pos.size else 0.0  # robust to outliers
    vmax = vmax if vmax > 0 else 1.0
    e = traj["edges"]
    fig, ax = plt.subplots(figsize=(13, 0.5 * M.shape[0] + 2.2), facecolor=C.BG)
    im = ax.imshow(M, aspect="auto", cmap=cmap, origin="upper", vmin=0, vmax=vmax,
                   extent=[e[0], e[-1], M.shape[0] - 0.5, -0.5],
                   interpolation="nearest")
    ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.6)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(traj["labels"], fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(f"fraction in state {tgt}", color=C.TEXT, fontsize=8)
    cbar.ax.tick_params(colors=C.MUTED, labelsize=7)
    cbar.outline.set_edgecolor(C.MUTED)
    _dark(ax, f"per-seizure state-{tgt} occurrence (10-min fraction; "
              f"{vmax:.0%}+ saturates){title_suffix}")
    ax.set_xlabel("minutes from seizure onset"); ax.set_ylabel("seizure")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def per_seizure_trajectory_fig(traj, out_png: str, *, title_suffix="") -> str:
    """Per-seizure dominant-state trajectory heatmap (one row per seizure; x =
    minutes from onset; colour = 10-min dominant state; white line at onset)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    M = traj["matrix"].astype(float)
    cmap = ListedColormap([C.NODATA_COLOR] + C.STATE_COLORS)
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(13, 0.5 * M.shape[0] + 2.2), facecolor=C.BG)
    e = traj["edges"]
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, origin="upper",
              extent=[e[0], e[-1], M.shape[0] - 0.5, -0.5], interpolation="nearest")
    ax.axvline(0, color=C.SEIZURE_COLOR, lw=1.6)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(traj["labels"], fontsize=7)
    bw = float(traj["edges"][1] - traj["edges"][0])          # bin width (minutes)
    blab = f"{bw:g}-min" if bw >= 1 else f"{int(round(bw * 60))}-s"
    _dark(ax, f"per-seizure pre-ictal state trajectory ({blab} dominant state)"
              f"{title_suffix}")
    ax.set_xlabel("minutes from seizure onset")
    ax.set_ylabel("seizure")
    handles = [Patch(color=C.STATE_COLORS[s], label=f"state {s}")
               for s in range(C.K_STATES)] + [Patch(color=C.NODATA_COLOR,
                                                     label="no data")]
    leg = ax.legend(handles=handles, fontsize=7.5, framealpha=0.1, ncol=5,
                    loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for t in leg.get_texts():
        t.set_color(C.TEXT)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _savefig(fig, out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png
