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

from . import config as C                   # noqa: E402


def _fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, facecolor=C.BG)
    return fig, ax


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
    _dark(ax, f"states (k={C.K_STATES}) on {C.N_PCS} PCs · PC{pcx+1} vs PC{pcy+1}")
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
    """Pre-ictal vs clean-baseline occupancy per state (grouped bars)."""
    k = C.K_STATES
    x = np.arange(k)
    w = 0.38
    ax.bar(x - w / 2, occ["preictal"], w, color="#f2c744",
           label=f"pre-ictal {int(C.PREICTAL_SEC//60)} min (n={occ['n_pre']})")
    ax.bar(x + w / 2, occ["baseline"], w, color="#5b8def",
           label=f"clean baseline ≥{int(C.BASELINE_MIN_SEC//3600)} h "
                 f"(n={occ['n_base']})")
    _dark(ax, "state occupancy · pre-ictal vs clean baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"state {s}" for s in range(k)])
    ax.set_xlabel("state")
    ax.set_ylabel("% of epochs in state")
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
                 f"({C.N_PCS} PCs, {100*ev[:C.N_PCS].sum():.0f}% var) · anchor check",
                 color=C.TEXT, fontsize=12)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
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
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
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
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
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
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


def mdates_num(epoch):
    import datetime as _dt
    import matplotlib.dates as _md
    return _md.date2num(_dt.datetime.fromtimestamp(float(epoch)))


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
    fig.savefig(out_png, dpi=130, facecolor=C.BG, bbox_inches="tight")
    plt.close(fig)
    return out_png
