"""Stage 2 to 4 figures for the evoked-shape analysis.

Dark house style (bg #1e1e2f, panel #26263a, text #f0f0f5, accent #5e7ce2), matching
src.evoked_typology and the notification figures. Every panel is titled with the
finding and its axes are labelled with units; legends carry n. Headless Agg backend.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

_BG, _PANEL, _TEXT, _MUTED, _ACCENT = "#1e1e2f", "#26263a", "#f0f0f5", "#9a9ab0", "#5e7ce2"


def _dark(ax) -> None:
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_MUTED)


def fig_structure(struct: dict, Z: np.ndarray, corr_vals: np.ndarray,
                  diff_coords: np.ndarray, out: str) -> str:
    """Stage 2: PC1/PC2 density, pairwise-correlation histogram, and the diffusion-map
    density, with the dip verdict in the title. The discrete-vs-continuum panel set."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=_BG)
    for ax in axes:
        _dark(ax)
    axes[0].hexbin(Z[:, 0], Z[:, 1], gridsize=40, cmap="magma", mincnt=1)
    axes[0].set_title(f"shape PCA density (n={Z.shape[0]})", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("PC1", color=_MUTED, fontsize=9)
    axes[0].set_ylabel("PC2", color=_MUTED, fontsize=9)

    axes[1].hist(corr_vals, bins=60, color=_ACCENT, alpha=0.85)
    axes[1].set_title(f"pairwise trial correlations "
                      f"(bimod coeff={struct['bimodality_coeff']:.2f})",
                      color=_TEXT, fontsize=11)
    axes[1].set_xlabel("Pearson r between trials", color=_MUTED, fontsize=9)
    axes[1].set_ylabel("pairs", color=_MUTED, fontsize=9)

    if diff_coords.shape[1] >= 2:
        axes[2].hexbin(diff_coords[:, 0], diff_coords[:, 1], gridsize=40,
                       cmap="viridis", mincnt=1)
        axes[2].set_ylabel("diffusion coord 2", color=_MUTED, fontsize=9)
    else:
        axes[2].hist(diff_coords[:, 0], bins=60, color="#4caf82")
        axes[2].set_ylabel("count", color=_MUTED, fontsize=9)
    axes[2].set_title("diffusion-map embedding", color=_TEXT, fontsize=11)
    axes[2].set_xlabel("diffusion coord 1", color=_MUTED, fontsize=9)

    fig.suptitle(f"Stage 2 structure: {struct['verdict']}  ·  "
                 f"PC1 dip p={struct['pc1_dip']['p']:.3f}, "
                 f"diffusion dip p={struct['diff_dip']['p']:.3f}",
                 color=_TEXT, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out


def fig_templates(templates: np.ndarray, labels: np.ndarray, time_ms: np.ndarray,
                  Xn: np.ndarray, out: str) -> str:
    """Stage 3: the k correlation-k-means template shapes with mean +/- SD and
    prevalence. Raw shape space, not summaries."""
    k = templates.shape[0]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, k))
    ncol = min(k, 4)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow),
                             facecolor=_BG, squeeze=False)
    n = labels.size
    t = time_ms if time_ms.size == templates.shape[1] else np.arange(templates.shape[1])
    for c in range(k):
        ax = axes[c // ncol][c % ncol]
        _dark(ax)
        m = labels == c
        if m.any():
            mean = np.nanmean(Xn[m], axis=0)
            sd = np.nanstd(Xn[m], axis=0)
            ax.fill_between(t, mean - sd, mean + sd, color=colors[c], alpha=0.22, lw=0)
            ax.plot(t, mean, color=colors[c], lw=1.8)
        ax.axhline(0, color=_MUTED, lw=0.5, alpha=0.4)
        ax.set_title(f"shape {c + 1}  ·  {m.sum() / max(1, n):.0%} (n={int(m.sum())})",
                     color=_TEXT, fontsize=10, loc="left")
        ax.set_xlabel("ms since stim", color=_MUTED, fontsize=8)
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Stage 3 templates: k={k} correlation k-means shapes "
                 f"(normalized mean +/- SD)", color=_TEXT, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out


def fig_select_k(bic: dict, heldout: dict, stability, maxcorr: dict,
                 k_lo: int, k_hi: int, out: str) -> str:
    """Stage 4: BIC, held-out log-likelihood, stability excess over null, and max
    template correlation vs k, with the chosen range shaded. All curves on one page."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=_BG)
    axes = axes.ravel()
    for ax in axes:
        _dark(ax)

    def _band(ax):
        ax.axvspan(k_lo, k_hi, color=_ACCENT, alpha=0.15,
                   label=f"chosen range {k_lo}-{k_hi}")

    ks = sorted(bic.keys())
    axes[0].plot(ks, [bic[k] for k in ks], "-o", color=_ACCENT)
    _band(axes[0])
    axes[0].set_title("GMM BIC (lower is better)", color=_TEXT, fontsize=11)
    axes[0].set_xlabel("k", color=_MUTED, fontsize=9)

    if heldout:
        hk = sorted(heldout.keys())
        axes[1].plot(hk, [heldout[k] for k in hk], "-o", color="#4caf82")
    _band(axes[1])
    axes[1].set_title("held-out log-likelihood / sample", color=_TEXT, fontsize=11)
    axes[1].set_xlabel("k", color=_MUTED, fontsize=9)

    if hasattr(stability, "empty") and not stability.empty:
        axes[2].plot(stability["k"], stability["ari_excess"], "-o", color=_ACCENT,
                     label="ARI excess")
        axes[2].plot(stability["k"], stability["jaccard_excess"], "-s",
                     color="#e0a75e", label="Jaccard excess")
        axes[2].axhline(0.05, color=_MUTED, lw=0.8, ls="--")
    _band(axes[2])
    axes[2].set_title("bootstrap stability, observed minus null", color=_TEXT,
                      fontsize=11)
    axes[2].set_xlabel("k", color=_MUTED, fontsize=9)
    leg = axes[2].legend(fontsize=8, frameon=False)
    for tx in leg.get_texts():
        tx.set_color(_TEXT)

    mk = sorted(maxcorr.keys())
    axes[3].plot(mk, [maxcorr[k] for k in mk], "-o", color="#e05e7c")
    axes[3].axhline(0.9, color=_MUTED, lw=0.8, ls="--", label="duplicate threshold 0.9")
    _band(axes[3])
    axes[3].set_title("max off-diagonal template correlation", color=_TEXT, fontsize=11)
    axes[3].set_xlabel("k", color=_MUTED, fontsize=9)
    leg = axes[3].legend(fontsize=8, frameon=False)
    for tx in leg.get_texts():
        tx.set_color(_TEXT)

    fig.suptitle(f"Stage 4 choosing k: defensible range {k_lo} to {k_hi}",
                 color=_TEXT, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, facecolor=_BG)
    plt.close(fig)
    return out
