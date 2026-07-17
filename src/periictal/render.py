"""Headless (matplotlib Agg) scatter PNGs of a 2-D embedding.

Mirrors src/evoked_figures/render.py's posture: pure Agg, no MATLAB, disposable
renders of the on-disk matrix. Used by the CLI to produce the first BCH111
picture before any dashboard exists.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")            # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402


def scatter_continuous(emb: np.ndarray, color: np.ndarray, path: str, *,
                       title: str, clabel: str, cmap: str = "plasma") -> None:
    """Scatter coloured by a continuous value + a colourbar."""
    assert emb.ndim == 2 and emb.shape[1] >= 2, "emb must be [N,>=2]"
    assert emb.shape[0] == np.asarray(color).shape[0], "len(color)!=len(emb)"
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=110)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap=cmap, s=4,
                    alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label=clabel)
    _finish(fig, ax, title)
    fig.savefig(path)
    plt.close(fig)


def scatter_categorical(emb: np.ndarray, labels, path: str, *,
                        title: str, legend_title: str) -> None:
    """Scatter coloured by a categorical label + a legend."""
    assert emb.ndim == 2 and emb.shape[1] >= 2, "emb must be [N,>=2]"
    cats = pd.Categorical(labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=110)
    cmap = plt.get_cmap("tab10")
    for k, name in enumerate(cats.categories):
        m = cats.codes == k
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5, linewidths=0,
                   color=cmap(k % 10), label=str(name))
    ax.legend(title=legend_title, markerscale=3, fontsize=8, framealpha=0.9,
              loc="best")
    _finish(fig, ax, title)
    fig.savefig(path)
    plt.close(fig)


def _finish(fig, ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()


# --------------------------------------------------------------------- #
#  Chang et al. 2026 preictal-vs-interictal report figures
# --------------------------------------------------------------------- #
_PRE_C = "#e8433f"       # preictal = red
_INT_C = "#3b7fc4"       # interictal = blue


def pdf_cdf_panel(pc: dict, path: str, *, feature: str, title: str) -> None:
    """Two-panel PDF (KDE) + CDF (ECDF) of *feature*, preictal vs interictal."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=120)
    grid = pc.get("grid")
    if grid is not None and getattr(grid, "size", 0):
        if pc.get("pre_kde") is not None:
            a1.fill_between(grid, pc["pre_kde"], color=_PRE_C, alpha=0.2)
            a1.plot(grid, pc["pre_kde"], color=_PRE_C, lw=2,
                    label=f"preictal (n={pc['n_pre']})")
        if pc.get("inter_kde") is not None:
            a1.fill_between(grid, pc["inter_kde"], color=_INT_C, alpha=0.15)
            a1.plot(grid, pc["inter_kde"], color=_INT_C, lw=2,
                    label=f"interictal (n={pc['n_inter']})")
    a1.set_title(f"PDF · {feature}", fontsize=10)
    a1.set_xlabel(feature); a1.set_ylabel("density"); a1.legend(fontsize=8)
    if pc.get("pre_cdf_x") is not None and pc["pre_cdf_x"].size:
        a2.step(pc["pre_cdf_x"], pc["pre_cdf_y"], where="post", color=_PRE_C,
                lw=2, label="preictal")
    if pc.get("inter_cdf_x") is not None and pc["inter_cdf_x"].size:
        a2.step(pc["inter_cdf_x"], pc["inter_cdf_y"], where="post", color=_INT_C,
                lw=2, label="interictal")
    a2.set_title(f"CDF · {feature}  (AUC {pc.get('auc_norm', float('nan')):.3f})",
                 fontsize=10)
    a2.set_xlabel(feature); a2.set_ylabel("cumulative fraction")
    a2.set_ylim(0, 1); a2.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


def feature_auc_bar(rows: list, path: str, *, title: str, top: int = 20) -> None:
    """Horizontal bar of normalised AUC per feature (paper Fig 2), red when the
    BH q-value < .05."""
    rows = [r for r in rows if np.isfinite(r.get("auc_norm", float("nan")))][:top]
    rows = rows[::-1]                                  # best at the top
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(rows) + 1)), dpi=120)
    y = np.arange(len(rows))
    vals = [r["auc_norm"] for r in rows]
    colors = [_PRE_C if (np.isfinite(r.get("q", 1)) and r["q"] < 0.05)
              else "#888888" for r in rows]
    ax.barh(y, vals, color=colors)
    ax.axvline(0.5, color="#bbbbbb", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r["feature"] for r in rows], fontsize=8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("normalised AUC (preictal vs interictal)")
    ax.set_title(title, fontsize=10)
    ax.text(0.99, 0.02, "red = BH q < .05", transform=ax.transAxes, fontsize=7,
            ha="right", color=_PRE_C)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def forecast_panel(res: dict, path: str, *, title: str) -> None:
    """Three-panel forecaster summary: ROC per phase, AUC-vs-phase, normalised
    coefficients (paper Fig 3 / Fig 5C)."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 4.4), dpi=120)
    phases = res.get("phases", [])
    roc_ph = [p for p in phases if p.get("fpr") is not None]
    a1.plot([0, 1], [0, 1], color="#bbbbbb", ls="--", lw=1.2,
            label="chance (AUC 0.5)")
    for i, p in enumerate(roc_ph):
        shade = 0.3 + 0.6 * (i / max(1, len(roc_ph) - 1))
        a1.plot(p["fpr"], p["tpr"], color=_PRE_C, alpha=shade, lw=1.8,
                label=f"P{p['train_phase']}→{p['test_phase']} ({p['auc']:.2f})")
    a1.set_title("ROC per phase (test on P+1)", fontsize=10)
    a1.set_xlabel("false-positive rate"); a1.set_ylabel("true-positive rate")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.legend(fontsize=7, loc="lower right")
    xs = [p["test_phase"] for p in phases]
    ys = [p["auc"] for p in phases]
    a2.axhline(0.5, color="#bbbbbb", ls="--", lw=1)
    a2.plot(xs, ys, "-o", color=_PRE_C, lw=2)
    a2.set_ylim(0.3, 1.0); a2.set_xlabel("test phase"); a2.set_ylabel("test AUC")
    a2.set_title(f"AUC vs phase (mean {res.get('mean_auc', float('nan')):.3f})",
                 fontsize=10)
    coef = sorted((res.get("coefficients") or {}).items(), key=lambda kv: kv[1])
    if coef:
        a3.barh(np.arange(len(coef)), [v for _, v in coef], color=_INT_C)
        a3.set_yticks(np.arange(len(coef)))
        a3.set_yticklabels([k for k, _ in coef], fontsize=8)
    a3.set_xlabel("normalised |coefficient| (Σ=1)")
    a3.set_title("Feature importance", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
