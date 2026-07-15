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
