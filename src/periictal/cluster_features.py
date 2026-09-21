"""Cluster the peri-ictal embedding and rank which feature explains the clusters.

This is COMPILED EVIDENCE, not a verdict. Given a 2-D embedding (what the operator
sees in the UMAP/PCA lens) and the aligned per-event feature frame, we:

  * cluster the 2-D coordinates (``cluster_embedding``) -- HDBSCAN or K-means,
  * rank every feature by how strongly cluster membership explains its variance
    (``feature_separability``, Kruskal-Wallis H + epsilon-squared effect size),
  * profile each cluster -- the robust-z median of every feature, its n, and its
    mean/median time-to-onset (``cluster_profiles``), which quantifies the
    "this cluster is homogeneous in feature X" pattern, and
  * report the raw cluster-quality numbers (``cluster_quality``).

Nothing here decides whether a number is meaningful -- it hands the operator the
epsilon-squared ranking, the per-cluster profiles and the silhouette so THEY can
interpret it. No p-value thresholds, no shuffled-null, no "this is/isn't evidence".

Dash-free and side-effect-free so it stays unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TTO = "time_to_onset_sec"
_NOISE = -1                       # HDBSCAN's noise label; excluded from stats


# --------------------------------------------------------------------- #
#  Clustering
# --------------------------------------------------------------------- #

def _default_min_cluster_size(n: int) -> int:
    """~2% of points, floored at 5 -- a stable HDBSCAN default across N."""
    return int(max(5, round(0.02 * n)))


def cluster_embedding(emb2d: np.ndarray, *, method: str = "hdbscan",
                      min_cluster_size: int | None = None,
                      k: int | None = None, seed: int = 0) -> np.ndarray:
    """Cluster the embedding COORDINATES (the 2-D picture, not the feature space).

    ``method="hdbscan"`` labels sparse points as noise (``-1``); ``method="kmeans"``
    partitions every point into ``k`` clusters. Returns an ``int`` label per row,
    aligned to ``emb2d``. An empty / single-point embedding returns all-noise.
    """
    assert method in ("hdbscan", "kmeans"), "method must be 'hdbscan' or 'kmeans'"
    emb2d = np.asarray(emb2d, dtype=np.float64)
    assert emb2d.ndim == 2, "emb2d must be 2-D [N, d]"
    n = emb2d.shape[0]
    if n < 3:
        return np.full(n, _NOISE, dtype=int)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        kk = int(k or max(2, min(8, round(np.sqrt(n / 2.0)))))
        kk = int(min(kk, n))                       # can't ask for more clusters than points
        km = KMeans(n_clusters=kk, random_state=seed, n_init=10)
        return km.fit_predict(emb2d).astype(int)
    from sklearn.cluster import HDBSCAN
    mcs = int(min_cluster_size or _default_min_cluster_size(n))
    mcs = int(min(max(2, mcs), n))
    hdb = HDBSCAN(min_cluster_size=mcs)
    return hdb.fit_predict(emb2d).astype(int)


# --------------------------------------------------------------------- #
#  Which feature explains the clusters (Kruskal-Wallis epsilon-squared)
# --------------------------------------------------------------------- #

def _groups(values: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """Finite feature values split by NON-noise cluster label; groups with no
    finite value dropped. Returns the per-cluster value arrays."""
    out = []
    for lab in np.unique(labels):
        if lab == _NOISE:
            continue
        v = values[labels == lab]
        v = v[np.isfinite(v)]
        if v.size:
            out.append(v)
    return out


def _epsilon_squared(h: float, n: int, k: int) -> float:
    """Kruskal-Wallis effect size epsilon^2 = (H - k + 1) / (n - k), clipped to
    [0, 1]. 0 = clusters explain none of the feature's rank variance, 1 = all."""
    if n <= k:
        return float("nan")
    return float(min(1.0, max(0.0, (h - k + 1.0) / (n - k))))


def feature_separability(df: pd.DataFrame, labels: np.ndarray,
                         features: list[str]) -> pd.DataFrame:
    """Rank *features* by how strongly the clusters explain each one.

    Per feature: Kruskal-Wallis H across the (non-noise) clusters plus the
    epsilon-squared effect size; sorted by epsilon-squared descending. Descriptive
    only -- the p-value is reported, never thresholded. Columns:
    ``feature, eps2, H, p, n, k_groups, rank``. Features with <2 usable groups are
    returned with NaN stats at the bottom.
    """
    from scipy.stats import kruskal
    labels = np.asarray(labels)
    assert len(labels) == len(df), "labels/df length mismatch"
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        groups = _groups(df[feat].to_numpy(dtype=np.float64), labels)
        n = int(sum(g.size for g in groups))
        k = len(groups)
        if k < 2 or _all_equal(groups):             # <2 groups, or no rank variance
            rows.append((feat, float("nan"), float("nan"), float("nan"), n, k))
            continue
        try:
            h, p = kruskal(*groups)
        except ValueError:                          # all identical across groups
            rows.append((feat, float("nan"), float("nan"), float("nan"), n, k))
            continue
        rows.append((feat, _epsilon_squared(float(h), n, k), float(h), float(p),
                     n, k))
    out = pd.DataFrame(rows, columns=["feature", "eps2", "H", "p", "n",
                                      "k_groups"])
    out = out.sort_values("eps2", ascending=False, na_position="last",
                          kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _all_equal(groups: list[np.ndarray]) -> bool:
    """True when every value across every group is the same constant (kruskal
    is undefined -- no rank variance to partition)."""
    cat = np.concatenate(groups)
    return cat.size > 0 and np.ptp(cat) == 0


# --------------------------------------------------------------------- #
#  Per-cluster feature profile (the "homogeneous colour" quantified)
# --------------------------------------------------------------------- #

def _robust_z(values: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) over the finite values; 0 where MAD == 0."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=np.float64)
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    scale = 1.4826 * mad
    if scale <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - med) / scale


def cluster_profiles(df: pd.DataFrame, labels: np.ndarray,
                     features: list[str]) -> pd.DataFrame:
    """One row per cluster (noise ``-1`` last, if present): the MEDIAN robust-z of
    every feature (how far/homogeneously that cluster sits vs the whole pool), the
    cluster ``n``, and its mean/median ``time_to_onset_sec``.

    A near-zero, tight column means the feature is at pool-median for that cluster;
    a large-magnitude one means the feature characterises it. Feature columns are
    the robust-z medians; ``n``/``mean_tto_sec``/``median_tto_sec`` are appended.
    """
    labels = np.asarray(labels)
    assert len(labels) == len(df), "labels/df length mismatch"
    zc = {f: _robust_z(df[f].to_numpy(dtype=np.float64))
          for f in features if f in df.columns}
    tto = (df[_TTO].to_numpy(dtype=np.float64) if _TTO in df.columns
           else np.full(len(df), np.nan))
    uniq = sorted(np.unique(labels).tolist(),
                  key=lambda x: (x == _NOISE, x))      # noise sorts last
    rows = []
    for lab in uniq:
        mask = labels == lab
        rec = {"cluster": int(lab), "n": int(mask.sum())}
        for f, z in zc.items():
            rec[f] = float(np.nanmedian(z[mask])) if mask.any() else float("nan")
        t = tto[mask]
        t = t[np.isfinite(t)]
        rec["mean_tto_sec"] = float(np.mean(t)) if t.size else float("nan")
        rec["median_tto_sec"] = float(np.median(t)) if t.size else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
#  Cluster quality (raw numbers, no verdict)
# --------------------------------------------------------------------- #

def cluster_quality(emb2d: np.ndarray, labels: np.ndarray) -> dict:
    """``{silhouette, n_clusters, n_noise, n_points}`` for the current run.

    Silhouette is computed over the non-noise points only and is ``None`` when
    there are <2 clusters or too few points -- a number to compare across runs
    (e.g. with vs without time-to-onset folded in), not a pass/fail gate."""
    emb2d = np.asarray(emb2d, dtype=np.float64)
    labels = np.asarray(labels)
    non_noise = labels != _NOISE
    uniq = np.unique(labels[non_noise])
    out = {"silhouette": None, "n_clusters": int(uniq.size),
           "n_noise": int((labels == _NOISE).sum()),
           "n_points": int(labels.size)}
    if uniq.size >= 2 and non_noise.sum() > uniq.size:
        from sklearn.metrics import silhouette_score
        try:
            out["silhouette"] = float(
                silhouette_score(emb2d[non_noise], labels[non_noise]))
        except ValueError:
            out["silhouette"] = None
    return out


# --------------------------------------------------------------------- #
#  One-call convenience for the dashboard worker
# --------------------------------------------------------------------- #

def analyze(emb2d: np.ndarray, df: pd.DataFrame, features: list[str], *,
            method: str = "hdbscan", min_cluster_size: int | None = None,
            k: int | None = None, seed: int = 0) -> dict:
    """Cluster + separability + profiles + quality in one call, aligned to
    ``emb2d``/``df`` (same row order). Returns a plain dict the UI can render:
    ``{labels, separability(DataFrame), profiles(DataFrame), quality(dict)}``."""
    assert len(df) == emb2d.shape[0], "df/emb2d length mismatch"
    labels = cluster_embedding(emb2d, method=method,
                               min_cluster_size=min_cluster_size, k=k, seed=seed)
    return {"labels": labels,
            "separability": feature_separability(df, labels, features),
            "profiles": cluster_profiles(df, labels, features),
            "quality": cluster_quality(emb2d, labels)}
