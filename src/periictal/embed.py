"""Embed the event x metric matrix into 2-D for exploration.

PCA is the DEFAULT: it is deterministic, ~instant at any N, and an honest linear
projection whose axes mean something. UMAP is OPT-IN -- it can manufacture
apparent clusters from noise and does not preserve global structure, so it is a
picture, never evidence. Above an interactive point cap the rows are STRATIFIED-
subsampled by (seizure, lead-time bin) rather than uniformly: at a fixed 0.5 Hz
stim rate the far-from-onset bins hold geometrically more points, so uniform
sampling would starve the near-onset window where any pre-ictal signal lives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.periictal import config as _cfg


def _prepare(df: pd.DataFrame, metrics: list[str]):
    """Metric matrix with all-NaN columns dropped and remaining NaNs filled with
    the column median. Returns (X[N,d], used_cols)."""
    cols = [m for m in metrics if m in df.columns and df[m].notna().any()]
    assert cols, "no usable metric columns (all-NaN)"
    X = df[cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    bad = np.isnan(X)
    if bad.any():
        X[bad] = np.take(med, np.where(bad)[1])
    return X, cols


def _stratified_subsample(df: pd.DataFrame, cap: int, seed: int) -> np.ndarray:
    """Row labels to keep: ~equal points per (seizure_idx, lead_bin) cell so no
    single seizure or far-field bin dominates. Returns all labels when N<=cap."""
    labels = df.index.to_numpy()
    if labels.size <= cap:
        return labels
    groups = list(df.groupby(["seizure_idx", "lead_bin"]).groups.values())
    per = max(1, cap // max(1, len(groups)))
    rng = np.random.default_rng(seed)
    picks = []
    for g in groups:
        g = np.asarray(g)
        take = min(g.size, per)
        picks.append(rng.choice(g, size=take, replace=False))
    out = np.concatenate(picks) if picks else labels
    if out.size > cap:                       # trim any overshoot deterministically
        out = rng.choice(out, size=cap, replace=False)
    return out


def _pca(X: np.ndarray, n_components: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    emb = pca.fit_transform(Xs)
    return emb, {"explained_var": pca.explained_variance_ratio_.tolist()}


def _umap(X: np.ndarray, n_components: int, n_neighbors: int, seed: int):
    import umap                                   # lazy: heavy import
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    n_neighbors = int(min(n_neighbors, max(2, Xs.shape[0] - 1)))
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        random_state=seed, low_memory=True)
    emb = reducer.fit_transform(Xs)
    return np.asarray(emb), {"n_neighbors": n_neighbors}


def embed(df: pd.DataFrame, metrics: list[str] | None = None, *,
          method: str = "pca", n_components: int = 2,
          cap: int = _cfg.INTERACTIVE_POINT_CAP, seed: int = 0,
          n_neighbors: int = 15) -> dict:
    """Embed *df*'s metric columns into ``n_components`` dims.

    Returns ``{"emb": ndarray[M,k], "rows": label array[M], "cols": [...],
    "method": str, "meta": {...}}`` where ``rows`` are the (subsampled) df index
    labels aligned to ``emb`` -- so the caller colours by df.loc[rows, ...].
    Empty df -> empty emb."""
    assert method in ("pca", "umap"), "method must be 'pca' or 'umap'"
    metrics = metrics or _cfg.CHEAP_METRICS
    if len(df) == 0:
        return {"emb": np.empty((0, n_components)), "rows": np.empty(0, dtype=int),
                "cols": [], "method": method, "meta": {}}
    rows = _stratified_subsample(df, cap, seed)
    sub = df.loc[rows]
    X, cols = _prepare(sub, metrics)
    if method == "umap" and X.shape[0] >= 5:
        emb, meta = _umap(X, n_components, n_neighbors, seed)
    else:
        emb, meta = _pca(X, min(n_components, X.shape[1]))
        method = "pca"
    meta["n_points"] = int(X.shape[0])
    meta["n_total"] = int(len(df))
    return {"emb": emb, "rows": rows, "cols": cols, "method": method, "meta": meta}


def confound_readout(emb: np.ndarray, df_sub: pd.DataFrame) -> dict:
    """Descriptive (NOT inferential) check that the embedding isn't just a
    confound: |Spearman rho| between each embedding axis and (a) time-of-day,
    (b) the stim-fingerprint group index. A high value means the picture is
    driven by that confound, not by seizure proximity. Returns a small dict."""
    from scipy.stats import rankdata
    out: dict = {}
    if emb.shape[0] < 10:
        return out
    tod = df_sub["hour_of_day"].to_numpy()
    fp = pd.Categorical(df_sub["stim_key"]).codes.astype(float)
    for name, v in (("time_of_day", tod), ("stim_fingerprint", fp)):
        if np.unique(v[np.isfinite(v)]).size < 2:
            out[name] = None
            continue
        rho = max(abs(_spearman(emb[:, k], v, rankdata))
                  for k in range(emb.shape[1]))
        out[name] = float(rho)
    return out


def _spearman(a: np.ndarray, b: np.ndarray, rankdata) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return 0.0
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else 0.0
