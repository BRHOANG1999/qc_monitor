"""Cluster-validity diagnostic: do discrete STATES exist in the evoked-feature
space, or is PCA->k-means just binning a continuum?

k-means always returns k Voronoi cells, even on a single Gaussian blob -- so before
interpreting "states" we must test whether discrete structure exists at all. Four
independent angles on the SAME space states.fit_states clusters (StandardScaler ->
PCA to VAR_TARGET):
  1. Hartigan dip test per PC         -- unimodal (continuum) vs multimodal (discrete)
  2. GMM BIC vs k                     -- an elbow/min => a natural k; monotone => continuum
  3. bootstrap stability - Gaussian   -- ARI excess over a covariance-matched unimodal
     null (reproducible partition beyond slicing a blob?)
  4. silhouette vs k                  -- cluster separation (relative only)

Reuses the committed evoked_shapes toolkit (structure, select_k).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config as C
from src.evoked_shapes import structure as _ST, select_k as _SK


def _log(m):
    print(f"[preictal_biomarker.validity] {m}", flush=True)


def cluster_validity(df: pd.DataFrame, *, features=None, var_target=None,
                     k_range=(2, 6), n_dip=4000, n_sil=8000, seed=0) -> dict:
    """Run the four diagnostics on the states.fit_states space. Returns a dict with
    the PC variance, per-PC dip results, GMM BIC curve, stability DataFrame and
    silhouette-by-k, plus a boolean `discrete` verdict."""
    feats = features or (list(C.FEATURES) + [c for c in C.PHFO_FEATURES
                                             if c in df.columns])
    feats = [f for f in feats if f in df.columns]
    vt = float(var_target or C.VAR_TARGET)
    X = df[feats].to_numpy(float)
    X = X[np.all(np.isfinite(X), axis=1)]
    assert X.shape[0] > 1000, "need enough epochs for a validity test"
    Xn = StandardScaler().fit_transform(X)
    pca = PCA(random_state=seed).fit(Xn)
    evr = pca.explained_variance_ratio_
    ncomp = int(np.searchsorted(np.cumsum(evr), vt) + 1)
    Z = pca.transform(Xn)[:, :ncomp]
    _log(f"N={Xn.shape[0]} features={len(feats)} PCs@{int(vt*100)}%={ncomp}")
    rng = np.random.default_rng(seed)
    di = np.sort(rng.choice(Z.shape[0], size=min(n_dip, Z.shape[0]), replace=False))
    dips = []
    for a in range(min(5, Z.shape[1])):
        r = _ST.dip_test(Z[di, a], n_boot=500, seed=seed)
        dips.append({"pc": a + 1, "dip": r["dip"], "p": r["p"],
                     "bimodality": float(_ST.bimodality_coefficient(Z[di, a]))})
    stab = _SK.stability_curve(Z, k_range, n_boot=100, seed=seed, max_n=2500)
    si = np.sort(rng.choice(Z.shape[0], size=min(n_sil, Z.shape[0]), replace=False))
    Zs = Z[si]
    sil = {}
    for k in range(k_range[0], k_range[1] + 1):
        lab = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Zs)
        sil[k] = float(silhouette_score(Zs, lab))
    try:
        bic = _SK.gmm_bic_curve(Zs, (1, k_range[1]), seed=seed)
    except Exception as e:                                 # noqa: BLE001
        _log(f"GMM BIC skipped: {e}"); bic = {}
    # Verdict: discrete only if at least one PC is clearly multimodal AND BIC has a
    # real interior minimum. Unimodal + monotone BIC => continuum.
    any_multimodal = any(d["p"] < 0.05 and d["bimodality"] > 0.555 for d in dips)
    bic_min_k = min(bic, key=bic.get) if bic else None
    bic_elbow = bic_min_k is not None and 1 < bic_min_k < k_range[1]
    discrete = bool(any_multimodal and bic_elbow)
    return {"evr": evr, "ncomp": ncomp, "n": int(Xn.shape[0]), "dips": dips,
            "stability": stab, "silhouette": sil, "bic": bic,
            "bic_min_k": bic_min_k, "discrete": discrete, "Zpc1": Z[di, 0],
            "var_target": vt}
