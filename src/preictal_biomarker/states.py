"""PCA + k-means state decomposition of the 8-feature matrix.

Standardize the 8 features -> PCA (6 PCs) -> k-means (k=4). Deterministic
(fixed seed). Returns PC coordinates and an integer ``state`` per epoch, plus a
per-state z-scored feature profile and an automatic character label (energy /
high-AR1 / low-energy / post-ictal), so the narrative does not depend on which
integer k-means happened to assign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as C


@dataclass
class StateModel:
    df: pd.DataFrame                 # input rows + pc1..pcN + state columns
    features: list                   # feature names used
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    explained_var: np.ndarray        # per-PC explained variance ratio
    profiles: pd.DataFrame           # state x feature, mean z-score
    labels: dict                     # state int -> character label
    fit_mask: np.ndarray             # rows that had all features finite


def fit_states(df: pd.DataFrame, *, features=None, n_pcs=None, k=None,
               seed=None) -> StateModel:
    """Fit PCA+k-means on the finite-feature rows; annotate every row with its PC
    coordinates and state (``-1`` where a feature was missing)."""
    features = list(features or C.FEATURES8)
    n_pcs = int(n_pcs or C.N_PCS)
    k = int(k or C.K_STATES)
    seed = C.SEED if seed is None else int(seed)
    assert set(features).issubset(df.columns), "missing feature columns"
    X = df[features].to_numpy(float)
    mask = np.all(np.isfinite(X), axis=1)
    assert mask.sum() > k * 10, "too few finite-feature rows to cluster"
    Xz = StandardScaler().fit_transform(X[mask])
    scaler = StandardScaler().fit(X[mask])
    pca = PCA(n_components=n_pcs, random_state=seed).fit(Xz)
    emb = pca.transform(Xz)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb)
    out = df.copy().reset_index(drop=True)
    pcs = np.full((len(out), n_pcs), np.nan)
    pcs[mask] = emb
    for j in range(n_pcs):
        out[f"pc{j + 1}"] = pcs[:, j]
    state = np.full(len(out), -1, dtype=int)
    state[mask] = km.labels_
    out["state"] = state
    profiles = _profiles(out, features, k)
    labels = _label_states(profiles)
    return StateModel(df=out, features=features, scaler=scaler, pca=pca,
                      kmeans=km, explained_var=pca.explained_variance_ratio_,
                      profiles=profiles, labels=labels, fit_mask=mask)


def _profiles(df: pd.DataFrame, features: list, k: int) -> pd.DataFrame:
    """Per-state mean of z-scored features (the state metric-profile heatmap)."""
    z = (df[features] - df[features].mean()) / (df[features].std(ddof=0) + 1e-9)
    z["state"] = df["state"].to_numpy()
    rows = {}
    for s in range(k):
        sub = z[z["state"] == s]
        rows[s] = sub[features].mean() if len(sub) else pd.Series(
            np.nan, index=features)
    return pd.DataFrame(rows).T[features]


def _label_states(profiles: pd.DataFrame) -> dict:
    """Heuristic character per state from its feature profile (for the narrative).
    Energy = mean of amplitude/line-length features; AR1 = csd_ar1 z."""
    amp = [f for f in ("rms_amplitude", "peak_to_trough", "line_length",
                       "variance") if f in profiles.columns]
    labels = {}
    for s in profiles.index:
        row = profiles.loc[s]
        energy = float(np.nanmean([row[f] for f in amp])) if amp else np.nan
        ar1 = float(row.get("csd_ar1", np.nan))
        if np.isfinite(ar1) and ar1 >= 0.6:
            tag = "high-AR1 (critical-slowing)"
        elif np.isfinite(energy) and energy >= 0.4:
            tag = "high-energy"
        elif np.isfinite(energy) and energy <= -0.4:
            tag = "low-energy"
        else:
            tag = "intermediate"
        labels[int(s)] = tag
    return labels


def preictal_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows in the pre-ictal window (0..PREICTAL_SEC before a lead/any onset)."""
    tto = pd.to_numeric(df.get("time_to_onset_sec"), errors="coerce").to_numpy()
    return np.isfinite(tto) & (tto >= 0) & (tto <= C.PREICTAL_SEC) \
        & (df["phase"].to_numpy() == "pre")


def baseline_mask(df: pd.DataFrame) -> np.ndarray:
    """Clean-baseline rows: >= BASELINE_MIN_SEC from the onset they flank."""
    tto = pd.to_numeric(df.get("time_to_onset_sec"), errors="coerce").to_numpy()
    return np.isfinite(tto) & (tto >= C.BASELINE_MIN_SEC)
