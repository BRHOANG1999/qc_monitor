"""Cluster-the-embedding + which-feature-explains-the-clusters (compiled evidence).

Covers the pure module ``src.periictal.cluster_features`` (clustering, Kruskal-Wallis
epsilon-squared separability ranking, per-cluster robust-z profiles, silhouette) and
the reusable ``matrix.time_to_onset_for`` join it shares with the evoked package.

Run with: pytest tests/test_cluster_features.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import cluster_features as cf        # noqa: E402
from src.periictal.matrix import time_to_onset_for, _assign_next_onset  # noqa: E402


# --------------------------------------------------------------------- #
#  Fixtures: three well-separated Gaussian blobs in 2-D + an aligned frame
# --------------------------------------------------------------------- #

def _blobs(seed: int = 0, per: int = 60):
    """3 blobs. `explains` is constant-within-blob but differs across blobs (a
    feature the clusters explain fully); `noise_feat` is blob-independent."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]])
    emb, blob = [], []
    for i, c in enumerate(centers):
        emb.append(c + rng.normal(0, 0.4, size=(per, 2)))
        blob.append(np.full(per, i))
    emb = np.vstack(emb)
    blob = np.concatenate(blob)
    df = pd.DataFrame({
        "explains": blob.astype(float) * 10.0 + rng.normal(0, 0.05, blob.size),
        "noise_feat": rng.normal(0, 1.0, blob.size),
        "time_to_onset_sec": blob.astype(float) * 100.0 + 50.0,
    })
    return emb, blob, df


# --------------------------------------------------------------------- #
#  cluster_embedding
# --------------------------------------------------------------------- #

def test_hdbscan_recovers_blobs():
    emb, blob, _ = _blobs()
    labels = cf.cluster_embedding(emb, method="hdbscan")
    non_noise = labels[labels != -1]
    assert len(np.unique(non_noise)) == 3, "should find the 3 injected blobs"
    # noise should be a small minority for cleanly separated blobs
    assert (labels == -1).mean() < 0.2


def test_kmeans_labels_every_point():
    emb, _, _ = _blobs()
    labels = cf.cluster_embedding(emb, method="kmeans", k=3)
    assert (labels == -1).sum() == 0, "k-means assigns every point"
    assert len(np.unique(labels)) == 3


def test_degenerate_embeddings():
    assert cf.cluster_embedding(np.empty((0, 2)), method="hdbscan").size == 0
    tiny = cf.cluster_embedding(np.zeros((2, 2)), method="hdbscan")
    assert (tiny == -1).all(), "too few points -> all noise, no crash"


# --------------------------------------------------------------------- #
#  feature_separability (Kruskal-Wallis epsilon-squared)
# --------------------------------------------------------------------- #

def test_separability_ranks_explaining_feature_top():
    emb, blob, df = _blobs()
    sep = cf.feature_separability(df, blob, ["explains", "noise_feat",
                                            "time_to_onset_sec"])
    top = sep.iloc[0]
    assert top["feature"] in ("explains", "time_to_onset_sec")
    assert top["eps2"] > 0.9, "a constant-within-cluster feature -> eps2 ~ 1"
    noise_row = sep[sep["feature"] == "noise_feat"].iloc[0]
    assert noise_row["eps2"] < 0.3, "a cluster-independent feature -> low eps2"
    # ranking is monotone in eps2
    assert list(sep["eps2"].dropna()) == sorted(sep["eps2"].dropna(),
                                                reverse=True)


def test_separability_handles_constant_and_single_group():
    emb, blob, df = _blobs()
    df = df.copy()
    df["flat"] = 7.0                                    # no rank variance anywhere
    sep = cf.feature_separability(df, blob, ["flat"])
    assert np.isnan(sep.iloc[0]["eps2"]), "a global constant has undefined eps2"
    one = cf.feature_separability(df, np.zeros(len(df), int), ["explains"])
    assert np.isnan(one.iloc[0]["eps2"]), "a single cluster -> no separability"


# --------------------------------------------------------------------- #
#  cluster_profiles
# --------------------------------------------------------------------- #

def test_profiles_recover_offsets_and_tto():
    emb, blob, df = _blobs()
    prof = cf.cluster_profiles(df, blob, ["explains", "noise_feat"])
    assert set(prof["cluster"]) == {0, 1, 2}
    assert (prof["n"] == 60).all()
    # `explains` monotonically increases with blob -> its robust-z median does too
    by_cluster = prof.set_index("cluster")["explains"].sort_index()
    assert by_cluster.is_monotonic_increasing
    # mean tto matches the injected per-blob value
    assert abs(prof.set_index("cluster").loc[1, "mean_tto_sec"] - 150.0) < 1e-6


def test_profiles_put_noise_last():
    emb, blob, df = _blobs()
    labels = blob.copy()
    labels[:5] = -1
    prof = cf.cluster_profiles(df, labels, ["explains"])
    assert prof.iloc[-1]["cluster"] == -1, "noise row sorts last"


# --------------------------------------------------------------------- #
#  cluster_quality + analyze
# --------------------------------------------------------------------- #

def test_quality_and_analyze():
    emb, blob, df = _blobs()
    q = cf.cluster_quality(emb, blob)
    assert q["n_clusters"] == 3 and q["n_points"] == len(df)
    assert q["silhouette"] is not None and q["silhouette"] > 0.5
    res = cf.analyze(emb, df, ["explains", "noise_feat", "time_to_onset_sec"],
                     method="kmeans", k=3)
    assert set(res) == {"labels", "separability", "profiles", "quality"}
    assert len(res["labels"]) == len(df)


# --------------------------------------------------------------------- #
#  time_to_onset_for -- shares the pre-onset join with build_matrix
# --------------------------------------------------------------------- #

def test_tto_matches_assign_next_onset():
    onsets = np.array([1000.0, 5000.0], dtype=float)
    ceilings = np.array([np.nan, 1000.0], dtype=float)   # 2nd seizure: 1000 s look-back
    t = np.array([4200.0, 4500.0, 4999.0, 3000.0, 6000.0], dtype=float)
    tto = time_to_onset_for(t, onsets, ceilings)
    _idx, ref_tto, keep = _assign_next_onset(t, onsets, ceilings)
    ref = np.where(keep, ref_tto, np.nan)
    np.testing.assert_allclose(tto, ref, equal_nan=True)
    # 3000 s is 2000 s before onset -> outside the 1000 s look-back -> NaN
    assert np.isnan(tto[3])
    # 6000 s is after the last onset -> NaN
    assert np.isnan(tto[4])
    # 4500 s is 500 s before onset -> inside -> 500
    assert abs(tto[1] - 500.0) < 1e-6


def test_tto_empty_inputs():
    assert np.isnan(time_to_onset_for(np.array([1.0]), np.empty(0),
                                      np.empty(0))).all()
    assert time_to_onset_for(np.empty(0), np.array([1.0]),
                             np.array([np.nan])).size == 0
