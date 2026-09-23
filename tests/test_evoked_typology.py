"""Evoked-response typology: the pure pieces of the unsupervised morphology
discovery (shape normalization, GMM shape clusters, NMF prototypes).

Run with: pytest tests/test_evoked_typology.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_typology import (normalize_shapes, shape_clusters,  # noqa: E402
                                 nmf_prototypes)


def _two_shapes(seed=0, per=120):
    """Two distinct morphologies at random amplitudes: a sharp early trough vs a
    slow broad bump. Shape (not amplitude) should separate them."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 80)
    a = -np.exp(-((t - 0.1) / 0.05) ** 2)                 # sharp early trough
    b = np.sin(np.pi * t) * 0.8                            # slow broad bump
    rows, lab = [], []
    for shape, k in ((a, 0), (b, 1)):
        for _ in range(per):
            amp = rng.uniform(0.2, 5.0)                    # random SIZE
            rows.append(amp * shape + rng.normal(0, 0.02, t.size))
            lab.append(k)
    return np.asarray(rows), np.asarray(lab)


def test_normalize_shapes_is_unit_peak_and_size_invariant():
    W = np.array([[0.0, 2.0, -4.0], [0.0, 1.0, -2.0]])    # same shape, 2x size
    Wn = normalize_shapes(W)
    assert np.allclose(np.nanmax(np.abs(Wn), axis=1), 1.0)
    np.testing.assert_allclose(Wn[0], Wn[1])              # size removed
    assert np.allclose(normalize_shapes(np.zeros((1, 4))), 0.0)  # flat -> zeros


def test_shape_clusters_recovers_two_types_despite_amplitude():
    W, lab = _two_shapes()
    res = shape_clusters(normalize_shapes(W), n_pca=5, k_range=(2, 4))
    assert res["k"] >= 2
    got = res["labels"]
    # cluster PURITY: each injected shape class is dominated by one predicted label
    pur = sum(np.bincount(got[lab == tc]).max() for tc in (0, 1))
    assert pur / len(lab) > 0.9                            # >90% purity


def test_nmf_prototypes_reconstructs_and_signs():
    W, _lab = _two_shapes()
    Wn = normalize_shapes(W)
    res = nmf_prototypes(Wn, k=2)
    assert res["components"].shape == (2, Wn.shape[1])
    assert res["weights"].shape == (len(Wn), 2)
    assert (res["weights"] >= 0).all()                    # non-negative mixture
    # signed prototypes capture both a negative-going and a positive-going shape
    comps = res["components"]
    assert comps.min() < -0.1 and comps.max() > 0.1
