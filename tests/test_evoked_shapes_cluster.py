"""Stage 3: correlation k-means recovers the planted templates and is scale/latency
robust.

Run: pytest tests/test_evoked_shapes_cluster.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_shapes import synth, preprocess as pp, cluster as cl   # noqa: E402


def _prep(mode="discrete", n=800, seed=0, jitter=4, align=False):
    X, meta = synth.generate(mode, n, n_samples=96, n_templates=3, jitter=jitter,
                             seed=seed)
    pre = pp.preprocess(X, mode="l2", align=align, qc_min_smoothness=0.3)
    return pre["Xn"], meta["true_label"].to_numpy()[pre["keep"]]


def test_recovers_three_templates():
    Xn, truth = _prep(seed=2)
    res = cl.correlation_kmeans(Xn, 3, seed=0)
    assert res["templates"].shape[0] == 3
    assert adjusted_rand_score(truth, res["labels"]) > 0.95


def test_gain_scaling_does_not_change_partition():
    X, meta = synth.generate("discrete", 700, n_samples=96, n_templates=3,
                             gain_range=(1.0, 1.0), jitter=0, noise=0.03, seed=3)
    rng = np.random.default_rng(1)
    Xg = X * rng.uniform(0.3, 3.0, size=(X.shape[0], 1))
    pre = pp.preprocess(Xg, mode="l2", qc_min_smoothness=0.3)
    res = cl.correlation_kmeans(pre["Xn"], 3, seed=0)
    truth = meta["true_label"].to_numpy()[pre["keep"]]
    assert adjusted_rand_score(truth, res["labels"]) > 0.95


def test_alignment_absorbs_latency_jitter():
    Xn, truth = _prep(seed=4, jitter=8, align=True)
    res = cl.correlation_kmeans(Xn, 3, seed=0)
    assert adjusted_rand_score(truth, res["labels"]) > 0.9


def test_assign_by_template_matches_fit_labels():
    Xn, _truth = _prep(seed=5)
    res = cl.correlation_kmeans(Xn, 3, seed=0)
    reassigned = cl.assign_by_template(Xn, res["templates"])
    assert adjusted_rand_score(res["labels"], reassigned) > 0.98


def test_max_offdiag_corr_rises_with_k():
    Xn, _truth = _prep(seed=6)
    sweep = cl.cluster_sweep(Xn, k_range=(2, 6), seed=0)
    m3 = cl.max_offdiag_corr(sweep[3]["templates"])
    m6 = cl.max_offdiag_corr(sweep[6]["templates"])
    assert m6 >= m3       # over-splitting yields more similar templates


def test_dtw_kmeans_optional():
    Xn, _truth = _prep(n=200, seed=7)
    notes = []
    out = cl.dtw_kmeans(Xn, 3, seed=0, log=notes.append)
    # tslearn is not installed in this env: must degrade to None with a note.
    if out is None:
        assert notes and "tslearn" in notes[0]
    else:
        assert out["labels"].shape[0] == Xn.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
