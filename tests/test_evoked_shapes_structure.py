"""Stage 2: the dip test and the discrete-vs-continuum verdict recover known truth.

Run: pytest tests/test_evoked_shapes_structure.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_shapes import synth, preprocess as pp, structure as st   # noqa: E402


def test_dip_flags_bimodal_not_unimodal():
    rng = np.random.default_rng(0)
    uni = rng.normal(0.0, 1.0, 1200)
    bi = np.concatenate([rng.normal(-3.0, 0.5, 600), rng.normal(3.0, 0.5, 600)])
    assert st.dip_test(uni, n_boot=200, seed=0)["p"] > 0.2
    assert st.dip_test(bi, n_boot=200, seed=0)["p"] < 0.05


def test_dip_statistic_ordering():
    rng = np.random.default_rng(1)
    uni = rng.normal(0.0, 1.0, 1000)
    bi = np.concatenate([rng.normal(-4.0, 0.4, 500), rng.normal(4.0, 0.4, 500)])
    assert st.dip_statistic(bi) > st.dip_statistic(uni)


def _pcs(mode, seed):
    X, meta = synth.generate(mode, 800, n_samples=96, n_templates=3, seed=seed)
    Xn = pp.preprocess(X, mode="l2", qc_min_smoothness=0.3)["Xn"]
    return st.shape_pca(Xn, seed=0)["Z"], Xn


def test_verdict_discrete_for_templates():
    Z, Xn = _pcs("discrete", 3)
    dips = st.dip_scan(Z, n_axes=4, n_boot=200, seed=0)
    out = st.summarize(pc_dips=dips, diff_dip={"p": 1.0},
                       corr_vals=st.pairwise_correlations(Xn, seed=0),
                       explained=np.array([0.5, 0.3]))
    assert out["discrete"] is True


def test_verdict_continuum_for_blend():
    Z, Xn = _pcs("continuum", 4)
    dips = st.dip_scan(Z, n_axes=4, n_boot=200, seed=0)
    out = st.summarize(pc_dips=dips, diff_dip={"p": 0.001},   # context, must be ignored
                       corr_vals=st.pairwise_correlations(Xn, seed=0),
                       explained=np.array([0.6, 0.2]))
    assert out["discrete"] is False


def test_preprocess_is_gain_invariant():
    X, _meta = synth.generate("discrete", 400, n_samples=96, n_templates=3,
                              gain_range=(1.0, 1.0), jitter=0, noise=0.02, seed=5)
    rng = np.random.default_rng(9)
    Xg = X * rng.uniform(0.3, 3.0, size=(X.shape[0], 1))
    a = pp.preprocess(X, mode="l2", qc_min_smoothness=0.0)["Xn"]
    b = pp.preprocess(Xg, mode="l2", qc_min_smoothness=0.0)["Xn"]
    # L2-normalized shapes are identical up to the injected scaling.
    assert np.allclose(a, b, atol=1e-6)


def test_diffusion_map_shapes():
    _Z, Xn = _pcs("mixture", 2)
    d = st.diffusion_map(Xn, n_components=2, cap=500, seed=0)
    assert d["coords"].shape[1] == 2
    assert d["coords"].shape[0] == d["rows"].size


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
