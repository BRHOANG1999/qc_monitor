"""Stage 4: the chosen k range brackets the true k and stability beats the null.

Run: pytest tests/test_evoked_shapes_select_k.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_shapes import synth, preprocess as pp, cluster as cl   # noqa: E402
from src.evoked_shapes import structure as st, select_k as sk          # noqa: E402


def _discrete(seed=1, n=800):
    X, meta = synth.generate("discrete", n, n_samples=96, n_templates=3, seed=seed)
    pre = pp.preprocess(X, mode="l2", qc_min_smoothness=0.3)
    return pre["Xn"], meta["session"].to_numpy()[pre["keep"]]


def test_range_brackets_true_k():
    Xn, sessions = _discrete()
    Z = st.shape_pca(Xn, seed=0)["Z"]
    templates_by_k = {k: r["templates"]
                      for k, r in cl.cluster_sweep(Xn, k_range=(2, 6), seed=0).items()}
    bic = sk.gmm_bic_curve(Z, (2, 6), seed=0)
    heldout = sk.gmm_heldout_ll(Z, sessions, (2, 6), seed=0)
    stab = sk.stability_curve(Xn, (2, 6), n_boot=30, seed=0)
    sel = sk.select_range(bic=bic, heldout=heldout, stability=stab,
                          templates_by_k=templates_by_k)
    assert sel["k_lo"] <= 3 <= sel["k_hi"]


def test_stability_excess_positive_at_true_k():
    Xn, _sessions = _discrete(seed=2)
    stab = sk.stability_curve(Xn, (2, 5), n_boot=30, seed=0)
    row = stab.loc[stab["k"] == 3].iloc[0]
    assert row["ari_excess"] > 0.05
    assert row["ari"] >= row["ari_null"] - 1e-9


def test_split_half_small_drop_at_true_k():
    Xn, sessions = _discrete(seed=3)
    out = sk.split_half(Xn, sessions, 3, seed=0)
    assert np.isfinite(out["corr_train"]) and np.isfinite(out["corr_test"])
    assert out["drop"] < 0.15      # templates generalize to held-out sessions


def test_bic_curve_is_dict_over_k():
    Xn, _sessions = _discrete(seed=4)
    Z = st.shape_pca(Xn, seed=0)["Z"]
    bic = sk.gmm_bic_curve(Z, (2, 5), seed=0)
    assert set(bic.keys()) == {2, 3, 4, 5}
    assert all(np.isfinite(v) for v in bic.values())


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
