"""UMAP is optional -- an unimportable UMAP must degrade to PCA, not fail.

Regression cover for the peri-ictal build dying with
``ImportError: Numba needs NumPy 2.4 or less. Got NumPy 2.5.``

umap-learn pulls in numba, which pins a maximum NumPy. Upgrading numpy past
that pin breaks ``import umap`` with a message naming numba -- so an optional,
explicitly "figure, not evidence" projection took down the whole build, and the
error read like a code bug rather than an environment one.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.periictal import embed as EM  # noqa: E402

_METRICS = ["m1", "m2"]


def _df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "m1": rng.normal(size=n),
        "m2": rng.normal(size=n),
        "seizure_idx": np.repeat([0, 1], n // 2),
        "lead_bin": np.tile([0, 1], n // 2),
    })


def test_falls_back_to_pca_when_umap_import_fails(monkeypatch):
    """The actual incident: numba/numpy conflict must not fail the build."""
    monkeypatch.setattr(EM, "_umap_unavailable",
                        lambda: "Numba needs NumPy 2.4 or less. Got NumPy 2.5.")
    res = EM.embed(_df(), metrics=_METRICS, method="umap")
    assert res["method"] == "pca"
    assert res["emb"].shape[0] == 40
    assert res["meta"]["fallback_from"] == "umap"
    assert "Numba" in res["meta"]["fallback_reason"]


def test_umap_is_not_even_attempted_when_unavailable(monkeypatch):
    """Guard the guard: _umap must not run, or we get the raw ImportError."""
    monkeypatch.setattr(EM, "_umap_unavailable", lambda: "nope")
    monkeypatch.setattr(EM, "_umap", lambda *a, **k: pytest.fail(
        "_umap called despite being reported unavailable"))
    assert EM.embed(_df(), metrics=_METRICS, method="umap")["method"] == "pca"


def test_too_few_points_falls_back_with_its_own_reason(monkeypatch):
    monkeypatch.setattr(EM, "_umap_unavailable", lambda: None)
    df = _df(n=4)
    res = EM.embed(df, metrics=_METRICS, method="umap", cap=4)
    assert res["method"] == "pca"
    assert "at least 5" in res["meta"]["fallback_reason"]


def test_pca_request_carries_no_fallback_marker(monkeypatch):
    monkeypatch.setattr(EM, "_umap_unavailable", lambda: "nope")
    meta = EM.embed(_df(), metrics=_METRICS, method="pca")["meta"]
    assert "fallback_from" not in meta


def test_umap_used_when_available(monkeypatch):
    """When the environment IS healthy, UMAP still runs and is labelled so."""
    monkeypatch.setattr(EM, "_umap_unavailable", lambda: None)
    monkeypatch.setattr(EM, "_umap",
                        lambda X, nc, nn, seed: (np.zeros((X.shape[0], nc)),
                                                 {"n_neighbors": nn}))
    res = EM.embed(_df(), metrics=_METRICS, method="umap")
    assert res["method"] == "umap"
    assert "fallback_from" not in res["meta"]


def test_unavailable_probe_reports_reason_not_raises():
    """Whatever this environment is, the probe returns a str or None."""
    out = EM._umap_unavailable()
    assert out is None or isinstance(out, str)
