"""Stage 0 oracle: the synthetic generators produce the structure they claim.

Run: pytest tests/test_evoked_shapes_synth.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_shapes import synth        # noqa: E402


def test_shapes_and_meta_columns():
    X, meta = synth.generate("discrete", 300, n_samples=96, n_templates=3, seed=0)
    assert X.shape == (300, 96)
    for col in ("animal", "session", "trial_time", "time_to_seizure",
                "stim_status", "kind", "true_label", "true_coord"):
        assert col in meta.columns
    assert set(np.unique(meta["true_label"])) <= {0, 1, 2}
    assert (meta["kind"] == "discrete").all()


def test_continuum_has_coords_not_labels():
    _X, meta = synth.generate("continuum", 300, n_samples=96, seed=1)
    assert (meta["true_label"] == -1).all()
    c = meta["true_coord"].to_numpy()
    assert np.all(np.isfinite(c)) and c.min() >= 0.0 and c.max() <= 1.0


def test_mixture_is_half_discrete_half_continuum():
    _X, meta = synth.generate("mixture", 400, n_samples=96, n_templates=3, seed=2)
    kinds = meta["kind"].to_numpy()
    assert (kinds == "discrete").sum() == 200
    assert (kinds == "continuum").sum() == 200


def test_templates_are_distinct_unit_peak():
    tpl = synth.make_templates(128, 3)
    assert np.allclose(np.max(np.abs(tpl), axis=1), 1.0)
    # pairwise correlations well below 1 (distinct shapes)
    c = np.corrcoef(tpl)
    off = c[np.triu_indices(3, k=1)]
    assert np.all(np.abs(off) < 0.9)


def test_seed_is_deterministic():
    a, _ = synth.generate("mixture", 200, n_samples=64, seed=7)
    b, _ = synth.generate("mixture", 200, n_samples=64, seed=7)
    assert np.array_equal(a, b)


def test_bad_mode_raises():
    with pytest.raises(AssertionError):
        synth.generate("bogus", 10)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
