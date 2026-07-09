"""Pre-ictal validation: leave-one-SEIZURE-out CV (whole seizures held out, no
sample leakage) + surrogate-null percentile/p.

Run with: pytest tests/test_preictal_validation.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal import validation  # noqa: E402


def _ramp_seizure(rng, n_bins=4, per=60):
    # nearest bin (0) highest -> a pre-ictal ramp toward onset.
    return [rng.normal(10 - 3 * b, 1.0, per) for b in range(n_bins)]


def _flat_seizure(rng, n_bins=4, per=60):
    return [rng.normal(5.0, 1.0, per) for _ in range(n_bins)]


def test_loso_ramp_positive_flat_zero():
    rng = np.random.default_rng(0)
    ramp = [_ramp_seizure(rng) for _ in range(6)]
    out = validation.loso_collapse(ramp)
    assert out["n_folds"] == 6                       # one fold per seizure
    assert out["mean"] > 0.2 and out["ci_lo"] > 0.0  # robust positive gradient
    flat = [_flat_seizure(rng) for _ in range(6)]
    assert abs(validation.loso_collapse(flat)["mean"]) < 0.06


def test_loso_holds_out_whole_seizure_no_leakage():
    # Each seizure's bin b carries a UNIQUE marker value; the held-out seizure's
    # values must be ENTIRELY absent from every fold's pool.
    n_bins = 3
    per_seizure = [[np.array([s * 100 + b], dtype=float) for b in range(n_bins)]
                   for s in range(4)]
    for h in range(4):
        keep = [i for i in range(4) if i != h]
        pooled = validation._pool(per_seizure, keep, n_bins)
        held = {h * 100 + b for b in range(n_bins)}
        present = set(np.concatenate(pooled).tolist())
        assert held.isdisjoint(present)              # zero leakage of seizure h


def test_null_stats_significant_and_chance():
    rng = np.random.default_rng(1)
    surr = rng.normal(0.0, 0.05, 300)
    sig = validation.null_stats(0.35, surr)          # well above the null
    assert sig["p"] < 0.01 and sig["percentile"] > 99
    chance = validation.null_stats(0.0, surr)        # smack in the null
    assert 0.3 < chance["p"] < 0.7
    assert 30 < chance["percentile"] < 70
    assert np.isnan(validation.null_stats(0.1, [])["p"])
