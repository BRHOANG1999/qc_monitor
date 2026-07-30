"""Inference-hardening stats layer (plan tasks #7-#8).

The load-bearing test is the COMPOSITION-ONLY synthetic: a between-seizure offset
with NO within-seizure proximity signal must show a high POOLED AUC that collapses
toward 0.5 after interictal-referenced within-seizure standardization, while the
per-seizure AUC (rank-invariant to that standardization) stays ~0.5 throughout.
The PLANTED-POSITIVE synthetic is the matched positive control: a real within-
seizure trend must be recovered by the per-seizure AUC and slope forests. Asserting
on the per-seizure AUC for the null would pass against a no-op helper (rank
invariance), so the null asserts on the POOLED AUC -- see
docs/periictal_methods_evidence.md, L3.

Run: pytest tests/test_periictal_inference.py -q
"""

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import forecast as fc          # noqa: E402
from src.periictal import resample as rs          # noqa: E402
from src.periictal import trendtest as tt         # noqa: E402

_DAY = 86400.0


def _synth(*, effect=0.0, offsets=(3.0, 0.0, -3.0),
           counts=((300, 60), (180, 180), (60, 300)), noise=1.0, seed=0):
    """A peri-ictal matrix: 3 seizures, each with preictal (tto in (0,1800]) and
    interictal (tto in [3600,5400]) rows. ``offsets`` set a between-seizure
    baseline; ``effect`` plants a within-seizure proximity trend (feature rises as
    onset approaches); ``counts`` = (n_pre, n_inter) per seizure (unequal counts
    correlated with offset drive the composition artifact)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, (off, (n_pre, n_inter)) in enumerate(zip(offsets, counts)):
        onset = s * 10.0 * _DAY               # far apart -> clean ISI
        for n, lo, hi, is_pre in ((n_pre, 1.0, 1800.0, True),
                                  (n_inter, 3600.0, 5400.0, False)):
            tto = np.linspace(lo, hi, n)
            sig = effect * (1800.0 - tto) / 1800.0 if is_pre else 0.0
            feat = off + sig + rng.normal(0.0, noise, n)
            for j in range(n):
                rows.append({"seizure_idx": s, "seizure_onset_epoch": onset,
                             "time_to_onset_sec": float(tto[j]), "phase": "pre",
                             "t_epoch": onset - float(tto[j]),
                             "hour_of_day": 12.0, "feat": float(feat[j])})
    return fc.label_classes(pd.DataFrame(rows))


def _pooled_auc(df, feature="feat"):
    return fc.feature_auc(df, feature)["auc"]


# --------------------------------------------------- composition null --- #

def test_null_pooled_auc_collapses_after_standardization():
    df = _synth(effect=0.0, seed=1)
    before = _pooled_auc(df)
    after = _pooled_auc(fc.standardize_to_interictal(df, features=["feat"]))
    assert before > 0.75, before                     # composition inflates pooled
    assert abs(after - 0.5) < 0.1, after             # ...and standardization kills it


def test_null_per_seizure_auc_is_chance_and_standardization_invariant():
    df = _synth(effect=0.0, seed=2)
    forest = fc.per_seizure_auc_forest(df, "feat", n_boot=200, min_n=5)
    aucs = np.array([f["auc"] for f in forest])
    assert forest and len(forest) == 3
    assert np.all(np.abs(aucs - 0.5) < 0.15), aucs   # no within-seizure signal
    # rank invariance: per-seizure AUC identical before/after standardization
    std = fc.per_seizure_auc_forest(fc.standardize_to_interictal(df, features=["feat"]), "feat",
                                    n_boot=200, min_n=5)
    assert [f["auc"] for f in std] == list(aucs)     # bit-identical (A2 / L3)


def test_null_per_seizure_slope_is_flat():
    df = _synth(effect=0.0, seed=3)
    forest = tt.per_seizure_slope_forest(df, "feat", n_boot=100, min_n=5)
    for f in forest:
        assert f["ci_lo"] <= 0.0 <= f["ci_hi"], f     # CI straddles zero slope


# ---------------------------------------------------- planted positive --- #

def test_positive_recovers_within_seizure_effect():
    df = _synth(effect=3.0, offsets=(0.0, 0.0, 0.0),
                counts=((200, 200), (200, 200), (200, 200)), seed=4)
    forest = fc.per_seizure_auc_forest(df, "feat", n_boot=200, min_n=5)
    aucs = np.array([f["auc"] for f in forest])
    assert np.all(aucs > 0.6), aucs                   # preictal > interictal
    slopes = tt.per_seizure_slope_forest(df, "feat", n_boot=100, min_n=5)
    assert np.all([s["slope"] < 0 for s in slopes])   # feature falls as tto rises
    assert np.all([s["rho"] < 0 for s in slopes])


# -------------------------------------------------- standardization unit --- #

def test_standardize_to_interictal_zeroes_the_reference():
    df = _synth(effect=0.0, seed=5)
    std = fc.standardize_to_interictal(df, features=["feat"])
    for s in (0, 1, 2):
        ref = std[(std["seizure_idx"] == s) & (std["class"] == "interictal")]["feat"]
        assert abs(float(ref.mean())) < 1e-9, s
        assert abs(float(ref.std(ddof=1)) - 1.0) < 1e-6, s


# ------------------------------------------------- dependence-aware CI --- #

def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_data():
    rng = np.random.default_rng(6)
    n, phi = 4000, 0.9
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()          # AR(1), strong autocorr
    assert rs.autocorr_time(x) > 3.0
    blk = rs.block_length(x)
    assert blk > 1
    # SE of the mean: block bootstrap must exceed the (too-tight) iid bootstrap
    r = np.random.default_rng(7)
    block_se = np.std([rs.moving_block_resample(x, blk, r).mean()
                       for _ in range(400)])
    iid_se = np.std([rs.moving_block_resample(x, 1, r).mean()
                     for _ in range(400)])
    assert block_se > 1.5 * iid_se, (block_se, iid_se)


# ------------------------------------------------------ sign-flip floor --- #

def test_sign_flip_floor_values():
    assert tt.sign_flip_floor(3) == 0.25
    assert tt.sign_flip_floor(5) == 0.0625
    assert tt.sign_flip_floor(6) == 0.03125
    assert tt.sign_flip_floor(6) < 0.05 <= tt.sign_flip_floor(5)
    assert tt.sign_flip_floor(1) == 1.0


# ------------------------------------------- circular-shift surrogate --- #

def test_circular_shift_surrogate_reports_effective_shifts():
    df = _synth(effect=3.0, offsets=(0.0, 0.0, 0.0),
                counts=((200, 200), (200, 200), (200, 200)), seed=8)
    out = tt.circular_shift_surrogate_p(df, "feat", n_surrogates=200, seed=0)
    assert 0.0 < out["p"] <= 1.0
    assert out["n_seizures"] == 3
    # effective independent offsets < the raw pre-onset length (autocorrelation)
    assert np.isfinite(out["n_eff_shifts"]) and out["n_eff_shifts"] < 400
    assert out["observed"] > 0.2                      # a real onset-aligned trend
