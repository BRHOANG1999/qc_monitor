"""Circadian-matched interictal reference (Part B, B4).

The decisive test: a feature that depends ONLY on clock time (pure circadian, no
preictal effect), with seizures recurring at a fixed clock time. The standard
same-day interictal window (60-90 min pre) sits ~1 h earlier than the preictal
window, so a circadian gradient inflates the same-day AUC; the circadian-MATCHED
interictal (same clock time, non-seizure days) must collapse it to ~0.5. A feature
independent of clock time must give ~0.5 both ways.

Run: pytest tests/test_periictal_circadian.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import circadian as cz          # noqa: E402
from src.periictal import trial_series as ts       # noqa: E402

_T0 = 1_600_000_000.0        # fixed epoch (deterministic)
_DAY = 86400.0


def _build(feature_fn, seed=0, dt=60.0, days=30):
    n = int(days * _DAY / dt)
    secs = _T0 + np.arange(n) * dt
    rng = np.random.default_rng(seed)
    vals = feature_fn(secs, rng)
    onsets = np.array([_T0 + d * _DAY + 0.5 * _DAY for d in (5, 15, 25)])  # same clock time
    return ts.assemble_trial_series("A", "feat", "ch", secs, vals, onsets)


def test_circadian_effect_collapses_under_matching():
    # feature == clock hour + noise: purely circadian, no preictal signal
    s = _build(lambda secs, rng: cz.hour_of_day(secs) + 0.3 * rng.standard_normal(secs.size))
    out = cz.circadian_matched_auc(s, min_n=15, hour_tol=0.75)
    assert out["n_seizures"] == 3, out["n_seizures"]
    # same-day AUC is inflated by the circadian gradient; matched collapses it
    assert out["sameday_pooled_auc"] > 0.75, out["sameday_pooled_auc"]
    assert abs(out["matched_pooled_auc"] - 0.5) < 0.15, out["matched_pooled_auc"]
    assert out["sameday_pooled_auc"] - out["matched_pooled_auc"] > 0.2


def test_non_circadian_feature_is_half_both_ways():
    # feature independent of clock time -> nothing to exploit, ~0.5 either way
    s = _build(lambda secs, rng: rng.standard_normal(secs.size), seed=1)
    out = cz.circadian_matched_auc(s, min_n=15, hour_tol=0.75)
    assert abs(out["matched_pooled_auc"] - 0.5) < 0.12, out["matched_pooled_auc"]
    assert abs(out["sameday_pooled_auc"] - 0.5) < 0.12, out["sameday_pooled_auc"]


def test_no_onsets_is_safe():
    secs = _T0 + np.arange(100) * 60.0
    s = ts.assemble_trial_series("A", "feat", "ch", secs, np.arange(100.0),
                                 np.array([]))
    out = cz.circadian_matched_auc(s)
    assert out["n_seizures"] == 0 and np.isnan(out["matched_pooled_auc"])
