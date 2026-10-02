"""Supervised log(time-to-seizure) regression (src/preictal_biomarker/regression.py):
target construction, pooled leave-one-seizure-out Ridge + held-out Spearman/AUC, the
circular-shift null, the incremental-beyond-clock test, and the trendtest adapter.

Synthetic-only (no DB). Run with: pytest tests/test_preictal_biomarker_regression.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.preictal_biomarker import regression as R      # noqa: E402
from src.preictal_biomarker import config as C          # noqa: E402


def _make_reg_df(n_seizures=6, per=150, far_per=250, ramp=1.0, clock=0.0, seed=0):
    """Synthetic df with a planted feature. *ramp*>0 => feature rises toward onset
    (so log-tto is recoverable from it); *clock*>0 injects a clock-hour dependence
    orthogonal to lead-time (the timing-only control). Returns (df, lead_onsets)."""
    rng = np.random.default_rng(seed)
    cap = C.REG_TARGET_CAP_SEC
    floor = C.REG_MIN_LEADTIME_SEC
    # IRREGULAR inter-onset gaps so time-since-last-seizure is NOT a clean proxy for
    # time-to-next (regular spacing would let the clock+tsl baseline win trivially).
    gaps = rng.uniform(2.5 * cap, 5.0 * cap, n_seizures)
    onsets = 1.0e6 + np.cumsum(gaps)
    t, feat = [], []
    for o, g in zip(onsets, gaps):
        tto = np.sort(rng.uniform(floor, cap, per))      # pre rows within cap
        hod = ((o - tto) % 86400.0) / 3600.0
        t.extend(o - tto)
        feat.extend(ramp * -np.log(tto) + clock * np.sin(hod / 24 * 2 * np.pi)
                    + rng.normal(0, 0.1, per))
        base = o - g / 2.0                               # far rows: > cap from any onset
        tf = base + rng.uniform(-g * 0.15, g * 0.15, far_per)
        hf = (tf % 86400.0) / 3600.0
        t.extend(tf)
        feat.extend(clock * np.sin(hf / 24 * 2 * np.pi) + rng.normal(0, 0.1, far_per))
    return (pd.DataFrame({"t_epoch": np.asarray(t, float),
                          "feat": np.asarray(feat, float)}), onsets)


# ------------------------------------------------------------------ #
#  target construction
# ------------------------------------------------------------------ #

def test_log_tto_target_pre_far_buffer():
    df, ons = _make_reg_df(seed=1)
    t = df["t_epoch"].to_numpy()
    y, grp, far, keep = R._log_tto_target(t, ons)
    assert (grp >= 0).sum() > 0 and far.sum() > 0        # both pre and far rows exist
    cap = C.REG_TARGET_CAP_SEC
    assert np.allclose(y[far], np.log(cap), atol=1e-6)   # far rows ceiling at log(cap)
    assert (grp[far] == -1).all()                        # far rows are not pre
    # a point 10 min after an onset is dropped by the post-ictal buffer
    t2 = np.array([ons[2] + 600.0])
    _, _, _, keep2 = R._log_tto_target(t2, ons)
    assert not keep2[0]


# ------------------------------------------------------------------ #
#  LOSO predict + metrics
# ------------------------------------------------------------------ #

def test_loso_recovers_planted_ramp():
    df, ons = _make_reg_df(ramp=1.0, seed=2)
    reg = R.regression_null(df, ons, features=["feat"], n_surr=50, seed=0)
    assert reg["spearman"] > 0.3                          # ramp is recoverable
    assert 0.0 < reg["p_spearman"] <= 0.1                # beats the shift null
    assert reg["y_true"].size == reg["y_pred"].size > 0


def test_no_signal_is_null():
    df, ons = _make_reg_df(ramp=0.0, seed=3)              # pure noise feature
    reg = R.regression_null(df, ons, features=["feat"], n_surr=50, seed=0)
    assert abs(reg["spearman"]) < 0.25
    assert reg["p_spearman"] > 0.1                        # inside the null


def test_metrics_both_computed():
    df, ons = _make_reg_df(ramp=1.0, seed=4)
    reg = R.regression_null(df, ons, features=["feat"], n_surr=10, seed=0)
    assert np.isfinite(reg["spearman"])
    assert set(reg["horizon_auc"]) == {float(h) for h in C.REG_HORIZONS_SEC}
    assert all(0.0 <= v <= 1.0 for v in reg["horizon_auc"].values()
               if np.isfinite(v))


def test_fewer_than_three_seizures_is_nan():
    df, ons = _make_reg_df(n_seizures=2, seed=5)
    try:
        R.regression_null(df, ons, features=["feat"], n_surr=5)
        raise AssertionError("should assert on < 3 seizures")
    except AssertionError as e:
        assert "3 lead seizures" in str(e)


# ------------------------------------------------------------------ #
#  incremental beyond clock + time-since-seizure
# ------------------------------------------------------------------ #

def test_incremental_evoked_adds_when_signal_present():
    df, ons = _make_reg_df(ramp=1.0, clock=0.0, seed=6)
    inc = R.incremental_evoked_regression(df, ons, features=["feat"], n_surr=60)
    assert inc["delta"] > 0.0                             # evoked adds beyond timing
    assert 0.0 < inc["p"] <= 1.0 and np.isfinite(inc["null_delta_med"])


def test_incremental_timing_only_adds_nothing():
    df, ons = _make_reg_df(ramp=0.0, clock=1.0, seed=7)   # feature ~ clock only
    inc = R.incremental_evoked_regression(df, ons, features=["feat"], n_surr=40)
    assert inc["delta"] < 0.1                             # ~no gain over clock+tsl


# ------------------------------------------------------------------ #
#  trendtest adapter round-trip
# ------------------------------------------------------------------ #

def test_trend_frame_roundtrip():
    df, ons = _make_reg_df(ramp=1.0, seed=8)
    frame = R._trend_frame(df, ons, feats=["feat"])
    for col in ("t_epoch", "time_to_onset_sec", "seizure_idx", "phase",
                "hour_of_day", "feat"):
        assert col in frame.columns
    assert (frame["phase"] == "pre").any()
    ps = R.per_seizure_regression(df, ons, feature="feat", n_boot=100, n_surr=50)
    assert len(ps["trend"]) >= 3
    assert all(r["rho"] < 0 for r in ps["trend"])        # rises to onset => rho<0
