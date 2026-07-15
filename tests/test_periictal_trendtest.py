"""Per-seizure trend test (src/periictal/trendtest.py): the seizure-as-unit
collapse, the across-seizure signed-rank / sign-test, the post-ictal positive
control, the BH-FDR feature scan, and the seeded permutation null.

Run with: pytest tests/test_periictal_trendtest.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import trendtest as tt              # noqa: E402


def _make_df(n_seizures=8, per=60, trend=0.0, seed=0, phase="pre",
             hour_effect=0.0, feature="line_length"):
    """Synthetic matrix slice. *trend* > 0 => feature RISES toward onset (Spearman
    rho vs time-to-onset is NEGATIVE); *hour_effect* > 0 injects a circadian
    dependence orthogonal to lead-time."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_seizures):
        tto = np.sort(rng.uniform(1.0, 3600.0, per))
        if phase == "post":
            tto = -tto
        hod = rng.uniform(0, 24, per)
        val = (trend * -np.log10(np.abs(tto))
               + hour_effect * np.sin(hod / 24.0 * 2 * np.pi)
               + rng.normal(0.0, 0.05, per))
        for k in range(per):
            rows.append({"seizure_idx": sid, "time_to_onset_sec": float(tto[k]),
                         "hour_of_day": float(hod[k]), "phase": phase,
                         feature: float(val[k])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  per_seizure_trend + across_seizure_test
# ------------------------------------------------------------------ #

def test_known_negative_trend_is_significant():
    df = _make_df(n_seizures=8, trend=1.0, seed=1)
    per = tt.per_seizure_trend(df, "line_length")
    assert len(per) == 8
    assert all(r["rho"] < 0 for r in per)              # every seizure rises to onset
    res = tt.across_seizure_test([r["rho"] for r in per])
    assert res["direction"] == "negative" and res["n_neg"] == 8
    assert res["p"] < 0.05                              # consistent -> significant


def test_null_trend_not_significant():
    df = _make_df(n_seizures=8, trend=0.0, seed=7)     # feature ⟂ lead-time
    res = tt.across_seizure_test(
        [r["rho"] for r in tt.per_seizure_trend(df, "line_length")])
    assert res["n_seizures"] == 8
    assert res["p"] > 0.05                              # scattered rhos -> null


def test_min_n_skips_sparse_seizures():
    df = _make_df(n_seizures=4, per=60, trend=1.0, seed=2)
    # shrink one seizure below min_n
    sparse = df[df["seizure_idx"] == 0].iloc[:3]
    rest = df[df["seizure_idx"] != 0]
    small = pd.concat([sparse, rest], ignore_index=True)
    per = tt.per_seizure_trend(small, "line_length", min_n=5)
    assert 0 not in {r["seizure_idx"] for r in per}    # sparse seizure dropped
    assert len(per) == 3


def test_hour_rho_detects_circadian():
    df = _make_df(n_seizures=6, trend=0.0, hour_effect=5.0, seed=3)
    per = tt.per_seizure_trend(df, "line_length")
    assert np.nanmedian([abs(r["hour_rho"]) for r in per]) > 0.5


def test_across_seizure_sign_fallback_small_n():
    # < 6 seizures -> binomial sign test, not Wilcoxon; still a valid p.
    res = tt.across_seizure_test([-0.5, -0.4, -0.6, -0.3])
    assert res["n_seizures"] == 4 and res["n_neg"] == 4
    assert 0.0 < res["p"] <= 1.0
    # all-zero rhos -> p == 1.0 (no evidence), no crash
    assert tt.across_seizure_test([0.0, 0.0, 0.0])["p"] == 1.0
    assert tt.across_seizure_test([])["n_seizures"] == 0


# ------------------------------------------------------------------ #
#  positive control + scan + surrogate
# ------------------------------------------------------------------ #

def test_positive_control_on_post_onset():
    post = _make_df(n_seizures=8, trend=1.0, seed=4, phase="post")
    pc = tt.positive_control(post, "line_length")
    assert pc["available"] is True and pc["n_seizures"] == 8
    assert pc["p"] < 0.05                              # post-ictal effect detected
    # a pre-only frame has no post rows -> control unavailable
    pre = _make_df(n_seizures=8, trend=1.0, seed=4, phase="pre")
    pc2 = tt.positive_control(pre, "line_length")
    assert pc2["available"] is False and pc2["n_seizures"] == 0


def test_scan_all_features_fdr_ranks_signal_first():
    strong = _make_df(n_seizures=8, trend=1.0, seed=5, feature="line_length")
    strong["rms_amplitude"] = np.random.default_rng(9).normal(size=len(strong))
    # add matching post rows so the positive-control column is populated
    post = _make_df(n_seizures=8, trend=1.0, seed=5, phase="post",
                    feature="line_length")
    post["rms_amplitude"] = np.random.default_rng(10).normal(size=len(post))
    df = pd.concat([strong, post], ignore_index=True)
    rows = tt.scan_all_features(df, ["line_length", "rms_amplitude"])
    assert {r["feature"] for r in rows} == {"line_length", "rms_amplitude"}
    assert all("q" in r for r in rows)
    # sorted by q ascending; the real-signal feature ranks first
    assert rows[0]["feature"] == "line_length"
    assert rows[0]["q"] <= rows[-1]["q"] or not np.isfinite(rows[-1]["q"])
    assert rows[0]["post_available"] is True


def test_surrogate_p_seeded_and_bounded():
    import pytest
    df = _make_df(n_seizures=8, trend=1.0, seed=6)
    a = tt.surrogate_p(df, "line_length", n_perm=80, seed=0)
    b = tt.surrogate_p(df, "line_length", n_perm=80, seed=0)
    assert a["p"] == b["p"] and a["n_perm"] == 80        # deterministic
    assert a["p"] < 0.1                                  # strong trend -> small p
    with pytest.raises(AssertionError):
        tt.surrogate_p(df, "line_length", n_perm=0)
    with pytest.raises(AssertionError):
        tt.surrogate_p(df, "line_length", n_perm=10**9)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
