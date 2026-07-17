"""Preictal-vs-interictal labeling, discrimination, and forecasting
(src.periictal.forecast) — the Chang et al. 2026 modeling layer.

Synthetic event x metric matrices with planted (or absent) preictal/interictal
separation; asserts on AUC recovery, permutation p, labeling guards, and the
prospective logistic forecaster. Mirrors tests/test_periictal_trendtest.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.periictal import forecast as F  # noqa: E402


def _matrix(n_seizures=12, per_class=40, sep=1.0, isi_sec=7200.0, seed=0,
            extra=None):
    """Build a synthetic matrix: each seizure has *per_class* preictal +
    interictal stimuli; feature 'sep' is +sep (preictal) / -sep (interictal),
    'noise' is pure noise. *extra* seizures overrides ISI-driven spacing."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_seizures):
        onset = 1_000_000.0 + s * isi_sec
        for cls, lo, hi, mu in ((True, 60, 1800, sep), (False, 3600, 5400, -sep)):
            for _ in range(per_class):
                tto = rng.uniform(lo, hi)
                rows.append(dict(seizure_idx=s, seizure_onset_epoch=onset,
                                 phase="pre", time_to_onset_sec=tto,
                                 hour_of_day=(onset / 3600) % 24,
                                 sep=rng.normal(mu, 1.0), noise=rng.normal(0, 1)))
    return pd.DataFrame(rows)


def test_label_classes_counts_and_windows():
    lab = F.label_classes(_matrix())
    vc = lab["class"].value_counts().to_dict()
    assert vc.get("preictal", 0) == 12 * 40
    assert vc.get("interictal", 0) == 12 * 40
    # No row is labeled in the 30-60 min buffer band.
    tto = lab["time_to_onset_sec"].to_numpy()
    buf = (tto > 1800) & (tto < 3600)
    assert set(lab.loc[buf, "class"]) <= {""}


def test_clustered_seizure_preictal_dropped():
    # ISI = 20 min < 30 min -> preictal rows dropped (no clean preictal period).
    df = _matrix(n_seizures=6, isi_sec=1200.0)
    lab = F.label_classes(df, require_clean_preictal=True)
    # Every seizure after the first has ISI 20 min -> only the first (ISI=inf)
    # keeps preictal; interictal needs ISI>=60min so also mostly dropped.
    assert (lab["class"] == "preictal").sum() == 40   # only seizure 0


def test_feature_auc_separates_signal_from_noise():
    lab = F.label_classes(_matrix(sep=1.2))
    assert F.feature_auc(lab, "sep")["auc_norm"] > 0.75
    assert 0.45 <= F.feature_auc(lab, "noise")["auc_norm"] <= 0.6


def test_permutation_p_signal_vs_null():
    lab = F.label_classes(_matrix(sep=1.2))
    assert F.permutation_p(lab, "sep", n_perm=500, seed=1)["p"] < 0.05
    assert F.permutation_p(lab, "noise", n_perm=500, seed=1)["p"] > 0.2


def test_paired_seizure_test_is_honest():
    lab = F.label_classes(_matrix(sep=1.5))
    ps = F.paired_seizure_test(lab, "sep")
    assert ps["n_seizures"] == 12
    assert ps["auc"] > 0.8 and ps["p"] < 0.05
    assert F.paired_seizure_test(lab, "noise")["p"] > 0.1


def test_scan_features_ranks_and_fdr():
    lab = F.label_classes(_matrix(sep=1.2))
    rows = F.scan_features(lab, ["noise", "sep"], n_perm=300)
    assert rows[0]["feature"] == "sep"          # sorted by auc_norm desc
    assert rows[0]["q"] < 0.05 and rows[-1]["q"] > 0.2


def test_pdf_cdf_shapes():
    lab = F.label_classes(_matrix(sep=1.0))
    pc = F.pdf_cdf(lab, "sep", bins=30)
    assert pc["centers"].size == 30
    assert pc["pre_kde"] is not None and pc["inter_kde"] is not None
    assert pc["pre_cdf_x"].size == pc["n_pre"]
    assert np.isclose(pc["pre_cdf_y"][-1], 1.0)


def test_assign_epi_phase_equal_counts():
    df = _matrix(n_seizures=20)
    d2, k = F.assign_epi_phase(df, n_phases=5)
    assert k == 5
    counts = np.bincount(d2["epi_phase"].to_numpy())
    assert counts.min() == counts.max()          # equal seizure split
    # n_phases clamped to seizure count.
    _d3, k3 = F.assign_epi_phase(_matrix(n_seizures=3), n_phases=10)
    assert k3 == 3


def _phase_growing_matrix(n_seizures=20, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    feats = ["sum_power_low", "sum_power_high", "expfit_initial"]
    for s in range(n_seizures):
        onset = 1e6 + s * 7200
        amt = 0.3 + 1.2 * (s / n_seizures)
        for cls, lo, hi, sign in ((1, 60, 1800, 1), (0, 3600, 5400, -1)):
            for _ in range(60):
                row = dict(seizure_idx=s, seizure_onset_epoch=onset, phase="pre",
                           time_to_onset_sec=rng.uniform(lo, hi), hour_of_day=0.0,
                           freq_moment_high=rng.normal(0, 1),
                           freq_moment_low=rng.normal(0, 1))
                for f in feats:
                    row[f] = rng.normal(sign * amt, 1.0)
                rows.append(row)
    return pd.DataFrame(rows)


def test_logistic_forecast_recovers_signal_and_rising_auc():
    lab = F.label_classes(_phase_growing_matrix())
    res = F.logistic_forecast(lab, n_phases=5)
    assert res["mean_auc"] > 0.8
    assert res["phase_auc_slope"] > 0            # AUC rises in later phases
    # Informative features carry ~all the weight; noise features ~0.
    coef = res["coefficients"]
    assert coef["expfit_initial"] > 0.1
    assert coef["freq_moment_high"] < 0.1


def test_logistic_forecast_null_is_chance():
    # No separation -> forecaster ~chance.
    lab = F.label_classes(_matrix(sep=0.0, n_seizures=16, per_class=80))
    lab["sum_power_low"] = lab["sep"]           # both features are pure noise
    lab["sum_power_high"] = lab["noise"]
    lab["expfit_initial"] = np.random.default_rng(9).normal(size=len(lab))
    lab["freq_moment_high"] = np.random.default_rng(8).normal(size=len(lab))
    lab["freq_moment_low"] = np.random.default_rng(7).normal(size=len(lab))
    res = F.logistic_forecast(lab, n_phases=4)
    assert abs(res["mean_auc"] - 0.5) < 0.12


def test_select_best_features_finds_informative():
    lab = F.label_classes(_phase_growing_matrix())
    best = F.select_best_features(
        lab, ["sum_power_low", "sum_power_high", "expfit_initial",
              "freq_moment_high", "freq_moment_low"], k=3, n_phases=4)
    assert set(best) == {"sum_power_low", "sum_power_high", "expfit_initial"}
