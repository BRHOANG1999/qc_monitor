"""Headless preictal/interictal report renderers + stats CSV
(src.periictal.render additions + src.periictal.forecast_report).

Renders to a tmp dir and asserts the PNGs/CSV are produced and well-formed;
does not touch real evoked data (the figure inputs are synthetic forecast
outputs).
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.periictal import forecast as F          # noqa: E402
from src.periictal import forecast_report as FR  # noqa: E402
from src.periictal import render                 # noqa: E402


def _labeled(n_sz=10, per=50, sep=1.0, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    feats = ["expfit_initial", "sum_power_low", "sum_power_high"]
    for s in range(n_sz):
        onset = 1e6 + s * 7200
        for lo, hi, sign in ((60, 1800, 1), (3600, 5400, -1)):
            for _ in range(per):
                row = dict(seizure_idx=s, seizure_onset_epoch=onset, phase="pre",
                           time_to_onset_sec=rng.uniform(lo, hi), hour_of_day=0.0,
                           freq_moment_high=rng.normal(0, 1),
                           freq_moment_low=rng.normal(0, 1))
                for fcol in feats:
                    row[fcol] = rng.normal(sign * sep, 1.0)
                rows.append(row)
    return F.label_classes(pd.DataFrame(rows))


def test_render_functions_write_pngs(tmp_path):
    lab = _labeled()
    metrics = ["expfit_initial", "sum_power_low", "sum_power_high",
               "freq_moment_high", "freq_moment_low"]
    scan = F.scan_features(lab, metrics, n_perm=100)
    res = F.logistic_forecast(lab, n_phases=4)
    render.feature_auc_bar(scan, str(tmp_path / "auc.png"), title="t")
    render.pdf_cdf_panel(F.pdf_cdf(lab, "expfit_initial"),
                         str(tmp_path / "pc.png"), feature="expfit_initial",
                         title="t")
    render.forecast_panel(res, str(tmp_path / "fc.png"), title="t")
    for name in ("auc.png", "pc.png", "fc.png"):
        p = tmp_path / name
        assert p.exists() and p.stat().st_size > 1000


def test_stats_csv_has_sections(tmp_path):
    lab = _labeled()
    metrics = ["expfit_initial", "sum_power_low", "sum_power_high"]
    scan = F.scan_features(lab, metrics, n_perm=100)
    res = F.logistic_forecast(lab, n_phases=4)
    path = tmp_path / "stats.csv"
    FR._write_stats_csv(str(path), scan, res, n_seizures=10)
    with path.open() as f:
        rows = list(csv.reader(f))
    sections = {r[0] for r in rows if r}
    assert {"feature", "forecaster", "phase", "coefficient"} <= sections
    # the informative feature ranks at the top of the feature section
    feat_rows = [r for r in rows if r and r[0] == "feature"]
    assert float(feat_rows[0][2]) >= 0.7
