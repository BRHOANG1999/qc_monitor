"""Peri-ictal ERP-image: the sliding trial-average and row block-mean reducers
(pure), and the raw-trace gather (integration, seeded dummy .mat).

Run with: pytest tests/test_periictal_erpimage.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal.erpimage import (column_waveform,  # noqa: E402
                                    decimate_rows, gather_leadup_trials,
                                    sliding_trial_average, window_starts)
from src.utils.evoked_features import FeatureConfig  # noqa: E402


# ------------------------------------------------------------------ #
#  sliding_trial_average
# ------------------------------------------------------------------ #

def test_sliding_trial_average_window_and_overlap():
    # 10 trials x 4 samples; every sample of trial i == i, so a column's mean
    # equals the mean trial index in its window.
    trials = np.arange(10)[:, None] * np.ones((10, 4))
    tto = np.arange(10, 0, -1, dtype=float)
    z, col_tto = sliding_trial_average(trials, tto, n=4, step=2)
    assert z.shape == (4, 4)                          # [samples x cols]
    assert np.allclose(z[0], [1.5, 3.5, 5.5, 7.5])    # means of 0-3,2-5,4-7,6-9
    assert col_tto.shape == (4,)


def test_sliding_always_includes_last_window():
    trials = np.arange(10)[:, None] * np.ones((10, 2))
    tto = np.arange(10, 0, -1, dtype=float)
    z, _ = sliding_trial_average(trials, tto, n=3, step=4)
    assert z.shape[1] == 3                            # starts 0,4 + last window 7


def test_sliding_clamps_n_to_trial_count():
    trials = np.ones((3, 5))
    z, ct = sliding_trial_average(trials, np.array([3., 2., 1.]), n=99, step=1)
    assert z.shape == (5, 1) and ct.shape == (1,)     # one column = all trials


def test_window_starts_and_column_waveform():
    assert window_starts(10, 3, 4) == [0, 4, 7]       # last window always included
    assert window_starts(3, 99, 1) == [0]             # n clamped to trial count
    trials = np.array([[1., 1], [3, 3], [5, 5], [7, 7]])
    tto = np.array([4., 3, 2, 1])
    mean, sd, ctto, ncol, nin = column_waveform(trials, tto, n=2, step=2, col=0)
    assert np.allclose(mean, [2, 2]) and nin == 2 and ncol == 2   # trials 0,1
    mean2, _, _, _, _ = column_waveform(trials, tto, 2, 2, 1)
    assert np.allclose(mean2, [6, 6])                 # col 1 = trials 2,3
    # out-of-range col is clamped, not an error.
    m3, _, _, _, _ = column_waveform(trials, tto, 2, 2, 99)
    assert np.allclose(m3, [6, 6])


# ------------------------------------------------------------------ #
#  decimate_rows
# ------------------------------------------------------------------ #

def test_decimate_rows_block_mean():
    z = np.arange(1800.0).reshape(900, 2)
    rm = np.linspace(0, 200, 900)
    z2, rm2 = decimate_rows(z, rm, max_rows=300)
    assert z2.shape == (300, 2) and rm2.shape == (300,)
    assert np.allclose(z2[0], z[0:3].mean(axis=0))    # 900//300 = 3 per block


def test_decimate_rows_noop_when_small():
    z = np.ones((100, 2))
    z2, rm2 = decimate_rows(z, np.arange(100.0), max_rows=300)
    assert z2.shape == (100, 2)


# ------------------------------------------------------------------ #
#  gather_leadup_trials (reads a real HDF5 dummy .mat)
# ------------------------------------------------------------------ #

def _seed_mat(evoked_dir, times_sec, n_samp=60):
    import h5py
    os.makedirs(evoked_dir, exist_ok=True)
    mat = os.path.join(
        evoked_dir, "sess__stimCopy_BCH040SR___2026_03_02__00_00_00_evoked.mat")
    ne = len(times_sec)
    traces = np.arange(ne)[:, None] * np.ones((ne, n_samp))   # trace i == i
    time_ms = np.linspace(-5.0, 200.0, n_samp)
    with h5py.File(mat, "w") as f:
        ch = f.create_group("allAnimalResults").create_group("BCH040SR")
        ch.create_dataset("stimulusTimes", data=np.asarray(times_sec, float))
        ch.create_dataset("stimulusPeakAmplitudes", data=np.zeros(ne))
        ch.create_dataset("stimulusTroughAmplitudes", data=np.zeros(ne))
        ch.create_dataset("evokedData", data=traces)          # [epochs x samples]
        ch.create_dataset("timeAxis", data=time_ms)
    return mat


def test_gather_orders_and_windows(tmp_path):
    evoked_dir = str(tmp_path / "evoked")
    _seed_mat(evoked_dir, times_sec=[10.0, 20.0, 30.0])
    base = datetime.fromisoformat("2026-03-02T00:00:00").timestamp()
    onset = base + 3600.0                                     # 1 h after rec start
    cfg = FeatureConfig(window_start_ms=1.0, window_end_ms=200.0)
    res = gather_leadup_trials(evoked_dir, "BCH040", onset, lookback_sec=3600.0,
                               cfg=cfg)
    assert res["trials"].shape[0] == 3                        # all 3 in window
    # ordered by DESCENDING time-to-onset: stim@10s (tto 3590) first, @30s last.
    assert np.all(np.diff(res["tto"]) < 0)
    assert res["trials"][0, 0] == 0.0 and res["trials"][-1, 0] == 2.0
    assert res["row_ms"].min() >= 1.0 and res["row_ms"].max() <= 200.0   # windowed


def test_gather_symmetric_spans_onset(tmp_path):
    # Trials before AND after onset; the default window is symmetric, so the
    # gather includes post-onset trials (negative time-to-onset).
    evoked_dir = str(tmp_path / "evoked")
    _seed_mat(evoked_dir, times_sec=[1800.0, 3000.0, 3600.0, 4200.0, 5400.0])
    base = datetime.fromisoformat("2026-03-02T00:00:00").timestamp()
    onset = base + 3600.0
    res = gather_leadup_trials(evoked_dir, "BCH040", onset, lookback_sec=1800.0)
    assert res["trials"].shape[0] == 5                # window [onset-30m, onset+30m]
    assert res["tto"].min() < 0 < res["tto"].max()    # spans the onset (tto=0)


def test_gather_empty_when_outside_window(tmp_path):
    evoked_dir = str(tmp_path / "evoked")
    _seed_mat(evoked_dir, times_sec=[10.0, 20.0])
    base = datetime.fromisoformat("2026-03-02T00:00:00").timestamp()
    onset = base + 100000.0                                   # far after the trials
    res = gather_leadup_trials(evoked_dir, "BCH040", onset, lookback_sec=60.0)
    assert res["trials"].shape[0] == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
