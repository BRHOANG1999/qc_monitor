"""matrix.build_null_matrix: fake-onset matrix scored by the unchanged
sliding_window_auc, and the per-fake independent assembly (FLAG 1).

Run with: pytest tests/test_periictal_null_matrix.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal.matrix import build_null_matrix, _empty_frame  # noqa: E402
from src.periictal.sliding_auc import sliding_window_auc, _REQUIRED  # noqa: E402

METRICS = ["m1", "m2"]
BAND_HI = 21600.0


def _epoch_cols(fake_onsets, *, step=30.0, seed=0):
    """Synthetic epoch columns: for each fake onset, dense stimuli every *step* s
    over the full (a-6h, a] lookback, with random feature values."""
    rng = np.random.default_rng(seed)
    ts, m1, m2 = [], [], []
    for a in fake_onsets:
        offs = np.arange(step, BAND_HI + 1, step)      # 30 s .. 6 h before a
        tt = a - offs
        ts.append(tt)
        m1.append(rng.normal(size=tt.size))
        m2.append(rng.normal(size=tt.size))
    t = np.concatenate(ts)
    n = t.size
    mcols = {"m1": np.concatenate(m1).astype(np.float32),
             "m2": np.concatenate(m2).astype(np.float32)}
    chan = np.array(["BCHxxxSLM"] * n, dtype=object)
    sess = np.array(["sess"] * n, dtype=object)
    rec = np.array(["rec"] * n, dtype=object)
    return (t, mcols, chan, sess, rec)


def test_null_matrix_scored_full_windows():
    T0 = 1_000_000_000.0 + BAND_HI
    fakes = np.array([T0, T0 + 8 * 3600, T0 + 16 * 3600])   # 8 h apart (>= 7 h)
    cols = _epoch_cols(fakes)
    m = build_null_matrix(fakes, cols, METRICS, window_sec=BAND_HI)
    # Contract: the columns sliding_window_auc needs, all pre-phase.
    assert set(_REQUIRED).issubset(m.columns)
    assert (m["phase"] == "pre").all()
    assert m["seizure_idx"].nunique() == 3
    res = sliding_window_auc(m, METRICS)
    assert res["n_seizures_used"] == 3, res["notes"]
    # Every fake is a clean, well-separated anchor -> all 12 windows valid.
    for ps in res["per_seizure"].values():
        assert ps["n_valid"] == res["offsets"].size == 12
    # AUCs are finite and >= 0.5 (rank AUC is normalised max(a,1-a)).
    g = np.array([res["group"][f] for f in METRICS])
    assert np.all(np.isfinite(g)) and np.all(g >= 0.5)


def test_flag1_independent_assembly_no_isi_truncation():
    # Two fakes only 2 h apart: the MATRIX must still carry pre rows for BOTH
    # (unique seizure_idx) -- per-fake assembly never lets inter-fake ISI truncate
    # a band at build time. (The scorer's guard may later drop the close one; that
    # is why the generator spaces fakes >= 7 h.)
    T0 = 1_000_000_000.0 + BAND_HI
    fakes = np.array([T0, T0 + 2 * 3600])
    cols = _epoch_cols(fakes)
    m = build_null_matrix(fakes, cols, METRICS, window_sec=BAND_HI)
    assert set(m["seizure_idx"].unique()) == {0, 1}
    # Each fake got its full (0, 6h] set of pre rows independently.
    for k in (0, 1):
        sub = m[m["seizure_idx"] == k]
        assert sub["time_to_onset_sec"].min() > 0
        assert sub["time_to_onset_sec"].max() <= BAND_HI + 1


def test_empty_epoch_cols_gives_empty_frame():
    empty = (np.empty(0), {m: np.empty(0, np.float32) for m in METRICS},
             np.empty(0, object), np.empty(0, object), np.empty(0, object))
    m = build_null_matrix(np.array([1.0, 2.0]), empty, METRICS)
    assert len(m) == 0
    assert set(_REQUIRED).issubset(m.columns) or m.equals(_empty_frame(METRICS))


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
