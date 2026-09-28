"""Unit tests for the sliding-window ROC-AUC test (src/periictal/sliding_auc.py).

Pure — no app, no DB. Builds a synthetic pre-onset matrix where one feature
cleanly separates preictal from interictal and another is null, and checks the
window geometry, the guard, and the per-seizure/group aggregation.
"""

import numpy as np
import pandas as pd
import pytest

from src.periictal import sliding_auc as sa


def _make_full(n_seizures=3, isi=30000.0, step=10.0, seed=0):
    """Synthetic 'pre'-phase matrix: stimuli every `step` s out to ~6 h before
    each onset. `signal` is high in the preictal window and low elsewhere;
    `null` is the same everywhere. Onsets spaced by `isi` (>7 h → every seizure
    gets the full window grid)."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_seizures):
        onset = 1_000_000.0 + s * isi
        tto = np.arange(step, 22000.0, step)
        n = tto.size
        is_pre = tto <= 1800.0
        signal = np.where(is_pre, rng.normal(5.0, 1.0, n), rng.normal(0.0, 1.0, n))
        frames.append(pd.DataFrame({
            "phase": "pre",
            "time_to_onset_sec": tto,
            "seizure_idx": s,
            "seizure_onset_epoch": onset,
            "signal": signal,
            "null": rng.normal(0.0, 1.0, n),
        }))
    return pd.concat(frames, ignore_index=True)


def test_canonical_offsets_grid():
    off = sa.canonical_offsets(band_lo=3600, band_hi=21600, width=1800, n=12)
    assert off.size == 12
    assert off[0] == 3600.0
    assert off[-1] == 21600.0 - 1800.0            # last window ends at band_hi
    d = np.diff(off)
    assert np.allclose(d, d[0])                    # evenly spaced
    assert sa.canonical_offsets(n=1, band_lo=3600).tolist() == [3600.0]


def test_canonical_offsets_bad_geometry():
    with pytest.raises(AssertionError):
        sa.canonical_offsets(band_lo=3600, band_hi=4000, width=1800, n=4)  # band too narrow


def test_valid_offset_mask_guard():
    off = sa.canonical_offsets(band_lo=3600, band_hi=21600, width=1800, n=12)
    # First seizure (ISI = inf): every window valid.
    assert sa.valid_offset_mask(off, np.inf).all()
    # Short ISI: far windows fall within 1 h of the previous seizure -> dropped.
    # hi = isi - width - guard = 10000 - 1800 - 3600 = 4600 -> only the 3600 offset.
    m = sa.valid_offset_mask(off, 10000.0, width=1800, postictal_guard=3600)
    assert m[0] and not m[-1]
    assert int(m.sum()) == int((off <= 4600 + 1e-6).sum()) == 1


def test_default_guard_is_two_hours_post_seizure():
    """The operator's rule: a window is interictal only >= 2 h after the previous
    seizure. Pins the config default and the boundary."""
    from src.periictal import config as cfg
    assert cfg.SLIDING_POSTICTAL_GUARD_SEC == 7200.0
    isi, width = 6 * 3600.0, cfg.SLIDING_WIDTH_SEC
    # An offset whose far edge sits T hours after the previous seizure.
    def off_for_post(t_h):
        return isi - width - t_h * 3600.0
    offs = np.array([off_for_post(t) for t in (1.0, 1.9, 2.1, 3.0)])
    # Default guard (2 h): the 1.0 h and 1.9 h windows are excluded; >=2 h kept.
    assert list(sa.valid_offset_mask(offs, isi)) == [False, False, True, True]
    # The old 1 h guard would have wrongly admitted the 1.9 h-post window.
    assert bool(sa.valid_offset_mask(offs, isi, postictal_guard=3600.0)[1]) is True


def test_signal_vs_null_auc():
    full = _make_full()
    res = sa.sliding_window_auc(full, ["signal", "null"], min_n=20)
    assert res["n_seizures_used"] == 3
    assert res["group"]["signal"] > 0.9          # cleanly separable
    assert res["group"]["null"] < 0.6            # ~chance
    assert res["group_top5"][0] == "signal"


def test_group_is_mean_of_per_seizure():
    full = _make_full()
    res = sa.sliding_window_auc(full, ["signal", "null"], min_n=20)
    per = [ps["mean_auc"]["signal"] for ps in res["per_seizure"].values()]
    assert np.isclose(res["group"]["signal"], float(np.mean(per)))


def test_best_per_seizure_from_group_top5():
    full = _make_full()
    res = sa.sliding_window_auc(full, ["signal", "null"], min_n=20)
    assert len(res["best_per_seizure"]) == 3
    for b in res["best_per_seizure"]:
        assert b["best_feature"] in res["group_top5"]
        assert b["best_feature"] == "signal"     # signal dominates
        assert b["best_auc"] > 0.9
        assert b["best_window"] in res["win_labels"]   # a concrete window, not a mean


def test_short_isi_seizure_dropped():
    # Two seizures 1000 s apart -> the 2nd has ISI < MIN_PREICTAL_SEC and is dropped.
    full = _make_full(n_seizures=1)
    close = _make_full(n_seizures=1, seed=1)
    close["seizure_idx"] = 1
    close["seizure_onset_epoch"] = 1_000_000.0 + 1000.0
    res = sa.sliding_window_auc(pd.concat([full, close], ignore_index=True),
                                ["signal", "null"], min_n=20)
    assert 1 not in res["per_seizure"]
    assert any("dropped" in n for n in res["notes"])
