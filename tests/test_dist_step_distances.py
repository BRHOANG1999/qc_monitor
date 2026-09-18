"""Frame-to-frame Wasserstein step size along the marching distributions.

Run with: pytest tests/test_dist_step_distances.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import dist_animation as dan          # noqa: E402


def _matrix(mu_of_tto, n_of_tto, *, tto_max=21600.0, feature="line_length",
            seed=0):
    """Synthetic event x metric matrix: one 'pre' seizure whose per-stimulus
    feature is N(mu(tto), 1), with n(tto) stimuli per ~300 s bin, out to
    *tto_max* seconds before onset (short tto_max => thin far baseline)."""
    rng = np.random.default_rng(seed)
    tto_list, val_list = [], []
    for center in np.arange(150.0, tto_max, 300.0):
        k = int(n_of_tto(center))
        if k <= 0:
            continue
        t = center + rng.uniform(-150, 150, k)
        tto_list.append(t)
        val_list.append(rng.normal(mu_of_tto(center), 1.0, k))
    tto = np.concatenate(tto_list)
    val = np.concatenate(val_list)
    return pd.DataFrame({"phase": ["pre"] * tto.size,
                         "time_to_onset_sec": tto,
                         "seizure_idx": np.ones(tto.size, dtype=int),
                         feature: val})


def test_step_rises_approaching_onset():
    # Stationary far from onset (tto > 2 h), a steep ramp near onset.
    def mu(t):
        return 10.0 if t > 7200 else 10.0 + 12.0 * (1.0 - t / 7200.0)
    full = _matrix(mu, lambda _t: 60)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert not d["insufficient"]
    h, step = d["hours_before"], d["step_w1"]
    far = step[h > 4.0]
    near = step[h < 1.0]
    assert np.nanmedian(near) > 2.5 * np.nanmedian(far)
    assert np.nanmedian(far) < 0.4


def test_normalized_to_baseline_jitter():
    # Stationary recording -> far-frame steps normalize to ~1.0 by construction.
    full = _matrix(lambda _t: 10.0, lambda _t: 60)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert not d["insufficient"]
    assert np.isfinite(d["baseline_step"]) and d["baseline_step"] > 0
    h, sn = d["hours_before"], d["step_w1_norm"]
    assert abs(np.nanmedian(sn[h > 4.0]) - 1.0) < 0.35   # far ~ 1x baseline


def test_equal_n_guard_neutralizes_sample_size():
    # SAME distribution, but n drops near onset -> the equal-n downsample keeps
    # the step from blowing up on sample size alone.
    def n_of(t):
        return 20 if t < 5400 else 70
    full = _matrix(lambda _t: 10.0, n_of)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert np.all(np.nan_to_num(d["step_w1"]) < 0.6)


def test_insufficient_baseline_is_excluded():
    # Data only within the last ~1.5 h -> the far-from-onset frames are empty,
    # so no baseline can be built and the seizure is flagged for exclusion.
    full = _matrix(lambda _t: 10.0, lambda _t: 60, tto_max=5400.0)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert d["insufficient"] is True
    assert "step_w1_norm" not in d
    assert d.get("reason")


def test_keys_and_ordering():
    full = _matrix(lambda _t: 10.0, lambda _t: 40)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert set(d) >= {"hours_before", "step_w1", "disp_w1", "n", "labels",
                      "step_w1_norm", "disp_w1_norm", "baseline_step"}
    h = d["hours_before"]
    assert h.size == d["step_w1"].size == d["step_w1_norm"].size
    assert np.all(np.diff(h) < 0)                          # far -> near (descending)


def test_straightening_drift_is_near_one():
    # Monotone drift: each frame shifts the same way -> path ≈ net -> ~1.
    def mu(t):
        return 10.0 + 10.0 * (1.0 - t / 21600.0)          # ramps 10 -> 20 to onset
    full = _matrix(mu, lambda _t: 60)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert np.isfinite(d["straightening"])
    assert d["straightening"] < 1.6                        # walked straight there


def test_straightening_wander_is_large():
    # Oscillates and returns near start: big path, tiny net -> >> 1.
    def mu(t):
        return 10.0 + 4.0 * np.sin(t / 900.0)
    full = _matrix(mu, lambda _t: 60)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0)
    assert np.isfinite(d["straightening"])
    assert d["straightening"] > 3.0                        # wandered, didn't drift


def test_non_overlapping_default_step():
    # step defaults to width -> non-overlapping frames -> ~lookback/width frames.
    full = _matrix(lambda _t: 10.0, lambda _t: 40)
    d = dan.frame_step_distances(full, 1, "line_length", width=600.0,
                                 lookback=21600.0)
    assert 30 <= d["n_frames"] <= 40                       # ~36 non-overlapping
