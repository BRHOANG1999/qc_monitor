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


def _matrix(mu_of_tto, n_of_tto, feature="line_length", seed=0):
    """Synthetic event x metric matrix: one 'pre' seizure whose per-stimulus
    feature is drawn N(mu(tto), 1), with n(tto) stimuli per ~900 s bin."""
    rng = np.random.default_rng(seed)
    tto_list, val_list = [], []
    for center in np.arange(450.0, 21600.0, 900.0):       # bin centers 0..6 h
        k = int(n_of_tto(center))
        t = center + rng.uniform(-450, 450, k)
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
        return 10.0 if t > 7200 else 10.0 + 6.0 * (1.0 - t / 7200.0)
    full = _matrix(mu, lambda _t: 90)
    d = dan.frame_step_distances(full, 1, "line_length")
    h, step = d["hours_before"], d["step_w1"]
    far = step[h > 4.0]
    near = step[h < 1.5]
    assert np.nanmedian(near) > 3.0 * np.nanmedian(far)   # movement near onset
    assert np.nanmedian(far) < 0.4                        # ~still far out


def test_equal_n_guard_neutralizes_sample_size():
    # SAME distribution everywhere, but n drops near onset. With the equal-n
    # downsample, the step size must NOT blow up just because n differs.
    def n_of(t):
        return 25 if t < 5400 else 90                     # fewer stimuli near onset
    full = _matrix(lambda _t: 10.0, n_of)
    d = dan.frame_step_distances(full, 1, "line_length")
    step = d["step_w1"]
    assert np.all(np.nan_to_num(step) < 0.6)              # no sample-size inflation


def test_keys_and_ordering():
    full = _matrix(lambda _t: 10.0, lambda _t: 40)
    d = dan.frame_step_distances(full, 1, "line_length")
    assert set(d) >= {"hours_before", "step_w1", "disp_w1", "n", "labels"}
    h = d["hours_before"]
    assert h.size == d["step_w1"].size == d["disp_w1"].size
    assert np.all(np.diff(h) < 0)                          # far -> near (descending)
    assert d["disp_w1"][-1] > d["disp_w1"][0] - 1e-9       # disp defined near onset


def test_too_few_points_is_nan_not_crash():
    full = _matrix(lambda _t: 10.0, lambda _t: 3)          # < min_n per frame
    d = dan.frame_step_distances(full, 1, "line_length", min_n=8)
    assert np.all(np.isnan(d["step_w1"]))
