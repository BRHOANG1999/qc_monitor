"""Windowed evoked package: the analysis WINDOW re-measures features (option 2),
so a wider window yields a larger line-length / different decile. Covers the pure
helpers and the recompute-changes-magnitude contract without touching the network.

Run with: pytest tests/test_evoked_windowed.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import evoked_windowed as ew          # noqa: E402
from src.utils import evoked_features as _ef                  # noqa: E402


def test_win_tag_and_cfg():
    assert ew._win_tag((2.0, 50.0)) == "2-50ms"
    assert ew._win_tag((2, 1000)) == "2-1000ms"
    cfg = ew._win_cfg((2.0, 500.0))
    assert cfg.window_start_ms == 2.0 and cfg.window_end_ms == 500.0
    # crop-only: no filtering enabled
    assert not cfg.bandpass and not cfg.notch and not cfg.smoothing


def test_window_recomputes_magnitude():
    """A wider window integrates more of the waveform, so line_length grows and
    the two windows disagree on magnitude -- the whole point of option 2."""
    fs = 20000.0
    time_ms = np.arange(-20.0, 200.0, 1000.0 / fs)     # -20..200 ms at 20 kHz
    rng = np.random.default_rng(1)
    # a decaying oscillation after t=0 so later ms still carry wiggles
    n = time_ms.size
    sig = np.zeros(n)
    post = time_ms > 0
    sig[post] = np.exp(-time_ms[post] / 60.0) * np.sin(time_ms[post] / 3.0)
    traces = sig[None, :] + rng.normal(0, 0.01, (3, n))
    avg = _ef.trial_moving_average(traces)
    ll_50 = _ef.compute_all(avg, time_ms, fs, cfg=ew._win_cfg((2, 50)),
                            include_wavelet=False)["line_length"]
    ll_200 = _ef.compute_all(avg, time_ms, fs, cfg=ew._win_cfg((2, 100)),
                             include_wavelet=False)["line_length"]
    assert np.all(np.isfinite(ll_50)) and np.all(np.isfinite(ll_200))
    assert np.mean(ll_200) > np.mean(ll_50), "wider window -> longer path length"


def test_win_amp_extent_and_finalize():
    fs = 20000.0
    time_ms = np.arange(-20.0, 200.0, 1000.0 / fs)
    traces = np.tile(np.sin(time_ms / 5.0), (4, 1))
    lo, hi = ew._win_amp_extent(traces, time_ms, ew._win_cfg((2, 100)))
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
    yl = ew._finalize_ylim({"2-100ms": [(lo, hi), (lo, hi)]})
    assert yl["2-100ms"][0] < yl["2-100ms"][1]


def test_finalize_values_sorts_by_time():
    secs = {"2-50ms": {"week": [3.0, 1.0, 2.0]}}
    vals = {"2-50ms": {"week": {"line_length": [30.0, 10.0, 20.0]}}}
    out = ew._finalize_values(secs, vals, ["line_length"])
    np.testing.assert_array_equal(out["2-50ms"]["week"]["secs"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        out["2-50ms"]["week"]["metrics"]["line_length"], [10.0, 20.0, 30.0])


def test_default_windows_are_the_four():
    tags = [ew._win_tag(w) for w in ew.WINDOWS_MS]
    assert tags == ["2-50ms", "2-100ms", "2-500ms", "2-1000ms"]
