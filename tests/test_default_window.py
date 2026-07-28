"""The default 'evoked' sidecar windows every feature to [1, 200] ms (v5).

Before v5 only the Chang morphology + area columns were windowed; the ~23
cheap/spectral/wavelet columns ran on the full +/-500 ms epoch, so "full trace
(fast)" and "custom window 1-200" silently disagreed and the peak/latency
features locked onto the t=0 stim artifact. The default now crops to [1, 200]
before any feature is computed.

Run: pytest tests/test_default_window.py -q
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import evoked_features as ef        # noqa: E402
from src.utils import evoked_output as EO          # noqa: E402


def _artifact_plus_bump():
    """A +/-500 ms epoch: a huge artifact spike at t=0 and a small real evoked
    bump near +50 ms -- the exact shape that fooled the full-epoch features."""
    t = np.linspace(-500.0, 500.0, 20001)
    y = np.zeros_like(t)
    y[np.argmin(np.abs(t))] = 100.0                 # stim artifact at t=0
    y[(t >= 48.0) & (t <= 52.0)] = 5.0              # evoked response ~50 ms
    fs = 1000.0 / float(np.median(np.diff(t)))
    return y[None, :], t, fs


# ------------------------------------------------------- constants --- #

def test_default_cfg_is_1_to_200():
    assert EO.DEFAULT_EVOKED_CFG.window_start_ms == 1.0
    assert EO.DEFAULT_EVOKED_CFG.window_end_ms == 200.0


def test_version_is_v5_and_older_rejected():
    assert EO._FEATURE_SIDECAR_VERSION == "5"
    assert EO._COMPATIBLE_SIDECAR_VERSIONS == {"5"}     # v4 full-epoch rejected
    assert EO._WAVELET_STABLE_VERSIONS == {"5"}


# --------------------------------------------------- windowing bites --- #

def test_full_epoch_peak_is_the_artifact():
    y, t, fs = _artifact_plus_bump()
    full = ef.compute_all(y, t, fs, cfg=None)          # passthrough / full trace
    assert abs(full["peak_latency_ms"][0]) < 2.0       # locks onto t=0 artifact


def test_default_window_finds_the_real_response():
    y, t, fs = _artifact_plus_bump()
    win = ef.compute_all(y, t, fs, cfg=EO.DEFAULT_EVOKED_CFG)
    pl = win["peak_latency_ms"][0]
    assert 1.0 <= pl <= 200.0                           # in-window
    assert pl > 10.0                                    # the bump, not the artifact
    assert abs(pl - 50.0) < 5.0


def test_default_and_explicit_1_200_agree():
    """cfg=None must resolve to DEFAULT_EVOKED_CFG in the sidecar builder, so the
    fast default equals an explicit [1,200] custom window."""
    y, t, fs = _artifact_plus_bump()
    a = ef.compute_all(y, t, fs, cfg=EO.DEFAULT_EVOKED_CFG)
    b = ef.compute_all(y, t, fs,
                       cfg=ef.FeatureConfig(window_start_ms=1.0,
                                            window_end_ms=200.0))
    assert a["peak_latency_ms"][0] == b["peak_latency_ms"][0]
    assert a["rms_amplitude"][0] == b["rms_amplitude"][0]
