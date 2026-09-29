"""Stimulus-delivery (stim P2P, Z_ss) vs evoked-response correlation.

Pure-logic checks: filename datetime parse, channel-key normalization, and that
``correlate`` recovers the sign of a planted monotone relationship for BOTH
predictors.

Run with: pytest tests/test_evoked_stim_corr.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications import evoked_stim_corr as C          # noqa: E402


def test_file_dt_parses_and_rejects():
    dt = C._file_dt("chronicStim-10nC__x___2026_09_28__08_15_30_evoked.mat")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2026, 9, 28, 8, 15)
    assert C._file_dt("no_timestamp_here.mat") is None


def test_match_key_normalizes():
    assert C._match_key(["BCH111 SR", "BCH117SR"], "bch111_sr") == "BCH111 SR"
    assert C._match_key(["BCH117SR"], "BCH111SR") is None


def test_correlate_epochs_ra_zss_signs():
    import numpy as np
    n = 40
    epochs = {"ra": np.arange(n, dtype=float),
              "zss": np.arange(n, dtype=float)[::-1].copy(),
              "line_length": np.arange(n, dtype=float) * 3.0}
    c = C.correlate_epochs(epochs, ["line_length"])
    assert c["ra"]["line_length"][0] > 0.99      # Ra ↑ → line_length ↑
    assert c["zss"]["line_length"][0] < -0.99    # Z_ss ↓ as line_length ↑
    assert c["ra"]["line_length"][1] == n        # n reported


def test_correlate_constant_predictor_is_na():
    import numpy as np
    import math
    n = 40
    epochs = {"ra": np.full(n, 0.4) + np.arange(n) * 1e-6,   # near-constant Ra
              "zss": np.arange(n, dtype=float),
              "line_length": np.arange(n, dtype=float)}
    c = C.correlate_epochs(epochs, ["line_length"])
    assert math.isnan(c["ra"]["line_length"][0])   # <1% range -> no dose -> n/a
    assert c["zss"]["line_length"][0] > 0.99


def test_ampl_metrics_are_gain_divided_set():
    # amplitude metrics must be the ones divided by gain; ratios/latency stay invariant
    assert "line_length" in C._AMPL_METRICS and "rms_amplitude" in C._AMPL_METRICS
    assert "peak_latency_ms" in C._INV_METRICS
    assert "early_late_ratio" in C._INV_METRICS
    assert set(C._AMPL_METRICS).isdisjoint(C._INV_METRICS)
