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


def test_correlate_recovers_both_predictor_signs():
    rows = []
    for i in range(30):                      # line_length tracks p2p up, zss down
        rows.append({"dt": datetime(2026, 9, 1), "p2p": float(i), "zss": float(-i),
                     "line_length": float(i) * 2.0, "peak_latency_ms": float(-i)})
    corr = C.correlate(rows, ["line_length", "peak_latency_ms"])
    assert corr["line_length"]["p2p"][0] > 0.99      # p2p ↑ → line_length ↑
    assert corr["line_length"]["zss"][0] < -0.99     # zss ↓ as line_length ↑
    assert corr["peak_latency_ms"]["p2p"][0] < -0.99
    assert corr["line_length"]["p2p"][1] == 30       # n reported


def test_correlate_constant_predictor_is_na():
    import math
    rows = [{"dt": datetime(2026, 9, 1), "p2p": 1.0, "zss": float(i),
             "line_length": float(i)} for i in range(20)]
    corr = C.correlate(rows, ["line_length"])
    assert math.isnan(corr["line_length"]["p2p"][0])   # constant P2P -> no dose-response
    assert corr["line_length"]["zss"][0] > 0.99


def test_ampl_metrics_are_gain_divided_set():
    # amplitude metrics must be the ones divided by gain; ratios/latency stay invariant
    assert "line_length" in C._AMPL_METRICS and "rms_amplitude" in C._AMPL_METRICS
    assert "peak_latency_ms" in C._INV_METRICS
    assert "early_late_ratio" in C._INV_METRICS
    assert set(C._AMPL_METRICS).isdisjoint(C._INV_METRICS)
