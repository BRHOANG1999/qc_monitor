"""Peri-ictal passive variant: the artifact guard that widens with smoothing,
the window-bounds validation (no silent full-trace fallback), the config-signed
sidecar round-trip, and the exclusion of the post-stim-band features that are
meaningless on a pre-stim window.

Run with: pytest tests/test_periictal_passive.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import config as _cfg              # noqa: E402
from src.periictal import passive as pv               # noqa: E402
from src.utils.evoked_output import (feature_sidecar_path,  # noqa: E402
                                     read_feature_sidecar,
                                     write_feature_sidecar)


def test_guard_widens_with_smoothing():
    assert pv.guard_ms(1.0, smoothing=False) == 1.0
    # A 20 ms boxcar spreads a point ~10 ms each side -> guard 1 + 10 = 11.
    assert pv.guard_ms(1.0, smoothing=True, smooth_ms=20.0) == 11.0


def test_window_validation_rejects_reversed_and_out_of_range():
    with pytest.raises(AssertionError):
        pv._validate_window(-1.0, -200.0)          # reversed
    with pytest.raises(AssertionError):
        pv._validate_window(5.0, 5000.0)           # beyond the trace


def test_passive_and_evoked_windows():
    p = pv.passive_config(pre_ms=200.0, artifact_half_ms=1.0)
    assert p.window_start_ms == -200.0 and p.window_end_ms == -1.0
    e = pv.evoked_config(post_ms=200.0, artifact_half_ms=1.0)
    assert e.window_start_ms == 1.0 and e.window_end_ms == 200.0
    # Smoothing pushes the guard out on both sides.
    ps = pv.passive_config(pre_ms=200.0, artifact_half_ms=1.0,
                           smoothing=True, smooth_ms=10.0)
    assert ps.window_end_ms == -6.0                # -(1 + 10/2)


def test_config_sig_sensitive_to_window():
    a = pv.config_sig(pv.passive_config(pre_ms=200.0))
    b = pv.config_sig(pv.passive_config(pre_ms=150.0))
    assert a != b                                  # window change -> new sig
    assert pv.config_sig(pv.passive_config(pre_ms=200.0)) == a  # stable


def test_passive_metric_set_drops_poststim_band_features():
    m = _cfg.metrics_for_variant("passive")
    assert "line_length" in m
    for bad in ("early_area", "late_area", "early_late_ratio"):
        assert bad not in m                        # meaningless pre-stim
    assert len(m) == len(_cfg.CHEAP_METRICS) - 3
    assert len(_cfg.metrics_for_variant("evoked")) == len(_cfg.CHEAP_METRICS)


def test_window_config_pushes_bound_off_artifact():
    # A post-stim window starting at 0 is pushed out to +guard; a pre-stim
    # window ending at 0 is pushed to -guard -- the artifact is always excluded.
    w = pv.window_config(0.0, 50.0, artifact_half_ms=2.0)
    assert w.window_start_ms == 2.0 and w.window_end_ms == 50.0
    w2 = pv.window_config(-100.0, 0.0, artifact_half_ms=2.0)
    assert w2.window_start_ms == -100.0 and w2.window_end_ms == -2.0


def test_window_config_rejects_reversed():
    with pytest.raises(AssertionError):
        pv.window_config(100.0, 10.0)


def test_build_variant_refuses_shared_default_name():
    # 'evoked' is the shared toolkit-default sidecar; a windowed variant must
    # never overwrite it (chronic_evoked / evoked_figures read it).
    with pytest.raises(AssertionError):
        pv.build_variant_sidecar("x.mat", "BCH040", "evoked", pv.evoked_config())


def test_variant_sidecar_roundtrip_and_config_gate(tmp_path):
    mat = str(tmp_path / "sess__stimCopy_BCH040SR___2026_03_02__00_00_00_evoked.mat")
    with open(mat, "wb") as f:
        f.write(b"\x00")
    rows = [{"channel": "BCH040SR", "abs_dt": "2026-03-02T00:00:00",
             "line_length": 1.0}]
    sig = pv.config_sig(pv.passive_config())
    write_feature_sidecar(mat, "BCH040", rows, variant="passive", config_sig=sig)
    # The passive sidecar is a DIFFERENT file from the evoked one.
    assert feature_sidecar_path(mat, "BCH040", "passive").endswith(".passive.json")
    assert feature_sidecar_path(mat, "BCH040") != \
        feature_sidecar_path(mat, "BCH040", "passive")
    # Right variant + matching sig -> rows; wrong sig -> None; evoked variant -> None.
    assert read_feature_sidecar(mat, "BCH040", "passive", sig) == rows
    assert read_feature_sidecar(mat, "BCH040", "passive", "OTHER") is None
    assert read_feature_sidecar(mat, "BCH040") is None      # no evoked sidecar


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
