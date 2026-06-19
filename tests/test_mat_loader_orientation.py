"""mat_loader orients sbuf to [samples x channels] regardless of how the
.mat stored it (v7 scipy = correct; v7.3 HDF5 = transposed).

Run with: pytest tests/test_mat_loader_orientation.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

scipy_io = pytest.importorskip("scipy.io")
from src.utils.mat_loader import load_mat  # noqa: E402


def _write_v7(path, sbuf, fs=20000.0):
    scipy_io.savemat(path, {"sbuf": sbuf, "fs": float(fs)})


def test_samples_major_unchanged(tmp_path):
    # Already [samples x channels] (samples >> channels) -> kept as-is.
    sbuf = np.random.randn(50000, 5).astype(np.float64)
    p = str(tmp_path / "ok.mat")
    _write_v7(p, sbuf)
    c = load_mat(p)
    assert c.num_samples == 50000 and c.num_channels == 5
    assert c.duration_sec == pytest.approx(2.5)


def test_channels_major_transposed(tmp_path):
    # [channels x samples] (the v7.3/HDF5 orientation) -> transposed so the
    # longer axis becomes samples. Without the fix this reads as 5 samples
    # x 50000 channels (a sub-millisecond degenerate trace).
    sbuf = np.random.randn(5, 50000).astype(np.float64)
    p = str(tmp_path / "transposed.mat")
    _write_v7(p, sbuf)
    c = load_mat(p)
    assert c.num_samples == 50000 and c.num_channels == 5
    assert c.duration_sec == pytest.approx(2.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
