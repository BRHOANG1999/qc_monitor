"""Full-density trace rasterizer (memory-safe all-traces overlay).

Every trace is rasterized into a fixed pixel buffer; a lone outlier trace must
stay visible above a dense smear (the whole point), and a single-trace pixel must
read at the per-trace opacity a0. Memory is O(W*H), independent of trace count.

Run with: pytest tests/test_trace_density.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.notifications.trace_density import LineDensity, robust_ylim  # noqa: E402


def _smear_plus_outlier():
    t = np.linspace(2, 200, 400)
    ld = LineDensity((2, 200), (-2, 2), width=300, height=200)
    rng = np.random.default_rng(0)
    for _ in range(500):                       # the dense smear near y=0
        ld.add(t, 0.2 * np.sin(t / 10) + rng.normal(0, 0.05, t.size))
    ld.add(t, 1.6 * np.ones_like(t))           # one lone trace up at y=1.6
    return ld


def test_outlier_trace_survives_the_smear():
    ld = _smear_plus_outlier()
    c = ld.counts()
    assert c.shape == (200, 300)
    assert ld.n == 501
    # the smear (near y=0 -> mid rows) is dense; the outlier row band gets ink too
    smear = c[95:105, :].sum()
    outlier = c[15:25, :].sum()                # y=1.6 -> row ~20
    assert smear > outlier > 0, "lone outlier must leave visible ink"


def test_single_trace_pixel_reads_at_a0():
    ld = _smear_plus_outlier()
    c = ld.counts()
    rgba = ld.rgba((0.4, 0.8, 1.0), a0=0.3)
    assert rgba.shape == (200, 300, 4)
    one = c == 1
    assert one.any()
    # alpha where exactly one trace passed == a0
    assert np.allclose(rgba[..., 3][one], 0.3)
    # dense pixels approach opaque
    assert rgba[..., 3].max() > 0.99


def test_counts_never_exceed_n_and_memory_is_fixed():
    small = LineDensity((0, 10), (-1, 1), width=64, height=48)
    big = LineDensity((0, 10), (-1, 1), width=64, height=48)
    t = np.linspace(0, 10, 50)
    for _ in range(10):
        small.add(t, np.zeros_like(t))
    for _ in range(100000):
        big.add(t, np.zeros_like(t))
    # buffer footprint is identical regardless of trace count
    assert small._delta.shape == big._delta.shape
    assert big.counts().max() <= big.n == 100000


def test_nan_gap_breaks_the_stroke():
    ld = LineDensity((0, 10), (-1, 1), width=40, height=40)
    y = np.zeros(50)
    t = np.linspace(0, 10, 50)
    y[20:30] = np.nan                          # blanked artifact window
    ld.add(t, y)
    c = ld.counts()
    # the blanked columns (~middle) carry no ink
    assert c[:, 18:22].sum() == 0 or c[:, 20:28].sum() < c[:, :5].sum() + 1


def test_robust_ylim():
    a = np.array([[-0.3, 0.3]] * 500 + [[1.6, 1.6]])
    y0, y1 = robust_ylim(a)
    assert y0 < 0 < y1
    assert y1 < 1.6                            # the 1-in-501 outlier doesn't set the axis
    assert robust_ylim(np.empty((0, 2))) == (-1.0, 1.0)
