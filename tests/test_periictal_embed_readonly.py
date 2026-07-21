"""Embedding prep must tolerate a read-only metric frame.

Regression cover for the peri-ictal build failing with
``ValueError: assignment destination is read-only``.

When every selected metric column shares one float64 block, ``to_numpy()``
returns a VIEW of the frame's buffer rather than a copy. A frame restored from
the parquet matrix cache carries a read-only buffer, so ``_prepare``'s
median-fill wrote into read-only memory -- and the build failed ONLY on a cache
hit, which is why it looked intermittent and never reproduced from a cold run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.periictal.embed import _prepare  # noqa: E402

_COLS = ["m1", "m2"]


def _frame(values, *, read_only: bool):
    base = np.asarray(values, dtype=np.float64)
    if read_only:
        base.setflags(write=False)
    return pd.DataFrame(base, columns=_COLS)


def test_readonly_frame_with_nans_still_prepares():
    """The exact failure: a read-only buffer plus a NaN needing the median fill."""
    df = _frame([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]], read_only=True)
    X, cols = _prepare(df, _COLS)
    assert cols == _COLS
    assert np.all(np.isfinite(X))
    assert X[1, 0] == pytest.approx(3.0)      # median of {1.0, 5.0}


def test_writeable_frame_unchanged():
    df = _frame([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]], read_only=False)
    X, _ = _prepare(df, _COLS)
    assert X[1, 0] == pytest.approx(3.0)


def test_prepare_never_mutates_the_caller_frame():
    """The fill must not write through to a cached frame shared by other lenses."""
    df = _frame([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]], read_only=False)
    _prepare(df, _COLS)
    assert bool(df["m1"].isna().iloc[1]), "median fill leaked into the source frame"


def test_all_nan_columns_are_dropped():
    df = pd.DataFrame({"m1": [1.0, 2.0], "m2": [np.nan, np.nan]})
    X, cols = _prepare(df, _COLS)
    assert cols == ["m1"] and X.shape == (2, 1)


def test_no_usable_columns_asserts():
    df = pd.DataFrame({"m1": [np.nan, np.nan]})
    with pytest.raises(AssertionError):
        _prepare(df, ["m1"])
