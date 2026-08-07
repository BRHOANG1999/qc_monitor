"""call_with_deadline: a hung call must not block the caller past the timeout.

Run with: pytest tests/test_deadline.py -q
"""

from __future__ import annotations

import os
import sys
import time

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.deadline import call_with_deadline  # noqa: E402


def test_fast_call_returns_value():
    assert call_with_deadline(lambda: 21 * 2, 5.0) == 42


def test_slow_call_returns_default_within_timeout():
    t0 = time.perf_counter()
    out = call_with_deadline(lambda: time.sleep(30), 0.3, default="TIMED_OUT")
    elapsed = time.perf_counter() - t0
    assert out == "TIMED_OUT"
    # Caller unblocked at ~the deadline, NOT after the 30s sleep.
    assert elapsed < 3.0, f"caller blocked {elapsed:.1f}s past the deadline"


def test_default_is_none_by_default():
    assert call_with_deadline(lambda: time.sleep(30), 0.2) is None


def test_exception_from_fn_is_reraised_when_it_finishes():
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        call_with_deadline(boom, 5.0)


def test_rejects_bad_args():
    with pytest.raises(AssertionError):
        call_with_deadline("not callable", 1.0)      # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        call_with_deadline(lambda: 1, 0)
    with pytest.raises(AssertionError):
        call_with_deadline(lambda: 1, -1.0)
