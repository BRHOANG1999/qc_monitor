"""peri_stim_artifact.seizure_windows: the three peri-ictal windows + the
no-overlap truncation against the neighbouring seizures.

Run: pytest tests/test_peri_stim_windows.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal.peri_stim_artifact import seizure_windows  # noqa: E402

_O = 10_000.0          # arbitrary onset epoch
W = 1800.0             # 30 min


def _rel(win):
    return {k: (round(lo - _O), round(hi - _O)) for k, (lo, hi) in win.items()}


def test_full_windows_when_isolated():
    assert _rel(seizure_windows(_O, None, None)) == {
        "pre": (-1800, 0), "at": (0, 1800), "post": (1800, 3600)}


def test_forward_truncated_at_next_seizure():
    # next seizure at +40 min -> post ends at +40 min, not +60
    assert _rel(seizure_windows(_O, None, _O + 2400)) == {
        "pre": (-1800, 0), "at": (0, 1800), "post": (1800, 2400)}


def test_post_dropped_when_next_is_within_30min():
    # next at +20 min -> 'at' truncated to +20, 'post' collapses (omitted)
    w = seizure_windows(_O, None, _O + 1200)
    assert _rel(w) == {"pre": (-1800, 0), "at": (0, 1200)}
    assert "post" not in w


def test_pre_clamped_at_previous_seizure():
    # previous seizure at -10 min -> pre starts at -10, not -30
    assert _rel(seizure_windows(_O, _O - 600, None))["pre"] == (-600, 0)


def test_at_and_post_dropped_when_next_at_onset():
    w = seizure_windows(_O, None, _O)     # degenerate: next == onset
    assert "at" not in w and "post" not in w and "pre" in w


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
