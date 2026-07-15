"""Peri-ictal selection summary: the lasso/box drill-down aggregation that tells
a real cluster from a confound (time-to-onset + hour-of-day distributions,
breakdown by source seizure and recording).

Run with: pytest tests/test_periictal_selection.py -q
"""

from __future__ import annotations

import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal.selection import summarize_selection  # noqa: E402


def _frame():
    # 3 seizures (onsets ascending), 2 recordings, varied lead bins/hours.
    return pd.DataFrame({
        "time_to_onset_sec": [60, 120, 3600, 7200, 100, 200],
        "hour_of_day": [19.0, 19.5, 3.0, 3.5, 11.0, 11.2],
        "seizure_idx": [2, 2, 0, 0, 1, 1],
        "seizure_onset_epoch": [300.0, 300.0, 100.0, 100.0, 200.0, 200.0],
        "rec": ["2026-07-13T18:00:00"] * 3 + ["2026-07-14T10:00:00"] * 3,
        "lead_bin": [0, 0, 5, 5, 1, 1],
    })


def test_empty_selection():
    s = summarize_selection(_frame(), [])
    assert s["n"] == 0 and s["n_total"] == 6
    assert s["by_seizure"] == [] and s["lead_frac"] is None


def test_selection_distributions_and_breakdowns():
    df = _frame()
    s = summarize_selection(df, [0, 1, 2, 3])       # seizures 2 (x2) and 0 (x2)
    assert s["n"] == 4 and s["n_total"] == 6
    assert len(s["tto_hours"]) == 4 and len(s["hour_of_day"]) == 4
    assert s["tto_hours"][0] == 60 / 3600.0
    # by_seizure ordered by ONSET time (seizure 0 onset 100 before seizure 2
    # onset 300); each contributes 2 selected points. (Labels are onset
    # datetimes -- timezone-dependent strings, so assert structure not text.)
    assert len(s["by_seizure"]) == 2
    assert [c for _, c in s["by_seizure"]] == [2, 2]
    assert sum(c for _, c in s["by_seizure"]) == 4
    # both recordings represented.
    assert sum(c for _, c in s["by_recording"]) == 4


def test_lead_frac_is_dominant_bin_share():
    df = _frame()
    # rows 0,1 (bin 0) + row 2 (bin 5): dominant bin 0 = 2/3.
    s = summarize_selection(df, [0, 1, 2])
    assert s["lead_frac"] == round(2 / 3, 3)


def test_out_of_range_indices_are_ignored():
    s = summarize_selection(_frame(), [0, 999, -1, 2])
    assert s["n"] == 2                              # only the two valid indices


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
