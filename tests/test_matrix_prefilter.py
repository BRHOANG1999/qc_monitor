"""Near-seizure file prefilter for the peri-ictal matrix build.

The prefilter must be a PROVABLE SUPERSET: it may skip files whose every
stimulus is dropped anyway, but must never drop a file that could contribute a
kept row. These tests pin the gate geometry (window_sec + slack around each
onset) and the fail-open behaviour on unparseable names.
"""

from datetime import datetime, timedelta

import numpy as np

from src.periictal.matrix import _near_seizure_file_filter


def _fn(dt: datetime) -> str:
    # Evoked sidecar name parse_recording_dt understands:
    # ...YYYY_MM_DD__HH_MM_SS_evoked.mat
    return (f"chronicStim_BCH111_{dt.year}_{dt.month:02d}_{dt.day:02d}__"
            f"{dt.hour:02d}_{dt.minute:02d}_{dt.second:02d}_evoked.mat")


def test_prefilter_keeps_near_and_drops_far():
    onset = datetime(2026, 7, 20, 12, 0, 0)
    onsets = np.array([onset.timestamp()])
    keep = _near_seizure_file_filter(onsets, window_sec=6 * 3600, slack=7200)
    # Kept: at onset, within the 6 h pre-window, within the post window, and a
    # file starting up to slack (2 h) before the 6 h window (chunk that contains
    # the window edge).
    assert keep(_fn(onset))
    assert keep(_fn(onset - timedelta(hours=5)))
    assert keep(_fn(onset + timedelta(hours=5)))
    assert keep(_fn(onset - timedelta(hours=7, minutes=30)))   # inside 6h+2h slack
    # Dropped: far outside [onset - 8h, onset + 8h] -> every stimulus is dropped
    # anyway, so it can't change the matrix.
    assert not keep(_fn(onset - timedelta(hours=20)))
    assert not keep(_fn(onset + timedelta(hours=20)))
    assert not keep(_fn(onset + timedelta(days=3)))


def test_prefilter_unions_multiple_seizures():
    a = datetime(2026, 7, 20, 12, 0, 0)
    b = datetime(2026, 7, 25, 9, 0, 0)
    keep = _near_seizure_file_filter(
        np.array([a.timestamp(), b.timestamp()]), window_sec=6 * 3600, slack=7200)
    assert keep(_fn(a + timedelta(hours=2)))                   # near seizure a
    assert keep(_fn(b - timedelta(hours=2)))                   # near seizure b
    assert not keep(_fn(a + timedelta(days=2)))                # between, far from both


def test_prefilter_keeps_unparseable_names():
    keep = _near_seizure_file_filter(np.array([datetime(2026, 7, 20).timestamp()]),
                                     window_sec=6 * 3600, slack=7200)
    # Never drop on uncertainty -- a name we can't time-stamp is read.
    assert keep("no_date_here.mat")
    assert keep("")
