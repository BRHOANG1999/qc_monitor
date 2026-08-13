"""null_onsets: deep-interictal fake onsets for the nonstationarity control.

Run with: pytest tests/test_null_onsets.py -q
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import null_onsets as no  # noqa: E402

BAND_HI = 21600.0        # 6 h
BUFFER = BAND_HI + 3600  # 7 h
PRE = 1800.0
BAND_LO = 3600.0
MIN_COV = 20


def _day_start(y, m, d):
    return datetime(y, m, d).timestamp()


def _synthetic(seizure_days=(1, 3, 5), stim_step=10.0):
    """3 seizure-days (Jan 1/3/5 2026), one real seizure at noon each; dense
    stimulus coverage every *stim_step* s across the whole span."""
    real = np.array([_day_start(2026, 1, d) + 12 * 3600 for d in seizure_days])
    day_counts = {date(2026, 1, d): 1 for d in seizure_days}
    lo = _day_start(2026, 1, seizure_days[0]) - 2 * 86400
    hi = _day_start(2026, 1, seizure_days[-1]) + 2 * 86400
    cov = np.arange(lo, hi, stim_step)
    return real, day_counts, cov


def _draw(real, day_counts, cov, *, seed=0, n=None, spacing=BAND_HI):
    return no.draw_null_onsets(
        None, "BCHxxx", "", seed=seed, n=(n if n is not None else real.size),
        buffer_sec=BUFFER, window_sec=BAND_HI, preictal_max_sec=PRE,
        band_lo_sec=BAND_LO, band_hi_sec=BAND_HI, min_coverage=MIN_COV,
        min_spacing_sec=spacing, _real=real, _day_counts=day_counts, _coverage=cov)


def test_places_matched_count():
    real, dc, cov = _synthetic()
    dr = _draw(real, dc, cov)
    assert dr.n_requested == 3
    assert dr.n_placed == 3, (dr.n_placed, dr.reason)
    assert dr.reason is None
    assert dr.onsets.size == 3
    assert np.all(np.diff(dr.onsets) > 0)  # sorted


def test_buffer_clearance_both_sides():
    real, dc, cov = _synthetic()
    dr = _draw(real, dc, cov)
    for a in dr.onsets:
        nearest = np.min(np.abs(real - a))
        assert nearest >= BUFFER, (a, nearest, BUFFER)


def test_stim_coverage_in_preictal_and_band():
    real, dc, cov = _synthetic()
    dr = _draw(real, dc, cov)
    cov = np.sort(cov)
    for a in dr.onsets:
        pre = np.searchsorted(cov, a, "right") - np.searchsorted(cov, a - PRE, "right")
        band = (np.searchsorted(cov, a - BAND_LO, "right")
                - np.searchsorted(cov, a - BAND_HI, "right"))
        assert pre >= MIN_COV and band >= MIN_COV, (pre, band)


def test_onsets_on_seizure_days():
    real, dc, cov = _synthetic()
    dr = _draw(real, dc, cov)
    days = set(dc)
    for a in dr.onsets:
        assert datetime.fromtimestamp(a).date() in days


def test_spacing_enforced():
    real, dc, cov = _synthetic()
    dr = _draw(real, dc, cov, spacing=BAND_HI)
    if dr.onsets.size >= 2:
        assert np.all(np.diff(np.sort(dr.onsets)) >= BAND_HI)


def test_determinism_and_seed_variation():
    real, dc, cov = _synthetic(seizure_days=(1, 3, 5, 7, 9), stim_step=10.0)
    a0 = _draw(real, dc, cov, seed=0, n=5).onsets
    a0b = _draw(real, dc, cov, seed=0, n=5).onsets
    a1 = _draw(real, dc, cov, seed=1, n=5).onsets
    assert np.array_equal(a0, a0b)                 # same seed -> identical
    # different seed generally differs (allow the rare tie by checking the set)
    assert not np.array_equal(a0, a1) or a0.size <= 1


def test_scarce_coverage_returns_shortfall():
    real, dc, _ = _synthetic()
    # Coverage only around ONE deep-interictal pocket -> can't match n=3.
    a_ok = _day_start(2026, 1, 1) + 2 * 3600            # 02:00 on a seizure-day
    cov = np.arange(a_ok - BAND_HI, a_ok, 10.0)         # covers just this anchor
    dr = _draw(real, dc, cov, n=3)
    assert dr.n_placed < 3
    assert dr.reason and "anchors" in dr.reason


def test_no_seizures_graceful():
    dr = no.draw_null_onsets(None, "BCHxxx", "", seed=0, n=0,
                             _real=np.array([]), _day_counts={}, _coverage=np.array([]))
    assert dr.n_placed == 0 and dr.reason


def test_draw_set_k_draws():
    real, dc, cov = _synthetic(seizure_days=(1, 3, 5, 7, 9))
    draws = no.draw_null_set(None, "BCHxxx", "", seeds=[0, 1, 2], n=5,
                             buffer_sec=BUFFER, window_sec=BAND_HI,
                             preictal_max_sec=PRE, band_lo_sec=BAND_LO,
                             band_hi_sec=BAND_HI, min_coverage=MIN_COV,
                             min_spacing_sec=BAND_HI,
                             _real=real, _day_counts=dc, _coverage=cov)
    assert len(draws) == 3
    assert all(d.seed == s for d, s in zip(draws, [0, 1, 2]))


def test_draw_set_reads_coverage_once(monkeypatch):
    """K draws must share ONE full-corpus coverage read, not one per draw --
    the fix for 'drawing null onsets took forever'."""
    real, dc, cov = _synthetic(seizure_days=(1, 3, 5, 7, 9))
    calls = {"cov": 0, "days": 0, "seiz": 0}

    def fake_cov(animal, evoked_dir):
        calls["cov"] += 1
        return cov

    def fake_days(store, animal):
        calls["days"] += 1
        return dc

    def fake_seiz(store, animal):
        calls["seiz"] += 1
        return [type("S", (), {"onset_epoch": float(x)})() for x in real]

    monkeypatch.setattr(no, "_coverage_epochs", fake_cov)
    monkeypatch.setattr(no, "_seizure_day_counts", fake_days)
    monkeypatch.setattr(no, "included_seizures", fake_seiz)
    draws = no.draw_null_set(None, "BCHxxx", "", seeds=list(range(8)), n=5,
                             buffer_sec=BUFFER, window_sec=BAND_HI,
                             preictal_max_sec=PRE, band_lo_sec=BAND_LO,
                             band_hi_sec=BAND_HI, min_coverage=MIN_COV,
                             min_spacing_sec=BAND_HI)
    assert len(draws) == 8
    assert calls == {"cov": 1, "days": 1, "seiz": 1}, calls


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
