"""Training-tab grading core.

Run with: pytest tests/test_training_grading.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.training import (  # noqa: E402
    grade_attempt, match_events, rolling_ready)


def _ev(eo=None, type_="LVF", racine=3, **extra):
    e = {"type": type_, "EO_sec": eo, "LAS_sec": None, "BO_sec": None,
         "PID_sec": None, "BB_sec": None, "racine": racine}
    e.update(extra)
    return e


# ---- Stage 1: detection -------------------------------------------- #

def test_stage1_detection_match_and_miss():
    val = [_ev(eo=12.0)]
    assert grade_attempt(1, val, {"events_present": True})["score"] == 1.0
    assert grade_attempt(1, val, {"events_present": False})["score"] == 0.0
    # No validated events -> correct call is "no".
    assert grade_attempt(1, [], {"events_present": False})["score"] == 1.0
    assert grade_attempt(1, [], {"events_present": True})["score"] == 0.0


# ---- Stage 2: type + racine ---------------------------------------- #

def test_stage2_perfect_match():
    val = [_ev(eo=10.0, type_="LVF", racine=3)]
    ans = {"events_present": True,
           "events": [_ev(eo=11.0, type_="LVF", racine=3)]}  # within tol
    r = grade_attempt(2, val, ans, onset_tol_s=10, racine_tol=1)
    assert r["score"] == 1.0
    assert r["breakdown"]["matched"] == 1


def test_stage2_racine_tolerance():
    val = [_ev(eo=10.0, type_="LVF", racine=3)]
    # Racine off by 1 -> agrees at tol=1, disagrees at tol=0.
    ans = {"events_present": True,
           "events": [_ev(eo=10.0, type_="LVF", racine=4)]}
    assert grade_attempt(2, val, ans, racine_tol=1)["score"] == 1.0
    # tol=0: only the type component (1/2) is right.
    assert grade_attempt(2, val, ans, racine_tol=0)["score"] == 0.5


def test_stage2_onset_tolerance_controls_match():
    val = [_ev(eo=10.0)]
    ans = {"events_present": True, "events": [_ev(eo=25.0)]}  # 15s away
    # Outside tol -> a miss AND a false positive -> 0/(1+1) = 0.
    assert grade_attempt(2, val, ans, onset_tol_s=10)["score"] == 0.0
    # Widen tol -> matched.
    assert grade_attempt(2, val, ans, onset_tol_s=20)["score"] == 1.0


def test_stage2_false_positive_penalised():
    val = [_ev(eo=10.0, type_="LVF", racine=3)]
    ans = {"events_present": True,
           "events": [_ev(eo=10.0, type_="LVF", racine=3),
                      _ev(eo=200.0, type_="HYP", racine=5)]}  # extra
    r = grade_attempt(2, val, ans, onset_tol_s=10)
    # credit 1 / (1 validated + 1 false_pos) = 0.5
    assert r["score"] == 0.5
    assert r["breakdown"]["false_pos"] == 1


def test_stage2_missed_event_penalised():
    val = [_ev(eo=10.0), _ev(eo=100.0)]
    ans = {"events_present": True, "events": [_ev(eo=10.0)]}  # missed 2nd
    r = grade_attempt(2, val, ans, onset_tol_s=10)
    assert r["score"] == 0.5  # 1 credit / 2 validated
    assert r["breakdown"]["missed"] == 1


def test_empty_vs_empty_is_perfect():
    assert grade_attempt(2, [], {"events_present": False,
                                  "events": []})["score"] == 1.0
    assert grade_attempt(3, [], {"events_present": False,
                                  "events": []})["score"] == 1.0


# ---- Stage 3: full landmarks --------------------------------------- #

def test_stage3_landmarks_scored():
    val = [_ev(eo=10.0, type_="LVF", racine=3,
               LAS_sec=11.0, BO_sec=12.0, PID_sec=20.0, BB_sec=30.0)]
    # Perfect.
    perfect = {"events_present": True,
               "events": [_ev(eo=10.0, type_="LVF", racine=3,
                              LAS_sec=11.0, BO_sec=12.0,
                              PID_sec=20.0, BB_sec=30.0)]}
    assert grade_attempt(3, val, perfect, onset_tol_s=2)["score"] == 1.0
    # One landmark (BB) way off -> 6/7 components right.
    off = {"events_present": True,
           "events": [_ev(eo=10.0, type_="LVF", racine=3,
                          LAS_sec=11.0, BO_sec=12.0,
                          PID_sec=20.0, BB_sec=999.0)]}
    r = grade_attempt(3, val, off, onset_tol_s=2)
    assert abs(r["score"] - 6 / 7) < 1e-9


def test_stage3_skips_absent_validated_landmarks():
    # HYP validated event with no LAS -> LAS not scored (5 comps total:
    # type, racine, EO, BO, PID, BB... LAS absent). Student also no LAS.
    val = [_ev(eo=10.0, type_="HYP", racine=4,
               BO_sec=12.0, PID_sec=20.0, BB_sec=30.0)]
    ans = {"events_present": True,
           "events": [_ev(eo=10.0, type_="HYP", racine=4,
                          BO_sec=12.0, PID_sec=20.0, BB_sec=30.0)]}
    assert grade_attempt(3, val, ans, onset_tol_s=2)["score"] == 1.0


# ---- matching + rolling_ready -------------------------------------- #

def test_match_events_greedy_nearest():
    val = [_ev(eo=10.0), _ev(eo=50.0)]
    stu = [_ev(eo=49.0), _ev(eo=11.0)]
    pairs, used = match_events(val, stu, onset_tol_s=5)
    assert sorted(pairs) == [(0, 1), (1, 0)]
    assert used == {0, 1}


def test_rolling_ready():
    assert rolling_ready([1.0] * 9, 85, 10) is False     # too few
    assert rolling_ready([1.0] * 10, 85, 10) is True
    assert rolling_ready([0.8] * 10, 85, 10) is False    # 80% < 85%
    # Only the last `window` count -> early misses drop off.
    assert rolling_ready([0.0] * 5 + [1.0] * 10, 85, 10) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
