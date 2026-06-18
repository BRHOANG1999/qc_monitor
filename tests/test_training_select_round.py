"""Balanced training-round selection (pure, deterministic with Random(0)).

Run with: pytest tests/test_training_select_round.py -q
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.training import select_round  # noqa: E402


def _cand(fid, *, seiz=False, typ=None, racine=None, animals=None, seen=None):
    return {"file_id": fid, "has_seizure": seiz, "rep_type": typ,
            "rep_racine": racine, "animals": animals or [], "last_seen": seen}


def _rng():
    return random.Random(0)


def test_stage1_75_25_split():
    pool = ([_cand(i, seiz=False) for i in range(1, 11)]
            + [_cand(i, seiz=True) for i in range(11, 21)])
    by = {c["file_id"]: c for c in pool}
    out = select_round(pool, 1, 10, {}, _rng())
    assert len(out) == 10 and len(set(out)) == 10
    n_seiz = sum(1 for f in out if by[f]["has_seizure"])
    assert n_seiz == 2          # round(0.25*10) -> 2 seizure, 8 no-seizure


def test_stage1_degrades_when_bucket_short():
    pool = [_cand(1, seiz=False)] + [_cand(i, seiz=True) for i in range(2, 12)]
    by = {c["file_id"]: c for c in pool}
    out = select_round(pool, 1, 10, {}, _rng())
    assert len(out) == 10 and len(set(out)) == 10
    assert 1 in out             # the lone no-seizure file is used
    assert sum(1 for f in out if by[f]["has_seizure"]) == 9


def test_stage2_balances_racine_and_type():
    pool = []
    fid = 1
    for racine in range(1, 9):
        for typ in ("LVF", "HYP"):
            for _ in range(2):          # 2 files per (racine, type) = 32
                pool.append(_cand(fid, typ=typ, racine=racine, seiz=True))
                fid += 1
    by = {c["file_id"]: c for c in pool}
    out = select_round(pool, 2, 16, {}, _rng())
    assert len(out) == 16 and len(set(out)) == 16
    racines = Counter(by[f]["rep_racine"] for f in out)
    types = Counter(by[f]["rep_type"] for f in out)
    assert max(racines.values()) - min(racines.values()) <= 1
    assert abs(types["LVF"] - types["HYP"]) <= 1


def test_stage2_sparse_pool_no_crash():
    # only racines 1-4 present; still returns a full round.
    pool = [_cand(i, typ="LVF" if i % 2 else "HYP",
                  racine=(i % 4) + 1, seiz=True) for i in range(1, 13)]
    out = select_round(pool, 2, 8, {}, _rng())
    assert len(out) == 8 and len(set(out)) == 8


def test_stage3_spreads_across_animals():
    pool = []
    fid = 1
    for animal in ("BCH040", "BCH060", "BCH062"):
        for _ in range(5):
            pool.append(_cand(fid, animals=[animal], seiz=True))
            fid += 1
    by = {c["file_id"]: c for c in pool}
    out = select_round(pool, 3, 9, {}, _rng())
    counts = Counter(by[f]["animals"][0] for f in out)
    assert set(counts) == {"BCH040", "BCH060", "BCH062"}
    assert max(counts.values()) - min(counts.values()) <= 1   # 3 each


def test_stage3_single_animal_is_random():
    pool = [_cand(i, animals=["BCH040"], seiz=True) for i in range(1, 11)]
    out = select_round(pool, 3, 5, {}, _rng())
    assert len(out) == 5 and len(set(out)) == 5


def test_least_seen_preferred():
    pool = [_cand(1, seiz=False, seen=9), _cand(2, seiz=False, seen=0)]
    out = select_round(pool, 1, 1, {1: 9, 2: 0}, _rng())
    assert out == [2]           # unseen wins over heavily-seen


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
