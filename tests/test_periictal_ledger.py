"""CONSORT trial/seizure accounting ledger (plan item 0 / task #9).

Guards the reconciliation (scored - dropped = included), the per-seizure per-class
counts, and the both-windows flag -- the counts that resolve "UMAP shows 4
seizures but the paired test used 3".

Run: pytest tests/test_periictal_ledger.py -q
"""

import os
import sys
from collections import namedtuple

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.periictal import ledger as lg          # noqa: E402

Sz = namedtuple("Sz", "file_id eo_sec onset_epoch racine")


def _rows(seizure_idx, onset, session, n_pre, n_inter):
    out = []
    for _ in range(n_pre):
        out.append({"seizure_idx": seizure_idx, "class": "preictal",
                    "seizure_onset_epoch": onset, "session": session})
    for _ in range(n_inter):
        out.append({"seizure_idx": seizure_idx, "class": "interictal",
                    "seizure_onset_epoch": onset, "session": session})
    return out


def _labelled():
    rows = (_rows(0, 1000.0, "S0", 10, 10)
            + _rows(1, 2000.0, "S1", 8, 0)      # only preictal -> both_windows False
            + _rows(2, 3000.0, "S2", 5, 5))
    return pd.DataFrame(rows)


def _scored():
    return [Sz(1, 5.0, 1000.0, 2), Sz(2, 5.0, 2000.0, 3),
            Sz(3, 5.0, 3000.0, 1), Sz(4, 5.0, 4000.0, 4)]   # #4 is the dropped one


def test_reconciles_scored_minus_dropped_equals_included():
    led = lg.build_ledger(_scored(), {4}, lambda s: s.file_id, _labelled())
    assert led["n_scored"] == 4 and led["n_excluded"] == 1
    assert led["n_included"] == 3
    assert led["reconciles"] is True
    assert led["excluded"][0]["file_id"] == 4
    assert "bad-stim" in led["excluded"][0]["reason"]


def test_per_seizure_per_class_counts_and_both_windows():
    led = lg.build_ledger(_scored(), {4}, lambda s: s.file_id, _labelled())
    per = {p["seizure_idx"]: p for p in led["per_seizure"]}
    assert per[0]["n_pre"] == 10 and per[0]["n_inter"] == 10 and per[0]["both_windows"]
    assert per[1]["n_pre"] == 8 and per[1]["n_inter"] == 0
    assert per[1]["both_windows"] is False          # the class-asymmetry made visible
    assert per[2]["session"] == "S2"
    assert led["totals"] == {"n_pre": 23, "n_inter": 15,
                             "n_seizures_both_windows": 2}


def test_empty_matrix_is_safe():
    led = lg.build_ledger(_scored(), set(), lambda s: s.file_id,
                          pd.DataFrame())
    assert led["n_included"] == 4 and led["per_seizure"] == []
    assert led["reconciles"] is True
