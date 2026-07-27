"""Per-seizure exclusion from the peri-ictal analysis.

A seizure scored during bad data collection can be marked out so it no longer
defines preictal windows / interictal boundaries in the embedding. Persistent
per animal; included_seizures is the single filter the matrix + its cache
signature use.

Run: pytest tests/test_seizure_exclusion.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                                # noqa: E402
from src.preictal.isi import included_seizures, scored_seizures  # noqa: E402

_ANIMAL = "BCH111"


def _seed(store, seizures):
    """seizures: list of (file_id, chunk_datetime, eo_sec, racine)."""
    with store.connection() as conn:
        for fid, cdt, eo, rac in seizures:
            conn.execute(
                "INSERT OR IGNORE INTO processed_files (id, file_path, "
                "session_dir, chunk_datetime) VALUES (?,?,?,?)",
                (fid, f"/s/{fid}.mat", "/sess", cdt))
            conn.execute(
                "INSERT INTO review_state (file_id, user_email, status, "
                "animal_id, markers_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (fid, "p", "pi_approved", _ANIMAL,
                 json.dumps([{"EO_sec": eo, "racine": rac, "type": "LVF"}]),
                 "t", "t"))
        conn.commit()


def _store(tmp_path):
    s = Store(str(tmp_path / "data" / "m.db"))
    _seed(s, [(1, "2026_03_01__00_00_00", 100.0, 3),
              (2, "2026_03_02__00_00_00", 200.0, 2),
              (3, "2026_03_03__00_00_00", 300.0, 4)])
    return s


# ------------------------------------------------------- store layer --- #

def test_exclude_then_unexclude_is_sparse(tmp_path):
    s = _store(tmp_path)
    assert s.excluded_seizure_keys(_ANIMAL) == set()
    s.set_seizure_excluded(_ANIMAL, 2, 200.0, True)
    assert s.excluded_seizure_keys(_ANIMAL) == {(2, 200.0)}
    # Un-excluding deletes the row (sparse table -- only exclusions stored).
    s.set_seizure_excluded(_ANIMAL, 2, 200.0, False)
    assert s.excluded_seizure_keys(_ANIMAL) == set()
    with s.connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM periictal_seizure_exclusion").fetchone()["c"]
    assert n == 0


def test_exclude_key_rounds_eo(tmp_path):
    s = _store(tmp_path)
    s.set_seizure_excluded(_ANIMAL, 2, 200.004, True)   # rounds to 200.0
    assert (2, 200.0) in s.excluded_seizure_keys(_ANIMAL)
    assert s.seizure_excl_key(2, 200.004) == (2, 200.0)


# ------------------------------------------------ included_seizures --- #

def test_included_seizures_drops_excluded(tmp_path):
    s = _store(tmp_path)
    assert len(scored_seizures(s, _ANIMAL)) == 3
    s.set_seizure_excluded(_ANIMAL, 2, 200.0, True)
    inc = included_seizures(s, _ANIMAL)
    assert len(inc) == 2
    assert all(sz.file_id != 2 for sz in inc)


def test_included_seizures_no_exclusions_is_identity(tmp_path):
    s = _store(tmp_path)
    assert [z.file_id for z in included_seizures(s, _ANIMAL)] == \
           [z.file_id for z in scored_seizures(s, _ANIMAL)]


def test_exclusion_is_per_animal(tmp_path):
    s = _store(tmp_path)
    s.set_seizure_excluded("BCH999", 2, 200.0, True)     # a different animal
    assert len(included_seizures(s, _ANIMAL)) == 3       # unaffected


# ----------------------------------------------- cache signature --- #

def test_signature_changes_with_exclusion(tmp_path):
    from src.periictal import persist as P
    s = _store(tmp_path)
    before = P._seizure_sig(s, _ANIMAL)
    s.set_seizure_excluded(_ANIMAL, 2, 200.0, True)
    after = P._seizure_sig(s, _ANIMAL)
    assert before != after, "excluding a seizure must invalidate the matrix cache"
