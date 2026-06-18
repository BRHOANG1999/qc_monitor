"""Historical-example ingest + candidate-pool union + EEG location cache.

Run with: pytest tests/test_past_events_store.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402


def _store(tmp_path):
    return Store(str(tmp_path / "data" / "monitor.db"))


def _pf(store, file_id, session="sessA"):
    with store.connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_files (id, file_path, "
            "session_dir, duration_sec) VALUES (?,?,?,?)",
            (file_id, f"/f/{file_id}.mat", session, 3600.0))
        conn.commit()


def _example(path, *, animal="BCH060", racine=3, typ="HYP",
             names=("stimCopy", "BCH060SR"), markers=None):
    return {
        "recorded_path": path,
        "folder": os.path.dirname(path),
        "session_dir": os.path.dirname(path),
        "session_name": os.path.basename(os.path.dirname(path)),
        "filename": os.path.basename(path),
        "channel_names": list(names),
        "fs": 20000.0,
        "animal": animal,
        "chunk_datetime": "2026-06-02T10:11:41",
        "markers": markers or [{"type": typ, "racine": racine,
                                "EO_sec": 2.0, "LAS_sec": None,
                                "BO_sec": None, "PID_sec": None,
                                "BB_sec": None}],
        "rep_type": typ,
        "rep_racine": racine,
        "source_csv": "x.csv",
        "peak_stamp": 1.0,
    }


def test_import_and_candidate_pool(tmp_path):
    store = _store(tmp_path)
    n = store.import_historical_examples([
        _example("D:/arch/sessA/rec1.mat", animal="BCH060", racine=4),
        _example("D:/arch/sessB/rec2.mat", animal="BCH061", racine=2,
                 typ="LVF", names=("stimCopy", "BCH061SLM")),
    ])
    assert n == 2
    pool = store.training_candidate_pool("stu@x")
    by_animal = {c["animals"][0]: c for c in pool}
    assert set(by_animal) == {"BCH060", "BCH061"}
    assert by_animal["BCH060"]["rep_racine"] == 4
    assert by_animal["BCH060"]["has_seizure"] is True
    assert by_animal["BCH061"]["rep_type"] == "LVF"


def test_import_is_idempotent(tmp_path):
    store = _store(tmp_path)
    ex = _example("D:/arch/sessA/rec1.mat")
    store.import_historical_examples([ex])
    # Re-import the same recording with an updated racine -> upsert, no dup.
    ex2 = _example("D:/arch/sessA/rec1.mat", racine=5)
    store.import_historical_examples([ex2])
    pool = store.training_candidate_pool("stu@x")
    assert len(pool) == 1
    assert pool[0]["rep_racine"] == 5


def test_validated_events_fallback_to_external(tmp_path):
    store = _store(tmp_path)
    store.import_historical_examples([_example("D:/arch/sessA/rec1.mat")])
    pool = store.training_candidate_pool("stu@x")
    fid = pool[0]["file_id"]
    ev = store.validated_events_for_file(fid)
    assert ev and ev[0]["type"] == "HYP" and ev[0]["EO_sec"] == 2.0


def test_real_approval_wins_over_external(tmp_path):
    store = _store(tmp_path)
    # Same file is both a real pi_approved review AND a historical import.
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (file_path, session_dir) "
            "VALUES ('D:/arch/sessA/rec1.mat', 'D:/arch/sessA')")
        fid = conn.execute(
            "SELECT id FROM processed_files WHERE file_path=?",
            ("D:/arch/sessA/rec1.mat",)).fetchone()["id"]
        conn.execute(
            """INSERT INTO review_state (file_id, user_email, status,
                   markers_json, created_at, updated_at)
               VALUES (?, 'pi@x', 'pi_approved', ?, '2026-01-01',
                       '2026-01-01')""",
            (fid, json.dumps([{"type": "LVF", "racine": 1,
                               "EO_sec": 9.0}])))
        conn.commit()
    store.import_historical_examples([_example("D:/arch/sessA/rec1.mat",
                                               racine=4, typ="HYP")])
    pool = store.training_candidate_pool("stu@x")
    assert len(pool) == 1                     # deduped by file_id
    # The real approval's answer wins.
    ev = store.validated_events_for_file(pool[0]["file_id"])
    assert ev[0]["type"] == "LVF" and ev[0]["EO_sec"] == 9.0


def test_resolve_training_file_lazy(tmp_path):
    store = _store(tmp_path)
    # Catalog with a stale recorded path; the real file lives elsewhere.
    real = tmp_path / "drive" / "MONTH" / "sess"
    real.mkdir(parents=True)
    fn = "rec1.mat"
    (real / fn).write_bytes(b"x")
    store.import_historical_examples(
        [_example("Z:/offline/wrong/rec1.mat")])
    pool = store.training_candidate_pool("stu@x")
    fid = pool[0]["file_id"]

    from src.utils import past_events as pe
    loc = pe.EEGLocator([str(tmp_path / "drive")], auto_drives=False,
                        cache_get=store.eeg_location_get,
                        cache_put=store.eeg_location_put)
    got = store.resolve_training_file(fid, loc)
    assert got == str(real / fn)
    # The processed_files row was repointed at the found path.
    with store.connection() as conn:
        row = conn.execute(
            "SELECT file_path FROM processed_files WHERE id=?",
            (fid,)).fetchone()
    assert row["file_path"] == str(real / fn)


def test_resolve_training_file_already_on_disk(tmp_path):
    store = _store(tmp_path)
    real = tmp_path / "sess"
    real.mkdir()
    (real / "rec1.mat").write_bytes(b"x")
    store.import_historical_examples(
        [_example(str(real / "rec1.mat"))])
    fid = store.training_candidate_pool("stu@x")[0]["file_id"]

    from src.utils import past_events as pe
    # auto_drives off + no roots: if it tried to search it'd find nothing,
    # but the path already exists so resolution is a no-op stat.
    loc = pe.EEGLocator(auto_drives=False)
    assert store.resolve_training_file(fid, loc) == str(real / "rec1.mat")


def test_historical_session_gets_stim_copy_channels(tmp_path):
    store = _store(tmp_path)
    store.import_historical_examples([_example(
        "D:/arch/sessA/rec1.mat",
        names=("stimCopy", "BCH060SR", "stimCopy", "BCH061SLM"))])
    from src.utils.stim_blank import stim_copy_channels
    got = stim_copy_channels(store, "D:/arch/sessA")
    assert got == {0, 2}             # name-derived, so blanking can run


def test_reset_student_lowers_stage_keeps_history(tmp_path):
    store = _store(tmp_path)
    _pf(store, 5)
    # Build some attempt history at stage 1, then advance + certify-ish.
    store.add_training_attempt("stu@x", 1, 5, json.dumps({}), 1.0)
    store.advance_student("stu@x", 3)
    assert store.get_training_progress("stu@x")["unlocked_stage"] == 3
    # Open a round so we can confirm reset closes it.
    store.create_training_round("stu@x", 1, [5])
    assert store.current_training_round("stu@x", 1) is not None
    # Reset back to start.
    store.reset_student("stu@x", 1)
    prog = store.get_training_progress("stu@x")
    assert prog["unlocked_stage"] == 1 and prog["certified_at"] is None
    assert store.current_training_round("stu@x", 1) is None   # round closed
    # History survives.
    assert store.training_lifetime_stats("stu@x", 1)["n"] == 1


def _pf_dur(store, file_id, duration, session="sessD"):
    with store.connection() as conn:
        conn.execute(
            "INSERT INTO processed_files (id, file_path, session_dir, "
            "duration_sec) VALUES (?,?,?,?)",
            (file_id, f"/f/{file_id}.mat", session, duration))
        conn.execute(
            """INSERT INTO review_state (file_id, user_email, status,
                   markers_json, created_at, updated_at)
               VALUES (?, 'pi@x', 'pi_approved', '[]',
                       '2026-01-01','2026-01-01')""", (file_id,))
        conn.commit()


def test_duration_filter_keeps_unknown_drops_short(tmp_path):
    store = _store(tmp_path)
    _pf_dur(store, 1, 3600.0)     # 60 min -> kept
    _pf_dur(store, 2, 600.0)      # 10 min -> dropped
    _pf_dur(store, 3, None)       # unknown -> kept
    # Historical (NULL duration) -> kept.
    store.import_historical_examples([_example("D:/arch/h/recH.mat")])
    pool = store.training_candidate_pool("stu@x", min_duration_sec=1800)
    ids = {c["file_id"] for c in pool}
    assert 1 in ids and 3 in ids and 2 not in ids
    # The historical NULL-duration file is present too.
    assert len(ids) == 3


def test_set_file_duration_only_when_unknown(tmp_path):
    store = _store(tmp_path)
    _pf_dur(store, 7, None)
    store.set_file_duration(7, 1234.0)
    with store.connection() as conn:
        d = conn.execute("SELECT duration_sec FROM processed_files WHERE id=7"
                         ).fetchone()["duration_sec"]
    assert d == 1234.0
    # Does not overwrite a known duration.
    store.set_file_duration(7, 9999.0)
    with store.connection() as conn:
        d = conn.execute("SELECT duration_sec FROM processed_files WHERE id=7"
                         ).fetchone()["duration_sec"]
    assert d == 1234.0


def test_eeg_location_cache_positive_and_negative(tmp_path):
    store = _store(tmp_path)
    assert store.eeg_location_get("rec1.mat") is None      # unknown
    store.eeg_location_put("rec1.mat", "D:/arch/sessA/rec1.mat")
    assert store.eeg_location_get("rec1.mat") == "D:/arch/sessA/rec1.mat"
    # Negative entry within TTL -> "" (skip re-walk).
    store.eeg_location_put("gone.mat", None)
    assert store.eeg_location_get("gone.mat") == ""


def test_eeg_location_negative_expires(tmp_path):
    store = _store(tmp_path)
    store.eeg_location_put("old.mat", None)
    # Backdate the negative entry past the TTL -> re-search (None).
    with store.connection() as conn:
        conn.execute(
            "UPDATE eeg_file_location SET checked_at='2000-01-01T00:00:00' "
            "WHERE filename='old.mat'")
        conn.commit()
    assert store.eeg_location_get("old.mat") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
