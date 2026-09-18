"""Re-key processed_files.session_dir back to the canonical session_config key
after a path-heal desynced it to the physical drive letter.

Run with: pytest tests/test_resync_session_dir.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                              # noqa: E402
from src.maintenance import resync_session_dir as rs        # noqa: E402


def _cfg(store, session_dir):
    return store.upsert_session_config(
        session_dir, {"channel_names": ["BCH111SR"], "eeg_channels": [1]})


def _pf(store, file_path, session_dir):
    return store.register_file(file_path, 0, 0.0, session_dir, "sess",
                               "2026_03_02__17_26_16")


# ------------------------------------------------------------------ #
#  tail / basename helpers
# ------------------------------------------------------------------ #

def test_tail2_and_basename_casefold():
    assert rs._tail2(r"H:\database\MARCH_2026\sessA") == r"march_2026\sessa"
    assert rs._tail2("//host/bhz/database/MARCH_2026/sessA") == r"march_2026\sessa"
    assert rs._base(r"H:\x\SessB") == "sessb"
    assert rs._is_unc("//host/share/x") and rs._is_unc(r"\\host\share\x")
    assert not rs._is_unc(r"H:\x")


# ------------------------------------------------------------------ #
#  re-key by unique 2-segment tail; file_path is left untouched
# ------------------------------------------------------------------ #

def test_rekey_by_tail_apply(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    canon = "//100.106.104.22/bhz/database/MARCH_2026/sessA"
    _cfg(store, canon)
    # a healed processed_files row: physical path on H:, session_dir desynced
    phys = r"H:\database\MARCH_2026\sessA\rec___2026_03_02__17_26_16.mat"
    fid = _pf(store, phys, r"H:\database\MARCH_2026\sessA")

    s = rs.resync_session_dirs(store, apply=True)
    assert s.scanned == 1 and s.rekeyed == 1
    assert s.ambiguous == 0 and s.unmatched == 0
    with store.connection() as conn:
        row = conn.execute("SELECT file_path, session_dir FROM processed_files "
                           "WHERE id=?", (fid,)).fetchone()
    assert row["session_dir"] == canon                 # re-keyed to canonical
    assert row["file_path"] == phys                    # physical path untouched


def test_dry_run_writes_nothing(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    _cfg(store, "//h/bhz/database/MARCH_2026/sessA")
    fid = _pf(store, r"H:\database\MARCH_2026\sessA\r.mat",
              r"H:\database\MARCH_2026\sessA")
    s = rs.resync_session_dirs(store, apply=False)
    assert s.rekeyed == 1
    with store.connection() as conn:
        sd = conn.execute("SELECT session_dir FROM processed_files WHERE id=?",
                          (fid,)).fetchone()["session_dir"]
    assert sd == r"H:\database\MARCH_2026\sessA"        # unchanged in a dry run


# ------------------------------------------------------------------ #
#  already-matched rows are a no-op; unknown sessions are left alone
# ------------------------------------------------------------------ #

def test_matched_row_is_noop(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    canon = "//h/bhz/database/MARCH_2026/sessMatched"
    _cfg(store, canon)
    _pf(store, r"H:\database\MARCH_2026\sessMatched\r.mat", canon)  # matches
    s = rs.resync_session_dirs(store, apply=True)
    assert s.scanned == 0 and s.rekeyed == 0           # not orphaned -> not scanned


def test_unmatched_session_left_alone(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    _cfg(store, "//h/bhz/database/MARCH_2026/knownSess")
    fid = _pf(store, r"H:\database\APRIL_2026\ghostSess\r.mat",
              r"H:\database\APRIL_2026\ghostSess")       # no session_config
    s = rs.resync_session_dirs(store, apply=True)
    assert s.scanned == 1 and s.rekeyed == 0 and s.unmatched == 1
    with store.connection() as conn:
        sd = conn.execute("SELECT session_dir FROM processed_files WHERE id=?",
                          (fid,)).fetchone()["session_dir"]
    assert sd == r"H:\database\APRIL_2026\ghostSess"    # untouched


# ------------------------------------------------------------------ #
#  fallback + tiebreak
# ------------------------------------------------------------------ #

def test_rekey_by_basename_fallback(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    canon = "//h/bhz/database/APRIL_2026/uniqueSess"     # tail2 differs, base unique
    _cfg(store, canon)
    fid = _pf(store, r"H:\weird\layout\uniqueSess\r.mat",
              r"H:\weird\layout\uniqueSess")
    s = rs.resync_session_dirs(store, apply=True)
    assert s.rekeyed == 1
    with store.connection() as conn:
        sd = conn.execute("SELECT session_dir FROM processed_files WHERE id=?",
                          (fid,)).fetchone()["session_dir"]
    assert sd == canon


def test_prefer_unc_when_tail_ambiguous(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    unc = "//h/bhz/database/MARCH_2026/sessDup"
    zdrive = r"Z:\database\MARCH_2026\sessDup"
    _cfg(store, unc)
    _cfg(store, zdrive)                                  # same tail, two canonicals
    fid = _pf(store, r"H:\database\MARCH_2026\sessDup\r.mat",
              r"H:\database\MARCH_2026\sessDup")
    s = rs.resync_session_dirs(store, apply=True)
    assert s.rekeyed == 1
    with store.connection() as conn:
        sd = conn.execute("SELECT session_dir FROM processed_files WHERE id=?",
                          (fid,)).fetchone()["session_dir"]
    assert sd == unc                                     # UNC canonical wins
