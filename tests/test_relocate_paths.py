"""Stale-path relocation: drive-swap candidate reconstruction + persisted
repointing, with reachable/unresolved accounting.

Run with: pytest tests/test_relocate_paths.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                         # noqa: E402
from src.maintenance import relocate_paths as rp       # noqa: E402


# ------------------------------------------------------------------ #
#  tail extraction + candidate generation
# ------------------------------------------------------------------ #

def test_tail_variants_strips_drive_and_bhz():
    v = rp._tail_variants(r"Z:\BHZ\sessA\subB\rec.mat")
    assert r"BHZ\sessA\subB\rec.mat" in v          # full tail below drive
    assert r"sessA\subB\rec.mat" in v              # BHZ-stripped variant


def test_candidate_paths_swaps_drive_roots():
    cands = list(rp.candidate_paths(r"Z:\BHZ\s\f.mat", ["E:\\", "J:\\"]))
    assert r"E:\s\f.mat" in cands                  # BHZ stripped, drive swapped
    assert r"J:\BHZ\s\f.mat" in cands              # full tail on another drive


# ------------------------------------------------------------------ #
#  relocate_one finds a file living under a different "drive" root
# ------------------------------------------------------------------ #

def test_relocate_one_finds_on_other_root(tmp_path):
    # simulate the live drive E:\ as tmp_path/live/  (root passed explicitly)
    live = tmp_path / "live"
    sess = live / "sessA" / "subB"
    sess.mkdir(parents=True)
    real = sess / "rec.mat"
    real.write_bytes(b"x")
    stored = r"Z:\BHZ\sessA\subB\rec.mat"          # dead mapped-drive path
    root = str(live) + os.sep
    found = rp.relocate_one(stored, [root])
    assert found == str(real)


def test_relocate_one_unresolved_returns_none(tmp_path):
    root = str(tmp_path / "empty") + os.sep
    os.makedirs(root, exist_ok=True)
    assert rp.relocate_one(r"Z:\BHZ\x\y.mat", [root]) is None


# ------------------------------------------------------------------ #
#  end-to-end over the store: dry-run vs apply, persistence, already_ok
# ------------------------------------------------------------------ #

def _register(store, path):
    return store.register_file(path, 0, 0.0, os.path.dirname(path),
                               "sess", "2026_02_01__00_00_00")


def test_relocate_missing_paths_apply_persists(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    # a real, reachable file (already_ok) ...
    ok = tmp_path / "ok.mat"
    ok.write_bytes(b"x")
    fid_ok = _register(store, str(ok))
    # ... and a dead mapped-drive path whose file lives under tmp_path/live
    live = tmp_path / "live" / "sessA"
    live.mkdir(parents=True)
    moved = live / "rec.mat"
    moved.write_bytes(b"x")
    fid_dead = _register(store, r"Z:\BHZ\sessA\rec.mat")

    # pretend tmp_path/live is a mounted drive root
    monkeypatch.setattr(rp, "available_drive_roots",
                        lambda: [str(tmp_path / "live") + os.sep])

    s = rp.relocate_missing_paths(store, apply=True)
    assert s.already_ok == 1
    assert s.relocated == 1
    assert s.unresolved == 0

    # the dead row was repointed to the real file
    with store.connection() as conn:
        p = conn.execute("SELECT file_path FROM processed_files WHERE id=?",
                         (fid_dead,)).fetchone()["file_path"]
    assert p == str(moved)
    assert os.path.isfile(p)


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    live = tmp_path / "live" / "sessA"
    live.mkdir(parents=True)
    (live / "rec.mat").write_bytes(b"x")
    fid = _register(store, r"Z:\BHZ\sessA\rec.mat")
    monkeypatch.setattr(rp, "available_drive_roots",
                        lambda: [str(tmp_path / "live") + os.sep])

    s = rp.relocate_missing_paths(store, apply=False)
    assert s.relocated == 1
    with store.connection() as conn:
        p = conn.execute("SELECT file_path FROM processed_files WHERE id=?",
                         (fid,)).fetchone()["file_path"]
    assert p == r"Z:\BHZ\sessA\rec.mat"            # unchanged in a dry run


def test_relocate_file_path_store_helper(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    target = tmp_path / "new" / "rec.mat"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    fid = _register(store, r"U:\OLD\rec.mat")
    assert store.relocate_file_path(fid, str(target)) is True
    with store.connection() as conn:
        row = conn.execute("SELECT file_path, session_dir FROM processed_files "
                           "WHERE id=?", (fid,)).fetchone()
    assert row["file_path"] == str(target)
    assert row["session_dir"] == os.path.dirname(str(target))
