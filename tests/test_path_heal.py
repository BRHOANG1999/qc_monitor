"""Automatic path self-heal: run_heal_pass (cursor batching, evoked relocation,
UNIQUE-collision + wall-clock accounting) and the EEGLocator wall-clock bound.

Run with: pytest tests/test_path_heal.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                           # noqa: E402
from src.maintenance import relocate_paths as rp         # noqa: E402
from src.utils import past_events as pe                  # noqa: E402


_BS = chr(92)


def _register(store, path):
    return store.register_file(path, 0, 0.0, os.path.dirname(path),
                               "sess", "2026_02_01__00_00_00")


# ------------------------------------------------------------------ #
#  UNC share-relative tails + recycle-bin/system-dir candidate guard
# ------------------------------------------------------------------ #

def test_unc_tail_variants_keep_and_drop_share():
    v = rp._tail_variants("//100.106.104.22/database/MARCH_2026/sessA/f.mat")
    assert r"database\MARCH_2026\sessA\f.mat" in v      # keeps 'database' folder
    assert r"MARCH_2026\sessA\f.mat" in v               # drops the share segment


def test_tail_guard_rejects_recycle_and_system():
    assert rp._tail_ok("database" + _BS + "s" + _BS + "f.mat")
    assert not rp._tail_ok("$RECYCLE.BIN" + _BS + "S-1" + _BS + "f.mat")
    assert not rp._tail_ok("System Volume Information" + _BS + "f.mat")


def test_recycle_tail_yields_no_candidates():
    # a stored path inside a recycle bin -> no drive-swap candidate at all
    cands = list(rp.candidate_paths(
        "//host/database/$RECYCLE.BIN/S-1/rec.mat", ["H:" + _BS]))
    assert cands == []


def test_unc_row_heals_onto_drive(tmp_path, monkeypatch):
    """A //host/database/... row whose file now lives at <drive>\\database\\...
    heals via the cheap candidate stat -- no recursive walk."""
    store = Store(str(tmp_path / "data" / "m.db"))
    live = _live_root(tmp_path, monkeypatch)
    (live / "database" / "MARCH_2026" / "sessA").mkdir(parents=True)
    moved = live / "database" / "MARCH_2026" / "sessA" / "rec.mat"
    moved.write_bytes(b"x")
    fid = _register(store, "//100.106.104.22/database/MARCH_2026/sessA/rec.mat")

    summary, _ = rp.run_heal_pass(store, {}, apply=True, include_evoked=False,
                                  use_locator=False)          # NO walk
    assert summary.relocated == 1
    with store.connection() as conn:
        p = conn.execute("SELECT file_path FROM processed_files WHERE id=?",
                         (fid,)).fetchone()["file_path"]
    assert p == str(moved)


def _live_root(tmp_path, monkeypatch):
    """Make tmp_path/live look like the one mounted drive."""
    root = str(tmp_path / "live") + os.sep
    os.makedirs(root, exist_ok=True)
    monkeypatch.setattr(rp, "available_drive_roots", lambda: [root])
    return tmp_path / "live"


# ------------------------------------------------------------------ #
#  1. repoint stale, leave healthy; report a cursor
# ------------------------------------------------------------------ #

def test_heal_pass_repoints_stale_keeps_healthy(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    ok = tmp_path / "ok.mat"
    ok.write_bytes(b"x")
    _register(store, str(ok))                            # healthy
    live = _live_root(tmp_path, monkeypatch)
    (live / "sessA").mkdir(parents=True)
    moved = live / "sessA" / "rec.mat"
    moved.write_bytes(b"x")
    fid = _register(store, r"Z:\BHZ\sessA\rec.mat")      # stale

    summary, cursor = rp.run_heal_pass(store, {}, apply=True,
                                       include_evoked=False, use_locator=False)
    assert summary.already_ok == 1
    assert summary.relocated == 1
    assert summary.unresolved == 0
    assert cursor == 0                                   # batch exhausted -> wrap
    with store.connection() as conn:
        p = conn.execute("SELECT file_path FROM processed_files WHERE id=?",
                         (fid,)).fetchone()["file_path"]
    assert p == str(moved)


# ------------------------------------------------------------------ #
#  2. rows_per_pass cursor batching + wrap
# ------------------------------------------------------------------ #

def test_heal_pass_cursor_batches_and_wraps(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    live = _live_root(tmp_path, monkeypatch)
    ids = []
    for i in range(3):
        d = live / f"s{i}"
        d.mkdir(parents=True)
        (d / "rec.mat").write_bytes(b"x")
        ids.append(_register(store, rf"Z:\BHZ\s{i}\rec.mat"))

    s1, c1 = rp.run_heal_pass(store, {}, apply=True, rows_per_pass=2,
                              include_evoked=False, use_locator=False)
    assert s1.scanned == 2 and s1.relocated == 2
    assert c1 == ids[1]                                  # cursor at 2nd row's id

    s2, c2 = rp.run_heal_pass(store, {}, apply=True, rows_per_pass=2,
                              after_id=c1, include_evoked=False,
                              use_locator=False)
    assert s2.scanned == 1 and s2.relocated == 1         # the 3rd row
    assert c2 == 0                                       # exhausted -> wrap


# ------------------------------------------------------------------ #
#  3. evoked relocation across BOTH tables (non-unique)
# ------------------------------------------------------------------ #

def test_evoked_relocation_updates_both_tables(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    fid = _register(store, str(tmp_path / "ok.mat"))
    (tmp_path / "ok.mat").write_bytes(b"x")
    live = _live_root(tmp_path, monkeypatch)
    (live / "ev").mkdir(parents=True)
    real_ev = live / "ev" / "rec_evoked.mat"
    real_ev.write_bytes(b"x")
    stale_ev = r"Z:\BHZ\ev\rec_evoked.mat"
    store.insert_matlab_result(fid, {"evoked_output_path": stale_ev})
    store.insert_evoked_summary(fid, {"evoked_output_path": stale_ev})

    # direct store helper: both rows updated
    n = store.relocate_evoked_output_path(stale_ev, str(real_ev))
    assert n == 2
    with store.connection() as conn:
        a = conn.execute("SELECT evoked_output_path FROM matlab_results "
                         "WHERE file_id=?", (fid,)).fetchone()[0]
        b = conn.execute("SELECT evoked_output_path FROM evoked_summary "
                         "WHERE file_id=?", (fid,)).fetchone()[0]
    assert a == str(real_ev) and b == str(real_ev)


def test_heal_pass_include_evoked(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    fid = _register(store, str(tmp_path / "ok.mat"))
    (tmp_path / "ok.mat").write_bytes(b"x")
    live = _live_root(tmp_path, monkeypatch)
    (live / "ev").mkdir(parents=True)
    (live / "ev" / "rec_evoked.mat").write_bytes(b"x")
    stale_ev = r"Z:\BHZ\ev\rec_evoked.mat"
    store.insert_matlab_result(fid, {"evoked_output_path": stale_ev})

    summary, _ = rp.run_heal_pass(store, {}, apply=True, include_evoked=True,
                                  use_locator=False)
    assert summary.evoked_relocated == 1
    with store.connection() as conn:
        p = conn.execute("SELECT evoked_output_path FROM matlab_results "
                         "WHERE file_id=?", (fid,)).fetchone()[0]
    assert p == str(live / "ev" / "rec_evoked.mat")


# ------------------------------------------------------------------ #
#  4. EEGLocator wall-clock bound
# ------------------------------------------------------------------ #

def test_locator_wall_clock_bound_bails(tmp_path, monkeypatch):
    # a tree deep enough to walk more than one dir
    root = tmp_path / "drive"
    deep = root / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    target = deep / "rec.mat"
    target.write_bytes(b"x")

    # monotonic jumps far past any deadline after the first read -> instant bail
    ticks = iter([0.0] + [1000.0] * 50)
    monkeypatch.setattr(pe.time, "monotonic", lambda: next(ticks))
    loc = pe.EEGLocator([str(root) + os.sep], auto_drives=False,
                        max_walk_seconds=0.01)
    assert loc.resolve("", "rec.mat") is None            # deadline cut it short


def test_locator_no_bound_still_finds(tmp_path):
    root = tmp_path / "drive"
    deep = root / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "rec.mat").write_bytes(b"x")
    loc = pe.EEGLocator([str(root) + os.sep], auto_drives=False)  # default None
    assert loc.resolve("", "rec.mat") == str(deep / "rec.mat")


# ------------------------------------------------------------------ #
#  5. UNIQUE collision: two stale rows -> one live path
# ------------------------------------------------------------------ #

def test_heal_pass_unique_collision_counted(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    live = _live_root(tmp_path, monkeypatch)
    (live / "sessA").mkdir(parents=True)
    (live / "sessA" / "rec.mat").write_bytes(b"x")
    # two DIFFERENT stale paths whose tail reconstructs to the SAME live file
    _register(store, r"Z:\BHZ\sessA\rec.mat")
    _register(store, r"U:\sessA\rec.mat")

    summary, _ = rp.run_heal_pass(store, {}, apply=True, include_evoked=False,
                                  use_locator=False)
    assert summary.relocated == 1                        # first wins
    assert summary.collided == 1                         # second collides
    assert summary.unresolved == 0


# ------------------------------------------------------------------ #
#  6. whole-pass wall-clock budget stops early with a partial cursor
# ------------------------------------------------------------------ #

def test_heal_pass_wall_clock_stops_early(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "data" / "m.db"))
    live = _live_root(tmp_path, monkeypatch)
    for i in range(3):
        d = live / f"s{i}"
        d.mkdir(parents=True)
        (d / "rec.mat").write_bytes(b"x")
        _register(store, rf"Z:\BHZ\s{i}\rec.mat")

    # monotonic: first call sets the deadline, next call is already past it, so
    # the row loop breaks before processing any row.
    ticks = iter([0.0, 1000.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(rp.time, "monotonic", lambda: next(ticks))
    summary, cursor = rp.run_heal_pass(store, {}, apply=True, rows_per_pass=10,
                                       wall_clock_sec=0.01, include_evoked=False,
                                       use_locator=False)
    assert summary.scanned == 0                          # cut off immediately
    assert cursor == 0                                   # nothing processed


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
