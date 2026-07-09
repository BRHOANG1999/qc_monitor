"""Duration backfill: header-only sample-count read + idempotent DB fill, with
reachable/unreachable accounting.

Run with: pytest tests/test_duration_backfill.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store                       # noqa: E402
from src.maintenance import duration_backfill as bf  # noqa: E402
from src.utils.mat_loader import read_mat_header      # noqa: E402


def _write_mat(path, n_samples, n_channels, fs):
    """A minimal KMrecorder-style v5 .mat (sbuf + fs) for header reads."""
    import scipy.io
    sbuf = np.random.default_rng(0).standard_normal((n_samples, n_channels))
    scipy.io.savemat(path, {"sbuf": sbuf, "fs": float(fs)})


def _register(store, path, cd="2026_03_02__00_00_00"):
    fid = store.register_file(path, 0, 0.0, os.path.dirname(path),
                              "sess", cd)
    return fid


# ------------------------------------------------------------------ #
#  header-only read matches load_mat's num_samples / duration
# ------------------------------------------------------------------ #

def test_read_mat_header_dims_and_fs(tmp_path):
    p = str(tmp_path / "chunk.mat")
    _write_mat(p, n_samples=1000, n_channels=4, fs=500.0)
    nsamp, nchan, fs = read_mat_header(p)
    assert nsamp == 1000
    assert nchan == 4                                  # longer axis = samples
    assert abs(fs - 500.0) < 1e-9
    assert abs((nsamp / fs) - 2.0) < 1e-9              # duration 2.0 s


# ------------------------------------------------------------------ #
#  apply fills num_samples + duration_sec
# ------------------------------------------------------------------ #

def test_backfill_apply_fills_duration(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    p = str(tmp_path / "rec.mat")
    _write_mat(p, n_samples=2500, n_channels=2, fs=500.0)
    fid = _register(store, p)
    # precondition: duration is NULL
    with store.connection() as conn:
        row = conn.execute("SELECT duration_sec, num_samples FROM processed_files "
                           "WHERE id=?", (fid,)).fetchone()
        assert row["duration_sec"] is None

    s = bf.backfill_durations(store, apply=True)
    assert s.filled == 1 and s.unreachable == 0 and s.unreadable == 0

    with store.connection() as conn:
        row = conn.execute("SELECT duration_sec, num_samples, sampling_rate "
                           "FROM processed_files WHERE id=?", (fid,)).fetchone()
    assert abs(row["duration_sec"] - 5.0) < 1e-6       # 2500/500
    assert row["num_samples"] == 2500
    assert abs(row["sampling_rate"] - 500.0) < 1e-9


# ------------------------------------------------------------------ #
#  dry-run writes nothing; unreachable files are counted per-root
# ------------------------------------------------------------------ #

def test_dry_run_and_unreachable_accounting(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    real = str(tmp_path / "real.mat")
    _write_mat(real, 1000, 2, 250.0)
    fid_real = _register(store, real)
    # a row pointing at a file that isn't on disk (simulates archived drive)
    ghost = str(tmp_path / "missing" / "gone.mat")
    _register(store, ghost)

    s = bf.backfill_durations(store, apply=False)      # DRY RUN
    assert s.scanned == 2
    assert s.filled == 1                               # would fill the real one
    assert s.unreachable == 1                          # the ghost
    # nothing persisted in a dry run
    with store.connection() as conn:
        d = conn.execute("SELECT duration_sec FROM processed_files WHERE id=?",
                         (fid_real,)).fetchone()["duration_sec"]
    assert d is None
    # per-root breakdown carries both outcomes
    roots = s.by_root
    assert sum(v["filled"] for v in roots.values()) == 1
    assert sum(v["unreachable"] for v in roots.values()) == 1


# ------------------------------------------------------------------ #
#  idempotent: a second apply fills nothing, never clobbers
# ------------------------------------------------------------------ #

def test_idempotent_second_run_fills_zero(tmp_path):
    store = Store(str(tmp_path / "data" / "m.db"))
    p = str(tmp_path / "rec.mat")
    _write_mat(p, 1200, 3, 400.0)
    fid = _register(store, p)

    s1 = bf.backfill_durations(store, apply=True)
    assert s1.filled == 1
    dur1 = _dur(store, fid)

    s2 = bf.backfill_durations(store, apply=True)      # already filled
    assert s2.scanned == 0 and s2.filled == 0          # row no longer qualifies
    assert _dur(store, fid) == dur1                    # unchanged


def _dur(store, fid):
    with store.connection() as conn:
        return conn.execute("SELECT duration_sec FROM processed_files WHERE id=?",
                            (fid,)).fetchone()["duration_sec"]


# ------------------------------------------------------------------ #
#  path_root classification (drive letter vs UNC share)
# ------------------------------------------------------------------ #

def test_path_root_classification():
    assert bf.path_root(r"Z:\BHZ\a\b.mat") == "Z:\\"
    assert bf.path_root(r"U:\FEB\x.mat") == "U:\\"
    assert bf.path_root(r"\\100.106.104.22\bhz\a\b.mat") == r"\\100.106.104.22\bhz"
    assert bf.path_root("//100.106.104.22/database/x.mat") == r"\\100.106.104.22\database"
    assert bf.path_root(None) == "(null)"
