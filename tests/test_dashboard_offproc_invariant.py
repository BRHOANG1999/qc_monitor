"""Enforce the dashboard off-proc invariant.

The dashboard process must never do a raw GIL-holding recording read or video
decode on a thread it owns -- seven freezes this year were that one class. Those
primitives (scipy.io.loadmat / h5py.File / cv2 / load_mat) belong ONLY behind the
off-process readers (chunk_reader.read_chunk_offproc, snapshot_reader.
decode_snapshot_offproc); the dashboard reaches them via get_chunk (which routes
off-proc when QC_DASHBOARD_ROLE is set). This is the mechanical guard against the
next feature reintroducing the class -- see src/utils/offproc_guard.py.

If test_no_raw_share_reads_in_dashboard fails: you added a raw read/decode to a
dashboard module. Route it through the off-proc reader instead.
"""

import os
import re

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DASH_DIR = os.path.join(_REPO, "src", "dashboard")

# Raw GIL-holding read/decode API that must not appear in the dashboard package.
_FORBIDDEN = [
    (re.compile(r"^\s*(import\s+cv2\b|from\s+cv2\b)"), "cv2 import (video decode)"),
    (re.compile(r"\bscipy\.io\.loadmat\s*\("), "scipy.io.loadmat("),
    (re.compile(r"\bh5py\.File\s*\("), "h5py.File("),
    (re.compile(r"(?<![\w.])load_mat\s*\("), "load_mat("),
]


def _dashboard_py_files():
    for root, _dirs, files in os.walk(_DASH_DIR):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_no_raw_share_reads_in_dashboard():
    """No dashboard module may call a raw share read / video decode directly."""
    offenders = []
    for path in _dashboard_py_files():
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if line.lstrip().startswith("#"):       # skip comment lines
                    continue
                for rx, what in _FORBIDDEN:
                    if rx.search(line):
                        rel = os.path.relpath(path, _REPO)
                        offenders.append(f"{rel}:{i}  [{what}]  {line.strip()[:90]}")
    assert not offenders, (
        "Dashboard modules must not do raw share reads / video decode on their "
        "own threads -- route through chunk_reader.read_chunk_offproc / "
        "snapshot_reader.decode_snapshot_offproc (offproc_guard).\n  "
        + "\n  ".join(offenders))


def test_require_offproc_raises_only_on_dashboard_owned_thread(monkeypatch):
    """The runtime guard: raises in the dashboard process on its own thread,
    passes in the daemon and inside an off-proc worker."""
    from src.utils import offproc_guard as g

    # Daemon (no dashboard role): allowed.
    monkeypatch.delenv("QC_DASHBOARD_ROLE", raising=False)
    monkeypatch.delenv("QC_OFFPROC_WORKER", raising=False)
    g.require_offproc("load_mat")            # no raise

    # Dashboard process, dashboard-owned thread: forbidden.
    monkeypatch.setenv("QC_DASHBOARD_ROLE", "1")
    monkeypatch.delenv("QC_OFFPROC_WORKER", raising=False)
    with pytest.raises(RuntimeError):
        g.require_offproc("load_mat")

    # Dashboard process, but inside an off-proc worker: allowed.
    g.mark_worker()
    g.require_offproc("load_mat")            # no raise


def test_read_file_evoked_impl_is_guarded(monkeypatch):
    """The evoked h5py read carries the runtime guard too: the raw
    ``_read_file_evoked_impl`` must refuse to run on a dashboard-owned thread
    (the public ``read_file_evoked`` router sends those reads off-process), and
    must run once marked as an off-proc worker."""
    from src.utils import evoked_output as eo

    # Dashboard-owned thread: the raw h5py read is forbidden.
    monkeypatch.setenv("QC_DASHBOARD_ROLE", "1")
    monkeypatch.delenv("QC_OFFPROC_WORKER", raising=False)
    with pytest.raises(RuntimeError):
        eo._read_file_evoked_impl("does_not_exist_evoked.mat")

    # Inside an off-proc worker: allowed -- a missing/unreadable file yields {}.
    monkeypatch.setenv("QC_OFFPROC_WORKER", "1")
    assert eo._read_file_evoked_impl("does_not_exist_evoked.mat") == {}
