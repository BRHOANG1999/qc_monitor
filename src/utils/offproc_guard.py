"""The invariant: nothing in the DASHBOARD process reads a recording off the
share or decodes video on a thread the dashboard owns.

Seven separate freezes this year were one class -- a GIL-holding C call
(scipy.io.loadmat / h5py / cv2 / SMB glob) on a waitress/request thread, which
pins the dashboard for the whole slow read. Fixing instances one at a time just
waits for the next feature to reintroduce it. This module enforces the invariant
MECHANICALLY at runtime, paired with a build test (tests/test_dashboard_offproc
_invariant.py) that fails CI on any new direct raw-read in the dashboard package.

Mechanism: the raw readers (load_mat, the cv2 decode) call require_offproc().
- Daemon process (QC_DASHBOARD_ROLE unset): allowed -- it has no UI to freeze.
- Dashboard process (QC_DASHBOARD_ROLE set) on one of its OWN threads: RAISES,
  loudly and immediately, instead of silently freezing the page.
- The off-process child readers (chunk_reader / snapshot_reader workers) call
  mark_worker() at startup, which stamps QC_OFFPROC_WORKER -- so the read is
  allowed there (that is the whole point: it happens off the dashboard's GIL).
Children inherit QC_DASHBOARD_ROLE from the parent via spawn, so the worker
marker is what distinguishes "off-proc read (fine)" from "dashboard-thread read
(forbidden)".
"""

from __future__ import annotations

import os

_ROLE = "QC_DASHBOARD_ROLE"
_WORKER = "QC_OFFPROC_WORKER"


def mark_worker() -> None:
    """Call once at the top of an off-proc reader worker (child process) to mark
    this process as an allowed reader."""
    os.environ[_WORKER] = "1"


def in_dashboard_owned_thread() -> bool:
    """True when running in the dashboard process on a thread it owns (i.e. NOT
    inside an off-proc reader worker)."""
    return bool(os.environ.get(_ROLE)) and not os.environ.get(_WORKER)


def require_offproc(what: str) -> None:
    """Raise if *what* (a raw share read / video decode) is about to run on a
    dashboard-owned thread. No-op in the daemon and in off-proc workers."""
    if in_dashboard_owned_thread():
        raise RuntimeError(
            f"{what} called on a dashboard-owned thread. In the dashboard "
            f"process this MUST go through the off-process reader "
            f"(src/utils/chunk_reader.read_chunk_offproc or "
            f"snapshot_reader.decode_snapshot_offproc) so the slow share read / "
            f"cv2 decode never holds the dashboard GIL. See "
            f"src/utils/offproc_guard.py and tests/test_dashboard_offproc_invariant.py.")
