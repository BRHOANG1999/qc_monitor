"""read_chunk_offproc must SURFACE the real read error (e.g. a missing file
after a drive remount), never mask it with the require_offproc guard by falling
back to an in-process load_mat on a dashboard-owned thread. Plus the LFP Browser
resolves a stale recording path to its live drive location before reading.

Run: pytest tests/test_offproc_fallback.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import chunk_reader as cr               # noqa: E402

_BS = chr(92)


class _FakeFuture:
    def __init__(self, exc):
        self._exc = exc

    def result(self, timeout=None):
        raise self._exc


class _FakePool:
    """A pool whose every submit yields a future that raises *exc* (a worker
    that dies / a child load_mat that hits a missing file)."""
    def __init__(self, exc):
        self._exc = exc
        self.submits = 0

    def submit(self, fn, *a):
        self.submits += 1
        return _FakeFuture(self._exc)


def _wire_failing_pool(monkeypatch, exc):
    pool = _FakePool(exc)
    monkeypatch.setattr(cr, "_get_pool", lambda: pool)
    monkeypatch.setattr(cr, "_reset_pool", lambda: None)
    return pool


# ---------------------------------------------------------------- guard --- #

def test_dashboard_thread_surfaces_real_error_not_guard(monkeypatch):
    """On a dashboard-owned thread, a failed off-proc read surfaces the REAL
    exception -- NOT a require_offproc RuntimeError from an in-process fallback."""
    monkeypatch.setenv("QC_DASHBOARD_ROLE", "1")
    monkeypatch.delenv("QC_OFFPROC_WORKER", raising=False)
    real = FileNotFoundError(2, "No such file or directory", r"Z:\gone\rec.mat")
    pool = _wire_failing_pool(monkeypatch, real)

    with pytest.raises(FileNotFoundError) as ei:
        cr.read_chunk_offproc(r"Z:\gone\rec.mat")
    assert "dashboard-owned thread" not in str(ei.value)   # not the guard mask
    assert pool.submits == 2                                # retried once


def test_timeout_is_raised_immediately_without_retry(monkeypatch):
    from concurrent.futures import TimeoutError as _FTimeout
    monkeypatch.setenv("QC_DASHBOARD_ROLE", "1")
    pool = _wire_failing_pool(monkeypatch, _FTimeout())
    with pytest.raises(TimeoutError):
        cr.read_chunk_offproc(r"Z:\slow\rec.mat")
    assert pool.submits == 1                                # no retry on timeout


def test_non_dashboard_caller_falls_back_in_process(monkeypatch):
    """Off a dashboard-owned thread (daemon), an in-process load_mat fallback is
    still allowed when the pool fails -- the guard is a no-op there."""
    monkeypatch.delenv("QC_DASHBOARD_ROLE", raising=False)
    monkeypatch.delenv("QC_OFFPROC_WORKER", raising=False)
    _wire_failing_pool(monkeypatch, RuntimeError("broken pool"))
    import src.utils.mat_loader as ml
    sentinel = object()
    monkeypatch.setattr(ml, "load_mat", lambda p: sentinel)
    assert cr.read_chunk_offproc("whatever.mat") is sentinel


# ------------------------------------------------- lfp_browser resolve --- #

def test_resolve_recording_returns_existing(tmp_path):
    from src.dashboard.tabs import lfp_browser as lb
    real = tmp_path / "rec.mat"
    real.write_bytes(b"x")
    assert lb._resolve_recording(str(real)) == str(real)


def test_resolve_recording_swaps_to_live_drive(tmp_path, monkeypatch):
    from src.dashboard.tabs import lfp_browser as lb
    from src.utils import past_events as pe
    live = tmp_path / "live"
    (live / "database" / "sessA").mkdir(parents=True)
    moved = live / "database" / "sessA" / "rec.mat"
    moved.write_bytes(b"x")
    monkeypatch.setattr(pe, "available_drive_roots",
                        lambda: [str(live) + os.sep])
    # a stale UNC path whose share-relative tail lives under the mounted drive
    stale = "//100.106.104.22/database/sessA/rec.mat"
    assert lb._resolve_recording(stale) == str(moved)


def test_resolve_recording_unresolvable_returns_original(tmp_path, monkeypatch):
    from src.dashboard.tabs import lfp_browser as lb
    from src.utils import past_events as pe
    monkeypatch.setattr(pe, "available_drive_roots",
                        lambda: [str(tmp_path / "empty") + os.sep])
    os.makedirs(tmp_path / "empty", exist_ok=True)
    stale = r"Z:\gone\rec.mat"
    assert lb._resolve_recording(stale) == stale           # honest fall-through


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
