"""matlab_bridge process-tree discipline.

Regression tests for the incident where a hung MATLAB permanently wedged the
dispatcher: ``matlab -batch`` is a thin launcher that spawns the real engine,
so killing only the direct child orphaned a multi-GB engine, and capturing via
PIPEs let that orphan hold the write handle open so ``communicate()`` blocked
forever (making the ``TimeoutExpired`` handler dead code on Windows).

These build a fake two-process "matlab" (launcher -> long-lived grandchild)
and assert the timeout is both RAISED and TOTAL.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import matlab_bridge as MB  # noqa: E402

psutil = pytest.importorskip("psutil")


def _write_fake_matlab(tmp_path):
    """A shim invocable as ``<shim> -batch <script>`` that spawns a grandchild
    which outlives its parent unless the whole tree is killed."""
    pid_file = tmp_path / "gc_pid.txt"
    launcher = tmp_path / "fake_matlab.py"
    launcher.write_text(
        "import subprocess, sys, time, os\n"
        "gc = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(600)'])\n"
        f"open(r'{pid_file}', 'w').write(str(gc.pid))\n"
        "time.sleep(600)\n", encoding="utf-8")
    if os.name == "nt":
        shim = tmp_path / "fakematlab.cmd"
        shim.write_text(f'@echo off\r\n"{sys.executable}" "{launcher}"\r\n',
                        encoding="utf-8")
    else:
        shim = tmp_path / "fakematlab.sh"
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{launcher}"\n',
                        encoding="utf-8")
        shim.chmod(0o755)
    return str(shim), pid_file


def _read_pid(pid_file, deadline=15.0):
    end = time.time() + deadline
    while time.time() < end:
        try:
            txt = pid_file.read_text().strip()
            if txt:
                return int(txt)
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    return None


def _dead(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return True
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


def test_timeout_raises_instead_of_deadlocking(tmp_path):
    """Before the fix this blocked forever inside communicate() and the
    TimeoutExpired handler was unreachable."""
    shim, pid_file = _write_fake_matlab(tmp_path)
    t0 = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        MB.run_matlab_batch("noop", timeout=3, matlab_exe=shim)
    elapsed = time.time() - t0
    # Must return promptly, not hang. Generous bound for CI noise.
    assert elapsed < 3 + MB._KILL_GRACE_SEC + 20
    gc_pid = _read_pid(pid_file)
    if gc_pid is not None:
        for _ in range(30):                       # allow the kill to land
            if _dead(gc_pid):
                break
            time.sleep(0.5)


def test_timeout_kills_the_whole_tree_not_just_the_launcher(tmp_path):
    """The grandchild (stand-in for bin/win64/MATLAB.exe) must not survive."""
    shim, pid_file = _write_fake_matlab(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        MB.run_matlab_batch("noop", timeout=3, matlab_exe=shim)
    gc_pid = _read_pid(pid_file)
    assert gc_pid is not None, "fake engine never recorded its pid"
    for _ in range(40):
        if _dead(gc_pid):
            break
        time.sleep(0.5)
    assert _dead(gc_pid), (
        f"grandchild {gc_pid} survived the timeout -- the process tree was "
        "not killed (this is the orphaned-MATLAB-engine leak)")


def test_capture_is_bounded(tmp_path):
    """Output is tailed from a file, so a chatty run can't balloon the heap."""
    assert MB._MAX_CAPTURE_BYTES <= 1_000_000
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (MB._MAX_CAPTURE_BYTES * 3))
    out = MB._tail_text(str(big))
    assert len(out) <= MB._MAX_CAPTURE_BYTES
    assert MB._tail_text(str(tmp_path / "missing.txt")) == ""
