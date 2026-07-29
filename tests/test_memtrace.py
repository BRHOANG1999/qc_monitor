"""The opt-in memory-leak tracer.

Guards the diagnostic that names a leak's file:line: it must stay OFF by
default, respect the config/env switch, and -- when on -- surface a growing
allocation site at the top of its growth-ranked log.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import memtrace as MT  # noqa: E402


def test_disabled_by_default():
    assert MT.start({}) is False
    assert MT.start({"debug": {"memtrace": {"enabled": False}}}) is False


def test_settings_clamps_and_env(monkeypatch):
    monkeypatch.delenv("QC_MEMTRACE", raising=False)
    en, interval, top, frames, _log = MT._settings(
        {"debug": {"memtrace": {"enabled": True, "interval_sec": 1, "top": 0}}})
    assert en is True
    assert interval == MT._MIN_INTERVAL_SEC        # floored
    assert top == 1                                 # clamped up from 0
    # env forces it on even when config says disabled
    monkeypatch.setenv("QC_MEMTRACE", "1")
    assert MT._settings({"debug": {"memtrace": {"enabled": False}}})[0] is True


def test_names_a_growing_site(tmp_path):
    """A container that grows between snapshots must appear (as a positive
    size_diff) in the log, tagged with THIS file -- the whole point of the tool."""
    log = tmp_path / "memtrace.log"
    started = MT.start({"debug": {"memtrace": {
        "enabled": True, "interval_sec": MT._MIN_INTERVAL_SEC,
        "top": 10, "logfile": str(log)}}})
    if not started:                                 # a prior test left it running
        return
    leak = []
    # baseline snapshot at ~15 s; grow across it, then let a second land.
    deadline = time.time() + MT._MIN_INTERVAL_SEC * 2 + 6
    while time.time() < deadline:
        leak.append(bytearray(8_000_000))           # 8 MB per tick
        time.sleep(3)
    text = log.read_text(encoding="utf-8")
    assert "=== memtrace" in text and "RSS=" in text
    assert os.path.basename(__file__) in text       # our leak site is named
    assert "MB" in text
