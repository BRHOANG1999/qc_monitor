"""FileWatcher: filename pattern + scan behaviour.

These lock the recorder filename conventions (the ___ vs __ separator that
silently broke ingest) and the format-drift canary.

Run: pytest tests/test_watcher.py -q
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.watcher import FileWatcher, MAT_PATTERN  # noqa: E402


# Both real-world conventions must match + yield the same captured datetime.
_OLD = "baseline__stimCopy_BCH062SR_BCH110SR___2026_06_29__14_41_46.mat"
_NEW = "stimStability__stimCopy_BCH110SR_BCH061SLM__2026_07_01__19_09_29.mat"


def test_pattern_matches_two_and_three_underscores():
    m_old = MAT_PATTERN.search(_OLD)
    m_new = MAT_PATTERN.search(_NEW)
    assert m_old and m_old.group(1) == "2026_06_29__14_41_46"
    assert m_new and m_new.group(1) == "2026_07_01__19_09_29"


def test_pattern_rejects_non_recording_mat():
    assert MAT_PATTERN.search("recorder_settings.mat") is None
    assert MAT_PATTERN.search("artifact_baseline.mat") is None


def _write(path: str, data: bytes = b"x"):
    with open(path, "wb") as f:
        f.write(data)


def _age(path: str, seconds: int):
    """Backdate a file's mtime so it clears the age gate."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_scan_registers_new_recording(tmp_path):
    session = tmp_path / "session1"
    session.mkdir()
    mat = session / _NEW
    _write(str(mat))
    _age(str(mat), 3600)                       # old enough to clear age gate
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    found = w.scan(known_paths=set())
    assert [os.path.basename(f.path) for f in found] == [_NEW]
    assert found[0].chunk_datetime == "2026_07_01__19_09_29"
    assert found[0].session_name == "session1"


def test_scan_skips_known_and_too_new(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    known = session / _OLD
    fresh = session / _NEW
    _write(str(known)); _age(str(known), 3600)
    _write(str(fresh))                          # just written -> within age gate
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    found = w.scan(known_paths={str(known)})
    # known one excluded; fresh one age-gated -> nothing new
    assert found == []


def test_scan_recurses_into_month_folders(tmp_path):
    month = tmp_path / "JULY_2026"
    session = month / "sess"
    session.mkdir(parents=True)
    mat = session / _NEW
    _write(str(mat)); _age(str(mat), 3600)
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    found = w.scan(known_paths=set())
    assert len(found) == 1


def test_recurses_past_stray_mat_in_parent(tmp_path):
    """A non-recording .mat in a PARENT folder (recorder_settings.mat,
    artifact_baseline.mat) must NOT stop recursion into the session subfolder
    -- the exact regression that stopped ingest mid-experiment."""
    parent = tmp_path / "database"
    session = parent / "stimStability__x_BCH110SR_BCH061SLM"
    session.mkdir(parents=True)
    # Stray non-recording .mat right next to the session folder.
    _write(str(parent / "recorder_settings.mat"))
    _write(str(parent / "artifact_baseline.mat"))
    rec = session / _NEW
    _write(str(rec)); _age(str(rec), 3600)
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    found = w.scan(known_paths=set())
    assert [os.path.basename(f.path) for f in found] == [_NEW]


def test_skip_canary_counts_recordinglike_nonmatches(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    # A recording-like name with a SINGLE-underscore separator (a hypothetical
    # future format drift) -> has a timestamp but MAT_PATTERN (which needs
    # 2-3 underscores before it) rejects it.
    drift = session / "baseline_2026_07_01__20_00_00.mat"
    _write(str(drift)); _age(str(drift), 3600)
    _write(str(session / "recorder_settings.mat"))   # not recording-like
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    found = w.scan(known_paths=set())
    assert found == []                              # nothing registered
    assert w.last_scan_stats["n_skipped_recordinglike"] == 1
    assert any("2026_07_01__20_00_00" in s
               for s in w.last_scan_stats["skipped_samples"])


def test_scan_stats_track_newest_recording_by_embedded_dt(tmp_path):
    """Newest recording is tracked by its EMBEDDED timestamp, not file mtime
    -- so a daily backup re-copying old files (fresh mtimes) doesn't look new."""
    session = tmp_path / "s"
    session.mkdir()
    older = session / "baseline__a_BCH110SR___2026_07_01__08_00_00.mat"
    newer = session / _NEW  # ...2026_07_01__19_09_29.mat
    _write(str(older)); _age(str(older), 3600)
    # The NEWER-by-recording-time file is re-copied "now" (fresh mtime) --
    # mtime order is opposite recording order, so embedded dt must win.
    _write(str(newer)); _age(str(newer), 120)
    w = FileWatcher([str(tmp_path)], min_file_age_sec=60)
    w.scan(known_paths=set())
    assert w.last_scan_stats["newest_chunk_dt"] == "2026_07_01__19_09_29"
    assert w.last_scan_stats["n_scanned"] == 2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
