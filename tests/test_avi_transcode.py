"""Companion-video enumeration + .avi->mp4 transcode cache (no ffmpeg run).

Run with: pytest tests/test_avi_transcode.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import avi_transcode as avi  # noqa: E402
from src.utils import video as vid  # noqa: E402


def _touch(p):
    p.write_bytes(b"x")
    return str(p)


# ---- companion enumeration ------------------------------------------ #

def test_companion_videos_multicam_avi(tmp_path):
    base = tmp_path / "rec___2025_03_03__10_46_58"
    mat = str(base) + ".mat"
    _touch(tmp_path / "rec___2025_03_03__10_46_58_v1.avi")
    _touch(tmp_path / "rec___2025_03_03__10_46_58_v2.avi")
    cams = vid.companion_videos(mat)
    assert [(c["cam"], c["kind"]) for c in cams] == [(1, "avi"), (2, "avi")]


def test_companion_videos_prefers_mp4(tmp_path):
    base = tmp_path / "rec"
    mat = str(base) + ".mat"
    _touch(tmp_path / "rec_v1.avi")
    _touch(tmp_path / "rec_v1.mp4")
    cams = vid.companion_videos(mat)
    assert cams == [{"cam": 1, "path": str(tmp_path / "rec_v1.mp4"),
                     "kind": "mp4"}]


def test_companion_videos_bare_single(tmp_path):
    mat = str(tmp_path / "rec.mat")
    _touch(tmp_path / "rec.avi")
    cams = vid.companion_videos(mat)
    assert cams == [{"cam": 1, "path": str(tmp_path / "rec.avi"),
                     "kind": "avi"}]


def test_companion_videos_none(tmp_path):
    assert vid.companion_videos(str(tmp_path / "rec.mat")) == []


def test_avi_companion_exists(tmp_path):
    mat = str(tmp_path / "rec.mat")
    assert vid.avi_companion_exists(mat) is False
    _touch(tmp_path / "rec_v1.avi")
    assert vid.avi_companion_exists(mat) is True


def test_video_path_for_mat_bare_mp4(tmp_path):
    mat = str(tmp_path / "rec.mat")
    _touch(tmp_path / "rec.mp4")          # not _v1.mp4
    assert vid.video_path_for_mat(mat) == str(tmp_path / "rec.mp4")


# ---- transcode cache (no ffmpeg) ------------------------------------ #

def _cfg(tmp_path):
    return {"video_transcode": {"cache_dir": str(tmp_path / "cache")},
            "event_clip": {"ffmpeg_bin": "ffmpeg"}}


def test_cache_dir_is_absolute(tmp_path, monkeypatch):
    # A relative config path must resolve to an absolute dir, else Flask
    # send_file resolves it against the app root_path and 404s.
    monkeypatch.chdir(tmp_path)
    cfg = {"video_transcode": {"cache_dir": "secrets/avi_mp4_cache"}}
    d = avi.cache_dir(cfg)
    assert os.path.isabs(str(d))
    assert avi.cache_path(_touch(tmp_path / "rec_v1.avi"), cfg).is_absolute()


def test_cache_path_deterministic_and_mtime_sensitive(tmp_path):
    cfg = _cfg(tmp_path)
    avi_path = _touch(tmp_path / "rec_v1.avi")
    p1 = avi.cache_path(avi_path, cfg)
    p2 = avi.cache_path(avi_path, cfg)
    assert p1 == p2 and p1.suffix == ".mp4"
    # A changed file (size) re-keys the cache.
    (tmp_path / "rec_v1.avi").write_bytes(b"xxxxxx")
    assert avi.cache_path(avi_path, cfg) != p1


def test_status_absent_then_ready(tmp_path):
    cfg = _cfg(tmp_path)
    avi_path = _touch(tmp_path / "rec_v1.avi")
    assert avi.status(avi_path, cfg) == "absent"
    assert avi.ready_path(avi_path, cfg) is None
    # Simulate a finished transcode by writing the cache file.
    cp = avi.cache_path(avi_path, cfg)
    cp.write_bytes(b"mp4data")
    assert avi.status(avi_path, cfg) == "ready"
    assert avi.ready_path(avi_path, cfg) == str(cp)


def test_ensure_async_returns_ready(tmp_path):
    cfg = _cfg(tmp_path)
    avi_path = _touch(tmp_path / "rec_v1.avi")
    avi.cache_path(avi_path, cfg).write_bytes(b"mp4data")
    # Already cached -> returns the path without spawning a worker.
    assert avi.ensure_async(avi_path, cfg) == str(
        avi.cache_path(avi_path, cfg))


def test_ensure_async_missing_source(tmp_path):
    cfg = _cfg(tmp_path)
    assert avi.ensure_async(str(tmp_path / "nope.avi"), cfg) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
