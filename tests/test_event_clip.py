"""Track B verification: event_clip.resolve_clip_segments.

Covers the four window-vs-boundary cases:
* middle of file (1 segment)
* EO near start (2 segments: prev tail + main head)
* BB near end (2 segments: main tail + next head)
* event spans both boundaries (3 segments)

Uses a tiny in-memory fake store -- no SQLite -- so the
test stays under 1 second.

Run with: pytest tests/test_event_clip.py -q
Or directly: python -m tests.test_event_clip
"""

from __future__ import annotations

import unittest

from src.utils.event_clip import (
    ClipSpec, Segment, resolve_clip_segments, spec_hash,
)


class _FakeStore:
    """Minimal store stub: only file_row and
    get_adjacent_files are needed."""

    def __init__(self, files: dict[int, dict],
                  prev_of: dict[int, int],
                  next_of: dict[int, int]):
        self._files = files
        self._prev = prev_of
        self._next = next_of

    def file_row(self, file_id: int) -> dict | None:
        return self._files.get(int(file_id))

    def get_adjacent_files(self, file_id: int,
                              direction: str) -> dict | None:
        if direction == "prev":
            other = self._prev.get(int(file_id))
        elif direction == "next":
            other = self._next.get(int(file_id))
        else:
            return None
        return self._files.get(int(other)) if other else None


def _three_files(dur: float = 3600.0):
    files = {
        1: {"id": 1, "file_path": "/fake/a.mat",
             "duration_sec": dur,
             "sampling_rate": 20000.0},
        2: {"id": 2, "file_path": "/fake/b.mat",
             "duration_sec": dur,
             "sampling_rate": 20000.0},
        3: {"id": 3, "file_path": "/fake/c.mat",
             "duration_sec": dur,
             "sampling_rate": 20000.0},
    }
    return _FakeStore(files=files,
                       prev_of={2: 1, 3: 2},
                       next_of={1: 2, 2: 3})


class MidFileEvent(unittest.TestCase):

    def test_one_segment(self):
        store = _three_files()
        spec = ClipSpec(file_id=2, eo_sec=1500.0, bb_sec=1530.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].file_id, 2)
        self.assertAlmostEqual(segs[0].start_sec, 1200.0)
        self.assertAlmostEqual(segs[0].end_sec, 1830.0)


class EoNearStart(unittest.TestCase):

    def test_two_segments_prev_tail_plus_main_head(self):
        store = _three_files()
        # EO at 100 s, BB at 200 s; pre=300 -> needs 200 s of
        # prev's tail.
        spec = ClipSpec(file_id=2, eo_sec=100.0, bb_sec=200.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 2)
        # First segment is the previous file's tail.
        self.assertEqual(segs[0].file_id, 1)
        self.assertAlmostEqual(segs[0].start_sec, 3400.0)
        self.assertAlmostEqual(segs[0].end_sec, 3600.0)
        # Second is current file from 0 to BB+post=500.
        self.assertEqual(segs[1].file_id, 2)
        self.assertAlmostEqual(segs[1].start_sec, 0.0)
        self.assertAlmostEqual(segs[1].end_sec, 500.0)


class BbNearEnd(unittest.TestCase):

    def test_two_segments_main_tail_plus_next_head(self):
        store = _three_files()
        # EO at 3500 s, BB at 3550 s; post=300 -> needs 250 s
        # of next's head.
        spec = ClipSpec(file_id=2, eo_sec=3500.0, bb_sec=3550.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].file_id, 2)
        self.assertAlmostEqual(segs[0].start_sec, 3200.0)
        self.assertAlmostEqual(segs[0].end_sec, 3600.0)
        self.assertEqual(segs[1].file_id, 3)
        self.assertAlmostEqual(segs[1].start_sec, 0.0)
        self.assertAlmostEqual(segs[1].end_sec, 250.0)


class SpansBothBoundaries(unittest.TestCase):

    def test_three_segments(self):
        # Very short file so the EO-pre/BB+post window
        # straddles both boundaries.
        store = _three_files(dur=100.0)
        spec = ClipSpec(file_id=2, eo_sec=10.0, bb_sec=90.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0].file_id, 1)  # prev tail
        self.assertEqual(segs[1].file_id, 2)  # main full
        self.assertAlmostEqual(segs[1].start_sec, 0.0)
        self.assertAlmostEqual(segs[1].end_sec, 100.0)
        self.assertEqual(segs[2].file_id, 3)  # next head


class MissingAdjacent(unittest.TestCase):

    def test_no_prev_clips_to_zero(self):
        store = _three_files()
        # file_id=1 has no prev; the window clips to start.
        spec = ClipSpec(file_id=1, eo_sec=100.0, bb_sec=200.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].file_id, 1)
        self.assertAlmostEqual(segs[0].start_sec, 0.0)
        self.assertAlmostEqual(segs[0].end_sec, 500.0)

    def test_no_next_clips_to_dur(self):
        store = _three_files()
        # file_id=3 has no next; window clips to end.
        spec = ClipSpec(file_id=3, eo_sec=3500.0, bb_sec=3550.0)
        segs = resolve_clip_segments(spec, store)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].file_id, 3)
        self.assertAlmostEqual(segs[0].start_sec, 3200.0)
        self.assertAlmostEqual(segs[0].end_sec, 3600.0)


class SpecHashStability(unittest.TestCase):

    def test_same_spec_same_segments_same_hash(self):
        store = _three_files()
        spec = ClipSpec(file_id=2, eo_sec=1500.0, bb_sec=1530.0)
        s1 = resolve_clip_segments(spec, store)
        s2 = resolve_clip_segments(spec, store)
        self.assertEqual(spec_hash(spec, s1), spec_hash(spec, s2))

    def test_different_eo_different_hash(self):
        store = _three_files()
        s1 = resolve_clip_segments(
            ClipSpec(file_id=2, eo_sec=1500.0, bb_sec=1530.0),
            store)
        s2 = resolve_clip_segments(
            ClipSpec(file_id=2, eo_sec=1501.0, bb_sec=1530.0),
            store)
        self.assertNotEqual(
            spec_hash(ClipSpec(2, 1500.0, 1530.0), s1),
            spec_hash(ClipSpec(2, 1501.0, 1530.0), s2),
        )


if __name__ == "__main__":
    unittest.main()
