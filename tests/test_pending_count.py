"""Queue-count correctness: count_pending_for_animal must be per-(file, animal).

Regression for the "0 pending despite 80 in queue" bug. Every recording is
multi-animal, so a SIBLING animal's terminal review must NOT hide the file from
another animal's pending count. count_pending_for_animal must also agree with
pending_files_for_animal (its own docstring claims they match).

Run with: pytest tests/test_pending_count.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.db.store import Store  # noqa: E402
from src.utils import mass_analyze as ma  # noqa: E402

# Two animals share every recording (BCH900 + BCH901).
_CHANNELS = '["stimCopy", "BCH900SR", "BCH901SLM"]'


def _store(tmp_path, n_files=5):
    s = Store(str(tmp_path / "data" / "m.db"))
    with s.connection() as conn:
        conn.execute(
            """INSERT INTO session_config (session_dir, channel_names,
               eeg_channels, discovered_at) VALUES (?, ?, ?, ?)""",
            ("/sess", _CHANNELS, "[1, 2]", "2026-05-27T00:00:00"))
        for fid in range(1, n_files + 1):
            conn.execute(
                """INSERT INTO processed_files (id, file_path, session_dir,
                   chunk_datetime, has_video)
                   VALUES (?, ?, '/sess', ?, 1)""",
                (fid, f"/sess/{fid}.mat", f"2026_05_27__0{fid}_00_00"))
        conn.commit()
    return s


def test_sibling_review_does_not_hide_files_from_other_animal(tmp_path):
    s = _store(tmp_path, n_files=5)
    # Baseline: both animals see all 5 as pending.
    assert ma.count_pending_for_animal(s, "BCH900") == 5
    assert ma.count_pending_for_animal(s, "BCH901") == 5

    # BCH901 finishes every file (its cage-mate is heavily reviewed).
    for fid in range(1, 6):
        s.mark_review(fid, "rev@lab", "pi_approved", markers=[],
                      animal_id="BCH901")

    # BCH901 now has 0 pending; BCH900 must STILL have all 5 (the pre-fix bug
    # collapsed this to 0 because the exclusion wasn't animal-scoped).
    assert ma.count_pending_for_animal(s, "BCH901") == 0
    assert ma.count_pending_for_animal(s, "BCH900") == 5


def test_count_matches_pending_files_list(tmp_path):
    s = _store(tmp_path, n_files=6)
    s.mark_review(1, "rev@lab", "pi_approved", markers=[], animal_id="BCH901")
    s.mark_review(2, "rev@lab", "no_events", markers=[], animal_id="BCH900")
    for a in ("BCH900", "BCH901"):
        assert (ma.count_pending_for_animal(s, a)
                == len(ma.pending_files_for_animal(s, a))), a


def test_own_review_removes_file_from_pending(tmp_path):
    s = _store(tmp_path, n_files=3)
    s.mark_review(1, "rev@lab", "pi_approved", markers=[], animal_id="BCH900")
    assert ma.count_pending_for_animal(s, "BCH900") == 2


def test_large_animal_does_not_crash_the_post_filter(tmp_path):
    # >4096 files (the old max_iter) for one animal must not crash
    # pending_files_for_animal; it should return them all.
    s = _store(tmp_path, n_files=0)
    with s.connection() as conn:
        for fid in range(1, 4201):
            conn.execute(
                """INSERT INTO processed_files (id, file_path, session_dir,
                   chunk_datetime, has_video)
                   VALUES (?, ?, '/sess', ?, 0)""",
                (fid, f"/sess/{fid}.mat", "2026_05_27__01_00_00"))
        conn.commit()
    files = ma.pending_files_for_animal(s, "BCH900")
    assert len(files) == 4200
    assert ma.count_pending_for_animal(s, "BCH900") == 4200
