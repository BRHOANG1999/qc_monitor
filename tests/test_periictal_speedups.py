"""Peri-ictal explorer read-path speedups.

Two network-latency wins on the matrix build:
  * ``persist._sidecar_inputs`` computes the cache signature from ONE
    ``os.scandir`` instead of a ``getmtime()`` per sidecar (~5.5 s -> ~0.02 s
    for BCH111's 568 files, paid on EVERY build incl. cache hits);
  * ``evoked_figures.data.iter_animal_sidecars`` reads sidecars through a thread
    pool (the reads are network-bound + independent), preserving order.

These lock the behaviour the optimisations must keep: identical inputs to the
signature, and identical (ordered, non-empty) rows out of the iterator.

Run: pytest tests/test_periictal_speedups.py -q
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evoked_figures import data as D               # noqa: E402
from src.periictal import persist as P                 # noqa: E402
from src.utils.evoked_output import (feature_sidecar_path,  # noqa: E402
                                     write_feature_sidecar)

_ANIMAL = "BCH111"


def _make_files(evoked_dir, n=6):
    os.makedirs(evoked_dir, exist_ok=True)
    mats = []
    for i in range(n):
        mat = os.path.join(
            evoked_dir,
            f"sess__stimCopy_{_ANIMAL}SR___2026_03_{i + 1:02d}__00_00_00_evoked.mat")
        with open(mat, "wb") as f:
            f.write(b"\x00")
        rows = [{"channel": f"{_ANIMAL}SR", "abs_dt": f"2026-03-{i + 1:02d}T00:59:00",
                 "session": "sess", "line_length": float(i)}]
        write_feature_sidecar(mat, _ANIMAL, rows)
        mats.append(mat)
    return mats


# ------------------------------------------------ scandir signature --- #

def test_sidecar_inputs_matches_getmtime(tmp_path):
    """scandir must yield the SAME (path, mtime) the per-file getmtime did,
    or the signature would change and needlessly invalidate every cache."""
    ed = str(tmp_path / "evoked")
    mats = _make_files(ed)
    got = dict(P._sidecar_inputs(_ANIMAL, ed, "evoked"))
    expect = {feature_sidecar_path(m, _ANIMAL, "evoked"): None for m in mats}
    assert set(got) == set(expect), "same sidecar set"
    for sp, mt in got.items():
        assert mt == os.path.getmtime(sp)          # exact mtime, not approximate


def test_sidecar_inputs_skips_absent_sidecars(tmp_path):
    ed = str(tmp_path / "evoked")
    mats = _make_files(ed, n=4)
    os.remove(feature_sidecar_path(mats[0], _ANIMAL, "evoked"))
    paths = {sp for sp, _ in P._sidecar_inputs(_ANIMAL, ed, "evoked")}
    assert feature_sidecar_path(mats[0], _ANIMAL, "evoked") not in paths
    assert len(paths) == 3


def test_dir_mtimes_empty_dir_is_safe(tmp_path):
    assert P._dir_mtimes(str(tmp_path / "nope")) == {}


# ------------------------------------------- parallel sidecar reads --- #

def test_iter_animal_sidecars_reads_all_in_order(tmp_path):
    """The thread pool must preserve file order and return every non-empty
    sidecar (order feeds the time axis; a dropped file loses a recording)."""
    ed = str(tmp_path / "evoked")
    _make_files(ed, n=6)
    out = list(D.iter_animal_sidecars(_ANIMAL, ed))
    assert len(out) == 6
    lls = [rows[0]["line_length"] for _fp, _sp, rows in out]
    assert lls == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]   # ascending = input order


def test_iter_animal_sidecars_drops_empty(tmp_path):
    ed = str(tmp_path / "evoked")
    mats = _make_files(ed, n=4)
    # An unreadable/absent sidecar reads as None -> skipped, not yielded.
    os.remove(feature_sidecar_path(mats[1], _ANIMAL, "evoked"))
    out = list(D.iter_animal_sidecars(_ANIMAL, ed))
    assert len(out) == 3
