"""Pre-build the Chronic Evoked Analyzer feature sidecars.

Reads every ``*_evoked.mat`` in the toolkit's evokedOutput folder, computes
the per-epoch feature set, and writes a per-(recording, animal) JSON sidecar
next to each ``.mat`` -- the same sidecars the dashboard AND the evoked-metric
figure export read (no database). Reruns are incremental: a file whose sidecar
is already present and fresh is skipped.

Each file is an independent HDF5 trace read + feature extraction (~15 s), so
the work is embarrassingly parallel -- it runs across a process pool by
default. Serially the full backlog is ~20 h; parallel it is a few hours.

The cheap vectorized features are always computed. Pass ``--expensive`` to
also compute the heavy per-epoch fits (recovery tau/slope, template
correlation, PCA recon error, AC width, exp-fit A); that forces a full
rebuild so the backlog gets the new columns.

Usage:
    python tools/warm_evoked_cache.py                  # all animals, parallel
    python tools/warm_evoked_cache.py BCH040 BCH062    # specific animals
    python tools/warm_evoked_cache.py --jobs 4         # cap the pool
    python tools/warm_evoked_cache.py --jobs 1         # serial (debug)
    python tools/warm_evoked_cache.py --expensive      # full feature set
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.evoked_output import (  # noqa: E402
    DEFAULT_EVOKED_DIR, animals_in_filename, compute_feature_rows,
    list_animals, list_evoked_files, read_feature_sidecar,
    write_feature_sidecar)

_MAX_WORKERS = 8            # memory: each worker holds one recording's traces


def _build_one(task):
    """Worker: (mat_path, animal, expensive) -> 1 written / 0 skipped / -1 failed.
    Each worker writes a DIFFERENT sidecar (atomic os.replace), so no contention."""
    fp, animal, expensive = task
    try:
        if not expensive and read_feature_sidecar(fp, animal) is not None:
            return 0                                   # already fresh
        rows = compute_feature_rows(fp, animal, None, expensive)
        if rows:
            write_feature_sidecar(fp, animal, rows)
            return 1
        return 0
    except Exception:                                  # noqa: BLE001
        return -1                                      # count it, never abort


def _default_jobs() -> int:
    try:
        return max(1, min(_MAX_WORKERS, (os.cpu_count() or 2) - 2))
    except Exception:                                  # noqa: BLE001
        return 2


def main(argv: list[str]) -> int:
    args = argv[1:]
    expensive = "--expensive" in args
    jobs = _default_jobs()
    if "--jobs" in args:
        try:
            jobs = max(1, int(args[args.index("--jobs") + 1]))
        except (IndexError, ValueError):
            pass
    animals = [a for a in args if not a.startswith("-") and a.isalnum()
               and not a.isdigit()]
    evoked_dir = DEFAULT_EVOKED_DIR
    files = list_evoked_files(evoked_dir)
    animals = animals or list_animals(evoked_dir)
    if not animals or not files:
        print(f"No evoked files under {evoked_dir}")
        return 1

    # Build the work list up front, skipping sidecars that are already fresh,
    # so the progress line reflects REAL remaining work (not a no-op scan).
    print(f"scanning {len(files)} recordings for "
          f"{len(animals)} animal(s): {', '.join(animals)} ...", flush=True)
    want = set(animals)
    tasks = []
    for fp in files:
        for a in (animals_in_filename(fp) & want):
            if not expensive and read_feature_sidecar(fp, a) is not None:
                continue
            tasks.append((fp, a, expensive))
    total = len(tasks)
    mode = "FULL (incl. expensive)" if expensive else "cheap features"
    print(f"{total} sidecar(s) to build [{mode}] with {jobs} worker(s)",
          flush=True)
    if not total:
        print("Nothing to do -- all sidecars fresh.")
        return 0

    done = written = failed = 0
    t0 = time.time()
    with mp.Pool(processes=jobs) as pool:
        for r in pool.imap_unordered(_build_one, tasks, chunksize=2):
            done += 1
            if r == 1:
                written += 1
            elif r == -1:
                failed += 1
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0.0
                eta_min = ((total - done) / rate / 60.0) if rate else 0.0
                print(f"  {done}/{total}  written={written} failed={failed}  "
                      f"{rate * 60:.0f}/min  ETA {eta_min:.0f} min", flush=True)
    el = time.time() - t0
    print(f"Done: {written} written, {failed} failed, in {el / 60:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
