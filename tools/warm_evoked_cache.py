"""Pre-warm the Chronic Evoked Analyzer cache.

Reads every ``*_evoked.mat`` in the toolkit's evokedOutput folder, extracts
the raw evoked traces, computes the per-epoch feature set + per-recording
mean waveform, and stores them in the sqlite cache the dashboard queries.
First run is slow (read + feature extraction per file); reruns are
incremental (only files whose mtime changed). Run it after a big batch of
new recordings so the tab opens instantly.

The cheap vectorized features are always computed. Pass ``--expensive`` to
also compute the heavy per-epoch fits (recovery tau/slope, template
correlation, PCA recon error, AC width, exp-fit A); that forces a full
recompute so the backlog gets the new columns.

Usage:
    python tools/warm_evoked_cache.py                  # all animals, cheap
    python tools/warm_evoked_cache.py BCH040 BCH062    # specific animals
    python tools/warm_evoked_cache.py --expensive      # all, full feature set
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.evoked_output import ChronicEvokedCache, list_animals  # noqa: E402


def main(argv: list[str]) -> int:
    args = argv[1:]
    expensive = "--expensive" in args
    animals = [a for a in args if not a.startswith("-")]
    cache = ChronicEvokedCache(compute_expensive=expensive)
    animals = animals or list_animals(cache.evoked_dir)
    if not animals:
        print(f"No evoked files under {cache.evoked_dir}")
        return 1
    mode = "FULL (incl. expensive)" if expensive else "cheap features"
    print(f"Warming {len(animals)} animal(s) [{mode}]: {', '.join(animals)}")
    for animal in animals:
        t0 = time.time()

        def _progress(done, total, _path, _a=animal):
            if done % 25 == 0 or done == total:
                print(f"  {_a}: {done}/{total}", flush=True)

        stats = cache.ensure_animal(animal, progress=_progress,
                                    force=expensive)
        print(f"{animal}: {stats['files']} files, "
              f"{stats['built']} (re)built in {time.time() - t0:.0f}s",
              flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
