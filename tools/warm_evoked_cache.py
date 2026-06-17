"""Pre-warm the Chronic Evoked Analyzer cache.

Reads every ``*_evoked.mat`` in the toolkit's evokedOutput folder and
extracts the per-epoch scalars into the sqlite cache the dashboard tab
queries. First run is slow (~0.2 s/file); reruns are incremental (only
files whose mtime changed). Run it after a big batch of new recordings
so the tab opens instantly.

Usage:
    python tools/warm_evoked_cache.py            # all animals
    python tools/warm_evoked_cache.py BCH040     # one or more animals
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
    cache = ChronicEvokedCache()
    animals = argv[1:] or list_animals(cache.evoked_dir)
    if not animals:
        print(f"No evoked files under {cache.evoked_dir}")
        return 1
    print(f"Warming {len(animals)} animal(s): {', '.join(animals)}")
    for animal in animals:
        t0 = time.time()

        def _progress(done, total, _path, _a=animal):
            if done % 50 == 0 or done == total:
                print(f"  {_a}: {done}/{total}", flush=True)

        stats = cache.ensure_animal(animal, progress=_progress)
        print(f"{animal}: {stats['files']} files, "
              f"{stats['built']} (re)built in {time.time() - t0:.0f}s",
              flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
