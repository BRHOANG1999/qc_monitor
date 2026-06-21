"""Pre-build the Chronic Evoked Analyzer feature sidecars.

Reads every ``*_evoked.mat`` in the toolkit's evokedOutput folder, computes
the per-epoch feature set, and writes a per-(recording, animal) JSON sidecar
next to each ``.mat`` -- the same sidecars the dashboard reads (no database).
First run is slow (read + feature extraction per file); reruns are incremental
(files whose sidecar is already present and fresh are skipped). Run it after a
big batch of new recordings so the tab plots instantly instead of building
sidecars on first open.

The cheap vectorized features are always computed. Pass ``--expensive`` to
also compute the heavy per-epoch fits (recovery tau/slope, template
correlation, PCA recon error, AC width, exp-fit A); that forces a full
rebuild so the backlog gets the new columns.

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

from src.utils.evoked_output import (  # noqa: E402
    DEFAULT_EVOKED_DIR, animals_in_filename, compute_feature_rows,
    list_animals, list_evoked_files, read_feature_sidecar,
    write_feature_sidecar)


def _build_animal(animal: str, files: list, expensive: bool) -> int:
    """Build/refresh *animal*'s sidecars; return how many were (re)written."""
    written = 0
    mine = [f for f in files if animal in animals_in_filename(f)]
    n = len(mine)
    for i, fp in enumerate(mine):
        if not expensive and read_feature_sidecar(fp, animal) is not None:
            continue
        rows = compute_feature_rows(fp, animal, None, expensive)
        if rows:
            write_feature_sidecar(fp, animal, rows)
            written += 1
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {animal}: {i + 1}/{n}", flush=True)
    return written


def main(argv: list[str]) -> int:
    args = argv[1:]
    expensive = "--expensive" in args
    animals = [a for a in args if not a.startswith("-")]
    evoked_dir = DEFAULT_EVOKED_DIR
    files = list_evoked_files(evoked_dir)
    animals = animals or list_animals(evoked_dir)
    if not animals or not files:
        print(f"No evoked files under {evoked_dir}")
        return 1
    mode = "FULL (incl. expensive)" if expensive else "cheap features"
    print(f"Building sidecars for {len(animals)} animal(s) [{mode}]: "
          f"{', '.join(animals)}")
    for animal in animals:
        t0 = time.time()
        written = _build_animal(animal, files, expensive)
        print(f"{animal}: {written} sidecar(s) (re)written in "
              f"{time.time() - t0:.0f}s", flush=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
