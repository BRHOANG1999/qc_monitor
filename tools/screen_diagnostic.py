"""Per-file diagnostic for the two Mass Analyze screens.

Recomputes BOTH screens FRESH (stim-blanked, bypassing
envelope_peak_cache / envelope_auc_cache) for an animal's pending files,
and prints the per-file envelope-max + AUC-max alongside the pool
verdicts. Use it to answer "why are the envelope and AUC screens flagging
the same files?" -- if the live scan's pool counts disagree with these
fresh numbers, the live caches are stale (restart QCMonitor to evict
them). The min/median/max summary also shows where your thresholds fall
relative to the data, so you can pick a threshold that actually separates
events from baseline.

Run from repo root:
    python tools/screen_diagnostic.py <db> <animal> <cutoff> <auc_thresh> <auc_win> [limit] [csv_out]

Example:
    python tools/screen_diagnostic.py data/monitor.db BCH062 0.018 0.065 5 40 diag.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.db.store import Store  # noqa: E402
from src.utils import stim_blank  # noqa: E402
from src.utils.chunk_cache import get_chunk  # noqa: E402
from src.utils.hilbert_envelope import (  # noqa: E402
    hilbert_envelope_20_200, windowed_auc)
from src.utils.peakseek import peakseek  # noqa: E402
from src.utils.mass_analyze import (  # noqa: E402
    pending_files_for_animal, first_animal_channel_index,
    DEFAULT_MIN_PEAK_DIST_SEC)


def _screen_one(store, file_id, channel, cutoff, auc_thresh, auc_win):
    """Fresh (blanked) envelope + AUC for one file -> a dict of values."""
    row = store.file_row(file_id)
    if not row or not row.get("file_path"):
        return None
    chunk = get_chunk(row["file_path"])
    fs = float(chunk.fs)
    if chunk.signal.ndim != 2 or chunk.signal.shape[1] <= channel:
        return None
    series = chunk.signal[:, channel].astype(np.float32, copy=False)
    if channel not in stim_blank.stim_copy_channels(store, row["session_dir"]):
        times = stim_blank.stim_times_for_file(store, file_id)
        series = stim_blank.blank_series_with_stim_times(
            series, fs, times, stim_blank.BLANK_PRE_MS,
            stim_blank.BLANK_POST_MS)
    env = hilbert_envelope_20_200(series, fs)
    auc = windowed_auc(env, fs, auc_win)
    min_dist = max(1, int(round(fs * DEFAULT_MIN_PEAK_DIST_SEC)))
    env_locs, _ = peakseek(env, minpeakdist=min_dist, minpeakh=cutoff)
    auc_locs, _ = peakseek(auc, minpeakdist=min_dist, minpeakh=auc_thresh)
    return {
        "file_id": file_id,
        "env_max": float(np.nanmax(env)) if env.size else 0.0,
        "env_n": int(len(env_locs)),
        "auc_max": float(np.nanmax(auc)) if auc.size else 0.0,
        "auc_n": int(len(auc_locs)),
    }


def main() -> int:
    if len(sys.argv) < 6:
        print(__doc__)
        return 2
    db, animal = sys.argv[1], sys.argv[2]
    cutoff, auc_thresh, auc_win = (float(sys.argv[3]),
                                    float(sys.argv[4]),
                                    float(sys.argv[5]))
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 9999
    csv_out = sys.argv[7] if len(sys.argv) > 7 else None

    store = Store(db)
    files = pending_files_for_animal(store, animal)[:limit]
    print(f"{animal}: scanning {len(files)} files fresh "
          f"(cutoff={cutoff:g}, auc_thresh={auc_thresh:g}, "
          f"win={auc_win:g})\n")
    print(f"{'file':>8}  {'env_max':>9} {'env>cut':>7}  "
          f"{'auc_max':>9} {'auc>thr':>7}  pools")

    rows = []
    p1 = p2 = p3 = 0
    for f in files:
        ch = first_animal_channel_index(store, f["session_dir"])
        r = _screen_one(store, int(f["file_id"]), ch,
                         cutoff, auc_thresh, auc_win)
        if r is None:
            continue
        env_pos = r["env_n"] > 0
        auc_pos = r["auc_n"] > 0
        p1 += int(env_pos)
        p2 += int(auc_pos)
        p3 += int(env_pos != auc_pos)
        pool = []
        if env_pos:
            pool.append("1")
        if auc_pos:
            pool.append("2")
        if env_pos != auc_pos:
            pool.append("3")
        print(f"{r['file_id']:>8}  {r['env_max']:>9.4f} "
              f"{('Y' if env_pos else '-'):>7}  "
              f"{r['auc_max']:>9.4f} "
              f"{('Y' if auc_pos else '-'):>7}  "
              f"{','.join(pool) or '(none)'}")
        rows.append(r)

    if not rows:
        print("no files scored.")
        return 1

    env_maxes = sorted(r["env_max"] for r in rows)
    auc_maxes = sorted(r["auc_max"] for r in rows)

    def _stats(vals):
        return (min(vals), statistics.median(vals), max(vals))

    e_lo, e_md, e_hi = _stats(env_maxes)
    a_lo, a_md, a_hi = _stats(auc_maxes)
    print(f"\nScored {len(rows)} files.")
    print(f"Pool 1 (envelope): {p1}  ·  Pool 2 (AUC): {p2}  ·  "
          f"Pool 3 (disagree): {p3}")
    print(f"env_max  min/median/max: {e_lo:.4f} / {e_md:.4f} / {e_hi:.4f}"
          f"   (cutoff {cutoff:g})")
    print(f"auc_max  min/median/max: {a_lo:.4f} / {a_md:.4f} / {a_hi:.4f}"
          f"   (auc_thresh {auc_thresh:g})")
    below_env = sum(1 for v in env_maxes if v < cutoff)
    below_auc = sum(1 for v in auc_maxes if v < auc_thresh)
    print(f"Files whose env_max is BELOW the cutoff (would be Pool-1 "
          f"negative): {below_env}/{len(rows)}")
    print(f"Files whose auc_max is BELOW the AUC thresh (would be Pool-2 "
          f"negative): {below_auc}/{len(rows)}")
    if below_auc == 0:
        print("\n>>> Every file's AUC peak exceeds the threshold. Either "
              "the threshold is below the AUC baseline (raise it toward "
              "the median above), or -- if the live scan disagrees with "
              "these fresh numbers -- the live cache is stale (restart "
              "QCMonitor).")

    if csv_out:
        with open(csv_out, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["file_id", "env_max", "env_n",
                                 "auc_max", "auc_n"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
