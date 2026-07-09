"""Backfill processed_files.num_samples + duration_sec for rows a shallow index
pass left blank.

WHY THIS EXISTS
---------------
~5266 processed_files rows have chunk_datetime + sampling_rate but a NULL
duration_sec (they were catalogued by a lightweight index pass that never read
the recording's sample count). The pre-ictal lead-up gather -- and the Training
min-duration gate -- decide which chunks overlap a window from
``f_end = f_start + duration_sec``; a row with no duration collapses to
zero-width and is silently skipped, even when the recording sits right in the
window. Filling duration makes those chunks usable again.

REACHABILITY REALITY
--------------------
Only some of the missing-duration recordings are reachable from the pipeline
host. The live share (//100.106.104.22/bhz|database) holds recent data; the bulk
of the blanks are older sessions on mapped drives (Z:, U:) archived off the live
server and not mounted in a background process. This backfill READS ONLY files
it can open, fills what it can, and reports the rest per drive-root -- turning an
opaque "no data" into an explicit, actionable count. It never writes a value it
didn't read from the file, and only fills rows still missing a duration
(idempotent -- safe to re-run).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class BackfillSummary:
    scanned: int = 0
    filled: int = 0
    unreachable: int = 0                 # file not found on disk (archived/offline)
    unreadable: int = 0                  # found but header read / duration failed
    elapsed_sec: float = 0.0
    # root -> {"filled":n, "unreachable":n, "unreadable":n}
    by_root: dict = field(default_factory=dict)

    def _bump(self, root: str, key: str) -> None:
        d = self.by_root.setdefault(
            root, {"filled": 0, "unreachable": 0, "unreadable": 0})
        d[key] += 1


def path_root(p: str | None) -> str:
    """Coarse storage root of a path -- drive letter ('Z:\\') or UNC share
    ('\\\\host\\share') -- for per-location reachability accounting."""
    if not p:
        return "(null)"
    bs = chr(92)
    q = p.replace("/", bs)
    if len(q) >= 2 and q[1] == ":":
        return q[:3].upper()
    if q.startswith(bs + bs):
        parts = q.lstrip(bs).split(bs)
        return bs + bs + bs.join(parts[:2])
    return q.split(bs)[0]


def missing_duration_rows(store, limit: int | None = None,
                          path_like: str | None = None) -> list:
    """Rows needing a duration: have a chunk_datetime + file_path but NULL/<=0
    duration_sec. Optional LIKE filter (e.g. '%100.106.104.22%' for the
    reachable share only) and row cap."""
    sql = ("SELECT id, file_path, sampling_rate FROM processed_files "
           "WHERE chunk_datetime IS NOT NULL AND file_path IS NOT NULL "
           "AND (duration_sec IS NULL OR duration_sec<=0)")
    params: list = []
    if path_like:
        sql += " AND file_path LIKE ?"
        params.append(path_like)
    sql += " ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with store.connection() as conn:
        return conn.execute(sql, params).fetchall()


def backfill_durations(store, apply: bool = False, limit: int | None = None,
                       path_like: str | None = None, progress=None,
                       cancel=None, log_every: int = 50) -> BackfillSummary:
    """Fill num_samples/duration_sec for missing-duration rows by reading each
    recording's shape HEADER-ONLY. ``apply=False`` (default) scans + classifies
    without writing. ``progress(done, total, summary)`` is called every
    *log_every* rows (and at the end) for live feedback. ``cancel()`` -> True
    stops early (partial summary returned)."""
    from src.utils.mat_loader import read_mat_header
    rows = missing_duration_rows(store, limit=limit, path_like=path_like)
    total = len(rows)
    s = BackfillSummary()
    t0 = time.time()
    max_iter = total + 1                        # fixed loop bound (NASA rule 2)
    for i, r in enumerate(rows):
        assert i < max_iter, "backfill scan runaway"
        if cancel and cancel():
            break
        s.scanned += 1
        raw = r["file_path"]
        root = path_root(raw)
        local = raw.replace("/", os.sep)
        if not os.path.exists(local):
            s.unreachable += 1
            s._bump(root, "unreachable")
        else:
            try:
                nsamp, _nchan, fs = read_mat_header(local)
                dur = (nsamp / fs) if fs else 0.0
                if dur and dur > 0:
                    if apply:
                        store.set_file_dimensions(
                            r["id"], num_samples=nsamp,
                            sampling_rate=fs, duration_sec=dur)
                    s.filled += 1
                    s._bump(root, "filled")
                else:
                    s.unreadable += 1
                    s._bump(root, "unreadable")
            except Exception:                   # noqa: BLE001 -- count, never crash
                s.unreadable += 1
                s._bump(root, "unreadable")
        if progress and (i % log_every == 0 or i == total - 1):
            progress(i + 1, total, s)
    s.elapsed_sec = time.time() - t0
    return s


# --------------------------------------------------------------------------- #
#  CLI  --  python -m src.maintenance.duration_backfill [--apply] [--limit N]
# --------------------------------------------------------------------------- #

def _print_summary(s: BackfillSummary, applied: bool) -> None:
    print()  # end the progress line
    print(f"\nscanned={s.scanned}  filled={s.filled}  "
          f"unreachable={s.unreachable}  unreadable={s.unreadable}  "
          f"in {s.elapsed_sec:.0f}s")
    if s.by_root:
        print("by storage root:")
        for root, d in sorted(s.by_root.items(),
                              key=lambda kv: -sum(kv[1].values())):
            print(f"  {root:30} filled={d['filled']:<6} "
                  f"unreachable={d['unreachable']:<6} "
                  f"unreadable={d['unreadable']}")
    if not applied:
        print("\n(dry-run: no writes. re-run with --apply to persist.)")
    elif s.unreachable:
        print(f"\n{s.unreachable} rows are on unreachable/archived drives -- "
              f"mount them (or restore to the live share) and re-run to fill.")


def _main(argv=None) -> int:
    import argparse
    from src.db.store import Store
    ap = argparse.ArgumentParser(
        description="Backfill processed_files.duration_sec from recording headers.")
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--apply", action="store_true",
                    help="persist durations (default is a dry-run)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--path-like", default=None,
                    help="only rows whose file_path LIKE this "
                         "(e.g. '%%100.106.104.22%%' for the reachable share)")
    a = ap.parse_args(argv)
    store = Store(a.db)

    def prog(done, total, s):
        pct = 100 * done / max(1, total)
        print(f"\r[{done}/{total} {pct:3.0f}%] filled={s.filled} "
              f"unreachable={s.unreachable} unreadable={s.unreadable}",
              end="", flush=True)

    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"{mode}: scanning processed_files for missing durations...", flush=True)
    s = backfill_durations(store, apply=a.apply, limit=a.limit,
                           path_like=a.path_like, progress=prog)
    _print_summary(s, applied=a.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
