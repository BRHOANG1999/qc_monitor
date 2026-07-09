"""Repair stale processed_files.file_path entries.

Recordings ingested under a mapped-drive letter (Z:, U:) that is no longer
mounted are physically present on a CURRENTLY-mounted drive under the SAME
session subfolder tree -- e.g. ``Z:\\BHZ\\<session>\\...\\x.mat`` now lives at
``E:\\<session>\\...\\x.mat`` and ``U:\\FEBRUARY_2026\\...`` at
``J:\\FEBRUARY_2026\\...``. The session / subsession / filename TAIL is stable
across drives; only the drive root (and an optional ``BHZ\\`` prefix) changes.

So we relocate by reconstructing the candidate path on each live drive and
stat-ing it -- NO directory walk (the expensive recursive EEGLocator is a
fallback only, for files whose folder layout genuinely differs). Once a row is
repointed, the duration backfill + the pre-ictal lead-up gather can read it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from src.utils.past_events import available_drive_roots

_BS = chr(92)


def _norm(p: str | None) -> str:
    return (p or "").replace("/", _BS)


def _tail_variants(stored_path: str) -> list[str]:
    """Path tail(s) below the drive root that identify the file within a session
    tree: 'X:\\a\\b\\c.mat' -> 'a\\b\\c.mat', plus a variant with a leading
    'BHZ\\' stripped ('Z:\\BHZ\\<session>\\...' layouts)."""
    p = _norm(stored_path)
    body = p[3:] if len(p) > 3 and p[1] == ":" else p.lstrip(_BS)
    outs = [body]
    parts = body.split(_BS)
    if parts and parts[0].upper() == "BHZ":
        outs.append(_BS.join(parts[1:]))
    return outs


def candidate_paths(stored_path: str, drive_roots: list[str]):
    """Drive-swap candidates: each tail joined onto each live drive root, both
    directly and under a 'BHZ\\' subfolder. Cheap stat targets, no walk."""
    for tail in _tail_variants(stored_path):
        for root in drive_roots:                       # root like 'E:\\'
            yield root + tail
            yield root + "BHZ" + _BS + tail


def relocate_one(stored_path: str, drive_roots: list[str],
                 locator=None) -> str | None:
    """First existing drive-swap candidate; else the recursive locator (if
    provided). Returns a real path or None (never the dead stored path)."""
    for cand in candidate_paths(stored_path, drive_roots):
        try:
            if os.path.isfile(cand):
                return cand
        except OSError:
            continue
    if locator is not None:
        p = _norm(stored_path)
        try:
            return locator.resolve(os.path.dirname(p), os.path.basename(p))
        except Exception:                              # noqa: BLE001
            return None
    return None


def _source_root(p: str) -> str:
    q = _norm(p)
    if len(q) >= 2 and q[1] == ":":
        return q[:3].upper()
    return q.split(_BS)[0] if q else "(null)"


@dataclass
class RelocateSummary:
    scanned: int = 0
    already_ok: int = 0            # stored path already on disk
    relocated: int = 0            # found on a live drive
    unresolved: int = 0           # not found anywhere reachable
    elapsed_sec: float = 0.0
    by_root: dict = field(default_factory=dict)        # src root -> counts

    def _bump(self, root: str, key: str) -> None:
        d = self.by_root.setdefault(root, {"relocated": 0, "unresolved": 0})
        d[key] += 1


def dead_path_rows(store, limit: int | None = None,
                   path_like: str | None = None) -> list:
    sql = "SELECT id, file_path FROM processed_files WHERE file_path IS NOT NULL"
    params: list = []
    if path_like:
        sql += " AND file_path LIKE ?"
        params.append(path_like)
    sql += " ORDER BY id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with store.connection() as conn:
        return conn.execute(sql, params).fetchall()


def relocate_missing_paths(store, apply: bool = False, limit: int | None = None,
                           path_like: str | None = None, use_locator: bool = False,
                           config: dict | None = None, progress=None,
                           cancel=None, log_every: int = 200) -> RelocateSummary:
    """Repoint every processed_files row whose file_path no longer exists to the
    live drive it's actually on. ``apply=False`` (default) scans/classifies
    without writing. ``use_locator=True`` adds the recursive EEGLocator fallback
    for stragglers (slower; needs *config* for search roots)."""
    rows = dead_path_rows(store, limit=limit, path_like=path_like)
    drive_roots = available_drive_roots()
    locator = None
    if use_locator:
        from src.utils.past_events import make_locator
        locator = make_locator(store, config or {})
    s = RelocateSummary()
    t0 = time.time()
    total = len(rows)
    max_iter = total + 1                               # fixed loop bound
    for i, r in enumerate(rows):
        assert i < max_iter, "relocate scan runaway"
        if cancel and cancel():
            break
        s.scanned += 1
        stored = r["file_path"]
        if os.path.isfile(_norm(stored)):
            s.already_ok += 1
        else:
            found = relocate_one(stored, drive_roots, locator)
            root = _source_root(stored)
            if found:
                if apply:
                    store.relocate_file_path(r["id"], found)
                s.relocated += 1
                s._bump(root, "relocated")
            else:
                s.unresolved += 1
                s._bump(root, "unresolved")
        if progress and (i % log_every == 0 or i == total - 1):
            progress(i + 1, total, s)
    s.elapsed_sec = time.time() - t0
    return s


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _print_summary(s: RelocateSummary, applied: bool) -> None:
    print()
    print(f"\nscanned={s.scanned}  already_ok={s.already_ok}  "
          f"relocated={s.relocated}  unresolved={s.unresolved}  "
          f"in {s.elapsed_sec:.0f}s")
    if s.by_root:
        print("by source root (dead paths only):")
        for root, d in sorted(s.by_root.items(),
                              key=lambda kv: -sum(kv[1].values())):
            print(f"  {root:30} relocated={d['relocated']:<6} "
                  f"unresolved={d['unresolved']}")
    if not applied:
        print("\n(dry-run: no writes. re-run with --apply to persist path fixes.)")
    elif s.relocated:
        print(f"\n{s.relocated} paths repointed to live drives -- now run the "
              f"duration backfill to fill their durations.")


def _main(argv=None) -> int:
    import argparse
    import yaml
    from src.db.store import Store
    ap = argparse.ArgumentParser(
        description="Repoint stale processed_files paths to live drives.")
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--path-like", default=None,
                    help="only rows whose file_path LIKE this (e.g. 'Z:%%')")
    ap.add_argument("--use-locator", action="store_true",
                    help="add the recursive EEGLocator fallback for stragglers")
    a = ap.parse_args(argv)
    store = Store(a.db)
    config = {}
    if a.use_locator:
        try:
            config = yaml.safe_load(open(a.config, encoding="utf-8")) or {}
        except Exception:                              # noqa: BLE001
            config = {}

    def prog(done, total, s):
        pct = 100 * done / max(1, total)
        print(f"\r[{done}/{total} {pct:3.0f}%] relocated={s.relocated} "
              f"unresolved={s.unresolved} already_ok={s.already_ok}",
              end="", flush=True)

    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"{mode}: scanning processed_files for stale paths "
          f"(live drives: {', '.join(available_drive_roots())})...", flush=True)
    s = relocate_missing_paths(store, apply=a.apply, limit=a.limit,
                               path_like=a.path_like, use_locator=a.use_locator,
                               config=config, progress=prog)
    _print_summary(s, applied=a.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
