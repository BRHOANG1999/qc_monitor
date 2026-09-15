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

from src.utils.past_events import _SKIP_DIRS, available_drive_roots

_BS = chr(92)


def _norm(p: str | None) -> str:
    return (p or "").replace("/", _BS)


def _tail_variants(stored_path: str) -> list[str]:
    """Path tail(s) below the volume root that identify the file within a session
    tree, so it can be reconstructed on whatever drive it now lives on.

    - Drive-letter origin ('X:\\a\\b\\c.mat') -> 'a\\b\\c.mat'.
    - UNC / share origin ('\\\\host\\database\\a\\b\\c.mat') -> two tails: one
      that KEEPS the share segment as a folder ('database\\a\\b\\c.mat', matches
      a drive that mirrors the share, e.g. H:\\database\\...) and one that DROPS
      it ('a\\b\\c.mat', matches a drive whose root holds the sessions directly).
    Each tail also gets a leading-'BHZ\\'-stripped variant ('Z:\\BHZ\\...' layouts)."""
    p = _norm(stored_path)
    if p.startswith(_BS + _BS):                        # UNC: \\host\share\...
        segs = [s for s in p.split(_BS) if s]          # [host, share, a, b, c]
        bodies = []
        if len(segs) >= 2:
            bodies.append(_BS.join(segs[1:]))          # keep share as a folder
        if len(segs) >= 3:
            bodies.append(_BS.join(segs[2:]))          # drop the share segment
        bodies = bodies or [p.lstrip(_BS)]
    else:
        bodies = [p[3:] if len(p) > 3 and p[1] == ":" else p.lstrip(_BS)]
    outs: list[str] = []
    for body in bodies:
        if body and body not in outs:
            outs.append(body)
        parts = body.split(_BS)
        if parts and parts[0].upper() == "BHZ":
            v = _BS.join(parts[1:])
            if v and v not in outs:
                outs.append(v)
    return outs


def _tail_ok(tail: str) -> bool:
    """Reject a reconstructed TAIL that dips into a recycle bin, a system-volume
    folder, or any '$'-prefixed segment -- those hold DELETED copies Windows may
    purge, so relinking a row there would silently rot. Applied to the tail only
    (from the stored path), never the trusted mounted-drive root. Mirrors the
    EEGLocator walk's _SKIP_DIRS / '$' pruning for the cheap-stat path."""
    for seg in tail.split(_BS):
        if seg.startswith("$") or seg.lower() in _SKIP_DIRS:
            return False
    return True


def candidate_paths(stored_path: str, drive_roots: list[str]):
    """Drive-swap candidates: each tail joined onto each live drive root, both
    directly and under a 'BHZ\\' subfolder. Cheap stat targets, no walk. Tails
    that dip into a recycle bin / system folder are dropped (see _tail_ok)."""
    for tail in _tail_variants(stored_path):
        if not _tail_ok(tail):
            continue
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
    relocated: int = 0            # found on a live drive + repointed
    collided: int = 0             # found, but another row already owns that path
    unresolved: int = 0           # not found anywhere reachable
    evoked_scanned: int = 0       # distinct evoked_output_path values checked
    evoked_relocated: int = 0     # evoked paths repointed (rows updated)
    evoked_unresolved: int = 0    # evoked paths not found on any live drive
    elapsed_sec: float = 0.0
    by_root: dict = field(default_factory=dict)        # src root -> counts

    def _bump(self, root: str, key: str) -> None:
        d = self.by_root.setdefault(root, {"relocated": 0, "unresolved": 0})
        d[key] += 1


def dead_path_rows(store, limit: int | None = None,
                   path_like: str | None = None, after_id: int = 0) -> list:
    """processed_files rows with a non-null file_path, id ascending. *after_id*
    (default 0) enables cursor batching -- pass the last id from the previous
    pass to continue where it left off."""
    sql = "SELECT id, file_path FROM processed_files WHERE file_path IS NOT NULL"
    params: list = []
    if after_id:
        sql += " AND id > ?"
        params.append(int(after_id))
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


def dead_evoked_paths(store) -> list[str]:
    """Every distinct non-empty ``evoked_output_path`` across ``matlab_results``
    and ``evoked_summary``. These are usually LOCAL (D:) and healthy; the pass
    skips the ones still on disk, so this is cheap and defensive."""
    sql = ("SELECT DISTINCT evoked_output_path AS p FROM matlab_results "
           "WHERE evoked_output_path IS NOT NULL AND evoked_output_path != '' "
           "UNION "
           "SELECT DISTINCT evoked_output_path AS p FROM evoked_summary "
           "WHERE evoked_output_path IS NOT NULL AND evoked_output_path != ''")
    with store.connection() as conn:
        return [r["p"] for r in conn.execute(sql).fetchall()]


def relocate_missing_evoked_paths(store, apply, drive_roots, locator=None,
                                  cancel=None) -> dict:
    """Repoint evoked_output_path values whose file no longer resolves. Same
    drive-swap + optional recursive-walk machinery as the recording relocation;
    matches by exact stale path across both evoked tables. Returns count dict."""
    paths = dead_evoked_paths(store)
    out = {"scanned": 0, "already_ok": 0, "relocated": 0, "unresolved": 0}
    max_iter = len(paths) + 1
    for i, stored in enumerate(paths):
        assert i < max_iter, "evoked relocate runaway"
        if cancel and cancel():
            break
        out["scanned"] += 1
        if os.path.isfile(_norm(stored)):
            out["already_ok"] += 1
            continue
        found = relocate_one(stored, drive_roots, locator)
        if found:
            if apply:
                store.relocate_evoked_output_path(stored, found)
            out["relocated"] += 1
        else:
            out["unresolved"] += 1
    return out


def run_heal_pass(store, config=None, *, apply: bool = True, after_id: int = 0,
                  rows_per_pass: int | None = None, use_locator: bool = True,
                  include_evoked: bool = True, max_walk_sec: float | None = None,
                  wall_clock_sec: float | None = None, cancel=None, progress=None,
                  log_every: int = 200) -> tuple[RelocateSummary, int]:
    """One BOUNDED self-heal pass, for the background path-heal worker.

    Repoints processed_files rows (id > *after_id*, at most *rows_per_pass*)
    whose file_path no longer resolves, via the cheap per-drive ``candidate_paths``
    stat then the recursive ``EEGLocator`` walk (only if *use_locator*). Honors a
    whole-pass ``wall_clock_sec`` budget and a per-file ``max_walk_sec`` walk cap.
    Optionally relocates the evoked_output_path columns too (*include_evoked*).

    Returns ``(summary, next_after_id)`` -- *next_after_id* is the last row id
    seen, or 0 to WRAP when the batch was exhausted (fewer than *rows_per_pass*
    rows remained), so the caller loops back to the start of the table."""
    rows = dead_path_rows(store, limit=rows_per_pass, after_id=after_id)
    drive_roots = available_drive_roots()
    locator = None
    if use_locator:
        from src.utils.past_events import make_locator
        locator = make_locator(store, config or {}, max_walk_sec=max_walk_sec)
    s = RelocateSummary()
    t0 = time.time()
    deadline = (time.monotonic() + wall_clock_sec) if wall_clock_sec else None
    total = len(rows)
    last_id = after_id
    max_iter = total + 1                               # fixed loop bound
    stopped_early = False
    for i, r in enumerate(rows):
        assert i < max_iter, "heal pass runaway"
        if cancel and cancel():
            stopped_early = True
            break
        if deadline is not None and time.monotonic() > deadline:
            stopped_early = True
            break
        last_id = r["id"]
        s.scanned += 1
        stored = r["file_path"]
        if os.path.isfile(_norm(stored)):
            s.already_ok += 1
        else:
            found = relocate_one(stored, drive_roots, locator)
            root = _source_root(stored)
            if found:
                repointed = store.relocate_file_path(r["id"], found) \
                    if apply else True
                if repointed:
                    s.relocated += 1
                    s._bump(root, "relocated")
                else:
                    s.collided += 1
            else:
                s.unresolved += 1
                s._bump(root, "unresolved")
        if progress and (i % log_every == 0 or i == total - 1):
            progress(i + 1, total, s)
    if include_evoked and not (cancel and cancel()) and \
            not (deadline is not None and time.monotonic() > deadline):
        ev = relocate_missing_evoked_paths(store, apply, drive_roots, locator,
                                           cancel=cancel)
        s.evoked_scanned = ev["scanned"]
        s.evoked_relocated = ev["relocated"]
        s.evoked_unresolved = ev["unresolved"]
    s.elapsed_sec = time.time() - t0
    # Wrap to the table start when we drained the batch (and weren't cut short).
    exhausted = (rows_per_pass is None or total < rows_per_pass)
    next_after_id = 0 if (exhausted and not stopped_early) else last_id
    return s, next_after_id


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
