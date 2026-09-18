"""Repair processed_files.session_dir rows desynced from session_config.

BACKGROUND. ``session_dir`` is the *logical* session key that JOINs
``processed_files`` to ``session_config`` (and the review / evoked / summary /
seizure-card / mass-analyze-pool queries). ``session_config.session_dir`` is
stored canonically -- usually the UNC network path
``//100.106.104.22/bhz/database/<MONTH_YEAR>/<session>`` (plus a few historical
Z:/U: rows). It is NEVER used as a filesystem read path (reads go through
``file_path``).

A path-heal bug (fixed in store.relocate_file_path) rewrote
``processed_files.session_dir`` to the *physical* healed drive
(``H:\\database\\<MONTH_YEAR>\\<session>``) when a remounted drive re-lettered.
That desynced it from the canonical ``session_config`` key, so every healed
recording silently dropped out of the seizure card and the pools.

THE REPAIR. The session TAIL (``<MONTH_YEAR>\\<session>``, or just the session
folder ``<session>``) is stable across drives; only the root/prefix changes. So
for every processed_files row whose ``session_dir`` has NO matching
``session_config`` row, we resolve its canonical ``session_config.session_dir``
by matching the tail and re-key ``session_dir`` back to it. ``file_path`` (the
physical read path, already healed) is left untouched.

Dry-run by default; ``--apply`` persists. Idempotent -- already-matched rows are
skipped.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

_BS = chr(92)


def _norm(p: str | None) -> str:
    """Backslash-normalized, trailing-separator-stripped path string."""
    return (p or "").replace("/", _BS).rstrip(_BS)


def _segs(p: str | None) -> list[str]:
    return [s for s in _norm(p).split(_BS) if s]


def _tail2(p: str | None) -> str:
    """Last two path segments, case-folded -- ``<MONTH_YEAR>\\<session>``. This
    disambiguates a session folder name reused under different months/roots."""
    s = _segs(p)
    return _BS.join(s[-2:]).casefold() if len(s) >= 2 else _base(p)


def _base(p: str | None) -> str:
    """Last path segment (the session folder), case-folded."""
    s = _segs(p)
    return s[-1].casefold() if s else ""


def _is_unc(p: str | None) -> bool:
    q = (p or "").lstrip()
    return q.startswith("//") or q.startswith(_BS + _BS)


@dataclass
class ResyncSummary:
    sc_rows: int = 0                 # session_config rows indexed
    scanned: int = 0                 # orphaned processed_files rows examined
    rekeyed: int = 0                 # resolved to a canonical session_config
    ambiguous: int = 0               # >1 canonical candidate, no UNC tiebreak
    unmatched: int = 0              # no canonical session_config for the tail
    by_src: dict = field(default_factory=dict)   # source prefix -> rekeyed count

    def _bump(self, root: str) -> None:
        self.by_src[root] = self.by_src.get(root, 0) + 1


def _src_root(p: str | None) -> str:
    q = _norm(p)
    if len(q) >= 2 and q[1] == ":":
        return q[:2].upper()
    if _is_unc(p):
        segs = _segs(p)
        return "//" + segs[0] if segs else "//?"
    return "(other)"


class _Canon:
    """A candidate session_config key + whether it carries usable channel
    metadata (channel_names AND eeg_channels non-null -- required for a session
    to appear in the seizure-card animal universe)."""
    __slots__ = ("sd", "usable")

    def __init__(self, sd: str, usable: bool):
        self.sd = sd
        self.usable = usable


def _build_canonical_index(store) -> tuple[dict, dict]:
    """Map session tail -> [_Canon, ...] by the 2-segment tail and (fallback) by
    basename, from the canonical ``session_config`` keys."""
    by_tail2: dict[str, list[_Canon]] = defaultdict(list)
    by_base: dict[str, list[_Canon]] = defaultdict(list)
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_dir, channel_names, eeg_channels "
            "FROM session_config WHERE session_dir IS NOT NULL").fetchall()
    for r in rows:
        c = _Canon(r["session_dir"],
                   bool(r["channel_names"]) and bool(r["eeg_channels"]))
        by_tail2[_tail2(c.sd)].append(c)
        by_base[_base(c.sd)].append(c)
    return by_tail2, by_base


def _pick_same_session(pool: list, dir_counts: dict | None) -> "_Canon":
    """Deterministically choose among candidates that are all the SAME logical
    session (they share a 2-segment tail, differing only by prefix / slash
    style). Prefer: a key with usable channel metadata, then the representation
    the session's other files already use most, then a forward-slash UNC, then
    the shortest / lexicographically-smallest key."""
    counts = dir_counts or {}

    def rank(c: "_Canon"):
        return (c.usable, counts.get(c.sd, 0), c.sd.startswith("//"),
                -len(c.sd), c.sd)
    return max(pool, key=rank)


def _canonical_for(sd: str, by_tail2: dict, by_base: dict,
                   dir_counts: dict | None = None) -> tuple[str | None, str]:
    """Resolve the canonical session_config.session_dir for a desynced *sd*.
    Returns (canonical_or_None, reason). Order: unique 2-seg tail, unique
    basename, then -- when every candidate is the SAME session under different
    prefixes -- a deterministic same-session tiebreak. Only genuinely-distinct
    sessions sharing a name remain ambiguous."""
    cand = by_tail2.get(_tail2(sd), [])
    if len(cand) == 1:
        return cand[0].sd, "tail2"
    cand_b = by_base.get(_base(sd), [])
    if len(cand_b) == 1:
        return cand_b[0].sd, "basename"
    pool = cand or cand_b
    if not pool:
        return None, "unmatched"
    if len({_tail2(c.sd) for c in pool}) == 1:           # same session, diff prefix
        return _pick_same_session(pool, dir_counts).sd, "same-session"
    return None, "ambiguous"                             # distinct sessions, same name


def _orphaned_rows(store, path_like: str | None, limit: int | None) -> list:
    """processed_files rows whose session_dir has NO session_config match."""
    sql = ("SELECT pf.id AS id, pf.session_dir AS sd "
           "FROM processed_files pf "
           "LEFT JOIN session_config sc ON sc.session_dir = pf.session_dir "
           "WHERE sc.session_dir IS NULL AND pf.session_dir IS NOT NULL")
    params: list = []
    if path_like:
        sql += " AND pf.session_dir LIKE ?"
        params.append(path_like)
    sql += " ORDER BY pf.id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with store.connection() as conn:
        return conn.execute(sql, params).fetchall()


def _pf_dir_counts(store) -> dict:
    """How many processed_files rows currently reference each session_dir -- so
    an orphan with duplicate same-session candidates unifies onto the
    representation the session's other files already use."""
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT session_dir, COUNT(*) AS n FROM processed_files "
            "WHERE session_dir IS NOT NULL GROUP BY session_dir").fetchall()
    return {r["session_dir"]: r["n"] for r in rows}


def resync_session_dirs(store, *, apply: bool = False, path_like: str | None = None,
                        limit: int | None = None, progress=None,
                        log_every: int = 500) -> ResyncSummary:
    """One repair pass. ``apply=False`` (default) classifies without writing."""
    by_tail2, by_base = _build_canonical_index(store)
    dir_counts = _pf_dir_counts(store)
    s = ResyncSummary(sc_rows=sum(len(v) for v in by_base.values()))
    rows = _orphaned_rows(store, path_like, limit)
    total = len(rows)
    max_iter = total + 1                                   # fixed loop bound
    pairs: list[tuple[int, str]] = []
    for i, r in enumerate(rows):
        assert i < max_iter, "resync scan runaway"
        s.scanned += 1
        canonical, reason = _canonical_for(r["sd"], by_tail2, by_base, dir_counts)
        if canonical is not None:
            s.rekeyed += 1
            s._bump(_src_root(r["sd"]))
            pairs.append((r["id"], canonical))
        elif reason == "ambiguous":
            s.ambiguous += 1
        else:
            s.unmatched += 1
        if progress and (i % log_every == 0 or i == total - 1):
            progress(i + 1, total, s)
    if apply and pairs:
        store.resync_processed_session_dirs(pairs)
    return s


def _print_summary(s: ResyncSummary, applied: bool) -> None:
    print("\n=== session_dir resync ===")
    print(f"  session_config keys indexed : {s.sc_rows}")
    print(f"  orphaned rows scanned        : {s.scanned}")
    print(f"  re-keyed to canonical        : {s.rekeyed}"
          f"  {'(APPLIED)' if applied else '(dry-run)'}")
    print(f"  ambiguous (skipped)          : {s.ambiguous}")
    print(f"  unmatched (skipped)          : {s.unmatched}")
    if s.by_src:
        print("  re-keyed by source prefix:")
        for k in sorted(s.by_src):
            print(f"     {k:8s} {s.by_src[k]}")
    if not applied:
        print("\n(dry-run: no writes. re-run with --apply to persist.)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--path-like", default=None,
                    help="only rows whose session_dir LIKE this (e.g. 'H:%%')")
    a = ap.parse_args(argv)
    from src.db.store import Store
    store = Store(a.db)
    mode = "APPLY" if a.apply else "DRY-RUN"
    print(f"[{mode}] resync processed_files.session_dir  db={a.db}")

    def progress(done, total, s):
        print(f"  ... {done}/{total} scanned, {s.rekeyed} re-keyed", flush=True)

    s = resync_session_dirs(store, apply=a.apply, path_like=a.path_like,
                            limit=a.limit, progress=progress)
    _print_summary(s, applied=a.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
