"""Clear cross-file REPLICATED onset drafts (the 1493.289 / 729.993 artifacts).

WHY THIS EXISTS
---------------
Before the scope-key guard shipped (video.py ``_events_scope_matches``), the
Video Review Submit / autosave callbacks read ``video-events-store`` without
checking it had rescoped to the newly-selected recording. A write firing in the
one-callback-round lag after navigation filed the PREVIOUS file's onsets under
the new file -- so a single onset got stamped, verbatim, onto several DIFFERENT
recordings:

  * BCH062  EO=1493.289  on files 4245, 4249, 4263, 4271  (a <1 s burst 2026-07-06)
  * BCH111  EO=729.993   on files 287307, 315809          (2 s apart 2026-07-13)

These EO values are physically impossible across distinct recordings (identical
to 3 decimals), carry no Racine, and are ``draft=true`` -- i.e. no real scored
work. This tool removes ONLY the events whose EO matches the known-bogus value
(per-event; any genuinely-different event on the same row is left untouched),
so the affected recordings open blank for honest re-scoring. It does NOT change
review_state ``status`` -- the files stay wherever they are (Flag / needs_scoring)
and simply lose the phantom onset.

The root cause is fixed separately (commit da3a6b3); this only cleans the data
already written. Dry-run by default; ``--apply`` writes, after dumping every
affected row (full) to a timestamped JSON backup so the change is reversible.
Idempotent: a second run finds nothing (the bogus events are gone).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

# The confirmed replication artifacts: (animal, bogus EO_sec, affected files).
# Each was verified to be the SAME value on multiple distinct recordings.
_TARGETS = [
    ("BCH062", 1493.289, [4245, 4249, 4263, 4271]),
    ("BCH111", 729.993, [287307, 315809]),
]
_EO_TOL = 1e-4                # float match tolerance for the stored EO
_MAX_ROWS = 1_000_000         # NASA Rule 2 runaway backstop.


def _strip_bogus_events(markers_json, eo: float) -> tuple[list, int]:
    """Return (kept_events, n_removed): every event whose EO_sec == *eo*
    (within tolerance) is dropped; all others are preserved verbatim."""
    try:
        evs = json.loads(markers_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return [], 0
    if not isinstance(evs, list):
        return [], 0
    kept, removed = [], 0
    for e in evs:
        eo_v = e.get("EO_sec") if isinstance(e, dict) else None
        if eo_v is not None and abs(float(eo_v) - eo) <= _EO_TOL:
            removed += 1
        else:
            kept.append(e)
    return kept, removed


def plan_cleanup(conn: sqlite3.Connection) -> list[dict]:
    """Every review_state row that still carries a bogus replicated onset, with
    the events it would keep. One dict per affected row."""
    conn.row_factory = sqlite3.Row
    out: list[dict] = []
    for animal, eo, files in _TARGETS:
        ph = ",".join("?" * len(files))
        rows = conn.execute(
            f"""SELECT id, file_id, status, markers_json FROM review_state
                WHERE animal_id=? AND file_id IN ({ph})
                  AND markers_json IS NOT NULL
                  AND markers_json NOT IN ('', '[]')""",
            [animal] + files).fetchall()
        assert len(rows) < _MAX_ROWS, "review_state scan runaway"
        for r in rows:
            kept, removed = _strip_bogus_events(r["markers_json"], eo)
            if removed:
                out.append({"row_id": r["id"], "file_id": r["file_id"],
                            "animal": animal, "status": r["status"],
                            "bogus_eo": eo, "removed": removed,
                            "kept_events": kept})
    return out


def _backup(conn, row_ids, path) -> int:
    conn.row_factory = sqlite3.Row
    ph = ",".join("?" * len(row_ids))
    rows = conn.execute(
        f"SELECT * FROM review_state WHERE id IN ({ph})", row_ids
    ).fetchall() if row_ids else []
    payload = {"at": datetime.now().isoformat(), "targets": _TARGETS,
               "review_state_rows": [dict(r) for r in rows]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return len(rows)


def apply_cleanup(conn: sqlite3.Connection, plan: list[dict], now: str) -> int:
    """Rewrite each affected row's markers_json to the kept-events list. Call
    inside a transaction. Returns the number of rows updated."""
    n = 0
    for p in plan:
        conn.execute(
            "UPDATE review_state SET markers_json=?, updated_at=? WHERE id=?",
            (json.dumps(p["kept_events"]), now, p["row_id"]))
        n += 1
    return n


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default: dry-run report only)")
    ap.add_argument("--backup-dir", default="data/backups")
    a = ap.parse_args(argv)

    conn = sqlite3.connect(a.db)
    try:
        plan = plan_cleanup(conn)
        print(f"Replicated-onset rows to clean: {len(plan)}")
        for p in plan:
            print(f"   row {p['row_id']:6}  file {p['file_id']:7}  "
                  f"{p['animal']}  {p['status']:14}  EO={p['bogus_eo']}  "
                  f"remove={p['removed']}  keep={len(p['kept_events'])}")
        if not plan:
            print("\nNothing to clean (already done, or DB has no artifacts).")
            return 0
        if not a.apply:
            print("\nDRY-RUN (no writes). Re-run with --apply to execute.")
            return 0
        os.makedirs(a.backup_dir, exist_ok=True)
        bpath = os.path.join(
            a.backup_dir,
            f"clear_replicated_onsets_{datetime.now():%Y%m%d_%H%M%S}.json")
        nb = _backup(conn, [p["row_id"] for p in plan], bpath)
        print(f"\n   backed up {nb} review_state rows -> {bpath}")
        now = datetime.now().isoformat()
        with conn:                       # transaction
            n = apply_cleanup(conn, plan, now)
        print(f"   APPLIED: {n} rows cleaned.")
        after = plan_cleanup(conn)
        print(f"   VERIFY: replicated-onset rows remaining = {len(after)} "
              f"(expected 0).")
        return 0 if not after else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(_main())
