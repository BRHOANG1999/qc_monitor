"""Revert autosave-stranded scoring drafts back to the reviewable Flag pool.

WHY THIS EXISTS
---------------
The Video Review autosave (``Store.upsert_scoring_draft``) SEEDS a
``needs_scoring`` review_state row the instant a reviewer drops an EEG onset --
before any Racine score. If the reviewer never finishes (adds Racine + submits),
the recording is stranded in the "⚠️ Needs more onsets" pool looking scored, but
carrying an onset with NO Racine. These are not "scored, just add landmarks"
files (what that pool is for) -- they are incomplete drafts that belong back in
the reviewable pool so they get scored from the start.

WHAT THIS DOES  (per animal)
----------------------------
Finds the animal's latest-``needs_scoring`` recordings whose events carry NO
Racine (the stranded set; a recording with >=1 Racine event is a genuine
"needs more onsets" file and is LEFT ALONE), and moves each into the auto-filter
"🚩 Flag (has events)" pool:

  1. Every review_state row for (animal, file) whose status is in the
     queue-excluding set (``needs_scoring`` and, for files that were previously
     submitted, ``pending_pi_review`` / ``pi_approved``) is set to
     ``abandoned`` -- a valid, non-queue-excluding status that PRESERVES
     ``markers_json`` (the surviving onset), so the recording leaves
     needs_scoring / approved and re-enters the queue.
  2. A ``auto_filter_flag`` review_event_log row is added for any file that
     lacks one, so the recording lands in the *Flag* subset (queue ∩
     auto_filter_flag), not just the plain queue. The payload is tagged
     ``source=manual_revert_stranded_draft`` so the scoring-loss audit can
     exclude these fabricated flags from its detector-flag signal.

Dry-run by default; ``--apply`` writes, after dumping every affected row to a
timestamped JSON backup. Idempotent: re-running finds nothing (the rows are no
longer ``needs_scoring``).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

# review_state statuses that keep a file OUT of the review queue (mirrors the
# NOT EXISTS in Store.get_review_queue). Setting these to 'abandoned' releases
# the file back to the queue while keeping its markers_json.
_QUEUE_EXCLUDING = ("no_events", "has_events", "pending_pi_review",
                    "pi_approved", "needs_scoring")
_FLAG_SOURCE = "manual_revert_stranded_draft"
_MAX_FILES = 1_000_000        # NASA Rule 2.


def _has_racine(markers_json) -> bool:
    try:
        d = json.loads(markers_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return False
    return any(isinstance(e, dict) and e.get("racine") not in (None, "", [])
               for e in d if isinstance(e, dict))


def find_stranded(conn: sqlite3.Connection, animal: str) -> dict:
    """The stranded-draft plan for *animal*: which files revert, which are kept,
    which review_state rows change, and which files need an auto_filter_flag."""
    assert animal, "animal required"
    conn.row_factory = sqlite3.Row
    latest = conn.execute(
        """SELECT rs.file_id, rs.markers_json FROM review_state rs
           WHERE rs.animal_id=? AND rs.status='needs_scoring'
             AND rs.id=(SELECT MAX(rs2.id) FROM review_state rs2
                        WHERE rs2.file_id=rs.file_id
                          AND rs2.animal_id=rs.animal_id)""",
        (animal,)).fetchall()
    assert len(latest) < _MAX_FILES, "needs_scoring scan runaway"
    target = sorted(r["file_id"] for r in latest
                    if not _has_racine(r["markers_json"]))
    keep = sorted(r["file_id"] for r in latest if _has_racine(r["markers_json"]))
    rows, need_flag = [], []
    if target:
        ph = ",".join("?" * len(target))
        rows = conn.execute(
            f"""SELECT id, file_id, status FROM review_state
                WHERE animal_id=? AND file_id IN ({ph})
                  AND status IN {_QUEUE_EXCLUDING}""",
            [animal] + target).fetchall()
        flagged = set(x[0] for x in conn.execute(
            "SELECT DISTINCT file_id FROM review_event_log "
            "WHERE action='auto_filter_flag' AND animal_id=?", (animal,)))
        need_flag = [f for f in target if f not in flagged]
    return {"animal": animal, "target_files": target, "keep_files": keep,
            "rows_to_abandon": [dict(r) for r in rows], "need_flag": need_flag}


def _backup_rows(conn, animal, target, path) -> int:
    """Dump every review_state row (full) for the target (animal, file) set +
    the plan to *path* so the change is precisely reversible."""
    conn.row_factory = sqlite3.Row
    ph = ",".join("?" * len(target))
    rows = conn.execute(
        f"""SELECT * FROM review_state WHERE animal_id=? AND file_id IN ({ph})""",
        [animal] + target).fetchall() if target else []
    payload = {"animal": animal, "at": datetime.now().isoformat(),
               "review_state_rows": [dict(r) for r in rows]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return len(rows)


def apply_revert(conn: sqlite3.Connection, animal: str, plan: dict,
                 user_email: str, now: str) -> dict:
    """Execute the revert (call inside a transaction). Returns counts."""
    target = plan["target_files"]
    if not target:
        return {"abandoned": 0, "flags_added": 0}
    ph = ",".join("?" * len(target))
    cur = conn.execute(
        f"""UPDATE review_state SET status='abandoned', updated_at=?
            WHERE animal_id=? AND file_id IN ({ph})
              AND status IN {_QUEUE_EXCLUDING}""",
        [now, animal] + target)
    n_ab = cur.rowcount
    pj = json.dumps({"animal_id": animal, "source": _FLAG_SOURCE,
                     "reason": "reverted from needs_scoring (no racine)"})
    for f in plan["need_flag"]:
        conn.execute(
            "INSERT INTO review_event_log (file_id,user_email,action,"
            "payload_json,at,animal_id) VALUES (?,?,?,?,?,?)",
            (f, user_email, "auto_filter_flag", pj, now, animal))
    return {"abandoned": n_ab, "flags_added": len(plan["need_flag"])}


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--animal", required=True)
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--user", default="hoang392@umn.edu",
                    help="user_email stamped on the added auto_filter_flag rows")
    ap.add_argument("--apply", action="store_true",
                    help="write the change (default: dry-run report only)")
    ap.add_argument("--backup-dir", default="data/backups")
    a = ap.parse_args(argv)

    conn = sqlite3.connect(a.db)
    try:
        plan = find_stranded(conn, a.animal)
        n_rows = len(plan["rows_to_abandon"])
        from collections import Counter
        by_status = Counter(r["status"] for r in plan["rows_to_abandon"])
        print(f"[{a.animal}] stranded needs_scoring files (no racine): "
              f"{len(plan['target_files'])}")
        print(f"   kept (real racine, untouched): {plan['keep_files']}")
        print(f"   review_state rows -> abandoned: {n_rows}  {dict(by_status)}")
        print(f"   auto_filter_flag to add: {len(plan['need_flag'])}")
        if not a.apply:
            print("\nDRY-RUN (no writes). Re-run with --apply to execute.")
            return 0
        if not plan["target_files"]:
            print("\nNothing to revert.")
            return 0
        os.makedirs(a.backup_dir, exist_ok=True)
        bpath = os.path.join(
            a.backup_dir,
            f"revert_{a.animal}_{datetime.now():%Y%m%d_%H%M%S}.json")
        nb = _backup_rows(conn, a.animal, plan["target_files"], bpath)
        print(f"\n   backed up {nb} review_state rows -> {bpath}")
        now = datetime.now().isoformat()
        with conn:                       # transaction
            res = apply_revert(conn, a.animal, plan, a.user, now)
        print(f"   APPLIED: {res['abandoned']} rows -> abandoned, "
              f"{res['flags_added']} auto_filter_flag added.")
        # verify
        after = find_stranded(conn, a.animal)
        print(f"   VERIFY: stranded remaining = {len(after['target_files'])} "
              f"(expect 0); still-needs_scoring kept = {after['keep_files']}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(_main())
