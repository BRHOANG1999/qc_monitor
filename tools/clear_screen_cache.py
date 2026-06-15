"""Force-clear the Mass Analyze envelope caches.

The one-time cache eviction in Store._init_db runs only once per DB
(guarded by PRAGMA user_version), so a plain restart won't re-clear it.
Use this when you need to guarantee the NEXT scan recomputes every file
from scratch -- e.g. after changing how the envelope/AUC is computed, or
to rule out a stale cache while troubleshooting.

Deletes every row from envelope_peak_cache + envelope_auc_cache and
resets user_version to 0 (so the app's own boot-time eviction also runs
again, belt and suspenders). The next scan repopulates them.

Run from repo root (stop the service first, or run against a copy):
    python tools/clear_screen_cache.py <db>
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: clear_screen_cache.py <db>", file=sys.stderr)
        return 2
    db = sys.argv[1]
    if not Path(db).is_file():
        print(f"db not found: {db}", file=sys.stderr)
        return 2
    conn = sqlite3.connect(db)
    try:
        before_p = conn.execute(
            "SELECT COUNT(*) FROM envelope_peak_cache").fetchone()[0]
        before_a = conn.execute(
            "SELECT COUNT(*) FROM envelope_auc_cache").fetchone()[0]
        conn.execute("DELETE FROM envelope_peak_cache")
        conn.execute("DELETE FROM envelope_auc_cache")
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    finally:
        conn.close()
    print(f"cleared envelope_peak_cache: {before_p} rows")
    print(f"cleared envelope_auc_cache: {before_a} rows")
    print("user_version reset to 0 (boot-time eviction will re-run).")
    print("Next Mass Analyze scan will recompute every file fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
