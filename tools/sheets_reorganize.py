"""Reorganize the BHZ Analysis Log tabs: rename, add the pending column, order.

One-time migration (safe to re-run -- every step is idempotent):

  1. RENAME each bare-MouseID summary tab  "BCH111"  ->  "BCH111 Daily"
     (a rename keeps the tab's sheetId, so #gid= links keep working).
  2. ENSURE the "Events Needing Onsets" column exists on every summary tab
     (appended at the end; the lab's hand-filled columns are untouched).
  3. REORDER every tab into the canonical flow: the cohort index tab(s) first,
     then all "<MouseID> Daily" sorted, then all "<MouseID> Events" sorted.

Dry-run by default -- nothing is written without --apply.

    python tools/sheets_reorganize.py            # show what would change
    python tools/sheets_reorganize.py --apply    # do it

IMPORTANT: run this only with the dashboard's new code deployed. Old code looks
for a tab titled the bare MouseID; if it pushes after the rename it will CREATE
a ghost "BCH111" tab. Restart the dashboard right after applying, and don't
Approve/Finalize in between.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import yaml  # noqa: E402

from src.utils import sheets_write as sw  # noqa: E402

_MOUSE_ID = re.compile(r"^[A-Z]{2,4}\d{2,4}$")   # BCH111, BCH040, ...


def _load_cfg(path: str = "config/config.yaml") -> dict:
    # utf-8 explicitly: the file has emoji tab names and cp1252 chokes.
    with io.open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                     help="write the changes (default: dry run)")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    gs = (_load_cfg(args.config) or {}).get("google_sheets") or {}
    sid = gs.get("spreadsheet_id")
    assert sid, "google_sheets.spreadsheet_id missing"
    sum_suffix = gs.get("summary_tab_suffix", " Daily")
    ev_suffix = gs.get("events_tab_suffix", " Events")
    pending_col = gs.get("summary_pending_column", "Events Needing Onsets")
    assert sum_suffix, "summary_tab_suffix must be non-empty to reorganize"

    print(f"Sheet {sid}")
    print(f"  summary suffix {sum_suffix!r} · events suffix {ev_suffix!r}")
    print(f"  pending column {pending_col!r}")
    print(f"  MODE: {'APPLY' if args.apply else 'DRY RUN (no writes)'}\n")

    svc = sw._sheets_api_rw(gs["service_account_file"])
    ids = sw._sheet_ids_by_title(svc, sid)
    current = [t for t, _ in sorted(ids.items(), key=lambda kv: kv[1][1])]

    # 1. Rename bare-MouseID tabs -> "<MouseID><suffix>".
    renames = [(t, f"{t}{sum_suffix}") for t in current
               if _MOUSE_ID.match(t or "")]
    print(f"[1/3] rename {len(renames)} bare-MouseID tab(s):")
    for old, new in renames:
        done = sw.rename_tab(svc, sid, old, new) if args.apply else None
        flag = "" if not args.apply else ("  ok" if done else "  (skipped)")
        print(f"      {old!r} -> {new!r}{flag}")
    if not renames:
        print("      (none -- already renamed)")

    # 2. Ensure the pending column on every summary tab.
    titles = (list(sw._sheet_ids_by_title(svc, sid)) if args.apply
              else [new for _o, new in renames]
                   + [t for t in current if t.endswith(sum_suffix)])
    summaries = sorted({t for t in titles if t.endswith(sum_suffix)})
    print(f"\n[2/3] ensure {pending_col!r} on {len(summaries)} summary tab(s):")
    for t in summaries:
        if args.apply:
            n = sw._ensure_header_columns(svc, sid, t, [pending_col],
                                           value_input_option="USER_ENTERED")
            print(f"      {t}: {'added' if n else 'already present'}")
        else:
            print(f"      {t}")

    # 3. Reorder into the canonical flow.
    print("\n[3/3] tab order:")
    if args.apply:
        final = sw.reorder_tabs(svc, sid, sum_suffix, ev_suffix)
    else:
        after_rename = [dict(renames).get(t, t) for t in current]
        final = sw._target_tab_order(after_rename, sum_suffix, ev_suffix)
    for i, t in enumerate(final):
        print(f"      {i:>2}  {t}")

    # Guard: the hazard this migration has to avoid.
    if args.apply:
        left = [t for t in sw._sheet_ids_by_title(svc, sid)
                if _MOUSE_ID.match(t or "")]
        if left:
            print(f"\n!! bare-MouseID tab(s) still present: {left}\n"
                   "   Old dashboard code likely re-created them -- restart "
                   "the dashboard and delete these ghosts.")
            return 1
        print("\nDone. Restart the dashboard now so it writes to the "
               "renamed tabs.")
    else:
        print("\nDry run only. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
