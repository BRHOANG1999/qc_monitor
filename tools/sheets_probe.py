"""Probe the Google Sheet sync target before enabling it.

Read-only. Uses the service account in config.google_sheets to:
  1. confirm auth + Sheets API access work,
  2. list every tab with its gid + title (so you can set tab_name for
     gid 256001691 -- the API needs the TITLE, not the gid),
  3. print the target tab's header row,
  4. preview how a sample (animal, day) stat row would map onto that
     header (so you can spot columns that need a column_map override).

Run from repo root:
    python tools/sheets_probe.py            # uses config/config.yaml
    python tools/sheets_probe.py 256001691  # find this gid's title
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from src.utils import sheets_write as sw  # noqa: E402


def main() -> int:
    target_gid = int(sys.argv[1]) if len(sys.argv) > 1 else 256001691

    cfg_path = REPO / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    gs = (cfg.get("google_sheets") or {})
    sa_file = gs.get("service_account_file")
    sheet_id = gs.get("spreadsheet_id")
    if not sa_file or not sheet_id:
        print("config.google_sheets missing service_account_file / "
              "spreadsheet_id", file=sys.stderr)
        return 2
    if not (REPO / sa_file).is_file():
        print(f"service account file not found: {sa_file}",
              file=sys.stderr)
        return 2

    print(f"service account file : {sa_file}")
    print(f"spreadsheet_id       : {sheet_id}")
    print(f"looking for gid      : {target_gid}\n")

    # Read-WRITE client (same one the finalize hook uses) -- proves the
    # write scope + creds load. We only READ here.
    try:
        svc = sw._sheets_api_rw(str(REPO / sa_file))
    except Exception as e:
        print(f"FAILED to build Sheets client: {e}", file=sys.stderr)
        print("\nLikely causes: google-api-python-client/google-auth "
              "not installed, or the service-account JSON is bad.")
        return 1

    # 1) Spreadsheet metadata -> every tab's gid + title.
    try:
        meta = svc.spreadsheets().get(
            spreadsheetId=sheet_id).execute()
    except Exception as e:
        print(f"FAILED to read spreadsheet metadata: {e}",
              file=sys.stderr)
        print("\nLikely causes:")
        print("  - the sheet is NOT shared (Editor) with the service "
              "account email,")
        print("  - the Sheets API is not enabled for the SA's project,")
        print("  - wrong spreadsheet_id.")
        return 1

    print(f"spreadsheet title    : {meta.get('properties', {}).get('title')}\n")
    print("tabs (gid -> title):")
    target_title = None
    for s in meta.get("sheets", []):
        p = s.get("properties", {})
        gid, title = p.get("sheetId"), p.get("title")
        mark = "  <-- gid match" if gid == target_gid else ""
        print(f"  {gid:>12}  {title!r}{mark}")
        if gid == target_gid:
            target_title = title
    print()

    print("Routing: one row PER DAY into each animal's own tab "
          "(tab title == MouseID). No single tab_name needed.\n")
    if not target_title:
        print(f"(gid {target_gid} not found -- not required for "
              "per-animal routing.)")
        target_title = next(
            (s["properties"]["title"] for s in meta.get("sheets", [])
             if s.get("properties", {}).get("title")), None)
    print(f"Reading a representative tab header: {target_title!r}\n")

    # 2) Header row of the target tab.
    try:
        grid = sw.read_grid(svc, sheet_id, target_title)
    except Exception as e:
        print(f"FAILED to read tab {target_title!r}: {e}",
              file=sys.stderr)
        return 1
    if not grid:
        print(f"Tab {target_title!r} is EMPTY -- add a header row "
              "(Date, Animal, ...) first.")
        return 1
    header = grid[0]
    print(f"header row ({len(header)} cols):")
    for c in header:
        print(f"   - {c!r}")
    print()

    # 3) Preview the mapping with a sample day-stat row. Cells that map
    # to a stat are WRITTEN; '(left as-is)' cells are never touched on
    # update (so the lab's hand-filled columns survive).
    sample = {
        "date": "2026-06-11", "animal": target_title,
        "csv_filename": "20260611_BCH062.csv",
        "n_files": 24, "n_files_with_events": 2, "n_events": 3,
        "max_racine": 5, "n_during_stim_events": 1,
        "n_no_event_files": 22, "reviewers": "you@umn.edu",
        "recording_location": "SR, SLM", "channels": "2, 3",
        "type_of_recording": "stim", "more_settings": "5nC, 100us, 130Hz",
        "exported_at": "2026-06-16T18:00:00",
    }
    column_map = gs.get("column_map") or {}
    cells = sw.map_cells(header, sample, column_map)
    print("column mapping (what the sync would do to each cell):")
    untouched = []
    for col, val in zip(header, cells):
        if val is None:
            print(f"   {col!r:<32} -> (left as-is)")
            untouched.append(col)
        else:
            print(f"   {col!r:<32} -> writes {val!r}")
    print()
    key_cols = gs.get("key_columns", ["Date"])
    print(f"key_columns (per-tab upsert match): {key_cols}")
    for kc in key_cols:
        norm = sw._norm(kc)
        ok = any(sw._norm(h) == norm for h in header)
        print(f"   {kc!r}: {'FOUND' if ok else 'MISSING -- upsert will '
                                                'fail'}")
    if untouched:
        print(f"\nLeft-as-is columns: {untouched}")
        print("If one of those SHOULD be auto-filled, add a column_map "
              "entry, e.g.:")
        print('   column_map: {"Channel(s)": "channels"}')
        print("stat keys: date, animal, csv_filename, n_events, "
              "reviewers, max_racine, n_during_stim_events, n_files, "
              "n_files_with_events, n_no_event_files, exported_at")
    print("\nProbe OK. Routing is per-animal-tab; no tab_name needed. "
          "If the mapping looks right, flip google_sheets.enabled: true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
