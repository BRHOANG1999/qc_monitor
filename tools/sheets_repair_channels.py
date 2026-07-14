"""One-time repair of the BHZ Analysis Log summary tabs.

Two data repairs, both dry-run by default:

  A. Channel(s) cells that Sheets silently coerced into DATES. Under
     USER_ENTERED, "2, 3" is parsed as Feb 3 (serial 46056) and "3/4" as Mar 4
     (45720); the cell still *displays* the original text, so the corruption is
     invisible. The true value is recoverable from the FORMATTED value, so we
     rewrite each corrupted cell as literal text (RAW -> no re-parsing).
     Affects both the app's rows and the lab's hand-typed ones.

  B. BCH040's duplicate day rows. An older non-idempotent push appended a
     second row for 2026-03-19 / 2026-03-23 alongside PM's hand row. Drop the
     app's row and keep PM's -- the upsert then merges into PM's row, so their
     "Recording File Name of event" / "Events (>=3)" notes stay live.

Going forward this can't recur: _session_meta wraps Channel(s) in
sheets_write.force_text, and the upsert is idempotent.

    python tools/sheets_repair_channels.py           # show what would change
    python tools/sheets_repair_channels.py --apply   # do it
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import re
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import yaml  # noqa: E402

from src.utils import sheets_write as sw  # noqa: E402

# Google date serials for 2024-2026 land around 45000-46600; real channel
# numbers are single/double digit. Anything this large in Channel(s) is a date.
_SERIAL_FLOOR = 1000

# Sheets' serial epoch (day 0 == 1899-12-30).
_EPOCH = datetime.date(1899, 12, 30)

# BCH040 rows the old buggy push appended next to PM's hand row.
_DUP_TAB = "BCH040 Daily"
_DUP_DATES = {"2026-03-19", "2026-03-23"}


# Collateral from an earlier off-by-one in THIS tool (now fixed + tested in
# tests/test_sheets_events.py::test_repair_tool_sheet_row_mapping): each write
# landed one row low, so the row just after a corrupted run was overwritten and
# a junk row was appended past the end. These undo that.
#
# BCH040 2025-04-02 was NOT date-corrupted (so PM typed a plain number, not
# "3/4"), and its only sibling -- same location SLM/SR, same type baseline-ER,
# the very next day -- reads 2. Restoring 2.
_CLOBBERED = [("BCH040 Daily", "2025-04-02", "Channel(s)", "2")]


def _load_cfg(path: str) -> dict:
    with io.open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _col(header: list, name: str) -> int:
    for i, h in enumerate(header):
        if sw._norm(h) == sw._norm(name):
            return i
    return -1


def _sheet_row(grid_index: int) -> int:
    """1-based SHEET row for grid index *grid_index* (raw[0] is the header, so
    raw[1] -- the first data row -- is sheet row 2). Off-by-one here silently
    writes/deletes the NEIGHBOURING row, so it lives in one tested helper."""
    assert grid_index >= 1, "grid_index 0 is the header row"
    return grid_index + 1


def _recover(serial, shown: str) -> str:
    """Original channel text behind a date-coerced cell.

    The displayed text is the cell's DATE FORMAT applied to the serial, so it
    usually reads back as the typed value ("2, 3") -- but some cells render the
    year too ("3, 4, 2005" for serial 38415). The year was never in the lab's
    typed value, so strip it. Returns "" if the result doesn't look like a
    channel list, so a weird cell is reported rather than silently rewritten.
    """
    d = _EPOCH + datetime.timedelta(days=int(serial))
    s = str(shown).strip()
    for suffix in (f", {d.year}", f"/{d.year}", f" {d.year}", f"-{d.year}"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    return s if re.fullmatch(r"[\d]+(\s*[,/]\s*[\d]+)*", s or "") else ""


def repair_channels(svc, sid: str, tabs: list[str], apply: bool) -> int:
    """Rewrite date-coerced Channel(s) cells as literal text."""
    fixed = 0
    for tab in tabs:
        raw = sw.read_grid(svc, sid, tab, unformatted=True)
        if not raw:
            continue
        fmt = svc.spreadsheets().values().get(
            spreadsheetId=sid, range=sw._q(tab),
            valueRenderOption="FORMATTED_VALUE").execute().get("values", [])
        ci = _col(raw[0], "Channel(s)")
        if ci < 0:
            continue
        batch = []
        # raw[r] is the r-th entry of the WHOLE grid, so raw[1] is the first
        # data row and lives at SHEET ROW 2 -> sheet_row == r + 1. (Getting
        # this wrong writes every value one row too low.)
        for r, row in enumerate(raw[1:], start=1):
            v = row[ci] if ci < len(row) else ""
            if not isinstance(v, (int, float)) or v < _SERIAL_FLOOR:
                continue
            shown = (fmt[r][ci] if r < len(fmt) and ci < len(fmt[r]) else "")
            cell = f"{sw._col_letter(ci + 1)}{_sheet_row(r)}"
            true_val = _recover(v, shown)
            if not true_val:
                print(f"      !! {tab} {cell}: {v} shows {shown!r} -- not a "
                       "channel list; SKIPPED, fix by hand")
                continue
            a1 = f"{sw._q(tab)}!{cell}"
            print(f"      {tab} {cell}: {v} -> {true_val!r}")
            batch.append({"range": a1, "values": [[true_val]]})
        if batch and apply:
            # RAW: store the string exactly, so it can never be re-parsed.
            svc.spreadsheets().values().batchUpdate(
                spreadsheetId=sid,
                body={"valueInputOption": "RAW", "data": batch}).execute()
        fixed += len(batch)
    return fixed


def drop_dup_rows(svc, sid: str, apply: bool) -> int:
    """Delete the app-authored duplicate day rows, keeping the lab's.

    The discriminator is "Recording File Name of event": PM fills it, the
    app never writes it. Do NOT key off "Who Completed Analysis" -- once the
    upsert merges into PM's row it overwrites that cell with the app user's
    email, so PM's OWN row then looks app-authored and would be deleted.

    Only acts on dates that really are duplicated, and never drops the last
    remaining row for a date.
    """
    raw = sw.read_grid(svc, sid, _DUP_TAB, unformatted=True)
    hdr = raw[0]
    di = _col(hdr, "Date")
    ei = _col(hdr, "Recording File Name of event")
    assert di >= 0 and ei >= 0, "BCH040 tab is missing its key columns"
    by_date: dict = {}
    for r, row in enumerate(raw[1:], start=1):
        date = sw._canon_key(row[di]) if di < len(row) else ""
        hand = str(row[ei]).strip() if ei < len(row) else ""
        by_date.setdefault(date, []).append((_sheet_row(r), bool(hand)))
    victims = []
    for date, rows in by_date.items():
        if len(rows) < 2 or not any(hand for _n, hand in rows):
            continue          # not duplicated, or no lab row to keep
        for rownum, hand in rows:
            if not hand:      # app-authored copy -> drop
                victims.append((rownum, date, "no Recording File Name"))
    for rownum, date, who in victims:
        print(f"      drop {_DUP_TAB} row{rownum}  ({date}, who={who!r})")
    if victims and apply:
        sheet_id = sw._sheet_ids_by_title(svc, sid)[_DUP_TAB][0]
        # Delete bottom-up so earlier indices stay valid.
        reqs = [{"deleteDimension": {"range": {
            "sheetId": sheet_id, "dimension": "ROWS",
            "startIndex": rownum - 1, "endIndex": rownum}}}
            for rownum, _d, _w in sorted(victims, reverse=True)]
        svc.spreadsheets().batchUpdate(
            spreadsheetId=sid, body={"requests": reqs}).execute()
    return len(victims)


def drop_junk_rows(svc, sid: str, tabs: list[str], apply: bool) -> int:
    """Remove rows the bad write appended past the end of a tab: they carry a
    lone Channel(s) value and nothing else (no Date)."""
    dropped = 0
    for tab in tabs:
        raw = sw.read_grid(svc, sid, tab, unformatted=True)
        if not raw:
            continue
        di = _col(raw[0], "Date")
        ci = _col(raw[0], "Channel(s)")
        if di < 0 or ci < 0:
            continue
        victims = []
        for r, row in enumerate(raw[1:], start=1):
            has_date = di < len(row) and str(row[di]).strip()
            has_ch = ci < len(row) and str(row[ci]).strip()
            filled = [c for c in row if str(c).strip()]
            if not has_date and has_ch and len(filled) == 1:
                victims.append(_sheet_row(r))
        for rownum in victims:
            print(f"      drop {tab} row{rownum} (lone Channel(s), no Date)")
        if victims and apply:
            sheet_id = sw._sheet_ids_by_title(svc, sid)[tab][0]
            svc.spreadsheets().batchUpdate(spreadsheetId=sid, body={
                "requests": [{"deleteDimension": {"range": {
                    "sheetId": sheet_id, "dimension": "ROWS",
                    "startIndex": n - 1, "endIndex": n}}}
                    for n in sorted(victims, reverse=True)]}).execute()
        dropped += len(victims)
    return dropped


def restore_clobbered(svc, sid: str, apply: bool) -> int:
    """Put back the cell the off-by-one overwrote (see _CLOBBERED)."""
    n = 0
    for tab, date_iso, colname, value in _CLOBBERED:
        raw = sw.read_grid(svc, sid, tab, unformatted=True)
        ci = _col(raw[0], colname)
        di = _col(raw[0], "Date")
        for r, row in enumerate(raw[1:], start=1):
            if di >= len(row) or sw._canon_key(row[di]) != date_iso:
                continue
            cur = row[ci] if ci < len(row) else ""
            a1 = f"{sw._q(tab)}!{sw._col_letter(ci + 1)}{_sheet_row(r)}"
            print(f"      {tab} {date_iso} {colname}: {cur!r} -> {value!r}")
            if apply:
                svc.spreadsheets().values().update(
                    spreadsheetId=sid, range=a1, valueInputOption="RAW",
                    body={"values": [[value]]}).execute()
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    gs = (_load_cfg(args.config) or {}).get("google_sheets") or {}
    sid = gs["spreadsheet_id"]
    suffix = gs.get("summary_tab_suffix", " Daily")
    svc = sw._sheets_api_rw(gs["service_account_file"])
    tabs = sorted(t for t in sw.sheet_titles(svc, sid)
                  if t and t.endswith(suffix))
    print(f"Sheet {sid}\n  MODE: {'APPLY' if args.apply else 'DRY RUN'}\n")

    print(f"[A] date-coerced Channel(s) cells across {len(tabs)} summary tab(s):")
    n = repair_channels(svc, sid, tabs, args.apply)
    print(f"    -> {n} cell(s) {'repaired' if args.apply else 'to repair'}\n")

    print("[B] junk rows appended by the earlier off-by-one:")
    j = drop_junk_rows(svc, sid, tabs, args.apply)
    print(f"    -> {j} row(s) {'dropped' if args.apply else 'to drop'}\n")

    print("[C] restore the cell the earlier off-by-one overwrote:")
    c = restore_clobbered(svc, sid, args.apply)
    print(f"    -> {c} cell(s) {'restored' if args.apply else 'to restore'}\n")

    print("[D] duplicate BCH040 day rows (keep PM's, drop the app's):")
    d = drop_dup_rows(svc, sid, args.apply)
    print(f"    -> {d} row(s) {'dropped' if args.apply else 'to drop'}")

    print("\nDone." if args.apply else "\nDry run only. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
