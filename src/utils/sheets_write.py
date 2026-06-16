"""Write per-(animal, day) summary rows into a Google Sheet.

Companion to ``sheets.py`` (which is read-only). The lab's final BHZ CSV
export, when the PI finalizes approved events, also upserts one row per
(animal, day) into a Google Sheet so the running tally lives where the
lab already looks.

Design:
* Service-account auth, **read-write** scope. Reuses the same SA JSON the
  read path uses; the target spreadsheet must be shared **Editor** with
  that service account's email and the SA's project must have the Sheets
  API enabled.
* **Upsert keyed on the sheet's own columns** (default ``Date`` +
  ``Animal``): read the tab, find the row whose key cells match the
  incoming day-row, ``update`` it in place, else ``append``.
* **Match the sheet's existing header**: we read row 1 and map each
  incoming canonical stat onto whichever header column it belongs to
  (normalized-name alias table, overridable via ``column_map``). Unknown
  headers are left blank; we never reorder or rename the sheet's columns.

All failures are non-fatal to the caller (the CSV write must not be
blocked by a Sheets hiccup): the public entry points raise, and the
finalize hook wraps them in try/except.
"""

from __future__ import annotations

import logging
import re
from threading import RLock

logger = logging.getLogger("qc_monitor.utils.sheets_write")

_RW_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_rw_cache: dict = {}
_rw_lock = RLock()

# Canonical stat key -> header aliases (normalized). The day-row dict the
# finalize hook builds uses these canonical keys; we match them onto the
# sheet's real header by normalizing both sides.
_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "day"),
    "animal": ("animal", "animalid", "subject", "mouse", "rat",
                "mouseid"),
    "csv_filename": ("csvfilename", "csvfile", "filename", "csv"),
    "n_files": ("files", "nfiles", "numfiles", "recordings"),
    "n_files_with_events": ("fileswithevents", "eventfiles"),
    "n_events": ("events", "nevents", "numevents", "seizures",
                  "nseizures", "seizurecount",
                  "numberofbehavioralevents", "behavioralevents",
                  "numbehavioralevents"),
    "max_racine": ("maxracine", "racine", "maxscore"),
    "n_during_stim_events": ("duringstim", "duringstimevents",
                              "stimevents", "eventsduringstim"),
    "n_no_event_files": ("noeventfiles", "nnoevents"),
    "n_needs_scoring": ("needsscoring", "flagged"),
    "reviewers": ("whocompletedanalysis", "reviewer", "reviewers",
                   "analyst", "scoredby", "completedby"),
    "channels": ("channels", "channel", "chans"),
    "recording_location": ("recordinglocation", "location",
                            "electrodelocation", "electrode"),
    "type_of_recording": ("typeofrecording", "recordingtype", "type",
                           "stimorbaseline", "stimbaseline"),
    "more_settings": ("moresettings", "moresettingsbch", "settings",
                       "stimsettings", "stimparams", "stimparameters"),
    "exported_at": ("exportedat", "exported", "lastexport",
                     "updatedat"),
}


def _norm(s: str) -> str:
    """Lowercase + strip everything but a-z0-9 for fuzzy header match."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _q(tab_name: str) -> str:
    """A1-quote a tab title. Titles like 'BCH039' collide with cell
    refs, so they MUST be single-quoted in a range; embedded quotes
    double per A1 rules."""
    return "'" + str(tab_name).replace("'", "''") + "'"


def _sheets_api_rw(service_account_file: str):
    """Lazy-build a read-WRITE Sheets client, cached by SA path."""
    assert service_account_file, "service_account_file required"
    with _rw_lock:
        svc = _rw_cache.get(service_account_file)
        if svc is not None:
            return svc
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_file(
        service_account_file, scopes=_RW_SCOPES)
    svc = build("sheets", "v4", credentials=creds,
                cache_discovery=False)
    with _rw_lock:
        _rw_cache[service_account_file] = svc
    return svc


def read_grid(svc, sheet_id: str, tab_name: str) -> list[list[str]]:
    """Return the tab as a list of rows (row 0 = header). The title is
    A1-quoted so cell-like titles ('BCH039') resolve to the whole
    sheet, not a bogus cell range."""
    resp = svc.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=_q(tab_name),
    ).execute()
    return resp.get("values", []) or []


def map_cells(header: list[str], stats: dict,
               column_map: dict | None = None) -> list:
    """Per header column: the stat value, or ``None`` when no stat maps
    to that column. ``None`` (vs "") lets the upsert distinguish
    "we own this cell" from "leave it alone" so human-maintained
    columns are never blanked on update.
    """
    overrides = {_norm(k): v for k, v in (column_map or {}).items()}
    alias_to_key: dict[str, str] = {}
    for key, aliases in _ALIASES.items():
        alias_to_key[_norm(key)] = key
        for a in aliases:
            alias_to_key[_norm(a)] = key
    out: list = []
    for col in header:
        n = _norm(col)
        key = overrides.get(n) or alias_to_key.get(n)
        out.append(stats[key] if (key is not None and key in stats)
                    else None)
    return out


def map_row_to_header(header: list[str], stats: dict,
                       column_map: dict | None = None) -> list:
    """Same as :func:`map_cells` but unmatched columns are "" -- used
    for previews / appending a brand-new row."""
    return ["" if c is None else c
             for c in map_cells(header, stats, column_map)]


def _key_index(header: list[str], key_columns: list[str]) -> list[int]:
    """Column indices in *header* for each name in *key_columns*."""
    norm_header = [_norm(h) for h in header]
    idxs: list[int] = []
    for kc in key_columns:
        nk = _norm(kc)
        idxs.append(norm_header.index(nk) if nk in norm_header else -1)
    return idxs


def _a1_row(tab_name: str, row_1indexed: int) -> str:
    return f"{_q(tab_name)}!A{row_1indexed}"


def sheet_titles(svc, sheet_id: str) -> set[str]:
    """Existing tab titles in the spreadsheet."""
    meta = svc.spreadsheets().get(spreadsheetId=sheet_id).execute()
    return {s.get("properties", {}).get("title")
            for s in meta.get("sheets", [])}


def upsert_rows_into_tab(svc, sheet_id: str, tab_name: str,
                           key_columns: list[str],
                           stats_rows: list[dict],
                           column_map: dict | None = None) -> dict:
    """Upsert *stats_rows* into one tab, keyed on *key_columns*.

    MERGE semantics on update: only the columns we have a value for are
    written; every other cell keeps its existing content, so columns the
    lab fills by hand (Recording Location, etc.) are never blanked.
    """
    grid = read_grid(svc, sheet_id, tab_name)
    if not grid:
        raise ValueError(
            f"tab {tab_name!r} is empty (needs a header row)")
    header = grid[0]
    body = grid[1:]
    key_idx = _key_index(header, key_columns)
    if any(i < 0 for i in key_idx):
        raise ValueError(
            f"key columns {key_columns} not all found in header "
            f"{header}")
    # key tuple -> (1-indexed sheet row, existing values list)
    existing: dict[tuple, tuple] = {}
    for r, vals in enumerate(body):
        ktuple = tuple(
            _norm(vals[i]) if i < len(vals) else "" for i in key_idx)
        existing[ktuple] = (r + 2, vals)

    updated = appended = 0
    appends: list[list] = []
    max_iter = len(stats_rows) + 1
    for n, stats in enumerate(stats_rows):
        assert n < max_iter, "upsert runaway"
        cells = map_cells(header, stats, column_map)  # value | None
        # Key cells must be present to upsert by them.
        if any(cells[i] is None for i in key_idx):
            logger.warning("row missing key column(s) %s; skipped",
                            key_columns)
            continue
        ktuple = tuple(_norm(cells[i]) for i in key_idx)
        hit = existing.get(ktuple)
        if hit is not None:
            sheet_row, exrow = hit
            merged = []
            for i in range(len(header)):
                if cells[i] is not None:
                    merged.append(cells[i])
                else:
                    merged.append(exrow[i] if i < len(exrow) else "")
            svc.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=_a1_row(tab_name, sheet_row),
                valueInputOption="USER_ENTERED",
                body={"values": [merged]},
            ).execute()
            updated += 1
        else:
            appends.append(["" if c is None else c for c in cells])
            appended += 1
    if appends:
        svc.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=_q(tab_name),
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": appends},
        ).execute()
    return {"updated": updated, "appended": appended}


def upsert_day_rows(service_account_file: str, sheet_id: str,
                      rows_by_tab: dict, key_columns: list[str],
                      column_map: dict | None = None) -> dict:
    """Upsert per-tab day rows. *rows_by_tab* maps a tab title (the
    animal's MouseID tab) to its list of canonical day-stat dicts.

    Tabs that don't exist in the spreadsheet are skipped with a warning
    (we never auto-create a tab in the lab's log). Returns
    ``{"updated", "appended", "skipped_tabs"}``. Raises on auth/API
    error -- the finalize hook wraps this non-fatally.
    """
    assert sheet_id, "sheet_id required"
    if not rows_by_tab:
        return {"updated": 0, "appended": 0, "skipped_tabs": []}
    svc = _sheets_api_rw(service_account_file)
    titles = sheet_titles(svc, sheet_id)
    updated = appended = 0
    skipped: list[str] = []
    for tab, rows in rows_by_tab.items():
        if tab not in titles:
            logger.warning("no tab %r in sheet; skipping %d row(s)",
                            tab, len(rows))
            skipped.append(tab)
            continue
        res = upsert_rows_into_tab(
            svc, sheet_id, tab, key_columns, rows, column_map)
        updated += res["updated"]
        appended += res["appended"]
    return {"updated": updated, "appended": appended,
             "skipped_tabs": skipped}
