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
    "animal": ("animal", "animalid", "subject", "mouse", "rat"),
    "n_files": ("files", "nfiles", "numfiles", "recordings"),
    "n_files_with_events": ("fileswithevents", "eventfiles"),
    "n_events": ("events", "nevents", "numevents", "seizures",
                  "nseizures", "seizurecount"),
    "max_racine": ("maxracine", "racine", "maxscore"),
    "n_during_stim_events": ("duringstim", "duringstimevents",
                              "stimevents", "eventsduringstim"),
    "n_no_event_files": ("noeventfiles", "nnoevents"),
    "n_needs_scoring": ("needsscoring", "flagged"),
    "exported_at": ("exportedat", "exported", "lastexport",
                     "updatedat"),
}


def _norm(s: str) -> str:
    """Lowercase + strip everything but a-z0-9 for fuzzy header match."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


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
    """Return the tab as a list of rows (row 0 = header)."""
    resp = svc.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=tab_name,
    ).execute()
    return resp.get("values", []) or []


def map_row_to_header(header: list[str], stats: dict,
                       column_map: dict | None = None) -> list:
    """Build a row (aligned to *header*) from a canonical *stats* dict.

    For each header cell, fill its value if we can resolve it to a stat
    key -- first via the explicit *column_map* (header-name -> stat-key),
    else via the normalized alias table. Unknown headers -> "".
    """
    # Normalized explicit overrides.
    overrides = {_norm(k): v for k, v in (column_map or {}).items()}
    # Normalized alias -> canonical key.
    alias_to_key: dict[str, str] = {}
    for key, aliases in _ALIASES.items():
        alias_to_key[_norm(key)] = key
        for a in aliases:
            alias_to_key[_norm(a)] = key
    row: list = []
    for col in header:
        n = _norm(col)
        key = overrides.get(n) or alias_to_key.get(n)
        if key is not None and key in stats:
            row.append(stats[key])
        else:
            row.append("")
    return row


def _key_index(header: list[str], key_columns: list[str]) -> list[int]:
    """Column indices in *header* for each name in *key_columns*."""
    norm_header = [_norm(h) for h in header]
    idxs: list[int] = []
    for kc in key_columns:
        nk = _norm(kc)
        idxs.append(norm_header.index(nk) if nk in norm_header else -1)
    return idxs


def _a1_row(tab_name: str, row_1indexed: int) -> str:
    return f"{tab_name}!A{row_1indexed}"


def upsert_day_rows(service_account_file: str, sheet_id: str,
                      tab_name: str, key_columns: list[str],
                      stats_rows: list[dict],
                      column_map: dict | None = None) -> dict:
    """Upsert each canonical *stats_rows* dict into the tab.

    Match on *key_columns* (e.g. ``["Date", "Animal"]``) against the
    sheet's existing key cells; ``update`` an existing row in place or
    ``append`` a new one. Returns ``{"updated": n, "appended": n}``.
    Raises on auth / API error -- caller wraps non-fatally.
    """
    assert sheet_id and tab_name, "sheet_id + tab_name required"
    if not stats_rows:
        return {"updated": 0, "appended": 0}
    svc = _sheets_api_rw(service_account_file)
    grid = read_grid(svc, sheet_id, tab_name)
    if not grid:
        raise ValueError(
            f"sheet tab {tab_name!r} is empty (need a header row)")
    header = grid[0]
    body = grid[1:]
    key_idx = _key_index(header, key_columns)
    if any(i < 0 for i in key_idx):
        raise ValueError(
            f"key columns {key_columns} not all found in header "
            f"{header}")

    # Index existing rows by their key tuple (normalized).
    existing: dict[tuple, int] = {}  # key tuple -> 1-indexed sheet row
    for r, vals in enumerate(body):
        ktuple = tuple(
            _norm(vals[i]) if i < len(vals) else "" for i in key_idx)
        existing[ktuple] = r + 2  # +1 header, +1 to 1-index

    updated = appended = 0
    appends: list[list] = []
    max_iter = len(stats_rows) + 1
    for n, stats in enumerate(stats_rows):
        assert n < max_iter, "upsert runaway"
        row_vals = map_row_to_header(header, stats, column_map)
        ktuple = tuple(
            _norm(row_vals[i]) if i < len(row_vals) else ""
            for i in key_idx)
        sheet_row = existing.get(ktuple)
        if sheet_row is not None:
            svc.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=_a1_row(tab_name, sheet_row),
                valueInputOption="USER_ENTERED",
                body={"values": [row_vals]},
            ).execute()
            updated += 1
        else:
            appends.append(row_vals)
            appended += 1
    if appends:
        svc.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=tab_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={"values": appends},
        ).execute()
    return {"updated": updated, "appended": appended}
