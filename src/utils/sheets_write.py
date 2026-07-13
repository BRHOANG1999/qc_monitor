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


# Fallback header for a brand-new animal summary tab when there's no existing
# tab to copy from. Column names normalize onto the canonical stat keys (see
# _ALIASES) so map_cells fills them.
_DEFAULT_SUMMARY_HEADER = (
    "Date", "MouseID", "CSV File Name", "Number of Behavioral Events",
    "Max Racine", "Who Completed Analysis", "Recording Location",
    "Channel(s)", "Type of Recording", "More Settings",
)


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


def read_grid(svc, sheet_id: str, tab_name: str,
               unformatted: bool = False) -> list[list]:
    """Return the tab as a list of rows (row 0 = header). The title is
    A1-quoted so cell-like titles ('BCH039') resolve to the whole sheet, not a
    bogus cell range. *unformatted* reads UNFORMATTED_VALUE (dates come back as
    Google serial numbers, not display strings) -- required for a stable key
    match against date-formatted columns."""
    kw = {"spreadsheetId": sheet_id, "range": _q(tab_name)}
    if unformatted:
        kw["valueRenderOption"] = "UNFORMATTED_VALUE"
    resp = svc.spreadsheets().values().get(**kw).execute()
    return resp.get("values", []) or []


def _canon_key(v) -> str:
    """Canonicalize a key cell so a date/datetime matches whether the sheet
    stored it as a Google serial number, an ISO string, or a reformatted
    display value -- e.g. the serial 46168 and the string '2026-05-26' both
    canonicalize to '2026-05-26'. Non-date values fall back to _norm. This is
    what makes the upsert idempotent against date-formatted columns."""
    from datetime import datetime, timedelta
    if isinstance(v, bool):
        return _norm(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if 20000 <= f <= 80000:            # plausible Sheets date-serial window
            dt = datetime(1899, 12, 30) + timedelta(days=f)
            if f.is_integer():
                return dt.strftime("%Y-%m-%d")
            return (dt + timedelta(seconds=0.5)).strftime("%Y-%m-%d %H:%M:%S")
        return _norm(v)
    s = str(v).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                 "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return (dt.strftime("%Y-%m-%d") if fmt == "%Y-%m-%d"
                     else dt.strftime("%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
    return _norm(s)


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


def _col_letter(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA (A1 column label for *n* columns)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def dedupe_tab(svc, sheet_id: str, tab_name: str, key_columns: list[str],
                dry_run: bool = True) -> dict:
    """Collapse rows sharing the same canonical key into ONE (merging the
    first non-empty value per column, keeping first-seen order), removing the
    surplus duplicate rows. Idempotent + safe to re-run. ``dry_run=True``
    computes the effect without writing. Returns
    ``{"tab","rows_before","rows_after","removed"}``."""
    from collections import OrderedDict
    grid = read_grid(svc, sheet_id, tab_name, unformatted=True)
    n_body = max(0, len(grid) - 1)
    if len(grid) < 2:
        return {"tab": tab_name, "rows_before": n_body,
                 "rows_after": n_body, "removed": 0}
    header, body = grid[0], grid[1:]
    ncol = len(header)
    key_idx = _key_index(header, key_columns)
    if any(i < 0 for i in key_idx):
        raise ValueError(
            f"key columns {key_columns} not all in header {header}")
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for vals in body:
        row = (list(vals) + [""] * ncol)[:ncol]
        k = tuple(_canon_key(row[i]) for i in key_idx)
        if k in groups:
            merged = groups[k]
            for i in range(ncol):
                if merged[i] in (None, "") and row[i] not in (None, ""):
                    merged[i] = row[i]
        else:
            groups[k] = row
    new_body = list(groups.values())
    removed = len(body) - len(new_body)
    res = {"tab": tab_name, "rows_before": len(body),
            "rows_after": len(new_body), "removed": removed}
    if dry_run or removed <= 0:
        return res
    last = _col_letter(ncol)
    svc.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f"{_q(tab_name)}!A2:{last}{1 + len(new_body)}",
        valueInputOption="RAW", body={"values": new_body},
    ).execute()
    svc.spreadsheets().values().clear(
        spreadsheetId=sheet_id,
        range=f"{_q(tab_name)}!A{2 + len(new_body)}:{last}{1 + len(body)}",
        body={},
    ).execute()
    return res


def sheet_titles(svc, sheet_id: str) -> set[str]:
    """Existing tab titles in the spreadsheet."""
    meta = svc.spreadsheets().get(spreadsheetId=sheet_id).execute()
    return {s.get("properties", {}).get("title")
            for s in meta.get("sheets", [])}


def upsert_rows_into_tab(svc, sheet_id: str, tab_name: str,
                           key_columns: list[str],
                           stats_rows: list[dict],
                           column_map: dict | None = None,
                           value_input_option: str = "USER_ENTERED") -> dict:
    """Upsert *stats_rows* into one tab, keyed on *key_columns*.

    MERGE semantics on update: only the columns we have a value for are
    written; every other cell keeps its existing content, so columns the
    lab fills by hand (Recording Location, etc.) are never blanked. Keys are
    canonicalized (_canon_key) so date columns match whether stored as a
    serial or a string -- this is what keeps the upsert idempotent.
    ``value_input_option='RAW'`` (used for the Events tabs we own) keeps
    dates/onsets as exact text so they round-trip without reformatting.
    """
    grid = read_grid(svc, sheet_id, tab_name, unformatted=True)
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
            _canon_key(vals[i]) if i < len(vals) else "" for i in key_idx)
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
        ktuple = tuple(_canon_key(cells[i]) for i in key_idx)
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
                valueInputOption=value_input_option,
                body={"values": [merged]},
            ).execute()
            updated += 1
        else:
            appends.append(["" if c is None else c for c in cells])
            appended += 1
    if appends:
        svc.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=_q(tab_name),
            valueInputOption=value_input_option,
            insertDataOption="INSERT_ROWS",
            body={"values": appends},
        ).execute()
    return {"updated": updated, "appended": appended}


def upsert_day_rows(service_account_file: str, sheet_id: str,
                      rows_by_tab: dict, key_columns: list[str],
                      column_map: dict | None = None,
                      create_missing: bool = False,
                      template_tab: str | None = None) -> dict:
    """Upsert per-tab day rows. *rows_by_tab* maps a tab title (the animal's
    MouseID tab) to its list of canonical day-stat dicts.

    When *create_missing* is True a NEW animal's tab is created (header
    copied from *template_tab*, else an existing animal tab, else a canonical
    default) so a new subject gets its own tab automatically. When False the
    missing tab is skipped with a warning (the legacy behavior). Returns
    ``{"updated", "appended", "skipped_tabs", "created_tabs"}``. Raises on
    auth/API error -- the caller wraps this non-fatally.
    """
    assert sheet_id, "sheet_id required"
    if not rows_by_tab:
        return {"updated": 0, "appended": 0,
                 "skipped_tabs": [], "created_tabs": []}
    svc = _sheets_api_rw(service_account_file)
    titles = sheet_titles(svc, sheet_id)
    header_for_new = None
    if create_missing and any(t not in titles for t in rows_by_tab):
        header_for_new = _resolve_summary_header(
            svc, sheet_id, titles, template_tab)
    updated = appended = 0
    skipped: list[str] = []
    created: list[str] = []
    for tab, rows in rows_by_tab.items():
        if tab not in titles:
            if create_missing and header_for_new:
                _create_tab_with_header(svc, sheet_id, tab, header_for_new)
                titles.add(tab)
                created.append(tab)
            else:
                logger.warning("no tab %r in sheet; skipping %d row(s)",
                                tab, len(rows))
                skipped.append(tab)
                continue
        res = upsert_rows_into_tab(
            svc, sheet_id, tab, key_columns, rows, column_map)
        updated += res["updated"]
        appended += res["appended"]
    return {"updated": updated, "appended": appended,
             "skipped_tabs": skipped, "created_tabs": created}


def _create_tab_with_header(svc, sheet_id: str, tab_name: str,
                             header: list[str]) -> None:
    """Create worksheet *tab_name* and write *header* as row 1."""
    svc.spreadsheets().batchUpdate(
        spreadsheetId=sheet_id,
        body={"requests": [
            {"addSheet": {"properties": {"title": tab_name}}}]},
    ).execute()
    svc.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=f"{_q(tab_name)}!A1",
        valueInputOption="USER_ENTERED",
        body={"values": [list(header)]},
    ).execute()


def ensure_tab(svc, sheet_id: str, tab_name: str,
                header: list[str]) -> bool:
    """Ensure *tab_name* exists with *header* as row 1. Creates it when
    missing. Returns True if it was created, False if it already existed."""
    if tab_name in sheet_titles(svc, sheet_id):
        return False
    _create_tab_with_header(svc, sheet_id, tab_name, header)
    return True


def _resolve_summary_header(svc, sheet_id: str, titles,
                             template_tab: str | None) -> list:
    """Header row for a NEWLY-created animal summary tab, so it matches the
    lab's existing tabs: the explicit *template_tab*'s header if given, else
    the first existing tab that has a 'Date' column, else a canonical
    default."""
    if template_tab and template_tab in titles:
        g = read_grid(svc, sheet_id, template_tab)
        if g and g[0]:
            return list(g[0])
    for t in sorted(t for t in titles if t):
        g = read_grid(svc, sheet_id, t)
        if g and g[0] and any(_norm(c) == "date" for c in g[0]):
            return list(g[0])
    return list(_DEFAULT_SUMMARY_HEADER)


def upsert_event_rows(service_account_file: str, sheet_id: str,
                       events_by_tab: dict, key_columns: list[str],
                       header: list[str],
                       column_map: dict | None = None) -> dict:
    """Upsert one row per seizure event into each animal's 'Events' tab.

    *events_by_tab* maps a tab title (e.g. ``"BCH062 Events"``) to a list of
    event-row dicts whose keys are the *header* column names. The tab is
    created with *header* if it doesn't exist yet, then rows are upserted
    keyed on *key_columns* (default (File, Onset) => idempotent, no dupes on
    re-approve). Returns ``{"updated", "appended", "created_tabs"}``. Raises
    on auth/API error -- callers wrap non-fatally.
    """
    assert sheet_id, "sheet_id required"
    if not events_by_tab:
        return {"updated": 0, "appended": 0, "created_tabs": []}
    svc = _sheets_api_rw(service_account_file)
    # Identity map: each header column pulls its own same-named key out of the
    # event-row dict (we own both the header and the dict, so no aliasing).
    cmap = column_map or {h: h for h in header}
    updated = appended = 0
    created: list[str] = []
    for tab, rows in events_by_tab.items():
        if not rows:
            continue
        if ensure_tab(svc, sheet_id, tab, header):
            created.append(tab)
        res = upsert_rows_into_tab(
            svc, sheet_id, tab, key_columns, rows, cmap,
            value_input_option="RAW")
        updated += res["updated"]
        appended += res["appended"]
    return {"updated": updated, "appended": appended,
             "created_tabs": created}
