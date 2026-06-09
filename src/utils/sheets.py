"""Google Sheets I/O helpers shared by dashboard tabs + notifications.

Previously these lived in ``src/dashboard/tabs/surgeries.py`` and were
imported (often by underscore name) from:

* ``src/dashboard/tabs/data_log_xref.py``
* ``src/dashboard/tabs/maintenance.py``
* ``src/dashboard/tabs/cage_index.py``
* ``src/utils/assignments.py``
* ``src/notifications/surgery_digest.py``

So a dashboard *tab* was acting as a shared library, and a non-dashboard
module (``notifications``) was depending on it. Moving these into
``utils`` cuts that cycle: tabs depend on utils, never the other way
around.

The function set is unchanged from the original surgeries.py
implementation -- this module is purely a relocation so existing
callers keep working via the thin re-export shim left behind in
``surgeries.py``.

Two fetch paths:

* ``_load_sheet(url, ttl_sec, header_row=1)`` -- public published-CSV
  path. Hits the lab's "File -> Share -> Publish to web" URL.
* ``_load_sheet_via_api(sheet_id, tab_name, service_account_file,
  ttl_sec, header_row=1)`` -- service-account path used by the
  Maintenance, Data Log and Reviewer Assignments tabs.

Both share the same TTL cache keyed by URL/sheet so a dashboard tick
doesn't re-fetch every tab independently.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from io import BytesIO
from threading import RLock

import pandas as pd
import requests

logger = logging.getLogger("qc_monitor.utils.sheets")

# Service-account lookup cache: SA-path -> built Sheets API client.
# google-api-python-client + google-auth are optional installs, imported
# lazily inside _sheets_api so that the published-CSV path still works
# on machines without the SA dep set.
_SHEETS_API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_sa_service_cache: dict = {}
_sa_service_lock = RLock()

# TTL cache: (epoch_sec_fetched, dataframe_or_None) keyed by source.
# Same cache backs both fetch paths so the warmth survives switching
# between published-CSV and API URLs for the same logical sheet.
_cache: "dict[str, tuple[float, pd.DataFrame | None]]" = {}
_cache_lock = RLock()
_FETCH_TIMEOUT_SEC = 10


# ===================================================================== #
#  Published-CSV path
# ===================================================================== #

def _load_sheet(url: str, ttl_sec: float, header_row: int = 1
                ) -> pd.DataFrame | None:
    """Return a DataFrame for *url*, hitting the cache when fresh.

    *header_row* is 1-indexed; rows above it are skipped so banded
    headers (group label on row 1, real names on row 2) still produce
    a clean frame.
    """
    assert isinstance(url, str) and url, "url required"
    assert ttl_sec >= 0, "ttl_sec must be non-negative"
    assert header_row >= 1, "header_row is 1-indexed"

    cache_key = f"csv:{url}:h{header_row}"
    now = time.time()
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit is not None and (now - hit[0]) < ttl_sec:
            return hit[1]

    try:
        resp = requests.get(url, timeout=_FETCH_TIMEOUT_SEC)
        resp.raise_for_status()
        df = pd.read_csv(BytesIO(resp.content), header=header_row - 1)
    except Exception as e:
        logger.warning("Surgery sheet fetch failed: %s (url=%s)", e, url)
        with _cache_lock:
            stale = _cache.get(cache_key)
        return stale[1] if stale is not None else None

    with _cache_lock:
        _cache[cache_key] = (now, df)
    return df


def _last_fetched(key: str) -> float | None:
    with _cache_lock:
        hit = _cache.get(key)
    return hit[0] if hit is not None else None


def cache_has_entry(key: str) -> bool:
    """True when *key* has already been fetched (warm cache)."""
    with _cache_lock:
        return key in _cache


# ===================================================================== #
#  Service-account API path
# ===================================================================== #

def _sheets_api(service_account_file: str):
    """Lazy-build a Sheets API client keyed by the SA JSON path.

    Cached so we don't re-parse the JSON / re-handshake on every fetch.
    Raises if google-api-python-client / google-auth aren't installed.
    """
    assert service_account_file, "service_account_file required"
    with _sa_service_lock:
        svc = _sa_service_cache.get(service_account_file)
        if svc is not None:
            return svc

    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_service_account_file(
        service_account_file, scopes=_SHEETS_API_SCOPES,
    )
    svc = build("sheets", "v4", credentials=creds, cache_discovery=False)
    with _sa_service_lock:
        _sa_service_cache[service_account_file] = svc
    return svc


def _load_sheet_via_api(sheet_id: str, tab_name: str,
                        service_account_file: str,
                        ttl_sec: float,
                        header_row: int = 1) -> pd.DataFrame | None:
    """Read a sheet via the Sheets API using a service-account key.

    *header_row* is 1-indexed. Rows above it are dropped so two-row
    banded headers (group label / real name) still produce a clean
    frame. Falls back to the previous cached value on network / auth
    errors.
    """
    assert sheet_id and tab_name, "sheet_id and tab_name required"
    assert header_row >= 1, "header_row is 1-indexed"
    cache_key = f"api:{sheet_id}:{tab_name}:h{header_row}"

    now = time.time()
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit is not None and (now - hit[0]) < ttl_sec:
            return hit[1]

    try:
        svc = _sheets_api(service_account_file)
        resp = svc.spreadsheets().values().get(
            spreadsheetId=sheet_id, range=tab_name,
        ).execute()
    except Exception as e:
        logger.warning("Sheets API fetch failed: %s (sheet=%s tab=%s)",
                       e, sheet_id, tab_name)
        with _cache_lock:
            stale = _cache.get(cache_key)
        return stale[1] if stale is not None else None

    values = resp.get("values", [])
    if len(values) < header_row:
        df = pd.DataFrame()
    else:
        header = [str(h) for h in values[header_row - 1]]
        data_rows = values[header_row:]
        # Some rows may be longer than the header (stray cells past the
        # rightmost named column) or shorter (trailing blanks dropped by
        # the API). Normalize to the max width seen, padding header with
        # placeholder names so no data is silently lost.
        max_w = max([len(header)] + [len(r) for r in data_rows])
        if len(header) < max_w:
            header = header + [f"_extra_{i}" for i in
                                range(len(header), max_w)]
        # Disambiguate duplicate column names. Lab sheets often repeat
        # "Notes" / "A/P" / etc. across injection-target blocks, and
        # itertuples()._asdict() returns None for every field when any
        # name repeats. Keep the first occurrence canonical so
        # _find_column still resolves it.
        header = _dedupe(header)
        rows = [r + [""] * (max_w - len(r)) for r in data_rows]
        df = pd.DataFrame(rows, columns=header)

    with _cache_lock:
        _cache[cache_key] = (now, df)
    return df


def _resolve_sa_path(config: dict, raw_path: str) -> str:
    """Make the SA path absolute relative to the project root if needed."""
    if not raw_path:
        return ""
    if os.path.isabs(raw_path):
        return raw_path
    here = os.path.dirname(os.path.abspath(__file__))
    # utils/sheets.py -> project root is ../..
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.normpath(os.path.join(project_root, raw_path))


# ===================================================================== #
#  Column discovery and parsing
# ===================================================================== #

def _norm_colname(name: str) -> str:
    """Normalize a column name for matching: lowercase, strip, collapse
    runs of underscores / dashes / whitespace to a single space."""
    s = str(name).lower().strip()
    out = []
    last_space = False
    for ch in s:
        if ch in (" ", "_", "-", "\t"):
            if not last_space:
                out.append(" ")
                last_space = True
        else:
            out.append(ch)
            last_space = False
    return "".join(out)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column (case-insensitive,
    underscore-/dash-/space-insensitive), else None."""
    if df is None or df.empty:
        return None
    norm_map = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        hit = norm_map.get(_norm_colname(cand))
        if hit is not None:
            return hit
    return None


def _safe_date(value) -> date | None:
    """Parse a heterogeneous date value to a date, else None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if isinstance(ts, datetime):
        return ts.date()
    return ts


def _normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _dedupe(names: list[str]) -> list[str]:
    """Return a copy with duplicates suffixed `.1`, `.2`, ... First
    occurrence of each name is kept canonical."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out
