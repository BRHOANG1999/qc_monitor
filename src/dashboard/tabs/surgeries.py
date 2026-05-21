"""Surgeries tab -- read-only "what's on my plate today" view.

Reads one or more Google Sheets (published as CSV via File -> Share ->
Publish to web) and computes today's required actions for every active
surgery:

* day -1 (implant surgeries only): Pre-op Motrin
* day  0                          : Surgery day
* days +1 / +2 / +3               : Post-op day N

The Sheets stay canonical -- this tab never writes. A small module-level
TTL cache around the CSV fetches keeps the page snappy without hammering
Google's endpoint.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from threading import RLock

import pandas as pd
import requests
from dash import Input, Output, callback_context, dash_table, dcc, html

from src.db.store import Store
from src.dashboard.components import (
    TABLE_STYLE as _SHARED_TABLE_STYLE, empty_state, refresh_bar,
)
from src.dashboard.design import (
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_TERTIARY,
    FONT_SIZE_CAPTION, SPACE_3, SPACE_5,
)

# Google Sheets API client is optional -- only required when a sheet entry
# uses the service-account path. Imported lazily so the published-CSV path
# still works on machines without the package set installed.
_SHEETS_API_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_sa_service_cache: dict = {}  # service_account_file -> Sheets API client
_sa_service_lock = RLock()

logger = logging.getLogger("qc_monitor.dashboard.surgeries")

# Default candidate-name lists. The user's sheets predate this code so we
# can't assume column names; we look for any case-insensitive match.
_DEFAULT_ALIASES = {
    "animal_id":    ["animal_id", "animal", "animal #", "id", "subject"],
    "surgery_date": ["surgery_date", "date", "surgery date", "op date",
                      "date of surgery"],
    "surgery_type": ["surgery_type", "type", "surgery", "procedure"],
    "notes":        ["notes", "comment", "comments", "note"],
}
_DEFAULT_IMPLANT_KEYWORDS = ["implant", "headstage", "headcap", "electrode"]

# (epoch_sec_fetched, dataframe_or_None) keyed by URL
_cache: "dict[str, tuple[float, pd.DataFrame | None]]" = {}
_cache_lock = RLock()
_FETCH_TIMEOUT_SEC = 10


# ===================================================================== #
#  Data path
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
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
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


# ===================================================================== #
#  Compute today's tasks
# ===================================================================== #

def _step_for_delta(delta_days: int, is_implant: bool) -> str | None:
    """Map (today - surgery_date) to the step due today, or None."""
    if delta_days == -1 and is_implant:
        return "Pre-op (Motrin)"
    if delta_days == 0:
        return "Surgery day"
    if 1 <= delta_days <= 3:
        return f"Post-op day {delta_days}"
    return None


def _resolve_date_columns(df: pd.DataFrame, date_columns_cfg: list[dict] | None,
                          aliases: dict) -> list[dict]:
    """Return a list of {column_name, is_implant, event_label} for every
    date column the user wants tracked on this sheet.

    Per-sheet *date_columns_cfg* (if given) wins. Otherwise we fall back
    to a single date column found via the global ``surgery_date`` alias
    list, with ``is_implant`` inferred row-by-row from the surgery-type
    column in the caller. The returned event_label is None in that
    fallback so the caller knows to use the row's surgery-type value.
    """
    out: list[dict] = []
    if date_columns_cfg:
        for entry in date_columns_cfg:
            name = entry.get("name") or entry.get("col")
            if not name or name not in df.columns:
                continue
            out.append({
                "column": name,
                "is_implant": bool(entry.get("is_implant", False)),
                "event_label": entry.get("event_label") or name,
            })
        return out
    col_date = _find_column(df, aliases["surgery_date"])
    if col_date is not None:
        out.append({"column": col_date, "is_implant": None,
                    "event_label": None})
    return out


def _tasks_from_frame(df: pd.DataFrame, sheet_label: str, today: date,
                      aliases: dict, implant_kw: list[str],
                      date_columns_cfg: list[dict] | None = None
                      ) -> tuple[list[dict], int]:
    """Walk a DataFrame and emit task dicts for rows due today.

    A row can contribute multiple events when *date_columns_cfg* lists
    more than one date column (e.g. injection + implant on the same
    animal). Returns ``(tasks, dropped)``; *dropped* counts rows that
    had neither an animal id nor any parseable date.
    """
    assert isinstance(today, date), "today must be a date"
    if df is None or df.empty:
        return [], 0

    col_animal = _find_column(df, aliases["animal_id"])
    col_type = _find_column(df, aliases["surgery_type"])
    col_notes = _find_column(df, aliases["notes"])
    date_cols = _resolve_date_columns(df, date_columns_cfg, aliases)

    if col_animal is None or not date_cols:
        return [], len(df)

    tasks: list[dict] = []
    dropped = 0
    # iterrows preserves the original column labels -- itertuples mangles
    # any name that isn't a valid Python identifier (spaces, "/" , "(s)")
    # to _0, _1, ... which breaks lookups by the real name.
    for _idx, row in df.iterrows():
        animal = _normalize_text(row.get(col_animal))
        if not animal:
            dropped += 1
            continue
        type_str = _normalize_text(row.get(col_type)) if col_type else ""
        notes = _normalize_text(row.get(col_notes)) if col_notes else ""
        any_event = False
        for dc in date_cols:
            d = _safe_date(row.get(dc["column"]))
            if d is None:
                continue
            any_event = True
            if dc["is_implant"] is None:
                is_implant = any(k in type_str.lower() for k in implant_kw)
            else:
                is_implant = dc["is_implant"]
            step = _step_for_delta((today - d).days, is_implant)
            if step is None:
                continue
            tasks.append({
                "step": step,
                "animal": animal,
                "type": dc["event_label"] or type_str or "—",
                "surgery_date": d.isoformat(),
                "sheet": sheet_label,
                "notes": notes,
            })
        if not any_event:
            dropped += 1
    return tasks, dropped


def _upcoming(df: pd.DataFrame, sheet_label: str, today: date,
              aliases: dict, date_columns_cfg: list[dict] | None = None,
              days_ahead: int = 7) -> list[dict]:
    """Surgeries scheduled in (today, today+days_ahead]."""
    if df is None or df.empty:
        return []
    col_animal = _find_column(df, aliases["animal_id"])
    col_type = _find_column(df, aliases["surgery_type"])
    date_cols = _resolve_date_columns(df, date_columns_cfg, aliases)
    if col_animal is None or not date_cols:
        return []
    horizon = today + timedelta(days=days_ahead)
    out: list[dict] = []
    for _idx, row in df.iterrows():
        animal = _normalize_text(row.get(col_animal))
        if not animal:
            continue
        type_str = _normalize_text(row.get(col_type)) if col_type else ""
        for dc in date_cols:
            d = _safe_date(row.get(dc["column"]))
            if d is None:
                continue
            if today < d <= horizon:
                out.append({
                    "animal": animal,
                    "type": dc["event_label"] or type_str or "",
                    "surgery_date": d.isoformat(),
                    "in_days": (d - today).days,
                    "sheet": sheet_label,
                })
    out.sort(key=lambda x: x["in_days"])
    return out


# ===================================================================== #
#  Rendering
# ===================================================================== #

_GROUP_ORDER = [
    "Pre-op (Motrin)",
    "Surgery day",
    "Post-op day 1",
    "Post-op day 2",
    "Post-op day 3",
]

def _empty_group(label: str) -> html.Div:
    return empty_state(f"No animals due for {label.lower()} today.")


def _task_table(tasks: list[dict]) -> html.Div:
    columns = [
        {"name": "Animal", "id": "animal"},
        {"name": "Type", "id": "type"},
        {"name": "Surgery date", "id": "surgery_date"},
        {"name": "Sheet", "id": "sheet"},
        {"name": "Notes", "id": "notes"},
    ]
    return html.Div([dash_table.DataTable(
        data=tasks, columns=columns, page_size=20,
        **_SHARED_TABLE_STYLE,
    )])


def _upcoming_table(rows: list[dict]) -> html.Div:
    if not rows:
        return empty_state("No surgeries scheduled in the next 7 days.")
    columns = [
        {"name": "Animal", "id": "animal"},
        {"name": "Type", "id": "type"},
        {"name": "Surgery date", "id": "surgery_date"},
        {"name": "In days", "id": "in_days"},
        {"name": "Sheet", "id": "sheet"},
    ]
    return html.Div([dash_table.DataTable(
        data=rows, columns=columns, page_size=20,
        **_SHARED_TABLE_STYLE,
    )])


def _cache_key_for(entry: dict) -> str:
    """Return the cache key string used to track last-fetched time."""
    header_row = int(entry.get("header_row", 1))
    if entry.get("sheet_id") and entry.get("tab_name"):
        return f"api:{entry['sheet_id']}:{entry['tab_name']}:h{header_row}"
    url = entry.get("url", "")
    return f"csv:{url}:h{header_row}" if url else ""


def _footer(sheets_cfg: list[dict], dropped_by_sheet: dict[str, int]
            ) -> html.Div:
    rows = []
    for s in sheets_cfg:
        key = _cache_key_for(s)
        ts = _last_fetched(key) if key else None
        when = (datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                if ts else "never")
        dropped = dropped_by_sheet.get(s.get("label", ""), 0)
        via = "API" if key.startswith("api:") else "CSV" if key.startswith("csv:") else "?"
        msg = f"{s.get('label', key)} ({via}) -- fetched {when}"
        if dropped:
            msg += f" -- {dropped} row(s) dropped (missing date/animal)"
        rows.append(html.Div(msg, style={
            "color": COLOR_TEXT_TERTIARY,
            "fontSize": FONT_SIZE_CAPTION, "marginBottom": "4px",
        }))
    return html.Div(rows, style={"marginTop": SPACE_5})


def _render_panels(today: date, grouped: dict[str, list[dict]],
                   upcoming: list[dict],
                   sheets_cfg: list[dict],
                   dropped_by_sheet: dict[str, int]) -> html.Div:
    section_h = {
        "color": COLOR_TEXT_SECONDARY, "marginTop": SPACE_5,
        "marginBottom": SPACE_3, "fontSize": FONT_SIZE_CAPTION,
        "textTransform": "uppercase", "letterSpacing": "0.5px",
    }
    children = [
        html.H3(f"Today's checklist -- {today.isoformat()}",
                style={"color": COLOR_TEXT_PRIMARY,
                       "marginBottom": SPACE_3}),
    ]
    for group in _GROUP_ORDER:
        children.append(html.H4(group, style=section_h))
        bucket = grouped.get(group, [])
        if bucket:
            children.append(_task_table(bucket))
        else:
            children.append(_empty_group(group))

    children.append(html.H4("Next 7 days", style=section_h))
    children.append(_upcoming_table(upcoming))

    children.append(_footer(sheets_cfg, dropped_by_sheet))
    return html.Div(children)


# ===================================================================== #
#  Public layout + callbacks
# ===================================================================== #

def _surgeries_cfg(config: dict) -> dict:
    return (config or {}).get("surgeries", {}) or {}


def _aliases(config: dict) -> dict:
    overrides = _surgeries_cfg(config).get("column_aliases", {}) or {}
    merged = {k: list(v) for k, v in _DEFAULT_ALIASES.items()}
    for key, vals in overrides.items():
        if key in merged and isinstance(vals, list):
            merged[key] = list(vals) + merged[key]
    return merged


def _implant_kw(config: dict) -> list[str]:
    custom = _surgeries_cfg(config).get("implant_keywords")
    if isinstance(custom, list) and custom:
        return [str(k).lower() for k in custom]
    return list(_DEFAULT_IMPLANT_KEYWORDS)


def layout(store: Store, config: dict | None = None) -> html.Div:
    cfg = _surgeries_cfg(config or {})
    if not cfg.get("enabled", False):
        return html.Div([
            html.H3("Surgeries", style={"color": COLOR_TEXT_PRIMARY}),
            html.P(
                "Surgery tracker is disabled. Enable it in "
                "config.yaml -> surgeries.enabled and add sheet URLs.",
                style={"color": COLOR_TEXT_SECONDARY},
            ),
        ], style={"padding": SPACE_5})

    refresh_min = float(cfg.get("refresh_minutes", 10))
    return html.Div([
        refresh_bar("surgeries-refresh-btn"),
        dcc.Interval(id="surgeries-tick",
                     interval=int(refresh_min * 60_000), n_intervals=0),
        html.Div(id="surgeries-pane"),
    ], style={"padding": SPACE_5})


def register_callbacks(app, store: Store, config: dict | None = None) -> None:
    cfg = _surgeries_cfg(config or {})
    if not cfg.get("enabled", False):
        return

    sheets_cfg = cfg.get("sheets", []) or []
    ttl_sec = float(cfg.get("refresh_minutes", 10)) * 60.0
    aliases = _aliases(config or {})
    implant_kw = _implant_kw(config or {})
    sa_file = _resolve_sa_path(config or {}, cfg.get("service_account_file", ""))

    @app.callback(
        Output("surgeries-pane", "children"),
        Input("surgeries-refresh-btn", "n_clicks"),
        Input("surgeries-tick", "n_intervals"),
    )
    def _render(_n_clicks, _n_intervals):
        # Manual Refresh forces a fresh pull (bypass cache TTL); the
        # interval tick respects the cache so we don't hammer Sheets.
        triggered = (callback_context.triggered_id
                     if callback_context.triggered else None)
        effective_ttl = 0.0 if triggered == "surgeries-refresh-btn" else ttl_sec
        today = date.today()
        grouped: dict[str, list[dict]] = {g: [] for g in _GROUP_ORDER}
        upcoming_all: list[dict] = []
        dropped: dict[str, int] = {}

        for s in sheets_cfg:
            label = s.get("label") or s.get("url") or s.get("sheet_id", "?")
            header_row = int(s.get("header_row", 1))
            date_cols_cfg = s.get("date_columns") or None
            df: pd.DataFrame | None = None

            if s.get("sheet_id") and s.get("tab_name"):
                if not sa_file:
                    logger.warning(
                        "Sheet '%s' uses sheet_id+tab_name but no "
                        "service_account_file is configured", label,
                    )
                    continue
                df = _load_sheet_via_api(
                    s["sheet_id"], s["tab_name"], sa_file, effective_ttl,
                    header_row=header_row,
                )
            elif s.get("url"):
                df = _load_sheet(s["url"], effective_ttl,
                                 header_row=header_row)
            else:
                logger.warning("Sheet '%s' has neither url nor sheet_id", label)
                continue

            tasks, n_drop = _tasks_from_frame(
                df, label, today, aliases, implant_kw,
                date_columns_cfg=date_cols_cfg,
            )
            dropped[label] = n_drop
            for t in tasks:
                grouped.setdefault(t["step"], []).append(t)
            upcoming_all.extend(_upcoming(
                df, label, today, aliases, date_columns_cfg=date_cols_cfg,
            ))

        upcoming_all.sort(key=lambda x: x["in_days"])
        return _render_panels(today, grouped, upcoming_all,
                              sheets_cfg, dropped)
