"""KMrecorder operator-log sync.

Pulls the operator-maintained "notes" tab of the KMrecorder Data Log
Google Sheet and upserts each row into ``annotations`` under
``category='operator_log'``. Rows whose ``Session_Filename`` carries the
``___YYYY_MM_DD__HH_MM_SS`` stamp are linked to the matching
``processed_files`` recording (so the note shows up on that file in the
Video tab); rows without a real session (battery changes, software notes)
are kept unlinked but browsable in the Operator Log panel.

Reuses the existing Sheets reader (``src.utils.sheets``) and service
account already configured for ``data_log_xref`` -- no new credentials.
The background sync is a daemon warmer thread cloned from
``src.utils.assignments`` (NASA Rule 2: bounded loop + assert).
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from datetime import datetime

from src.utils.sheets import (
    _load_sheet_via_api, _resolve_sa_path, _find_column, _normalize_text,
)

logger = logging.getLogger("qc_monitor.operator_log")

# Same trailing-timestamp key the data-log cross-ref uses to join the
# Sheet to monitor.db (src/dashboard/tabs/data_log_xref.py).
_TS_RE = re.compile(r"___(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})")

_CATEGORY = "operator_log"

_warmer_lock = threading.Lock()
_warmer_started = False
_last_sync_at = 0.0
_last_sync_stats: dict = {}


def _cfg(config: dict) -> dict:
    return (config or {}).get("operator_log", {}) or {}


def _extract_key(name) -> str | None:
    """Return the YYYY_MM_DD__HH_MM_SS substring of a filename, or None."""
    if not name:
        return None
    m = _TS_RE.search(str(name))
    return m.group(1) if m else None


def _source_key(date: str, time_s: str, operator: str, note: str) -> str:
    """Stable de-dup key for an operator-log row. Identical rows map to
    the same key (idempotent re-sync); an edited note becomes a new row."""
    h = hashlib.md5(note.encode("utf-8")).hexdigest()[:10]
    return f"oplog:{date}|{time_s}|{operator}|{h}"


def _iso_timestamp(date: str, time_s: str) -> str:
    """Best-effort ISO timestamp from the sheet's Date + Time cells.
    Falls back to a raw join so a row is never dropped for a bad clock."""
    raw = f"{date} {time_s}".strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return raw or datetime.now().isoformat()


def _db_file_index(store) -> dict:
    """Map trailing-timestamp key -> (file_id, session_dir) for every
    processed recording, so a log row can be linked to its file."""
    index: dict = {}
    with store.connection() as conn:
        rows = conn.execute(
            "SELECT id, file_path, session_dir FROM processed_files"
        ).fetchall()
    for r in rows:
        key = _extract_key(r["file_path"])
        if key is not None:
            index[key] = (int(r["id"]), r["session_dir"])
    return index


def sync_once(store, config: dict, ttl_sec: float = 0.0) -> dict:
    """Pull the operator-log tab and upsert its rows. Returns stats
    ``{total, synced, linked, ok}``. Never raises -- a bad tab/network
    logs and returns ``ok=False`` so the warmer keeps running."""
    cfg = _cfg(config)
    sheet_id = cfg.get("sheet_id", "")
    tab_name = cfg.get("tab_name", "")
    sa_file = _resolve_sa_path(config, cfg.get("service_account_file", ""))
    stats = {"total": 0, "synced": 0, "linked": 0, "ok": False}
    if not (sheet_id and tab_name and sa_file):
        logger.warning("operator_log sync skipped (sheet not configured)")
        return stats
    try:
        df = _load_sheet_via_api(sheet_id, tab_name, sa_file, ttl_sec)
    except Exception:
        logger.exception("operator_log sheet fetch failed")
        return stats
    if df is None or df.empty:
        stats["ok"] = True
        return stats

    cols = _resolve_columns(df)
    if cols["note"] is None:
        logger.warning("operator_log: no Note column found in tab '%s'",
                        tab_name)
        return stats
    index = _db_file_index(store)
    stats["total"] = len(df)
    for _i, row in df.iterrows():
        if _sync_row(store, row, cols, index):
            stats["synced"] += 1
            if cols["_linked_last"]:
                stats["linked"] += 1
    stats["ok"] = True
    global _last_sync_at, _last_sync_stats
    _last_sync_at = time.time()
    _last_sync_stats = dict(stats)
    logger.info("operator_log synced: %d rows, %d linked",
                 stats["synced"], stats["linked"])
    return stats


def _resolve_columns(df) -> dict:
    """Fuzzy-match the sheet's columns to our fields."""
    return {
        "date": _find_column(df, ["Date"]),
        "time": _find_column(df, ["Time"]),
        "file": _find_column(df, ["Session_Filename", "Filename",
                                   "Session File", "File"]),
        "note": _find_column(df, ["Note_Text", "Note", "Notes",
                                   "Note Text"]),
        "operator": _find_column(df, ["Operator"]),
        "event": _find_column(df, ["Event_Trigger", "Event", "Trigger"]),
        "active": _find_column(df, ["Recording_Active", "Active"]),
        "pc": _find_column(df, ["PC Name", "PC"]),
        "version": _find_column(df, ["Version"]),
        "_linked_last": False,
    }


def _cell(row, col) -> str:
    return _normalize_text(row.get(col)) if col else ""


def _sync_row(store, row, cols, index) -> bool:
    """Upsert one log row. Returns True if it was newly inserted."""
    note = _cell(row, cols["note"])
    if not note:
        return False
    date = _cell(row, cols["date"])
    time_s = _cell(row, cols["time"])
    operator = _cell(row, cols["operator"])
    fname = _cell(row, cols["file"])
    key = _extract_key(fname)
    file_id, session_dir = (index.get(key) or (None, None))
    cols["_linked_last"] = file_id is not None
    import json as _json
    context = _json.dumps({
        "source": "operator_log",
        "operator": operator,
        "pc": _cell(row, cols["pc"]),
        "version": _cell(row, cols["version"]),
        "event": _cell(row, cols["event"]),
        "recording_active": _cell(row, cols["active"]),
        "session_filename": fname,
    })
    try:
        return store.upsert_synced_note(
            source_key=_source_key(date, time_s, operator, note),
            timestamp=_iso_timestamp(date, time_s),
            note=note, category=_CATEGORY,
            session_dir=session_dir, file_id=file_id,
            user_email=operator or None, context=context)
    except Exception:
        logger.exception("operator_log upsert failed for a row")
        return False


# --------------------------------------------------------------------- #
# Background warmer (clone of src/utils/assignments.py)
# --------------------------------------------------------------------- #

def _interval_sec(config: dict) -> float:
    minutes = float(_cfg(config).get("refresh_minutes", 15) or 15)
    return max(60.0, minutes * 60.0)


def start_warmer(config: dict, store) -> None:
    """Spawn the background operator-log sync thread. Idempotent."""
    assert isinstance(config, dict), "config must be a dict"
    cfg = _cfg(config)
    if not cfg or not cfg.get("enabled", False) or not cfg.get("sheet_id"):
        logger.debug("operator_log warmer not started (disabled/unset)")
        return
    global _warmer_started
    with _warmer_lock:
        if _warmer_started:
            return
        _warmer_started = True
    t = threading.Thread(target=_warmer_loop, args=(config, store),
                          daemon=True, name="qc-operator-log-sync")
    t.start()
    logger.info("Started operator-log sync (every %.0f s)",
                 _interval_sec(config))


def _warmer_loop(config: dict, store) -> None:
    """Daemon loop. Bounded per NASA Rule 2 (assert at top)."""
    assert isinstance(config, dict), "config must be a dict"
    interval = _interval_sec(config)
    max_iter = 10 ** 9
    i = 0
    while True:
        assert i < max_iter, "operator_log loop runaway"
        i += 1
        try:
            sync_once(store, config, ttl_sec=0.0)
        except Exception:
            logger.exception("operator_log sync iteration failed; "
                              "retry in %.0f s", interval)
        time.sleep(interval)
