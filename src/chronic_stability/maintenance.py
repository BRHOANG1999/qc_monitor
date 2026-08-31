"""Maintenance events (battery / cage changes) for the BCH111 stability map.

Two sources, per the operator:

* the recorder's software **Notes** tab -- ``Recording_Stop`` /
  ``Recording_Start`` events with free text. These carry the GROUND-TRUTH
  software timestamp of when the recording was actually halted and resumed
  around a change, so they double as the "Notes" section of the report.
* the Maintenance Tracker's per-rig **battery / cage** tabs -- HUMAN-reported
  change times (approximate, NOT ground truth) plus which batteries were
  swapped. BCH111 is on Rig C.

All timestamps are naive local wall-clock, matching the evoked / DB timestamps
used elsewhere. Sheets are read with the shared sheet-sync service account; any
failure (offline, auth, missing dep) degrades to ``available=False`` so the
pipeline never breaks on a maintenance fetch.
"""

from __future__ import annotations

from datetime import datetime

_MAX_ROWS = 100_000         # NASA Rule 2: explicit bound on sheet scans.
_BATTERY_KW = ("batter",)
_CAGE_KW = ("cage", "clean")

# Built-in defaults so the overlay works without a config block. BCH111 is on
# Rig C. Override / disable via config: chronic_stability.maintenance.
_DEFAULTS = {
    "enabled": True,
    "service_account_file": "secrets/sheet-sync-473717-fb1897b8f02f.json",
    "notes": {"sheet_id": "1a7E0t_Go3IXUpJj0AFfq2KSlXkjMrw1KZa1GdIWnoe0",
              "tab": "Notes"},
    "battery": {"sheet_id": "1EyECDE7Uk-a2o9YDtTL3zvIZ6xH-OCfN_ZT7vvB-m4U",
                "tab": "🐱 Rig C Battery Replacement"},
    "cage": {"sheet_id": "1EyECDE7Uk-a2o9YDtTL3zvIZ6xH-OCfN_ZT7vvB-m4U",
             "tab": "🐱 Rig C Cage Cleaning"},
}


def _cfg(config: dict) -> dict:
    """Merge the built-in defaults with any config override."""
    user = (config or {}).get("chronic_stability", {}).get("maintenance", {})
    return {**_DEFAULTS, **(user or {})}


def load_events(config: dict, lo_epoch: float, hi_epoch: float) -> dict:
    """Battery / cage maintenance events in ``[lo_epoch, hi_epoch]``.

    Returns ``{"available": bool, "software": [...], "battery_human": [...],
    "cage_human": [...], "error": str|None}``. Each software event is
    ``{epoch, date_iso, event('stop'|'start'), kind('battery'|'cage'|'both'|
    'other'), text}``; human events are ``{epoch, date_iso, text}``.
    """
    mc = _cfg(config)
    if not mc.get("enabled", False):
        return _empty("disabled in config")
    try:
        from src.utils.sheets import _sheets_api
        svc = _sheets_api(mc["service_account_file"])
        soft = _load_notes(svc, mc["notes"], lo_epoch, hi_epoch)
        bat = _load_human(svc, mc["battery"], lo_epoch, hi_epoch)
        cage = _load_human(svc, mc["cage"], lo_epoch, hi_epoch)
    except Exception as e:                       # noqa: BLE001 - degrade cleanly
        return _empty(f"{type(e).__name__}: {e}")
    return {"available": True, "software": soft, "battery_human": bat,
            "cage_human": cage, "error": None}


def _empty(reason: str) -> dict:
    return {"available": False, "software": [], "battery_human": [],
            "cage_human": [], "error": reason}


def _values(svc, sheet_id: str, tab: str) -> list:
    return svc.spreadsheets().values().get(
        spreadsheetId=sheet_id, range=tab).execute().get("values", [])


def _load_notes(svc, ref: dict, lo: float, hi: float) -> list:
    """Software Recording_Stop/Start events (battery/cage only) in the window."""
    rows = _values(svc, ref["sheet_id"], ref["tab"])
    out = []
    for i, r in enumerate(rows[1:]):
        assert i < _MAX_ROWS, "notes scan runaway"
        ep = _notes_epoch(_col(r, 0), _col(r, 1))
        if ep is None or not (lo <= ep <= hi):
            continue
        kind = _classify(_col(r, 5))
        if kind == "other":
            continue
        out.append({"epoch": ep, "date_iso": _iso(ep),
                    "event": _event(_col(r, 4)), "kind": kind,
                    "text": _col(r, 5).strip()})
    out.sort(key=lambda d: d["epoch"])
    return out


def _load_human(svc, ref: dict, lo: float, hi: float) -> list:
    """Human-reported change times (col 0 = timestamp) in the window."""
    rows = _values(svc, ref["sheet_id"], ref["tab"])
    out = []
    for i, r in enumerate(rows[1:]):
        assert i < _MAX_ROWS, "human scan runaway"
        ep = _sheet_epoch(_col(r, 0))
        if ep is None or not (lo <= ep <= hi):
            continue
        out.append({"epoch": ep, "date_iso": _iso(ep),
                    "text": _row_comment(r)})
    out.sort(key=lambda d: d["epoch"])
    return out


def _col(row: list, i: int) -> str:
    return row[i] if i < len(row) and row[i] is not None else ""


def _row_comment(row: list) -> str:
    """Last non-empty cell (the free-text note), trimmed."""
    for cell in reversed(row):
        if isinstance(cell, str) and cell.strip():
            return cell.strip()
    return ""


def _classify(text: str) -> str:
    t = (text or "").lower()
    bat = any(k in t for k in _BATTERY_KW)
    cage = any(k in t for k in _CAGE_KW)
    if bat and cage:
        return "both"
    if bat:
        return "battery"
    if cage:
        return "cage"
    return "other"


def _event(trigger: str) -> str:
    t = (trigger or "").lower()
    if "stop" in t:
        return "stop"
    if "start" in t:
        return "start"
    return "manual"


def _notes_epoch(date_s: str, time_s: str) -> float | None:
    """Notes 'YYYY-MM-DD' + 'HH:MM:SS' (or 'H:MM' / 'HH:MM') -> epoch."""
    date_s, time_s = (date_s or "").strip(), (time_s or "").strip()
    if not date_s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date_s} {time_s}", fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.strptime(date_s, "%Y-%m-%d").timestamp()
    except ValueError:
        return None


def _sheet_epoch(s: str) -> float | None:
    """Maintenance tab 'M/D/YYYY H:MM:SS' (or without seconds) -> epoch."""
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def figure_lines(events: dict) -> dict:
    """Collapse software STOP events to battery / cage line epochs for the
    figure overlays (one line per interruption onset)."""
    if not events or not events.get("available"):
        return {"battery": [], "cage": []}
    stops = [e for e in events["software"] if e["event"] == "stop"]
    return {
        "battery": [e["epoch"] for e in stops if e["kind"] in ("battery", "both")],
        "cage": [e["epoch"] for e in stops if e["kind"] in ("cage", "both")],
    }
