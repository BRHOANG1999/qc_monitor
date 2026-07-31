"""Auto-populate the 'Summary' tab of the KA-cohort spreadsheet.

Per animal, everything is derived from data QC Monitor already has:

  * KA location / Surgeon / KA (injection) date  -> surgery 'BCH Log' sheet
        (Target 1 / Surgeon / Date of Injection)
  * Time from KA -> start of recording           -> KA date -> min(chunk_datetime)
  * # overt Bh Sz in first 2 weeks of recording  -> seizure_days_per_animal
  * Total recording (and analyzed) time          -> processed_files.duration_sec
  * Total overt Bh Sz                            -> seizure_days_per_animal
  * Overall Bh Sz rate (sz / recording-day)      -> total sz / total recording days

A "Bh Sz" is a scored event carrying BOTH an EEG onset (EO_sec) AND a Racine
score -- the SAME definition Store.seizure_days_per_animal uses for the Overview
seizure card. "Analyzed" time = recording that has been dispositioned in Video
Review (any terminal/scored review_state status).

Rows are matched by MouseID (tolerates the 'BCH' prefix being present or not);
only the DERIVED columns are written, so Comments/Notes and any manual cell is
never blanked. Dry-run by default -- prints the computed table; ``--apply``
writes it to the sheet.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger("qc_monitor.utils.summary_sheet")

# The target spreadsheet + tab (the same workbook QC Monitor writes the BHZ
# Daily/Events tabs to). Overridable via config['summary_sheet'].
_DEFAULT_SHEET_ID = "1HwQlqGPERHCOVUcJDzv8yMm-vZzjqfsh25ud7rfWes8"
_DEFAULT_TAB = "Summary"

# review_state statuses that mean "this (file, animal) has been reviewed"
# (counts toward analyzed time). 'claimed' (in progress) and 'abandoned'
# (returned to the pool) do NOT count.
_ANALYZED_STATUSES = {"no_events", "has_events", "needs_scoring",
                      "pending_pi_review", "pi_approved", "pi_flagged"}

# Summary-tab header -> the value key we compute. Columns not listed here
# (MouseID, Comments/Notes) are left untouched.
_DERIVED_COLUMNS = {
    "KA location": "ka_location",
    "Surgeon": "surgeon",
    "Time from KA to start of recording": "time_from_ka",
    "# of overt Bh Sz in first 2 weeks": "bh_sz_2wk",
    "Total recording (and analyzed) time": "recording_time",
    "Total overt bh sz during that time": "bh_sz_total",
    "Overall Bh SZ rate": "bh_sz_rate",
}


def _norm_animal(raw: str) -> str:
    """Canonical animal id: strip spaces, upper, add the 'BCH' prefix when the
    sheet wrote just the number ('111' -> 'BCH111', 'BCH110' -> 'BCH110')."""
    s = str(raw or "").strip().upper().replace(" ", "")
    if not s:
        return ""
    if s.isdigit():
        return f"BCH{s}"
    return s


def _parse_chunk_dt(ts: str | None) -> datetime | None:
    """Parse the embedded recording timestamp 'YYYY_MM_DD__HH_MM_SS' (or ISO)."""
    if not ts:
        return None
    for fmt in ("%Y_%m_%d__%H_%M_%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(ts)[:19], fmt)
        except ValueError:
            continue
    return None


def _recording_stats_per_animal(store) -> dict:
    """{animal: {'total_sec', 'analyzed_sec', 'first_dt' (datetime|None)}}.

    A file's duration counts toward EVERY animal recorded on it (channels are
    per-animal); 'analyzed' additionally requires a reviewed status for that
    (file, animal). Mirrors the file->animal scoping used elsewhere via
    Store._animal_ids_for_config.
    """
    latest = store._sql_latest_row_correlated("rs.file_id")
    conn = store._connect()
    try:
        files = conn.execute(
            """SELECT pf.id, pf.duration_sec, pf.chunk_datetime,
                      sc.channel_names, sc.eeg_channels
               FROM processed_files pf
               JOIN session_config sc ON sc.session_dir = pf.session_dir
               WHERE pf.duration_sec IS NOT NULL
                 AND sc.channel_names IS NOT NULL""").fetchall()
        rev = conn.execute(
            f"""SELECT rs.file_id, rs.animal_id, rs.status
                FROM review_state rs
                WHERE rs.animal_id IS NOT NULL AND {latest}""").fetchall()
    finally:
        conn.close()
    # (file, animal) -> reviewed?
    reviewed: set = set()
    for r in rev:
        if r["status"] in _ANALYZED_STATUSES:
            reviewed.add((int(r["file_id"]), r["animal_id"]))
    out: dict = {}
    for f in files:
        animals = store._animal_ids_for_config(
            f["channel_names"], f["eeg_channels"])
        dur = float(f["duration_sec"] or 0.0)
        dt = _parse_chunk_dt(f["chunk_datetime"])
        for a in animals:
            slot = out.setdefault(
                a, {"total_sec": 0.0, "analyzed_sec": 0.0, "first_dt": None})
            slot["total_sec"] += dur
            if (int(f["id"]), a) in reviewed:
                slot["analyzed_sec"] += dur
            if dt and (slot["first_dt"] is None or dt < slot["first_dt"]):
                slot["first_dt"] = dt
    return out


def _surgery_meta(config: dict) -> dict:
    """{animal: {'ka_location', 'surgeon', 'ka_date' (datetime|None)}} from the
    surgery 'BCH Log' sheet (Target 1 / Surgeon / Date of Injection)."""
    from src.dashboard.tabs.surgeries import (
        _load_sheet_via_api, _resolve_sa_path, _find_column)
    sheets = (config.get("surgeries", {}) or {}).get("sheets", []) or []
    if not sheets:
        return {}
    s = sheets[0]
    sa = _resolve_sa_path(
        config, (config.get("surgeries", {}) or {}).get(
            "service_account_file", ""))
    if not sa:
        return {}
    df = _load_sheet_via_api(s["sheet_id"], s["tab_name"], sa,
                             float(s.get("refresh_minutes", 15)) * 60.0,
                             header_row=int(s.get("header_row", 1)))
    if df is None or df.empty:
        return {}
    c_animal = _find_column(df, ["Animal ID", "animal_id", "animal"])
    c_loc = _find_column(df, ["Target 1", "KA location", "target"])
    c_surg = _find_column(df, ["Surgeon", "surgeon"])
    c_date = _find_column(df, ["Date of Injection", "date of injection",
                               "injection date", "surgery_date"])
    out: dict = {}
    for _, row in df.iterrows():
        a = _norm_animal(row.get(c_animal)) if c_animal else ""
        if not a:
            continue
        out[a] = {
            "ka_location": (str(row.get(c_loc)).strip() if c_loc else "") or "",
            "surgeon": (str(row.get(c_surg)).strip() if c_surg else "") or "",
            "ka_date": _parse_date(row.get(c_date)) if c_date else None,
        }
    return out


def _parse_date(v) -> datetime | None:
    s = str(v or "").strip()
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _fmt_hours(sec: float) -> float:
    return round(sec / 3600.0, 1)


def compute_rows(store, config: dict) -> list[dict]:
    """One dict per animal that has any recording, with the derived summary
    values (plus the raw pieces, for the dry-run preview)."""
    # Same exclusion the Overview seizure card uses (default ['Randles'] --
    # a non-experimental subject); case-insensitive substring match.
    excl = [e.lower() for e in
            ((config.get("overview", {}) or {}).get("exclude_animals")
             or ["Randles"]) if e]
    rec = _recording_stats_per_animal(store)
    surg = _surgery_meta(config)
    szd = store.seizure_days_per_animal()      # {animal: {'YYYY-MM-DD': n}}
    rows: list[dict] = []
    for animal in sorted(rec.keys()):
        if any(e in animal.lower() for e in excl):
            continue                            # skip excluded subjects
        r = rec[animal]
        total_h = _fmt_hours(r["total_sec"])
        analyzed_h = _fmt_hours(r["analyzed_sec"])
        first_dt = r["first_dt"]
        days = szd.get(animal, {})
        bh_total = sum(days.values())
        # first 2 weeks of RECORDING
        bh_2wk = 0
        if first_dt is not None:
            cutoff = (first_dt + timedelta(days=14)).date().isoformat()
            start = first_dt.date().isoformat()
            bh_2wk = sum(n for d, n in days.items() if start <= d <= cutoff)
        meta = surg.get(animal, {})
        ka_date = meta.get("ka_date")
        time_from_ka = ""
        if ka_date and first_dt:
            time_from_ka = f"{(first_dt.date() - ka_date.date()).days} days"
        # rate = seizures per recording-day
        rec_days = r["total_sec"] / 86400.0
        rate = f"{(bh_total / rec_days):.2f} /day" if rec_days > 0 else ""
        rows.append({
            "animal": animal,
            "ka_location": meta.get("ka_location", ""),
            "surgeon": meta.get("surgeon", ""),
            "time_from_ka": time_from_ka,
            "bh_sz_2wk": bh_2wk,
            "recording_time": f"{total_h} h ({analyzed_h} h)",
            "bh_sz_total": bh_total,
            "bh_sz_rate": rate,
            # extras for preview
            "_first_dt": first_dt.date().isoformat() if first_dt else "",
            "_ka_date": ka_date.date().isoformat() if ka_date else "",
        })
    return rows


def _print_preview(rows: list[dict]) -> None:
    hdr = ("animal", "ka_location", "surgeon", "_ka_date", "_first_dt",
           "time_from_ka", "bh_sz_2wk", "recording_time", "bh_sz_total",
           "bh_sz_rate")
    print("  ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r.get(h,'')):>14}" for h in hdr))


def write_summary(store, config: dict, *, dry_run: bool = True) -> dict:
    """Compute + (unless dry_run) write the derived columns to the Summary tab,
    matched by MouseID. Only animals already present as rows are updated."""
    rows = compute_rows(store, config)
    by_animal = {r["animal"]: r for r in rows}
    cfg = config.get("summary_sheet", {}) or {}
    sheet_id = cfg.get("sheet_id", _DEFAULT_SHEET_ID)
    tab = cfg.get("tab_name", _DEFAULT_TAB)
    from src.dashboard.tabs.surgeries import _resolve_sa_path
    sa = _resolve_sa_path(
        config, (config.get("surgeries", {}) or {}).get(
            "service_account_file", ""))
    import src.utils.sheets_write as SW
    svc = SW._sheets_api_rw(sa)
    grid = SW.read_grid(svc, sheet_id, tab, unformatted=True)
    if not grid:
        raise ValueError(f"Summary tab {tab!r} is empty (no header row)")
    header = grid[0]
    body = grid[1:]

    def _col(name):
        for i, h in enumerate(header):
            if str(h).strip().lower() == name.lower():
                return i
        return -1
    id_idx = _col("MouseID")
    # column index -> derived-value key, for the columns we OWN
    owned = {_col(name): key for name, key in _DERIVED_COLUMNS.items()
             if _col(name) >= 0}
    batch = []              # {'range': A1, 'values': [merged_row]}  (updates)
    matched, present, n_cells = [], set(), 0
    for r_i, existing in enumerate(body):
        raw_id = existing[id_idx] if 0 <= id_idx < len(existing) else ""
        animal = _norm_animal(raw_id)
        if animal:
            present.add(animal)
        r = by_animal.get(animal)
        if not r:
            continue
        matched.append(animal)
        # MERGE: our columns get the computed value; every other cell keeps its
        # existing content (Comments/Notes + MouseID are never touched).
        merged = []
        for ci in range(len(header)):
            if ci in owned:
                merged.append(str(r[owned[ci]]))
                n_cells += 1
            else:
                merged.append(existing[ci] if ci < len(existing) else "")
        batch.append({"range": SW._a1_row(tab, r_i + 2), "values": [merged]})
    # APPEND a fresh row for every recorded animal NOT already in the sheet, so
    # the tab covers all animals recorded (past + present). New rows carry the
    # canonical 'BCHxxx' MouseID + our derived columns; other cells blank.
    appends: list[list] = []
    for animal in sorted(by_animal):
        if animal in present:
            continue
        new_row = [""] * len(header)
        if id_idx >= 0:
            new_row[id_idx] = animal
        for ci, key in owned.items():
            new_row[ci] = str(by_animal[animal][key])
        appends.append(new_row)
    result = {"rows": rows, "matched": matched, "appended": [r[id_idx] for r
              in appends] if id_idx >= 0 else [], "n_updates": n_cells,
              "dry_run": dry_run, "sheet_id": sheet_id, "tab": tab}
    if dry_run:
        return result
    if batch:
        svc.spreadsheets().values().batchUpdate(
            spreadsheetId=sheet_id,
            body={"valueInputOption": "USER_ENTERED", "data": batch},
        ).execute()
    if appends:
        svc.spreadsheets().values().append(
            spreadsheetId=sheet_id, range=f"{tab}!A1",
            valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS",
            body={"values": appends},
        ).execute()
    logger.info("summary_sheet: updated %d rows (%s), appended %d (%s)",
                len(batch), matched, len(appends), result["appended"])
    return result


def start_worker(store, config: dict) -> None:
    """Spawn a daemon thread that writes the Summary tab once per interval
    (default nightly). Enabled unless config['summary_sheet']['enabled'] is
    False. Idempotent-ish: the caller (main.py) may call this once. Best-effort:
    a write failure (e.g. a Sheets blip) is logged and retried next cycle,
    never fatal to the daemon.
    """
    cfg = config.get("summary_sheet", {}) or {}
    if cfg.get("enabled", True) is False:
        logger.info("summary_sheet worker disabled by config")
        return
    interval = max(3600.0, float(cfg.get("interval_hours", 24)) * 3600.0)
    startup_delay = float(cfg.get("startup_delay_sec", 300))

    def _run():
        it, mx = 0, 10 ** 9
        time.sleep(startup_delay)       # keep off the boot-contention window
        while True:
            assert it < mx, "summary worker runaway"
            it += 1
            try:
                res = write_summary(store, config, dry_run=False)
                logger.info("summary_sheet: nightly write ok -- %d cells, %s",
                            res["n_updates"], res["matched"])
            except Exception:
                logger.exception("summary_sheet: nightly write failed")
            time.sleep(interval)

    threading.Thread(target=_run, daemon=True,
                     name="qc-summary-sheet").start()
    logger.info("summary_sheet worker started (every %.0f h, +%.0fs delay)",
                interval / 3600.0, startup_delay)


def _main(argv=None) -> int:
    import argparse
    import yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--db", default="data/monitor.db")
    ap.add_argument("--apply", action="store_true",
                    help="write to the sheet (default: dry-run preview)")
    a = ap.parse_args(argv)
    import logging as _l
    _l.basicConfig(level=_l.INFO)
    with open(a.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    from src.db.store import Store
    store = Store(a.db)
    res = write_summary(store, config, dry_run=not a.apply)
    print(f"\nSummary tab: {res['sheet_id']} / {res['tab']}")
    _print_preview(res["rows"])
    print(f"\nupdated existing rows: {res['matched']}")
    print(f"append new rows for:  {res['appended']}")
    if res["dry_run"]:
        print(f"DRY-RUN: would update {res['n_updates']} cells + append "
              f"{len(res['appended'])} rows. Re-run with --apply.")
    else:
        print(f"WROTE {res['n_updates']} cells + appended "
              f"{len(res['appended'])} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
