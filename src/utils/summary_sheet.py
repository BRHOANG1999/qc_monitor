"""Auto-populate the 'Summary' tab of the KA-cohort spreadsheet.

Per animal, everything is derived from data QC Monitor already has:

  * KA location / Surgeon / KA (injection) date  -> surgery 'BCH Log' sheet
        (Target 1 / Surgeon / Date of Injection)
  * Time from KA -> start of recording           -> KA date -> min(chunk_datetime)
  * # overt Bh Sz in first 2 weeks of recording  -> seizure_days_per_animal
  * Total recording time                         -> processed_files.duration_sec
  * Total overt Bh Sz                            -> seizure_days_per_animal
  * Overall Bh Sz rate (sz / recording-day)      -> total sz / total recording days

A "Bh Sz" is a scored event carrying BOTH an EEG onset (EO_sec) AND a Racine
score -- the SAME definition Store.seizure_days_per_animal uses for the Overview
seizure card. There is deliberately NO "analyzed"/"reviewed" figure: the DB only
tracks review done in THIS app's Video Review, which under-reports animals
analyzed in the old software, so such a number would read as false zeros.

Rows are matched by MouseID (tolerates the 'BCH' prefix being present or not);
only the DERIVED columns are written, so Comments/Notes and any manual cell is
never blanked. Dry-run by default -- prints the computed table; ``--apply``
writes it to the sheet.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta

from src.utils.version import qc_monitor_version

logger = logging.getLogger("qc_monitor.utils.summary_sheet")

# A "gap" in an animal's recording timeline is a break between consecutive
# recording starts longer than this. Chunks are normally back-to-back hourly,
# so >2h means at least one missing chunk.
_GAP_THRESHOLD_H = 2.0

# The target spreadsheet + tab (the same workbook QC Monitor writes the BHZ
# Daily/Events tabs to). Overridable via config['summary_sheet'].
_DEFAULT_SHEET_ID = "1HwQlqGPERHCOVUcJDzv8yMm-vZzjqfsh25ud7rfWes8"
_DEFAULT_TAB = "Summary"

# Recordings are hourly chunks (median duration_sec is exactly 3600s; 97% are
# within +/-100s), but only ~66% of files have duration_sec computed. Estimate
# the missing ones at one nominal chunk so "total recording time" reflects ALL
# recorded chunks, not just the ones that happened to get a duration written.
_NOMINAL_CHUNK_SEC = 3600.0

# Summary-tab header -> the value key we compute. Columns not listed here
# (MouseID, Comments/Notes) are left untouched.
# The recording electrodes in this cohort (the lab records each animal on the
# SR and/or SLM electrode). Each gets its OWN gap columns because an animal is
# often recorded on one electrode while the other pair is idle, so a per-
# electrode timeline has different gaps than the cumulative animal timeline.
_GAP_ELECTRODES = ("SLM", "SR")

_DERIVED_COLUMNS = {
    "KA location": "ka_location",
    "Surgeon": "surgeon",
    "Time from KA to start of recording": "time_from_ka",
    "# of overt Bh Sz in first 2 weeks": "bh_sz_2wk",
    "Total recording time": "recording_time",
    "Total overt bh sz during that time": "bh_sz_total",
    "Overall Bh SZ rate": "bh_sz_rate",
    # NOTE: no "analyzed"/"reviewed" columns -- the DB only tracks review done in
    # THIS app's Video Review, which under-reports animals analyzed in the old
    # software, so an "analyzed" figure here would be misleading. Total recording
    # time + the seizure rate stand on their own.
    # Validation / provenance columns (each its own cell, ordered easiest to
    # digest first) so a reviewer can independently verify the numbers above.
    "Electrode(s) recorded": "electrodes",
    "Earliest recording": "earliest_date",
    "Latest recording": "latest_date",
    # Cumulative animal timeline: a recording counts once regardless of how many
    # electrodes it carried.
    "Animal coverage %": "coverage_pct",
    "Animal # gaps (>2h)": "gap_count",
    "Animal total gap (h)": "gap_total_h",
    "Animal largest gap (h)": "gap_max_h",
    "Animal smallest gap (h)": "gap_min_h",
}
# Per-electrode gap columns (core 3: coverage %, # gaps, total gap) for each
# electrode, built here so the set stays DRY and extensible.
for _e in _GAP_ELECTRODES:
    _DERIVED_COLUMNS[f"{_e} coverage %"] = f"{_e.lower()}_coverage_pct"
    _DERIVED_COLUMNS[f"{_e} # gaps (>2h)"] = f"{_e.lower()}_gap_count"
    _DERIVED_COLUMNS[f"{_e} total gap (h)"] = f"{_e.lower()}_gap_total_h"
_DERIVED_COLUMNS.update({
    "Earliest file": "earliest_file",
    "Latest file": "latest_file",
    "Last updated": "last_updated",
    "QC Monitor version": "qc_version",
    # Plain-language finalize-readiness verdict for the advisor (see
    # _reliability_flag): 'OK', or 'Check: <reasons>'.
    "Reliability": "reliability",
})

# Thresholds that make a row "Check" (not finalize-ready). Kept as named
# constants so the verdict logic is auditable and easy to tune.
_MIN_COVERAGE_PCT = 80.0       # <80% coverage -> gappy recording


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


def _animals_on_file(channel_names_json: str) -> set:
    """Real animal ids recorded on a file, parsed straight from channel_names
    (e.g. 'BCH040slm' -> 'BCH040'). Independent of session_config.eeg_channels,
    which is NULL for many (esp. older) sessions -- Store._animal_ids_for_config
    needs it and silently returned [] for those, so an animal's early recordings
    were dropped (BCH040's first date read a year late). Filters non-animal
    channels (stimCopy/saline) and label-only 'NULL' (no digit)."""
    from src.utils.animal import is_animal_channel, split_animal_electrode
    try:
        names = json.loads(channel_names_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return set()
    out: set = set()
    for ch in (names if isinstance(names, list) else []):
        if not is_animal_channel(ch):
            continue
        animal, _ = split_animal_electrode(ch)
        if animal and any(c.isdigit() for c in animal):   # real id has a number
            out.add(animal)
    return out


def _canon_electrode(electrode: str | None) -> str:
    """Canonical RECORDING location for a channel-name suffix.

    Lab labels vary: 'SLM', 'RecSLM', and 'stimSRRecSLM' all denote a recording
    on SLM (the token after 'Rec'; any 'stim<X>' prefix marks where stim was
    applied, not where we recorded). Normalize to the recording site so the
    Summary column reads a clean 'SLM, SR' instead of the raw variants. When
    there's no 'Rec' marker the whole suffix IS the recording location.
    """
    if not electrode:
        return ""
    up = str(electrode).upper()
    idx = up.rfind("REC")
    if idx >= 0 and idx + 3 < len(up):
        return up[idx + 3:]
    return up


def _animal_electrodes_on_file(channel_names_json: str) -> dict:
    """{animal_id: set(recording_location)} parsed from a file's channel_names.

    Same parse as _animals_on_file, but keeps the electrode suffix instead of
    discarding it (e.g. 'BCH062SLM' -> {'BCH062': {'SLM'}}), canonicalized to the
    recording location via _canon_electrode, so the caller learns both which
    animals were recorded AND on which electrode(s). Non-animal channels
    (stimCopy/saline) and label-only ids (no digit) are filtered out.
    """
    from src.utils.animal import is_animal_channel, split_animal_electrode
    try:
        names = json.loads(channel_names_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return {}
    out: dict = {}
    for ch in (names if isinstance(names, list) else []):
        if not is_animal_channel(ch):
            continue
        animal, electrode = split_animal_electrode(ch)
        if not (animal and any(c.isdigit() for c in animal)):
            continue
        slot = out.setdefault(animal, set())
        loc = _canon_electrode(electrode)
        if loc:
            slot.add(loc)
    return out


def _update_extrema(slot: dict, dt: datetime, basename: str) -> None:
    """Track earliest/latest (dt, filename) for an animal, in place."""
    assert slot is not None, "slot required"
    assert dt is not None, "dt required"
    if slot["first_dt"] is None or dt < slot["first_dt"]:
        slot["first_dt"] = dt
        slot["first_file"] = basename
    if slot["last_dt"] is None or dt > slot["last_dt"]:
        slot["last_dt"] = dt
        slot["last_file"] = basename


def _recording_gaps(dts: list) -> dict:
    """Gap stats for a recording timeline (animal-cumulative or per-electrode).
    A gap = a break between consecutive recording starts longer than
    _GAP_THRESHOLD_H; its size is the elapsed time minus one nominal chunk (so a
    contiguous hourly run yields ~0). Bounds (first/last) are the min/max of
    *dts* -- pass the full list for the animal, or one electrode's subset.

    Returns {gap_count, gap_total_h, gap_max_h, gap_min_h, coverage_pct}; the
    numeric gap fields are '' when there are no gaps and coverage_pct is '' when
    the timeline is empty (electrode never recorded). Coverage = recorded span
    minus total gap time, over the full span, clamped to [0, 100]%.
    """
    assert isinstance(dts, list), "dts must be a list"
    empty = {"gap_count": 0, "gap_total_h": "", "gap_max_h": "",
             "gap_min_h": "", "coverage_pct": ""}
    if len(dts) < 2:
        # 0 files -> '' coverage (not recorded); 1 file -> spans nothing -> 100%.
        empty["coverage_pct"] = "100%" if len(dts) == 1 else ""
        return empty
    ordered = sorted(dts)
    first_dt, last_dt = ordered[0], ordered[-1]
    gaps: list = []
    nominal_h = _NOMINAL_CHUNK_SEC / 3600.0
    for i in range(len(ordered) - 1):
        delta_h = (ordered[i + 1] - ordered[i]).total_seconds() / 3600.0
        if delta_h > _GAP_THRESHOLD_H:
            gaps.append(max(0.0, delta_h - nominal_h))
    span_h = (last_dt - first_dt).total_seconds() / 3600.0
    total = sum(gaps)
    cov = 100.0 * (span_h - total) / span_h if span_h > 0 else 100.0
    cov = max(0.0, min(100.0, cov))
    return {
        "gap_count": len(gaps),
        "gap_total_h": round(total, 1) if gaps else "",
        "gap_max_h": round(max(gaps), 1) if gaps else "",
        "gap_min_h": round(min(gaps), 1) if gaps else "",
        "coverage_pct": f"{cov:.0f}%",
    }


def _new_stat_slot() -> dict:
    """A per-animal accumulator: recording time plus earliest/latest file, the
    electrode set, every parsed recording datetime (cumulative, for animal-level
    gaps), and per-electrode datetime lists (for per-electrode gaps)."""
    return {"total_sec": 0.0, "first_dt": None,
            "first_file": "", "last_dt": None, "last_file": "",
            "electrodes": set(), "dts": [], "dts_by_elec": {}}


def _recording_stats_per_animal(store) -> dict:
    """{animal: stat-slot} (see _new_stat_slot). Earliest/latest file + all
    recording datetimes + electrode set are tracked alongside recording time.

    A file's duration counts toward EVERY animal recorded on it (channels are
    per-animal). Earliest/latest consider ALL files (even those whose duration
    wasn't computed); total time only sums non-null durations.
    """
    assert store is not None, "store required"
    conn = store._connect()
    try:
        files = conn.execute(
            """SELECT pf.id, pf.duration_sec, pf.chunk_datetime, pf.file_path,
                      sc.channel_names
               FROM processed_files pf
               JOIN session_config sc ON sc.session_dir = pf.session_dir
               WHERE sc.channel_names IS NOT NULL
                 AND pf.chunk_datetime IS NOT NULL""").fetchall()
    finally:
        conn.close()
    out: dict = {}
    for f in files:
        elecs = _animal_electrodes_on_file(f["channel_names"])
        dur = (float(f["duration_sec"]) if f["duration_sec"] is not None
               else _NOMINAL_CHUNK_SEC)   # estimate missing durations at 1 chunk
        dt = _parse_chunk_dt(f["chunk_datetime"])
        basename = os.path.basename(f["file_path"] or "")
        for a, electrodes in elecs.items():
            slot = out.setdefault(a, _new_stat_slot())
            slot["total_sec"] += dur
            slot["electrodes"].update(electrodes)
            if dt:
                slot["dts"].append(dt)          # cumulative animal timeline
                for e in electrodes:            # per-electrode timelines
                    slot["dts_by_elec"].setdefault(e, []).append(dt)
                _update_extrema(slot, dt, basename)
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


def _reliability_flag(r: dict, coverage_pct: str) -> str:
    """Finalize-readiness verdict for one animal: 'OK', or 'Check: <reasons>'.

    Flags a gappy recording (low coverage), which makes the auto Sz/rate numbers
    less trustworthy. Electrode count is NOT a reason -- a single-electrode
    implant can still be reliable and is already visible in the Electrode(s)
    column. (No 'analyzed' reason: review done in the old software isn't tracked
    here, so an analyzed-fraction flag would fire falsely.)
    """
    assert isinstance(r, dict), "stat slot required"
    tot = r["total_sec"]
    if tot <= 0:
        return "Check: no recording"
    reasons: list = []
    cov = None
    if coverage_pct and coverage_pct.endswith("%"):
        try:
            cov = float(coverage_pct[:-1])
        except ValueError:
            cov = None
    if cov is not None and cov < _MIN_COVERAGE_PCT:
        reasons.append(f"{coverage_pct} coverage")
    return "OK" if not reasons else "Check: " + "; ".join(reasons)


def _validation_cols(r: dict, last_updated: str, qc_version: str) -> dict:
    """The per-animal validation / provenance columns (electrodes, earliest &
    latest date + file, cumulative + per-electrode gap stats, run stamp +
    version) from a stat slot *r*."""
    assert isinstance(r, dict), "stat slot required"
    assert last_updated, "last_updated required"
    first_dt, last_dt = r["first_dt"], r["last_dt"]
    gaps = _recording_gaps(r["dts"])            # cumulative animal timeline
    cols = {
        "electrodes": ", ".join(sorted(r["electrodes"])),
        "earliest_date": first_dt.strftime("%Y-%m-%d %H:%M") if first_dt else "",
        "latest_date": last_dt.strftime("%Y-%m-%d %H:%M") if last_dt else "",
        "coverage_pct": gaps["coverage_pct"],
        "gap_count": gaps["gap_count"],
        "gap_total_h": gaps["gap_total_h"],
        "gap_max_h": gaps["gap_max_h"],
        "gap_min_h": gaps["gap_min_h"],
        "earliest_file": r["first_file"],
        "latest_file": r["last_file"],
        "last_updated": last_updated,
        "qc_version": qc_version,
        "reliability": _reliability_flag(r, gaps["coverage_pct"]),
    }
    # Per-electrode gaps: a blank row means the animal was never recorded on
    # that electrode (its pair was used instead).
    for e in _GAP_ELECTRODES:
        eg = _recording_gaps(r["dts_by_elec"].get(e, []))
        key = e.lower()
        cols[f"{key}_coverage_pct"] = eg["coverage_pct"]
        cols[f"{key}_gap_count"] = eg["gap_count"] if eg["coverage_pct"] else ""
        cols[f"{key}_gap_total_h"] = eg["gap_total_h"]
    return cols


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
    # Run-level provenance -- identical on every row written this run.
    last_updated = datetime.now().isoformat(timespec="seconds")
    qc_version = qc_monitor_version()
    rows: list[dict] = []
    for animal in sorted(rec.keys()):
        if any(e in animal.lower() for e in excl):
            continue                            # skip excluded subjects
        r = rec[animal]
        total_h = _fmt_hours(r["total_sec"])
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
        row = {
            "animal": animal,
            "ka_location": meta.get("ka_location", ""),
            "surgeon": meta.get("surgeon", ""),
            "time_from_ka": time_from_ka,
            "bh_sz_2wk": bh_2wk,
            "recording_time": f"{total_h} h",
            "bh_sz_total": bh_total,
            "bh_sz_rate": rate,
            # extras for preview
            "_first_dt": first_dt.date().isoformat() if first_dt else "",
            "_ka_date": ka_date.date().isoformat() if ka_date else "",
        }
        row.update(_validation_cols(r, last_updated, qc_version))
        rows.append(row)
    return rows


def _print_preview(rows: list[dict]) -> None:
    hdr = ("animal", "electrodes", "earliest_date", "latest_date",
           "coverage_pct", "gap_count", "gap_total_h",
           "slm_coverage_pct", "slm_gap_count", "sr_coverage_pct",
           "sr_gap_count", "qc_version")
    print("  ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(r.get(h,'')):>14}" for h in hdr))


def _owned_columns(header: list) -> tuple:
    """Return (id_idx, {col_index: value_key}) for the columns we own, matching
    _DERIVED_COLUMNS headers case-insensitively. *header* must already contain
    every owned column (self-provisioned by the caller)."""
    def _col(name):
        for i, h in enumerate(header):
            if str(h).strip().lower() == name.lower():
                return i
        return -1
    id_idx = _col("MouseID")
    owned = {_col(name): key for name, key in _DERIVED_COLUMNS.items()
             if _col(name) >= 0}
    return id_idx, owned


def _build_row_writes(header, body, by_animal, id_idx, owned, tab) -> tuple:
    """Build the (batch, matched, present, appends, n_cells) write payload:
    a MERGE update for every existing sheet row whose MouseID matches (owned
    columns overwritten, all other cells preserved), plus an appended row for
    every recorded animal not yet present. Pure -- issues no API calls."""
    import src.utils.sheets_write as SW
    batch: list = []                     # {'range': A1, 'values': [merged_row]}
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
        merged = []
        for ci in range(len(header)):
            if ci in owned:
                merged.append(str(r[owned[ci]]))
                n_cells += 1
            else:
                merged.append(existing[ci] if ci < len(existing) else "")
        batch.append({"range": SW._a1_row(tab, r_i + 2), "values": [merged]})
    appends: list = []
    for animal in sorted(by_animal):
        if animal in present:
            continue
        new_row = [""] * len(header)
        if id_idx >= 0:
            new_row[id_idx] = animal
        for ci, key in owned.items():
            new_row[ci] = str(by_animal[animal][key])
        appends.append(new_row)
    return batch, matched, present, appends, n_cells


def write_summary(store, config: dict, *, dry_run: bool = True) -> dict:
    """Compute + (unless dry_run) write the derived columns to the Summary tab,
    matched by MouseID. Only animals already present as rows are updated;
    recorded animals not yet present are appended."""
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
    assert isinstance(header, list), "header must be a list"
    # Self-provision any owned column whose header doesn't exist yet: extend the
    # in-memory header (in place) so the row merge/append size correctly and
    # every owned column resolves; the live sheet's row 1 gets the same new
    # cells written below (skipped in dry-run). Idempotent across runs.
    lower_hdr = {str(h).strip().lower() for h in header}
    missing_headers = [name for name in _DERIVED_COLUMNS
                       if name.lower() not in lower_hdr]
    header.extend(missing_headers)
    id_idx, owned = _owned_columns(header)
    batch, matched, present, appends, n_cells = _build_row_writes(
        header, grid[1:], by_animal, id_idx, owned, tab)
    result = {"rows": rows, "matched": matched, "appended": [r[id_idx] for r
              in appends] if id_idx >= 0 else [], "n_updates": n_cells,
              "dry_run": dry_run, "sheet_id": sheet_id, "tab": tab}
    if dry_run:
        return result
    # Write the new header cells to the live sheet before the body so the owned
    # columns exist when the rows land. Idempotent (appends only truly-missing
    # columns) and never disturbs existing columns / hand-entered cells.
    if missing_headers:
        SW._ensure_header_columns(
            svc, sheet_id, tab, list(_DERIVED_COLUMNS.keys()),
            value_input_option="USER_ENTERED")
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
