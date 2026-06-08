"""Append-only writer for the BHZ_DETECTOR-compatible CSV.

Mirrors the lab's existing MATLAB tool's CSV format byte-for-
byte so the downstream analysis pipeline (which reads these
files keyed on the 44-column header) doesn't notice anything
changed. Reference: ``G:\\BHZ\\BHZ_CSV_Exports\\20260224_BCH040.csv``.

Encoding rules verified by reading the reference:
* Numeric columns: unset = literal ``NaN`` (not empty, not
  ``nan``).
* String columns: unset = empty cell. Strings containing
  commas are CSV-quoted via ``csv.QUOTE_MINIMAL``.
* Timestamps stored as **integer sample indices** (e.g.
  ``EventEO=28429513`` at fs=20000 == 1421.5 s). Conversion is
  ``round(time_sec * fs)``.
* Onset stays the full string -- ``LVF (Low Voltage Fast
  Onset)`` / ``Hyp (Hypersynchronous)``.
* ``Mode`` = ``manual`` for event rows; empty for "No events"
  rows.
* Column 33 ``ScoreComment`` stays empty; column 44 lowercase
  ``scorecomment`` carries the reviewer note. This duplicate is
  intentional and mirrors the MATLAB original (do not "fix").

One row per event. Files with N events emit N rows sharing
folder/filename/Peak_*. Files with 0 events emit one "No
events" row so the per-file audit trail is complete.

Idempotency: dedupe by (filename, EventEO). Saving the same
file twice doesn't double its rows; editing an event in place
(same EventEO) is a no-op, so the contract is "delete + re-add"
if a sample-index actually changes (caveat #3 in the plan).
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger("qc_monitor.utils.bhz_csv")


# 44 columns, exact order verified against
# 20260224_BCH040.csv (Plan, Data model section).
COLUMNS: tuple[str, ...] = (
    "folder", "filename", "fs", "Cutoff", "Channel",
    "Peak_Index", "Peak_Stamp", "Peak_Date", "Peak_Time",
    "Peak_Start", "Peak_Stop", "Duration",
    "Eventstart", "Eventstop",
    "EventStim", "EventSS", "EventIF",
    "EventEO", "EventPID", "EventBB", "EventBO", "EventLAS",
    "EventPAs", "EventPAe", "EventCustom",
    "EventPAs_1", "EventPAe_1", "EventPAs_2", "EventPAe_2",
    "EventPAs_3", "EventPAe_3",
    "Score", "ScoreComment", "Onset", "OnsetComment",
    "Roomlight", "VideoQuality", "VideoComment",
    "BehaviorOnsetComment", "Comment", "Light", "Mode", "Target",
    "scorecomment",
)

ONSET_LVF = "LVF (Low Voltage Fast Onset)"
ONSET_HYP = "Hyp (Hypersynchronous)"
NO_EVENTS_COMMENT = "No events in file for current threshold"

# Column groups that are numeric (NaN when unset) vs string
# (empty when unset). Used by ``_fmt_cell`` to pick the right
# unset sentinel.
_NUMERIC_COLS: frozenset[str] = frozenset((
    "fs", "Cutoff", "Channel",
    "Peak_Index", "Peak_Stamp",
    "Peak_Start", "Peak_Stop", "Duration",
    "EventStim", "EventSS", "EventIF",
    "EventEO", "EventPID", "EventBB", "EventBO", "EventLAS",
    "EventPAs", "EventPAe", "EventCustom",
    "EventPAs_1", "EventPAe_1", "EventPAs_2", "EventPAe_2",
    "EventPAs_3", "EventPAe_3",
    "Score", "Light",
))


def _to_sample_index(t_sec: float | None,
                       fs: float) -> int | None:
    """Convert seconds -> integer sample index, or None if t_sec
    is None / NaN."""
    if t_sec is None:
        return None
    try:
        tf = float(t_sec)
    except (TypeError, ValueError):
        return None
    if tf != tf:  # NaN
        return None
    assert fs > 0, "fs must be > 0"
    return int(round(tf * fs))


def _fmt_cell(col: str, val) -> str:
    """Render one cell value with the right unset sentinel.

    Numeric columns get ``NaN`` when unset; strings get empty.
    Floats with finite values keep their natural repr (MATLAB's
    ``num2str`` default).
    """
    if val is None or val == "":
        return "NaN" if col in _NUMERIC_COLS else ""
    if isinstance(val, float):
        if val != val:  # NaN
            return "NaN"
        # MATLAB writes integer-valued floats without a decimal
        # (fs=20000, not 20000.0). Match that for cleanliness.
        if val.is_integer():
            return str(int(val))
        return repr(val)
    return str(val)


def _read_existing_eos(csv_path: Path) -> set[tuple[str, str]]:
    """Return the ``{(filename, EventEO_str)}`` pairs already in
    *csv_path*, used for idempotent append. Empty set if the
    file doesn't exist.
    """
    assert isinstance(csv_path, Path), "Path required"
    if not csv_path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_iter = 100_000  # NASA Rule 2
            for i, row in enumerate(reader):
                assert i < max_iter, "CSV scan runaway"
                fn = (row.get("filename") or "").strip()
                eo = (row.get("EventEO") or "").strip()
                if fn:
                    seen.add((fn, eo))
    except Exception as e:
        # Corrupt CSV is the operator's problem; don't blow up
        # the save path -- log and treat as empty.
        logger.warning("Failed to scan existing CSV %s: %s",
                        csv_path, e)
        return set()
    return seen


def _event_to_row(event: dict, file_meta: dict,
                    fs: float) -> dict:
    """Build one event row from a structured event dict."""
    assert isinstance(event, dict), "event must be dict"
    assert isinstance(file_meta, dict), "file_meta must be dict"
    assert fs > 0, "fs > 0"
    etype = event.get("type") or ""
    onset_str = (ONSET_LVF if etype == "LVF"
                  else ONSET_HYP if etype == "HYP"
                  else "")
    row: dict = {c: "" for c in COLUMNS}
    row.update(file_meta)
    # Sample-index conversions
    row["EventEO"] = _to_sample_index(event.get("EO_sec"), fs)
    row["EventLAS"] = _to_sample_index(event.get("LAS_sec"), fs)
    row["EventBO"] = _to_sample_index(event.get("BO_sec"), fs)
    row["EventPID"] = _to_sample_index(event.get("PID_sec"), fs)
    row["EventBB"] = _to_sample_index(event.get("BB_sec"), fs)
    row["Score"] = event.get("racine")
    row["Onset"] = onset_str
    row["OnsetComment"] = event.get("onset_comment", "")
    row["BehaviorOnsetComment"] = event.get("behavior_comment", "")
    row["scorecomment"] = event.get("score_comment", "")
    row["Roomlight"] = event.get("roomlight", "")
    row["VideoQuality"] = event.get("video_quality", "")
    row["Mode"] = "manual"
    row["Target"] = file_meta.get("Target") or "LFP"
    return row


def _no_events_row(file_meta: dict) -> dict:
    """Row variant for files the reviewer marked as having no
    events (Mode empty, Comment populated, Target=LFP)."""
    assert isinstance(file_meta, dict), "file_meta must be dict"
    row: dict = {c: "" for c in COLUMNS}
    row.update(file_meta)
    row["Comment"] = NO_EVENTS_COMMENT
    row["Mode"] = ""
    row["Target"] = file_meta.get("Target") or "LFP"
    return row


def write_event_rows(csv_path: str | Path,
                       file_meta: dict,
                       events: list[dict],
                       fs: float) -> int:
    """Append rows for one ``.mat`` file's review.

    *file_meta* must populate at minimum:
      folder, filename, fs, Cutoff, Channel,
      Peak_Index, Peak_Stamp, Peak_Date, Peak_Time,
      Peak_Start, Peak_Stop, Duration.

    *events* is a list of structured event dicts (the JSON
    schema from the plan); empty list -> one "No events" row.

    Returns the number of rows actually written (after dedup).
    Creates the file with the header on first write; appends
    otherwise.
    """
    assert isinstance(file_meta, dict), "file_meta dict required"
    assert isinstance(events, list), "events list required"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_eos(path)
    # Always carry the file-level meta into every row.
    is_new = not path.exists()
    rows_to_write: list[dict] = []
    if not events:
        key = (str(file_meta.get("filename") or ""), "")
        if key not in existing:
            rows_to_write.append(_no_events_row(file_meta))
    else:
        max_iter = 1024  # NASA Rule 2 cap on events per file
        for i, ev in enumerate(events):
            assert i < max_iter, "too many events for one file"
            row = _event_to_row(ev, file_meta, fs)
            key = (str(file_meta.get("filename") or ""),
                    _fmt_cell("EventEO", row["EventEO"]))
            if key in existing:
                continue
            rows_to_write.append(row)
            existing.add(key)
    if not rows_to_write:
        return 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL,
                              lineterminator="\n")
        if is_new:
            writer.writerow(COLUMNS)
        for row in rows_to_write:
            writer.writerow([_fmt_cell(c, row.get(c))
                              for c in COLUMNS])
    return len(rows_to_write)


def build_file_meta(*, folder: str, filename: str,
                      fs: float, cutoff: float,
                      channel: int,
                      peak_index: int | None,
                      peak_stamp: float | None,
                      peak_dt: datetime | None,
                      target: str = "LFP",
                      ) -> dict:
    """Assemble the file-level meta dict.

    Encapsulates the Peak_Date / Peak_Time split and the
    folder-trailing-slash convention from the reference
    (``U:\\FEBRUARY_2026\\<session>\\`` with the trailing ``\\``).
    """
    assert filename, "filename required"
    assert fs > 0, "fs > 0"
    folder_norm = folder if folder.endswith(os.sep) else folder + os.sep
    if peak_dt is not None:
        peak_date = peak_dt.date().isoformat()
        peak_time = peak_dt.strftime("%H:%M:%S")
    else:
        peak_date, peak_time = "", ""
    return {
        "folder": folder_norm,
        "filename": filename,
        "fs": int(fs) if float(fs).is_integer() else float(fs),
        "Cutoff": cutoff,
        "Channel": int(channel),
        "Peak_Index": peak_index,
        "Peak_Stamp": peak_stamp,
        "Peak_Date": peak_date,
        "Peak_Time": peak_time,
        "Peak_Start": None,
        "Peak_Stop": None,
        "Duration": None,
        "Target": target,
    }


def resolve_csv_path(base_dir: str | Path,
                       template: str,
                       chunk_date: date,
                       animal: str) -> Path:
    """Build the per-(animal, day) CSV path. ``template`` is a
    format string with ``{date}`` and ``{animal}`` slots."""
    assert isinstance(chunk_date, date), "chunk_date required"
    assert animal, "animal required"
    name = template.format(
        date=chunk_date.strftime("%Y%m%d"), animal=animal,
    )
    return Path(base_dir) / name
