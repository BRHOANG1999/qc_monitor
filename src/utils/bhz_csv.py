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
import threading
from dataclasses import dataclass
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


# --------------------------------------------------------------- #
# Overwrite mode (Track D of the PI verification plan)
# --------------------------------------------------------------- #

@dataclass
class CsvDiff:
    """What an overwrite would change. ``removed_filenames``
    being non-empty is the trigger for the PI's confirmation
    modal."""
    added_filenames: list[str]
    removed_filenames: list[str]
    modified_filenames: list[str]


_OVERWRITE_LOCK = threading.Lock()


def existing_filenames_in_csv(csv_path: str | Path
                                ) -> dict[str, list[dict]]:
    """Return ``{filename: [row, ...]}`` for every row in
    *csv_path*, grouped by the ``filename`` column. Empty dict
    when the file doesn't exist."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    out: dict[str, list[dict]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_iter = 100_000  # NASA Rule 2
            for i, row in enumerate(reader):
                assert i < max_iter, "csv scan runaway"
                fn = (row.get("filename") or "").strip()
                if not fn:
                    continue
                out.setdefault(fn, []).append(dict(row))
    except Exception as e:
        logger.warning("existing_filenames_in_csv failed: %s", e)
        return {}
    return out


def _row_canonical(row: dict) -> tuple:
    """Return a hashable tuple of the comparable columns.

    Floating-point quirks aside, the canonical row is just
    the formatted cells in COLUMNS order -- this is what the
    on-disk CSV stores."""
    return tuple(_fmt_cell(c, row.get(c)) for c in COLUMNS)


def overwrite_day_csv(csv_path: str | Path,
                       events_by_filename: dict[str, list[dict]],
                       file_meta_by_filename: dict[str, dict],
                       fs: float, *,
                       dry_run: bool = False) -> CsvDiff:
    """Rewrite *csv_path* with the canonical set of approved
    rows.

    *events_by_filename* maps each .mat filename to the list of
    structured events the PI has approved; an empty list ->
    one "No events" row. *file_meta_by_filename* maps the same
    filenames to the file_meta dicts (folder, Channel, etc.).

    Returns a CsvDiff describing which filenames the overwrite
    adds, removes, and modifies relative to the file on disk.
    The PI tab uses this diff to gate the actual write behind
    a confirmation modal when ``removed_filenames`` is
    non-empty.

    ``dry_run=True`` computes + returns the diff without
    writing -- the PI tab uses this to preview the destructive
    overwrite BEFORE asking for confirmation. The same call
    with ``dry_run=False`` is the actual write.

    Thread-safe via a module-level lock so two simultaneous PI
    overwrites can't trample each other.
    """
    assert isinstance(events_by_filename, dict)
    assert isinstance(file_meta_by_filename, dict)
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    path = Path(csv_path)
    with _OVERWRITE_LOCK:
        # Snapshot what's on disk.
        existing = existing_filenames_in_csv(path)
        existing_fns = set(existing.keys())
        new_fns = set(events_by_filename.keys())
        # Build the canonical new rows in deterministic order.
        new_rows: list[dict] = []
        max_iter = 4096
        for i, fn in enumerate(sorted(new_fns)):
            assert i < max_iter, "filename loop runaway"
            meta = file_meta_by_filename.get(fn, {})
            evs = events_by_filename.get(fn, [])
            if not evs:
                new_rows.append(_no_events_row(meta))
                continue
            for ev in evs:
                new_rows.append(_event_to_row(ev, meta, fs))
        # Compute the diff before writing.
        added = sorted(new_fns - existing_fns)
        removed = sorted(existing_fns - new_fns)
        modified: list[str] = []
        for fn in sorted(new_fns & existing_fns):
            old_canonical = [_row_canonical(r)
                              for r in existing.get(fn, [])]
            new_canonical = [_row_canonical(r)
                              for r in new_rows
                              if (r.get("filename") or "")
                                  .strip() == fn]
            if old_canonical != new_canonical:
                modified.append(fn)
        diff = CsvDiff(
            added_filenames=added,
            removed_filenames=removed,
            modified_filenames=modified,
        )
        if dry_run:
            return diff
        # Write atomically: tmp file, then os.replace.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL,
                                  lineterminator="\n")
            writer.writerow(COLUMNS)
            for row in new_rows:
                writer.writerow(
                    [_fmt_cell(c, row.get(c)) for c in COLUMNS])
        # Atomic-ish replace; Windows lacks O_TMPFILE so this
        # is the simplest portable approach.
        os.replace(tmp, path)
        return diff
