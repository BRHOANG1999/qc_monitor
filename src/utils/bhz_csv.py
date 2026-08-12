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

import contextlib
import csv
import hashlib
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path

from src.utils.version import qc_monitor_version

logger = logging.getLogger("qc_monitor.utils.bhz_csv")


# First 44 columns: exact order verified against
# 20260224_BCH040.csv (Plan, Data model section). The 45th,
# ``SoftwareVersion``, is a lab-local addition appended at the
# END so MATLAB ``readtable`` (which reads by header name) keeps
# working and the first-44 contract is untouched.
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
    # Lab-local trailing additions (after the 44-col reference). All are
    # appended at the END so MATLAB readtable + the first-44 contract are
    # untouched. EventEO_WallClock = the EEG/electrographic onset (EventEO)
    # rendered as an absolute wall-clock datetime, for chronic alignment.
    # AUC_Threshold / AUC_Window_s = the sliding-window-AUC screen settings
    # used for this file (the raw-envelope threshold is the Cutoff column);
    # recording both makes each row self-documenting about how it was detected.
    # EO_HourOfDay = the onset's fractional hour of day (0-23.999) so seizures
    # can be binned by circadian time directly, no datetime parsing.
    "EventEO_WallClock", "EO_HourOfDay",
    "AUC_Threshold", "AUC_Window_s",
    "SoftwareVersion",
)

ONSET_LVF = "LVF (Low Voltage Fast Onset)"
ONSET_HYP = "Hyp (Hypersynchronous)"
ONSET_UNDEFINED = "Undefined"
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
    "EO_HourOfDay",
    "AUC_Threshold", "AUC_Window_s",
))

# Scoring cells that distinguish a FULLY-scored event from a partial
# onset-only one. ``write_event_rows`` counts how many of these are
# populated to decide whether a new same-EventEO row is "more complete"
# than the one already on disk, so a later full score upgrades a partial
# onset row in place instead of being blocked by the append dedup.
_COMPLETENESS_COLS: tuple[str, ...] = (
    "Score", "Onset", "OnsetComment", "BehaviorOnsetComment",
    "scorecomment", "EventLAS", "EventBO", "EventPID", "EventBB",
    "Eventstart", "Eventstop", "Roomlight", "VideoQuality", "Light",
)

# One lock serialises ALL day-CSV mutations (append + overwrite), since
# both read-modify-rewrite the same per-(animal, day) files.
_CSV_WRITE_LOCK = threading.Lock()

# Same-host cross-PROCESS lock for the day CSVs. `_CSV_WRITE_LOCK` only serialises
# threads within ONE process, but the split topology has TWO writers: the
# dashboard (Video Review Submit / PI approve) and the daemon
# (needs_scoring_flush timer) both read-modify-write the same {date}_{animal}.csv.
# Without a cross-process lock a concurrent Submit + flush can lose one side's
# appended rows (classic read-modify-write race). Both services run on the same
# host as the same user, so a LOCAL advisory lock file (keyed by the CSV's
# absolute path) coordinates them without depending on network-share lock
# semantics. Best-effort: if the OS lock can't be taken it degrades to the thread
# lock alone (today's behaviour) rather than blocking the write.
_CSV_LOCK_DIR = os.path.join(tempfile.gettempdir(), "qc_bhz_csv_locks")
_CSV_OS_LOCK_TRIES = 3          # NASA Rule 2: bounded. msvcrt LK_LOCK waits ~10 s
                                # per try, so ~30 s max before it degrades.


def _os_lock(fd: int) -> bool:
    """Take an exclusive advisory lock on *fd* (blocking, bounded). True on
    success, False if it could not be acquired within the retry budget."""
    try:
        import msvcrt
    except ImportError:                              # non-Windows (e.g. CI Linux)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
            return True
        except Exception:  # noqa: BLE001
            return False
    for _ in range(_CSV_OS_LOCK_TRIES):
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            return True
        except OSError:
            continue
    return False


def _os_unlock(fd: int) -> None:
    try:
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    except ImportError:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            pass


@contextlib.contextmanager
def _cross_proc_csv_lock(path: Path):
    """Serialise day-CSV read-modify-write ACROSS processes (dashboard vs daemon),
    keyed on the CSV's absolute path. The slow cross-process wait happens here,
    OUTSIDE ``_CSV_WRITE_LOCK``, so it never blocks writes to OTHER day files.
    Degrades to a no-op (thread lock only) if the lock file / OS lock can't be
    obtained -- it never blocks a write outright."""
    fd = None
    try:
        os.makedirs(_CSV_LOCK_DIR, exist_ok=True)
        key = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        fd = os.open(os.path.join(_CSV_LOCK_DIR, key + ".lock"),
                     os.O_RDWR | os.O_CREAT, 0o600)
        _os_lock(fd)
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        fd = None                       # degrade: proceed under the thread lock
    try:
        yield
    finally:
        if fd is not None:
            try:
                _os_unlock(fd)
            finally:
                os.close(fd)


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


def _wall_clock_eo(eo_sec, file_meta: dict, fs: float) -> str:
    """Absolute wall-clock datetime of the EEG onset (EventEO).

    ``EventEO`` is a sample index from the recording start. The recording
    start = ``Peak_Date Peak_Time`` minus ``Peak_Index/fs`` (when a peak
    index is present; the app's review flows set Peak_* to the recording
    chunk start with no peak index, so the offset is 0). Returns "" when
    the onset or the peak timestamp is missing.
    """
    if eo_sec is None or eo_sec == "":
        return ""
    try:
        eo = float(eo_sec)
    except (TypeError, ValueError):
        return ""
    if eo != eo:  # NaN
        return ""
    pdate = (file_meta.get("Peak_Date") or "").strip()
    ptime = (file_meta.get("Peak_Time") or "00:00:00").strip()
    if not pdate:
        return ""
    peak_dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            peak_dt = datetime.strptime(f"{pdate} {ptime}", fmt)
            break
        except ValueError:
            continue
    if peak_dt is None:
        return ""
    pidx = file_meta.get("Peak_Index")
    offset = (float(pidx) / fs) if (pidx not in (None, "") and fs > 0) else 0.0
    wall = peak_dt - timedelta(seconds=offset) + timedelta(seconds=eo)
    return wall.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _eo_hour_of_day(eo_sec, file_meta: dict, fs: float):
    """Fractional hour of day (0-23.999) of the EEG onset, so seizures can be
    binned by circadian time directly. Derived from the same wall-clock the
    ``EventEO_WallClock`` column uses. Returns None (blank cell) when there's
    no onset / no recording timestamp."""
    wall = _wall_clock_eo(eo_sec, file_meta, fs)
    if not wall:
        return None
    try:
        dt = datetime.strptime(wall, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None
    return round(dt.hour + dt.minute / 60.0
                 + (dt.second + dt.microsecond / 1e6) / 3600.0, 4)


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


def _read_existing_rows(csv_path: Path) -> list[dict]:
    """Return every row of *csv_path* as an ordered list of dicts.

    Empty list if the file doesn't exist. Reading full rows (not just the
    ``(filename, EventEO)`` keys) lets ``write_event_rows`` compare an
    existing row's completeness against a new one and upgrade a partial
    onset row in place. On a corrupt/unreadable CSV we log and return [],
    which degrades to the plain-append behaviour (never blows up the save
    path).
    """
    assert isinstance(csv_path, Path), "Path required"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_iter = 100_000  # NASA Rule 2
            rows: list[dict] = []
            for i, row in enumerate(reader):
                assert i < max_iter, "CSV scan runaway"
                rows.append(dict(row))
            return rows
    except Exception as e:
        # Corrupt CSV is the operator's problem; don't blow up
        # the save path -- log and treat as empty.
        logger.warning("Failed to scan existing CSV %s: %s",
                        csv_path, e)
        return []


def _row_completeness(row: dict) -> int:
    """Count of populated scoring cells in *row* (see ``_COMPLETENESS_COLS``).

    Works for both a freshly-built row (native values from ``_event_to_row``)
    and an on-disk row (already-formatted strings from ``csv.DictReader``):
    both are run through ``_fmt_cell``, so an unset cell reads as ``""`` or
    ``"NaN"`` either way and is not counted."""
    assert isinstance(row, dict), "row must be dict"
    n = 0
    for c in _COMPLETENESS_COLS:
        if _fmt_cell(c, row.get(c)) not in ("", "NaN"):
            n += 1
    return n


def _write_all_rows(path: Path, rows: list[dict]) -> None:
    """Atomically (re)write *path* = header + *rows* via a temp file.

    Existing on-disk rows are already formatted strings, so re-running them
    through ``_fmt_cell`` is a no-op and preserves them byte-for-byte; only
    upgraded/new rows carry native values that get formatted here."""
    assert isinstance(path, Path), "Path required"
    tmp = path.with_suffix(path.suffix + ".rewrtmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_fmt_cell(c, row.get(c)) for c in COLUMNS])
    tmp.replace(path)


def _event_to_row(event: dict, file_meta: dict,
                    fs: float) -> dict:
    """Build one event row from a structured event dict."""
    assert isinstance(event, dict), "event must be dict"
    assert isinstance(file_meta, dict), "file_meta must be dict"
    assert fs > 0, "fs > 0"
    etype = event.get("type") or ""
    onset_str = (ONSET_LVF if etype == "LVF"
                  else ONSET_HYP if etype == "HYP"
                  else ONSET_UNDEFINED if etype == "Undefined"
                  else "")
    row: dict = {c: "" for c in COLUMNS}
    row.update(file_meta)
    # Sample-index conversions
    row["EventEO"] = _to_sample_index(event.get("EO_sec"), fs)
    row["EventEO_WallClock"] = _wall_clock_eo(
        event.get("EO_sec"), file_meta, fs)
    row["EO_HourOfDay"] = _eo_hour_of_day(event.get("EO_sec"), file_meta, fs)
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
    row["Light"] = event.get("light")
    row["Mode"] = "manual"
    row["Target"] = file_meta.get("Target") or "LFP"
    row["SoftwareVersion"] = qc_monitor_version()
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
    row["SoftwareVersion"] = qc_monitor_version()
    return row


def _ensure_header(path: Path) -> None:
    """If *path* exists with a header that differs from the current COLUMNS,
    rewrite it in place so newly appended rows stay column-aligned. Old rows
    are preserved (keyed by name); columns added since (e.g. EO_HourOfDay,
    EventEO_WallClock) are blank-filled, dropped columns discarded. No-op for a
    new file or one already on the current schema -- this repairs the schema
    drift that would otherwise spill columns when appending to an older CSV."""
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
    except (OSError, csv.Error, StopIteration):
        return
    if header is None or tuple(header) == COLUMNS:
        return
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            old_rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return
    tmp = path.with_suffix(path.suffix + ".hdrtmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in old_rows:
            writer.writerow([r.get(c, "") for c in COLUMNS])
    tmp.replace(path)


def write_event_rows(csv_path: str | Path,
                       file_meta: dict,
                       events: list[dict],
                       fs: float) -> int:
    """Append (or upgrade) rows for one ``.mat`` file's review.

    *file_meta* must populate at minimum:
      folder, filename, fs, Cutoff, Channel,
      Peak_Index, Peak_Stamp, Peak_Date, Peak_Time,
      Peak_Start, Peak_Stop, Duration.

    *events* is a list of structured event dicts (the JSON
    schema from the plan); empty list -> one "No events" row.

    Dedup is by ``(filename, EventEO)``. A new event whose EventEO is
    already on disk is normally a no-op -- EXCEPT when the new row is more
    complete (more scoring cells populated; see ``_row_completeness``), in
    which case it REPLACES the on-disk row in place. This lets a later full
    score upgrade a partial onset-only row (quick-flag / needs_scoring
    export) instead of being silently blocked by the append dedup. A less-
    or equally-complete row never overwrites a fuller one.

    Returns the number of rows written OR upgraded. Creates the file with
    the header on first write; appends otherwise, or rewrites atomically
    when a row is upgraded.
    """
    assert isinstance(file_meta, dict), "file_meta dict required"
    assert isinstance(events, list), "events list required"
    assert isinstance(fs, (int, float)) and fs > 0, "fs > 0"
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fname = str(file_meta.get("filename") or "")
    with _cross_proc_csv_lock(path), _CSV_WRITE_LOCK:
        # Upgrade a legacy/older-schema CSV to the current COLUMNS before we
        # append, so the new row's extra columns don't spill past the header.
        _ensure_header(path)
        existing_rows = _read_existing_rows(path)
        # (filename, EventEO_str) -> position of the row in existing_rows.
        index: dict[tuple[str, str], int] = {}
        for pos, r in enumerate(existing_rows):
            fn = (r.get("filename") or "").strip()
            if fn:
                eo = (r.get("EventEO") or "").strip()
                index[(fn, eo)] = pos

        to_append: list[dict] = []
        upgraded = 0
        if not events:
            if (fname, "") not in index:
                to_append.append(_no_events_row(file_meta))
        else:
            max_iter = 1024  # NASA Rule 2 cap on events per file
            for i, ev in enumerate(events):
                assert i < max_iter, "too many events for one file"
                row = _event_to_row(ev, file_meta, fs)
                key = (fname, _fmt_cell("EventEO", row["EventEO"]))
                pos = index.get(key)
                if pos is None:
                    to_append.append(row)
                elif _row_completeness(row) > _row_completeness(
                        existing_rows[pos]):
                    existing_rows[pos] = row     # in-place upgrade
                    upgraded += 1
                # else: same-or-less-complete duplicate -> no-op.

        if upgraded == 0:
            # Fast path: pure append (or nothing changed).
            if not to_append:
                return 0
            is_new = not path.exists()
            with path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL,
                                     lineterminator="\n")
                if is_new:
                    writer.writerow(COLUMNS)
                for row in to_append:
                    writer.writerow([_fmt_cell(c, row.get(c))
                                      for c in COLUMNS])
            return len(to_append)

        # An upgrade means we must rewrite the whole file (CSV append can't
        # edit a row in place). Existing rows are preserved verbatim; the
        # upgraded ones already sit in existing_rows.
        _write_all_rows(path, existing_rows + to_append)
        return upgraded + len(to_append)


def build_file_meta(*, folder: str, filename: str,
                      fs: float, cutoff: float,
                      channel: int,
                      peak_index: int | None,
                      peak_stamp: float | None,
                      peak_dt: datetime | None,
                      target: str = "LFP",
                      auc_threshold: float | None = None,
                      auc_window: float | None = None,
                      ) -> dict:
    """Assemble the file-level meta dict.

    Encapsulates the Peak_Date / Peak_Time split and the
    folder-trailing-slash convention from the reference
    (``U:\\FEBRUARY_2026\\<session>\\`` with the trailing ``\\``).

    ``cutoff`` is the raw-envelope (Hilbert) peak threshold -> the ``Cutoff``
    column. ``auc_threshold`` / ``auc_window`` are the sliding-window-AUC
    screen settings; both are file-level, so they repeat on every event row.
    Left None when no AUC screen was applied (-> NaN cell).
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
        "AUC_Threshold": auc_threshold,
        "AUC_Window_s": auc_window,
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
    with _cross_proc_csv_lock(path), _CSV_WRITE_LOCK:
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
