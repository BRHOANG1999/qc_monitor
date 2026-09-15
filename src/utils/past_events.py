"""Read past scored BHZ events and locate their EEG ``.mat`` files.

Faithful Python port of the lab MATLAB toolkit's *BHZ Seizure Event Review
Workstation* -- specifically ``src/ui/bhz/BHZCSVParser.m`` (load the scored-
event CSVs, filter to real seizures, extract animal IDs, dedupe) and
``src/ui/bhz/BHZFileDiscovery.m`` (map a CSV row to the recording file on
disk via ``resolveMatFile`` -> ``recursiveFind``).

The MATLAB reviewer searches a single user-picked override root. Here that
is generalized to a list of roots -- the event row's own ``folder`` first,
then configured share roots, then every mounted fixed drive -- because the
lab's recordings are scattered across HDDs (``Z:\\``, ``I:\\``, ``U:\\`` ...)
and network shares that get remounted under different letters. Resolved
locations are cached (via injected callables) so a drive is walked once, not
once per event.

Pure functions + one ``EEGLocator`` class; no Dash, no DB import -- the store
supplies the cache get/put callables and performs the ingest.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import string
import time
from datetime import datetime
from typing import Callable, Iterable

logger = logging.getLogger("qc_monitor.utils.past_events")

# Animal ID = 2-4 uppercase letters + 2-4 digits (e.g. BCH040), skipping the
# electrode/label tokens that share that shape. Mirrors
# BHZCSVParser.extractAnimalIDFromName.
_ANIMAL_RE = re.compile(r"([A-Z]{2,4}\d{2,4})")
_EXCLUDE_PREFIXES: frozenset[str] = frozenset(
    {"CH", "SR", "HZ", "HP", "LP", "FS", "BW"})

# Recording-start timestamp embedded in the filename: ___YYYY_MM_DD__HH_MM_SS
_DT_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")

# CSV onset/landmark sample-index columns -> the marker-schema *_sec keys the
# Training tab grades against (EO/LAS/BO/PID/BB).
_LANDMARK_COLS: dict[str, str] = {
    "EO": "eventeo", "LAS": "eventlas", "BO": "eventbo",
    "PID": "eventpid", "BB": "eventbb",
}

# Two seizures with Peak_Stamp (MATLAB datenum, units = days) within 10 min
# are the same event -- BHZCSVParser.deduplicateByOnsetType.
_CLUSTER_GAP_DAYS = 10.0 / (24.0 * 60.0)

# Directories never worth walking when recursively hunting for a recording.
_SKIP_DIRS: frozenset[str] = frozenset({
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "appdata", "node_modules",
    ".git", "__pycache__", "perflogs", "recovery",
})

_MAX_ROWS = 5_000_000          # NASA Rule 2 loop bounds.
_MAX_GROUPS = 1_000_000


# ===================================================================== #
#  Filename parsing (animal id, channels, datetime)
# ===================================================================== #

def extract_channel_names(filename: str) -> list[str]:
    """Channel name list parsed from a BHZ filename:
    ``YYYYMMDD_expID__ch1_ch2_..___YYYY_MM_DD__HH_MM_SS.mat`` -> [ch1, ch2 ..].
    Port of BHZCSVParser.extractChannelNamesFromFilename."""
    name = os.path.splitext(os.path.basename(str(filename)))[0]
    prefix = _DT_RE.sub("", name)
    # Drop a trailing ``___`` left where the datetime used to be.
    prefix = re.sub(r"_+$", "", prefix.split("___")[0])
    dbl = prefix.find("__")
    if dbl < 0:
        return []
    section = prefix[dbl + 2:]
    return [c for c in section.split("_") if c]


def animal_from_name(name: str) -> str:
    """Animal id from a channel name or filename, or "" if none matches."""
    for cand in _ANIMAL_RE.findall(str(name).upper()):
        letters = re.match(r"^[A-Z]+", cand)
        if letters and letters.group(0) in _EXCLUDE_PREFIXES:
            continue
        return cand
    return ""


def animal_for_event(filename: str, channel: int | None) -> str:
    """Animal id for one event: index the filename's channel list by the
    1-based ``Channel`` column; fall back to scanning the whole filename.
    Port of BHZCSVParser.extractAnimalIDsFromColumn."""
    names = extract_channel_names(filename)
    if channel is not None and names and 1 <= channel <= len(names):
        got = animal_from_name(names[channel - 1])
        if got:
            return got
    return animal_from_name(filename)


def file_datetime(filename: str) -> datetime | None:
    """Recording-start datetime parsed from the filename, or None."""
    m = _DT_RE.search(str(filename))
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return None


# ===================================================================== #
#  CSV row parsing + scored-event loading
# ===================================================================== #

def _to_float(val) -> float | None:
    """Parse a CSV cell to float; NaN / empty / non-numeric -> None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def _to_int(val) -> int | None:
    f = _to_float(val)
    return int(f) if f is not None else None


def _seizure_type(onset: str, comment_parts: list[str]) -> str | None:
    """LVF/HYP from the new-format ``Onset`` column or the legacy pipe-
    delimited ``Comment`` ("HYP | EEG-ONSET | ...").

    The lab writes low-voltage onsets as "LVHF" (Low-Voltage High-Frequency);
    the app's canonical label is "LVF" (Low-Voltage Fast) -- the SAME onset
    pattern. "LVHF" does not contain the substring "LVF" (the H splits it), so
    without an explicit case it parsed to None -> stored as no type -> rendered
    "?" AND mis-graded a student who correctly picked LVF. Map every low-voltage
    spelling to LVF."""
    hay = onset or (comment_parts[0] if comment_parts else "")
    up = hay.upper()
    if "LVHF" in up or "LVF" in up or "LVH" in up:
        return "LVF"
    if "HYP" in up:
        return "HYP"
    return None


def _has_eeg_onset(*fields: str) -> bool:
    return any("EEG-ONSET" in (f or "").upper() for f in fields)


def parse_row(norm: dict, source_csv: str) -> dict | None:
    """One normalized (lowercased-key) CSV row -> a scored-event dict, or
    None when it isn't a confirmed seizure. Matches BHZCSVParser's filters:
    Peak_Index present AND Score > 0."""
    if _to_float(norm.get("peak_index")) is None:
        return None
    score = _to_float(norm.get("score"))
    if score is None or score <= 0:
        return None
    filename = (norm.get("filename") or "").strip()
    if not filename:
        return None
    comment = (norm.get("comment") or "").strip()
    parts = [p.strip() for p in comment.split("|")] if "|" in comment else []
    channel = _to_int(norm.get("channel"))
    ev = {
        "folder": (norm.get("folder") or "").strip(),
        "filename": filename,
        "fs": _to_float(norm.get("fs")) or 0.0,
        "channel": channel,
        "animal": animal_for_event(filename, channel),
        "type": _seizure_type((norm.get("onset") or "").strip(), parts),
        "racine": int(score),
        "peak_stamp": _to_float(norm.get("peak_stamp")),
        "peak_index": _to_float(norm.get("peak_index")),
        "eeg_onset": _has_eeg_onset(norm.get("onset"),
                                     norm.get("onsetcomment"), comment),
        "source_csv": source_csv,
    }
    for lm, col in _LANDMARK_COLS.items():
        ev[f"{lm}_idx"] = _to_float(norm.get(col))
    return ev


def _read_one_csv(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="",
                  errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                assert i < _MAX_ROWS, "CSV row scan runaway"
                norm = {(k or "").strip().lower(): v for k, v in row.items()}
                ev = parse_row(norm, os.path.basename(path))
                if ev is not None:
                    out.append(ev)
    except (OSError, csv.Error) as e:
        logger.warning("past_events: failed to read %s: %s", path, e)
    return out


def load_scored_events(base_dir: str) -> list[dict]:
    """Every confirmed-seizure event across all CSVs in *base_dir*
    (unsorted, not yet deduped). Empty list if the dir is missing."""
    assert isinstance(base_dir, str), "base_dir must be a str"
    if not base_dir or not os.path.isdir(base_dir):
        logger.info("past_events: CSV dir not found: %s", base_dir)
        return []
    out: list[dict] = []
    names = sorted(n for n in os.listdir(base_dir)
                   if n.lower().endswith(".csv"))
    for i, name in enumerate(names):
        assert i < _MAX_GROUPS, "CSV file scan runaway"
        out.extend(_read_one_csv(os.path.join(base_dir, name)))
    logger.info("past_events: %d scored events across %d CSVs",
                len(out), len(names))
    return out


def parse_no_event_row(norm: dict) -> dict | None:
    """A normalized CSV row that records a recording with NO seizure
    (Peak_Index unset -- the lab's "No events in file" rows) -> a recording
    meta dict, or None. These are the negatives stage-1 detection needs."""
    if _to_float(norm.get("peak_index")) is not None:
        return None                      # has a peak -> a scored event row
    filename = (norm.get("filename") or "").strip()
    if not filename:
        return None
    channel = _to_int(norm.get("channel"))
    return {"folder": (norm.get("folder") or "").strip(),
            "filename": filename,
            "fs": _to_float(norm.get("fs")) or 0.0,
            "channel": channel,
            "animal": animal_for_event(filename, channel)}


def load_no_event_recordings(base_dir: str,
                             exclude_keys: set | None = None) -> list[dict]:
    """Distinct recordings that have a 'No events' row and are NOT in
    *exclude_keys* (the seizure files). One meta per (folder, filename),
    first row wins. These become has_seizure=False practice negatives."""
    assert isinstance(base_dir, str), "base_dir must be a str"
    if not base_dir or not os.path.isdir(base_dir):
        return []
    exclude = exclude_keys or set()
    by_key: dict[tuple[str, str], dict] = {}
    names = sorted(n for n in os.listdir(base_dir)
                   if n.lower().endswith(".csv"))
    for i, name in enumerate(names):
        assert i < _MAX_GROUPS, "no-event CSV scan runaway"
        for meta in _read_no_event_rows(os.path.join(base_dir, name)):
            key = (meta["folder"], meta["filename"])
            if key in exclude or key in by_key:
                continue
            by_key[key] = meta
    logger.info("past_events: %d no-event recordings (negatives)",
                len(by_key))
    return list(by_key.values())


def _read_no_event_rows(path: str) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="",
                  errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                assert i < _MAX_ROWS, "no-event row scan runaway"
                norm = {(k or "").strip().lower(): v for k, v in row.items()}
                meta = parse_no_event_row(norm)
                if meta is not None:
                    out.append(meta)
    except (OSError, csv.Error) as e:
        logger.warning("past_events: failed to read %s: %s", path, e)
    return out


# ===================================================================== #
#  Dedup + grouping + markers
# ===================================================================== #

def _flush_cluster(cluster: list[dict], out: list[dict]) -> None:
    """Keep one row per cluster, preferring an EEG-ONSET row."""
    if not cluster:
        return
    eeg = [e for e in cluster if e.get("eeg_onset")]
    out.append(eeg[0] if eeg else cluster[0])


def dedup_events(events: list[dict]) -> list[dict]:
    """Collapse duplicate scorings of the same seizure: cluster by
    (animal, filename) within a 10-min Peak_Stamp gap, prefer EEG-ONSET.
    Port of BHZCSVParser.deduplicateByOnsetType."""
    assert isinstance(events, list), "events must be a list"
    if not events:
        return []
    ordered = sorted(
        events,
        key=lambda e: (e["animal"], e["filename"], e.get("peak_stamp") or 0.0))
    out: list[dict] = []
    cluster: list[dict] = []
    prev: dict | None = None
    for i, e in enumerate(ordered):
        assert i < _MAX_ROWS, "dedup scan runaway"
        if prev is not None:
            same = (e["animal"] == prev["animal"]
                    and e["filename"] == prev["filename"])
            ps, pps = e.get("peak_stamp"), prev.get("peak_stamp")
            close = (ps is not None and pps is not None
                     and abs(ps - pps) < _CLUSTER_GAP_DAYS)
            if not (same and close):
                _flush_cluster(cluster, out)
                cluster = []
        cluster.append(e)
        prev = e
    _flush_cluster(cluster, out)
    logger.info("past_events: deduped %d -> %d events", len(events), len(out))
    return out


def group_by_file(events: Iterable[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group events by (folder, filename), preserving first-seen order --
    one recording can hold several seizures."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in events:
        groups.setdefault((e["folder"], e["filename"]), []).append(e)
    return groups


def markers_for_event(ev: dict) -> dict:
    """A scored event -> the Training marker schema (type, racine, and each
    landmark in SECONDS). Sample indices become seconds via fs. When the
    electrographic onset (EventEO) wasn't recorded -- common for behavioral-
    only Racine scores -- EO_sec falls back to the behavioral Peak_Index so
    the event is still time-locatable for stage-2/3 matching."""
    fs = ev.get("fs") or 0.0
    m: dict = {"type": ev.get("type"), "racine": ev.get("racine")}
    for lm in _LANDMARK_COLS:
        idx = ev.get(f"{lm}_idx")
        if lm == "EO" and idx is None:
            idx = ev.get("peak_index")
        m[f"{lm}_sec"] = (idx / fs if (idx is not None and fs > 0) else None)
    return m


# ===================================================================== #
#  EEG file location (recursive, multi-root, cached)
# ===================================================================== #

def available_drive_roots() -> list[str]:
    """Every mounted ``X:\\`` root on this machine (Windows). Unmounted
    letters and errors are skipped, so an offline HDD costs nothing."""
    roots: list[str] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        try:
            if os.path.isdir(root):
                roots.append(root)
        except OSError:
            continue
    return roots


class EEGLocator:
    """Resolve a CSV (folder, filename) to the recording ``.mat`` on disk.

    Cascade (port of BHZFileDiscovery.resolveMatFile, multi-root):
      1. ``folder/filename`` (and ``+ .mat``) -- the path captured at scoring.
      2. cached location from a previous resolve.
      3. recursive walk of each root (the folder's own drive, configured
         roots, then every mounted fixed drive); first match wins.
    A negative result is cached too so an offline file isn't re-walked every
    round; the store may expire negatives on its own clock.
    """

    def __init__(self, search_roots: list[str] | None = None, *,
                 cache_get: Callable[[str], str | None] | None = None,
                 cache_put: Callable[[str, str | None], None] | None = None,
                 auto_drives: bool = True,
                 max_dirs: int = 200_000,
                 max_walk_seconds: float | None = None) -> None:
        assert max_dirs > 0, "max_dirs must be > 0"
        self.roots = [r for r in (search_roots or []) if r]
        self.cache_get = cache_get
        self.cache_put = cache_put
        self.auto_drives = bool(auto_drives)
        self.max_dirs = int(max_dirs)
        # Wall-clock cap for ONE resolve()'s walk across all its roots. None (the
        # default) preserves the historic behaviour: bounded only by max_dirs.
        self.max_walk_seconds = (float(max_walk_seconds)
                                 if max_walk_seconds else None)

    def resolve(self, folder: str, filename: str) -> str | None:
        assert filename, "filename required"
        name = str(filename).strip()
        direct = self._direct(folder, name)
        if direct:
            return direct
        if self.cache_get is not None:
            cached = self.cache_get(name)
            if isinstance(cached, str):
                if cached and os.path.isfile(cached):
                    return cached     # positive hit, still on disk
                if cached == "":
                    return None        # known-missing within the cache TTL
                # else: stale positive -> fall through and re-search
        deadline = (time.monotonic() + self.max_walk_seconds
                    if self.max_walk_seconds else None)
        for root in self._roots_for(folder):
            hit = self._recursive_find(root, name, deadline)
            if hit:
                self._remember(name, hit)
                return hit
            if deadline is not None and time.monotonic() > deadline:
                # Budget spent across the roots walked so far; do NOT cache a
                # negative (the file may sit under an unwalked root).
                return None
        self._remember(name, None)
        return None

    def _remember(self, name: str, path: str | None) -> None:
        if self.cache_put is not None:
            try:
                self.cache_put(name, path)
            except Exception as e:  # noqa: BLE001 -- cache must never break resolve
                logger.debug("eeg location cache_put failed: %s", e)

    @staticmethod
    def _direct(folder: str, name: str) -> str | None:
        folder = (folder or "").strip()
        if not folder:
            return None
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
        if not name.lower().endswith(".mat"):
            cand2 = os.path.join(folder, name + ".mat")
            if os.path.isfile(cand2):
                return cand2
        return None

    def _roots_for(self, folder: str) -> list[str]:
        out: list[str] = []

        def add(r: str | None) -> None:
            if not r or r in out:
                return
            try:
                if os.path.isdir(r):
                    out.append(r)
            except OSError:
                pass

        if folder:
            add(folder)
            drive = os.path.splitdrive(folder)[0]
            if drive:
                add(drive + os.sep)
        for r in self.roots:
            add(r)
        if self.auto_drives:
            for r in available_drive_roots():
                add(r)
        return out

    def _recursive_find(self, root: str, name: str,
                        deadline: float | None = None) -> str | None:
        want = {name}
        if not name.lower().endswith(".mat"):
            want.add(name + ".mat")
        seen = 0
        try:
            walker = os.walk(root)
        except OSError:
            return None
        for dirpath, dirnames, filenames in walker:
            seen += 1
            if seen > self.max_dirs:
                logger.warning("eeg search hit max_dirs in %s", root)
                break
            if deadline is not None and time.monotonic() > deadline:
                logger.warning("eeg search hit max_walk_seconds in %s", root)
                break
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in _SKIP_DIRS
                           and not d.startswith("$")]
            for w in want:
                if w in filenames:
                    return os.path.join(dirpath, w)
        return None


# ===================================================================== #
#  Example assembly + Training ingest orchestration
# ===================================================================== #

def _recorded_path(folder: str, filename: str) -> str:
    """The recording path as captured in the CSV (folder + filename, ``.mat``
    ensured). This is the *best-known* location; the actual file is only
    confirmed/recursively relocated lazily, when an example is loaded."""
    name = filename if filename.lower().endswith(".mat") else filename + ".mat"
    folder = (folder or "").rstrip("/\\")
    return os.path.join(folder, name) if folder else name


def catalog_examples(events: list[dict],
                     progress: Callable[[int, int, str], None] | None
                     = None) -> list[dict]:
    """Assemble one catalog payload per distinct recording from the parsed
    events -- metadata + markers + the *recorded* path, with NO disk access.

    Cataloging is cheap, so the whole scored-event population is available
    for balanced round selection; the expensive recursive EEG search runs
    lazily, only for the handful of files a student actually loads
    (Store.resolve_training_file). *progress* is ``(done, total, filename)``.
    """
    assert isinstance(events, list), "events must be a list"
    groups = group_by_file(events)
    total = len(groups)
    out: list[dict] = []
    for i, ((folder, filename), evs) in enumerate(groups.items()):
        assert i < _MAX_GROUPS, "catalog assembly runaway"
        if progress is not None:
            progress(i, total, filename)
        rep = max(evs, key=lambda e: (e.get("racine") or 0))
        dt = file_datetime(filename)
        rec = _recorded_path(folder, filename)
        out.append({
            "recorded_path": rec,
            "folder": folder,
            "filename": filename,
            "session_dir": os.path.dirname(rec),
            "session_name": os.path.basename(os.path.dirname(rec)),
            "channel_names": extract_channel_names(filename),
            "fs": evs[0].get("fs") or 0.0,
            "animal": rep.get("animal"),
            "chunk_datetime": dt.isoformat() if dt else None,
            "markers": [markers_for_event(e) for e in evs],
            "has_seizure": True,
            "rep_type": rep.get("type"),
            "rep_racine": rep.get("racine"),
            "source_csv": evs[0].get("source_csv"),
            "peak_stamp": evs[0].get("peak_stamp"),
        })
    if progress is not None:
        progress(total, total, "")
    return out


def catalog_no_event_examples(metas: list[dict]) -> list[dict]:
    """Catalog payloads for no-event recordings: has_seizure=False, empty
    markers (the stage-1 detection negatives). No disk access."""
    assert isinstance(metas, list), "metas must be a list"
    out: list[dict] = []
    for i, m in enumerate(metas):
        assert i < _MAX_GROUPS, "no-event catalog runaway"
        folder, filename = m["folder"], m["filename"]
        dt = file_datetime(filename)
        rec = _recorded_path(folder, filename)
        out.append({
            "recorded_path": rec,
            "folder": folder,
            "filename": filename,
            "session_dir": os.path.dirname(rec),
            "session_name": os.path.basename(os.path.dirname(rec)),
            "channel_names": extract_channel_names(filename),
            "fs": m.get("fs") or 0.0,
            "animal": m.get("animal"),
            "chunk_datetime": dt.isoformat() if dt else None,
            "markers": [],
            "has_seizure": False,
            "rep_type": None,
            "rep_racine": None,
            "source_csv": None,
            "peak_stamp": None,
        })
    return out


def make_locator(store, config: dict, max_walk_sec: float | None = None
                 ) -> EEGLocator:
    """An EEGLocator wired to the store's persistent location cache and the
    configured search roots -- used for lazy, on-demand resolution. *max_walk_sec*
    (default None) caps one resolve()'s recursive walk by wall-clock; the path
    self-heal passes its own budget so a single stubborn file can't stall a pass."""
    pe = (config or {}).get("past_events", {}) or {}
    return EEGLocator(
        search_roots(config),
        cache_get=getattr(store, "eeg_location_get", None),
        cache_put=getattr(store, "eeg_location_put", None),
        auto_drives=bool(pe.get("auto_drives", True)),
        max_dirs=int(pe.get("max_search_dirs", 200_000)),
        max_walk_seconds=max_walk_sec)


def bhz_csv_dir(config: dict) -> str:
    """The scored-event CSV directory (reuses the bhz_csv export base)."""
    bhz = (config or {}).get("bhz_csv", {}) or {}
    return str(bhz.get("base_dir") or "")


def search_roots(config: dict) -> list[str]:
    """Configured recursive-search roots: explicit past_events.search_roots
    if set, else the watcher's network-share paths."""
    pe = (config or {}).get("past_events", {}) or {}
    roots = pe.get("search_roots")
    if roots:
        return [str(r) for r in roots if r]
    watch = ((config or {}).get("watch", {}) or {}).get("paths", []) or []
    return [str(r) for r in watch if r]


def import_into_training(store, config: dict,
                         progress: Callable[[int, int, str], None] | None
                         = None) -> int:
    """Catalog past scored events as Training practice examples -- cheap,
    metadata only, NO drive scan. Each event's EEG is located lazily when a
    student loads it (Store.resolve_training_file). Returns recordings
    cataloged. Safe to re-run -- the store upserts by recorded path."""
    assert store is not None, "store required"
    base = bhz_csv_dir(config)
    events = dedup_events(load_scored_events(base))
    seizures = catalog_examples(events, progress=progress)
    # No-event recordings -> stage-1 detection negatives (the lab's "No
    # events in file" rows), excluding any file that also has a seizure.
    seizure_keys = {(e["folder"], e["filename"]) for e in seizures}
    negatives = catalog_no_event_examples(
        load_no_event_recordings(base, seizure_keys))
    n = store.import_historical_examples(seizures + negatives)
    logger.info("past_events: cataloged %d recordings (%d seizure, %d none; "
                "EEG resolved on load)", n, len(seizures), len(negatives))
    return n
