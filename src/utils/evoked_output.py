"""Chronic evoked source: the MATLAB toolkit's ``evokedOutput`` folder.

For an implanted animal the *full* chronic record (back to the implant
baseline) lives as per-recording ``*_evoked.mat`` files in the toolkit's
``evokedOutput`` directory -- not in the rolling monitor DB, which only
retains the recent window it ingests. The Chronic Evoked Analyzer tab
reads those files at **per-evoked-response resolution**: one point per
stimulus, not one per session.

Each ``*_evoked.mat`` is a MATLAB v7.3 (HDF5) file::

    allAnimalResults/<channel>/stimulusTimes            (1, nEpoch)
    allAnimalResults/<channel>/stimulusPeakAmplitudes   (nEpoch, 1)
    allAnimalResults/<channel>/stimulusTroughAmplitudes (nEpoch, 1)
    allAnimalResults/<channel>/evokedData               (nEpoch, nSamp)  # big

``<channel>`` is a full channel name (e.g. ``BCH062SR``); a multi-animal
recording carries several channels. h5py reads datasets lazily, so we
pull only the three tiny scalar arrays and never touch the big
``evokedData`` trace matrix.

Opening an HDF5 file still costs ~0.2 s, so a first pass over an animal's
~1.5k files would stall the UI. We therefore extract the per-epoch
scalars once into a compact sqlite cache (keyed by file mtime); rebuilds
are incremental and re-reads are instant.
"""

from __future__ import annotations

import glob
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta

from src.utils.animal import split_animal_electrode, is_animal_channel

# Default location of the toolkit's evoked output (Windows path). Override
# via ``config.chronic_evoked.evoked_output_dir``.
DEFAULT_EVOKED_DIR = (
    r"D:\code\Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit"
    r"\daqSignalGenerator\evokedOutput"
)
DEFAULT_CACHE_DB = os.path.join("data", "evoked_chronic_cache.db")

# Trailing ``..._YYYY_MM_DD__HH_MM_SS_evoked.mat`` recording timestamp.
_DT_RX = re.compile(
    r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})_evoked\.mat$")
# Animal-id tokens anywhere in the filename (BCH040, BCH062, ...).
_ANIMAL_RX = re.compile(r"(BCH\d+)")

_MAX_FILES = 100000          # NASA Rule 2: explicit loop bound.

# Serialize cache builds: the dashboard fires the chronic callback from
# multiple worker threads (page auto-refresh), and two concurrent builds
# would race on the file_meta INSERT. One builder at a time; the others
# wait, then find the work already done.
_BUILD_LOCK = threading.Lock()


def parse_recording_dt(filename: str) -> datetime | None:
    """Recording datetime from the KMrecorder-style filename, or None."""
    if not filename:
        return None
    m = _DT_RX.search(os.path.basename(filename))
    if not m:
        return None
    try:
        return datetime(*(int(g) for g in m.groups()))
    except (ValueError, TypeError):
        return None


def _add_seconds(rec_iso: str | None, seconds) -> str:
    """ISO recording time + a stim offset (seconds) -> ISO, or '' / rec."""
    if not rec_iso:
        return ""
    if seconds is None:
        return rec_iso
    try:
        base = datetime.fromisoformat(rec_iso)
        return (base + timedelta(seconds=float(seconds))).isoformat()
    except (ValueError, TypeError):
        return rec_iso


def animals_in_filename(filename: str) -> set[str]:
    """Animal-id tokens (``BCH###``) present in *filename*."""
    if not filename:
        return set()
    return set(_ANIMAL_RX.findall(os.path.basename(filename)))


def list_evoked_files(evoked_dir: str) -> list[str]:
    """All ``*_evoked.mat`` files directly in *evoked_dir* (sorted)."""
    assert isinstance(evoked_dir, str) and evoked_dir, "evoked_dir required"
    if not os.path.isdir(evoked_dir):
        return []
    return sorted(glob.glob(os.path.join(evoked_dir, "*_evoked.mat")))


def list_animals(evoked_dir: str) -> list[str]:
    """Sorted animal ids that appear in any evoked filename."""
    seen: set[str] = set()
    files = list_evoked_files(evoked_dir)
    for i, f in enumerate(files):
        assert i < _MAX_FILES, "evoked file count exceeds bound"
        seen |= animals_in_filename(f)
    return sorted(seen)


def read_file_epochs(path: str) -> dict[str, dict]:
    """Per-channel scalar arrays for one ``*_evoked.mat`` (lazy h5py read).

    Returns ``{channel_name: {"times": [...], "peak": [...],
    "trough": [...]}}``. Only the small scalar datasets are read; the big
    ``evokedData`` matrix is never touched. Unreadable files -> ``{}``.
    """
    assert isinstance(path, str) and path, "path required"
    import h5py  # lazy: keeps the dependency off non-chronic code paths.
    out: dict[str, dict] = {}
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return {}
            for ch in list(grp.keys()):
                node = grp.get(ch)
                if node is None or "stimulusTimes" not in node:
                    continue
                times = node["stimulusTimes"][()].ravel().tolist()
                peak = node["stimulusPeakAmplitudes"][()].ravel().tolist()
                trough = node["stimulusTroughAmplitudes"][()].ravel().tolist()
                out[str(ch)] = {"times": times, "peak": peak,
                                "trough": trough}
    except (OSError, KeyError, ValueError):
        return {}
    return out


class ChronicEvokedCache:
    """Per-epoch evoked scalars cached from ``evokedOutput`` into sqlite.

    The cache is animal-agnostic on disk (one row per epoch per channel);
    callers query by animal id. ``ensure_animal`` is incremental -- it
    skips files whose mtime already matches the cached value.
    """

    def __init__(self, evoked_dir: str | None = None,
                 cache_db: str | None = None) -> None:
        self.evoked_dir = evoked_dir or DEFAULT_EVOKED_DIR
        self.cache_db = cache_db or DEFAULT_CACHE_DB
        assert isinstance(self.evoked_dir, str), "evoked_dir must be str"
        assert isinstance(self.cache_db, str), "cache_db must be str"
        d = os.path.dirname(self.cache_db)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        # WAL + a generous busy timeout so the dashboard can read while a
        # warm pass (tools/warm_evoked_cache.py) writes concurrently.
        conn = sqlite3.connect(self.cache_db, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init_schema(self) -> None:
        # epoch rows reference file_meta by integer id (not the long
        # file_path) so neither the column nor its index bloats the cache.
        conn = self._conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_meta (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    mtime     REAL NOT NULL,
                    rec_dt    TEXT,
                    built_at  TEXT
                );
                CREATE TABLE IF NOT EXISTS epoch (
                    file_id       INTEGER NOT NULL,
                    animal        TEXT NOT NULL,
                    electrode     TEXT,
                    channel       TEXT,
                    stim_time_sec REAL,
                    peak          REAL,
                    trough        REAL
                );
                CREATE INDEX IF NOT EXISTS idx_epoch_animal
                    ON epoch(animal);
                CREATE INDEX IF NOT EXISTS idx_epoch_file
                    ON epoch(file_id);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def files_for_animal(self, animal: str) -> list[str]:
        """Evoked files whose filename names *animal* (sorted by time)."""
        if not animal:
            return []
        files = [f for f in list_evoked_files(self.evoked_dir)
                 if animal in animals_in_filename(f)]
        return sorted(files, key=lambda f: (parse_recording_dt(f)
                                            or datetime.min))

    def _needs_build(self, conn, path: str) -> bool:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        row = conn.execute(
            "SELECT mtime FROM file_meta WHERE file_path = ?",
            (path,)).fetchone()
        return row is None or abs(float(row["mtime"]) - mtime) > 1e-6

    def ensure_animal(self, animal: str, progress=None) -> dict:
        """Cache every not-yet-current evoked file for *animal*.

        *progress* (optional) is called ``progress(done, total, path)``
        after each file. Returns ``{"files": n, "built": k}``.
        """
        assert isinstance(animal, str) and animal, "animal required"
        files = self.files_for_animal(animal)
        built = 0
        # One builder at a time across threads (see _BUILD_LOCK). A second
        # caller waits, then re-checks _needs_build and finds nothing to do.
        with _BUILD_LOCK:
            conn = self._conn()
            try:
                for i, path in enumerate(files):
                    assert i < _MAX_FILES, "file loop exceeds bound"
                    if self._needs_build(conn, path):
                        self._build_one(conn, path)
                        conn.commit()   # release the write lock per file so
                        built += 1       # the dashboard can read mid-warm.
                    if progress is not None:
                        progress(i + 1, len(files), path)
            finally:
                conn.close()
        return {"files": len(files), "built": built}

    def _build_one(self, conn, path: str) -> None:
        """(Re)extract one file's epochs into the cache (keyed by file_id)."""
        rec_dt = parse_recording_dt(path)
        rec_iso = rec_dt.isoformat() if rec_dt else ""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        file_id = self._upsert_file(conn, path, mtime, rec_iso)
        conn.execute("DELETE FROM epoch WHERE file_id = ?", (file_id,))
        channels = read_file_epochs(path)
        rows = []
        for ch, arr in channels.items():
            if not is_animal_channel(ch):
                continue
            animal, electrode = split_animal_electrode(ch)
            rows.extend(self._epoch_rows(file_id, animal, electrode, ch, arr))
        if rows:
            conn.executemany(
                """INSERT INTO epoch (file_id, animal, electrode,
                       channel, stim_time_sec, peak, trough)
                   VALUES (?,?,?,?,?,?,?)""", rows)

    @staticmethod
    def _upsert_file(conn, path, mtime, rec_iso) -> int:
        """Insert/refresh the file_meta row, returning its integer id.

        Idempotent: ``INSERT OR IGNORE`` never collides on the file_path
        UNIQUE, then we set the fresh values and read the id back -- safe
        even if two builders touch the same path.
        """
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO file_meta (file_path, mtime, rec_dt, "
            "built_at) VALUES (?,?,?,?)", (path, mtime, rec_iso, now))
        conn.execute(
            "UPDATE file_meta SET mtime=?, rec_dt=?, built_at=? "
            "WHERE file_path=?", (mtime, rec_iso, now, path))
        row = conn.execute("SELECT id FROM file_meta WHERE file_path = ?",
                           (path,)).fetchone()
        assert row is not None, "file_meta upsert lost its row"
        return int(row["id"])

    @staticmethod
    def _epoch_rows(file_id, animal, electrode, ch, arr):
        """Flatten one channel's arrays into epoch insert tuples."""
        times = arr.get("times") or []
        peak = arr.get("peak") or []
        trough = arr.get("trough") or []
        n = min(len(times), len(peak), len(trough))
        rows = []
        for j in range(n):
            assert j < _MAX_FILES, "epoch loop exceeds bound"
            rows.append((file_id, animal, electrode, ch,
                         times[j], peak[j], trough[j]))
        return rows

    def query(self, animal: str, hours: int | None = None) -> list[dict]:
        """Per-epoch rows for *animal*, oldest first (optionally last N h).

        ``abs_dt`` (recording time + stim offset) is computed here from the
        joined ``rec_dt``; ``hours`` filters on recording datetime. Each
        dict: channel, electrode, rec_dt, abs_dt, stim_time_sec, peak,
        trough, peak_to_trough.
        """
        if not animal:
            return []
        conn = self._conn()
        try:
            sql = ("SELECT e.channel, e.electrode, m.rec_dt, "
                   "e.stim_time_sec, e.peak, e.trough "
                   "FROM epoch e JOIN file_meta m ON e.file_id = m.id "
                   "WHERE e.animal = ?")
            params: list = [animal]
            if hours:
                cutoff = (datetime.now()
                          - timedelta(hours=hours)).isoformat()
                sql += " AND m.rec_dt > ?"
                params.append(cutoff)
            sql += " ORDER BY m.rec_dt ASC, e.stim_time_sec ASC"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(r) -> dict:
        """Add abs_dt + peak_to_trough to a queried epoch row."""
        d = dict(r)
        rec = d.get("rec_dt")
        t = d.get("stim_time_sec")
        d["abs_dt"] = _add_seconds(rec, t)
        pk, tr = d.get("peak"), d.get("trough")
        d["peak_to_trough"] = (pk - tr) if (pk is not None
                                           and tr is not None) else None
        return d
