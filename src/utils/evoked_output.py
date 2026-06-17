"""Chronic evoked source: the MATLAB toolkit's ``evokedOutput`` folder.

For an implanted animal the *full* chronic record (back to the implant
baseline) lives as per-recording ``*_evoked.mat`` files in the toolkit's
``evokedOutput`` directory -- not in the rolling monitor DB, which only
retains the recent window it ingests. The Chronic Evoked Analyzer tab
reads those files at **per-evoked-response resolution**: one point per
stimulus, not one per session.

Each ``*_evoked.mat`` is a MATLAB v7.3 (HDF5) file::

    allAnimalResults/<channel>/stimulusTimes            (1, nEpoch)
    allAnimalResults/<channel>/stimulusPeakAmplitudes   (nEpoch, 1)  # raw stim
    allAnimalResults/<channel>/stimulusTroughAmplitudes (nEpoch, 1)  # raw stim
    allAnimalResults/<channel>/evokedData               (nEpoch, nSamp)
    allAnimalResults/<channel>/timeAxis                 (nSamp, 1)   # ms, t0=stim

``<channel>`` is a full channel name (e.g. ``BCH062SR``); a multi-animal
recording carries several channels.

To faithfully reproduce the toolkit's "Chronic Evoked Features" tab we read
the (already filtered + baseline-corrected) ``evokedData`` traces and
compute the full ~25-feature set per epoch (``src.utils.evoked_features``)
plus a per-recording mean/std waveform. Opening + feature-extracting each
file is slow, so results are cached once into a compact sqlite DB keyed by
file mtime; rebuilds are incremental and re-reads are instant. The cheap
vectorized features are always computed; the expensive per-epoch fits are
opt-in (``compute_expensive``).
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta

import numpy as np

from src.utils.animal import split_animal_electrode, is_animal_channel
from src.utils import evoked_features as ef

# Default location of the toolkit's evoked output (Windows path). Override
# via ``config.chronic_evoked.evoked_output_dir``.
DEFAULT_EVOKED_DIR = (
    r"D:\code\Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit"
    r"\daqSignalGenerator\evokedOutput"
)
DEFAULT_CACHE_DB = os.path.join("data", "evoked_chronic_cache.db")

# Bump when the cache layout changes; a mismatch wipes + rebuilds (the DB
# is a throwaway cache, gitignored).
SCHEMA_VERSION = "3"

# Per-recording mean waveforms are stored downsampled to this many points;
# the overlay never needs the raw ~20k-sample resolution, and full traces
# as JSON would bloat the cache (~16 MB/file).
_MEAN_TRACE_POINTS = 1000

# epoch columns = stim scalars (raw) + the computed evoked feature set.
_BASE_EPOCH_COLS = ["file_id", "animal", "electrode", "channel",
                    "stim_time_sec", "peak", "trough"]
_EPOCH_COLS = _BASE_EPOCH_COLS + ef.ALL_COLUMNS

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


def _num(v):
    """Cache-friendly scalar: NaN/inf/None -> None (SQL NULL), else float."""
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def _downsample(time_ms, mean_tr, std_tr, target):
    """Stride-decimate the three aligned arrays to <= *target* points."""
    n = len(time_ms)
    if n <= target or target <= 0:
        return time_ms, mean_tr, std_tr
    step = (n // target) + 1
    return time_ms[::step], mean_tr[::step], std_tr[::step]


def _round(arr) -> list:
    """Compact JSON: round to 4 sig-ish figures to keep the cache small."""
    return [round(float(v), 4) for v in np.asarray(arr).ravel()]


def _mean_row_to_dict(r) -> dict:
    d = dict(r)
    for k in ("time_axis", "mean_trace", "std_trace"):
        try:
            d[k] = json.loads(d[k]) if d.get(k) else []
        except (ValueError, TypeError):
            d[k] = []
    return d


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


def read_file_evoked(path: str) -> dict[str, dict]:
    """Per-channel data for one ``*_evoked.mat`` (h5py read).

    Returns ``{channel: {"times":[...], "stim_peak":[...],
    "stim_trough":[...], "traces": ndarray[E,T] or None,
    "time_ms": ndarray[T] or None}}``. ``traces``/``time_ms`` are None when
    the file has no ``evokedData`` (scalars-only files still yield rows).
    Unreadable files -> ``{}``. h5py gives ``evokedData`` as ``[epochs x
    samples]`` already (HDF5 is transposed vs MATLAB), so no transpose.
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
                rec = _read_channel(grp.get(ch))
                if rec is not None:
                    out[str(ch)] = rec
    except (OSError, KeyError, ValueError):
        return {}
    return out


def _read_channel(node):
    """One channel's datasets, or None if it lacks stim times."""
    if node is None or "stimulusTimes" not in node:
        return None
    times = node["stimulusTimes"][()].ravel().tolist()
    sp = node["stimulusPeakAmplitudes"][()].ravel().tolist()
    st = node["stimulusTroughAmplitudes"][()].ravel().tolist()
    traces = None
    time_ms = None
    if "evokedData" in node and "timeAxis" in node:
        ev = np.asarray(node["evokedData"][()], dtype=np.float64)
        if ev.ndim == 2 and ev.shape[1] >= 2:
            traces = ev
            time_ms = np.asarray(node["timeAxis"][()],
                                 dtype=np.float64).ravel()
    return {"times": times, "stim_peak": sp, "stim_trough": st,
            "traces": traces, "time_ms": time_ms}


class ChronicEvokedCache:
    """Per-epoch evoked scalars cached from ``evokedOutput`` into sqlite.

    The cache is animal-agnostic on disk (one row per epoch per channel);
    callers query by animal id. ``ensure_animal`` is incremental -- it
    skips files whose mtime already matches the cached value.
    """

    def __init__(self, evoked_dir: str | None = None,
                 cache_db: str | None = None,
                 compute_expensive: bool = False) -> None:
        self.evoked_dir = evoked_dir or DEFAULT_EVOKED_DIR
        self.cache_db = cache_db or DEFAULT_CACHE_DB
        self.compute_expensive = bool(compute_expensive)
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
        # A schema-version mismatch wipes the data tables and rebuilds.
        conn = self._conn()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta "
                "(key TEXT PRIMARY KEY, value TEXT)")
            row = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None or row["value"] != SCHEMA_VERSION:
                self._reset_schema(conn)
            conn.commit()
        finally:
            conn.close()

    def _reset_schema(self, conn) -> None:
        """Drop + recreate the data tables for the current schema version."""
        feat_ddl = ",\n                ".join(
            f"{c} REAL" for c in ef.ALL_COLUMNS)
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS epoch;
            DROP TABLE IF EXISTS recording_mean;
            DROP TABLE IF EXISTS file_meta;
            CREATE TABLE file_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL, rec_dt TEXT, built_at TEXT,
                n_epochs INTEGER, fs REAL
            );
            CREATE TABLE epoch (
                file_id INTEGER NOT NULL, animal TEXT NOT NULL,
                electrode TEXT, channel TEXT, stim_time_sec REAL,
                peak REAL, trough REAL,
                {feat_ddl}
            );
            CREATE TABLE recording_mean (
                file_id INTEGER NOT NULL, animal TEXT NOT NULL,
                channel TEXT, time_axis TEXT, mean_trace TEXT,
                std_trace TEXT, n_epochs INTEGER
            );
            CREATE INDEX idx_epoch_animal ON epoch(animal);
            CREATE INDEX idx_epoch_file ON epoch(file_id);
            CREATE INDEX idx_recmean_animal ON recording_mean(animal);
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES "
            "('schema_version', ?)", (SCHEMA_VERSION,))

    def files_for_animal(self, animal: str) -> list[str]:
        """Evoked files whose filename names *animal* (sorted by time)."""
        if not animal:
            return []
        files = [f for f in list_evoked_files(self.evoked_dir)
                 if animal in animals_in_filename(f)]
        return sorted(files, key=lambda f: (parse_recording_dt(f)
                                            or datetime.min))

    def _needs_build(self, conn, path: str, force: bool) -> bool:
        if force:
            return True
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        row = conn.execute(
            "SELECT mtime FROM file_meta WHERE file_path = ?",
            (path,)).fetchone()
        return row is None or abs(float(row["mtime"]) - mtime) > 1e-6

    def ensure_animal(self, animal: str, progress=None,
                      force: bool = False) -> dict:
        """Cache every not-yet-current evoked file for *animal*.

        *progress* (optional) is called ``progress(done, total, path)``
        after each file. *force* recomputes even unchanged files (used to
        backfill expensive features after enabling them). Returns
        ``{"files": n, "built": k}``.
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
                    if self._needs_build(conn, path, force):
                        self._build_one(conn, path)
                        conn.commit()   # release the write lock per file so
                        built += 1       # the dashboard can read mid-warm.
                    if progress is not None:
                        progress(i + 1, len(files), path)
            finally:
                conn.close()
        return {"files": len(files), "built": built}

    def _build_one(self, conn, path: str) -> None:
        """(Re)extract one file's epochs + mean waveforms (keyed by file_id)."""
        rec_dt = parse_recording_dt(path)
        rec_iso = rec_dt.isoformat() if rec_dt else ""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        file_id = self._upsert_file(conn, path, mtime, rec_iso)
        conn.execute("DELETE FROM epoch WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM recording_mean WHERE file_id = ?", (file_id,))
        channels = read_file_evoked(path)
        rows: list = []
        means: list = []
        n_total = 0
        fs_seen = None
        for ch, info in channels.items():
            if not is_animal_channel(ch):
                continue
            animal, electrode = split_animal_electrode(ch)
            erows, mrow, fs = self._channel_rows(
                file_id, animal, electrode, ch, info)
            rows.extend(erows)
            n_total += len(erows)
            if mrow is not None:
                means.append(mrow)
            fs_seen = fs_seen or fs
        if rows:
            ph = ",".join("?" * len(_EPOCH_COLS))
            conn.executemany(
                f"INSERT INTO epoch ({','.join(_EPOCH_COLS)}) VALUES ({ph})",
                rows)
        if means:
            conn.executemany(
                "INSERT INTO recording_mean (file_id, animal, channel, "
                "time_axis, mean_trace, std_trace, n_epochs) "
                "VALUES (?,?,?,?,?,?,?)", means)
        conn.execute("UPDATE file_meta SET n_epochs=?, fs=? WHERE id=?",
                     (n_total, fs_seen, file_id))

    def _channel_rows(self, file_id, animal, electrode, ch, info):
        """Epoch insert tuples + a recording_mean tuple for one channel.

        Returns ``(epoch_rows, recording_mean_row_or_None, fs_or_None)``.
        """
        times = info.get("times") or []
        sp = info.get("stim_peak") or []
        st = info.get("stim_trough") or []
        traces = info.get("traces")
        time_ms = info.get("time_ms")
        feats, fs, mrow = None, None, None
        if traces is not None and time_ms is not None and traces.shape[0] >= 1:
            fs = 1000.0 / float(np.mean(np.diff(time_ms)))
            feats = ef.compute_all(traces, time_ms, fs, self.compute_expensive)
            mrow = self._mean_row(file_id, animal, ch, traces, time_ms)
        n = min(len(times), len(sp), len(st))
        if traces is not None:
            n = min(n, traces.shape[0])
        rows = self._epoch_tuples(file_id, animal, electrode, ch,
                                  times, sp, st, feats, n)
        return rows, mrow, fs

    @staticmethod
    def _mean_row(file_id, animal, ch, traces, time_ms):
        mean_tr = np.nanmean(traces, axis=0)
        std_tr = np.nanstd(traces, axis=0)
        t, m, s = _downsample(time_ms, mean_tr, std_tr, _MEAN_TRACE_POINTS)
        return (file_id, animal, ch, json.dumps(_round(t)),
                json.dumps(_round(m)), json.dumps(_round(s)),
                int(traces.shape[0]))

    @staticmethod
    def _epoch_tuples(file_id, animal, electrode, ch, times, sp, st, feats, n):
        rows = []
        for j in range(n):
            assert j < _MAX_FILES * 100, "epoch loop exceeds bound"
            base = [file_id, animal, electrode, ch, _num(times[j]),
                    _num(sp[j]), _num(st[j])]
            if feats is None:
                base.extend([None] * len(ef.ALL_COLUMNS))
            else:
                base.extend(_num(feats[c][j]) for c in ef.ALL_COLUMNS)
            rows.append(tuple(base))
        return rows

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

    def query(self, animal: str, hours: int | None = None) -> list[dict]:
        """Per-epoch rows for *animal*, oldest first (optionally last N h).

        ``abs_dt`` (recording time + stim offset) is computed here from the
        joined ``rec_dt``; ``hours`` filters on recording datetime. Each
        dict carries channel/electrode/rec_dt/abs_dt/stim_time_sec, the raw
        stim ``peak``/``trough``, and every computed evoked feature column.
        """
        if not animal:
            return []
        feat_sel = ", ".join("e." + c for c in ef.ALL_COLUMNS)
        conn = self._conn()
        try:
            sql = ("SELECT e.channel, e.electrode, m.rec_dt, "
                   "e.stim_time_sec, e.peak, e.trough, " + feat_sel +
                   " FROM epoch e JOIN file_meta m ON e.file_id = m.id "
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

    def query_recording_means(self, animal: str,
                              hours: int | None = None) -> list[dict]:
        """Per-recording mean/std evoked waveforms for *animal*, oldest first.

        Each dict: channel, rec_dt, n_epochs, and ``time_axis``/``mean_trace``/
        ``std_trace`` decoded back to Python float lists.
        """
        if not animal:
            return []
        conn = self._conn()
        try:
            sql = ("SELECT rm.channel, m.rec_dt, rm.n_epochs, rm.time_axis, "
                   "rm.mean_trace, rm.std_trace "
                   "FROM recording_mean rm "
                   "JOIN file_meta m ON rm.file_id = m.id "
                   "WHERE rm.animal = ?")
            params: list = [animal]
            if hours:
                cutoff = (datetime.now()
                          - timedelta(hours=hours)).isoformat()
                sql += " AND m.rec_dt > ?"
                params.append(cutoff)
            sql += " ORDER BY m.rec_dt ASC"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [_mean_row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(r) -> dict:
        """Add abs_dt to a queried epoch row."""
        d = dict(r)
        d["abs_dt"] = _add_seconds(d.get("rec_dt"), d.get("stim_time_sec"))
        return d
