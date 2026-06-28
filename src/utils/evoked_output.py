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
compute the full ~25-feature set per epoch (``src.utils.evoked_features``).

This module is pure: filename parsing, an h5py trace reader
(``read_file_evoked``), and the per-(recording, animal) feature **sidecar**
(``read_feature_sidecar`` / ``write_feature_sidecar``) -- a small JSON written
next to each ``*_evoked.mat`` holding that animal's computed feature rows.
The sidecar co-locates the derived features with their source, so the chronic
view reads them without re-reading the heavy traces and without any database;
the one-time trace read per recording is paid once, then the sidecar makes
every re-plot instant. (The former sqlite feature cache was removed in favor
of these sidecars.)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from datetime import datetime, timedelta

import numpy as np

from src.utils.animal import split_animal_electrode, is_animal_channel
from src.utils import evoked_features as ef

logger = logging.getLogger("qc_monitor.utils.evoked_output")

# Default location of the toolkit's evoked output (Windows path). Override
# via ``config.chronic_evoked.evoked_output_dir``.
DEFAULT_EVOKED_DIR = (
    r"D:\code\Stimulation-Telemetry-Modulation-NeuroEngineering-Toolkit"
    r"\daqSignalGenerator\evokedOutput"
)
# Trailing ``..._YYYY_MM_DD__HH_MM_SS_evoked.mat`` recording timestamp.
_DT_RX = re.compile(
    r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})_evoked\.mat$")
# Animal-id tokens anywhere in the filename (BCH040, BCH062, ...).
_ANIMAL_RX = re.compile(r"(BCH\d+)")

_MAX_FILES = 100000          # NASA Rule 2: explicit loop bound.


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


def animals_in_filename(filename: str) -> set[str]:
    """Animal-id tokens (``BCH###``) present in *filename*."""
    if not filename:
        return set()
    return set(_ANIMAL_RX.findall(os.path.basename(filename)))


def parse_session(filename: str) -> str:
    """Session/protocol label = the filename prefix before the channel
    list (the first ``__``). E.g. ``20251210_stimBaseline__stimCopy_...``
    -> ``20251210_stimBaseline``; ``stimTest-40nC__...`` -> ``stimTest-40nC``.
    """
    b = os.path.basename(filename or "")
    return b.split("__", 1)[0] if b else ""


def sessions_for_animal(evoked_dir: str, animal: str) -> list[str]:
    """Distinct session labels for *animal*, read straight from the evoked
    filenames -- so the session dropdown can populate the moment an animal is
    picked, without waiting for that animal's cache to warm."""
    if not animal or not evoked_dir:
        return []
    seen: set[str] = set()
    for i, f in enumerate(list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "evoked file count exceeds bound"
        if animal in animals_in_filename(f):
            s = parse_session(f)
            if s:
                seen.add(s)
    return sorted(seen)


def sessions_with_dt_for_animal(evoked_dir: str,
                                animal: str) -> list[dict]:
    """[{session, first, last}] for *animal* (ISO recording datetimes parsed
    from the filenames), sorted by first time -- so the session dropdown can
    show each session's timestamp without a cache warm."""
    if not animal or not evoked_dir:
        return []
    by_sess: dict[str, list] = {}
    for i, f in enumerate(list_evoked_files(evoked_dir)):
        assert i < _MAX_FILES, "evoked file count exceeds bound"
        if animal in animals_in_filename(f):
            s = parse_session(f)
            if s:
                by_sess.setdefault(s, []).append(parse_recording_dt(f))
    out: list[dict] = []
    for s, dts in by_sess.items():
        valid = sorted(d for d in dts if d is not None)
        out.append({"session": s,
                    "first": valid[0].isoformat() if valid else "",
                    "last": valid[-1].isoformat() if valid else ""})
    out.sort(key=lambda r: (r["first"] or r["session"]))
    return out


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


# Per-epoch feature sidecar: a small JSON written next to each *_evoked.mat,
# one per animal, holding that animal's computed feature rows. It co-locates
# the derived features with their source so the chronic view can read them
# WITHOUT re-reading the heavy traces or warming the sqlite cache -- and it
# doubles as the on-disk export. Bump the version if the row schema changes.
# v2: added wavelet_power_{slow_gamma,gamma,high_gamma} columns.
_FEATURE_SIDECAR_VERSION = "2"


def feature_sidecar_path(mat_path: str, animal: str) -> str:
    """Path of *animal*'s per-epoch feature sidecar next to a *_evoked.mat."""
    assert mat_path and animal, "mat_path and animal required"
    base = mat_path[:-4] if mat_path.lower().endswith(".mat") else mat_path
    return f"{base}.features.{animal}.json"


def write_feature_sidecar(mat_path: str, animal: str, rows: list) -> str:
    """Atomically write *animal*'s feature rows beside the source .mat as
    JSON, stamped with the source mtime so a stale sidecar is detectable."""
    assert mat_path and animal, "mat_path and animal required"
    assert isinstance(rows, list), "rows must be a list"
    sp = feature_sidecar_path(mat_path, animal)
    try:
        src_mtime = os.path.getmtime(mat_path)
    except OSError:
        src_mtime = 0.0
    payload = {"version": _FEATURE_SIDECAR_VERSION, "animal": animal,
               "source": os.path.basename(mat_path), "source_mtime": src_mtime,
               "columns": list(ef.ALL_COLUMNS), "rows": rows}
    tmp = sp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, sp)        # atomic on the same volume
    return sp


def read_feature_sidecar(mat_path: str, animal: str) -> list | None:
    """*animal*'s feature rows from the sidecar, or None when it's missing,
    unreadable, a different schema version, or stale (source .mat changed)."""
    assert mat_path and animal, "mat_path and animal required"
    sp = feature_sidecar_path(mat_path, animal)
    if not os.path.exists(sp):
        return None
    try:
        with open(sp, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if payload.get("version") != _FEATURE_SIDECAR_VERSION:
        return None
    try:
        if abs(float(payload.get("source_mtime", -1.0))
               - os.path.getmtime(mat_path)) > 1e-6:
            return None
    except OSError:
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def read_file_evoked(path: str,
                     only_animals: list | None = None) -> dict[str, dict]:
    """Per-channel data for one ``*_evoked.mat`` (h5py read).

    When *only_animals* is given, channels whose animal id isn't in it are
    skipped BEFORE their (large) ``evokedData`` is read -- so warming one
    animal of a multi-animal file (e.g. 4 animals x 1795 epochs x 20 k
    samples = ~600 MB) reads/decodes only that animal's channel, not all.

    Returns ``{channel: {"times":[...], "stim_peak":[...],
    "stim_trough":[...], "traces": ndarray[E,T] or None,
    "time_ms": ndarray[T] or None}}``. ``traces``/``time_ms`` are None when
    the file has no ``evokedData`` (scalars-only files still yield rows).
    Unreadable files -> ``{}``. h5py gives ``evokedData`` as ``[epochs x
    samples]`` already (HDF5 is transposed vs MATLAB), so no transpose.
    """
    assert isinstance(path, str) and path, "path required"
    import h5py  # lazy: keeps the dependency off non-chronic code paths.
    keep = set(only_animals) if only_animals else None
    out: dict[str, dict] = {}
    try:
        with h5py.File(path, "r") as g:
            grp = g.get("allAnimalResults")
            if grp is None:
                return {}
            for ch in list(grp.keys()):
                if keep is not None:
                    a, _e = split_animal_electrode(str(ch))
                    if a not in keep:
                        continue
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


def _abs_dt(rec_iso: str, seconds) -> str:
    """ISO recording time + a stim offset (seconds) -> ISO, or '' / rec."""
    if not rec_iso:
        return ""
    if seconds is None:
        return rec_iso
    try:
        return (datetime.fromisoformat(rec_iso)
                + timedelta(seconds=float(seconds))).isoformat()
    except (ValueError, TypeError):
        return rec_iso


def compute_feature_rows(path: str, animal: str, cfg=None,
                         expensive: bool = False) -> list:
    """Per-epoch feature rows for one ``*_evoked.mat`` / *animal* -- the
    canonical sidecar payload. Pure: reads the animal's traces, computes the
    full feature set (optionally with *cfg* window/filter/smoothing/baseline),
    and returns oldest-first rows. Shared by the dashboard preview/load path
    and the offline sidecar-build tool.
    """
    assert path and animal, "path and animal required"
    chans = read_file_evoked(path, only_animals=[animal])
    rec_iso = (parse_recording_dt(path) or datetime.min).isoformat()
    session = parse_session(path)
    rows: list = []
    for ch, rec in chans.items():
        a, electrode = split_animal_electrode(ch)
        if a != animal or not is_animal_channel(ch):
            continue
        traces, tms = rec.get("traces"), rec.get("time_ms")
        if traces is None or tms is None or len(traces) < 1 or tms.size < 2:
            continue
        fs = 1000.0 / float(np.mean(np.diff(tms)))
        feats = ef.compute_all(traces, tms, fs, expensive, cfg)
        times = rec.get("times") or []
        pk, tr = rec.get("stim_peak") or [], rec.get("stim_trough") or []
        for j in range(traces.shape[0]):
            st = times[j] if j < len(times) else None
            row = {"channel": ch, "electrode": electrode, "rec_dt": rec_iso,
                   "session": session, "stim_time_sec": st,
                   "abs_dt": _abs_dt(rec_iso, st),
                   "peak": pk[j] if j < len(pk) else None,
                   "trough": tr[j] if j < len(tr) else None}
            for col in ef.ALL_COLUMNS:
                v = float(feats[col][j])
                row[col] = v if np.isfinite(v) else None
            rows.append(row)
    rows.sort(key=lambda r: r.get("abs_dt") or "")
    return rows
