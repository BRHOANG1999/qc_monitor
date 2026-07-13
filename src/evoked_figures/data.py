"""Gather per-(animal, channel, metric) evoked time-series from the feature
sidecars and compute rolling percentile bands.

Dash-free, but mirrors the chronic-evoked tab's data path exactly
(`read_feature_sidecar` -> `_feature_points` -> `_rolling_percentiles`) so the
exported figures match what the dashboard shows. Reading only FRESH sidecars
(`read_feature_sidecar` returns None when the source .mat changed) is the
readiness gate — no rendering off a half-written recording.
"""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime

import numpy as np

from src.utils import evoked_features as ef
from src.utils.evoked_output import (animals_in_filename, feature_sidecar_path,
                                     list_evoked_files, read_feature_sidecar)

_META_COLS = ["channel", "electrode", "rec_dt", "session", "stim_time_sec",
              "abs_dt", "peak", "trough"]


def _finite(v) -> float | None:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def parse_iso(s):
    try:
        return datetime.fromisoformat(s) if s else None
    except (TypeError, ValueError):
        return None


def animal_rows(animal: str, evoked_dir: str) -> tuple[list, list]:
    """(rows, inputs): every FRESH-sidecar feature row for *animal* across its
    recordings, chronological by abs_dt, plus inputs = [(sidecar_path, mtime)]
    for provenance + change detection. Stale/missing sidecars are skipped."""
    assert animal, "animal required"
    rows: list = []
    inputs: list = []
    files = list_evoked_files(evoked_dir)
    for i, fp in enumerate(files):
        assert i < 1_000_000, "evoked file scan runaway"
        if animal not in animals_in_filename(fp):
            continue
        sc = read_feature_sidecar(fp, animal)
        if not sc:
            continue
        sp = feature_sidecar_path(fp, animal)
        try:
            inputs.append((sp, os.path.getmtime(sp)))
        except OSError:
            pass
        rows.extend(sc)
    rows.sort(key=lambda r: r.get("abs_dt") or "")
    return rows, inputs


def channels_in(rows: list) -> list[str]:
    return sorted({r.get("channel") for r in rows if r.get("channel")})


def primary_channel(animal: str, rows: list, metrics: list[str],
                    override: str | None = None) -> str | None:
    """Override wins; else the channel with the most finite datapoints across
    *metrics*. NOTE: a completeness heuristic, NOT a scientific one (SR != SLM);
    the manifest records the pick so it can be eyeballed + overridden."""
    chans = channels_in(rows)
    if override and override in chans:
        return override
    best, best_n = None, -1
    for ch in chans:
        n = 0
        for r in rows:
            if r.get("channel") != ch:
                continue
            for m in metrics:
                if _finite(r.get(m)) is not None:
                    n += 1
        if n > best_n:
            best, best_n = ch, n
    return best


def series_for(rows: list, channel: str, metric: str):
    """(dts, secs, vals) for finite (channel, metric) points, sorted by time.
    Mirrors chronic_evoked._feature_points, filtered to one channel."""
    trips = []
    for r in rows:
        if r.get("channel") != channel:
            continue
        fv = _finite(r.get(metric))
        dt = parse_iso(r.get("abs_dt"))
        if fv is None or dt is None:
            continue
        trips.append((dt, dt.timestamp(), fv))
    trips.sort(key=lambda t: t[1])
    dts = [t[0] for t in trips]
    secs = np.asarray([t[1] for t in trips], dtype=float)
    vals = np.asarray([t[2] for t in trips], dtype=float)
    return dts, secs, vals


def rolling_percentiles(secs, vals, win, n_eval=400):
    """(band_secs, p10, p25, p50, p75, p90) in a centered count window, or None.
    Identical algorithm to chronic_evoked._rolling_percentiles."""
    n = int(vals.size)
    win = max(11, int(win))
    if n < max(11, win // 2):
        return None
    half = win // 2
    lo_i, hi_i = half, n - half - 1
    if hi_i <= lo_i:
        lo_i, hi_i = 0, n - 1
    idxs = np.unique(np.linspace(lo_i, hi_i,
                                 min(n_eval, hi_i - lo_i + 1)).astype(int))
    qs = np.array([10, 25, 50, 75, 90])
    out = np.empty((idxs.size, 5))
    for k, i in enumerate(idxs):
        seg = vals[max(0, i - half):min(n, i + half + 1)]
        out[k] = np.percentile(seg, qs)
    return (secs[idxs], out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4])


def rows_to_csv(rows: list) -> str:
    """meta columns + every feature -> CSV text. Mirrors
    chronic_evoked._rows_to_csv; abs_dt stays the ISO string it was stored as
    (the shipped time representation, exercised by the epoch round-trip test).
    For large exports use write_rows_csv_gz (streams, no giant string)."""
    assert isinstance(rows, list), "rows must be a list"
    cols = _META_COLS + list(ef.ALL_COLUMNS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c) for c in cols])
    return buf.getvalue()


def write_rows_csv_gz(rows: list, path: str) -> str:
    """Stream the same columns as rows_to_csv to a GZIPPED CSV — one animal's
    per-epoch data can be hundreds of MB uncompressed (BCH062 ≈ 582k epochs ≈
    311 MB), so we compress AND stream (never build the whole string in memory,
    which would OOM on the big animals). Same ISO abs_dt representation."""
    import gzip
    cols = _META_COLS + list(ef.ALL_COLUMNS)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])
    os.replace(tmp, path)               # atomic
    return path
