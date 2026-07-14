"""Gather per-(animal, channel, metric) evoked time-series from the feature
sidecars and compute rolling percentile bands.

Dash-free, but mirrors the chronic-evoked tab's math exactly
(`read_feature_sidecar` -> `_feature_points` -> `_rolling_percentiles`) so the
exported figures match what the dashboard shows. Reading only FRESH sidecars
(`read_feature_sidecar` returns None when the source .mat changed) is the
readiness gate -- no rendering off a half-written recording.

SCALE: the big animals are millions of epochs (BCH040 ~= 2.3M across 1450
recordings). So we make ONE pass over the sidecars and accumulate straight into
float arrays per (channel, metric) -- never holding the row dicts (GBs) and
never re-scanning the rows once per metric. The CSV is written by a separate
STREAMING pass so it, too, never materialises everything in memory.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
from collections import defaultdict
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


def _nan(v) -> float:
    fv = _finite(v)
    return fv if fv is not None else np.nan


def parse_iso(s):
    try:
        return datetime.fromisoformat(s) if s else None
    except (TypeError, ValueError):
        return None


def iter_animal_sidecars(animal: str, evoked_dir: str):
    """Yield (mat_path, sidecar_path, rows) for each FRESH sidecar of *animal*,
    oldest recording first. Stale/missing sidecars are skipped (readiness gate).
    One file's rows at a time -- the caller never holds them all."""
    assert animal, "animal required"
    for i, fp in enumerate(list_evoked_files(evoked_dir)):
        assert i < 1_000_000, "evoked file scan runaway"
        if animal not in animals_in_filename(fp):
            continue
        rows = read_feature_sidecar(fp, animal)
        if not rows:
            continue
        yield fp, feature_sidecar_path(fp, animal), rows


def animal_series(animal: str, evoked_dir: str, metrics: list[str]):
    """ONE pass over *animal*'s sidecars -> per-channel float arrays.

    Returns ``(series, inputs)`` where
      series[channel] = {"secs": float[n], "metrics": {m: float[n] (NaN-filled)}}
    sorted ascending by time, and inputs = [(sidecar_path, mtime)] for
    provenance + change detection. Cost is ~(1 + len(metrics)) floats per epoch
    (~420 MB for 2.3M epochs x 22 metrics), not GBs of dicts.
    """
    chunk_secs: dict = defaultdict(list)          # channel -> [per-file arrays]
    chunk_vals: dict = defaultdict(lambda: defaultdict(list))
    inputs: list = []
    for _fp, sp, rows in iter_animal_sidecars(animal, evoked_dir):
        try:
            inputs.append((sp, os.path.getmtime(sp)))
        except OSError:
            pass
        by_ch: dict = defaultdict(list)
        for r in rows:
            ch = r.get("channel")
            if ch:
                by_ch[ch].append(r)
        for ch, rws in by_ch.items():
            secs = np.fromiter(
                ((parse_iso(r.get("abs_dt")) or datetime.min).timestamp()
                 if r.get("abs_dt") else np.nan for r in rws),
                dtype=float, count=len(rws))
            chunk_secs[ch].append(secs)
            for m in metrics:
                chunk_vals[ch][m].append(
                    np.fromiter((_nan(r.get(m)) for r in rws),
                                dtype=float, count=len(rws)))
        # rows (and their dicts) go out of scope here -> memory stays flat

    series: dict = {}
    for ch, parts in chunk_secs.items():
        secs = np.concatenate(parts)
        order = np.argsort(secs, kind="stable")
        series[ch] = {
            "secs": secs[order],
            "metrics": {m: np.concatenate(chunk_vals[ch][m])[order]
                        for m in metrics},
        }
    return series, inputs


def primary_channel(animal: str, series: dict, metrics: list[str],
                    override: str | None = None) -> str | None:
    """Override wins; else the channel with the most finite datapoints across
    *metrics*. NOTE: a completeness heuristic, NOT a scientific one (SR != SLM);
    the manifest records the pick so it can be eyeballed + overridden."""
    if override and override in series:
        return override
    best, best_n = None, -1
    for ch, d in series.items():
        n = int(sum(np.isfinite(d["metrics"][m]).sum() for m in metrics
                    if m in d["metrics"]))
        if n > best_n:
            best, best_n = ch, n
    return best


def finite_series(series: dict, channel: str, metric: str):
    """(dts, secs, vals) for the finite points of one (channel, metric),
    time-ascending. Mirrors chronic_evoked._feature_points."""
    d = series.get(channel) or {}
    vals = (d.get("metrics") or {}).get(metric)
    secs = d.get("secs")
    if vals is None or secs is None or secs.size == 0:
        return [], np.empty(0), np.empty(0)
    ok = np.isfinite(vals) & np.isfinite(secs)
    s, v = secs[ok], vals[ok]
    dts = [datetime.fromtimestamp(x) for x in s]
    return dts, s, v


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
    """meta columns + every feature -> CSV text (small/in-memory; used by tests).
    abs_dt stays the ISO string it was stored as -- the shipped time
    representation, exercised by the epoch round-trip test."""
    assert isinstance(rows, list), "rows must be a list"
    cols = _META_COLS + list(ef.ALL_COLUMNS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c) for c in cols])
    return buf.getvalue()


def stream_channel_csv_gz(animal: str, evoked_dir: str, channel: str,
                          path: str) -> int:
    """STREAM one channel's per-epoch rows to a gzipped CSV, one sidecar at a
    time -- a big animal is millions of epochs (BCH040 ~2.3M ~= 1.2 GB raw), so
    it must never be materialised. Returns the row count. Same columns + ISO
    abs_dt as rows_to_csv."""
    cols = _META_COLS + list(ef.ALL_COLUMNS)
    tmp = path + ".tmp"
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for _fp, _sp, rows in iter_animal_sidecars(animal, evoked_dir):
            for r in rows:
                if r.get("channel") != channel:
                    continue
                w.writerow([r.get(c) for c in cols])
                n += 1
    os.replace(tmp, path)                              # atomic
    return n
