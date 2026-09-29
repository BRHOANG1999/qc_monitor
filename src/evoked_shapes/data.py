"""Stage 1 (real data): assemble the trial matrix X, fs, and the meta frame.

Raw per-trial evoked traces live only in the MATLAB evokedOutput .mat files (the DB
stores per-file means, not per-trial epochs), so we read them through the shared
reader used everywhere else: src.evoked_typology.gather_responses (baseline-subtract,
crop to the analysis window, reservoir-subsample to a cap). We then attach
time-to-next-seizure with the peri-ictal machinery: cluster LEADERS only (followers
within the lead gap dropped), each seizure's ISI-ceilinged lookback window.

Two grounded choices, recorded so the report can flag them:
  * ``session`` is the recording DAY (meta from gather_responses exposes the day, not
    a session token). Held-out-by-day is a stricter generalization test than by an
    arbitrary session id, so this is a feature, not a fallback.
  * ``stim_status`` is resolved at the animal level (per-token fingerprint) and
    returned as ``stim_coverage``; it is not attached per trial (the subsampled trace
    reader does not carry each trial's session_dir). It matters for the deferred risk
    stage, not for the shape-discovery stages built here.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from src.evoked_shapes import config as _cfg
from src.evoked_typology import gather_responses
from src.evoked_figures import data as _efd
from src.preictal import isi as _isi
from src.periictal import matrix as _mx
from src.periictal import stim_map as _sm

_CACHE_SCHEMA = "1"
_CHANNEL_METRIC = "peak_to_trough"      # a cheap always-present metric for channel pick


def _default_days(config: dict) -> int:
    try:
        return max(1, int((config.get("evoked_shapes", {}) or {}).get("days", 400)))
    except (TypeError, ValueError):
        return 400


def _resolve_channel(animal: str, evoked_dir: str, config: dict,
                     log: Callable[[str], None] | None) -> str | None:
    """Primary channel for *animal* via the shared evoked_figures picker. Honors an
    ``evoked_shapes.primary_channel`` override, else the most-populated channel."""
    override = ((config.get("evoked_shapes", {}) or {})
               .get("primary_channel", {}) or {}).get(animal)
    if log is not None:
        log(f"resolving primary channel for {animal}...")
    series, _inputs = _efd.animal_series(animal, evoked_dir, [_CHANNEL_METRIC])
    if not series:
        return None
    return _efd.primary_channel(animal, series, [_CHANNEL_METRIC], override)


def _seizure_onsets(config: dict, animal: str,
                    log: Callable[[str], None] | None) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-leader seizure onsets and their ISI lookback ceilings for *animal*.
    Returns ``(onsets_sorted, ceilings_aligned)``; both empty when <2 leaders."""
    from src.db.store import Store
    store = Store(_cfg.db_path(config))
    included = _isi.included_seizures(store, animal)
    leaders = _isi.leading_seizures(included, _cfg.lead_gap_sec(config))
    if log is not None:
        log(f"seizures: {len(included)} scored, {len(leaders)} cluster leaders")
    if len(leaders) < 2:
        return np.empty(0), np.empty(0)
    onsets = np.array([s.onset_epoch for s in leaders], dtype=np.float64)
    ceilings = _isi.lookback_ceilings(
        leaders, _cfg.post_ictal_buffer_sec(config),
        max_lookback_sec=_cfg.lookback_window_sec(config))
    ceil = np.array([np.nan if c is None else c for c in ceilings], dtype=np.float64)
    order = np.argsort(onsets)
    return onsets[order], ceil[order]


def _stim_coverage(config: dict, animal: str) -> dict:
    """Per session-token stim fingerprint status for *animal* (mapped / record_only /
    unknown), for the report's coverage note."""
    from src.db.store import Store
    store = Store(_cfg.db_path(config))
    tokens = _sm.session_dir_by_token(store, animal)
    out: dict[str, str] = {}
    for tok, sd in tokens.items():
        out[tok] = _sm.resolve_fingerprint(store, sd, animal).status
    return out


def _signature(config: dict, animal: str, channel: str, onsets: np.ndarray) -> str:
    """Cache key over the parameters + the seizure timeline (a new seizure relabels
    time_to_seizure, so it must invalidate the cache)."""
    h = hashlib.sha256()
    parts = [_CACHE_SCHEMA, animal, channel or "", str(_cfg.window_ms(config)),
             str(_cfg.cap(config)), str(_cfg.seed(config)),
             str(_default_days(config)), np.asarray(onsets).tobytes()]
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
    return h.hexdigest()[:16]


def build_dataset(config: dict, animal: str, *, end_day: datetime | None = None,
                  cache: bool = True,
                  log: Callable[[str], None] | None = None) -> dict:
    """Assemble ``{"X", "fs", "time_ms", "meta", "stim_coverage", "channel",
    "n_gathered"}`` for *animal*. ``X`` is ``[n_trials, n_samples]`` amplitude-
    preserving windowed traces; ``meta`` has ``animal, session (day), rec, trial_time,
    time_to_seizure, stim_status, hour``. Empty ``X`` when no traces are found."""
    assert animal, "animal required"
    evoked_dir = _cfg.evoked_dir(config)
    win = _cfg.window_ms(config)
    channel = _resolve_channel(animal, evoked_dir, config, log)
    if channel is None:
        if log is not None:
            log(f"no evoked data for {animal}")
        return {"X": np.empty((0, 0)), "fs": float("nan"), "time_ms": np.empty(0),
                "meta": pd.DataFrame(), "stim_coverage": {}, "channel": None,
                "n_gathered": 0}
    onsets, ceilings = _seizure_onsets(config, animal, log)

    sig = _signature(config, animal, channel, onsets)
    cache_path = os.path.join(_cfg.out_root(config), animal, "cache",
                              f"dataset_{sig}.pkl")
    if cache and os.path.exists(cache_path):
        if log is not None:
            log(f"loading cached dataset ({sig})")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    end = end_day or datetime.now()
    days = _default_days(config)
    if log is not None:
        log(f"gathering traces on {channel} over {days} d (cap {_cfg.cap(config)})...")
    W, tm, meta_g = gather_responses(
        animal, evoked_dir, end, days, channel, win_ms=(float(win[0]), float(win[1])),
        cap=_cfg.cap(config), seed=_cfg.seed(config),
        progress=(lambda m: log(m)) if log is not None else None)
    if W is None:
        return {"X": np.empty((0, 0)), "fs": float("nan"), "time_ms": np.empty(0),
                "meta": pd.DataFrame(), "stim_coverage": _stim_coverage(config, animal),
                "channel": channel, "n_gathered": 0}
    fs = 1000.0 / float(np.mean(np.diff(tm)))
    secs = meta_g["secs"].astype(np.float64)
    tto = (_mx.time_to_onset_for(secs, onsets, ceilings) if onsets.size
           else np.full(secs.shape, np.nan))
    meta = pd.DataFrame({
        "animal": animal,
        "session": meta_g["day"],
        "rec": meta_g["day"],
        "trial_time": secs,
        "time_to_seizure": tto,
        "stim_status": "unknown",
        "hour": meta_g["hour"],
    })
    out = {"X": np.asarray(W, dtype=np.float64), "fs": fs, "time_ms": np.asarray(tm),
           "meta": meta, "stim_coverage": _stim_coverage(config, animal),
           "channel": channel, "n_gathered": int(W.shape[0])}
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_path)
    if log is not None:
        n_lab = int(np.isfinite(tto).sum())
        log(f"gathered {out['n_gathered']} trials; {n_lab} within a pre-ictal window")
    return out
