"""Per-epoch pHFO occurrence feature, from the riding_event detector.

For each in-scope, near-seizure recording, `riding_event.periictal.
evoked_riding_recording` subtracts the robust-median evoked response and gates the
2-50 ms residual for a riding pHFO/LFD packet (HF-SNR, or prominence+fraction),
returning the absolute epoch-second times of pHFO-flagged vs all stimuli (cached
per file). We turn that into a per-epoch 0/1 `phfo_present` and join it to the
feature matrix by epoch time. (pHFO / LFD -- never "ripple".)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.riding_event.periictal import animal_recordings, evoked_riding_recording
from src.periictal import matrix as _mx, config as _pcfg
from . import config as C, features as F

_PHFO_CACHE = C.CACHE_DIR + "/BCH111_phfo.pkl"


def _log(m):
    print(f"[preictal_biomarker.phfo] {m}", flush=True)


def build_phfo(store, *, force=False) -> pd.DataFrame:
    """Per-epoch pHFO table [t_epoch, phfo_present] for BCH111SR, in-scope +
    near-seizure recordings. Cached."""
    if not force and os.path.exists(_PHFO_CACHE):
        _log(f"loading cached pHFO table: {_PHFO_CACHE}")
        return pd.read_pickle(_PHFO_CACHE)
    onsets = F.scoped_onsets(store)
    keep = _mx._near_seizure_file_filter(onsets, C.LOOKBACK_SEC,
                                         _pcfg.PERIICTAL_PREFILTER_SLACK_SEC)
    s0, s1 = C.ANALYSIS_START.timestamp(), C.ANALYSIS_END.timestamp()
    recs = [r for r in animal_recordings(store, C.ANIMAL)
            if s0 <= float(r.get("start_epoch", 0)) < s1]
    _log(f"{len(recs)} in-scope recordings; detecting riding pHFO per epoch "
         f"(evoked read, cached per file) ...")
    ts, pres = [], []
    for i, rec in enumerate(recs):
        if i % 20 == 0:
            _log(f"  recording {i}/{len(recs)}")
        try:
            flag, stim = evoked_riding_recording(store, C.ANIMAL, rec,
                                                 win_ms=C.WINDOW_MS)
        except Exception as e:                     # noqa: BLE001 - skip bad file
            _log(f"  skip {rec.get('file_id')}: {type(e).__name__}")
            continue
        if stim.size == 0:
            continue
        ts.append(np.asarray(stim, float))
        pres.append(np.isin(stim, flag).astype(np.int8))
    assert ts, "no pHFO epochs extracted"
    df = pd.DataFrame({"t_epoch": np.concatenate(ts),
                       "phfo_present": np.concatenate(pres)})
    df = df.sort_values("t_epoch").reset_index(drop=True)
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    df.to_pickle(_PHFO_CACHE)
    _log(f"cached -> {_PHFO_CACHE}  ({len(df)} epochs, "
         f"{100*df['phfo_present'].mean():.1f}% pHFO+)")
    return df


def attach_phfo(feat_df: pd.DataFrame, phfo_df: pd.DataFrame, *,
                tol_sec: float = 2.0) -> pd.DataFrame:
    """Join phfo_present onto feat_df by nearest epoch time within tol_sec. Both
    t_epoch are recording_start + per-stim offset, so they match to sub-second;
    unmatched rows get NaN (dropped later if phfo is in the feature set)."""
    out = feat_df.copy()
    pt = phfo_df["t_epoch"].to_numpy(float)
    pv = phfo_df["phfo_present"].to_numpy(float)
    order = np.argsort(pt)
    pt, pv = pt[order], pv[order]
    ft = pd.to_numeric(out["t_epoch"], errors="coerce").to_numpy(float)
    pos = np.clip(np.searchsorted(pt, ft), 1, pt.size - 1)
    left = pt[pos - 1]; right = pt[pos]
    near = np.where(np.abs(ft - left) <= np.abs(ft - right), pos - 1, pos)
    val = np.full(ft.size, np.nan)
    good = np.abs(pt[near] - ft) <= tol_sec
    val[good] = pv[near][good]
    out["phfo_present"] = val
    return out


if __name__ == "__main__":
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    from src.db.store import Store
    from src.dashboard.data_helpers import load_config
    cfg = load_config()
    build_phfo(Store(cfg["database"]["path"]), force="--force" in sys.argv)
