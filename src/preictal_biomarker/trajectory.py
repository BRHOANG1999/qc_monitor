"""Per-seizure dominant-state trajectory (states don't flip every stimulus).

For each seizure, bin epochs into fixed windows relative to onset and take the
dominant (most-common) state per bin. This is the substrate for the per-seizure
trajectory heatmap (Fig 06a) and for the continuous timeline (timeline.py).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from . import config as C


def dominant_state(states: np.ndarray, k: int) -> int:
    """Most-common valid state in a bin; -1 if none."""
    s = np.asarray(states)
    s = s[s >= 0]
    if s.size == 0:
        return -1
    return int(np.bincount(s, minlength=k).argmax())


def per_seizure_trajectory(df: pd.DataFrame, onsets: np.ndarray, *,
                           pre_min: float = 120.0, post_min: float = 20.0,
                           bin_min: float = 10.0, k: int | None = None) -> dict:
    """[n_onsets x n_bins] dominant-state matrix, minutes-from-onset on x.
    Rows ordered by onset time; -1 = no data in that bin."""
    k = int(k or C.K_STATES)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    edges = np.arange(-pre_min, post_min + bin_min, bin_min)      # minutes
    centers = (edges[:-1] + edges[1:]) / 2.0
    ons = np.sort(np.asarray(onsets, float))
    M = np.full((ons.size, centers.size), -1, dtype=int)
    for r, o in enumerate(ons):
        rel = (t - o) / 60.0                                      # minutes
        for c in range(centers.size):
            m = (rel >= edges[c]) & (rel < edges[c + 1])
            M[r, c] = dominant_state(st[m], k)
    labels = [_dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M") for o in ons]
    return {"matrix": M, "edges": edges, "centers": centers,
            "onsets": ons, "labels": labels}


def per_seizure_state_fraction(df: pd.DataFrame, onsets: np.ndarray, *,
                               target: int = 1, pre_min: float = 120.0,
                               post_min: float = 20.0, bin_min: float = 10.0) -> dict:
    """[n_onsets x n_bins] FRACTION of valid epochs in *target* state per bin (NaN if
    no data) -- tracks occurrence of a single (possibly rare) state, since a rare
    state is never the dominant one. Rows ordered by onset time."""
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    edges = np.arange(-pre_min, post_min + bin_min, bin_min)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ons = np.sort(np.asarray(onsets, float))
    M = np.full((ons.size, centers.size), np.nan)
    for r, o in enumerate(ons):
        rel = (t - o) / 60.0
        for c in range(centers.size):
            s = st[(rel >= edges[c]) & (rel < edges[c + 1])]
            s = s[s >= 0]
            if s.size:
                M[r, c] = float(np.mean(s == target))
    labels = [_dt.datetime.fromtimestamp(o).strftime("%m-%d %H:%M") for o in ons]
    return {"matrix": M, "edges": edges, "centers": centers, "onsets": ons,
            "labels": labels, "target": int(target)}


def state_fraction_timeline(df: pd.DataFrame, *, target: int = 1,
                            bin_sec: float = 600.0) -> dict:
    """Continuous FRACTION-in-*target* series over absolute time (NaN = empty bin)."""
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    ok = np.isfinite(t)
    t, st = t[ok], st[ok]
    lo, hi = t.min(), t.max()
    edges = np.arange(lo, hi + bin_sec, bin_sec)
    centers = edges[:-1] + bin_sec / 2.0
    frac = np.full(centers.size, np.nan)
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, centers.size - 1)
    for c in range(centers.size):
        s = st[which == c]; s = s[s >= 0]
        if s.size:
            frac[c] = float(np.mean(s == target))
    return {"edges": edges, "centers": centers, "frac": frac,
            "bin_sec": float(bin_sec), "target": int(target)}


def dominant_state_timeline(df: pd.DataFrame, *, bin_sec: float = 600.0,
                            k: int | None = None) -> dict:
    """Continuous dominant-state series over absolute time: one bin per *bin_sec*
    from first to last epoch; -1 where a bin has no epochs."""
    k = int(k or C.K_STATES)
    t = pd.to_numeric(df["t_epoch"], errors="coerce").to_numpy(float)
    st = df["state"].to_numpy()
    ok = np.isfinite(t)
    t, st = t[ok], st[ok]
    lo, hi = t.min(), t.max()
    edges = np.arange(lo, hi + bin_sec, bin_sec)
    centers = edges[:-1] + bin_sec / 2.0
    dom = np.full(centers.size, -1, dtype=int)
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0,
                    centers.size - 1)
    for c in range(centers.size):
        dom[c] = dominant_state(st[which == c], k)
    return {"edges": edges, "centers": centers, "dominant": dom,
            "bin_sec": float(bin_sec)}
