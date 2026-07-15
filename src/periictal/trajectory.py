"""Count-agnostic lead-time trajectory.

At 0.5 Hz, stimuli are uniform in LINEAR time, so near-onset log-lead-time bins
are sparse and far-onset bins are abundant -- colouring a per-stimulus scatter by
time is dominated by the far field. Binning lead-time into dyadic log bins and
summarising each bin with ONE value gives every time-scale equal weight, so the
count imbalance can't distort the trend. Per-seizure medians are kept alongside
the aggregate, because the effective unit of replication is the SEIZURE, not the
stimulus -- a 'trend' driven by a single seizure must be visible.

Pure numpy so it can be unit-tested; the tab renders the returned summary.
"""

from __future__ import annotations

import numpy as np

_MAX_BINS = 128        # NASA Rule 2.


def default_edges(window_sec: float, min_leadtime_sec: float = 2.0) -> np.ndarray:
    """Dyadic bin edges (seconds-before-onset) from *min_leadtime_sec* up to
    *window_sec*, reusing the pre-ictal engine's binning. Falls back to a manual
    dyadic ladder if the window is too short for the shared helper."""
    from src.preictal.isi import leadtime_bins
    edges = leadtime_bins(float(window_sec), float(min_leadtime_sec)).edges_sec
    if len(edges) >= 2:
        return np.asarray(edges, dtype=float)
    lo, hi = float(min_leadtime_sec), float(max(window_sec, min_leadtime_sec * 2))
    ladder = [lo]
    while ladder[-1] < hi and len(ladder) < _MAX_BINS:
        ladder.append(ladder[-1] * 2.0)
    return np.asarray(ladder, dtype=float)


def lead_time_trajectory(tto_sec, values, seizure_idx, edges,
                         min_n: int = 5) -> dict:
    """Median (+ p25/p75) of *values* per lead-time bin, plus per-seizure medians.

    *edges* are ascending seconds-before-onset. Returns
    ``{centers, median, p25, p75, n, low_n, per_seizure}`` where centers are the
    bins' geometric means, low_n marks bins with < *min_n* finite values
    (near-onset bins are naturally sparse -- flagged, not silently trusted), and
    per_seizure maps seizure id -> its own per-bin median (NaN where empty)."""
    tto = np.asarray(tto_sec, dtype=float)
    v = np.asarray(values, dtype=float)
    sz = np.asarray(seizure_idx)
    edges = np.asarray(edges, dtype=float)
    assert edges.ndim == 1 and edges.size >= 2, "need >= 2 bin edges"
    assert tto.shape == v.shape == sz.shape, "tto/values/seizure_idx mismatch"
    nb = int(edges.size - 1)
    assert nb < _MAX_BINS, "too many bins"
    centers = np.sqrt(edges[:-1] * edges[1:])            # geometric-mean centres
    idx = np.digitize(tto, edges) - 1                    # bin index per point
    med = np.full(nb, np.nan)
    p25 = np.full(nb, np.nan)
    p75 = np.full(nb, np.nan)
    n = np.zeros(nb, dtype=int)
    for b in range(nb):
        vb = v[(idx == b) & np.isfinite(v)]
        n[b] = vb.size
        if vb.size:
            med[b], p25[b], p75[b] = np.percentile(vb, [50, 25, 75])
    return {"centers": centers, "median": med, "p25": p25, "p75": p75,
            "n": n, "low_n": n < int(min_n),
            "per_seizure": _per_seizure(idx, v, sz, nb)}


def _per_seizure(idx: np.ndarray, v: np.ndarray, sz: np.ndarray,
                 nb: int) -> dict:
    """Each seizure's per-bin median (NaN where it has no point in a bin)."""
    out: dict = {}
    for s in np.unique(sz):
        m = np.full(nb, np.nan)
        sel = sz == s
        for b in range(nb):
            vb = v[sel & (idx == b) & np.isfinite(v)]
            if vb.size:
                m[b] = np.median(vb)
        out[int(s)] = m
    return out
