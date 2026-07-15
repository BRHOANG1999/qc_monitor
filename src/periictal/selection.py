"""Summarize a lasso/box selection from the embedding for the details panel.

The whole point of letting the user select a cluster is to answer "is this a
real signal or a confound?" -- so the summary reports the selection's
distribution in TIME-TO-ONSET (is it one lead-time band?), in HOUR-OF-DAY (is it
just one time of day -> circadian confound?), and its breakdown by source
SEIZURE and RECORDING (is it just one recording -> a batch/drift effect?).

Pure + Dash-free so it can be unit-tested; the callback stays thin.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

_MAX_ROWS = 5_000_000        # NASA Rule 2.


def summarize_selection(sub: pd.DataFrame, idx) -> dict:
    """Aggregate the selected rows (positional indices *idx* into *sub*).

    Returns a dict with: n / n_total; tto_hours + hour_of_day value lists (for
    histograms); by_seizure and by_recording as ordered (label, count) lists;
    and lead_frac = the fraction of the selection in its single most-common
    lead-time bin (a quick 'how concentrated in time' scalar)."""
    assert sub is not None, "sub required"
    idx = np.asarray(idx if idx is not None else [], dtype=int)
    idx = idx[(idx >= 0) & (idx < len(sub))]
    assert idx.size < _MAX_ROWS, "selection too large"
    sel = sub.iloc[idx]
    out = {"n": int(len(sel)), "n_total": int(len(sub))}
    if len(sel) == 0:
        return {**out, "tto_hours": [], "hour_of_day": [],
                "by_seizure": [], "by_recording": [], "lead_frac": None}
    out["tto_hours"] = (sel["time_to_onset_sec"].to_numpy() / 3600.0).tolist()
    out["hour_of_day"] = sel["hour_of_day"].to_numpy().tolist()
    out["by_seizure"] = _by_seizure(sel)
    out["by_recording"] = _top_counts(sel.get("rec"), top=6)
    out["lead_frac"] = _dominant_fraction(sel.get("lead_bin"))
    return out


def _by_seizure(sel: pd.DataFrame) -> list:
    """(seizure label, count) ordered by seizure onset. Label = onset datetime
    so the user can see WHICH seizure's lead-up the cluster came from."""
    if "seizure_idx" not in sel or "seizure_onset_epoch" not in sel:
        return []
    counts = Counter(int(i) for i in sel["seizure_idx"].to_numpy())
    onset = dict(zip(sel["seizure_idx"].to_numpy(),
                     sel["seizure_onset_epoch"].to_numpy()))
    rows = []
    for sz in sorted(counts, key=lambda s: onset.get(s, 0.0)):
        rows.append((_fmt_epoch(onset.get(sz)), counts[sz]))
    return rows


def _top_counts(series, top: int) -> list:
    """Top-*top* (value, count) pairs of a categorical column, most common
    first; remaining folded into ('other (k)', total)."""
    if series is None:
        return []
    counts = Counter(str(v) for v in series.to_numpy())
    common = counts.most_common(top)
    rest = len(counts) - len(common)
    out = [(_short_rec(v), c) for v, c in common]
    if rest > 0:
        out.append((f"other ({rest})", sum(counts.values())
                    - sum(c for _, c in common)))
    return out


def _dominant_fraction(series) -> float | None:
    """Fraction of the selection sitting in its single most-common lead-time
    bin -- high => the cluster is concentrated at one lead time."""
    if series is None or len(series) == 0:
        return None
    counts = Counter(int(b) for b in series.to_numpy())
    return round(max(counts.values()) / sum(counts.values()), 3)


def _fmt_epoch(epoch) -> str:
    try:
        return datetime.fromtimestamp(float(epoch)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _short_rec(rec_iso: str) -> str:
    """Recording id (its start datetime) shortened for the table."""
    try:
        return datetime.fromisoformat(rec_iso).strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return rec_iso[:16] if rec_iso else "?"
