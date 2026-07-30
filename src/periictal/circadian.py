"""Circadian-matched interictal reference (Part B, B4) -- the highest-value control.

The evoked features are circadian at |rho|~0.67 (far above the ~0.2 time-to-onset
association), and the standard interictal window (60-90 min before onset) sits at a
DIFFERENT clock time than the preictal window (0-30 min), so a circadian gradient
leaks straight onto the preictal-vs-interictal axis. This module instead draws the
interictal reference from the **same clock time on non-seizure days** -- holding
time-of-day fixed BY CONSTRUCTION rather than regressing it out. If a feature's AUC
survives circadian matching it means something; if it collapses to ~0.5 the effect
was circadian. Settled in an afternoon, not a cohort.

Consumes a TrialSeries (B0), which already carries secs/values/onsets/
time_to_onset. Returns BOTH the matched AUC and the standard same-day AUC so the
survives-or-collapses comparison is one call.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.preictal.scoring import rank_auc

_PREICTAL_MAX = 1800.0                 # 0-30 min before onset
_INTER_LO, _INTER_HI = 3600.0, 5400.0  # 60-90 min before onset (same-day baseline)
_MAX_SEIZURES = 1_000_000


def hour_of_day(secs) -> np.ndarray:
    """Local clock hour (0-24) for epoch seconds -- matches matrix._hour_of_day."""
    secs = np.asarray(secs, dtype=float)
    out = np.full(secs.size, np.nan)
    for i, s in enumerate(secs):
        assert i < 500_000_000, "hour-of-day scan runaway"
        try:
            d = datetime.fromtimestamp(float(s))
            out[i] = d.hour + d.minute / 60.0 + d.second / 3600.0
        except (OSError, ValueError, OverflowError):
            pass
    return out


def _circ_dist_h(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 24.0
    return np.minimum(d, 24.0 - d)


def _auc(pre, other, min_n):
    pre = pre[np.isfinite(pre)]
    other = other[np.isfinite(other)]
    if pre.size < min_n or other.size < min_n:
        return float("nan"), int(pre.size), int(other.size)
    return float(rank_auc(pre, other)), int(pre.size), int(other.size)


def circadian_matched_auc(series, feature=None, *, preictal_max_sec=_PREICTAL_MAX,
                          hour_tol=0.75, exclude_onset_sec=6 * 3600.0,
                          min_n=20) -> dict:
    """For each seizure, AUC of preictal vs (a) the CIRCADIAN-MATCHED interictal
    set -- stimuli at the same hour-of-day (+/- hour_tol) that are far
    (> exclude_onset_sec) from EVERY onset -- and (b) the standard same-day
    interictal window (60-90 min pre). Returns per-seizure + pooled for both, so a
    caller can see whether the effect survives matching. *feature* is informational
    (values come from the TrialSeries)."""
    secs = np.asarray(series.secs, dtype=float)
    vals = np.asarray(series.values, dtype=float)
    onsets = np.asarray(series.onsets, dtype=float)
    onsets = onsets[np.isfinite(onsets)]
    base = {"feature": feature or getattr(series, "feature", None),
            "per_seizure": [], "matched_pooled_auc": float("nan"),
            "sameday_pooled_auc": float("nan"), "n_seizures": 0,
            "hour_tol": hour_tol, "exclude_onset_sec": exclude_onset_sec}
    if secs.size == 0 or onsets.size == 0:
        return base
    assert onsets.size < _MAX_SEIZURES, "seizure count runaway"
    hod = hour_of_day(secs)
    far = np.abs(np.asarray(series.time_to_onset, dtype=float)) > exclude_onset_sec
    per, pre_m, matched_m, pre_s, same_s = [], [], [], [], []
    for o in onsets:
        d = o - secs
        pre = vals[(d > 0) & (d <= preictal_max_sec)]
        pre = pre[np.isfinite(pre)]
        if pre.size < min_n:
            continue
        center = float(np.nanmedian(hod[(d > 0) & (d <= preictal_max_sec)]))
        matched = vals[far & (_circ_dist_h(hod, center) <= hour_tol)]
        sameday = vals[(d >= _INTER_LO) & (d <= _INTER_HI)]
        m_auc, n_pre, n_m = _auc(pre, matched, min_n)
        s_auc, _n2, n_s = _auc(pre, sameday, min_n)
        per.append({"onset_epoch": float(o), "hod": center,
                    "matched_auc": m_auc, "sameday_auc": s_auc,
                    "n_pre": n_pre, "n_matched": n_m, "n_sameday": n_s})
        if np.isfinite(m_auc):
            pre_m.append(pre)
            matched_m.append(matched[np.isfinite(matched)])
        if np.isfinite(s_auc):
            pre_s.append(pre)
            same_s.append(sameday[np.isfinite(sameday)])
    base["per_seizure"] = per
    base["n_seizures"] = len(per)
    if pre_m:
        base["matched_pooled_auc"] = float(
            rank_auc(np.concatenate(pre_m), np.concatenate(matched_m)))
    if pre_s:
        base["sameday_pooled_auc"] = float(
            rank_auc(np.concatenate(pre_s), np.concatenate(same_s)))
    return base
