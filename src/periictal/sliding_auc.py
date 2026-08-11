"""Sliding-window ROC-AUC test for the peri-ictal explorer.

The fixed preictal window (0, PREICTAL_MAX] stays the positive class; the
interictal reference is sampled as N evenly-spaced WIDTH-wide windows across a
fixed lookback BAND before onset (config.SLIDING_*), each window dropped when its
far edge would fall within POSTICTAL_GUARD of the PREVIOUS seizure. For every
selected seizure we score each feature's preictal-vs-window rank AUC (normalised
to max(AUC, 1-AUC)), then aggregate across seizures.

Pure + Dash-free so it is unit-testable; the lens calls in here. Operates on the
``event x metric`` matrix (``matrix.build_matrix``) whose relevant meta columns
are ``time_to_onset_sec`` (signed, positive = pre-onset), ``phase`` (pre/post),
``seizure_idx``, ``seizure_onset_epoch``.

Statistical unit is the SEIZURE: the group number is the mean of per-seizure
AUCs, not a pooled-across-stimuli AUC (which would reintroduce the within-window
pseudo-replication ``forecast.py`` warns about).
"""

from __future__ import annotations

import numpy as np

from src.periictal import config as _cfg
from src.preictal.scoring import rank_auc

_MAX_SEIZURES = 1_000_000
_REQUIRED = ("time_to_onset_sec", "seizure_idx", "seizure_onset_epoch", "phase")


def _isi_map(onsets) -> dict:
    """Map each onset epoch to its inter-seizure interval (seconds since the
    previous distinct onset); the first seizure maps to +inf."""
    o = np.unique(np.asarray(onsets, dtype=float))
    o = o[np.isfinite(o)]
    out, prev = {}, None
    for x in o:
        out[float(x)] = (x - prev) if prev is not None else np.inf
        prev = x
    return out


def canonical_offsets(*, band_lo=_cfg.SLIDING_BAND_LO_SEC,
                      band_hi=_cfg.SLIDING_BAND_HI_SEC,
                      width=_cfg.SLIDING_WIDTH_SEC,
                      n=_cfg.SLIDING_N_WINDOWS) -> np.ndarray:
    """The canonical grid of interictal-window START offsets (seconds before
    onset), shared across seizures so the AUC columns align. N evenly spaced in
    [band_lo, band_hi-width]."""
    assert width > 0, "window width must be positive"
    assert int(n) >= 1, "need at least one window"
    hi = band_hi - width
    assert hi >= band_lo, "band too narrow for the window width"
    if int(n) == 1:
        return np.array([band_lo], dtype=float)
    return np.linspace(band_lo, hi, int(n), dtype=float)


def valid_offset_mask(offsets, isi_sec, *, width=_cfg.SLIDING_WIDTH_SEC,
                      postictal_guard=_cfg.SLIDING_POSTICTAL_GUARD_SEC) -> np.ndarray:
    """Which canonical offsets are usable for a seizure with this ISI: a window's
    far edge (offset+width before onset) must stay >= postictal_guard after the
    previous seizure, i.e. offset <= isi - width - guard. First seizure (inf) keeps
    all."""
    offsets = np.asarray(offsets, dtype=float)
    hi = isi_sec - width - postictal_guard      # +inf for the first seizure
    return offsets <= hi + 1e-6


def _fmt_offset(sec: float) -> str:
    """Short human label for a window offset, e.g. '3.5 h'."""
    return f"{sec / 3600.0:.1f} h"


def _seizure_auc(fv_s, tto_s, offsets, vmask, *, width, preictal_max, min_n):
    """AUC (auc_norm) matrix [feature x window] for ONE seizure. fv_s is the
    seizure's feature matrix (rows x n_features); invalid/undersized cells NaN."""
    n_feat = fv_s.shape[1]
    auc = np.full((n_feat, offsets.size), np.nan, dtype=float)
    pre_sel = (tto_s > 0) & (tto_s <= preictal_max)
    if not pre_sel.any():
        return auc
    for wj in range(offsets.size):              # bounded: N windows
        if not vmask[wj]:
            continue
        o = offsets[wj]
        inter_sel = (tto_s >= o) & (tto_s <= o + width)
        if not inter_sel.any():
            continue
        for fi in range(n_feat):                # bounded: n features
            a = fv_s[pre_sel, fi]
            b = fv_s[inter_sel, fi]
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size >= min_n and b.size >= min_n:
                r = rank_auc(a, b)
                auc[fi, wj] = max(r, 1.0 - r)
    return auc


def _rank_features(score_by_feature, features, k=5):
    """Feature names sorted by descending score (NaN last); top-k slice."""
    order = sorted(features,
                   key=lambda f: (-score_by_feature[f]
                                  if np.isfinite(score_by_feature[f]) else np.inf))
    return order, order[:k]


def sliding_window_auc(full, features, *,
                       n_windows=_cfg.SLIDING_N_WINDOWS,
                       band_lo=_cfg.SLIDING_BAND_LO_SEC,
                       band_hi=_cfg.SLIDING_BAND_HI_SEC,
                       width=_cfg.SLIDING_WIDTH_SEC,
                       postictal_guard=_cfg.SLIDING_POSTICTAL_GUARD_SEC,
                       preictal_max=_cfg.PREICTAL_MAX_SEC,
                       min_preictal_isi=_cfg.MIN_PREICTAL_SEC,
                       min_n=_cfg.SLIDING_MIN_N, top_k=5) -> dict:
    """Per-seizure and group sliding-window preictal-vs-interictal AUC. See the
    module docstring for the result shape."""
    assert set(_REQUIRED) <= set(full.columns), "matrix missing required columns"
    feats = [f for f in features if f in full.columns]
    assert feats, "no requested feature columns present in the matrix"
    offsets = canonical_offsets(band_lo=band_lo, band_hi=band_hi,
                                width=width, n=n_windows)
    win_labels = [_fmt_offset(o) for o in offsets]

    pre = full[full["phase"].to_numpy() == "pre"]
    tto = pre["time_to_onset_sec"].to_numpy(dtype=float)
    sid = pre["seizure_idx"].to_numpy()
    onset = pre["seizure_onset_epoch"].to_numpy(dtype=float)
    fv = pre[feats].to_numpy(dtype=float)               # rows x n_features, once
    isi = _isi_map(onset)

    sids = np.unique(sid)
    assert sids.size < _MAX_SEIZURES, "seizure count runaway"
    per_seizure, notes = {}, []
    for s in sids:                                       # bounded: seizure count
        m = sid == s
        isi_s = float(isi.get(float(onset[m][0]), np.inf))
        if isi_s < min_preictal_isi:
            notes.append(f"seizure {int(s)}: dropped (ISI "
                         f"{isi_s / 60:.0f} min < {min_preictal_isi / 60:.0f} min)")
            continue
        vmask = valid_offset_mask(offsets, isi_s, width=width,
                                  postictal_guard=postictal_guard)
        auc = _seizure_auc(fv[m], tto[m], offsets, vmask, width=width,
                           preictal_max=preictal_max, min_n=min_n)
        n_valid = int(np.isfinite(auc).any(axis=0).sum())
        if n_valid == 0:
            notes.append(f"seizure {int(s)}: no scorable interictal window "
                         "(short ISI / too few stimuli)")
            continue
        mean_auc = {f: float(np.nanmean(auc[fi])) if np.isfinite(auc[fi]).any()
                    else float("nan") for fi, f in enumerate(feats)}
        ranked, top5 = _rank_features(mean_auc, feats, k=top_k)
        per_seizure[int(s)] = {"auc": auc, "valid": vmask, "n_valid": n_valid,
                               "mean_auc": mean_auc, "ranked": ranked, "top5": top5}

    group = _group_from_per_seizure(per_seizure, feats)
    group_ranked, group_top5 = _rank_features(group, feats, k=top_k)
    best = _best_per_seizure(per_seizure, group_top5)
    return {"offsets": offsets, "win_labels": win_labels, "features": feats,
            "per_seizure": per_seizure, "group": group,
            "group_ranked": group_ranked, "group_top5": group_top5,
            "best_per_seizure": best, "n_seizures_used": len(per_seizure),
            "notes": notes}


def _group_from_per_seizure(per_seizure, feats) -> dict:
    """Group AUC per feature = MEAN across seizures of the per-seizure mean AUC
    (seizure is the unit). NaN when no seizure scored the feature."""
    out = {}
    for f in feats:
        vals = [ps["mean_auc"][f] for ps in per_seizure.values()
                if np.isfinite(ps["mean_auc"][f])]
        out[f] = float(np.mean(vals)) if vals else float("nan")
    return out


def sliding_class_column(full, *, n_windows=_cfg.SLIDING_N_WINDOWS,
                         band_lo=_cfg.SLIDING_BAND_LO_SEC,
                         band_hi=_cfg.SLIDING_BAND_HI_SEC,
                         width=_cfg.SLIDING_WIDTH_SEC,
                         postictal_guard=_cfg.SLIDING_POSTICTAL_GUARD_SEC,
                         preictal_max=_cfg.PREICTAL_MAX_SEC,
                         min_preictal_isi=_cfg.MIN_PREICTAL_SEC) -> np.ndarray:
    """A 'class' array over ``full``'s rows -- 'preictal' / 'interictal' / '' --
    using the SAME sliding geometry, for a POOLED PDF/CDF. Interictal = the union
    of a seizure's valid sampled windows (so the density matches the AUC test's
    reference, not the fixed 60-90 min band)."""
    assert set(_REQUIRED) <= set(full.columns), "matrix missing required columns"
    cls = np.full(len(full), "", dtype=object)
    is_pre = full["phase"].to_numpy() == "pre"
    tto = full["time_to_onset_sec"].to_numpy(dtype=float)
    sid = full["seizure_idx"].to_numpy()
    onset = full["seizure_onset_epoch"].to_numpy(dtype=float)
    offsets = canonical_offsets(band_lo=band_lo, band_hi=band_hi,
                                width=width, n=n_windows)
    isi = _isi_map(onset[is_pre])
    for s in np.unique(sid[is_pre]):                     # bounded: seizure count
        m = is_pre & (sid == s)
        isi_s = float(isi.get(float(onset[m][0]), np.inf))
        if isi_s < min_preictal_isi:
            continue
        vmask = valid_offset_mask(offsets, isi_s, width=width,
                                  postictal_guard=postictal_guard)
        tto_s = tto[m]
        idx = np.where(m)[0]
        cls[idx[(tto_s > 0) & (tto_s <= preictal_max)]] = "preictal"
        inter = np.zeros(tto_s.size, dtype=bool)
        for wj in range(offsets.size):                   # bounded: N windows
            if vmask[wj]:
                inter |= (tto_s >= offsets[wj]) & (tto_s <= offsets[wj] + width)
        cls[idx[inter]] = "interictal"
    return cls


def _best_per_seizure(per_seizure, group_top5) -> list:
    """For each seizure, its best mean AUC among the GROUP top-5 features (the
    seizure's best achievable discrimination using the consensus feature set)."""
    out = []
    for s, ps in per_seizure.items():
        cand = [(ps["mean_auc"][f], f) for f in group_top5
                if np.isfinite(ps["mean_auc"].get(f, float("nan")))]
        if not cand:
            out.append({"seizure_idx": int(s), "best_auc": float("nan"),
                        "best_feature": None})
            continue
        best_auc, best_f = max(cand, key=lambda t: t[0])
        out.append({"seizure_idx": int(s), "best_auc": float(best_auc),
                    "best_feature": best_f})
    out.sort(key=lambda r: int(r["seizure_idx"]))
    return out
