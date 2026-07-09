"""Validation: leave-one-SEIZURE-out CV + surrogate-null statistics.

Two guards turn a per-scale collapse number into a defensible result:

* **Leave-one-SEIZURE-out** (never leave-one-sample-out). CWT coefficients are
  autocorrelated within a seizure -- worse at long scales -- so sample-level
  folds leak pre-ictal structure across the split and fake tight CIs. Whole
  seizures are held out; the spread of the collapse across folds is the CI.
* **Surrogate null.** The observed collapse is placed in a distribution of
  collapses computed the SAME way around non-seizure-locked anchors (generated
  by the engine -- random/interictal anchors, the strong null; NOT a bare
  circular shift). Structure that survives (low empirical p) is seizure-locked;
  structure that doesn't is generic drift.

Data model: ``per_seizure_bins`` is a list over seizures; each element is a list
of ``n_bins`` 1-D arrays -- the |CWT coefficient| values that fell in each
lead-time bin FOR THAT SEIZURE (nearest bin first).
"""

from __future__ import annotations

import numpy as np

from src.preictal import scoring


def _pool(per_seizure_bins: list, seiz_idx, n_bins: int) -> list:
    """Concatenate, per bin, the values from the given seizures -> pooled bin
    arrays (the input to one AUC matrix)."""
    pooled = []
    for b in range(n_bins):
        parts = [np.asarray(per_seizure_bins[s][b], dtype=float)
                 for s in seiz_idx
                 if b < len(per_seizure_bins[s])
                 and len(per_seizure_bins[s][b])]
        pooled.append(np.concatenate(parts) if parts
                      else np.array([], dtype=float))
    return pooled


def collapse_over(per_seizure_bins: list, seiz_idx, n_bins: int,
                   collapse_fn=scoring.collapse_gradient) -> float:
    """Collapse statistic pooled over the given seizures."""
    return collapse_fn(scoring.auc_matrix(_pool(per_seizure_bins, seiz_idx,
                                                n_bins)))


def loso_collapse(per_seizure_bins: list,
                   collapse_fn=scoring.collapse_gradient) -> dict:
    """Leave-one-SEIZURE-out jackknife of the collapse statistic. Returns
    ``{mean, ci_lo, ci_hi, n_folds}`` (2.5/97.5 percentile CI across folds).
    With < 2 seizures there's nothing to hold out -> the all-seizure value with
    NaN CI."""
    s = len(per_seizure_bins)
    n_bins = len(per_seizure_bins[0]) if s else 0
    if s < 2:
        c = collapse_over(per_seizure_bins, range(s), n_bins,
                          collapse_fn) if s else float("nan")
        return {"mean": c, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_folds": s}
    folds = []
    for h in range(s):
        keep = [i for i in range(s) if i != h]      # whole seizure held out
        folds.append(collapse_over(per_seizure_bins, keep, n_bins, collapse_fn))
    folds = np.asarray([f for f in folds if np.isfinite(f)], dtype=float)
    if folds.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_folds": 0}
    return {"mean": float(folds.mean()),
            "ci_lo": float(np.percentile(folds, 2.5)),
            "ci_hi": float(np.percentile(folds, 97.5)),
            "n_folds": int(folds.size)}


def null_stats(observed: float, surrogates) -> dict:
    """Where does *observed* fall in the surrogate-null distribution? Returns
    ``{mean, std, percentile, p}`` -- percentile = fraction of surrogates below
    observed (x100); p = one-sided upper empirical p with +1 smoothing
    ``(#surrogate >= observed + 1) / (n + 1)``. NaN when no surrogates."""
    s = np.asarray([x for x in surrogates if np.isfinite(x)], dtype=float)
    if s.size == 0 or not np.isfinite(observed):
        return {"mean": float("nan"), "std": float("nan"),
                "percentile": float("nan"), "p": float("nan")}
    return {"mean": float(s.mean()), "std": float(s.std()),
            "percentile": float(np.mean(s < observed) * 100.0),
            "p": float((np.sum(s >= observed) + 1) / (s.size + 1))}
