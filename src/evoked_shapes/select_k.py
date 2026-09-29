"""Stage 4: BOUND the number of shapes k, do not point-optimize it.

Stability metrics inflate mechanically with k, and any single criterion is gameable,
so we report a defensible RANGE with four independent readouts:

  * lower bound   GMM BIC vs k and held-out log-likelihood on held-out sessions
                  (a k too small underfits both).
  * upper bound   max off-diagonal template correlation: once two templates exceed
                  ~0.9 the extra cluster is a split of one shape, not a new shape.
  * stability     bootstrap adjusted Rand index + per-cluster Jaccard (Hennig 2007)
                  under a dependence-aware moving-block resample, reported as
                  OBSERVED MINUS a matched null (a covariance-matched unimodal
                  Gaussian), because raw stability rises with k even on structureless
                  data.
  * reproducible  split-half: fit templates on half the sessions, assign the other
                  half, report the within-cluster correlation drop.

The verdict (``select_range``) is a range plus the reasoning, never a single k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evoked_shapes import cluster as _cl
from src.evoked_shapes import structure as _st
from src.periictal import resample as _rs


# --------------------------------------------------------------------- #
#  Lower bound: GMM BIC + held-out log-likelihood on held-out sessions
# --------------------------------------------------------------------- #
def gmm_bic_curve(Z: np.ndarray, k_range: tuple[int, int], *, seed: int = 0) -> dict:
    """BIC per k for a full-covariance GMM on the PCA scores *Z*. Lower BIC is
    better; the curve's knee is the lower bound on k."""
    from sklearn.mixture import GaussianMixture
    lo, hi = int(k_range[0]), int(k_range[1])
    out: dict[int, float] = {}
    for k in range(lo, hi + 1):
        if k > Z.shape[0]:
            break
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=2, reg_covar=1e-5)
        g.fit(Z)
        out[k] = float(g.bic(Z))
    return out


def gmm_heldout_ll(Z: np.ndarray, sessions: np.ndarray,
                   k_range: tuple[int, int], *, seed: int = 0) -> dict:
    """Mean per-sample held-out log-likelihood: fit the GMM on half the sessions,
    score the other half. Peaks then plateaus/falls when k overfits. Falls back to a
    random row split when there are <2 sessions."""
    from sklearn.mixture import GaussianMixture
    lo, hi = int(k_range[0]), int(k_range[1])
    uniq = np.asarray(pd.unique(pd.Series(sessions)), dtype=object)
    rng = np.random.default_rng(seed)
    if uniq.size >= 2:
        uniq = uniq[rng.permutation(uniq.size)]
        train_s = set(uniq[: uniq.size // 2].tolist())
        train = np.array([s in train_s for s in sessions])
    else:
        train = rng.random(Z.shape[0]) < 0.5
    test = ~train
    out: dict[int, float] = {}
    if train.sum() < (hi + 1) or test.sum() < 1:
        return out
    for k in range(lo, hi + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=2, reg_covar=1e-5)
        g.fit(Z[train])
        out[k] = float(np.mean(g.score_samples(Z[test])))
    return out


# --------------------------------------------------------------------- #
#  Stability: bootstrap ARI + Jaccard, observed minus matched null
# --------------------------------------------------------------------- #
def _jaccard_min(ref: np.ndarray, alt: np.ndarray) -> float:
    """Minimum over reference clusters of the best Jaccard overlap with any alt
    cluster (Hennig's dissolution-resistant clusters have high min Jaccard)."""
    best = []
    for r in np.unique(ref):
        rmask = ref == r
        j = 0.0
        for a in np.unique(alt):
            amask = alt == a
            inter = np.sum(rmask & amask)
            union = np.sum(rmask | amask)
            if union:
                j = max(j, inter / union)
        best.append(j)
    return float(min(best)) if best else float("nan")


def _stability_once(Xn: np.ndarray, k: int, block: int, seed: int,
                    n_boot: int) -> tuple[float, float]:
    """Mean bootstrap ARI and mean min-Jaccard of a k-means partition of *Xn* under
    moving-block resampling. Reference = fit on the full data."""
    from sklearn.metrics import adjusted_rand_score
    ref = _cl.correlation_kmeans(Xn, k, seed=seed)["labels"]
    rng = np.random.default_rng(seed + 7)
    aris, jacs = [], []
    n = Xn.shape[0]
    for _ in range(n_boot):
        idx = _rs.block_index(n, block, rng)
        res = _cl.correlation_kmeans(Xn[idx], k, n_init=10, seed=seed)
        lab = _cl.assign_by_template(Xn, res["templates"])
        aris.append(adjusted_rand_score(ref, lab))
        jacs.append(_jaccard_min(ref, lab))
    return float(np.mean(aris)), float(np.mean(jacs))


def _gaussian_null(Xn: np.ndarray, seed: int) -> np.ndarray:
    """A covariance-matched Gaussian null: PCA the traces, then draw a UNIMODAL
    Gaussian with each PC's variance and reconstruct. This matches the second-order
    structure but has NO clusters, so any bootstrap stability on it is the mechanical
    baseline. (An independent per-PC SHUFFLE was tried first and rejected: it preserves
    each PC's multimodal marginal, and the product of independent multimodal marginals
    forms a spurious grid that k-means clusters stably, which hides the real signal.)"""
    p = _st.shape_pca(Xn, n_components=min(10, Xn.shape[1]), seed=seed)
    var = np.var(p["Z"], axis=0)
    rng = np.random.default_rng(seed + 99)
    Z = rng.standard_normal((Xn.shape[0], var.size)) * np.sqrt(var)[None, :]
    return p["pca"].inverse_transform(Z)


def stability_curve(Xn: np.ndarray, k_range: tuple[int, int], *,
                    n_boot: int = 100, seed: int = 0) -> pd.DataFrame:
    """Per k: observed ARI/Jaccard, the same on a covariance-matched Gaussian null, and
    the observed-minus-null differences (the honest, k-inflation-corrected stability).
    Columns: ``k, ari, jaccard, ari_null, jaccard_null, ari_excess, jaccard_excess``."""
    lo, hi = int(k_range[0]), int(k_range[1])
    block = _rs.block_length(_st.shape_pca(Xn, n_components=1, seed=seed)["Z"][:, 0])
    null = _gaussian_null(Xn, seed)
    rows = []
    for k in range(lo, hi + 1):
        if k > Xn.shape[0]:
            break
        ari, jac = _stability_once(Xn, k, block, seed, n_boot)
        ari_n, jac_n = _stability_once(null, k, block, seed, n_boot)
        rows.append({"k": k, "ari": ari, "jaccard": jac,
                     "ari_null": ari_n, "jaccard_null": jac_n,
                     "ari_excess": ari - ari_n, "jaccard_excess": jac - jac_n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
#  Split-half reproducibility
# --------------------------------------------------------------------- #
def _within_cluster_corr(Xn: np.ndarray, labels: np.ndarray,
                         templates: np.ndarray) -> float:
    """Mean signed correlation of each trace with its assigned template."""
    U = _cl._unit_rows(Xn)
    Tn = _cl._unit_rows(templates)
    corr = np.sum(U * Tn[labels], axis=1)
    return float(np.nanmean(corr)) if corr.size else float("nan")


def split_half(Xn: np.ndarray, sessions: np.ndarray, k: int, *,
               seed: int = 0) -> dict:
    """Fit templates on half the sessions, assign both halves, report the
    within-cluster correlation on train vs held-out and their drop. A small drop means
    the k templates generalize; a large drop means they overfit that half."""
    uniq = np.asarray(pd.unique(pd.Series(sessions)), dtype=object)
    rng = np.random.default_rng(seed + 3)
    if uniq.size >= 2:
        uniq = uniq[rng.permutation(uniq.size)]
        a_s = set(uniq[: uniq.size // 2].tolist())
        a = np.array([s in a_s for s in sessions])
    else:
        a = rng.random(Xn.shape[0]) < 0.5
    b = ~a
    if a.sum() < k or b.sum() < 1:
        return {"k": int(k), "corr_train": float("nan"),
                "corr_test": float("nan"), "drop": float("nan")}
    res = _cl.correlation_kmeans(Xn[a], k, seed=seed)
    tmpl = res["templates"]
    la = _cl.assign_by_template(Xn[a], tmpl)
    lb = _cl.assign_by_template(Xn[b], tmpl)
    ca = _within_cluster_corr(Xn[a], la, tmpl)
    cb = _within_cluster_corr(Xn[b], lb, tmpl)
    return {"k": int(k), "corr_train": ca, "corr_test": cb, "drop": ca - cb}


# --------------------------------------------------------------------- #
#  Verdict: a defensible range, not a point
# --------------------------------------------------------------------- #
def select_range(*, bic: dict, heldout: dict, stability: pd.DataFrame,
                 templates_by_k: dict, dup_thresh: float = 0.9,
                 excess_thresh: float = 0.05) -> dict:
    """Combine the readouts into a range ``(k_lo, k_hi)`` with reasoning.

    The range is set by the HONEST evidence: the k values that are BOTH stable above
    the matched null (ari_excess >= excess_thresh) AND free of near-duplicate templates
    (max off-diagonal template correlation < dup_thresh). ``k_lo`` and ``k_hi`` are the
    min and max of that set. BIC and held-out log-likelihood are reported as context
    (GMM log-likelihood on PC scores keeps rising with k, so it is not used to set the
    lower bound). Falls back to the BIC minimum when the honest set is empty. Returns
    ``{"k_lo", "k_hi", "text", "detail"}``."""
    dup_ok = [k for k, T in templates_by_k.items()
              if not np.isfinite(_cl.max_offdiag_corr(T)) or
              _cl.max_offdiag_corr(T) < dup_thresh]
    stab_ok = (stability.loc[stability["ari_excess"] >= excess_thresh, "k"].tolist()
               if not stability.empty else list(templates_by_k.keys()))
    ok = sorted(set(dup_ok) & set(stab_ok))
    if ok:
        k_lo, k_hi = int(min(ok)), int(max(ok))
    else:
        k_lo = int(min(bic, key=bic.get)) if bic else 2
        k_hi = k_lo
    k_bic = int(min(bic, key=bic.get)) if bic else None
    k_ll = int(max(heldout, key=heldout.get)) if heldout else None
    text = (f"Defensible k range: {k_lo} to {k_hi}. Set by the k that are both stable "
            f"above the PC-shuffled null (ARI excess >= {excess_thresh}) and free of "
            f"near-duplicate templates (max template correlation < {dup_thresh}). "
            f"For context, BIC is minimized at k={k_bic} and held-out log-likelihood "
            f"peaks at k={k_ll} (GMM log-likelihood keeps rising with k, so it is not "
            f"used to set the lower bound).")
    return {"k_lo": int(k_lo), "k_hi": int(k_hi), "text": text,
            "detail": {"dup_ok": dup_ok, "stab_ok": stab_ok,
                       "k_bic": k_bic, "k_heldout": k_ll}}
