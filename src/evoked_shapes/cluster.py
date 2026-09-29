"""Stage 3: correlation-based clustering of the SHAPE space.

Primary method is correlation (spherical) k-means: L2-normalize each trace to the unit
sphere so Euclidean k-means becomes cosine/correlation clustering, then take each
centroid (renormalized) as a response template. Correlation keeps SIGN by default (a
trough-first and a peak-first response are different shapes); an ``absolute`` flag
aligns each trace's polarity first, for the rare case where only the waveform envelope
matters.

Two complementary views are reused from src.evoked_typology rather than re-implemented:
the GMM-by-BIC mixture (``gmm_view``) and the signed NMF prototypes (``nmf_view``).
DTW k-means (tslearn) is offered only if tslearn is importable, and skipped with a
logged note otherwise (no new dependency, and it protects the numpy<2.5 pin).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _unit_rows(Xn: np.ndarray) -> np.ndarray:
    """L2-normalize each row to the unit sphere (flat rows left as zeros)."""
    norm = np.sqrt(np.nansum(Xn * Xn, axis=1, keepdims=True))
    norm[norm == 0] = 1.0
    return Xn / norm


def _sign_align(U: np.ndarray) -> np.ndarray:
    """Flip each row so it correlates non-negatively with the mean-|shape| reference,
    for absolute-correlation clustering (envelope, ignoring polarity)."""
    ref = np.nanmean(np.abs(U), axis=0)
    sign = np.sign(U @ ref)
    sign[sign == 0] = 1.0
    return U * sign[:, None]


def correlation_kmeans(Xn: np.ndarray, k: int, *, n_init: int = 50,
                       seed: int = 0, absolute: bool = False) -> dict:
    """Correlation (spherical) k-means at a single *k*. Returns
    ``{"labels", "templates", "inertia", "k"}`` where ``templates`` are the unit-norm
    centroid shapes ``[k, T]``. ``n_init`` restarts keep the best objective."""
    from sklearn.cluster import KMeans
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    assert 2 <= k <= Xn.shape[0], "k must be in [2, n_trials]"
    U = _unit_rows(Xn)
    if absolute:
        U = _sign_align(U)
    km = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed))
    labels = km.fit_predict(U).astype(int)
    cent = km.cluster_centers_
    tnorm = np.sqrt((cent * cent).sum(axis=1, keepdims=True))
    tnorm[tnorm == 0] = 1.0
    templates = cent / tnorm
    return {"labels": labels, "templates": templates,
            "inertia": float(km.inertia_), "k": int(k)}


def assign_by_template(Xn: np.ndarray, templates: np.ndarray, *,
                       absolute: bool = False) -> np.ndarray:
    """Assign each trace to the most-correlated template (signed correlation by
    default). Used to carry a fit onto held-out data (split-half reproducibility)."""
    assert templates.ndim == 2, "templates must be 2-D [k, T]"
    U = _unit_rows(Xn)
    if absolute:
        U = _sign_align(U)
    Tn = _unit_rows(templates)
    corr = U @ Tn.T
    if absolute:
        corr = np.abs(corr)
    return np.argmax(corr, axis=1).astype(int)


def cluster_sweep(Xn: np.ndarray, *, k_range: tuple[int, int] = (2, 10),
                  n_init: int = 50, seed: int = 0, absolute: bool = False) -> dict:
    """Correlation k-means for every k in ``k_range`` (inclusive). Returns
    ``{k: result_dict}``. k values above the trial count are skipped."""
    lo, hi = int(k_range[0]), int(k_range[1])
    assert 2 <= lo <= hi, "bad k_range"
    out: dict[int, dict] = {}
    for k in range(lo, hi + 1):
        if k > Xn.shape[0]:
            break
        out[k] = correlation_kmeans(Xn, k, n_init=n_init, seed=seed,
                                    absolute=absolute)
    return out


def template_corr_matrix(templates: np.ndarray, *, absolute: bool = False) -> np.ndarray:
    """Signed (or absolute) correlation matrix between templates. Its max off-diagonal
    flags a k where two templates are near-duplicates (Stage-4 upper bound)."""
    Tn = _unit_rows(templates)
    C = Tn @ Tn.T
    return np.abs(C) if absolute else C


def max_offdiag_corr(templates: np.ndarray, *, absolute: bool = True) -> float:
    """Largest off-diagonal template correlation (absolute by default: a near-mirror
    pair is as redundant as a near-duplicate). NaN for a single template."""
    if templates.shape[0] < 2:
        return float("nan")
    C = template_corr_matrix(templates, absolute=absolute)
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.max(C[iu]))


# --------------------------------------------------------------------- #
#  Reused complementary views (src.evoked_typology)
# --------------------------------------------------------------------- #
def gmm_view(Xn: np.ndarray, *, n_pca: int = 8, k_range: tuple[int, int] = (2, 8),
             seed: int = 0) -> dict:
    """PCA then GMM with k chosen by BIC (reused from src.evoked_typology). The
    mixture comparison to the k-means partition. Returns that function's dict."""
    from src.evoked_typology import shape_clusters
    return shape_clusters(Xn, n_pca=n_pca, k_range=k_range, seed=seed)


def nmf_view(Xn: np.ndarray, *, k: int = 4, seed: int = 0) -> dict:
    """Signed NMF prototypes (reused from src.evoked_typology): the archetype /
    continuum view where each trace is a non-negative mix of prototype shapes."""
    from src.evoked_typology import nmf_prototypes
    # NMF needs the pos/neg split to be non-negative; that is handled inside.
    return nmf_prototypes(Xn, k=int(k), seed=int(seed))


def dtw_kmeans(Xn: np.ndarray, k: int, *, seed: int = 0,
               log: Callable[[str], None] | None = None) -> dict | None:
    """DTW k-means (latency-robust) via tslearn, ONLY if tslearn is importable.
    Returns ``{"labels", "k", "method": "dtw"}`` or ``None`` (with a logged note) when
    tslearn is absent, so no dependency is added and the numpy pin is protected."""
    try:
        from tslearn.clustering import TimeSeriesKMeans
    except ImportError:
        if log is not None:
            log("Stage 3: tslearn not installed; skipping DTW k-means "
                "(correlation k-means and GMM/NMF cover the shape structure).")
        return None
    U = _unit_rows(Xn)[:, :, None]
    km = TimeSeriesKMeans(n_clusters=int(k), metric="dtw", random_state=int(seed))
    labels = km.fit_predict(U).astype(int)
    return {"labels": labels, "k": int(k), "method": "dtw"}
