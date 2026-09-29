"""Stage 2: is the shape space DISCRETE (a few modes) or a CONTINUUM?

Four independent readouts, so no single method decides it:

  * PCA on the NORMALIZED traces (plain, un-standardized: standardizing each sample
    would reweight the waveform and distort shape, so we center-only like
    src.evoked_typology.shape_clusters). Report variance explained and PC1/PC2.
  * Pairwise correlation histogram: the distribution of trial-to-trial correlations.
    Bimodal (a high-within / low-between split) implies discrete families; unimodal
    implies a continuum. Computed on a capped subsample because it is O(n^2).
  * Hartigan dip test (implemented inline, GCM/LCM iteration) on PC1 and on the 1-D
    diffusion-map coordinate, with a uniform-null bootstrap p-value. A significant dip
    is positive evidence for at least two modes.
  * Diffusion map: a nonlinear 1-D ordering (and 2-D density) that separates modes
    better than linear PCA when the modes are not linearly arranged.

Everything is deterministic given the seed. The verdict is COMPILED EVIDENCE
(``summarize``), never a hard label: it states what each readout shows.
"""

from __future__ import annotations

import numpy as np

_MAX_PAIRWISE = 2000        # O(n^2) memory/compute cap for the correlation histogram
_MAX_DIP_N = 4000           # dip is O(n log n) but the bootstrap multiplies it
_DIP_ITER_MAX = 100         # NASA Rule 2: bounded modal-interval search


# --------------------------------------------------------------------- #
#  PCA on normalized shapes (center-only, no per-sample standardization)
# --------------------------------------------------------------------- #
def shape_pca(Xn: np.ndarray, n_components: int = 8, seed: int = 0) -> dict:
    """Plain PCA of the normalized traces. Returns ``{"Z", "explained", "pca"}``
    where ``Z`` is the score matrix ``[n, k]`` and ``explained`` the variance ratios."""
    from sklearn.decomposition import PCA
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    k = int(min(n_components, Xn.shape[1], max(1, Xn.shape[0] - 1)))
    p = PCA(n_components=k, random_state=seed)
    Z = p.fit_transform(Xn)
    return {"Z": Z, "explained": p.explained_variance_ratio_, "pca": p}


# --------------------------------------------------------------------- #
#  Pairwise correlation histogram
# --------------------------------------------------------------------- #
def pairwise_correlations(Xn: np.ndarray, *, cap: int = _MAX_PAIRWISE,
                          seed: int = 0) -> np.ndarray:
    """Off-diagonal pairwise Pearson correlations between trials (a capped random
    subsample of trials when n > cap). Returns a 1-D array of correlation values.
    Bimodality here is the discrete-families signature."""
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    n = Xn.shape[0]
    rng = np.random.default_rng(seed)
    if n > cap:
        sel = rng.choice(n, size=cap, replace=False)
        X = Xn[sel]
    else:
        X = Xn
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.sqrt((Xc * Xc).sum(axis=1))
    norm[norm == 0] = np.nan
    C = (Xc @ Xc.T) / np.outer(norm, norm)
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    return vals[np.isfinite(vals)]


def bimodality_coefficient(v: np.ndarray) -> float:
    """Sarle's bimodality coefficient of a 1-D sample: ``(skew^2 + 1) / kurtosis``
    (with the small-sample correction). Values above ~0.555 (the uniform value)
    suggest bimodality; the normal distribution sits at ~0.33. Descriptive only."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return float("nan")
    m = v.mean()
    s = v.std(ddof=1)
    if s == 0:
        return float("nan")
    z = (v - m) / s
    skew = (n / ((n - 1) * (n - 2))) * np.sum(z ** 3)
    g2 = (np.sum(z ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
          - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    kurt = g2 + 3.0
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# --------------------------------------------------------------------- #
#  Hartigan dip statistic (inline GCM/LCM iteration) + bootstrap p-value
# --------------------------------------------------------------------- #
def _minorant_chain(x: np.ndarray) -> np.ndarray:
    """Convex-minorant predecessor chain over the points (x[k], k)."""
    n = x.size
    mn = np.zeros(n, dtype=np.intp)
    for j in range(1, n):
        mn[j] = j - 1
        it = 0
        while it < n:
            it += 1
            mnj = int(mn[j])
            mnmnj = int(mn[mnj])
            if mnj == 0 or (x[j] - x[mnj]) * (mnj - mnmnj) < \
                    (x[mnj] - x[mnmnj]) * (j - mnj):
                break
            mn[j] = mnmnj
    return mn


def _majorant_chain(x: np.ndarray) -> np.ndarray:
    """Concave-majorant successor chain over the points (x[k], k)."""
    n = x.size
    mj = np.zeros(n, dtype=np.intp)
    mj[n - 1] = n - 1
    for k in range(n - 2, -1, -1):
        mj[k] = k + 1
        it = 0
        while it < n:
            it += 1
            mjk = int(mj[k])
            mjmjk = int(mj[mjk])
            if mjk == n - 1 or (x[k] - x[mjk]) * (mjk - mjmjk) < \
                    (x[mjk] - x[mjmjk]) * (k - mjk):
                break
            mj[k] = mjmjk
    return mj


def dip_statistic(samples: np.ndarray) -> float:
    """Hartigan and Hartigan (1985) dip statistic: the sup-distance between the
    empirical CDF and the closest unimodal CDF, found by iterating the greatest
    convex minorant (below the mode) against the least concave majorant (above it)
    and shrinking to the modal interval. Returned in ECDF (probability) units.
    0 for degenerate / tiny input."""
    x = np.sort(np.asarray(samples, dtype=np.float64))
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 or x[-1] <= x[0]:
        return 0.0
    mn = _minorant_chain(x)
    mj = _majorant_chain(x)
    low, high = 0, n - 1
    dip = 0.0
    for _ in range(_DIP_ITER_MAX):
        # GCM (convex minorant) touch indices from high down to low, then ascending.
        gcm = [high]
        while gcm[-1] > low:
            gcm.append(int(mn[gcm[-1]]))
        gidx = np.array(gcm[::-1], dtype=np.intp)          # ascending low..high
        # LCM (concave majorant) touch indices from low up to high.
        lcm = [low]
        while lcm[-1] < high:
            lcm.append(int(mj[lcm[-1]]))
        lidx = np.array(lcm, dtype=np.intp)                # ascending low..high
        gx, gh = x[gidx], gidx.astype(np.float64)
        lx, lh = x[lidx], lidx.astype(np.float64)
        # Vertical gap between the two hulls, evaluated at every knot of each.
        gap_at_g = np.interp(gx, lx, lh) - gh              # LCM above ECDF at GCM knots
        gap_at_l = lh - np.interp(lx, gx, gh)              # ECDF above GCM at LCM knots
        ix = int(gidx[int(np.argmax(gap_at_g))]) if gap_at_g.size else low
        iv = int(lidx[int(np.argmax(gap_at_l))]) if gap_at_l.size else high
        d_g = float(gap_at_g.max()) if gap_at_g.size else 0.0
        d_l = float(gap_at_l.max()) if gap_at_l.size else 0.0
        dip = max(d_g, d_l)
        # Refine the modal interval to [gcm-argmax, lcm-argmax] and iterate; the dip
        # SHRINKS toward the flat modal region (near 0 for a smooth unimodal shape).
        if ix >= iv or (ix == low and iv == high):
            break
        low, high = ix, iv
    return float(dip / (2.0 * n))


def dip_test(samples: np.ndarray, *, n_boot: int = 500, seed: int = 0) -> dict:
    """Dip statistic plus a Monte-Carlo p-value against the uniform null (the least
    favourable unimodal distribution, per Hartigan). p = fraction of uniform-sample
    dips >= the observed dip. Returns ``{"dip", "p", "n", "n_boot"}``. The SAME
    ``dip_statistic`` is used for observed and null, so the constant convention does
    not affect the p-value."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n > _MAX_DIP_N:
        rng0 = np.random.default_rng(seed)
        x = x[rng0.choice(n, size=_MAX_DIP_N, replace=False)]
        n = _MAX_DIP_N
    if n < 4:
        return {"dip": 0.0, "p": 1.0, "n": int(n), "n_boot": 0}
    obs = dip_statistic(x)
    rng = np.random.default_rng(seed + 1)
    ge = 0
    for _ in range(n_boot):
        u = rng.random(n)
        if dip_statistic(u) >= obs:
            ge += 1
    p = (ge + 1.0) / (n_boot + 1.0)
    return {"dip": float(obs), "p": float(p), "n": int(n), "n_boot": int(n_boot)}


# --------------------------------------------------------------------- #
#  Diffusion map (correlation/Gaussian affinity -> 1-D coord + 2-D)
# --------------------------------------------------------------------- #
def diffusion_map(Xn: np.ndarray, *, n_components: int = 2, cap: int = 3000,
                  seed: int = 0) -> dict:
    """Diffusion-map embedding of the normalized traces. Gaussian affinity on a
    correlation-distance metric, row-normalized to a Markov matrix, eigendecomposed.
    Returns ``{"coords", "rows", "eigvals"}`` where ``coords`` is ``[m, k]`` and
    ``rows`` the (subsampled) row indices aligned to it. Nonlinear, so it can order a
    curved continuum or separate modes that PCA smears."""
    assert Xn.ndim == 2, "Xn must be 2-D [n, T]"
    n = Xn.shape[0]
    rng = np.random.default_rng(seed)
    rows = (rng.choice(n, size=cap, replace=False) if n > cap
            else np.arange(n))
    X = Xn[rows]
    Xc = X - X.mean(axis=1, keepdims=True)
    norm = np.sqrt((Xc * Xc).sum(axis=1))
    norm[norm == 0] = 1.0
    C = (Xc @ Xc.T) / np.outer(norm, norm)
    C = np.clip(C, -1.0, 1.0)
    dist2 = 2.0 * (1.0 - C)                       # squared correlation distance
    med = np.median(dist2[dist2 > 0]) if np.any(dist2 > 0) else 1.0
    eps = med if med > 0 else 1.0
    W = np.exp(-dist2 / eps)
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    P = W / d[:, None]                            # row-stochastic Markov matrix
    # Symmetric conjugate P_sym = D^-1/2 W D^-1/2 shares P's eigenvalues.
    ds = np.sqrt(d)
    Psym = W / np.outer(ds, ds)
    Psym = 0.5 * (Psym + Psym.T)                 # guard tiny asymmetry
    vals, vecs = np.linalg.eigh(Psym)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    phi = vecs / ds[:, None]                      # back to P's right eigenvectors
    k = int(min(n_components, phi.shape[1] - 1))
    coords = phi[:, 1:1 + k] * vals[1:1 + k][None, :]   # drop trivial 1st component
    return {"coords": coords, "rows": rows, "eigvals": vals[:1 + k]}


# --------------------------------------------------------------------- #
#  Verdict: compile the evidence
# --------------------------------------------------------------------- #
def dip_scan(Z: np.ndarray, *, n_axes: int = 4, n_boot: int = 200,
             seed: int = 0) -> list[dict]:
    """Dip test on each of the top ``n_axes`` PCA scores. Scanning several axes (not
    only PC1) catches modes that separate along a later component; the caller applies a
    multiple-comparison correction over the returned list."""
    assert Z.ndim == 2, "Z must be 2-D [n, k]"
    axes = int(min(n_axes, Z.shape[1]))
    return [dip_test(Z[:, j], n_boot=n_boot, seed=seed) for j in range(axes)]


def summarize(*, pc_dips: list[dict], diff_dip: dict, corr_vals: np.ndarray,
              explained: np.ndarray, alpha: float = 0.05) -> dict:
    """Compile the Stage-2 readouts into a plain dict and a human verdict string.

    Primary signal: the dip test scanned across the top PCA axes, Bonferroni-corrected
    (a mode can separate along PC2+, not only PC1). DISCRETE when any axis dips at
    ``alpha / n_axes``; otherwise CONTINUUM (leaning). The diffusion-coordinate dip and
    the pairwise-correlation bimodality are reported as CONTEXT only, not the verdict:
    the diffusion coordinate piles points up at its boundaries and so dips even for a
    genuine continuum, and Sarle's coefficient is insensitive to a spike-plus-mass
    correlation histogram. Never a hard classifier."""
    n_ax = max(1, len(pc_dips))
    thr = alpha / n_ax
    ps = [d.get("p", 1.0) for d in pc_dips]
    arg = int(np.argmin(ps)) if ps else 0
    min_p = float(ps[arg]) if ps else 1.0
    discrete = bool(min_p < thr)
    bc = bimodality_coefficient(corr_vals)
    pc1 = pc_dips[0] if pc_dips else {"dip": float("nan"), "p": float("nan"), "n": 0}
    verdict = "DISCRETE (>=2 modes)" if discrete else "CONTINUUM (leaning)"
    lines = [
        f"Verdict: {verdict}.",
        f"  dip scan over top {n_ax} PCs: min p={min_p:.3f} on PC{arg + 1} "
        f"(Bonferroni threshold {thr:.3f}); PC1 dip={pc1.get('dip', float('nan')):.4f} "
        f"p={pc1.get('p', float('nan')):.3f} (n={pc1.get('n', 0)}).",
        f"  [context] diffusion-coord dip p={diff_dip.get('p', float('nan')):.3f} "
        f"(dips on continua too); pairwise-corr bimodality coeff={bc:.3f}.",
        f"  PC1/PC2 variance explained="
        f"{explained[0] * 100:.0f}%/{explained[1] * 100:.0f}%."
        if explained.size >= 2 else "  PCA variance: <2 components.",
    ]
    return {"discrete": discrete, "verdict": verdict, "bimodality_coeff": bc,
            "pc_dips": pc_dips, "pc1_dip": pc1, "diff_dip": diff_dip,
            "min_p": min_p, "min_p_axis": arg + 1, "text": "\n".join(lines)}
