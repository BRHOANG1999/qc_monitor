"""Discover the KINDS of evoked response (unsupervised morphology typology).

Sorting responses by one feature's deciles only ranks them along a single axis. To
find *types* we work in multivariate SHAPE space, and — because the deciles are
mostly amplitude scaling — we AMPLITUDE-NORMALIZE each response first, so the types
reflect shape, not size.

Two complementary views (the operator asked for both):

  A. SHAPE CLUSTERS. Peak-normalize -> PCA -> GMM (k by BIC) -> each cluster is a
     response "type"; we show its mean±SD template, prevalence, and how it breaks
     down by circadian bin / amplitude / day (so a "type" that is really one stim
     condition is visible as such, not mistaken for an intrinsic mode).

  C. NMF PROTOTYPES (archetypal parts). We split each normalized response into its
     positive and negative parts (so the data is non-negative), run NMF, and read
     the signed components back out as a few additive PROTOTYPE shapes; every
     response is a non-negative mixture of them. Answers "what building-block shapes
     compose the responses, and where does each response sit between them."

Pure-ish (numpy/sklearn); a gather step reads traces via the shared readers. Run:
``python -m src.evoked_typology --animal BCH111 --date 2026-09-20 --days 3``
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.notifications import evoked_digest as _ed        # noqa: E402
from src.notifications import evoked_windowed as _ew       # noqa: E402
from src.evoked_figures import data as _d                 # noqa: E402
from src.utils import evoked_features as _ef               # noqa: E402
from src.utils.evoked_output import read_feature_sidecar, read_file_evoked  # noqa: E402

_BG, _PANEL, _TEXT, _MUTED = "#1e1e2f", "#26263a", "#f0f0f5", "#9a9ab0"


# --------------------------------------------------------------------- #
#  gather the response-waveform matrix + per-response metadata
# --------------------------------------------------------------------- #
def gather_responses(animal, evoked_dir, end_day, days, channel, win_ms=(2.0, 120.0),
                     cap=15000, seed=0, progress=None):
    """Return (W[n,T] amplitude-preserving baseline-corrected+blanked responses in
    the window, tm[T], meta dict of arrays). Reads traces once per file; subsamples
    to *cap* responses (reservoir) so the clustering stays quick."""
    files = _ed.list_window_files(animal, evoked_dir, end_day, days)
    rng = np.random.default_rng(seed)
    W, secs, amp = [], [], []
    seen = 0
    tm_win = [None]
    for i, fp in enumerate(files):
        if progress and i % 5 == 0:
            progress(f"gathering responses… ({i}/{len(files)})")
        rows = [r for r in (read_feature_sidecar(fp, animal) or [])
                if r.get("channel") == channel]
        if not rows:
            continue
        ev = read_file_evoked(fp, only_animals=[animal]).get(channel)
        if not ev or ev.get("traces") is None:
            continue
        traces = np.asarray(ev["traces"], dtype=float)
        time_ms = np.asarray(ev["time_ms"], dtype=float)
        st_times = np.asarray(ev["times"], dtype=float)
        base = (time_ms >= -50) & (time_ms <= -5)
        wm = (time_ms >= win_ms[0]) & (time_ms <= win_ms[1])
        if wm.sum() < 8:
            continue
        if tm_win[0] is None:
            tm_win[0] = time_ms[wm]
        st2dt = {round(float(_d._nan(r.get("stim_time_sec"))), 4):
                 (_d.parse_iso(r.get("abs_dt")) if r.get("abs_dt") else None)
                 for r in rows if np.isfinite(_d._nan(r.get("stim_time_sec")))}
        for j, stt in enumerate(st_times):
            dt = st2dt.get(round(float(stt), 4))
            if dt is None:
                continue
            y = traces[j].astype(float)
            if base.any():
                y = y - np.nanmean(y[base])
            yw = y[wm]
            if yw.size != tm_win[0].size or not np.all(np.isfinite(yw)):
                continue
            seen += 1
            rec = (yw, dt.timestamp(), float(np.nanmax(yw) - np.nanmin(yw)))
            if len(W) < cap:
                W.append(rec[0]); secs.append(rec[1]); amp.append(rec[2])
            else:                                  # reservoir replace
                r_ = int(rng.integers(0, seen))
                if r_ < cap:
                    W[r_], secs[r_], amp[r_] = rec
    if not W:
        return None, None, None
    W = np.asarray(W)
    secs = np.asarray(secs)
    meta = {"secs": secs, "amp": np.asarray(amp),
            "hour": np.array([datetime.fromtimestamp(s).hour for s in secs]),
            "cbin": np.array([_ew._circadian(datetime.fromtimestamp(s))[1]
                              for s in secs]),
            "day": np.array([datetime.fromtimestamp(s).date().isoformat()
                             for s in secs])}
    return W, tm_win[0], meta


def normalize_shapes(W):
    """Peak-normalize each response to unit max|amplitude| (sign kept) so clustering
    sees SHAPE, not size. Flat rows -> left as zeros."""
    pk = np.nanmax(np.abs(W), axis=1, keepdims=True)
    pk[pk == 0] = 1.0
    return W / pk


# --------------------------------------------------------------------- #
#  A. shape clustering (PCA -> GMM, k by BIC)
# --------------------------------------------------------------------- #
def shape_clusters(Wn, n_pca=8, k_range=(2, 8), seed=0):
    """PCA the normalized shapes, fit GMMs over *k_range*, pick k by BIC. Returns
    ``{labels, k, pcs, Z(scores), bic, probs}``."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    p = PCA(n_components=min(n_pca, Wn.shape[1]), random_state=seed)
    Z = p.fit_transform(Wn)
    best = None
    bic = {}
    for k in range(k_range[0], k_range[1] + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=seed, n_init=2, reg_covar=1e-5)
        g.fit(Z)
        bic[k] = float(g.bic(Z))
        if best is None or bic[k] < bic[best[0]]:
            best = (k, g)
    k, g = best
    return {"labels": g.predict(Z), "k": k, "pca": p, "Z": Z, "bic": bic,
            "probs": g.predict_proba(Z),
            "explained": p.explained_variance_ratio_}


# --------------------------------------------------------------------- #
#  C. NMF prototypes (non-negative pos/neg split -> signed components)
# --------------------------------------------------------------------- #
def nmf_prototypes(Wn, k=4, seed=0):
    """Split each normalized response into positive+negative parts (so it is
    non-negative), run NMF, and recover k SIGNED prototype shapes plus each
    response's non-negative mixture weights. Returns ``{components[k,T], weights[n,k]}``."""
    from sklearn.decomposition import NMF
    T = Wn.shape[1]
    X = np.hstack([np.maximum(Wn, 0.0), np.maximum(-Wn, 0.0)])   # n x 2T, >=0
    m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=600)
    Wt = m.fit_transform(X)               # n x k weights
    H = m.components_                      # k x 2T
    comps = H[:, :T] - H[:, T:]           # signed prototype shapes
    return {"components": comps, "weights": Wt}


# --------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------- #
def _dark(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)


def fig_type_atlas(Wn, tm, meta, res, out):
    """Per discovered TYPE: mean±SD normalized shape template, prevalence, and its
    circadian-bin composition (to expose condition-driven 'types')."""
    labels, k = res["labels"], res["k"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, k))
    ncol = min(k, 4)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                             facecolor=_BG, squeeze=False)
    n = len(labels)
    for c in range(k):
        ax = axes[c // ncol][c % ncol]
        _dark(ax)
        m = labels == c
        mean = np.nanmean(Wn[m], axis=0)
        sd = np.nanstd(Wn[m], axis=0)
        ax.fill_between(tm, mean - sd, mean + sd, color=colors[c], alpha=0.22, lw=0)
        ax.plot(tm, mean, color=colors[c], lw=1.8)
        frac = m.sum() / n
        comp = np.bincount(meta["cbin"][m], minlength=4) / max(1, m.sum())
        comp_s = " ".join(f"{v:.0%}" for v in comp)
        ax.set_title(f"type {c + 1}  ·  {frac:.0%} of responses (n={m.sum()})",
                     color=_TEXT, fontsize=10, loc="left")
        ax.set_xlabel(f"ms · circadian mix [dE dL nE nL]= {comp_s}",
                      color=_MUTED, fontsize=8)
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Response TYPE atlas (A) — {k} shape clusters (peak-normalized "
                 f"mean±SD)", color=_TEXT, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=125, facecolor=_BG)
    plt.close(fig)
    return out


def fig_pc_scatter(res, meta, out):
    """PC1×PC2 of the shape space, colored by TYPE and (2nd panel) by hour-of-day —
    the confound check: do the types just track time?"""
    Z, labels, k = res["Z"], res["labels"], res["k"]
    colors = plt.cm.turbo(np.linspace(0.12, 0.92, k))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    _dark(axes[0]); _dark(axes[1])
    for c in range(k):
        m = labels == c
        axes[0].scatter(Z[m, 0], Z[m, 1], s=4, color=colors[c], alpha=0.4,
                        label=f"type {c + 1}")
    ev = res["explained"]
    axes[0].set_title(f"shape PCA — colored by type (k={k} by BIC)", color=_TEXT,
                      fontsize=11)
    axes[0].set_xlabel(f"PC1 ({ev[0]:.0%})", color=_MUTED, fontsize=9)
    axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})", color=_MUTED, fontsize=9)
    leg = axes[0].legend(fontsize=8, frameon=False, markerscale=2)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    sc = axes[1].scatter(Z[:, 0], Z[:, 1], s=4, c=meta["hour"], cmap="twilight",
                         alpha=0.5)
    axes[1].set_title("same space — colored by hour-of-day (confound check)",
                      color=_TEXT, fontsize=11)
    axes[1].set_xlabel(f"PC1 ({ev[0]:.0%})", color=_MUTED, fontsize=9)
    cb = fig.colorbar(sc, ax=axes[1]); cb.set_label("hour", color=_MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=125, facecolor=_BG)
    plt.close(fig)
    return out


def fig_nmf(Wn, tm, nmf, out):
    """The NMF PROTOTYPE shapes (signed) + how strongly each response loads on each
    (weight distributions) — the 'building blocks' every response mixes."""
    comps, Wt = nmf["components"], nmf["weights"]
    k = comps.shape[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, k))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=_BG)
    _dark(axes[0]); _dark(axes[1])
    for c in range(k):
        axes[0].plot(tm, comps[c], color=colors[c], lw=1.8,
                     label=f"prototype {c + 1}")
    axes[0].axhline(0, color=_MUTED, lw=0.5, alpha=0.4)
    axes[0].set_title("NMF prototype shapes (C) — signed building blocks",
                      color=_TEXT, fontsize=11)
    axes[0].set_xlabel("ms since stim", color=_MUTED, fontsize=9)
    leg = axes[0].legend(fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    # dominant-prototype prevalence
    dom = np.argmax(Wt, axis=1)
    frac = np.bincount(dom, minlength=k) / len(dom)
    axes[1].bar(range(1, k + 1), frac, color=colors)
    axes[1].set_title("prevalence: responses whose dominant mix is each prototype",
                      color=_TEXT, fontsize=11)
    axes[1].set_xlabel("prototype", color=_MUTED, fontsize=9)
    axes[1].set_ylabel("fraction", color=_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=125, facecolor=_BG)
    plt.close(fig)
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--date", default=None, help="end day YYYY-MM-DD (default: now)")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--cap", type=int, default=15000)
    ap.add_argument("--nmf-k", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evoked_dir = cfg["chronic_evoked"]["evoked_output_dir"]
    end = datetime.fromisoformat(a.date) if a.date else datetime.now()
    out = a.out or os.path.join("data", "evoked_typology")
    os.makedirs(out, exist_ok=True)
    prog = lambda m: print("  ", m)                          # noqa: E731
    _f, wser = _ew._pkg._period_series(a.animal, evoked_dir, end, a.days,
                                       ["peak_to_trough"])
    channel = _d.primary_channel(a.animal, wser, ["peak_to_trough"], None)
    print("channel:", channel)
    W, tm, meta = gather_responses(a.animal, evoked_dir, end, a.days, channel,
                                   cap=a.cap, progress=prog)
    if W is None:
        print("no responses"); return 1
    print(f"gathered {len(W)} responses, T={W.shape[1]}")
    Wn = normalize_shapes(W)
    res = shape_clusters(Wn)
    print(f"A: {res['k']} shape types (BIC), sizes:",
          np.bincount(res["labels"]).tolist())
    nmf = nmf_prototypes(Wn, k=a.nmf_k)
    p1 = fig_type_atlas(Wn, tm, meta, res, os.path.join(out, "A_type_atlas.png"))
    p2 = fig_pc_scatter(res, meta, os.path.join(out, "A_pc_scatter.png"))
    p3 = fig_nmf(Wn, tm, nmf, os.path.join(out, "C_nmf_prototypes.png"))
    print("SAVED:", p1); print("SAVED:", p2); print("SAVED:", p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
