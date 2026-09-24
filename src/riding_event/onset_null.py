"""Null-onset control for the pre-ictal ripple-rate trend.

The per-seizure ripple-rate pipeline (``periictal.py``) reports the matched-filter
rate rising toward seizure onset. Before believing that is a PRE-ICTAL process (and
not drift + a binning selection effect), we run the SAME detection + binning on
random DEEP-INTERICTAL "null" onsets (``periictal.null_onsets.draw_null_set``:
count-matched to the real seizures, a seizure-free buffer on both sides, day-matched)
and ask whether the real toward-onset fold-change exceeds the null distribution.

Detection is cached per-recording (onset-independent), so the real side is free;
each null onset only re-reads the deep-interictal recordings it overlaps (that IS
the cost -- those weren't near a seizure). We reuse ``seizure_trajectory`` per null
onset in a process pool, aggregate each draw with ``periictal_stats`` (baseline vs
last-hour fold), pool K draws into a null band, and get an empirical p via
``preictal.validation.null_stats``. The real per-seizure trajectories are the ones
already cached; we also give the real fold a leave-one-seizure-out CI.

CLI: ``python -m src.riding_event.onset_null --animal BCH111 --k 6 --jobs 3``
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.riding_event.periictal import (                       # noqa: E402
    ensure_template, seizure_trajectory, load_trajectories, periictal_stats,
    _root)
from src.periictal.null_onsets import draw_null_set            # noqa: E402
from src.preictal.validation import null_stats                 # noqa: E402

_BG, _TEXT, _MUTED, _ACC = "#1e1e2f", "#f0f0f5", "#9a9ab0", "#5e7ce2"


# --------------------------------------------------------------------- #
#  process-pool worker (module-level so it pickles)
# --------------------------------------------------------------------- #
def _null_traj(onset, animal, config_path, horizon_h, post_h, bin_min, thresh):
    """One null-onset trajectory. Opens its own Store + the cached template; returns
    only the small arrays needed (picklable)."""
    import yaml
    from src.db.store import Store
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    tmpl = ensure_template(store, animal)
    t = seizure_trajectory(store, animal, float(onset), tmpl, horizon_h=horizon_h,
                           post_h=post_h, bin_min=bin_min, thresh=thresh)
    return {"onset_epoch": float(onset), "centers_min": t["centers_min"],
            "phfo_rate": t["phfo_rate"], "mf_rate": t["mf_rate"],
            "cover_sec": t["cover_sec"], "n_recs": t["n_recs"]}


def _group_rate(trajs, key):
    """Across-seizure per-bin (mean, sem) of *key* -- mean where >=1 seizure covers
    the bin, SEM = std/sqrt(n) where >=2 (mirrors periictal._agg_panel)."""
    trajs = [t for t in trajs if t.get(key) is not None]
    if not trajs:
        return None, None, None
    ref = trajs[0]["centers_min"]
    Y = np.vstack([t[key] for t in trajs if t[key].size == ref.size])
    n = np.sum(np.isfinite(Y), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Y, axis=0)
        sem = np.nanstd(Y, axis=0) / np.sqrt(np.maximum(n, 1))
    mean[n < 1] = np.nan
    sem[n < 2] = np.nan
    return ref / 60.0, mean, sem


def _fold_per_seizure(trajs, key, baseline_h=(6.0, 4.0), last_h=1.0):
    """Per-seizure last-hour/baseline fold, using EACH trajectory's OWN time axis
    (horizon-independent, so trajectories with different bin counts still count).
    Finite pairs only."""
    out = []
    for t in trajs:
        y = t.get(key)
        c = t.get("centers_min")
        if y is None or c is None or np.asarray(y).size != np.asarray(c).size:
            continue
        ref = np.asarray(c, dtype=float) / 60.0
        bm = (ref <= -baseline_h[1]) & (ref >= -baseline_h[0])
        lm = ref >= -last_h
        b = np.nanmean(y[bm]) if bm.any() and np.isfinite(y[bm]).any() else np.nan
        ll = np.nanmean(y[lm]) if lm.any() and np.isfinite(y[lm]).any() else np.nan
        if np.isfinite(b) and np.isfinite(ll) and b > 0:
            out.append(ll / b)
    return np.asarray(out)


def _boot_median_ci(x, *, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the MEDIAN across seizures (the unit)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(x, x.size, replace=True))
                     for _ in range(n_boot)])
    return (float(np.percentile(meds, 100 * alpha / 2)),
            float(np.percentile(meds, 100 * (1 - alpha / 2))))


def run_null_control(config_path, animal="BCH111", *, k_draws=6, seed=0, jobs=3,
                     horizon_h=6.0, post_h=3.0, bin_min=10.0, thresh=0.7,
                     key="phfo_rate", out_dir=None, progress=None):
    """Compare the real toward-onset ripple-rate fold against K matched deep-
    interictal null draws. Returns a result dict + writes a figure."""
    import yaml
    from concurrent.futures import ProcessPoolExecutor
    from src.db.store import Store
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    store = Store(cfg["database"]["path"])
    evoked_dir = (cfg.get("chronic_evoked", {}) or {}).get("evoked_output_dir", "")
    out_dir = out_dir or os.path.join(_root(animal), "onset_null")
    os.makedirs(out_dir, exist_ok=True)

    # REAL side: the already-cached per-seizure trajectories.
    real = load_trajectories(animal)
    if not real:
        return {"error": "no real trajectories -- run periictal first"}
    real_stats = periictal_stats(real, key=key)
    real_folds = _fold_per_seizure(real, key)
    n_real = len(real_folds)
    if progress:
        progress(f"real: {len(real)} seizures, {n_real} with finite fold; "
                 f"median fold {real_stats['median_fold']:.2f}")

    # NULL draws (count-matched to the real seizures used).
    draws = draw_null_set(store, animal, evoked_dir,
                          seeds=list(range(seed, seed + k_draws)),
                          n=max(3, n_real), window_sec=horizon_h * 3600.0,
                          band_hi_sec=horizon_h * 3600.0, progress=progress)
    tasks = [(float(o), di) for di, d in enumerate(draws) for o in d.onsets]
    if progress:
        progress(f"{len(draws)} null draws, {len(tasks)} null onsets -> detecting "
                 f"(deep-interictal recordings, cached per file)…")
    by_draw: dict = {di: [] for di in range(len(draws))}
    with ProcessPoolExecutor(max_workers=max(1, int(jobs))) as ex:
        futs = {ex.submit(_null_traj, o, animal, config_path, horizon_h, post_h,
                          bin_min, thresh): di for (o, di) in tasks}
        done = 0
        for fu in futs:
            di = futs[fu]
            try:
                by_draw[di].append(fu.result())
            except Exception as e:                 # noqa: BLE001
                if progress:
                    progress(f"null onset failed: {e}")
            done += 1
            if progress and done % 5 == 0:
                progress(f"null onsets done {done}/{len(tasks)}")

    # per-draw median fold; pool K draws into the null distribution of median folds.
    null_folds = []
    for di in range(len(draws)):
        f = _fold_per_seizure(by_draw[di], key)
        if f.size:
            null_folds.append(float(np.median(f)))
    null_folds = np.asarray(null_folds)
    real_median = float(np.median(real_folds)) if real_folds.size else np.nan
    ns = null_stats(real_median, null_folds) if null_folds.size \
        else {"p": np.nan, "percentile": np.nan}
    ci_lo, ci_hi = _boot_median_ci(real_folds, seed=seed)

    png = os.path.join(out_dir, f"{animal}_onset_null_{key}.png")
    _fig(real_folds, null_folds, real_median, (ci_lo, ci_hi), ns, key, animal, png)
    res = {"animal": animal, "key": key, "n_real_folds": int(real_folds.size),
           "real_median_fold": real_median, "loso_fold_ci": (ci_lo, ci_hi),
           "null_median_fold": float(np.median(null_folds)) if null_folds.size
           else np.nan, "null_p_empirical": ns.get("p"),
           "k_draws": len(draws), "n_null_onsets": len(tasks), "figure": png}
    return res


def _fig(real_folds, null_folds, real_median, ci, ns, key, animal, out):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), facecolor=_BG,
                                  gridspec_kw={"width_ratios": [1, 1.2]})
    for a in (ax, ax2):
        a.set_facecolor("#26263a")
        a.tick_params(colors=_MUTED, labelsize=8)
    # left: per-seizure fold forest (the effective n)
    y = np.arange(real_folds.size)
    ax.scatter(real_folds, y, s=40, color=_ACC, zorder=3)
    if np.isfinite(ci[0]):
        ax.axvspan(ci[0], ci[1], color=_ACC, alpha=0.15,
                   label=f"median {real_median:.2f}  95% CI [{ci[0]:.2f},{ci[1]:.2f}]")
    ax.axvline(real_median, color=_ACC, lw=2)
    ax.axvline(1.0, color=_MUTED, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"sz {i + 1}" for i in y], fontsize=7)
    ax.set_xlabel("last-hour / baseline fold", color=_TEXT, fontsize=10)
    ax.set_title(f"{animal} · real per-seizure fold (n={real_folds.size})",
                 color=_TEXT, fontsize=11)
    leg = ax.legend(fontsize=8, frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(_TEXT)
    # right: real median vs the null distribution of median folds
    if null_folds.size:
        ax2.hist(null_folds, bins=min(20, max(5, null_folds.size)),
                 color="#e08a3c", alpha=0.6, label=f"null (K={null_folds.size})")
    ax2.axvline(real_median, color=_ACC, lw=2.5,
                label=f"real median {real_median:.2f}")
    ax2.axvline(1.0, color=_MUTED, lw=0.8, ls=":")
    ax2.set_xlabel(f"median {key} fold (last-hour / baseline)", color=_TEXT,
                   fontsize=10)
    ax2.set_title("real vs deep-interictal NULL onsets", color=_TEXT, fontsize=12)
    p = ns.get("p")
    nm = float(np.median(null_folds)) if null_folds.size else float("nan")
    verdict = ("SURVIVES: real > null" if np.isfinite(p) and p < 0.05
               else "CONFOUNDED: within null" if np.isfinite(p)
               else "insufficient null")
    ax2.text(0.98, 0.98, f"null median {nm:.2f}\nempirical p(real>null)={p:.3g}\n"
             f"→ {verdict}", transform=ax2.transAxes, ha="right", va="top",
             color=_TEXT, fontsize=9,
             bbox=dict(boxstyle="round", fc="#1e1e2f", ec="#3a3a52"))
    leg2 = ax2.legend(fontsize=8, frameon=False, loc="upper left")
    for t in leg2.get_texts():
        t.set_color(_TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=125, facecolor=_BG)
    plt.close(fig)
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--animal", default="BCH111")
    ap.add_argument("--k", type=int, default=6, help="null draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--horizon-h", type=float, default=6.0)
    ap.add_argument("--key", default="phfo_rate",
                    choices=["phfo_rate", "mf_rate"])
    a = ap.parse_args(argv)
    res = run_null_control(a.config, a.animal, k_draws=a.k, seed=a.seed,
                           jobs=a.jobs, horizon_h=a.horizon_h, key=a.key,
                           progress=lambda m: print("  ", m, flush=True))
    print("RESULT:", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
